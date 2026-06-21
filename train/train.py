#!/usr/bin/env python3
"""Zell: coherence-core trainer (continued pretrain on the mixed blend).

Full fine-tune of Qwen2.5-0.5B-Instruct on the packed uint32 blend memmap from
tools/build_blend.py. Mixed text + code + chat + synthetic tool-call tokens are
trained together, so coherence/code/chat/tools are all present from step 1.

Single GPU:
    python train/train.py --meta /kaggle/working/meta.json --train-tokens 50000000
Dual T4 (DDP):
    accelerate launch --multi_gpu --num_processes 2 train/train.py \
        --meta /kaggle/working/meta.json --train-tokens 50000000

Speed notes for Kaggle 2xT4 (Turing, sm_75):
- T4 has fp16 tensor cores but NOT bf16 tensor cores, so --precision fp16 is the
  fast default here; pass --precision bf16 to match the design-of-record dtype.
- Packed fixed-length sequences = zero padding waste. Gradient checkpointing +
  paged_adamw_8bit keep the 0.5B full fine-tune inside 16GB/GPU.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

NP_DTYPE = {"uint16": np.uint16, "uint32": np.uint32}


class PackedBlocks(Dataset):
    """Contiguous fixed-length blocks over a packed token memmap. Causal-LM
    labels == input_ids; HF shifts internally."""

    def __init__(self, path, dtype, seq_len, n_tokens):
        self.arr = np.memmap(path, dtype=dtype, mode="r")
        usable = min(n_tokens, len(self.arr)) if n_tokens else len(self.arr)
        self.seq_len = seq_len
        self.n_blocks = max(0, usable // seq_len)

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, i):
        s = i * self.seq_len
        block = np.asarray(self.arr[s:s + self.seq_len], dtype=np.int64)
        return {"input_ids": block, "labels": block.copy(),
                "attention_mask": np.ones_like(block)}


@torch.no_grad()
def sample_generations(model, tok, device, prompts, max_new=120):
    model.eval()
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n  [user] {p}\n  [zell] {gen.strip()}", flush=True)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-dir", default="/kaggle/working/zell-core")
    ap.add_argument("--train-tokens", type=int, default=50_000_000,
                    help="token budget; max_steps derived from this and the effective batch")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)
    dtype = NP_DTYPE[meta["dtype"]]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, default_data_collator)

    tok = AutoTokenizer.from_pretrained(meta["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        meta["model_id"],
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ds = PackedBlocks(meta["train_path"], dtype, args.seq_len, meta["train_tokens"])
    tokens_per_step = args.per_device_batch * args.grad_accum * args.seq_len * world
    max_steps = max(1, args.train_tokens // tokens_per_step)
    if is_main:
        print(f"  data: {len(ds):,} blocks of {args.seq_len} | {tokens_per_step:,} tok/step "
              f"| {max_steps:,} steps for {args.train_tokens:,} tokens (world={world})", flush=True)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=2,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=default_data_collator)
    trainer.train(resume_from_checkpoint=args.resume)

    if is_main:
        trainer.save_model(args.out_dir)
        tok.save_pretrained(args.out_dir)
        print(f"  saved core -> {args.out_dir}", flush=True)
        sample_generations(model, tok, model.device, [
            "Explain how a suffix array works, briefly.",
            "Write a Python function that returns the n-th Fibonacci number.",
            "What's the weather like in Tokyo right now?",
            "Give me three ideas for a weekend project.",
        ])


if __name__ == "__main__":
    main()
