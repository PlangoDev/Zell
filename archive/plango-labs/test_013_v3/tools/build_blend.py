#!/usr/bin/env python3
"""BrainFormer v14 — blended multi-source dataset builder.

Streams several high-quality Hugging Face datasets, interleaves them by
deterministic per-source TOKEN QUOTAS into one uint16 memmap, serializes the
chat/instruction sources into a ChatML-lite format so the brain learns the chat
skeleton, holds out a chat eval split (no leakage), and writes a meta.json the
C/CUDA brain reads unchanged.

Run on Kaggle in a dedicated **Internet-ON** notebook (separate from training):
    python build_blend.py --pool-tokens 4500000000 --out-dir /kaggle/working

Then train internet-OFF: ./brain --meta /kaggle/working/meta.json ...

Notes:
- uint16, no header (token i = bytes [2i,2i+2)). doc separator = id 0 (Pythia EOS).
- NEVER pass num_proc to a streaming map (deadlocks). Batched tokenization gives
  multi-core throughput on its own.
- HF_TOKEN read from env if set (higher rate limits); not required for these
  public sources.
"""
import argparse
import json
import os
import time
import hashlib

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hfcache")

MODEL_ID = "EleutherAI/pythia-410m"

# kind: "text" (plain field) | "chat_messages" (list of {role,content}) | "chat_dolly"
# weight = fraction of the pool token budget. (oasst1 dropped: streaming tree
# reconstruction is error-prone; its share folded into ultrachat.)
SOURCES = [
    {"name": "fineweb-edu", "hf": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
     "split": "train", "field": "text", "kind": "text", "weight": 0.42},
    {"name": "wikipedia", "hf": "wikimedia/wikipedia", "config": "20231101.en",
     "split": "train", "field": "text", "kind": "text", "weight": 0.13},
    {"name": "cosmopedia", "hf": "HuggingFaceTB/cosmopedia", "config": "web_samples_v2",
     "split": "train", "field": "text", "kind": "text", "weight": 0.13},
    {"name": "open-web-math", "hf": "open-web-math/open-web-math", "config": None,
     "split": "train", "field": "text", "kind": "text", "weight": 0.10},
    {"name": "github-code", "hf": "codeparrot/github-code-clean", "config": "Python-all",
     "split": "train", "field": "code", "kind": "text", "weight": 0.10},
    {"name": "ultrachat", "hf": "HuggingFaceH4/ultrachat_200k", "config": None,
     "split": "train_sft", "field": "messages", "kind": "chat_messages", "weight": 0.075},
    {"name": "smoltalk", "hf": "HuggingFaceTB/smoltalk", "config": "all",
     "split": "train", "field": "messages", "kind": "chat_messages", "weight": 0.04},
    {"name": "dolly", "hf": "databricks/databricks-dolly-15k", "config": None,
     "split": "train", "field": None, "kind": "chat_dolly", "weight": 0.005},
]

EVAL_HASH_MOD = 500      # hold out ~1/500 of ultrachat for chat eval (no leakage)
EVAL_MAX_CONV = 2000     # cap held-out conversations


def chat_from_messages(msgs):
    """Serialize a [{role,content}] conversation into ChatML-lite text."""
    out = []
    for m in msgs:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            out.append("<|system|>\n" + content)
        elif role == "user":
            out.append("<|user|>\n" + content)
        elif role == "assistant":
            out.append("<|assistant|>\n" + content + "<|end|>")
    return "\n".join(out)


def chat_from_dolly(row):
    instr = (row.get("instruction") or "").strip()
    ctx = (row.get("context") or "").strip()
    resp = (row.get("response") or "").strip()
    if not instr or not resp:
        return ""
    user = instr + ("\n\n" + ctx if ctx else "")
    return "<|user|>\n" + user + "\n<|assistant|>\n" + resp + "<|end|>"


def conv_key(text):
    return int(hashlib.md5(text.encode("utf-8", "ignore")).hexdigest(), 16)


def serialize(src, row):
    """Return (train_text_or_None, eval_text_or_None). eval_text is set only for
    held-out ultrachat conversations (and excluded from train)."""
    k = src["kind"]
    if k == "text":
        t = row.get(src["field"])
        return (t if t else None), None
    if k == "chat_dolly":
        return (chat_from_dolly(row) or None), None
    if k == "chat_messages":
        msgs = row.get("messages") or []
        if not msgs:
            return None, None
        # held-out chat eval: only from ultrachat, deterministic by hash
        if src["name"] == "ultrachat":
            first_user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            if conv_key(first_user) % EVAL_HASH_MOD == 0:
                return None, chat_from_messages(msgs)   # eval, exclude from train
        return chat_from_messages(msgs), None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-tokens", type=int, default=4_500_000_000)
    ap.add_argument("--train-tokens", type=int, default=6_000_000_000,
                    help="training budget (the sampler re-samples the pool to reach it)")
    ap.add_argument("--eval-tokens", type=int, default=1_500_000)
    ap.add_argument("--out-dir", default="/kaggle/working")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.model_max_length = 10 ** 9
    vocab = int(tok.vocab_size)
    assert vocab < 65536, "uint16 requires vocab < 65536"

    train_path = os.path.join(args.out_dir, "tokens.bin")
    eval_path = os.path.join(args.out_dir, "chat_eval.bin")
    meta_path = os.path.join(args.out_dir, "meta.json")

    from datasets import load_dataset

    # ---- per-source quotas + streaming iterators ----
    quota = {s["name"]: int(round(s["weight"] * args.pool_tokens)) for s in SOURCES}
    written = {s["name"]: 0 for s in SOURCES}
    iters = {}
    for s in SOURCES:
        ds = load_dataset(s["hf"], s["config"], split=s["split"], streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
        iters[s["name"]] = iter(ds)
    active = [s["name"] for s in SOURCES]

    train_arr = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(args.pool_tokens,))
    cur = 0
    # eval written separately (small)
    eval_ids = []
    eval_convs = 0

    def emit(arr_list, toks):
        arr_list.extend(toks)

    print(f"  blend: pool {args.pool_tokens:,} tokens across {len(SOURCES)} sources", flush=True)
    t0 = time.time()
    src_by_name = {s["name"]: s for s in SOURCES}
    docbuf = []   # batched docs to tokenize: list of strings (train)
    BATCH = 1000

    def flush_train():
        nonlocal cur
        if not docbuf:
            return
        encs = tok(docbuf, add_special_tokens=False)["input_ids"]
        docbuf.clear()
        flat = []
        for e in encs:
            flat.extend(e)
            flat.append(0)            # doc separator (Pythia EOS)
        if flat:
            m = min(len(flat), args.pool_tokens - cur)
            if m > 0:
                train_arr[cur:cur + m] = np.asarray(flat[:m], dtype=np.uint16)
                cur += m

    rr = 0
    while active and cur < args.pool_tokens:
        name = active[rr % len(active)]
        rr += 1
        s = src_by_name[name]
        if written[name] >= quota[name]:
            active.remove(name)
            continue
        try:
            row = next(iters[name])
        except StopIteration:
            active.remove(name)
            continue
        train_text, eval_text = serialize(s, row)
        if eval_text is not None and eval_convs < EVAL_MAX_CONV:
            ids = tok(eval_text, add_special_tokens=False)["input_ids"]
            eval_ids.extend(ids); eval_ids.append(0)
            eval_convs += 1
            continue
        if not train_text:
            continue
        # rough token accounting via the batched flush; approximate per-doc here
        docbuf.append(train_text)
        # estimate tokens to update quota (cheap len proxy; exactness from flush clamp)
        approx = max(1, len(train_text) // 4)
        written[name] += approx
        if len(docbuf) >= BATCH:
            flush_train()
            if (cur // 50_000_000) != ((cur - 1) // 50_000_000):
                dt = time.time() - t0
                print(f"    {cur:,}/{args.pool_tokens:,} tokens  ({dt:.0f}s, {cur/max(dt,1):,.0f} tok/s)", flush=True)
    flush_train()
    train_arr.flush()
    del train_arr
    real_pool = cur
    print(f"  blend: built {real_pool:,} pool tokens in {time.time()-t0:.0f}s", flush=True)

    # eval bin
    eval_ids = eval_ids[: args.eval_tokens]
    np.asarray(eval_ids, dtype=np.uint16).tofile(eval_path)
    print(f"  blend: {len(eval_ids):,} held-out chat-eval tokens ({eval_convs} convs) -> {eval_path}", flush=True)

    meta = {
        "model_id": MODEL_ID,
        "vocab": vocab,
        "dtype": "uint16",
        "train_path": train_path,
        "train_tokens": int(real_pool),     # the brain re-samples this pool to its --train-tokens budget
        "test_path": eval_path,
        "test_tokens": int(len(eval_ids)),
        "eval_window": args.window,
        "eval_stride": args.stride,
        "doc_separator_id": 0,
        "build_seed": args.seed,
        "pool_tokens": int(real_pool),
        "train_budget": int(args.train_tokens),
        "sources": [{"name": s["name"], "hf": s["hf"], "weight": s["weight"],
                      "tokens": written[s["name"]]} for s in SOURCES],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  blend: wrote {meta_path}")
    print(json.dumps({k: meta[k] for k in ("train_tokens", "test_tokens", "vocab")}, indent=2))
    print("\n  Train with:  ./brain --meta", meta_path,
          "--train-tokens", args.train_tokens, "--n-gpus 2 --n-classes 256 --n-layers 4 "
          "--multiscale 1 --neg-word 8 --adapt 1")


if __name__ == "__main__":
    main()
