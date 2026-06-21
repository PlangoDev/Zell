#!/usr/bin/env python3
"""Run the pretrained LLM opponent's strided-window perplexity on the WikiText-103
test .bin (the same stream the brain is scored on) and write llm_result.json.

This is the ONLY place torch/transformers run in v3 — the brain is pure C/CUDA.

    python run_llm.py --meta /kaggle/working/meta.json --out /kaggle/working/llm_result.json
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def strided_ppl(model, ids, window, stride, device):
    model.eval()
    ids = ids.long().to(device)
    n = ids.numel()
    nll = torch.zeros((), device=device)
    ntok = torch.zeros((), dtype=torch.long, device=device)
    prev_end = 0
    t0 = time.time()
    for begin in range(0, n - 1, stride):
        end = min(begin + window, n)
        inp = ids[begin:end].unsqueeze(0)
        tgt = inp.clone()
        new = end - prev_end
        tgt[:, :-new] = -100
        logits = model(inp).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               tgt[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        nll += loss
        ntok += (tgt[:, 1:] != -100).sum()
        prev_end = end
        if end == n:
            break
    nt = int(ntok.item())
    return math.exp(nll.item() / max(nt, 1)), nt, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="/kaggle/working/llm_result.json")
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(meta["model_id"])
    model = base.to(device).half() if device == "cuda" else base

    ids = np.fromfile(meta["test_path"], dtype=np.uint16).astype(np.int64)
    ids = torch.from_numpy(ids[: meta["test_tokens"]])
    win = min(meta["eval_window"],
              int(getattr(model.config, "max_position_embeddings", meta["eval_window"])))
    ppl, ntok, secs = strided_ppl(model, ids, win, meta["eval_stride"], device)
    macs = sum(p.numel() for p in model.parameters())
    out = {"llm_ppl": ppl, "llm_macs": int(macs), "window": win,
           "stride": meta["eval_stride"], "ntok": ntok, "seconds": secs}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  LLM ({meta['model_id']}) strided ppl {ppl:.2f}  {macs:,} params  ({secs:.1f}s)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
