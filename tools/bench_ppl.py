#!/usr/bin/env python3
"""Zell: strided-window perplexity benchmark on the WikiText-103 test bin.

Scores any set of HF model ids or local checkpoints with the SAME protocol that
gives Pythia-410M 17.19 (window 512, stride 256, no leakage between windows), so
results are directly comparable. Use it for the baselines and for the trained
Zell core.

    python tools/bench_ppl.py \
        --test-bin /kaggle/working/wikitext103_test_Qwen_Qwen2.5-0.5B-Instruct.bin \
        --dtype uint32 \
        --models EleutherAI/pythia-410m,Qwen/Qwen2.5-0.5B-Instruct,\
HuggingFaceTB/SmolLM2-1.7B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0,\
/kaggle/working/zell-core \
        --out /kaggle/working/scoreboard.json

Each model is tokenized-agnostic only if the test bin is in ITS tokenizer; for a
fair cross-tokenizer comparison the standard is to score each model on a test
bin built in its own tokenizer (per-token ppl is tokenizer-dependent). For the
Zell-vs-Qwen comparison both use the Qwen tokenizer, so that pair is exact;
Pythia/TinyLlama/SmolLM2 numbers are indicative and flagged as such.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

NP_DTYPE = {"uint16": np.uint16, "uint32": np.uint32}


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
    ap.add_argument("--test-bin", required=True)
    ap.add_argument("--dtype", default="uint32", choices=list(NP_DTYPE))
    ap.add_argument("--models", required=True, help="comma-separated HF ids or local paths")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = use all test tokens")
    ap.add_argument("--out", default="/kaggle/working/scoreboard.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM

    raw = np.fromfile(args.test_bin, dtype=NP_DTYPE[args.dtype]).astype(np.int64)
    if args.max_tokens:
        raw = raw[: args.max_tokens]
    ids = torch.from_numpy(raw)
    print(f"  test bin: {ids.numel():,} tokens ({args.dtype})", flush=True)

    rows = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16)
            model = model.to(device)
            win = min(args.window,
                      int(getattr(model.config, "max_position_embeddings", args.window)))
            ppl, ntok, secs = strided_ppl(model, ids, win, args.stride, device)
            params = sum(p.numel() for p in model.parameters())
            rows.append({"model": mid, "ppl": round(ppl, 3), "params": int(params),
                         "window": win, "stride": args.stride, "ntok": ntok,
                         "seconds": round(secs, 1)})
            print(f"  {mid:55s} ppl {ppl:8.3f}  {params:,} params  ({secs:.1f}s)", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # one bad model should not sink the board
            print(f"  {mid:55s} FAILED: {e}", flush=True)
            rows.append({"model": mid, "ppl": None, "error": str(e)})

    rows_ok = sorted([r for r in rows if r.get("ppl") is not None], key=lambda r: r["ppl"])
    board = {"reference": {"Pythia-410M (design-of-record)": 17.19},
             "window": args.window, "stride": args.stride, "results": rows_ok + [r for r in rows if r.get("ppl") is None]}
    with open(args.out, "w") as f:
        json.dump(board, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
