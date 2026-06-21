#!/usr/bin/env python3
"""Tokenize the training corpus + the WikiText-103 test split into uint16 .bin
files the C/CUDA brain mmaps, and write meta.json describing them.

This is the ONLY data-prep step; the C binary never tokenizes. Reuses the v13/v2
streaming-memmap approach. Run on Kaggle with Internet ON.

    python build_data.py --model EleutherAI/pythia-410m \
        --train-tokens 50000000 --out-dir /kaggle/working
"""
import argparse
import json
import os
import time

import numpy as np


def build_train(tok, corpus, corpus_config, text_key, budget, path):
    from datasets import load_dataset
    print(f"  data: building train memmap {path} ({budget:,} tokens, uint16)", flush=True)
    t0 = time.time()
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(budget,))
    cur = 0
    docs = []

    def flush():
        nonlocal cur
        if not docs:
            return
        enc = tok(docs, add_special_tokens=False)["input_ids"]
        docs.clear()
        flat = []
        for e in enc:
            flat.extend(e)
        if flat:
            m = min(len(flat), budget - cur)
            if m > 0:
                arr[cur:cur + m] = np.asarray(flat[:m], dtype=np.uint16)
                cur += m

    ds = load_dataset(corpus, corpus_config, split="train", streaming=True)
    for row in ds:
        t = row.get(text_key)
        if t:
            docs.append(t)
        if len(docs) >= 2000:
            flush()
        if cur >= budget:
            break
    if cur < budget:
        flush()
    arr.flush()
    del arr
    dt = time.time() - t0
    print(f"  data: built {cur:,} train tokens in {dt:.0f}s ({cur/max(dt,1):,.0f} tok/s)", flush=True)
    return int(cur)


def build_test(tok, max_tokens, path):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    ids = []
    for row in ds:
        if row["text"]:
            ids.extend(tok(row["text"], add_special_tokens=False)["input_ids"])
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    np.asarray(ids, dtype=np.uint16).tofile(path)
    print(f"  data: built {len(ids):,} WikiText-103 test tokens -> {path}", flush=True)
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--corpus", default="wikimedia/wikipedia")
    ap.add_argument("--corpus-config", default="20231101.en")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--train-tokens", type=int, default=50_000_000)
    ap.add_argument("--eval-tokens", type=int, default=250_000)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--out-dir", default="/kaggle/working")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = 10 ** 9
    vocab = int(tok.vocab_size)

    safe = args.model.replace("/", "_")
    train_path = os.path.join(args.out_dir, f"wiki_train_{safe}_{args.train_tokens}.bin")
    test_path = os.path.join(args.out_dir, f"wiki_test_{safe}.bin")
    meta_path = os.path.join(args.out_dir, "meta.json")

    train_tokens = args.train_tokens
    if not os.path.exists(train_path):
        train_tokens = build_train(tok, args.corpus, args.corpus_config, args.text_key,
                                   args.train_tokens, train_path)
    else:
        train_tokens = os.path.getsize(train_path) // 2
        print(f"  data: reusing {train_path} ({train_tokens:,} tokens)")

    if not os.path.exists(test_path):
        test_tokens = build_test(tok, args.eval_tokens, test_path)
    else:
        test_tokens = os.path.getsize(test_path) // 2
        print(f"  data: reusing {test_path} ({test_tokens:,} tokens)")

    meta = {
        "model_id": args.model,
        "vocab": vocab,
        "dtype": "uint16",
        "train_path": train_path,
        "train_tokens": int(train_tokens),
        "test_path": test_path,
        "test_tokens": int(test_tokens),
        "eval_window": args.window,
        "eval_stride": args.stride,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  data: wrote {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
