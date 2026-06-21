#!/usr/bin/env python3
"""Detokenize the brain's generated token ids into readable text.

    python detok.py --meta /kaggle/working/meta.json --gen /kaggle/working/brain_gen.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--gen", default="/kaggle/working/brain_gen.json")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    g = json.load(open(args.gen))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["model_id"])

    seed = tok.decode(g.get("seed", []))
    gen = tok.decode(g.get("tokens", []))
    print("=" * 70)
    print(f"  GENERATION  (temp={g.get('temp','?')})")
    print("=" * 70)
    print(f"\n  seed: \033[90m{seed!r}\033[0m\n")
    print(f"  brain continues:\n\n  \033[1;35m{gen}\033[0m\n")
    print("=" * 70)
    print("  (at high perplexity this is locally-plausible word flow, not global")
    print("   coherence yet -- the baseline to watch improve.)")


if __name__ == "__main__":
    main()
