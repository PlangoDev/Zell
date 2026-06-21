#!/usr/bin/env python3
"""Merge brain_result.json (from the C/CUDA binary) and llm_result.json (from
run_llm.py) into the head-to-head scoreboard.

    python scoreboard.py --brain brain_result.json --llm llm_result.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="/kaggle/working/brain_result.json")
    ap.add_argument("--llm", default="/kaggle/working/llm_result.json")
    args = ap.parse_args()

    with open(args.brain) as f:
        b = json.load(f)
    with open(args.llm) as f:
        l = json.load(f)

    print("\n" + "=" * 70)
    print("  SCOREBOARD   (WikiText-103 test, same tokenizer/stream)")
    print("=" * 70)
    print(f"    {'system':<28}{'perplexity':>12}{'MACs/token':>16}{'backprop':>10}")
    print(f"    {'LLM (pretrained)':<28}{l['llm_ppl']:>12.2f}{l['llm_macs']:>16,}{'yes':>10}")
    consumed_m = b['consumed_tokens'] // 1_000_000
    print(f"    {'Brain ('+str(consumed_m)+'M tok, no-bp)':<28}"
          f"{b['brain_ppl']:>12.2f}{b['macs_per_token']:>16,}{'NO':>10}")
    print(f"    {'-'*64}")
    for i, (p, w) in enumerate(zip(b.get("per_layer_ppl", []), b.get("mix", []))):
        print(f"      layer {i} ppl {p:>10.2f}   w={w:.3f}")
    print(f"    {'-'*64}")
    if l['llm_macs']:
        print(f"    brain runs at {b['macs_per_token']/l['llm_macs']:.5f}x the LLM's compute/token.")
    print(f"    brain trained at {b['tok_per_sec']:,.0f} tok/s "
          f"({b['consumed_tokens']:,} tok in {b['train_seconds']:.0f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
