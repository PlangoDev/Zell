#!/usr/bin/env python3
"""BrainFormer v15 — infini-gram suffix-array builder.

Builds a suffix array over a uint16 token corpus so the C eval can compute, for
any context, the next-token distribution from the LONGEST matching suffix in the
corpus (unbounded n-gram / infini-gram). Interpolating this with the brain's
distribution is the single highest-EV quality lever in the v15 research
(web-verified ~42% relative ppl cut on GPT-2 when mixed with a neural LM).

Output (alongside the token .bin):
    <bin>.sa      int64 suffix array, SA[i] = start offset of the i-th smallest suffix
    <bin>.sa.json meta {n_tokens, token_path, sa_dtype, vocab}

The brain then trains/evaluates as usual and adds  --ngram <bin>  --ngram-lambda L.

Construction uses pydivsufsort (O(n), pip-installable, internet-on builder) with a
correct pure-numpy prefix-doubling fallback for small corpora / offline tests.

Usage:
    python build_ngram.py --tokens /kaggle/working/tokens.bin            # from a raw .bin
    python build_ngram.py --meta   /kaggle/working/meta.json             # uses meta.train_path
    python build_ngram.py --selftest                                     # local correctness check
"""
import argparse
import json
import os
import sys

import numpy as np


def suffix_array_prefix_doubling(t):
    """O(n log n) suffix array via prefix doubling, FULLY VECTORIZED (no Python loop):
    each doubling is a lexsort + a cumsum rank-recompute. Builds ~50M tokens in a
    couple minutes in pure numpy — no pydivsufsort / no internet needed. Returns int64 SA."""
    n = len(t)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    _, inv = np.unique(t, return_inverse=True)
    rank = inv.astype(np.int64)
    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        if k < n:
            second[:n - k] = rank[k:]
        sa = np.lexsort((second, rank))            # primary rank, secondary second
        ra = rank[sa]; rb = second[sa]
        changed = np.empty(n, dtype=np.int64)
        changed[0] = 0
        changed[1:] = (ra[1:] != ra[:-1]) | (rb[1:] != rb[:-1])
        rank_new = np.empty(n, dtype=np.int64)
        rank_new[sa] = np.cumsum(changed)          # dense rank in SA order -> scatter back
        rank = rank_new
        if rank[sa[-1]] == n - 1:                   # all suffixes distinct -> done
            break
        k <<= 1
        if k >= n:
            break
    return sa.astype(np.int64)


def build_sa(tokens):
    try:
        import pydivsufsort  # type: ignore
        # pydivsufsort works on integer numpy arrays; uint16 is fine.
        sa = pydivsufsort.divsufsort(tokens)
        return np.asarray(sa, dtype=np.int64)
    except Exception as e:  # noqa: BLE001
        if len(tokens) > 50_000_000:
            print(f"  ngram: pydivsufsort unavailable ({e}); the numpy fallback is too "
                  f"slow for {len(tokens):,} tokens. pip install pydivsufsort.", file=sys.stderr)
            raise
        print(f"  ngram: pydivsufsort unavailable ({e}); using numpy prefix-doubling.")
        return suffix_array_prefix_doubling(tokens)


def ngram_prob_ref(tokens, sa, context, y, max_match=16, min_count=1):
    """Reference infini-gram next-token probability P(y | context) by longest-suffix
    backoff over the suffix array. Mirrors the C query for validation."""
    n = len(tokens)
    # try suffixes of `context` from longest to shortest
    L = min(len(context), max_match)
    for ln in range(L, 0, -1):
        pat = context[-ln:]
        lo, hi = sa_range(tokens, sa, pat)
        cnt = hi - lo
        if cnt >= min_count:
            # next tokens are tokens[sa[i] + ln] for i in [lo,hi) with sa[i]+ln < n
            yes = tot = 0
            for i in range(lo, hi):
                p = sa[i] + ln
                if p < n:
                    tot += 1
                    if tokens[p] == y:
                        yes += 1
            if tot > 0:
                return (yes + 1e-6) / (tot + 1e-6 * 65536)
    return 1.0 / 65536.0


def sa_range(tokens, sa, pat):
    """Return [lo,hi) range of suffixes (in SA order) that start with `pat`."""
    n = len(sa)
    lo = _lower_bound(tokens, sa, pat, False)
    hi = _lower_bound(tokens, sa, pat, True)
    return lo, hi


def _cmp_suffix_pat(tokens, start, pat):
    n = len(tokens)
    for j in range(len(pat)):
        if start + j >= n:
            return -1  # suffix shorter -> smaller
        a, b = int(tokens[start + j]), int(pat[j])
        if a != b:
            return -1 if a < b else 1
    return 0


def _lower_bound(tokens, sa, pat, upper):
    lo, hi = 0, len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        c = _cmp_suffix_pat(tokens, sa[mid], pat)
        if c < 0 or (upper and c == 0):
            lo = mid + 1
        else:
            hi = mid
    return lo


def selftest():
    # "abracadabra"-like token stream; check a few infini-gram probabilities.
    s = "abracadabra abracadabra mississippi"
    tokens = np.frombuffer(s.encode(), dtype=np.uint8).astype(np.uint16)
    sa = build_sa(tokens)
    # sanity: SA is a permutation and sorted
    assert sorted(sa.tolist()) == list(range(len(tokens)))
    for i in range(1, len(sa)):
        a = tokens[sa[i - 1]:].tolist()
        b = tokens[sa[i]:].tolist()
        assert a <= b, "SA not sorted"
    # 'a' is most often followed by 'b' (abra) in the corpus
    ctx = list(np.frombuffer(b"a", dtype=np.uint8).astype(np.uint16))
    pa_b = ngram_prob_ref(tokens, sa, ctx, ord("b"))
    pa_z = ngram_prob_ref(tokens, sa, ctx, ord("z"))
    assert pa_b > pa_z, (pa_b, pa_z)
    # longer context 'abra' -> next is 'c' or end/space; 'c' should be likely
    ctx2 = list(np.frombuffer(b"abra", dtype=np.uint8).astype(np.uint16))
    pc = ngram_prob_ref(tokens, sa, ctx2, ord("c"))
    print(f"  selftest OK: P(b|a)={pa_b:.3f} > P(z|a)={pa_z:.4f}; P(c|abra)={pc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", help="path to a uint16 token .bin")
    ap.add_argument("--meta", help="meta.json (uses train_path)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap corpus size (0 = all); SA memory ~8 bytes/token")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    vocab = 65536
    if args.meta:
        meta = json.load(open(args.meta))
        bin_path = meta["train_path"]
        vocab = int(meta.get("vocab", 65536))
    elif args.tokens:
        bin_path = args.tokens
    else:
        ap.error("need --tokens or --meta or --selftest")

    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    if args.max_tokens and len(tokens) > args.max_tokens:
        tokens = tokens[: args.max_tokens]
    tokens = np.ascontiguousarray(tokens)
    print(f"  ngram: building suffix array over {len(tokens):,} tokens from {bin_path}", flush=True)
    import time
    t0 = time.time()
    sa = build_sa(tokens)
    print(f"  ngram: SA built in {time.time()-t0:.0f}s", flush=True)

    sa_path = bin_path + ".sa"
    sa.astype(np.int64).tofile(sa_path)
    meta = {"n_tokens": int(len(tokens)), "token_path": os.path.abspath(bin_path),
            "sa_dtype": "int64", "vocab": vocab, "max_match": 16}
    json.dump(meta, open(sa_path + ".json", "w"), indent=2)
    print(f"  ngram: wrote {sa_path} ({sa.nbytes/1e9:.2f} GB) + meta")
    print(f"  Eval with:  ./brain --meta ... --ngram {bin_path} --ngram-lambda 0.3")


if __name__ == "__main__":
    main()
