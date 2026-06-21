# Infini-gram Retrieval and Interpolation

## Scope

This document records the suffix-array infini-gram component of BrainFormer: its construction, the longest-suffix backoff query, the smoothing estimate, the Python-versus-C parity validation, the evaluation-time interpolation `P = (1-lambda)*P_brain + lambda*P_ngram` for static and adaptive scoring in a single pass, the lambda sweep, the measured perplexity results, and the bounded distributional scope of the gain. The tokenizer-alignment requirement that governs reuse of the index in the Zell hybrid is stated last.

Source files of record:

- `test_013_v3/src/ngram.h` — host query: mmap, longest-suffix binary search, smoothed estimate.
- `test_013_v3/tools/build_ngram.py` — suffix-array builder (pydivsufsort with a vectorized numpy prefix-doubling fallback), reference query, and self-test.

Relevant commits: `25aedc1`, `c30462a`, `ef28dd5`.

## Motivation

The no-backprop cerebellar model reaches the low thousands of perplexity on WikiText-103 and is structurally barred from approaching the backprop baseline (EleutherAI/Pythia-410M, strided WikiText-103 perplexity 17.19). The infini-gram is a non-parametric retrieval distribution placed beside the model at inference time. It contributes nothing to training and adds no learned parameters. On text that resembles the indexed corpus it supplies a sharp, low-entropy next-token distribution drawn from exact long-context matches; the model and the infini-gram are combined by linear interpolation of probabilities. The infini-gram produced the single largest realized quality jump in the project.

An infini-gram is an unbounded-order n-gram. Rather than fixing an order n, it answers each query from the **longest suffix of the current context that still occurs at least once in the corpus**. A suffix array over the token corpus makes this longest-suffix lookup a pair of binary searches.

## Index construction (`build_ngram.py`)

### Inputs and outputs

The builder consumes a `uint16` token corpus — either a raw `.bin` (`--tokens`) or the `train_path` named by a `meta.json` (`--meta`). It writes two artifacts alongside the token file:

| Artifact | Contents |
| --- | --- |
| `<bin>.sa` | `int64` suffix array. `SA[i]` is the start offset of the i-th lexicographically smallest suffix. |
| `<bin>.sa.json` | metadata `{n_tokens, token_path, sa_dtype, vocab, max_match}`. |

`--max-tokens` caps corpus size for memory control; the suffix array costs 8 bytes per token (`int64`). At 50M tokens the `.sa` file is approximately 0.4 GB.

### Two construction backends

`build_sa(tokens)` (`build_ngram.py:64`) selects a backend at runtime:

```python
def build_sa(tokens):
    try:
        import pydivsufsort
        sa = pydivsufsort.divsufsort(tokens)        # O(n), C extension
        return np.asarray(sa, dtype=np.int64)
    except Exception as e:
        if len(tokens) > 50_000_000:
            raise                                    # fallback too slow at this size
        return suffix_array_prefix_doubling(tokens)  # pure-numpy, no internet
```

1. **pydivsufsort** — the SA-IS / divsufsort C extension, O(n), pip-installable. It operates directly on the integer numpy array (`uint16` is accepted). This is the path used for full-scale builds and requires network access to install.
2. **Vectorized numpy prefix-doubling** — the offline / small-corpus fallback, used when pydivsufsort is unavailable and the corpus is at or below 50M tokens. Above 50M tokens the fallback is judged too slow and the builder raises with an instruction to install pydivsufsort.

### Vectorized prefix-doubling

`suffix_array_prefix_doubling(t)` (`build_ngram.py:32`) is an O(n log n) suffix array with no Python-level loop over positions. Each doubling round is a `lexsort` plus a `cumsum` rank recomputation:

```python
_, inv = np.unique(t, return_inverse=True)
rank = inv.astype(np.int64)
k = 1
while True:
    second = np.full(n, -1, dtype=np.int64)
    if k < n:
        second[:n - k] = rank[k:]
    sa = np.lexsort((second, rank))            # primary rank, secondary "second"
    ra = rank[sa]; rb = second[sa]
    changed = np.empty(n, dtype=np.int64)
    changed[0] = 0
    changed[1:] = (ra[1:] != ra[:-1]) | (rb[1:] != rb[:-1])
    rank_new = np.empty(n, dtype=np.int64)
    rank_new[sa] = np.cumsum(changed)          # dense rank in SA order, scatter back
    rank = rank_new
    if rank[sa[-1]] == n - 1:                   # all suffixes distinct -> done
        break
    k <<= 1
    if k >= n:
        break
```

The construction:

- Compacts the alphabet with `np.unique(..., return_inverse=True)` so the initial rank is the dense order of the raw token id.
- At each round forms a key pair `(rank[i], rank[i+k])`, where the second component is a sentinel `-1` for positions whose `i+k` falls off the end (a shorter suffix sorts first).
- Sorts by the pair with `lexsort`, recomputes dense ranks by flagging adjacent key-pair changes and taking a `cumsum`, then scatters the new ranks back to text order.
- Terminates when every suffix has a distinct rank (`rank[sa[-1]] == n-1`) or when the doubling window `k` reaches `n`.

Per the builder's docstring, this fully vectorized form builds approximately 50M tokens in a couple of minutes in pure numpy, with no native dependency and no internet.

### Metadata sidecar

The builder writes `max_match: 16` into `<bin>.sa.json`. The host reader (below) reads `max_match` and `vocab` from this sidecar; `max_match` is accepted only in the range `(0, 64]`.

## Host query (`ngram.h`)

The query path is pure host (CPU) code using POSIX `mmap`, matching `data.c`. Evaluation is not throughput-critical, so a CPU suffix-array probe per position is acceptable.

### Mapping and open

`ngram_open(token_path, vocab_hint)` (`ngram.h:50`) mmaps `<token_path>` and `<token_path>.sa` read-only (`PROT_READ`, `MAP_PRIVATE`). It derives:

- `n` = token count, taken as the smaller of `sa_bytes/8` and `toks_bytes/2` (guards a token file longer than the SA).
- `max_match` = 16 by default, overridden from the `.sa.json` sidecar (clamped to `(0, 64]`).
- `vocab` = `vocab_hint` if positive, else the sidecar value, else 50277 (the Pythia tokenizer vocabulary).
- `scan_cap` = 8192.

The `Ngram` struct (`ngram.h:25`):

```c
typedef struct {
    const uint16_t *toks;   /* mmapped corpus tokens [n]   */
    const int64_t  *sa;     /* mmapped suffix array [n]     */
    long n;                 /* number of tokens            */
    int  max_match;         /* cap on suffix length probed */
    int  vocab;             /* smoothing denominator hint  */
    int  scan_cap;          /* max corpus positions per query */
    size_t toks_bytes, sa_bytes;
} Ngram;
```

### Suffix comparison and bounds

`ng_cmp(toks, n, start, pat, plen)` (`ngram.h:90`) compares the suffix beginning at `start` against the pattern `pat[0..plen)`. It returns `-1` if the suffix sorts before the pattern (including the case where the suffix runs off the end of the corpus before `plen` tokens, which makes it smaller), `0` if the pattern is a prefix of the suffix, and `1` otherwise.

`ng_lb(g, pat, plen, upper)` (`ngram.h:101`) is a standard lower-bound binary search over SA indices. With `upper=0` it returns the first SA index whose suffix is `>= pat`; with `upper=1` it returns the first index strictly past the block of suffixes that have `pat` as a prefix. The half-open interval `[lo, hi)` from the two calls is exactly the set of corpus positions whose suffix begins with `pat`, and `hi - lo` is the occurrence count of `pat`.

### Longest-suffix backoff and the estimate

`ngram_prob(g, ctx, ctxlen, y)` (`ngram.h:114`) returns the infini-gram probability of next token `y` given the context:

```c
int Lmax = ctxlen < g->max_match ? ctxlen : g->max_match;
for (int ln = Lmax; ln >= 1; ln--) {
    const int *pat = ctx + (ctxlen - ln);     /* the last ln tokens of ctx */
    long lo = ng_lb(g, pat, ln, 0);
    long hi = ng_lb(g, pat, ln, 1);
    long cnt = hi - lo;
    if (cnt < 1) continue;                     /* this suffix never occurs; back off */
    long step = 1;
    if (cnt > g->scan_cap) step = cnt / g->scan_cap;   /* strided subsample */
    long yes = 0, tot = 0;
    for (long i = lo; i < hi; i += step) {
        long p = g->sa[i] + ln;
        if (p < g->n) { tot++; if ((int)g->toks[p] == y) yes++; }
    }
    if (tot > 0)
        return ((double)yes + 1e-6) / ((double)tot + 1e-6 * 65536.0);
}
return 1.0 / 65536.0;                          /* nothing matched: uniform */
```

Mechanism, step by step:

1. **Backoff order.** Probe suffix lengths from `Lmax = min(ctxlen, max_match)` down to 1. The first length that occurs in the corpus is used; shorter suffixes are not consulted once a match is found. This is the infini-gram principle: trust the longest match.
2. **Match block.** For suffix length `ln`, the pattern is the last `ln` context tokens. The two bound searches give the SA block `[lo, hi)` of all corpus positions whose forward window equals that pattern. `cnt = hi - lo` is the raw occurrence count.
3. **Strided subsample.** When `cnt > scan_cap` (8192) the block is sampled with stride `step = cnt / scan_cap`, bounding the per-query scan regardless of how common the suffix is. For matches at or below 8192 occurrences every position is scanned.
4. **Next-token tally.** For each scanned occurrence at SA position `i`, the following corpus token is `toks[sa[i] + ln]`. Positions whose successor index `sa[i] + ln` falls off the end are skipped (`tot` is not incremented). `yes` counts successors equal to `y`; `tot` counts valid successors scanned.
5. **Smoothed estimate.** When at least one valid successor was seen,

   ```
   P_ngram(y | ctx) = (yes + 1e-6) / (tot + 1e-6 * 65536)
   ```

   This is a Laplace-style estimate with a vanishing pseudocount `1e-6` per type over a fixed 65536-symbol denominator. The floor `1e-6` is small relative to integer counts, so the estimate is dominated by the empirical ratio `yes/tot`; the floor only prevents an exact zero so the interpolated distribution stays positive. The choice to weight the longest match directly, rather than recursively interpolate across backoff orders, is deliberate (`ngram.h:129`): "trust the longest match (epsilon floor only)."
6. **No match.** If no suffix length down to 1 occurs (the final context token itself is unseen in the corpus), return the uniform `1/65536`.

The denominator constant `65536` corresponds to the `uint16` symbol space, not the model's true vocabulary; it is a fixed smoothing constant rather than a calibrated normalizer. It is identical on both the C and Python sides, which is what the parity test depends on.

## Python-versus-C parity validation

The query exists twice: the production C path in `ngram.h` and a reference implementation `ngram_prob_ref` in `build_ngram.py:79`. Both are written to the same arithmetic so the C result can be validated byte-for-byte against the Python reference.

The reference mirrors the C exactly:

```python
def ngram_prob_ref(tokens, sa, context, y, max_match=16, min_count=1):
    n = len(tokens)
    L = min(len(context), max_match)
    for ln in range(L, 0, -1):
        pat = context[-ln:]
        lo, hi = sa_range(tokens, sa, pat)
        cnt = hi - lo
        if cnt >= min_count:
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
```

Correspondences that make the two numerically identical on matched ranges:

| Concern | C (`ngram.h`) | Python (`build_ngram.py`) |
| --- | --- | --- |
| Backoff order | `for ln = Lmax..1` | `for ln in range(L, 0, -1)` |
| Bounds search | `ng_lb` lower/upper | `_lower_bound` / `sa_range` |
| Suffix compare | `ng_cmp` (off-end => smaller) | `_cmp_suffix_pat` (off-end => `-1`) |
| Successor index | `sa[i] + ln`, skip if `>= n` | `sa[i] + ln`, skip if `>= n` |
| Estimate | `(yes+1e-6)/(tot+1e-6*65536)` | `(yes+1e-6)/(tot+1e-6*65536)` |
| Empty fallback | `1.0/65536` | `1.0/65536` |

The deliberate non-parity is the subsampling: the C path caps the scan at `scan_cap = 8192` with a stride, while the reference scans the full `[lo, hi)`. For any match block at or below 8192 occurrences the two scan identical positions and return bit-identical estimates; above that cap the C path computes the estimate on a strided sample of the same block. The reference is therefore the ground truth for small and moderate match blocks, which is the regime the self-test exercises.

`--selftest` (`build_ngram.py:134`) builds an SA over the byte stream `"abracadabra abracadabra mississippi"` and asserts:

- The SA is a permutation of `range(n)` and is lexicographically sorted (each suffix `<=` the next).
- `P("b" | "a") > P("z" | "a")` — after `a`, `b` (from `abra`) is more likely than an unseen symbol.
- `P("c" | "abra")` is recovered for the longer context.

This validates the builder (both backends), the bounds search, the backoff loop, and the estimate arithmetic against hand-checkable cases.

## Evaluation interpolation

At evaluation the infini-gram distribution is mixed with the model's mixture distribution per position:

```
P(y | ctx) = (1 - lambda) * P_brain(y | ctx) + lambda * P_ngram(y | ctx)
```

`P_brain` is the learned probabilistic mixture over the deep cerebellar layers; `P_ngram` is `ngram_prob`. Properties of the interpolation:

- It is a convex combination of two proper distributions, so the result is a proper distribution for any `lambda` in `[0, 1]`.
- When no suffix matches, `P_ngram` is uniform (`1/65536`) and contributes a near-flat term that only mildly flattens `P_brain`. The gain therefore concentrates on positions with a real long-suffix match.
- No leakage: the infini-gram is built over the training corpus and queried with the eval context; the scored next token `y` is read from the eval stream, not from the index.

### Static and adaptive in one pass

The eval scores both the static and the adaptive (test-time-adaptation, `--adapt`) variants of the model in a single pass over the documents, and interpolates both with the same infini-gram. Adaptive scoring continues to apply the local delta rule during evaluation on the document being read (predict-then-learn; the token is scored before the update, so there is still no leakage). The infini-gram term is identical for both variants at a given position; only `P_brain` differs (frozen weights versus continuously adapted weights). Computing both in the same pass shares the suffix-array queries and avoids re-reading the corpus.

### Lambda sweep

The eval sweeps `lambda` over `{0.1, 0.2, ..., 0.9}` and reports perplexity at each value for both static and adaptive scoring. The selected operating point is `lambda = 0.3` (also the value printed by the builder's usage hint, `build_ngram.py:194`).

Flags: `--ngram <bin>` supplies the token file whose `.sa` index is loaded; `--ngram-lambda <L>` sets the mixing weight (`config.h`).

## Measured results

Configuration: Kaggle dual NVIDIA T4, WikiText-103, 4-layer multiscale model, 50M-token suffix-array index, `lambda = 0.3`. Evaluation is deterministic.

| Scoring | Without infini-gram | With infini-gram (lambda=0.3) | Relative change |
| --- | --- | --- | --- |
| Static (frozen) | 3412 | 1272 | -63% |
| Adaptive (`--adapt`) | 1930 | 762 | -60% |

This is the largest single quality improvement recorded in the program. The two inference-time augmentations stack: from the static frozen model at 3412, test-time adaptation alone reaches 1930 (-43%), and adding the infini-gram on top of the adaptive model reaches 762.

## Scope and limitations

The gain is realized only on corpus-like, in-distribution text where a long suffix of the context occurs in the index. The backoff query returns a sharp, low-entropy distribution exactly when a long match exists; for novel input — open-ended conversation, text unlike the indexed corpus — no long suffix matches, the query backs off toward very short suffixes or to the uniform `1/65536` fallback, and the interpolation reduces to the bare model. The 762 perplexity therefore does **not** transfer to conversation; it is a measurement on WikiText-style in-distribution text.

Further limitations recorded for completeness:

- **Single-document semantics, not just matching.** The estimate trusts the longest match directly with only an epsilon floor; there is no recursive backoff smoothing across orders and no discounting. On rare long suffixes the next-token distribution can be sharp from a small sample.
- **Subsample bias at scale.** Above `scan_cap = 8192` occurrences the C estimate is computed on a strided sample; for very common short suffixes this introduces a small sampling error relative to the exact Python reference. This regime is the least informative one (short, high-count suffixes contribute the most diffuse distributions), so the practical effect is minor, but the C and reference numbers are not bit-identical there.
- **Host POSIX dependency.** `ngram.h` uses `mmap`/`open`/`fstat` and is Linux/POSIX-only, matching the Kaggle eval environment. It is not a Windows path.
- **Fixed smoothing denominator.** The `65536` denominator is the `uint16` symbol-space constant, not the model vocabulary; it is a smoothing constant shared by both implementations rather than a calibrated normalizer.

## Tokenizer-alignment requirement for reuse in Zell

The index is built over a specific token stream and queried with context in that same token id space. The suffix array, the `uint16` token corpus, the smoothing denominator, and the eval context must all be in one tokenizer. The BrainFormer index is built in the Pythia-410m tokenizer.

For the Zell hybrid the coherence core is a backprop-trained efficient transformer using the Qwen2.5 BPE tokenizer (approximately 151k vocabulary). Reusing infini-gram interpolation there requires rebuilding the index in the core's tokenizer: re-tokenize the corpus to the Qwen ids, rebuild the `.sa` over that stream, and query with Qwen-tokenized context. A Pythia-tokenized suffix array cannot be queried with Qwen-tokenized context — the integer ids do not correspond. The carryover is the tool (`build_ngram.py`, re-tokenized) and the query design (`ngram.h`), not the existing `.sa` artifact. In the Zell plan this is milestone M4: the infini-gram interpolation rebuilt in the Qwen tokenizer with an adaptive lambda. The honest-scope caveat carries over unchanged: the interpolation gain is real on in-distribution text and near-zero on novel chat input.
