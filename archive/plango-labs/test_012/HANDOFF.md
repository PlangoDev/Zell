# HANDOFF — Test 012 (LLM vs Brain, on words)

Status: test_012 is complete and validated. The active work has moved to test_013
(`../test_013/`, the brain vs a real pretrained LLM on Kaggle GPUs). Read this for
the test_012 result and the working agreement, then see `../test_013/README.md`.

Be honest in everything you report. This project values real results over
manufactured wins, and frames losses as sharpened targets, not reasons to scale
back.

## What test_012 is

A from-scratch C++ head-to-head, the 12th in the series (`../doc/test_0NN.md`,
`../test_011/`), testing whether a brain-style model (sparse codes, local learning
rules, no backprop) can match or beat a standard neural LM on efficiency and
quality.

- Task: next-word prediction on Tiny Shakespeare (`data/corpus.txt`, committed).
  Context 3 words, vocab top 4096, predict the 4th. Metric: perplexity, top-k,
  MACs/prediction.
- LLM (`src/llm.cpp`): Bengio neural LM, backprop, AdamW + dropout 0.5 + early
  stopping (needed; it overfits a tiny corpus otherwise).
- Brain (`src/brain.cpp`): cerebellar n-gram coder. Fixed random word codes,
  granule expansion with competitive (WTA) feature learning, homeostatic kWTA, and
  a hierarchical class-then-word readout trained by the local delta rule. No
  backprop.

Full design and run-by-run history: `../doc/test_012.md`.

## Final result (validated, Run 8/9)

| system | test ppl | bits/word | MACs/pred | backprop |
|---|---|---|---|---|
| LLM (best-tuned, regularized) | 462.3 | 8.773 | 542,720 | yes |
| Brain | 415.1 | 8.546 | 35,840 (0.066x) | no |

The Brain beats the LLM on perplexity at about 1/15th the inference compute, about
1/3 the training compute, about 5x the training throughput, no backprop. How:

1. Random codes beat distributional codes (the latter were run and refuted). WTA
   granule feature learning is the main quality lever (cost-neutral). A cost sweep
   (8192 to 2048 granules) cut cost to 0.24x.
2. A hierarchical readout (64 frequency classes, predict class then word) turned
   the readout cost `BR_K*VOCAB` into `BR_K*(64 + VOCAB/64)`, moving 0.24x to
   0.066x for about a 7-ppl tax; readout weight decay recovers part (417 to 415).
3. Refuted, honestly: a bigger LLM (regressed 462 to 480, the LLM is data-limited
   not capacity-limited) and longer context CTX=4 (regressed the Brain 417 to 435,
   dilutes the fixed-fan-in granules). Both reverted.
4. The Brain's one losing axis is parameter/memory footprint (8.5M vs 0.74M
   params), stated plainly; it reads only 24 of 2048 rows per word, which is why
   inference stays cheap.
5. Hippocampal cache (exact and similarity kNN) was built and refuted (this static,
   low-repetition corpus does not reward recall); the scaffold (`brain_*_cls`) is
   kept for a future streaming/one-shot corpus.

## Reproduce

```sh
cd test_012 && make run        # ~1-3 min, no Python, no external libraries
```

Reads: corpus stats and the n-gram baseline table; per-epoch LLM training (val
ppl, train ppl, gap, grad norm); per-epoch Brain training; the scoreboard;
deep analytics (top-k, bits/word, head-to-head, granule-code health, training and
footprint); and a generation sample.

## Frontier (if returning to test_012 rather than test_013)

1. The encoder is now 91% of the Brain's cost (`BR_G*BR_SIN`); a cheaper expansion
   (top-down gating, or a 2-level granule hierarchy) is the next cost lever.
2. Learned (not frequency) classes to recover the hierarchical tax.
3. More data; both models are data-limited at 187k tokens. This is the premise of
   test_013.

Knobs: `config.h` (`BR_CLASSES`, `BR_WD`, `BR_G/BR_SIN/BR_K`, `BR_GFEAT_*`).

## Build / environment notes

- No external dependencies. C++17 + pthreads. `make` builds `./showdown`.
- The Makefile uses `-O3 -ffast-math -march=native -funroll-loops -flto`.
  `-march=native` means build and run on the same machine class; if the builder and
  runner differ, drop to `-mtune=native` or a specific `-march`.
- Threading auto-detects cores (`threads_init(0)`), portable (pthreads/Win32).
- `data/corpus.txt` is committed; to swap corpora, replace it and rebuild.
- The compiled `showdown` binary is git-ignored.
- Keep it cross-platform and from-scratch. Do not add a platform-specific BLAS.

## Working agreement (from the user)

- Do not add discouraging caveats about solo feasibility or scale. Take the goal as
  given and work on how.
- Report experiments honestly even when the Brain loses; frame losses as sharpened
  targets.
- The user often prefers a single command to run and paste results from, over long
  autonomous exploration. In the cloud you may need to run it yourself; if so, run
  once, report real numbers, iterate.
- Keep writing free of AI tells.
