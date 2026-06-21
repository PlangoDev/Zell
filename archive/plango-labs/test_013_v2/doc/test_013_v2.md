# Test 013 v2 — The Deep Brain (results)

*A no-backprop, brain-style language model goes DEEP: a stack of locally-trained
cerebellar layers with hyperdimensional long context, versus a real pretrained
LLM, on WikiText-103.*

> STATUS: implementation complete, awaiting first Kaggle run. This document is the
> template; fill the tables from the run output. The v13 baseline is recorded so
> every v2 number has something to beat.

## 1. The baseline to beat (v13, single layer)

| training data | Brain ppl | LLM ppl | Brain MACs/tok | compute vs LLM |
|---|---|---|---|---|
| 50M tokens  | 3284.45 | 17.19 | 2,388,288 | 0.589% |
| 100M tokens | 3086.33 | 17.19 | 2,388,288 | 0.589% |
| 500M tokens | 2680.41 | 17.19 | 2,388,288 | 0.589% |

v13 is one learned layer on a fixed random expansion. It learns (ppl falls with
data) but the 10× data sweep only cut perplexity ~18% — a one-layer ceiling.

## 2. What v2 changes (all no-backprop)

1. **Depth**: N cerebellar layers, each trained by the local delta rule against
   the next token (deep supervision); layer L+1 reads a random projection of
   layer L's sparse code plus a context skip. Final = learned probabilistic
   mixture of all layers.
2. **Long context via hyperdimensional binding**: token×position conjunctive
   codes summed under multi-scale temporal decay → context 6 → 32 with no
   fan-in dilution, and a smaller granule matmul.
3. **Continuous granule learning**: competitive refinement keeps adapting the
   granules across the whole stream instead of freezing after 1M tokens.

Engineering for fast iteration: fused activation, two-level approximate top-k,
flat scatter-add word-head update, async double-buffered prefetch, readout EMA.

## 3. Configuration

| knob | v13 | v2 default |
|---|---|---|
| layers | 1 | 4 |
| granules / layer | 49,152 | 12,288 |
| context | 6 | 32 |
| code dim | 96 | 128 |
| fan-in | 48 | 48 |
| kWTA K | 64 | 64 |
| classes | 256 | 256 |
| relay dim | — | 128 |

Total word-head VRAM is matched to v13 (~9.9 GB) by trading granules/layer for
depth: `4 × 12288 ≈ 1 × 49152`.

## 4. Results (fill from the run)

| training data | Brain ppl | LLM ppl | Brain MACs/tok | compute vs LLM | vs v13 |
|---|---|---|---|---|---|
| 50M  | _tbd_ | 17.19 | _tbd_ | _tbd_ | _tbd_ |
| 100M | _tbd_ | 17.19 | _tbd_ | _tbd_ | _tbd_ |
| 500M | _tbd_ | 17.19 | _tbd_ | _tbd_ | _tbd_ |

### Per-layer perplexity + learned mixture (500M)

| layer | ppl | mixture weight |
|---|---|---|
| 0 | _tbd_ | _tbd_ |
| 1 | _tbd_ | _tbd_ |
| 2 | _tbd_ | _tbd_ |
| 3 | _tbd_ | _tbd_ |

### Wall-clock per block (v13 vs v2)

| | v13 | v2 |
|---|---|---|
| s / 1M tokens | _tbd_ | _tbd_ |

## 5. Ablations to run

- `--no-refine` (freeze granules): isolates continuous-learning gain.
- `--no-ema`: isolates the EMA contribution.
- `--n-layers 1`: collapse to v13-shaped depth with the new front-end (isolates
  the HD-binding + engineering gains from the depth gain).
- `--ctx 6` vs `--ctx 32` vs `--ctx 64`: the long-context curve.

## 6. Open questions

- Does the random (unlearned) relay carry enough signal for deep layers to add
  value, or does it need a learned (still-local) relay?
- Where does the mixture put its weight — do deep layers earn their keep?
- Does continuous refinement keep helping at 500M, or saturate early?
