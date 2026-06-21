# Test 013 v2 — The Deep Brain

A **deep, no-backprop** brain-style language model that goes head-to-head with a
real pretrained LLM (default Pythia-410M) on WikiText-103 perplexity, at a tiny
fraction of the compute. This is the multi-file successor to `test_013/showdown.py`
(kept as the v1 single-layer baseline).

## What changed from v13, and why

v13's brain plateaued at ~2680 perplexity (vs the LLM's 17). The cause was
architectural, not a tuning bug: v13 has exactly **one** learned layer (a shallow
readout on a fixed random expansion), while an LLM has 24 layers of depth. Three
changes attack that directly, all staying strictly no-backprop:

1. **Depth — a stack of locally-trained cerebellar layers.**
   Each layer has its own granule expansion, kWTA sparse code, and class→word
   readout, and is trained by the **local delta rule against the next token**
   (deep supervision). No gradient crosses a layer. Layer *L+1*'s input is a
   fixed random projection ("relay") of layer *L*'s sparse code, concatenated
   with the context (a skip connection). Early layers learn surface n-gram
   structure; deeper layers, fed the codes below them, learn longer-range
   structure. The final prediction is a **probabilistic mixture** of all layers,
   `P(y) = Σ_l w_l · P_l(y)`, with the mixture weights `w` learned by a local
   EM-style rule (responsibility − prior). Neuroscience: laminar cortical
   microcircuits stacked on cerebellar expansion.

2. **Long context — hyperdimensional binding.**
   v13 concatenated token codes, so a fixed-fan-in granule saw a shrinking slice
   of the input as context grew (the CTX=4 regression in test_012). Instead we
   **bind** each token code to its position by an element-wise sign flip and sum
   under several temporal-decay scales (recent = sharp, distant = gist). The
   result is one `code_dim` vector regardless of context length, so context can
   grow from 6 to **32+** with no feature dilution — and the granule matmul
   shrinks too. Neuroscience: entorhinal/hippocampal conjunctive ("what × where")
   codes.

3. **Continuous granule learning.**
   v13's competitive k-means ran once on ~1M tokens then froze. Here granules
   keep adapting via lightweight competitive refinement every N blocks across all
   500M tokens.

Plus engineering for fast iteration on Kaggle 2×T4: a **fused** activation
(`F.linear`), **two-level approximate top-k** (group → candidates → final),
**flat 1-D scatter-add** for the word-head update (kills the v13 atomic-scatter
hot spot), **async double-buffered** data prefetch, and EMA on the cheap readout
tensors.

## Architecture at a glance

```
X (ctx token ids)
  └─ ContextCoder.bind ──────────────► bound [B, code_dim]      (HD binding)
        │
        ▼
   ┌─ Layer 0: granules→kWTA→readout₀ ─ predicts next token (delta rule)
   │     relay₀ = randproj(sparse₀);  x₁ = [relay₀ ; bound]
   ├─ Layer 1: granules→kWTA→readout₁ ─ predicts next token
   │     …
   └─ Layer L-1
        │
        ▼
   mixture  P(y) = Σ_l softmax(mix)_l · P_l(y)   (mix learned by local EM rule)
```

Nothing in here uses backprop. Every learning rule is local (delta rule for the
readouts, competitive k-means for the granules, EM responsibilities for the
mixture).

## Running on Kaggle ("GPU T4 × 2")

```bash
!python run.py --build-data        # tokenize Wikipedia once into the memmap cache (Internet ON)
!python run.py                     # auto dual-GPU when 2 are present; trains + scoreboard
!python run.py --single            # force single GPU
!python run.py --smoke             # tiny, CPU-OK; exercises the whole multi-layer path
```

Useful knobs (all have CLI flags):

| flag | meaning | default |
|---|---|---|
| `--n-layers` | depth of the stack | 4 |
| `--n-gran` | granules **per layer** (drop if VRAM OOMs) | 12288 |
| `--ctx` | context length (HD binding makes this cheap) | 32 |
| `--train-tokens` | training budget | 500M |
| `--no-refine` | freeze granules after setup (v13 behavior) | off |
| `--no-ema` | disable readout EMA | off |
| `--no-feat` | disable competitive granule learning | off |

### VRAM note

The word head `Ww[n_gran, V+1]` fp32 dominates memory, **summed across layers**:
`n_layers × n_gran × (V+1) × 4 bytes`. The default `4 × 12288 × ~50305 × 4 ≈
9.9 GB` matches v13's single-layer footprint and fits one T4 (the LLM is loaded
for eval only after the brain's transients are freed). Raising `--n-layers` or
`--n-gran` scales this linearly — lower one if you OOM.

## Module map

| file | what it is |
|---|---|
| `run.py` | CLI entrypoint; data build / smoke / single / dual-GPU dispatch |
| `bf/config.py` | `Cfg` dataclass, smoke scaling, CLI→Cfg |
| `bf/data.py` | memmap builder + async double-buffer `WindowSampler` |
| `bf/codes.py` | fixed token codes + hyperdimensional context binding |
| `bf/kernels.py` | fused activation, two-level approx top-k, flat scatter-add |
| `bf/granules.py` | one layer's expansion + competitive k-means + kWTA encode |
| `bf/readout.py` | class hierarchy + hierarchical delta-rule head + EMA |
| `bf/layer.py` | `CerebellarLayer` = granules + readout + inter-layer relay |
| `bf/brain.py` | `DeepBrain` = stacked layers + learned mixture |
| `bf/distributed.py` | NCCL init + local-SGD averaging across all layers |
| `bf/evaluation.py` | LLM strided perplexity, WikiText loader, scoreboard |
| `bf/train.py` | per-rank worker: setup, loop, sync, rank-0 eval |

## Honest accounting

The scoreboard reports the brain's **sparse** MACs/token summed across layers
(granule expansion + relays + readouts + context bind), exactly as a brain-style
accelerator that skips absent wires would pay it — not the dense matmul used to
compute it fast on a GPU. Per-layer perplexities and the learned mixture weights
are printed too, so you can see which depths are doing the work.

## Status

Code compiles clean and is written for the Kaggle 2×T4 box. **Validate with
`--smoke` first** (CPU-OK, runs the full multi-layer path on tiny-gpt2), then
`--build-data`, then the full run. Results go in `doc/test_013_v2.md`.
