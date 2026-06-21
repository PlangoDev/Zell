# Research Program and Technique Survey

## 1. Purpose and frame

This document records the web-verified technique surveys conducted for the BrainFormer program in June 2026, the disposition of each candidate technique (built, deferred, or refuted), and the engineering reason behind every disposition. It is the IP record of *what was considered and why it was or was not adopted*, distinct from the implementation records of the mechanisms that shipped.

The program objective was to approach useful language-model quality at a small fraction of the compute and parameter cost of a standard transformer, using neuroscience-derived mechanisms (cerebellar Marr-Albus granule coding, sparse distributed representations, k-winners-take-all selection, local learning rules, multi-timescale binding, retrieval/associative memory, and test-time plasticity) and **no backpropagation**. The reference architecture is the cerebellar model in `test_013_v3/src/main.cu` with configuration in `test_013_v3/src/config.h`. The baseline against which BrainFormer is measured is EleutherAI/Pythia-410M (405,334,016 parameters) at WikiText-103 strided perplexity 17.19 (window 512, measured by `run_llm.py`).

The central empirical finding governs the entire survey. The perplexity gap between the best BrainFormer configuration (low thousands; best in-distribution figures around 762–1930 with augmentations) and the 17.19 baseline is **structural**, not a matter of tuning or scale. The no-backprop parallel-expert design lacks compositional credit assignment: each cerebellar layer is an independent next-token expert, the layers are combined by a learned probabilistic mixture, and no gradient flows between them. Every technique below was evaluated against the question of whether it crosses that structural gap. None does. The realistic ceiling for the pure model is low hundreds of perplexity on in-distribution text, not coherent open-ended generation. This conclusion is what motivated the Zell hybrid pivot recorded separately.

All work was conducted on Kaggle dual NVIDIA T4 GPUs against WikiText-103. New modules in v15 are flag-gated and default-off, so the proven v3 path is unchanged (`test_013_v3/doc/v15.md`).

## 2. Survey scope

Four technique families were surveyed:

| Family | Source domain | Question asked | Section |
| --- | --- | --- | --- |
| DeepSeek technique transfer | Frontier dense/MoE transformer training | Which DeepSeek efficiency techniques transfer to a no-backprop sparse model | 3 |
| Gradient-free credit assignment | Biologically-plausible and gradient-free learning | Can any non-backprop learning rule close the structural gap | 4 |
| Missing efficient-LM techniques | Retrieval-augmented and linear-attention LMs | Which efficient-LM techniques graft onto the cerebellar model | 5 |
| 2xT4 speed levers | Hardware-specific optimization | What raises throughput on the specific tensor-core hardware | 6 |

Section 7 records the structural-ceiling conclusion that closes the survey.

## 3. DeepSeek technique transfer

DeepSeek's published training stack was surveyed technique-by-technique against the constraints of the cerebellar architecture: no backpropagation, sparse k-winners-take-all activation, fixed random front-end, parallel-expert layer composition, and a linear local-delta-rule readout. The governing filter is that any technique presupposing dense attention or end-to-end gradient flow cannot transfer.

### 3.1 Transfers in principle (four techniques)

| Technique | DeepSeek role | BrainFormer fit | Status |
| --- | --- | --- | --- |
| Aux-loss-free load-balancing | Bias-controlled expert routing without an auxiliary loss term | Maps to a bias controller on the per-granule selection bias `gbias` | BUILT, did not help (3.1.1) |
| Multi-token prediction (MTP) | Auxiliary heads predict tokens at offset h to densify the training signal | Does not fit; see 3.2 | DEFERRED (structural) |
| Shared/always-on experts | A fraction of experts fire for every token to capture common structure | Maps to always-on granules; not yet built | DEFERRED |
| int8 readout (FP8 spirit) | Low-precision matmul for throughput | Maps to int8 quantization of the readout heads | DEFERRED (numerical constraint, see 3.3) |

#### 3.1.1 Aux-loss-free load-balancing (BUILT, `--balance`, commit 4683af4)

The per-granule selection bias `gbias` existed in the architecture as scaffolding but was always zero, so there was no load-balancing in effect. The DeepSeek aux-loss-free controller was implemented as a self-correcting negative-feedback rule on `gbias`: an over-firing granule receives a higher suppression bias, lowering its future selection probability. The control law is relative-error based; the patch was hardened by a four-agent adversarial review covering the atomic-path accumulation, dual-GPU integrator reset at each sync, refine reset, a u64 accumulator, a `gran_frac` guard, and the relative-error control law. The feature is dual-GPU safe (controller state resets at each sync) and incompatible with `--gran-frac < 1` (auto-disabled).

Result: usage-Gini fell from 0.348 to 0.251, confirming the controller flattens the hot-granule tail as designed. Perplexity got slightly **worse** (3470 / 2037 with balancing vs 3412 / 1930 without, static / adaptive). The feature was DROPPED from the quality configuration for a diagnosable reason: the dead-granule fraction was 0.0%, so there was no wasted capacity to recover. The 0.348 Gini reflected useful specialization, not waste — natural language is correlated, so a healthy sparse coder will use granules at non-uniform rates. Forcing uniform fire-rates removed that specialization. The architecture was already healthy; the technique solves a problem this model does not have.

### 3.2 Does not fit: multi-token prediction

MTP densifies the training signal by adding auxiliary heads that predict tokens at future offsets, with the benefit flowing back through the shared trunk during backpropagation. Neither precondition holds here. The cerebellar layers are parallel experts combined by a probabilistic mixture `P(y) = sum_l w_l P_l(y)`, not a shared-trunk hierarchy, so making layer *l* predict token *t+h(l)* makes that layer useless for the next-token objective it is supposed to serve. There is no backprop through which an auxiliary head's benefit could be shared with the rest of the model. The forward cross-layer cascade that MTP would informally approximate is already available as the `--boost` (NoProp-style residual cascade) flag (`test_013_v3/doc/v15.md`).

### 3.3 Numerical constraint on int8 readout

int8 readout transfers in principle and is the closest analogue of DeepSeek's FP8 spirit, but it inherits a hard numerical constraint established empirically in the readout path. The local delta-rule updates to the word head are on the order of 3.7e-5. A bf16 word-head master rounds these updates to zero, so the word head is kept in fp32. Any low-precision readout scheme must preserve the dynamic range of the delta-rule update or accumulate in higher precision; a naive int8 master would fail the same way bf16 did. int8 readout remains DEFERRED pending an accumulation design that respects this constraint.

### 3.4 Does not transfer (presuppose attention or backprop)

The following DeepSeek techniques were surveyed and rejected because they presuppose a dense attention mechanism or end-to-end gradient flow that the cerebellar model does not contain:

| Technique | Presupposition the model lacks |
| --- | --- |
| Multi-head Latent Attention (MLA) | Attention KV cache |
| FP8 training | Backprop gradient precision management |
| GRPO / R1 reinforcement learning | Policy-gradient backprop |
| DualPipe | Pipeline-parallel backprop schedule |
| NSA / DSA sparse attention | Attention |
| RoPE / YaRN | Attention positional encoding |

None of the transferable DeepSeek techniques, individually or combined, closes the structural perplexity gap. Load-balancing was tested and did not help; MTP does not fit; shared experts and int8 readout are throughput/capacity refinements, not credit-assignment fixes.

## 4. Gradient-free credit assignment

The structural gap is a credit-assignment gap: with no backprop, the model cannot assign blame across its depth in a way that composes. This section surveys the gradient-free and biologically-plausible learning rules that claim to substitute for backprop, and records why none reaches baseline quality.

### 4.1 Direct Feedback Alignment (DFA) — closest fit, conditioning-only

DFA broadcasts the output error directly to each layer through a fixed random feedback matrix, removing the need for a symmetric backward weight transport. It is the closest fit to the cerebellar model's existing fixed-random-projection machinery. It is nonetheless **conditioning-only**: the random feedback aligns the forward weights enough to descend, but it does not recover the exact gradient and cannot perform the compositional credit assignment that deep language modeling needs. The web-verified figure is decisive — DFA applied to a language model reached perplexity 93 against 30 for backprop on the same setup, a ~3x quality penalty that does not approach the 17.19 baseline, let alone cross it. DFA's cross-layer credit is already approximated in the codebase by `--boost` (`test_013_v3/doc/v15.md`). Status: surveyed, not adopted as a separate mechanism.

### 4.2 Other gradient-free rules — rejected with reasons

| Technique | Mechanism | Why it does not apply |
| --- | --- | --- |
| Forward-Forward | Two forward passes (positive/negative data), per-layer goodness objective | Presupposes a layer topology and per-layer contrastive objective the parallel-expert model does not have |
| Predictive coding | Iterative inference to a fixed point that approximates backprop | Multiplies the bandwidth-bound activation computation by an iteration count; the model is already HBM-bandwidth-bound (Section 6) |
| Target propagation | Learned inverses propagate targets backward | Presupposes invertible layer-to-layer mappings the fixed-random front-end does not provide |
| Equilibrium propagation | Energy-based relaxation to equilibrium | Presupposes an energy-based topology the model lacks; same per-iteration bandwidth multiplier as predictive coding |
| NoProp residual cascade | Forward-only residual refinement | Exists in the codebase as `--boost`; a forward cascade, not compositional credit assignment |
| MeZO / SPSA | Zeroth-order gradient estimation by random perturbation | Useful only for tuning a small set of hand-set hyperparameters, not for learning the readout or granule weights at scale |

The two rules that multiply the activation by an iteration count (predictive coding, equilibrium propagation) are doubly disqualified: the activation read over all G granules is already a bandwidth bottleneck (Section 6), so an iterative inner loop is both quality-insufficient and throughput-prohibitive on T4. MeZO/SPSA is retained only as a hyperparameter-tuning option, not a learning rule.

### 4.3 Conclusion for Section 4

No gradient-free credit-assignment rule crosses the gap. DFA, the closest fit, lands at roughly 3x the backprop perplexity in the web-verified reference. The rules that claim closer fidelity to backprop (predictive coding, target prop, equilibrium prop) presuppose topologies the model does not have and/or impose an iteration-count multiplier on the dominant cost. The absence of compositional credit assignment is intrinsic to the no-backprop design.

## 5. Missing efficient-LM techniques (gap-denters)

This family is distinct from the previous two: these techniques do not claim to replace backprop or to fix credit assignment. They are inference-time or readout-time augmentations from the retrieval-augmented and linear-attention literature that *narrow* the gap on in-distribution text without altering the training paradigm. One produced the single largest realized quality win in the project.

### 5.1 Infini-gram interpolation (BUILT — largest realized win)

Commits 25aedc1 / c30462a / ef28dd5. A suffix array over the training corpus yields the next-token distribution from the **longest matching suffix** of the current context, with no training and no backprop. The eval interpolates the two distributions:

```
P = (1 - lambda) * P_brain + lambda * P_ngram
```

Build and components:

- `tools/build_ngram.py` builds the suffix array, with a vectorized numpy prefix-doubling fallback or `pydivsufsort` when available; it also provides `--selftest` for validation.
- `src/ngram.h` is the host query: mmap of the index, binary-search longest-suffix backoff, probability estimate `(yes + 1e-6) / (tot + 1e-6 * 65536)`, validated byte-for-byte against the Python reference via `build_ngram.py --selftest`.
- Eval interpolates for both static and adaptive scoring in a single pass, with a lambda sweep over {0.1 .. 0.9}.

Result (50M index, lambda = 0.3): static perplexity 3412 -> 1272 (-63%); adaptive 1930 -> 762 (-60%). This is the single largest quality jump in the program.

Honest scope, stated plainly: the gain is realized on corpus-like, in-distribution text where a long suffix actually matches. For novel chat inputs there is no matching suffix and the interpolation falls back to the bare model. The 762 figure does not transfer to open-ended conversation. The technique narrows the in-distribution gap; it does not cross the structural gap for generation.

### 5.2 Test-time adaptation (BUILT — the differentiated axis)

`--adapt` ("hyper-fixation"). The local delta rule continues to apply during evaluation on the document being read, predict-then-learn: each token is scored *before* the update is applied, so there is no leakage. The frozen baseline LM structurally cannot do this. Best result: 4-layer multiscale static 3412 -> adaptive 1930 (-43%). This is the model's differentiated axis — a property the no-backprop local-learning design has that a backprop-frozen model does not.

### 5.3 DeltaNet-style forgetting gate (DEFERRED — bottleneck conflict)

A DeltaNet-style subtract-old-value / forgetting gate on the readout was surveyed as a way to let the local delta rule unlearn stale associations rather than only accumulate. The correct full-row version (decaying the entire word-head row, not just the updated entries) roughly **doubles the word-head scatter**, which is the dominant cost in the pipeline (Section 6, scatter ~33.5% of step time). The marginal quality gain over the already-built `--wd` weight decay plus `--lr-final` cosine annealing did not justify doubling the bottleneck. Status: NOT built; deferred for cost-vs-gain (`test_013_v3/doc/v15.md`).

### 5.4 kNN / SDM retrieval head (DEFERRED)

A k-nearest-neighbor or sparse-distributed-memory retrieval head keyed on the kWTA bitmask was surveyed. The kWTA active-set bitmask is a natural sparse key for associative recall, aligning with the program's retrieval/associative-memory thesis. It was not prioritized over infini-gram, which delivered the larger and already-validated retrieval win. Status: NOT built.

### 5.5 Denser kWTA K (DEFERRED — hypothesis)

Hypothesis: K = 64 active granules may be too sparse for language, which is highly correlated; a denser code could carry more usable signal per token. The v15 A/B plan lists `--k-active {128, 256}` as a sweep to test this (`test_013_v3/doc/v15.md`). Status: NOT built / not yet swept; labeled a hypothesis pending measurement.

### 5.6 Refuted hypothesis: context representation is not the wall

A separate hypothesis — that the context binding representation was the quality bottleneck — was tested and refuted. Increasing `code_dim` from 128 to 512 (a 4x richer context vector) gained only about 4% perplexity. The context representation is not the wall; the wall is credit assignment.

## 6. 2xT4 speed levers

Throughput levers were surveyed against the measured hardware profile. The system is HBM-bandwidth-bound on the fp32 word-head scatter plus the all-G activation read. Per-phase profile for the 4-layer configuration:

| Phase | Share of step time |
| --- | --- |
| bind | 1.1% |
| activate | ~21% |
| topk | ~24% |
| relay | 6% |
| readout-fwd | ~12% |
| scatter | ~33.5% |
| bias | 0.7% |
| mixture | 0.9% |

Throughput improved from ~44k tok/s (v13) to ~334k tok/s peak (v3), with ~171k tok/s for the 4-layer quality configuration (50M tokens in ~5 min). The choice `n_classes ~ sqrt(V)` minimizes readout cost `K*(C + V/C)` and is the architectural lever already taken.

### 6.1 IVF / coarse granule routing (DEFERRED — break-even on tensor cores)

The largest theoretical speed lever is to stop computing all G granule activations by using an IVF/coarse-routing scheme that visits only candidate granules. In a non-tensor-core setting this is reported to give a >5x speedup. It does not transfer to the T4 because it fights the design that earns the current throughput. The activation is a dense cuBLAS fp16 tensor-core GEMM (`cublasGemmEx`, `CUBLAS_COMPUTE_32F`); skipping granules forfeits tensor-core throughput, converting a dense tensor-core matmul into a sparse gather that the T4 executes far less efficiently. The realistic outcome on T4 is roughly break-even rather than the >5x the research assumed for non-tensor-core hardware. Status: NOT built, pending Nsight profiling on T4 to confirm the break-even estimate before any implementation effort (`test_013_v3/doc/v15.md`).

### 6.2 Incremental levers (DEFERRED)

CUDA graphs, coalesced scatter, dual-GPU layer-sharding, and int8 readout (subject to the numerical constraint in 3.3) are incremental throughput improvements. They do not change quality and were not prioritized over the quality and scale-recovery work. Status: NOT built / incremental.

### 6.3 Scale-recovery (BUILT — adjacent to speed)

The "more data hurts" regression — the same configuration giving static 3412 / adaptive 1930 at 50M tokens but a worse static 3703 / adaptive 2057 at 500M tokens, on deterministic eval (not noise) — was diagnosed as constant learning rate over-training and fixed weight decay over-regularizing at scale (`keep = (1 - lr*wd)^decay_every` per event). Fix in commit 0e542be: `--lr-final` cosine annealing of the learning rate, plus CLI exposure of `--wd` and `--decay-every` (previously hardcoded 1e-5 / 200). This is the v15 "run this first" item: if static ppl falls below 3412 at scale, scaling is recovered and the larger overnight run is greenlit (`test_013_v3/doc/v15.md`). The batch-invariant step `st = lr * lr_scale / LR_REF` with `LR_REF = 8192` (commit ac5179e) makes batch size a pure speed knob rather than a learning-rate confound (linear scaling rule); the earlier `st = lr / B` made a bigger batch mean less learning.

## 7. Structural-ceiling conclusion

The survey converges on one conclusion across all four families. The perplexity gap from the BrainFormer regime (low thousands; best in-distribution figures 762–1930 with augmentations) to the Pythia-410M baseline of 17.19 is **structural**. It originates in the no-backprop parallel-expert design — fixed random codes, a sparse kWTA bottleneck, a linear local-delta-rule readout, and independent layers combined by a probabilistic mixture with no gradient flowing between them — which lacks compositional credit assignment.

No surveyed technique crosses it:

- **DeepSeek transfer:** only four techniques transfer in principle; the one tested (load-balancing) did not help because the architecture had no wasted capacity to recover; MTP does not fit the parallel-expert topology; shared experts and int8 readout are capacity/throughput refinements, not credit-assignment fixes.
- **Gradient-free credit assignment:** DFA, the closest fit, lands at roughly 3x baseline perplexity (93 vs 30 web-verified); the higher-fidelity rules presuppose absent topologies or impose a prohibitive iteration-count multiplier on the bandwidth bottleneck.
- **Efficient-LM grafts:** infini-gram and test-time adaptation narrow the gap substantially on in-distribution text but fall back to the bare model on novel inputs; DeltaNet forgetting doubles the bottleneck; kNN/SDM and denser K are unbuilt and, at best, incremental.
- **Speed levers:** the largest (IVF routing) is roughly break-even on tensor-core hardware; the rest are incremental and quality-neutral.

The realistic ceiling for the pure no-backprop model is low hundreds of perplexity on in-distribution text, not coherent open-ended generation. This is the finding that motivated the Zell hybrid decision: use a real backprop-trained efficient transformer as the coherence core, and retain the BrainFormer mechanisms that *do* produce realized wins — infini-gram retrieval and test-time adaptation — as inference-time augmentations.

## 8. Built vs deferred summary

| Technique | Family | Status | Reason |
| --- | --- | --- | --- |
| Aux-loss-free load-balancing (`--balance`) | DeepSeek | BUILT, dropped from quality config | Dead-granule fraction 0.0%; Gini reflected useful specialization; ppl slightly worse |
| Infini-gram interpolation (`--ngram`) | Efficient-LM | BUILT | Largest realized win: static -63%, adaptive -60% in-distribution |
| Test-time adaptation (`--adapt`) | Efficient-LM | BUILT | Differentiated axis; -43% adaptive; frozen baseline cannot do it |
| LR annealing + WD exposure (`--lr-final`, `--wd`, `--decay-every`) | Speed/scale | BUILT | Fixes "more data hurts" regression at scale |
| Checkpoint/resume (`--save` / `--load`) | Engineering | BUILT | 12h Kaggle session limit; saves ~0.4GB learned state |
| NoProp residual cascade (`--boost`) | Gradient-free | BUILT (flag) | Forward cross-layer cascade; covers DFA's informal role |
| Multi-token prediction | DeepSeek | DEFERRED (structural) | Parallel-expert topology, no backprop to share the gain |
| Shared/always-on experts | DeepSeek | DEFERRED | Not yet built; capacity refinement |
| int8 readout | DeepSeek | DEFERRED (numerical) | bf16 rounds 3.7e-5 delta-rule updates to zero; needs HP accumulation design |
| DFA as separate mechanism | Gradient-free | NOT adopted | Conditioning-only; 93 vs 30 ppl; `--boost` covers it |
| Forward-Forward / predictive coding / target prop / equilibrium prop | Gradient-free | NOT applicable | Presuppose absent topologies; iterative variants multiply the bandwidth bottleneck |
| MeZO / SPSA | Gradient-free | Retained for HP tuning only | Not a learning rule for readout/granule weights |
| DeltaNet forgetting gate | Efficient-LM | DEFERRED (bottleneck) | Full-row decay doubles the word-head scatter for marginal gain |
| kNN / SDM retrieval head | Efficient-LM | DEFERRED | Lower priority than infini-gram |
| Denser kWTA K | Efficient-LM | DEFERRED (hypothesis) | `--k-active {128,256}` sweep pending |
| IVF / coarse granule routing | Speed | DEFERRED (break-even) | Fights tensor-core dense GEMM; needs Nsight profiling first |
| CUDA graphs / coalesced scatter / layer-sharding | Speed | DEFERRED (incremental) | Quality-neutral; lower priority |

## 9. Source references

- `test_013_v3/doc/v15.md` — v15 feature set, the "honest frame", and the deliberately-not-built list with reasons.
- `test_013_v3/src/main.cu` — cerebellar pipeline, CUDA kernels, readout scatter, mixture update.
- `test_013_v3/src/config.h` — configuration flags and defaults.
- `tools/build_ngram.py`, `src/ngram.h` — infini-gram suffix array build and host query.
- Commits: 4683af4 (load-balancing), 25aedc1 / c30462a / ef28dd5 (infini-gram), 0e542be (LR annealing + WD), ac5179e (batch-invariant step), d5177d0 (multiscale), c4b901b (sampled-softmax word head).
