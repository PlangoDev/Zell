# Experimental Results

Scope: a consolidated record of all measured results from the BrainFormer test_013 series, conducted in June 2026 on Kaggle dual NVIDIA T4 (sm_75) GPUs against WikiText-103. Sections cover the per-version lineage, the multi-timescale depth ablation, the sampled-softmax word head, test-time adaptation, the context-dimension refutation, the more-data-hurts regression and its fix, the homeostatic load-balancing negative result, the infini-gram interpolation win, throughput history, the per-phase profile, granule-health metrics, and a perplexity-to-capability calibration table. Every row pairs a configuration with a measurement. Code references are to `test_013_v3/src/main.cu`, `test_013_v3/src/config.h`, and `test_013_v3/doc/v15.md`, with commit hashes where a result is tied to a specific change. All perplexity figures are word-level WikiText-103 unless otherwise stated. The metric and its caveats are defined in Section 1.

## 1. Metric, baseline, and evaluation protocol

The evaluation metric is per-token perplexity on the WikiText-103 validation split. Two distinct evaluation modes are reported throughout:

- **Static** — the learned state (granule weights `gwt`, biases, class head `Wc`, word head `Ww`, mixture logits) is frozen after training; the model scores the evaluation stream with no further updates. This is the conventional language-model evaluation.
- **Adaptive** — the local delta rule continues to apply during evaluation on the document being read (the `--adapt` path, Section 5). Each token is scored *before* the update that token induces, so no label leaks into its own prediction. The frozen baseline cannot perform this mode.

The baseline is **EleutherAI/Pythia-410M** (405,334,016 parameters), measured by `run_llm.py` with strided WikiText-103 perplexity at window 512, giving **17.19**. This is the reference number against which every BrainFormer result is compared. The gap between BrainFormer's best static figures (low thousands) and 17.19 is the central quantitative fact of the program and is addressed structurally in Section 13.

Two properties of the protocol matter for interpreting the tables:

1. Evaluation is **deterministic** (fixed eval stream, fixed seed for the regenerated front-end). Differences between runs reported below are therefore not sampling noise; a change of a few percent is a real effect of the configuration change, not variance.
2. The strided-perplexity definition and window match between the baseline harness and the BrainFormer eval so that the numbers are comparable in kind. They are not comparable in magnitude in any favorable sense: BrainFormer is two orders of magnitude worse on static in-distribution perplexity. The interest of the results is in the *differentiated axes* (adaptation, retrieval interpolation) and in the *efficiency* of reaching the perplexities it does reach, not in matching the baseline.

## 2. Version lineage

The series progressed through three implementations of the same cerebellar core, separated by the learning topology and the host language.

| Version | Description | Role |
|---|---|---|
| test_013 v1 | Single learned cerebellar layer | First learned-readout result |
| test_013 v2 | Python multi-file deep stack | Perplexity oracle (correctness reference, slow) |
| test_013 v3 | C/CUDA single-translation-unit rewrite | Throughput vehicle; all v15 features |

The v1 single-layer model establishes the entry point of the lineage. Its static perplexity falls monotonically with training tokens, confirming the local delta rule is learning rather than drifting:

| Configuration | Train tokens | Static ppl |
|---|---|---|
| v1, single learned layer | 50M | 3284 |
| v1, single learned layer | 100M | 3086 |
| v1, single learned layer | 500M | 2680 |

v2 reproduced the multi-layer deep-supervision design in Python and served as the numerical oracle for the v3 rewrite: v3 kernels were validated against v2 outputs before throughput optimization. v3 is the implementation all subsequent results in this document are measured on, except where a row is explicitly attributed to v1.

## 3. Multi-timescale depth (`--multiscale`)

The deep stack is `n_layers` independent cerebellar experts (default quality config 4 layers), each with its own next-token objective, combined by a learned probabilistic mixture `P(y) = sum_l w_l P_l(y)` (mixture logits updated on-device by the local EM-style rule `k_mixture_update` on `dmix`). No gradient flows between layers; they are parallel experts, not a shared trunk.

Before multiscale, the layers were near-redundant: their individual perplexities were tightly clustered (per-layer 4076 / 4081), indicating each layer was solving the same problem at the same temporal resolution. The multiscale change makes layer `l` bind context with exponential decay `decay0 / 2^l`, so each layer attends to a different temporal window. The result is both an aggregate perplexity improvement and a real divergence of the layers into a temporal hierarchy.

| Configuration | Static ppl | Per-layer ppl (sample) | Commit |
|---|---|---|---|
| 4-layer, single-rate bind (pre-multiscale) | 3858 | 4076 / 4081 (near-tied) | — |
| 4-layer, `--multiscale 1` | 3478 (-10%) | L0 3539 / L1 3919 (diverged) | d5177d0 |

The -10% aggregate improvement and the per-layer divergence are reported together because they are the same effect observed two ways: the mixture gains because its experts stopped being redundant. The mechanism is `k_bind_rate` (single-rate bind per layer) applied with a per-layer decay schedule.

## 4. Sampled-softmax word head (`--neg-word`)

The word-head readout is the bandwidth bottleneck (Section 11). `--neg-word 8` replaces the full word-head softmax with a sampled-softmax over 8 negatives, reducing the per-step word-head work.

| Configuration | Throughput effect | Quality effect | Commit |
|---|---|---|---|
| `--neg-word 8` | ~+14% speed | quality-neutral | c4b901b |

This is the standard quality configuration's word head: the speed is taken for free because the perplexity is unchanged within measurement.

## 5. Test-time adaptation (`--adapt`, "hyper-fixation")

Test-time adaptation is the model's differentiated axis. The delta-rule update continues during evaluation, fitting the model to the specific document under the predict-then-score ordering described in Section 1. The effect is large and consistent.

| Configuration | Static ppl | Adaptive ppl | Relative change |
|---|---|---|---|
| 4-layer multiscale, 50M tokens | 3412 | 1930 | -43% |

The 43% reduction is the single most reproducible quality lever the bare model has, and it is structurally unavailable to a frozen backprop-trained baseline: Pythia-410M cannot rewrite its weights to the document it is reading at inference time without an explicit dynamic-evaluation training loop. The honest scope is that this gain is realized on the document being read (in-context fitting); it is real on any text the model evaluates, including held-out WikiText, but it does not by itself produce coherent open-ended generation.

## 6. Context-representation refutation (code dimension)

A natural hypothesis was that the quality ceiling is set by the context representation, i.e. that the hyperdimensional bound vector at `code_dim = 128` is too low-dimensional to carry enough context. This was tested directly by widening the code dimension fourfold.

| Configuration | code_dim | Static ppl effect |
|---|---|---|
| baseline | 128 | reference |
| widened | 512 | ~-4% ppl only |

A 4x increase in the context vector dimension bought only about 4% perplexity. The hypothesis is **refuted**: the context representation is not the quality wall. The remaining gap lies in credit assignment (Section 13), not in the width of the bound vector. This is recorded as a negative result so the program does not re-spend effort widening `code_dim`.

## 7. The "more data hurts" regression and its fix

A scaling regression was observed and is reported in full because it inverts the expected direction. The identical 4-layer multiscale configuration was worse at 500M tokens than at 50M tokens, on deterministic evaluation, so it is not noise.

| Configuration | Train tokens | Static ppl | Adaptive ppl |
|---|---|---|---|
| 4-layer multiscale | 50M | 3412 | 1930 |
| 4-layer multiscale (same config) | 500M | 3703 (worse) | 2057 (worse) |

Root cause: a constant learning rate over-trains as the token budget grows, and the fixed per-event weight decay `keep = (1 - lr*wd)^decay_every` (previously hardcoded `wd = 1e-5`, `decay_every = 200`) over-regularizes at scale. Both pathologies grow with the number of update events, so more data made the model worse rather than better.

The fix (commit 0e542be) adds cosine annealing of the learning rate and exposes the decay schedule:

```
--lr-final F      # cosine-anneal lr -> lr*F over the run (default 1.0 = old behavior)
--wd W            # weight-decay rate (was hardcoded 1e-5)
--decay-every N   # decay application interval (was hardcoded 200)
```

The recommended scaling-recovery configuration (`test_013_v3/doc/v15.md`, "run this first") is:

```
./brain --meta meta.json --train-tokens 500000000 \
  --n-classes 256 --n-layers 4 --multiscale 1 --neg-word 8 --adapt 1 \
  --lr-final 0.1 --wd 0
```

The keep/kill criterion attached to this fix is explicit: static ppl below the 50M figure of 3412 at 500M tokens means scaling has recovered and the larger overnight run is greenlit. The batch-invariant step normalization (`st = lr * lr_scale / LR_REF`, `LR_REF = 8192`, commit ac5179e) is a prerequisite for this analysis: before it, `st = lr / B` meant a larger batch produced *less* learning, conflating the batch-size knob with the learning-rate knob. After ac5179e, batch size is a pure speed knob (linear scaling rule) and the LR-annealing study is interpretable.

## 8. Homeostatic load-balancing (`--balance`) — negative result

The DeepSeek aux-loss-free load-balancing controller was ported onto the per-granule selection bias `gbias` (the bias was scaffolded but always zero, so there had been no load-balancing of any kind). The controller is a self-correcting negative feedback loop on per-granule fire rate: an over-firing granule has its bias raised and is suppressed at the next selection (`k_mixture_update`-adjacent path; controller in the `--balance` block, commit 4683af4).

| Configuration | Usage-Gini | Static ppl | Adaptive ppl |
|---|---|---|---|
| baseline (no balance) | 0.348 | 3412 | 1930 |
| `--balance 1 --balance-lr 0.1 --balance-every 200` | 0.251 | 3470 (worse) | 2037 (worse) |

The controller did exactly what it was designed to do at the level of statistics — usage-Gini fell from 0.348 to 0.251, flattening the hot-granule tail — but perplexity got slightly worse on both static and adaptive. The intervention was **dropped**, and the reasoning is the substance of the result:

- The dead-granule fraction was measured at **0.0%**. There was no idle capacity to recover, so the premise of load-balancing (reclaim wasted experts) did not hold here.
- The 0.348 Gini was therefore not a symptom of imbalance to be corrected; it reflected useful specialization. Language is correlated, so some granules legitimately fire more than others. Forcing fire rates toward uniform destroyed that specialization, which is why perplexity rose as the Gini fell.

The conclusion recorded is that the architecture is *healthy* on this axis: the skew is signal, not waste. The patch was nonetheless hardened by a four-agent adversarial review covering the atomic-path accumulation, dual-GPU integrator reset at each sync, refine-time reset, a u64 accumulator to prevent overflow, the `gran_frac` guard (the controller is incompatible with `--gran-frac < 1` and auto-disables), and a relative-error control law. It remains flag-gated and default-off so the proven path is unchanged.

## 9. Infini-gram interpolation (`--ngram`) — largest realized win

A suffix array is built over the training corpus (`tools/build_ngram.py`; pydivsufsort, with a vectorized numpy prefix-doubling fallback). At eval, the host query (`test_013_v3/src/ngram.h`: mmap plus binary-search longest-suffix backoff) returns the next-token distribution from the longest matching suffix of the current context, with estimate `(yes + 1e-6) / (tot + 1e-6*65536)`. The eval interpolates `P = (1-lambda)*P_brain + lambda*P_ngram`, scoring static and adaptive in a single pass with a lambda sweep over {0.1 .. 0.9}. The C query is validated byte-for-byte against a Python reference (`build_ngram.py --selftest`).

| Configuration | Static ppl | Adaptive ppl |
|---|---|---|
| 4-layer multiscale, 50M (no ngram) | 3412 | 1930 |
| + `--ngram`, 50M index, `lambda = 0.3` | 1272 (-63%) | 762 (-60%) |

This is the single largest quality jump in the project. The commits are 25aedc1 / c30462a / ef28dd5, with the host query in `src/ngram.h`. The build-once / eval-many workflow is:

```
python tools/build_ngram.py --meta meta.json       # writes tokens.bin.sa (+ .json)
./brain --meta meta.json ... --adapt 1 --ngram /kaggle/working/tokens.bin --ngram-lambda 0.3
```

Honest scope, stated plainly: the gain is on corpus-like, in-distribution text where a long suffix of the context actually occurs in the indexed corpus. For novel chat inputs there is no long match, the n-gram distribution backs off toward uniform, and the interpolation falls back to the bare model. The 762 adaptive figure therefore does **not** transfer to open-ended conversation. It is a real and large win on the benchmark distribution and near-zero on out-of-distribution chat.

## 10. Throughput history

Throughput is reported in tokens per second on the dual-T4 system. The progression from v13 to v3 is the result of the CUDA engineering described in `04-cuda-engineering.md` (cuBLAS fp16 tensor-core activation, bitonic-sort top-k, atomic-free bucketed readout scatter, on-device mixture update, resident VRAM training shard).

| Build / configuration | Throughput | Note |
|---|---|---|
| v13 | ~44k tok/s | pre-rewrite baseline |
| v3, peak | ~334k tok/s | best-case config |
| v3, 4-layer quality config | ~171k tok/s | 50M tokens in ~5 min |

The ~7.6x v13->v3 peak speedup and the ~3.9x speedup at the 4-layer quality config are what make the 50M-token A/B loop fast enough to run the ablations in this document (a 50M-token run completes in roughly five minutes at 171k tok/s).

## 11. Per-phase profile and the bandwidth ceiling

The per-phase profile for the 4-layer configuration, captured by the lag-free analytics path (timing flag plus a post-training profiling burst, no per-step host sync), is:

| Phase | Share of step time |
|---|---|
| bind | 1.1% |
| activate | ~21% |
| topk | ~24% |
| relay | 6% |
| readout-fwd | ~12% |
| scatter | ~33.5% |
| bias | 0.7% |
| mixture | 0.9% |

The system is **HBM-bandwidth-bound** on two reads: the fp32 word-head scatter (33.5%, the largest single phase) and the all-G activation read inside the dense GEMM (folded into activate at ~21%). The readout cost is minimized by the choice `n_classes ~ sqrt(V)`, which minimizes the readout term `K*(C + V/C)`; the default `n_classes = 256` is this minimum for the working vocabulary. The bandwidth ceiling is why several research grafts were declined: any technique that adds a second full-row pass over the word head (for example DeltaNet true forgetting) roughly doubles the dominant phase for marginal gain, and any technique that skips granule activations (IVF routing) forfeits the tensor-core throughput that makes the dense activation GEMM cheap on Turing.

A separate numerical finding constrains the word-head dtype: a bf16 word-head master rounds the delta-rule updates (magnitude ~3.7e-5) to zero, so the word head must remain fp32. This is the reason the dominant bandwidth phase cannot be halved by lowering its precision.

## 12. Granule-health metrics

The granule population was measured for dead units and usage skew (the metrics that motivated and then refuted the load-balancing intervention in Section 8).

| Metric | Value | Interpretation |
|---|---|---|
| Dead-granule fraction | 0.0% | No idle capacity; every granule participates |
| Usage-Gini (baseline) | 0.348 | Specialization, not waste |
| Usage-Gini (`--balance`) | 0.251 | Flattened, but ppl regressed |

The combination — zero dead granules with a moderate, productive usage skew — is the evidence that the k-winners-take-all sparse code is well-tuned for this corpus and that the architecture has no wasted capacity to reclaim. The denser-K hypothesis (that K=64 is too sparse for correlated language, suggested as a future A/B in `test_013_v3/doc/v15.md`) is noted but not yet measured.

## 13. Perplexity-to-capability calibration

The perplexity figures above are large by modern language-model standards, so a calibration is recorded to keep the numbers interpretable as capability rather than as a single abstract score. The mapping below is a qualitative anchor, not a measured curve; the BrainFormer column states which configuration reaches that band on in-distribution WikiText.

| Perplexity band | Qualitative capability on in-distribution text | BrainFormer configuration reaching this band |
|---|---|---|
| ~17 (baseline) | Coherent local syntax and short-range semantics; usable generation | Not reached by any BrainFormer config (structural gap) |
| ~100 | Strong next-token statistics; word- and phrase-level structure; not coherent generation | Web-verified DFA-on-LM ceiling for gradient-free credit assignment (reference, not BrainFormer) |
| low hundreds | Frequency and local-collocation structure captured; the realistic ceiling for the pure no-backprop model on in-distribution text | Stated program ceiling for the bare model (not yet reached) |
| ~760–1300 | Long-suffix recall dominates; strong on corpus-like text, falls back on novel text | 4-layer multiscale + infini-gram, 50M (adaptive 762 / static 1272) |
| ~1900–2000 | Captures local statistics; in-context fitting visible | 4-layer multiscale + adaptation, 50M (adaptive 1930) |
| ~2700–3900 | Bare next-token statistics; the un-augmented learned model | v1 single-layer 500M (2680) to 4-layer pre-multiscale (3858) |

The structural reading of this table: the bare no-backprop parallel-expert model lives in the ~2700–3900 band; adaptation moves it to ~1900; infini-gram interpolation moves it to ~760–1300 on in-distribution text only. None of these crosses the ~17 baseline, and the program's honest position is that no data-side or grafted technique does. The gap is **structural** — the parallel-expert mixture with fixed random codes and a linear local-delta readout lacks compositional credit assignment. This is the finding that motivated the Zell hybrid pivot (a backprop-trained efficient transformer as the coherence core, with BrainFormer's infini-gram retrieval and test-time adaptation as inference-time augmentations), recorded separately in the program's pivot documentation.

## 14. Consolidated results table

A single table of every headline measurement in this document, for cross-reference. Static/adaptive perplexities are word-level WikiText-103 on dual T4.

| # | Configuration | Train tokens | Static ppl | Adaptive ppl | Other measurement | Commit |
|---|---|---|---|---|---|---|
| 1 | v1 single learned layer | 50M | 3284 | — | — | — |
| 2 | v1 single learned layer | 100M | 3086 | — | — | — |
| 3 | v1 single learned layer | 500M | 2680 | — | — | — |
| 4 | 4-layer pre-multiscale | — | 3858 | — | per-layer 4076 / 4081 | — |
| 5 | 4-layer `--multiscale 1` | — | 3478 | — | per-layer 3539 / 3919; -10% | d5177d0 |
| 6 | `--neg-word 8` | — | neutral | neutral | ~+14% speed | c4b901b |
| 7 | 4-layer multiscale + `--adapt` | 50M | 3412 | 1930 | -43% adaptive | — |
| 8 | code_dim 128 -> 512 | — | ~-4% | — | refutes context-wall hypothesis | — |
| 9 | 4-layer multiscale (regression) | 500M | 3703 | 2057 | worse than 50M (deterministic) | — |
| 10 | LR-anneal + wd fix | 500M | < 3412 target | — | `--lr-final 0.1 --wd 0` | 0e542be |
| 11 | `--balance 1` | 50M | 3470 | 2037 | Gini 0.348 -> 0.251; dropped | 4683af4 |
| 12 | + `--ngram` lambda=0.3 (50M index) | 50M | 1272 | 762 | -63% static / -60% adaptive | 25aedc1 / c30462a / ef28dd5 |
| 13 | v3 peak throughput | — | — | — | ~334k tok/s (vs ~44k v13) | — |
| 14 | v3 4-layer throughput | — | — | — | ~171k tok/s (50M in ~5 min) | — |
| 15 | granule health | — | — | — | dead 0.0%; Gini 0.348 | — |
| 16 | batch-invariant step | — | — | — | `st = lr*lr_scale/8192` | ac5179e |

All figures are ground-truth measurements from the test_013 series on Kaggle dual T4 in June 2026. The single largest quality jump is row 12 (infini-gram interpolation, in-distribution only); the most transferable mechanism is row 7 (test-time adaptation); the most consequential negative results are row 8 (context width is not the wall) and row 11 (the architecture has no idle capacity to load-balance).
