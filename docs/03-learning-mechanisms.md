# BrainFormer Learning Mechanisms

## Scope

This document records every parameter-updating rule in the BrainFormer cerebellar language model (`test_013_v3`), together with the empirical findings that shaped them. No update in the system uses backpropagation. There is no chain of partial derivatives through the stack, no autograd graph, and no gradient flowing between layers. Every learned tensor is updated by a local rule that reads only quantities physically present at the point of update: the input that drove a granule, the active top-K granule set, the readout error at a single token, or a per-granule selection count. The layers are parallel experts under deep supervision, combined by a learned probabilistic mixture; the only cross-layer signal is a per-example boost weight (optional) and the mixture's responsibility estimate.

The learned state is small. Per layer the system trains the granule projection weights `gwt[G, fan_in]`, an optional per-granule selection bias `gbias[G]`, the class readout `Wc[G, C]`, the word readout `Ww[G, Gcols]` with `Gcols = C*S`, and the readout biases `bc[C]`, `bw[Gcols]`. A single device mixture vector `dmix[L]` is shared across the stack. Everything else — token codes `codes[V, D]`, position sign-codes `pos[ctx, D]`, decay weights, granule wiring `gidx[G, fan_in]`, and the relay projection `relayR[G, relay_dim]` — is fixed-random, seed-derived, and never updated. The checkpoint therefore serializes only `gwt, gbias, Wc, Ww, bc, bw, dmix` and regenerates the front-end from the seed (`save_brain`/`load_brain`, `main.cu:1761`/`main.cu:1782`).

Symbols used throughout: `V` vocabulary size; `D` code/bind width (`code_dim`, default 128); `G` granules per layer (`n_gran`, default 12288); `K` active granules per token (`k_active`, default 64); `fan_in` sparse input connections per granule (default 48); `C` readout classes (`n_classes`, default 256); `S = ceil(V/C)` slots per class; `Gcols = C*S`; `L` layers (`n_layers`, default 4 in the quality config); `B` batch; `lr` base learning rate; `st` the realized per-step scale.

All measurements were taken in June 2026 on Kaggle dual NVIDIA T4 GPUs against WikiText-103, with the Pythia-410M baseline at strided perplexity 17.19 (window 512, `run_llm.py`).

---

## 1. Competitive granule learning (k-means under winner-take-all)

### 1.1 Mechanism

Each granule `g` reads `fan_in` fixed input coordinates `gidx[g, :]` with weights `gwt[g, :]`. The wiring `gidx` is fixed-random for the run; the weights `gwt` are the learned feature dictionary. Learning is online k-means under the winner-take-all selection already imposed by the kWTA stage: a granule that fires (is selected into the top-K) moves its weight vector toward the mean of the inputs that activated it.

The accumulation kernel `k_comp_accum` (`main.cu:686`) runs one thread per active `(example, k_slot)` pair. For each fired granule `g = idx[t]` it increments a win count `counts[g]` and adds the activating input coordinates into a per-granule sum:

```
counts[g] += 1
for f in 0..fan_in:  sums[g, f] += x[e, gidx[g, f]]
```

The apply kernel `k_comp_apply` (`main.cu:700`) moves each weight toward the conditional mean and renormalizes the row to `sqrt(fan_in)` norm:

```
if counts[g] == 0: skip                       # dead granule this pass: untouched
mean = sums[g, f] / counts[g]
gwt[g, f] += eta * (mean - gwt[g, f])         # k-means step toward the centroid
gwt[g, :] *= sqrt(fan_in) / (||gwt[g, :]|| + 1e-6)   # fixed-norm rows
```

The row renormalization keeps activations on a comparable scale across granules so that the kWTA comparison is not dominated by weight magnitude — the homeostatic counterpart to fixed-norm cerebellar parallel-fiber synapses.

### 1.2 Initial learning and periodic refinement

`brain_competitive` (`main.cu:1604`) orchestrates the pass. It samples `feat_sample` (default 40000) windows, flows them up the full stack with `encode_stack` so each layer learns on its *true* input (the relay output of the layer below, not a proxy), and applies `feat_passes` (default 2) k-means passes per layer. The learning rate is annealed within the pass, `et = eta * (1 - p/passes)`, so later passes refine rather than relocate centroids. After each layer's apply, the dense projection `projT` and its fp16 mirror `projT_h` are rebuilt from the updated `gwt` via `k_build_projT` + `k_f32_to_f16` (`main.cu:1628`).

During training, refinement fires every `refine_every` blocks (default 50): a single pass over `refine_sample` (default 4000) windows at the smaller `refine_eta` (default 0.01) keeps the dictionary tracking the data without destabilizing the readout that sits on top of it (`main.cu:1932`). After each refine the projection is rebuilt and, in the dual-GPU case, the encoders are re-averaged so all replicas share one granule basis.

| Parameter | Flag | Default | Role |
|---|---|---|---|
| `learn_feat` | `--no-feat` disables | 1 | enable competitive learning at all |
| `feat_passes` | — | 2 | k-means passes in the initial pass |
| `feat_sample` | — | 40000 | windows sampled for initial learning |
| `feat_eta` | — | 0.05 | initial-pass step size |
| `refine_every` | `--refine-every` | 50 | blocks between refinements |
| `refine_sample` | — | 4000 | windows per refinement |
| `refine_eta` | — | 0.01 | refinement step size |

### 1.3 Relation to the M1 baseline

The earliest single-translation-unit milestone (M1) used purely random granule weights, with competitive learning layered on as M4. The header note at `main.cu:9` records that the kernels were written "correct first, fast second." On `--load`, competitive learning is skipped entirely: the loaded `gwt` already defines the basis, so re-running k-means would only drift it away from the readout it was paired with (`main.cu:1862`).

---

## 2. Readout: the local delta rule

### 2.1 Error definition

The readout is hierarchical. The vocabulary, sorted by descending corpus frequency, is partitioned into `C` classes of `S` contiguous slots (`build_hierarchy`, `main.cu:917`). A token's class is `w2class[y]` and its within-class slot is `w2slot[y]`. The probability of a word factorizes as `P(word) = P(class) * P(word | class)`, so the readout forward computes class logits over all `C` classes and word logits over the `S` slots of the *target* class only — a hierarchical cheap path that costs `K*(C + S)` per token rather than `K*V`. The choice `C ~ sqrt(V)` minimizes this gather cost `K*(C + V/C)`.

The forward kernel `k_readout_fwd` (`main.cu:341`) produces, per example, a class softmax and a within-class softmax, the true-word log-probability `logp[e]` (consumed by the mixture), and two delta-rule error vectors:

```
errc[e, c] = softmax(lc)[c] - onehot(class)[c]      # class error
errw[e, j] = softmax(lw)[j] - onehot(slot)[j]        # within-class word error
```

This is the gradient of the cross-entropy loss with respect to the *logits*, and only that — it is not propagated through any upstream weight. The local delta rule is exact for the single linear readout layer it touches and is not an approximation of backprop through the rest of the network; there is no rest of the network to propagate through, because the granule code feeding the readout is produced by fixed-random and competitively-learned (not gradient-learned) stages.

### 2.2 Weight update

Both readouts are updated by the same form: the outer product of the per-token error with the active granule activations, scattered into the rows of the active top-K granules only. For granule `g` in the active set with activation value `val[e, k]`:

```
Wc[g, c] -= st * weight[e] * val[e, k] * errc[e, c]            # for all c
Ww[g, base + j] -= st * weight[e] * val[e, k] * errw[e, j]      # base = class*S, for all j in 0..S
```

where `st` is the batch-invariant step (Section 4) and `weight[e]` is the per-example boost weight (1 unless residual boosting is active). Only `K` granule rows are touched per token, so the readout update is structurally sparse on the granule axis. Biases follow the same error directly (`k_bias_update`, `main.cu:670`):

```
bc[c] -= st * weight[e] * errc[e, c]
bw[base + j] -= st * weight[e] * errw[e, j]
```

The readout biases are initialized to the log unigram statistics of the corpus (`bw[c*S + slot] = log(freq[w] + 0.5)`, `bc[c] = log(sum freq in class)`, `main.cu:938`), so the model starts at the unigram prior and the delta rule learns the conditional structure on top of it.

### 2.3 Numerical finding: the word head must stay fp32

The realized per-step update magnitude is on the order of `3.7e-5` (driven by `st` with `LR_REF = 8192`). A bf16 master for the word head rounds updates of this size to zero, so accumulation stalls and the word head never learns. `Ww` and `bw` are therefore kept in fp32 masters. The word head is the dominant memory tensor (`G * Gcols * 4` bytes, the term that sets the ~0.4 GB checkpoint size), and it is also the scatter bottleneck (Section 3.3) — but it cannot be demoted to bf16 without losing the signal.

---

## 3. Readout scatter: bucketed and atomic paths

The delta-rule update is a scatter: many `(example, k_slot)` pairs write into shared granule rows, and the same granule can be selected by multiple examples in a chunk. Two implementations exist, selected by `--fast-scatter` (default 1 = bucketed). The host wrapper `readout_scatter` (`main.cu:1203`) chooses the path.

### 3.1 Atomic path (M1 fallback)

`k_readout_scatter` (`main.cu:501`) runs one block per example and uses `atomicAdd` into `Wc` and `Ww` to absorb cross-example collisions on shared granules. Correct, simple, and the reference the bucketed path is validated against; slower because of atomic contention on hot granules. Selected by `--fast-scatter 0`. The smoke config forces this path first (`config.h:157`).

### 3.2 Atomic-free bucketed path (M2 default)

The bucketed path inverts the scatter into a per-granule gather so that one block owns each granule row and no atomics touch the dominant `Ww` payload. Four device stages (`main.cu:1208`):

1. `k_bucket_count` (`main.cu:617`): count selections per granule into `gcount[G]` (one small atomic per pair, on the count array only).
2. `k_prefix_sum` (`main.cu:742`): device-side exclusive prefix sum `gcount -> goff[G+1]`. Single-threaded but off the bandwidth-bound path; it exists specifically to remove the host round-trip and sync that an earlier host prefix-sum imposed.
3. `k_bucket_fill` (`main.cu:622`): scatter each `(e, k)` pair into its granule's bucket slot using a per-granule cursor (`atomicAdd` on `cursor[g]`, not on the weights).
4. `k_scatter_class_bucketed` (`main.cu:633`) and `k_scatter_word_bucketed` (`main.cu:651`): one block per granule reads its bucket and accumulates `-st * sum_pairs(weight*val*err)` directly into the granule's row — no atomics, because the block exclusively owns row `g`.

This is the "atomic-free bucketed readout scatter with device-side prefix sum" of the CUDA engineering record.

### 3.3 Profile

The 4-layer profile (`run_rank` analytics, `main.cu:1984`) attributes scatter ~33.5% of step time, the single largest phase, ahead of topk ~24% and activation ~21%. The system is HBM-bandwidth-bound on the fp32 word-head scatter plus the all-`G` activation read.

### 3.4 Sampled-softmax word head

`--neg-word N` (default 0; quality config uses 8) cuts the word-head scatter from `K*S` to `K*(1 + N)` by updating the true word plus `N` sampled negatives within the target class instead of all `S` slots. The class head stays full and bucketed; the word head uses an atomic scatter over the small `(1+N)` column set (`readout_scatter_negw`, `main.cu:1253`; forward `k_readout_fwd_negw`, `main.cu:418`; negative sampling `k_sample_negw`, `main.cu:408`; word scatter `k_scatter_word_negw`, `main.cu:469`). Eval always uses the full exact readout. Measured effect: ~+14% throughput, quality-neutral.

---

## 4. The batch-invariant step (LR_REF = 8192)

### 4.1 The rule

The realized step inside `brain_step` (`main.cu:1290`) is:

```
const float LR_REF = 8192.0f;
float st = c.lr * b->lr_scale / LR_REF;
```

The scatter kernels then *accumulate* the per-example contributions across the batch chunk (the bucketed kernels sum over all pairs in a granule's bucket; the atomic kernels add each example's contribution). The total move applied per granule row is therefore `st * sum_over_examples(...)`, i.e. proportional to `(lr / LR_REF) * B * mean_contribution`.

### 4.2 Why a fixed reference

The earlier rule was `st = lr / B`, which normalizes by the *actual* batch. With that rule the total move per step is `lr * mean` regardless of `B`, so a larger batch means fewer steps over a fixed token budget and therefore *less total learning per token*. Batch size became a quality knob, which is wrong: it should be a pure speed knob.

Normalizing by a fixed reference batch `LR_REF = 8192` (the batch at which `lr` was tuned) makes the per-step move scale linearly with the actual `B`. Total learning per token is then constant, and batch size is decoupled from optimization (the linear scaling rule). The comment block at `main.cu:1284` states this directly. Fixed in commit `ac5179e`.

The test-time adaptation pass uses a different normalization, `st = adapt_lr / bc` (`main.cu:1481`), because adaptation reads the document in fixed-size chunks and the per-chunk move should be averaged over the chunk, not scaled to a reference batch.

---

## 5. Learning-rate annealing, weight decay, and the "more data hurts" regression

### 5.1 Cosine annealing of the learning rate

`lr_scale` (default 1.0 = constant) multiplies `st` and is updated each step by a cosine schedule from `lr` down to `lr * lr_final` over the full step count (`main.cu:1913`):

```
prog = step / sched_total
lr_scale = lr_final + (1 - lr_final) * 0.5 * (1 + cos(pi * prog))
```

`sched_total = n_blocks * steps_per_block` is the true number of optimization steps (`main.cu:1875`). With `--lr-final 1.0` the schedule is flat and reproduces the constant-`lr` behavior.

### 5.2 Weight decay

`brain_decay` (`main.cu:1375`) multiplies `Wc` and `Ww` by a keep factor every `decay_every` steps (`main.cu:1920`):

```
keep = (1 - lr * wd) ^ decay_every
W *= keep
```

Defaults are `wd = 1e-5`, `decay_every = 200`. Both were previously hardcoded; commit `0e542be` exposed `--wd` and `--decay-every`. Decay is applied to the readout weights only, not the biases or the granule dictionary.

### 5.3 The regression

The same 4-layer multiscale configuration produced *worse* perplexity with a larger token budget under deterministic evaluation (not noise):

| Tokens | Static ppl | Adaptive ppl |
|---|---|---|
| 50M | 3412 | 1930 |
| 500M | 3703 | 2057 |

### 5.4 Diagnosis and fix

With a constant `lr`, the delta rule keeps applying full-size updates long after the readout has converged, over-training on the tail of the schedule; simultaneously the per-event weight decay `keep = (1 - lr*wd)^decay_every`, applied more times at scale, over-regularizes. The two effects compound and the model gets worse with more data. The fix (commit `0e542be`) introduced the cosine `--lr-final` anneal — shrinking the step toward the end of the run so late updates refine rather than overshoot — and exposed `--wd` and `--decay-every` so the decay strength is tunable to the budget rather than fixed at the 50M-token setting.

---

## 6. The mixture: a local EM update

### 6.1 Architecture

The `L` layers are parallel experts, each with its own next-token objective (deep supervision). Their predictions are combined by a learned probabilistic mixture `P(y) = sum_l w_l * P_l(y)`, with weights `w = softmax(mix)` over the device logit vector `dmix[L]`. No gradient flows between layers; the mixture is the only place their outputs interact.

### 6.2 The update

`k_mixture_update` (`main.cu:752`) runs as a single block after each chunk, with no host synchronization (the logits live on-device and are updated on-device, so the training hot loop never stalls on the mixture). Per chunk it computes the current weights `w = softmax(mix)`, then for each example the per-layer responsibility

```
lw_l = log(w_l) + logp[l, e]
r_l(e) = softmax_l(lw)                # responsibility of layer l for example e
```

and updates each logit toward `(mean responsibility - prior weight)`:

```
mix[l] += mix_lr * ( mean_e r_l(e) - w_l )
```

This is the M-step of an EM fit of the mixture weights to the per-layer likelihoods: at a fixed point `mean_e r_l = w_l`, the standard EM stationarity condition. `mix_lr` defaults to 0.05. The `logp[l, e]` inputs are the true-word log-probabilities the readout forward already produced, so the mixture costs nothing beyond a reduction. The profile attributes ~0.9% of step time to the mixture.

### 6.3 Eval and adaptation

At eval the host reads `dmix`, forms `w = softmax(mix)`, and mixes per-layer log-probs in log-space with a log-sum-exp (`brain_eval_ppl`, `main.cu:1387`). The adaptive pass re-reads `dmix` each chunk because it is still being updated during adaptation (`main.cu:1476`), and applies the same `k_mixture_update` after scoring (`main.cu:1516`).

---

## 7. Homeostatic load-balancing (built, tested, dropped)

### 7.1 Mechanism

`gbias[G]` is a per-granule selection bias *subtracted* from the activation before kWTA (`k_sub_gbias`, `main.cu:720`; subtracted at `a -= gbias`). It was scaffolded in the architecture but held at zero until the load-balancing controller was added (commit `4683af4`). The controller is the DeepSeek aux-loss-free load-balancing rule, recast as cerebellar intrinsic-excitability homeostasis: nudge each granule's selection bias toward an even fire-rate, with no auxiliary loss and no gradient.

Per-granule selection counts accumulate into `usage_acc[G]` via `k_accum_counts` (`main.cu:588`), fed from the same `gcount` the bucketed scatter already builds (so the integrator is nearly free; the atomic path builds `gcount` explicitly only when balance is on, `main.cu:1233`). Every `balance_every` steps `k_balance_apply` (`main.cu:598`) runs a negative-feedback control law on a scale-free relative error:

```
fr = usage_acc[g] / n_examples            # measured per-example fire rate
e  = fr / target - 1                       # target = K/G
e  = clamp(e, -1, +4)                       # asymmetric: hot granule pushes hard
gbias[g] = clamp(gbias[g] + gamma * e, -8, +8)
usage_acc[g] = 0                            # reset integrator
```

Because `gbias` is subtracted in the activation, an over-firing granule gets a *higher* bias (suppressed) and a starved granule a *lower* bias (boosted). The relative-error form makes a fixed gain `gamma` work regardless of the `K/G` ratio.

| Parameter | Flag | Default | Role |
|---|---|---|---|
| `balance` | `--balance` | 0 (off) | enable the controller |
| `balance_lr` | `--balance-lr` | 0.1 | controller gain `gamma` |
| `balance_every` | `--balance-every` | 200 | steps between apply + integrator reset |

### 7.2 Adversarial hardening

The patch was hardened by a four-agent adversarial review covering: the atomic-path accumulation (build `gcount` even when bucketed scatter is off); dual-GPU integrator reset (reset `usage_acc` and `bal_consumed` after readout averaging so the controller restarts from the averaged `gbias` without windup, `main.cu:1746`); refine reset (clear the integrator after competitive refine retunes the basis, `main.cu:1938`); a `u64` accumulator to avoid count overflow; a `--gran-frac` guard (subsampling makes the per-example fire-rate denominator wrong, so balance is disabled under `gran_frac < 1`, `main.cu:1896`); and the relative-error control law itself.

### 7.3 Empirical finding: dropped

| Config | Static ppl | Adaptive ppl | Usage-Gini |
|---|---|---|---|
| baseline (no balance) | 3412 | 1930 | 0.348 |
| with `--balance` | 3470 | 2037 | 0.251 |

Balancing reduced the usage-Gini from 0.348 to 0.251 but made perplexity slightly *worse*. The diagnosis: the dead-granule fraction was 0.0% (measured by `brain_health`, `main.cu:1645`), so there was no wasted capacity to recover by spreading fire-rates. The Gini was not waste — it reflected useful specialization, because language is correlated and some granules legitimately fire more. Forcing uniform fire-rates removed that specialization. The architecture is healthy without the controller, so it was dropped from the quality config (`balance` defaults to 0). The mechanism remains in the codebase, guarded and hardened, but off by default.

---

## 8. Test-time adaptation ("hyper-fixation")

### 8.1 Mechanism

`--adapt` keeps the delta rule running during evaluation, on the document being read. This is the model's differentiated axis: a frozen backprop-trained baseline cannot improve as it reads a document, but a model whose learning rule is cheap enough to run at inference can.

`brain_eval_adapt` (`main.cu:1455`) reads the document in order, in chunks. For each chunk and each layer it:

1. runs `k_readout_fwd` to produce both the prediction `dlogp` (captured *before* any update) and the delta-rule error;
2. applies `readout_scatter` at `st = adapt_lr / bc` to learn from the chunk;
3. after scoring the chunk for perplexity, applies the mixture EM update.

### 8.2 No leakage

The scoring order guarantees no information leakage. The per-layer log-prob `dlogp` is written by the forward pass *before* `readout_scatter` mutates the weights (`main.cu:1486` forward, `main.cu:1493` learn), and `dlogp` is copied to host *after* the scatter only to read the pre-update values it already holds (`main.cu:1495`, commented "pre-update predictions"). Each token is therefore scored by the model state that existed before that token was seen — predict-then-learn. The token's own label never influences its own score.

### 8.3 Result

The headline result of the pure model: 4-layer multiscale, static 3412 -> adaptive 1930, a 43% perplexity reduction (`main.cu:2028`). The adaptive pass runs only in benchmark/eval mode and mutates the readout in place; it is run after the static eval, which does not mutate.

| Parameter | Flag | Default | Role |
|---|---|---|---|
| `adapt` | `--adapt` | 0 | run the adaptive eval pass |
| `adapt_lr` | `--adapt-lr` | 0.1 | step for the test-time update |

### 8.4 Honest scope

The adaptive gain is real on in-distribution / corpus-like text where the document being read resembles training data. For the infini-gram interpolation that compounds with it (static 3412 -> 1272, adaptive 1930 -> 762 at 50M index, lambda 0.3), the same scope caveat applies and is sharper: the gain comes from a long matching suffix in the corpus, so it does not transfer to novel chat inputs where no suffix matches. The 762 figure is in-distribution only. Test-time adaptation likewise specializes to the document's own statistics; it does not manufacture compositional ability the architecture lacks.

---

## 9. Optional cross-layer signal: residual boosting

Boosting (`--boost`, NoProp-style residual cascade) is the only mechanism that lets one layer's outcome influence another's *update*. It does so through a scalar per-example weight, not a gradient. `k_boost_weight` (`main.cu:783`) sets layer `l`'s per-example update weight to `weight[e] = max(1 - p_prev(e), wmin)`, where `p_prev` is the probability the layer below assigned to the true word. Examples the previous layer already predicts well are down-weighted, so each layer specializes on the residual and depth becomes additive (`main.cu:1321`). Layer 0 and the non-boost path use `k_fill_ones` (`weight = 1`, `main.cu:779`). The weight multiplies `st` inside every scatter and bias kernel. Default off (`boost = 0`, `boost_wmin = 0.1`).

---

## 10. Summary of update rules

| State | Rule | Kernel | Signal used | Backprop |
|---|---|---|---|---|
| `gwt[G, fan_in]` | competitive k-means under WTA + row renorm | `k_comp_accum`/`k_comp_apply` | activating inputs of fired granules | no |
| `Wc[G, C]`, `Ww[G, Gcols]` | local delta rule, scattered into active K rows | `k_scatter_*` | `softmax - onehot` at one token | no |
| `bc[C]`, `bw[Gcols]` | delta rule on the error directly | `k_bias_update`/`k_bias_class` | `softmax - onehot` | no |
| `gbias[G]` | homeostatic negative feedback to target fire-rate (off by default) | `k_balance_apply` | per-granule selection counts | no |
| `dmix[L]` | EM M-step toward mean responsibility | `k_mixture_update` | per-layer true-word log-probs | no |

Fixed-random, never updated: `codes`, `pos`, `decay`, `gidx`, `relayR`. Schedule knobs that modulate the delta rule: the batch-invariant `LR_REF = 8192`, the cosine `lr_final` anneal, and the periodic weight decay `keep = (1 - lr*wd)^decay_every`.

The structural ceiling is recorded plainly: the perplexity gap from ~2000 down to the 17.19 baseline is not closed by any of these rules or by the data-side grafts (infini-gram, adaptation). The no-backprop parallel-expert design lacks compositional credit assignment, and the local rules above — each exact for the single quantity it touches — cannot supply it.
