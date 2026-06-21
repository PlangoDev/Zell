# Cerebellar Architecture (BrainFormer, no-backprop)

This document specifies the no-backprop language model at the center of the
BrainFormer program: a deep stack of independent cerebellar layers, each trained
by local learning rules, combined by a learned probabilistic mixture. The
reference implementation is the C/CUDA rewrite `test_013_v3`. All code-level
references below cite `test_013_v3/src/main.cu` and `test_013_v3/src/config.h`.
The model is the no-backprop axis of the project; the hybrid chatbot built on a
backprop-trained core is named Zell and is out of scope here.

The design intent is a model whose **forward front-end is entirely fixed-random
or learned by competition (no gradient)** and whose **only trained tensors are a
linear readout updated by a delta rule and a mixture controller updated by a
local EM rule**. No error signal is ever propagated backward across a nonlinear
boundary, and no error signal flows between layers. The layers are parallel
experts, not a shared-trunk hierarchy.

---

## 1. Symbol table and default dimensions

All symbols and their defaults are read from `cfg_default` (`config.h:120`).

| Symbol | Meaning | Field / flag | Default | Quality config |
|---|---|---|---|---|
| `V` | vocabulary size (from `meta.json`, raised to max observed id) | `meta.vocab` | corpus | corpus |
| `D` | token code width = context-bind width | `code_dim` / `--code-dim` | 128 | 128 |
| `ctx` | context length (tokens) | `ctx` / `--ctx` | 32 | 32 |
| `n_scales` | temporal decay scales (summed bind) | `n_scales` | 3 | 3 |
| `decay0,1,2` | per-scale exponential decay rates | `decay0/1/2` | 0.20, 0.05, 0.0125 | as default |
| `L` | layers (parallel experts) | `n_layers` / `--n-layers` | 2 | 4 |
| `G` | granules per layer | `n_gran` / `--n-gran` | 12288 | 12288 |
| `fan_in` | sparse input connections per granule | `fan_in` / `--fan-in` | 48 | 48 |
| `K` | active granules per token (kWTA) | `k_active` / `--k-active` | 64 | 64 |
| `C` | readout classes | `n_classes` / `--n-classes` | 256 | 256 |
| `S` | slots per class = `ceil(V/C)` | derived | derived | derived |
| `Gcols` | word-head columns = `C*S` | derived | derived | derived |
| `relay_dim` | Johnson-Lindenstrauss relay width | `relay_dim` / `--relay-dim` | 128 | 128 |
| `topk_groups` | top-k candidate groups | `topk_groups` | 256 | 256 |
| `topk_per_group` | candidates kept per group | `topk_per_group` | 2 | 2 |
| `multiscale` | per-layer timescale binding | `multiscale` / `--multiscale` | 0 | 1 |
| `neg_word` | sampled-softmax negatives (train only) | `neg_word` / `--neg-word` | 0 | 8 |

Derived layer input width: layer 0 consumes the bind vector directly
(`in_dim = D`); every deeper layer consumes `[relay ; bind]`
(`in_dim = relay_dim + D`). See `layer_alloc` (`main.cu:954`):

```
ly->in_dim = (depth == 0) ? b->D : (b->relay_dim + b->D);
```

`brain_init` (`main.cu:995`) computes `S = (V + C - 1) / C` and `Gcols = C*S`.

---

## 2. Per-token pipeline overview

Each token is processed by the same seven-stage pipeline. Stages 1-2 are shared
across all layers; stages 3-6 are per layer; stage 7 combines the layers.

```
  context window X[ctx]  (token ids)
        |
   (1) fixed RANDOM token codes        codes[V, D]          never trained
        |
   (2) hyperdimensional context bind    bound[D]            multi-scale decay
        |                                                     x position sign-codes
   ============ per layer l = 0..L-1 (parallel, independent) ==============
        |                       input  x_l = (l==0 ? bound : [relay_{l-1} ; bound_l])
   (3) granule expansion        a_l[G] = x_l . projT_l^T - gbias   fan_in-sparse wiring
        |                                                          gwt by competition
   (4) kWTA sparse activation   top-K granules fire -> (idx_l[K], val_l[K])
        |                                                          one of 4 top-k kernels
   (5) relay projection         r_l[relay_dim] = L2norm( sum_k val.relayR[idx] )
        |                       x_{l+1} = [ r_l ; bound_{l+1} ]    fixed random JL
        |
   (6) hierarchical readout     P_l(word) = P_l(class) . P_l(word|class)
        |                       Wc_l[G,C], Ww_l[G,Gcols]           local delta rule
   =======================================================================
        |
   (7) learned mixture          P(y) = sum_l w_l . P_l(y)          local EM on logits
```

For training the per-layer readout error is computed and scattered immediately
(deep supervision); for evaluation each layer emits only the true-word logprob
and the mixture combines them.

---

## 3. Stage 1 — fixed random token codes

Every token id `t` is assigned a fixed random Gaussian code vector
`codes[t, :] in R^D`, drawn once at init and **never trained** (`brain_init`,
`main.cu:1004`):

```
std::vector<float> codes((size_t)V * b->D);
for (size_t i = 0; i < codes.size(); i++) codes[i] = frand(&r);
```

`frand` returns a standard Gaussian (`rng_gauss`, `main.cu:914`). The code matrix
is `codes[V, D]`, `D` defaulting to 128. Because the codes are seed-derived they
are regenerated rather than stored in a checkpoint (Section 11).

Refuted hypothesis: code width is **not** the quality wall. Raising `code_dim`
from 128 to 512 improved perplexity by only ~4%, so the representational
bottleneck lies in credit assignment, not in the context vector's dimensionality.

---

## 4. Stage 2 — hyperdimensional context binding

The context window `X[ctx]` (the `ctx` token ids preceding the target) is bound
into a single vector `bound[D]` by hyperdimensional binding: per-dimension
multiplication of each token's code by a per-position sign-code, weighted by an
exponential temporal decay, then summed over positions and scales.

### 4.1 Position sign-codes

`pos[ctx, D]` is a fixed random `+/-1` matrix (`brain_init`, `main.cu:1009`):

```
for (size_t i = 0; i < pos.size(); i++) pos[i] = (rng_u64(&r) & 1) ? 1.0f : -1.0f;
```

Multiplying a token code by its position's sign-code is the binding operation:
it makes the bound vector sensitive to **where** a token appeared, while keeping
the bind cheap (`ctx*D` multiply-adds, independent of `G`).

### 4.2 Multi-scale decay (summed variant)

The default summed bind uses `n_scales` decay rates. Distance is measured from
the most recent token (`dist = ctx-1-p`, so the last token has `dist=0`). The
precomputed weights `decay[n_scales, ctx]` are built at init (`main.cu:1013`):

```
decay[s*ctx + p] = expf(-(float)dist * rates[s]);   rates = {decay0, decay1, decay2}
```

The bind kernel `k_bind` (`main.cu:53`) implements

```
bound[e,d] = (1/n_scales) * sum_p ( sum_s decay[s,p] ) * codes[X[e,p],d] * pos[p,d]
```

one thread per `(example, dimension)` pair. Three scales (0.20 / 0.05 / 0.0125)
give one vector that simultaneously emphasizes recent tokens and retains a long
tail of older context.

### 4.3 Per-layer single-rate binding (multiscale flag)

When `--multiscale 1`, the summed bind is replaced by `L` independent
single-rate binds: layer `l` binds context at rate `decay0 / 2^l` (`encode_stack`,
`main.cu:1090`):

```
float rate = b->cfg.decay0 / (float)(1 << l);   /* deeper = longer range */
k_bind_rate<<<...>>>(b->dX, b->codes, b->pos, rate,
                     b->dboundL + l*chunk*D, bc, b->ctx, D);
```

`k_bind_rate` (`main.cu:75`) computes `bound[e,d] = sum_p exp(-dist*rate) *
codes[X,d] * pos[d]`. Deeper layers therefore see longer-range context. This is
what turned the stack from `L` near-identical experts into a real hierarchy:
static perplexity fell from 3858 to 3478 (-10%) and the per-layer perplexities
diverged (L0 3539 / L1 3919, versus the near-tied 4076 / 4081 before the change),
commit `d5177d0`. Storage is per-layer: `dboundL[L, chunk, D]` (`brain_init`,
`main.cu:1044`).

---

## 5. Stage 3 — granule expansion and sparse wiring

Each layer expands its `in_dim`-wide input into `G` granule activations
(`G = 12288`/layer by default). This is the Marr-Albus expansion: a few mossy-fibre
inputs fan out into a much larger, sparse granule layer.

### 5.1 Sparse random wiring

Each granule `g` has exactly `fan_in` (default 48) input connections chosen
uniformly at random over `[0, in_dim)`, with weights `gwt[g, fan_in]`
(`layer_alloc`, `main.cu:957`):

```
gidx[i] = rng_int(r, ly->in_dim);   /* which input dims this granule reads */
gwt[i]  = frand(r);                 /* connection weight */
```

Each granule's weight row is normalized to unit L2 then scaled by `sqrt(fan_in)`
(`main.cu:964`). The wiring indices `gidx` are fixed; the weights `gwt` are the
only part of the front-end that can be tuned, and that tuning is **competitive,
not gradient** (Section 9.1).

### 5.2 Dense projection and the fp16 mirror

For throughput the sparse wiring is materialized into a dense projection-transpose
`projT[G, in_dim]` so the activation is a dense GEMM. `k_build_projT`
(`main.cu:40`) scatters the sparse weights, with `atomicAdd` to absorb duplicate
input indices:

```
projT[g, gidx[g,f]] += gwt[g,f];
```

A half-precision mirror `projT_h[G, in_dim]` is produced by `k_f32_to_f16`
(`main.cu:728`). `projT` is the fp32 master rebuilt by competitive learning;
`projT_h` feeds the tensor-core activation.

### 5.3 Activation

The granule pre-activation is the projection minus a per-granule selection bias:

```
a[e,g] = sum_d x[e,d] * projT[g,d] - gbias[g]
```

Two paths exist (`encode_stack`, `main.cu:1119`). The default cuBLAS path casts
`x` to fp16, runs `cublasGemmEx` (`CUBLAS_OP_T, CUBLAS_OP_N`, compute type
`CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`) into an fp16 activation
matrix `dact[chunk, G]`, then subtracts `gbias` with `k_sub_gbias`
(`main.cu:720`). The fallback custom kernel `k_activate` (`main.cu:92`) computes
the same inner product in fp32. fp16 is safe here because the activation matrix
only feeds an argmax (top-k); precision of the dot products beyond ranking does
not matter.

`gbias[g]` defaults to zero and is the actuator for the homeostatic load-balancing
controller (Section 12). The `--gran-frac` flag activates only a rotating window
of `Geff = G*gran_frac` granules per step (granule dropout), cutting the
activate+topk bandwidth at a quality cost; winners are shifted back to global ids
by `k_add_offset` (`main.cu:734`, called at `main.cu:1158`).

---

## 6. Stage 4 — k-winners-take-all (kWTA)

Only the top `K` granules (default 64) by activation fire; the rest are zero.
The fired granules emit `idx[K]` (granule ids) and `val[K]` (a ReLU margin: the
winner's activation minus the `(K+1)`th activation, floored at zero). The margin,
not the raw activation, is the value carried forward; subtracting the threshold
makes the code contrast-normalized and non-negative.

Four interchangeable top-k kernels exist; `encode_stack` (`main.cu:1145`) selects
one per step:

| Kernel | Selector | Method | Notes |
|---|---|---|---|
| `k_topk` (`main.cu:107`) | `--exact-topk` | `K+1` sequential block-argmax passes over all `G` | exact reference; slow |
| `k_topk_approx` (`main.cu:156`) | fallback when `M` not 2^n | per-group top-`per_group`, then `K+1` argmax over `M=groups*per_group` candidates | ~1 scan of `G` |
| `k_topk_bitonic` (`main.cu:212`) | default | same candidates, then a bitonic **sort** of `M` in ~`log2(M)^2/2` parallel stages | exact over candidates; default |
| `k_topk_groupmax` (`main.cu:275`) | `--fast-topk` | split `G` into exactly `K` groups, take each group's max in a single scan | fastest, crudest; needs `K\|G`, `K` power-of-two |

With the defaults `topk_groups=256`, `topk_per_group=2`, the candidate count is
`M = groups*per_group = 512`, a power of two `>= K+1`, so the bitonic kernel is
selected. The two-level scheme (many small groups, a few per group) captures
nearly all of the true top-K at roughly one scan; the bitonic sort replaces the
`K+1` sequential argmax passes that had made the approx kernel the bottleneck.
`k_topk_groupmax` exploits the fact that granule index is uncorrelated with
activation (random wiring), so one-winner-per-contiguous-group closely
approximates the true top-K. The selection guards live in `main.cu` (the
`exact_topk` / `fast_topk` / bitonic / approx cascade at `main.cu:1145-1156`),
with parameter validation in `main()` (`main.cu:2102`).

The margin (`val`) computation is identical across kernels: ReLU of (winner
activation - threshold), where threshold is the `(K+1)`th value.

---

## 7. Stage 5 — relay / Johnson-Lindenstrauss projection

Layers communicate through a fixed random low-dimensional projection of the
sparse code, not through the sparse code itself. Each layer holds
`relayR[G, relay_dim]`, a fixed Gaussian matrix scaled by `1/sqrt(relay_dim)`
(`layer_alloc`, `main.cu:981`). `k_relay_concat` (`main.cu:304`) forms

```
r[e,:] = normalize( sum_k val[e,k] * relayR[idx[e,k], :] ) * sqrt(relay_dim)
x_next[e,:] = concat( r[e,:] , bound_{l+1}[e,:] )
```

The sparse activation is read out as a dense `relay_dim`-vector by summing the
relay rows of the active granules weighted by their margins, L2-normalizing, and
rescaling by `sqrt(relay_dim)`. The next layer's input is this relay concatenated
with that layer's context bind (the multiscale variant supplies the `l+1` bind;
the summed variant reuses the single `dbound`, `main.cu:1170`). This is a
Johnson-Lindenstrauss embedding: a random linear map that preserves pairwise
distances of the sparse codes in `relay_dim` dimensions, so layer `l+1` sees a
compressed, distance-preserving summary of layer `l`'s firing pattern plus fresh
context. The relay matrix is never trained; the only learned signal across the
stack is carried by the readout and the mixture, both of which are local.

The relay is **not** a residual trunk. No gradient or error crosses it; it is a
forward-only feature transform feeding an independent expert.

---

## 8. Stage 6 — hierarchical class/slot readout

### 8.1 Factored vocabulary

The readout factors the next-token distribution into a class distribution and a
within-class word distribution:

```
P(word) = P(class) * P(word | class)
```

The vocabulary is partitioned by descending unigram frequency into `C` classes
of `S = ceil(V/C)` contiguous slots each (`build_hierarchy`, `main.cu:917`). A
word's rank `r` maps to `class = r/S`, `slot = r%S`; the inverse map
`class_word[C*S]` (rank-ordered, padded with `V`) supports generation. The class
and word biases are initialized to the log unigram counts so the untrained model
already reproduces the unigram distribution (`main.cu:937`):

```
bw[class*S + slot] = logf(freq[word] + 0.5f);
bc[class]          = logf(sum_freq_in_class + 0.5f);
```

### 8.2 Why n_classes ~ sqrt(V)

The readout gather cost per token is `K*(C + V/C)`: the class head touches `C`
columns and the word head touches `S = V/C` columns of the target class, for each
of the `K` active granules. Minimizing `C + V/C` over `C` gives `C = sqrt(V)`,
hence `n_classes ~ sqrt(V)`. The same optimum reduces the hierarchical "quality
tax" (a coarser-than-necessary class layer loses information; an over-fine one
makes per-class statistics sparse). `config.h:127` records this directly:

```
c.n_classes = 256; c.relay_dim = 128;
/* ~sqrt(V): minimizes readout gather K*(C+V/C) AND the hierarchical quality tax */
```

`brain_macs_per_token` (`main.cu:1676`) charges the readout exactly as
`K*(C + S)` per layer.

### 8.3 Readout tensors and forward pass

Per layer: class head `Wc[G, C]`, word head `Ww[G, Gcols]` (class-major), biases
`bc[C]`, `bw[Gcols]`. Training-forward `k_readout_fwd` (`main.cu:341`) computes,
for one example, the full class logits over all `C` classes and the word logits
over the **target class's** `S` slots only (a cheap hierarchical path: the word
head is evaluated for one class, not all `C`). It emits the delta-rule errors and
the true-word logprob:

```
errc[e,c] = softmax(lc)[c] - onehot(target_class)[c]
errw[e,j] = softmax(lw)[j] - onehot(target_slot)[j]
logp[e]   = (lc[ce] - lse_c) + (lw[slot] - lse_w)
```

Eval-forward `k_readout_logp` (`main.cu:528`) is identical but writes only
`logp[e]` (no error, no scatter). Generation uses full per-class distributions
(`k_gen_class` `main.cu:792`, `k_gen_within` `main.cu:813`).

### 8.4 Numerical constraint: fp32 word head

The delta-rule updates to `Ww` are on the order of `3.7e-5`. A bf16 word-head
master rounds those increments to zero, so the word head **must** remain fp32.
The class head and biases are also fp32. Only the activation GEMM and the relay
inputs run in fp16. (`ww_fp16` exists in `config.h` for inter-rank transport, not
for the master copy.)

---

## 9. Local learning rules (no backprop)

No gradient is computed anywhere. Three local mechanisms account for all learning.

### 9.1 Granule weights — competitive k-means under WTA

`gwt` is tuned by online k-means restricted to the WTA winners (`brain_competitive`,
`main.cu:1604`). For each fired granule, `k_comp_accum` (`main.cu:686`) accumulates
the sum of the inputs that activated it; `k_comp_apply` (`main.cu:700`) moves each
weight toward the per-granule mean and renormalizes the row:

```
w += eta * (mean_of_activating_inputs - w);   then  row *= sqrt(fan)/||row||
```

After each update the dense `projT` and its fp16 mirror are rebuilt
(`main.cu:1628`). This runs off the hot path: an initial pass (`feat_passes=2`,
`feat_sample=40000`, `feat_eta=0.05`) plus periodic refines (`refine_every=50`
blocks, `refine_sample=4000`, `refine_eta=0.01`). It is skipped entirely when a
checkpoint is loaded (`main.cu:1862`).

### 9.2 Readout — local delta rule

The readout is trained by the delta (perceptron/LMS) rule: `error = softmax -
onehot`, scattered into the rows of `Wc` / `Ww` belonging to the active granules,
weighted by the granule margin. There is no backward pass; the error is a direct
local difference at the output layer.

Two scatter paths produce identical math:

- **Atomic** (`k_readout_scatter`, `main.cu:501`): one block per example, `atomicAdd`
  absorbs collisions when granules are shared across the chunk.
- **Bucketed, atomic-free** (default, `fast_scatter=1`): a counting sort assigns
  each `(example, kslot)` pair to its granule bucket. `k_bucket_count`
  (`main.cu:617`) counts, `k_prefix_sum` (`main.cu:742`) builds offsets on-device
  (no host round-trip), `k_bucket_fill` (`main.cu:622`) places pairs, then
  `k_scatter_class_bucketed` (`main.cu:633`) and `k_scatter_word_bucketed`
  (`main.cu:651`) run one block per granule so each block owns its `Wc`/`Ww` row
  and needs no atomics on the dominant payload.

Biases are updated by `k_bias_update` (`main.cu:670`) with atomics on the tiny
`bc`/`bw` arrays. The sampled-softmax word head (`--neg-word`, Section 10) uses
`k_readout_fwd_negw` (`main.cu:418`), `k_scatter_word_negw` (`main.cu:469`) and
`k_bias_class` (`main.cu:488`).

#### Batch-invariant step size

The per-step learning increment is normalized by a fixed reference batch, not the
actual batch (`brain_step`, `main.cu:1290`):

```
const float LR_REF = 8192.0f;
float st = c.lr * b->lr_scale / LR_REF;   /* lr_scale anneals over the run */
```

Earlier the step was `lr/B`, which made a larger batch take the same-sized move
across fewer steps, so a larger batch meant less total learning. Normalizing by
`LR_REF=8192` makes a larger batch take a proportionally larger step, so total
learning per token is constant and batch becomes a pure speed knob (the linear
scaling rule), commit `ac5179e`. `lr_scale` is the cosine-anneal multiplier driven
by `--lr-final` (`main.cu:1913`), the fix for the "more data hurts" regression
(commit `0e542be`). Weight decay (`brain_decay`, `main.cu:1375`) multiplies the
readout by `keep = (1 - lr*wd)^decay_every` every `decay_every` steps.

### 9.3 Mixture — local EM rule

The mixture logits are updated on-device by an EM-style rule (`k_mixture_update`,
`main.cu:752`): compute each layer's posterior responsibility for the observed
token, then move each logit toward `responsibility - prior`. The update lives on
the device (`dmix`) so the training hot loop has no per-step host sync.

```
mix[l] += mix_lr * ( mean_responsibility[l] - softmax(mix)[l] )
```

---

## 10. Stage 7 — deep parallel-expert stack and learned mixture

`L` cerebellar layers are trained, each with its **own** next-token objective
(deep supervision): every layer runs its own readout forward, error, and scatter
in the per-layer loop of `brain_step` (`main.cu:1299`). No gradient or error flows
between layers; the relay (Section 7) is a forward-only feature transform. The
layers are **parallel experts**, not a shared trunk.

Predictions are combined by a learned probabilistic mixture:

```
P(y) = sum_{l=0}^{L-1} w_l * P_l(y),   w = softmax(mix)
```

The mixture weights are the only cross-layer trained quantity, and they are
updated locally (Section 9.3). At eval the per-layer logprobs are combined in log
space (`brain_eval_ppl`, `main.cu:1387`). Because credit is assigned per layer and
combined only at the output, the architecture cannot perform the compositional
credit assignment that backprop through a shared trunk provides; this is the
structural source of the perplexity gap.

Optional residual boosting (`--boost`, `k_boost_weight` `main.cu:783`) reweights
each example for layer `l>0` by `1 - p_prev` (floored at `boost_wmin`) so deeper
layers specialize on what shallower layers missed. It is the codebase's
NoProp-style residual cascade and is off by default.

The layer count is clamped to 8 (`main.cu:2101`); several device structures
hard-size arrays to 8 layers (`k_mixture_update` shared arrays `w[8]`/`rs[8]`,
eval `lw[8]`).

---

## 11. Test-time adaptation ("hyper-fixation")

The differentiated axis of the model. During evaluation, the delta rule keeps
running on the document being read: each chunk is **predicted first** (scored for
perplexity with the current weights, so the scored token is never seen by the
update — no leakage) and **then learned from** at `adapt_lr` (`brain_eval_adapt`,
`main.cu:1455`). The frozen baseline LM cannot do this. Best measured result on
the 4-layer multiscale static config: 3412 -> 1930 perplexity (-43%). The
adaptive pass mutates the readout (`readout_scatter` at `main.cu:1493`) and the
mixture; the static pass (`brain_eval_ppl`) does not mutate and is always run
first.

---

## 12. Homeostatic load-balancing (scaffolded, dropped)

`gbias[g]` is a per-granule selection bias subtracted in the activation. A
DeepSeek aux-loss-free controller (`k_balance_apply`, `main.cu:598`) nudges it
toward an even fire-rate (`target = K/G`): an over-firing granule gets a higher
bias (suppressed), a starved one a lower bias (boosted) — pure negative feedback,
no gradient. It reduced usage-Gini from 0.348 to 0.251 but made perplexity
slightly worse (3470/2037 vs 3412/1930) and was dropped: the dead-granule fraction
was 0.0%, so there was no wasted capacity to recover, and the Gini reflected
useful specialization (language is correlated). The controller remains in the code
behind `--balance` (commit `4683af4`); by default `balance=0` and `gbias` stays
zero. It is auto-disabled under `--gran-frac<1` because subsampling biases the
fire-rate denominator (`main.cu:1896`).

---

## 13. Data shapes and device-resident state

Per-layer trained/fixed tensors (`struct Layer`, `main.cu:836`):

| Tensor | Shape | Precision | Trained by |
|---|---|---|---|
| `gidx` | `[G, fan_in]` | int | fixed (wiring) |
| `gwt` | `[G, fan_in]` | fp32 | competitive k-means |
| `gbias` | `[G]` | fp32 | balance controller (default 0) |
| `projT` | `[G, in_dim]` | fp32 | rebuilt from `gwt` |
| `projT_h` | `[G, in_dim]` | fp16 | mirror of `projT` |
| `Wc` | `[G, C]` | fp32 | delta rule |
| `Ww` | `[G, Gcols]` | fp32 (mandatory) | delta rule |
| `bc` / `bw` | `[C]` / `[Gcols]` | fp32 | delta rule + unigram init |
| `relayR` | `[G, relay_dim]` | fp32 | fixed (JL) |

Shared front-end (`struct Brain`, `main.cu:851`): `codes[V,D]`, `pos[ctx,D]`,
`decay[n_scales,ctx]`, `w2class[V]`, `w2slot[V]`, `class_word[C*S]`, host/device
mixture logits `mix[L]`/`dmix[L]`. Per-step transients are sized for `step_chunk`
(`brain_init`, `main.cu:1039`): `dX[chunk,ctx]`, `dY[chunk]`, `dbound[chunk,D]`
(or `dboundL[L,chunk,D]` under multiscale), `dact[chunk,G]` (fp16),
`dxh[chunk,relay_dim+D]` (fp16), `didx[l][chunk,K]`, `dval[l][chunk,K]`,
`derrc[chunk,C]`, `derrw[chunk,S]`, `dlogp[L,chunk]`, plus bucketed-scatter
scratch (`gcount[G]`, `goff[G+1]`, `gcursor[G]`, `bk_e/bk_k[chunk*K]`) and
competitive scratch (`csum[G*fan_in]`, `ccount[G]`).

### Checkpoint/resume

`save_brain` (`main.cu:1761`) writes only the learned state — `gwt`, `gbias`,
`Wc`, `Ww`, `bc`, `bw`, mixture — roughly 0.4 GB (`Ww` dominates). The
fixed-random front-end and the frequency hierarchy are regenerated from the seed
by `brain_init`, and `projT`/`projT_h` are rebuilt from `gwt` on load. The header
(magic `BFf1`, then `V, G, K, C, S, Gcols, relay_dim, L, fan_in`) is validated
against the live config and corpus before loading (`load_brain`, `main.cu:1782`);
a loaded checkpoint skips competitive feature learning. The feature exists for the
12-hour Kaggle session limit (commits in the v15 line).

---

## 14. Quality configuration and reference numbers

The defaults are speed-first (`L=2`, large `n_classes`). The quality config is:

```
--n-layers 4 --n-classes 256 --multiscale 1 --neg-word 8 --adapt 1
```

Measured on Kaggle dual T4, WikiText-103 (window 512 strided where comparable),
4-layer multiscale static config at 50M training tokens:

| Configuration | Static ppl | Adaptive ppl |
|---|---|---|
| 4-layer multiscale (50M tokens) | 3412 | 1930 |
| same config at 500M tokens ("more data hurts") | 3703 | 2057 |
| + infini-gram interpolation, 50M index, lambda=0.3 | 1272 | 762 |

The infini-gram interpolation (`P = (1-lambda)*P_brain + lambda*P_ngram`, scored
in one pass over the lambda sweep in `brain_eval_ppl`/`brain_eval_adapt`) is the
single largest realized quality jump, but it is an in-distribution gain (a long
suffix must match the corpus) and does not transfer to novel conversation. The
baseline for comparison is EleutherAI/Pythia-410M at WikiText-103 strided
perplexity 17.19.

Honest ceiling: the gap from ~2000 to ~17 is **structural**. The no-backprop
parallel-expert design lacks compositional credit assignment, and no data-side or
grafted technique observed in the program crosses it. The realistic ceiling for
the pure model is low hundreds of perplexity on in-distribution text, not
coherent open-ended generation. This is the reason the chatbot (Zell) adopts a
backprop-trained coherence core and retains only the BrainFormer inference-time
augmentations.

---

## 15. Throughput and bottleneck (engineering summary)

Single CUDA translation unit (`main.cu`), `sm_75`. Throughput rose from ~44k
tok/s (v13) to ~334k tok/s peak (v3); the 4-layer config runs ~171k tok/s (50M
tokens in ~5 min). Per-phase profile for the 4-layer config (from the lag-free
profiling burst, `brain_profile` `main.cu:1339`): bind 1.1%, activate ~21%,
topk ~24%, relay 6%, readout-fwd ~12%, scatter ~33.5%, bias 0.7%, mixture 0.9%.
The system is HBM-bandwidth-bound on the fp32 word-head scatter plus the all-`G`
activation read; `n_classes ~ sqrt(V)` minimizes the readout cost. The mixture
update and analytics run without per-step host sync; per-phase timing fires only
during the post-training profiling burst (`timing_on`).
