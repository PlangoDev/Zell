# Novel Contributions

This document is an intellectual-property-oriented enumeration of the distinct technical
contributions of the BrainFormer program (the no-backprop cerebellar language model) and
the Zell hybrid system. Each entry states the **mechanism**, the **closest prior art**,
and the **specific distinction**, and labels the contribution as either a *novel mechanism*
(a learning rule, kernel, or data path not found in the cited prior art) or a *novel
combination* (a configuration of known parts whose conjunction is the contribution).

The work was conducted in June 2026 on Kaggle dual NVIDIA T4 GPUs. Source references are to
`test_013_v3/src/` unless noted. Perplexity figures are WikiText-103 strided perplexity
(window 512), the same protocol under which `run_llm.py` measured the Pythia-410M baseline
at 17.19.

Scope note carried through every claim below: the perplexity gap from the no-backprop
model's operating range (low thousands, falling to the 760–1930 band with adaptation and
retrieval) to the 17.19 baseline is **structural**, attributable to the parallel-expert
mixture, fixed random codes, and linear readout. None of the contributions below crosses
that gap on novel open-ended text. The contributions are claimed on their own terms:
gradient-free training feasibility, an inference-time plasticity axis the frozen baseline
lacks, retrieval gains on in-distribution text, and a class of GPU kernels for local
learning rules.

---

## Contribution index

| # | Contribution | Type | Primary source |
|---|---|---|---|
| C1 | No-backprop deep parallel-expert cerebellar stack with per-layer deep supervision and a learned EM mixture | Novel combination | `main.cu`, `config.h` |
| C2 | Multi-timescale per-layer context decay producing layer divergence | Novel combination | `main.cu:75`, `main.cu:1090` |
| C3 | Test-time delta-rule adaptation ("hyper-fixation") as an inference-time plasticity axis | Novel combination | `main.cu:1455` |
| C4 | Atomic-free bucketed delta-rule scatter with device-side prefix sum | Novel mechanism | `main.cu:613`–`main.cu:668` |
| C5 | n_classes ≈ √V hierarchical readout cost-minimization for a local-rule head | Novel combination | `main.cu`, `config.h:127` |
| C6 | Batch-invariant local-rule step (LR_REF normalization) | Novel mechanism (narrow) | `main.cu:1290` |
| C7 | Infini-gram × sparse-model × test-time-adaptation hybrid eval | Novel combination | `ngram.h`, `main.cu:1431` |
| C8 | On-device EM mixture-of-experts update with no host sync | Novel mechanism (narrow) | `main.cu:752` |
| C9 | Zell: real efficient transformer core with BrainFormer mechanisms as inference-time augmentations | Novel combination | program-level |

---

## C1 — No-backprop deep parallel-expert cerebellar stack with per-layer deep supervision and a learned EM mixture

### Mechanism

The model is a stack of `n_layers` independent cerebellar layers (code default 2,
speed-first; the quality config uses 4). Each layer implements the Marr-Albus
granule-cell pipeline per token:

1. **Fixed random token codes** `codes[V, D]`, `D = code_dim` (default 128), never trained.
2. **Context binding** by hyperdimensional bind of token codes with position sign-codes and
   multi-scale exponential decay into one vector `bound[D]` (`k_bind`, `main.cu:53`).
3. **Granule expansion**: `G` granules (default 12288/layer), each with `fan_in` (default 48)
   sparse random input connections; granule weights `gwt` are set by competitive k-means
   under winner-take-all (`brain_competitive`), not by gradient. The dense projection
   `projT[G, in_dim]` is rebuilt from `gwt`+`gidx` (`k_build_projT`, `main.cu:40`).
4. **k-winners-take-all** sparse activation: the top-`K` (default 64) granules by
   `a = x·projTᵀ − gbias` fire (`k_topk_bitonic` default; `main.cu:212`).
5. **Relay** of the sparse code to the next layer by a fixed random Johnson-Lindenstrauss
   projection `relayR`, L2-normalized and concatenated with the next layer's context bind
   (`k_relay_concat`, `main.cu:304`).
6. **Hierarchical readout**: vocabulary `V` partitioned into `C` classes (default 256) ×
   `S` slots; class head `Wc[G, C]`, word head `Ww[G, Gcols]`, `Gcols = C·S`; prediction
   `P(word) = P(class)·P(word|class)` (`k_readout_fwd`, `main.cu:341`).

The defining structural choice (`main.cu:14`, `main.cu:8`): **every layer carries its own
next-token objective (deep supervision)**, and the per-layer predictions are combined by a
**learned probabilistic mixture** `P(y) = Σ_l w_l P_l(y)`. No gradient flows between layers.
The layers are parallel experts, not a shared-trunk hierarchy. Mixture logits are updated
on-device by a local EM-style rule (`k_mixture_update`, `main.cu:752`; see C8). Readout
weights are updated by a local delta rule (see C4). Granule weights are set by competition.
There is no backpropagation anywhere in the system.

Measured lineage (Kaggle 2×T4, WikiText-103): test_013 v1, a single learned layer, reached
perplexity 3284 / 3086 / 2680 at 50M / 100M / 500M tokens. v2 was a Python multi-file deep
version used as a perplexity oracle. v3 is the C/CUDA rewrite for throughput.

### Closest prior art

- **Marr-Albus / cerebellar models.** Marr (1969) and Albus (1971) describe the cerebellar
  cortex as a sparse expansion (granule cells) followed by a single trainable readout layer
  (Purkinje cells with a delta-rule synapse driven by climbing-fiber error). Modern
  instantiations (e.g. sparse-coding/perceptron treatments of cerebellum) keep the two-stage
  expand-then-readout structure.
- **Deeply-supervised nets** (Lee et al., 2015) attach auxiliary classifiers to intermediate
  layers of a backprop-trained network.
- **Mixture of experts** (Jacobs et al., 1991; Shazeer et al., 2017) combines specialist
  sub-models, typically with a learned gating network trained by backprop.

### Specific distinction

The contribution is a **novel combination**. Prior cerebellar models are single-readout and
single-stage; deeply-supervised nets and mixtures-of-experts are trained by backprop with a
shared trunk or a backprop-trained gate. The combination claimed here is:

1. a **deep stack of full Marr-Albus modules** (expand + readout per layer), where
2. each module is independently next-token-supervised (deep supervision **without** a shared
   trunk and **without** backprop to couple the modules), and
3. the modules are combined by a **mixture whose gate is itself fit by a local EM rule**, so
4. the entire system — granule features, per-layer readouts, and the gate — is trained by
   three distinct **local** rules (competition, delta, EM) and **zero** backpropagation.

No prior cerebellar, deeply-supervised, or mixture-of-experts system is trained end-to-end
without backprop in this configuration. The parallel-expert framing also constrains what can
be grafted: multi-token prediction does not fit, because there is no shared trunk and no
backprop through which an auxiliary head's benefit could propagate to the shared
representation (`v15.md:71`).

---

## C2 — Multi-timescale per-layer context decay producing layer divergence

### Mechanism

With `--multiscale 1`, layer `l` binds context at decay rate `decay0 / 2^l`
(`main.cu:1092`: `float rate = b->cfg.decay0 / (float)(1 << l)`), so deeper layers integrate
longer-range context. The per-layer bind uses a single-rate exponential kernel
`bound[e,d] = Σ_p exp(−dist·rate)·codes[X[e,p],d]·pos[p,d]` (`k_bind_rate`, `main.cu:75`),
and the per-layer binds are materialized into `dboundL[L, chunk, D]` (`main.cu:1044`) so each
layer's readout sees a context representation at its own timescale.

This converts the stack from a set of near-identical experts into a genuine temporal
hierarchy. Measured effect (commit `d5177d0`): static perplexity 3858 → 3478 (−10%), and the
layers diverged — per-layer perplexities went from near-tied (4076 / 4081) to separated
(L0 3539 / L1 3919). The mixture then has distinct experts to combine rather than redundant
copies.

A separate control experiment refuted the hypothesis that the **context representation** is
the quality wall: increasing `code_dim` from 128 to 512 gained only ~4% perplexity. The wall
is structural (C1), not a binding-capacity limit.

### Closest prior art

- **Multi-timescale / multi-rate recurrent models** (e.g. Clockwork RNN, Koutník et al.,
  2014; hierarchical multiscale RNNs) assign different update rates to different hidden
  groups in a backprop-trained recurrent network.
- **Exponentially-decayed context / leaky integration** appears in linear-attention and
  state-space families.

### Specific distinction

The contribution is a **novel combination**. The multi-rate idea (different timescales at
different depths) is prior art for backprop-trained recurrent nets. Here it is applied to a
**stack of parallel, independently-supervised, non-backprop experts**, where the per-layer
timescale is the mechanism that **forces the experts to specialize** (because each expert is
otherwise structurally identical and would converge to the same function). The measured
layer divergence — experts moving from tied to separated perplexity — is the specific,
verifiable consequence claimed. The novelty is using a fixed per-layer decay schedule as the
specialization driver in a system that has no gradient to differentiate the experts.

---

## C3 — Test-time delta-rule adaptation ("hyper-fixation")

### Mechanism

With `--adapt 1`, evaluation does not freeze the model. As the document is read in order, the
model **predicts** each chunk with its current weights and **then learns** from it with the
delta rule, at `adapt_lr`. The ordering is strict predict-then-learn and is the basis of the
no-leakage claim: in `brain_eval_adapt` (`main.cu:1455`) the forward kernel `k_readout_fwd`
writes `dlogp` (the per-layer true-word logprob used for scoring) **before** `readout_scatter`
applies the weight update for that same chunk (`main.cu:1486`–`main.cu:1493`), and the host
reads `hlogp` as the *pre-update* predictions (`main.cu:1495`, comment "pre-update
predictions"). The token is therefore scored on weights that have not yet seen it.

Measured effect: 4-layer multiscale, static perplexity 3412 → adaptive 1930 (−43%). This is
the model's differentiated axis: a frozen baseline language model cannot do this, because it
has no local learning rule to run at inference and no safe (leakage-free) way to update on the
text being scored.

Adaptation interacts with scale through the same saturation mechanism described in C6 / the
results record: at 500M tokens the constant-learning-rate, fixed-weight-decay configuration
regressed (static 3703 / adaptive 2057 versus 3412 / 1930 at 50M), which motivated the
cosine learning-rate anneal `--lr-final` and the CLI-exposed `--wd` / `--decay-every`
(commit `0e542be`).

### Closest prior art

- **Dynamic evaluation** (Mikolov et al., 2010; Krause et al., 2018, "Dynamic Evaluation of
  Neural Sequence Models"): continue **SGD/backprop** on the model during evaluation, adapting
  to the local text. This is the closest prior art and the direct ancestor of the idea.
- **Test-time training** (Sun et al., 2020) adapts a network at inference via a self-supervised
  auxiliary loss, by backprop.

### Specific distinction

The contribution is a **novel combination**. Dynamic evaluation and test-time training both
adapt at inference, but both do so by **backpropagation** of a loss into the model's weights.
The distinction here is that the inference-time adaptation is performed by a **local delta
rule** scattered into the active top-K granule rows of the readout (`readout_scatter`), with
no gradient and no backward pass — the same rule used in training, kept running at
eval. Two consequences follow:

1. Adaptation costs one extra scatter per chunk, not a full backward pass; it runs at
   training-throughput, not at backprop cost.
2. Because the granule front-end is fixed-random and only the linear readout adapts, the
   adaptation is a bounded, interpretable update of the output layer's class/word weights for
   exactly the granules that fired.

The product framing for Zell ("keeps learning during your session") rests on this axis. The
honest scope is stated plainly: the gain is real on the document being read; it does not by
itself produce coherent open-ended generation, and the adaptive number does not transfer to
novel chat where there is no in-distribution structure to over-fit to.

---

## C4 — Atomic-free bucketed delta-rule scatter with device-side prefix sum

### Mechanism

The readout learning rule is a **local delta rule**: `error = softmax − onehot`, scattered
into the active top-K granule rows of the class head `Wc` and word head `Ww`. The naive
implementation (`k_readout_scatter`, `main.cu:501`) accumulates with `atomicAdd` into `Wc`
and `Ww`, because many examples in a batch select the same granule row and would otherwise
race. On the word head — the dominant payload and the program's bandwidth bottleneck — those
atomics are expensive.

The atomic-free path (`main.cu:613`–`main.cu:668`) restructures the scatter as a counting
sort over granules:

1. `k_bucket_count` (`main.cu:617`) counts how many (example, k-slot) pairs select each
   granule.
2. `k_prefix_sum` (`main.cu:742`) computes the exclusive prefix sum `gcount[G] → goff[G+1]`
   **on the device** in a single thread, removing the host round-trip and sync that had made
   an earlier bucketed version slow.
3. `k_bucket_fill` (`main.cu:622`) places each (example, k-slot) pair into its granule's
   bucket using one `atomicAdd` on a small per-granule cursor (not on the weight payload).
4. `k_scatter_class_bucketed` / `k_scatter_word_bucketed` (`main.cu:633`, `main.cu:651`)
   launch **one block per granule**; each block owns row `g` exclusively, sums the
   contributions of all pairs in that granule's bucket, and writes the row with **no atomics
   on the weight matrix** (`Wc[g·C + c] -= st·acc`).

The result is that the dominant write traffic — the word-head update — is conflict-free.
Per-phase profiling of the 4-layer config attributes ~33.5% of runtime to the scatter,
confirming it as the bottleneck this kernel targets.

The patch was hardened by a 4-agent adversarial review (atomic-path accumulation correctness,
dual-GPU integrator reset, refine reset, u64 accumulator width, `gran_frac` guard, and a
relative-error control law for the related balance controller).

A numerical finding constrains the data type: a bf16 word-head master rounds the
~3.7e-5-magnitude delta-rule updates to zero, so the word head must stay fp32. The scatter is
therefore an fp32 write into the largest matrix, which is why it is HBM-bandwidth-bound.

### Closest prior art

- **Atomic-free / sort-based scatter and segmented reduction** (counting sort, segmented
  reduce) are standard GPU primitives (e.g. CUB, Thrust).
- **Sparse outer-product / embedding-gradient accumulation** in deep-learning frameworks uses
  either atomics or sort-by-key + segmented sum for backprop into embedding tables.

### Specific distinction

The contribution is a **novel mechanism** in the narrow sense that the kernel family is
designed for a **local delta rule on a kWTA sparse code**, not for a backprop gradient. The
structure exploited is specific to this model: the active set is exactly the top-K granules
per example, so the "keys" are kWTA indices, the buckets are granules, and the per-granule
block writes a full readout row that includes the **hierarchical** class/word partition
(`base = w2class[y]·S`, word-head writes only into the target class's column block,
`main.cu:651`). The device-side single-thread prefix sum over `G` (small, off the
bandwidth-bound path) deliberately trades a tiny serial cost to eliminate a host sync, which
matters because the rest of the step is engineered to have **no per-step host syncs** at all
(the mixture update also runs on-device, C8). The combination — counting sort keyed on kWTA,
device prefix sum to avoid host sync, one-block-per-granule conflict-free write of a
hierarchical readout row, fp32-forced because the update magnitude underflows bf16 — is not a
reuse of a stock embedding-gradient kernel.

---

## C5 — n_classes ≈ √V hierarchical readout cost-minimization for a local-rule head

### Mechanism

The vocabulary is factored as `V = C · S` (with `S = ceil(V/C)`), and prediction is
`P(word) = P(class)·P(word|class)`. The readout gather cost per example is `K·(C + V/C)`:
`K` granules contribute to `C` class logits and to `S = V/C` word logits within the target
class. Minimizing `C + V/C` over `C` gives `C = √V`. The default `n_classes = 256` is chosen
as ~√V for the Pythia-tokenizer vocabulary (`config.h:127`, comment "~sqrt(V): minimizes
readout gather K·(C+V/C) AND the hierarchical quality tax").

This sets both the **compute** cost (gather work) and the **memory** footprint of the word
head `Ww[G, C·S]`, and it is the lever that keeps the readout — already the bandwidth
bottleneck (C4) — as small as the factorization allows.

### Closest prior art

- **Hierarchical softmax / class-based language models** (Goodman, 2001; Morin & Bengio,
  2005) factor the output distribution into class and within-class terms to reduce softmax
  cost, with the same `√V` two-level optimum.

### Specific distinction

The contribution is a **novel combination**, and the document is explicit that the `√V`
factorization itself is prior art (class-based LMs). The distinction is the **setting**: the
two-level factorization is applied to a head trained by a **local delta rule scattered over a
kWTA sparse code on GPU**, where the factorization simultaneously (1) minimizes the gather
cost `K·(C + V/C)`, (2) bounds the size of the fp32 word matrix that the bandwidth-bound
scatter must write, and (3) shapes the per-granule bucketed scatter of C4 (the word-head
block per class). In a backprop softmax, `√V` is a softmax-cost optimization; here it is the
joint compute/bandwidth/learning-rule design point for a non-backprop sparse readout. The
contribution is the alignment of the classical optimum with the C4 kernel structure and the
fp32 constraint, not the optimum itself.

---

## C6 — Batch-invariant local-rule step (LR_REF normalization)

### Mechanism

The per-step weight update is `st = lr · lr_scale / LR_REF` with `LR_REF = 8192`
(`main.cu:1290`: `const float LR_REF = 8192.0f; float st = c.lr * b->lr_scale / LR_REF;`),
where `lr_scale` carries the cosine anneal across the run. Earlier code used `st = lr / B`
(divide by batch size), which made a larger batch produce a smaller effective update — i.e.
batch size silently changed the learning dynamics. Normalizing by a fixed reference instead
of the actual batch (commit `ac5179e`) makes the effective learning rate independent of batch
size, so batch becomes a pure speed knob. This is the linear-scaling rule expressed for a
local delta rule rather than for SGD.

### Closest prior art

- **Linear scaling rule** for minibatch SGD (Goyal et al., 2017): scale the learning rate
  with batch size so that large-batch training matches small-batch dynamics.

### Specific distinction

The contribution is a **novel mechanism in a narrow sense**: the linear-scaling principle is
prior art for backprop SGD, but here it is derived and applied to a **local delta-rule
accumulator** whose natural (buggy) normalization was `1/B`. The fix replaces an
accumulation-dependent normalization with a fixed reference `LR_REF`, decoupling the local
rule's step size from batch size. The contribution is the recognition that a local-rule
readout scatter, which sums per-example contributions, exhibits the same batch-coupling
pathology as un-scaled SGD, and the specific constant-reference normalization that fixes it
without changing the rule.

---

## C7 — Infini-gram × sparse-model × test-time-adaptation hybrid eval

### Mechanism

A suffix array over the corpus tokens (`tools/build_ngram.py`) supports, at each eval
position, the next-token distribution from the **longest matching suffix** of the context.
The host query (`ngram.h`) mmaps the token file and the suffix array, then for suffix lengths
from `Lmax` down to 1 binary-searches the longest suffix that occurs in the corpus
(`ng_lb`, `ngram.h:101`), counts next-token outcomes at the matching positions (with a strided
subsample capped at `scan_cap = 8192` for very frequent suffixes), and returns the Laplace-
floored estimate `(yes + 1e-6)/(tot + 1e-6·65536)` (`ngram_prob`, `ngram.h:114`). The C query
is validated byte-for-byte against a Python reference (`build_ngram.py --selftest`).

Eval interpolates `P = (1−λ)·P_brain + λ·P_ngram`. The interpolation is scored for **both**
the static and the adaptive (C3) brain in a **single pass**, for every `λ` in a sweep
simultaneously (`brain_eval_ppl` and `brain_eval_adapt`, `main.cu:1431`–`main.cu:1438` and
`main.cu:1507`–`main.cu:1514`): for each example the brain probability `pb` and the n-gram
probability `pg` are computed once, then `nll_lam[i] += −log((1−λᵢ)·pb + λᵢ·pg)` accumulates
all `λ` at once.

Result (50M index, λ = 0.3): static 3412 → 1272 (−63%); adaptive 1930 → 762 (−60%). This is
the single largest quality jump in the program. Honest scope: the gain is on corpus-like /
in-distribution text where a long suffix matches; for novel chat inputs there is no match and
the interpolation falls back to the bare model, so 762 does not transfer to conversation
(`v15.md:7`).

### Closest prior art

- **Infini-gram** (Liu et al., 2024): unbounded-`n` n-gram language modeling from a suffix
  array over a large corpus, with longest-suffix backoff; demonstrated to lower perplexity
  when interpolated with a neural LM.
- **kNN-LM** (Khandelwal et al., 2020) and **retrieval-augmented LMs** interpolate a neural
  LM with a nearest-neighbor / retrieved distribution.

### Specific distinction

The contribution is a **novel combination**, and the document credits infini-gram and
retrieval interpolation as the prior art. The distinct elements are:

1. **The base model is gradient-free.** Infini-gram's published interpolation result is with
   a backprop-trained neural LM. Here both the base model (a no-backprop cerebellar mixture)
   and the retrieval term are trained without backprop, so the **entire** scoring path
   contains no gradient learning.
2. **Two-axis interpolation in one pass.** The retrieval term is interpolated against the
   **test-time-adapted** brain (C3), not only the static brain, and both axes plus a full `λ`
   sweep are scored in a single eval pass by reusing one `pg` per example. The combination
   of inference-time plasticity (C3) with longest-suffix retrieval is the specific hybrid.
3. It is the empirical anchor for the program's honest ceiling: the retrieval gain is large
   and real on in-distribution text and near-zero on novel chat, which is the finding that
   drove the Zell pivot (C9).

---

## C8 — On-device EM mixture-of-experts update with no host sync

### Mechanism

The mixture gate over the `L` parallel experts is updated on-device by a local EM-style rule
(`k_mixture_update`, `main.cu:752`). One block, threads split over the examples of the chunk:
the current mixture weights `w_l = softmax(mix)` are formed; for each example the per-expert
responsibility is computed from `log w_l + logp[l,e]` and reduced (via atomics into shared
`rs[l]`); the gate logits are updated by `mix[l] += mix_lr·(rs[l]/bc − w_l)` (`main.cu:773`),
i.e. **responsibility minus prior**, the standard EM M-step direction for mixture weights.
The update runs entirely on the device — no per-step host synchronization — which is what
allows the training step to have no host syncs in the hot path (the readout uses the
device-prefix-sum scatter of C4 for the same reason).

### Closest prior art

- **EM for mixture models** (Dempster et al., 1977): the M-step for mixing coefficients is
  the mean responsibility, which the rule above tracks by a gradient-toward-responsibility
  step.
- **Mixture-of-experts gating** is conventionally a backprop-trained softmax gate.

### Specific distinction

The contribution is a **novel mechanism in a narrow sense**: EM for mixture weights is
classical, but it is applied here as the **gate of a non-backprop mixture-of-experts** whose
experts are independently supervised, and it is implemented as a **single-block, host-sync-
free CUDA kernel** that performs the responsibility reduction and the gate update on-device
each step. The contribution is the use of a local EM rule (rather than a backprop-trained
gate) to combine independently-trained sparse experts, fused into the no-sync training step.
Together with the competition rule (granule features) and the delta rule (readout), it
completes the system's set of three local learning rules with zero backpropagation (C1).

---

## C9 — Zell: real efficient transformer core with BrainFormer mechanisms as inference-time augmentations

### Mechanism

Because the perplexity gap is structural (stated throughout, `v15.md:7`), Zell does not use
the no-backprop model as the coherence core. The recorded design (June 2026) is a **hybrid**:
a real (backprop-trained) efficient transformer — a full fine-tune of `Qwen2.5-0.5B-Instruct`
(Apache-2.0, native Hermes `<tool_call>` tool-calling, ChatML) — serves as the coherence
core, and the BrainFormer mechanisms are retained as **inference-time augmentations**:

- **Infini-gram interpolation (C7)**, rebuilt in the core's tokenizer, with adaptive `λ`.
- **Test-time adaptation (C3)**, reframed as dynamic-evaluation SGD on a small parameter
  subset (benchmark mode) and as RAG in-context session memory (the "keeps learning during
  your session" product framing).

The single mandatory porting cost is the tokenizer switch from the Pythia-410M tokenizer to
the Qwen2.5 BPE tokenizer (~151k vocab), which requires the infini-gram index to be rebuilt
in the core's tokenizer and the data blend to be re-tokenized. Carryover artifacts:
`build_ngram.py` (re-tokenized) and `build_blend.py` (re-templated to Qwen ChatML); retired:
the no-backprop cerebellar model as the coherence core.

### Closest prior art

- **Retrieval-augmented generation** and **kNN-LM** (Khandelwal et al., 2020) augment a
  neural LM at inference with retrieval.
- **Dynamic evaluation** (Krause et al., 2018) adapts a backprop LM during evaluation.

### Specific distinction

The contribution is a **novel combination** at the system level: pairing a small, full-fine-
tuned, tool-calling instruction transformer with **two specific inference-time mechanisms
developed in the gradient-free program** — longest-suffix infini-gram interpolation (C7) and
local/dynamic test-time adaptation (C3) — under a single product. The distinguishing decision
is **scope honesty**: the design explicitly assigns the structural coherence work to the
backprop core and confines the BrainFormer mechanisms to where they are measured to help
(in-distribution perplexity), rather than overstating their transfer to open-ended chat. The
novelty is the integration pattern and the empirically-grounded boundary between the two,
not any single component.

---

## Negative results and non-contributions (for completeness)

These were investigated and are recorded as **not** contributions, to bound the claims above
and document the prior-art search.

| Item | Status | Reason |
|---|---|---|
| Homeostatic load-balancing (`--balance`, DeepSeek aux-loss-free bias controller on `gbias`, `main.cu:598`) | Implemented, dropped | Reduced usage-Gini 0.348 → 0.251 but perplexity got slightly worse (3470/2037 vs 3412/1930); dead-granule fraction was 0.0%, so there was no wasted capacity to recover and the Gini reflected useful specialization. |
| Multi-token prediction (DeepSeek MTP) | Not built | Layers are parallel experts, not a shared trunk; no backprop to share an auxiliary head's gain (`v15.md:71`). |
| DeltaNet true forgetting on the readout | Not built | The correct full-row decay roughly doubles the word-head scatter (the bottleneck) for marginal gain over `--wd` + `--lr-final` (`v15.md:75`). |
| IVF / coarse granule routing ("skip computing all G granules") | Not built | Fights the tensor-core dense GEMM; realistic outcome on T4 is ~break-even, not the >5× seen in non-tensor-core settings (`v15.md:77`). |
| Direct Feedback Alignment ("fake backprop") | Not adopted as core | Cross-layer credit it adds is conditioning-only; web-verified DFA-on-LM reached perplexity 93 vs 30 for backprop. `--boost` already provides a forward cross-layer cascade. |

The honest ceiling, stated as a limitation: the no-backprop parallel-expert design lacks
compositional credit assignment, so no data-side or grafted technique crosses the perplexity
gap to the 17.19 baseline on novel text. The realistic ceiling for the pure model is low
hundreds of perplexity on in-distribution text, not coherent open-ended generation. The
contributions above are claimed within that boundary.
