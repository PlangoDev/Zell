# Test-Time Adaptation (Hyper-fixation)

## 1. Scope and claim

Test-time adaptation is the mechanism by which BrainFormer continues to apply its
local **delta rule** during evaluation, learning from each document as it is read.
The model is not frozen at inference. As it scores a held-out stream it specializes
its readout weights to the statistics of that specific stream. The internal name for
this behavior is **hyper-fixation**: the model fixates on the document in front of it.

This is the differentiated axis of the project. The claim is narrow and is stated as
an adaptation property, not a static-quality property:

- A frozen backprop-trained baseline language model produces one perplexity number on
  a held-out stream and cannot improve on that stream without an offline gradient
  step. BrainFormer's learning rule is local, cheap, and gradient-free, so it can be
  run forward during evaluation at negligible incremental cost.
- On the 4-layer multiscale configuration at the 50M-token training budget, test-time
  adaptation moved WikiText-103 perplexity from **3412 (static) to 1930 (adaptive)**, a
  **-43%** reduction, in a single deterministic pass over the test stream.
- This does not close the structural gap to the Pythia-410M baseline (WikiText-103
  strided perplexity **17.19**, window 512, measured by `run_llm.py` on
  EleutherAI/Pythia-410M, 405,334,016 params). The adaptive number is a property of the
  no-backprop model relative to its own static number; it is not a competitive static
  perplexity claim.

The mechanism is the conceptual carryover into the Zell hybrid (Section 7): the
no-backprop cerebellar model is retired as the coherence core, but the test-time
adaptation idea survives as (a) RAG in-context session memory framed as a product
feature and (b) dynamic-evaluation SGD as a benchmark-mode capability.

## 2. Definition and no-leakage guarantee

Adaptation proceeds in **predict-then-learn** order on a per-chunk basis. For each
chunk of the test stream the model performs two operations in sequence:

1. **Predict.** Compute the per-layer true-word log-probability `logp` for every
   example in the chunk under the *current* weights. This log-probability is what is
   accumulated into the negative log-likelihood used for perplexity.
2. **Learn.** Apply the delta rule (the same readout scatter used during training) to
   move the readout weights toward the just-observed targets.

The token is **scored before the update that uses it**. The delta-rule scatter for a
chunk runs only after `logp` for that chunk has been captured. Therefore no example is
ever scored using a weight state that has already seen its own target. There is no
label leakage. The ordering is enforced structurally in code, not by a flag.

Within a chunk the forward pass and the scatter are interleaved per layer, but the
forward (`k_readout_fwd`) always precedes `readout_scatter` for the same layer and the
same chunk, and the per-layer `dlogp` rows are read back to the host only after all
layers have run. The host-side perplexity accumulation in `brain_eval_adapt`
(`main.cu:1495`) consumes `hlogp`, which is the **pre-update** prediction copied at
`main.cu:1495` immediately following the per-layer learn loop. Each `logp` value in
that buffer was written by the forward call at `main.cu:1486` before the scatter at
`main.cu:1493` modified any weight.

## 3. Implementation

The adaptive evaluator is `brain_eval_adapt` (`test_013_v3/src/main.cu:1455`). Its
signature mirrors the static evaluator so the two can be compared directly:

```c
static double brain_eval_adapt(Brain *b, const uint16_t *stream, long n,
                               float adapt_lr, double *per_layer_ppl,
                               const Ngram *ng, const float *lams, int nlam,
                               double *out_ppl);
```

### 3.1 Per-chunk loop

The stream is walked in contiguous chunks of `step_chunk` examples. For each chunk
(`main.cu:1465`):

1. **Window gather (host).** For each example `e` and position `p`,
   `hX[e,p] = stream[base + e + p]` and `hY[e] = stream[base + e + ctx]`
   (`main.cu:1468`). Windows are read in document order; this is what makes the
   adaptation see the document as a coherent sequence rather than a shuffled batch.
2. **Encode.** `encode_stack(b, bc)` runs the full forward front-end (context bind,
   granule activation, kWTA top-K, relay) for all layers, producing the per-layer
   sparse codes `didx[l]`, `dval[l]` (`main.cu:1474`).
3. **Read current mixture weights.** The mixture logits `dmix` are copied to host and
   softmaxed into `w[l]` (`main.cu:1476`). These are the *current* (adapting) mixture
   weights, because `dmix` is updated on-device at the end of each chunk.
4. **Per-layer predict-then-learn** (`main.cu:1483`):
   ```c
   for (int l = 0; l < L; l++) {
       k_readout_fwd<<<bc, 256, (C+S)*sizeof(float)>>>(...,
               b->dlogp + (size_t)l * bc, ...);   /* capture logp[l] (pre-update) */
       k_fill_ones<<<...>>>(b->dweight, bc);       /* boost weight = 1 (no boosting) */
       readout_scatter(b, ly, l, bc, st);          /* delta-rule update from this chunk */
   }
   ```
   `k_readout_fwd` (`main.cu:341`) computes class logits over all `C` classes and word
   logits over the target class's `S` slots, writes the delta-rule error
   `errc = softmax(lc) - onehot(class)` and `errw = softmax(lw) - onehot(slot)`, and
   writes `logp[e] = (lc[class] - lse_c) + (lw[slot] - lse_w)`. The scatter then
   subtracts `st * error * val` into the active top-K granule rows of `Wc` and `Ww` and
   updates the biases (`readout_scatter`, `main.cu:1203`).
5. **Read predictions, accumulate NLL** (`main.cu:1495`). `dlogp` (all layers, this
   chunk) is copied to `hlogp`. For each example the per-layer log-probabilities are
   combined under the mixture `P(y) = sum_l w_l P_l(y)` via a log-sum-exp
   (`main.cu:1496`), and `-log P_brain(y)` is added to `nll`. Per-layer NLL is also
   accumulated so per-layer adaptive perplexity is reported.
6. **Mixture EM update** (`main.cu:1516`). `k_mixture_update` runs on-device on `dmix`
   using this chunk's `dlogp`, so the mixture weights themselves also adapt to the
   document. This is the only adaptation of the mixture; it uses the same local EM rule
   as training.

### 3.2 Learning-rate convention

The adaptive step size is set per chunk at `main.cu:1481`:

```c
float st = adapt_lr / (float)bc;
```

This is the simple `lr / B` convention, normalizing the summed delta-rule update by the
chunk size `bc`. It is deliberately distinct from the batch-invariant training step
`st = lr * lr_scale / LR_REF` with `LR_REF = 8192` (`main.cu:1290`, commit `ac5179e`).
During training the batch-invariant rule makes batch a pure speed knob; during
adaptation the chunk is a unit of document context, so per-chunk normalization is the
intended semantics — the model takes one bounded step per observed chunk regardless of
chunk width. `adapt_lr` is the only learning-rate knob exposed for adaptation.

### 3.3 Defaults and flags

| Flag | Field (`config.h`) | Default | Meaning |
|---|---|---|---|
| `--adapt` | `adapt` | `0` | `1` runs an additional adaptive eval pass after static eval |
| `--adapt-lr` | `adapt_lr` | `0.1` | delta-rule step for the test-time update (`st = adapt_lr / bc`) |

Defaults are set at `config.h:140` (`c.adapt = 0; c.adapt_lr = 0.1f;`) and parsed at
`config.h:205`–`206`. The documented quality configuration is 4 layers, `n_classes 256`,
`--multiscale 1 --neg-word 8 --adapt 1`.

## 4. Contrast with the static evaluator

The static evaluator `brain_eval_ppl` (`main.cu:1387`) and the adaptive evaluator
`brain_eval_adapt` (`main.cu:1455`) share the chunked stream walk, the mixture
log-sum-exp, and the per-layer/per-lambda accumulation. They differ in exactly the
following way:

| Aspect | Static (`brain_eval_ppl`) | Adaptive (`brain_eval_adapt`) |
|---|---|---|
| Per-layer kernel | `k_readout_logp` (logp only, no error) | `k_readout_fwd` (logp + delta-rule error) |
| Weight mutation | none | `readout_scatter` after each chunk |
| Mixture | fixed at the eval-snapshot `mix` | `k_mixture_update` on `dmix` each chunk |
| Step size | n/a | `st = adapt_lr / bc` |
| Scoring order | predict only | predict (scored), then learn |

The static path calls `k_readout_logp` (`main.cu:528`), a forward that writes only
`logp[e]` and performs no scatter. The adaptive path calls `k_readout_fwd`, the same
forward augmented to also emit the delta-rule error vectors, and then runs the scatter.
Because the only weight-changing step is the post-scoring scatter, swapping the kernel
and adding the scatter is the entire difference between a frozen pass and an adapting
pass. The static pass at `main.cu:2005` runs first and does not mutate state, so the
adaptive pass at `main.cu:2024` starts from the same trained weights the static pass
reported.

## 5. Why a frozen baseline LM cannot match this axis

A standard backprop-trained transformer is frozen at inference for two reasons that
together make per-document adaptation impractical in the same regime:

1. **No local learning rule.** The transformer learns by backpropagation through the
   full computational graph. There is no per-token local update that can be run forward
   during evaluation. Any adaptation requires constructing and applying a gradient,
   which means a backward pass over the network for every adapted token — the cost of a
   training step, not an inference step.
2. **Coupled credit assignment.** Updating a backprop model on the document being read
   perturbs a single shared parameter tensor through which all predictions flow, so an
   in-place update is a controlled fine-tune with its own stability and catastrophic-
   forgetting concerns. BrainFormer's readout update is a rank-structured scatter into
   the rows of `Wc`/`Ww` indexed by the active top-K granules; it is local to the
   firing pattern and adds the same arithmetic the training step already performs.

BrainFormer's design — fixed random front-end, sparse k-winners-take-all codes, and a
delta-rule readout — makes the forward learning step cheap and bounded. The adaptation
is therefore not a bolt-on; it is the same machinery the model uses to train, run one
more time, in order, over the test document. The frozen baseline reports `17.19` and
stays there. The phrase logged at `main.cu:2030`, "LLM is frozen at this number," names
exactly this asymmetry. The asymmetry is the axis; the absolute perplexities are not
comparable across it.

## 6. Relationship to dynamic evaluation (Krause)

The mechanism is an instance of **dynamic evaluation** in the sense of Krause et al.:
continue to adapt model parameters on the evaluation stream, scoring each token before
updating on it, so that the model exploits local repetition and topical drift in the
test document. BrainFormer differs from the classical formulation in the credit-
assignment substrate:

- Classical dynamic evaluation performs SGD (a gradient step) on the language model's
  own parameters between scored tokens.
- BrainFormer performs a **gradient-free local delta-rule** update on the readout only;
  the fixed-random front-end and the kWTA codes are never adapted. The update is a
  scatter into the active granule rows, not a gradient through the network.

The two share the predict-then-learn protocol and the no-leakage guarantee. The
distinction matters for Zell: the dynamic-evaluation SGD form (Section 7) is the
faithful Krause mechanism applied to the backprop core, and the delta-rule form here is
the gradient-free analogue native to the no-backprop model. Both are benchmark-time, in-
distribution-text mechanisms; neither is claimed to help on novel open-ended chat where
there is no local repetition to fixate on.

## 7. Planned Zell forms

The Zell pivot (recorded June 2026) is a hybrid: a real backprop-trained efficient
transformer (full fine-tune of Qwen/Qwen2.5-0.5B-Instruct) is the coherence core, and
BrainFormer's inference-time mechanisms are augmentations on top of it. Test-time
adaptation carries over in two forms, scheduled as milestone M5:

### 7.1 M5a — RAG in-context session memory (product framing)

In-context session memory: as a session proceeds, content is retrieved and placed into
the context window so the core conditions on what it has already seen in the session.
This is adaptation without a weight update — the "learning" lives in retrieved context.
The product framing is that Zell "keeps learning during your session." This is the
default, always-on form because it carries no training-time risk and no per-token
gradient cost.

### 7.2 M5b — Dynamic-evaluation SGD (benchmark mode)

The faithful Krause dynamic-evaluation mechanism applied to the core: a small SGD step
on a restricted parameter subset of the transformer, run on the evaluation stream in
predict-then-learn order. This is gated to **benchmark mode** because it mutates model
weights and is appropriate for measuring in-distribution perplexity, not for serving
open-ended chat. Expected effect: WikiText-103 perplexity in the ~9–12 range at M5b,
versus ~10–13 at M4 (infini-gram rebuilt in the Qwen tokenizer) and the first sub-17.19
win at M1.

### 7.3 Honest framing carried into Zell

The same scope limit that applies to the no-backprop model applies to the Zell forms:
the adaptation and infini-gram gains are real on corpus-like / in-distribution text
where the document repeats and a long suffix matches, and are near-zero on novel chat
inputs where there is nothing to fixate on. Test-time adaptation is an adaptation axis,
not a static-quality claim, in both the BrainFormer measurements and the Zell plan.

## 8. Measured results

All figures below are from deterministic single-pass evaluation on Kaggle dual NVIDIA
T4 GPUs against WikiText-103. The configuration is 4-layer multiscale unless noted.

| Configuration | Static ppl | Adaptive ppl | Change |
|---|---|---|---|
| 4-layer multiscale, 50M-token train budget | 3412 | 1930 | -43% |
| 4-layer multiscale, 500M-token train budget | 3703 | 2057 | worse than 50M |

The 50M result (`3412 -> 1930`) is the headline adaptation number and the model's
differentiated axis. The 500M regression — a *higher* static and adaptive perplexity at
a larger training budget — is the "more data hurts" finding: a constant learning rate
over-trains and the weight-decay schedule (`keep = (1 - lr*wd)^decay_every` per event)
over-regularizes at scale. It is deterministic, not noise. The fix (commit `0e542be`)
exposed `--lr-final` for cosine annealing of `lr` plus `--wd` and `--decay-every`. The
regression is recorded here because it sets the budget at which the `3412 / 1930` pair
was obtained; the adaptive mechanism itself is unchanged across budgets.

### 8.1 Composition with infini-gram

Because adaptation mutates the readout in a lambda-independent way, `brain_eval_adapt`
scores the bare adaptive model and every infini-gram interpolation weight in a single
pass (`main.cu:1507`–`1514`): for each example it forms `P_brain` from the adapted
mixture, queries the suffix-array next-token estimate `P_ngram`, and accumulates
`-log((1-lambda) P_brain + lambda P_ngram)` for each lambda in the sweep
`{0.1, 0.2, 0.3, 0.5, 0.7, 0.9}` plus any custom value (`main.cu:1999`). The
interpolation result is reported at `main.cu:2034`. The standalone infini-gram result
(50M index, lambda = 0.3) was static `3412 -> 1272` (-63%) and adaptive `1930 -> 762`
(-60%); the same in-distribution scope caveat applies — `762` does not transfer to
conversation.

## 9. Source references

| Element | Location |
|---|---|
| Adaptive evaluator (predict-then-learn) | `test_013_v3/src/main.cu:1455` |
| Pre-update `logp` capture | `test_013_v3/src/main.cu:1486`, copied at `:1495` |
| Per-chunk step size `st = adapt_lr / bc` | `test_013_v3/src/main.cu:1481` |
| Delta-rule readout scatter | `test_013_v3/src/main.cu:1203` |
| Readout forward + error (`k_readout_fwd`) | `test_013_v3/src/main.cu:341` |
| Static evaluator (no mutation) | `test_013_v3/src/main.cu:1387` |
| Logp-only forward (`k_readout_logp`) | `test_013_v3/src/main.cu:528` |
| On-device mixture EM update | `test_013_v3/src/main.cu:752`, called at `:1516` |
| Eval driver (static then adaptive) | `test_013_v3/src/main.cu:2005`, `:2024` |
| "LLM is frozen at this number" log line | `test_013_v3/src/main.cu:2030` |
| Flags `--adapt` / `--adapt-lr` | `test_013_v3/src/config.h:93`–`94`, `:205`–`206` |
| Defaults `adapt=0`, `adapt_lr=0.1` | `test_013_v3/src/config.h:140` |
