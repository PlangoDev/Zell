# 01 — Thesis and Motivation

Status: permanent technical and IP record. Project: BrainFormer (the no-backprop cerebellar language model) and Zell (the hybrid chatbot that succeeds it). Repository: PlangoDev/plango-labs. Period of work: June 2026, Kaggle dual NVIDIA T4 GPUs. This document records the problem the program set out to attack, the neuroscience hypothesis that defined the architecture, the efficiency thesis and its original numerical targets, the test_013 experiment lineage (v1/v2/v3), and the empirical finding — a structural perplexity ceiling — that motivated the pivot to the Zell hybrid.

---

## 1. The problem: transformer compute and the efficiency wall

The dominant language-model design couples next-token quality to two scaling axes that both grow super-linearly in cost: parameter count and training compute (FLOPs over tokens). Quality is delivered by dense self-attention and dense feed-forward blocks trained end-to-end by backpropagation. Three properties of that design set the cost floor the program treated as the target to undercut.

1. **Dense activation.** A standard transformer activates the full parameter set for every token. There is no per-token sparsity in the base architecture; mixture-of-experts adds routed sparsity but retains a backprop-trained dense trunk and dense attention.
2. **Backpropagation.** End-to-end credit assignment requires storing activations for the backward pass, synchronous gradient flow across all layers, and an optimizer state several times the size of the parameters. This is the primary driver of training memory and of the inability to learn online at inference time.
3. **Global synchronization.** Attention mixes all positions; backprop mixes all layers. Both forbid the kind of local, parallel, asynchronous computation that biological neural tissue uses.

The program's working position was that these three properties, taken together, account for the bulk of the compute bill and that a substantial fraction of language-model quality is reachable without any of them. The chosen comparison point is **EleutherAI/Pythia-410M** (405,334,016 parameters), measured at **WikiText-103 strided perplexity 17.19** (window 512, measured by `run_llm.py`). Every BrainFormer perplexity in this record is stated against that baseline.

---

## 2. The neuroscience hypothesis

BrainFormer is constructed from mechanisms taken from systems neuroscience rather than from the transformer literature. The hypothesis is that several biological motifs, composed, can substitute for the dense-attention/backprop stack at a small fraction of its cost. Five families of mechanism define the design.

### 2.1 Cerebellar expansion coding (Marr–Albus)

The Marr–Albus model of the cerebellum maps a low-dimensional mossy-fibre input onto an enormous, sparsely active granule-cell layer (in mammals, granule cells outnumber their inputs by roughly two orders of magnitude), then reads that sparse expanded code out through Purkinje cells under a local error signal carried by climbing fibres. The expansion makes arbitrary input patterns linearly separable; the sparsity makes the readout cheap and interference-resistant; the climbing-fibre error makes learning a local delta rule rather than a global gradient.

BrainFormer instantiates this directly. Each layer expands its input across `G` granules (default 12288 per layer), each granule reading a small fixed fan-in (default 48) of sparse random input connections. The readout (the analogue of Purkinje cells) is a linear head trained by a **local delta rule** on the active granules only — `error = softmax − onehot`, scattered into the rows of the readout weights belonging to the granules that fired. No gradient propagates through the expansion. The expansion weights themselves are tuned by competitive k-means under winner-take-all, not by any error signal.

### 2.2 Sparse distributed representations and k-winners-take-all

Kanerva's Sparse Distributed Memory and the broader sparse-coding literature argue that high-dimensional sparse codes give large capacity, graceful degradation, and low cross-pattern interference. BrainFormer enforces sparsity explicitly: only the top-K granules by activation (default K = 64 of 12288, roughly 0.5 %) are allowed to fire per token (**k-winners-take-all**, kWTA). The kWTA stage is both the biological commitment (only a few granules are active at once) and the computational lever (the readout and its updates touch only K rows).

### 2.3 Predictive coding and local learning

The program adopts the predictive-coding stance that a cortical/cerebellar circuit is fundamentally a prediction machine whose only learning signal is a locally available prediction error. Every learned component in BrainFormer is updated by a rule that uses only quantities present at the synapse: the local delta rule on the readout, competitive k-means on the granule weights, and a local EM-style update on the layer-mixture logits. There is no backward pass and no cross-layer gradient. (Predictive coding and related gradient-free credit-assignment schemes — Direct Feedback Alignment, Forward-Forward, target propagation, equilibrium propagation — were later surveyed as ways to add cross-layer credit; the survey conclusion is recorded in §6 and in the dedicated research-program document.)

### 2.4 Multi-timescale processing

Cortical and cerebellar circuits integrate information over a spectrum of timescales rather than a single context window. BrainFormer binds context with **multi-scale exponential decay**: token codes are bound with position sign-codes and a decay kernel, and under the `--multiscale` flag layer *l* binds context at `decay0 / 2^l`, so shallow layers see fast/local context and deep layers see slow/global context. This was the change that turned the layer stack from a set of near-identical experts into a genuine hierarchy (see §5).

### 2.5 Complementary learning systems and retrieval/associative memory

Complementary Learning Systems theory separates a slow, structured neocortical store from a fast, episodic hippocampal store. BrainFormer's analogue of the fast episodic store is the combination of (a) an **associative/retrieval** path — the infini-gram suffix-array head that supplies the next-token distribution from the longest matching suffix of the context — and (b) **test-time plasticity**: the delta rule keeps running during evaluation on the document being read ("hyper-fixation"), so the model adapts to the current text without any offline retraining. Both are inference-time mechanisms that the frozen-weights baseline cannot perform.

These five families compose into the per-token pipeline documented in the architecture record: fixed random token codes → hyperdimensional context binding → sparse granule expansion → kWTA → fixed random Johnson–Lindenstrauss relay to the next layer → hierarchical class/word readout → a learned probabilistic mixture over `n_layers` parallel cerebellar experts (default 4), each carrying its own next-token objective (deep supervision). The single most consequential structural fact, which recurs throughout this record, is that the layers are **parallel experts combined by a mixture, not a shared-trunk hierarchy with cross-layer credit assignment**.

---

## 3. The efficiency thesis and original targets

The thesis is that the neuroscience-derived stack reaches useful language-model quality at a small fraction of the compute and parameter cost of a backprop transformer, by replacing dense activation, backpropagation, and global synchronization with sparse activation, local learning, and parallel locality.

The numerical targets stated earlier in the program were:

| Quantity | Target |
| --- | --- |
| Nominal parameters | ~2B |
| Active parameters per token | ~50–100M |
| Training budget | hundreds to ~1500 USD |
| Quality reference | approach Pythia-410M, WikiText-103 ppl 17.19 |

The active-parameter target follows directly from kWTA: with only K granules firing, the per-token readout and update cost is set by K rather than by `G`, so a large nominal capacity can be carried at small active cost. The cost-of-readout analysis that fixes the hierarchical head is `K·(C + V/C)`, minimized at `n_classes ≈ sqrt(V)` (default 256 classes), which is why the readout is hierarchical (class head times word-given-class head) rather than a flat softmax over the full vocabulary.

---

## 4. Architecture and learning in brief

The design and its learning rules are recorded in full in the architecture and CUDA-engineering documents; the minimal summary needed to read the lineage is below. Reference implementation: `test_013_v3/src/main.cu`; configuration `test_013_v3/src/config.h`.

- **Fixed random front-end.** Token codes `codes[V, D]` (D = `code_dim`, default 128), position sign-codes, and the inter-layer relay are random projections that are never trained. Only `gwt` (granule weights, by competitive k-means), `gbias`, the readout weights `Wc`/`Ww`, the biases, and the mixture logits are learned.
- **Local learning only.** Readout by the delta rule (atomic-free bucketed scatter with device-side prefix sum, or an atomic fallback); granule weights by competitive k-means with periodic refine; mixture by a local EM update. No backprop anywhere.
- **Batch-invariant step.** The learning step is `st = lr · lr_scale / LR_REF` with `LR_REF = 8192` (commit `ac5179e`), so batch size is a pure speed knob rather than a learning-rate knob (linear scaling rule). The earlier `st = lr / B` made a larger batch learn less.
- **Numerical constraint.** The word head must stay fp32: a bf16 master rounds the ~3.7e-5 delta-rule updates to zero.

---

## 5. Experiment lineage: test_013 v1 → v2 → v3

The program is a thirteen-experiment series (`test_001`..`test_013`). The architecture above is the test_013 line; its three sub-versions trace the path from a single learned layer to the production CUDA engine.

### 5.1 v1 — single learned cerebellar layer

A single learned layer established the floor. WikiText-103 perplexity by training-token budget:

| Tokens | Perplexity |
| --- | --- |
| 50M | 3284 |
| 100M | 3086 |
| 500M | 2680 |

The result confirmed the mechanism produces a working language model (perplexity well below uniform over the vocabulary, improving with data) while sitting two orders of magnitude above the 17.19 baseline.

### 5.2 v2 — Python multi-file deep version (perplexity oracle)

A multi-layer deep-supervision version written in Python served as the **perplexity oracle**: the readable reference whose numerical behavior the CUDA rewrite had to reproduce. It introduced the parallel-expert stack with the learned mixture, but was too slow for the token budgets the thesis required.

### 5.3 v3 — C/CUDA rewrite for throughput

v3 is a single-translation-unit C/CUDA rewrite (`test_013_v3/src/main.cu`, `Makefile` targeting `sm_75`) built for the throughput the experiments needed on dual T4s. Engineering details are in the CUDA document; the headline is throughput from ~44k tok/s (v13) to ~334k tok/s peak, ~171k tok/s for the 4-layer quality config (50M tokens in ~5 minutes). v3 is the platform on which every mechanism below was measured.

### 5.4 Mechanisms and results measured on v3 (Kaggle 2×T4, WikiText-103)

| Mechanism / flag | Effect | Commit |
| --- | --- | --- |
| Multi-timescale depth (`--multiscale`) | static 3858 → 3478 (−10 %); layers diverged into a real hierarchy (L0 3539 / L1 3919 vs near-tied 4076/4081) | `d5177d0` |
| Sampled-softmax word head (`--neg-word 8`) | ~+14 % speed, quality-neutral | `c4b901b` |
| **Test-time adaptation (`--adapt`, "hyper-fixation")** | 4-layer multiscale static 3412 → adaptive **1930 (−43 %)**; predict-then-learn, token scored before update, no leakage | — |
| Context-dim refutation | `code_dim` 128 → 512 gained only ~4 %; context representation is **not** the quality wall | — |
| "More data hurts" regression | 50M static 3412 / adaptive 1930, but 500M static 3703 / adaptive 2057 (worse); deterministic, not noise | — |
| LR anneal + exposed weight decay (`--lr-final`, `--wd`, `--decay-every`) | fix for the regression: cosine-anneal lr, expose wd/decay-every (were hardcoded 1e-5 / 200) | `0e542be` |
| Homeostatic load-balancing (`--balance`) | Gini 0.348 → 0.251 but ppl slightly worse (3470/2037 vs 3412/1930); **dropped** — dead-granule fraction was 0.0 %, so the Gini reflected useful specialization | `4683af4` |
| **Infini-gram interpolation (`--ngram`)** | 50M index, λ=0.3: static 3412 → **1272 (−63 %)**; adaptive 1930 → **762 (−60 %)**; largest single quality jump | `25aedc1` / `c30462a` / `ef28dd5` |
| Checkpoint / resume (`--save`/`--load`) | saves only learned state (~0.4 GB), regenerates the fixed-random front-end from seed; for the 12h Kaggle limit | — |

Two of these results define the program's differentiated axis and its limit, and both carry an explicit scope caveat.

- **Test-time adaptation** is the mechanism the frozen baseline cannot do: continuing the local delta rule during evaluation cut perplexity 43 % on the document being read. Because the token is scored before the update, there is no leakage.
- **Infini-gram interpolation** delivered the largest realized win (static −63 %, adaptive −60 %) by mixing `P = (1−λ)·P_brain + λ·P_ngram` from the longest matching corpus suffix. The honest scope: the gain is on corpus-like / in-distribution text where a long suffix matches. On novel chat input there is no match and the blend falls back to the bare model, so the 762 figure does **not** transfer to conversation.

---

## 6. The structural ceiling and why it motivates a hybrid

The program's terminal finding is that the gap from BrainFormer's ~2000-range perplexity to the 17.19 baseline is **structural**, not a tuning or data deficit, and that no data-side or grafted technique surveyed crosses it. The frame is stated plainly in `test_013_v3/doc/v15.md`: the ppl ~2000 → 17 gap is structural — parallel-expert mixture, fixed random codes, linear readout — and no single graft crosses it; v15's realistic target is lower perplexity and unlocking scale, not a coherent chatbot.

The root cause is the same fact noted in §2: the layers are **parallel experts combined by a mixture, not a shared-trunk hierarchy**, and there is no backpropagation to perform compositional credit assignment across them. The supporting evidence from the surveys (full detail in the research-program document):

- **DeepSeek technique transfer.** Of the surveyed techniques, only four transfer in principle to a no-backprop sparse model; aux-loss-free load-balancing was tested and did not help here (§5.4), multi-token prediction does not fit (parallel experts, no shared trunk, no backprop to share an auxiliary head's benefit), leaving shared/always-on experts and int8 readout. None closes the perplexity gap.
- **Gradient-free credit assignment.** Direct Feedback Alignment is the closest fit but is conditioning-only — web-verified DFA on a language model reached perplexity 93 versus 30 for backprop — so it cannot approach the baseline. Forward-Forward, predictive coding, target propagation, and equilibrium propagation either presuppose a topology the model lacks or multiply the bandwidth-bound activation by an iteration count.
- **Efficient-LM grafts.** Infini-gram interpolation (built; largest realized win) and retrieval/denser-kWTA ideas reduce perplexity on in-distribution text but do not give compositional generation.

The honest ceiling: the realistic floor for the pure model is **low hundreds of perplexity on in-distribution text, not coherent open-ended generation**. No grafted technique observed in the program crosses the structural gap to ~17.

This is the decision the rest of the project rests on. Because the structural gap cannot be closed by the no-backprop model, **Zell** is defined as a **hybrid** (decision recorded June 2026): a real, backprop-trained efficient transformer serves as the coherence core, with the BrainFormer mechanisms that proved genuinely differentiated — infini-gram retrieval and test-time adaptation — kept as inference-time augmentations.

- **Core:** full fine-tune of `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0; native Hermes `<tool_call>` tool-calling; ChatML template; ~1 GB fp16). Backup: `SmolLM2-1.7B-Instruct`.
- **Mandatory port:** from the Pythia-410m tokenizer to the Qwen2.5 BPE tokenizer (~151k vocab); the infini-gram index must be rebuilt in the core's tokenizer. This is the single real porting cost.
- **Carryover:** `build_ngram.py` (re-tokenized), `build_blend.py` (re-templated to Qwen ChatML), and the test-time-adaptation concept (as dynamic evaluation / RAG session memory).
- **Retired:** the no-backprop cerebellar model as the coherence core.

The expected WikiText-103 trajectory for Zell: off-the-shelf Qwen2.5-0.5B-Instruct ~mid-teens; first solid win below 17.19 after continued-pretrain + assistant-only SFT; further drops from rebuilt infini-gram interpolation and test-time adaptation — with the standing caveat that the infini-gram and adaptation gains are real on in-distribution text and near-zero on novel chat. The detailed milestone plan (M0–M5) is recorded in the Zell pivot document.

---

## 7. Position against prior art (high level)

- **Dense transformers (the baseline class).** BrainFormer keeps the next-token objective but removes dense attention, dense feed-forward, and backpropagation; it trades global mixing for sparse local expansion. The cost is the structural ceiling above; the benefit is per-token sparse activation and online (test-time) learning the dense model cannot do.
- **Mixture-of-experts.** Both use routed sparsity, but MoE routes within a backprop-trained dense trunk and learns its router by gradient. BrainFormer's "experts" are whole parallel cerebellar layers with no shared trunk and no gradient between them — which is also exactly the property that blocks compositional credit assignment.
- **Gradient-free learning (DFA, Forward-Forward, predictive coding, target/equilibrium propagation).** BrainFormer is in this family by construction (local rules only). The surveyed evidence places these methods materially above backprop in perplexity, consistent with BrainFormer's observed ceiling.
- **Retrieval / n-gram interpolation (infini-gram, kNN-LM).** Adopted as an augmentation, not a substitute. It produced the largest realized perplexity win but only on in-distribution text, which is precisely why it is an inference-time add-on to a coherent core rather than the core itself.
- **Test-time / dynamic evaluation (Krause-style).** BrainFormer's hyper-fixation is a local-rule instance of dynamic evaluation; it is the mechanism carried forward into Zell as both dynamic-evaluation SGD on a small parameter subset and RAG session memory.

---

## 8. Summary of the motivating chain

1. The transformer cost floor is set by dense activation, backpropagation, and global synchronization.
2. Neuroscience offers a stack — cerebellar expansion coding, sparse distributed representations, predictive-coding local learning, multi-timescale processing, complementary learning systems — that removes all three.
3. BrainFormer (test_013 v1 → v2 → v3) realized that stack as a working, fast, no-backprop language model with two genuinely differentiated capabilities: test-time adaptation and infini-gram retrieval.
4. The perplexity gap to the 17.19 baseline proved structural: parallel-expert mixture with no compositional credit assignment, and no surveyed graft crosses it.
5. Therefore Zell keeps the differentiated mechanisms but moves the coherence burden to a small backprop-trained transformer core — a hybrid, not a replacement.

Sources read for code-level accuracy: `test_013_v3/doc/v15.md`; architecture and config references `test_013_v3/src/main.cu` and `test_013_v3/src/config.h` as cited.
