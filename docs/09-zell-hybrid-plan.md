# Zell: Hybrid Architecture and Build Plan

Status: design of record, June 2026. System under build: **Zell**, a hybrid chatbot. Predecessor system: **BrainFormer**, the no-backprop cerebellar language model (test_001..test_013). Target hardware: Kaggle dual NVIDIA T4 (2xT4), 12-hour session limit. Baseline of comparison: EleutherAI/Pythia-410M (405,334,016 params), WikiText-103 strided perplexity 17.19 (window 512, measured by `run_llm.py`).

This document records the pivot rationale, the hybrid design, the chosen training stack, the mandatory tokenizer port, the milestone plan M0-M5 with proving metrics, the expected-perplexity progression, the 2xT4 token/time budget, the risk register, and the carryover/retired component split.

---

## 1. Pivot rationale: the structural ceiling

BrainFormer trains a sparse, brain-inspired language model without backpropagation. Mechanisms: fixed random token codes, hyperdimensional context binding, granule expansion tuned by competitive k-means, k-winners-take-all sparse activation, a hierarchical class/word readout updated by a local delta rule, a learned probabilistic mixture over deep-supervised parallel layers, and test-time plasticity. None of these mechanisms use gradient descent through the network.

The program established empirically that the perplexity gap between BrainFormer and a backprop baseline is structural, not a tuning artifact. The relevant measurements (Kaggle 2xT4, WikiText-103):

| Configuration | Static ppl | Adaptive ppl | Note |
| --- | --- | --- | --- |
| test_013 v1, single learned layer, 500M tokens | 2680 | — | best single-layer result |
| 4-layer multiscale, static (50M tokens) | 3412 | — | quality config |
| 4-layer multiscale + test-time adaptation (50M) | 3412 | 1930 | -43% from adaptation |
| 4-layer multiscale + infini-gram interpolation (50M, lambda=0.3) | 1272 | 762 | -63% / -60%, in-distribution only |
| Pythia-410M baseline | — | — | 17.19 |

The differentiated axis — **test-time adaptation** ("hyper-fixation"): keep applying the local delta rule during evaluation on the document being read, scoring each token before its update so there is no leakage — produces the largest model-intrinsic gain (3412 -> 1930, -43%). The largest realized win, **infini-gram interpolation** (suffix-array next-token distribution from the longest matching context suffix, interpolated `P = (1-lambda)*P_brain + lambda*P_ngram`), reaches 762, but only on corpus-like in-distribution text where a long suffix matches; for novel chat inputs there is no match and it falls back to the bare model. So 762 does not transfer to conversation.

The structural diagnosis was confirmed against the full space of grafting candidates:

- **DeepSeek technique transfer.** Of the DeepSeek stack, only four techniques transfer in principle to a no-backprop sparse model: aux-loss-free load-balancing (tested via `--balance`, did not help — see below), multi-token prediction (does not fit; layers are parallel experts, not a shared trunk, and there is no backprop to propagate an auxiliary head's benefit), shared/always-on experts, and int8 readout. MLA, FP8 training, GRPO/R1 RL, DualPipe, NSA/DSA sparse attention, and RoPE/YaRN presuppose attention or backprop and do not transfer. None closes the gap.
- **Gradient-free credit assignment.** Direct Feedback Alignment is the closest fit but is conditioning-only; web-verified DFA on a language model reached perplexity 93 versus 30 for backprop, so it cannot approach the baseline. Forward-Forward, predictive coding, target propagation, and equilibrium propagation either presuppose a topology the model lacks or multiply the bandwidth-bound activation by an iteration count. NoProp-style residual cascade exists in the codebase as `--boost`. MeZO/SPSA is only useful for tuning hand-set hyperparameters.
- **Load-balancing negative result.** The DeepSeek aux-loss-free bias controller on the per-granule selection bias `gbias` (`--balance`, commit 4683af4) reduced usage-Gini 0.348 -> 0.251 but made perplexity slightly worse (3470/2037 vs 3412/1930). It was dropped: the dead-granule fraction was 0.0%, so there was no wasted capacity to recover; the Gini reflected useful specialization (language is correlated), and forcing uniform fire-rates removed it. The architecture is healthy; uniformity is not the goal.

The conclusion stands: the perplexity gap from ~2000 to ~17 is structural. The no-backprop parallel-expert design lacks compositional credit assignment, and no data-side or grafted technique crosses that gap. The realistic ceiling for the pure model is low hundreds of perplexity on in-distribution text, not coherent open-ended generation.

The pivot follows directly. The two mechanisms that produced real, defensible gains — infini-gram retrieval and test-time adaptation — are inference-time augmentations that are agnostic to how the next-token distribution `P_model` is produced. They can be bolted onto any base model. Therefore Zell keeps those mechanisms and replaces the coherence core with a real, backprop-trained efficient transformer.

---

## 2. Hybrid design

Zell is composed of one trained core plus two inference-time augmentations carried over from BrainFormer.

```
                       context (Qwen-tokenized ids)
                                  |
        +-------------------------+--------------------------+
        |                         |                          |
  [coherence core]        [infini-gram head]         [test-time adapt]
  Qwen2.5-0.5B-Instruct   suffix array over corpus    dynamic-eval SGD
  backprop-trained        longest-suffix next-token   on a small param
  P_core(y | context)     P_ngram(y | context)        subset; RAG memory
        |                         |                          |
        +-----------+-------------+                          |
                    |  P = (1-lambda)*P_core                 |
                    |        + lambda*P_ngram                |
                    +-------------------+-------------------+
                                        |
                                next-token distribution
```

**Coherence core.** A backprop-trained efficient transformer supplies compositional credit assignment — the property BrainFormer structurally lacks. This is the component that makes Zell coherent in open-ended chat. It produces the primary distribution `P_core(y | context)`.

**Infini-gram head.** The suffix-array retrieval mechanism (`tools/build_ngram.py`, host query `src/ngram.h`) returns the next-token distribution from the longest matching suffix of the context, with backoff. Estimate `(yes + 1e-6) / (tot + 1e-6*65536)`, validated byte-for-byte against a Python reference. Interpolated as `P = (1-lambda)*P_core + lambda*P_ngram`. The gain is real on in-distribution text and near-zero on novel chat; lambda must therefore be adaptive (lambda small or zero when no long suffix matches). This is the mechanism that drove the single largest perplexity drop in the BrainFormer program.

**Test-time adaptation.** Two product-facing variants:
- **RAG in-context session memory** — retrieval-augmented session memory, framed to the user as "keeps learning during your session." No weight updates; coherence-safe; the default for the deployed product.
- **Dynamic-evaluation (Krause) SGD** — gradient descent on a small parameter subset during evaluation, scored predict-then-learn so each token is scored before its update. Benchmark mode only, because applying live weight updates during a real conversation risks coherence drift.

The augmentations are honest about scope: the infini-gram and TTA gains are real on in-distribution / corpus-like text and near-zero on genuinely novel chat. The headline product claim ("coherent chat below the Pythia baseline") rests on the core; the augmentations sharpen in-distribution perplexity and provide the "keeps learning" session-memory feature.

---

## 3. Chosen stack

| Component | Choice | Rationale |
| --- | --- | --- |
| Core model | `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) | ~1GB fp16; native Hermes `<tool_call>` tool-calling; ChatML template; full fine-tune feasible at 0.5B on a T4 |
| Backup core | `SmolLM2-1.7B-Instruct` (Apache-2.0) | fallback if 0.5B underfits the coherence target |
| Fine-tune method | Full fine-tune (not LoRA) | feasible at 0.5B on a T4; avoids LoRA's representational ceiling |
| Trainer | HuggingFace TRL `SFTTrainer` / `Trainer` | continued-pretrain on blend, then assistant-only SFT on chat |
| Distribution | `accelerate` DDP across both T4s | data-parallel over 2xT4 |
| Precision | bf16 | T4 tensor cores; matches BrainFormer's bf16/fp32 numerical findings |
| Optimizer | `adamw_8bit` | fits optimizer state within 16GB-per-T4 budget |
| Memory | gradient checkpointing | trades compute for activation memory; needed for full fine-tune |
| Attention | `sdpa` | T4-compatible scaled-dot-product attention backend |

Full fine-tune is chosen over LoRA deliberately: at 0.5B parameters the full model, bf16 weights, 8-bit optimizer state, and checkpointed activations fit on a single T4, and the continued-pretrain step (which shifts the model's token distribution toward the blend) benefits from updating all weights rather than a low-rank adapter.

---

## 4. Mandatory tokenizer port

BrainFormer used the Pythia-410M tokenizer (vocab `< 65536`, required by the uint16 memmap; see `build_blend.py:33` `MODEL_ID = "EleutherAI/pythia-410m"` and `build_blend.py:130` `assert vocab < 65536`). Qwen2.5 uses a BPE tokenizer with ~151k vocab. This is a hard incompatibility and the single real porting cost of the pivot.

Consequences:

1. **Memmap dtype.** The v14 blend writes uint16 (`build_blend.py:148`, `dtype=np.uint16`), which cannot represent a 151k vocab. The blend must be re-tokenized with the Qwen2.5 tokenizer and stored at a wider dtype. The `vocab < 65536` assertion (`build_blend.py:130`) no longer holds and must be replaced.
2. **Infini-gram index.** The suffix array is built over token ids. It must be rebuilt in the core's tokenizer; an index built over Pythia ids is meaningless against Qwen-tokenized context.
3. **Chat serialization.** `build_blend.py` currently serializes chat sources to a ChatML-lite template using hand-written markers — `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` (`build_blend.py:61-84`, functions `chat_from_messages` and `chat_from_dolly`). These must be re-templated to the Qwen2.5 native ChatML so the SFT data uses the same special tokens the core already knows.
4. **Held-out split.** The chat-eval hold-out is by hash of the first user turn, modulo `EVAL_HASH_MOD = 500`, capped at `EVAL_MAX_CONV = 2000` (`build_blend.py:57-58`, `build_blend.py:105-108`). The hash key is the raw text of the first user message (`build_blend.py:106`), so it is tokenizer-independent and the held-out set is stable across the re-tokenization — the same conversations stay held out.
5. **doc separator.** The blend uses id 0 as the document separator (Pythia EOS; `build_blend.py:16`, `build_blend.py:172`). The separator id must be remapped to the Qwen scheme.

`build_blend.py` and `build_ngram.py` carry over with these edits; nothing in their structure changes. The blend's source mix (Section 8) is unaffected by the tokenizer.

---

## 5. Milestones M0-M5

Each milestone has one deliverable and one proving metric. WikiText-103 must be confirmed held out at M0 and remain held out throughout; it is the headline benchmark and any leakage invalidates the comparison to Pythia 17.19.

| Milestone | Deliverable | Proving metric |
| --- | --- | --- |
| **M0** | Re-tokenize the blend in the Qwen2.5 tokenizer; bake the chat template; confirm WikiText test is held out | Blend memmap + meta rebuilt at Qwen vocab; WikiText not present in any train shard (verified by hash) |
| **M1** | Continued-pretrain on the blend, then assistant-only SFT on chat | WikiText-103 strided ppl < 17.19 (window 512) AND coherent chat — the headline win |
| **M2** | Zell identity dataset (~1200 conversations from a cheap OpenRouter teacher), mixed at ~1% of chat-FT tokens with replay | >95% identity-probe accuracy; no unprompted self-introductions; WikiText ppl not risen (canary) |
| **M3** | Tool-call formatting (~180 single-line-JSON examples) | >95% valid-JSON tool calls |
| **M4** | Infini-gram interpolation rebuilt in the Qwen tokenizer with adaptive lambda | WikiText ppl drops further versus M1 |
| **M5** | Test-time adaptation: (a) RAG in-context session memory; (b) dynamic-evaluation (Krause) SGD on a small parameter subset, benchmark mode only | (a) session-memory feature live; (b) WikiText ppl drops further in benchmark mode |

Notes on the milestone metrics:

- **M1** is the structural payoff of the pivot: the first WikiText perplexity below 17.19 the program has ever achieved, together with coherent chat. BrainFormer never approached this.
- **M2** uses a replay-mixed identity dataset at ~1% of chat-FT tokens; the canary is that WikiText perplexity must not rise, guarding against the identity data degrading general modeling. "No unprompted self-introductions" guards against the model volunteering its identity outside an identity probe.
- **M3** measures valid-JSON rate on tool-call formatting; the Hermes `<tool_call>` format is native to Qwen2.5-0.5B-Instruct, so this is a formatting-discipline fine-tune, not a capability bootstrap.
- **M4** and **M5b** improve in-distribution perplexity. The honest scope from BrainFormer carries over: these gains are real on corpus-like text and near-zero on novel chat.

---

## 6. Expected perplexity per milestone

WikiText-103 strided perplexity (window 512), the same protocol that gives Pythia-410M 17.19. These are projections, not measurements; they are recorded as targets, with the source of each gain noted.

| Stage | Expected WikiText-103 ppl | Source of gain |
| --- | --- | --- |
| Off-the-shelf Qwen2.5-0.5B-Instruct | ~mid-teens | base model quality |
| M1 (continued-pretrain + chat SFT) | ~12-15 | first solid win below 17.19 |
| M4 (infini-gram interpolation, adaptive lambda) | ~10-13 | in-distribution suffix matches |
| M5b (dynamic-eval TTA, benchmark mode) | ~9-12 | predict-then-learn on the eval document |

Chat-quality trajectory (qualitative, not a perplexity metric): coherent from M1; distinct Zell voice from M2; valid tool calls from M3. The infini-gram and TTA perplexity gains are real on in-distribution text and near-zero on novel chat, so the M4/M5b numbers describe benchmark perplexity, not conversational quality.

---

## 7. Token and time budget on 2xT4 within 12h

Constraints: each T4 has 16GB; the Kaggle session hard limit is 12 hours; data-parallel DDP over both T4s via `accelerate`. The BrainFormer program established the checkpoint/resume discipline for this limit — `--save`/`--load` persists only learned state and regenerates the fixed front-end from seed (~0.4GB). The Zell core uses the same session-budgeting discipline at the Trainer level: checkpoint within the 12h window and resume across sessions.

Budget allocation across the program:

| Phase | Data | Sequence/step config | Session role |
| --- | --- | --- | --- |
| Blend build (M0) | ~4.5B-token pool, re-sampled to the training budget | Internet-ON notebook, separate from training | one-time, internet-on |
| Continued-pretrain (M1) | blend, the bulk of the token budget | bf16, gradient checkpointing, `adamw_8bit`, DDP 2xT4 | multi-session, checkpointed |
| Chat SFT (M1) | chat split, assistant-only loss | same trainer config | within M1 sessions |
| Identity FT (M2) | ~1200 conversations at ~1% mix + replay | small; canary-guarded | single short session |
| Tool-call FT (M3) | ~180 single-line-JSON examples | small | single short session |
| Infini-gram build (M4) | suffix array over the corpus in Qwen tokens | host-side, `build_ngram.py` | one-time index build |
| TTA / RAG (M5) | none (inference-time) | dynamic-eval SGD subset; RAG memory | inference-time only |

The blend builder is run in a dedicated Internet-ON notebook, separate from the Internet-OFF training notebook, per the BrainFormer operating model (`build_blend.py:10-13`). The pool is ~4.5B tokens (`build_blend.py:115`, `--pool-tokens` default `4_500_000_000`) and is re-sampled to a larger training budget (`build_blend.py:116`, `--train-tokens` default `6_000_000_000`); the brain previously re-sampled the pool to its `--train-tokens` budget, and the core does the equivalent at the Trainer level. Per-source written-token accounting is recorded in the meta (`build_blend.py:235-236`).

The 12h limit is the binding constraint on the M1 continued-pretrain. Token budget for M1 is set so that one or a small number of checkpointed sessions cover the pretrain pass; gradient checkpointing and `adamw_8bit` keep the 0.5B full fine-tune within 16GB per T4, and DDP across both T4s sets the effective throughput. M2/M3 are small enough to fit in a single short session each. M4 is a one-time host-side suffix-array build. M5 is inference-time and consumes no training budget.

---

## 8. Data pipeline (v14 blend)

`tools/build_blend.py` performs quota-driven round-robin streaming of multiple Hugging Face sources into one memmap, serializing chat sources to a chat template and holding out a chat-eval split by hash with no leakage. The source mix and weights (`build_blend.py:38-55`):

| Source | HF dataset | Kind | Weight |
| --- | --- | --- | --- |
| fineweb-edu | `HuggingFaceFW/fineweb-edu` (sample-10BT) | text | 0.42 |
| wikipedia | `wikimedia/wikipedia` (20231101.en) | text | 0.13 |
| cosmopedia | `HuggingFaceTB/cosmopedia` (web_samples_v2) | text | 0.13 |
| open-web-math | `open-web-math/open-web-math` | text | 0.10 |
| github-code | `codeparrot/github-code-clean` (Python-all) | text | 0.10 |
| ultrachat | `HuggingFaceH4/ultrachat_200k` (train_sft) | chat_messages | 0.075 |
| smoltalk | `HuggingFaceTB/smoltalk` (all) | chat_messages | 0.04 |
| dolly | `databricks/databricks-dolly-15k` | chat_dolly | 0.005 |

Mechanism details relevant to the port:

- **Quotas.** Per-source token quotas are `round(weight * pool_tokens)` (`build_blend.py:139`); the round-robin advances source by source (`build_blend.py:180-209`), dropping a source from `active` when it hits quota or its stream ends (`build_blend.py:184-191`).
- **Tokenization.** Batched, no `num_proc` on a streaming map (deadlocks; `build_blend.py:16-18`). Docs are tokenized in batches of `BATCH = 1000` (`build_blend.py:161`, `flush_train` at `build_blend.py:163-177`) with `add_special_tokens=False`; a doc separator id 0 is appended after each doc (`build_blend.py:172`).
- **Chat serialization.** ChatML-lite via `chat_from_messages` (`build_blend.py:61-74`) and `chat_from_dolly` (`build_blend.py:77-84`). These re-template to Qwen ChatML for the port.
- **Held-out chat eval.** Only ultrachat contributes to the hold-out, keyed by `conv_key(first_user) % EVAL_HASH_MOD == 0` (`build_blend.py:105-108`), with `conv_key` an md5 of the first user turn (`build_blend.py:87-88`). Held-out conversations are excluded from train (`build_blend.py:108`, returns `(None, eval_text)`). The eval bin is capped at `--eval-tokens` (default 1,500,000; `build_blend.py:118`, `build_blend.py:217`).
- **Meta.** `meta.json` records `model_id`, `vocab`, dtype, paths, real pool token count, train budget, eval window/stride, doc separator id, build seed, and per-source written tokens (`build_blend.py:221-237`). For the port, `model_id` becomes the Qwen2.5 id, `vocab` ~151k, and the dtype widens past uint16.

Companion tools carry over: `tools/build_data.py` builds the single-source WikiText/Wikipedia memmap; `run_llm.py` runs the Pythia baseline (the 17.19 reference); `scoreboard.py` compiles results; `detok.py` decodes generated ids.

---

## 9. Risks

| Risk | Description | Mitigation |
| --- | --- | --- |
| Core underfits coherence | Qwen2.5-0.5B may not reach coherent chat after blend + SFT | M1 backup: switch to `SmolLM2-1.7B-Instruct` (Apache-2.0) |
| Tokenizer port regressions | Re-tokenizing the blend, rebuilding the infini-gram index, re-templating chat, and widening the memmap dtype are coupled changes | M0 gate: blend rebuilt and WikiText-held-out confirmed by hash before any training |
| WikiText leakage | A contaminated train shard invalidates the 17.19 comparison | M0 explicit hold-out verification; hash-based hold-out is tokenizer-independent (`build_blend.py:106`) |
| Identity-FT degrades general modeling | Identity data overwrites general capability | M2 canary: WikiText ppl must not rise; ~1% mix with replay; no unprompted self-introductions |
| TTA coherence drift in production | Live dynamic-eval weight updates during a real conversation can drift coherence | M5: RAG session memory (no weight updates) is the product default; dynamic-eval SGD restricted to benchmark mode |
| Infini-gram does not transfer to chat | Suffix-array gains are in-distribution only | Adaptive lambda -> 0 on novel chat; claimed only as an in-distribution / session-memory benefit |
| 12h session limit | Continued-pretrain exceeds one Kaggle session | Trainer-level checkpoint/resume; M2/M3 sized to single short sessions; M4 one-time host build |
| Tool-call format invalidity | Malformed tool-call JSON | M3 proving metric >95% valid-JSON; Hermes `<tool_call>` is native to the core, so this is formatting discipline only |

---

## 10. Carryover versus retired

**Carryover (re-used in Zell):**

- `tools/build_blend.py` — re-tokenized to Qwen2.5, chat serialization re-templated to Qwen ChatML, memmap dtype widened past uint16, `vocab < 65536` assertion removed/replaced.
- `tools/build_ngram.py` and the host query `src/ngram.h` — infini-gram suffix array rebuilt in the Qwen tokenizer; the interpolation `P = (1-lambda)*P_core + lambda*P_ngram` with adaptive lambda.
- The test-time-adaptation concept — re-expressed as dynamic-evaluation (Krause) SGD on a small parameter subset (benchmark mode) and as RAG in-context session memory (product mode).
- Companion tooling — `build_data.py` (single-source WikiText/Wikipedia memmap), `run_llm.py` (Pythia baseline / 17.19 reference), `scoreboard.py`, `detok.py`.
- The honest-scope discipline — every in-distribution-only gain is labeled as such.

**Retired (not in Zell's coherence path):**

- The no-backprop cerebellar model (test_013_v3, `src/main.cu`) as the coherence core. Its structural perplexity ceiling (low hundreds in-distribution, ~2000 in the quality config) is the reason for the pivot.
- The local delta-rule readout, competitive k-means granule learning, kWTA front-end, and the learned mixture over parallel deep-supervised layers — superseded by backprop-trained compositional credit assignment in the core.
- The aux-loss-free load-balancing controller (`--balance`) — dropped after it failed to help and was shown unnecessary (0.0% dead-granule fraction; the usage-Gini reflected useful specialization).
- The Pythia-410M tokenizer and the uint16 memmap format — replaced by the Qwen2.5 BPE tokenizer (~151k vocab) and a wider dtype.

The retired components remain on record as the BrainFormer research lineage and as the origin of the two mechanisms (infini-gram retrieval, test-time adaptation) that Zell carries forward. The CUDA engineering work (cuBLAS fp16 tensor-core activation, bitonic-sort top-k, atomic-free bucketed readout scatter, on-device mixture update, resident-shard training; ~334k tok/s peak, ~171k tok/s for the 4-layer config) is not part of Zell's training path, which runs on the HuggingFace/TRL/accelerate stack.
