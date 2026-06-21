# Zell — project context for Claude Code

This file orients a fresh session. Read it before assuming anything about the repo.

## What this is

**Zell** is a hybrid chatbot: a backprop-trained efficient transformer core (full fine-tune of `Qwen/Qwen2.5-0.5B-Instruct`) for coherence, plus two inference-time augmentations carried over from the predecessor system — **infini-gram retrieval** and **test-time adaptation** ("hyper-fixation"). Target hardware: Kaggle dual NVIDIA T4 (2xT4, `sm_75`), 12-hour sessions.

**BrainFormer** is the predecessor: a no-backprop, brain-inspired (cerebellar Marr-Albus) language model built across the `test_001`..`test_013` series. Its perplexity gap to a backprop baseline was proven *structural* — the no-backprop parallel-expert design lacks compositional credit assignment. That finding is why Zell exists. The two augmentations Zell keeps are the only mechanisms that produced real, in-distribution wins.

Baseline of comparison everywhere: EleutherAI/Pythia-410M (405,334,016 params), WikiText-103 strided perplexity **17.19**.

## Layout

- `docs/` — the live BrainFormer + Zell documentation/IP record. Entry point `docs/README.md`; Zell design of record is `docs/09-zell-hybrid-plan.md` (milestones M0–M5, the carryover/retired split, the 2xT4 budget).
- `archive/plango-labs/` — frozen snapshot of all prior research (every `test_*.py`, `test_011/`..`test_013_v3/`, and per-test writeups under `doc/`). Reference, not an active build target. The `test_013_v3/src/main.cu` single-translation-unit CUDA BrainFormer is the most advanced archived implementation.
- Large data blobs under `archive/plango-labs/test_011/data/*.bin` are on disk but git-ignored.

## How this project works (carried from the research method)

- Honest science over manufactured wins. Report results faithfully even when an idea loses; frame a loss as a sharpened research target, never as a reason to scale back ambition.
- Every experiment must teach something genuinely new, not just move a number.
- **Writing style: no AI tells.** Plain prose. No em-dashes as connectors, no emoji, no hype words ("delve", "leverage", "robust", "seamless"). The docs were deliberately scrubbed of these — keep it that way.
- Do not add discouraging caveats about solo feasibility or competing with big labs. Take stated goals as given and work on *how*, not *whether*.

## Origin

Forked from the `PlangoDev/plango-labs` research repo (gh CLI authenticated under `PlangoDev`). This `Zell/` folder is a fresh git repo; the archive preserves the prior work as code.
