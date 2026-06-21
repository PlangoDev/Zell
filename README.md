# Zell

Zell is a hybrid chatbot. It pairs a backprop-trained efficient transformer core (full fine-tune of `Qwen/Qwen2.5-0.5B-Instruct`) for coherence with the two inference-time augmentations that produced real, defensible wins in the predecessor system: infini-gram retrieval and test-time adaptation ("hyper-fixation").

The predecessor is **BrainFormer**, the no-backprop cerebellar language model built across the `test_001`..`test_013` experiment series. BrainFormer is a working, fast language model whose perplexity gap to a backprop baseline was established as *structural*: the parallel-expert mixture with fixed random codes and a linear local-delta readout lacks compositional credit assignment, and no surveyed data-side or grafted technique crosses it. That finding is what motivates the Zell pivot — move the coherence burden to a backprop-trained core, keep the two augmentations that work.

Baseline of comparison throughout: EleutherAI/Pythia-410M (405,334,016 params), WikiText-103 strided perplexity 17.19.

## Layout

| Path | What it is |
|---|---|
| `docs/` | The BrainFormer + Zell documentation set (the live technical and IP record). Start at [`docs/README.md`](docs/README.md). The Zell design of record is [`docs/09-zell-hybrid-plan.md`](docs/09-zell-hybrid-plan.md). |
| `archive/plango-labs/` | Frozen copy of all prior research from the `PlangoDev/plango-labs` repo: every test (`test_001.py`..`test_010.py`, plus `test_011/` … `test_013_v3/`) and the per-test writeups in `archive/plango-labs/doc/`. |

## Notes on the archive

- It is a snapshot of the research as code, not an active build target. The current work lives at the top level (`docs/`) and in whatever Zell code is added here going forward.
- Large data blobs (`archive/plango-labs/test_011/data/*.bin`, ~52 MB Fashion-MNIST) are present on disk but git-ignored to keep this repo light. Un-ignore them in `.gitignore` if you want them tracked/pushed.
- The original lives at `PlangoDev/plango-labs` (the `test_013_v3/` CUDA single-translation-unit BrainFormer is the latest there).
