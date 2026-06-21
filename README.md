# Zell

Zell is a hybrid chatbot. It pairs a backprop-trained efficient transformer core (full fine-tune of `Qwen/Qwen2.5-0.5B-Instruct`) for coherence with the two inference-time augmentations that produced real, defensible wins in the predecessor system: infini-gram retrieval and test-time adaptation ("hyper-fixation").

The predecessor is **BrainFormer**, the no-backprop cerebellar language model built across the `test_001`..`test_013` experiment series. BrainFormer is a working, fast language model whose perplexity gap to a backprop baseline was established as *structural*: the parallel-expert mixture with fixed random codes and a linear local-delta readout lacks compositional credit assignment, and no surveyed data-side or grafted technique crosses it. That finding is what motivates the Zell pivot: move the coherence burden to a backprop-trained core, keep the two augmentations that work.

Baseline of comparison throughout: EleutherAI/Pythia-410M (405,334,016 params), WikiText-103 strided perplexity 17.19.

## Layout

| Path | What it is |
|---|---|
| `docs/` | The BrainFormer + Zell documentation set (the live technical and IP record). Start at [`docs/README.md`](docs/README.md). The Zell design of record is [`docs/09-zell-hybrid-plan.md`](docs/09-zell-hybrid-plan.md). |
| `tools/` | Data + benchmark pipeline. `build_blend.py` builds the mixed pretrain blend (general text + code + math + chat + synthetic tool-call data) into one `uint32` memmap in the Qwen2.5 tokenizer; `build_data.py` builds the WikiText-103 test bin; `bench_ppl.py` scores any model set with the strided-ppl protocol (the 17.19 reference). |
| `synth/` | `generate.py`: OpenRouter synthetic chat + tool-call corpus generator (cheap/free teacher), written as JSONL the blend ingests as one weighted source. |
| `train/` | `train.py`: full fine-tune of the core on the blend memmap (fp16/bf16, gradient checkpointing, `paged_adamw_8bit`, DDP over 2xT4) with end-of-run generation samples. |
| `notebooks/` | `zell_train_kaggle.ipynb`: the end-to-end 50M-token validation run (build → train → benchmark → generate). |
| `archive/plango-labs/` | Frozen copy of all prior research from the `PlangoDev/plango-labs` repo: every test (`test_001.py`..`test_010.py`, plus `test_011/` … `test_013_v3/`) and the per-test writeups in `archive/plango-labs/doc/`. |

## Build pipeline (M0 → M1)

```
# 1. (optional, Internet-ON) generate synthetic chat + tool-call data
export OPENROUTER_API_KEY=sk-or-...
python synth/generate.py --target-tokens 50000000 --out-dir synth/out \
    --model "inclusionai/ling-2.6-flash" --concurrency 8   # ~$1.65 for the full 50M

# 2. (Internet-ON) build the WikiText-103 test bin + the mixed blend
python tools/build_data.py  --out-dir /kaggle/working --test-only
python tools/build_blend.py --pool-tokens 50000000 --out-dir /kaggle/working \
    --synth-glob "synth/out/*.jsonl"

# 3. (Internet-OFF) continued-pretrain the core on the blend, across both T4s
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
    train/train.py --meta /kaggle/working/meta.json --train-tokens 50000000

# 4. benchmark vs the baselines on the same protocol that gives Pythia 17.19
python tools/bench_ppl.py --test-bin /kaggle/working/wikitext103_test_*.bin \
    --models EleutherAI/pythia-410m,Qwen/Qwen2.5-0.5B-Instruct,/kaggle/working/zell-core
```

Or run all four stages from [`notebooks/zell_train_kaggle.ipynb`](notebooks/zell_train_kaggle.ipynb).

## Notes on the archive

- It is a snapshot of the research as code, not an active build target. The current work lives at the top level (`docs/`) and in whatever Zell code is added here going forward.
- Large data blobs (`archive/plango-labs/test_011/data/*.bin`, ~52 MB Fashion-MNIST) are present on disk but git-ignored to keep this repo light. Un-ignore them in `.gitignore` if you want them tracked/pushed.
- The original lives at `PlangoDev/plango-labs` (the `test_013_v3/` CUDA single-translation-unit BrainFormer is the latest there).
