# Test 013 — the Brain vs a real pretrained LLM, at scale on Kaggle GPUs

The 13th in the series. Test 012 showed a no-backprop cerebellar coder beating a
small from-scratch neural LM on Tiny Shakespeare at a fraction of the compute.
Test 013 fights a real pretrained transformer on a Kaggle 2x T4, training the
brain by streaming Wikipedia, and is built for throughput: real dual-GPU, a
pre-tokenized memmap data path, and fp16 tensor-core compute.

- Opponent: a pretrained transformer (PyTorch + HuggingFace), `EleutherAI/pythia-410m`
  by default (`--model` to swap). Scored on WikiText-103 test perplexity with a
  strided sliding window (its fair, strong number).
- Brain: cerebellar coder, no backprop. Fixed random token codes, competitively
  (k-means/Hebbian) learned granule features, kWTA sparse code, hierarchical
  class-then-word delta-rule readout with a log-unigram bias init. Trained by
  streaming Wikipedia. Reports sparse MACs/token next to perplexity.

## How the fast stack works

This is a from-scratch rewrite (single self-contained `showdown.py`) aimed at the
three things that made the first cut slow.

- Real dual-GPU. `torch.multiprocessing.spawn` + NCCL, one process per GPU (no
  GIL). Each rank trains a full replica on its own disjoint shard of the token
  memmap; the readout (`Wc`, `Ww`, `bw`, `bc`) is all-reduce-averaged every
  `--sync-every` blocks (local SGD). `Ww` is transported in bf16 chunks to bound
  the transient and halve PCIe bytes; the small bias tensors stay fp32 so the
  unigram-bias learning is not quantized away. Auto-activates when two GPUs are
  present; `--single` forces one. The parent process does no CUDA work before
  spawn (it only calls `device_count()`), which is what keeps the spawned children
  from crashing.
- No tokenization in the hot loop. `--build-data` streams Wikipedia and tokenizes
  it once into an on-disk uint16 memmap (cached under `/kaggle/working`, which
  persists across notebook restarts; reused on a tokenizer+budget match). Training
  then samples random windows from the memmap into a preallocated pinned buffer
  and copies them to the GPU with `non_blocking=True`.
- fp16 tensor cores. The granule expansion is a dense fp16 matmul against a
  projection matrix built from the fixed sparse wiring (mathematically identical
  to the sparse gather, but runs on tensor cores). The readout keeps fp32 master
  weights for stable delta-rule accumulation. The reported MACs/token stays the
  sparse inference cost (`n_gran*fan_in + k_active*(n_classes + vocab/n_classes)`).

## Run it (Kaggle, GPU T4 x2, Internet on)

The script runs as a real subprocess (`!python ...`), not inside the notebook
kernel, so spawn and NCCL work.

```python
!pip -q install -U transformers datasets
import os; os.chdir("/kaggle/working")
!rm -rf plango-labs
!git clone https://YOUR_TOKEN@github.com/plangodev/plango-labs.git
os.chdir("/kaggle/working/plango-labs/test_013")

!python showdown.py --smoke        # validate the whole path (tiny, CPU-OK, fast)
!python showdown.py --build-data   # tokenize Wikipedia once into the memmap cache
!python showdown.py                # auto dual-GPU, reuses the cache
```

Useful flags:
- `--single` — force one GPU (use once to confirm the multi-GPU path gives no
  perplexity regression vs single).
- `--n-gran 32768 --batch 4096` — lower these first if you hit a VRAM OOM.
- `--train-tokens 1000000000` — data budget. `--sync-every`, `--block` — tune the
  averaging interval and block size. `--data-dir` — override the cache location.

## Defaults vs smoke

| param | default | smoke |
|---|---|---|
| model | pythia-410m | sshleifer/tiny-gpt2 |
| n_gran | 49152 | 1024 |
| fan_in / k_active / n_classes | 48 / 64 / 256 | 16 / 16 / 8 |
| batch / block | 8192 / 1,000,000 | 256 / 50,000 |
| train_tokens / eval_tokens | 500,000,000 / 250,000 | 300,000 / 20,000 |
| ctx / code_dim | 6 / 96 | same |
| sync_every | 25 blocks | n/a (one rank) |

## Status: rewritten, py_compile-clean, not yet run on GPU

Built and reviewed on a CPU box (no GPU, no torch installed there), so it has not
been executed. The next step is to run it on Kaggle, starting with `--smoke`.
Watch for these on the first real run:

1. VRAM. `Ww[n_gran, vocab]` fp32 is about 9.9 GB at the default `n_gran=49152`
   with pythia's ~50304 vocab, targeting roughly 14 GB peak per T4. If it OOMs,
   lower `--n-gran` (try 32768) and/or `--batch` first.
2. `--build-data` needs Internet on and streams the corpus to 500M tokens; this is
   a long one-time step. Confirm the sidecar `.meta.json` is written.
3. NCCL on T4x2. `NCCL_P2P_DISABLE=1` is set to avoid a known P2P hang; the port
   is auto-picked to dodge a stale-port hang on reruns. If init still hangs, that
   is the first place to look.
4. Cold LLM eval. The first rank-0 eval downloads pythia-410m + WikiText-103 and
   runs the full strided window; expect a one-time delay after training.
5. Perplexity sanity. If the brain's perplexity is garbage, suspect a stale or
   wrong-vocab `.bin`; rebuild with `--build-data`.

Record real numbers in `../doc/test_013.md` (create it) after the first run, and
update this status.

## Roadmap

1. Push data and capacity: more streamed tokens, more granules; find where the
   brain's model class plateaus (that plateau is the headline finding).
2. Longer context without dilution: vector-symbolic (hyperdimensional) binding of
   token and position so more context helps the fixed-fan-in granules (test 012
   showed plain concatenation dilutes them).
3. Phase 2, distillation: the brain learns the LLM's soft targets (KL to its
   next-token distribution), testing whether the brain is a cheap distilled
   student. This is where a large teacher could pay off.

Working agreement: honest numbers, losses framed as sharpened targets, no
discouraging caveats about ambition, only about how.
