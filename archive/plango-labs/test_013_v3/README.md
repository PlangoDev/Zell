# Test 013 v3 — C/CUDA Deep Brain

The brain rewritten in **C/CUDA** for maximum throughput and control. Python is
used **only** to tokenize data and to run the pretrained LLM opponent; all brain
training and evaluation is in a single compiled binary. Same no-backprop deep
cerebellar architecture as `test_013_v2` (the Python reference oracle).

**Goal: tokens/sec — as high as the 2×T4 allows (target: high hundreds of
thousands → 1M+).** Quality is held at "match v2" as a correctness gate; quality
levers wait until the v2 50M run tells us what to tune.

## Why C/CUDA

The torch v2 ran at a few % of the hardware's bandwidth ceiling — lost to kernel
launches, atomic-scatter contention, and a materialized `[B,K,S]` gather. v3's
custom kernels remove all three. The readout (word head) is the HBM-bandwidth
bottleneck; everything else (granule GEMM, binding, relay) is cheap.

## Architecture (no backprop)

Fixed random token codes → hyperdimensional context binding (token×position,
multi-scale decay) → stack of cerebellar layers (granule expansion → kWTA → local
delta-rule class→word readout, deep supervision) → learned probabilistic mixture
(local EM rule). A fixed random relay of each sparse code + a context skip feeds
the next layer.

## Milestone status

- **M1 ✓** single-GPU, full depth + mixture, fp32 masters, custom kernels.
- **M2 ✓** atomic-free bucketed readout scatter (`--fast-scatter`, default on),
  resident-VRAM shard + on-GPU window gather (`--resident`), class-major `Ww`
  layout, and full **per-phase timing + granule-health analytics** (`--analytics`).
  (Still TODO inside M2 for more speed: device-side bucket prefix-sum instead of
  the per-step host round-trip, cuBLAS fp16 tensor-core activation, CUDA graphs.)
- **M3 ✓** depth + learned mixture (in M1).
- **M4 ✓** competitive granule learning (setup) + online refinement (periodic).
- **M5 (next): dual-GPU local-SGD.** Deliberately gated on the single-GPU binary
  compiling and producing sane numbers first — dual-GPU is a ~2× throughput
  multiplier on a base that must work, so validating two unknowns at once is the
  wrong order. `--n-gpus 2` currently warns and runs single-GPU.
- **M6 ✓** notebook (with analytics + knob sweep) + tools + docs.

### Safe fallbacks (if a fast path misbehaves)
`--fast-scatter 0` (atomic scatter), `--resident 0` (host gather), `--no-feat`
(random granules), `--analytics 0` (skip timing syncs). Smoke uses the atomic
scatter by default to validate the simplest path first.

## Run on Kaggle

Open `brainformer_v13_v3_kaggle.ipynb` (Accelerator `GPU T4 x2`, Internet On, add
a `GITHUB_TOKEN` Kaggle secret). Cells: clone → build data → LLM ppl → `nvcc`
compile → `--smoke` → train+eval → scoreboard.

Manual:
```bash
python tools/build_data.py --train-tokens 50000000 --out-dir /kaggle/working
python tools/run_llm.py    --meta /kaggle/working/meta.json --out /kaggle/working/llm_result.json
make
./brain --smoke --meta /kaggle/working/meta.json --out /kaggle/working/brain_smoke.json
./brain        --meta /kaggle/working/meta.json --out /kaggle/working/brain_result.json --train-tokens 50000000
python tools/scoreboard.py --brain /kaggle/working/brain_result.json --llm /kaggle/working/llm_result.json
```

### Speed dials (CLI)
`--n-classes` (↑ shrinks S = V/n_classes, the dominant traffic), `--n-layers`,
`--k-active`, `--batch`, `--step-chunk`, `--bench` (throughput only, no eval).
Quality config: `--n-layers 4 --n-classes 256`.

## Module map

| file | what |
|---|---|
| `src/common.h` | RNG, timer, softmax/argmax (reused from test_012) |
| `src/threads.{h,cpp}` | CPU thread pool (reused) |
| `src/cuda_util.h` | CUDA error macros, device-buffer helpers |
| `src/config.h` | `Cfg` + CLI parse (speed-first defaults) |
| `src/data.{h,c}` | meta.json reader + uint16 token-bin mmap |
| `src/main.cu` | all CUDA kernels + DeepBrain + train/eval + main |
| `tools/build_data.py` | tokenize train + WikiText-103 test → bins + meta.json |
| `tools/run_llm.py` | LLM strided perplexity → llm_result.json |
| `tools/scoreboard.py` | merge brain + LLM results |

## Honest accounting

MACs/token is the **sparse** cost summed across layers (granule expansion +
relays + readouts + bind), printed in `brain_result.json` and the scoreboard,
alongside live `tok/s`. The dense matmul is just how it's computed fast.

## Status caveat

Written for the Kaggle T4 toolchain; **not yet compiled/run** (no local nvcc/GPU
in dev). The `nvcc` cell is the first real validation, then `--smoke`. The Python
v2 (`test_013_v2`, `--n-layers N --no-feat`) on the same bins is the perplexity
oracle for the correctness gate.
