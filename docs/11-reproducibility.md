# Reproducibility and Operations

This document specifies how to build, configure, train, evaluate, checkpoint, and augment the BrainFormer cerebellar language model (the no-backprop system) and its data pipeline. It is written so that a third party with the repository and a Kaggle dual-NVIDIA-T4 session can reproduce every reported result. The canonical implementation is the C/CUDA rewrite at `test_013_v3/`. Code references are to that subtree unless noted.

All work was conducted in June 2026 on Kaggle dual NVIDIA T4 GPUs (compute capability 7.5, `sm_75`). Where a measured figure appears it is reproduced from the result record; where a number is a build-time constant it is read from the cited source file.

---

## 1. Repository file map

The repository is `PlangoDev/plango-labs`, a 13-experiment series (`test_001` .. `test_013`). The operational artifact is `test_013_v3`. The relevant tree:

```
plango-labs/
├── test_013/                     v1: single learned cerebellar layer (Python)
├── test_013_v2/                  v2: Python multi-file deep version (perplexity oracle)
│   └── bf/config.py              the config the v3 config.h mirrors
├── test_013_v3/                  v3: C/CUDA rewrite for throughput (canonical)
│   ├── Makefile                  nvcc build, sm_75, single TU + C data layer + C++ pool
│   ├── src/
│   │   ├── main.cu               single CUDA translation unit (whole model + kernels)
│   │   ├── config.h              every runtime knob; cfg_default / cfg_parse / cfg_smoke
│   │   ├── common.h              shared structs / constants (LR_REF etc.)
│   │   ├── data.c / data.h       C data layer: memmap load, window gather
│   │   ├── threads.cpp / threads.h   C++ thread pool (host-side parallelism)
│   │   ├── cuda_util.h           CUDA helpers (error checks, launch wrappers)
│   │   └── ngram.h               host infini-gram query (mmap + longest-suffix backoff)
│   └── tools/
│       ├── build_data.py         single-source WikiText/Wikipedia uint16 memmap + meta.json
│       ├── build_blend.py        v14 blended multi-source corpus + held-out chat eval
│       ├── build_ngram.py        v15 suffix-array (infini-gram) builder + --selftest
│       ├── run_llm.py            Pythia-410M baseline (strided WikiText-103 perplexity)
│       ├── scoreboard.py         compiles result records
│       └── detok.py              decodes generated token-id JSON back to text
└── docs/                         technical / IP record (this file: 11-reproducibility.md)
```

`main.cu` is compiled as one translation unit; the C data layer (`data.c`) and C++ thread pool (`threads.cpp`) are compiled and linked alongside it. There is no separate kernel library and no header-only split of the model.

---

## 2. Build

### 2.1 Toolchain

| Component | Requirement |
|---|---|
| CUDA toolkit | `nvcc` with `sm_75` codegen support (Kaggle T4 default toolkit) |
| Host compiler | C++14, pthreads |
| Libraries | cuBLAS (`-lcublas`), pthreads (`-lpthread`) |
| GPU | NVIDIA T4 (compute 7.5); two for the dual-GPU path |

### 2.2 Makefile

The build is defined in `test_013_v3/Makefile`. The exact flags:

```make
NVCC      ?= nvcc
ARCH      := -gencode arch=compute_75,code=sm_75
NVCCFLAGS := -O3 $(ARCH) -std=c++14 --use_fast_math \
             -Xcompiler "-O3 -funroll-loops -pthread"
LDFLAGS   := -lcublas -lpthread
SRC       := src/main.cu src/data.c src/threads.cpp
```

The `brain` target compiles the three sources together into one binary:

```bash
cd test_013_v3
make            # -> ./brain
```

This expands to:

```bash
nvcc -O3 -gencode arch=compute_75,code=sm_75 -std=c++14 --use_fast_math \
     -Xcompiler "-O3 -funroll-loops -pthread" \
     src/main.cu src/data.c src/threads.cpp -o brain -lcublas -lpthread
```

`--use_fast_math` is intentional: the model tolerates reduced-precision intrinsics in the front-end. The one precision constraint that is NOT relaxed is the fp32 word head (Section 9). Clean with `make clean` (removes `./brain`).

`NVCC` is overridable (e.g. `make NVCC=/usr/local/cuda-12/bin/nvcc`). The build header dependency list is `src/common.h src/config.h src/data.h src/cuda_util.h src/threads.h`; editing any of these triggers a rebuild.

### 2.3 Smoke test

`config.h` provides a tiny configuration (`cfg_smoke`, selected by `--smoke`) that exercises the entire pipeline at sizes small enough to run quickly and validate correctness, including the simple atomic readout-scatter path (`fast_scatter=0`) rather than the bucketed path. Use it after any build to confirm the binary runs end to end:

```bash
./brain --meta /path/to/meta.json --smoke
```

The smoke configuration sets `n_layers=2`, `n_gran=1024`, `fan_in=16`, `k_active=16`, `n_classes=64`, `relay_dim=32`, `ctx=8`, `code_dim=32`, `train_tokens=300000`, `eval_tokens=20000` (full list in `config.h:150`).

---

## 3. Configuration flag reference

All knobs live in one struct, `Cfg`, defined in `test_013_v3/src/config.h:13`. Defaults are set by `cfg_default` (`config.h:120`) and are **speed-first**: large `n_classes` (small per-class slot count `S`) and few layers. The quality configuration (Section 6) is reached by overriding flags. Parsing is order-independent: each flag is an independent `if` (no else-chaining), via the `ARG_INT` / `ARG_LONG` / `ARG_FLT` / `ARG_STR` macros (`config.h:161`).

Symbols used below: `V` = vocabulary size; `D` = `code_dim`; `G` = `n_gran` per layer; `K` = `k_active`; `C` = `n_classes`; `S = ceil(V/C)` = slots per class; `B` = `batch`; `L` = `n_layers`.

### 3.1 Context and token codes

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--ctx` | `ctx` | 32 | Context length. HD binding makes long context cheap. |
| `--code-dim` | `code_dim` | 128 | Per-token random code width = bound-vector width `D`. |
| `--multiscale` | `multiscale` | 0 | 1 = per-layer timescale: layer `l` binds context decayed at `decay0/2^l`, so deeper layers see longer range and layers stop being identical. |

`hd_binding` (bind+sum vs concat) and `n_scales` (number of temporal decay scales, default 3) plus `decay0/decay1/decay2` (0.20 / 0.05 / 0.0125) are set in the struct but not exposed as CLI flags in the current `cfg_parse`.

### 3.2 The deep stack

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--n-layers` | `n_layers` | 2 | Number of independent cerebellar layers (parallel experts, deep supervision; no gradient between layers). |
| `--n-gran` | `n_gran` | 12288 | Granules per layer `G` (keep `G % topk_groups == 0`). |
| `--fan-in` | `fan_in` | 48 | Sparse random input connections per granule. |
| `--k-active` | `k_active` | 64 | kWTA: top-`K` granules fire per token. |
| `--n-classes` | `n_classes` | 256 | Hierarchical-readout classes `C`. `C ~ sqrt(V)` minimizes readout cost `K*(C + V/C)`. |
| `--relay-dim` | `relay_dim` | 128 | Width of the fixed random Johnson-Lindenstrauss relay to the next layer. |

### 3.3 Top-k selection

Two-level approximate top-k by default: many small groups, a few candidates per group, capturing almost all true top-`K` in one scan.

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--topk-groups` | `topk_groups` | 256 | Number of candidate groups. |
| `--topk-per-group` | `topk_per_group` | 2 | Candidates retained per group. |
| `--exact-topk` | `exact_topk` | 0 | 1 = exact K-pass top-k (slow, reference). |
| `--fast-topk` | `fast_topk` | 0 | 1 = group-max top-k (fastest, cruder); opt in when quality does not matter. |

### 3.4 Local learning

All learning is local. No backpropagation. `lr` drives the delta-rule readout; granule weights are competitive k-means.

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--lr` | `lr` | 0.3 | Base learning rate for the local delta rule. |
| `--lr-final` | `lr_final` | 1.0 | Cosine-anneals `lr` to `lr*lr_final` over the run. 1.0 = constant (old behavior). `<1` fixes the "more data hurts" delta-rule saturation at scale. |
| `--wd` | `wd` | 1e-5 | Weight decay; per-event keep factor `(1 - lr*wd)^decay_every`. |
| `--decay-every` | `decay_every` | 200 | Apply decay every N steps. |

`mix_lr` (mixture EM rate, default 0.05) is set in the struct but not exposed as a CLI flag.

`wd` and `decay_every` were hardcoded (1e-5 / 200) before commit `0e542be`; that commit exposed them and added `--lr-final` to address the regression in Section 7.

### 3.5 Homeostatic load-balancing (DeepSeek aux-loss-free controller)

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--balance` | `balance` | 0 | 1 = enable a per-granule fire-rate controller on the selection bias `gbias`. |
| `--balance-lr` | `balance_lr` | 0.1 | Controller gain (gamma). |
| `--balance-every` | `balance_every` | 200 | Apply controller and reset the usage accumulator every N steps. |

Added in commit `4683af4`. Reduced usage-Gini 0.348 -> 0.251 but perplexity got slightly worse (3470/2037 vs 3412/1930); dead-granule fraction was 0.0%, so it was dropped. The Gini reflected useful specialization, not wasted capacity. Retained as a runtime-off flag.

### 3.6 Batching and step

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--batch` | `batch` | 8192 | Examples per step `B`. Pure speed knob (see batch-invariant step below). |
| `--step-chunk` | `step_chunk` | 2048 | Sub-batch chunking for the device step. |
| `--block` | `block` | 1000000 | Training block size (tokens) between progress / sync points. |
| `--train-tokens` | `train_tokens` | 50000000 | Total training token budget. |
| `--eval-tokens` | `eval_tokens` | 250000 | Tokens scored at evaluation. |

The step is batch-invariant: `st = lr * lr_scale / LR_REF` with `LR_REF = 8192` (commit `ac5179e`). Before this, `st = lr/B` made a larger batch learn less; afterward, batch is a pure speed knob (linear scaling rule).

### 3.7 Multi-GPU

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--n-gpus` | `n_gpus` | 1 | 1 or 2 (dual T4). |
| `--sync-every` | `sync_every` | 25 | Local-SGD parameter-averaging interval, in blocks. |
| `--ww-fp16` | `ww_fp16` | 1 | Store/transport the word head in fp16/bf16 for cross-GPU transport (the on-GPU master stays fp32; see Section 9). |

### 3.8 Speed paths (runtime-selectable)

These default ON so the speed paths are exercised; the safe fallback (atomic scatter, host gather) is one flag away.

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--fast-scatter` | `fast_scatter` | 1 | 1 = atomic-free bucketed readout scatter; 0 = atomic. |
| `--resident` | `resident` | 1 | 1 = keep the training shard in VRAM, sample windows on GPU. |
| `--analytics` | `analytics` | 1 | 1 = run a short profiling burst for per-phase timing. |
| `--cublas` | `use_cublas` | 1 | 1 = cuBLAS GEMM activation; 0 = custom-kernel fallback. |
| `--progress-every` | `progress_every` | 1 | Progress line every N blocks (0 = auto). |
| `--gran-frac` | `gran_frac` | 1.0 | `<1` = activate+top-k only a rotating fraction of granules per step (granule dropout); cuts activate+topk bandwidth ~`1/frac`. Quality tradeoff. |
| `--neg-word` | `neg_word` | 0 | `>0` = sampled-softmax word head: update the true word + this many negatives instead of all `S`; cuts word-head scatter ~`S/(1+neg)`. Training only; eval stays full and exact. |

`--neg-word 8` gave ~+14% speed, quality-neutral (commit `c4b901b`).

### 3.9 Residual boosting and test-time adaptation

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--boost` | `boost` | 0 | 1 = residual boosting (layer `l` focuses on what `l-1` missed); NoProp-style cascade. |
| `--adapt` | `adapt` | 0 | 1 = also run an adaptive eval pass (predict-then-learn during evaluation). |
| `--adapt-lr` | `adapt_lr` | 0.1 | Learning rate for the test-time update. |

`boost_wmin` (floor on the per-example boost weight, default 0.1) is exposed as `--boost-wmin`.

Test-time adaptation ("hyper-fixation") keeps applying the delta rule during evaluation on the document being read. The token is scored BEFORE the update, so there is no leakage. Best measured: 4-layer multiscale static 3412 -> adaptive 1930 (-43%). This is the model's differentiated axis; a frozen baseline LM cannot do it.

### 3.10 Generation

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--gen-tokens` | `gen_tokens` | 0 | `>0` = generate this many tokens after eval. |
| `--gen-temp` | `gen_temp` | 0.8 | Sampling temperature (0 = greedy/argmax). |
| `--gen-out` | `gen_out` | "" | Output path for generated token ids (JSON), decoded by `detok.py`. |

### 3.11 Infini-gram interpolation (v15)

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--ngram` | `ngram_path` | "" | Path to the token `.bin` indexed by `build_ngram.py` (expects `<path>.sa` alongside). |
| `--ngram-lambda` | `ngram_lambda` | 0.3 | Mixing weight: `P = (1-lambda)*P_brain + lambda*P_ngram`. |

Commits `25aedc1` / `c30462a` / `ef28dd5`. Best measured (50M index, lambda=0.3): static 3412 -> 1272 (-63%); adaptive 1930 -> 762 (-60%). Gains hold on in-distribution / corpus-like text where a long suffix matches; for novel chat input there is no match and it falls back to the bare model, so 762 does not transfer to conversation.

### 3.12 Checkpoint / resume (v15)

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--save` | `save_path` | "" | Save learned state after the run. |
| `--load` | `load_path` | "" | Load learned state; regenerate the fixed-random front-end from the seed; skip competitive feature learning. |

Saved tensors: `gwt`, `gbias`, `Wc`, `Ww`, `bc`, `bw`, mixture. Seed-derived fixed-random parts are regenerated by `brain_init`, so a checkpoint is ~0.4GB, not GBs (Section 8).

### 3.13 Miscellaneous and value-less flags

| Flag | Field | Default | Meaning |
|---|---|---|---|
| `--meta` | `meta_path` | "" | Path to `meta.json` (produced by the data builders). |
| `--out` | `out_path` | "" | Result output path. |
| `--seed` | `seed` | 13 | Master seed for all fixed-random structure. |
| `--refine-every` | `refine_every` | 50 | Competitive-learning refine interval. |
| `--bench` | `bench` | 0 | Throughput benchmark mode (no eval). |
| `--smoke` | (selects `cfg_smoke`) | off | Tiny end-to-end smoke configuration. |
| `--no-feat` | `learn_feat=0` | on | Disable competitive feature learning. |
| `--no-refine` | `refine_every=0` | on | Disable periodic refine. |

Competitive-learning struct fields not exposed as CLI flags: `feat_passes` (2), `feat_sample` (40000), `feat_eta` (0.05), `refine_sample` (4000), `refine_eta` (0.01).

---

## 4. Data pipeline

Three builders produce the uint16 token memmaps and `meta.json` files the brain reads; three tools score, compile, and decode. All token files are headerless uint16 (token `i` = bytes `[2i, 2i+2)`); the doc separator is id 0 (Pythia EOS).

### 4.1 `build_data.py` — single-source corpus

Builds the single-source WikiText-103 / Wikipedia uint16 memmap and the `meta.json` that the C brain reads unchanged. Used for the headline WikiText-103 comparison against Pythia-410M. (Run from `test_013_v3/tools/`.)

### 4.2 `build_blend.py` — v14 blended corpus

`tools/build_blend.py` streams multiple Hugging Face datasets, interleaves them by deterministic per-source token quotas into one uint16 memmap, serializes chat sources to a ChatML-lite template, holds out a chat-eval split by hash (no leakage), and writes `meta.json`. Tokenizer: `EleutherAI/pythia-410m` (`build_blend.py:33`); asserts `vocab < 65536` so uint16 is valid.

Source quotas (weights are fractions of the pool token budget, `build_blend.py:38`):

| Source | HF dataset | Weight |
|---|---|---|
| fineweb-edu | `HuggingFaceFW/fineweb-edu` (sample-10BT) | 0.42 |
| wikipedia | `wikimedia/wikipedia` (20231101.en) | 0.13 |
| cosmopedia | `HuggingFaceTB/cosmopedia` (web_samples_v2) | 0.13 |
| open-web-math | `open-web-math/open-web-math` | 0.10 |
| github-code | `codeparrot/github-code-clean` (Python-all) | 0.10 |
| ultrachat | `HuggingFaceH4/ultrachat_200k` (train_sft) | 0.075 |
| smoltalk | `HuggingFaceTB/smoltalk` (all) | 0.04 |
| dolly | `databricks/databricks-dolly-15k` | 0.005 |

Chat serialization is ChatML-lite: `<|system|>` / `<|user|>` / `<|assistant|>` ... `<|end|>` (`build_blend.py:61`). The held-out chat-eval split is drawn ONLY from ultrachat, deterministically by `md5(first_user_message) % 500 == 0` (`EVAL_HASH_MOD = 500`), capped at `EVAL_MAX_CONV = 2000` conversations; held-out conversations are excluded from train (`build_blend.py:104`). Outputs: `tokens.bin` (train), `chat_eval.bin` (held-out), `meta.json`. The brain re-samples the pool to its `--train-tokens` budget.

Run on a dedicated Internet-ON Kaggle notebook, separate from training:

```bash
python build_blend.py --pool-tokens 4500000000 --out-dir /kaggle/working
```

Defaults: `--pool-tokens 4_500_000_000`, `--train-tokens 6_000_000_000`, `--eval-tokens 1_500_000`, `--window 512`, `--stride 256`, `--seed 1337`. Do not pass `num_proc` to a streaming `map` (deadlocks); batched tokenization (`BATCH = 1000`) provides multi-core throughput on its own. `HF_TOKEN` is read from env if set (higher rate limits, not required for these public sources). The builder prints the suggested training command on completion.

### 4.3 `build_ngram.py` — v15 suffix-array (infini-gram) builder

`tools/build_ngram.py` builds a suffix array over a uint16 token corpus so the C eval can compute, for any context, the next-token distribution from the longest matching suffix in the corpus (unbounded n-gram / infini-gram). Outputs alongside the token `.bin`:

- `<bin>.sa` — int64 suffix array, `SA[i]` = start offset of the i-th smallest suffix
- `<bin>.sa.json` — meta `{n_tokens, token_path, sa_dtype, vocab, max_match}` (`max_match` = 16)

Construction prefers `pydivsufsort` (O(n), pip-installable, Internet-ON builder); the fallback is a fully vectorized pure-numpy prefix-doubling SA (`suffix_array_prefix_doubling`, `build_ngram.py:32`) that builds ~50M tokens in a couple of minutes with no internet. For corpora over 50M tokens the numpy fallback raises and demands `pydivsufsort` (`build_ngram.py:71`).

```bash
python build_ngram.py --tokens /kaggle/working/tokens.bin     # from a raw .bin
python build_ngram.py --meta   /kaggle/working/meta.json      # uses meta.train_path
python build_ngram.py --selftest                              # local correctness check
python build_ngram.py --tokens tokens.bin --max-tokens 50000000   # cap corpus (SA ~8 bytes/token)
```

The probability estimate is `(yes + 1e-6) / (tot + 1e-6 * 65536)` over next tokens of suffixes that match the longest suffix with count `>= min_count` (`build_ngram.py:79`). The host C query lives in `src/ngram.h` (mmap + binary-search longest-suffix backoff) and is validated byte-for-byte against the Python reference `ngram_prob_ref` via `--selftest`.

### 4.4 `run_llm.py` — Pythia baseline

`tools/run_llm.py` runs the EleutherAI/Pythia-410M baseline (405,334,016 params) and produces the WikiText-103 strided perplexity 17.19 (window 512). This is the number BrainFormer is compared against.

### 4.5 `scoreboard.py` and `detok.py`

`tools/scoreboard.py` compiles result records into the scoreboard. `tools/detok.py` decodes generated token-id JSON (from `--gen-out`) back into text using the corpus tokenizer.

---

## 5. Kaggle dual-T4 workflow

The session limit is 12 hours; the build and training are separated from the Internet-ON data build.

1. **Data build (Internet-ON notebook).** Run `build_blend.py` (or `build_data.py` for the single-source WikiText comparison) to produce `tokens.bin`, optional `chat_eval.bin`, and `meta.json` in `/kaggle/working`. Optionally run `build_ngram.py` to add `<bin>.sa`. Persist these as a Kaggle dataset so training runs Internet-OFF.

2. **Build the binary (training notebook).** `cd test_013_v3 && make`. Confirm with `./brain --meta <meta.json> --smoke`.

3. **Train Internet-OFF.** Use both T4s with `--n-gpus 2`. The resident-shard path (`--resident 1`, default) keeps the training shard in VRAM and gathers windows on GPU. Dual-GPU is local-SGD with parameter averaging every `--sync-every` blocks (default 25).

4. **Checkpoint against the 12h wall.** Pass `--save` so a 12h cutoff leaves a ~0.4GB resumable checkpoint; resume in the next session with `--load` (Section 8).

5. **Score.** Static and adaptive perplexity are produced in the same eval pass when `--adapt 1`; the infini-gram lambda sweep runs in one pass when `--ngram` is set. Compile with `scoreboard.py`.

Throughput context (4-layer config): ~171k tok/s, i.e. 50M tokens in ~5 minutes; peak ~334k tok/s (v3) up from ~44k tok/s (v13). The system is HBM-bandwidth-bound on the fp32 word-head scatter plus the all-`G` activation read.

---

## 6. Canonical quality-config command

The quality configuration is 4 layers, `n_classes 256`, multiscale on, sampled-softmax word head, adaptive eval. `build_blend.py` prints this command on completion (`build_blend.py:242`):

```bash
./brain --meta /kaggle/working/meta.json \
        --train-tokens 50000000 \
        --n-gpus 2 \
        --n-classes 256 \
        --n-layers 4 \
        --multiscale 1 \
        --neg-word 8 \
        --adapt 1
```

This config produced static 3412 / adaptive 1930 at 50M tokens (WikiText-103, deterministic eval). Token budget is the speed/quality dial via `--train-tokens`; note the "more data hurts" regression (Section 7) before increasing it without annealing.

To anneal `lr` and tune regularization at larger budgets, add:

```bash
        --lr 0.3 --lr-final 0.1 --wd 1e-5 --decay-every 200
```

---

## 7. The "more data hurts" regression and its fix

The same config gave static 3412 / adaptive 1930 at 50M tokens but static 3703 / adaptive 2057 at 500M tokens — worse, and deterministic, so not noise. Cause: a constant learning rate over-trains and weight decay (`keep = (1 - lr*wd)^decay_every` per event) over-regularizes at scale. Fix (commit `0e542be`): `--lr-final` cosine-anneals `lr` over the run; `--wd` and `--decay-every` were exposed (previously hardcoded 1e-5 / 200). Reproduce the regression by running the canonical command at `--train-tokens 500000000` with `--lr-final 1.0` (constant), then re-run with `--lr-final 0.1` to recover.

---

## 8. Checkpoint / resume workflow

The checkpoint saves only learned state (`gwt`, `gbias`, `Wc`, `Ww`, biases, mixture; ~0.4GB). The fixed-random front-end (token codes, granule connectivity indices, relay matrices) is regenerated deterministically from `--seed` by `brain_init`; `--load` skips the competitive feature-learning phase. The header is validated against the config and corpus on load.

Save at end of session:

```bash
./brain --meta /kaggle/working/meta.json --train-tokens 50000000 \
        --n-gpus 2 --n-classes 256 --n-layers 4 --multiscale 1 --neg-word 8 \
        --adapt 1 --save /kaggle/working/ckpt.bf
```

Resume next session (same seed, same config, same corpus):

```bash
./brain --meta /kaggle/working/meta.json --train-tokens 50000000 \
        --n-gpus 2 --n-classes 256 --n-layers 4 --multiscale 1 --neg-word 8 \
        --adapt 1 --load /kaggle/working/ckpt.bf
```

The seed and the architecture flags must match the saved run, because the front-end is regenerated rather than stored; a mismatch is rejected by the header validation.

---

## 9. Infini-gram build + eval workflow

End to end, on the same corpus the brain trains on:

1. Build the index over the training token file (50M-token index reproduces the reported result):

   ```bash
   python tools/build_ngram.py --tokens /kaggle/working/tokens.bin --max-tokens 50000000
   # -> /kaggle/working/tokens.bin.sa  +  /kaggle/working/tokens.bin.sa.json
   ```

2. Evaluate the brain with interpolation; the lambda sweep `{0.1 .. 0.9}` is scored for static and adaptive in a single pass:

   ```bash
   ./brain --meta /kaggle/working/meta.json --train-tokens 50000000 \
           --n-gpus 2 --n-classes 256 --n-layers 4 --multiscale 1 --neg-word 8 \
           --adapt 1 \
           --ngram /kaggle/working/tokens.bin --ngram-lambda 0.3
   ```

The host query (`src/ngram.h`) mmaps `<bin>` and `<bin>.sa` and backs off from the longest matching suffix (max match length 16). Eval interpolates `P = (1-lambda)*P_brain + lambda*P_ngram`. Reported result (50M index, lambda=0.3): static 3412 -> 1272 (-63%); adaptive 1930 -> 762 (-60%) — the single largest quality jump in the project. Scope: the gain holds on corpus-like / in-distribution text where a long suffix matches; novel chat inputs have no match and fall back to the bare model.

Validate the index logic against the Python reference before trusting an eval:

```bash
python tools/build_ngram.py --selftest
```

---

## 10. Numerical and engineering constraints to preserve on reproduction

- **Word head must stay fp32.** A bf16 word-head master rounds the ~3.7e-5 delta-rule updates to zero. `--ww-fp16 1` controls only fp16/bf16 storage/transport across GPUs; the on-GPU accumulating master remains fp32.
- **Batch is a speed knob, not a learning knob.** The batch-invariant step (`st = lr * lr_scale / LR_REF`, `LR_REF = 8192`, commit `ac5179e`) makes results invariant to `--batch`. Changing `--batch` should not change perplexity, only throughput.
- **`G % topk_groups == 0`.** Granule count per layer must be divisible by `topk_groups` for the two-level top-k grouping.
- **`C ~ sqrt(V)`.** `n_classes` near `sqrt(V)` minimizes the readout gather cost `K*(C + V/C)`; large `C` (small `S`) is faster but taxes hierarchical quality.
- **Deterministic eval.** Evaluation is deterministic; reported perplexity differences (including the regressions) are reproducible, not sampling noise.
- **Per-phase profile (4-layer, `--analytics 1`):** bind 1.1%, activate ~21%, topk ~24%, relay 6%, readout-fwd ~12%, scatter ~33.5%, bias 0.7%, mixture 0.9%. The scatter and all-`G` activation read are the HBM-bandwidth bottleneck.
