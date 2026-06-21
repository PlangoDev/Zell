# CUDA Implementation and Performance Engineering

Scope: the C/CUDA implementation of the deep cerebellar language model (BrainFormer test_013_v3), covering the single-translation-unit design, the device-kernel inventory, the cuBLAS fp16 tensor-core activation path, the bitonic top-k selector, the atomic-free bucketed readout scatter with device-side prefix sum, the on-device mixture update, the resident VRAM training shard with on-GPU window gather, dual-GPU local-SGD with host-staged averaging, lag-free per-phase analytics, the throughput history, the per-phase profile, the HBM-bandwidth ceiling analysis, the fp32 word-head numerical finding, and the checkpoint format. All references are to `test_013_v3/src/main.cu`, `test_013_v3/src/config.h`, `test_013_v3/src/cuda_util.h`, `test_013_v3/src/threads.h`, and `test_013_v3/Makefile`. Measurements were taken in June 2026 on Kaggle dual NVIDIA T4 (sm_75) GPUs against WikiText-103.

## 1. Build and translation-unit structure

The model compiles as one CUDA translation unit. `main.cu` contains all device kernels, the host orchestration (`Brain`/`Layer` structs and their methods), and `main()`. Two non-CUDA support files link alongside: `data.c` (memory-mapped token-bin reader plus `meta.json` parsing) and `threads.cpp` (a persistent CPU thread pool). The build is a single `nvcc` invocation (`test_013_v3/Makefile`):

```make
ARCH := -gencode arch=compute_75,code=sm_75
NVCCFLAGS := -O3 $(ARCH) -std=c++14 --use_fast_math \
             -Xcompiler "-O3 -funroll-loops -pthread"
LDFLAGS := -lcublas -lpthread
$(NVCC) $(NVCCFLAGS) src/main.cu src/data.c src/threads.cpp -o brain $(LDFLAGS)
```

The single-TU choice removes the relocatable-device-code link step: every `__global__` kernel and every host caller is visible in one compile, so `nvcc` resolves launches directly with no separate device-link pass. `--use_fast_math` and `sm_75` (Turing, the T4 architecture) are the load-bearing flags; `sm_75` is required for the FP16 tensor cores that the activation GEMM targets and for the `atomicAdd(double*)` used by the analytics margin accumulator (`k_usage_accum`, main.cu:575).

CUDA error handling is centralized in `cuda_util.h`. `CUDA_CHECK` and `CUBLAS_CHECK` wrap every runtime/cuBLAS call and abort with file:line on failure; `CUDA_KERNEL_CHECK()` (`cudaGetLastError()`) follows each kernel launch. Typed device allocation helpers (`dev_alloc<T>`, `dev_zalloc<T>`, `dev_free<T>`, `h2d<T>`, `d2h<T>`) take element counts rather than byte counts, which keeps the size arithmetic in the call sites in units of the logical tensor rather than bytes.

## 2. State layout

Two structs hold all state (main.cu:836-904).

`Layer` holds the per-layer parameters: the fixed sparse wiring `gidx[G,fan_in]` and its weights `gwt[G,fan_in]`; the per-granule selection bias `gbias[G]`; the running selection-count integrator `usage_acc[G]` for the load-balancing controller; the dense projection-transpose `projT[G,in_dim]` (fp32 master, rebuilt from `gwt`+`gidx`) and its fp16 mirror `projT_h[G,in_dim]`; the class head `Wc[G,C]`; the word head `Ww[G,Gcols]` (class-major, `Gcols = C*S`); the class/word biases `bc[C]`, `bw[Gcols]`; and the relay projection `relayR[G,relay_dim]`.

`Brain` holds the shared fixed front-end (`codes[V,D]`, `pos[ctx,D]`, `decay[n_scales,ctx]`), the frequency-rank hierarchy maps (`w2class[V]`, `w2slot[V]`, `class_word[C*S]`), the layer vector, the mixture logits (host `mix[L]` and device `dmix[L]`), the cuBLAS handle, and a large set of reusable device transients all sized for `step_chunk` (the in-flight micro-batch). Keeping the transients resident across steps avoids per-step `cudaMalloc`/`cudaFree`.

Defaults (config.h:120-147): `code_dim` (D) 128, `n_gran` (G) 12288/layer, `fan_in` 48, `k_active` (K) 64, `n_classes` (C) 256, `relay_dim` 128, `batch` 8192, `step_chunk` 2048, `ctx` 32. The default `n_layers` is 2 (speed-first); the quality configuration uses 4 layers with `--multiscale 1 --neg-word 8 --adapt 1`.

## 3. Kernel inventory

The following table enumerates every `__global__` kernel, its role, its launch geometry, and its phase. "B" is the chunk size `bc`; geometry is given as `<<<grid, block>>>`.

| Kernel (main.cu) | Role | Launch geometry | Phase |
|---|---|---|---|
| `k_build_projT` (40) | scatter sparse `gwt` into dense `projT` via `atomicAdd` | `(G*fan+255)/256, 256` | setup / refine |
| `k_f32_to_f16` (728) | fp32→fp16 cast (projT mirror, GEMM input) | `(n+255)/256, 256` | setup / activate |
| `k_bind` (53) | summed multi-scale HD context bind, one thread per (e,d) | `(B, ceil(D/128)), 128` | bind |
| `k_bind_rate` (75) | single-rate bind per layer for multiscale | `(B, ceil(D/128)), 128` | bind |
| `k_activate` (92) | custom fp32 granule activation (cuBLAS fallback) | `(ceil(Geff/256), B), 256` | activate |
| `k_sub_gbias` (720) | subtract `gbias` from fp16 activation after GEMM | `(B*Geff+255)/256, 256` | activate |
| `k_topk` (107) | exact K+1-pass argmax kWTA | `B, 256, smem` | topk |
| `k_topk_approx` (156) | two-level group top-`per_group` + K+1 argmax | `B, 256, smem` | topk |
| `k_topk_bitonic` (212) | two-level candidates + bitonic sort (default) | `B, 256, smem` | topk |
| `k_topk_groupmax` (275) | one-max-per-group single scan (`--fast-topk`) | `B, K, K*4` | topk |
| `k_add_offset` (734) | shift window-local winners to global ids (subsampling) | `(B*K+255)/256, 256` | topk |
| `k_relay_concat` (304) | JL relay + L2-norm, concat next-layer bind | `B, 128, relay_dim*4` | relay |
| `k_readout_fwd` (341) | full class + target-class word forward, errors, logp | `B, 256, (C+S)*4` | readout-fwd |
| `k_sample_negw` (408) | hash-sample negative word slots | `(B*N+255)/256, 256` | readout-fwd |
| `k_readout_fwd_negw` (418) | class full + sampled-softmax word forward | `B, 256, smem` | readout-fwd |
| `k_readout_logp` (528) | eval-only true-word logp per layer | `B, 256, (C+S)*4` | eval |
| `k_readout_scatter` (501) | atomic delta-rule scatter (M1 path) | `B, 256` | scatter |
| `k_bucket_count` (617) | per-granule selection counts | `(B*K+255)/256, 256` | scatter |
| `k_prefix_sum` (742) | device exclusive prefix sum `gcount→goff` | `1, 1` | scatter |
| `k_bucket_fill` (622) | fill (example,kslot) pairs into bucket slots | `(B*K+255)/256, 256` | scatter |
| `k_scatter_class_bucketed` (633) | one block per granule, class head, no atomics | `G, 256` | scatter |
| `k_scatter_word_bucketed` (651) | one block per granule, word head, no atomics | `G, 128` | scatter |
| `k_scatter_word_negw` (469) | atomic word scatter over (1+N) sampled cols | `B, 64` | scatter |
| `k_bias_update` (670) | atomic class+word bias delta | `B, 256` | bias |
| `k_bias_class` (488) | atomic class-bias delta (negw path) | `B, 256` | bias |
| `k_mixture_update` (752) | on-device EM mixture-logit update | `1, 256` | mixture |
| `k_gather_windows` (564) | on-GPU window gather from resident shard | `B, 64` | data |
| `k_usage_accum` (575) | granule usage + margin sum (analytics) | `(B*K+255)/256, 256` | analytics |
| `k_accum_counts` (588) | fold `gcount` into `usage_acc` (balance) | `(G+255)/256, 256` | balance |
| `k_balance_apply` (598) | homeostatic `gbias` controller step | `(G+255)/256, 256` | balance |
| `k_comp_accum` (686) | competitive k-means win-sum accumulation | `(B*K+255)/256, 256` | feature learn |
| `k_comp_apply` (700) | move `gwt` toward win-mean + renormalize | `(G+255)/256, 256` | feature learn |
| `k_decay` (1371) | multiplicative weight decay | `(n+255)/256, 256` | regularize |
| `k_fill_ones` (779) | reset per-example boost weights to 1 | `(B+255)/256, 256` | boost |
| `k_boost_weight` (783) | residual boost weight `1 - p_prev` | `(B+255)/256, 256` | boost |
| `k_gen_class` (792) | per-layer class softmax (generation) | `1, 256, C*4` | generation |
| `k_gen_within` (813) | per-layer within-class softmax (generation) | `1, 256, S*4` | generation |

The per-token pipeline through the hot loop is `bind → activate → topk → relay → readout-fwd → scatter → bias → mixture`, executed inside `encode_stack` (main.cu:1080) for the forward path and `brain_step` (main.cu:1282) for the learning path.

## 4. cuBLAS fp16 tensor-core activation

The granule activation is the dense projection `a[B,Geff] = x[B,in_dim] @ projT[Geff,in_dim]^T - gbias`. With G=12288 granules per layer and `in_dim` up to `relay_dim+D = 256`, this is the largest GEMM in the pipeline. It runs on the Turing FP16 tensor cores via `cublasGemmEx` (main.cu:1123):

```c
k_f32_to_f16<<<(xn+255)/256,256>>>(xin, b->dxh, xn);   /* cast input to fp16 */
cublasGemmEx(b->cub, CUBLAS_OP_T, CUBLAS_OP_N, Geff, bc, in_dim,
             &one, ly.projT_h + (size_t)off*in_dim, CUDA_R_16F, in_dim,
             b->dxh,                                    CUDA_R_16F, in_dim,
             &zero, b->dact,                            CUDA_R_16F, Geff,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
k_sub_gbias<<<(n+255)/256,256>>>(b->dact, ly.gbias + off, bc, Geff);
```

Both operands and the output are fp16 (`CUDA_R_16F`); the accumulator is fp32 (`CUBLAS_COMPUTE_32F`), and `CUBLAS_GEMM_DEFAULT_TENSOR_OP` selects the tensor-core path. The precision argument is justified by the downstream consumer: the activation feeds only the kWTA argmax (a comparison of granule scores), not a gradient. Half precision on the inputs and outputs perturbs the score magnitudes but rarely reorders the top-K, so the argmax winners are essentially unchanged while the matmul runs roughly an order of magnitude faster than fp32 on the T4 (the kernel comment notes ~8x; the realized end-to-end activation share is ~21% of step time, Section 9). The `gbias` subtraction is fused as a separate elementwise pass (`k_sub_gbias`, main.cu:720) because cuBLAS has no per-row bias hook for this layout; it reads and rewrites the fp16 activation matrix once.

The fp16 weight mirror `projT_h` is materialized once at allocation (`layer_alloc`, main.cu:991) by `k_build_projT` (dense scatter of the sparse wiring into fp32 `projT`) followed by `k_f32_to_f16`. It is rebuilt whenever competitive learning changes `gwt` (`brain_competitive`, main.cu:1636; `load_brain`, main.cu:1812; `average_granules`, main.cu:1729) so the tensor-core path always reflects the current granule basis. The fp32 `projT` master is retained for the non-cuBLAS fallback `k_activate` (main.cu:92), selectable with `--cublas 0`, which computes the same activation in fp32 with one thread per (e,g) and no tensor cores.

When `--gran-frac < 1` (granule dropout), the GEMM is sliced: only `Geff = G*gran_frac` rows of `projT_h` are activated, starting at a rotating offset `off` (`encode_stack`, main.cu:1108-1124), and `k_add_offset` shifts the window-local winners back to global granule ids afterward. This cuts both the activation and top-k bandwidth by ~`1/gran_frac` at a quality cost.

## 5. Top-k (kWTA) selection

Four top-k kernels exist; `encode_stack` (main.cu:1145-1156) selects among them at launch time:

1. `k_topk` (exact, main.cu:107). K+1 sequential block-argmax passes over all G activations, removing each winner (set to `-inf`) between passes. The (K+1)th value is the ReLU margin threshold subtracted from the winner values. Correctness reference; selected by `--exact-topk` or when the approx parameters are invalid.

2. `k_topk_approx` (main.cu:156). Phase 1: each thread scans a contiguous group of `G/groups` granules and keeps its local top-`per_group` by insertion sort into a register list (`MAX_PG = 32` cap). Phase 2: K+1 block-argmax passes over the `groups*per_group` candidates. ~1 scan of G plus K+1 short passes. Fallback when the candidate count is not a power of two.

3. `k_topk_bitonic` (default, main.cu:212). Same phase-1 candidate generation, then a bitonic sort of the M=`groups*per_group` candidates (M a power of two; default 256*2=512). The sort runs in `log2(M)*(log2(M)+1)/2` parallel compare-exchange stages with a `__syncthreads()` per stage, after which the largest K sit at the array tail and the (K+1)th largest is the margin threshold. This replaces the K+1 sequential argmax passes of the approx kernel — which made selection the throughput bottleneck — with `O(log^2 M)` parallel stages, producing the same winners as approx (exact over the candidate set) in far fewer steps.

4. `k_topk_groupmax` (`--fast-topk`, main.cu:275). G is split into exactly K contiguous groups; each of K threads takes its group's max, giving K winners in a single scan with no selection passes. The margin uses the largest per-group runner-up. Requires `K | G` and K a power of two (else `encode_stack` falls back). Because granule index is uncorrelated with activation under the random wiring, top-1-per-group closely approximates the true top-K; this is the fastest and crudest selector.

All four emit `idx[B,K]` (winning granule ids) and `val[B,K]` (ReLU margins = winner activation minus the (K+1)th value, floored at 0). The activation matrix is read in fp16, halving the top-k read bandwidth versus an fp32 activation.

## 6. Atomic-free bucketed readout scatter

The readout learning step is a local delta rule: `error = softmax - onehot`, scattered into the active top-K granule rows of `Wc` and `Ww`. The naive realization (`k_readout_scatter`, main.cu:501) uses `atomicAdd` to absorb the collisions that occur when two examples in a chunk select the same granule. With K=64 winners per example and `step_chunk`=2048 examples, the word head `Ww[G,Gcols]` receives `B*K*S` atomic updates per layer per step into a `G*Gcols`-element fp32 tensor — the dominant write traffic in the system.

The default fast path (`fast_scatter=1`, `readout_scatter`, main.cu:1203-1243) eliminates the atomics on this dominant payload by reorganizing the scatter as a counting sort over granules:

```
Stage 0  k_bucket_count   gcount[g] += #(example,kslot) pairs selecting g   (atomic, small array)
Stage 1  k_prefix_sum     goff[0..G] = exclusive prefix sum of gcount        (device, single thread)
Stage 2  k_bucket_fill    write (e,k) into bucket slot goff[g]+cursor[g]++   (atomic on cursor only)
Stage 3a k_scatter_class_bucketed   one block per granule g, threads over C  (no atomics)
Stage 3b k_scatter_word_bucketed    one block per granule g, threads over S  (no atomics)
```

After bucketing, each granule row is owned by exactly one block (`g = blockIdx.x`, main.cu:633,651), so the dominant updates into `Wc[g,:]` and `Ww[g,base:base+S]` have no cross-block contention and need no atomics. The only remaining atomics are on the small `gcount[G]`/`gcursor[G]` counters (Stage 0/2) and on the tiny shared biases `bc[C]`/`bw[Gcols]` (`k_bias_update`, main.cu:670).

The prefix sum is on-device (`k_prefix_sum`, main.cu:742) — a single-thread serial scan of G=12288 elements. G is small and the scan is off the bandwidth-bound path, but running it on-device removes the host round-trip and the synchronizing `cudaMemcpy`/`cudaDeviceSynchronize` that an earlier host-prefix-sum version required; that stall is what made the first bucketed version slow. The cursor reset (`gcursor`) uses `cudaMemsetAsync` (main.cu:1219) to overlap with the preceding kernel.

A sampled-softmax variant (`readout_scatter_negw`, main.cu:1253, enabled by `--neg-word`) keeps the class head bucketed but reverts the word head to a small atomic scatter (`k_scatter_word_negw`, main.cu:469) over only the true slot plus N sampled negatives, cutting the word-head write volume by ~`S/(1+N)`. With `--neg-word 8` this gave ~+14% throughput at quality-neutral perplexity.

## 7. On-device mixture update

The L cerebellar layers are parallel experts combined by a learned probabilistic mixture `P(y) = sum_l w_l P_l(y)`, where the mixture logits live in `dmix[L]` on the device. The update is an EM-style local rule run entirely on-GPU by `k_mixture_update` (main.cu:752): a single block whose threads split over the chunk's examples, compute per-example responsibilities (`logf(w_l) + logp_l`, softmaxed over l), reduce them into shared memory, and step `mix[l] += mix_lr*(mean_responsibility_l - prior_l)`. Each layer's true-word logprob `logp[L,B]` is produced by the readout-forward kernels in the same step.

The point of doing this on-device is that the mixture is the only cross-layer coupling in the model, and a host-side EM step would force a `dlogp` device-to-host copy and a synchronize on every step. Keeping it on-device (one block, no host transfer) means `brain_step` (main.cu:1282) issues the entire forward + all-layer learn + mixture update as an uninterrupted stream of kernel launches with zero host syncs in the hot loop. The host mixture copy `mix[L]` is synced from `dmix` only at eval/JSON boundaries (`d2h(brain->mix, brain->dmix, ...)`, main.cu:1969).

## 8. Resident VRAM shard and on-GPU window gather

Training samples random `ctx+1`-length windows from the corpus. The naive approach gathers each chunk's windows on the host and copies `B*ctx` int ids to the device every step. The resident path (`--resident 1`, default) instead keeps the entire training shard in VRAM as a `uint16` array (`dtoks`) and gathers windows directly on the GPU.

At startup, `run_rank` (main.cu:1843-1854) queries free VRAM with `cudaMemGetInfo` and uploads the shard only if it fits within 3/4 of free memory (`bytes < freeb*3/4`); otherwise it logs the fallback and uses host gather. When resident, `fill_windows` (main.cu:1183) copies only the B random start positions (`dpos`, a `long[chunk]`) to the device and launches `k_gather_windows` (main.cu:564), which fills `dX[B,ctx]` and `dY[B]` from `dtoks` with one block per example and `blockDim`=64 threads cooperatively copying the context. This replaces a `B*ctx*4`-byte H2D per step with a `B*8`-byte H2D, removing per-step PCIe traffic from the hot path. The host-gather fallback (main.cu:1190) reconstructs windows into reused host buffers and does the full H2D.

## 9. Per-phase profile and HBM-bandwidth ceiling

Per-phase timing is gated by `Brain::timing_on`, which is 0 in the training hot loop (no inserted syncs) and 1 only during `brain_profile` (main.cu:1339), a short timed burst run AFTER training (Section 10). `phase_tic`/`phase_toc` (main.cu:907) bracket each phase with `cudaDeviceSynchronize` and accumulate elapsed seconds into per-phase counters.

The profiled per-phase breakdown for the 4-layer quality configuration:

| Phase | Share of step time |
|---|---|
| bind | 1.1% |
| activate | ~21% |
| topk | ~24% |
| relay | 6% |
| readout-fwd | ~12% |
| scatter | ~33.5% |
| bias | 0.7% |
| mixture | 0.9% |

Scatter dominates at ~33.5%, followed by topk (~24%) and activate (~21%). The system is HBM-bandwidth-bound on two reads/writes: the fp32 word-head scatter (writing into the `G*Gcols` `Ww` tensor) and the all-G activation read (the kWTA must read every granule's score). These are memory-traffic limited, not compute limited — the tensor-core GEMM saturates bandwidth before it saturates FLOPs at this shape. The readout cost scales as `K*(C + V/C)`, minimized at `C ≈ sqrt(V)`; the default `n_classes`=256 sits near `sqrt(V)` for the working vocabularies, which is why the hierarchical readout (class head plus per-class word head) is far cheaper than a flat `K*V` softmax scatter would be (`brain_macs_per_token`, main.cu:1676).

The bandwidth ceiling bounds the obvious speed lever. The largest theoretical win is "stop computing all G granule activations" via coarse routing (IVF-style), but skipping granules forfeits the dense tensor-core GEMM throughput; on the T4 the realistic outcome is roughly break-even rather than the >5x seen in non-tensor-core settings, so it was not built. `--gran-frac` is the implemented partial version: it shrinks the activated set but trades quality.

## 10. Lag-free analytics

All diagnostic instrumentation is kept off the training critical path. The hot loop runs with `timing_on=0`, so no phase has a synchronize. After training completes, `brain_profile` (main.cu:1339) sets `timing_on=1`, runs a short burst of `brain_step` calls on freshly sampled positions, and accumulates the per-phase timings; `brain_health` (main.cu:1645) then samples granule usage (`k_usage_accum`) to compute the dead-granule fraction, the usage Gini coefficient, and the mean kWTA margin. Because this profiling burst runs after the timed training section, the per-phase syncs never touch the reported training throughput. The progress line during training reports both a running-average and an instantaneous tok/s (`print_progress`, main.cu:1357), and the single `cudaDeviceSynchronize` it needs fires only once per `progress_every` blocks (main.cu:1950).

The health metrics from this path informed the load-balancing decision: the dead-granule fraction measured 0.0%, so there was no idle capacity to reclaim, and the usage Gini reflected useful specialization rather than waste. The homeostatic `gbias` controller (`k_balance_apply`, main.cu:598) lowered Gini from 0.348 to 0.251 but slightly worsened perplexity, and was dropped.

## 11. Dual-GPU local-SGD with host-staged averaging

Multi-GPU (`--n-gpus 2`) uses data-parallel local-SGD: one `Brain` replica per T4, each training on a disjoint corpus shard, with periodic parameter averaging. `main` (main.cu:2149) spawns rank 1 in a `std::thread` and runs rank 0 inline; each `run_rank` calls `cudaSetDevice(rank)` (main.cu:1837) to bind its replica to its GPU. Shards are split by rank in `run_rank` (main.cu:1879).

Averaging avoids P2P entirely and goes through host staging (`avg_tensor`, main.cu:1701):

```c
d2h(sh->hbuf[rank], Tdev, n);                  /* each rank stages its tensor */
pthread_barrier_wait(&sh->bar);
if (rank == 0)                                  /* rank 0 averages both copies */
    for (i) sh->hbuf[0][i] = 0.5f*(sh->hbuf[0][i] + sh->hbuf[1][i]);
pthread_barrier_wait(&sh->bar);
h2d(Tdev, sh->hbuf[0], n);                      /* both ranks upload the mean */
pthread_barrier_wait(&sh->bar);
```

A `pthread_barrier_t` (with `world`=2 participants) synchronizes the two host threads around the three phases (stage, average, upload). The host staging buffers `hbuf[2]` are sized to the largest tensor, `Ww` (`G*Gcols`), allocated in `main` (main.cu:2153).

Two averaging scopes exist. `average_granules` (main.cu:1715) is called once after competitive feature learning to make every replica share the same granule encoder — readout averaging across replicas is only valid if their granule bases match, so `gwt` (and `gbias` when balancing) is synced first and `projT`/`projT_h` rebuilt. `average_readouts` (main.cu:1735) runs every `sync_every` blocks (default 25) and at the end of training (main.cu:1960), averaging `Wc`, `Ww`, `bc`, `bw`, and `dmix`. When load-balancing is active, `gbias` is averaged in lock-step and each replica's `usage_acc` integrator is reset (main.cu:1748) so the controller restarts cleanly from the averaged bias rather than carrying stale windup.

The step is batch-invariant so that dual-GPU and single-GPU learn at the same rate per token. The per-step learning rate is `st = lr * lr_scale / LR_REF` with `LR_REF = 8192` (main.cu:1290), normalizing by a fixed reference batch rather than the actual batch. Under the older `st = lr/B`, a larger batch took the same-magnitude step over fewer steps and therefore learned less in total; the fixed reference makes batch (and GPU count) a pure speed knob (linear scaling rule).

## 12. fp32 word-head numerical finding

The word head `Ww` must remain fp32. The delta-rule update magnitude is `st * val * err`, on the order of `3.7e-5` per event for the default learning rate. A bf16 master for `Ww` rounds updates of that magnitude to zero — the increment is smaller than the bf16 representable spacing near the accumulated weight value — so learning silently stalls. The word head therefore stays fp32 (`Ww` is `float*` throughout, e.g. `Layer::Ww`, main.cu:846; the bucketed word scatter writes fp32, main.cu:651). This is the binding precision constraint: the activation GEMM tolerates fp16 because it feeds only an argmax, but the readout accumulation cannot, because it integrates many tiny signed deltas whose information is destroyed by low-mantissa rounding. The class head `Wc` and the biases are fp32 for the same reason. The `--ww-fp16` config flag exists (config.h:73) but the fp32 finding governs the production configuration.

## 13. Checkpoint format

Checkpointing (`--save`/`--load`) persists only the learned state; the entire fixed-random front-end (`codes`, `pos`, `decay`, `gidx`, `relayR`) and the frequency-derived hierarchy are regenerated deterministically from the seed by `brain_init`, so a checkpoint is ~0.4 GB (dominated by `Ww`) rather than the full model footprint.

The file format (`save_brain`/`load_brain`, main.cu:1761-1822) begins with an 11-int header:

```
[ BF_CKPT_MAGIC=0x42466631 ('BFf1'), version=1, V, G, K, C, S, Gcols, relay_dim, L, fan_in ]
```

followed by one `in_dim` int per layer, then per layer the six learned tensors in order — `gwt[G*fan]`, `gbias[G]`, `Wc[G*C]`, `Ww[G*Gcols]`, `bc[C]`, `bw[Gcols]` — all fp32, and finally `dmix[L]`. On load, the header is validated field-by-field against the current config and corpus (V, G, K, C, S, Gcols, L, fan_in) and each layer's `in_dim`; a mismatch refuses the load rather than producing a silently corrupt model (main.cu:1787). After loading `gwt`, the dense `projT` and its fp16 mirror are rebuilt (main.cu:1808-1813), and competitive feature learning is skipped (main.cu:1862). Both ranks load the same checkpoint in dual-GPU mode (main.cu:1858). This format is the mechanism for resuming across the 12-hour Kaggle session limit.

## 14. Throughput history

| Stage | Throughput | Notes |
|---|---|---|
| v13 (pre-rewrite) | ~44k tok/s | baseline single-layer learned implementation |
| v3 peak | ~334k tok/s | speed-first config with all M2 paths enabled |
| v3 4-layer quality | ~171k tok/s | 50M tokens in ~5 min; the quality configuration |

The ~7.6x improvement from v13 to the v3 peak is attributable to the cuBLAS fp16 tensor-core activation (Section 4), the bitonic top-k replacing K+1 sequential argmax passes (Section 5), the atomic-free bucketed scatter with on-device prefix sum (Section 6), the on-device mixture update removing per-step host syncs (Section 7), and the resident shard removing per-step H2D traffic (Section 8). The remaining ceiling is HBM bandwidth on the word-head scatter and the all-G activation read (Section 9); incremental levers beyond this point (CUDA graphs, coalesced scatter, dual-GPU layer-sharding, int8 readout) are bandwidth-incremental rather than structural.
