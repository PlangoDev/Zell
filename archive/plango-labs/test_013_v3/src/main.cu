/* ─────────────────────────────────────────────────────────────────────────────
 * main.cu — BrainFormer v13-v3, the C/CUDA deep cerebellar LM (single CUDA TU).
 *
 * This file holds ALL device code + the DeepBrain host orchestration + main(),
 * compiled as one translation unit so no relocatable-device-code linking is
 * needed. data.c (mmap + meta) and threads.cpp (CPU pool) link alongside.
 *
 * MILESTONE STATE: M1 (single-GPU, full depth + learned mixture, fp32 masters,
 * RANDOM granules, simple atomic readout scatter, custom kernels). The fast
 * bucketed scatter / CUDA graphs / resident-shard / cuBLAS tensor cores (M2) and
 * dual-GPU (M5) and competitive granule learning (M4) layer on top of this; the
 * kernels here are written to be correct first, fast second.
 *
 * No backprop anywhere: per-layer delta-rule readout (deep supervision) + a
 * learned probabilistic mixture (local EM rule). Architecture mirrors
 * test_013_v2/bf/*.py exactly so its perplexity is the correctness oracle.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "common.h"
#include "config.h"
#include "data.h"
#include "cuda_util.h"
#include "threads.h"
#include "ngram.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <thread>
#include <pthread.h>

#define NEG_INF (-1e30f)

/* ─────────────────────────── device kernels ───────────────────────────────── */

/* Build the dense projection-transpose projT[G, in_dim] from the sparse wiring:
 * projT[g, gidx[g,f]] += gwt[g,f]. atomicAdd handles duplicate input indices. */
__global__ void k_build_projT(const int *gidx, const float *gwt, float *projT,
                              int G, int fan_in, int in_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   /* one thread per (g,f) */
    if (t >= G * fan_in) return;
    int g = t / fan_in;
    int d = gidx[t];
    if (d < 0 || d >= in_dim) return;
    atomicAdd(&projT[(size_t)g * in_dim + d], gwt[t]);
}

/* Hyperdimensional context binding:
 * bound[e,d] = (1/n_scales) * sum_s sum_p decay[s,p] * codes[X[e,p],d] * pos[p,d]
 * X[e,p] in [0,V). One thread per (e,d). */
__global__ void k_bind(const int *X, const float *codes, const float *pos,
                        const float *decay, float *bound,
                        int B, int ctx, int D, int n_scales) {
    int e = blockIdx.x;                       /* example */
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= B || d >= D) return;
    float acc = 0.0f;
    for (int p = 0; p < ctx; p++) {
        int tok = X[e * ctx + p];
        float cv = codes[(size_t)tok * D + d];
        float pv = pos[p * D + d];
        float cp = cv * pv;
        float w = 0.0f;
        for (int s = 0; s < n_scales; s++) w += decay[s * ctx + p];
        acc += w * cp;
    }
    bound[(size_t)e * D + d] = acc / (float)n_scales;
}

/* single-rate context binding (for multi-timescale: each layer gets its own rate
 * so deeper layers see longer-range context). bound[e,d] = sum_p exp(-dist*rate)
 * * codes[X[e,p],d] * pos[p,d]. */
__global__ void k_bind_rate(const int *X, const float *codes, const float *pos,
                            float rate, float *bound, int B, int ctx, int D) {
    int e = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= B || d >= D) return;
    float acc = 0.0f;
    for (int p = 0; p < ctx; p++) {
        int tok = X[e * ctx + p];
        float cp = codes[(size_t)tok * D + d] * pos[p * D + d];
        int dist = ctx - 1 - p;
        acc += expf(-(float)dist * rate) * cp;
    }
    bound[(size_t)e * D + d] = acc;
}

/* Granule activation: a[e,g] = sum_d x[e,d]*projT[g,d] - gbias[g].
 * One thread per (e,g); in_dim is small (<=256). */
__global__ void k_activate(const float *x, const float *projT, const float *gbias,
                           __half *a, int B, int G, int in_dim) {
    int e = blockIdx.y;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= B || g >= G) return;
    const float *xr = x + (size_t)e * in_dim;
    const float *pr = projT + (size_t)g * in_dim;
    float acc = 0.0f;
    for (int d = 0; d < in_dim; d++) acc += xr[d] * pr[d];
    a[(size_t)e * G + g] = __float2half(acc - gbias[g]);
}

/* kWTA top-(K+1) per example by K+1 argmax passes (exact; M1 correctness).
 * Mutates `a` (sets winners to -inf). Emits idx[e,K] and val[e,K] = winner -
 * (K+1)th value (relu margin). One block per example. */
__global__ void k_topk(__half *a, int *idx, float *val, int B, int G, int K) {
    int e = blockIdx.x;
    if (e >= B) return;
    __half *ar = a + (size_t)e * G;
    extern __shared__ char smem[];
    float *sval = (float *)smem;               /* blockDim.x floats */
    int   *sidx = (int *)(sval + blockDim.x);  /* blockDim.x ints   */
    int tid = threadIdx.x, nth = blockDim.x;

    float thr = NEG_INF;
    for (int kk = 0; kk <= K; kk++) {          /* K winners + 1 threshold */
        float best = NEG_INF; int bi = -1;
        for (int g = tid; g < G; g += nth) {
            float v = __half2float(ar[g]);
            if (v > best) { best = v; bi = g; }
        }
        sval[tid] = best; sidx[tid] = bi;
        __syncthreads();
        for (int off = nth / 2; off > 0; off >>= 1) {
            if (tid < off) {
                if (sval[tid + off] > sval[tid]) {
                    sval[tid] = sval[tid + off];
                    sidx[tid] = sidx[tid + off];
                }
            }
            __syncthreads();
        }
        int win = sidx[0];
        float winv = sval[0];
        if (kk < K) {
            if (tid == 0) { idx[e * K + kk] = win; val[e * K + kk] = winv; }
            if (tid == 0 && win >= 0) ar[win] = __float2half(NEG_INF);   /* remove winner */
            __syncthreads();
        } else {
            thr = winv;                         /* (K+1)th value = margin threshold */
        }
    }
    /* apply relu margin */
    for (int kk = tid; kk < K; kk += nth) {
        float m = val[e * K + kk] - thr;
        val[e * K + kk] = m > 0.0f ? m : 0.0f;
    }
}

/* Two-level APPROX kWTA: split G into `groups` blocks of gs=G/groups; each thread
 * finds the top `per_group` within its groups (local insertion sort), then a block
 * argmax over the groups*per_group candidates picks the top K+1. ~1 scan of G
 * instead of the exact kernel's K+1 scans. One block per example. */
#define MAX_PG 32
__global__ void k_topk_approx(const __half *a, int *idx, float *val, int B, int G, int K,
                              int groups, int per_group) {
    int e = blockIdx.x;
    if (e >= B) return;
    const __half *ar = a + (size_t)e * G;
    int gs = G / groups;
    int Ccand = groups * per_group;
    extern __shared__ char sm[];
    float *cval = (float *)sm;                 /* [Ccand] */
    int   *cidx = (int *)(cval + Ccand);       /* [Ccand] */
    float *rval = (float *)(cidx + Ccand);     /* [nth]   */
    int   *ridx = (int *)(rval + blockDim.x);  /* [nth]   */
    int tid = threadIdx.x, nth = blockDim.x;

    /* phase 1: per-group local top-per_group (insertion into a sorted-desc list) */
    for (int gp = tid; gp < groups; gp += nth) {
        float lv[MAX_PG]; int li[MAX_PG];
        for (int j = 0; j < per_group; j++) { lv[j] = NEG_INF; li[j] = -1; }
        int base = gp * gs;
        for (int i = 0; i < gs; i++) {
            float x = __half2float(ar[base + i]);
            if (x > lv[per_group - 1]) {
                int p = per_group - 1;
                while (p > 0 && lv[p - 1] < x) { lv[p] = lv[p - 1]; li[p] = li[p - 1]; p--; }
                lv[p] = x; li[p] = base + i;
            }
        }
        for (int j = 0; j < per_group; j++) { cval[gp * per_group + j] = lv[j]; cidx[gp * per_group + j] = li[j]; }
    }
    __syncthreads();

    /* phase 2: top K+1 among the candidates via K+1 block-argmax passes */
    float thr = NEG_INF;
    for (int kk = 0; kk <= K; kk++) {
        float best = NEG_INF; int bi = -1;
        for (int c = tid; c < Ccand; c += nth) if (cval[c] > best) { best = cval[c]; bi = c; }
        rval[tid] = best; ridx[tid] = bi;
        __syncthreads();
        for (int off = nth / 2; off > 0; off >>= 1) {
            if (tid < off && rval[tid + off] > rval[tid]) { rval[tid] = rval[tid + off]; ridx[tid] = ridx[tid + off]; }
            __syncthreads();
        }
        int wc = ridx[0]; float wv = rval[0];
        if (kk < K) {
            if (tid == 0) { idx[e * K + kk] = (wc >= 0 ? cidx[wc] : 0); val[e * K + kk] = wv; if (wc >= 0) cval[wc] = NEG_INF; }
            __syncthreads();
        } else thr = wv;
    }
    for (int kk = tid; kk < K; kk += nth) { float m = val[e * K + kk] - thr; val[e * K + kk] = m > 0.0f ? m : 0.0f; }
}

/* BITONIC kWTA: phase-1 builds groups*per_group candidates (many small groups =
 * near-exact top-K), then a bitonic SORT of those M candidates (M a power of two)
 * in ~log2(M)^2/2 PARALLEL stages -> take the largest K. Replaces the K+1
 * sequential argmax passes that made approx-topk the bottleneck. Same winners as
 * approx (exact over candidates), far fewer steps. One block per example. */
__global__ void k_topk_bitonic(const __half *a, int *idx, float *val, int B, int G, int K,
                               int groups, int per_group) {
    int e = blockIdx.x;
    if (e >= B) return;
    const __half *ar = a + (size_t)e * G;
    int gs = G / groups;
    int M = groups * per_group;                 /* power of two (checked by caller) */
    extern __shared__ char sm[];
    float *sv = (float *)sm;                     /* [M] candidate values */
    int   *si = (int *)(sv + M);                 /* [M] candidate granule ids */
    int tid = threadIdx.x, nth = blockDim.x;

    /* phase 1: per-group local top-per_group */
    for (int gp = tid; gp < groups; gp += nth) {
        float lv[MAX_PG]; int li[MAX_PG];
        for (int j = 0; j < per_group; j++) { lv[j] = NEG_INF; li[j] = -1; }
        int base = gp * gs;
        for (int i = 0; i < gs; i++) {
            float x = __half2float(ar[base + i]);
            if (x > lv[per_group - 1]) {
                int p = per_group - 1;
                while (p > 0 && lv[p - 1] < x) { lv[p] = lv[p - 1]; li[p] = li[p - 1]; p--; }
                lv[p] = x; li[p] = base + i;
            }
        }
        for (int j = 0; j < per_group; j++) { sv[gp * per_group + j] = lv[j]; si[gp * per_group + j] = li[j]; }
    }
    __syncthreads();

    /* phase 2: bitonic sort ASCENDING (largest end up at the top of the array) */
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < M; i += nth) {
                int l = i ^ j;
                if (l > i) {
                    bool ascend = ((i & k) == 0);
                    bool swap = ascend ? (sv[i] > sv[l]) : (sv[i] < sv[l]);
                    if (swap) {
                        float tv = sv[i]; sv[i] = sv[l]; sv[l] = tv;
                        int ti = si[i]; si[i] = si[l]; si[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
    /* largest K are at the end; threshold = the (K+1)th largest */
    float thr = (M - K - 1 >= 0) ? sv[M - K - 1] : NEG_INF;
    for (int kk = tid; kk < K; kk += nth) {
        int pos = M - 1 - kk;                    /* kk-th largest */
        idx[e * K + kk] = si[pos];
        float m = sv[pos] - thr;
        val[e * K + kk] = m > 0.0f ? m : 0.0f;
    }
}

/* FAST kWTA: split G into exactly K contiguous groups; each thread takes its
 * group's MAX -> K winners in a SINGLE scan of G (no K-pass selection). The
 * margin uses the largest per-group runner-up as the threshold. Requires K | G
 * and K a power of two (else encode_stack falls back to approx/exact). Because
 * granule index is uncorrelated with activation (random wiring), top-1-per-group
 * closely approximates the true top-K. This is the kernel that kills the topk
 * bottleneck. One block (K threads) per example. */
__global__ void k_topk_groupmax(const __half *a, int *idx, float *val, int B, int G, int K) {
    int e = blockIdx.x;
    if (e >= B) return;
    const __half *ar = a + (size_t)e * G;
    int tid = threadIdx.x;                 /* 0..K-1, one group each */
    int gs = G / K;
    int base = tid * gs;
    int end = (tid == K - 1) ? G : (base + gs);
    float m1 = NEG_INF, m2 = NEG_INF; int i1 = base;
    for (int i = base; i < end; i++) {
        float x = __half2float(ar[i]);
        if (x > m1) { m2 = m1; m1 = x; i1 = i; }
        else if (x > m2) m2 = x;
    }
    extern __shared__ float s2[];           /* K runner-ups */
    s2[tid] = m2;
    __syncthreads();
    for (int off = K / 2; off > 0; off >>= 1) {
        if (tid < off && s2[tid + off] > s2[tid]) s2[tid] = s2[tid + off];
        __syncthreads();
    }
    float thr = s2[0];
    idx[e * K + tid] = i1;
    float m = m1 - thr;
    val[e * K + tid] = m > 0.0f ? m : 0.0f;
}

/* Relay: r[e,:] = normalize( sum_k val[e,k] * relayR[idx[e,k], :] ) * sqrt(relay_dim),
 * then write next-layer input = concat(r, bound). One block per example. */
__global__ void k_relay_concat(const int *idx, const float *val, const float *relayR,
                                const float *bound, float *xnext,
                                int B, int K, int relay_dim, int D) {
    int e = blockIdx.x;
    if (e >= B) return;
    extern __shared__ float sr[];              /* relay_dim floats */
    int tid = threadIdx.x, nth = blockDim.x;
    for (int j = tid; j < relay_dim; j += nth) sr[j] = 0.0f;
    __syncthreads();
    for (int k = 0; k < K; k++) {
        int g = idx[e * K + k];
        float v = val[e * K + k];
        if (v == 0.0f) continue;
        const float *rr = relayR + (size_t)g * relay_dim;
        for (int j = tid; j < relay_dim; j += nth) atomicAdd(&sr[j], v * rr[j]);
    }
    __syncthreads();
    /* L2 norm (block reduction in shared scratch reusing sr is unsafe; recompute) */
    __shared__ float ssum;
    if (tid == 0) ssum = 0.0f;
    __syncthreads();
    float local = 0.0f;
    for (int j = tid; j < relay_dim; j += nth) local += sr[j] * sr[j];
    atomicAdd(&ssum, local);
    __syncthreads();
    float scale = sqrtf((float)relay_dim) / (sqrtf(ssum) + 1e-6f);
    int in_dim = relay_dim + D;
    float *xr = xnext + (size_t)e * in_dim;
    for (int j = tid; j < relay_dim; j += nth) xr[j] = sr[j] * scale;
    for (int j = tid; j < D; j += nth) xr[relay_dim + j] = bound[(size_t)e * D + j];
}

/* Readout R1 forward for one example over the TARGET class only (hierarchical
 * cheap path): class logits lc[C] over all classes, word logits lw[S] over the
 * target class's S contiguous columns. Produces errc[B,C], errw[B,S] (delta-rule
 * error = softmax - onehot) and logp[B] (true-word logprob, for the mixture).
 * One block per example; blockDim.x threads cooperate. */
__global__ void k_readout_fwd(const int *idx, const float *val,
                              const float *Wc, const float *Ww,
                              const float *bc, const float *bw,
                              const int *w2class, const int *w2slot, const int *y,
                              float *errc, float *errw, float *logp,
                              int B, int K, int C, int S, int Gcols) {
    int e = blockIdx.x;
    if (e >= B) return;
    int tid = threadIdx.x, nth = blockDim.x;
    int yy = y[e];
    int ce = w2class[yy];
    int slot = w2slot[yy];
    int base = ce * S;

    extern __shared__ float sh[];
    float *lc = sh;            /* C floats */
    float *lw = sh + C;        /* S floats */

    /* class logits */
    for (int c = tid; c < C; c += nth) {
        float acc = bc[c];
        for (int k = 0; k < K; k++) {
            int g = idx[e * K + k];
            float v = val[e * K + k];
            acc += v * Wc[(size_t)g * C + c];
        }
        lc[c] = acc;
    }
    /* word logits over target class */
    for (int j = tid; j < S; j += nth) {
        float acc = bw[base + j];
        for (int k = 0; k < K; k++) {
            int g = idx[e * K + k];
            float v = val[e * K + k];
            acc += v * Ww[(size_t)g * Gcols + base + j];
        }
        lw[j] = acc;
    }
    __syncthreads();

    /* softmax(lc) and softmax(lw) + logprob of true (ce, slot) — done by tid 0
     * for numerical simplicity (C,S are modest: 2048 / 25). */
    __shared__ float s_lse_c, s_lse_w;
    if (tid == 0) {
        float mc = lc[0]; for (int c = 1; c < C; c++) if (lc[c] > mc) mc = lc[c];
        float sc = 0.0f;  for (int c = 0; c < C; c++) sc += expf(lc[c] - mc);
        s_lse_c = mc + logf(sc + 1e-30f);
        float mw = lw[0]; for (int j = 1; j < S; j++) if (lw[j] > mw) mw = lw[j];
        float sw = 0.0f;  for (int j = 0; j < S; j++) sw += expf(lw[j] - mw);
        s_lse_w = mw + logf(sw + 1e-30f);
        logp[e] = (lc[ce] - s_lse_c) + (lw[slot] - s_lse_w);
    }
    __syncthreads();
    /* errc = softmax(lc) - onehot(ce) ; errw = softmax(lw) - onehot(slot) */
    for (int c = tid; c < C; c += nth) {
        float p = expf(lc[c] - s_lse_c);
        errc[(size_t)e * C + c] = p - (c == ce ? 1.0f : 0.0f);
    }
    for (int j = tid; j < S; j += nth) {
        float p = expf(lw[j] - s_lse_w);
        errw[(size_t)e * S + j] = p - (j == slot ? 1.0f : 0.0f);
    }
}

/* ── sampled-softmax word head (training only; eval stays full+exact) ──
 * cuts the word-head scatter from K*S to K*(1+neg): update the true word + `neg`
 * sampled negatives within the target class instead of all S words. */
__global__ void k_sample_negw(int *negw, int bc, int N, int S, unsigned long seed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bc * N) return;
    unsigned int x = (unsigned int)(seed + (unsigned long)t * 2654435761u);
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    negw[t] = (int)(x % (unsigned int)S);
}

/* class head FULL + word head over {true slot + N negatives}. Writes errc[bc,C]
 * (full), werr[bc,1+N], wslot[bc,1+N] (the within-class slots used), logp[bc]. */
__global__ void k_readout_fwd_negw(const int *idx, const float *val,
                                   const float *Wc, const float *Ww,
                                   const float *bc, const float *bw,
                                   const int *w2class, const int *w2within, const int *y,
                                   const int *negw, float *errc, float *werr, int *wslot,
                                   float *logp, int B, int K, int C, int S, int N, int Gcols) {
    int e = blockIdx.x;
    if (e >= B) return;
    int tid = threadIdx.x, nth = blockDim.x;
    int yy = y[e]; int ce = w2class[yy]; int within = w2within[yy]; int base = ce * S;
    int M = N + 1;
    extern __shared__ float sh[];
    float *lc = sh;            /* C */
    float *lw = sh + C;        /* M */
    int   *sl = (int *)(lw + M);  /* M slots */
    /* class logits (full) */
    for (int c = tid; c < C; c += nth) {
        float acc = bc[c];
        for (int k = 0; k < K; k++) acc += val[e * K + k] * Wc[(size_t)idx[e * K + k] * C + c];
        lc[c] = acc;
    }
    /* word slots: 0 = true, 1..N = negatives */
    if (tid == 0) sl[0] = within;
    for (int m = tid; m < N; m += nth) sl[m + 1] = negw[e * N + m];
    __syncthreads();
    for (int m = tid; m < M; m += nth) {
        int col = base + sl[m];
        float acc = bw[col];
        for (int k = 0; k < K; k++) acc += val[e * K + k] * Ww[(size_t)idx[e * K + k] * Gcols + col];
        lw[m] = acc;
        wslot[e * M + m] = sl[m];
    }
    __syncthreads();
    __shared__ float lse_c, lse_w;
    if (tid == 0) {
        float mc = lc[0]; for (int c = 1; c < C; c++) if (lc[c] > mc) mc = lc[c];
        float sc = 0.0f; for (int c = 0; c < C; c++) sc += expf(lc[c] - mc);
        lse_c = mc + logf(sc + 1e-30f);
        float mw = lw[0]; for (int m = 1; m < M; m++) if (lw[m] > mw) mw = lw[m];
        float sw = 0.0f; for (int m = 0; m < M; m++) sw += expf(lw[m] - mw);
        lse_w = mw + logf(sw + 1e-30f);
        logp[e] = (lc[ce] - lse_c) + (lw[0] - lse_w);   /* sampled-softmax logp */
    }
    __syncthreads();
    for (int c = tid; c < C; c += nth)
        errc[(size_t)e * C + c] = expf(lc[c] - lse_c) - (c == ce ? 1.0f : 0.0f);
    for (int m = tid; m < M; m += nth)
        werr[(size_t)e * M + m] = expf(lw[m] - lse_w) - (m == 0 ? 1.0f : 0.0f);
}

/* atomic scatter for the sampled word head: K*(1+N) writes/example (small) */
__global__ void k_scatter_word_negw(const int *idx, const float *val,
                                    const float *werr, const int *wslot,
                                    const int *w2class, const int *y, const float *weight,
                                    float *Ww, float *bw, float st, int B, int K, int M,
                                    int S, int Gcols) {
    int e = blockIdx.x;
    if (e >= B) return;
    int base = w2class[y[e]] * S;
    float we = st * weight[e];
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        int col = base + wslot[e * M + m];
        float dw = -we * werr[(size_t)e * M + m];
        for (int k = 0; k < K; k++)
            atomicAdd(&Ww[(size_t)idx[e * K + k] * Gcols + col], dw * val[e * K + k]);
        atomicAdd(&bw[col], dw);
    }
}

/* class-bias update only (word bias handled in k_scatter_word_negw) */
__global__ void k_bias_class(const float *errc, const float *weight, float *bc,
                             float st, int B, int C) {
    int e = blockIdx.x;
    if (e >= B) return;
    float we = st * weight[e];
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        atomicAdd(&bc[c], -we * errc[(size_t)e * C + c]);
}

/* Readout R2 scatter (M1 simple atomic version): for each example, for each
 * active granule, add -st*val*err into Wc and the target class's Ww columns,
 * and into the biases. Atomics absorb shared-granule collisions across the
 * chunk. (M2 replaces this with the atomic-free bucketed kernel.) */
__global__ void k_readout_scatter(const int *idx, const float *val,
                                  const float *errc, const float *errw,
                                  const int *w2class, const int *y, const float *weight,
                                  float *Wc, float *Ww,
                                  float st, int B, int K, int C, int S, int Gcols) {
    int e = blockIdx.x;
    if (e >= B) return;
    int tid = threadIdx.x, nth = blockDim.x;
    int ce = w2class[y[e]];
    int base = ce * S;
    float we = st * weight[e];
    /* class head (biases handled separately by k_bias_update) */
    for (int c = tid; c < C; c += nth) {
        float dc = -we * errc[(size_t)e * C + c];
        for (int k = 0; k < K; k++)
            atomicAdd(&Wc[(size_t)idx[e * K + k] * C + c], dc * val[e * K + k]);
    }
    /* word head (target class only) */
    for (int j = tid; j < S; j += nth) {
        float dw = -we * errw[(size_t)e * S + j];
        for (int k = 0; k < K; k++)
            atomicAdd(&Ww[(size_t)idx[e * K + k] * Gcols + base + j], dw * val[e * K + k]);
    }
}

/* Eval-only true-word logprob per layer (no scatter): same as R1 forward but
 * writes only logp[e]. Reused by the mixture eval. */
__global__ void k_readout_logp(const int *idx, const float *val,
                               const float *Wc, const float *Ww,
                               const float *bc, const float *bw,
                               const int *w2class, const int *w2slot, const int *y,
                               float *logp, int B, int K, int C, int S, int Gcols) {
    int e = blockIdx.x;
    if (e >= B) return;
    int tid = threadIdx.x, nth = blockDim.x;
    int yy = y[e]; int ce = w2class[yy]; int slot = w2slot[yy]; int base = ce * S;
    extern __shared__ float sh[];
    float *lc = sh; float *lw = sh + C;
    for (int c = tid; c < C; c += nth) {
        float acc = bc[c];
        for (int k = 0; k < K; k++) acc += val[e*K+k] * Wc[(size_t)idx[e*K+k]*C + c];
        lc[c] = acc;
    }
    for (int j = tid; j < S; j += nth) {
        float acc = bw[base + j];
        for (int k = 0; k < K; k++) acc += val[e*K+k] * Ww[(size_t)idx[e*K+k]*Gcols + base + j];
        lw[j] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float mc = lc[0]; for (int c = 1; c < C; c++) if (lc[c] > mc) mc = lc[c];
        float sc = 0.0f;  for (int c = 0; c < C; c++) sc += expf(lc[c] - mc);
        float lse_c = mc + logf(sc + 1e-30f);
        float mw = lw[0]; for (int j = 1; j < S; j++) if (lw[j] > mw) mw = lw[j];
        float sw = 0.0f;  for (int j = 0; j < S; j++) sw += expf(lw[j] - mw);
        float lse_w = mw + logf(sw + 1e-30f);
        logp[e] = (lc[ce] - lse_c) + (lw[slot] - lse_w);
    }
}

/* ── M2: resident-shard GPU window gather (F5 — no per-step H2D) ──
 * Given the whole training shard resident on-device (uint16) and B random start
 * positions, fill dX[B,ctx] (int) and dY[B] (int). One thread per (e,p). */
__global__ void k_gather_windows(const unsigned short *toks, const long *pos,
                                 int *X, int *Y, int B, int ctx) {
    int e = blockIdx.x;
    if (e >= B) return;
    long start = pos[e];
    for (int p = threadIdx.x; p < ctx; p += blockDim.x)
        X[(size_t)e * ctx + p] = (int)toks[start + p];
    if (threadIdx.x == 0) Y[e] = (int)toks[start + ctx];
}

/* ── analytics: granule usage + margin accumulation over a sample ── */
__global__ void k_usage_accum(const int *idx, const float *val, unsigned int *usage,
                              double *margin_sum, int B, int K) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * K) return;
    int g = idx[t];
    atomicAdd(&usage[g], 1u);
    /* margin_sum is a single double; accumulate with atomic add on double (sm_60+) */
    atomicAdd(margin_sum, (double)val[t]);
}

/* load-balancing: fold this chunk's per-granule selection counts (already computed
 * by the bucketed scatter as gcount) into a persistent usage accumulator. Cheap:
 * G adds, no per-pair atomics. */
__global__ void k_accum_counts(unsigned long long *usage_acc, const unsigned int *gcount, int G) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < G) usage_acc[g] += (unsigned long long)gcount[g];
}

/* homeostatic load-balancing (DeepSeek aux-loss-free controller, cast as cerebellar
 * intrinsic-excitability homeostasis): nudge each granule's SELECTION bias toward an
 * even fire-rate. gbias is SUBTRACTED in the activation, so an over-firing granule
 * gets a HIGHER bias (suppressed) and a starved one a LOWER bias (boosted). Pure
 * negative-feedback control, no gradient. Clamps for safety; resets the accumulator. */
__global__ void k_balance_apply(float *gbias, unsigned long long *usage_acc, int G,
                                float target, float gamma, float inv_n, float clampv) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    float fr = (float)usage_acc[g] * inv_n;        /* measured per-example fire rate */
    /* scale-free RELATIVE error so a fixed gain works regardless of K/G: +inf-capped
     * above (a hot granule pushes hard), -1 below (a starved one boosts by <= gamma). */
    float e = fr / target - 1.0f;
    e = e > 4.0f ? 4.0f : (e < -1.0f ? -1.0f : e);
    float nb = gbias[g] + gamma * e;               /* gbias subtracted in activation: hot -> up -> suppressed */
    nb = nb > clampv ? clampv : (nb < -clampv ? -clampv : nb);
    gbias[g] = nb;
    usage_acc[g] = 0ULL;
}

/* ── M2: atomic-free bucketed readout scatter ──
 * Stage 1 counts per-granule bucket sizes; host prefix-sums to offsets; stage 2
 * fills (example,kslot) pairs; stage 3 has ONE block per granule write its row
 * (no cross-block conflict on Ww -> no atomics on the dominant payload). */
__global__ void k_bucket_count(const int *idx, unsigned int *gcount, int B, int K) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * K) return;
    atomicAdd(&gcount[idx[t]], 1u);
}
__global__ void k_bucket_fill(const int *idx, const int *off, unsigned int *cursor,
                             int *bk_e, int *bk_k, int B, int K) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * K) return;
    int e = t / K, k = t % K;
    int g = idx[t];
    unsigned int p = atomicAdd(&cursor[g], 1u);
    int slot = off[g] + (int)p;
    bk_e[slot] = e; bk_k[slot] = k;
}
/* class head: one block per granule, threads over C; no atomics (block owns row g) */
__global__ void k_scatter_class_bucketed(const int *off, const int *bk_e, const int *bk_k,
                                         const float *val, const float *errc, const float *weight,
                                         float *Wc, float st, int G, int C, int K) {
    int g = blockIdx.x;
    if (g >= G) return;
    int lo = off[g], hi = off[g + 1];
    if (lo == hi) return;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (int p = lo; p < hi; p++) {
            int e = bk_e[p], k = bk_k[p];
            acc += weight[e] * val[e * K + k] * errc[(size_t)e * C + c];
        }
        Wc[(size_t)g * C + c] -= st * acc;
    }
}
/* word head: one block per granule; for each pair, add into that example's class
 * column block. No atomics (block owns row g). */
__global__ void k_scatter_word_bucketed(const int *off, const int *bk_e, const int *bk_k,
                                        const float *val, const float *errw,
                                        const int *w2class, const int *y, const float *weight,
                                        float *Ww, float st, int G, int S, int K, int Gcols) {
    int g = blockIdx.x;
    if (g >= G) return;
    int lo = off[g], hi = off[g + 1];
    if (lo == hi) return;
    for (int p = lo; p < hi; p++) {
        int e = bk_e[p], k = bk_k[p];
        int base = w2class[y[e]] * S;
        float v = val[e * K + k] * weight[e];
        const float *er = errw + (size_t)e * S;
        float *row = Ww + (size_t)g * Gcols + base;
        for (int j = threadIdx.x; j < S; j += blockDim.x)
            row[j] -= st * v * er[j];
    }
}
/* shared bias update (both scatter modes): atomics on the tiny bc/bw arrays */
__global__ void k_bias_update(const float *errc, const float *errw,
                              const int *w2class, const int *y, const float *weight,
                              float *bc, float *bw, float st, int B, int C, int S) {
    int e = blockIdx.x;
    if (e >= B) return;
    int base = w2class[y[e]] * S;
    float we = st * weight[e];
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        atomicAdd(&bc[c], -we * errc[(size_t)e * C + c]);
    for (int j = threadIdx.x; j < S; j += blockDim.x)
        atomicAdd(&bw[base + j], -we * errw[(size_t)e * S + j]);
}

/* ── M4: competitive granule learning (online k-means under WTA) ──
 * accumulate per-granule sums of the activating inputs (over topk winners),
 * count wins, then move gwt toward the mean and renormalize rows. */
__global__ void k_comp_accum(const int *idx, const int *X_unused,
                            const float *x, const int *gidx,
                            float *sums, unsigned int *counts,
                            int B, int K, int fan, int in_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   /* one thread per (e,k) */
    if (t >= B * K) return;
    int e = t / K;
    int g = idx[t];
    atomicAdd(&counts[g], 1u);
    const float *xr = x + (size_t)e * in_dim;
    const int *gi = gidx + (size_t)g * fan;
    float *sr = sums + (size_t)g * fan;
    for (int f = 0; f < fan; f++) atomicAdd(&sr[f], xr[gi[f]]);
}
__global__ void k_comp_apply(float *gwt, const float *sums, const unsigned int *counts,
                            float eta, int G, int fan) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    unsigned int cnt = counts[g];
    if (cnt == 0) return;
    float inv = 1.0f / (float)cnt;
    float n2 = 0.0f;
    for (int f = 0; f < fan; f++) {
        float mean = sums[(size_t)g * fan + f] * inv;
        float w = gwt[(size_t)g * fan + f];
        w += eta * (mean - w);
        gwt[(size_t)g * fan + f] = w;
        n2 += w * w;
    }
    float scale = sqrtf((float)fan) / (sqrtf(n2) + 1e-6f);
    for (int f = 0; f < fan; f++) gwt[(size_t)g * fan + f] *= scale;
}

/* subtract per-granule homeostatic bias after the activation matmul (fp16 act) */
__global__ void k_sub_gbias(__half *a, const float *gbias, int B, int G) {
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= (size_t)B * G) return;
    a[t] = __float2half(__half2float(a[t]) - gbias[t % G]);
}

/* fp32 -> fp16 conversion (for fp16 tensor-core activation: the matmul only feeds
 * an argmax, so half precision on inputs is harmless and ~8x faster on T4). */
__global__ void k_f32_to_f16(const float *in, __half *out, size_t n) {
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t < n) out[t] = __float2half(in[t]);
}

/* shift granule indices by a window offset (granule subsampling) */
__global__ void k_add_offset(int *idx, int off, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) idx[t] += off;
}

/* device-side exclusive prefix sum of gcount[G] -> goff[G+1] (single thread;
 * G is small and this is off the bandwidth-bound path, but it removes the host
 * round-trip + sync that made the bucketed scatter slow). */
__global__ void k_prefix_sum(const unsigned int *gcount, int *goff, int G) {
    if (threadIdx.x || blockIdx.x) return;
    int acc = 0;
    for (int g = 0; g < G; g++) { goff[g] = acc; acc += (int)gcount[g]; }
    goff[G] = acc;
}

/* device mixture EM update: P(y)=sum_l w_l P_l(y); update mix logits by
 * (responsibility - prior). One block, threads split over examples (parallel),
 * responsibilities reduced into shared memory. Off-host => no per-step sync. */
__global__ void k_mixture_update(const float *logp, float *mix, float mix_lr, int L, int bc) {
    __shared__ float w[8];
    __shared__ float rs[8];
    int tid = threadIdx.x, nth = blockDim.x;
    if (tid == 0) {
        float mx = -1e30f;
        for (int l = 0; l < L; l++) if (mix[l] > mx) mx = mix[l];
        float zs = 0.0f;
        for (int l = 0; l < L; l++) { w[l] = expf(mix[l] - mx); zs += w[l]; }
        for (int l = 0; l < L; l++) { w[l] /= zs; rs[l] = 0.0f; }
    }
    __syncthreads();
    for (int e = tid; e < bc; e += nth) {
        float lw[8], m = -1e30f;
        for (int l = 0; l < L; l++) { lw[l] = logf(w[l] + 1e-30f) + logp[(size_t)l * bc + e]; if (lw[l] > m) m = lw[l]; }
        float z = 0.0f;
        for (int l = 0; l < L; l++) z += expf(lw[l] - m);
        for (int l = 0; l < L; l++) atomicAdd(&rs[l], expf(lw[l] - m) / z);
    }
    __syncthreads();
    if (tid == 0)
        for (int l = 0; l < L; l++) mix[l] += mix_lr * (rs[l] / (float)bc - w[l]);
}

/* boosting: per-example update weight. Layer 0 uses 1; deeper layers downweight
 * examples the layer below already predicts well (weight = 1 - p_prev, floored),
 * so each layer specializes on the residual -> depth becomes additive. */
__global__ void k_fill_ones(float *w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = 1.0f;
}
__global__ void k_boost_weight(const float *logp_prev, float *weight, int bc, float wmin) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= bc) return;
    float p = expf(logp_prev[e]);          /* prob the previous layer gave the true word */
    float w = 1.0f - p;
    weight[e] = w < wmin ? wmin : w;
}

/* generation (single context): per-layer class probabilities over all C classes */
__global__ void k_gen_class(const int *idx, const float *val, const float *Wc,
                            const float *bc, float *pc, int C, int K) {
    int tid = threadIdx.x, nth = blockDim.x;
    extern __shared__ float sh[];            /* C logits */
    for (int c = tid; c < C; c += nth) {
        float acc = bc[c];
        for (int k = 0; k < K; k++) acc += val[k] * Wc[(size_t)idx[k] * C + c];
        sh[c] = acc;
    }
    __syncthreads();
    __shared__ float ssum;
    if (tid == 0) {
        float m = sh[0]; for (int c = 1; c < C; c++) if (sh[c] > m) m = sh[c];
        float s = 0.0f; for (int c = 0; c < C; c++) { sh[c] = expf(sh[c] - m); s += sh[c]; }
        ssum = s;
    }
    __syncthreads();
    for (int c = tid; c < C; c += nth) pc[c] = sh[c] / ssum;
}

/* generation: per-layer within-class word probabilities for one class c */
__global__ void k_gen_within(const int *idx, const float *val, const float *Ww,
                             const float *bw, int c, float *pw, int S, int K, int Gcols) {
    int tid = threadIdx.x, nth = blockDim.x;
    int base = c * S;
    extern __shared__ float sh[];            /* S logits */
    for (int j = tid; j < S; j += nth) {
        float acc = bw[base + j];
        for (int k = 0; k < K; k++) acc += val[k] * Ww[(size_t)idx[k] * Gcols + base + j];
        sh[j] = acc;
    }
    __syncthreads();
    __shared__ float ssum;
    if (tid == 0) {
        float m = sh[0]; for (int j = 1; j < S; j++) if (sh[j] > m) m = sh[j];
        float s = 0.0f; for (int j = 0; j < S; j++) { sh[j] = expf(sh[j] - m); s += sh[j]; }
        ssum = s;
    }
    __syncthreads();
    for (int j = tid; j < S; j += nth) pw[j] = sh[j] / ssum;
}

/* ─────────────────────────── host: layer + brain ──────────────────────────── */

struct Layer {
    int in_dim;
    int   *gidx;      /* [G, fan_in] */
    float *gwt;       /* [G, fan_in] */
    float *gbias;     /* [G] per-granule selection bias (homeostatic load-balancing) */
    unsigned long long *usage_acc; /* [G] running selection counts for the balance controller */
    float *projT;     /* [G, in_dim] fp32 master (rebuilt by competitive learning) */
    __half *projT_h;  /* [G, in_dim] fp16 mirror for tensor-core activation */
    float *Wc;        /* [G, C] */
    float *Ww;        /* [G, C*S] class-major */
    float *bc;        /* [C] */
    float *bw;        /* [C*S] class-major */
    float *relayR;    /* [G, relay_dim] */
};

struct Brain {
    Cfg cfg;
    int V, D, ctx, G, K, C, S, Gcols, relay_dim, L;
    /* shared fixed front-end */
    float *codes;     /* [V, D] */
    float *pos;       /* [ctx, D] */
    float *decay;     /* [n_scales, ctx] */
    int   *w2class;   /* [V] */
    int   *w2slot;    /* [V] */
    int   *class_word;/* [C*S] (class,slot)->word id (V = padding); for generation */
    std::vector<Layer> layers;
    float *mix;       /* host [L] mixture logits (synced from dmix for eval/JSON) */
    float *dmix;      /* device [L] mixture logits (updated on-device each step) */
    cublasHandle_t cub;
    int   timing_on;  /* 1 only during brain_profile; 0 in the hot loop (no syncs) */
    float lr_scale;   /* per-step lr multiplier (cosine anneal); 1.0 = constant lr */
    unsigned long negctr;  /* varies the negative-sampling seed per call */
    unsigned long subctr;  /* rotates the granule-subsample window offset */
    /* reusable device transients (sized for step_chunk) */
    int   *dX;        /* [chunk, ctx] */
    int   *dY;        /* [chunk] */
    float *dbound;    /* [chunk, D] (single summed bind) */
    float *dboundL;   /* [L, chunk, D] per-layer timescale binds (multiscale) */
    float *dx[8];     /* per-layer input buffers (layer0 aliases dbound) */
    __half *dact;     /* [chunk, G] fp16 activation (feeds argmax; halves topk read) */
    __half *dxh;      /* [chunk, relay_dim+D] fp16 activation-GEMM input */
    int   *didx[8];   /* [chunk, K] per layer */
    float *dval[8];   /* [chunk, K] per layer */
    float *derrc;     /* [chunk, C] */
    float *derrw;     /* [chunk, S] */
    float *dlogp;     /* [L, chunk] */
    float *dweight;   /* [chunk] per-example boost weight (all 1 when boost off) */
    int   *dnegw;     /* [chunk, neg_word] sampled negative slots (sampled softmax) */
    float *dwerr;     /* [chunk, 1+neg_word] sampled word errors */
    int   *dwslot;    /* [chunk, 1+neg_word] sampled word slots */
    /* M2: resident shard + GPU window gather */
    unsigned short *dtoks;  /* whole train shard on device (uint16), or NULL */
    long  dtoks_n;
    const unsigned short *htoks;  /* host shard pointer (fallback gather) */
    long  htoks_n;
    long  *dpos;            /* [chunk] random start positions */
    /* M2: bucketed-scatter scratch (sized [chunk*K] and [G+1]) */
    unsigned int *gcount;   /* [G] */
    int   *goff;            /* [G+1] (host-prefix-summed) */
    unsigned int *gcursor;  /* [G] */
    int   *bk_e;            /* [chunk*K] */
    int   *bk_k;            /* [chunk*K] */
    /* M4: competitive-learning scratch */
    float *csum;            /* [G*fan_in] */
    unsigned int *ccount;   /* [G] */
    /* analytics: per-phase timings (seconds) and granule health */
    double t_bind, t_act, t_topk, t_relay, t_rfwd, t_scatter, t_bias, t_mix;
    long   n_steps_timed;
};

/* analytics timing helper: sync + accumulate elapsed into *acc when on */
static inline void phase_tic(double *t0) { CUDA_CHECK(cudaDeviceSynchronize()); *t0 = now_sec(); }
static inline void phase_toc(int on, double *acc, double t0) {
    if (!on) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    *acc += now_sec() - t0;
}

static float frand(rng_t *r) { return (float)rng_gauss(r); }

/* build the frequency-rank class hierarchy (host), matching v2 ClassHierarchy */
static void build_hierarchy(Brain *b, const long *freq) {
    int V = b->V, C = b->C, S = b->S;
    /* sort word ids by descending frequency */
    std::vector<int> order(V);
    for (int w = 0; w < V; w++) order[w] = w;
    std::sort(order.begin(), order.end(),
              [&](int a, int c) { return freq[a] > freq[c]; });
    std::vector<int> w2class(V), w2slot(V);
    for (int rank = 0; rank < V; rank++) {
        int w = order[rank];
        w2class[w] = rank / S;
        w2slot[w]  = rank % S;
    }
    h2d(b->w2class, w2class.data(), V);
    h2d(b->w2slot,  w2slot.data(),  V);
    /* (class,slot)->word: flat rank index r maps to class r/S slot r%S, so the
     * flat (c*S+slot) cell is exactly rank r -> order[r]. Pad tail with V. */
    std::vector<int> cw((size_t)C * S, V);
    for (int r = 0; r < V; r++) cw[r] = order[r];
    h2d(b->class_word, cw.data(), (size_t)C * S);
    /* per-layer unigram bias init (class-major) */
    std::vector<float> bw(C * S, -30.0f), bc(C, 0.0f);
    std::vector<double> cf(C, 0.0);
    for (int w = 0; w < V; w++) {
        int cc = w2class[w], sl = w2slot[w];
        bw[cc * S + sl] = logf((float)freq[w] + 0.5f);
        cf[cc] += (double)freq[w];
    }
    for (int c = 0; c < C; c++) bc[c] = logf((float)cf[c] + 0.5f);
    for (int l = 0; l < b->L; l++) {
        h2d(b->layers[l].bw, bw.data(), (size_t)C * S);
        h2d(b->layers[l].bc, bc.data(), C);
    }
}

static void layer_alloc(Brain *b, Layer *ly, int depth, rng_t *r) {
    Cfg &c = b->cfg;
    ly->in_dim = (depth == 0) ? b->D : (b->relay_dim + b->D);
    int G = b->G, fan = c.fan_in;
    /* fixed wiring + (random for M1) granule weights */
    std::vector<int> gidx((size_t)G * fan);
    std::vector<float> gwt((size_t)G * fan);
    for (size_t i = 0; i < (size_t)G * fan; i++) {
        gidx[i] = rng_int(r, ly->in_dim);
        gwt[i]  = frand(r);
    }
    /* unit-norm rows scaled by sqrt(fan) */
    for (int g = 0; g < G; g++) {
        double n2 = 0.0;
        for (int f = 0; f < fan; f++) { float w = gwt[(size_t)g*fan+f]; n2 += (double)w*w; }
        float inv = (float)(sqrt((double)fan) / (sqrt(n2) + 1e-6));
        for (int f = 0; f < fan; f++) gwt[(size_t)g*fan+f] *= inv;
    }
    ly->gidx = dev_alloc<int>((size_t)G * fan);   h2d(ly->gidx, gidx.data(), (size_t)G*fan);
    ly->gwt  = dev_alloc<float>((size_t)G * fan); h2d(ly->gwt, gwt.data(), (size_t)G*fan);
    ly->gbias = dev_zalloc<float>(G);
    ly->usage_acc = dev_zalloc<unsigned long long>(G);
    ly->projT = dev_zalloc<float>((size_t)G * ly->in_dim);
    ly->projT_h = dev_alloc<__half>((size_t)G * ly->in_dim);
    ly->Wc = dev_zalloc<float>((size_t)G * b->C);
    ly->Ww = dev_zalloc<float>((size_t)G * b->Gcols);
    ly->bc = dev_alloc<float>(b->C);
    ly->bw = dev_alloc<float>((size_t)b->Gcols);
    /* relay projection */
    std::vector<float> rR((size_t)G * b->relay_dim);
    float rs = 1.0f / sqrtf((float)b->relay_dim);
    for (size_t i = 0; i < rR.size(); i++) rR[i] = frand(r) * rs;
    ly->relayR = dev_alloc<float>((size_t)G * b->relay_dim);
    h2d(ly->relayR, rR.data(), (size_t)G * b->relay_dim);
    /* build dense projT + its fp16 mirror */
    int nt = G * fan;
    k_build_projT<<<(nt + 255) / 256, 256>>>(ly->gidx, ly->gwt, ly->projT, G, fan, ly->in_dim);
    CUDA_KERNEL_CHECK();
    size_t pn = (size_t)G * ly->in_dim;
    k_f32_to_f16<<<(pn + 255) / 256, 256>>>(ly->projT, ly->projT_h, pn);
    CUDA_KERNEL_CHECK();
}

static void brain_init(Brain *b, const Cfg &cfg, int V, const long *freq) {
    b->cfg = cfg;
    b->V = V; b->D = cfg.code_dim; b->ctx = cfg.ctx; b->G = cfg.n_gran;
    b->K = cfg.k_active; b->C = cfg.n_classes; b->relay_dim = cfg.relay_dim;
    b->L = cfg.n_layers;
    b->S = (V + b->C - 1) / b->C;
    b->Gcols = b->C * b->S;
    rng_t r; rng_seed(&r, cfg.seed);

    /* fixed token codes [V,D] */
    std::vector<float> codes((size_t)V * b->D);
    for (size_t i = 0; i < codes.size(); i++) codes[i] = frand(&r);
    b->codes = dev_alloc<float>((size_t)V * b->D);
    h2d(b->codes, codes.data(), (size_t)V * b->D);
    /* position sign codes [ctx,D] */
    std::vector<float> pos((size_t)b->ctx * b->D);
    for (size_t i = 0; i < pos.size(); i++) pos[i] = (rng_u64(&r) & 1) ? 1.0f : -1.0f;
    b->pos = dev_alloc<float>(pos.size()); h2d(b->pos, pos.data(), pos.size());
    /* decay [n_scales, ctx] */
    float rates[3] = {cfg.decay0, cfg.decay1, cfg.decay2};
    std::vector<float> decay((size_t)cfg.n_scales * b->ctx);
    for (int s = 0; s < cfg.n_scales; s++)
        for (int p = 0; p < b->ctx; p++) {
            int dist = b->ctx - 1 - p;             /* most recent token => smallest distance */
            decay[s * b->ctx + p] = expf(-(float)dist * rates[s]);
        }
    b->decay = dev_alloc<float>(decay.size()); h2d(b->decay, decay.data(), decay.size());

    b->w2class = dev_alloc<int>(V);
    b->w2slot  = dev_alloc<int>(V);
    b->class_word = dev_alloc<int>((size_t)b->C * b->S);

    b->layers.resize(b->L);
    for (int l = 0; l < b->L; l++) layer_alloc(b, &b->layers[l], l, &r);
    build_hierarchy(b, freq);

    b->mix = (float *)calloc(b->L, sizeof(float));
    b->dmix = dev_zalloc<float>(b->L);
    CUBLAS_CHECK(cublasCreate(&b->cub));
    b->timing_on = 0;
    b->lr_scale = 1.0f;
    b->negctr = 0;
    b->subctr = 0;

    /* transients sized for step_chunk */
    int chunk = cfg.step_chunk;
    b->dX = dev_alloc<int>((size_t)chunk * b->ctx);
    b->dY = dev_alloc<int>(chunk);
    b->dbound = dev_alloc<float>((size_t)chunk * b->D);
    b->dboundL = cfg.multiscale ? dev_alloc<float>((size_t)b->L * chunk * b->D) : NULL;
    b->dact = dev_alloc<__half>((size_t)chunk * b->G);
    b->dxh = dev_alloc<__half>((size_t)chunk * (b->relay_dim + b->D));  /* fp16 GEMM input */
    b->derrc = dev_alloc<float>((size_t)chunk * b->C);
    b->derrw = dev_alloc<float>((size_t)chunk * b->S);
    b->dlogp = dev_alloc<float>((size_t)b->L * chunk);
    b->dweight = dev_alloc<float>(chunk);
    for (int l = 0; l < b->L; l++) {
        b->didx[l] = dev_alloc<int>((size_t)chunk * b->K);
        b->dval[l] = dev_alloc<float>((size_t)chunk * b->K);
        b->dx[l] = (l == 0) ? (cfg.multiscale ? b->dboundL : b->dbound)
                            : dev_alloc<float>((size_t)chunk * (b->relay_dim + b->D));
    }
    /* M2/M4 scratch */
    b->dtoks = NULL; b->dtoks_n = 0;
    b->dpos = dev_alloc<long>(chunk);
    b->gcount = dev_alloc<unsigned int>(b->G);
    b->goff = dev_alloc<int>(b->G + 1);
    b->gcursor = dev_alloc<unsigned int>(b->G);
    b->bk_e = dev_alloc<int>((size_t)chunk * b->K);
    b->bk_k = dev_alloc<int>((size_t)chunk * b->K);
    b->csum = dev_alloc<float>((size_t)b->G * cfg.fan_in);
    b->ccount = dev_alloc<unsigned int>(b->G);
    if (cfg.neg_word > 0) {
        b->dnegw = dev_alloc<int>((size_t)chunk * cfg.neg_word);
        b->dwerr = dev_alloc<float>((size_t)chunk * (cfg.neg_word + 1));
        b->dwslot = dev_alloc<int>((size_t)chunk * (cfg.neg_word + 1));
    } else { b->dnegw = NULL; b->dwerr = NULL; b->dwslot = NULL; }
    b->t_bind = b->t_act = b->t_topk = b->t_relay = 0.0;
    b->t_rfwd = b->t_scatter = b->t_bias = b->t_mix = 0.0;
    b->n_steps_timed = 0;
}

/* encode a chunk through the stack: fills didx[l], dval[l] and per-layer inputs.
 * Per-phase timing fires ONLY when b->timing_on (set during brain_profile), so
 * the training hot loop has zero added syncs. */
static void encode_stack(Brain *b, int bc) {
    int D = b->D, G = b->G, K = b->K;
    int tm = b->timing_on;
    double t0;
    const float one = 1.0f, zero = 0.0f;
    /* bind: one summed multi-scale vector, OR per-layer single-rate binds so
     * deeper layers see longer-range context (multiscale). */
    int chunk = b->cfg.step_chunk;
    if (tm) phase_tic(&t0);
    dim3 bb(128); dim3 bg(bc, (D + 127) / 128);
    if (b->cfg.multiscale) {
        for (int l = 0; l < b->L; l++) {
            float rate = b->cfg.decay0 / (float)(1 << l);   /* deeper = longer range */
            k_bind_rate<<<bg, bb>>>(b->dX, b->codes, b->pos, rate,
                                    b->dboundL + (size_t)l * chunk * D, bc, b->ctx, D);
        }
    } else {
        k_bind<<<bg, bb>>>(b->dX, b->codes, b->pos, b->decay, b->dbound,
                           bc, b->ctx, D, b->cfg.n_scales);
    }
    CUDA_KERNEL_CHECK();
    phase_toc(tm, &b->t_bind, t0);
    for (int l = 0; l < b->L; l++) {
        Layer &ly = b->layers[l];
        float *xin = b->dx[l];        /* dx[0] aliases the layer-0 bind (see brain_init) */
        int in_dim = ly.in_dim;
        /* granule subsampling: activate + top-k only a rotating window of Geff
         * granules (offset off), then shift the winners back to global ids. */
        int Geff = G, off = 0;
        if (b->cfg.gran_frac < 1.0f) {
            Geff = (int)(G * b->cfg.gran_frac);
            Geff -= Geff % b->cfg.topk_groups;
            if (Geff < b->cfg.topk_groups) Geff = b->cfg.topk_groups;
            int range = G - Geff;
            unsigned long h = (b->subctr++) * 2654435761u; h ^= h >> 13;
            off = range > 0 ? (int)(h % (unsigned long)(range + 1)) : 0;
        }
        /* activation a[B,Geff] = x[B,in] @ projT[off:off+Geff,in]^T - gbias */
        if (tm) phase_tic(&t0);
        if (b->cfg.use_cublas) {
            size_t xn = (size_t)bc * in_dim;
            k_f32_to_f16<<<(xn + 255) / 256, 256>>>(xin, b->dxh, xn);
            CUDA_KERNEL_CHECK();
            CUBLAS_CHECK(cublasGemmEx(b->cub, CUBLAS_OP_T, CUBLAS_OP_N, Geff, bc, in_dim,
                                      &one, ly.projT_h + (size_t)off * in_dim, CUDA_R_16F, in_dim,
                                      b->dxh, CUDA_R_16F, in_dim,
                                      &zero, b->dact, CUDA_R_16F, Geff,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            size_t n = (size_t)bc * Geff;
            k_sub_gbias<<<(n + 255) / 256, 256>>>(b->dact, ly.gbias + off, bc, Geff);
            CUDA_KERNEL_CHECK();
        } else {
            dim3 ab(256); dim3 ag((Geff + 255) / 256, bc);
            k_activate<<<ag, ab>>>(xin, ly.projT + (size_t)off * in_dim, ly.gbias + off,
                                   b->dact, bc, Geff, in_dim);
            CUDA_KERNEL_CHECK();
        }
        phase_toc(tm, &b->t_act, t0);
        /* topk: bitonic-sorted candidates by default (accurate + parallel);
         * --fast-topk = single-pass group-max (faster, cruder); --exact-topk =
         * slow reference; approx (K-pass) used only if candidate count isn't 2^n. */
        if (tm) phase_tic(&t0);
        int nth = 256;
        int groups = b->cfg.topk_groups, pg = b->cfg.topk_per_group;
        int M = groups * pg;
        if (b->cfg.exact_topk) {
            size_t sh = nth * (sizeof(float) + sizeof(int));
            k_topk<<<bc, nth, sh>>>(b->dact, b->didx[l], b->dval[l], bc, Geff, K);
        } else if (b->cfg.fast_topk && Geff % K == 0 && (K & (K - 1)) == 0) {
            k_topk_groupmax<<<bc, K, K * sizeof(float)>>>(b->dact, b->didx[l], b->dval[l], bc, Geff, K);
        } else if ((M & (M - 1)) == 0 && M >= K + 1) {
            size_t sh = (size_t)M * (sizeof(float) + sizeof(int));
            k_topk_bitonic<<<bc, nth, sh>>>(b->dact, b->didx[l], b->dval[l], bc, Geff, K, groups, pg);
        } else {
            size_t sh = ((size_t)M + nth) * (sizeof(float) + sizeof(int));
            k_topk_approx<<<bc, nth, sh>>>(b->dact, b->didx[l], b->dval[l], bc, Geff, K, groups, pg);
        }
        CUDA_KERNEL_CHECK();
        if (off > 0) {            /* shift window-local winners to global granule ids */
            int np = bc * K;
            k_add_offset<<<(np + 255) / 256, 256>>>(b->didx[l], off, np);
            CUDA_KERNEL_CHECK();
        }
        phase_toc(tm, &b->t_topk, t0);
        /* relay -> next input */
        if (l < b->L - 1) {
            if (tm) phase_tic(&t0);
            int rn = 128;
            size_t rsh = b->relay_dim * sizeof(float);
            /* next layer's input = [relay ; next layer's context bind] */
            float *ctxb = b->cfg.multiscale ? (b->dboundL + (size_t)(l + 1) * chunk * D)
                                            : b->dbound;
            k_relay_concat<<<bc, rn, rsh>>>(b->didx[l], b->dval[l], ly.relayR,
                                            ctxb, b->dx[l + 1],
                                            bc, K, b->relay_dim, D);
            CUDA_KERNEL_CHECK();
            phase_toc(tm, &b->t_relay, t0);
        }
    }
}

/* fill dX[bc,ctx], dY[bc] for a chunk from start positions cp (host) — resident
 * GPU gather when the shard is on-device, else host gather + H2D. */
static void fill_windows(Brain *b, const long *cp, int bc) {
    int ctx = b->ctx;
    if (b->dtoks && b->cfg.resident) {
        h2d(b->dpos, cp, bc);
        k_gather_windows<<<bc, 64>>>(b->dtoks, b->dpos, b->dX, b->dY, bc, ctx);
        CUDA_KERNEL_CHECK();
    } else {
        static std::vector<int> hX, hY;
        hX.resize((size_t)bc * ctx); hY.resize(bc);
        for (int e = 0; e < bc; e++) {
            long s = cp[e];
            for (int p = 0; p < ctx; p++) hX[(size_t)e * ctx + p] = (int)b->htoks[s + p];
            hY[e] = (int)b->htoks[s + ctx];
        }
        h2d(b->dX, hX.data(), (size_t)bc * ctx);
        h2d(b->dY, hY.data(), bc);
    }
}

/* run the readout scatter for one layer using the selected mode (atomic / bucketed) */
static void readout_scatter(Brain *b, Layer &ly, int l, int bc, float st) {
    int C = b->C, S = b->S, K = b->K, G = b->G;
    int tm = b->timing_on;
    double t0;
    if (tm) phase_tic(&t0);
    if (b->cfg.fast_scatter) {
        /* build per-granule buckets (counting sort + DEVICE prefix sum: no host
         * round-trip, no stall) */
        CUDA_CHECK(cudaMemset(b->gcount, 0, (size_t)G * sizeof(unsigned int)));
        int npairs = bc * K;
        k_bucket_count<<<(npairs + 255) / 256, 256>>>(b->didx[l], b->gcount, bc, K);
        CUDA_KERNEL_CHECK();
        if (b->cfg.balance)
            k_accum_counts<<<(G + 255) / 256, 256>>>(ly.usage_acc, b->gcount, G);
        k_prefix_sum<<<1, 1>>>(b->gcount, b->goff, G);
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaMemsetAsync(b->gcursor, 0, (size_t)G * sizeof(unsigned int)));
        k_bucket_fill<<<(npairs + 255) / 256, 256>>>(b->didx[l], b->goff, b->gcursor,
                                                     b->bk_e, b->bk_k, bc, K);
        CUDA_KERNEL_CHECK();
        k_scatter_class_bucketed<<<G, 256>>>(b->goff, b->bk_e, b->bk_k, b->dval[l],
                                             b->derrc, b->dweight, ly.Wc, st, G, C, K);
        CUDA_KERNEL_CHECK();
        k_scatter_word_bucketed<<<G, 128>>>(b->goff, b->bk_e, b->bk_k, b->dval[l],
                                            b->derrw, b->w2class, b->dY, b->dweight, ly.Ww, st,
                                            G, S, K, b->Gcols);
        CUDA_KERNEL_CHECK();
    } else {
        /* atomic path doesn't build gcount; do it explicitly so load-balancing works
         * regardless of scatter mode (B1). */
        if (b->cfg.balance) {
            CUDA_CHECK(cudaMemset(b->gcount, 0, (size_t)G * sizeof(unsigned int)));
            k_bucket_count<<<(bc * K + 255) / 256, 256>>>(b->didx[l], b->gcount, bc, K);
            k_accum_counts<<<(G + 255) / 256, 256>>>(ly.usage_acc, b->gcount, G);
        }
        k_readout_scatter<<<bc, 256>>>(b->didx[l], b->dval[l], b->derrc, b->derrw,
                                       b->w2class, b->dY, b->dweight, ly.Wc, ly.Ww,
                                       st, bc, K, C, S, b->Gcols);
        CUDA_KERNEL_CHECK();
    }
    phase_toc(tm, &b->t_scatter, t0);
    if (tm) phase_tic(&t0);
    k_bias_update<<<bc, 256>>>(b->derrc, b->derrw, b->w2class, b->dY, b->dweight,
                               ly.bc, ly.bw, st, bc, C, S);
    CUDA_KERNEL_CHECK();
    phase_toc(tm, &b->t_bias, t0);
}

/* sampled-softmax scatter: class head bucketed (full), word head atomic over the
 * sampled (1+neg) columns. Cuts the dominant word-head scatter ~S/(1+neg)x. */
static void readout_scatter_negw(Brain *b, Layer &ly, int l, int bc, float st) {
    int C = b->C, S = b->S, K = b->K, G = b->G, M = b->cfg.neg_word + 1;
    int tm = b->timing_on; double t0;
    if (tm) phase_tic(&t0);
    CUDA_CHECK(cudaMemset(b->gcount, 0, (size_t)G * sizeof(unsigned int)));
    int npairs = bc * K;
    k_bucket_count<<<(npairs + 255) / 256, 256>>>(b->didx[l], b->gcount, bc, K);
    if (b->cfg.balance)
        k_accum_counts<<<(G + 255) / 256, 256>>>(ly.usage_acc, b->gcount, G);
    k_prefix_sum<<<1, 1>>>(b->gcount, b->goff, G);
    CUDA_CHECK(cudaMemsetAsync(b->gcursor, 0, (size_t)G * sizeof(unsigned int)));
    k_bucket_fill<<<(npairs + 255) / 256, 256>>>(b->didx[l], b->goff, b->gcursor,
                                                 b->bk_e, b->bk_k, bc, K);
    k_scatter_class_bucketed<<<G, 256>>>(b->goff, b->bk_e, b->bk_k, b->dval[l],
                                         b->derrc, b->dweight, ly.Wc, st, G, C, K);
    k_scatter_word_negw<<<bc, 64>>>(b->didx[l], b->dval[l], b->dwerr, b->dwslot,
                                    b->w2class, b->dY, b->dweight, ly.Ww, ly.bw, st,
                                    bc, K, M, S, b->Gcols);
    CUDA_KERNEL_CHECK();
    phase_toc(tm, &b->t_scatter, t0);
    if (tm) phase_tic(&t0);
    k_bias_class<<<bc, 256>>>(b->derrc, b->dweight, ly.bc, st, bc, C);
    CUDA_KERNEL_CHECK();
    phase_toc(tm, &b->t_bias, t0);
}

/* one training step over a full batch, chunked. hpos = B random start positions.
 * NO host syncs in the hot path: the mixture EM update runs on-device (k_mixture_
 * update on b->dmix). Per-phase timing only when b->timing_on (brain_profile). */
static void brain_step(Brain *b, const long *hpos, int B) {
    Cfg &c = b->cfg;
    /* batch-INVARIANT step: normalize by a fixed reference batch (8192, where lr
     * was tuned), NOT the actual batch. Otherwise a bigger batch = fewer steps =
     * less total learning (the per-step move is lr*mean regardless of B). With a
     * fixed reference, a bigger batch takes a proportionally bigger step, so total
     * learning per token is constant and batch is a pure speed knob (linear
     * scaling rule). */
    const float LR_REF = 8192.0f;
    float st = c.lr * b->lr_scale / LR_REF;   /* lr_scale anneals over the run */
    int C = b->C, S = b->S, K = b->K, tm = b->timing_on;
    double t0;
    for (int lo = 0; lo < B; lo += c.step_chunk) {
        int bc = (lo + c.step_chunk <= B) ? c.step_chunk : (B - lo);
        fill_windows(b, hpos + lo, bc);
        encode_stack(b, bc);
        int N = c.neg_word;
        for (int l = 0; l < b->L; l++) {
            Layer &ly = b->layers[l];
            if (tm) phase_tic(&t0);
            if (N > 0) {
                int M = N + 1;
                k_sample_negw<<<(bc * N + 255) / 256, 256>>>(b->dnegw, bc, N, S,
                                                             c.seed + (b->negctr++));
                size_t shf = (size_t)(C + M) * sizeof(float) + (size_t)M * sizeof(int);
                k_readout_fwd_negw<<<bc, 256, shf>>>(b->didx[l], b->dval[l], ly.Wc, ly.Ww,
                                                     ly.bc, ly.bw, b->w2class, b->w2slot, b->dY,
                                                     b->dnegw, b->derrc, b->dwerr, b->dwslot,
                                                     b->dlogp + (size_t)l * bc, bc, K, C, S, N, b->Gcols);
            } else {
                size_t shf = (size_t)(C + S) * sizeof(float);
                k_readout_fwd<<<bc, 256, shf>>>(b->didx[l], b->dval[l], ly.Wc, ly.Ww,
                                                ly.bc, ly.bw, b->w2class, b->w2slot, b->dY,
                                                b->derrc, b->derrw, b->dlogp + (size_t)l * bc,
                                                bc, K, C, S, b->Gcols);
            }
            CUDA_KERNEL_CHECK();
            phase_toc(tm, &b->t_rfwd, t0);
            /* boosting: layer l>0 focuses on what layer l-1 got wrong */
            if (b->cfg.boost && l > 0)
                k_boost_weight<<<(bc + 255) / 256, 256>>>(b->dlogp + (size_t)(l - 1) * bc,
                                                          b->dweight, bc, b->cfg.boost_wmin);
            else
                k_fill_ones<<<(bc + 255) / 256, 256>>>(b->dweight, bc);
            CUDA_KERNEL_CHECK();
            if (N > 0) readout_scatter_negw(b, ly, l, bc, st);
            else       readout_scatter(b, ly, l, bc, st);
        }
        if (tm) phase_tic(&t0);
        k_mixture_update<<<1, 256>>>(b->dlogp, b->dmix, c.mix_lr, b->L, bc);
        CUDA_KERNEL_CHECK();
        phase_toc(tm, &b->t_mix, t0);
    }
}

/* lag-free analytics: run a short timed burst (timing_on=1) AFTER the hot loop,
 * so per-phase syncs never touch training throughput. */
static void brain_profile(Brain *b, int steps) {
    if (!b->cfg.analytics) return;
    b->t_bind = b->t_act = b->t_topk = b->t_relay = 0.0;
    b->t_rfwd = b->t_scatter = b->t_bias = b->t_mix = 0.0;
    int B = b->cfg.batch;
    std::vector<long> pos(B);
    long span = b->ctx + 1, hi = b->htoks_n - span; if (hi < 1) hi = 1;
    rng_t r; rng_seed(&r, b->cfg.seed + 123);
    b->timing_on = 1;
    for (int s = 0; s < steps; s++) {
        for (int e = 0; e < B; e++) pos[e] = (long)(rng_u64(&r) % (uint64_t)hi);
        brain_step(b, pos.data(), B);
    }
    b->timing_on = 0;
}

/* a colorful, frequent progress line (purple). Shows the running average AND the
 * instantaneous rate (since the last line) so "slowing down" is unambiguous. */
static void print_progress(long done, long total, double avg_tps, double inst_tps, double elapsed) {
    const int barw = 24;
    double frac = total > 0 ? (double)done / (double)total : 0.0;
    if (frac > 1.0) frac = 1.0;
    int fill = (int)(frac * barw);
    double eta = inst_tps > 1.0 ? (double)(total - done) / inst_tps : 0.0;
    printf("\033[1;35m  \xF0\x9F\xA7\xA0 [");
    for (int i = 0; i < barw; i++) printf("%s", i < fill ? "\xE2\x96\x88" : "\xE2\x96\x91");
    printf("] %3.0f%%\033[0m  \033[95m%.1f/%.1fM\033[0m  "
           "\033[1;96m%.0fk now\033[0m \033[35m(%.0fk avg)\033[0m  \033[35mETA %.0fs\033[0m\n",
           frac * 100.0, done / 1e6, total / 1e6, inst_tps / 1e3, avg_tps / 1e3, eta);
}

/* weight decay (call every decay_every steps) */
__global__ void k_decay(float *W, size_t n, float keep) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) W[i] *= keep;
}
static void brain_decay(Brain *b) {
    float keep = powf(1.0f - b->cfg.lr * b->cfg.wd, (float)b->cfg.decay_every);
    for (int l = 0; l < b->L; l++) {
        size_t nW = (size_t)b->G * b->C, nWw = (size_t)b->G * b->Gcols;
        k_decay<<<(nW + 255) / 256, 256>>>(b->layers[l].Wc, nW, keep);
        k_decay<<<(nWw + 255) / 256, 256>>>(b->layers[l].Ww, nWw, keep);
    }
}

/* eval perplexity on a host token stream (mixture). If ng!=NULL, ALSO scores the
 * infini-gram interpolation P=(1-lam)*P_brain + lam*P_ngram for EVERY lam in lams[]
 * in the same pass (out_ppl[i]); returns the brain-only ppl. */
static double brain_eval_ppl(Brain *b, const uint16_t *stream, long n, double *per_layer_ppl,
                             const Ngram *ng, const float *lams, int nlam, double *out_ppl) {
    int ctx = b->ctx, K = b->K, C = b->C, S = b->S, L = b->L;
    int chunk = b->cfg.step_chunk;
    std::vector<int> hX((size_t)chunk * ctx), hY(chunk);
    double nll = 0.0; long ntok = 0;
    std::vector<double> nll_l(L, 0.0), nll_lam(nlam > 0 ? nlam : 1, 0.0);
    std::vector<float> hlogp((size_t)L * chunk);
    float w[8];                                   /* L <= 8 (see Cfg/transient sizing) */
    float mmax = b->mix[0]; for (int l = 1; l < L; l++) if (b->mix[l] > mmax) mmax = b->mix[l];
    double zs = 0.0; for (int l = 0; l < L; l++) { w[l] = expf(b->mix[l] - mmax); zs += w[l]; }
    for (int l = 0; l < L; l++) w[l] = (float)(w[l] / zs);

    long last = n - ctx - 1;
    for (long base = 0; base < last; base += chunk) {
        int bc = (int)((base + chunk <= last) ? chunk : (last - base));
        if (bc <= 0) break;
        for (int e = 0; e < bc; e++) {
            for (int p = 0; p < ctx; p++) hX[(size_t)e * ctx + p] = stream[base + e + p];
            hY[e] = stream[base + e + ctx];
        }
        h2d(b->dX, hX.data(), (size_t)bc * ctx);
        h2d(b->dY, hY.data(), bc);
        encode_stack(b, bc);
        for (int l = 0; l < L; l++) {
            Layer &ly = b->layers[l];
            size_t shf = (C + S) * sizeof(float);
            k_readout_logp<<<bc, 256, shf>>>(b->didx[l], b->dval[l], ly.Wc, ly.Ww,
                                             ly.bc, ly.bw, b->w2class, b->w2slot, b->dY,
                                             b->dlogp + (size_t)l * bc, bc, K, C, S, b->Gcols);
            CUDA_KERNEL_CHECK();
        }
        d2h(hlogp.data(), b->dlogp, (size_t)L * bc);
        for (int e = 0; e < bc; e++) {
            float lw[8]; float mx = NEG_INF;
            for (int l = 0; l < L; l++) {
                float v = hlogp[(size_t)l * bc + e];
                nll_l[l] += -(double)v;
                lw[l] = logf(w[l] + 1e-30f) + v;
                if (lw[l] > mx) mx = lw[l];
            }
            double z = 0.0; for (int l = 0; l < L; l++) z += exp((double)(lw[l] - mx));
            double mixed = (double)mx + log(z);           /* log P_brain(y_e) */
            nll += -mixed;                                 /* brain-only (always) */
            if (ng) {
                double pb = exp(mixed);
                double pg = ngram_prob(ng, &hX[(size_t)e * ctx], ctx, hY[e]);
                for (int i = 0; i < nlam; i++) {
                    double Lm = lams[i];
                    nll_lam[i] += -log((1.0 - Lm) * pb + Lm * pg + 1e-30);
                }
            }
        }
        ntok += bc;
    }
    double denom = (double)(ntok > 0 ? ntok : 1);
    for (int l = 0; l < L; l++) per_layer_ppl[l] = exp(nll_l[l] / denom);
    if (ng && out_ppl) for (int i = 0; i < nlam; i++) out_ppl[i] = exp(nll_lam[i] / denom);
    return exp(nll / denom);
}

/* TEST-TIME ADAPTATION ("hyper-fixation"): read the doc in order, PREDICT each
 * chunk with the current weights (scored for ppl, no leakage), THEN learn from
 * it with the delta rule at adapt_lr. The brain specializes to THIS document as
 * it reads -- the frozen LLM cannot. Mutates the readout. */
/* adapt mutates the readout (lam-independent), so brain-only ppl and the infini-gram
 * interpolation for every lam in lams[] are accumulated in ONE pass: returns brain-only
 * ppl; out_ppl[i] gets the interpolated ppl for lams[i]. */
static double brain_eval_adapt(Brain *b, const uint16_t *stream, long n,
                               float adapt_lr, double *per_layer_ppl,
                               const Ngram *ng, const float *lams, int nlam, double *out_ppl) {
    int ctx = b->ctx, K = b->K, C = b->C, S = b->S, L = b->L;
    int chunk = b->cfg.step_chunk;
    std::vector<int> hX((size_t)chunk * ctx), hY(chunk);
    std::vector<float> hlogp((size_t)L * chunk);
    double nll = 0.0; long ntok = 0;
    std::vector<double> nll_l(L, 0.0), nll_lam(nlam > 0 ? nlam : 1, 0.0);
    long last = n - ctx - 1;
    for (long base = 0; base < last; base += chunk) {
        int bc = (int)((base + chunk <= last) ? chunk : (last - base));
        if (bc <= 0) break;
        for (int e = 0; e < bc; e++) {
            for (int p = 0; p < ctx; p++) hX[(size_t)e * ctx + p] = stream[base + e + p];
            hY[e] = stream[base + e + ctx];
        }
        h2d(b->dX, hX.data(), (size_t)bc * ctx);
        h2d(b->dY, hY.data(), bc);
        encode_stack(b, bc);
        /* current mixture weights from the (adapting) device mixture */
        d2h(b->mix, b->dmix, L);
        float w[8]; float mmax = b->mix[0];
        for (int l = 1; l < L; l++) if (b->mix[l] > mmax) mmax = b->mix[l];
        double zs = 0.0; for (int l = 0; l < L; l++) { w[l] = expf(b->mix[l] - mmax); zs += w[l]; }
        for (int l = 0; l < L; l++) w[l] = (float)(w[l] / zs);
        float st = adapt_lr / (float)bc;
        /* per layer: forward (captures dlogp BEFORE the update) then learn */
        for (int l = 0; l < L; l++) {
            Layer &ly = b->layers[l];
            size_t shf = (C + S) * sizeof(float);
            k_readout_fwd<<<bc, 256, shf>>>(b->didx[l], b->dval[l], ly.Wc, ly.Ww,
                                            ly.bc, ly.bw, b->w2class, b->w2slot, b->dY,
                                            b->derrc, b->derrw, b->dlogp + (size_t)l * bc,
                                            bc, K, C, S, b->Gcols);
            CUDA_KERNEL_CHECK();
            k_fill_ones<<<(bc + 255) / 256, 256>>>(b->dweight, bc);
            CUDA_KERNEL_CHECK();
            readout_scatter(b, ly, l, bc, st);     /* learn from this chunk */
        }
        d2h(hlogp.data(), b->dlogp, (size_t)L * bc);  /* pre-update predictions */
        for (int e = 0; e < bc; e++) {
            float lw[8]; float mx = NEG_INF;
            for (int l = 0; l < L; l++) {
                float v = hlogp[(size_t)l * bc + e];
                nll_l[l] += -(double)v;
                lw[l] = logf(w[l] + 1e-30f) + v;
                if (lw[l] > mx) mx = lw[l];
            }
            double z = 0.0; for (int l = 0; l < L; l++) z += exp((double)(lw[l] - mx));
            double mixed = (double)mx + log(z);
            nll += -mixed;                              /* brain-only (always) */
            if (ng) {                                   /* + infini-gram, all lam in one pass */
                double pb = exp(mixed);
                double pg = ngram_prob(ng, &hX[(size_t)e * ctx], ctx, hY[e]);
                for (int i = 0; i < nlam; i++) {
                    double Lm = lams[i];
                    nll_lam[i] += -log((1.0 - Lm) * pb + Lm * pg + 1e-30);
                }
            }
        }
        k_mixture_update<<<1, 256>>>(b->dlogp, b->dmix, b->cfg.mix_lr, L, bc);
        CUDA_KERNEL_CHECK();
        ntok += bc;
    }
    double denom = (double)(ntok > 0 ? ntok : 1);
    for (int l = 0; l < L; l++) per_layer_ppl[l] = exp(nll_l[l] / denom);
    if (ng && out_ppl) for (int i = 0; i < nlam; i++) out_ppl[i] = exp(nll_lam[i] / denom);
    return exp(nll / denom);
}

/* sample an index from a (possibly unnormalized) prob vector with temperature.
 * temp<=0 -> argmax; else reweight by p^(1/temp) and sample. */
static int sample_dist(const std::vector<double> &p, float temp, rng_t *r) {
    int n = (int)p.size();
    if (temp <= 1e-6f) {
        int best = 0; for (int i = 1; i < n; i++) if (p[i] > p[best]) best = i;
        return best;
    }
    double inv = 1.0 / (double)temp, tot = 0.0;
    std::vector<double> q(n);
    for (int i = 0; i < n; i++) { q[i] = pow(p[i] > 0 ? p[i] : 0, inv); tot += q[i]; }
    if (tot <= 0) return 0;
    double u = rng_unif(r) * tot, acc = 0.0;
    for (int i = 0; i < n; i++) { acc += q[i]; if (u <= acc) return i; }
    return n - 1;
}

/* GENERATION: autoregressively sample `N` tokens from a seed context. Per step,
 * ancestral sampling over the layer mixture: sample a class from the mixed class
 * distribution, then a word from the mixed within-class distribution. Returns the
 * generated vocab ids (Python detokenizes). */
static void brain_generate(Brain *b, const uint16_t *seed, int N, float temp,
                           std::vector<int> &out) {
    int ctx = b->ctx, K = b->K, C = b->C, S = b->S, L = b->L;
    std::vector<int> ctxbuf(ctx);
    for (int i = 0; i < ctx; i++) ctxbuf[i] = (int)seed[i];
    float *d_pc = dev_alloc<float>((size_t)L * C);
    float *d_pw = dev_alloc<float>((size_t)L * S);
    std::vector<float> h_pc((size_t)L * C), h_pw((size_t)L * S);
    std::vector<int> cw((size_t)C * S);
    d2h(cw.data(), b->class_word, (size_t)C * S);
    d2h(b->mix, b->dmix, L);
    float w[8]; float mmax = b->mix[0];
    for (int l = 1; l < L; l++) if (b->mix[l] > mmax) mmax = b->mix[l];
    double zs = 0.0; for (int l = 0; l < L; l++) { w[l] = expf(b->mix[l] - mmax); zs += w[l]; }
    for (int l = 0; l < L; l++) w[l] = (float)(w[l] / zs);
    rng_t r; rng_seed(&r, b->cfg.seed + 777);
    out.clear();
    for (int t = 0; t < N; t++) {
        h2d(b->dX, ctxbuf.data(), (size_t)ctx);     /* bc = 1 */
        encode_stack(b, 1);
        for (int l = 0; l < L; l++)
            k_gen_class<<<1, 256, C * sizeof(float)>>>(b->didx[l], b->dval[l],
                       b->layers[l].Wc, b->layers[l].bc, d_pc + (size_t)l * C, C, K);
        CUDA_KERNEL_CHECK();
        d2h(h_pc.data(), d_pc, (size_t)L * C);
        std::vector<double> Pc(C, 0.0);
        for (int c = 0; c < C; c++) { double s = 0; for (int l = 0; l < L; l++) s += w[l] * h_pc[(size_t)l * C + c]; Pc[c] = s; }
        int c = sample_dist(Pc, temp, &r);
        for (int l = 0; l < L; l++)
            k_gen_within<<<1, 256, S * sizeof(float)>>>(b->didx[l], b->dval[l],
                        b->layers[l].Ww, b->layers[l].bw, c, d_pw + (size_t)l * S, S, K, b->Gcols);
        CUDA_KERNEL_CHECK();
        d2h(h_pw.data(), d_pw, (size_t)L * S);
        std::vector<double> Pw(S, 0.0);
        for (int j = 0; j < S; j++) { double s = 0; for (int l = 0; l < L; l++) s += w[l] * h_pc[(size_t)l * C + c] * h_pw[(size_t)l * S + j]; Pw[j] = s; }
        int j = sample_dist(Pw, temp, &r);
        int word = cw[(size_t)c * S + j];
        if (word >= b->V) word = 0;
        out.push_back(word);
        for (int i = 0; i < ctx - 1; i++) ctxbuf[i] = ctxbuf[i + 1];
        ctxbuf[ctx - 1] = word;
    }
    dev_free(d_pc); dev_free(d_pw);
}

/* sample M random start positions into a host vector */
static void sample_positions(Brain *b, std::vector<long> &pos, int M, unsigned long seed) {
    long span = b->ctx + 1;
    long hi = b->htoks_n - span;
    if (hi < 1) hi = 1;
    rng_t r; rng_seed(&r, seed);
    pos.resize(M);
    for (int i = 0; i < M; i++) pos[i] = (long)(rng_u64(&r) % (uint64_t)hi);
}

/* M4: competitive (online k-means under WTA) granule learning. Flows the sample
 * up the stack so each layer learns on its true input. Off the hot path. */
static void brain_competitive(Brain *b, int M, int passes, float eta) {
    Cfg &c = b->cfg;
    int K = b->K, fan = c.fan_in, G = b->G, chunk = c.step_chunk;
    std::vector<long> pos;
    sample_positions(b, pos, M, c.seed + 7);
    for (int p = 0; p < passes; p++) {
        float et = eta * (1.0f - (float)p / (float)passes);
        for (int l = 0; l < b->L; l++) {
            CUDA_CHECK(cudaMemset(b->csum, 0, (size_t)G * fan * sizeof(float)));
            CUDA_CHECK(cudaMemset(b->ccount, 0, (size_t)G * sizeof(unsigned int)));
            for (int off = 0; off < M; off += chunk) {
                int bc = (off + chunk <= M) ? chunk : (M - off);
                fill_windows(b, pos.data() + off, bc);
                encode_stack(b, bc);
                int np = bc * K;
                k_comp_accum<<<(np + 255) / 256, 256>>>(b->didx[l], NULL, b->dx[l],
                                                        b->layers[l].gidx, b->csum,
                                                        b->ccount, bc, K, fan,
                                                        b->layers[l].in_dim);
                CUDA_KERNEL_CHECK();
            }
            k_comp_apply<<<(G + 255) / 256, 256>>>(b->layers[l].gwt, b->csum, b->ccount,
                                                   et, G, fan);
            CUDA_KERNEL_CHECK();
            /* rebuild dense projection (+ fp16 mirror) from updated weights */
            CUDA_CHECK(cudaMemset(b->layers[l].projT, 0,
                                  (size_t)G * b->layers[l].in_dim * sizeof(float)));
            int nt = G * fan;
            k_build_projT<<<(nt + 255) / 256, 256>>>(b->layers[l].gidx, b->layers[l].gwt,
                                                     b->layers[l].projT, G, fan,
                                                     b->layers[l].in_dim);
            CUDA_KERNEL_CHECK();
            size_t pn = (size_t)G * b->layers[l].in_dim;
            k_f32_to_f16<<<(pn + 255) / 256, 256>>>(b->layers[l].projT, b->layers[l].projT_h, pn);
            CUDA_KERNEL_CHECK();
        }
    }
}

/* analytics: granule-code health on layer 0 over a sample (dead frac, usage Gini,
 * mean kWTA margin) — mirrors test_012 brain_diagnose. */
static void brain_health(Brain *b, int M, double *dead_frac, double *gini, double *mean_margin) {
    int K = b->K, G = b->G, chunk = b->cfg.step_chunk;
    std::vector<long> pos; sample_positions(b, pos, M, b->cfg.seed + 11);
    unsigned int *usage = dev_zalloc<unsigned int>(G);
    double *dmsum = dev_zalloc<double>(1);
    long ntok = 0;
    for (int off = 0; off < M; off += chunk) {
        int bc = (off + chunk <= M) ? chunk : (M - off);
        fill_windows(b, pos.data() + off, bc);
        encode_stack(b, bc);
        int np = bc * K;
        k_usage_accum<<<(np + 255) / 256, 256>>>(b->didx[0], b->dval[0], usage, dmsum, bc, K);
        CUDA_KERNEL_CHECK();
        ntok += bc;
    }
    std::vector<unsigned int> h(G);
    d2h(h.data(), usage, G);
    double msum; d2h(&msum, dmsum, 1);
    long dead = 0; double tot = 0.0;
    for (int g = 0; g < G; g++) { if (h[g] == 0) dead++; tot += (double)h[g]; }
    *dead_frac = (double)dead / G;
    *mean_margin = (ntok > 0) ? msum / ((double)ntok * K) : 0.0;
    std::sort(h.begin(), h.end());
    double cum = 0.0;
    if (tot > 0) {
        for (int g = 0; g < G; g++) cum += (double)(2 * (g + 1) - G - 1) * (double)h[g];
        *gini = cum / ((double)G * tot);
    } else *gini = 0.0;
    dev_free(usage); dev_free(dmsum);
}

static long brain_macs_per_token(Brain *b) {
    Cfg &c = b->cfg;
    /* granule subsampling reduces the granules activated per token (eval subsamples
     * too), so the honest sparse cost uses the effective granule count. */
    long Geff = c.n_gran;
    if (c.gran_frac < 1.0f) {
        Geff = (long)(c.n_gran * c.gran_frac);
        Geff -= Geff % c.topk_groups;
        if (Geff < c.topk_groups) Geff = c.topk_groups;
    }
    long per_layer = Geff * c.fan_in + (long)c.k_active * (c.n_classes + b->S);
    long relays = (long)(c.n_layers - 1) * c.k_active * c.relay_dim;
    long bind = (long)c.ctx * c.code_dim;
    return (long)c.n_layers * per_layer + relays + bind + c.n_layers;
}

/* ─────────────────────── dual-GPU local-SGD (M5) ──────────────────────────── */
struct Shared {
    float *hbuf[2];          /* host staging, sized to the largest tensor (Ww) */
    pthread_barrier_t bar;
    int world;
};

/* average one device tensor across ranks via host staging (no P2P needed):
 * each rank copies its tensor to its host buffer; rank 0 averages; both upload. */
static void avg_tensor(Shared *sh, int rank, float *Tdev, long n) {
    if (!sh || sh->world < 2) return;
    d2h(sh->hbuf[rank], Tdev, (size_t)n);
    pthread_barrier_wait(&sh->bar);
    if (rank == 0)
        for (long i = 0; i < n; i++) sh->hbuf[0][i] = 0.5f * (sh->hbuf[0][i] + sh->hbuf[1][i]);
    pthread_barrier_wait(&sh->bar);
    h2d(Tdev, sh->hbuf[0], (size_t)n);
    pthread_barrier_wait(&sh->bar);
}

/* sync the learned encoders (gwt, gbias) ONCE after competitive learning, then
 * rebuild projT, so every replica is the SAME encoder (readout averaging across
 * a shared granule basis is only valid if the bases match). */
static void average_granules(Shared *sh, Brain *b, int rank) {
    if (!sh || sh->world < 2) return;
    int fan = b->cfg.fan_in;
    for (int l = 0; l < b->L; l++) {
        avg_tensor(sh, rank, b->layers[l].gwt, (long)b->G * fan);
        if (b->cfg.balance) avg_tensor(sh, rank, b->layers[l].gbias, b->G);  /* M3: only when active */
        CUDA_CHECK(cudaMemset(b->layers[l].projT, 0,
                              (size_t)b->G * b->layers[l].in_dim * sizeof(float)));
        int nt = b->G * fan;
        k_build_projT<<<(nt + 255) / 256, 256>>>(b->layers[l].gidx, b->layers[l].gwt,
                                                 b->layers[l].projT, b->G, fan,
                                                 b->layers[l].in_dim);
        CUDA_KERNEL_CHECK();
        size_t pn = (size_t)b->G * b->layers[l].in_dim;
        k_f32_to_f16<<<(pn + 255) / 256, 256>>>(b->layers[l].projT, b->layers[l].projT_h, pn);
        CUDA_KERNEL_CHECK();
    }
}

/* average the learned readouts (local-SGD) across ranks */
static void average_readouts(Shared *sh, Brain *b, int rank) {
    if (!sh || sh->world < 2) return;
    for (int l = 0; l < b->L; l++) {
        avg_tensor(sh, rank, b->layers[l].Wc, (long)b->G * b->C);
        avg_tensor(sh, rank, b->layers[l].Ww, (long)b->G * b->Gcols);
        avg_tensor(sh, rank, b->layers[l].bc, b->C);
        avg_tensor(sh, rank, b->layers[l].bw, b->Gcols);
        /* keep the selection bias in lock-step so the shared granule basis (and thus
         * readout averaging) stays valid across ranks. Reset the per-rank usage
         * integrator so the controller restarts cleanly from the averaged gbias
         * (M2: avoids integrator/actuator windup). Caller resets bal_consumed. */
        if (b->cfg.balance) {
            avg_tensor(sh, rank, b->layers[l].gbias, b->G);
            CUDA_CHECK(cudaMemset(b->layers[l].usage_acc, 0,
                                  (size_t)b->G * sizeof(unsigned long long)));
        }
    }
    avg_tensor(sh, rank, b->dmix, b->L);
}

/* ─────────────────────── checkpoint / resume (v15) ─────────────────────────────
 * Save only the LEARNED state (gwt, gbias, Wc, Ww, bc, bw, mixture). The fixed
 * random front-end (codes, pos, decay, gidx, relayR) and the freq-derived hierarchy
 * are regenerated by brain_init, so a checkpoint is ~0.4GB (Ww dominates) and a load
 * skips competitive feature learning. Valid only for a MATCHING config + corpus. */
#define BF_CKPT_MAGIC 0x42466631   /* 'BFf1' */
static bool save_brain(Brain *b, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "  ckpt: cannot open %s for write\n", path); return false; }
    int fan = b->cfg.fan_in;
    int hdr[11] = {BF_CKPT_MAGIC, 1, b->V, b->G, b->K, b->C, b->S, b->Gcols, b->relay_dim, b->L, fan};
    fwrite(hdr, sizeof(int), 11, f);
    for (int l = 0; l < b->L; l++) { int id = b->layers[l].in_dim; fwrite(&id, sizeof(int), 1, f); }
    std::vector<float> buf((size_t)b->G * b->Gcols);          /* Ww is the largest tensor */
    for (int l = 0; l < b->L; l++) {
        Layer &ly = b->layers[l];
        d2h(buf.data(), ly.gwt,  (size_t)b->G * fan);       fwrite(buf.data(), 4, (size_t)b->G * fan, f);
        d2h(buf.data(), ly.gbias, (size_t)b->G);            fwrite(buf.data(), 4, (size_t)b->G, f);
        d2h(buf.data(), ly.Wc,   (size_t)b->G * b->C);      fwrite(buf.data(), 4, (size_t)b->G * b->C, f);
        d2h(buf.data(), ly.Ww,   (size_t)b->G * b->Gcols);  fwrite(buf.data(), 4, (size_t)b->G * b->Gcols, f);
        d2h(buf.data(), ly.bc,   (size_t)b->C);             fwrite(buf.data(), 4, (size_t)b->C, f);
        d2h(buf.data(), ly.bw,   (size_t)b->Gcols);         fwrite(buf.data(), 4, (size_t)b->Gcols, f);
    }
    d2h(buf.data(), b->dmix, (size_t)b->L); fwrite(buf.data(), 4, (size_t)b->L, f);
    fclose(f);
    return true;
}
static bool load_brain(Brain *b, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  ckpt: cannot open %s for read\n", path); return false; }
    int fan = b->cfg.fan_in, hdr[11];
    if (fread(hdr, sizeof(int), 11, f) != 11) { fclose(f); return false; }
    if (hdr[0] != BF_CKPT_MAGIC || hdr[2] != b->V || hdr[3] != b->G || hdr[4] != b->K ||
        hdr[5] != b->C || hdr[6] != b->S || hdr[7] != b->Gcols || hdr[9] != b->L || hdr[10] != fan) {
        fprintf(stderr, "  ckpt: header mismatch (config/corpus differ) — not loading %s\n", path);
        fclose(f); return false;
    }
    for (int l = 0; l < b->L; l++) {
        int id = 0; if (fread(&id, sizeof(int), 1, f) != 1 || id != b->layers[l].in_dim) {
            fprintf(stderr, "  ckpt: layer %d in_dim mismatch\n", l); fclose(f); return false; }
    }
    std::vector<float> buf((size_t)b->G * b->Gcols);
    bool ok = true;
    for (int l = 0; l < b->L && ok; l++) {
        Layer &ly = b->layers[l];
        size_t sizes[6] = {(size_t)b->G * fan, (size_t)b->G, (size_t)b->G * b->C,
                           (size_t)b->G * b->Gcols, (size_t)b->C, (size_t)b->Gcols};
        float *dsts[6] = {ly.gwt, ly.gbias, ly.Wc, ly.Ww, ly.bc, ly.bw};
        for (int t = 0; t < 6 && ok; t++) {
            if (fread(buf.data(), 4, sizes[t], f) != sizes[t]) ok = false;
            else h2d(dsts[t], buf.data(), sizes[t]);
        }
        if (ok) {   /* rebuild the dense projection + fp16 mirror from loaded gwt */
            CUDA_CHECK(cudaMemset(ly.projT, 0, (size_t)b->G * ly.in_dim * sizeof(float)));
            int nt = b->G * fan;
            k_build_projT<<<(nt + 255) / 256, 256>>>(ly.gidx, ly.gwt, ly.projT, b->G, fan, ly.in_dim);
            size_t pn = (size_t)b->G * ly.in_dim;
            k_f32_to_f16<<<(pn + 255) / 256, 256>>>(ly.projT, ly.projT_h, pn);
            CUDA_KERNEL_CHECK();
        }
    }
    if (ok && fread(buf.data(), 4, (size_t)b->L, f) == (size_t)b->L) {
        h2d(b->dmix, buf.data(), (size_t)b->L);
        for (int l = 0; l < b->L; l++) b->mix[l] = buf[l];
    }
    fclose(f);
    return ok;
}

/* ─────────────────────── per-rank train + (rank 0) eval ────────────────────── */
struct RankCtx {
    Cfg cfg; Meta meta;
    const uint16_t *train_data; long train_n;
    const uint16_t *test_data;  long test_n;
    int V; const long *freq;
    int rank, world; Shared *sh;
};

static void run_rank(RankCtx *rc) {
    Cfg cfg = rc->cfg;   /* mutable copy: the balance/gran-frac guard may flip a flag */
    int rank = rc->rank, world = rc->world;
    Shared *sh = rc->sh;
    if (world > 1) CUDA_CHECK(cudaSetDevice(rank));

    Brain brain;
    brain_init(&brain, cfg, rc->V, rc->freq);
    brain.htoks = rc->train_data; brain.htoks_n = rc->train_n;

    if (cfg.resident) {
        size_t bytes = (size_t)rc->train_n * sizeof(unsigned short);
        size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
        if (bytes < freeb * 3 / 4) {
            brain.dtoks = dev_alloc<unsigned short>(rc->train_n);
            h2d(brain.dtoks, rc->train_data, rc->train_n);
            brain.dtoks_n = rc->train_n;
            if (rank == 0) printf("  resident: train shard in VRAM (%.2f GB)\n", bytes / 1e9);
        } else if (rank == 0) {
            printf("  resident: shard too big for VRAM, using host gather\n");
        }
    }

    bool loaded = false;
    if (cfg.load_path[0]) {
        loaded = load_brain(&brain, cfg.load_path);   /* both ranks load the same checkpoint */
        if (loaded && rank == 0)
            printf("  brain: loaded checkpoint %s (skipping feature learning)\n", cfg.load_path);
    }
    if (cfg.learn_feat && !loaded) {
        if (rank == 0) printf("  brain: competitive granule learning (%d layers) ...\n", brain.L);
        double tf = now_sec();
        brain_competitive(&brain, cfg.feat_sample, cfg.feat_passes, cfg.feat_eta);
        average_granules(sh, &brain, rank);    /* make all replicas the same encoder */
        CUDA_CHECK(cudaDeviceSynchronize());
        if (rank == 0) printf("  brain: features learned (%.1fs)\n", now_sec() - tf);
    }

    int B = cfg.batch, ctx = cfg.ctx;
    long steps_per_block = cfg.block / B; if (steps_per_block < 1) steps_per_block = 1;
    long total_steps = cfg.train_tokens / (B * (long)world); if (total_steps < 1) total_steps = 1;
    long n_blocks = total_steps / steps_per_block; if (n_blocks < 1) n_blocks = 1;
    long sched_total = n_blocks * steps_per_block;   /* actual #steps, for lr anneal */

    /* this rank's disjoint data shard */
    long span = ctx + 1;
    long shard_lo = rank * rc->train_n / world;
    long shard_hi = (rank + 1) * rc->train_n / world - span;
    if (shard_hi <= shard_lo) { shard_lo = 0; shard_hi = rc->train_n - span; }
    if (shard_hi < 1) shard_hi = 1;

    std::vector<long> hpos(B);
    rng_t sr; rng_seed(&sr, cfg.seed + 2 + rank);
    if (rank == 0)
        printf("  brain: %d layers x %ld blocks x %ld steps x batch %d x %d gpu(s)  "
               "(scatter=%s, resident=%d, cublas=%d)\n",
               brain.L, n_blocks, steps_per_block, B, world,
               cfg.fast_scatter ? "bucketed" : "atomic", brain.dtoks ? 1 : 0, cfg.use_cublas);
    if (rank == 0 && cfg.lr_final != 1.0f)
        printf("  brain: lr cosine-anneal %.3f -> %.3f over %ld steps\n",
               cfg.lr, cfg.lr * cfg.lr_final, sched_total);
    /* M1: subsampling makes the per-example fire-rate denominator wrong (a granule is
     * only eligible in Geff/G of examples). Disable balance under gran_frac<1. */
    if (cfg.balance && cfg.gran_frac < 1.0f) {
        if (rank == 0) printf("  brain: load-balancing disabled (incompatible with --gran-frac<1)\n");
        cfg.balance = 0; brain.cfg.balance = 0;   /* keep both copies in sync (kernels read brain.cfg) */
    }
    if (rank == 0 && cfg.balance)
        printf("  brain: homeostatic load-balancing on (gain=%.3f every %d steps, "
               "target fire-rate %.4f)\n", cfg.balance_lr, cfg.balance_every,
               (float)brain.K / (float)brain.G);
    double t0 = now_sec();
    long consumed = 0; int step = 0; long bal_consumed = 0;
    int pe = cfg.progress_every > 0 ? cfg.progress_every : 1;
    double prev_el = 0.0; long prev_tot = 0;     /* for instantaneous tok/s */

    for (long blk = 0; blk < n_blocks; blk++) {
        for (long s = 0; s < steps_per_block; s++) {
            for (int e = 0; e < B; e++)
                hpos[e] = shard_lo + (long)(rng_u64(&sr) % (uint64_t)(shard_hi - shard_lo));
            if (cfg.lr_final != 1.0f) {           /* cosine anneal lr -> lr*lr_final */
                double prog = sched_total > 1 ? (double)step / (double)sched_total : 0.0;
                brain.lr_scale = cfg.lr_final + (1.0f - cfg.lr_final) * 0.5f *
                                 (1.0f + cosf((float)(3.14159265358979 * prog)));
            }
            brain_step(&brain, hpos.data(), B);
            consumed += B; bal_consumed += B;
            if (++step % cfg.decay_every == 0 && cfg.wd > 0) brain_decay(&brain);
            if (cfg.balance && step % cfg.balance_every == 0) {
                float target = (float)brain.K / (float)brain.G;
                float inv_n = bal_consumed > 0 ? 1.0f / (float)bal_consumed : 0.0f;
                for (int l = 0; l < brain.L; l++)
                    k_balance_apply<<<(brain.G + 255) / 256, 256>>>(
                        brain.layers[l].gbias, brain.layers[l].usage_acc,
                        brain.G, target, cfg.balance_lr, inv_n, 8.0f);
                CUDA_KERNEL_CHECK();
                bal_consumed = 0;
            }
        }
        if (cfg.learn_feat && cfg.refine_every && (blk + 1) % cfg.refine_every == 0) {
            brain_competitive(&brain, cfg.refine_sample, 1, cfg.refine_eta);
            /* re-sync encoders after refine so readout averaging stays valid */
            average_granules(sh, &brain, rank);
            /* M3: competitive refine retunes the granule basis; restart the balance
             * controller so it doesn't carry stale fire-rate history (single + dual GPU). */
            if (cfg.balance) {
                for (int l = 0; l < brain.L; l++)
                    CUDA_CHECK(cudaMemset(brain.layers[l].usage_acc, 0,
                                          (size_t)brain.G * sizeof(unsigned long long)));
                bal_consumed = 0;
            }
        }
        if (world > 1 && (blk + 1) % cfg.sync_every == 0) {
            average_readouts(sh, &brain, rank);
            if (cfg.balance) bal_consumed = 0;   /* M2: usage_acc reset inside; reset its denom too */
        }
        if (rank == 0 && (blk + 1) % pe == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            double el = now_sec() - t0;
            long tot = consumed * world;
            double avg = tot / (el > 1e-6 ? el : 1e-6);
            double dt = el - prev_el;
            double inst = dt > 1e-6 ? (tot - prev_tot) / dt : avg;
            print_progress(tot, cfg.train_tokens, avg, inst, el);
            prev_el = el; prev_tot = tot;
        }
    }
    if (world > 1) average_readouts(sh, &brain, rank);   /* final mandatory average */
    CUDA_CHECK(cudaDeviceSynchronize());
    double train_s = now_sec() - t0;
    long total_consumed = consumed * world;
    double tok_s = total_consumed / (train_s > 1e-6 ? train_s : 1e-6);

    if (rank != 0) { dev_free(brain.dtoks); return; }

    /* ---- rank 0: analytics + eval + scoreboard + JSON ---- */
    d2h(brain.mix, brain.dmix, brain.L);          /* sync device mixture for eval/JSON */

    if (cfg.save_path[0]) {                        /* checkpoint the trained state */
        if (save_brain(&brain, cfg.save_path))
            printf("  brain: saved checkpoint -> %s (%.2f GB)\n", cfg.save_path,
                   (double)brain.L * brain.G * brain.Gcols * 4.0 / 1e9);
    }

    double dead_frac = -1, gini = -1, mean_margin = -1;
    if (cfg.analytics) {
        brain_profile(&brain, 8);                 /* lag-free: timed burst AFTER training */
        brain_health(&brain, 8000, &dead_frac, &gini, &mean_margin);
        double tt = brain.t_bind + brain.t_act + brain.t_topk + brain.t_relay +
                    brain.t_rfwd + brain.t_scatter + brain.t_bias + brain.t_mix;
        if (tt < 1e-9) tt = 1e-9;
        printf("\n\033[1;35m  --- per-phase timing (profiled) ---\033[0m\n");
        printf("    bind %.1f%%  activate %.1f%%  topk %.1f%%  relay %.1f%%\n",
               100*brain.t_bind/tt, 100*brain.t_act/tt, 100*brain.t_topk/tt, 100*brain.t_relay/tt);
        printf("    readout-fwd %.1f%%  scatter %.1f%%  bias %.1f%%  mixture %.1f%%\n",
               100*brain.t_rfwd/tt, 100*brain.t_scatter/tt, 100*brain.t_bias/tt, 100*brain.t_mix/tt);
        printf("    granule health (layer 0): dead %.1f%%  usage-Gini %.3f  mean-margin %.3f\n",
               100*dead_frac, gini, mean_margin);
    }

    /* optional infini-gram for eval interpolation (highest-EV quality lever) */
    Ngram *ng = NULL;
    if (cfg.ngram_path[0] && !cfg.bench) ng = ngram_open(cfg.ngram_path, brain.V);

    /* infini-gram lambda sweep (scored in one eval pass; cfg.ngram_lambda is added so
     * a custom value is always in the table) */
    float lams[8] = {0.1f, 0.2f, 0.3f, 0.5f, 0.7f, 0.9f, cfg.ngram_lambda, 0.0f};
    int nlam = ng ? 7 : 0;
    std::vector<double> ng_sweep(8, -1.0), adapt_sweep(8, -1.0);

    double br_ppl = -1.0;
    std::vector<double> per_layer(brain.L, -1.0);
    if (rc->test_data && !cfg.bench)
        br_ppl = brain_eval_ppl(&brain, rc->test_data, rc->test_n, per_layer.data(),
                                ng, lams, nlam, ng_sweep.data());
    long macs = brain_macs_per_token(&brain);
    printf("\n\033[1;35m  Brain (no-bp, %dL): perplexity %.2f  %ld MACs/token  "
           "(trained %ld tok in %.0fs, %.0f tok/s)\033[0m\n",
           brain.L, br_ppl, macs, total_consumed, train_s, tok_s);
    if (ng) {
        int bi = 0; for (int i = 1; i < nlam; i++) if (ng_sweep[i] < ng_sweep[bi]) bi = i;
        printf("\033[1;35m  Brain + infini-gram lambda sweep (static):\033[0m\n");
        for (int i = 0; i < nlam; i++)
            printf("\033[35m    lambda %.2f -> ppl %8.2f  (%+.1f%%)%s\033[0m\n", lams[i], ng_sweep[i],
                   br_ppl > 0 ? 100.0 * (ng_sweep[i] - br_ppl) / br_ppl : 0.0, i == bi ? "  <- best" : "");
    }

    /* test-time adaptation: the table-turner. Runs AFTER static eval (which does
     * not mutate); this one specializes the readout to the test doc as it reads. */
    double adapt_ppl = -1.0;
    std::vector<double> adapt_pl(brain.L, -1.0);
    if (cfg.adapt && rc->test_data && !cfg.bench) {
        /* single pass: adapt mutates (lam-independent), so score brain-only + every lam together */
        adapt_ppl = brain_eval_adapt(&brain, rc->test_data, rc->test_n, cfg.adapt_lr,
                                     adapt_pl.data(), ng, lams, nlam, adapt_sweep.data());
        double gain = (br_ppl > 0) ? 100.0 * (br_ppl - adapt_ppl) / br_ppl : 0.0;
        printf("\033[1;35m  Brain ADAPTIVE (hyper-fixation, lr=%.2f): perplexity %.2f  "
               "(static %.2f -> %+.1f%%; LLM is frozen at this number)\033[0m\n",
               cfg.adapt_lr, adapt_ppl, br_ppl, -gain);
        if (ng) {   /* adaptation + infini-gram: the full v15 stack */
            int bi = 0; for (int i = 1; i < nlam; i++) if (adapt_sweep[i] < adapt_sweep[bi]) bi = i;
            printf("\033[1;35m  Brain ADAPTIVE + infini-gram lambda sweep:\033[0m\n");
            for (int i = 0; i < nlam; i++)
                printf("\033[35m    lambda %.2f -> ppl %8.2f  (%+.1f%% vs adaptive)%s\033[0m\n",
                       lams[i], adapt_sweep[i],
                       adapt_ppl > 0 ? 100.0 * (adapt_sweep[i] - adapt_ppl) / adapt_ppl : 0.0,
                       i == bi ? "  <- best" : "");
        }
    }
    if (ng) ngram_close(ng);

    /* generation: autoregressive sampling from the test seed -> token-id file */
    if (cfg.gen_tokens > 0 && rc->test_data && !cfg.bench) {
        std::vector<int> gen;
        brain_generate(&brain, rc->test_data, cfg.gen_tokens, cfg.gen_temp, gen);
        const char *gp = cfg.gen_out[0] ? cfg.gen_out : "/kaggle/working/brain_gen.json";
        FILE *gf = fopen(gp, "w");
        if (gf) {
            fprintf(gf, "{\"temp\": %.2f, \"seed\": [", cfg.gen_temp);
            for (int i = 0; i < brain.ctx; i++) fprintf(gf, "%s%d", i ? "," : "", (int)rc->test_data[i]);
            fprintf(gf, "], \"tokens\": [");
            for (size_t i = 0; i < gen.size(); i++) fprintf(gf, "%s%d", i ? "," : "", gen[i]);
            fprintf(gf, "]}\n");
            fclose(gf);
            printf("  generated %d tokens -> %s\n", cfg.gen_tokens, gp);
        }
    }

    if (cfg.out_path[0]) {
        FILE *f = fopen(cfg.out_path, "w");
        if (f) {
            fprintf(f, "{\"brain_ppl\": %.4f, \"macs_per_token\": %ld, "
                       "\"consumed_tokens\": %ld, \"train_seconds\": %.1f, "
                       "\"tok_per_sec\": %.0f, \"n_gpus\": %d, \"n_layers\": %d, \"n_gran\": %d, "
                       "\"n_classes\": %d, \"per_layer_ppl\": [",
                    br_ppl, macs, total_consumed, train_s, tok_s, world, brain.L, cfg.n_gran, cfg.n_classes);
            for (int l = 0; l < brain.L; l++) fprintf(f, "%s%.4f", l ? ", " : "", per_layer[l]);
            fprintf(f, "], \"mix\": [");
            float mmax = brain.mix[0]; for (int l = 1; l < brain.L; l++) if (brain.mix[l] > mmax) mmax = brain.mix[l];
            double zs = 0.0; std::vector<float> w(brain.L);
            for (int l = 0; l < brain.L; l++) { w[l] = expf(brain.mix[l] - mmax); zs += w[l]; }
            for (int l = 0; l < brain.L; l++) fprintf(f, "%s%.4f", l ? ", " : "", w[l] / (float)zs);
            fprintf(f, "], ");
            double tt = brain.t_bind + brain.t_act + brain.t_topk + brain.t_relay +
                        brain.t_rfwd + brain.t_scatter + brain.t_bias + brain.t_mix;
            if (tt < 1e-9) tt = 1e-9;
            fprintf(f, "\"scatter\": \"%s\", \"resident\": %d, \"cublas\": %d, ",
                    cfg.fast_scatter ? "bucketed" : "atomic", brain.dtoks ? 1 : 0, cfg.use_cublas);
            fprintf(f, "\"phase_pct\": {\"bind\": %.2f, \"activate\": %.2f, \"topk\": %.2f, "
                       "\"relay\": %.2f, \"readout_fwd\": %.2f, \"scatter\": %.2f, "
                       "\"bias\": %.2f, \"mixture\": %.2f}, ",
                    100*brain.t_bind/tt, 100*brain.t_act/tt, 100*brain.t_topk/tt,
                    100*brain.t_relay/tt, 100*brain.t_rfwd/tt, 100*brain.t_scatter/tt,
                    100*brain.t_bias/tt, 100*brain.t_mix/tt);
            fprintf(f, "\"adapt_ppl\": %.4f, ", adapt_ppl);
            fprintf(f, "\"health\": {\"dead_frac\": %.4f, \"usage_gini\": %.4f, "
                       "\"mean_margin\": %.4f}}\n", dead_frac, gini, mean_margin);
            fclose(f);
            printf("  wrote %s\n", cfg.out_path);
        }
    }
    dev_free(brain.dtoks);
}

/* ─────────────────────────────── main ─────────────────────────────────────── */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    Cfg cfg = cfg_parse(argc, argv);
    if (cfg.n_layers > 8) { cfg.n_layers = 8; fprintf(stderr, "n_layers clamped to 8\n"); }
    if (cfg.topk_per_group > 32) cfg.topk_per_group = 32;   /* MAX_PG local array */
    if (!cfg.exact_topk &&
        (cfg.topk_groups * cfg.topk_per_group < cfg.k_active + 1 ||
         cfg.n_gran % cfg.topk_groups != 0)) {
        fprintf(stderr, "  note: approx-topk params invalid (need groups*per_group>=K+1 and "
                        "n_gran%%groups==0); using exact top-k.\n");
        cfg.exact_topk = 1;
    }
    threads_init(0);

    if (cfg.meta_path[0] == 0) { fprintf(stderr, "need --meta path/to/meta.json\n"); return 1; }
    Meta meta;
    if (meta_load(cfg.meta_path, &meta)) return 1;

    TokenBin train, test;
    if (tokenbin_open(meta.train_path, cfg.train_tokens > 0 ? cfg.train_tokens : meta.train_tokens, &train)) return 1;
    test.data = NULL;
    if (meta.test_path[0]) tokenbin_open(meta.test_path, meta.test_tokens, &test);

    int V = meta.vocab;
    int tmax = tokenbin_max(&train) + 1;
    if (tmax > V) V = tmax;

    /* how many GPUs to actually use */
    int ngpu = 0; cudaGetDeviceCount(&ngpu);
    int world = (cfg.n_gpus >= 2 && ngpu >= 2 && !cfg.bench) ? 2 : 1;
    if (cfg.n_gpus >= 2 && ngpu < 2)
        fprintf(stderr, "  note: --n-gpus 2 but only %d GPU visible; running single-GPU.\n", ngpu);

    printf("\033[1;35m  v3: V=%d ctx=%d code_dim=%d layers=%d n_gran=%d n_classes=%d S=%d K=%d "
           "batch=%d gpus=%d\033[0m\n",
           V, cfg.ctx, cfg.code_dim, cfg.n_layers, cfg.n_gran, cfg.n_classes,
           (V + cfg.n_classes - 1) / cfg.n_classes, cfg.k_active, cfg.batch, world);

    long fn = train.n < cfg.block ? train.n : cfg.block;
    std::vector<long> freq(V, 0);
    for (long i = 0; i < fn; i++) { int t = train.data[i]; if (t < V) freq[t]++; }

    RankCtx base;
    base.cfg = cfg; base.meta = meta;
    base.train_data = train.data; base.train_n = train.n;
    base.test_data = test.data; base.test_n = test.data ? (test.n < meta.test_tokens ? test.n : meta.test_tokens) : 0;
    base.V = V; base.freq = freq.data();

    if (world == 1) {
        base.rank = 0; base.world = 1; base.sh = NULL;
        run_rank(&base);
    } else {
        /* dual-GPU local-SGD: one replica per T4, disjoint shards, periodic average */
        Shared sh; sh.world = 2;
        size_t big = (size_t)cfg.n_gran * (((V + cfg.n_classes - 1) / cfg.n_classes) * cfg.n_classes);
        sh.hbuf[0] = (float *)malloc(big * sizeof(float));
        sh.hbuf[1] = (float *)malloc(big * sizeof(float));
        if (!sh.hbuf[0] || !sh.hbuf[1]) { fprintf(stderr, "host staging alloc failed\n"); return 1; }
        pthread_barrier_init(&sh.bar, NULL, 2);
        RankCtx rc0 = base, rc1 = base;
        rc0.rank = 0; rc0.world = 2; rc0.sh = &sh;
        rc1.rank = 1; rc1.world = 2; rc1.sh = &sh;
        std::thread t1([&]() { run_rank(&rc1); });
        run_rank(&rc0);
        t1.join();
        pthread_barrier_destroy(&sh.bar);
        free(sh.hbuf[0]); free(sh.hbuf[1]);
    }

    tokenbin_close(&train);
    if (test.data) tokenbin_close(&test);
    return 0;
}
