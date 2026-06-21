/* ─────────────────────────────────────────────────────────────────────────────
 * brain.cpp — the cerebellar n-gram coder, multithreaded. No backprop in here.
 *
 * The front-end (word codes -> granule expansion -> kWTA sparse code) is FIXED
 * random, so we encode every training context exactly once and then sweep those
 * stored sparse codes while only the readout learns — the same "encode once,
 * train the readout many times" trick test_011 used for images.
 *
 * The readout update is the interesting part. The weight matrix Wr is big
 * (BR_G x vocab), so we never make per-thread copies of it. Instead each example
 * only touches BR_K granule rows, so per mini-batch we bucket the touched
 * (granule -> examples) pairs and parallelize the update OVER GRANULE ROWS: each
 * granule row is written by exactly one thread, so there are no locks and no
 * race conditions, and the writes stay contiguous (cache-friendly).
 * ───────────────────────────────────────────────────────────────────────────*/
#include "brain.h"
#include "common.h"
#include "threads.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>

#define BATCH       1024
#define COOC_WIN    4          /* co-occurrence window (+/- words) for word codes */
#define LIVE_EVAL   6000       /* per-epoch eval slice (full eval runs at the end) */

/* granule activations for one context: dot of fixed random wiring with the
 * concatenated word codes, minus the homeostatic boost (intrinsic plasticity). */
static void granule_act(const Brain *b, const int *ctx, float *RESTRICT a) {
    float cvec[BR_IN];
    for (int t = 0; t < CTX; t++)
        memcpy(cvec + t * BR_EMB, b->Eb + (size_t)ctx[t] * BR_EMB, BR_EMB * sizeof(float));
    for (int g = 0; g < BR_G; g++) {
        const int *RESTRICT gi = b->gidx + (size_t)g * BR_SIN;
        const float *RESTRICT gw = b->gwt + (size_t)g * BR_SIN;
        float s = 0.0f;
        for (int k = 0; k < BR_SIN; k++) s += gw[k] * cvec[gi[k]];
        a[g] = s - b->gbias[g];
    }
}

/* ── DISTRIBUTIONAL WORD CODES (no backprop) ──────────────────────────────────
 * The distributional hypothesis, done the brain's way: words that occur in
 * similar contexts should get similar vectors. We count co-occurrence within a
 * window, weight by PPMI (positive pointwise mutual information — a Hebbian
 * correlation measure), and random-project that high-dim context profile down to
 * BR_EMB dims (Johnson-Lindenstrauss / dentate-style sparse projection). The
 * result: "king" and "queen" land near each other, so the granule layer and the
 * readout can finally SHARE structure across related words. */
static void learn_word_codes(Brain *b, const Corpus *co, rng_t *r) {
    int V = b->vocab; const int *tr = co->train; int n = co->ntrain;

#if BR_RANDOM_CODES
    /* FIXED RANDOM codes (the Run-4 winner): each word gets an i.i.d. gaussian
     * vector — a dentate-gyrus-style random projection. Orthogonal in expectation,
     * so distinct words stay maximally separable (the discriminability this
     * memorization task rewards). Sharing is delegated to the hippocampal cache. */
    (void)tr; (void)n;
    b->Eb = (float *)malloc((size_t)V * BR_EMB * sizeof(float));
    for (long i = 0; i < (long)V * BR_EMB; i++) b->Eb[i] = (float)rng_gauss(r);
    return;
#endif

    std::vector<long> f(V, 0);
    for (int i = 0; i < n; i++) f[tr[i]]++;
    double total = (double)n;

    /* windowed co-occurrence counts (symmetric) */
    std::unordered_map<uint64_t, int> cc;
    cc.reserve((size_t)n * 2);
    for (int i = 0; i < n; i++)
        for (int d = 1; d <= COOC_WIN; d++) {
            if (i + d >= n) break;
            int a = tr[i], c = tr[i + d];
            cc[(uint64_t)a * V + c]++;
            cc[(uint64_t)c * V + a]++;
        }

    /* a fixed random projection matrix R: BR_EMB x V (column per context word) */
    float *R = (float *)malloc((size_t)BR_EMB * V * sizeof(float));
    for (long i = 0; i < (long)BR_EMB * V; i++) R[i] = (float)rng_gauss(r);

    /* Eb[w] = sum_c PPMI(w,c) * R[:,c] */
    b->Eb = (float *)calloc((size_t)V * BR_EMB, sizeof(float));
    for (auto &kv : cc) {
        int w = (int)(kv.first / V), c = (int)(kv.first % V);
        double pmi = log((double)kv.second * total / ((double)f[w] * (double)f[c] + 1e-9));
        if (pmi <= 0.0) continue;                       /* positive PMI only */
        float *RESTRICT ew = b->Eb + (size_t)w * BR_EMB; float p = (float)pmi;
        for (int e = 0; e < BR_EMB; e++) ew[e] += p * R[(size_t)e * V + c];
    }
    /* Normalize each word code to a FIXED magnitude. We scale to sqrt(BR_EMB) (the
     * expected norm of the old unit-gaussian random codes), NOT to unit norm:
     * unit-norm codes are ~sqrt(BR_EMB)≈7x smaller, which shrinks the granule
     * activations, the kWTA margins, and therefore the delta-rule step by the same
     * factor — leaving the readout badly undertrained in a fixed epoch budget. By
     * matching the old code magnitude the learning DYNAMICS are unchanged, so any
     * difference is due to the codes' SEMANTICS, not an accidental learning-rate cut.
     * Never-seen words fall back to a random code at the same scale. */
    const float CODE_NORM = sqrtf((float)BR_EMB);
    for (int w = 0; w < V; w++) {
        float *ew = b->Eb + (size_t)w * BR_EMB; float nrm = 0.0f;
        for (int e = 0; e < BR_EMB; e++) nrm += ew[e] * ew[e];
        if (nrm < 1e-12f) { for (int e = 0; e < BR_EMB; e++) ew[e] = (float)rng_gauss(r); }
        else { float inv = CODE_NORM / sqrtf(nrm); for (int e = 0; e < BR_EMB; e++) ew[e] *= inv; }
    }
    free(R);
}

/* ── Competitive granule feature learning (no backprop) ───────────────────────
 * Each granule samples BR_SIN fixed inputs (its wiring stays random). Here we tune
 * its WEIGHTS by online competitive learning: present a context, let the kWTA pick
 * the BR_K winners, and move each winner's weights toward that context (in its
 * sampled dims), then renormalize. Winners that recur on similar contexts converge
 * to prototypes of the data — a self-organized dictionary, like test_011's k-means
 * visual features, but driven by the same WTA the encoder already uses. */
static void learn_granule_features(Brain *b, rng_t *r, const Corpus *co) {
    int navail = co->ntrain - CTX;
    int sample = BR_GFEAT_SAMPLE; if (sample > navail) sample = navail;
    float *a = (float *)malloc((size_t)BR_G * sizeof(float));
    float cvec[BR_IN];
    for (int pass = 0; pass < BR_GFEAT_PASSES; pass++) {
        float eta = BR_GFEAT_ETA * (1.0f - (float)pass / BR_GFEAT_PASSES);  /* anneal */
        for (int s = 0; s < sample; s++) {
            const int *ctx = co->train + rng_int(r, navail);
            for (int t = 0; t < CTX; t++)
                memcpy(cvec + t * BR_EMB, b->Eb + (size_t)ctx[t] * BR_EMB, BR_EMB * sizeof(float));
            for (int g = 0; g < BR_G; g++) {
                const int *RESTRICT gi = b->gidx + (size_t)g * BR_SIN;
                const float *RESTRICT gw = b->gwt + (size_t)g * BR_SIN;
                float acc = 0.0f;
                for (int k = 0; k < BR_SIN; k++) acc += gw[k] * cvec[gi[k]];
                a[g] = acc;
            }
            for (int rk = 0; rk < BR_K; rk++) {        /* nudge the BR_K winners */
                int best = 0; float bv = a[0];
                for (int g = 1; g < BR_G; g++) if (a[g] > bv) { bv = a[g]; best = g; }
                a[best] = -1e30f;
                float *RESTRICT gw = b->gwt + (size_t)best * BR_SIN;
                const int *RESTRICT gi = b->gidx + (size_t)best * BR_SIN;
                float nrm = 0.0f;
                for (int k = 0; k < BR_SIN; k++) { gw[k] += eta * (cvec[gi[k]] - gw[k]); nrm += gw[k] * gw[k]; }
                if (nrm > 1e-12f) {                    /* keep ~sqrt(BR_SIN) scale (Oja-style) */
                    float inv = sqrtf((float)BR_SIN) / sqrtf(nrm);
                    for (int k = 0; k < BR_SIN; k++) gw[k] *= inv;
                }
            }
        }
    }
    free(a);
}

void brain_init(Brain *b, const Corpus *co, uint64_t seed) {
    rng_t r; rng_seed(&r, seed);
    int vocab = co->vocab;
    b->vocab = vocab;
    b->hippo = NULL;

    /* 1. distributional word codes (semantic, no backprop) */
    learn_word_codes(b, co, &r);

    /* 2. fixed sparse granule wiring: each granule samples BR_SIN random inputs */
    b->gidx = (int *)malloc((size_t)BR_G * BR_SIN * sizeof(int));
    b->gwt  = (float *)malloc((size_t)BR_G * BR_SIN * sizeof(float));
    for (long g = 0; g < BR_G; g++)
        for (int s = 0; s < BR_SIN; s++) {
            b->gidx[g * BR_SIN + s] = rng_int(&r, BR_IN);
            b->gwt[g * BR_SIN + s]  = (float)rng_gauss(&r);
        }

    /* 2b. competitive feature learning: tune the granule weights (not the wiring)
     * so the expansion tiles the data manifold. Cost-neutral at inference. */
#if BR_LEARN_GRANULES
    learn_granule_features(b, &r, co);
#endif

    /* 3. homeostatic boosting (intrinsic plasticity): set each granule's bias to
     * its average activation over a sample, so no granule dominates and dead
     * granules come back to life — evens out the sparse code's load. */
    b->gbias = (float *)calloc((size_t)BR_G, sizeof(float));     /* zero during the probe */
    int sample = 4000, navail = co->ntrain - CTX;
    if (sample > navail) sample = navail;
    double *acc = (double *)calloc((size_t)BR_G, sizeof(double));
    float *a = (float *)malloc((size_t)BR_G * sizeof(float));
    for (int s = 0; s < sample; s++) {
        int start = rng_int(&r, navail);
        granule_act(b, co->train + start, a);
        for (int g = 0; g < BR_G; g++) acc[g] += a[g];
    }
    for (int g = 0; g < BR_G; g++) b->gbias[g] = (float)(acc[g] / (sample ? sample : 1));
    free(acc); free(a);

    /* 4. the only thing that learns: a two-level linear readout, started at zero.
     *    Wr = within-class word logits, Wc = class logits. */
    b->Wr = (float *)calloc((size_t)BR_G * vocab, sizeof(float));
    b->br = (float *)calloc((size_t)vocab, sizeof(float));
    b->Wc = (float *)calloc((size_t)BR_G * BR_CLASSES, sizeof(float));
    b->bc = (float *)calloc((size_t)BR_CLASSES, sizeof(float));

    /* 5. word -> class assignment: balanced contiguous frequency-rank bins. Sorting
     * by frequency keeps each class to ~vocab/BR_CLASSES words (so within-class cost
     * is bounded and even) and groups words of similar frequency — function words
     * cluster, rare content words cluster — which keeps the class predictable from
     * context, so the split costs little perplexity. */
    b->w2class  = (int *)malloc((size_t)vocab * sizeof(int));
    b->w2slot   = (int *)malloc((size_t)vocab * sizeof(int));
    b->cls_word = (int *)malloc((size_t)vocab * sizeof(int));
    b->cls_off  = (int *)malloc((size_t)(BR_CLASSES + 1) * sizeof(int));
    {
        long *freq = (long *)calloc((size_t)vocab, sizeof(long));
        for (int i = 0; i < co->ntrain; i++) freq[co->train[i]]++;
        int *rank = (int *)malloc((size_t)vocab * sizeof(int));
        for (int w = 0; w < vocab; w++) rank[w] = w;
        /* sort word ids by descending frequency (simple, vocab is small) */
        std::sort(rank, rank + vocab, [&](int a, int c){ return freq[a] > freq[c]; });
        /* contiguous bins of near-equal size */
        for (int c = 0; c <= BR_CLASSES; c++)
            b->cls_off[c] = (int)((long)c * vocab / BR_CLASSES);
        for (int c = 0; c < BR_CLASSES; c++)
            for (int p = b->cls_off[c]; p < b->cls_off[c + 1]; p++) {
                int w = rank[p];
                b->cls_word[p] = w; b->w2class[w] = c; b->w2slot[w] = p - b->cls_off[c];
            }
        free(freq); free(rank);
    }
}

void brain_encode(const Brain *b, const int *ctx, int *out_idx, float *out_val) {
    float a[BR_G];
    granule_act(b, ctx, a);

    /* kWTA: keep the BR_K strongest granules; one extra max gives the threshold
     * so every kept activation is strictly positive (a clean sparse code). */
    float tmpv[BR_K];
    for (int rk = 0; rk < BR_K; rk++) {
        int best = 0; float bv = a[0];
        for (int g = 1; g < BR_G; g++) if (a[g] > bv) { bv = a[g]; best = g; }
        out_idx[rk] = best; tmpv[rk] = bv; a[best] = -1e30f;
    }
    float thr = a[0];
    for (int g = 1; g < BR_G; g++) if (a[g] > thr) thr = a[g];
    for (int rk = 0; rk < BR_K; rk++) out_val[rk] = tmpv[rk] - thr;   /* > 0 */
}

/* generic sparse readout: logits[0..W) = bias + sum over the BR_K active granules of
 * val * weight[row g], where each granule row has stride `W`. */
static void readout_W(const float *Wt, const float *bias, int W,
                      const int *idx, const float *val, float *RESTRICT logits) {
    memcpy(logits, bias, (size_t)W * sizeof(float));
    for (int k = 0; k < BR_K; k++) {
        const float *RESTRICT row = Wt + (size_t)idx[k] * W; float vv = val[k];
        for (int v = 0; v < W; v++) logits[v] += vv * row[v];
    }
}
static inline void readout(const Brain *b, const int *idx, const float *val, float *RESTRICT logits) {
    readout_W(b->Wr, b->br, b->vocab, idx, val, logits);   /* full flat word logits */
}

/* ── hierarchical inference ───────────────────────────────────────────────────
 * P(word) = P(class) * P(word | class). The cheap path only ever scores BR_CLASSES
 * class logits + the words of ONE class, so it costs BR_K*(BR_CLASSES + |class|)
 * instead of BR_K*VOCAB. */

/* class log-probabilities for a code (length BR_CLASSES, already log-softmaxed) */
static void class_logprobs(const Brain *b, const int *idx, const float *val, float *RESTRICT lp) {
    readout_W(b->Wc, b->bc, BR_CLASSES, idx, val, lp);
    softmax(lp, BR_CLASSES);
    for (int c = 0; c < BR_CLASSES; c++) lp[c] = logf(lp[c] < 1e-30f ? 1e-30f : lp[c]);
}

/* P(word | class c): softmax over the words of class c only. Writes into pw[0..size). */
static void within_class_probs(const Brain *b, const int *idx, const float *val,
                               int c, float *RESTRICT pw) {
    int beg = b->cls_off[c], size = b->cls_off[c + 1] - beg;
    for (int j = 0; j < size; j++) {
        int w = b->cls_word[beg + j]; float s = b->br[w];
        for (int k = 0; k < BR_K; k++) s += val[k] * b->Wr[(size_t)idx[k] * b->vocab + w];
        pw[j] = s;
    }
    softmax(pw, size);
}

/* fill a FULL probability vector over the vocabulary (used by analytics / generation;
 * this is the dense O(VOCAB) path, not the cheap inference cost). */
static void full_probs(const Brain *b, const int *idx, const float *val, float *RESTRICT probs) {
    float clp[BR_CLASSES]; class_logprobs(b, idx, val, clp);
    float pw[VOCAB];
    for (int c = 0; c < BR_CLASSES; c++) {
        int beg = b->cls_off[c], size = b->cls_off[c + 1] - beg;
        within_class_probs(b, idx, val, c, pw);
        float pc = expf(clp[c]);
        for (int j = 0; j < size; j++) probs[b->cls_word[beg + j]] = pc * pw[j];
    }
}

/* perplexity + top-1 over PRE-ENCODED codes (used for the fast per-epoch eval:
 * the front-end is fixed, so the test contexts are encoded once, not every epoch) */
static void eval_codes(const Brain *b, const int *idx, const float *val, const int *tgt,
                       int nex, double *ppl, double *acc) {
    double nll = 0.0; long correct = 0;
    float clp[BR_CLASSES], pw[VOCAB];
    for (int e = 0; e < nex; e++) {
        const int *ix = idx + (size_t)e * BR_K; const float *vl = val + (size_t)e * BR_K;
        int y = tgt[e];
        class_logprobs(b, ix, vl, clp);
        /* perplexity: exact prob of the true word via its own class (cheap) */
        int cy = b->w2class[y];
        within_class_probs(b, ix, vl, cy, pw);
        float py = expf(clp[cy]) * pw[b->w2slot[y]]; if (py < 1e-12f) py = 1e-12f;
        nll += -log((double)py);
        /* top-1: greedy decode — best class, then best word in it (cheap path) */
        int cbest = 0; for (int c = 1; c < BR_CLASSES; c++) if (clp[c] > clp[cbest]) cbest = c;
        if (cbest != cy) within_class_probs(b, ix, vl, cbest, pw);
        int beg = b->cls_off[cbest], size = b->cls_off[cbest + 1] - beg;
        int jbest = 0; for (int j = 1; j < size; j++) if (pw[j] > pw[jbest]) jbest = j;
        if (b->cls_word[beg + jbest] == y) correct++;
    }
    *ppl = exp(nll / nex);
    *acc = (double)correct / nex;
}

/* ---- one-time encoding of every training context (parallel over examples) ---- */
typedef struct { const Brain *b; const int *stream; int *oidx; float *oval; } EncJob;
static void enc_range(int begin, int end, int tid, void *arg) {
    (void)tid; EncJob *j = (EncJob *)arg;
    for (int e = begin; e < end; e++)
        brain_encode(j->b, j->stream + e, j->oidx + (size_t)e * BR_K, j->oval + (size_t)e * BR_K);
}

/* ---- mini-batch forwards: one for the WORD head (within-class softmax) and one
 *      for the CLASS head. Each writes a delta-rule error signal the shared
 *      granule-bucketed updater then applies to Wr/Wc. ---- */
typedef struct {
    const Brain *b; const int *order; int b0; const int *tgt;
    const int *allidx; const float *allval; float *dlogw; float *dlogc;
} FwdJob;

/* WORD head: softmax over the target's class members only; dlogw is zero elsewhere,
 * so the bucketed updater naturally touches only that class's word columns. */
static void fwd_word_range(int begin, int end, int tid, void *arg) {
    (void)tid; FwdJob *j = (FwdJob *)arg; const Brain *b = j->b; int V = b->vocab;
    float pw[VOCAB];
    for (int s = begin; s < end; s++) {
        int e = j->order[j->b0 + s]; int y = j->tgt[e];
        const int *ix = j->allidx + (size_t)e * BR_K; const float *vl = j->allval + (size_t)e * BR_K;
        int c = b->w2class[y], beg = b->cls_off[c], size = b->cls_off[c + 1] - beg;
        within_class_probs(b, ix, vl, c, pw);
        float *dl = j->dlogw + (size_t)s * V;
        memset(dl, 0, (size_t)V * sizeof(float));
        for (int jj = 0; jj < size; jj++) {
            int w = b->cls_word[beg + jj];
            dl[w] = pw[jj] - (w == y ? 1.0f : 0.0f);
        }
    }
}

/* CLASS head: plain softmax over BR_CLASSES, target = the word's class. */
static void fwd_class_range(int begin, int end, int tid, void *arg) {
    (void)tid; FwdJob *j = (FwdJob *)arg; const Brain *b = j->b;
    float logits[BR_CLASSES];
    for (int s = begin; s < end; s++) {
        int e = j->order[j->b0 + s]; int cy = b->w2class[j->tgt[e]];
        readout_W(b->Wc, b->bc, BR_CLASSES, j->allidx + (size_t)e * BR_K, j->allval + (size_t)e * BR_K, logits);
        softmax(logits, BR_CLASSES);
        float *dl = j->dlogc + (size_t)s * BR_CLASSES;
        for (int c = 0; c < BR_CLASSES; c++) dl[c] = logits[c] - (c == cy ? 1.0f : 0.0f);
    }
}

/* ---- race-free readout update, parallelized OVER granule rows ----
 * Each row touched this batch is first shrunk by `decay` (L2 weight decay /
 * synaptic homeostasis), then the delta-rule gradient is subtracted. Rows with no
 * active example this batch are left alone — use-dependent forgetting. */
typedef struct {
    float *Wr; const int *off; const int *sb; const float *sv;
    const float *dlog; int V; float step; float decay;
} UpdJob;
static void upd_range(int begin, int end, int tid, void *arg) {
    (void)tid; UpdJob *j = (UpdJob *)arg; int V = j->V; float keep = 1.0f - j->decay;
    for (int g = begin; g < end; g++) {
        if (j->off[g] == j->off[g + 1]) continue;          /* untouched row */
        float *RESTRICT row = j->Wr + (size_t)g * V;
        if (j->decay > 0.0f) for (int v = 0; v < V; v++) row[v] *= keep;
        for (int e = j->off[g]; e < j->off[g + 1]; e++) {
            float coef = j->step * j->sv[e];
            const float *RESTRICT dl = j->dlog + (size_t)j->sb[e] * V;
            for (int v = 0; v < V; v++) row[v] -= coef * dl[v];
        }
    }
}

typedef struct { float *br; const float *dlog; int bs, V; float step; } BiasJob;
static void bias_range(int begin, int end, int tid, void *arg) {
    (void)tid; BiasJob *j = (BiasJob *)arg;
    for (int v = begin; v < end; v++) {
        float s = 0.0f;
        for (int b = 0; b < j->bs; b++) s += j->dlog[(size_t)b * j->V + v];
        j->br[v] -= j->step * s;
    }
}

void brain_train(Brain *b, const Corpus *co, int epochs, float lr0) {
    int V = b->vocab;
    int nex = corpus_examples(co->ntrain);

    /* targets, then encode every context once (the fixed front-end) */
    int *tgt = (int *)malloc((size_t)nex * sizeof(int));
    for (int e = 0; e < nex; e++) tgt[e] = co->train[e + CTX];
    int   *allidx = (int *)malloc((size_t)nex * BR_K * sizeof(int));
    float *allval = (float *)malloc((size_t)nex * BR_K * sizeof(float));
    double t0 = now_sec();
    EncJob ej = { b, co->train, allidx, allval };
    parallel_for(nex, enc_range, &ej);

    /* encode the TEST contexts once too, so the per-epoch eval is just a cheap
     * readout sweep (the front-end is fixed — no need to re-encode every epoch) */
    int   ntex = corpus_examples(co->ntest);
    int   *teidx = (int *)malloc((size_t)ntex * BR_K * sizeof(int));
    float *teval = (float *)malloc((size_t)ntex * BR_K * sizeof(float));
    int   *tetgt = (int *)malloc((size_t)ntex * sizeof(int));
    for (int e = 0; e < ntex; e++) tetgt[e] = co->test[e + CTX];
    EncJob tej = { b, co->test, teidx, teval };
    parallel_for(ntex, enc_range, &tej);
    printf("    encoded %d train + %d test contexts into sparse codes (%.1fs)\n",
           nex, ntex, now_sec() - t0);
    fflush(stdout);

    /* scratch for the mini-batch update (one error buffer per head) */
    float *dlogw = (float *)malloc((size_t)BATCH * V * sizeof(float));
    float *dlogc = (float *)malloc((size_t)BATCH * BR_CLASSES * sizeof(float));
    int   *order = (int *)malloc((size_t)nex * sizeof(int));
    for (int i = 0; i < nex; i++) order[i] = i;
    int   *cnt = (int *)malloc((size_t)(BR_G + 1) * sizeof(int));
    int   *off = (int *)malloc((size_t)(BR_G + 1) * sizeof(int));
    int   *cur = (int *)malloc((size_t)BR_G * sizeof(int));
    int   *sb  = (int *)malloc((size_t)BATCH * BR_K * sizeof(int));
    float *sv  = (float *)malloc((size_t)BATCH * BR_K * sizeof(float));
    rng_t r; rng_seed(&r, 909090);

    for (int ep = 0; ep < epochs; ep++) {
        float lr = lr0 / (1.0f + 0.05f * ep);
        for (int i = nex - 1; i > 0; i--) { int k = rng_int(&r, i + 1); int t = order[i]; order[i] = order[k]; order[k] = t; }

        for (int b0 = 0; b0 < nex; b0 += BATCH) {
            int bs = (b0 + BATCH <= nex) ? BATCH : (nex - b0);

            /* forward both heads: within-class word errors + class errors */
            FwdJob fj = { b, order, b0, tgt, allidx, allval, dlogw, dlogc };
            parallel_for(bs, fwd_word_range, &fj);
            parallel_for(bs, fwd_class_range, &fj);

            /* bucket the touched (granule -> example) pairs by granule id (shared) */
            memset(cnt, 0, (size_t)(BR_G + 1) * sizeof(int));
            for (int s = 0; s < bs; s++) {
                const int *ix = allidx + (size_t)order[b0 + s] * BR_K;
                for (int k = 0; k < BR_K; k++) cnt[ix[k]]++;
            }
            off[0] = 0;
            for (int g = 0; g < BR_G; g++) { off[g + 1] = off[g] + cnt[g]; cur[g] = off[g]; }
            for (int s = 0; s < bs; s++) {
                const int *ix = allidx + (size_t)order[b0 + s] * BR_K;
                const float *vl = allval + (size_t)order[b0 + s] * BR_K;
                for (int k = 0; k < BR_K; k++) { int g = ix[k]; int p = cur[g]++; sb[p] = s; sv[p] = vl[k]; }
            }

            /* update both heads from the same buckets (one granule row per thread) */
            float step = lr / bs;
            UpdJob uw = { b->Wr, off, sb, sv, dlogw, V, step, BR_WD };
            parallel_for(BR_G, upd_range, &uw);
            BiasJob bw = { b->br, dlogw, bs, V, step };
            parallel_for(V, bias_range, &bw);
            UpdJob uc = { b->Wc, off, sb, sv, dlogc, BR_CLASSES, step, BR_WD };
            parallel_for(BR_G, upd_range, &uc);
            BiasJob bcj = { b->bc, dlogc, bs, BR_CLASSES, step };
            parallel_for(BR_CLASSES, bias_range, &bcj);
        }

        double ppl, acc, trppl, tracc;
        int teprobe = ntex < LIVE_EVAL ? ntex : LIVE_EVAL;
        eval_codes(b, teidx, teval, tetgt, teprobe, &ppl, &acc);
        int probe = nex < LIVE_EVAL ? nex : LIVE_EVAL;  /* train ppl on a slice = overfit probe */
        eval_codes(b, allidx, allval, tgt, probe, &trppl, &tracc);
        printf("    epoch %2d/%2d   test ppl %7.2f   top1 %.2f%%   train ppl %7.2f (gap %+.0f)   (%.1fs)\n",
               ep + 1, epochs, ppl, acc * 100, trppl, ppl - trppl, now_sec() - t0);
        fflush(stdout);
    }
    free(tgt); free(allidx); free(allval); free(dlogw); free(dlogc); free(order);
    free(cnt); free(off); free(cur); free(sb); free(sv);
    free(teidx); free(teval); free(tetgt);
}

/* ── Hippocampal SIMILARITY recall + CLS fusion ───────────────────────────────
 * Run 5 showed that recalling by EXACT context is just a trigram and hurts here.
 * The hippocampus does pattern COMPLETION: recall the episodes whose sparse codes
 * OVERLAP the query's, and vote their next words by overlap. Two contexts that share
 * granules (i.e. are similar in the learned feature space) recall each other, so the
 * memory GENERALIZES instead of memorizing exact continuations. We keep it cheap with
 * an inverted index (granule -> episodes), bounded per granule, so recall touches at
 * most BR_K * HIPPO_CAP episodes — a hash-scale lookup, not a full kNN scan.
 *   probs <- (1-lam)*cortex + lam*recall, lam scaled by how well the best memory
 *   matches the query (perfect match -> HIPPO_LMAX, no match -> 0). */
#define HIPPO_CAP   48        /* max episodes filed under each granule (bounds cost)  */
#define HIPPO_LMAX  0.30f     /* cap on the recall's vote share                       */
#define HIPPO_POW   3.0f      /* sharpen the overlap weighting of the vote            */

struct HippoKNN {
    int  nex;                 /* number of stored training episodes                  */
    int *ep_word;             /* [nex]   next word of each episode                   */
    int *post_off;            /* [BR_G+1] CSR offsets into post_ep                   */
    int *post_ep;             /* episode ids filed under each granule (capped)        */
    /* per-query scratch (mutable cache state; recall is single-threaded) */
    float *score;             /* [nex] overlap score, kept zero between queries       */
    int   *touched;           /* [nex] list of episodes touched this query           */
};

void brain_build_hippocampus(Brain *b, const Corpus *co) {
    int nex = corpus_examples(co->ntrain);
    HippoKNN *h = new HippoKNN();
    h->nex = nex;
    h->ep_word  = (int *)malloc((size_t)nex * sizeof(int));
    h->score    = (float *)calloc((size_t)nex, sizeof(float));
    h->touched  = (int *)malloc((size_t)nex * sizeof(int));
    h->post_off = (int *)malloc((size_t)(BR_G + 1) * sizeof(int));

    /* encode every training context once, recording its code + next word, and count
     * how many episodes fall under each granule (capped at HIPPO_CAP). */
    int *allcode = (int *)malloc((size_t)nex * BR_K * sizeof(int));
    int *cnt = (int *)calloc((size_t)BR_G, sizeof(int));
    int idx[BR_K]; float val[BR_K];
    for (int e = 0; e < nex; e++) {
        brain_encode(b, co->train + e, idx, val);
        int *code = allcode + (size_t)e * BR_K;
        for (int k = 0; k < BR_K; k++) { code[k] = idx[k]; if (cnt[idx[k]] < HIPPO_CAP) cnt[idx[k]]++; }
        h->ep_word[e] = co->train[e + CTX];
    }
    h->post_off[0] = 0;
    for (int g = 0; g < BR_G; g++) h->post_off[g + 1] = h->post_off[g] + cnt[g];
    h->post_ep = (int *)malloc((size_t)h->post_off[BR_G] * sizeof(int));
    int *fill = (int *)calloc((size_t)BR_G, sizeof(int));
    for (int e = 0; e < nex; e++) {
        const int *code = allcode + (size_t)e * BR_K;
        for (int k = 0; k < BR_K; k++) {
            int g = code[k];
            if (fill[g] < HIPPO_CAP) h->post_ep[h->post_off[g] + fill[g]++] = e;
        }
    }
    free(allcode); free(cnt); free(fill);
    b->hippo = h;
}

/* fuse an already-softmaxed cortical distribution with similarity recall over codes:
 * cortex <- (1-lam)*cortex + lam*recall, where recall is the overlap-weighted vote of
 * the matching episodes' next words and lam grows with the best match quality. */
static void cls_fuse_code(const Brain *b, const int *idx, const float *val,
                          float *RESTRICT cortex, float *RESTRICT recall) {
    int V = b->vocab;
    for (int v = 0; v < V; v++) recall[v] = 0.0f;
    if (!b->hippo) return;
    HippoKNN *h = (HippoKNN *)b->hippo;
    int nt = 0; float self = 0.0f;
    for (int k = 0; k < BR_K; k++) {
        int g = idx[k]; float vk = val[k]; self += vk;
        for (int p = h->post_off[g]; p < h->post_off[g + 1]; p++) {
            int e = h->post_ep[p];
            if (h->score[e] == 0.0f) h->touched[nt++] = e;
            h->score[e] += vk;
        }
    }
    if (nt == 0 || self <= 0.0f) return;
    float best = 0.0f, mass = 0.0f;
    for (int i = 0; i < nt; i++) {
        int e = h->touched[i]; float s = h->score[e] / self;
        if (s > best) best = s;
        float w = powf(s, HIPPO_POW);
        recall[h->ep_word[e]] += w; mass += w;
        h->score[e] = 0.0f;
    }
    float lam = HIPPO_LMAX * powf(best, HIPPO_POW);   /* only near-perfect matches earn weight */
    float inv = mass > 0.0f ? 1.0f / mass : 0.0f;
    for (int v = 0; v < V; v++) cortex[v] = (1.0f - lam) * cortex[v] + lam * (recall[v] * inv);
}

void brain_next_cls(const Brain *b, const int *ctx, float *probs) {
    int idx[BR_K]; float val[BR_K]; float recall[VOCAB];
    brain_encode(b, ctx, idx, val);
    readout(b, idx, val, probs);
    softmax(probs, b->vocab);
    cls_fuse_code(b, idx, val, probs, recall);
}

void brain_eval_cls(const Brain *b, const int *stream, int ntok, double *ppl, double *acc) {
    int V = b->vocab, nex = corpus_examples(ntok);
    double nll = 0.0; long correct = 0;
    int idx[BR_K]; float val[BR_K], logits[VOCAB], recall[VOCAB];
    for (int e = 0; e < nex; e++) {
        brain_encode(b, stream + e, idx, val);
        readout(b, idx, val, logits);
        softmax(logits, V);
        cls_fuse_code(b, idx, val, logits, recall);
        int y = stream[e + CTX];
        float py = logits[y]; if (py < 1e-12f) py = 1e-12f;
        nll += -log((double)py);
        if (argmax(logits, V) == y) correct++;
    }
    *ppl = exp(nll / nex);
    *acc = (double)correct / nex;
}

/* headline eval: the cheap HIERARCHICAL path (class -> word). Perplexity is exact;
 * top-1 is the greedy class-then-word decode (what a deployed model would emit). */
void brain_eval(const Brain *b, const int *stream, int ntok, double *ppl, double *acc) {
    int nex = corpus_examples(ntok);
    double nll = 0.0; long correct = 0;
    int idx[BR_K]; float val[BR_K]; float clp[BR_CLASSES], pw[VOCAB];
    for (int e = 0; e < nex; e++) {
        brain_encode(b, stream + e, idx, val);
        int y = stream[e + CTX];
        class_logprobs(b, idx, val, clp);
        int cy = b->w2class[y];
        within_class_probs(b, idx, val, cy, pw);
        float py = expf(clp[cy]) * pw[b->w2slot[y]]; if (py < 1e-12f) py = 1e-12f;
        nll += -log((double)py);
        int cbest = 0; for (int c = 1; c < BR_CLASSES; c++) if (clp[c] > clp[cbest]) cbest = c;
        if (cbest != cy) within_class_probs(b, idx, val, cbest, pw);
        int beg = b->cls_off[cbest], size = b->cls_off[cbest + 1] - beg;
        int jbest = 0; for (int j = 1; j < size; j++) if (pw[j] > pw[jbest]) jbest = j;
        if (b->cls_word[beg + jbest] == y) correct++;
    }
    *ppl = exp(nll / nex);
    *acc = (double)correct / nex;
}

/* full distribution over the vocabulary (analytics / generation; dense O(VOCAB)) */
void brain_next(const Brain *b, const int *ctx, float *probs) {
    int idx[BR_K]; float val[BR_K];
    brain_encode(b, ctx, idx, val);
    full_probs(b, idx, val, probs);
}

/* compare(int) for qsort on granule-usage counts (ascending) */
static int cmp_long(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}

void brain_diagnose(const Brain *b, const int *stream, int ntok, BrainDiag *d) {
    int V = b->vocab, nex = corpus_examples(ntok);
    int sample = nex < 8000 ? nex : 8000;          /* a representative slice */

    /* granule usage + average code margin over the sample */
    long *use = (long *)calloc((size_t)BR_G, sizeof(long));
    double margin_sum = 0.0; long codes = 0;
    int idx[BR_K]; float val[BR_K];
    for (int e = 0; e < sample; e++) {
        brain_encode(b, stream + e, idx, val);
        for (int k = 0; k < BR_K; k++) { use[idx[k]]++; margin_sum += val[k]; codes++; }
    }
    int dead = 0; for (int g = 0; g < BR_G; g++) if (use[g] == 0) dead++;
    d->dead_granules = dead;
    d->density = 1.0 - (double)dead / BR_G;
    d->mean_margin = codes ? margin_sum / codes : 0.0;

    /* Gini of granule usage: 0 = perfectly even, 1 = a few granules do all work */
    qsort(use, BR_G, sizeof(long), cmp_long);
    double cum = 0.0, tot = 0.0;
    for (int g = 0; g < BR_G; g++) tot += (double)use[g];
    if (tot > 0) {
        for (int g = 0; g < BR_G; g++) cum += (double)(2 * (g + 1) - BR_G - 1) * (double)use[g];
        d->usage_gini = cum / ((double)BR_G * tot);
    } else d->usage_gini = 0.0;
    free(use);

    /* readout weight norm + how many vocab rows the readout never learned */
    double wn2 = 0.0; int dead_ro = 0;
    double *colnorm = (double *)calloc((size_t)V, sizeof(double));
    for (long g = 0; g < BR_G; g++) {
        const float *row = b->Wr + (size_t)g * V;
        for (int v = 0; v < V; v++) { double w = row[v]; wn2 += w * w; colnorm[v] += w * w; }
    }
    for (int v = 0; v < V; v++) if (colnorm[v] < 1e-12 && b->br[v] * b->br[v] < 1e-12) dead_ro++;
    d->readout_wnorm = sqrt(wn2);
    d->dead_readout = dead_ro;
    free(colnorm);
}

/* hierarchical inference cost: granule expansion + class scores + the words of one
 * (average-sized) class. The flat readout's BR_K*VOCAB term is replaced by
 * BR_K*(BR_CLASSES + VOCAB/BR_CLASSES). */
long brain_ops_per_pred(void) {
    return (long)BR_G * BR_SIN + (long)BR_K * (BR_CLASSES + (VOCAB + BR_CLASSES - 1) / BR_CLASSES);
}

void brain_free(Brain *b) {
    free(b->Eb); free(b->gidx); free(b->gwt); free(b->gbias); free(b->Wr); free(b->br);
    free(b->Wc); free(b->bc); free(b->w2class); free(b->w2slot); free(b->cls_word); free(b->cls_off);
    if (b->hippo) {
        HippoKNN *h = (HippoKNN *)b->hippo;
        free(h->ep_word); free(h->post_off); free(h->post_ep); free(h->score); free(h->touched);
        delete h;
    }
    memset(b, 0, sizeof(*b));
}
