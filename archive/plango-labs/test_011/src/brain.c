/* ─────────────────────────────────────────────────────────────────────────────
 * brain.c — the Cerebellar Sparse Coder, multithreaded. No backprop in here.
 *
 * Stages (each timed and, where it pays, parallelized across cores):
 *   patch_contrast    : normalize a 5x5 patch for shape, not brightness
 *   jacobi            : eigen-decompose the patch covariance (for whitening)
 *   learn_dictionary  : whiten + spherical k-means -> detectors (threaded assign)
 *   extract           : image -> location-specific sparse code (threaded)
 *   train_readout     : one linear layer by the delta rule (threaded mini-batch)
 * ───────────────────────────────────────────────────────────────────────────*/
#include "brain.h"
#include "common.h"
#include "threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PATCH_SAMPLES 50000
#define KMEANS_ITERS  12
#define RD_EPOCHS     16
#define RD_LR0        0.30f
#define RD_BATCH      256       /* small enough for accuracy, big enough for low overhead */
#define ZCA_EPS       0.10

static int grid_pos(int g) { return (g * (IMG_W - BR_PATCH) + (BR_GRID - 1) / 2) / (BR_GRID - 1); }

static void patch_contrast(const float *img, int r, int c, float *out) {
    float mean = 0.0f;
    for (int pr = 0; pr < BR_PATCH; pr++)
        for (int pc = 0; pc < BR_PATCH; pc++)
            out[pr * BR_PATCH + pc] = img[(r + pr) * IMG_W + (c + pc)];
    for (int i = 0; i < BR_PDIM; i++) mean += out[i];
    mean /= BR_PDIM;
    float var = 0.0f;
    for (int i = 0; i < BR_PDIM; i++) { out[i] -= mean; var += out[i] * out[i]; }
    float inv = 1.0f / (sqrtf(var / BR_PDIM) + 1e-4f);
    for (int i = 0; i < BR_PDIM; i++) out[i] *= inv;
}

/* symmetric eigen-decomposition by cyclic Jacobi rotations (small n, exact enough) */
static void jacobi(double *a, int n, double *v, double *d) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) v[i * n + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
        if (off < 1e-18) break;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) {
            double apq = a[p * n + q];
            if (fabs(apq) < 1e-300) continue;
            double app = a[p * n + p], aqq = a[q * n + q];
            double phi = 0.5 * atan2(2.0 * apq, aqq - app);
            double cc = cos(phi), ss = sin(phi);
            for (int i = 0; i < n; i++) {
                double aip = a[i * n + p], aiq = a[i * n + q];
                a[i * n + p] = cc * aip - ss * aiq; a[i * n + q] = ss * aip + cc * aiq;
            }
            for (int i = 0; i < n; i++) {
                double api = a[p * n + i], aqi = a[q * n + i];
                a[p * n + i] = cc * api - ss * aqi; a[q * n + i] = ss * api + cc * aqi;
            }
            for (int i = 0; i < n; i++) {
                double vip = v[i * n + p], viq = v[i * n + q];
                v[i * n + p] = cc * vip - ss * viq; v[i * n + q] = ss * vip + cc * viq;
            }
        }
    }
    for (int i = 0; i < n; i++) d[i] = a[i * n + i];
}

/* ---- threaded k-means assignment (in the reduced BR_PROJ space) ---- */
typedef struct { const float *cw; const float *patches; float *tsum; int *tcnt; } KmJob;
static void km_assign(int begin, int end, int tid, void *arg) {
    KmJob *j = (KmJob *)arg;
    float *sum = j->tsum + (size_t)tid * BR_K * BR_PROJ;
    int   *cnt = j->tcnt + (size_t)tid * BR_K;
    for (int p = begin; p < end; p++) {
        const float *x = j->patches + (size_t)p * BR_PROJ;
        int best = 0; float bd = -1e30f;
        for (int k = 0; k < BR_K; k++) {
            const float *c = j->cw + (size_t)k * BR_PROJ;
            float d = 0.0f;
            for (int i = 0; i < BR_PROJ; i++) d += c[i] * x[i];
            if (d > bd) { bd = d; best = k; }
        }
        float *acc = sum + (size_t)best * BR_PROJ;
        for (int i = 0; i < BR_PROJ; i++) acc[i] += x[i];
        cnt[best]++;
    }
}

static void learn_dictionary(BrainExpert *e, const Dataset *tr, rng_t *r) {
    int P = PATCH_SAMPLES, N = BR_PDIM, T = threads_count();
    float *patch = (float *)malloc((size_t)P * N * sizeof(float));
    for (int p = 0; p < P; p++) {
        int img = rng_int(r, tr->n);
        int rr = rng_int(r, IMG_H - BR_PATCH + 1), cc = rng_int(r, IMG_W - BR_PATCH + 1);
        patch_contrast(tr->X + (size_t)img * IMG_DIM, rr, cc, patch + (size_t)p * N);
    }

    /* mean + covariance, then the whitening matrix M (symmetric) */
    double pmean[BR_PDIM]; for (int i = 0; i < N; i++) pmean[i] = 0.0;
    for (int p = 0; p < P; p++) for (int i = 0; i < N; i++) pmean[i] += patch[(size_t)p * N + i];
    for (int i = 0; i < N; i++) pmean[i] /= P;
    double *cov = (double *)calloc((size_t)N * N, sizeof(double));
    for (int p = 0; p < P; p++) {
        const float *x = patch + (size_t)p * N;
        for (int i = 0; i < N; i++) { double xi = x[i] - pmean[i];
            for (int j = 0; j < N; j++) cov[i * N + j] += xi * (x[j] - pmean[j]); }
    }
    for (int i = 0; i < N * N; i++) cov[i] /= P;
    double Vv[BR_PDIM * BR_PDIM], dv[BR_PDIM];
    jacobi(cov, N, Vv, dv);

    /* keep the BR_PROJ directions with the largest variance (rest is noise) */
    int order[BR_PDIM]; for (int i = 0; i < N; i++) order[i] = i;
    for (int a = 0; a < BR_PROJ; a++) {                  /* partial selection sort, descending */
        int best = a; for (int b = a + 1; b < N; b++) if (dv[order[b]] > dv[order[best]]) best = b;
        int t = order[a]; order[a] = order[best]; order[best] = t;
    }
    /* projection row p = whitened top-p eigenvector; store proj + patch mean */
    e->proj  = (float *)malloc((size_t)BR_PROJ * N * sizeof(float));
    e->pmean = (float *)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) e->pmean[i] = (float)pmean[i];
    for (int p = 0; p < BR_PROJ; p++) {
        int k = order[p]; double s = 1.0 / sqrt(dv[k] + ZCA_EPS);
        for (int i = 0; i < N; i++) e->proj[(size_t)p * N + i] = (float)(Vv[i * N + k] * s);
    }

    /* project every sampled patch into the reduced BR_PROJ space */
    float *pp = (float *)malloc((size_t)P * BR_PROJ * sizeof(float));
    for (int q = 0; q < P; q++) {
        const float *x = patch + (size_t)q * N; float *o = pp + (size_t)q * BR_PROJ;
        for (int p = 0; p < BR_PROJ; p++) { double s = 0.0; const float *pr = e->proj + (size_t)p * N;
            for (int i = 0; i < N; i++) s += pr[i] * (x[i] - e->pmean[i]); o[p] = (float)s; }
    }

    /* spherical k-means on the reduced patches (threaded assignment) */
    float *cw  = (float *)malloc((size_t)BR_K * BR_PROJ * sizeof(float));
    for (int k = 0; k < BR_K; k++)
        memcpy(cw + (size_t)k * BR_PROJ, pp + (size_t)rng_int(r, P) * BR_PROJ, BR_PROJ * sizeof(float));
    float *tsum = (float *)malloc((size_t)T * BR_K * BR_PROJ * sizeof(float));
    int   *tcnt = (int *)malloc((size_t)T * BR_K * sizeof(int));
    float *sum  = (float *)malloc((size_t)BR_K * BR_PROJ * sizeof(float));
    int   *cnt  = (int *)malloc((size_t)BR_K * sizeof(int));

    for (int it = 0; it < KMEANS_ITERS; it++) {
        memset(tsum, 0, (size_t)T * BR_K * BR_PROJ * sizeof(float));
        memset(tcnt, 0, (size_t)T * BR_K * sizeof(int));
        KmJob job = { cw, pp, tsum, tcnt };
        parallel_for(P, km_assign, &job);
        memset(sum, 0, (size_t)BR_K * BR_PROJ * sizeof(float));
        memset(cnt, 0, (size_t)BR_K * sizeof(int));
        for (int t = 0; t < T; t++)
            for (int k = 0; k < BR_K; k++) {
                cnt[k] += tcnt[(size_t)t * BR_K + k];
                const float *ts = tsum + ((size_t)t * BR_K + k) * BR_PROJ;
                float *ac = sum + (size_t)k * BR_PROJ;
                for (int i = 0; i < BR_PROJ; i++) ac[i] += ts[i];
            }
        for (int k = 0; k < BR_K; k++) {
            float *c = cw + (size_t)k * BR_PROJ;
            if (cnt[k] == 0) { memcpy(c, pp + (size_t)rng_int(r, P) * BR_PROJ, BR_PROJ * sizeof(float)); continue; }
            const float *ac = sum + (size_t)k * BR_PROJ; float nrm = 0.0f;
            for (int i = 0; i < BR_PROJ; i++) { c[i] = ac[i]; nrm += c[i] * c[i]; }
            float inv = 1.0f / (sqrtf(nrm) + 1e-8f);
            for (int i = 0; i < BR_PROJ; i++) c[i] *= inv;
        }
    }
    e->dict = cw;                                        /* detectors live in BR_PROJ space */
    free(patch); free(cov); free(pp); free(tsum); free(tcnt); free(sum); free(cnt);
}

/* ---- image -> location-specific sparse code (project to 12-D, then detect) ---- */
static void extract(const BrainExpert *e, const float *img, float *feat) {
    float patch[BR_PDIM], red[BR_PROJ], sims[BR_K];
    for (int gr = 0; gr < BR_GRID; gr++) {
        for (int gc = 0; gc < BR_GRID; gc++) {
            patch_contrast(img, grid_pos(gr), grid_pos(gc), patch);
            /* project the 36-D patch down to 12 whitened dims (shared by all detectors) */
            for (int p = 0; p < BR_PROJ; p++) { float s = 0.0f; const float *pr = e->proj + (size_t)p * BR_PDIM;
                for (int i = 0; i < BR_PDIM; i++) s += pr[i] * (patch[i] - e->pmean[i]); red[p] = s; }
            float mean = 0.0f;
            for (int k = 0; k < BR_K; k++) {
                const float *c = e->dict + (size_t)k * BR_PROJ;
                float d = 0.0f;
                for (int i = 0; i < BR_PROJ; i++) d += c[i] * red[i];
                sims[k] = d; mean += d;
            }
            mean /= BR_K;
            float *bin = feat + (size_t)(gr * BR_GRID + gc) * BR_K;
            for (int k = 0; k < BR_K; k++) { float a = sims[k] - mean; bin[k] = a > 0.0f ? a : 0.0f; }
        }
    }
}

typedef struct { const BrainExpert *e; const float *X; float *F; } FeatJob;
static void feat_range(int begin, int end, int tid, void *arg) {
    (void)tid; FeatJob *j = (FeatJob *)arg;
    for (int s = begin; s < end; s++)
        extract(j->e, j->X + (size_t)s * IMG_DIM, j->F + (size_t)s * BR_FEAT);
}

/* ---- training-time data augmentation (free at inference) ----
 * type 1 = horizontal flip, type 2 = small random shift. Seeing flipped/shifted
 * versions teaches the readout to generalize, which is worth a couple of points
 * and costs nothing when classifying. */
#define BR_AUG 1
static void augment(const float *src, float *dst, int type, int s) {
    if (type == 1) {                                  /* horizontal flip */
        for (int r = 0; r < IMG_H; r++)
            for (int c = 0; c < IMG_W; c++) dst[r * IMG_W + c] = src[r * IMG_W + (IMG_W - 1 - c)];
    } else {                                          /* random small shift */
        rng_t rr; rng_seed(&rr, 12345u + (uint64_t)s);
        int dx = rng_int(&rr, 7) - 3, dy = rng_int(&rr, 7) - 3;
        for (int r = 0; r < IMG_H; r++)
            for (int c = 0; c < IMG_W; c++) {
                int sr = r - dy, sc = c - dx;
                dst[r * IMG_W + c] = (sr >= 0 && sr < IMG_H && sc >= 0 && sc < IMG_W) ? src[sr * IMG_W + sc] : 0.0f;
            }
    }
}
typedef struct { const BrainExpert *e; const float *X; float *F; int aug; } AugJob;
static void aug_feat_range(int begin, int end, int tid, void *arg) {
    (void)tid; AugJob *j = (AugJob *)arg;
    float img[IMG_DIM];
    for (int s = begin; s < end; s++) {
        const float *src = j->X + (size_t)s * IMG_DIM, *use = src;
        if (j->aug != 0) { augment(src, img, j->aug, s); use = img; }
        extract(j->e, use, j->F + (size_t)s * BR_FEAT);
    }
}

/* ---- threaded mini-batch readout (the only learning, by the delta rule). ────
 * Each worker accumulates the gradient for its slice of the batch into its OWN
 * buffer (no locks), then the main thread sums the buffers and takes one step.
 * A BIG batch (512) means few reductions, so the per-batch threading overhead is
 * tiny — that is the difference between this being fast and being slow. */
typedef struct {
    const float *F; const uint8_t *y; const int *order; int b0;
    const float *W; const float *b; float *tgW; float *tgb;
} RdJob;
static void rd_grad(int begin, int end, int tid, void *arg) {
    RdJob *j = (RdJob *)arg;
    float *gW = j->tgW + (size_t)tid * NCLASS * BR_FEAT;
    float *gb = j->tgb + (size_t)tid * NCLASS;
    float p[NCLASS];
    for (int s = begin; s < end; s++) {
        const float *f = j->F + (size_t)j->order[j->b0 + s] * BR_FEAT;
        int yy = j->y[j->order[j->b0 + s]];
        for (int c = 0; c < NCLASS; c++) {
            const float *w = j->W + (size_t)c * BR_FEAT; float a = j->b[c];
            for (int k = 0; k < BR_FEAT; k++) a += w[k] * f[k];
            p[c] = a;
        }
        softmax(p, NCLASS);
        for (int c = 0; c < NCLASS; c++) {
            float g = p[c] - (c == yy ? 1.0f : 0.0f);
            float *gw = gW + (size_t)c * BR_FEAT;
            for (int k = 0; k < BR_FEAT; k++) gw[k] += g * f[k];
            gb[c] += g;
        }
    }
}

/* parallel reduction of the T per-thread gradient copies into the weights */
typedef struct { float *W; const float *tg; int T, sz; float step; } RdApply;
static void rd_apply(int begin, int end, int tid, void *arg) {
    (void)tid; RdApply *j = (RdApply *)arg;
    for (int i = begin; i < end; i++) {
        float s = 0.0f;
        for (int t = 0; t < j->T; t++) s += j->tg[(size_t)t * j->sz + i];
        j->W[i] -= j->step * s;
    }
}

static void train_readout(BrainExpert *e, const float *F, const uint8_t *y, int n, rng_t *r) {
    int T = threads_count();
    e->W = (float *)calloc((size_t)NCLASS * BR_FEAT, sizeof(float));
    e->b = (float *)calloc(NCLASS, sizeof(float));
    float *tgW = (float *)malloc((size_t)T * NCLASS * BR_FEAT * sizeof(float));
    float *tgb = (float *)malloc((size_t)T * NCLASS * sizeof(float));
    int *order = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;

    for (int ep = 0; ep < RD_EPOCHS; ep++) {
        float lr = RD_LR0 / (1.0f + 0.05f * ep);
        for (int i = n - 1; i > 0; i--) { int jx = rng_int(r, i + 1); int t = order[i]; order[i] = order[jx]; order[jx] = t; }
        for (int b0 = 0; b0 < n; b0 += RD_BATCH) {
            int bs = (b0 + RD_BATCH <= n) ? RD_BATCH : (n - b0);
            memset(tgW, 0, (size_t)T * NCLASS * BR_FEAT * sizeof(float));
            memset(tgb, 0, (size_t)T * NCLASS * sizeof(float));
            RdJob job = { F, y, order, b0, e->W, e->b, tgW, tgb };
            parallel_for(bs, rd_grad, &job);
            float step = lr / bs;
            RdApply aj = { e->W, tgW, T, NCLASS * BR_FEAT, step };
            parallel_for(NCLASS * BR_FEAT, rd_apply, &aj);     /* parallel reduce */
            for (int c = 0; c < NCLASS; c++) {                 /* biases (tiny) serial */
                float gb = 0.0f;
                for (int t = 0; t < T; t++) gb += tgb[(size_t)t * NCLASS + c];
                e->b[c] -= step * gb;
            }
        }
    }
    free(tgW); free(tgb); free(order);
}

void brain_train(BrainExpert *e, const Dataset *tr, uint64_t seed, BrainTiming *tm) {
    rng_t r; rng_seed(&r, seed);
    memset(e, 0, sizeof(*e));
    double t0;

    t0 = now_sec();
    learn_dictionary(e, tr, &r);
    if (tm) tm->dict_s = now_sec() - t0;
    printf("dict %.2fs | ", now_sec() - t0); fflush(stdout);   /* live progress */

    t0 = now_sec();
    long NA = (long)tr->n * BR_AUG;                          /* original + augmented copies */
    float *F = (float *)malloc((size_t)NA * BR_FEAT * sizeof(float));
    uint8_t *yA = (uint8_t *)malloc((size_t)NA);
    for (int a = 0; a < BR_AUG; a++) {
        AugJob aj = { e, tr->X, F + (size_t)a * tr->n * BR_FEAT, a };
        parallel_for(tr->n, aug_feat_range, &aj);
        for (int s = 0; s < tr->n; s++) yA[(size_t)a * tr->n + s] = tr->y[s];
    }

    /* standardize each feature over all (augmented) codes; remember the stats */
    e->fmean = (float *)calloc(BR_FEAT, sizeof(float));
    e->fstd  = (float *)malloc(BR_FEAT * sizeof(float));
    for (int j = 0; j < BR_FEAT; j++) {
        double m = 0.0; for (long s = 0; s < NA; s++) m += F[(size_t)s * BR_FEAT + j]; m /= NA;
        double v = 0.0; for (long s = 0; s < NA; s++) { double d = F[(size_t)s * BR_FEAT + j] - m; v += d * d; }
        e->fmean[j] = (float)m; e->fstd[j] = (float)sqrt(v / NA) + 1e-6f;
    }
    for (long s = 0; s < NA; s++) { float *f = F + (size_t)s * BR_FEAT;
        for (int j = 0; j < BR_FEAT; j++) f[j] = (f[j] - e->fmean[j]) / e->fstd[j]; }
    if (tm) tm->feat_s = now_sec() - t0;
    printf("features %.2fs | ", now_sec() - t0); fflush(stdout);

    t0 = now_sec();
    train_readout(e, F, yA, (int)NA, &r);                    /* learn on the augmented set */
    if (tm) tm->readout_s = now_sec() - t0;
    printf("readout %.2fs", now_sec() - t0); fflush(stdout);

    /* hippocampus from the ORIGINAL (un-augmented) codes — rows 0..n-1 */
    mem_init(&e->hippo, 5000, BR_FEAT);
    { rng_t hr; rng_seed(&hr, seed * 2654435761u + 1);
      for (int i = 0; i < e->hippo.cap; i++) { int ix = rng_int(&hr, tr->n);
          mem_add(&e->hippo, F + (size_t)ix * BR_FEAT, tr->y[ix]); } }
    free(F); free(yA);
}

void brain_predict(const BrainExpert *e, const float *x, float *out) {
    float f[BR_FEAT];
    extract(e, x, f);
    for (int j = 0; j < BR_FEAT; j++) f[j] = (f[j] - e->fmean[j]) / e->fstd[j];
    for (int c = 0; c < NCLASS; c++) {
        const float *w = e->W + (size_t)c * BR_FEAT; float a = e->b[c];
        for (int j = 0; j < BR_FEAT; j++) a += w[j] * f[j];
        out[c] = a;
    }
    softmax(out, NCLASS);
}

void brain_encode(const BrainExpert *e, const float *x, float *code) {
    extract(e, x, code);
    for (int j = 0; j < BR_FEAT; j++) code[j] = (code[j] - e->fmean[j]) / e->fstd[j];
}

int brain_readout(const float *W, const float *b, const float *code) {
    float p[NCLASS];
    for (int c = 0; c < NCLASS; c++) {
        const float *w = W + (size_t)c * BR_FEAT; float a = b[c];
        for (int j = 0; j < BR_FEAT; j++) a += w[j] * code[j];
        p[c] = a;
    }
    return argmax(p, NCLASS);
}

double brain_accuracy(const BrainExpert *e, const Dataset *te) {
    float p[NCLASS]; int correct = 0;
    for (int s = 0; s < te->n; s++) {
        brain_predict(e, te->X + (size_t)s * IMG_DIM, p);
        if (argmax(p, NCLASS) == te->y[s]) correct++;
    }
    return (double)correct / te->n;
}

void brain_feature_stats(const BrainExpert *e, const Dataset *te,
                         double *avg_density, int *dead_protos) {
    float feat[BR_FEAT];
    double *protoact = (double *)calloc(BR_K, sizeof(double));
    long nonzero = 0, total = 0;
    for (int s = 0; s < te->n; s++) {
        extract(e, te->X + (size_t)s * IMG_DIM, feat);          /* raw sparse code */
        for (int loc = 0; loc < BR_NLOC; loc++)
            for (int k = 0; k < BR_K; k++) {
                float v = feat[(size_t)loc * BR_K + k];
                if (v > 0.0f) nonzero++;
                protoact[k] += v;
            }
        total += BR_FEAT;
    }
    *avg_density = (double)nonzero / (double)total;
    int dead = 0;
    for (int k = 0; k < BR_K; k++) if (protoact[k] < 1e-6) dead++;
    *dead_protos = dead;
    free(protoact);
}

long brain_ops_per_image(void) {
    /* per location: project (BR_PROJ x BR_PDIM) once, then BR_K detectors in 12-D */
    long feat = (long)BR_NLOC * (BR_PROJ * BR_PDIM + BR_K * BR_PROJ);
    return feat + (long)BR_FEAT * NCLASS;
}

void brain_free(BrainExpert *e) {
    free(e->proj); free(e->pmean); free(e->dict);
    free(e->fmean); free(e->fstd); free(e->W); free(e->b);
    mem_free(&e->hippo);
    memset(e, 0, sizeof(*e));
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  THE BRAIN DOING WHAT BRAINS DO — memory + sleep, now part of the brain.
 *  (These used to be a separate experiments.c; they belong here.)
 * ═══════════════════════════════════════════════════════════════════════════*/

/* one night of replay: a single SGD pass over stored codes, optionally dreaming
 * (random feature dropout) so the cortex generalizes rather than memorizes */
static void night(float *W, float *b, const float *codes, const uint8_t *y,
                  int n, float lr, float dream, rng_t *r) {
    int *order = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = n - 1; i > 0; i--) { int j = rng_int(r, i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
    float in[BR_FEAT], p[NCLASS];
    float scale = dream > 0 ? 1.0f / (1.0f - dream) : 1.0f;
    for (int s = 0; s < n; s++) {
        const float *c = codes + (size_t)order[s] * BR_FEAT;
        if (dream > 0) { for (int k = 0; k < BR_FEAT; k++) in[k] = (rng_unif(r) < dream) ? 0.0f : c[k] * scale; }
        else memcpy(in, c, BR_FEAT * sizeof(float));
        for (int cl = 0; cl < NCLASS; cl++) {
            const float *w = W + (size_t)cl * BR_FEAT; float a = b[cl];
            for (int k = 0; k < BR_FEAT; k++) a += w[k] * in[k];
            p[cl] = a;
        }
        softmax(p, NCLASS);
        int yy = y[order[s]];
        for (int cl = 0; cl < NCLASS; cl++) {
            float g = (p[cl] - (cl == yy ? 1.0f : 0.0f)) * lr;
            float *w = W + (size_t)cl * BR_FEAT;
            for (int k = 0; k < BR_FEAT; k++) w[k] -= g * in[k];
            b[cl] -= g;
        }
    }
    free(order);
}

static double eval_codes(const float *W, const float *b, const float *codes, const uint8_t *y, int n) {
    int correct = 0;
    for (int s = 0; s < n; s++) if (brain_readout(W, b, codes + (size_t)s * BR_FEAT) == y[s]) correct++;
    return (double)correct / n;
}

void brain_oneshot(const BrainExpert *e, const Dataset *tr, const Dataset *te) {
    const int NOVEL[2] = {8, 9};
    int *pool[2], poolN[2] = {0, 0}, *tst, tstN = 0;
    for (int c = 0; c < 2; c++) pool[c] = (int *)malloc((size_t)tr->n * sizeof(int));
    tst = (int *)malloc((size_t)te->n * sizeof(int));
    for (int s = 0; s < tr->n; s++) for (int c = 0; c < 2; c++) if (tr->y[s] == NOVEL[c]) pool[c][poolN[c]++] = s;
    for (int s = 0; s < te->n; s++) if (te->y[s] == NOVEL[0] || te->y[s] == NOVEL[1]) tst[tstN++] = s;

    printf("  ONE-SHOT — learn 'bag' vs 'boot', never trained on (a net can't, 50%% guess):\n");
    printf("    %-16s %s\n", "examples shown", "brain memory");
    float *ncodes = (float *)malloc((size_t)tstN * BR_FEAT * sizeof(float));
    for (int j = 0; j < tstN; j++) brain_encode(e, te->X + (size_t)tst[j] * IMG_DIM, ncodes + (size_t)j * BR_FEAT);

    Memory mem; mem_init(&mem, 64, BR_FEAT);
    float code[BR_FEAT], probs[NCLASS];
    int shots[4] = {1, 3, 5, 10};
    for (int si = 0; si < 4; si++) {
        int K = shots[si]; double acc_sum = 0.0; int trials = 25;
        for (int t = 0; t < trials; t++) {
            rng_t r; rng_seed(&r, 7000u + (uint64_t)K * 131 + t);
            mem_clear(&mem);
            for (int c = 0; c < 2; c++) for (int k = 0; k < K; k++) {
                int idx = pool[c][rng_int(&r, poolN[c])];
                brain_encode(e, tr->X + (size_t)idx * IMG_DIM, code);
                mem_add(&mem, code, NOVEL[c]);
            }
            int correct = 0;
            for (int j = 0; j < tstN; j++) {
                mem_knn(&mem, ncodes + (size_t)j * BR_FEAT, K < 3 ? 1 : 3, probs);
                if (argmax(probs, NCLASS) == te->y[tst[j]]) correct++;
            }
            acc_sum += (double)correct / tstN;
        }
        char lbl[32]; snprintf(lbl, sizeof(lbl), "%d each%s", K, K == 1 ? " (one-shot)" : "");
        printf("    %-16s %.1f%%\n", lbl, 100.0 * acc_sum / trials);
    }
    mem_free(&mem); free(ncodes); for (int c = 0; c < 2; c++) free(pool[c]); free(tst);
    printf("\n");
}

typedef struct { const BrainExpert *e; const Memory *mem; const float *tcodes;
                 int knn; float alpha; int *cpred; int *hpred; int *fpred; } FuseJob;
static void fuse_range(int begin, int end, int tid, void *arg) {
    (void)tid; FuseJob *j = (FuseJob *)arg;
    float cp[NCLASS], hp[NCLASS];
    for (int s = begin; s < end; s++) {
        const float *code = j->tcodes + (size_t)s * BR_FEAT;
        for (int c = 0; c < NCLASS; c++) {
            const float *w = j->e->W + (size_t)c * BR_FEAT; float a = j->e->b[c];
            for (int k = 0; k < BR_FEAT; k++) a += w[k] * code[k];
            cp[c] = a;
        }
        softmax(cp, NCLASS);
        mem_knn(j->mem, code, j->knn, hp);
        int bi = 0; float best = -1e30f;
        for (int c = 0; c < NCLASS; c++) { float f = j->alpha * cp[c] + (1 - j->alpha) * hp[c]; if (f > best) { best = f; bi = c; } }
        j->cpred[s] = argmax(cp, NCLASS); j->hpred[s] = argmax(hp, NCLASS); j->fpred[s] = bi;
    }
}

double brain_fusion(const BrainExpert *e, const Dataset *te, long cortex_ops) {
    float *tcodes = (float *)malloc((size_t)te->n * BR_FEAT * sizeof(float));
    for (int s = 0; s < te->n; s++) brain_encode(e, te->X + (size_t)s * IMG_DIM, tcodes + (size_t)s * BR_FEAT);
    int *cpred = malloc((size_t)te->n * sizeof(int));
    int *hpred = malloc((size_t)te->n * sizeof(int));
    int *fpred = malloc((size_t)te->n * sizeof(int));
    FuseJob job = { e, &e->hippo, tcodes, 15, 0.55f, cpred, hpred, fpred };   /* uses the brain's OWN hippocampus */
    parallel_for(te->n, fuse_range, &job);

    int cc = 0, hc = 0, fc = 0;
    for (int s = 0; s < te->n; s++) { cc += cpred[s] == te->y[s]; hc += hpred[s] == te->y[s]; fc += fpred[s] == te->y[s]; }
    double ca = (double)cc / te->n, ha = (double)hc / te->n, fa = (double)fc / te->n;
    long hippo_ops = (long)e->hippo.n * BR_FEAT;
    printf("  CORTEX + HIPPOCAMPUS (complementary learning systems, integrated):\n");
    printf("    %-26s %-9s %s\n", "system", "accuracy", "MACs / image");
    printf("    %-26s %7.2f%%   %ld\n", "cortex alone (rules)", ca * 100, cortex_ops);
    printf("    %-26s %7.2f%%   %ld   (kNN over %d memories)\n", "hippocampus alone (recall)", ha * 100, hippo_ops, e->hippo.n);
    printf("    %-26s %7.2f%%   %ld\n", "FUSED (cortex+memory)", fa * 100, cortex_ops + hippo_ops);
    printf("    note: recall costs ~%ldx the cortex — the honest price of episodic memory.\n", hippo_ops / cortex_ops);
    free(tcodes); free(cpred); free(hpred); free(fpred);
    return fa;
}

void brain_sleep(const BrainExpert *e, const Dataset *tr, const Dataset *te, double full_acc) {
    const int M = 4000, CYCLES = 5, NIGHTS = 6;
    const float DREAM = 0.5f;
    float *tcodes = (float *)malloc((size_t)te->n * BR_FEAT * sizeof(float));
    for (int s = 0; s < te->n; s++) brain_encode(e, te->X + (size_t)s * IMG_DIM, tcodes + (size_t)s * BR_FEAT);
    rng_t r; rng_seed(&r, 2024);
    float *mcodes = (float *)malloc((size_t)M * BR_FEAT * sizeof(float));
    uint8_t *mlab = (uint8_t *)malloc((size_t)M);
    float *W = (float *)calloc((size_t)NCLASS * BR_FEAT, sizeof(float));
    float *b = (float *)calloc(NCLASS, sizeof(float));

    printf("  WAKE-SLEEP — each 'day' grab %d fresh memories, sleep on them. Over and over:\n", M);
    for (int i = 0; i < M; i++) { int ix = rng_int(&r, tr->n);
        brain_encode(e, tr->X + (size_t)ix * IMG_DIM, mcodes + (size_t)i * BR_FEAT); mlab[i] = tr->y[ix]; }
    for (int it = 0; it < CYCLES * NIGHTS; it++) night(W, b, mcodes, mlab, M, 0.30f / (1 + 0.04f * it), 0.0f, &r);
    double rote = eval_codes(W, b, tcodes, te->y, te->n);

    memset(W, 0, (size_t)NCLASS * BR_FEAT * sizeof(float)); memset(b, 0, (size_t)NCLASS * sizeof(float));
    int step = 0; double slept = 0;
    for (int cyc = 0; cyc < CYCLES; cyc++) {
        for (int i = 0; i < M; i++) { int ix = rng_int(&r, tr->n);
            brain_encode(e, tr->X + (size_t)ix * IMG_DIM, mcodes + (size_t)i * BR_FEAT); mlab[i] = tr->y[ix]; }
        for (int nn = 0; nn < NIGHTS; nn++) { night(W, b, mcodes, mlab, M, 0.30f / (1 + 0.04f * step), DREAM, &r); step++; }
        slept = eval_codes(W, b, tcodes, te->y, te->n);
        printf("    day %d/%d (slept %d nights on fresh memories):  cortex %.2f%%\n", cyc + 1, CYCLES, NIGHTS, slept * 100);
        fflush(stdout);
    }
    printf("    rote replay %.2f%%  vs  wake-sleep+dream %.2f%%   (all-data cortex: %.2f%%)\n",
           rote * 100, slept * 100, full_acc * 100);
    printf("    => sleeping on fresh memories %s rote by %.2f points.\n\n",
           slept > rote ? "BEAT" : "trailed", (slept - rote) * 100);
    free(tcodes); free(mcodes); free(mlab); free(W); free(b);
}
