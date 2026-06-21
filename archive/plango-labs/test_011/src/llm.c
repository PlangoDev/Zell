/* ─────────────────────────────────────────────────────────────────────────────
 * llm.c — dense 2-layer MLP with backpropagation. The reference contestant.
 *
 * MINI-BATCH SGD (batch 100), exactly like the Python reference that scored
 * ~88.4%. The expensive per-sample forward+backward is spread across cores:
 * each worker accumulates gradients into its OWN buffer (no locks), then the
 * main thread sums the buffers and applies one averaged step. Identical maths
 * to the serial version, just parallel — so accuracy is unchanged and it runs
 * in a few seconds instead of minutes.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "llm.h"
#include "common.h"
#include "threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BATCH 100

static float *he_alloc(int out, int in, rng_t *r) {
    float *w = (float *)malloc((size_t)out * in * sizeof(float));
    float scale = (float)sqrt(2.0 / in);
    for (long i = 0; i < (long)out * in; i++) w[i] = (float)rng_gauss(r) * scale;
    return w;
}
static float *zeros(int n) { return (float *)calloc((size_t)n, sizeof(float)); }

void llm_init(LLM *m, uint64_t seed) {
    rng_t r; rng_seed(&r, seed);
    m->W1 = he_alloc(LLM_H1, IMG_DIM, &r); m->b1 = zeros(LLM_H1);
    m->W2 = he_alloc(LLM_H2, LLM_H1,  &r); m->b2 = zeros(LLM_H2);
    m->W3 = he_alloc(NCLASS, LLM_H2,  &r); m->b3 = zeros(NCLASS);
}

static void forward(const LLM *m, const float *x, float *h1, float *h2, float *p) {
    for (int j = 0; j < LLM_H1; j++) {
        const float *w = m->W1 + (size_t)j * IMG_DIM; float a = m->b1[j];
        for (int i = 0; i < IMG_DIM; i++) a += w[i] * x[i];
        h1[j] = a > 0.0f ? a : 0.0f;
    }
    for (int k = 0; k < LLM_H2; k++) {
        const float *w = m->W2 + (size_t)k * LLM_H1; float a = m->b2[k];
        for (int j = 0; j < LLM_H1; j++) a += w[j] * h1[j];
        h2[k] = a > 0.0f ? a : 0.0f;
    }
    for (int c = 0; c < NCLASS; c++) {
        const float *w = m->W3 + (size_t)c * LLM_H2; float a = m->b3[c];
        for (int k = 0; k < LLM_H2; k++) a += w[k] * h2[k];
        p[c] = a;
    }
    softmax(p, NCLASS);
}

/* ---- per-thread gradient accumulation over a slice of the mini-batch ---- */
typedef struct {
    const LLM *m; const Dataset *tr; const int *order; int b0;
    float *tgW1, *tgb1, *tgW2, *tgb2, *tgW3, *tgb3;   /* T-strided scratch */
} GradJob;

static void grad_range(int begin, int end, int tid, void *arg) {
    GradJob *j = (GradJob *)arg; const LLM *m = j->m;
    float *gW1 = j->tgW1 + (size_t)tid * LLM_H1 * IMG_DIM, *gb1 = j->tgb1 + (size_t)tid * LLM_H1;
    float *gW2 = j->tgW2 + (size_t)tid * LLM_H2 * LLM_H1, *gb2 = j->tgb2 + (size_t)tid * LLM_H2;
    float *gW3 = j->tgW3 + (size_t)tid * NCLASS * LLM_H2, *gb3 = j->tgb3 + (size_t)tid * NCLASS;
    float h1[LLM_H1], h2[LLM_H2], p[NCLASS], e[NCLASS], d2[LLM_H2], d1[LLM_H1];

    for (int s = begin; s < end; s++) {
        const float *x = j->tr->X + (size_t)j->order[j->b0 + s] * IMG_DIM;
        int y = j->tr->y[j->order[j->b0 + s]];
        forward(m, x, h1, h2, p);
        for (int c = 0; c < NCLASS; c++) e[c] = p[c] - (c == y ? 1.0f : 0.0f);

        memset(d2, 0, sizeof(d2));
        for (int c = 0; c < NCLASS; c++) {
            float ec = e[c]; const float *w = m->W3 + (size_t)c * LLM_H2; float *gw = gW3 + (size_t)c * LLM_H2;
            for (int k = 0; k < LLM_H2; k++) { d2[k] += w[k] * ec; gw[k] += ec * h2[k]; }
            gb3[c] += ec;
        }
        for (int k = 0; k < LLM_H2; k++) if (h2[k] <= 0.0f) d2[k] = 0.0f;

        memset(d1, 0, sizeof(d1));
        for (int k = 0; k < LLM_H2; k++) {
            float dk = d2[k]; const float *w = m->W2 + (size_t)k * LLM_H1; float *gw = gW2 + (size_t)k * LLM_H1;
            for (int jj = 0; jj < LLM_H1; jj++) { d1[jj] += w[jj] * dk; gw[jj] += dk * h1[jj]; }
            gb2[k] += dk;
        }
        for (int jj = 0; jj < LLM_H1; jj++) if (h1[jj] <= 0.0f) d1[jj] = 0.0f;

        for (int jj = 0; jj < LLM_H1; jj++) {
            float dj = d1[jj]; float *gw = gW1 + (size_t)jj * IMG_DIM;
            for (int i = 0; i < IMG_DIM; i++) gw[i] += dj * x[i];
            gb1[jj] += dj;
        }
    }
}

/* reduce T per-thread gradient copies into the weights, scaled by step.
 * Kept serial: with a big batch there are few batches, so this is cheap, and a
 * parallel_for here just adds barrier overhead. The win came from BATCH size. */
static void apply(float *W, const float *tg, int T, int sz, float step) {
    for (int t = 0; t < T; t++) { const float *g = tg + (size_t)t * sz;
        for (int i = 0; i < sz; i++) W[i] -= step * g[i]; }
}

void llm_train(LLM *m, const Dataset *tr, const Dataset *te, int epochs, float lr0) {
    int T = threads_count();
    double t_start = now_sec();
    int s1 = LLM_H1 * IMG_DIM, s2 = LLM_H2 * LLM_H1, s3 = NCLASS * LLM_H2;
    float *tgW1 = zeros(T * s1), *tgb1 = zeros(T * LLM_H1);
    float *tgW2 = zeros(T * s2), *tgb2 = zeros(T * LLM_H2);
    float *tgW3 = zeros(T * s3), *tgb3 = zeros(T * NCLASS);
    int *order = (int *)malloc((size_t)tr->n * sizeof(int));
    for (int i = 0; i < tr->n; i++) order[i] = i;
    rng_t r; rng_seed(&r, 1234);

    for (int ep = 0; ep < epochs; ep++) {
        float lr = lr0 / (1.0f + 0.02f * ep);
        for (int i = tr->n - 1; i > 0; i--) { int j = rng_int(&r, i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }

        for (int b0 = 0; b0 < tr->n; b0 += BATCH) {
            int bs = (b0 + BATCH <= tr->n) ? BATCH : (tr->n - b0);
            memset(tgW1, 0, (size_t)T * s1 * sizeof(float)); memset(tgb1, 0, (size_t)T * LLM_H1 * sizeof(float));
            memset(tgW2, 0, (size_t)T * s2 * sizeof(float)); memset(tgb2, 0, (size_t)T * LLM_H2 * sizeof(float));
            memset(tgW3, 0, (size_t)T * s3 * sizeof(float)); memset(tgb3, 0, (size_t)T * NCLASS * sizeof(float));

            GradJob job = { m, tr, order, b0, tgW1, tgb1, tgW2, tgb2, tgW3, tgb3 };
            parallel_for(bs, grad_range, &job);

            float step = lr / bs;
            apply(m->W1, tgW1, T, s1, step); apply(m->b1, tgb1, T, LLM_H1, step);
            apply(m->W2, tgW2, T, s2, step); apply(m->b2, tgb2, T, LLM_H2, step);
            apply(m->W3, tgW3, T, s3, step); apply(m->b3, tgb3, T, NCLASS, step);
        }
        if (te && (ep % 2 == 1 || ep == epochs - 1)) {  /* live progress every 2 epochs */
            printf("    epoch %2d/%2d   test %.2f%%   (%.1fs)\n",
                   ep + 1, epochs, llm_accuracy(m, te) * 100, now_sec() - t_start);
            fflush(stdout);
        }
    }
    free(order);
    free(tgW1); free(tgb1); free(tgW2); free(tgb2); free(tgW3); free(tgb3);
}

double llm_accuracy(const LLM *m, const Dataset *te) {
    float h1[LLM_H1], h2[LLM_H2], p[NCLASS]; int correct = 0;
    for (int s = 0; s < te->n; s++) {
        forward(m, te->X + (size_t)s * IMG_DIM, h1, h2, p);
        if (argmax(p, NCLASS) == te->y[s]) correct++;
    }
    return (double)correct / te->n;
}

/* probabilities for one image (used by the analytics report) */
void llm_predict(const LLM *m, const float *x, float *out) {
    float h1[LLM_H1], h2[LLM_H2];
    forward(m, x, h1, h2, out);
}

long llm_ops_per_image(void) {
    return (long)IMG_DIM * LLM_H1 + (long)LLM_H1 * LLM_H2 + (long)LLM_H2 * NCLASS;
}

void llm_free(LLM *m) {
    free(m->W1); free(m->b1); free(m->W2); free(m->b2); free(m->W3); free(m->b3);
    memset(m, 0, sizeof(*m));
}
