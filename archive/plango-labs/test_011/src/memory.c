/* ─────────────────────────────────────────────────────────────────────────────
 * memory.c — the hippocampus, implemented. Store and recall, nothing trained.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "memory.h"
#include "common.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

void mem_init(Memory *m, int cap, int dim) {
    m->cap = cap; m->n = 0; m->dim = dim;
    m->codes  = (float *)malloc((size_t)cap * dim * sizeof(float));
    m->labels = (uint8_t *)malloc((size_t)cap);
}

void mem_clear(Memory *m) { m->n = 0; }

void mem_add(Memory *m, const float *code, int label) {
    if (m->n >= m->cap) return;                      /* bank is full, drop it */
    float *dst = m->codes + (size_t)m->n * m->dim;
    float nrm = 0.0f;
    for (int i = 0; i < m->dim; i++) nrm += code[i] * code[i];
    float inv = 1.0f / (sqrtf(nrm) + 1e-8f);
    for (int i = 0; i < m->dim; i++) dst[i] = code[i] * inv;   /* store unit-norm */
    m->labels[m->n] = (uint8_t)label;
    m->n++;
}

/* recall: cosine-similarity to every memory, let the k closest vote (weighted
 * by similarity) for their labels, return class probabilities */
void mem_knn(const Memory *m, const float *q, int knn, float *out_probs) {
    for (int c = 0; c < NCLASS; c++) out_probs[c] = 0.0f;
    if (m->n == 0) return;

    /* normalize the query so the dot product is a cosine */
    float qn[1 << 14];                               /* dim <= 16384 */
    float nrm = 0.0f;
    for (int i = 0; i < m->dim; i++) nrm += q[i] * q[i];
    float inv = 1.0f / (sqrtf(nrm) + 1e-8f);
    for (int i = 0; i < m->dim; i++) qn[i] = q[i] * inv;

    /* similarity to each memory */
    int K = knn < m->n ? knn : m->n;
    float *sim = (float *)malloc((size_t)m->n * sizeof(float));
    for (int j = 0; j < m->n; j++) {
        const float *c = m->codes + (size_t)j * m->dim;
        float d = 0.0f;
        for (int i = 0; i < m->dim; i++) d += qn[i] * c[i];
        sim[j] = d;
    }

    /* take the top-K by a simple partial selection, vote weighted by similarity */
    for (int t = 0; t < K; t++) {
        int best = -1; float bv = -1e30f;
        for (int j = 0; j < m->n; j++) if (sim[j] > bv) { bv = sim[j]; best = j; }
        if (best < 0) break;
        out_probs[m->labels[best]] += bv > 0 ? bv : 0.0f;
        sim[best] = -1e30f;                          /* remove from contention */
    }
    free(sim);

    float s = 0.0f;
    for (int c = 0; c < NCLASS; c++) s += out_probs[c];
    if (s > 0) for (int c = 0; c < NCLASS; c++) out_probs[c] /= s;
}

void mem_free(Memory *m) {
    free(m->codes); free(m->labels);
    m->codes = NULL; m->labels = NULL; m->n = m->cap = 0;
}
