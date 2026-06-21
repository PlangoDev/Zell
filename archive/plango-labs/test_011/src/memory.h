/* ─────────────────────────────────────────────────────────────────────────────
 * memory.h — the HIPPOCAMPUS: a fast episodic memory over feature codes.
 *
 * It stores (code, label) pairs as they are experienced — one exposure is
 * enough (that is the "one-shot" the brain is famous for) — and recalls by
 * cosine similarity (k-nearest-neighbour vote). Codes are L2-normalized on the
 * way in so recall is a plain dot product.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef MEMORY_H
#define MEMORY_H

#include <stdint.h>

typedef struct {
    int      cap;     /* maximum memories                       */
    int      n;       /* memories currently held                */
    int      dim;     /* length of each code (= BR_FEAT)        */
    float   *codes;   /* cap * dim, each L2-normalized          */
    uint8_t *labels;  /* cap                                    */
} Memory;

void mem_init(Memory *m, int cap, int dim);
void mem_clear(Memory *m);                                  /* forget everything */
void mem_add(Memory *m, const float *code, int label);      /* store one episode */
void mem_knn(const Memory *m, const float *q, int knn, float *out_probs);  /* recall */
void mem_free(Memory *m);

#endif /* MEMORY_H */
