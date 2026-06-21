/* ─────────────────────────────────────────────────────────────────────────────
 * llm.h — the LLM-style contestant: a dense, fully-connected, 2-hidden-layer
 * neural network trained by BACKPROPAGATION. This is the standard recipe every
 * mainstream AI uses, shrunk to laptop size. It is the bar the brain must beat.
 *
 *   196 inputs -> 256 ReLU -> 256 ReLU -> 10 classes (softmax)
 *
 * Every neuron fires on every image, and learning flows backward through every
 * weight. Powerful, and our reference for "what good accuracy looks like."
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef LLM_H
#define LLM_H

#include "dataset.h"

#define LLM_H1 256
#define LLM_H2 256

typedef struct {
    /* weight matrices (row-major) and bias vectors */
    float *W1, *b1;     /* H1 x IMG_DIM , H1   */
    float *W2, *b2;     /* H2 x H1      , H2   */
    float *W3, *b3;     /* NCLASS x H2  , NCLASS */
} LLM;

/* allocate + He-initialize the weights */
void  llm_init(LLM *m, uint64_t seed);

/* train in place by mini-batch backprop. If `te` is non-NULL, prints live
 * per-epoch test accuracy so progress can be watched as it runs. */
void  llm_train(LLM *m, const Dataset *tr, const Dataset *te, int epochs, float lr0);

/* fraction correct on a held-out split */
double llm_accuracy(const LLM *m, const Dataset *te);

/* class probabilities for one image (for the analytics report) */
void  llm_predict(const LLM *m, const float *x, float *out);

/* multiply-accumulate operations to classify ONE image (the energy proxy) */
long  llm_ops_per_image(void);

void  llm_free(LLM *m);

#endif /* LLM_H */
