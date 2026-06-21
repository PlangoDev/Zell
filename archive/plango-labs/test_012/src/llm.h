/* ─────────────────────────────────────────────────────────────────────────────
 * llm.h — the LLM-style contestant: a NEURAL LANGUAGE MODEL trained by
 * BACKPROPAGATION (Bengio et al. 2003, the direct ancestor of today's LLMs),
 * shrunk to laptop size. This is the bar the brain must beat.
 *
 *   [w-3,w-2,w-1] --embed--> 144 --W1--> 128 tanh --W2--> VOCAB softmax
 *
 * Every weight is trained end-to-end by gradients flowing backward — including
 * the word embeddings, so similar words can learn similar vectors. That learned
 * generalization is exactly the LLM's edge over the brain's fixed codes.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef LLM_H
#define LLM_H

#include "config.h"
#include "corpus.h"
#include <stdint.h>

typedef struct {
    float *E;           /* VOCAB * LM_EMB : learned word embeddings              */
    float *W1, *b1;     /* LM_HID * LM_IN , LM_HID                               */
    float *W2, *b2;     /* VOCAB  * LM_HID, VOCAB                                */
    int    vocab;       /* actual vocabulary size for this corpus               */

    /* Adam optimizer state (first/second moment per weight) + step counter.
     * Adam converges much faster and more stably than plain SGD, which is what
     * the data asked for — the SGD run was still falling at epoch 20. */
    float *mE, *vE, *mW1, *vW1, *mb1, *vb1, *mW2, *vW2, *mb2, *vb2;
    long   tstep;

    /* diagnostics, refreshed each epoch by llm_train (read by the report) */
    double last_gnorm;  /* gradient L2 norm of the last batch                   */
    double train_ppl;   /* perplexity on a train subset (vs test = overfit gap) */
    int    best_ep;     /* epoch whose checkpoint was kept (early stopping)     */
    int    epochs_run;  /* epochs actually executed (for the training-cost tally)*/
    long   train_ex;    /* training examples swept per epoch                    */
} LLM;

void   llm_init(LLM *m, int vocab, uint64_t seed);

/* Train by mini-batch Adam over the corpus's train stream. Prints live per-epoch
 * test perplexity, top-1, overfit gap, and gradient norm so progress is legible. */
void   llm_train(LLM *m, const Corpus *co, int epochs, float lr0);

/* perplexity (lower is better) and next-word top-1 accuracy on a token stream */
void   llm_eval(const LLM *m, const int *stream, int ntok, double *ppl, double *acc);

/* probabilities of the next word given a CTX-long context (for generation) */
void   llm_next(const LLM *m, const int *ctx, float *probs);

long   llm_ops_per_pred(void);     /* multiply-adds to predict one word         */
void   llm_free(LLM *m);

#endif /* LLM_H */
