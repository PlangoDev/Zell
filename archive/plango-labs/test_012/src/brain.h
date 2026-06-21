/* ─────────────────────────────────────────────────────────────────────────────
 * brain.h — the brain-style contestant for WORDS: a CEREBELLAR N-GRAM CODER.
 *
 * This is the same Marr-Albus cerebellum that won test_010/011, mapped onto
 * language. NO BACKPROPAGATION anywhere.
 *
 *   1. WORD CODES        — each word gets a FIXED random vector (BR_EMB dims),
 *                          like a dentate/granule random projection. Not learned.
 *   2. MOSSY FIBERS      — the CTX word codes are concatenated into one context
 *                          vector (BR_IN dims).
 *   3. GRANULE EXPANSION — BR_G granule cells, each wired to only BR_SIN inputs
 *                          (real granule cells have ~4 dendrites) with fixed
 *                          random weights. A huge, cheap, fixed expansion.
 *   4. kWTA SPARSE CODE  — only the BR_K strongest granules fire (sparse), so the
 *                          readout only ever touches BR_K rows — that is why the
 *                          brain stays cheap even with a big vocabulary.
 *   5. PURKINJE READOUT  — one linear layer to the vocabulary, trained by the
 *                          local delta rule. The ONLY thing that learns.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef BRAIN_H
#define BRAIN_H

#include "config.h"
#include "corpus.h"
#include <stdint.h>

typedef struct {
    float *Eb;      /* vocab * BR_EMB   : DISTRIBUTIONAL word codes (learned,      */
                    /*                    no backprop: co-occurrence -> PPMI ->    */
                    /*                    random projection. Similar words ~ near) */
    int   *gidx;    /* BR_G  * BR_SIN   : which inputs each granule samples        */
    float *gwt;     /* BR_G  * BR_SIN   : fixed random granule weights             */
    float *gbias;   /* BR_G             : homeostatic boost (intrinsic plasticity) */
    float *Wr;      /* BR_G  * vocab    : WITHIN-CLASS word readout (row per granule)*/
    float *br;      /* vocab            : word readout bias                        */
    float *Wc;      /* BR_G  * BR_CLASSES : CLASS readout (the "what stream")      */
    float *bc;      /* BR_CLASSES        : class readout bias                      */
    int   *w2class; /* vocab            : each word's class id                     */
    int   *cls_off; /* BR_CLASSES+1     : CSR offsets into cls_word                */
    int   *cls_word;/* vocab            : word ids grouped by class                */
    int   *w2slot;  /* vocab            : a word's position within its class group */
    int    vocab;
    void  *hippo;   /* opaque hippocampal episodic cache (context -> next-word     */
                    /* counts). Built once; fused with the cortical readout (CLS). */
} Brain;

/* builds the fixed front-end FROM the corpus: distributional word codes +
 * granule wiring + homeostatic boosting. The readout starts at zero. */
void   brain_init(Brain *b, const Corpus *co, uint64_t seed);

/* encode a CTX-long context into its sparse code: the BR_K active granule ids
 * and their (positive) activations. This is the fixed front-end — no learning. */
void   brain_encode(const Brain *b, const int *ctx, int *out_idx, float *out_val);

/* train ONLY the readout, by the local delta rule, over the corpus train stream.
 * Encodes every context once, then sweeps the sparse codes for `epochs`. */
void   brain_train(Brain *b, const Corpus *co, int epochs, float lr0);

void   brain_eval(const Brain *b, const int *stream, int ntok, double *ppl, double *acc);
void   brain_next(const Brain *b, const int *ctx, float *probs);

/* ── Hippocampus (Complementary Learning Systems) ─────────────────────────────
 * The cortical readout above generalizes slowly across episodes; the hippocampus
 * stores SPECIFIC episodes (an exact context -> next-word table) and recalls them
 * by pattern completion. brain_next_cls fuses the two: when a context is familiar
 * the sharp episodic recall dominates; when it is novel the cortex carries it.
 * The cache is a hash lookup — effectively free in MACs, so the Brain stays cheap. */
void   brain_build_hippocampus(Brain *b, const Corpus *co);
void   brain_next_cls(const Brain *b, const int *ctx, float *probs);
void   brain_eval_cls(const Brain *b, const int *stream, int ntok, double *ppl, double *acc);

/* granule-code health, for diagnosing why the brain plateaus where it does */
typedef struct {
    double density;        /* fraction of granules that ever fire (1 - dead frac) */
    int    dead_granules;  /* granules never in the top-K over the sample          */
    double usage_gini;     /* 0 = every granule used equally, 1 = a few dominate   */
    double mean_margin;    /* average winner activation after threshold (code size) */
    double readout_wnorm;  /* L2 norm of the learned readout (saturation check)    */
    int    dead_readout;   /* vocabulary rows the readout never learned (all ~0)    */
} BrainDiag;
void   brain_diagnose(const Brain *b, const int *stream, int ntok, BrainDiag *d);

long   brain_ops_per_pred(void);
void   brain_free(Brain *b);

#endif /* BRAIN_H */
