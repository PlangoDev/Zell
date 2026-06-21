/* ─────────────────────────────────────────────────────────────────────────────
 * analytics.h — diagnostics for the word head-to-head. None of this trains
 * anything; it exists so we can SEE why each model wins or stalls:
 *   - count-based n-gram baselines (the bar every neural model must clear)
 *   - top-1 / top-5 / top-10 accuracy and bits-per-word for a model
 *   - a head-to-head on who puts more probability on the true next word
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef ANALYTICS_H
#define ANALYTICS_H

#include "config.h"
#include "corpus.h"
#include <functional>

/* a model is anything that fills `probs` (length vocab) from a CTX-long context */
typedef std::function<void(const int *ctx, float *probs)> NextFn;

typedef struct {
    double ppl;                 /* perplexity (lower better)                      */
    double bits;                /* bits per word = log2(perplexity)               */
    double top1, top5, top10;   /* fraction where the true word is in the top-k   */
} TopkReport;

/* rich evaluation of one model over a token stream */
TopkReport ana_eval(NextFn next, const int *stream, int ntok, int vocab);

/* count-based reference perplexities (unigram / bigram / trigram / interpolated),
 * built from train and scored on test. Prints a table + trigram coverage. */
void ana_ngram_baselines(const Corpus *co);

/* on the test set: who assigns more probability to the TRUE next word, and how
 * often the two models' top-1 picks agree */
void ana_head_to_head(NextFn llm_next, NextFn brain_next, const Corpus *co);

#endif /* ANALYTICS_H */
