/* ─────────────────────────────────────────────────────────────────────────────
 * analytics.h — deep diagnostics for the head-to-head. Given the two models'
 * predictions on the test set, it prints everything we need to actually find
 * the bottleneck: per-class accuracy, a confusion matrix, where the two models
 * agree and disagree, and the committee's internal diversity.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef ANALYTICS_H
#define ANALYTICS_H

#include <stdint.h>

/* the ten Fashion-MNIST class names, for readable tables */
extern const char *FASHION_NAMES[10];

/* per-class precision / recall / F1, printed as a table */
void ana_per_class(const char *who, const int *pred, const uint8_t *y, int n);

/* 10x10 confusion matrix (rows = true class, cols = predicted) */
void ana_confusion(const char *who, const int *pred, const uint8_t *y, int n);

/* how often the two models agree, and who is right when they disagree */
void ana_agreement(const int *a_pred, const int *b_pred, const uint8_t *y, int n);

/* committee internals: each expert's solo accuracy + average pairwise agreement */
void ana_committee(const double *solo_acc, const int *expert_pred, int E,
                   const uint8_t *y, int n);

#endif /* ANALYTICS_H */
