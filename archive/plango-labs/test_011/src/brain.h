/* ─────────────────────────────────────────────────────────────────────────────
 * brain.h — the brain-style contestant: the CEREBELLAR SPARSE CODER.
 *
 * Fuses every idea that actually worked across tests 001-010, drops the rest.
 * NO BACKPROPAGATION anywhere.
 *
 *   1. LOCAL RECEPTIVE FIELDS — 5x5 patches on a 5x5 grid (V1-style).
 *   2. WHITENING              — decorrelate patches first (the Coates-Ng trick),
 *                               folded into the detectors so inference is cheap.
 *   3. COMPETITIVE LEARNING   — learn the detectors by spherical k-means, a local
 *                               winner-take-all Hebbian rule. No gradients.
 *   4. SPARSE CODING          — each spot keeps only above-average matches.
 *   5. LOCATION-SPECIFIC CODE — every grid spot keeps its own activations (no
 *                               pooling), so the readout knows WHERE things are.
 *   6. LOCAL-RULE READOUT     — one linear layer, delta rule. No deep learning.
 *
 * One expert; main.c runs several as a voting committee. Training is fully
 * multithreaded (see threads.h).
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef BRAIN_H
#define BRAIN_H

#include "dataset.h"
#include "memory.h"
#include <stdint.h>

#define BR_PATCH 6                          /* 6x6 receptive field (28x28 input)  */
#define BR_PDIM  (BR_PATCH * BR_PATCH)      /* 36 pixels per patch                */
#define BR_PROJ  12                         /* keep only the top 12 whitened patch dims */
#define BR_GRID  6                          /* 6x6 grid of patch locations        */
#define BR_NLOC  (BR_GRID * BR_GRID)        /* 36 locations                       */
#define BR_K     64                         /* prototypes per location            */
#define BR_FEAT  (BR_NLOC * BR_K)           /* 36 * 64 = 2304 location-specific feats */

typedef struct {
    float *proj;     /* BR_PROJ * BR_PDIM : whitening + dim-reduction projection   */
    float *pmean;    /* BR_PDIM           : patch mean (subtracted before project) */
    float *dict;     /* BR_K * BR_PROJ    : detectors, in the reduced 12-D space   */
    float *fmean;    /* BR_FEAT        : per-feature mean   (standardization)      */
    float *fstd;     /* BR_FEAT        : per-feature stddev (standardization)      */
    float *W;        /* NCLASS * BR_FEAT : the linear readout weights             */
    float *b;        /* NCLASS         : readout biases                           */
    Memory hippo;    /* the brain's own HIPPOCAMPUS — a sample of what it learned */
} BrainExpert;

/* per-stage wall-clock timing, filled by brain_train for the analytics report */
typedef struct { double dict_s, feat_s, readout_s; } BrainTiming;

void   brain_train(BrainExpert *e, const Dataset *tr, uint64_t seed, BrainTiming *tm);
void   brain_predict(const BrainExpert *e, const float *x, float *out);

/* encode an image into its standardized BR_FEAT feature code (what the readout
 * sees). The hippocampus stores these codes; sleep replays them. */
void   brain_encode(const BrainExpert *e, const float *x, float *code);

/* run a readout (W: NCLASS*BR_FEAT, b: NCLASS) on a pre-computed code */
int    brain_readout(const float *W, const float *b, const float *code);
double brain_accuracy(const BrainExpert *e, const Dataset *te);
long   brain_ops_per_image(void);

/* analytics: average fraction of nonzero features, and how many of the BR_K
 * prototypes are "dead" (essentially never the local winner). */
void   brain_feature_stats(const BrainExpert *e, const Dataset *te,
                           double *avg_density, int *dead_protos);

/* ---- the brain doing what brains do (memory + sleep, now INSIDE the brain) ---- */
/* learn brand-new categories from K examples via episodic memory */
void   brain_oneshot(const BrainExpert *e, const Dataset *tr, const Dataset *te);
/* wake-sleep consolidation: keep a few memories, sleep on them over and over */
void   brain_sleep(const BrainExpert *e, const Dataset *tr, const Dataset *te, double full_acc);
/* fuse the cortex with the brain's own hippocampus (complementary learning systems) */
double brain_fusion(const BrainExpert *e, const Dataset *te, long cortex_ops);

void   brain_free(BrainExpert *e);

#endif /* BRAIN_H */
