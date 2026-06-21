/* ─────────────────────────────────────────────────────────────────────────────
 * llm.cpp — the neural language model, portable BATCHED training (no BLAS).
 *
 * The first cut did one example at a time, streaming the 2 MB W2 matrix from
 * memory PER example and keeping a full per-thread copy of every gradient. Both
 * are killers. This version is batch-major:
 *   - the output forward/backward thread OVER THE VOCABULARY, so each W2 row is
 *     loaded once and reused across all B examples in the batch (cache win), and
 *   - each output row's weight gradient is owned by exactly one thread, so there
 *     are NO per-thread gradient buffers and NO giant memsets (the old version's
 *     biggest hidden cost). Same maths, same quality — just laid out for speed.
 *
 * Still fully regularized (dropout + AdamW + early stopping), since the SGD/Adam
 * runs showed the model memorizes the corpus without it. Builds everywhere.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "llm.h"
#include "common.h"
#include "threads.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define BATCH       512
#define ADAM_B1     0.9f
#define ADAM_B2     0.999f
#define ADAM_EPS    1e-8f
#define LM_DROPOUT  0.5f
#define LM_WD       1e-4f
#define LM_VALFRAC  12

static float *gauss_alloc(long n, float scale, rng_t *r) {
    float *w = (float *)malloc((size_t)n * sizeof(float));
    for (long i = 0; i < n; i++) w[i] = (float)rng_gauss(r) * scale;
    return w;
}
static float *zeros(long n) { return (float *)calloc((size_t)n, sizeof(float)); }

void llm_init(LLM *m, int vocab, uint64_t seed) {
    rng_t r; rng_seed(&r, seed);
    m->vocab = vocab;
    m->E  = gauss_alloc((long)vocab * LM_EMB, 0.10f, &r);
    m->W1 = gauss_alloc((long)LM_HID * LM_IN, (float)sqrt(1.0 / LM_IN), &r);
    m->b1 = zeros(LM_HID);
    m->W2 = gauss_alloc((long)vocab * LM_HID, (float)sqrt(1.0 / LM_HID), &r);
    m->b2 = zeros(vocab);

    m->mE  = zeros((long)vocab * LM_EMB); m->vE  = zeros((long)vocab * LM_EMB);
    m->mW1 = zeros((long)LM_HID * LM_IN); m->vW1 = zeros((long)LM_HID * LM_IN);
    m->mb1 = zeros(LM_HID);               m->vb1 = zeros(LM_HID);
    m->mW2 = zeros((long)vocab * LM_HID); m->vW2 = zeros((long)vocab * LM_HID);
    m->mb2 = zeros(vocab);                m->vb2 = zeros(vocab);
    m->tstep = 0; m->last_gnorm = 0; m->train_ppl = 0;
}

/* single-example forward (no dropout) — used only by generation */
static void forward1(const LLM *m, const int *ctx, float *p) {
    int V = m->vocab; float x[LM_IN], h[LM_HID];
    for (int t = 0; t < CTX; t++)
        memcpy(x + t * LM_EMB, m->E + (size_t)ctx[t] * LM_EMB, LM_EMB * sizeof(float));
    for (int k = 0; k < LM_HID; k++) {
        const float *RESTRICT w = m->W1 + (size_t)k * LM_IN; float a = m->b1[k];
        for (int i = 0; i < LM_IN; i++) a += w[i] * x[i];
        h[k] = tanhf(a);
    }
    for (int v = 0; v < V; v++) {
        const float *RESTRICT w = m->W2 + (size_t)v * LM_HID; float a = m->b2[v];
        for (int k = 0; k < LM_HID; k++) a += w[k] * h[k];
        p[v] = a;
    }
    softmax(p, V);
}

/* ── batch scratch, shared by the training step and batched eval ──────────── */
typedef struct {
    const LLM *m; int bs;
    const float *X, *H, *Hpre, *mask;     /* B x {IN,HID,HID,HID}  */
    float *logits;                        /* B x V (becomes dlogits in training) */
    float *dH, *dX;                       /* B x {HID,IN}          */
    float *gE, *gW1, *gb1, *gW2, *gb2;    /* full gradient tensors */
} BJob;

/* hidden layer for the batch: H = tanh(W1·x + b1) then dropout (thread over b) */
static void hidden_range(int begin, int end, int tid, void *arg) {
    (void)tid; BJob *j = (BJob *)arg; const LLM *m = j->m;
    for (int b = begin; b < end; b++) {
        const float *RESTRICT x = j->X + (size_t)b * LM_IN;
        float *RESTRICT hpre = (float *)j->Hpre + (size_t)b * LM_HID;
        float *RESTRICT h    = (float *)j->H    + (size_t)b * LM_HID;
        /* mask may be NULL (eval has no dropout); base the per-row pointer on the
         * BASE pointer, not an offset of it — NULL + b*LM_HID is a bogus non-NULL
         * pointer for b>0 and would slip past the `mk ?` guard below. */
        const float *RESTRICT mk = j->mask ? j->mask + (size_t)b * LM_HID : NULL;
        for (int k = 0; k < LM_HID; k++) {
            const float *RESTRICT w = m->W1 + (size_t)k * LM_IN; float a = m->b1[k];
            for (int i = 0; i < LM_IN; i++) a += w[i] * x[i];
            float t = tanhf(a); hpre[k] = t; h[k] = mk ? t * mk[k] : t;
        }
    }
}

/* output logits for the batch: thread OVER VOCAB so each W2 row is reused across
 * every example (the cache win). logits stored row-major [b*V + v]. */
static void logits_range(int begin, int end, int tid, void *arg) {
    (void)tid; BJob *j = (BJob *)arg; const LLM *m = j->m; int bs = j->bs;
    for (int v = begin; v < end; v++) {
        const float *RESTRICT w = m->W2 + (size_t)v * LM_HID; float bias = m->b2[v];
        for (int b = 0; b < bs; b++) {
            const float *RESTRICT h = j->H + (size_t)b * LM_HID; float a = bias;
            for (int k = 0; k < LM_HID; k++) a += w[k] * h[k];
            j->logits[(size_t)b * j->m->vocab + v] = a;
        }
    }
}

/* softmax each row (thread over b) */
static void softmax_range(int begin, int end, int tid, void *arg) {
    (void)tid; BJob *j = (BJob *)arg; int V = j->m->vocab;
    for (int b = begin; b < end; b++) softmax(j->logits + (size_t)b * V, V);
}

/* dW2 + db2: each vocab row owned by one thread (race-free, no per-thread copy) */
static void gW2_range(int begin, int end, int tid, void *arg) {
    (void)tid; BJob *j = (BJob *)arg; int bs = j->bs, V = j->m->vocab;
    for (int v = begin; v < end; v++) {
        float acc[LM_HID]; for (int k = 0; k < LM_HID; k++) acc[k] = 0.0f;
        float gb = 0.0f;
        for (int b = 0; b < bs; b++) {
            float e = j->logits[(size_t)b * V + v];          /* dlogit */
            const float *RESTRICT h = j->H + (size_t)b * LM_HID;
            for (int k = 0; k < LM_HID; k++) acc[k] += e * h[k];
            gb += e;
        }
        memcpy(j->gW2 + (size_t)v * LM_HID, acc, LM_HID * sizeof(float));
        j->gb2[v] = gb;
    }
}

/* dH = dlogits·W2, then back through dropout + tanh' (thread over b) */
static void dH_range(int begin, int end, int tid, void *arg) {
    (void)tid; BJob *j = (BJob *)arg; const LLM *m = j->m; int V = m->vocab;
    for (int b = begin; b < end; b++) {
        float *RESTRICT dh = j->dH + (size_t)b * LM_HID;
        for (int k = 0; k < LM_HID; k++) dh[k] = 0.0f;
        const float *RESTRICT dl = j->logits + (size_t)b * V;
        for (int v = 0; v < V; v++) {
            float e = dl[v]; const float *RESTRICT w = m->W2 + (size_t)v * LM_HID;
            for (int k = 0; k < LM_HID; k++) dh[k] += e * w[k];
        }
        const float *RESTRICT hpre = j->Hpre + (size_t)b * LM_HID;
        const float *RESTRICT mk = j->mask + (size_t)b * LM_HID;
        for (int k = 0; k < LM_HID; k++) dh[k] *= mk[k] * (1.0f - hpre[k] * hpre[k]);
    }
}

/* dW1 + db1: each hidden row owned by one thread (race-free) */
static void gW1_range(int begin, int end, int tid, void *arg) {
    (void)tid; BJob *j = (BJob *)arg; int bs = j->bs;
    for (int k = begin; k < end; k++) {
        float acc[LM_IN]; for (int i = 0; i < LM_IN; i++) acc[i] = 0.0f;
        float gb = 0.0f;
        for (int b = 0; b < bs; b++) {
            float d = j->dH[(size_t)b * LM_HID + k];
            const float *RESTRICT x = j->X + (size_t)b * LM_IN;
            for (int i = 0; i < LM_IN; i++) acc[i] += d * x[i];
            gb += d;
        }
        memcpy(j->gW1 + (size_t)k * LM_IN, acc, LM_IN * sizeof(float));
        j->gb1[k] = gb;
    }
}

/* dX = dH·W1 (thread over b) — needed to push the gradient into the embeddings */
static void dX_range(int begin, int end, int tid, void *arg) {
    (void)tid; BJob *j = (BJob *)arg; const LLM *m = j->m;
    for (int b = begin; b < end; b++) {
        float *RESTRICT dx = j->dX + (size_t)b * LM_IN;
        for (int i = 0; i < LM_IN; i++) dx[i] = 0.0f;
        const float *RESTRICT dh = j->dH + (size_t)b * LM_HID;
        for (int k = 0; k < LM_HID; k++) {
            float d = dh[k]; const float *RESTRICT w = m->W1 + (size_t)k * LM_IN;
            for (int i = 0; i < LM_IN; i++) dx[i] += d * w[i];
        }
    }
}

/* ── parallel AdamW update of one tensor; returns the grad L2 norm² ────────── */
typedef struct {
    float *W, *mv, *vv; const float *g; long sz;
    float lr, invbs, bc1, bc2, wd; double *tnorm;
} AdamJob;
static void adam_range(int begin, int end, int tid, void *arg) {
    AdamJob *j = (AdamJob *)arg; double ns = 0.0;
    for (int i = begin; i < end; i++) {
        float g = j->g[i] * j->invbs;
        ns += (double)g * g;
        float mm = ADAM_B1 * j->mv[i] + (1.0f - ADAM_B1) * g;
        float vv = ADAM_B2 * j->vv[i] + (1.0f - ADAM_B2) * g * g;
        j->mv[i] = mm; j->vv[i] = vv;
        float mh = mm / j->bc1, vh = vv / j->bc2;
        j->W[i] -= j->lr * (mh / (sqrtf(vh) + ADAM_EPS) + j->wd * j->W[i]);
    }
    j->tnorm[tid] += ns;
}
static double adam_step(float *W, float *mv, float *vv, const float *g, long sz,
                        float lr, float invbs, float bc1, float bc2, float wd, double *tnorm, int T) {
    for (int t = 0; t < T; t++) tnorm[t] = 0.0;
    AdamJob aj = { W, mv, vv, g, sz, lr, invbs, bc1, bc2, wd, tnorm };
    parallel_for((int)sz, adam_range, &aj);
    double s = 0.0; for (int t = 0; t < T; t++) s += tnorm[t];
    return s;
}

/* fill probs for a batch of examples (no dropout) — used by eval/probe */
static void probs_batch(const LLM *m, const int *stream, const int *idx, int bs,
                        float *X, float *H, float *logits) {
    for (int b = 0; b < bs; b++) {
        const int *ctx = stream + idx[b];
        for (int t = 0; t < CTX; t++)
            memcpy(X + (size_t)b * LM_IN + t * LM_EMB, m->E + (size_t)ctx[t] * LM_EMB, LM_EMB * sizeof(float));
    }
    BJob j; memset(&j, 0, sizeof(j));
    j.m = m; j.bs = bs; j.X = X; j.H = H; j.Hpre = H; j.mask = NULL; j.logits = logits;
    parallel_for(bs, hidden_range, &j);
    parallel_for(m->vocab, logits_range, &j);
    parallel_for(bs, softmax_range, &j);
}

/* perplexity over an index list of examples (batched) */
static double ppl_indices(const LLM *m, const int *stream, const int *idx, int n,
                          float *X, float *H, float *logits) {
    int V = m->vocab; double nll = 0.0;
    for (int b0 = 0; b0 < n; b0 += BATCH) {
        int bs = (b0 + BATCH <= n) ? BATCH : (n - b0);
        probs_batch(m, stream, idx + b0, bs, X, H, logits);
        for (int b = 0; b < bs; b++) {
            float py = logits[(size_t)b * V + stream[idx[b0 + b] + CTX]];
            if (py < 1e-12f) py = 1e-12f;
            nll += -log((double)py);
        }
    }
    return exp(nll / (n ? n : 1));
}

void llm_train(LLM *m, const Corpus *co, int epochs, float lr0) {
    int T = threads_count(), V = m->vocab;
    long sE = (long)V * LM_EMB, sW1 = (long)LM_HID * LM_IN, sW2 = (long)V * LM_HID;

    /* batch scratch (allocated once) */
    float *X = (float *)malloc((size_t)BATCH * LM_IN * sizeof(float));
    float *Hp = (float *)malloc((size_t)BATCH * LM_HID * sizeof(float));
    float *H  = (float *)malloc((size_t)BATCH * LM_HID * sizeof(float));
    float *mask = (float *)malloc((size_t)BATCH * LM_HID * sizeof(float));
    float *logits = (float *)malloc((size_t)BATCH * V * sizeof(float));
    float *dH = (float *)malloc((size_t)BATCH * LM_HID * sizeof(float));
    float *dX = (float *)malloc((size_t)BATCH * LM_IN * sizeof(float));
    float *gE = (float *)malloc((size_t)sE * sizeof(float));
    float *gW1 = (float *)malloc((size_t)sW1 * sizeof(float)), *gb1 = (float *)malloc(LM_HID * sizeof(float));
    float *gW2 = (float *)malloc((size_t)sW2 * sizeof(float)), *gb2 = (float *)malloc((size_t)V * sizeof(float));
    double *tnorm = (double *)malloc((size_t)T * sizeof(double));

    /* hold out the last 1/LM_VALFRAC of train tokens as a validation slice */
    int val_tok = co->ntrain / LM_VALFRAC;
    int split   = co->ntrain - val_tok;
    int nex = corpus_examples(split);
    int *order = (int *)malloc((size_t)nex * sizeof(int));
    for (int i = 0; i < nex; i++) order[i] = i;

    int vprobe = corpus_examples(val_tok); if (vprobe > 4000) vprobe = 4000;
    int *vidx = (int *)malloc((size_t)vprobe * sizeof(int));
    for (int i = 0; i < vprobe; i++) vidx[i] = i;          /* into val_stream */
    int tprobe = nex < 4000 ? nex : 4000;
    int *tidx = (int *)malloc((size_t)tprobe * sizeof(int));
    for (int i = 0; i < tprobe; i++) tidx[i] = i;
    const int *val_stream = co->train + split;

    rng_t r; rng_seed(&r, 4321);
    double t0 = now_sec();

    /* best-on-validation checkpoint (early stopping) */
    float *bE = (float *)malloc((size_t)sE * sizeof(float));
    float *bW1 = (float *)malloc((size_t)sW1 * sizeof(float)), *bb1 = (float *)malloc(LM_HID * sizeof(float));
    float *bW2 = (float *)malloc((size_t)sW2 * sizeof(float)), *bb2 = (float *)malloc((size_t)V * sizeof(float));
    double best_val = 1e300; int best_ep = 0;

    for (int ep = 0; ep < epochs; ep++) {
        for (int i = nex - 1; i > 0; i--) { int k = rng_int(&r, i + 1); int t = order[i]; order[i] = order[k]; order[k] = t; }

        double gn2 = 0.0;
        for (int b0 = 0; b0 < nex; b0 += BATCH) {
            int bs = (b0 + BATCH <= nex) ? BATCH : (nex - b0);

            /* gather inputs + dropout mask for this batch */
            for (int b = 0; b < bs; b++) {
                const int *ctx = co->train + order[b0 + b];
                for (int t = 0; t < CTX; t++)
                    memcpy(X + (size_t)b * LM_IN + t * LM_EMB, m->E + (size_t)ctx[t] * LM_EMB, LM_EMB * sizeof(float));
            }
            rng_t mr; rng_seed(&mr, (uint64_t)(m->tstep + 1) * 2654435761u + (uint64_t)b0);
            float inv = 1.0f / (1.0f - LM_DROPOUT);
            for (int i = 0; i < bs * LM_HID; i++) mask[i] = (rng_unif(&mr) < LM_DROPOUT) ? 0.0f : inv;

            BJob j; memset(&j, 0, sizeof(j));
            j.m = m; j.bs = bs; j.X = X; j.H = H; j.Hpre = Hp; j.mask = mask;
            j.logits = logits; j.dH = dH; j.dX = dX;
            j.gE = gE; j.gW1 = gW1; j.gb1 = gb1; j.gW2 = gW2; j.gb2 = gb2;

            parallel_for(bs, hidden_range, &j);            /* H = dropout(tanh(W1x+b1)) */
            parallel_for(V, logits_range, &j);             /* logits (reuse W2 rows)    */
            parallel_for(bs, softmax_range, &j);           /* P                         */
            for (int b = 0; b < bs; b++)                   /* dlogits = P - onehot      */
                logits[(size_t)b * V + co->train[order[b0 + b] + CTX]] -= 1.0f;
            parallel_for(V, gW2_range, &j);                /* dW2, db2 (race-free)      */
            parallel_for(bs, dH_range, &j);                /* dH                        */
            parallel_for(LM_HID, gW1_range, &j);           /* dW1, db1 (race-free)      */
            parallel_for(bs, dX_range, &j);                /* dX                        */
            memset(gE, 0, (size_t)sE * sizeof(float));     /* scatter dX -> gE          */
            for (int b = 0; b < bs; b++) {
                const int *ctx = co->train + order[b0 + b]; const float *dx = dX + (size_t)b * LM_IN;
                for (int t = 0; t < CTX; t++) {
                    float *ge = gE + (size_t)ctx[t] * LM_EMB; const float *dxt = dx + t * LM_EMB;
                    for (int d = 0; d < LM_EMB; d++) ge[d] += dxt[d];
                }
            }

            m->tstep++;
            float bc1 = 1.0f - powf(ADAM_B1, (float)m->tstep);
            float bc2 = 1.0f - powf(ADAM_B2, (float)m->tstep);
            float ib = 1.0f / bs;
            gn2  = adam_step(m->E,  m->mE,  m->vE,  gE,  sE,  lr0, ib, bc1, bc2, LM_WD, tnorm, T);
            gn2 += adam_step(m->W1, m->mW1, m->vW1, gW1, sW1, lr0, ib, bc1, bc2, LM_WD, tnorm, T);
            gn2 += adam_step(m->b1, m->mb1, m->vb1, gb1, LM_HID, lr0, ib, bc1, bc2, 0.0f, tnorm, T);
            gn2 += adam_step(m->W2, m->mW2, m->vW2, gW2, sW2, lr0, ib, bc1, bc2, LM_WD, tnorm, T);
            gn2 += adam_step(m->b2, m->mb2, m->vb2, gb2, V,  lr0, ib, bc1, bc2, 0.0f, tnorm, T);
        }
        m->last_gnorm = sqrt(gn2);
        m->train_ppl  = ppl_indices(m, co->train, tidx, tprobe, X, H, logits);
        double val_ppl = ppl_indices(m, val_stream, vidx, vprobe, X, H, logits);

        int improved = val_ppl < best_val;
        if (improved) {
            best_val = val_ppl; best_ep = ep + 1;
            memcpy(bE, m->E, (size_t)sE * sizeof(float));
            memcpy(bW1, m->W1, (size_t)sW1 * sizeof(float)); memcpy(bb1, m->b1, LM_HID * sizeof(float));
            memcpy(bW2, m->W2, (size_t)sW2 * sizeof(float)); memcpy(bb2, m->b2, (size_t)V * sizeof(float));
        }
        printf("    epoch %2d/%2d   val ppl %7.2f%s   train ppl %7.2f (gap %+.0f)   |grad| %.3f   (%.1fs)\n",
               ep + 1, epochs, val_ppl, improved ? " *best" : "      ", m->train_ppl,
               val_ppl - m->train_ppl, m->last_gnorm, now_sec() - t0);
        fflush(stdout);
    }

    memcpy(m->E, bE, (size_t)sE * sizeof(float));
    memcpy(m->W1, bW1, (size_t)sW1 * sizeof(float)); memcpy(m->b1, bb1, LM_HID * sizeof(float));
    memcpy(m->W2, bW2, (size_t)sW2 * sizeof(float)); memcpy(m->b2, bb2, (size_t)V * sizeof(float));
    m->best_ep = best_ep; m->epochs_run = epochs; m->train_ex = nex;
    printf("    -> early stop: kept epoch %d (best val ppl %.2f)\n", best_ep, best_val);

    free(X); free(Hp); free(H); free(mask); free(logits); free(dH); free(dX);
    free(gE); free(gW1); free(gb1); free(gW2); free(gb2); free(tnorm);
    free(order); free(vidx); free(tidx);
    free(bE); free(bW1); free(bb1); free(bW2); free(bb2);
}

void llm_eval(const LLM *m, const int *stream, int ntok, double *ppl, double *acc) {
    int V = m->vocab, nex = corpus_examples(ntok);
    float *X = (float *)malloc((size_t)BATCH * LM_IN * sizeof(float));
    float *H = (float *)malloc((size_t)BATCH * LM_HID * sizeof(float));
    float *logits = (float *)malloc((size_t)BATCH * V * sizeof(float));
    int *idx = (int *)malloc((size_t)BATCH * sizeof(int));
    double nll = 0.0; long correct = 0;
    for (int b0 = 0; b0 < nex; b0 += BATCH) {
        int bs = (b0 + BATCH <= nex) ? BATCH : (nex - b0);
        for (int b = 0; b < bs; b++) idx[b] = b0 + b;
        probs_batch(m, stream, idx, bs, X, H, logits);
        for (int b = 0; b < bs; b++) {
            int y = stream[b0 + b + CTX]; const float *p = logits + (size_t)b * V;
            float py = p[y]; if (py < 1e-12f) py = 1e-12f;
            nll += -log((double)py);
            if (argmax(p, V) == y) correct++;
        }
    }
    *ppl = exp(nll / nex);
    *acc = (double)correct / nex;
    free(X); free(H); free(logits); free(idx);
}

void llm_next(const LLM *m, const int *ctx, float *probs) { forward1(m, ctx, probs); }

long llm_ops_per_pred(void) {
    return (long)LM_HID * LM_IN + (long)VOCAB * LM_HID;
}

void llm_free(LLM *m) {
    free(m->E); free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m->mE); free(m->vE); free(m->mW1); free(m->vW1); free(m->mb1); free(m->vb1);
    free(m->mW2); free(m->vW2); free(m->mb2); free(m->vb2);
    memset(m, 0, sizeof(*m));
}
