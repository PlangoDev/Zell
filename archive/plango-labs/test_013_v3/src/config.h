/* ─────────────────────────────────────────────────────────────────────────────
 * config.h — every v3 knob in one struct, mirroring test_013_v2/bf/config.py.
 * Defaults are SPEED-FIRST (the v3 goal): large n_classes (small S) and few
 * layers. Quality-config (4 layers, n_classes=256) is reachable via flags.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef BF_CONFIG_H
#define BF_CONFIG_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    /* context + token codes */
    int   ctx;            /* HD binding makes long ctx cheap */
    int   code_dim;       /* per-token code width = bound width */
    int   hd_binding;     /* 1 = bind+sum, 0 = concat (v1 behavior) */
    int   n_scales;       /* number of temporal decay scales */
    int   multiscale;     /* 1 = per-layer timescale: layer l sees context decayed
                           * at decay0/2^l, so deeper layers see longer range and
                           * the layers stop being identical (real hierarchy). */
    float decay0, decay1, decay2;   /* up to 3 decay rates (n_scales of them used) */

    /* the deep stack */
    int   n_layers;
    int   n_gran;         /* granules per layer (keep % topk_groups == 0) */
    int   fan_in;
    int   k_active;
    int   n_classes;      /* readout classes; large -> small S -> fast */
    int   relay_dim;

    /* two-level approx topk: many small groups + a few per group captures ~all of
     * the true top-K (low miss rate) at ~1 scan. (Group-max = K groups x1 is
     * faster but cruder; opt into it with --fast-topk when quality doesn't matter.) */
    int   topk_groups;
    int   topk_per_group;

    /* local learning */
    float lr;
    float lr_final;       /* lr is annealed (cosine) from lr to lr*lr_final over the
                           * run. 1.0 = constant (old behavior). <1 fixes the
                           * "more data hurts" delta-rule saturation at scale. */
    float wd;
    int   decay_every;
    float mix_lr;
    /* homeostatic granule load-balancing (DeepSeek aux-loss-free controller): a
     * per-granule selection bias nudged toward an even fire-rate. 0 = off. */
    int   balance;        /* 1 = enable the fire-rate controller on gbias */
    float balance_lr;     /* controller gain (gamma) */
    int   balance_every;  /* apply + reset the usage accumulator every N steps */

    /* batching */
    int   batch;
    int   step_chunk;
    long  block;
    long  train_tokens;

    /* competitive granule learning */
    int   learn_feat;
    int   feat_passes;
    int   feat_sample;
    float feat_eta;
    int   refine_every;
    int   refine_sample;
    float refine_eta;

    /* evaluation */
    long  eval_tokens;

    /* multi-gpu */
    int   n_gpus;         /* 1 or 2 */
    int   sync_every;     /* local-SGD averaging interval, in blocks */
    int   ww_fp16;        /* store/transport word head in fp16/bf16 */

    /* M2 speed paths (runtime-selectable so the safe M1 path stays a fallback) */
    int   fast_scatter;   /* 1 = atomic-free bucketed readout scatter, 0 = atomic */
    int   resident;       /* 1 = keep train shard in VRAM, sample windows on GPU */
    int   analytics;      /* 1 = run a short profiling burst for per-phase timing */
    int   use_cublas;     /* 1 = cuBLAS sgemm activation, 0 = custom kernel fallback */
    int   progress_every; /* print a progress line every N blocks (0 = auto) */
    int   exact_topk;     /* 1 = exact K-pass top-k (slow), 0 = two-level approx */
    int   fast_topk;      /* 1 = group-max top-k (fastest, cruder), 0 = two-level approx */
    float gran_frac;      /* <1 = each step only activates + top-k's a rotating
                           * fraction of granules (granule dropout): cuts the
                           * activate+topk bandwidth ~1/frac. Quality tradeoff. */
    int   neg_word;       /* >0 = sampled-softmax word head: update true + this many
                           * negatives instead of all S (cuts the word-head scatter
                           * ~S/(1+neg)x). Training only; eval stays full+exact. */
    int   boost;          /* 1 = residual boosting (layer l focuses on what l-1 missed) */
    float boost_wmin;     /* floor on the per-example boost weight */
    /* test-time adaptation ("hyper-fixation"): keep learning during eval on the
     * doc being read (predict-then-learn). The LLM is frozen; we are not. */
    int   adapt;          /* 1 = also run an adaptive eval pass */
    float adapt_lr;       /* learning rate for the test-time update */
    /* generation: autoregressive sampling to a token-id file (detok in Python) */
    int   gen_tokens;     /* >0 = generate this many tokens after eval */
    float gen_temp;       /* sampling temperature (0 = greedy/argmax) */
    char  gen_out[1024];  /* where to write generated token ids (json) */

    /* infini-gram interpolation (v15): mix a suffix-array n-gram into eval.
     * P = (1-lambda)*P_brain + lambda*P_ngram. ngram_path = the token .bin that
     * was indexed by tools/build_ngram.py (expects <path>.sa alongside). */
    char  ngram_path[1024];
    float ngram_lambda;

    /* checkpoint/resume (v15): save/load the LEARNED state (gwt, gbias, Wc, Ww, bc,
     * bw, mixture). Seed-derived fixed-random parts are regenerated by brain_init,
     * so a checkpoint is ~0.4GB not GBs. --load skips competitive feature learning. */
    char  save_path[1024];
    char  load_path[1024];

    unsigned long seed;

    /* IO paths (filled from meta.json / CLI) */
    char  meta_path[1024];
    char  out_path[1024];
    int   bench;          /* 1 = throughput benchmark mode (no eval) */
} Cfg;

static inline Cfg cfg_default(void) {
    Cfg c;
    memset(&c, 0, sizeof(c));
    c.ctx = 32; c.code_dim = 128; c.hd_binding = 1;
    c.n_scales = 3; c.decay0 = 0.20f; c.decay1 = 0.05f; c.decay2 = 0.0125f;
    c.multiscale = 0;
    c.n_layers = 2; c.n_gran = 12288; c.fan_in = 48; c.k_active = 64;
    c.n_classes = 256; c.relay_dim = 128;   /* ~sqrt(V): minimizes readout gather K*(C+V/C) AND the hierarchical quality tax */
    c.topk_groups = 256; c.topk_per_group = 2;
    c.lr = 0.3f; c.lr_final = 1.0f; c.wd = 1e-5f; c.decay_every = 200; c.mix_lr = 0.05f;
    c.balance = 0; c.balance_lr = 0.1f; c.balance_every = 200;
    c.batch = 8192; c.step_chunk = 2048; c.block = 1000000; c.train_tokens = 50000000;
    c.learn_feat = 1; c.feat_passes = 2; c.feat_sample = 40000; c.feat_eta = 0.05f;
    c.refine_every = 50; c.refine_sample = 4000; c.refine_eta = 0.01f;
    c.eval_tokens = 250000;
    c.n_gpus = 1; c.sync_every = 25; c.ww_fp16 = 1;
    c.fast_scatter = 1; c.resident = 1; c.analytics = 1;
    c.use_cublas = 1; c.progress_every = 1; c.exact_topk = 0;
    c.boost = 0; c.boost_wmin = 0.1f; c.fast_topk = 0;
    c.gran_frac = 1.0f; c.neg_word = 0;
    c.adapt = 0; c.adapt_lr = 0.1f;
    c.gen_tokens = 0; c.gen_temp = 0.8f; c.gen_out[0] = 0;
    c.ngram_path[0] = 0; c.ngram_lambda = 0.3f;
    c.save_path[0] = 0; c.load_path[0] = 0;
    c.seed = 13;
    c.meta_path[0] = 0; c.out_path[0] = 0; c.bench = 0;
    return c;
}

/* tiny CPU-OK smoke config to exercise the whole path */
static inline void cfg_smoke(Cfg *c) {
    c->n_layers = 2; c->n_gran = 1024; c->fan_in = 16; c->k_active = 16;
    c->n_classes = 64; c->relay_dim = 32; c->ctx = 8; c->code_dim = 32;
    c->n_scales = 2; c->topk_groups = 16; c->topk_per_group = 8;
    c->batch = 256; c->block = 50000; c->step_chunk = 256;
    c->train_tokens = 300000; c->eval_tokens = 20000;
    c->feat_sample = 8000; c->refine_every = 5; c->refine_sample = 1000;
    c->fast_scatter = 0;   /* smoke validates the simple atomic path first */
}

/* standalone-statement arg macros (no else-if chaining: robust, order-independent) */
#define ARG_INT(flag, field)   if (!strcmp(a, flag) && i+1 < argc) c.field = atoi(argv[++i]);
#define ARG_LONG(flag, field)  if (!strcmp(a, flag) && i+1 < argc) c.field = atol(argv[++i]);
#define ARG_FLT(flag, field)   if (!strcmp(a, flag) && i+1 < argc) c.field = (float)atof(argv[++i]);
#define ARG_STR(flag, field)   if (!strcmp(a, flag) && i+1 < argc) { strncpy(c.field, argv[++i], sizeof(c.field)-1); }

static inline Cfg cfg_parse(int argc, char **argv) {
    Cfg c = cfg_default();
    int smoke = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--smoke")) smoke = 1;
        if (!strcmp(a, "--bench")) c.bench = 1;
        ARG_STR("--meta", meta_path);
        ARG_STR("--out", out_path);
        ARG_INT("--n-layers", n_layers);
        ARG_INT("--n-gran", n_gran);
        ARG_INT("--fan-in", fan_in);
        ARG_INT("--k-active", k_active);
        ARG_INT("--n-classes", n_classes);
        ARG_INT("--ctx", ctx);
        ARG_INT("--code-dim", code_dim);
        ARG_INT("--multiscale", multiscale);
        ARG_INT("--relay-dim", relay_dim);
        ARG_INT("--topk-groups", topk_groups);
        ARG_INT("--topk-per-group", topk_per_group);
        ARG_INT("--batch", batch);
        ARG_INT("--step-chunk", step_chunk);
        ARG_LONG("--block", block);
        ARG_LONG("--train-tokens", train_tokens);
        ARG_LONG("--eval-tokens", eval_tokens);
        ARG_INT("--n-gpus", n_gpus);
        ARG_INT("--sync-every", sync_every);
        ARG_INT("--ww-fp16", ww_fp16);
        ARG_INT("--fast-scatter", fast_scatter);
        ARG_INT("--resident", resident);
        ARG_INT("--analytics", analytics);
        ARG_INT("--cublas", use_cublas);
        ARG_INT("--progress-every", progress_every);
        ARG_INT("--exact-topk", exact_topk);
        ARG_INT("--fast-topk", fast_topk);
        ARG_INT("--neg-word", neg_word);
        ARG_FLT("--gran-frac", gran_frac);
        ARG_INT("--boost", boost);
        ARG_FLT("--boost-wmin", boost_wmin);
        ARG_INT("--adapt", adapt);
        ARG_FLT("--adapt-lr", adapt_lr);
        ARG_INT("--gen-tokens", gen_tokens);
        ARG_FLT("--gen-temp", gen_temp);
        ARG_STR("--gen-out", gen_out);
        ARG_STR("--ngram", ngram_path);
        ARG_FLT("--ngram-lambda", ngram_lambda);
        ARG_STR("--save", save_path);
        ARG_STR("--load", load_path);
        ARG_INT("--refine-every", refine_every);
        ARG_FLT("--lr", lr);
        ARG_FLT("--lr-final", lr_final);
        ARG_FLT("--wd", wd);
        ARG_INT("--decay-every", decay_every);
        ARG_INT("--balance", balance);
        ARG_FLT("--balance-lr", balance_lr);
        ARG_INT("--balance-every", balance_every);
        ARG_INT("--seed", seed);
    }
    /* value-less flags */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-feat")) c.learn_feat = 0;
        if (!strcmp(argv[i], "--no-refine")) c.refine_every = 0;
    }
    if (smoke) cfg_smoke(&c);
    return c;
}

#endif /* BF_CONFIG_H */
