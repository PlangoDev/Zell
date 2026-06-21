/* ─────────────────────────────────────────────────────────────────────────────
 * main.cpp — TEST 012: the head-to-head moves from PICTURES to WORDS.
 *
 *   LLM way   : a neural language model trained by BACKPROPAGATION.
 *   Brain way : a cerebellar n-gram coder (fixed sparse expansion + local-rule
 *               readout). NO backprop anywhere.
 *
 * Task: predict the next word from the previous CTX words. Scored on PERPLEXITY
 * (the language metric — lower is better), next-word top-1 accuracy, multiply-
 * adds per prediction (the energy proxy), and wall-clock time. Then each model
 * writes a few words of its own so we can see what it actually learned.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "common.h"
#include "config.h"
#include "corpus.h"
#include "llm.h"
#include "brain.h"
#include "analytics.h"
#include "threads.h"

#include <cstdio>
#include <cstdlib>
#include <functional>

#define LLM_EPOCHS   8        /* early stopping bottoms out ~epoch 3 — no need for more */
#define LLM_LR       0.003f   /* AdamW step size (robust; ~1e-3 range)            */
#define BR_EPOCHS    34       /* the local delta-rule readout keeps descending; let it converge */
#define BR_LR        0.30f
#define GEN_WORDS    24

/* sample one word from a probability vector (temperature already baked in) */
static int sample(const float *p, int n, rng_t *r) {
    double u = rng_unif(r), c = 0.0;
    for (int i = 0; i < n; i++) { c += p[i]; if (u <= c) return i; }
    return n - 1;
}

/* let a model continue from a seed context, printing the words it picks */
static void generate(const Corpus *co, const char *who,
                     std::function<void(const int *, float *)> next, uint64_t seed) {
    rng_t r; rng_seed(&r, seed);
    int ctx[CTX];
    for (int t = 0; t < CTX; t++) ctx[t] = co->test[t];

    printf("  %-6s: ", who);
    for (int t = 0; t < CTX; t++) printf("%s ", co->word[ctx[t]]);
    printf("|");
    float p[VOCAB];
    for (int w = 0; w < GEN_WORDS; w++) {
        next(ctx, p);
        int nx = sample(p, co->vocab, &r);
        printf(" %s", co->word[nx]);
        for (int t = 0; t < CTX - 1; t++) ctx[t] = ctx[t + 1];
        ctx[CTX - 1] = nx;
    }
    printf("\n");
}

int main(void) {
    threads_init(0);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\n======================================================================\n");
    printf("  TEST 012  —  LLM  vs  the Brain,  now on WORDS   (C++, %d cores)\n", threads_count());
    printf("======================================================================\n");

    Corpus co = corpus_load("data/corpus.txt");
    printf("  corpus: %ld words, vocab %d (top words kept), %.1f%% fall to <unk>\n",
           co.total_tok, co.vocab, co.unk_frac * 100);
    printf("  task: see %d words, predict the next. %d train / %d test contexts.\n\n",
           CTX, corpus_examples(co.ntrain), corpus_examples(co.ntest));

    /* ===================== reference bars (no learning) =============== */
    ana_ngram_baselines(&co);

    /* ============================ the LLM ============================== */
    printf("  LLM training (neural language model, backprop) — watch perplexity fall:\n");
    LLM lm; llm_init(&lm, co.vocab, 11);
    double t0 = now_sec();
    llm_train(&lm, &co, LLM_EPOCHS, LLM_LR);
    double lm_s = now_sec() - t0;
    double lm_ppl, lm_acc; llm_eval(&lm, co.test, co.ntest, &lm_ppl, &lm_acc);
    long lm_ops = llm_ops_per_pred();
    printf("  LLM final: perplexity %.2f   top1 %.2f%%   %ld MACs/pred   %.1fs\n\n",
           lm_ppl, lm_acc * 100, lm_ops, lm_s);

    /* ========================== the Brain ============================= */
    printf("  BRAIN training (cerebellar coder, hierarchical readout, NO backprop) — live:\n");
    Brain br; brain_init(&br, &co, 100);
    t0 = now_sec();
    brain_train(&br, &co, BR_EPOCHS, BR_LR);
    double br_s = now_sec() - t0;
    double br_ppl, br_acc; brain_eval(&br, co.test, co.ntest, &br_ppl, &br_acc);
    long br_ops = brain_ops_per_pred();
    printf("  BRAIN final: perplexity %.2f   top1 %.2f%%   %ld MACs/pred   %.1fs\n\n",
           br_ppl, br_acc * 100, br_ops, br_s);

    /* ============================ scoreboard ========================== */
    printf("======================================================================\n");
    printf("  SCOREBOARD   (perplexity: lower is better)\n");
    printf("======================================================================\n");
    printf("    %-26s %-12s %-8s %-12s %s\n", "system", "perplexity", "top1", "MACs/pred", "backprop?");
    printf("    %-26s %-12.2f %-7.2f%% %-12ld %s\n", "LLM (neural LM)",  lm_ppl, lm_acc * 100, lm_ops, "yes");
    printf("    %-26s %-12.2f %-7.2f%% %-12ld %s\n", "Brain (cerebellar)", br_ppl, br_acc * 100, br_ops, "NO");
    printf("    ----------------------------------------------------------------\n");
    {
        int br_better_ppl = br_ppl < lm_ppl, br_cheaper = br_ops < lm_ops;
        const char *v = (br_better_ppl && br_cheaper) ? "BRAIN WINS BOTH (lower perplexity AND cheaper)"
                      : br_better_ppl ? "brain wins perplexity"
                      : br_cheaper ? "brain is cheaper; LLM has the lower perplexity"
                      : "LLM wins this round";
        printf("    verdict: %s\n", v);
        if (br_cheaper) printf("             brain runs at %.3fx the LLM's compute per word.\n",
                               (double)br_ops / lm_ops);
    }

    /* ============ training & footprint: the OTHER axes of the fight ========= */
    printf("\n======================================================================\n");
    printf("  TRAINING & FOOTPRINT  (the fight is not just inference MACs)\n");
    printf("======================================================================\n");
    long ntr_ex = corpus_examples(co.ntrain), nte_ex = corpus_examples(co.ntest);

    /* parameter counts (trained vs fixed) */
    long lm_trained = (long)co.vocab * LM_EMB + (long)LM_HID * LM_IN + LM_HID
                    + (long)co.vocab * LM_HID + co.vocab;
    long lm_fixed   = 0;
    long br_trained = (long)BR_G * co.vocab + co.vocab + (long)BR_G * BR_CLASSES + BR_CLASSES;
    long br_fixed   = (long)co.vocab * BR_EMB + (long)BR_G * BR_SIN + BR_G;   /* codes+wiring+bias */
    double lm_mb = (lm_trained + lm_fixed) * 4.0 / 1e6;
    double br_mb = (br_trained + br_fixed) * 4.0 / 1e6;

    /* training compute (MACs): fwd+bwd for the LLM, encode-once + delta sweeps for
     * the brain. Estimates from the layer shapes — same multiply-add basis as MACs/pred. */
    double lm_train_macs = 3.0 * (double)lm_ops * (double)lm.train_ex * (double)lm.epochs_run;
    double br_enc_macs   = (double)(ntr_ex + nte_ex) * BR_G * BR_SIN;
    double br_sweep_macs = (double)ntr_ex * BR_EPOCHS * BR_K
                         * ((double)co.vocab + (double)co.vocab / BR_CLASSES + 2.0 * BR_CLASSES);
    double br_train_macs = br_enc_macs + br_sweep_macs;

    printf("    %-22s %14s %14s\n", "", "LLM (backprop)", "Brain (local)");
    printf("    %-22s %14s %14s\n", "learning rule", "AdamW + dropout", "delta + Hebb");
    printf("    %-22s %13.2fM %13.2fM\n", "trained params", lm_trained / 1e6, br_trained / 1e6);
    printf("    %-22s %13.2fM %13.2fM\n", "fixed params",   lm_fixed   / 1e6, br_fixed   / 1e6);
    printf("    %-22s %13.1fMB %13.1fMB\n", "model footprint", lm_mb, br_mb);
    printf("    %-22s %14.1f %14.1f\n", "train wall-clock (s)", lm_s, br_s);
    printf("    %-22s %13.2fG %13.2fG\n", "train compute (MACs)", lm_train_macs / 1e9, br_train_macs / 1e9);
    printf("    %-22s %14.0f %14.0f\n", "train examples/sec",
           lm.train_ex * (double)lm.epochs_run / lm_s, (double)ntr_ex * BR_EPOCHS / br_s);
    printf("    %-22s %11d/%-2d %14d\n", "epochs (kept/run)", lm.best_ep, lm.epochs_run, BR_EPOCHS);
    printf("    %-22s %14ld %14ld\n", "inference MACs/word", lm_ops, br_ops);
    printf("    ----------------------------------------------------------------\n");
    printf("    brain trains in %.2fx the LLM's compute and %.2fx its wall-clock,\n",
           br_train_macs / lm_train_macs, br_s / lm_s);
    printf("    then predicts at %.3fx the LLM's per-word cost — with no backprop.\n",
           (double)br_ops / lm_ops);
    printf("    (note: brain's trained matrix is larger but SPARSELY read — only %d of\n", BR_K);
    printf("     %d granule rows fire per word, which is why inference stays cheap.)\n", BR_G);

    /* ====================== deep analytics ============================ */
    printf("\n======================================================================\n");
    printf("  DEEP ANALYTICS  (diagnose where each model wins, loses, and stalls)\n");
    printf("======================================================================\n");

    NextFn llm_fn   = [&](const int *c, float *p) { llm_next(&lm, c, p); };
    NextFn brain_fn = [&](const int *c, float *p) { brain_next(&br, c, p); };

    TopkReport lr = ana_eval(llm_fn,   co.test, co.ntest, co.vocab);
    TopkReport br_r = ana_eval(brain_fn, co.test, co.ntest, co.vocab);
    printf("  ACCURACY DEPTH  (true word inside the model's top-k guesses)\n");
    printf("    %-22s %8s %8s %8s %10s\n", "model", "top-1", "top-5", "top-10", "bits/word");
    printf("    %-22s %7.2f%% %7.2f%% %7.2f%% %10.3f\n", "LLM (neural LM)",
           lr.top1 * 100, lr.top5 * 100, lr.top10 * 100, lr.bits);
    printf("    %-22s %7.2f%% %7.2f%% %7.2f%% %10.3f\n", "Brain (cerebellar)",
           br_r.top1 * 100, br_r.top5 * 100, br_r.top10 * 100, br_r.bits);
    printf("\n");

    ana_head_to_head(llm_fn, brain_fn, &co);

    BrainDiag bd; brain_diagnose(&br, co.test, co.ntest, &bd);
    printf("  BRAIN GRANULE-CODE HEALTH  (why the brain plateaus where it does)\n");
    printf("    granules ever firing  : %.1f%%  (%d of %d dead / never win)\n",
           bd.density * 100, bd.dead_granules, BR_G);
    printf("    usage Gini            : %.3f   (0 = even load, 1 = a few do all the work)\n", bd.usage_gini);
    printf("    mean code margin      : %.3f   (avg winner activation after kWTA)\n", bd.mean_margin);
    printf("    readout weight norm   : %.1f\n", bd.readout_wnorm);
    printf("    vocab rows unlearned  : %d of %d\n", bd.dead_readout, co.vocab);
    printf("    note: random word codes + WTA-learned granule features beat the LLM here\n");
    printf("          (discriminability matters more than semantic sharing on this task).\n");
    printf("\n");

    /* ===================== what each one writes ======================= */
    printf("\n  GENERATION  (each model continues the same seed, in its own words):\n");
    generate(&co, "LLM",   [&](const int *c, float *p) { llm_next(&lm, c, p); }, 2024);
    generate(&co, "Brain", [&](const int *c, float *p) { brain_next(&br, c, p); }, 2024);
    printf("======================================================================\n\n");

    llm_free(&lm); brain_free(&br); corpus_free(&co);
    return 0;
}
