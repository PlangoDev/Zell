/* ─────────────────────────────────────────────────────────────────────────────
 * analytics.cpp — the diagnostics, implemented. Pure measurement and printing.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "analytics.h"
#include "common.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <unordered_map>

#define ANA_CAP 8000     /* diagnostics run on a representative slice for speed */

TopkReport ana_eval(NextFn next, const int *stream, int ntok, int vocab) {
    int nex = corpus_examples(ntok);
    if (nex > ANA_CAP) nex = ANA_CAP;
    double nll = 0.0; long t1 = 0, t5 = 0, t10 = 0;
    std::vector<float> p(vocab);
    for (int e = 0; e < nex; e++) {
        next(stream + e, p.data());
        int y = stream[e + CTX];
        float py = p[y]; if (py < 1e-12f) py = 1e-12f;
        nll += -log((double)py);
        /* rank of the true word = how many words beat it */
        int better = 0;
        for (int v = 0; v < vocab; v++) if (p[v] > py) better++;
        if (better < 1)  t1++;
        if (better < 5)  t5++;
        if (better < 10) t10++;
    }
    TopkReport r;
    r.ppl  = exp(nll / nex);
    r.bits = (nll / nex) / log(2.0);
    r.top1 = (double)t1 / nex; r.top5 = (double)t5 / nex; r.top10 = (double)t10 / nex;
    return r;
}

/* ── count-based n-gram baselines ─────────────────────────────────────────── */
#define ADD_K 0.05

void ana_ngram_baselines(const Corpus *co) {
    int V = co->vocab;
    const int *tr = co->train; int ntr = co->ntrain;

    /* unigram counts */
    std::vector<long> uni(V, 0);
    for (int i = 0; i < ntr; i++) uni[tr[i]]++;
    long N = ntr;

    /* bigram + trigram counts (sparse, in hash maps) */
    std::unordered_map<uint64_t, int> bi, tri;
    bi.reserve(ntr); tri.reserve(ntr);
    for (int i = 0; i + 1 < ntr; i++)
        bi[(uint64_t)tr[i] * V + tr[i + 1]]++;
    for (int i = 0; i + 2 < ntr; i++)
        tri[((uint64_t)tr[i] * V + tr[i + 1]) * V + tr[i + 2]]++;

    /* score every test example under each model */
    const int *te = co->test; int nex = corpus_examples(co->ntest);
    double nll_u = 0, nll_b = 0, nll_t = 0, nll_i = 0; long seen_ctx = 0;
    const double l1 = 0.1, l2 = 0.3, l3 = 0.6;          /* interpolation weights */

    for (int e = 0; e < nex; e++) {
        int wl = te[e + CTX - 1];          /* last context word (predicts y)        */
        int wp = te[e + CTX - 2];          /* second-last (the trigram's first slot) */
        int y  = te[e + CTX];

        double pu = (uni[y] + ADD_K) / (N + ADD_K * V);

        long cwl = uni[wl];
        auto itb = bi.find((uint64_t)wl * V + y);
        long cb = (itb == bi.end()) ? 0 : itb->second;
        double pb = (cb + ADD_K) / (cwl + ADD_K * V);

        auto itc = bi.find((uint64_t)wp * V + wl);          /* trigram context count */
        long cctx = (itc == bi.end()) ? 0 : itc->second;
        auto itt = tri.find(((uint64_t)wp * V + wl) * V + y);
        long ct = (itt == tri.end()) ? 0 : itt->second;
        double pt = (ct + ADD_K) / (cctx + ADD_K * V);
        if (cctx > 0) seen_ctx++;

        double pi = l1 * pu + l2 * pb + l3 * pt;

        nll_u += -log(pu); nll_b += -log(pb); nll_t += -log(pt); nll_i += -log(pi);
    }

    printf("  N-GRAM BASELINES  (count-based, add-%.2f smoothed — the bar to beat)\n", ADD_K);
    printf("    %-26s %s\n", "model", "test perplexity");
    printf("    %-26s %.2f\n", "unigram (word freq)",     exp(nll_u / nex));
    printf("    %-26s %.2f\n", "bigram  (prev 1 word)",   exp(nll_b / nex));
    printf("    %-26s %.2f\n", "trigram (prev 2 words)",  exp(nll_t / nex));
    printf("    %-26s %.2f\n", "interpolated 1+2+3",      exp(nll_i / nex));
    printf("    trigram context seen in train: %.1f%% of test cases\n",
           100.0 * seen_ctx / nex);
    printf("\n");
}

/* ── head-to-head on the test set ─────────────────────────────────────────── */
void ana_head_to_head(NextFn llm_next, NextFn brain_next, const Corpus *co) {
    int V = co->vocab, nex = corpus_examples(co->ntest);
    if (nex > ANA_CAP) nex = ANA_CAP;
    const int *te = co->test;
    std::vector<float> pl(V), pb(V);
    long llm_win = 0, brain_win = 0, tie = 0, agree = 0, both_right = 0;

    for (int e = 0; e < nex; e++) {
        llm_next(te + e, pl.data());
        brain_next(te + e, pb.data());
        int y = te[e + CTX];
        float a = pl[y], b = pb[y];
        if (a > b * 1.001f)      llm_win++;
        else if (b > a * 1.001f) brain_win++;
        else                     tie++;
        int la = argmax(pl.data(), V), ba = argmax(pb.data(), V);
        if (la == ba) { agree++; if (la == y) both_right++; }
    }
    printf("  HEAD-TO-HEAD  (probability mass each model puts on the TRUE next word)\n");
    printf("    LLM more confident on truth : %6.2f%%\n", 100.0 * llm_win / nex);
    printf("    Brain more confident on truth: %6.2f%%\n", 100.0 * brain_win / nex);
    printf("    effectively tied            : %6.2f%%\n", 100.0 * tie / nex);
    printf("    top-1 picks agree           : %6.2f%%  (both correct %.2f%%)\n",
           100.0 * agree / nex, 100.0 * both_right / nex);
    printf("\n");
}
