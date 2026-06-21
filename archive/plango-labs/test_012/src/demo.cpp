/* ─────────────────────────────────────────────────────────────────────────────
 * demo.cpp — watch the Brain actually read.
 *
 * Trains the cerebellar coder (no backprop) on Tiny Shakespeare, then shows what
 * it learned on real words: given a 3-word context it prints its top-5 next-word
 * guesses with probabilities, and continues a few seed phrases in its own words.
 *
 * Build:  c++ -O3 -ffast-math -march=native -std=c++17 src/demo.cpp src/corpus.cpp \
 *             src/brain.cpp src/threads.cpp -o demo -pthread
 * Run:    ./demo      (from the test_012 directory, needs data/corpus.txt)
 * ───────────────────────────────────────────────────────────────────────────*/
#include "common.h"
#include "config.h"
#include "corpus.h"
#include "brain.h"
#include "threads.h"

#include <cstdio>
#include <cstring>
#include <cctype>

#define DEMO_EPOCHS  20
#define DEMO_LR      0.30f

/* map a word to its vocabulary id (lowercased, linear scan; 0 = <unk>) */
static int word_id(const Corpus *co, const char *w) {
    char low[64]; int i = 0;
    for (; w[i] && i < 63; i++) low[i] = (char)tolower((unsigned char)w[i]);
    low[i] = 0;
    for (int v = 0; v < co->vocab; v++)
        if (strcmp(co->word[v], low) == 0) return v;
    return 0;
}

/* turn "to be or" into a CTX-long context of ids; prints how it resolved */
static void parse_ctx(const Corpus *co, const char *phrase, int *ctx) {
    char buf[256]; strncpy(buf, phrase, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
    int n = 0; char *tok = strtok(buf, " ");
    int tmp[64];
    while (tok && n < 64) { tmp[n++] = word_id(co, tok); tok = strtok(NULL, " "); }
    /* right-align the last CTX words into ctx (pad with <unk> on the left) */
    for (int t = 0; t < CTX; t++) {
        int src = n - CTX + t;
        ctx[t] = (src >= 0) ? tmp[src] : 0;
    }
}

/* print the top-k next-word predictions for a context */
static void show_topk(const Corpus *co, const Brain *br, const char *phrase, int k) {
    int ctx[CTX]; parse_ctx(co, phrase, ctx);
    float p[VOCAB]; brain_next(br, ctx, p);

    printf("  \"%s\"  ->\n", phrase);
    for (int rank = 0; rank < k; rank++) {
        int best = 0; float bv = -1.0f;
        for (int v = 0; v < co->vocab; v++) if (p[v] > bv) { bv = p[v]; best = v; }
        int bar = (int)(bv * 40.0f + 0.5f);
        printf("      %-14s %5.1f%%  ", co->word[best], bv * 100.0f);
        for (int b = 0; b < bar; b++) putchar('#');
        putchar('\n');
        p[best] = -1.0f;                          /* remove and find the next */
    }
    printf("\n");
}

/* continue a seed phrase, sampling from the Brain's distribution */
static void generate(const Corpus *co, const Brain *br, const char *seed, int nwords, uint64_t s) {
    int ctx[CTX]; parse_ctx(co, seed, ctx);
    rng_t r; rng_seed(&r, s);
    printf("  %s |", seed);
    float p[VOCAB];
    for (int w = 0; w < nwords; w++) {
        brain_next(br, ctx, p);
        double u = rng_unif(&r), c = 0.0; int nx = co->vocab - 1;
        for (int v = 0; v < co->vocab; v++) { c += p[v]; if (u <= c) { nx = v; break; } }
        printf(" %s", co->word[nx]);
        for (int t = 0; t < CTX - 1; t++) ctx[t] = ctx[t + 1];
        ctx[CTX - 1] = nx;
    }
    printf("\n");
}

int main(void) {
    threads_init(0);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\n====================================================================\n");
    printf("  THE BRAIN READS  —  cerebellar coder, no backprop (%d cores)\n", threads_count());
    printf("====================================================================\n");

    Corpus co = corpus_load("data/corpus.txt");
    printf("  trained on Tiny Shakespeare: %ld words, vocab %d\n\n", co.total_tok, co.vocab);

    printf("  training the brain (delta-rule readout, no backprop):\n");
    Brain br; brain_init(&br, &co, 100);
    double t0 = now_sec();
    brain_train(&br, &co, DEMO_EPOCHS, DEMO_LR);
    printf("  trained in %.1fs\n\n", now_sec() - t0);

    printf("====================================================================\n");
    printf("  NEXT-WORD PREDICTION  (top-5 guesses for a 3-word context)\n");
    printf("====================================================================\n");
    const char *prompts[] = {
        "to be or", "i pray you", "my good lord", "what is the",
        "i will not", "the king is", "let us go", "i am the",
    };
    for (unsigned i = 0; i < sizeof(prompts) / sizeof(prompts[0]); i++)
        show_topk(&co, &br, prompts[i], 5);

    printf("====================================================================\n");
    printf("  GENERATION  (the brain continues a seed, in its own words)\n");
    printf("====================================================================\n");
    generate(&co, &br, "to be or", 24, 7);
    generate(&co, &br, "my lord the", 24, 19);
    generate(&co, &br, "i love thee", 24, 33);
    printf("====================================================================\n\n");

    brain_free(&br);
    corpus_free(&co);
    return 0;
}
