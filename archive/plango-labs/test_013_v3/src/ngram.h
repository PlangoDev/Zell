/* ─────────────────────────────────────────────────────────────────────────────
 * ngram.h — infini-gram next-token probability from a suffix array (host/CPU).
 *
 * Built by tools/build_ngram.py (suffix array over the uint16 corpus). At eval we
 * compute, for each position, P_ngram(y | context) from the LONGEST suffix of the
 * context that occurs in the corpus, and interpolate it with the brain's mixture
 * probability:  P = (1-lambda)*P_brain + lambda*P_ngram.  This is the v15 research's
 * highest-EV quality lever (web-verified ~42% rel ppl cut interpolated with a neural
 * LM). Pure host code (Linux/POSIX mmap, matching data.c); eval is not throughput-
 * critical, so a CPU suffix-array query per position is fine.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef BF_NGRAM_H
#define BF_NGRAM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct {
    const uint16_t *toks;   /* mmapped corpus tokens [n] */
    const int64_t  *sa;     /* mmapped suffix array [n] */
    long n;                 /* number of tokens */
    int  max_match;         /* cap on suffix length probed */
    int  vocab;             /* for Laplace smoothing denominator */
    int  scan_cap;          /* max corpus positions scanned per query */
    size_t toks_bytes, sa_bytes;
} Ngram;

/* mmap a file read-only; returns NULL on failure and sets *len to 0. */
static const void *ng_map(const char *path, size_t *len) {
    *len = 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return NULL; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *len = (size_t)st.st_size;
    return p;
}

/* open <token_path> + <token_path>.sa (+ optional .sa.json for n/max_match/vocab). */
static Ngram *ngram_open(const char *token_path, int vocab_hint) {
    char sapath[1100];
    snprintf(sapath, sizeof(sapath), "%s.sa", token_path);
    Ngram *g = (Ngram *)calloc(1, sizeof(Ngram));
    g->toks = (const uint16_t *)ng_map(token_path, &g->toks_bytes);
    g->sa   = (const int64_t  *)ng_map(sapath, &g->sa_bytes);
    if (!g->toks || !g->sa) {
        fprintf(stderr, "  ngram: failed to map %s / %s\n", token_path, sapath);
        free(g); return NULL;
    }
    g->n = (long)(g->sa_bytes / sizeof(int64_t));
    long ntok = (long)(g->toks_bytes / sizeof(uint16_t));
    if (ntok < g->n) g->n = ntok;          /* be safe if token file is longer */
    g->max_match = 16;
    g->vocab = vocab_hint > 0 ? vocab_hint : 50277;
    g->scan_cap = 8192;
    /* refine from the json sidecar if present (best-effort, simple parse) */
    char jpath[1160]; snprintf(jpath, sizeof(jpath), "%s.json", sapath);
    FILE *jf = fopen(jpath, "r");
    if (jf) {
        char buf[4096]; size_t r = fread(buf, 1, sizeof(buf) - 1, jf); buf[r] = 0; fclose(jf);
        const char *m = strstr(buf, "\"max_match\"");
        if (m) { const char *c = strchr(m, ':'); if (c) { int v = atoi(c + 1); if (v > 0 && v <= 64) g->max_match = v; } }
        const char *vv = strstr(buf, "\"vocab\"");
        if (vv) { const char *c = strchr(vv, ':'); if (c) { int v = atoi(c + 1); if (v > 0) g->vocab = v; } }
    }
    fprintf(stderr, "  ngram: opened %ld-token suffix array (max_match=%d, vocab=%d)\n",
            g->n, g->max_match, g->vocab);
    return g;
}

static void ngram_close(Ngram *g) {
    if (!g) return;
    if (g->toks) munmap((void *)g->toks, g->toks_bytes);
    if (g->sa)   munmap((void *)g->sa, g->sa_bytes);
    free(g);
}

/* compare the suffix at `start` against pattern `pat[0..plen)`:
 *  <0 suffix sorts before pat, 0 pat is a prefix of the suffix, >0 after. */
static inline int ng_cmp(const uint16_t *toks, long n, long start, const int *pat, int plen) {
    for (int j = 0; j < plen; j++) {
        long p = start + j;
        if (p >= n) return -1;                 /* suffix shorter -> smaller */
        int a = (int)toks[p], b = pat[j];
        if (a != b) return a < b ? -1 : 1;
    }
    return 0;
}

/* first SA index whose suffix is >= pat (upper=0) or > pat-as-prefix (upper=1). */
static long ng_lb(const Ngram *g, const int *pat, int plen, int upper) {
    long lo = 0, hi = g->n;
    while (lo < hi) {
        long mid = lo + ((hi - lo) >> 1);
        int c = ng_cmp(g->toks, g->n, g->sa[mid], pat, plen);
        if (c < 0 || (upper && c == 0)) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/* infini-gram P(y | ctx): longest suffix of ctx that occurs in the corpus, Laplace-
 * smoothed over the next tokens at the matching positions (count-capped sample). */
static double ngram_prob(const Ngram *g, const int *ctx, int ctxlen, int y) {
    int Lmax = ctxlen < g->max_match ? ctxlen : g->max_match;
    for (int ln = Lmax; ln >= 1; ln--) {
        const int *pat = ctx + (ctxlen - ln);
        long lo = ng_lb(g, pat, ln, 0);
        long hi = ng_lb(g, pat, ln, 1);
        long cnt = hi - lo;
        if (cnt < 1) continue;                 /* this suffix never occurs; back off */
        long step = 1;
        if (cnt > g->scan_cap) step = cnt / g->scan_cap;   /* strided subsample of a huge range */
        long yes = 0, tot = 0;
        for (long i = lo; i < hi; i += step) {
            long p = g->sa[i] + ln;
            if (p < g->n) { tot++; if ((int)g->toks[p] == y) yes++; }
        }
        /* infini-gram estimate: trust the longest match (epsilon floor only). MUST
         * match build_ngram.py:ngram_prob_ref exactly so the local validation holds. */
        if (tot > 0)
            return ((double)yes + 1e-6) / ((double)tot + 1e-6 * 65536.0);
    }
    return 1.0 / 65536.0;                       /* nothing matched: uniform */
}

#endif /* BF_NGRAM_H */
