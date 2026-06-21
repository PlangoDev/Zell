/* ─────────────────────────────────────────────────────────────────────────────
 * common.h — small shared helpers (same toolbox test_011 used, minus the image
 * constants): a fast RNG, a wall-clock timer, and a numerically-stable softmax /
 * argmax that work over any length n.  Header-only (all `static inline`).
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Tell the compiler hot-loop pointers don't alias, so it can vectorize freely. */
#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

/* ── Fast RNG: xorshift64*. One 64-bit word of state, must be non-zero. ───────
 * A handful of instructions per number — far faster than libc rand(). */
typedef struct { uint64_t s; } rng_t;

static inline void rng_seed(rng_t *r, uint64_t seed) {
    r->s = seed ? seed : 0x9E3779B97F4A7C15ull;
}
static inline uint64_t rng_u64(rng_t *r) {
    uint64_t x = r->s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1Dull;
}
static inline double rng_unif(rng_t *r) {            /* uniform in [0,1)         */
    return (rng_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}
static inline int rng_int(rng_t *r, int n) {         /* uniform in [0,n)         */
    return (int)(rng_u64(r) % (uint64_t)n);
}
static inline double rng_gauss(rng_t *r) {           /* standard normal          */
    double u1 = rng_unif(r), u2 = rng_unif(r);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ── Wall-clock seconds, monotonic. ───────────────────────────────────────── */
static inline double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* ── Softmax over n logits, in place. Subtract the max first for stability. ── */
static inline void softmax(float *z, int n) {
    float m = z[0];
    for (int i = 1; i < n; i++) if (z[i] > m) m = z[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { z[i] = expf(z[i] - m); sum += z[i]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < n; i++) z[i] *= inv;
}

/* ── Index of the largest value. ──────────────────────────────────────────── */
static inline int argmax(const float *z, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (z[i] > z[best]) best = i;
    return best;
}

#endif /* COMMON_H */
