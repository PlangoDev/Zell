/* ─────────────────────────────────────────────────────────────────────────────
 * common.h — small shared helpers used by every part of the showdown:
 *   - a fast pseudo-random number generator (no libc rand(), which is slow)
 *   - a wall-clock timer
 *   - softmax + argmax used by both contestants
 *
 * Header-only (everything is `static inline`) so there is no common.c to link.
 * Commented heavily on purpose — this file is meant to be read top to bottom.
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

/* ── Dataset shape constants (one place to change them) ───────────────────── */
#define IMG_W    28            /* full-resolution 28x28 Fashion-MNIST           */
#define IMG_H    28
#define IMG_DIM  (IMG_W * IMG_H)   /* 196 pixels per image                      */
#define NCLASS   10            /* ten kinds of clothing                          */

/* ── Fast RNG: xorshift64*. One 64-bit word of state, must be non-zero. ─────
 * Plenty random for shuffling, weight init and patch sampling, and it is a
 * handful of instructions per number — far faster than the C library rand(). */
typedef struct { uint64_t s; } rng_t;

static inline void rng_seed(rng_t *r, uint64_t seed) {
    r->s = seed ? seed : 0x9E3779B97F4A7C15ull;   /* never allow a zero state   */
}

static inline uint64_t rng_u64(rng_t *r) {
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* uniform double in [0,1) */
static inline double rng_unif(rng_t *r) {
    return (rng_u64(r) >> 11) * (1.0 / 9007199254740992.0);   /* 53-bit mantissa */
}

/* uniform integer in [0, n) */
static inline int rng_int(rng_t *r, int n) {
    return (int)(rng_u64(r) % (uint64_t)n);
}

/* standard-normal sample via Box–Muller (used for weight initialization) */
static inline double rng_gauss(rng_t *r) {
    double u1 = rng_unif(r), u2 = rng_unif(r);
    if (u1 < 1e-12) u1 = 1e-12;                    /* guard log(0)               */
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ── Wall-clock seconds, monotonic. For timing each contestant. ───────────── */
static inline double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* ── Softmax over `n` logits, in place. Subtract the max first for stability. */
static inline void softmax(float *z, int n) {
    float m = z[0];
    for (int i = 1; i < n; i++) if (z[i] > m) m = z[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { z[i] = expf(z[i] - m); sum += z[i]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < n; i++) z[i] *= inv;
}

/* ── Index of the largest value (the predicted class). ────────────────────── */
static inline int argmax(const float *z, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (z[i] > z[best]) best = i;
    return best;
}

#endif /* COMMON_H */
