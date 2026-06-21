/* ─────────────────────────────────────────────────────────────────────────────
 * dataset.c — implementation of the tiny binary loader.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "dataset.h"
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read exactly `count` items of `size` bytes from `path`, or die trying.
 * Keeping this in one helper means every file read is checked the same way. */
static void *read_or_die(const char *path, size_t size, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        fprintf(stderr, "       run `make data` first (needs python + sklearn once).\n");
        exit(1);
    }
    void *buf = malloc(size * count);
    if (!buf) { fprintf(stderr, "ERROR: out of memory for %s\n", path); exit(1); }
    size_t got = fread(buf, size, count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "ERROR: %s holds %zu items, expected %zu\n", path, got, count);
        exit(1);
    }
    return buf;
}

Dataset dataset_load(const char *dir, const char *split, int n) {
    char path[512];
    Dataset d;
    d.n = n;

    /* "data/<split>_X.bin" : n * IMG_DIM uint8 pixels (0-255). Convert to
     * float in [0,1] — keeps the on-disk file small while the model uses floats. */
    snprintf(path, sizeof(path), "%s/%s_X.bin", dir, split);
    uint8_t *raw = (uint8_t *)read_or_die(path, sizeof(uint8_t), (size_t)n * IMG_DIM);
    d.X = (float *)malloc((size_t)n * IMG_DIM * sizeof(float));
    for (size_t i = 0; i < (size_t)n * IMG_DIM; i++) d.X[i] = raw[i] * (1.0f / 255.0f);
    free(raw);

    /* e.g. "data/train_y.bin" : n uint8 labels */
    snprintf(path, sizeof(path), "%s/%s_y.bin", dir, split);
    d.y = (uint8_t *)read_or_die(path, sizeof(uint8_t), (size_t)n);

    return d;
}

void dataset_free(Dataset *d) {
    free(d->X); d->X = NULL;
    free(d->y); d->y = NULL;
    d->n = 0;
}
