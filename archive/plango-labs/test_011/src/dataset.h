/* ─────────────────────────────────────────────────────────────────────────────
 * dataset.h — loads the flat binary Fashion-MNIST files produced by
 * tools/export_data.py into plain C arrays.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>

/* A loaded split: `n` images, each IMG_DIM floats, plus `n` labels. */
typedef struct {
    int       n;        /* number of examples                         */
    float    *X;        /* n * IMG_DIM floats, pixel values in [0,1]  */
    uint8_t  *y;        /* n labels in [0,9]                          */
} Dataset;

/* Load one split. `dir` is the folder holding the .bin files (e.g. "data"),
 * `split` is "train" or "test", `n` is how many examples that file holds.
 * Exits the program with a message if a file is missing. */
Dataset dataset_load(const char *dir, const char *split, int n);

/* Free the arrays inside a Dataset. */
void dataset_free(Dataset *d);

#endif /* DATASET_H */
