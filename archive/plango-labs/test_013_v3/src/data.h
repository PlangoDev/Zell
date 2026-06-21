/* ─────────────────────────────────────────────────────────────────────────────
 * data.h — token memmap + window sampler. Python (tools/build_data.py) writes a
 * uint16 .bin and a meta.json; here we mmap the train .bin read-only and sample
 * random [ctx+1] windows into a host buffer for H2D upload. (F5 — keeping the
 * shard resident in VRAM — is layered on top in brain.cu; this is the host path.)
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef BF_DATA_H
#define BF_DATA_H

#include <stdint.h>
#include <stddef.h>   /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char  model_id[256];
    int   vocab;
    char  dtype[16];          /* "uint16" (vocab < 65536) */
    char  train_path[1024];
    long  train_tokens;
    char  test_path[1024];
    long  test_tokens;
    int   eval_window;
    int   eval_stride;
} Meta;

/* Parse the small JSON written by build_data.py. Returns 0 on success. */
int meta_load(const char *path, Meta *m);

/* A read-only mmap of a uint16 token bin. */
typedef struct {
    const uint16_t *data;     /* mmapped tokens */
    long            n;        /* token count */
    int             fd;
    size_t          map_bytes;
} TokenBin;

int  tokenbin_open(const char *path, long n_tokens, TokenBin *tb);
void tokenbin_close(TokenBin *tb);
int  tokenbin_max(const TokenBin *tb);   /* max token id (to size V safely) */

#ifdef __cplusplus
}
#endif

#endif /* BF_DATA_H */
