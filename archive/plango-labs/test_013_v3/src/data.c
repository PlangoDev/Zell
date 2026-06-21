/* ─────────────────────────────────────────────────────────────────────────────
 * data.c — meta.json reader + uint16 token-bin mmap (POSIX / Kaggle Linux).
 * The JSON parse is a tiny purpose-built scanner for the known keys, not a full
 * parser — build_data.py controls the format.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* find "key" in the buffer and return a pointer just past the ':' after it */
static const char *find_val(const char *buf, const char *key) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(buf, pat);
    if (!p) return NULL;
    p = strchr(p + strlen(pat), ':');
    if (!p) return NULL;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static void get_str(const char *buf, const char *key, char *out, int cap) {
    const char *p = find_val(buf, key);
    out[0] = 0;
    if (!p || *p != '"') return;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < cap - 1) out[i++] = *p++;
    out[i] = 0;
}

static long get_long(const char *buf, const char *key, long dflt) {
    const char *p = find_val(buf, key);
    if (!p) return dflt;
    return atol(p);
}

int meta_load(const char *path, Meta *m) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "meta_load: cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return 1; }
    buf[sz] = 0;
    fclose(f);

    memset(m, 0, sizeof(*m));
    get_str(buf, "model_id", m->model_id, sizeof(m->model_id));
    get_str(buf, "dtype", m->dtype, sizeof(m->dtype));
    get_str(buf, "train_path", m->train_path, sizeof(m->train_path));
    get_str(buf, "test_path", m->test_path, sizeof(m->test_path));
    m->vocab        = (int)get_long(buf, "vocab", 0);
    m->train_tokens = get_long(buf, "train_tokens", 0);
    m->test_tokens  = get_long(buf, "test_tokens", 0);
    m->eval_window  = (int)get_long(buf, "eval_window", 512);
    m->eval_stride  = (int)get_long(buf, "eval_stride", 256);
    free(buf);
    if (m->dtype[0] && strcmp(m->dtype, "uint16") != 0)
        fprintf(stderr, "data: warning, dtype %s (expected uint16)\n", m->dtype);
    return 0;
}

int tokenbin_open(const char *path, long n_tokens, TokenBin *tb) {
    memset(tb, 0, sizeof(*tb));
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "tokenbin_open: cannot open %s\n", path); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 1; }
    long avail = (long)(st.st_size / sizeof(uint16_t));
    if (n_tokens <= 0 || n_tokens > avail) n_tokens = avail;
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { close(fd); fprintf(stderr, "tokenbin_open: mmap failed\n"); return 1; }
    tb->data = (const uint16_t *)p;
    tb->n = n_tokens;
    tb->fd = fd;
    tb->map_bytes = st.st_size;
    return 0;
}

void tokenbin_close(TokenBin *tb) {
    if (tb->data) munmap((void *)tb->data, tb->map_bytes);
    if (tb->fd >= 0) close(tb->fd);
    memset(tb, 0, sizeof(*tb));
}

int tokenbin_max(const TokenBin *tb) {
    int mx = 0;
    /* sample-scan (full scan of 500M would be slow but is only done once if needed);
     * scan a stride to bound cost while still catching the true max with high
     * probability. The caller also takes max(meta.vocab, this, len(tok)). */
    long step = tb->n > 5000000 ? tb->n / 5000000 : 1;
    for (long i = 0; i < tb->n; i += step) {
        int v = (int)tb->data[i];
        if (v > mx) mx = v;
    }
    return mx;
}
