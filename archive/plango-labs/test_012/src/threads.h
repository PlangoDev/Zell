/* ─────────────────────────────────────────────────────────────────────────────
 * threads.h — a tiny, portable parallel-for. Splits a range [0,n) across the
 * machine's cores. Uses POSIX threads on macOS/Linux and the Win32 API on
 * Windows, chosen at compile time, so the exact same source builds and runs
 * fast on the Windows PC too.
 *
 * The callback gets its own slice [begin,end) and a thread id `tid` in
 * [0, threads_count). Use `tid` to index per-thread scratch buffers and avoid
 * locks: each thread writes only its own slot, then the caller reduces.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef THREADS_H
#define THREADS_H

typedef void (*par_fn)(int begin, int end, int tid, void *arg);

/* Pick the worker count. Pass 0 to auto-detect the number of hardware threads. */
void threads_init(int nthreads);

/* How many workers parallel_for will use. */
int  threads_count(void);

/* Run fn over [0,n) split into threads_count contiguous chunks; wait for all. */
void parallel_for(int n, par_fn fn, void *arg);

#endif /* THREADS_H */
