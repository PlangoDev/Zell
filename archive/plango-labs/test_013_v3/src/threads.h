/* ─────────────────────────────────────────────────────────────────────────────
 * threads.h — portable parallel-for over a persistent thread pool (from test_012).
 * Used host-side for the window-gather and any CPU-side bucket prep; the GPU does
 * the heavy lifting. POSIX threads on Linux (Kaggle), Win32 on Windows.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef BF_THREADS_H
#define BF_THREADS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*par_fn)(int begin, int end, int tid, void *arg);

void threads_init(int nthreads);   /* 0 = auto-detect hardware threads */
int  threads_count(void);
void parallel_for(int n, par_fn fn, void *arg);

#ifdef __cplusplus
}
#endif

#endif /* BF_THREADS_H */
