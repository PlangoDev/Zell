/* ─────────────────────────────────────────────────────────────────────────────
 * threads.cpp — persistent thread pool + portable parallel_for (from test_012).
 * Workers are spawned once and parked on a condition variable; each parallel_for
 * wakes them, runs a chunk on the main thread too, and waits.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "threads.h"
#include <stdlib.h>

#ifdef _WIN32
  #include <windows.h>
  #define MUTEX            CRITICAL_SECTION
  #define COND             CONDITION_VARIABLE
  #define LOCK(m)          EnterCriticalSection(m)
  #define UNLOCK(m)        LeaveCriticalSection(m)
  #define WAIT(c, m)       SleepConditionVariableCS(c, m, INFINITE)
  #define WAKE_ALL(c)      WakeAllConditionVariable(c)
  #define MUTEX_INIT(m)    InitializeCriticalSection(m)
  #define COND_INIT(c)     InitializeConditionVariable(c)
#else
  #include <pthread.h>
  #include <unistd.h>
  #define MUTEX            pthread_mutex_t
  #define COND             pthread_cond_t
  #define LOCK(m)          pthread_mutex_lock(m)
  #define UNLOCK(m)        pthread_mutex_unlock(m)
  #define WAIT(c, m)       pthread_cond_wait(c, m)
  #define WAKE_ALL(c)      pthread_cond_broadcast(c)
  #define MUTEX_INIT(m)    pthread_mutex_init(m, NULL)
  #define COND_INIT(c)     pthread_cond_init(c, NULL)
#endif

static int   g_n = 1;
static MUTEX g_lock;
static COND  g_wake, g_done;
static par_fn g_fn; static void *g_arg;
static int   g_total, g_chunk;
static long  g_jobid = 0;
static int   g_finished = 0;
static int   g_stop = 0;

static void run_chunk(int idx) {
    int b = idx * g_chunk, e = b + g_chunk;
    if (e > g_total) e = g_total;
    if (b < e) g_fn(b, e, idx, g_arg);
}

#ifdef _WIN32
static DWORD WINAPI worker(LPVOID arg) {
#else
static void *worker(void *arg) {
#endif
    int idx = (int)(long)arg;
    long seen = 0;
    for (;;) {
        LOCK(&g_lock);
        while (g_jobid == seen && !g_stop) WAIT(&g_wake, &g_lock);
        if (g_stop) { UNLOCK(&g_lock); break; }
        seen = g_jobid;
        UNLOCK(&g_lock);

        run_chunk(idx);

        LOCK(&g_lock);
        if (++g_finished == g_n - 1) WAKE_ALL(&g_done);
        UNLOCK(&g_lock);
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

extern "C" void threads_init(int n) {
    if (n <= 0) {
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si); n = (int)si.dwNumberOfProcessors;
#else
        long c = sysconf(_SC_NPROCESSORS_ONLN); n = (c > 0) ? (int)c : 1;
#endif
    }
    g_n = n < 1 ? 1 : n;
    MUTEX_INIT(&g_lock); COND_INIT(&g_wake); COND_INIT(&g_done);
    for (int i = 1; i < g_n; i++) {
#ifdef _WIN32
        CreateThread(NULL, 0, worker, (LPVOID)(long)i, 0, NULL);
#else
        pthread_t th; pthread_create(&th, NULL, worker, (void *)(long)i); pthread_detach(th);
#endif
    }
}

extern "C" int threads_count(void) { return g_n; }

extern "C" void parallel_for(int n, par_fn fn, void *arg) {
    if (g_n <= 1 || n <= 1) { fn(0, n, 0, arg); return; }
    LOCK(&g_lock);
    g_fn = fn; g_arg = arg; g_total = n;
    g_chunk = (n + g_n - 1) / g_n;
    g_finished = 0; g_jobid++;
    WAKE_ALL(&g_wake);
    UNLOCK(&g_lock);

    run_chunk(0);

    LOCK(&g_lock);
    while (g_finished < g_n - 1) WAIT(&g_done, &g_lock);
    UNLOCK(&g_lock);
}
