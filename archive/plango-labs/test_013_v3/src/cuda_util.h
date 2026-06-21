/* ─────────────────────────────────────────────────────────────────────────────
 * cuda_util.h — thin CUDA helpers: error-checking macros, device-buffer alloc,
 * a cuBLAS handle wrapper, and fp16 conversion shims. Header-only.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef BF_CUDA_UTIL_H
#define BF_CUDA_UTIL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,        \
                    __LINE__, (int)_s);                                         \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_e));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* Launch-error check after a kernel (use in debug; cheap enough to keep on). */
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

/* Typed device allocation helpers (counts, not bytes). */
template <typename T>
static inline T *dev_alloc(size_t n) {
    T *p = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&p, n * sizeof(T)));
    return p;
}
template <typename T>
static inline T *dev_zalloc(size_t n) {
    T *p = dev_alloc<T>(n);
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template <typename T>
static inline void dev_free(T *&p) {
    if (p) { cudaFree(p); p = nullptr; }
}
template <typename T>
static inline void h2d(T *dst, const T *src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
static inline void d2h(T *dst, const T *src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost));
}

/* fp16 helpers usable from host and device. */
static inline __half h2half(float f) { return __float2half(f); }

#endif /* BF_CUDA_UTIL_H */
