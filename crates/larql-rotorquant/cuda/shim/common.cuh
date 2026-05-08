#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define WARP_SIZE 32

[[noreturn]] inline void ggml_cuda_error(
    const char * stmt,
    const char * func,
    const char * file,
    int line,
    const char * msg
) {
    std::fprintf(stderr, "CUDA error: %s failed in %s at %s:%d: %s\n", stmt, func, file, line, msg);
    std::abort();
}

#define CUDA_CHECK(err)                                                            \
    do {                                                                          \
        cudaError_t err_ = (err);                                                  \
        if (err_ != cudaSuccess) {                                                 \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        }                                                                         \
    } while (0)
