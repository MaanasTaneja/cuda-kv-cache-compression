#pragma once

//cuda is inherenltly a c++ compiler layer. not a c one.
#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <stdint.h>
#include "matrix.h"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

void quantize_matrix_gpu(const FP32Matrix* src, INT8Matrix* dest, const float* scales);
void dequantize_matrix_gpu(const INT8Matrix* src, FP32Matrix* dest, const float* scales); 

#ifdef __cplusplus
}
#endif