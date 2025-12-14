#pragma once

//cuda is inherenltly a c++ compiler layer. not a c one.
#ifdef __cplusplus
extern "C" {
#endif

#include "matrix.h"
#include <omp.h>

#define SCALE_COMPUTATION_NUM_THREADS 64

void compute_scales(const FP32Matrix* matrix, float* scales);

void quantize_matrix(const FP32Matrix* src, INT8Matrix* dest, const float* scales);
void dequantize_matrix(const INT8Matrix* src, FP32Matrix* dest, const float* scales); 


#ifdef __cplusplus
}
#endif