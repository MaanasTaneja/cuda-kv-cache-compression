#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "quant_cpu.h"
#include "quant_gpu.cuh"

/*
preliminary results 

KV-cache quantization is memory-bound
CPUs scale poorly with large T
GPUs excel even with naive kernels
Quantization error is independent of backend
Attention quality degradation remains small


can also do kernel fusion??? hhmm.
*/

typedef struct {
    const char* name;
    int T;
    int D;
} TestCase;

int main(int argc, char** argv) {

    TestCase tests[] = {
        {"Small",      2048,   128},
        {"Medium",          16384,   256},
        {"Large",     65536,   256},
        {"Realistic LLM workload",    131072,   256}
    };

    int num_tests = sizeof(tests) / sizeof(TestCase);

    srand((unsigned int) time(NULL));

    for (int t = 0; t < num_tests; t++) {

        int T = tests[t].T;
        int D = tests[t].D;

        printf("\n=============================================\n");
        printf("Test %d: %s\n", t + 1, tests[t].name);
        printf("KV matrix: T = %d, D = %d\n", T, D);
        printf("=============================================\n");

        /* ---------- ALLOCATION ---------- */

        FP32Matrix* K_cpu       = createFP32Matrix(T, D);
        FP32Matrix* K_cpu_rec   = createFP32Matrix(T, D);
        FP32Matrix* K_gpu_rec   = createFP32Matrix(T, D);
        INT8Matrix* K_int8_cpu  = createINT8Matrix(T, D);
        INT8Matrix* K_int8_gpu  = createINT8Matrix(T, D);

        float* scales = (float*) malloc(D * sizeof(float));
        float* Q      = (float*) malloc(D * sizeof(float));

        random_fill_FP32Matrix(K_cpu, -1.0f, 1.0f);
        random_fill_query_vector(Q, D, -1.0f, 1.0f);

        /* ---------------- CPU PATH ---------------- */

        double cpu_t0 = omp_get_wtime();

        compute_scales(K_cpu, scales);
        quantize_matrix(K_cpu, K_int8_cpu, scales);
        dequantize_matrix(K_int8_cpu, K_cpu_rec, scales);

        double cpu_t1 = omp_get_wtime();
        double cpu_time = cpu_t1 - cpu_t0;

        /* ---------------- GPU PATH ---------------- */

        // Warm-up
        quantize_matrix_gpu(K_cpu, K_int8_gpu, scales);
        dequantize_matrix_gpu(K_int8_gpu, K_gpu_rec, scales);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        quantize_matrix_gpu(K_cpu, K_int8_gpu, scales);
        dequantize_matrix_gpu(K_int8_gpu, K_gpu_rec, scales);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
        double gpu_time = gpu_ms / 1000.0;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        /* ---------------- ERROR CHECKS ---------------- */

        float l2_cpu  = l2_error(K_cpu, K_cpu_rec);
        float l2_gpu  = l2_error(K_cpu, K_gpu_rec);
        float max_cpu = max_abs_error(K_cpu, K_cpu_rec);
        float max_gpu = max_abs_error(K_cpu, K_gpu_rec);

        float attn_cpu = attention_dot_product_error(Q, K_cpu, K_cpu_rec);
        float attn_gpu = attention_dot_product_error(Q, K_cpu, K_gpu_rec);

        /* ---------------- REPORT ---------------- */

        printf("\n===== PERFORMANCE =====\n");
        printf("CPU quant+dequant time : %.6f s\n", cpu_time);
        printf("GPU quant+dequant time : %.6f s\n", gpu_time);
        printf("Speedup                : %.2fx\n", cpu_time / gpu_time);

        printf("\n===== RECONSTRUCTION ERROR =====\n");
        printf("CPU L2 error           : %f\n", l2_cpu);
        printf("GPU L2 error           : %f\n", l2_gpu);
        printf("CPU Max abs error      : %f\n", max_cpu);
        printf("GPU Max abs error      : %f\n", max_gpu);

        printf("\n===== ATTENTION SURROGATE ERROR =====\n");
        printf("CPU Mean |Q·K - Q·K̂|   : %f\n", attn_cpu);
        printf("GPU Mean |Q·K - Q·K̂|   : %f\n", attn_gpu);

        /* ---------------- CLEANUP ---------------- */

        free(Q);
        free(scales);

        freeFP32Matrix(K_cpu);
        freeFP32Matrix(K_cpu_rec);
        freeFP32Matrix(K_gpu_rec);
        freeINT8Matrix(K_int8_cpu);
        freeINT8Matrix(K_int8_gpu);
    }

    return 0;
}
