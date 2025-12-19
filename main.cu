// main.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "quant_cpu.h"
#include "quant_gpu.cuh"

typedef struct {
    const char* name;
    int T;
    int D;
} TestCase;

typedef struct {
    const char* name;
    int mode;  // 1=naive, 2=tiled, 3=coarsened, 4=vectorized
} GpuVariant;

int main(int argc, char** argv) {

    TestCase tests[] = {
        {"Trivial Correctness Test", 1024, 64},
        {"Small",                  2048,    128},
        {"Medium",                16384,    256},
        {"Large",                 65536,    256},
        {"Very Large",            131072,    256},
        {"Realistic Small LLM workload",131072,   1024},
        {"Realistic Medium LLM workload",131072,   2048},
        {"Realistic Large LLM workload",131072,   4096},
        {"Realistic V. Large LLM workload",131072,  8192}, //this is an estimate to claude's kv cache matrix size.
        {"Massive Attention", 262144, 128} //long context window
    };
    const int num_tests = (int)(sizeof(tests) / sizeof(TestCase));

    GpuVariant variants[] = {
        {"Naive",     1},
        {"Tiled",     2},
        {"Coarsened", 3},
        {"Vectorized", 4}
    };
    const int num_variants = (int)(sizeof(variants) / sizeof(GpuVariant));

    srand((unsigned int) time(NULL));

    // You can tune these (since your wrappers include malloc/memcpy/free, large iters may be slow)
    const int WARMUP_ITERS = 1;
    const int TIMED_ITERS  = 3;

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

        float* scales = (float*) malloc((size_t)D * sizeof(float));
        float* Q      = (float*) malloc((size_t)D * sizeof(float));
        if (!scales || !Q) {
            fprintf(stderr, "Host malloc failed\n");
            return 1;
        }

        random_fill_FP32Matrix(K_cpu, -1.0f, 1.0f);
        random_fill_query_vector(Q, D, -1.0f, 1.0f);

        /* ---------------- CPU PATH ---------------- */

        compute_scales(K_cpu, scales);

        double cpu_q_t0 = omp_get_wtime();
        quantize_matrix(K_cpu, K_int8_cpu, scales);
        double cpu_q_t1 = omp_get_wtime();
        double cpu_q_time = cpu_q_t1 - cpu_q_t0;

        double cpu_dq_t0 = omp_get_wtime();
        dequantize_matrix(K_int8_cpu, K_cpu_rec, scales);
        double cpu_dq_t1 = omp_get_wtime();
        double cpu_dq_time = cpu_dq_t1 - cpu_dq_t0;

        double cpu_time = cpu_q_time + cpu_dq_time;

        float l2_cpu   = l2_error(K_cpu, K_cpu_rec);
        float max_cpu  = max_abs_error(K_cpu, K_cpu_rec);
        float attn_cpu = attention_dot_product_error(Q, K_cpu, K_cpu_rec);

        /* ---------------- GPU PATH: run all 4 kernels ---------------- */
        // We will store result times here
        double gpu_time_s[4]    = {0};
        double gpu_q_time_s[4]  = {0};
        double gpu_dq_time_s[4] = {0};

        float  l2_gpu[4]     = {0};
        float  max_gpu[4]    = {0};
        float  attn_gpu[4]   = {0};

        for (int v = 0; v < num_variants; v++) {
            int mode = variants[v].mode;
            float dummy_ms = 0.0f; // placeholder for warmup

            // 1. Warm-up
            for (int w = 0; w < WARMUP_ITERS; w++) {
                quantize_matrix_gpu(K_cpu, K_int8_gpu, scales, mode, &dummy_ms);
                dequantize_matrix_gpu(K_int8_gpu, K_gpu_rec, scales, mode, &dummy_ms);
            }

            // 2. Timed Execution
            float total_q_ms = 0.0f;
            float total_dq_ms = 0.0f;

            for (int it = 0; it < TIMED_ITERS; it++) {
                float q_ms = 0.0f;
                float dq_ms = 0.0f;

                // Run Quant (returns kernel ms)
                quantize_matrix_gpu(K_cpu, K_int8_gpu, scales, mode, &q_ms);
                
                // Run Dequant (returns kernel ms)
                dequantize_matrix_gpu(K_int8_gpu, K_gpu_rec, scales, mode, &dq_ms);

                total_q_ms  += q_ms;
                total_dq_ms += dq_ms;
            }

            // Calculate Averages (converting ms to seconds)
            gpu_q_time_s[v]  = (total_q_ms / (double)TIMED_ITERS) / 1000.0;
            gpu_dq_time_s[v] = (total_dq_ms / (double)TIMED_ITERS) / 1000.0;
            gpu_time_s[v]    = gpu_q_time_s[v] + gpu_dq_time_s[v];

            // 3. Error Checking (uses the result from the last iteration)
            l2_gpu[v]   = l2_error(K_cpu, K_gpu_rec);
            max_gpu[v]  = max_abs_error(K_cpu, K_gpu_rec);
            attn_gpu[v] = attention_dot_product_error(Q, K_cpu, K_gpu_rec);
        }

        /* ---------------- REPORT (your format + extra GPU modes) ---------------- */

        printf("\n===== PERFORMANCE =====\n");
        printf("CPU quant time         : %.6f s\n", cpu_q_time);
        printf("CPU dequant time       : %.6f s\n", cpu_dq_time);
        printf("CPU quant+dequant time : %.6f s\n", cpu_time);

        printf("GPU (Naive) quant time             : %.6f s\n", gpu_q_time_s[0]);
        printf("GPU (Naive) dequant time           : %.6f s\n", gpu_dq_time_s[0]);
        printf("GPU (Naive) quant+dequant time     : %.6f s\n", gpu_time_s[0]);

        printf("GPU (Tiled) quant time             : %.6f s\n", gpu_q_time_s[1]);
        printf("GPU (Tiled) dequant time           : %.6f s\n", gpu_dq_time_s[1]);
        printf("GPU (Tiled) quant+dequant time     : %.6f s\n", gpu_time_s[1]);

        printf("GPU (Coarsened) quant time         : %.6f s\n", gpu_q_time_s[2]);
        printf("GPU (Coarsened) dequant time       : %.6f s\n", gpu_dq_time_s[2]);
        printf("GPU (Coarsened) quant+dequant time : %.6f s\n", gpu_time_s[2]);

        printf("GPU (Vectorized) quant time         : %.6f s\n", gpu_q_time_s[3]);
        printf("GPU (Vectorized) dequant time       : %.6f s\n", gpu_dq_time_s[3]);
        printf("GPU (Vectorized) quant+dequant time : %.6f s\n", gpu_time_s[3]);

        printf("Speedup (Naive)      : %.2fx\n", cpu_time / gpu_time_s[0]);
        printf("Speedup (Tiled)      : %.2fx\n", cpu_time / gpu_time_s[1]);
        printf("Speedup (Coarsened)  : %.2fx\n", cpu_time / gpu_time_s[2]);
        printf("Speedup (Vectorized) : %.2fx\n", cpu_time / gpu_time_s[3]);

        printf("\n===== RECONSTRUCTION ERROR =====\n");
        printf("CPU L2 error                 : %f\n", l2_cpu);
        printf("GPU (Naive) L2 error         : %f\n", l2_gpu[0]);
        printf("GPU (Tiled) L2 error         : %f\n", l2_gpu[1]);
        printf("GPU (Coarsened) L2 error     : %f\n", l2_gpu[2]);
        printf("GPU (Vectorized) L2 error    : %f\n", l2_gpu[3]);

        printf("CPU Max abs error            : %f\n", max_cpu);
        printf("GPU (Naive) Max abs error    : %f\n", max_gpu[0]);
        printf("GPU (Tiled) Max abs error    : %f\n", max_gpu[1]);
        printf("GPU (Coarsened) Max abs error: %f\n", max_gpu[2]);
        printf("GPU (Vectorized) Max abs error: %f\n", max_gpu[3]);

        printf("\n===== ATTENTION SURROGATE ERROR =====\n");
        printf("CPU Mean |Q·K - Q·K̂|               : %f\n", attn_cpu);
        printf("GPU (Naive) Mean |Q·K - Q·K̂|       : %f\n", attn_gpu[0]);
        printf("GPU (Tiled) Mean |Q·K - Q·K̂|       : %f\n", attn_gpu[1]);
        printf("GPU (Coarsened) Mean |Q·K - Q·K̂|   : %f\n", attn_gpu[2]);
        printf("GPU (Vectorized) Mean |Q·K - Q·K̂|  : %f\n", attn_gpu[3]);

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