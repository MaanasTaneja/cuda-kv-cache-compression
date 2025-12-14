#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "quant_cpu.h"

int main(int argc, char** argv) {
    //set dimensions for a test KV matrix
    int T = 16384;   //number of tokens (rows)
    //var T or num rows of kv depend on input sequence length, this T is capped by max context limit of model
    int D = 256;    //head dimension (columns)
    //head dimension means the size of each key vector per token
    //this is constant based on model architecture, 

    // Seed RNG
    srand((unsigned int) time(NULL));

    FP32Matrix* K       = createFP32Matrix(T, D);
    FP32Matrix* K_recon = createFP32Matrix(T, D);
    INT8Matrix* K_int8  = createINT8Matrix(T, D);

    //allocate scales (one per column)
    float* scales = (float*) malloc(D * sizeof(float));

    //allocate query vector (length D)
    float* Q = (float*) malloc(D * sizeof(float));

    random_fill_FP32Matrix(K, -1.0f, 1.0f);
    random_fill_query_vector(Q, D, -1.0f, 1.0f);


    compute_scales(K, scales);

    //serial timing
    double t0 = omp_get_wtime();
    compute_scales(K, scales);
    double t1 = omp_get_wtime();

    double serial_s = (t1 - t0);
    printf("Scale Computation Time: %f seconds\n", serial_s);



    quantize_matrix(K, K_int8, scales);
    dequantize_matrix(K_int8, K_recon, scales);

    //compute reconstruction errors
    float l2 = l2_error(K, K_recon);
    float maxerr = max_abs_error(K, K_recon);

    printf("=== Reconstruction Error Metrics ===\n");
    printf("L2 error           : %f\n", l2);
    printf("Max absolute error : %f\n", maxerr);

    //compute attention surrogate error
    float attn_err = attention_dot_product_error(Q, K, K_recon);
    printf("\n=== Attention Surrogate Error ===\n");
    printf("Mean |Q·K - Q·K_recon| : %f\n", attn_err);

    free(Q);
    free(scales);
    freeFP32Matrix(K);
    freeFP32Matrix(K_recon);
    freeINT8Matrix(K_int8);

    return 0;
}
