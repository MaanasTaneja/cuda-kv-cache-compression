#include "quant_gpu.cuh"

/*
for (block_y = 0; block_y < grid.y; block_y++) {
  for (block_x = 0; block_x < grid.x; block_x++) {
    for (thread_y = 0; thread_y < block.y; thread_y++) {
      for (thread_x = 0; thread_x < block.x; thread_x++) {

        row = block_y * block.y + thread_y;
        col = block_x * block.x + thread_x;

        if (row < T && col < D)
            work(row, col);
      }
    }
  }
}

this is basiclaly kernel <<<grid, block>>>() fuck off cuda.
*/

//create kernels here
__global__ void quantize_naive_kernel(const float* k_orignal, const float* scales, int T, int D, int8_t* k_quantized){
    //so i want to quantize this as -> each thread will quantize one element basically
    //since we access to many threads.  and x, y are no the final matrix indices.

    //each kernel is one thread perspective
    //thread idx is local thrad id in this current block.
    const int col = blockIdx.x * blockDim.x + threadIdx.x; 
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    //cuda gives two stupid arbitrary counters thats it, 0 -> max threads in block in one dimension
    //so threadidx * threadidy = 1024 (bnumber of threadsin one block)
    //and we have to decide to what to do with this ordering.
    //and thread idx always changes fatser than thread idy.... 
    //only rule is threadIdx.x * threadIdx.y * threadIdx.z less than or equal to 1024

    //so (0,0), (1,0), (2,0)... (32, 0) and then (0, 1), (1, 1) (wrap around limit depends on block size, here assumption is block size is 32)
    //in any case thats why we need to do row is slow changer - y and col (can change quick) so x

    //if we incremenet col at a particular row, then you are accessing contiguious
    //but if you increment row at each thread, then you need to travel to next row (skipping num columns worth of elemnts)
    //so this is memory coalescing.

    if(col >= 0 && row >= 0 && col < D && row < T){
        float scale = scales[col]; //get scal
        float value = k_orignal[row * D + col]; //its row index * columns, since we need to cross all those cols (elements)
        //quantize
        int q_value = (int) __float2int_rn(value / scale);
        //clamp to int8 range (avoid wrap around)
        if(q_value > 127) q_value = 127;
        if(q_value < -128) q_value = -128;
        k_quantized[row * D + col] = (int8_t) q_value;
    }
    //if this particualr threadindex is beyond our bounds, let this thread die.
}

__global__ void dequantize_naive_kernel(const int8_t* k_quantized, const float* scales, int T, int D, float* k_reconstructed){
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col >= 0 && row >= 0 && col < D && row < T){
        float scale = scales[col];
        int8_t q_value = k_quantized[row * D + col];
        float value = ((float) q_value) * scale;
        k_reconstructed[row * D + col] = value;
    }
    //if this particualr threadindex is beyond our bounds, let this thread die.

}


void quantize_matrix_gpu(const FP32Matrix* src, INT8Matrix* dest, const float* scales){
    float* d_kf32;
    float* d_scales;
    int8_t* d_ki8;

    int T = src->rows;
    int D = src->columns;

    CUDA_CHECK(cudaMalloc(&d_kf32, T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ki8, T * D * sizeof(int8_t))); //size should be same right. of quantized and orignal kv matrix.
    CUDA_CHECK(cudaMalloc(&d_scales, D * sizeof(float))); //scales is num columns of kv matrix.

    CUDA_CHECK(cudaMemcpy(d_kf32, src->data, T * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales, D * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16); 
    dim3 grid(
        (D + block.x - 1) / block.x,
        (T + block.y - 1) / block.y //since x direction covers cols, and y durection covers rows.
    ); //grid has to cover entire src matrix basially. so lets pad out a block?

    quantize_naive_kernel<<<grid, block>>>(d_kf32, d_scales, T, D, d_ki8);

    CUDA_CHECK(cudaGetLastError());        // launch errors
    CUDA_CHECK(cudaDeviceSynchronize());   // runtime errors inside kernel

    CUDA_CHECK(cudaMemcpy(dest->data, d_ki8,
              T * D * sizeof(int8_t),
              cudaMemcpyDeviceToHost));

    dest->rows = T;
    dest->columns = D;

    CUDA_CHECK(cudaFree(d_kf32));
    CUDA_CHECK(cudaFree(d_ki8));
    CUDA_CHECK(cudaFree(d_scales));
    //end
}

void dequantize_matrix_gpu(const INT8Matrix* src, FP32Matrix* dest, const float* scales){
    float*  d_kf32   = NULL;
    float*  d_scales = NULL;
    int8_t* d_ki8    = NULL;

    int T = src->rows;
    int D = src->columns;

    CUDA_CHECK(cudaMalloc(&d_kf32,   T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ki8,    T * D * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_scales, D * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_ki8,    src->data, T * D * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales,    D * sizeof(float),     cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (D + block.x - 1) / block.x,
        (T + block.y - 1) / block.y
    );

    dequantize_naive_kernel<<<grid, block>>>(d_ki8, d_scales, T, D, d_kf32);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dest->data, d_kf32, T * D * sizeof(float), cudaMemcpyDeviceToHost));

    dest->rows = T;
    dest->columns = D;

    CUDA_CHECK(cudaFree(d_kf32));
    CUDA_CHECK(cudaFree(d_ki8));
    CUDA_CHECK(cudaFree(d_scales));
}
