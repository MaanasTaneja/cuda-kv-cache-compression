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

//shsred memeory tiling may not help us at all, since we are not reusing memeory at all
//but thread coarsening might go increase arithmetic intensity 

__global__ void quantize_tiled_kernel(const float* k_orignal, const float* scales, int T, int D, int8_t* k_quantized){
   __shared__ float s_tile[TILE_DIM][TILE_DIM];
   const int tile_col = blockIdx.x; //x direction is the column direction --->>>>
   const int tile_row = blockIdx.y; // y direction goes down so its a row direction..

   const int tile_start_col = tile_col * TILE_DIM; //tile start index of thread (global thread ppool)
   const int tile_start_row = tile_row * TILE_DIM;
   //start row is column ofcuse since x direction
   //start col is blockidx y 

   const int global_thread_col = tile_start_col + threadIdx.x;
   const int global_thread_row = tile_start_row + threadIdx.y;

   //load this thread's tile elemnt
   if(global_thread_row >= 0 && global_thread_row < T && 
        global_thread_col >= 0 && global_thread_col < D){
            //if we are in bounds then load
            //thread y is row i and thread x is the j
            s_tile[threadIdx.y][threadIdx.x] = k_orignal[global_thread_row * D + global_thread_col];
        }
    else{
        s_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    //gaurd the global thread bounds.
    if(global_thread_row < T && global_thread_col < D){
        //now we can compute. (scaled)
        float scale = scales[global_thread_col]; //get scal
        float value = s_tile[threadIdx.y][threadIdx.x];
        //extract from tile.
        int q_value = (int) __float2int_rn(value / scale);
        //clamp to int8 range (avoid wrap around)
        if(q_value > 127) q_value = 127;
        if(q_value < -128) q_value = -128;
        k_quantized[global_thread_row * D + global_thread_col] = (int8_t) q_value;

    }
}

__global__ void dequantize_tiled_kernel(const int8_t* k_quantized, const float* scales, int T, int D, float* k_reconstructed){
    __shared__ int8_t s_tile[TILE_DIM][TILE_DIM];
    const int tile_col = blockIdx.x; //x direction is the column direction --->>>>
   const int tile_row = blockIdx.y; // y direction goes down so its a row direction..

   const int tile_start_col = tile_col * TILE_DIM; //tile start index of thread (global thread ppool)
   const int tile_start_row = tile_row * TILE_DIM;
   //start row is column ofcuse since x direction
   //start col is blockidx y 

   const int global_thread_col = tile_start_col + threadIdx.x;
   const int global_thread_row = tile_start_row + threadIdx.y;

   //load this thread's tile elemnt
   if(global_thread_row >= 0 && global_thread_row < T && 
        global_thread_col >= 0 && global_thread_col < D){
            //if we are in bounds then load
            //thread y is row i and thread x is the j
            s_tile[threadIdx.y][threadIdx.x] = k_quantized[global_thread_row * D + global_thread_col];
        }
    else{
        s_tile[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    if(global_thread_row < T && global_thread_col < D){
        float scale = scales[global_thread_col];
        int8_t q_value = s_tile[threadIdx.y][threadIdx.x];
        float value = ((float) q_value) * scale;
        k_reconstructed[global_thread_row * D + global_thread_col] = value;
    }
}

//expected performance gain from shared is minimal, possibly worse... since shared memeory is useful if threads
//in a block share and resuse data, int eh same block(local) but here each element is a discrete operation
//but thread coarsening is going to help hopefulyy.

__global__ void quantize_coarsened_kernel(const float* k_orignal, const float* scales, int T, int D, int8_t* k_quantized){

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    // const int col = blockIdx.x * blockDim.x + threadIdx.x;

    //to coarsen we will coarsen in the x direction, this is 1d coarsen btw.
    //like since we makde our grid to be grid.x is d and grid.y is t 
    //we shoudl coarsen in the d direction -> (columns of kv basically) 
    //as we can load multiple scales at the same time and then cache them. (accorss columns)

    //int start_column = blockIdx.x * blockDim.x + threadIdx.x;
    int start_column = blockIdx.x * blockDim.x * COARSEN + threadIdx.x;

    if (row >= T) return;

    for (int i = 0; i < COARSEN; i++) {

        //actual column index this iteration
        int d = start_column + i * blockDim.x; //GLOBAL COLUMN INDEX.

        if (d < D) {
            float scale = scales[d];
            float value = k_orignal[row * D + d];
            int q_value = (int) __float2int_rn(value / scale);
            //clamp to int8 range (avoid wrap around)
            if(q_value > 127) q_value = 127;
            if(q_value < -128) q_value = -128;

            k_quantized[row * D + d] = (int8_t) q_value;
            
        }
    }
}

__global__ void dequantize_coarsened_kernel(const int8_t* k_quantized, const float* scales, int T, int D, float* k_reconstructed){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    // const int col = blockIdx.x * blockDim.x + threadIdx.x;

    //to coarsen we will coarsen in the x direction, this is 1d coarsen btw.
    //like since we makde our grid to be grid.x is d and grid.y is t 
    //we shoudl coarsen in the d direction -> (columns of kv basically) 
    //as we can load multiple scales at the same time and then cache them. (accorss columns)

    //int start_column = blockIdx.x * blockDim.x + threadIdx.x;
    int start_column = blockIdx.x * blockDim.x * COARSEN + threadIdx.x;

    if (row >= T) return;

    for (int i = 0; i < COARSEN; i++) {
        //actual column index this iteration
        int d = start_column + i * blockDim.x; //GLOBAL COLUMN INDEX.

        if (d < D) {
            float scale = scales[d];
            int8_t q    = k_quantized[row * D + d];
            k_reconstructed[row * D + d] = ((float) q) * scale;
        }
    }
}

// --- vectorized kernels .. most perfomant.

//our operation is a SOLELY MEMORY BOUND OPERATION. since our arithmetic intensity is super low, and we need to do element ise processing 
//and tiled wont help (it might a bit, since cache benfit) btu we arent reusing aything too much, so not the best
//coarsening helps us since we inrease the airthmetic intensity for each thread, gives us gains
//but vectorizatomn should help us better than any other technique, since we taclle the memory loading and offloading and make
//it efficient.
//only one problem, will work only if d % 4 == 0, since we vectorize loads float4..
//so may run into issues such as when d =2042 it will blow up here. unless you pad the matrix.


//helper to pack 4 int8s into a single 32-bit int
__device__ __forceinline__ int pack_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    int32_t packed = 0;
    //we mask to ensure we only grab the bottom 8 bits, then shift
    packed |= (a & 0xFF);
    packed |= ((b & 0xFF) << 8);
    packed |= ((c & 0xFF) << 16);
    packed |= ((d & 0xFF) << 24);
    return packed;
}

__global__ void quantize_vectorized_kernel(const float* k_orignal, const float* scales, int T, int D, int8_t* k_quantized){
    //lets first reinterpent cast our float and int arrays into float 4 and int4 vectors 
    //so we can load 4 floats at once from our array..
    //make pointer now look at float array data as a float 4 (x,y,z,w) data.

    const float4* vec_input = reinterpret_cast<const float4*>(k_orignal);
    const float4* vec_scales = reinterpret_cast<const float4*>(scales);
    int* vec_output = reinterpret_cast<int*>(k_quantized); // so we will pack 4 int8ts together make an inteeger (in bits) 
    //and push those bits into the memeory (as ints) but eventually we can read that memeory as int8ts.

    //now effceivelly we process 4 elemnts per thread
    int vec_D = D / 4; // the number of columns are reduced.. (simialr to coarsen)
    //we are basically iterating 0 to d/4 (and float 4 is grabbing in intervals of 4 - the float elements)

    int col = blockIdx.x * blockDim.x + threadIdx.x; //column should increase  ->> this diretion and x does 
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row goes down 

    if(col < vec_D && row < T){
        //ensure we are in bounds.
        //load in 4 elemtns at once..
        float4 values = vec_input[row * vec_D + col];
        //so here each pointer skips forward 4 values right.
        //index i loads 4 then i + 1 (skips the three) and loads the next four after last four
        //so no reuse and waste.
        //extract the 4 set of elements (or one four.)
        float4 s_values = vec_scales[col];

        int8_t q0, q1, q2, q3;
        int temp;
        temp = __float2int_rn(values.x / s_values.x);
        if(temp > 127) temp = 127;
        if(temp < -128) temp = -128;
        q0 = (int8_t) temp;

        temp = __float2int_rn(values.y / s_values.y);
        if(temp > 127) temp = 127;
        if(temp < -128) temp = -128;
        q1 = (int8_t) temp;

        temp = __float2int_rn(values.z / s_values.z);
        if(temp > 127) temp = 127;
        if(temp < -128) temp = -128;
        q2 = (int8_t) temp;

        temp = __float2int_rn(values.w / s_values.w);
        if(temp > 127) temp = 127;
        if(temp < -128) temp = -128;
        q3 = (int8_t) temp;

        vec_output[row * vec_D + col] = pack_i8(q0, q1, q2, q3);
    }
}

__global__ void dequantize_vectorized_kernel(const int8_t* k_quantized, const float* scales, int T, int D, float* k_reconstructed){
    //lets first reinterpent cast our float and int arrays into float 4 and int4 vectors 
    //so we can load 4 floats at once from our array..
    //make pointer now look at float array data as a float 4 (x,y,z,w) data.

    const char4* vec_input = reinterpret_cast<const char4*>(k_quantized);
    const float4* vec_scales = reinterpret_cast<const float4*>(scales);
    float4* vec_output = reinterpret_cast<float4*>(k_reconstructed); // so we will pack 4 int8ts together make an inteeger (in bits) 
    //and push those bits into the memeory (as ints) but eventually we can read that memeory as int8ts.

    //now effceivelly we process 4 elemnts per thread
    int vec_D = D / 4; // the number of columns are reduced.. (simialr to coarsen)
    //we are basically iterating 0 to d/4 (and float 4 is grabbing in intervals of 4 - the float elements)

    int col = blockIdx.x * blockDim.x + threadIdx.x; //column should increase  ->> this diretion and x does 
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row goes down 

    if(col < vec_D && row < T){
        //ensure we are in bounds.
        //load in 4 elemtns at once..
        char4 values = vec_input[row * vec_D + col];
        //so here each pointer skips forward 4 values right.
        //index i loads 4 then i + 1 (skips the three) and loads the next four after last four
        //so no reuse and waste.
        //extract the 4 set of elements (or one four.)
        float4 s_values = vec_scales[col];
        float4 result;

        result.x = ((float) values.x) * s_values.x;
        result.y = ((float) values.y) * s_values.y;
        result.z = ((float) values.z) * s_values.z;
        result.w = ((float) values.w) * s_values.w;

        vec_output[row * vec_D + col] = result;
    }

}

void quantize_matrix_gpu(const FP32Matrix* src, INT8Matrix* dest, const float* scales, int kernel_version, float* kernel_ms){
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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(TILE_DIM, TILE_DIM); 
    dim3 grid;
    if (kernel_version == 3) {
        // coarsened: each block covers block.x * COARSEN columns
        grid = dim3(
            (D + block.x * COARSEN - 1) / (block.x * COARSEN),
            (T + block.y - 1) / block.y
        );
    } 
    else if(kernel_version == 4){
        int vec_D = D / 4; // Ensure D % 4 == 0
        grid =  dim3(
            (vec_D + block.x - 1) / block.x, // Grid width is now divided by 4
            (T + block.y - 1) / block.y
        );
    }
    else {
        // naive / tiled: each block covers block.x columns
        grid = dim3(
            (D + block.x - 1) / block.x, //since x direction covers cols, and y durection covers rows.
            (T + block.y - 1) / block.y
        );  //grid has to cover entire src matrix basially. so lets pad out a block?
    }

    CUDA_CHECK(cudaEventRecord(start));
    switch (kernel_version) {
        case 1:
            quantize_naive_kernel<<<grid, block>>>(d_kf32, d_scales, T, D, d_ki8);
            break;
        case 2:
            quantize_tiled_kernel<<<grid, block>>>(d_kf32, d_scales, T, D, d_ki8);
            break;
        case 3:
            quantize_coarsened_kernel<<<grid, block>>>(d_kf32, d_scales, T, D, d_ki8);
            break;
        case 4:
            quantize_vectorized_kernel<<<grid, block>>>(d_kf32, d_scales, T, D, d_ki8);
            break;
        default:
            // handle invalid mode
            fprintf(stderr, "Invalid mode %d\n", kernel_version);
            return;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(kernel_ms, start, stop));

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

void dequantize_matrix_gpu(const INT8Matrix* src, FP32Matrix* dest, const float* scales, int kernel_version, float* kernel_ms){
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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid;
    if (kernel_version == 3) {
        // coarsened: each block covers block.x * COARSEN columns
        grid = dim3(
            (D + block.x * COARSEN - 1) / (block.x * COARSEN),
            (T + block.y - 1) / block.y
        );
    } 
    else if(kernel_version == 4){
        int vec_D = D / 4; // Ensure D % 4 == 0
        grid =  dim3(
            (vec_D + block.x - 1) / block.x, // Grid width is now divided by 4
            (T + block.y - 1) / block.y
        );
    }
    else {
        // naive / tiled: each block covers block.x columns
        grid = dim3(
            (D + block.x - 1) / block.x, //since x direction covers cols, and y durection covers rows.
            (T + block.y - 1) / block.y
        );  //grid has to cover entire src matrix basially. so lets pad out a block?
    }

    CUDA_CHECK(cudaEventRecord(start));
    switch (kernel_version) {
        case 1:
            dequantize_naive_kernel<<<grid, block>>>(d_ki8, d_scales, T, D, d_kf32);
            break;
        case 2:
            dequantize_tiled_kernel<<<grid, block>>>(d_ki8, d_scales, T, D, d_kf32);
            break;
        case 3:
            dequantize_coarsened_kernel<<<grid, block>>>(d_ki8, d_scales, T, D, d_kf32);
            break;
        case 4:
            dequantize_vectorized_kernel<<<grid, block>>>(d_ki8, d_scales, T, D, d_kf32);
            break;
        default:
            // handle invalid mode
            fprintf(stderr, "Invalid mode %d\n", kernel_version);
            return;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(kernel_ms, start, stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dest->data, d_kf32, T * D * sizeof(float), cudaMemcpyDeviceToHost));

    dest->rows = T;
    dest->columns = D;

    CUDA_CHECK(cudaFree(d_kf32));
    CUDA_CHECK(cudaFree(d_ki8));
    CUDA_CHECK(cudaFree(d_scales));
}
