# GPU-Accelerated KV Cache Quantization

This project implements **INT8 KV-cache quantization** for transformer inference and benchmarks multiple CUDA kernel variants against a CPU baseline.

The core idea is **per-channel (column-wise) symmetric linear quantization**:

* Compute one scale per column
  [
  s_d = \frac{\max_t |K_{t,d}|}{127}
  ]
* Quantize each element: `q = clamp(round(K / s), [-128, 127])`
* Dequantize: `K̃ = q * s`

This mirrors common inference-engine practice: keep KV compressed in memory, and reconstruct values when needed. 

---

## What’s in this repo

### CPU implementation

* **`compute_scales()`**: OpenMP-parallel over columns (each thread scans one column) 
* **`quantize_matrix()` / `dequantize_matrix()`**: serial reference loops (conservative baseline)

### GPU implementation

All GPU kernels treat quantization/dequantization as an **embarrassingly parallel, memory-bound** workload (one thread ≈ one element, or a small bundle of elements).

1. **Naive kernel**
   Each thread handles one `(row, col)` element; good coalescing by mapping `col` to the fast-changing x dimension. 

2. **Tiled (shared memory)**
   Loads a TILE into shared memory, then quantizes from shared. In this workload, there is **little/no reuse**, so tiling often helps minimally (or hurts). 

3. **Thread-coarsened**
   Each thread processes `COARSEN` columns for the same row, reducing overhead and improving per-thread work. 

4. **Vectorized (float4 / char4)**
   Each thread processes **4 values per load/store** using vector types, improving effective bandwidth. Requires **`D % 4 == 0`**. 

---

## Benchmark driver

The benchmark runs multiple matrix sizes (from small synthetic to “realistic LLM workload” shapes) and reports:

* quant time, dequant time, total
* speedup over CPU
* reconstruction error (L2, max-abs)
* “attention surrogate” error: mean |Q·K − Q·K̃| over rows 

The driver cycles through 4 GPU modes: Naive, Tiled, Coarsened, Vectorized. 

Timing approach:

* 1 warmup iteration
* 3 timed iterations (average kernel milliseconds via CUDA events) 

---

## Build & run

### Requirements

* NVIDIA GPU + CUDA toolkit
* `nvcc`
* OpenMP-capable host compiler (used for CPU scale computation)

### Compile

Example (adjust filenames if needed):

```bash
nvcc -O3 -Xcompiler -fopenmp main.cu quant_gpu.cu quant_cpu.c matrix.c -o kv_quant_bench
```

### Run

### Run Benchmarks (Stress Tests)
The compiled binary `kv_quant_bench` will automatically run all stress test cases mentioned in the paper, covering a wide range of workloads from small synthetic tests to realistic large-context LLM scenarios. No additional arguments or configuration are required.

```bash
./kv_quant_bench
```

## Benchmark Inputs (Matrix Sizes)

The benchmark evaluates performance across various KV-cache dimensions ($T \times D$), representing different stages of inference and model sizes.

| Test Case | Dimensions ($T \times D$) | Description |
| :--- | :--- | :--- |
| **Trivial Correctness** | $1,024 \times 64$ | Minimal case for basic validation. |
| **Small** | $2,048 \times 128$ | Baseline synthetic workload. |
| **Medium** | $16,384 \times 256$ | Intermediate synthetic workload. |
| **Large** | $65,536 \times 256$ | Large context synthetic workload. |
| **Very Large** | $131,072 \times 256$ | Extended context synthetic workload. |
| **Realistic Small LLM** | $131,072 \times 1,024$ | Realistic hidden dimension for small models. |
| **Realistic Medium LLM** | $131,072 \times 2,048$ | Realistic hidden dimension for medium models. |
| **Realistic Large LLM** | $131,072 \times 4,096$ | Realistic hidden dimension for large models (e.g., Llama 2 70B). |
| **Realistic V. Large LLM** | $131,072 \times 8,192$ | Estimate for very large models (e.g., Claude, GPT-4 class). |
| **Massive Attention** | $262,144 \times 128$ | Testing extreme sequence length handling. |

### Run Unit Tests (Correctness)
```bash
nvcc -O3 -Xcompiler -fopenmp tests.cu quant_gpu.cu quant_cpu.c matrix.c -o unit_tests
./unit_tests
```

---

## Unit Tests Details (25 Correctness Tests)

To ensure robustness and compliance with correctness requirements, `tests.cu` implements **25 distinct unit tests**. These cover basic allocation, core logic, GPU kernel correctness, edge cases, pattern checks, and stress testing.

| Category | Count | Tests | Rationale/Description |
| :--- | :---: | :--- | :--- |
| **Basic Allocations** | 2 | `test_create_fp32`<br>`test_create_int8` | Verify that FP32 and INT8 matrix structs can be allocated, initialized with correct dimensions, and freed without errors. |
| **Data Helpers** | 2 | `test_fill_range`<br>`test_query_fill` | Ensure random number generators (matrix & vector) produce values strictly within specified min/max bounds. |
| **Metric Identity** | 3 | `test_l2_identical`<br>`test_max_abs_identical`<br>`test_attn_identical` | **Sanity check**: The error between a matrix and itself must be 0 (or $< 10^{-6}$). Validates the error-checking code itself. |
| **CPU Core Logic** | 3 | `test_compute_scales_simple`<br>`test_cpu_quant_values`<br>`test_cpu_dequant_values` | **White-box testing** of the reference CPU implementation. Checks if specific known inputs (e.g., 63.5) produce expected quantized integers (e.g., 64) and scales. |
| **GPU Correctness** | 4 | `test_gpu_naive`<br>`test_gpu_tiled`<br>`test_gpu_coarsened`<br>`test_gpu_vectorized` | **Cross-implementation checks**: Runs each GPU kernel variant on random data and verifies the output matches the CPU reference implementation within integer-rounding tolerance. |
| **Edge Cases** | 3 | `test_1x1_cpu`<br>`test_1x1_gpu_naive`<br>`test_1x4_gpu_vec` | **Boundary testing**: Ensures the code handles minimal matrix sizes ($1 \times 1$) without segfaulting or producing NaNs. |
| **Patterns** | 3 | `test_all_zeros`<br>`test_all_ones`<br>`test_alternating` | **Deterministic patterns**: Checks behavior on structured data (all 0s, max 127s, alternating $\pm 127$) to catch logic errors that random data might miss. |
| **Consistency** | 3 | `test_consistency_naive_tiled`<br>`test_consistency_naive_coarsened`<br>`test_consistency_naive_vectorized` | **Kernel-to-kernel validation**: Ensures that optimized kernels (Shared Mem, Coarsened, Vectorized) produce bit-exact (or near-exact) outputs compared to the simple Naive kernel. |
| **Stress Tests** | 2 | `test_stress_cpu_large`<br>`test_stress_gpu_large` | **Scalability check**: Runs on larger matrices ($2048 \times 128$ and $4096 \times 256$) to ensure no heap corruption or scaling artifacts occur. |

**Total Tests: 25**

Executing `./unit_tests` runs all of the above and reports a `PASS/FAIL` status for each.

---

## Results

### Key takeaway

* GPU kernels achieve **~200× to ~1700× speedup** vs CPU across tested sizes.
* **Vectorized** is consistently best (when alignment holds).
* **Tiled** provides minimal benefit, matching the “no reuse → tiling won’t help much” expectation. 

### Speedup summary (CPU vs GPU total quant+dequant)

| Test                   | KV Shape (T×D) | Best GPU Variant |       Speedup |
| ---------------------- | -------------: | ---------------- | ------------: |
| Small                  |       2048×128 | Vectorized       |  **211.97×**  |
| Medium                 |      16384×256 | Vectorized       |  **659.65×**  |
| Large                  |      65536×256 | Vectorized       | **1175.89×**  |
| Very Large             |     131072×256 | Vectorized       | **1147.34×**  |
| Realistic Small LLM    |    131072×1024 | Vectorized       | **1494.63×**  |
| Realistic Medium LLM   |    131072×2048 | Vectorized       | **1600.12×**  |
| Realistic Large LLM    |    131072×4096 | Vectorized       | **1632.33×**  |
| Realistic V. Large LLM |    131072×8192 | Vectorized       | **1694.08×**  |

### Correctness

Reconstruction and attention-surrogate errors match across CPU and GPU variants in the logs (functional equivalence), consistent with the report’s conclusion. 

---

## Notes / gotchas

* **Vectorized kernel requires `D % 4 == 0`** (or you must pad). 
* This benchmark measures **kernel time** (CUDA events). If you care about end-to-end performance, you’d also measure:

  * H2D/D2H copies
  * allocations/frees in wrappers
  * interaction with downstream attention computation
* Scale computation is CPU OpenMP-parallel over columns and is typically amortized over decoding steps (since scales can be reused until KV changes). 

---

## File pointers (from this snapshot)

* `main.cu`: benchmark harness + reporting 
* `quant_gpu.cu`: CUDA kernels + wrappers (naive/tiled/coarsened/vectorized) 
* `matrix.c/.h`: matrix alloc, random fill, error metrics, attention surrogate error 

