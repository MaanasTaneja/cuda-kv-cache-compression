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

