# vLLM Hybrid MoE

An inference-optimization project built on vLLM for Mixture-of-Experts models
whose BF16 weights exceed GPU memory. It improves expert placement, heterogeneous
execution, and long-context prefill on NVIDIA A100 GPUs.

## Optimizations

### Expert cache

A model-wide LRU cache retains frequently used experts in GPU memory and replaces
static whole-parameter placement with expert-granularity loading and eviction.
Startup memory profiling jointly allocates expert and KV-cache capacity.

### CPU/GPU expert computation

During C1 decode, cached experts and selected misses run on GPU while remaining
misses run concurrently on CPU from a shared weight bank. CUDA Graph replay and
double-buffered prefill are supported.

### FlashPrefill

An A100 Triton sparse-prefill path implements mean-pooled K/V block selection,
exact sink/local attention, and mean correction for skipped blocks. Decode remains
dense.

## Benchmarks

### 1. Qwen3-30B-A3B — 1K prefill chunks

**1× A100 40GB, TP1, BF16, CUDA Graph decode, C1; two sequential 8192-input +
256-output requests; 1024-token chunks; native generated prompts.**

| Metric | Native UVA | Ours | Speedup |
| --- | ---: | ---: | ---: |
| Mean TTFT | 7.95 s | 8.95 s | 0.89× |
| Decode (1 / mean TPOT) | 15.24 tok/s | 86.81 tok/s | **5.70×** |
| End-to-end output throughput | 10.37 tok/s | 21.54 tok/s | **2.08×** |
| Peak GPU memory | 38.14 GiB | 37.95 GiB | — |
| CPU expert backing | 22.5 GiB | 54 GiB | — |

KV allocation is 3.03 GiB; GPU expert allocation is 31.50 GiB for UVA and
31.28 GiB for ours, including our prefill buffers.

### 2. Qwen3-30B-A3B — 8K prefill chunks

**1× A100 40GB, TP1, BF16, dense FlashAttention 2, Triton MoE, CUDA Graph decode,
C1; two sequential 8192-input + 256-output requests; 8192-token chunks; native
generated prompts.**

| Metric | Native UVA | Ours | Speedup |
| --- | ---: | ---: | ---: |
| Mean TTFT | 4.844 s | 1.627 s | — |
| Prefill proxy (input tokens / TTFT) | 1,691 tok/s | 5,035 tok/s | **2.98×** |
| Decode (1 / mean TPOT) | 14.59 tok/s | 87.35 tok/s | **5.98×** |
| End-to-end output throughput | 11.47 tok/s | 56.29 tok/s | **4.91×** |
| Peak GPU memory | 37.77 GiB | 37.61 GiB | — |
| CPU engine PSS | 25.50 GiB | 55.95 GiB | — |

Both backends use 30.375 GiB GPU expert allocation and 3.028 GiB KV allocation;
ours includes 2.25 GiB of prefill buffers. CPU expert backing is 23.625 GiB for
UVA and 54 GiB for ours.

### 3. Qwen3-235B-A22B-Instruct-2507

**4× A100 80GB, TP4, BF16, CUDA Graph decode, C1; two sequential 8192-input +
256-output requests; 1024-token chunks; archive/chat and code prompts.**

| Metric | Native UVA | Ours | Speedup |
| --- | ---: | ---: | ---: |
| Mean TTFT | 23.55 s | 28.12 s | 0.84× |
| Decode (1 / mean TPOT) | 4.62 tok/s | 23.67 tok/s | **5.12×** |
| End-to-end output throughput | 3.25 tok/s | 6.58 tok/s | **2.02×** |
| Peak GPU memory per device | 76.83 GiB | 76.91 GiB | — |
| CPU expert backing per host | 153 GiB | 423 GiB | — |

KV allocation is 3.54 GiB per device; both backends use 67.50 GiB GPU expert
allocation per device.

### 4. Expert cache vs expert cache + FlashPrefill

**Qwen3-30B-A3B-Instruct-2507, 1× A100 40GB, TP1, BF16, CUDA Graph decode, C1;
two requests per configuration; 8192-token chunks; FlashPrefill threshold 0.1.**

| Input | Expert cache tok/s | + FlashPrefill tok/s | Speedup | TTFT, cache → + FlashPrefill |
| --- | ---: | ---: | ---: | ---: |
| 32K | 24.46 | 25.74 | **1.05×** | 7.07 → 6.52 s |
| 64K | 11.78 | 14.23 | **1.21×** | 16.76 → 12.89 s |
| 128K | 4.03 | 6.64 | **1.65×** | 54.98 → 30.66 s |

| Input | Expert allocation | KV | Peak GPU, cache → + FlashPrefill | CPU PSS, cache → + FlashPrefill |
| --- | ---: | ---: | ---: | ---: |
| 32K | 31.23 GiB | 3.03 GiB | 38.52 → 39.04 GiB | 55.94 → 55.95 GiB |
| 64K | 28.23 GiB | 6.02 GiB | 38.52 → 39.04 GiB | 55.94 → 55.95 GiB |
| 128K | 22.23 GiB | 12.03 GiB | 38.52 → 39.07 GiB | 55.95 → 55.95 GiB |

### 5. Qwen3-30B-A3B — additional C1 workloads

| Input / output tokens | TTFT s, UVA → ours | Decode tok/s, UVA → ours | E2E output tok/s, UVA → ours |
| --- | ---: | ---: | ---: |
| 32,768 / 256 | 21.77 → 6.81 | 13.04 → 72.84 | 6.19 → 24.83 |
| 1,024 / 1,024 | 1.20 → 1.36 | 14.58 → 94.34 | 14.35 → 83.89 |

At 32K, prefill-proxy / decode / E2E speedups are **3.20× / 5.59× / 4.01×**.
At 1K, TTFT is **13.2% slower**, while decode / E2E improve **6.47× / 5.85×**.

| Input / output tokens | Peak GPU GiB, UVA → ours | CPU PSS GiB, UVA → ours | Exact output matches |
| --- | ---: | ---: | ---: |
| 32,768 / 256 | 37.77 → 37.61 | 25.50 → 55.91 | 1/2 |
| 1,024 / 1,024 | 37.46 → 37.05 | 25.50 → 55.91 | 2/2 |

## References

- [FreeToken implementation](https://github.com/FlashML-org/FreeToken)
- [FreeToken paper](https://arxiv.org/abs/2608.16157)
- [FlashPrefill V2 implementation](https://github.com/qhfan/FlashPrefillv2)
- [FlashPrefill V2 paper](https://arxiv.org/abs/2608.19758)
