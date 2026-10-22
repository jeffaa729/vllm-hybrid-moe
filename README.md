# vLLM Hybrid MoE

A personal inference-optimization project built on [vLLM](https://github.com/vllm-project/vllm)
for Mixture-of-Experts models whose BF16 weights exceed GPU memory. It combines
CPU/GPU expert execution with a persistent GPU expert cache.
Long-context prefill also supports a FlashPrefill V2-style sparse attention path.

## Approach

### 1. CPU/GPU expert execution

Native vLLM UVA offloading keeps selected weights in CPU RAM but executes expert
computation on the GPU, accessing host weights over the interconnect.

This project splits C1 decode work between GPU and CPU: cached experts and
selected cache misses run on GPU, while remaining misses run on CPU according
to a configured miss-count policy. CPU computation overlaps GPU work and shares
the existing CPU weight bank. The raw-weight CPU adapter reuses vLLM's vector
primitives and SiLU kernel; the handoff supports CUDA Graph replay.

### 2. GPU expert cache

Native UVA uses static, whole-parameter CPU/GPU placement. This project adds a
model-wide LRU cache at expert granularity: protect active experts, fetch selected
misses into reusable GPU slots, and retain experts across decoding steps.

Startup uses vLLM memory profiling to reserve KV capacity for the configured
context and concurrency, then fills the remaining budget with expert slots.
Cache initialization preloads resident experts; prefill uses double-buffered
layer streaming. Cache capacities remain fixed after startup.

Existing vLLM GPU MoE kernels, TP weight sharding, all-reduce and graph capture
are reused.

### 3. Sparse prefill

A custom A100 Triton implementation follows FlashPrefill V2: mean-pooled K/V
block selection, exact sink/local attention and mean correction for skipped
blocks. It reuses vLLM's paged KV cache and split reduction; decode remains dense.

## Benchmarks

### Expert offloading

Baseline: **native vLLM UVA offloading**. Both models use BF16, CUDA Graph decode,
C1, two sequential requests of **8192 input + 256 output tokens**, 1024-token
chunked prefill, temperature 0 and forced output length. Each backend has one
measured pass, no client warmups, and matched KV capacity.

| Model | Hardware | Prompts | UVA output tok/s | Ours output tok/s | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-30B-A3B | 1× A100 40GB, TP1 | Native generated | 10.37 | 21.54 | **2.08×** |
| Qwen3-235B-A22B-Instruct-2507 | 4× A100 80GB, TP4 | Archive/chat + code | 3.25 | 6.58 | **2.02×** |

Output throughput includes prefill and decode. Results compare the complete
implementation with UVA; they do not isolate the benefit of CPU computation.

| Model | Mean TTFT, UVA → ours | Mean TPOT, UVA → ours | Peak GPU GiB/device, UVA → ours | CPU expert backing GiB/host, UVA → ours |
| --- | ---: | ---: | ---: | ---: |
| 30B | 7.95 → 8.95 s | 65.61 → 11.52 ms | 38.14 → 37.95 | 22.5 → 54 |
| 235B | 23.55 → 28.12 s | 216.27 → 42.25 ms | 76.83 → 76.91 | 153 → 423 |

The improvement comes with higher CPU-memory use and slower time to first token.
KV allocation is 3.03 GiB/device for 30B and 3.54 GiB/device for 235B. GPU expert
allocation, including our prefill buffers, is 31.50/31.28 GiB for UVA/ours on 30B,
and 67.50/67.50 GiB on 235B.

The 30B result is the 2026-10-19 fresh-clone run. The 235B result is the saved
2026-10-18 TP4 experiment; the latest CPU-executor revision has been revalidated
on 30B only. These small runs do not establish general model accuracy.

### Expert cache vs expert cache + FlashPrefill

Qwen3-30B-A3B-Instruct-2507 on **1× A100 40GB**, TP1, BF16; EPYC 7K62,
23.04-core CPU quota, 16 CPU expert threads. CUDA Graph decode, C1, 8K prefill
chunks, prefix cache disabled, temperature/seed 0, and 256 forced output tokens.
Two sequential requests per configuration, one measured pass, no client warmups.
FlashPrefill threshold is 0.1; CPU/GPU expert execution is enabled in both modes.

| Input | Expert cache tok/s | Expert cache + FlashPrefill tok/s | Speedup | TTFT s, cache → +FlashPrefill |
| --- | ---: | ---: | ---: | ---: |
| 32K | 24.46 | 25.74 | **1.05×** | 7.07 → 6.52 |
| 64K | 11.78 | 14.23 | **1.21×** | 16.76 → 12.89 |
| 128K | 4.03 | 6.64 | **1.65×** | 54.98 → 30.66 |

Rates include prefill and decode. The 32K inputs are native generated prompts;
64K/128K use two fixed RULER retrieval examples, respectively 65,377–65,486 and
131,033–131,041 input tokens. Each pair uses identical inputs and expert/KV budgets.

| Input | Expert allocation GiB | KV GiB | Peak GPU GiB, cache → +FlashPrefill | CPU PSS GiB, cache → +FlashPrefill |
| --- | ---: | ---: | ---: | ---: |
| 32K | 31.23 | 3.03 | 38.52 → 39.04 | 55.94 → 55.95 |
| 64K | 28.23 | 6.02 | 38.52 → 39.04 | 55.94 → 55.95 |
| 128K | 22.23 | 12.03 | 38.52 → 39.07 | 55.95 → 55.95 |

Expert allocations include 2.25 GiB of prefill buffers; CPU expert backing is
54 GiB in both modes. GPU peaks are sampled every 500 ms; CPU PSS is the worker
snapshot after each pass.

Accuracy checks: both modes retrieved **4/4 access codes** in a separate single
32K prompt (32 output tokens), and **2/2 reference answers at each of 64K/128K**
from the measured RULER requests, including the correct first UUID. Generated
continuations differ; these small checks do not establish general accuracy.
Results were measured on 2026-10-21.

[Installation and benchmark commands](docs/benchmarking/hybrid_moe.md).

## References

The expert-cache, heterogeneous execution and double-buffered prefill designs
are informed by FreeToken:

- [FreeToken implementation](https://github.com/FlashML-org/FreeToken)
- [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://arxiv.org/abs/2608.16157)
- [FlashPrefill V2 implementation](https://github.com/qhfan/FlashPrefillv2)
- [FlashPrefill V2 paper](https://arxiv.org/abs/2608.19758)
