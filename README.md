# vLLM Hybrid MoE

## Implementation

| Technique | Implementation |
| --- | --- |
| Expert cache | Model-wide LRU, expert-granularity transfers, cache preloading and double-buffered GPU prefill |
| CPU/GPU execution | GPU-resident/fetched experts and CPU residual misses share one CPU weight bank; graph-enabled decode and TP support |
| Memory planning | Startup profiling-based expert/KV sizing; fixed matched allocations in these benchmarks |
| Sparse prefill | FlashPrefill V2-style block selection and mean correction, with an A100 compute kernel |

## Qwen3-30B benchmark conditions

| Setting | Value |
| --- | --- |
| Model / GPU | Qwen3-30B-A3B-Instruct-2507 / 1× A100 40GB |
| Execution | BF16, TP1, CUDA Graph decode, 8K prefill chunks; prefix caching off |
| GPU allocations | Utilization 0.95; expert 21.375 GiB; KV 12.028 GiB, matched across modes |
| CPU | EPYC 7532; 12 expert threads |
| Requests | 2K: C1/C8/C32 with 2/8/32 requests; 32K/64K/128K: C1 with 2 requests each |
| Generation / timing | Greedy, fixed 256 output tokens, ignore EOS; one pass; no client warmups |
| Metrics | Native vllm bench serve; decode = 1 / mean TPOT; E2E includes prefill + decode |

## Qwen3-30B — 2K serving concurrency

| Concurrency | Backend | TTFT s ↓ | Decode/request tok/s ↑ | E2E output tok/s ↑ | E2E vs UVA |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 | Native UVA | 2.24 | 11.05 | 10.11 | 1.00× |
| C1 | Expert cache + CPU/GPU | 1.70 | 86.18 | 54.98 | **5.44×** |
| C8 | Native UVA | 10.31 | 4.30 | 29.31 | 1.00× |
| C8 | Expert cache + CPU/GPU | 4.26 | 40.86 | 194.91 | **6.65×** |
| C32 | Native UVA | 30.54 | 1.69 | 44.94 | 1.00× |
| C32 | Expert cache + CPU/GPU | 10.26 | 10.30 | 233.05 | **5.19×** |

## Qwen3-235B-A22B-Instruct-2507 — TP4 expert cache

| Setting | Value |
| --- | --- |
| Hardware / execution | 4× A100 SXM4 80GB, TP4, BF16, CUDA Graph decode, 8K chunks |
| Workload | C1; one pass, 2 requests each at 8K / 128K input + 256 output; greedy, ignore EOS; no client warmups |
| GPU allocations per device | Utilization 0.95; expert 64.125 GiB + KV 5.887 GiB, matched across modes |
| CPU | EPYC 7763; 12 expert threads per rank |
| Accuracy | 20 fixed 128K RULER examples, normal EOS |

| Input | Backend | TTFT s ↓ | Decode tok/s ↑ | E2E output tok/s ↑ | E2E vs UVA |
| --- | --- | ---: | ---: | ---: | ---: |
| 8K | Native UVA | 16.11 | 4.43 | 3.48 | 1.00× |
| 8K | Expert cache + CPU/GPU | 3.89 | 26.27 | 18.83 | **5.42×** |
| 128K | Native UVA | 274.51 | 4.34 | 0.77 | 1.00× |
| 128K | Expert cache + CPU/GPU | 63.55 | 23.60 | 3.44 | **4.48×** |

| Input | Backend | GPU peak GiB/device | CPU workers PSS GiB/host | Retrieval |
| --- | --- | ---: | ---: | ---: |
| 8K | Native UVA | 77.03 | 173.44 | 2/2 |
| 8K | Expert cache + CPU/GPU | 76.67 | 430.05 | 2/2 |
| 128K | Native UVA | 77.03 | 173.53 | 2/2 |
| 128K | Expert cache + CPU/GPU | 76.67 | 430.11 | 2/2 |

| Weight storage | Native UVA | Expert cache + CPU/GPU |
| --- | ---: | ---: |
| CPU expert bank per host | 166.5 GiB | 423 GiB |
| GPU-resident experts per device | 64.125 GiB | 61.875 GiB + 2.250 GiB prefill buffers |

| 128K RULER backend | Answer recall | Fully correct | Mean TTFT s |
| --- | ---: | ---: | ---: |
| Expert cache + CPU/GPU | **100%** | 20/20 | 63.35 |

## Qwen3-30B — long context and FlashPrefill, C1

| Input | Backend | TTFT s ↓ | Decode tok/s ↑ | E2E output tok/s ↑ | E2E vs UVA |
| --- | --- | ---: | ---: | ---: | ---: |
| 32K | Native UVA | 27.31 | 10.77 | 5.02 | 1.00× |
| 32K | Expert cache + CPU/GPU | 7.39 | 46.03 | 19.79 | **3.94×** |
| 32K | Expert cache + sparse prefill | 7.11 | 41.75 | 19.36 | 3.86× |
| 64K | Native UVA | 56.93 | 10.53 | 3.15 | 1.00× |
| 64K | Expert cache + CPU/GPU | 14.87 | 48.92 | 12.75 | **4.04×** |
| 64K | Expert cache + sparse prefill | 13.25 | 48.67 | 13.84 | **4.39×** |
| 128K | Native UVA | 130.66 | 10.07 | 1.64 | 1.00× |
| 128K | Expert cache + CPU/GPU | 45.27 | 36.12 | 4.89 | **2.98×** |
| 128K | Expert cache + sparse prefill | 27.83 | 36.75 | 7.36 | **4.49×** |

### Sparse prefill — versus dense expert cache

| Input | TTFT speedup | E2E throughput speedup |
| --- | ---: | ---: |
| 32K | 1.04× | 0.98× |
| 64K | 1.12× | 1.09× |
| 128K | **1.63×** | **1.50×** |

### Approximation tradeoffs

Official FlashPrefill also lost accuracy: **100% → 99%** on our saved 20-example 128K RULER subset; our 30B implementation scored **99% → 97%** against its dense baseline. The repeated 32K four-code retrieval prompt scored **2/2 dense vs 0/2 sparse**.

## Qwen3-30B — memory and resident concurrency

| Workload | GPU peak GiB, UVA / cache / cache + sparse | CPU worker PSS peak GiB, UVA / cache / cache + sparse | Peak running |
| --- | ---: | ---: | ---: |
| 2K C1 | 37.55 / 37.18 / — | 34.47 / 55.91 / — | 1 |
| 2K C8 | 37.97 / 37.83 / — | 34.49 / 55.94 / — | 8 |
| 2K C32 | 37.99 / 37.85 / — | 34.54 / 55.97 / — | 32 |
| 32K C1 | 37.91 / 37.76 / 38.20 | 34.50 / 55.91 / 55.92 | 1 |
| 64K C1 | 37.91 / 37.76 / 38.20 | 34.51 / 55.91 / 55.92 | 1 |
| 128K C1 | 37.91 / 37.76 / 38.20 | 34.51 / 55.92 / 55.88 | 1 |

| Weight storage | Native UVA | Expert cache / cache + sparse |
| --- | ---: | ---: |
| CPU expert bank | 32.625 GiB | 54 GiB |
| GPU-resident experts | 21.375 GiB | 19.125 GiB + 2.250 GiB prefill buffers |

## References

| Work | Paper | Code |
| --- | --- | --- |
| FreeToken | [Paper](https://arxiv.org/abs/2608.16157) | [Official implementation](https://github.com/FlashML-org/FreeToken) |
| FlashPrefill V2 | [Paper](https://arxiv.org/abs/2608.19758) | [Official implementation](https://github.com/qhfan/FlashPrefillv2) |
