# vLLM Hybrid MoE

## Implementation

| Technique | Implementation |
| --- | --- |
| Expert cache | Model-wide LRU, expert-granularity transfers, cache preloading and double-buffered GPU prefill |
| CPU/GPU execution | GPU-resident/fetched experts and CPU residual misses share one CPU weight bank; graph-enabled decode and TP support |
| Memory planning | Startup profiling-based expert/KV sizing; fixed matched allocations in these benchmarks |
| Sparse prefill | FlashPrefill V2-style block selection and mean correction, with an A100 compute kernel |

## Launch — expert cache + CPU/GPU + sparse prefill

With this checkout installed in the active virtual environment (1× A100 40GB):

```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 127.0.0.1 --port 18080 --dtype bfloat16 \
  --max-num-seqs 8 --max-model-len 131328 \
  --max-num-batched-tokens 8192 --no-enable-prefix-caching \
  --generation-config vllm --gpu-memory-utilization 0.95 \
  --attention-config '{"backend":"FLASH_ATTN","sparse_prefill":{}}' \
  --kernel-config '{"moe_backend":"triton"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --kv-cache-memory-bytes 12914786304 \
  --additional-config '{"expert_cache":{"num_slots":2432,"cpu":{"num_threads":6,"fetch_counts":[0,1,1,1,1,2,2,2,3]}}}'
```

Sparse defaults: threshold 0.1, 4 dense layers, 16 local blocks, 2 sink blocks, 8 dense tail tiles.

## Qwen3-30B — Expert cache

| Setting | Value |
| --- | --- |
| Hardware | 1× A100 40GB; EPYC 7532, 12 expert threads |
| Execution | BF16, TP1, CUDA Graph decode, 8K chunks |
| Matched GPU allocations | Expert 21.375 GiB; KV 12.028 GiB |
| Workload | C1; 2 requests/length + 256 output tokens, ignore EOS |

| Input | Backend | TTFT s ↓ | Decode tok/s ↑ | E2E output tok/s ↑ | E2E vs UVA |
| --- | --- | ---: | ---: | ---: | ---: |
| 32K | Native UVA | 27.31 | 10.77 | 5.02 | 1.00× |
| 32K | Expert cache + CPU/GPU | **7.39** | **46.03** | **19.79** | **3.94×** |
| 64K | Native UVA | 56.93 | 10.53 | 3.15 | 1.00× |
| 64K | Expert cache + CPU/GPU | **14.87** | **48.92** | **12.75** | **4.04×** |
| 128K | Native UVA | 130.66 | 10.07 | 1.64 | 1.00× |
| 128K | Expert cache + CPU/GPU | **45.27** | **36.12** | **4.89** | **2.98×** |

| Weight storage | Native UVA | Expert cache |
| --- | ---: | ---: |
| CPU expert bank | 32.625 GiB | 54 GiB |
| GPU-resident experts | 21.375 GiB | 19.125 GiB + 2.250 GiB prefill buffers |

## Qwen3-30B — Expert cache + FlashPrefill-style sparse prefill

### Prefill performance

| Setting | Value |
| --- | --- |
| Hardware / execution | 1× A100 40GB; EPYC 7343, 12 expert threads; BF16, TP1, CUDA Graph decode, 8K chunks |
| Matched GPU allocations | Expert 21.375 GiB; KV 12.028 GiB |
| Workload | C1; 2 retrieval requests/length + 256 output tokens, ignore EOS |
| Sparse settings | Threshold 0.1; 4 dense layers, 16 local blocks |

| Input | Backend | TTFT s ↓ | TTFT speedup | Decode tok/s ↑ | E2E output tok/s ↑ | E2E speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32K | Expert cache + CPU/GPU | 7.62 | 1.00× | 38.02 | 17.87 | 1.00× |
| 32K | Expert cache + CPU/GPU + sparse prefill | **7.37** | **1.03×** | 38.11 | **18.21** | **1.02×** |
| 64K | Expert cache + CPU/GPU | 18.52 | 1.00× | 45.05 | 10.59 | 1.00× |
| 64K | Expert cache + CPU/GPU + sparse prefill | **13.48** | **1.37×** | 42.51 | **13.15** | **1.24×** |
| 128K | Expert cache + CPU/GPU | 63.81 | 1.00× | 32.78 | 3.58 | 1.00× |
| 128K | Expert cache + CPU/GPU + sparse prefill | **29.50** | **2.16×** | 32.86 | **6.87** | **1.92×** |

### RULER accuracy — 16 local blocks

| Setting | Value |
| --- | --- |
| Hardware | 1× A100 40GB; EPYC 7343, 6 expert threads |
| Execution | BF16, TP1, CUDA Graph decode, 8K chunks |
| GPU allocations | Expert 21.375 GiB; KV 12.028 GiB |
| Workload | 20 fixed 128K RULER examples, C1; normal EOS, 256-token cap |
| Sparse settings | Threshold 0.1; 4 dense layers, 16 local blocks, 2 sink blocks, 8 dense tail tiles |

| Backend | TTFT s ↓ | Request latency s ↓ | RULER recall ↑ | All references | UUID recall | Variable tracking recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert cache + CPU/GPU + sparse prefill | **29.40** | **29.91** | **99%** | **19/20** | **100%** | **96%** |

## Qwen3-235B-A22B-Instruct-2507 — TP4 expert cache

| Setting | Value |
| --- | --- |
| Hardware | 4× A100 80GB; EPYC 7763, 12 expert threads/rank |
| Execution | BF16, TP4, CUDA Graph decode, 8K chunks |
| Workload | C1; 2 requests/length, 8K / 128K input + 256 output, ignore EOS |
| Matched GPU allocations/device | Expert 64.125 GiB; KV 5.887 GiB; utilization 0.95 |

| Input | Backend | TTFT s ↓ | Decode tok/s ↑ | E2E output tok/s ↑ | E2E vs UVA |
| --- | --- | ---: | ---: | ---: | ---: |
| 8K | Native UVA | 16.11 | 4.43 | 3.48 | 1.00× |
| 8K | Expert cache + CPU/GPU | **3.89** | **26.27** | **18.83** | **5.42×** |
| 128K | Native UVA | 274.51 | 4.34 | 0.77 | 1.00× |
| 128K | Expert cache + CPU/GPU | **63.55** | **23.60** | **3.44** | **4.48×** |

| Input | Backend | GPU peak GiB/device | CPU workers PSS GiB/host |
| --- | --- | ---: | ---: |
| 8K | Native UVA | 77.03 | 173.44 |
| 8K | Expert cache + CPU/GPU | 76.67 | 430.05 |
| 128K | Native UVA | 77.03 | 173.53 |
| 128K | Expert cache + CPU/GPU | 76.67 | 430.11 |

| Weight storage | Native UVA | Expert cache + CPU/GPU |
| --- | ---: | ---: |
| CPU expert bank per host | 166.5 GiB | 423 GiB |
| GPU-resident experts per device | 64.125 GiB | 61.875 GiB + 2.250 GiB prefill buffers |

## References

| Work | Paper | Code |
| --- | --- | --- |
| FreeToken | [Paper](https://arxiv.org/abs/2608.16157) | [Official implementation](https://github.com/FlashML-org/FreeToken) |
| FlashPrefill V2 | [Paper](https://arxiv.org/abs/2608.19758) | [Official implementation](https://github.com/qhfan/FlashPrefillv2) |
| ShareGPT | — | [Dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| RULER | — | [Official implementation](https://github.com/NVIDIA/RULER) |

[RTX 5090 benchmarks](5090.md)
