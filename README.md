# vLLM Hybrid MoE

## Implementation

| Technique | Implementation |
| --- | --- |
| Expert cache | Model-wide LRU, expert-granularity transfers, cache preloading and double-buffered GPU prefill |
| CPU/GPU execution | GPU-resident/fetched experts and CPU residual misses share one CPU weight bank; graph-enabled decode and TP support |
| Memory planning | Startup profiling-based expert/KV sizing; fixed matched allocations in these benchmarks |
| Sparse prefill | FlashPrefill V2-style block selection and mean correction, with an A100 compute kernel |

## Qwen3-30B — RTX 5090 benchmark conditions

| Setting | Value |
| --- | --- |
| Model / GPU | Qwen3-30B-A3B-Instruct-2507 / 1× RTX 5090 32GB |
| CPU / PCIe | EPYC 7402P; 12 expert threads; PCIe 4.0 ×16 |
| Execution | BF16, TP1, CUDA Graph decode, 8K chunks; prefix caching off |
| Attention | Dense FA2 / sparse prefill at threshold 0.1; sparse activates at 4K, first 4 layers dense |
| Matched GPU allocations | Expert 24.75 GiB, including cache prefill buffers; KV 1.286 GiB / 14,048 tokens |
| ShareGPT | 20 fixed real prompts, 278–916 input tokens; C1 / C8 |
| RULER | 20 fixed 8K-target examples, four tasks; C1 |
| Generation / timing | Greedy, normal EOS, 256-token cap; one pass; no client warmups |
| Metrics | Native vllm bench serve; decode = 1 / mean TPOT; E2E includes prefill + decode |

## Qwen3-30B — ShareGPT serving

| Concurrency | Backend | TTFT s ↓ | Decode/request tok/s ↑ | E2E output tok/s ↑ | E2E vs UVA |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 | Native UVA | 1.60 | 11.85 | 11.07 | 1.00× |
| C1 | Expert cache + CPU/GPU | 1.33 | **77.21** | **55.29** | **4.99×** |
| C1 | Expert cache + CPU/GPU + sparse enabled | 1.22 | 77.63 | 56.87 | 5.14× |
| C8 | Native UVA | 4.16 | 2.33 | 15.66 | 1.00× |
| C8 | Expert cache + CPU/GPU | 2.44 | **10.30** | **67.44** | **4.31×** |
| C8 | Expert cache + CPU/GPU + sparse enabled | 2.47 | 11.18 | 71.98 | 4.60× |

| Concurrency | TTFT speedup | Decode speedup | Mean output tokens, UVA / ours | Peak resident requests |
| --- | ---: | ---: | ---: | ---: |
| C1 | 1.20× | **6.52×** | 256 / 256 | 1 |
| C8 | 1.70× | **4.41×** | 256 / 256 | 8 |

| Sparse activation on ShareGPT | Value |
| --- | --- |
| Sparse attention calls | None: all inputs are below 4K; these rows use dense attention |

## Qwen3-30B — RTX 5090 RULER accuracy and prefill

| Backend | Answer recall | All reference answers | TTFT s ↓ | Request latency s ↓ |
| --- | ---: | ---: | ---: | ---: |
| Native UVA | **100%** | 20/20 | 5.92 | 7.51 |
| Expert cache + CPU/GPU | **100%** | 20/20 | **1.29 (4.58×)** | **1.57 (4.77×)** |
| Expert cache + CPU/GPU + sparse prefill | **100%** | 20/20 | 1.30 (4.57×) | 1.57 (4.79×) |

| Subset | Value |
| --- | --- |
| Tasks | `niah_single_1`, `niah_multikey_2`, `niah_multikey_3`, `vt`; 5 examples each |
| Input / output | 6,139–8,161 input tokens; mean 19.75 output tokens in all three modes; no output-cap hits |
| Sparse vs dense cache | Recall 100% → 100%; TTFT 1.293 → 1.295 s |

## Qwen3-30B — RTX 5090 memory

| Workload | GPU peak GiB, UVA / cache / cache + sparse | CPU worker PSS peak GiB, UVA / cache / cache + sparse |
| --- | ---: | ---: |
| ShareGPT C1 | 30.40 / 29.96 / 30.47 | 31.73 / 56.54 / 56.55 |
| ShareGPT C8 | 30.59 / 30.36 / 30.88 | 31.69 / 56.50 / 56.51 |
| RULER C1 | 30.74 / 30.77 / 31.29 | 31.65 / 56.46 / 56.46 |

| Weight storage | Native UVA | Expert cache + CPU/GPU |
| --- | ---: | ---: |
| CPU expert bank | 29.25 GiB | 54 GiB |
| GPU expert allocation | 24.75 GiB | 22.50 GiB + 2.25 GiB prefill buffers |

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

## Qwen3-30B — RTX 5090 long-context RULER

| Setting | Value |
| --- | --- |
| Model / GPU | Qwen3-30B-A3B-Instruct-2507 / 1× RTX 5090 32GB |
| Execution | BF16, TP1, CUDA Graph decode, 8K chunks; sparse threshold 0.1 |
| Matched allocations, both lengths | Expert 13.5 GiB (11.25 GiB resident + 2.25 GiB prefill buffers); KV 12.028 GiB / 131,376 tokens |
| Workload | C1; 20 fixed RULER examples per length and mode, same four tasks and seed 42; one pass |
| Generation | Greedy, normal EOS, 256-token cap; no client warmups |

| Input | Backend | TTFT s ↓ | TTFT speedup | Request latency s ↓ | Latency speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 64K | Expert cache + CPU/GPU | 14.85 | 1.00× | 15.28 | 1.00× |
| 64K | Expert cache + CPU/GPU + sparse prefill | **13.42** | **1.11×** | **13.84** | **1.10×** |
| 128K | Expert cache + CPU/GPU | 45.27 | 1.00× | 45.75 | 1.00× |
| 128K | Expert cache + CPU/GPU + sparse prefill | **28.06** | **1.61×** | **28.97** | **1.58×** |

| Input | GPU peak GiB, dense / sparse | CPU worker PSS peak GiB, dense / sparse | Mean output tokens, dense / sparse | Output-cap hits, dense / sparse |
| --- | ---: | ---: | ---: | ---: |
| 64K | 30.02 / 30.56 | 56.46 / 56.47 | 18.95 / 19.05 | 0 / 0 |
| 128K | 30.02 / 30.56 | 56.43 / 56.44 | 18.65 / 30.85 | 0 / 1 |

<details>
<summary>RULER accuracy and approximation tradeoffs</summary>

Mean reference-answer recall; all refs = every expected answer present.

| Input | Backend | Recall | All refs |
| --- | --- | ---: | ---: |
| 64K | Expert cache + CPU/GPU | 100% | 20/20 |
| 64K | Expert cache + CPU/GPU + sparse prefill | 100% | 20/20 |
| 128K | Expert cache + CPU/GPU | 100% | 20/20 |
| 128K | Expert cache + CPU/GPU + sparse prefill | 91% | 16/20 |

| Task recall | 64K dense / sparse | 128K dense / sparse |
| --- | ---: | ---: |
| Number retrieval, `niah_single_1` | 100% / 100% | 100% / 100% |
| Number retrieval with distractors, `niah_multikey_2` | 100% / 100% | 100% / 100% |
| UUID retrieval, `niah_multikey_3` | 100% / 100% | 100% / 80% |
| Variable tracking, `vt` | 100% / 100% | 100% / 84% |

</details>

## Qwen3-30B — A100 long context and FlashPrefill, C1

| Setting | Value |
| --- | --- |
| Hardware / execution | 1× A100 40GB; BF16, TP1, CUDA Graph decode, 8K chunks |
| Matched GPU allocations | Expert 21.375 GiB; KV 12.028 GiB |
| CPU | EPYC 7532; 12 expert threads |
| Workload | 2 custom retrieval requests per length; 256 output tokens, ignore EOS; one pass |

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

## Qwen3-30B — A100 memory

| Workload | GPU peak GiB, UVA / cache / cache + sparse | CPU worker PSS peak GiB, UVA / cache / cache + sparse | Peak running |
| --- | ---: | ---: | ---: |
| 32K C1 | 37.91 / 37.76 / 38.20 | 34.50 / 55.91 / 55.92 | 1 |
| 64K C1 | 37.91 / 37.76 / 38.20 | 34.51 / 55.91 / 55.92 | 1 |
| 128K C1 | 37.91 / 37.76 / 38.20 | 34.51 / 55.92 / 55.88 | 1 |

| Weight storage | Native UVA | Expert cache / cache + sparse |
| --- | ---: | ---: |
| CPU expert bank | 32.625 GiB | 54 GiB |
| GPU-resident experts | 21.375 GiB | 19.125 GiB + 2.250 GiB prefill buffers |

## Benchmark command

Saved fixtures, server commands and environments: `benchmark-results/rtx5090-real-20261029/`.
Use `sharegpt-20.jsonl` at C1/C8, or `ruler-8k-20.jsonl` at C1.

```bash
FIXTURE=benchmark-results/rtx5090-real-20261029/datasets/sharegpt-20.jsonl
CONCURRENCY=1
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend openai --host 127.0.0.1 --port 18080 --model moe-cv-5090 \
  --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --dataset-name custom --dataset-path "$FIXTURE" --custom-output-len 256 \
  --num-prompts 20 --max-concurrency "$CONCURRENCY" --request-rate inf \
  --seed 0 --temperature 0 --skip-chat-template --disable-shuffle --no-oversample \
  --num-warmups 0 --ready-check-timeout-sec 0 --save-result --save-detailed
```

## References

| Work | Paper | Code |
| --- | --- | --- |
| FreeToken | [Paper](https://arxiv.org/abs/2608.16157) | [Official implementation](https://github.com/FlashML-org/FreeToken) |
| FlashPrefill V2 | [Paper](https://arxiv.org/abs/2608.19758) | [Official implementation](https://github.com/qhfan/FlashPrefillv2) |
| ShareGPT | — | [Dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| RULER | — | [Official implementation](https://github.com/NVIDIA/RULER) |
