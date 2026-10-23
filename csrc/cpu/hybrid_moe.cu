// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace {
__global__ void publish(const __nv_bfloat16* x, const int32_t* ids,
                        const float* weights, __nv_bfloat16* host_x,
                        int32_t* host_ids, float* host_weights, int32_t* flags,
                        int hidden, int top_k, int layer) {
  __shared__ bool active;
  if (threadIdx.x == 0) {
    active = false;
    for (int i = 0; i < top_k; ++i) active |= ids[i] >= 0;
    flags[1] = active;
    flags[32] = active ? 0 : 1;
  }
  __syncthreads();
  if (!active) return;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) host_x[i] = x[i];
  for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
    host_ids[i] = ids[i];
    host_weights[i] = weights[i];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *reinterpret_cast<volatile int32_t*>(flags) = layer + 1;
  }
}

__global__ void combine(__nv_bfloat16* output, const __nv_bfloat16* cpu_output,
                        const int32_t* flags, int hidden) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < hidden && flags[1]) {
    output[index] = __float2bfloat16(__bfloat162float(output[index]) +
                                     __bfloat162float(cpu_output[index]));
  }
}

void check(cudaError_t rc) {
  if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
}  // namespace

void hybrid_publish(cudaStream_t stream, const void* x, const int32_t* ids,
                    const float* weights, void* host_x, int32_t* host_ids,
                    float* host_weights, int32_t* flags, int hidden, int top_k,
                    int layer) {
  publish<<<1, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), ids,
                                 weights, static_cast<__nv_bfloat16*>(host_x),
                                 host_ids, host_weights, flags, hidden, top_k,
                                 layer);
  check(cudaGetLastError());
}

void hybrid_combine(cudaStream_t stream, void* output, const void* cpu_output,
                    const int32_t* flags, int hidden) {
  combine<<<(hidden + 255) / 256, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(output),
      static_cast<const __nv_bfloat16*>(cpu_output), flags, hidden);
  check(cudaGetLastError());
}
