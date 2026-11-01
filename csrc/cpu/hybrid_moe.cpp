// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <barrier>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

#include "cpu/cpu_fused_moe_activations.hpp"
#include "cpu/micro_gemm/cpu_micro_gemm_vec.hpp"

void hybrid_publish(cudaStream_t stream, const void* x, const int32_t* ids,
                    const float* weights, void* host_x, int32_t* host_ids,
                    float* host_weights, int32_t* flags, int hidden, int top_k,
                    int layer, int num_tokens);
void hybrid_combine(cudaStream_t stream, void* output, const void* cpu_output,
                    const int32_t* flags, int size);

namespace {
using bf16 = c10::BFloat16;

void check_cuda(cudaError_t rc) {
  TORCH_CHECK(rc == cudaSuccess, cudaGetErrorString(rc));
}

void pin_thread(int core) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(core, &mask);
  TORCH_CHECK(pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) == 0,
              "Cannot pin CPU expert thread");
}

template <typename T>
T* device_pointer(const torch::Tensor& tensor) {
  void* pointer = nullptr;
  check_cuda(cudaHostGetDevicePointer(&pointer, tensor.data_ptr(), 0));
  return static_cast<T*>(pointer);
}

class CPUExecutor {
  using WaitValue = CUresult (*)(CUstream, CUdeviceptr, unsigned int,
                                 unsigned int);
  std::vector<std::pair<torch::Tensor, torch::Tensor>> banks_;
  int hidden_, intermediate_, top_k_, max_tokens_, threads_;
  torch::Tensor x_, ids_, weights_, output_, flags_;
  std::vector<float> partial_;
  std::vector<bf16> intermediate_values_;
  void *device_x_, *device_output_;
  int32_t *device_ids_, *device_flags_;
  float* device_weights_;
  WaitValue wait_value_;
  std::barrier<> start_, phase_, finish_;
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> served_{0};
  int layer_ = 0;
  int num_tokens_ = 0;
  std::vector<std::thread> workers_;
  std::thread coordinator_;

  void compute(int tid) {
    const auto* ids = ids_.data_ptr<int32_t>();
    const auto* input = x_.data_ptr<bf16>();
    const auto* w13 = banks_[layer_].first.data_ptr<bf16>();
    const auto* w2 = banks_[layer_].second.data_ptr<bf16>();
    for (int index = tid * 32; index < num_tokens_ * top_k_ * intermediate_;
         index += threads_ * 32) {
      int route = index / intermediate_;
      int row = index % intermediate_;
      if (ids[route] >= 0) {
        alignas(64) float gate_up[64];
        alignas(64) float up[32];
        const auto* token_input = input + (route / top_k_) * hidden_;
        for (int j = 0; j < 32; ++j) {
          auto* weight =
              w13 +
              (int64_t(ids[route]) * 2 * intermediate_ + row + j) * hidden_;
          gate_up[j] = float(
              bf16(cpu_micro_gemm::dot_unpacked(weight, token_input, hidden_)));
          up[j] = float(bf16(cpu_micro_gemm::dot_unpacked(
              weight + int64_t(intermediate_) * hidden_, token_input,
              hidden_)));
          gate_up[j + 32] = 1.0f;
        }
        alignas(64) float activated[32];
        cpu_fused_moe_utils::silu_and_mul(gate_up, activated, 1, 64, 64, 32);
        for (int j = 0; j < 32; ++j) {
          // CUDA rounds SiLU to BF16 before multiplying by the up projection.
          intermediate_values_[index + j] =
              bf16(float(bf16(activated[j])) * up[j]);
        }
      }
    }
    phase_.arrive_and_wait();
    for (int first = tid * 32; first < num_tokens_ * top_k_ * hidden_;
         first += threads_ * 32) {
      for (int index = first; index < first + 32; ++index) {
        int route = index / hidden_;
        int row = index % hidden_;
        if (ids[route] >= 0) {
          partial_[index] = cpu_micro_gemm::dot_unpacked(
              w2 + (int64_t(ids[route]) * hidden_ + row) * intermediate_,
              intermediate_values_.data() + route * intermediate_,
              intermediate_);
        }
      }
    }
    phase_.arrive_and_wait();
    const float* weights = weights_.data_ptr<float>();
    auto* output = output_.data_ptr<bf16>();
    for (int first = tid * 32; first < num_tokens_ * hidden_;
         first += threads_ * 32) {
      for (int index = first; index < first + 32; ++index) {
        int row = index % hidden_;
        int first_route = (index / hidden_) * top_k_;
        float sum = 0;
        for (int route = first_route; route < first_route + top_k_; ++route) {
          if (ids[route] >= 0)
            sum +=
                float(bf16(partial_[route * hidden_ + row] * weights[route]));
        }
        output[index] = bf16(sum);
      }
    }
  }

 public:
  CPUExecutor(std::vector<std::pair<torch::Tensor, torch::Tensor>> banks,
              int top_k, int max_tokens, std::vector<int> cores,
              int coordinator)
      : banks_(std::move(banks)),
        hidden_(banks_[0].first.size(2)),
        intermediate_(banks_[0].second.size(2)),
        top_k_(top_k),
        max_tokens_(max_tokens),
        threads_(cores.size()),
        partial_(max_tokens * top_k * hidden_),
        intermediate_values_(max_tokens * top_k * intermediate_),
        start_(threads_ + 1),
        phase_(threads_),
        finish_(threads_ + 1) {
    TORCH_CHECK(hidden_ % 32 == 0 && intermediate_ % 32 == 0);
    for (const auto& [w13, w2] : banks_) {
      TORCH_CHECK(w13.is_cpu() && w2.is_cpu() && w13.is_pinned() &&
                  w2.is_pinned() && w13.is_contiguous() && w2.is_contiguous());
      TORCH_CHECK(w13.scalar_type() == at::kBFloat16 &&
                  w2.scalar_type() == at::kBFloat16);
      TORCH_CHECK(w13.size(1) == 2 * intermediate_ && w13.size(2) == hidden_ &&
                  w2.size(0) == w13.size(0) && w2.size(1) == hidden_ &&
                  w2.size(2) == intermediate_);
    }
    auto options =
        torch::TensorOptions().device(torch::kCPU).pinned_memory(true);
    x_ = torch::empty({max_tokens_, hidden_}, options.dtype(torch::kBFloat16));
    ids_ = torch::empty({max_tokens_, top_k_}, options.dtype(torch::kInt32));
    weights_ =
        torch::empty({max_tokens_, top_k_}, options.dtype(torch::kFloat32));
    output_ =
        torch::empty({max_tokens_, hidden_}, options.dtype(torch::kBFloat16));
    flags_ = torch::zeros({64}, options.dtype(torch::kInt32));
    device_x_ = device_pointer<bf16>(x_);
    device_ids_ = device_pointer<int32_t>(ids_);
    device_weights_ = device_pointer<float>(weights_);
    device_output_ = device_pointer<bf16>(output_);
    device_flags_ = device_pointer<int32_t>(flags_);
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(driver, "CUDA driver is unavailable");
    wait_value_ =
        reinterpret_cast<WaitValue>(dlsym(driver, "cuStreamWaitValue32_v2"));
    if (!wait_value_) {
      wait_value_ =
          reinterpret_cast<WaitValue>(dlsym(driver, "cuStreamWaitValue32"));
    }
    TORCH_CHECK(wait_value_, "CUDA stream memory operations are unavailable");
    for (int tid = 0; tid < threads_; ++tid) {
      workers_.emplace_back([this, tid, core = cores[tid]] {
        pin_thread(core);
        while (true) {
          start_.arrive_and_wait();
          if (layer_ < 0) break;
          compute(tid);
          finish_.arrive_and_wait();
        }
      });
    }
    coordinator_ = std::thread([this, coordinator] {
      pin_thread(coordinator);
      auto ready = std::atomic_ref<int32_t>(flags_.data_ptr<int32_t>()[0]);
      auto done = std::atomic_ref<int32_t>(flags_.data_ptr<int32_t>()[32]);
      while (!stop_.load()) {
        int task = ready.load(std::memory_order_acquire);
        if (!task) {
          _mm_pause();
          continue;
        }
        layer_ = task - 1;
        num_tokens_ = flags_.data_ptr<int32_t>()[2];
        ready.store(0, std::memory_order_relaxed);
        start_.arrive_and_wait();
        finish_.arrive_and_wait();
        served_.fetch_add(1, std::memory_order_relaxed);
        done.store(1, std::memory_order_release);
      }
    });
  }

  ~CPUExecutor() {
    stop_.store(true);
    coordinator_.join();
    layer_ = -1;
    start_.arrive_and_wait();
    for (auto& worker : workers_) worker.join();
  }

  void submit(uintptr_t stream, uintptr_t x, uintptr_t ids, uintptr_t weights,
              int layer, int num_tokens) {
    TORCH_CHECK(num_tokens > 0 && num_tokens <= max_tokens_);
    hybrid_publish(reinterpret_cast<cudaStream_t>(stream),
                   reinterpret_cast<void*>(x), reinterpret_cast<int32_t*>(ids),
                   reinterpret_cast<float*>(weights), device_x_, device_ids_,
                   device_weights_, device_flags_, hidden_, top_k_, layer,
                   num_tokens);
  }

  void combine(uintptr_t stream, uintptr_t output, int num_tokens) {
    TORCH_CHECK(num_tokens > 0 && num_tokens <= max_tokens_);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    TORCH_CHECK(wait_value_(reinterpret_cast<CUstream>(stream),
                            reinterpret_cast<CUdeviceptr>(device_flags_ + 32),
                            1, CU_STREAM_WAIT_VALUE_EQ) == CUDA_SUCCESS);
    hybrid_combine(cuda_stream, reinterpret_cast<void*>(output), device_output_,
                   device_flags_, num_tokens * hidden_);
  }

  int64_t served() const { return served_.load(); }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<CPUExecutor>(module, "CPUExecutor")
      .def(pybind11::init<std::vector<std::pair<torch::Tensor, torch::Tensor>>,
                          int, int, std::vector<int>, int>())
      .def("submit", &CPUExecutor::submit)
      .def("combine", &CPUExecutor::combine)
      .def("served", &CPUExecutor::served);
}
