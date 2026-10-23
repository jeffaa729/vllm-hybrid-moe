# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert offloading with a model-wide, device-managed LRU cache.

Follows FreeToken's offload-cache algorithm: protect this step's routes, fetch
misses into least-recently-used slots, optionally computing residual misses on CPU.
Reference: FlashML-org/FreeToken, commit f02fbb16a213f5c81094e4017c988d41f0164141.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import CPUExpertExecutor

logger = init_logger(__name__)


@triton.jit(do_not_specialize=["layer_id"])
def _ensure_experts(
    routes,
    slot_for_id,
    id_of_slot,
    usage,
    clock,
    sources,
    destinations,
    stats,
    layer_id,
    E: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    PREFILL: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
    cpu_routes=None,
    recency=None,
    fetch_counts=None,
    cpu_stats=None,
    HYBRID: tl.constexpr = False,
):
    experts = tl.arange(0, BLOCK_E)
    slots = tl.arange(0, BLOCK_C)
    base = layer_id * E
    step = tl.load(clock) + 1
    tl.store(clock, step)
    active = (experts < E) & PREFILL
    if not PREFILL:
        for i in range(K):
            active |= experts == tl.load(routes + i)
    mapped = tl.load(slot_for_id + base + experts, experts < E, other=-1)
    missing = active & (mapped < 0)
    ranks = tl.cumsum(missing.to(tl.int32)) - 1
    count = tl.sum(missing.to(tl.int32))
    fetch = count
    if HYBRID:
        fetch = tl.load(fetch_counts + count)
        recent = tl.load(recency + base + experts, experts < E, other=0)
        priority = tl.where(missing, recent * E + E - 1 - experts, -1)
    tl.store(sources + experts, -1, experts < E)
    tl.store(usage + mapped, step, active & (mapped >= 0))
    tl.debug_barrier()

    owners = tl.load(id_of_slot + slots, slots < C, other=-1)
    ages = tl.load(usage + slots, slots < C, other=0)
    protected = (owners >= base) & (owners < base + E) & PREFILL
    if not PREFILL:
        for i in range(K):
            protected |= owners == base + tl.load(routes + i)
    ages = tl.where(protected | (slots >= C), 0x7FFFFFFFFFFFFFFF, ages)
    for i in range(fetch):
        victim = tl.argmin(ages, 0).to(tl.int32)
        old_id = tl.sum(tl.where(slots == victim, owners, 0))
        if HYBRID:
            expert = tl.argmax(priority, 0).to(tl.int32)
            priority = tl.where(experts == expert, -1, priority)
        else:
            expert = tl.min(tl.where(missing & (ranks == i), experts, E), 0)
        tl.store(slot_for_id + old_id, -1, old_id >= 0)
        tl.store(slot_for_id + base + expert, victim)
        tl.store(id_of_slot + victim, base + expert)
        tl.store(usage + victim, step)
        tl.store(sources + i, expert)
        tl.store(destinations + i, victim)
        ages = tl.where(slots == victim, 0x7FFFFFFFFFFFFFFF, ages)
    if HYBRID:
        tl.debug_barrier()
        for i in range(K):
            expert = tl.load(routes + i)
            slot = tl.load(slot_for_id + base + expert)
            tl.store(cpu_routes + i, tl.where(slot < 0, expert, -1))
        tl.store(recency + base + experts, step, active)
        tl.store(cpu_stats, tl.load(cpu_stats) + count - fetch)
        tl.store(cpu_stats + 1, tl.load(cpu_stats + 1) + (count > fetch))
        tl.store(cpu_stats + 2, tl.load(cpu_stats + 2) + fetch)
    # Separate decode/prefill counters: unique active experts, misses, calls.
    stats += 3 * PREFILL
    tl.store(stats, tl.load(stats) + tl.sum(active.to(tl.int64)))
    tl.store(stats + 1, tl.load(stats + 1) + count)
    tl.store(stats + 2, tl.load(stats + 2) + 1)


@triton.jit
def _copy_experts(
    host_w13,
    host_w2,
    gpu_w13,
    gpu_w2,
    sources,
    destinations,
    N13: tl.constexpr,
    N2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    source = tl.load(sources + row).to(tl.int64)
    if source >= 0:
        destination = tl.load(destinations + row).to(tl.int64)
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        up = tl.load(host_w13 + source * N13 + offsets, offsets < N13, other=0)
        tl.store(gpu_w13 + destination * N13 + offsets, up, offsets < N13)
        down = tl.load(host_w2 + source * N2 + offsets, offsets < N2, other=0)
        tl.store(gpu_w2 + destination * N2 + offsets, down, offsets < N2)


class GPUExpertCache:
    """Shared GPU slots backed by one pinned, GPU-layout host bank per layer.

    Calls are serialized by the model runner, including decode graph replay.
    Prefill borrows two layer-sized buffers without replacing the remaining cache.
    """

    def __init__(self, w13, w2, num_layers: int, capacity: int | str, cpu_config=None):
        self.cpu_config = cpu_config
        self.cpu_executor: CPUExpertExecutor | None = None
        self.num_experts = w13.shape[0]
        self.auto_size = capacity == "auto"
        if self.auto_size:
            capacity = min(2 * self.num_experts, num_layers * self.num_experts)
        elif not isinstance(capacity, int):
            raise ValueError("Expert cache capacity must be an integer or 'auto'")
        if not self.num_experts <= capacity <= num_layers * self.num_experts:
            raise ValueError("Expert cache must hold at least one complete layer")
        self.capacity = capacity
        self.w13 = torch.empty(
            (capacity, *w13.shape[1:]), dtype=w13.dtype, device="cuda"
        )
        self.w2 = torch.empty((capacity, *w2.shape[1:]), dtype=w2.dtype, device="cuda")
        self.slot_for_id = torch.full(
            (num_layers, self.num_experts), -1, dtype=torch.int32, device="cuda"
        )
        self.id_of_slot = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
        self.usage = torch.zeros(capacity, dtype=torch.int64, device="cuda")
        self.clock = torch.zeros((), dtype=torch.int64, device="cuda")
        self.sources = torch.empty(self.num_experts, dtype=torch.int32, device="cuda")
        self.destinations = torch.empty_like(self.sources)
        self.stats = torch.zeros(6, dtype=torch.int64, device="cuda")
        self.host_weights: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.host_views: list[tuple[torch.Tensor, torch.Tensor]] = []
        if cpu_config:
            self.recency = torch.zeros_like(self.slot_for_id, dtype=torch.int64)
            self.fetch_counts = torch.tensor(
                cpu_config["fetch_counts"], dtype=torch.int32, device="cuda"
            )
            self.cpu_stats = torch.zeros(3, dtype=torch.int64, device="cuda")
        self.expert_bytes = w13[0].nbytes + w2[0].nbytes
        self.prefill_weights = [
            (
                self.w13[i * self.num_experts : (i + 1) * self.num_experts],
                self.w2[i * self.num_experts : (i + 1) * self.num_experts],
            )
            for i in range(
                2
                if 2 * self.num_experts <= capacity < num_layers * self.num_experts
                else 0
            )
        ]
        self.prefill_stream = torch.cuda.Stream(device=self.w13.device)
        self.prefill_ready = [torch.cuda.Event() for _ in self.prefill_weights]
        self.prefill_done = [torch.cuda.Event() for _ in self.prefill_weights]
        self.prefill_layers = [-1] * len(self.prefill_weights)
        self.prefill_ids = torch.arange(
            self.num_experts, dtype=torch.int32, device=self.w13.device
        )
        logger.info(
            "GPU expert cache: %d slots, %.6f GiB GPU, %.6f GiB host backing",
            capacity,
            capacity * self.expert_bytes / 2**30,
            num_layers * self.num_experts * self.expert_bytes / 2**30,
        )

    def add_layer(self, w13, w2) -> int:
        if len(self.host_weights) >= self.slot_for_id.shape[0]:
            raise ValueError("More expert layers than the configured host bank")
        for host, gpu in ((w13, self.w13), (w2, self.w2)):
            assert host.is_pinned() and host.is_contiguous()
            assert host.shape[1:] == gpu.shape[1:] and host.dtype == gpu.dtype
        layer_id = len(self.host_weights)
        self.host_weights.append((w13, w2))
        self.host_views.append(
            (
                get_accelerator_view_from_cpu_tensor(w13),
                get_accelerator_view_from_cpu_tensor(w2),
            )
        )
        reserved = len(self.prefill_weights) * self.num_experts
        first_id = self.slot_for_id.numel() - (self.capacity - reserved)
        first_expert = max(0, first_id - layer_id * self.num_experts)
        if first_expert < self.num_experts:
            first_slot = (
                reserved + layer_id * self.num_experts + first_expert - first_id
            )
            last_slot = first_slot + self.num_experts - first_expert
            self.w13[first_slot:last_slot].copy_(w13[first_expert:], non_blocking=True)
            self.w2[first_slot:last_slot].copy_(w2[first_expert:], non_blocking=True)
            slots = torch.arange(
                first_slot, last_slot, dtype=torch.int32, device=self.w13.device
            )
            self.slot_for_id[layer_id, first_expert:] = slots
            self.id_of_slot[first_slot:last_slot] = (
                layer_id * self.num_experts + first_expert + slots - first_slot
            )
        return layer_id

    def finish_startup_sizing(self, config, kv_specs, available_memory):
        """Reserve native maximum-context KV, then fill remaining bytes with experts."""
        from vllm.v1.core.kv_cache_planning import max_memory_usage_bytes

        if not self.auto_size:
            return available_memory, 0
        required_kv = max_memory_usage_bytes(config, kv_specs.values())
        required_kv *= config.scheduler_config.max_num_seqs
        # BlockPool permanently reserves one null block.
        required_kv += sum(spec.page_size_bytes for spec in kv_specs.values())
        if available_memory < required_kv:
            raise ValueError(
                "Insufficient profiled memory for the requested KV capacity"
            )
        slot_bytes = (
            self.expert_bytes
            + self.id_of_slot.element_size()
            + self.usage.element_size()
        )
        capacity = min(
            self.slot_for_id.numel(),
            self.capacity + (available_memory - required_kv) // slot_bytes,
        )
        banks, executor = self.host_weights, self.cpu_executor
        before = torch.accelerator.memory_allocated(self.w13.device)
        torch.accelerator.synchronize(self.w13.device)
        self.prefill_stream.synchronize()
        self.prefill_weights.clear()
        del self.w13, self.w2, self.id_of_slot, self.usage
        torch.accelerator.empty_cache()
        GPUExpertCache.__init__(self, *banks[0], len(banks), capacity, self.cpu_config)
        for weights in banks:
            self.add_layer(*weights)
        self.host_weights, self.cpu_executor = banks, executor
        torch.accelerator.synchronize(self.w13.device)
        delta = torch.accelerator.memory_allocated(self.w13.device) - before
        available_memory -= delta
        if available_memory < required_kv:
            raise ValueError("Final expert allocation exceeds the profiled KV budget")
        logger.info(
            "Expert cache auto-sized: %d slots, %.6f GiB GPU (buffers included), "
            "%.6f GiB KV available, %.6f GiB required for %d tokens x %d requests",
            capacity,
            capacity * self.expert_bytes / 2**30,
            available_memory / 2**30,
            required_kv / 2**30,
            config.model_config.max_model_len,
            config.scheduler_config.max_num_seqs,
        )
        return available_memory, delta

    def _begin_prefill(self):
        owners = self.id_of_slot[: 2 * self.num_experts]
        self.slot_for_id.view(-1)[owners[owners >= 0].long()] = -1
        owners.fill_(-1)
        self.usage[: 2 * self.num_experts].zero_()
        self.prefill_layers = [-1, -1]
        self.prefill_snapshot = self.slot_for_id.cpu().tolist()
        self.prefill_stream.wait_stream(torch.cuda.current_stream())

    def _prefetch_layer(self, layer_id):
        if layer_id >= len(self.host_weights):
            return
        buffer_id = layer_id % 2
        if self.prefill_layers[buffer_id] == layer_id:
            return
        weights = self.prefill_weights[buffer_id]
        mapped = self.slot_for_id[layer_id]
        sources = torch.where(mapped >= 2 * self.num_experts, mapped, -1)
        n13, n2 = self.w13[0].numel(), self.w2[0].numel()
        _copy_experts[(triton.cdiv(max(n13, n2), 4096), self.num_experts)](
            self.w13, self.w2, *weights, sources, self.prefill_ids, n13, n2, 4096
        )
        missing = [
            i
            for i, slot in enumerate(self.prefill_snapshot[layer_id])
            if slot < 2 * self.num_experts
        ]
        runs: list[tuple[int, int]] = []
        for expert in missing:
            if runs and runs[-1][1] == expert:
                runs[-1] = (runs[-1][0], expert + 1)
            else:
                runs.append((expert, expert + 1))
        with torch.cuda.stream(self.prefill_stream):
            if self.prefill_layers[buffer_id] >= 0:
                self.prefill_stream.wait_event(self.prefill_done[buffer_id])
            for source, target in zip(self.host_weights[layer_id], weights):
                for first, last in runs:
                    target[first:last].copy_(source[first:last], non_blocking=True)
            self.prefill_ready[buffer_id].record()
        self.prefill_layers[buffer_id] = layer_id
        self.stats[3].add_(self.num_experts)
        self.stats[4].add_(len(missing))
        self.stats[5].add_(1)

    def _apply_prefill(self, layer_id, layer, x, topk_weights, topk_ids):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        if layer_id == 0:
            self._begin_prefill()
        self._prefetch_layer(layer_id)
        buffer_id = layer_id % 2
        torch.cuda.current_stream().wait_event(self.prefill_ready[buffer_id])
        self._prefetch_layer(layer_id + 1)
        output = fused_experts(
            x,
            self.prefill_weights[buffer_id][0],
            self.prefill_weights[buffer_id][1],
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=self.num_experts,
        )
        self.prefill_done[buffer_id].record()
        return output

    def apply(self, layer_id, layer, x, topk_weights, topk_ids, *, is_prefill):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        assert topk_ids.is_contiguous()
        assert is_prefill or x.shape[0] == 1
        if self.cpu_config and self.cpu_executor is None:
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertExecutor,
            )

            if len(self.host_weights) != self.slot_for_id.shape[0]:
                raise ValueError("CPU expert execution requires the complete host bank")
            counts = self.cpu_config["fetch_counts"]
            if len(counts) != topk_ids.shape[1] + 1 or any(
                not isinstance(n, int) or not 0 <= n <= misses
                for misses, n in enumerate(counts)
            ):
                raise ValueError("fetch_counts must map each miss count to 0..misses")
            self.cpu_executor = CPUExpertExecutor(
                self.host_weights, topk_ids.shape[1], self.cpu_config
            )
        if is_prefill and self.prefill_weights:
            return self._apply_prefill(layer_id, layer, x, topk_weights, topk_ids)
        cpu = self.cpu_executor if not is_prefill else None
        _ensure_experts[(1,)](
            topk_ids,
            self.slot_for_id,
            self.id_of_slot,
            self.usage,
            self.clock,
            self.sources,
            self.destinations,
            self.stats,
            layer_id,
            self.num_experts,
            self.capacity,
            topk_ids.shape[1],
            is_prefill,
            triton.next_power_of_2(self.num_experts),
            triton.next_power_of_2(self.capacity),
            cpu_routes=cpu.routes if cpu else None,
            recency=self.recency if cpu else None,
            fetch_counts=self.fetch_counts if cpu else None,
            cpu_stats=self.cpu_stats if cpu else None,
            HYBRID=cpu is not None,
        )
        if cpu:
            cpu.submit(layer_id, x, topk_weights)
        n13, n2 = self.w13[0].numel(), self.w2[0].numel()
        rows = self.num_experts if is_prefill else topk_ids.shape[1]
        _copy_experts[(triton.cdiv(max(n13, n2), 4096), rows)](
            *self.host_views[layer_id],
            self.w13,
            self.w2,
            self.sources,
            self.destinations,
            n13,
            n2,
            4096,
        )
        output = fused_experts(
            x,
            self.w13,
            self.w2,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=self.num_experts,
            expert_map=self.slot_for_id[layer_id],
        )
        return cpu.merge(output) if cpu else output
