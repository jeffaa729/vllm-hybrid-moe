# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80+ paged sparse prefill, following FlashPrefill V2 (arXiv:2608.19758).

Reference: https://github.com/qhfan/FlashPrefillv2, commit 75b58f2.
Selection aggregates 128 packed query rows; two 64-row SM80 compute tiles
share that selection and compensate pruned blocks with their K/V means.
"""

import torch

from vllm.config.attention import SparsePrefillConfig
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.triton_attention_helpers import resolve_seq_and_query_len
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments
from vllm.v1.worker.workspace import current_workspace_manager


@triton.jit
def _pool_blocks(
    K,
    V,
    Table,
    Lengths,
    KM,
    VM,
    stride_kp: tl.int64,
    stride_kt: tl.int64,
    stride_kh: tl.int64,
    stride_vp: tl.int64,
    stride_vt: tl.int64,
    stride_vh: tl.int64,
    stride_tb: tl.int64,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lengths + seq)
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, DIM)
    pages = tl.load(
        Table + seq * stride_tb + tokens // PAGE, tokens < length, other=0
    ).to(tl.int64)
    k = tl.load(
        K
        + pages[:, None] * stride_kp
        + (tokens % PAGE)[:, None] * stride_kt
        + head * stride_kh
        + dims[None, :],
        tokens[:, None] < length,
        other=0,
    ).to(tl.float32)
    v = tl.load(
        V
        + pages[:, None] * stride_vp
        + (tokens % PAGE)[:, None] * stride_vt
        + head * stride_vh
        + dims[None, :],
        tokens[:, None] < length,
        other=0,
    ).to(tl.float32)
    count = tl.maximum(tl.minimum(length - block * BLOCK, BLOCK), 1)
    offset = ((seq.to(tl.int64) * CAPACITY + block) * HEADS + head) * DIM + dims
    tl.store(KM + offset, tl.sum(k, 0) / count)
    tl.store(VM + offset, tl.sum(v, 0) / count)


@triton.jit
def _select_blocks(
    Q,
    KM,
    Starts,
    Lengths,
    Energies,
    Maxima,
    Indices,
    Selected,
    Counts,
    stride_qt: tl.int64,
    stride_qh: tl.int64,
    seqs,
    scale,
    threshold,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    CAPACITY: tl.constexpr,
    CHUNKS: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
    TAIL: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    SCAN: tl.constexpr,
):
    tile, head = tl.program_id(0), tl.program_id(1)
    seq, local, begin, qlen, klen = resolve_seq_and_query_len(
        Starts, Lengths, tile, seqs, M // GROUP
    )
    if local * (M // GROUP) >= qlen:
        return
    first = klen - qlen + local * (M // GROUP)
    last = tl.minimum(first + M // GROUP, klen) - 1
    last_block = last // B
    active_blocks = (last + 1) // B
    row = (tile * HEADS + head) * CAPACITY
    columns = tl.arange(0, SCAN)
    dense = (threshold == 0) | (local >= tl.cdiv(qlen * GROUP, M) - TAIL)
    keep = columns <= last_block
    if not dense:
        packed = local * M + tl.arange(0, M)
        pos = packed // GROUP
        dims = tl.arange(0, DIM)
        q = tl.load(
            Q
            + (begin + pos)[:, None] * stride_qt
            + (head * GROUP + packed % GROUP)[:, None] * stride_qh
            + dims[None, :],
            pos[:, None] < qlen,
            other=0,
        )
        running_max = tl.full((), -1.0e30, tl.float32)
        for base in range(0, active_blocks, N):
            blocks = base + tl.arange(0, N)
            km = tl.load(
                KM
                + ((seq * CAPACITY + blocks[:, None]) * HEADS + head) * DIM
                + dims[None, :],
                blocks[:, None] < active_blocks,
                other=0,
            )
            score = tl.dot(q, tl.trans(km)) * (scale * 1.4426950408889634)
            score = tl.where(
                (pos[:, None] < qlen)
                & (blocks[None, :] < active_blocks)
                & (klen - qlen + pos[:, None] >= blocks[None, :] * B),
                score,
                float("-inf"),
            )
            running_max = tl.maximum(running_max, tl.max(score))
            energy = tl.sum(tl.exp2(score - running_max), 0)
            tl.store(Energies + row + blocks, energy, blocks < active_blocks)
            tl.store(Maxima + (tile * HEADS + head) * CHUNKS + base // N, running_max)
        tl.debug_barrier()
        energy = tl.load(Energies + row + columns, columns < active_blocks, other=0)
        chunk_max = tl.load(
            Maxima + (tile * HEADS + head) * CHUNKS + columns // N,
            columns < active_blocks,
            other=-1.0e30,
        )
        energy *= tl.exp2(tl.minimum(chunk_max - running_max, 0.0))
        keep &= (
            (energy >= threshold * tl.max(energy))
            | (columns < SINK)
            | (columns >= first // B - WINDOW + 1)
        )
    keep &= columns < CAPACITY
    compact = tl.cumsum(keep.to(tl.int32)) - 1
    tl.store(Indices + row + compact, columns, keep)
    tl.store(Selected + row + columns, keep, columns < CAPACITY)
    tl.store(Counts + tile * HEADS + head, tl.sum(keep.to(tl.int32)))


@triton.jit
def _sparse_attention(
    Q,
    K,
    V,
    Table,
    Starts,
    Lengths,
    KM,
    VM,
    Indices,
    Selected,
    Counts,
    Partial,
    Maxima,
    Sums,
    stride_qt: tl.int64,
    stride_qh: tl.int64,
    stride_kp: tl.int64,
    stride_kt: tl.int64,
    stride_kh: tl.int64,
    stride_vp: tl.int64,
    stride_vt: tl.int64,
    stride_vh: tl.int64,
    stride_tb: tl.int64,
    seqs,
    scale,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    CAPACITY: tl.constexpr,
    SPLITS: tl.constexpr,
    M: tl.constexpr,
    SELECT_M: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
):
    tile, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, local, begin, qlen, klen = resolve_seq_and_query_len(
        Starts, Lengths, tile, seqs, M // GROUP
    )
    if local * (M // GROUP) >= qlen:
        return
    packed = local * M + tl.arange(0, M)
    pos = packed // GROUP
    qhead = head * GROUP + packed % GROUP
    dims = tl.arange(0, DIM)
    offsets = tl.arange(0, N)
    q = tl.load(
        Q
        + (begin + pos)[:, None] * stride_qt
        + qhead[:, None] * stride_qh
        + dims[None, :],
        pos[:, None] < qlen,
        other=0,
    )
    selection_local = local // (SELECT_M // M)
    selection_tile = begin // (SELECT_M // GROUP) + seq + selection_local
    row = (selection_tile * HEADS + head) * CAPACITY
    count = tl.load(Counts + selection_tile * HEADS + head)
    per_split = tl.cdiv(count, SPLITS)
    m = tl.full((M,), float("-inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, DIM), tl.float32)
    for i in tl.range(
        split * per_split * (B // N),
        tl.minimum((split + 1) * per_split, count) * (B // N),
        num_stages=2,
    ):
        block = tl.load(Indices + row + i // (B // N))
        tokens = block * B + (i % (B // N)) * N + offsets
        pages = tl.load(
            Table + seq * stride_tb + tokens // PAGE, tokens < klen, other=0
        ).to(tl.int64)
        k = tl.load(
            K
            + pages[None, :] * stride_kp
            + (tokens % PAGE)[None, :] * stride_kt
            + head * stride_kh
            + dims[:, None],
            tokens[None, :] < klen,
            other=0,
        )
        v = tl.load(
            V
            + pages[:, None] * stride_vp
            + (tokens % PAGE)[:, None] * stride_vt
            + head * stride_vh
            + dims[None, :],
            tokens[:, None] < klen,
            other=0,
        )
        scores = tl.dot(q, k) * (scale * 1.4426950408889634)
        scores = tl.where(
            (tokens[None, :] < klen) & (tokens[None, :] <= klen - qlen + pos[:, None]),
            scores,
            float("-inf"),
        )
        next_m = tl.maximum(m, tl.max(scores, 1))
        # Empty causal rows can occur in later KV splits.
        safe_m = tl.where(next_m == float("-inf"), 0.0, next_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(scores - safe_m[:, None])
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        denominator = denominator * alpha + tl.sum(p, 1)
        m = next_m
    # Both compute halves use the visibility boundary of their selection tile.
    full_blocks = (klen - qlen + selection_local * (SELECT_M // GROUP)) // B
    correction_chunks = tl.cdiv(full_blocks, N)
    per_split = tl.cdiv(correction_chunks, SPLITS)
    for chunk in range(
        split * per_split, tl.minimum((split + 1) * per_split, correction_chunks)
    ):
        blocks = chunk * N + offsets
        selected = tl.load(Selected + row + blocks, blocks < full_blocks, other=True)
        dropped = (blocks < full_blocks) & ~selected
        if tl.sum(dropped.to(tl.int32)) > 0:
            mean_offset = (
                (seq * CAPACITY + blocks[:, None]) * HEADS + head
            ) * DIM + dims[None, :]
            km = tl.load(KM + mean_offset, dropped[:, None], other=0)
            vm = tl.load(VM + mean_offset, dropped[:, None], other=0)
            scores = tl.dot(q, tl.trans(km)) * (scale * 1.4426950408889634)
            scores += 7.0  # log2(128): one mean vector represents 128 tokens.
            scores = tl.where(dropped[None, :], scores, float("-inf"))
            next_m = tl.maximum(m, tl.max(scores, 1))
            alpha = tl.exp2(m - next_m)
            p = tl.exp2(scores - next_m[:, None])
            acc = tl.dot(p.to(vm.dtype), vm, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            m = next_m
    partial_offset = (
        (begin + pos).to(tl.int64) * (HEADS * GROUP) + qhead
    ) * SPLITS + split
    tl.store(
        Partial + partial_offset[:, None] * DIM + dims[None, :],
        acc,
        pos[:, None] < qlen,
    )
    # The native reducer consumes natural-log maxima and unnormalized sums.
    tl.store(Maxima + partial_offset, m * 0.6931471805599453, pos < qlen)
    tl.store(Sums + partial_offset, denominator, pos < qlen)


class SparsePrefillAttention:
    """Paged, mean-corrected sparse prefill with shared native scratch storage."""

    def __init__(
        self,
        config: SparsePrefillConfig,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_model_len: int,
        max_num_tokens: int,
        max_num_seqs: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.group = num_heads // num_kv_heads
        self.max_num_tokens = max_num_tokens
        self.block_m = 64
        self.block_q = self.block_m // self.group
        self.selection_m = 128
        self.selection_q = self.selection_m // self.group
        self.max_blocks = cdiv(max_model_len, 128)
        self.max_tiles = cdiv(max_num_tokens, self.selection_q) + max_num_seqs
        index_shape = (self.max_tiles, num_kv_heads, self.max_blocks)
        mean_shape = (max_num_seqs, self.max_blocks, num_kv_heads, head_size)
        partial_shape = (max_num_tokens, num_heads, config.num_splits)
        self.workspace_shapes = (
            (mean_shape, dtype),
            (mean_shape, dtype),
            (index_shape, torch.float32),
            ((self.max_tiles, num_kv_heads, cdiv(self.max_blocks, 32)), torch.float32),
            (index_shape, torch.int32),
            (index_shape, torch.bool),
            ((self.max_tiles, num_kv_heads), torch.int32),
            ((*partial_shape, head_size), torch.float32),
            (partial_shape, torch.float32),
            (partial_shape, torch.float32),
        )

    def reserve_workspace(self) -> list[torch.Tensor]:
        return current_workspace_manager().get_simultaneous(*self.workspace_shapes)

    def forward(self, q, k, v, out, starts, lengths, table, max_seq_len, scale):
        km, vm, energy, chunk_max, indices, selected, counts, partial, maxima, sums = (
            self.reserve_workspace()
        )
        num_seqs = lengths.numel()
        tiles = q.shape[0] // self.block_q + num_seqs
        selection_tiles = q.shape[0] // self.selection_q + num_seqs
        common = dict(
            HEADS=self.num_kv_heads, DIM=self.head_size, CAPACITY=self.max_blocks
        )
        _pool_blocks[(cdiv(max_seq_len, 128), self.num_kv_heads, num_seqs)](
            k,
            v,
            table,
            lengths,
            km,
            vm,
            *k.stride()[:3],
            *v.stride()[:3],
            table.stride(0),
            PAGE=k.shape[1],
            BLOCK=128,
            **common,
            num_warps=4,
        )
        _select_blocks[(selection_tiles, self.num_kv_heads)](
            q,
            km,
            starts,
            lengths,
            energy,
            chunk_max,
            indices,
            selected,
            counts,
            *q.stride()[:2],
            num_seqs,
            scale,
            self.config.threshold,
            GROUP=self.group,
            CHUNKS=cdiv(self.max_blocks, 32),
            SINK=self.config.sink_blocks,
            WINDOW=self.config.window_blocks,
            TAIL=self.config.dense_tail_tiles,
            M=self.selection_m,
            B=128,
            N=32,
            SCAN=triton.next_power_of_2(self.max_blocks),
            **common,
            num_warps=4,
        )
        _sparse_attention[(tiles, self.num_kv_heads, self.config.num_splits)](
            q,
            k,
            v,
            table,
            starts,
            lengths,
            km,
            vm,
            indices,
            selected,
            counts,
            partial,
            maxima,
            sums,
            *q.stride()[:2],
            *k.stride()[:3],
            *v.stride()[:3],
            table.stride(0),
            num_seqs,
            scale,
            PAGE=k.shape[1],
            GROUP=self.group,
            SPLITS=self.config.num_splits,
            M=self.block_m,
            SELECT_M=self.selection_m,
            N=64,
            B=128,
            **common,
            num_warps=4,
            num_stages=2,
        )
        reduce_segments[(q.shape[0], self.num_heads)](
            out,
            partial,
            maxima,
            sums,
            lengths,
            num_seqs,
            self.num_heads,
            1.0,
            out.stride(0),
            out.stride(1),
            table.stride(0),
            TILE_SIZE=1,
            HEAD_SIZE=self.head_size,
            HEAD_SIZE_PADDED=self.head_size,
            query_start_len_ptr=starts,
            BLOCK_Q=self.block_q,
            NUM_SEGMENTS_PER_SEQ=self.config.num_splits,
            USE_FP8=False,
        )


def warmup_sparse_prefill(model: torch.nn.Module) -> None:
    """Warm native kernels against the allocated KV layout before graph capture."""
    from vllm.model_executor.layers.attention import Attention

    warmed: set[tuple[int, int, torch.Size, tuple[int, ...]]] = set()
    for layer in model.modules():
        if not isinstance(layer, Attention):
            continue
        sparse = getattr(layer.impl, "sparse_prefill", None)
        if sparse is None:
            continue
        k, v = layer.kv_cache.transpose(1, 2).split(sparse.head_size, dim=-1)
        key = (sparse.num_heads, sparse.num_kv_heads, k.shape[1:], k.stride())
        if key in warmed:
            continue
        warmed.add(key)
        tokens = min(128, sparse.max_num_tokens)
        q = torch.zeros(
            (tokens, sparse.num_heads, sparse.head_size), dtype=k.dtype, device=k.device
        )
        # These cache pages hold no requests during startup.
        pages = min(k.shape[0], cdiv(sparse.config.min_seq_len, k.shape[1]))
        layer.kv_cache[:pages].zero_()
        starts = torch.tensor([0, tokens], dtype=torch.int32, device=k.device)
        lengths = torch.tensor([pages * k.shape[1]], dtype=torch.int32, device=k.device)
        table = torch.arange(pages, dtype=torch.int32, device=k.device).view(1, -1)
        sparse.forward(
            q,
            k,
            v,
            torch.empty_like(q),
            starts,
            lengths,
            table,
            pages * k.shape[1],
            layer.impl.scale,
        )
