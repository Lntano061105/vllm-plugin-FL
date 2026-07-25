# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from the vllm-ascend project (PR #12096, slot mapping Triton kernel
# optimization for Ascend NPU), ported to the vLLM 0.13.0 block table layout.

"""Ascend-optimized slot mapping computation for the vLLM v1 block table.

The upstream vLLM 0.13.0 ``vllm.v1.worker.block_table.BlockTable`` computes
slot mapping with NumPy on the CPU and then copies the result to the device
(``commit_slot_mapping``). This module backports the Ascend-local Triton
kernel from vllm-ascend PR #12096 so the computation runs directly on the
NPU and the H2D commit of slot mapping is skipped.

Key differences from the original PR (adaptation decisions):

1. vLLM 0.13.0 ``BlockTable.compute_slot_mapping`` takes NumPy
   ``(req_indices, positions)`` instead of on-device metadata. The caller
   therefore prepares the runner's existing ``query_start_loc.gpu`` and
   ``seq_lens.gpu`` buffers before launching this kernel. A request's scheduled
   positions are consecutive, so the kernel derives its first position as
   ``seq_len - scheduled_tokens``. This avoids a dedicated positions H2D copy,
   which is especially important for M-RoPE models.
2. vLLM 0.13.0 allocates ``slot_mapping`` as ``torch.int64`` (the PR targets
   an int32 buffer). The kernel therefore casts ``slot_ids`` to the pointer's
   element type at store time instead of assuming int32.
3. In vLLM 0.13.0 the block table already stores kernel-block ids (physical
   blocks are expanded by ``map_to_kernel_blocks`` when hybrid blocks are
   used), so ``KV_CACHE_BLOCK_SIZE`` is passed as
   ``block_size * blocks_per_kv_block`` and the runtime ``block_size``
   argument is the kernel block size, keeping the kernel's index math
   identical to the upstream NumPy path.
4. PCP/DCP (context parallelism) is not covered by the PR kernel launch path;
   when ``pcp_world_size * dcp_world_size > 1`` we fall back to the upstream
   NumPy implementation.
5. Only active tokens are written. vLLM 0.13.0 already pads slot mapping to the
   selected graph size while building attention metadata, so clearing the whole
   ``max_num_batched_tokens`` tail here would be redundant decode-step work.
"""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import cdiv

# TILE size of the token loop inside each request program. Same value as the
# original PR (and upstream vLLM).
_BLOCK_SIZE = 1024


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _compute_slot_mapping_kernel(
    query_start_loc_ptr,  # [num_reqs + 1], int32
    seq_lens_ptr,  # [num_reqs], int32
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32
    block_table_stride,
    block_size,
    slot_mapping_ptr,  # [max_num_tokens]
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    start_idx = tl.load(query_start_loc_ptr + req_idx)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1)
    num_req_tokens = end_idx - start_idx
    request_start_pos = tl.load(seq_lens_ptr + req_idx) - num_req_tokens
    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)

    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        positions = request_start_pos + offsets - start_idx

        physical_block_indices = positions // KV_CACHE_BLOCK_SIZE
        local_block_offsets = positions - (
            KV_CACHE_BLOCK_SIZE * physical_block_indices
        )

        local_block_indices = local_block_offsets // block_size
        block_indices = (
            physical_block_indices * BLOCKS_PER_KV_BLOCK + local_block_indices
        )

        # Non-contiguous block_table loads degrade to scalar on Ascend. Positions
        # are grouped by request, so a token tile only spans a small block window.
        valid_block_indices = tl.where(mask, block_indices, 2147483647)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)

        slot_offsets = local_block_offsets - block_size * local_block_indices
        slot_ids = block_numbers * block_size + slot_offsets
        # NOTE(FL): vLLM 0.13.0 allocates slot_mapping as int64 (the PR targets
        # an int32 buffer); cast to the pointer element type so both work.
        tl.store(slot_mapping_ptr + offsets,
                 slot_ids.to(slot_mapping_ptr.dtype.element_ty),
                 mask=mask)


def _launch_kernel(block_table,
                   num_reqs: int,
                   query_start_loc: torch.Tensor,
                   seq_lens: torch.Tensor) -> None:
    """Launch the Triton kernel for a single vLLM 0.13.0 ``BlockTable``."""
    # NOTE(FL): in vLLM 0.13.0 ``block_table.block_size`` is already the
    # kernel block size and the block table stores kernel-block ids, so the
    # physical block size is reconstructed as block_size * blocks_per_kv_block.
    blocks_per_kv_block = block_table.blocks_per_kv_block
    physical_block_size = block_table.block_size * blocks_per_kv_block
    _compute_slot_mapping_kernel[(num_reqs, )](
        query_start_loc,
        seq_lens,
        block_table.block_table.gpu,
        block_table.block_table.gpu.stride(0),
        block_table.block_size,
        block_table.slot_mapping.gpu,
        KV_CACHE_BLOCK_SIZE=physical_block_size,
        BLOCKS_PER_KV_BLOCK=blocks_per_kv_block,
        BLOCK_SIZE=_BLOCK_SIZE,
        BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2(
            cdiv(_BLOCK_SIZE, block_table.block_size) + 1),
    )


def compute_slot_mapping_npu(
    block_tables,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    """Compute slot mapping on the NPU for selected ``BlockTable`` objects.

    Returns True when the slot mapping was computed on the device (the caller
    must then skip ``commit_slot_mapping``), or False when the caller should
    use the upstream NumPy path (non-NPU platform, Triton unavailable, or
    context parallelism enabled).
    """
    if current_platform.device_type != "npu" or not HAS_TRITON:
        return False
    if not block_tables:
        return True
    first = block_tables[0]
    if first.pcp_world_size * first.dcp_world_size > 1:
        # The PR kernel launch path only covers CP==1; keep the upstream
        # NumPy implementation for PCP/DCP.
        return False

    for block_table in block_tables:
        _launch_kernel(block_table, num_reqs, query_start_loc, seq_lens)
    return True
