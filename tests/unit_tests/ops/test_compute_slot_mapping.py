# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for the Ascend slot mapping Triton kernel launcher.

Adapted from vllm-ascend PR #12096 (tests/ut/worker/a2/test_block_table.py).
The launch-grid assertions run on CPU by mocking the Triton kernel; numerical
correctness of the kernel itself requires an NPU and is covered by e2e tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


def test_seq_lens_reconstruct_scheduled_positions():
    """seq_len - query_len is the first absolute position of each request."""
    query_start_loc = torch.tensor([0, 3, 4, 8], dtype=torch.int32)
    seq_lens = torch.tensor([13, 21, 104], dtype=torch.int32)

    reconstructed = []
    for req_idx in range(seq_lens.numel()):
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        first_pos = int(seq_lens[req_idx]) - (end - start)
        reconstructed.extend(range(first_pos, int(seq_lens[req_idx])))

    assert reconstructed == [10, 11, 12, 20, 100, 101, 102, 103]


def _make_fake_block_table(max_num_batched_tokens: int,
                           block_size: int = 128,
                           blocks_per_kv_block: int = 1):
    return SimpleNamespace(
        block_size=block_size,
        blocks_per_kv_block=blocks_per_kv_block,
        max_num_batched_tokens=max_num_batched_tokens,
        cp_kv_cache_interleave_size=1,
        block_table=SimpleNamespace(
            gpu=torch.zeros((4, 8), dtype=torch.int32)),
        slot_mapping=SimpleNamespace(
            gpu=torch.zeros((max_num_batched_tokens, ), dtype=torch.int64)),
    )


def _run_launch_case(num_reqs: int, max_num_batched_tokens: int,
                     block_size: int):
    from vllm_fl.dispatch.backends.vendor.ascend.impl import (
        compute_slot_mapping as csm)

    block_table = _make_fake_block_table(max_num_batched_tokens,
                                         block_size=block_size)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    seq_lens = torch.zeros(num_reqs, dtype=torch.int32)

    kernel_mock = MagicMock()
    launcher_mock = MagicMock()
    kernel_mock.__getitem__.return_value = launcher_mock
    with patch.object(csm, "_compute_slot_mapping_kernel", kernel_mock):
        csm._launch_kernel(block_table, num_reqs, query_start_loc, seq_lens)

    # The model runner pads only to the selected graph size later. The slot
    # kernel must not clear the full max_num_batched_tokens tail every step.
    kernel_mock.__getitem__.assert_called_once_with((num_reqs, ))
    launcher_mock.assert_called_once()
    args, kwargs = launcher_mock.call_args
    assert args[0] is query_start_loc
    assert args[1] is seq_lens
    assert kwargs["BLOCK_SIZE"] == 1024
    assert kwargs["BLOCK_TABLE_WINDOW_SIZE"] == 16
    assert kwargs["KV_CACHE_BLOCK_SIZE"] == block_size
    assert kwargs["BLOCKS_PER_KV_BLOCK"] == 1


@pytest.mark.parametrize("block_size", [128])
def test_launches_one_program_per_request_without_full_tail_padding(block_size):
    """A small decode batch must not clear the entire graph-capacity tail."""
    _run_launch_case(num_reqs=2,
                     max_num_batched_tokens=4096,
                     block_size=block_size)


@pytest.mark.parametrize("block_size", [128])
def test_full_batch_needs_no_padding_programs(block_size):
    """Grid size remains request-based when graph capacity is small."""
    _run_launch_case(num_reqs=2,
                     max_num_batched_tokens=512,
                     block_size=block_size)


def test_hybrid_blocks_pass_physical_block_size():
    """With hybrid blocks the kernel receives the physical (kv-cache) block size."""
    from vllm_fl.dispatch.backends.vendor.ascend.impl import (
        compute_slot_mapping as csm)

    # kernel block size 64, 2 kernel blocks per 128-token physical block
    block_table = _make_fake_block_table(512,
                                         block_size=64,
                                         blocks_per_kv_block=2)
    kernel_mock = MagicMock()
    kernel_mock.__getitem__.return_value = MagicMock()
    with patch.object(csm, "_compute_slot_mapping_kernel", kernel_mock):
        csm._launch_kernel(block_table, 1,
                           torch.zeros(2, dtype=torch.int32),
                           torch.zeros(1, dtype=torch.int32))
    _, kwargs = kernel_mock.__getitem__.return_value.call_args
    assert kwargs["KV_CACHE_BLOCK_SIZE"] == 128
    assert kwargs["BLOCKS_PER_KV_BLOCK"] == 2


def test_compute_slot_mapping_reuses_runner_device_buffers():
    """The launcher should receive the caller's persistent device tensors."""
    from vllm_fl.dispatch.backends.vendor.ascend.impl import (
        compute_slot_mapping as csm)

    block_table = _make_fake_block_table(512)
    block_table.pcp_world_size = 1
    block_table.dcp_world_size = 1
    multi_group = SimpleNamespace(block_tables=[block_table])
    query_start_loc = torch.zeros(3, dtype=torch.int32)
    seq_lens = torch.zeros(2, dtype=torch.int32)

    with (
        patch.object(csm, "current_platform",
                     SimpleNamespace(device_type="npu")),
        patch.object(csm, "HAS_TRITON", True),
        patch.object(csm, "_launch_kernel") as launch_kernel,
    ):
        used_npu = csm.compute_slot_mapping_npu(
            multi_group.block_tables,
            num_reqs=2,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
        )

    assert used_npu
    launch_kernel.assert_called_once_with(
        block_table, 2, query_start_loc, seq_lens)
