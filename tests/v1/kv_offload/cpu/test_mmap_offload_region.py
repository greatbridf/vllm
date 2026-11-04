# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage-contract tests for anonymous mmap KV offloading."""

import mmap

import pytest
import torch

from vllm.v1.kv_offload.cpu.mmap_offload_region import MmapOffloadRegion

PAGE = mmap.PAGESIZE


def test_tensor_views_preserve_chunk_stride_and_do_not_overlap():
    region = MmapOffloadRegion(3, 2 * PAGE, PAGE)
    try:
        first = region.create_next_worker_view(17)
        second = region.create_next_worker_view(PAGE - 17)
        assert region.base_tensor.device.type == "cpu"
        assert region.base_tensor.data_ptr() % PAGE == 0
        assert region.base_tensor.numel() == region.total_size_bytes == 6 * PAGE
        assert first.stride() == second.stride() == (2 * PAGE, 1)
        assert second.data_ptr() == first.data_ptr() + 17
        first.fill_(42)
        second.fill_(7)
        for chunk in range(3):
            start = chunk * 2 * PAGE
            assert torch.all(region.base_tensor[start : start + 17] == 42)
            assert torch.all(region.base_tensor[start + 17 : start + PAGE] == 7)
            assert torch.all(region.base_tensor[start + PAGE : start + 2 * PAGE] == 0)
    finally:
        region.cleanup()


def test_regions_do_not_share_storage():
    first = MmapOffloadRegion(1, PAGE, PAGE, rank=0)
    second = MmapOffloadRegion(1, PAGE, PAGE, rank=1)
    try:
        first.base_tensor.fill_(42)
        assert torch.all(second.base_tensor == 0)
        assert first.base_tensor.data_ptr() != second.base_tensor.data_ptr()
    finally:
        first.cleanup()
        second.cleanup()


def test_exported_view_remains_valid_after_cleanup():
    region = MmapOffloadRegion(2, PAGE, PAGE)
    view = region.create_kv_memoryview()
    region.base_tensor[PAGE + 3] = 19
    region.cleanup()
    assert view.shape == (2, PAGE)
    assert view[1, 3] == 19
    view[0, 1] = 21
    assert view[0, 1] == 21
    view.release()
    region.cleanup()
    with pytest.raises(RuntimeError, match="has been released"):
        _ = region.base_tensor


def test_cleanup_closes_mapping_without_exported_views():
    region = MmapOffloadRegion(1, PAGE, PAGE)
    mapping = region._mmap
    region.cleanup()
    assert mapping is not None and mapping.closed
    assert region._mmap is None


def test_view_bounds_do_not_consume_space_on_failure():
    region = MmapOffloadRegion(1, PAGE, PAGE)
    try:
        with pytest.raises(ValueError, match="exceeds worker area"):
            region.create_next_worker_view(PAGE + 1)
        with pytest.raises(ValueError, match="positive"):
            region.create_next_worker_view(0)
        assert region.create_next_worker_view(PAGE).shape == (1, PAGE)
    finally:
        region.cleanup()


@pytest.mark.parametrize(
    "chunks,row,worker",
    [(0, PAGE, PAGE), (1, PAGE - 1, 1), (1, PAGE, 0), (1, PAGE, PAGE + 1)],
)
def test_invalid_dimensions_are_rejected(chunks, row, worker):
    with pytest.raises(ValueError):
        MmapOffloadRegion(chunks, row, worker)


def test_private_region_rejects_cross_rank_canonical_pages():
    region = MmapOffloadRegion(1, PAGE, PAGE)
    try:
        with pytest.raises(NotImplementedError, match="requires shared pages"):
            region.create_next_canonical_view(PAGE)
    finally:
        region.cleanup()
