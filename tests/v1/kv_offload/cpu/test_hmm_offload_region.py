# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HMMOffloadRegion."""

import mmap
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region
from vllm.v1.kv_offload.cpu.hmm_offload_region import HMMOffloadRegion

PAGE_SIZE = mmap.PAGESIZE


def _make_region(
    num_chunks: int = 4,
    kv_bytes_per_chunk: int = 2 * PAGE_SIZE,
    cpu_page_size: int = PAGE_SIZE,
    rank: int | None = 0,
) -> HMMOffloadRegion:
    return HMMOffloadRegion(
        num_chunks=num_chunks,
        kv_bytes_per_chunk=kv_bytes_per_chunk,
        cpu_page_size=cpu_page_size,
        rank=rank,
    )


def test_region_is_unpinned_host_memory():
    region = _make_region()
    try:
        base = region.base_tensor
        assert base.device.type == "cpu"
        assert base.numel() == region.total_size_bytes == 4 * 2 * PAGE_SIZE
        # The whole point: plain host memory the kernel may migrate.
        assert not base.is_pinned()
        assert region.is_pinned is False
        assert region.pinned_addresses == []
        assert HMMOffloadRegion.PIN_HOST_MEMORY is False
    finally:
        region.cleanup()


def test_worker_views_are_strided_over_the_shared_allocation():
    region = _make_region(
        num_chunks=3, kv_bytes_per_chunk=4 * PAGE_SIZE, cpu_page_size=3 * PAGE_SIZE
    )
    try:
        first = region.create_next_worker_view(PAGE_SIZE)
        second = region.create_next_worker_view(2 * PAGE_SIZE)

        assert first.shape == (3, PAGE_SIZE)
        assert second.shape == (3, 2 * PAGE_SIZE)
        assert first.stride() == (4 * PAGE_SIZE, 1)
        assert second.stride() == (4 * PAGE_SIZE, 1)
        # Views alias the single allocation, at the offsets they were carved at.
        assert first.data_ptr() == region.base_tensor.data_ptr()
        assert second.data_ptr() == region.base_tensor.data_ptr() + PAGE_SIZE

        first[1, 0] = 42
        assert region.base_tensor[4 * PAGE_SIZE] == 42
    finally:
        region.cleanup()


def test_worker_views_overflowing_the_worker_area_are_rejected():
    region = _make_region(cpu_page_size=PAGE_SIZE)
    try:
        region.create_next_worker_view(PAGE_SIZE)
        with pytest.raises(AssertionError, match="exceeds worker area end"):
            region.create_next_worker_view(1)
    finally:
        region.cleanup()


def test_kv_memoryview_is_zero_copy():
    region = _make_region(num_chunks=2, kv_bytes_per_chunk=2 * PAGE_SIZE)
    try:
        view = region.create_kv_memoryview()
        assert view.shape == (2, 2 * PAGE_SIZE)
        region.base_tensor[2 * 2 * PAGE_SIZE - 1] = 0
        assert view[1, 2 * PAGE_SIZE - 1] == 0
        region.base_tensor[2 * 2 * PAGE_SIZE - 1] = 7
        assert view[1, 2 * PAGE_SIZE - 1] == 7
        view.release()
    finally:
        region.cleanup()


def test_canonical_view_is_rejected():
    # Canonical pages are shared by all ranks; an HMM region is private.
    region = _make_region()
    try:
        with pytest.raises(NotImplementedError, match="canonical_layout"):
            region.create_next_canonical_view(PAGE_SIZE)
    finally:
        region.cleanup()


def test_cleanup_releases_the_region():
    region = _make_region()
    region.cleanup()
    with pytest.raises(RuntimeError, match="has been released"):
        _ = region.base_tensor


def test_pin_mmap_region_skips_host_registration(monkeypatch):
    region = _make_region()
    try:
        cudart = MagicMock()
        monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
        pin_mmap_region(region)
        cudart.cudaHostRegister.assert_not_called()
        assert region.is_pinned is False
    finally:
        region.cleanup()
