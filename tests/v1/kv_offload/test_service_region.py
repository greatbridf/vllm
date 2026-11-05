# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometry, pinning and teardown of the service-owned worker mapping."""

import mmap
import os
from unittest.mock import MagicMock, call

import pytest

import vllm.v1.kv_offload.service.region as region_module
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.service.region import ServiceOffloadRegion

PAGE = mmap.PAGESIZE


def _mapping(num_chunks: int, row_bytes: int) -> int:
    """Sizes a memfd the way the service seals one namespace's mapping."""
    fd = os.memfd_create("service-region-test")
    os.ftruncate(fd, num_chunks * row_bytes)
    return fd


def _region(
    num_chunks: int = 4,
    row_bytes: int = 2 * PAGE,
    worker_bytes: int = PAGE,
    world_size: int = 2,
    rank: int = 0,
    size: int | None = None,
) -> ServiceOffloadRegion:
    fd = _mapping(num_chunks, row_bytes if size is None else size)
    return ServiceOffloadRegion(
        MagicMock(), fd, num_chunks, row_bytes, worker_bytes, world_size, rank
    )


@pytest.mark.parametrize(
    "geometry",
    [
        # rank outside world_size
        {"rank": 2, "world_size": 2},
        # row_bytes not a whole number of pages
        {"row_bytes": 2 * PAGE + 1},
        # row_bytes cannot hold one slice per rank
        {"row_bytes": PAGE, "worker_bytes": PAGE, "world_size": 2},
        # no slice to map
        {"worker_bytes": 0},
    ],
)
def test_region_rejects_an_impossible_geometry(geometry):
    with pytest.raises(ValueError, match="Invalid service shared-memory geometry"):
        _region(**geometry)


def test_region_rejects_a_mapping_size_mismatch():
    """The mapping must be exactly one row per chunk, as the service sealed it."""
    with pytest.raises(ValueError, match="Invalid service shared-memory geometry"):
        _region(num_chunks=4, size=3)


def test_region_maps_this_ranks_slice_of_every_chunk():
    """Rank 1's views start one slice into each row of the shared mapping."""
    region = _region(num_chunks=3, rank=1)
    try:
        assert region.total_size_bytes == 3 * 2 * PAGE
        assert region.cpu_page_size == PAGE
        view = region.create_next_worker_view(PAGE)
        assert view.data_ptr() == region.base_tensor.data_ptr() + PAGE
        assert list(region.iter_worker_slot_ranges()) == [
            ((2 * row + 1) * PAGE, PAGE) for row in range(3)
        ]
    finally:
        region.cleanup()


def test_region_merges_contiguous_slices_into_one_range():
    """Rows holding only this rank's bytes register as a single range."""
    region = _region(num_chunks=3, row_bytes=PAGE, worker_bytes=PAGE, world_size=1)
    try:
        assert list(region.iter_worker_slot_ranges()) == [(0, 3 * PAGE)]
    finally:
        region.cleanup()


def test_pin_registers_only_this_ranks_slices(monkeypatch):
    """Each rank pins its own slot per chunk instead of the whole region."""
    register = MagicMock(return_value=True)
    monkeypatch.setattr(gpu_worker, "host_register", register)
    monkeypatch.setattr(gpu_worker, "MIN_HOST_REGISTER_SAVED_BYTES", PAGE)
    region = _region(num_chunks=2, rank=1)
    try:
        gpu_worker.pin_mmap_region(region, worker_slot_only=True)
        base = region.base_tensor.data_ptr() + PAGE
        assert register.call_args_list == [
            call(base + row * 2 * PAGE, PAGE) for row in range(2)
        ]
        assert region.pinned_addresses == [c.args[0] for c in register.call_args_list]
        assert region.is_pinned
    finally:
        region.cleanup()


def test_cleanup_unregisters_through_the_shared_helper(monkeypatch):
    """Teardown warns through host_unregister instead of raising a runtime error."""
    unregister = MagicMock()
    monkeypatch.setattr(region_module, "host_unregister", unregister)
    region = _region()
    region.pinned_addresses = [0x1000, 0x2000]
    region.is_pinned = True
    region.cleanup()

    assert unregister.call_args_list == [call(0x2000), call(0x1000)]
    assert region.pinned_addresses == []
    assert not region.is_pinned


def test_cleanup_is_idempotent():
    """Worker shutdown may reach a region a failed setup already cleaned up."""
    client = MagicMock()
    fd = _mapping(2, 2 * PAGE)
    region = ServiceOffloadRegion(client, fd, 2, 2 * PAGE, PAGE, 2, 0)
    region.cleanup()
    region.cleanup()

    assert region.pinned_addresses == []
    assert client.close.call_count == 2
