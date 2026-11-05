# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker mapping of CPU cache memory owned by the Rust service."""

import mmap
import os
from collections.abc import Iterator

import numpy as np
import torch

from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.cpu.host_register import host_unregister
from vllm.v1.kv_offload.cpu.mmap_offload_region import MmapOffloadRegion
from vllm.v1.kv_offload.service.client import CacheServiceClient


class ServiceOffloadRegion(MmapOffloadRegion):
    """A rank-private slice of each service-owned shared chunk."""

    def __init__(
        self,
        client: CacheServiceClient,
        fd: int,
        num_chunks: int,
        row_bytes: int,
        worker_bytes: int,
        world_size: int,
        rank: int,
    ):
        self.client = client
        self.is_pinned = False
        self.pinned_addresses: list[int] = []
        self._views: list[torch.Tensor] = []
        self._base = None
        self._mmap = None
        try:
            if (
                num_chunks <= 0
                or row_bytes % mmap.PAGESIZE
                or row_bytes < worker_bytes * world_size
                or worker_bytes <= 0
                or not 0 <= rank < world_size
                or os.fstat(fd).st_size != num_chunks * row_bytes
            ):
                raise ValueError("Invalid service shared-memory geometry")
            self.num_chunks = num_chunks
            self.rank = rank
            self._row_stride = row_bytes
            # This rank's slice of every row. `pin_mmap_region` reads it to
            # register those slices instead of the whole multi-rank region.
            self.cpu_page_size = worker_bytes
            self._worker_offset = rank * worker_bytes
            self._worker_area_end = (rank + 1) * worker_bytes
            self.total_size_bytes = num_chunks * row_bytes
            self._mmap = mmap.mmap(fd, self.total_size_bytes, flags=mmap.MAP_SHARED)
            self._base = torch.from_numpy(np.frombuffer(self._mmap, dtype=np.int8))
        except Exception:
            self.cleanup()
            raise
        finally:
            os.close(fd)

    def iter_worker_slot_ranges(self) -> Iterator[tuple[int, int]]:
        """Yields this rank's page-aligned slice of every chunk.

        Consecutive slices are merged into one range, so a layout whose rows
        hold nothing but this rank's bytes registers once, not once per chunk.
        """
        page = mmap.PAGESIZE
        rank = self.rank
        assert rank is not None
        pending: tuple[int, int] | None = None
        for chunk in range(self.num_chunks):
            row = chunk * self._row_stride
            start = row + rank * self.cpu_page_size
            start -= start % page
            end = min(
                round_up(row + self._worker_area_end, page), self.total_size_bytes
            )
            if pending is not None and start <= pending[1]:
                pending = (pending[0], max(pending[1], end))
                continue
            if pending is not None:
                yield pending[0], pending[1] - pending[0]
            pending = (start, end)
        if pending is not None:
            yield pending[0], pending[1] - pending[0]

    def cleanup(self) -> None:
        """Releases the mapping and this rank's registration; safe to repeat."""
        if self.is_pinned:
            # Pinning is a bandwidth optimization the region has already given
            # up if unregistering fails, so a failure warns instead of raising.
            for address in reversed(self.pinned_addresses):
                host_unregister(address)
            self.pinned_addresses.clear()
            self.is_pinned = False
        super().cleanup()
        # Worker shutdown drains DMA before disconnecting. The service
        # invalidates this namespace before it can reuse abandoned slots.
        self.client.close()
