# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker mapping of CPU cache memory owned by the Rust service."""

import mmap
import os

import numpy as np
import torch

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
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

    def cleanup(self) -> None:
        try:
            if self.is_pinned:
                runtime = CudaRTLibrary()
                while self.pinned_addresses:
                    address = self.pinned_addresses[-1]
                    code = runtime.cudaHostUnregister(address)
                    if code != 0:
                        raise RuntimeError(f"Host unregistration failed: {code}")
                    self.pinned_addresses.pop()
                self.is_pinned = False
            super().cleanup()
        finally:
            # Worker shutdown drains DMA before disconnecting. The service
            # invalidates this namespace before it can reuse abandoned slots.
            self.client.close()
