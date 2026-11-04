# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-private, anonymous mmap storage for the CPU KV cache."""

import mmap

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class MmapOffloadRegion:
    """Page-aligned DRAM rows, with consecutive tensor slices in each row.

    CPUOffloadingWorker pins the mapping for DMA before using it. No files,
    cross-worker sharing, or secondary storage tiers are involved.
    """

    BLOCK_SIZE_ALIGNMENT = mmap.PAGESIZE
    PIN_HOST_MEMORY = True

    def __init__(
        self,
        num_chunks: int,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
        rank: int | None = None,
    ) -> None:
        if num_chunks <= 0:
            raise ValueError("num_chunks must be positive.")
        if kv_bytes_per_chunk <= 0 or kv_bytes_per_chunk % mmap.PAGESIZE:
            raise ValueError("kv_bytes_per_chunk must be positive and page-aligned.")
        if not 0 < cpu_page_size <= kv_bytes_per_chunk:
            raise ValueError("cpu_page_size must fit within a chunk.")

        self.num_chunks = num_chunks
        self.rank = rank
        self._row_stride = kv_bytes_per_chunk
        self._worker_area_end = cpu_page_size
        self._worker_offset = 0
        self.total_size_bytes = num_chunks * kv_bytes_per_chunk
        self.is_pinned = False
        self.pinned_addresses: list[int] = []
        self._views: list[torch.Tensor] = []
        self._mmap: mmap.mmap | None = mmap.mmap(
            -1,
            self.total_size_bytes,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            # NumPy retains a buffer export, preventing unmapping while a
            # tensor view still references the allocation.
            self._base: torch.Tensor | None = torch.from_numpy(
                np.frombuffer(self._mmap, dtype=np.int8)
            )
        except Exception:
            self._mmap.close()
            self._mmap = None
            raise

    @property
    def base_tensor(self) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("Mmap offload region has been released.")
        return self._base

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:
        """Carve one tensor's bytes from every chunk without copying."""
        if tensor_page_size <= 0:
            raise ValueError("tensor_page_size must be positive.")
        new_offset = self._worker_offset + tensor_page_size
        if new_offset > self._worker_area_end:
            raise ValueError("Tensor view exceeds worker area end.")
        view = torch.as_strided(
            self.base_tensor,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(view)
        return view

    def create_next_canonical_view(self, tensor_page_size: int) -> torch.Tensor:
        raise NotImplementedError(
            "canonical_layout requires shared pages; offload_region='mmap' "
            "is process-private."
        )

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy view with one byte row per chunk."""
        return memoryview(
            self.base_tensor.view(self.num_chunks, self._row_stride).numpy()
        )

    def cleanup(self) -> None:
        """Release registration and storage after transfers have drained."""
        if self.is_pinned:
            for address in reversed(self.pinned_addresses):
                result = torch.cuda.cudart().cudaHostUnregister(address)
                if result.value != 0:
                    raise RuntimeError(
                        f"cudaHostUnregister failed at {address:#x}: {result.value}"
                    )
            self.pinned_addresses.clear()
            self.is_pinned = False
        self._views.clear()
        self._base = None
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # Outstanding tensors keep the NumPy buffer export alive;
                # the mapping closes when their last reference is released.
                logger.debug(
                    "Deferring mmap release until exported views are released."
                )
            self._mmap = None
