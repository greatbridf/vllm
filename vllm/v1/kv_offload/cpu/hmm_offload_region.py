# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host offload region whose placement is left to the kernel (Linux HMM)."""

import mmap
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class HMMOffloadRegion:
    """Host KV offload region that the kernel places and migrates.

    Counterpart to `SharedOffloadRegion` for deployments that offload host KV
    pages through Linux HMM (a driver migrating pages on device fault) instead
    of through vLLM's own secondary tiers. The differences are deliberate:

    * Plain host allocation: no file-backed mmap in /dev/shm, so the pages are
      ordinary anonymous memory the kernel can migrate.
    * Never pinned: cudaHostRegister would make every page resident and
      unmovable, which is exactly what HMM migration needs to avoid.
    * Never pre-faulted: the pages are left untouched on allocation, both to
      keep the region from being resident up front and to let the kernel
      observe real access patterns.
    * Process-private: every worker owns its own region, so no cross-worker
      sharing, barrier, or /dev/shm accounting is involved.

    Layout matches the per-worker slice of `SharedOffloadRegion`: one
    `kv_bytes_per_chunk`-byte row per chunk, with this worker's tensors carved
    consecutively from offset 0 of each row.

    The region's address and size are exposed as `base_tensor.data_ptr()` and
    `total_size_bytes`, which is what an out-of-tree driver or helper needs to
    register the range with the kernel.
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    # CPUOffloadingWorker skips cudaHostRegister when this is False.
    PIN_HOST_MEMORY: bool = False

    def __init__(
        self,
        num_chunks: int,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
        rank: int | None = None,
    ) -> None:
        assert num_chunks > 0
        assert kv_bytes_per_chunk > 0
        assert 0 < cpu_page_size <= kv_bytes_per_chunk
        self.page_size = mmap.PAGESIZE
        # Page-aligned rows keep HMM migration granularity page-sized.
        assert kv_bytes_per_chunk % self.page_size == 0

        self.num_chunks = num_chunks
        self.rank = rank
        self._row_stride = kv_bytes_per_chunk
        self.total_size_bytes = self.num_chunks * self._row_stride

        t0 = time.monotonic()
        self._base: torch.Tensor | None = torch.empty(
            self.total_size_bytes, dtype=torch.int8, device="cpu"
        )
        logger.info(
            "Allocated HMM offload region: %d chunks x %d bytes (%.2f GB) "
            "in %.3f s",
            self.num_chunks,
            self._row_stride,
            self.total_size_bytes / 1e9,
            time.monotonic() - t0,
        )

        self._views: list[torch.Tensor] = []
        self._worker_offset = 0
        self._worker_area_end = cpu_page_size
        # Never registered with cudaHostRegister, but kept for interface parity
        # with SharedOffloadRegion.
        self.is_pinned: bool = False
        self.pinned_addresses: list[int] = []

    @property
    def base_tensor(self) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("HMM offload region has been released.")
        return self._base

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. All views share the row and
        are carved consecutively from its start:

            | tensor0_data | tensor1_data | ... | tensor{L-1}_data |

        Returns an int8 tensor of shape (num_chunks, tensor_page_size) with
        stride (row_stride, 1). Using int8 keeps stride == bytes, so
        swap_blocks address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per chunk for this tensor.

        """
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
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
            "canonical_layout needs one page shared by all ranks, but an HMM "
            "region is process-private: ranks cannot write disjoint bytes "
            "into a single canonical page. Use offload_region='shared'."
        )

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_chunks, row_stride_bytes).
        """
        base = self.base_tensor
        kv_tensor = base.view(self.num_chunks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the host buffer"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        # Release views before _base: each view holds a reference to the base
        # storage, so the allocation is only freed once both are dropped.
        self._views.clear()
        self._base = None
