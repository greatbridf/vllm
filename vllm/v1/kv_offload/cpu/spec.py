# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from typing_extensions import override

from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingInfo, CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker, OffloadRegion
from vllm.v1.kv_offload.cpu.hmm_offload_region import HMMOffloadRegion
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.mmap_offload_region import MmapOffloadRegion
from vllm.v1.kv_offload.cpu.shared_offload_region import (
    REGION_SEGMENTS,
    SharedOffloadRegion,
)

# Host region selected by kv_connector_extra_config["offload_region"].
OFFLOAD_REGION_SHARED = "shared"
OFFLOAD_REGION_HMM = "hmm"
OFFLOAD_REGION_MMAP = "mmap"
OFFLOAD_REGION_TYPES = (
    OFFLOAD_REGION_SHARED,
    OFFLOAD_REGION_HMM,
    OFFLOAD_REGION_MMAP,
)


def _all_workers_barrier() -> None:
    """Block until every worker rank has reached this point (gloo cpu group).

    A superset of the node-local mmap openers suffices: once the barrier
    releases, every worker sharing the region file has mapped it."""
    from vllm.distributed.parallel_state import (
        get_inner_dp_world_group,
        get_world_group,
    )

    try:
        group = get_inner_dp_world_group()
    except AssertionError:
        group = get_world_group()
    group.barrier()


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU chunks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        self.region_type: str = self.extra_config.get(
            "offload_region", OFFLOAD_REGION_SHARED
        )
        if self.region_type not in OFFLOAD_REGION_TYPES:
            raise ValueError(
                f"Unsupported offload_region '{self.region_type}'; "
                f"expected one of {list(OFFLOAD_REGION_TYPES)}"
            )
        if self.region_type == OFFLOAD_REGION_MMAP and config.canonical_layout:
            raise ValueError(
                "canonical_layout is not supported with offload_region='mmap': "
                "canonical pages require a shared region."
            )

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_chunks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        self.replicated_layout = config.replicated_layout and self._uses_shared_region()
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            num_copies = 1 if self.replicated_layout else world_size
            kv_bytes_per_block = config.worker_kv_bytes_per_block * num_copies
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // num_copies

            # calculate num_chunks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            if self.region_type == OFFLOAD_REGION_MMAP:
                # Every private mapping has its own page-alignment padding.
                # Budget for all workers, not just one aligned aggregate row.
                aligned_kv_bytes_per_chunk = round_up(
                    self.cpu_page_size_per_worker, self.BLOCK_SIZE_ALIGNMENT
                )
                self.num_chunks = int(cpu_bytes_to_use) // (
                    aligned_kv_bytes_per_chunk * world_size
                )
            else:
                self.num_chunks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded chunk is of the form,
            # |--- W0-C0---|---- W1-C0---| ... |---- Wn-C0---| *** maybe-pad *** |
            # or |--- C0 (single copy) ---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        self.cpu_info = CPUOffloadingInfo.from_config(
            config, self.num_chunks, self.kv_bytes_per_chunk
        )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

    @classmethod
    @override
    def config_info_keys(cls, extra_config: dict[str, Any]) -> tuple[str, ...]:
        """Return the label names CPUOffloadingManager.config_info() fills."""
        return CPUOffloadingInfo.config_info_keys()

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a chunk must be offered for
            # storage before it is eligible for CPU offloading.  Values < 2
            # disable filtering (a threshold of 1 equals no filter; 0 is the
            # default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_chunks=self.num_chunks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                info=self.cpu_info,
            )
        return self._manager

    def _uses_shared_region(self) -> bool:
        """Whether workers can deduplicate replicated KV in shared storage."""
        return self.region_type == OFFLOAD_REGION_SHARED and (
            (current_platform.is_cuda_alike() and not current_platform.is_rocm())
            or current_platform.is_xpu()
        )

    def _create_offload_region(self, rank: int) -> OffloadRegion:
        """Allocate the host region backing this worker's offload side.

        `shared` uses /dev/shm; `mmap` uses private anonymous DRAM; `hmm` uses
        an unpinned allocation left for kernel-managed migration.
        """
        if self.region_type == OFFLOAD_REGION_MMAP:
            return MmapOffloadRegion(
                num_chunks=self.num_chunks,
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                rank=rank,
            )
        if self.region_type == OFFLOAD_REGION_HMM:
            if self.config.canonical_layout:
                raise ValueError(
                    "canonical_layout is not supported with "
                    "offload_region='hmm': canonical pages are shared by all "
                    "ranks, but an HMM region is process-private."
                )
            return HMMOffloadRegion(
                num_chunks=self.num_chunks,
                kv_bytes_per_chunk=round_up(
                    self.cpu_page_size_per_worker,
                    HMMOffloadRegion.BLOCK_SIZE_ALIGNMENT,
                ),
                cpu_page_size=self.cpu_page_size_per_worker,
                rank=rank,
            )
        return SharedOffloadRegion(
            engine_id=self.config.engine_id,
            num_chunks=self.num_chunks,
            rank=rank,
            kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
            barrier=_all_workers_barrier,
            num_segments=REGION_SEGMENTS,
            populate_shard=(
                (self.config.parallel.rank, self.config.parallel.world_size)
                if self.replicated_layout
                else None
            ),
        )

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        region: OffloadRegion | None = None
        # num_chunks == 0 would size the region to zero bytes, which cannot be
        # mmap'd; fall back to the tensor path (empty tensors) as before.
        if self.num_chunks > 0 and (
            self.region_type in (OFFLOAD_REGION_HMM, OFFLOAD_REGION_MMAP)
            or self._uses_shared_region()
        ):
            rank = 0 if self.replicated_layout else self.config.parallel.rank
            region = self._create_offload_region(rank)
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=self.num_chunks,
                mmap_region=region,
            )
        except Exception:
            if region is not None:
                region.cleanup()
            raise

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
