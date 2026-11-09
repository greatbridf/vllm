# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rust CPU cache service backing the native OffloadingConnector."""

import hashlib
import json
import mmap
from dataclasses import asdict, replace
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingInfo
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.service.client import CacheServiceClient
from vllm.v1.kv_offload.service.manager import ServiceOffloadingManager
from vllm.v1.kv_offload.service.region import ServiceOffloadRegion

logger = init_logger(__name__)

# `kv_connector_extra_config` keys this backend honours, either itself (the
# `service_*` settings and the geometry) or through the CPU settings it inherits
# and the connector that drives it. Anything else is rejected, so a typo or a
# setting meant for another backend cannot be silently ignored.
_SUPPORTED_KEYS = frozenset(
    {
        "block_size",
        "blocks_per_chunk",
        "canonical_layout",
        "cpu_bytes_to_use",
        "eviction_policy",
        "offload_prompt_only",
        "offload_region",
        "pin_host_memory",
        "prefetch_on_load",
        "prefetch_on_lookup",
        "self_describing_kv_events",
        "service_connect_timeout",
        "service_socket",
        "service_timeout",
        "spec_name",
        "store_threshold",
        "sync_load_budget_bytes",
    }
)

# Keys another CPU backend reads that this one cannot serve, with what to do
# instead. Every other unsupported key is reported as unknown.
_UNSUPPORTED_KEYS = {
    "cache_policy_module_path": (
        "the service owns eviction, so remove cache_policy_module_path"
    ),
    "max_tracker_size": (
        "the service keeps no lookup tracker, so remove max_tracker_size"
    ),
    "secondary_tiers": (
        "the service has no secondary tiers; use TieringOffloadingSpec for them"
    ),
}


def _validated_extra_config(config: OffloadingConfig) -> dict[str, Any]:
    """Returns the extra config, rejecting every setting this backend cannot honour."""
    extra = dict(config.extra_config)
    for key in extra:
        if key in _UNSUPPORTED_KEYS:
            raise ValueError(
                f"RustServiceOffloadingSpec does not support {key!r}: "
                f"{_UNSUPPORTED_KEYS[key]}"
            )
        if key not in _SUPPORTED_KEYS:
            raise ValueError(
                f"RustServiceOffloadingSpec does not support "
                f"kv_connector_extra_config key {key!r}; it honours "
                f"{sorted(_SUPPORTED_KEYS)}"
            )
    region = extra.get("offload_region", "shared")
    if region != "shared":
        raise ValueError(
            "RustServiceOffloadingSpec keeps one service-owned copy per worker, so "
            f"it always uses offload_region='shared'; remove offload_region={region!r}"
        )
    eviction_policy = extra.get("eviction_policy", "lru")
    if eviction_policy != "lru":
        raise ValueError(
            "RustServiceOffloadingSpec evicts the least recently used key only; "
            f"remove eviction_policy={eviction_policy!r}"
        )
    store_threshold = int(extra.get("store_threshold", 0))
    if not 0 <= store_threshold < 2:
        raise ValueError(
            "RustServiceOffloadingSpec stores every offered chunk instead of "
            f"counting reuse; remove store_threshold={store_threshold} (a value "
            "below 2 means no filter)"
        )
    return extra


class RustServiceOffloadingSpec(CPUOffloadingSpec):
    """Single-node, topology-specific, service-owned DRAM cache."""

    def __init__(self, config: OffloadingConfig):
        if config.canonical_layout:
            raise ValueError(
                "RustServiceOffloadingSpec stores a private copy per worker, so "
                "canonical pages are not supported; remove canonical_layout"
            )
        if config.parallel.pp_size != 1 or config.parallel.pcp_size != 1:
            raise ValueError("Rust cache service initially supports TP-only execution")
        if config.parallel.dcp_size != 1:
            raise ValueError("Rust cache service does not support context parallelism")
        extra = _validated_extra_config(config)
        extra["offload_region"] = "shared"
        super().__init__(replace(config, extra_config=extra, replicated_layout=False))
        path = extra.get("service_socket")
        if not isinstance(path, str) or not path:
            raise ValueError("service_socket is required for RustServiceOffloadingSpec")
        self.path: str = path
        self.timeout = float(extra.get("service_timeout", 5.0))
        if self.timeout <= 0:
            raise ValueError("service_timeout must be positive")
        self.connect_timeout = float(extra.get("service_connect_timeout", 5.0))
        if self.connect_timeout <= 0:
            raise ValueError("service_connect_timeout must be positive")
        pin_host_memory = extra.get("pin_host_memory", True)
        if not isinstance(pin_host_memory, bool):
            raise ValueError("pin_host_memory must be a boolean")
        # Registering the region page-locks it, which makes the cache
        # unevictable; set `pin_host_memory: false` when the pages must stay
        # migratable for kernel-managed offload (rswap).
        self.pin_host_memory: bool = pin_host_memory
        prefetch = extra.get("prefetch_on_load", True)
        if not isinstance(prefetch, bool):
            raise ValueError("prefetch_on_load must be a boolean")
        self.prefetch_on_load: bool = prefetch
        prefetch_on_lookup = extra.get("prefetch_on_lookup", True)
        if not isinstance(prefetch_on_lookup, bool):
            raise ValueError("prefetch_on_lookup must be a boolean")
        # A `pending` lookup means the chunk is committed but not in DRAM, so the
        # connector defers that request; the manager asks the service to read the
        # chunk back at the step's end instead of leaving the page faults to the
        # deferred load.
        self.prefetch_on_lookup: bool = prefetch_on_lookup
        self.worker_bytes = self.cpu_page_size_per_worker
        world = config.parallel.world_size
        self.kv_bytes_per_chunk = (
            (self.worker_bytes * world + mmap.PAGESIZE - 1)
            // mmap.PAGESIZE
            * mmap.PAGESIZE
        )
        if self.kv_bytes_per_chunk == 0:
            raise ValueError("Rust cache service requires a nonempty KV layout")
        requested_bytes = int(extra["cpu_bytes_to_use"])
        self.num_chunks = requested_bytes // self.kv_bytes_per_chunk
        if self.num_chunks == 0:
            raise ValueError(
                f"cpu_bytes_to_use={requested_bytes} is smaller than one "
                f"{self.kv_bytes_per_chunk}-byte chunk; reserve at least "
                f"{self.kv_bytes_per_chunk} bytes"
            )
        if self.num_chunks > 2**31 - 1:
            raise ValueError(
                f"cpu_bytes_to_use={requested_bytes} needs more chunks than int32 "
                f"slot IDs address; reserve at most "
                f"{(2**31 - 1) * self.kv_bytes_per_chunk} bytes"
            )
        self.num_chunks = self._cap_to_service_budget(self.num_chunks, requested_bytes)
        logger.info(
            "Rust cache service namespace %s: worker_bytes=%d row_bytes=%d "
            "num_chunks=%d bytes=%d world_size=%d",
            self.config.engine_id,
            self.worker_bytes,
            self.kv_bytes_per_chunk,
            self.num_chunks,
            self.num_chunks * self.kv_bytes_per_chunk,
            world,
        )
        # The inherited info was built before the overrides above, so rebuild it
        # from the geometry the service is asked to allocate.
        self.cpu_info = CPUOffloadingInfo.from_config(
            self.config, self.num_chunks, self.kv_bytes_per_chunk
        )
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "model": asdict(config.model),
                    "groups": [asdict(group) for group in config.groups],
                    "cache": asdict(config.cache),
                    "worker_bytes": self.worker_bytes,
                    "world_size": world,
                    "layout": config.kv_cache_layout,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def _client(self) -> CacheServiceClient:
        return CacheServiceClient(self.path, self.timeout, self.connect_timeout)

    def _service_capacity_bytes(self) -> int:
        """Reads the aggregate budget the service reserves namespaces from."""
        client = self._client()
        try:
            return int(client.call("info", retry=True)["capacity_bytes"])
        finally:
            client.close()

    def _cap_to_service_budget(self, num_chunks: int, requested_bytes: int) -> int:
        """Caps the chunk count to what the service's budget can hold."""
        capacity_bytes = self._service_capacity_bytes()
        budget_chunks = capacity_bytes // self.kv_bytes_per_chunk
        if num_chunks <= budget_chunks:
            return num_chunks
        if budget_chunks == 0:
            raise ValueError(
                f"the service reports --capacity-bytes={capacity_bytes}, which leaves "
                f"no room for a {self.kv_bytes_per_chunk}-byte chunk; raise "
                f"--capacity-bytes or lower cpu_bytes_to_use={requested_bytes}"
            )
        logger.warning(
            "cpu_bytes_to_use=%d exceeds the service's --capacity-bytes=%d: using "
            "%d chunk(s) of %d bytes instead of %d",
            requested_bytes,
            capacity_bytes,
            budget_chunks,
            self.kv_bytes_per_chunk,
            num_chunks,
        )
        return budget_chunks

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            client = self._client()
            try:
                client.call(
                    "open",
                    version=1,
                    namespace=self.config.engine_id,
                    fingerprint=self.fingerprint,
                    worker_bytes=self.worker_bytes,
                    world_size=self.config.parallel.world_size,
                    num_chunks=self.num_chunks,
                )
            except Exception:
                client.close()
                raise
            self._manager = ServiceOffloadingManager(
                client,
                self.num_chunks,
                info=self.cpu_info,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                prefetch=self.prefetch_on_load,
                prefetch_on_lookup=self.prefetch_on_lookup,
            )
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        client = self._client()
        region = None
        try:
            rank = self.config.parallel.rank
            geometry = client.call(
                "attach",
                version=1,
                namespace=self.config.engine_id,
                fingerprint=self.fingerprint,
                rank=rank,
            )
            if (
                geometry.get("version") != 1
                or geometry.get("worker_bytes") != self.worker_bytes
                or geometry.get("world_size") != self.config.parallel.world_size
                or geometry.get("row_bytes") != self.kv_bytes_per_chunk
                or geometry.get("num_chunks") != self.num_chunks
            ):
                raise ValueError("Rust service returned incompatible cache geometry")
            region = ServiceOffloadRegion(
                client,
                client.receive_fd(),
                self.num_chunks,
                self.kv_bytes_per_chunk,
                self.worker_bytes,
                self.config.parallel.world_size,
                rank,
                pin_host_memory=self.pin_host_memory,
            )
            return CPUOffloadingWorker(
                kv_caches,
                self.blocks_per_chunk,
                self.num_chunks,
                mmap_region=region,
            )
        except Exception:
            if region is not None:
                region.cleanup()
            else:
                client.close()
            raise
