# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rust CPU cache service backing the native OffloadingConnector."""

import hashlib
import json
import mmap
from dataclasses import asdict, replace

from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingInfo
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.service.client import CacheServiceClient
from vllm.v1.kv_offload.service.manager import ServiceOffloadingManager
from vllm.v1.kv_offload.service.region import ServiceOffloadRegion


class RustServiceOffloadingSpec(CPUOffloadingSpec):
    """Single-node, topology-specific, service-owned DRAM cache."""

    def __init__(self, config: OffloadingConfig):
        if config.canonical_layout:
            raise ValueError("Rust cache service does not support canonical_layout")
        if config.parallel.pp_size != 1 or config.parallel.pcp_size != 1:
            raise ValueError("Rust cache service initially supports TP-only execution")
        if config.parallel.dcp_size != 1:
            raise ValueError("Rust cache service does not support context parallelism")
        if config.extra_config.get("secondary_tiers"):
            raise ValueError("Rust cache service has no secondary tiers")
        if config.extra_config.get("eviction_policy", "lru") != "lru":
            raise ValueError("Rust cache service initially supports only LRU eviction")
        if config.extra_config.get("store_threshold", 0) >= 2:
            raise ValueError("Rust cache service does not support store filtering")
        if config.extra_config.get("canonical_layout"):
            raise ValueError("Rust cache service does not support canonical_layout")
        # The service owns one copy per worker, without replicated-rank dedup.
        extra = dict(config.extra_config)
        extra["offload_region"] = "shared"
        super().__init__(replace(config, extra_config=extra, replicated_layout=False))
        path = extra.get("service_socket")
        if not isinstance(path, str) or not path:
            raise ValueError("service_socket is required for RustServiceOffloadingSpec")
        self.path: str = path
        self.timeout = float(extra.get("service_timeout", 5.0))
        if self.timeout <= 0:
            raise ValueError("service_timeout must be positive")
        self.worker_bytes = self.cpu_page_size_per_worker
        world = config.parallel.world_size
        self.kv_bytes_per_chunk = (
            (self.worker_bytes * world + mmap.PAGESIZE - 1)
            // mmap.PAGESIZE
            * mmap.PAGESIZE
        )
        if self.kv_bytes_per_chunk == 0:
            raise ValueError("Rust cache service requires a nonempty KV layout")
        self.num_chunks = int(extra["cpu_bytes_to_use"]) // self.kv_bytes_per_chunk
        if not 0 < self.num_chunks <= 2**31 - 1:
            raise ValueError(
                "CPU budget must fit at least one chunk and int32 slot IDs"
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
        return CacheServiceClient(self.path, self.timeout)

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
