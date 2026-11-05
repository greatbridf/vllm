# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual Rust subprocess tests for storage ownership and lease publication."""

import mmap
import os
import socket
import struct
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.service.client import (
    CacheServiceClient,
    CacheServiceUnavailable,
)
from vllm.v1.kv_offload.service.region import ServiceOffloadRegion
from vllm.v1.kv_offload.service.spec import RustServiceOffloadingSpec

PAGE = mmap.PAGESIZE
BINARY = Path(
    os.environ.get(
        "VLLM_KV_CACHE_SERVICE_BINARY", "rust/target/debug/vllm-kv-cache-service"
    )
).resolve()


@pytest.fixture
def service(tmp_path):
    if not BINARY.is_file():
        pytest.skip("Build vllm-kv-cache-service or set VLLM_KV_CACHE_SERVICE_BINARY")
    tmp_path.chmod(0o700)
    path = str(tmp_path / "cache.sock")
    process = subprocess.Popen(
        [str(BINARY), "--socket", path, "--capacity-bytes", str(64 * PAGE)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not Path(path).exists():
            if process.poll() is not None:
                pytest.fail(f"Service exited: {process.communicate()[1]!r}")
            if time.monotonic() >= deadline:
                pytest.fail("Service startup timed out")
            time.sleep(0.01)
        yield path
    finally:
        process.terminate()
        process.communicate(timeout=10)


def _spec(path, rank=0):
    return RustServiceOffloadingSpec(
        replace(
            make_offloading_config(
                cpu_bytes_to_use=2 * PAGE,
                worker_kv_bytes_per_block=64,
                world_size=2,
                rank=rank,
                extra_config={"service_socket": path},
            ),
            engine_id="test-service",
        )
    )


def _attach(spec):
    client = CacheServiceClient(spec.path)
    geometry = client.call(
        "attach",
        version=1,
        namespace=spec.config.engine_id,
        fingerprint=spec.fingerprint,
        rank=spec.config.parallel.rank,
    )
    return ServiceOffloadRegion(
        client,
        client.receive_fd(),
        geometry["num_chunks"],
        geometry["row_bytes"],
        geometry["worker_bytes"],
        geometry["world_size"],
        spec.config.parallel.rank,
    )


def test_rust_service_shares_rank_slices_and_publishes_only_completed_stores(service):
    spec = _spec(service)
    manager = spec.get_manager()
    regions = [_attach(_spec(service, rank)) for rank in range(2)]
    try:
        views = [region.create_next_worker_view(64) for region in regions]
        request = ReqContext("first")
        key = OffloadKey(b"block-a")
        output = manager.prepare_store([key], request)
        assert output is not None
        slot = int(output.store_spec.block_ids[0])
        assert manager.lookup(key, request) is LookupResult.HIT_PENDING
        views[0][slot].fill_(17)
        views[1][slot].fill_(29)
        assert torch.all(regions[0].base_tensor[slot * PAGE : slot * PAGE + 64] == 17)
        assert torch.all(
            regions[0].base_tensor[slot * PAGE + 64 : slot * PAGE + 128] == 29
        )
        manager.complete_store([key], request)
        assert manager.lookup(key, request) is LookupResult.HIT
        load = manager.prepare_load([key], request)
        assert int(load.block_ids[0]) == slot
        manager.on_request_finished(request)
        assert manager.client.call("stats")["stats"]["leased"] == 1
        manager.complete_load([key], request)
        assert manager.client.call("stats")["stats"]["leased"] == 0
        manager.reset_cache()
        # The connector flushes in-flight transfers, then reports the flush.
        manager.on_reset_flushed()
        assert manager.lookup(key, ReqContext("next")) is LookupResult.MISS
    finally:
        for region in regions:
            region.cleanup()
        manager.shutdown()


def test_store_commit_round_trip_publishes_the_leased_slot(service):
    manager = _spec(service).get_manager()
    request = ReqContext("first")
    try:
        key = OffloadKey(b"round-trip")
        output = manager.prepare_store([key], request)
        assert output is not None
        assert manager.lookup(key, request) is LookupResult.HIT_PENDING
        manager.complete_store([key], request)
        assert manager.lookup(key, request) is LookupResult.HIT
        load = manager.prepare_load([key], request)
        assert load.block_ids == output.store_spec.block_ids
        stats = manager.client.call("stats")["stats"]
        assert stats["commits"] == 1
        assert stats["failed_commits"] == 0
        assert stats["pending"] == 0
        assert stats["pending_hits"] == 1
        assert stats["leased"] == 1
        manager.on_request_finished(request)
        manager.complete_load([key], request)
    finally:
        manager.shutdown()


def test_commit_from_a_request_that_did_not_reserve_is_rejected(service):
    manager = _spec(service).get_manager()
    request = ReqContext("first")
    try:
        key = OffloadKey(b"foreign-commit")
        assert manager.prepare_store([key], request) is not None
        with pytest.raises(RuntimeError, match="does not own the reservation"):
            manager.client.call(
                "commit", keys=[key.hex()], request="another", success=True
            )
        # A rejected request keeps the connection and the reservation alive.
        assert manager.client.call("stats")["stats"]["pending"] == 1
    finally:
        manager.shutdown()


def test_lookup_lease_prevents_eviction_and_failed_store_is_removed(service):
    manager = _spec(service).get_manager()
    request = ReqContext("first")
    try:
        keys = [OffloadKey(b"aa"), OffloadKey(b"bb")]
        assert manager.prepare_store(keys, request) is not None
        manager.complete_store(keys, request)
        assert manager.lookup(keys[0], request) is LookupResult.HIT
        third = OffloadKey(b"cc")
        output = manager.prepare_store([third], request)
        assert output is not None and output.evicted_keys == [keys[1]]
        manager.complete_store([third], request, success=False)
        assert manager.lookup(third, request) is LookupResult.MISS
        manager.on_request_finished(request)
        manager.reset_cache()
        manager.on_reset_flushed()
    finally:
        manager.shutdown()


def test_worker_disconnect_invalidates_namespace_without_recycling_old_mapping(service):
    spec = _spec(service)
    manager = spec.get_manager()
    region = _attach(spec)
    try:
        original = region.base_tensor
        original.fill_(13)
        # Force disconnect, leaving an outstanding mapped tensor alive.
        region.client.socket.shutdown(socket.SHUT_RDWR)
        region.client.close()
        # The service processes disconnect asynchronously; use observable RPC
        # invalidation rather than assuming an arbitrary delay is sufficient.
        deadline = time.monotonic() + 5
        while True:
            try:
                manager.client.call("stats")
            except (RuntimeError, ConnectionError):
                break
            if time.monotonic() > deadline:
                pytest.fail("Worker disconnect did not invalidate namespace")
        assert torch.all(original == 13)
        new_manager = _spec(service).get_manager()
        new_region = _attach(spec)
        try:
            assert torch.all(new_region.base_tensor == 0)
            assert torch.all(original == 13)
        finally:
            new_region.cleanup()
            new_manager.shutdown()
    finally:
        region.cleanup()
        manager.shutdown()


def test_service_spec_worker_uses_memfd_and_existing_dma_worker(service, monkeypatch):
    import vllm.v1.kv_offload.service.spec as module

    spec = _spec(service)
    manager = spec.get_manager()
    constructor = MagicMock()
    monkeypatch.setattr(module, "CPUOffloadingWorker", constructor)
    caches = MagicMock()
    try:
        spec.create_worker(caches)
        region = constructor.call_args.kwargs["mmap_region"]
        assert isinstance(region, ServiceOffloadRegion)
        assert region.total_size_bytes == 2 * PAGE
        assert constructor.call_args.args[0] is caches
        region.cleanup()
    finally:
        manager.shutdown()


def test_service_rejects_layout_mismatch(service):
    spec = _spec(service)
    manager = spec.get_manager()
    client = CacheServiceClient(service)
    try:
        with pytest.raises(RuntimeError, match="incompatible"):
            client.call(
                "attach",
                version=1,
                namespace=spec.config.engine_id,
                fingerprint="f" * 64,
                rank=0,
            )
    finally:
        client.close()
        manager.shutdown()


def test_service_rejects_capacity_overflow(service):
    client = CacheServiceClient(service)
    try:
        with pytest.raises(RuntimeError, match="capacity"):
            client.call(
                "open",
                version=1,
                namespace="oversized",
                fingerprint="a" * 64,
                worker_bytes=PAGE,
                world_size=1,
                num_chunks=65,
            )
    finally:
        client.close()


def test_info_reports_the_capacity_of_the_service(service):
    client = CacheServiceClient(service)
    try:
        info = client.call("info")
    finally:
        client.close()

    assert info["capacity_bytes"] == 64 * PAGE
    assert info["used_bytes"] == 0
    assert info["retired_bytes"] == 0
    assert info["namespaces"] == 0
    assert info["sessions"] == 0
    assert info["max_connections"] >= 1
    assert isinstance(info["version"], str)
    assert info["uptime_ms"] >= 0


def test_oversized_control_frame_does_not_terminate_service(service):
    client = CacheServiceClient(service)
    try:
        client.socket.sendall(struct.pack("!I", 1024 * 1024 + 1))
        assert client.socket.recv(1) == b""
        manager = _spec(service).get_manager()
        try:
            assert manager.client.call("stats")["stats"]["entries"] == 0
        finally:
            manager.shutdown()
    finally:
        client.close()


def test_client_connect_error_names_the_socket_path(tmp_path, monkeypatch):
    import vllm.v1.kv_offload.service.client as client_module

    monkeypatch.setattr(client_module, "_CONNECT_BACKOFF_S", 0.0)
    with pytest.raises(CacheServiceUnavailable, match="missing.sock"):
        CacheServiceClient(str(tmp_path / "missing.sock"))
