# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live rswap test of the real offload store/load path, through a mock engine.

The Rust service owns the DRAM; rswap stores its cold pages in the backend
configured on the kernel command line and brings them back on the next access.
`test_service_rswap.py` proves the digest protocol against the service's own
mapping. This file drives the production path instead: the real
`RustServiceOffloadingSpec`, the real `ServiceOffloadingManager` RPCs, and a
real `ServiceOffloadRegion`, with a stand-in for the GPU copy (host bytes
through the region views) because no CUDA device is required to exercise
storage, eviction, reclaim, and reload.

Both halves of the pinning contract are covered:

* a page-locked region is unevictable, so the kernel never publishes a digest
  for it (`test_locked_region_is_not_reclaimable`);
* with host pinning released for the service region, a store survives swap-out
  and a load returns the same bytes (`test_mock_engine_store_load_round_trip`).

It needs root (CAP_BPF, CAP_SYS_ADMIN), an rswap-capable kernel with a readable
backend, and `VLLM_TEST_KV_SERVICE_RSWAP=1`; it creates and destroys its own
rswap device and leaves any other swap device alone.
"""

import ctypes
import errno
import mmap
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region
from vllm.v1.kv_offload.service.client import CacheServiceClient
from vllm.v1.kv_offload.service.region import ServiceOffloadRegion
from vllm.v1.kv_offload.service.spec import RustServiceOffloadingSpec

PAGE = mmap.PAGESIZE
PAGES_PER_CHUNK = 4
CHUNKS = int(os.environ.get("VLLM_TEST_KV_SERVICE_RSWAP_CHUNKS", "8"))
CACHE_BYTES = CHUNKS * PAGES_PER_CHUNK * PAGE
DEVICE_PRIORITY = 32767
MADV_PAGEOUT = getattr(mmap, "MADV_PAGEOUT", 21)
BINARY = Path(
    os.environ.get(
        "VLLM_KV_CACHE_SERVICE_BINARY", "rust/target/debug/vllm-kv-cache-service"
    )
).resolve()
CREATE_PATH = Path("/sys/kernel/mm/xswap/create")
DESTROY_PATH = Path("/sys/kernel/mm/xswap/destroy")
DEBUGFS = Path("/sys/kernel/debug/rswap")
LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def rswap_types() -> list[str]:
    return [
        row.split()[0].removeprefix("xswap")
        for row in Path("/proc/swaps").read_text().splitlines()[1:]
        if row.startswith("xswap")
    ]


def read_stats(device: str) -> dict[str, int]:
    stats = {}
    for line in (DEBUGFS / f"type{device}" / "stats").read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            stats[fields[0]] = int(fields[1])
    return stats


@pytest.fixture
def device():
    if os.environ.get("VLLM_TEST_KV_SERVICE_RSWAP") != "1":
        pytest.skip("set VLLM_TEST_KV_SERVICE_RSWAP=1 to run the live rswap test")
    if os.geteuid() != 0:
        pytest.skip("the rswap digest path needs root (CAP_BPF, CAP_SYS_ADMIN)")
    if not CREATE_PATH.is_file() or not DESTROY_PATH.is_file():
        pytest.skip("kernel has no CONFIG_RSWAP swap interface")
    if rswap_types():
        pytest.skip("another rswap device is active; destroy it first")
    CREATE_PATH.write_text(f"rswap {DEVICE_PRIORITY}\n")
    created = rswap_types()
    if len(created) != 1:
        pytest.skip("kernel added no rswap device: check dmesg for rswap_enable")
    if not (DEBUGFS / f"type{created[0]}" / "stats").is_file():
        DESTROY_PATH.write_text(created[0] + "\n")
        pytest.skip("kernel has no CONFIG_DEBUG_FS rswap statistics")
    try:
        yield created[0]
    finally:
        if created[0] in rswap_types():
            DESTROY_PATH.write_text(created[0] + "\n")


@pytest.fixture
def service(tmp_path):
    if not BINARY.is_file():
        pytest.skip("build vllm-kv-cache-service or set VLLM_KV_CACHE_SERVICE_BINARY")
    tmp_path.chmod(0o700)
    path = str(tmp_path / "cache.sock")
    process = subprocess.Popen(
        [
            str(BINARY),
            "--socket",
            path,
            "--capacity-bytes",
            str(CACHE_BYTES),
            "--rswap-attach",
        ],
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


def _spec(path: str, pin_host_memory: bool) -> RustServiceOffloadingSpec:
    return RustServiceOffloadingSpec(
        replace(
            make_offloading_config(
                cpu_bytes_to_use=CACHE_BYTES,
                worker_kv_bytes_per_block=PAGES_PER_CHUNK * PAGE,
                world_size=1,
                extra_config={
                    "service_socket": path,
                    "pin_host_memory": pin_host_memory,
                },
            ),
            engine_id="service-rswap",
        )
    )


def _region(spec: RustServiceOffloadingSpec) -> ServiceOffloadRegion:
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
        pin_host_memory=spec.pin_host_memory,
    )


class MockEngine:
    """The connector's scheduler/worker split with a host copy instead of DMA.

    `CPUOffloadingWorker` needs accelerator tensors to build its copy list, so it
    cannot run without a GPU; everything else here is the real path: the real
    manager, its RPCs, and the real region the workers map. The one simulated
    step is the copy between "device" blocks and the region slot.
    """

    def __init__(self, spec: RustServiceOffloadingSpec):
        # The scheduler opens the namespace; a worker can only attach to one.
        self.manager = spec.get_manager()
        self.region = _region(spec)
        self.view = self.region.create_next_worker_view(PAGES_PER_CHUNK * PAGE).numpy()
        # Simulated GPU KV blocks, keyed by key bytes.
        self.device: dict[bytes, np.ndarray] = {}
        self.block = 0

    def store(self, key: OffloadKey, request: ReqContext, payload: np.ndarray) -> int:
        output = self.manager.prepare_store([key], request)
        assert output is not None
        assert isinstance(output.store_spec, CPULoadStoreSpec)
        slot = int(output.store_spec.block_ids[0])
        self.view[slot] = payload
        self.device[bytes(key)] = payload
        self.manager.complete_store([key], request)
        return slot

    def load(self, key: OffloadKey, request: ReqContext) -> np.ndarray:
        load_spec = self.manager.prepare_load([key], request)
        assert isinstance(load_spec, CPULoadStoreSpec)
        slots = load_spec.block_ids
        loaded = self.view[int(slots[0])].copy()
        self.manager.complete_load([key], request)
        return loaded

    def cleanup(self) -> None:
        self.manager.shutdown()
        self.region.cleanup()


def _payload(index: int) -> np.ndarray:
    rng = np.random.default_rng(index)
    return rng.integers(0, 256, CACHE_BYTES // CHUNKS, dtype=np.uint8).view(np.int8)


def _madvise_page_out(region: ServiceOffloadRegion) -> int:
    """Asks the kernel to reclaim the region, returning the errno it reports."""
    ctypes.set_errno(0)
    advised = LIBC.madvise(
        ctypes.c_void_p(region.base_tensor.data_ptr()),
        ctypes.c_size_t(region.total_size_bytes),
        ctypes.c_int(MADV_PAGEOUT),
    )
    return 0 if advised == 0 else ctypes.get_errno()


def _page_out(region: ServiceOffloadRegion) -> None:
    error = _madvise_page_out(region)
    assert error == 0, os.strerror(error)


def _reclaimed(device: str, before: dict[str, int]) -> int:
    after = read_stats(device)
    return after.get("pages_written", 0) - before.get("pages_written", 0)


def test_mock_engine_store_load_round_trip(device, service):
    """Store through the manager, swap the pages out, load them back."""
    spec = _spec(service, pin_host_memory=False)
    engine = MockEngine(spec)
    request = ReqContext("round-trip")
    pages = CHUNKS * PAGES_PER_CHUNK
    try:
        assert engine.region.PIN_HOST_MEMORY is False
        keys = [OffloadKey(bytes([0x30, index])) for index in range(CHUNKS)]
        payloads = [_payload(index) for index in range(CHUNKS)]
        for key, payload in zip(keys, payloads):
            engine.store(key, request, payload)
        assert all(
            engine.manager.lookup(key, request) is LookupResult.HIT for key in keys
        )

        # The backend may be read back page by page, but counting this device's
        # own writes is what proves the pages left DRAM instead of an unrelated
        # swap device taking them.
        before = read_stats(device)
        _page_out(engine.region)
        written = _reclaimed(device, before)
        assert written >= pages, (
            f"only {written} of {pages} pages reached rswap device {device}"
        )

        # Destroying the device swaps every entry back in.
        DESTROY_PATH.write_text(device + "\n")
        assert device not in rswap_types()

        for key, payload in zip(keys, payloads):
            np.testing.assert_array_equal(engine.load(key, request), payload)
    finally:
        engine.cleanup()


def test_locked_region_is_not_reclaimable(device, service):
    """A page-locked cache is invisible to rswap, which is why pinning defaults
    to pageable for this backend when kernel offload is wanted."""
    spec = _spec(service, pin_host_memory=False)
    engine = MockEngine(spec)
    request = ReqContext("locked")
    try:
        key = OffloadKey(b"locked")
        engine.store(key, request, _payload(0))
        base = engine.region.base_tensor.data_ptr()
        size = engine.region.total_size_bytes
        assert LIBC.mlock(ctypes.c_void_p(base), ctypes.c_size_t(size)) == 0, (
            os.strerror(ctypes.get_errno())
        )
        try:
            # `can_madv_lru_vma()` excludes VM_LOCKED ranges, so the kernel
            # refuses the request outright instead of silently skipping pages.
            before = read_stats(device)
            error = _madvise_page_out(engine.region)
            assert error in (0, errno.EINVAL), os.strerror(error)
            assert _reclaimed(device, before) == 0
        finally:
            assert LIBC.munlock(ctypes.c_void_p(base), ctypes.c_size_t(size)) == 0

        # The same pages are reclaimable once the lock is gone.
        before = read_stats(device)
        _page_out(engine.region)
        assert _reclaimed(device, before) >= PAGES_PER_CHUNK
    finally:
        engine.cleanup()


def test_released_pinning_skips_host_registration(service, monkeypatch):
    """`pin_host_memory: false` keeps the region out of the accelerator's pins."""
    import vllm.v1.kv_offload.cpu.gpu_worker as worker_module

    registered: list[int] = []

    def record_register(ptr: int, size: int) -> bool:
        registered.append(ptr)
        return False

    monkeypatch.setattr(worker_module, "host_register", record_register)

    unpinned = _spec(service, pin_host_memory=False)
    unpinned_manager = unpinned.get_manager()
    region = _region(unpinned)
    try:
        pin_mmap_region(region, worker_slot_only=True)
        assert registered == []
        assert region.PIN_HOST_MEMORY is False
    finally:
        region.cleanup()
        unpinned_manager.shutdown()

    pinned = _spec(service, pin_host_memory=True)
    pinned_manager = pinned.get_manager()
    region = _region(pinned)
    try:
        pin_mmap_region(region, worker_slot_only=True)
        assert registered, "a pinned service region must be registered"
        assert region.PIN_HOST_MEMORY is True
    finally:
        region.cleanup()
        pinned_manager.shutdown()
