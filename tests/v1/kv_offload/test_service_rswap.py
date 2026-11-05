# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live rswap reclaim test for the Rust KV cache service.

The service publishes one arena digest per cache page and attaches the digest
program to its shared mapping. This test fills the cache through a client
mapping, forces the kernel to write exactly those pages into an rswap device,
and then checks both halves of the kernel contract: every page reached the
remote store under a digest of its own, and every page still reads back
unchanged.

It needs root (CAP_BPF, CAP_SYS_ADMIN) and an rswap-capable kernel, so it only
runs when ``VLLM_TEST_KV_SERVICE_RSWAP=1`` is set. It creates and destroys its
own rswap device and leaves other swap devices alone.

A kernel that resolves the shmem mapping of an already-swapped-out folio in
``rswap_shmem_move_hashes()`` drops every memfd digest and stores all pages
under one all-zero key; this test fails there by design.
"""

import ctypes
import mmap
import os
import socket
import struct
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.service.client import CacheServiceClient
from vllm.v1.kv_offload.service.region import ServiceOffloadRegion
from vllm.v1.kv_offload.service.spec import RustServiceOffloadingSpec

PAGE = mmap.PAGESIZE
PAGES_PER_CHUNK = 4
CHUNKS = 8
CACHE_BYTES = CHUNKS * PAGES_PER_CHUNK * PAGE
# Above the default xswap priority so this test's pages land on its own device.
DEVICE_PRIORITY = 32767
CANONICAL_KEY_LEN = 36
MADV_PAGEOUT = getattr(mmap, "MADV_PAGEOUT", 21)
BINARY = Path(
    os.environ.get(
        "VLLM_KV_CACHE_SERVICE_BINARY", "rust/target/debug/vllm-kv-cache-service"
    )
).resolve()
ATTACH_PATH = Path("/sys/kernel/mm/xswap/rswap_attach")
CREATE_PATH = Path("/sys/kernel/mm/xswap/create")
DESTROY_PATH = Path("/sys/kernel/mm/xswap/destroy")
DEBUGFS = Path("/sys/kernel/debug/rswap")
MODULE = Path("/sys/module/rswap/parameters")
LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def rswap_types():
    return [
        row.split()[0].removeprefix("xswap")
        for row in Path("/proc/swaps").read_text().splitlines()[1:]
        if row.startswith("xswap")
    ]


def read_stats(device):
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
    if not ATTACH_PATH.is_file() or not CREATE_PATH.is_file():
        pytest.skip("kernel has no CONFIG_RSWAP swap interface")
    if not BINARY.is_file():
        pytest.skip("build vllm-kv-cache-service or set VLLM_KV_CACHE_SERVICE_BINARY")
    help_text = subprocess.run(
        [str(BINARY), "--help"], capture_output=True, text=True
    ).stdout
    if "--rswap-attach" not in help_text:
        pytest.skip("service was built without clang, so it has no --rswap-attach")
    # Per-CPU swap clusters outlive the allocation that cached them, so pages
    # can land on a previously used rswap device regardless of priority.
    busy = [
        row.split()[0]
        for row in Path("/proc/swaps").read_text().splitlines()[1:]
        if row.split()[3] != "0"
    ]
    if busy:
        pytest.skip(
            "swap devices are in use, so allocation may bypass the test device: "
            + ", ".join(busy)
        )
    existing = rswap_types()
    if existing:
        pytest.skip(
            "another rswap device is active; destroy it first: "
            + " ".join(f"echo {device} > {DESTROY_PATH}" for device in existing)
        )
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


def _spec(path):
    return RustServiceOffloadingSpec(
        replace(
            make_offloading_config(
                cpu_bytes_to_use=CACHE_BYTES,
                worker_kv_bytes_per_block=PAGES_PER_CHUNK * PAGE,
                extra_config={"service_socket": path},
            ),
            engine_id="rswap-reclaim",
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


def _pattern(key, page):
    seed = bytes([key[1], page]) + bytes(key)
    return (seed * (PAGE // len(seed) + 1))[:PAGE]


def _page_digest(key, page):
    """Mirror of `PageHash` in rust/src/kv-cache-service/src/hash.rs."""
    canonical = bytes(key)[:CANONICAL_KEY_LEN].ljust(CANONICAL_KEY_LEN, b"\0")
    return (canonical + struct.pack("<I", len(key)) + struct.pack("<Q", page)).hex()


def _redis_get(keys):
    """Reads `keys` from the rswap Redis backend; None if it is not one."""
    params = {}
    for name in ("backend", "host", "port", "redis_user", "redis_password"):
        path = MODULE / name
        if path.is_file():
            params[name] = path.read_text().strip()
    if params.get("backend") != "redis":
        return None
    try:
        connection = socket.create_connection(
            (params["host"], int(params["port"])), timeout=5
        )
    except OSError:
        return None

    def command(*args):
        frame = f"*{len(args)}\r\n".encode()
        for arg in args:
            frame += b"$%d\r\n%s\r\n" % (len(arg), arg)
        connection.sendall(frame)
        head, _, rest = connection.recv(64).partition(b"\r\n")
        if head.startswith((b"+", b"-", b":")):
            return head
        if head.startswith(b"$-1"):
            return None
        payload = rest
        while len(payload) < int(head[1:]) + 2:
            payload += connection.recv(int(head[1:]) + 2 - len(payload))
        return payload[: int(head[1:])]

    try:
        command(
            b"AUTH",
            params.get("redis_user", "default").encode(),
            params.get("redis_password", "").encode(),
        )
        return {key: command(b"GET", key.encode()) for key in keys}
    finally:
        connection.close()


def test_reclaimed_pages_round_trip_through_rswap(device, service):
    spec = _spec(service)
    manager = spec.get_manager()
    region = _attach(spec)
    view = region.create_next_worker_view(PAGES_PER_CHUNK * PAGE).numpy()
    request = ReqContext("rswap")
    pages = {}
    for index in range(CHUNKS):
        key = OffloadKey(bytes([0x20, index]))
        output = manager.prepare_store([key], request)
        assert output is not None
        slot = int(output.store_spec.block_ids[0])
        for page in range(PAGES_PER_CHUNK):
            payload = _pattern(key, page)
            view[slot, page * PAGE : (page + 1) * PAGE] = np.frombuffer(
                payload, dtype=np.int8
            )
            pages[(slot * PAGES_PER_CHUNK + page) * PAGE] = (key, page, payload)
        manager.complete_store([key], request)
        assert manager.lookup(key, request) is LookupResult.HIT

    before = read_stats(device)
    advised = LIBC.madvise(
        ctypes.c_void_p(region.base_tensor.data_ptr()),
        ctypes.c_size_t(CACHE_BYTES),
        ctypes.c_int(MADV_PAGEOUT),
    )
    assert advised == 0, os.strerror(ctypes.get_errno())
    stats = {
        name: value - before.get(name, 0) for name, value in read_stats(device).items()
    }
    assert stats["pages_written"] >= len(pages), stats
    assert stats["errors_written"] == 0, stats
    # A page without a digest cannot be stored at all: the service's arena
    # program has to resolve one for every cache page.
    assert stats["no_digest"] == 0, stats

    # The backend audit is what proves the digest program drove the store keys:
    # a page stored under a digest the service never published, or under a
    # digest shared with another page, is data loss on the next swap-in.
    digests = {
        offset: _page_digest(key, page) for offset, (key, page, _) in pages.items()
    }
    stored = _redis_get(list(digests.values()))
    if stored is None:
        print(f"no readable rswap backend: {len(pages)} pages stored unaudited")
    else:
        missing = [
            offset for offset, digest in digests.items() if stored[digest] is None
        ]
        assert not missing, f"pages {missing} are not under their published digest"
        assert len(set(digests.values())) == len(pages)
        wrong = [
            offset
            for offset, digest in digests.items()
            if stored[digest] != pages[offset][2]
        ]
        assert not wrong, f"pages {wrong} hold another page's bytes in rswap"

    # Destroying the device swaps every entry back in, so this compares what
    # the service published with what survived the round trip.
    DESTROY_PATH.write_text(device + "\n")
    assert device not in rswap_types(), "the rswap device did not go away"
    mismatched = [
        offset // PAGE
        for offset, (_, _, payload) in pages.items()
        if bytes(region.base_tensor[offset : offset + PAGE].numpy()) != payload
    ]
    assert not mismatched, f"pages {mismatched} did not round trip"
