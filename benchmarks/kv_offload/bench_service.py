# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulated KV-offload workload against the process-separated Rust service.

The harness drives the real deployment path -- `RustServiceOffloadingSpec`, the
`ServiceOffloadingManager` RPCs, and the `ServiceOffloadRegion` the workers map --
with a numpy copy standing in for the GPU DMA, because the copy engine is not what
this measures. `--mode cpu` runs the same workload against the in-process
`CPUOffloadingManager`: that is the upper bound for the same policy, so the gap is
the process boundary plus the wire protocol.

One step per request mirrors the connector:

* read: `lookup_many(prefix)` (truncating at the first miss), `prepare_load`,
  copy the chunks out, `complete_load`;
* write: `prepare_store(tail)`, copy them in, `complete_store`;
* `on_request_finished`, then the step boundary (`on_schedule_end`).

`--rswap` runs the same workload against an rswap attach point with host pinning
released, and forces every written chunk out to the remote backend at the end of
each round, so the next round's reads fault them back in. The page-out and the
resulting read cost are reported separately.

Examples:
    # One engine, 4k-token prompts (256 chunks of 16 tokens), 64 MiB cache.
    .venv/bin/python benchmarks/kv_offload/bench_service.py --mode both \\
        --prefix-chunks 192 --tail-chunks 64 --rounds 20

    # Eight engines sharing one service.
    .venv/bin/python benchmarks/kv_offload/bench_service.py --mode rust --threads 8

    # Cold chunks in the rswap remote tier (root, rswap kernel, --rswap-attach).
    sudo VLLM_TEST_KV_SERVICE_RSWAP=1 .venv/bin/python \\
        benchmarks/kv_offload/bench_service.py --mode rust --rswap

"""

import argparse
import ctypes
import json
import mmap
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.service.client import CacheServiceClient
from vllm.v1.kv_offload.service.region import ServiceOffloadRegion
from vllm.v1.kv_offload.service.spec import RustServiceOffloadingSpec

PAGE = mmap.PAGESIZE
BINARY = Path(
    os.environ.get(
        "VLLM_KV_CACHE_SERVICE_BINARY", "rust/target/debug/vllm-kv-cache-service"
    )
).resolve()
MADV_PAGEOUT = getattr(mmap, "MADV_PAGEOUT", 21)
LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
CSWAP_CREATE = Path("/sys/kernel/mm/xswap/create")
CSWAP_DESTROY = Path("/sys/kernel/mm/xswap/destroy")
RSWAP_DEBUGFS = Path("/sys/kernel/debug/rswap")
DEVICE_PRIORITY = 32767


def rswap_devices() -> list[str]:
    return [
        row.split()[0].removeprefix("xswap")
        for row in Path("/proc/swaps").read_text().splitlines()[1:]
        if row.startswith("xswap")
    ]


def rswap_stats(device: str) -> dict[str, int]:
    path = RSWAP_DEBUGFS / f"type{device}" / "stats"
    stats: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            stats[fields[0]] = int(fields[1])
    return stats


class RswapDevice:
    """An rswap device of its own, so the workload's pages reach the backend.

    Without a device, the digest program still runs but reclaim has nowhere to
    send the pages, and an unrelated swap device takes them instead -- the
    backend then sees no traffic at all.
    """

    def __init__(self) -> None:
        if os.geteuid() != 0:
            raise RuntimeError("--rswap needs root (BPF and xswap sysfs)")
        active = rswap_devices()
        if active:
            raise RuntimeError(f"another rswap device is active: {active}")
        CSWAP_CREATE.write_text(f"rswap {DEVICE_PRIORITY}\n")
        created = rswap_devices()
        if len(created) != 1:
            raise RuntimeError("kernel added no rswap device")
        self.device = created[0]
        self.before = rswap_stats(self.device)

    def stats(self) -> dict[str, int]:
        after = rswap_stats(self.device)
        return {
            "rswap_pages_written": after["pages_written"]
            - self.before["pages_written"],
            "rswap_pages_read": after["pages_read"] - self.before["pages_read"],
            "rswap_errors": after["errors_written"]
            - self.before["errors_written"]
            + after["errors_read"]
            - self.before["errors_read"],
            "rswap_no_digest": after["no_digest"] - self.before["no_digest"],
        }

    def close(self) -> None:
        if self.device in rswap_devices():
            CSWAP_DESTROY.write_text(self.device + "\n")


class Service:
    def __init__(self, capacity_bytes: int, rswap: bool):
        self.dir = tempfile.TemporaryDirectory()
        Path(self.dir.name).chmod(0o700)
        self.path = str(Path(self.dir.name) / "cache.sock")
        argv = [
            str(BINARY),
            "--socket",
            self.path,
            "--capacity-bytes",
            str(capacity_bytes),
        ]
        if rswap:
            argv.append("--rswap-attach")
        self.process = subprocess.Popen(argv, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 10
        while not Path(self.path).exists():
            if self.process.poll() is not None:
                raise RuntimeError("service exited during startup")
            if time.monotonic() > deadline:
                raise RuntimeError("service startup timed out")
            time.sleep(0.005)

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)
        self.dir.cleanup()


class Engine:
    """One engine's scheduler-side manager and worker-side region."""

    def __init__(self, socket_path: str, chunks: int, worker_bytes: int, name: str):
        self.spec = RustServiceOffloadingSpec(
            replace(
                make_offloading_config(
                    cpu_bytes_to_use=chunks * worker_bytes,
                    worker_kv_bytes_per_block=worker_bytes,
                    world_size=1,
                    engine_id=name,
                    extra_config={
                        "service_socket": socket_path,
                        "pin_host_memory": not RS_WORKLOAD.rswap,
                    },
                ),
                engine_id=name,
            )
        )
        self.manager = self.spec.get_manager()
        client = CacheServiceClient(socket_path)
        geometry = client.call(
            "attach",
            version=1,
            namespace=name,
            fingerprint=self.spec.fingerprint,
            rank=0,
        )
        self.region = ServiceOffloadRegion(
            client,
            client.receive_fd(),
            geometry["num_chunks"],
            geometry["row_bytes"],
            geometry["worker_bytes"],
            geometry["world_size"],
            0,
            pin_host_memory=self.spec.pin_host_memory,
        )
        self.view = self.region.create_next_worker_view(worker_bytes).numpy()

    def page_out(self) -> float:
        started = time.perf_counter()
        LIBC.madvise(
            ctypes.c_void_p(self.region.base_tensor.data_ptr()),
            ctypes.c_size_t(self.region.total_size_bytes),
            ctypes.c_int(MADV_PAGEOUT),
        )
        return time.perf_counter() - started

    def close(self) -> None:
        self.region.cleanup()
        self.manager.shutdown()


class Runs:
    def __init__(self) -> None:
        self.read: list[float] = []
        self.write: list[float] = []
        self.read_chunks = 0
        self.write_chunks = 0
        self.evicted = 0
        self.hits = 0
        self.misses = 0
        self.store_failures = 0
        self.page_out_s = 0.0
        self.page_outs = 0
        self.degraded = False


class Workload:
    rswap = False


RS_WORKLOAD = Workload()


def prefix_keys(count: int) -> list[OffloadKey]:
    """The shared prefix every request reuses: stored once, then read."""
    return [OffloadKey(bytes([0x40, i])) for i in range(count)]


def tail_keys(serial: int, count: int) -> list[OffloadKey]:
    """Freshly computed chunks of one request, unique across the whole run."""
    return [
        OffloadKey(bytes([0x50, serial >> 8, serial & 0xFF, i])) for i in range(count)
    ]


def warm(manager, view, count: int) -> None:
    """Publishes the shared prefix so the measured rounds are read-heavy."""
    if not count:
        return
    request = ReqContext("warmup")
    keys = prefix_keys(count)
    output = manager.prepare_store(keys, request)
    assert output is not None, "the shared prefix did not fit the cache"
    for slot in output.store_spec.block_ids:
        view[int(slot)][:] = 0x33
    manager.complete_store(output.keys_to_store, request)
    manager.on_request_finished(request)
    manager.on_schedule_end(None)


def step(
    manager, view, runs: Runs, prefix: int, tail: int, serial: int, copy: bool = True
) -> None:
    request = ReqContext(f"req-{serial}")
    started = time.perf_counter()
    if prefix:
        keys = prefix_keys(prefix)
        statuses = manager.lookup_many(keys, request, until_miss=True)
        # A truncating scan ends *at* the miss, so only the HIT prefix is loadable;
        # a HIT_PENDING or RETRY would defer the request, as the connector does.
        hit_keys = [
            key for key, status in zip(keys, statuses) if status is LookupResult.HIT
        ]
        runs.hits += len(hit_keys)
        runs.misses += prefix - len(hit_keys)
        if hit_keys:
            load_spec = manager.prepare_load(hit_keys, request)
            for slot in load_spec.block_ids:
                # Read a byte so the copy is not folded away, and so a remote
                # page is faulted in exactly as a DMA read would fault it.
                if copy:
                    view[int(slot)][0] &= 0x7F
            manager.complete_load(hit_keys, request)
            runs.read_chunks += len(hit_keys)
    runs.read.append(time.perf_counter() - started)

    started = time.perf_counter()
    if tail:
        keys = tail_keys(serial, tail)
        output = manager.prepare_store(keys, request)
        if output is None:
            runs.store_failures += 1
        else:
            slots = [int(slot) for slot in output.store_spec.block_ids]
            if copy:
                for slot in slots:
                    view[slot][:] = 0x5A
            manager.complete_store(output.keys_to_store, request)
            runs.write_chunks += len(slots)
            runs.evicted += len(output.evicted_keys)
    runs.write.append(time.perf_counter() - started)
    manager.on_request_finished(request)
    manager.on_schedule_end(None)


def run_rust(args, runs: Runs, index: int) -> None:
    engine = Engine(args.socket, args.chunks, args.worker_bytes, f"bench-{index}")
    try:
        warm(engine.manager, engine.view, args.prefix_chunks)
        for round_index in range(args.rounds):
            for request_index in range(args.requests):
                step(
                    engine.manager,
                    engine.view,
                    runs,
                    args.prefix_chunks,
                    args.tail_chunks,
                    (index * args.rounds + round_index) * args.requests + request_index,
                    not args.no_copy,
                )
            if args.rswap:
                runs.page_out_s += engine.page_out()
                runs.page_outs += 1
                runs.degraded = getattr(engine.manager, "_degraded", runs.degraded)
    finally:
        engine.close()


def run_cpu(args, runs: Runs) -> None:
    manager = CPUOffloadingManager(num_chunks=args.chunks)
    view = np.zeros((args.chunks, args.worker_bytes), dtype=np.uint8)
    warm(manager, view, args.prefix_chunks)
    for round_index in range(args.rounds):
        for request_index in range(args.requests):
            step(
                manager,
                view,
                runs,
                args.prefix_chunks,
                args.tail_chunks,
                round_index * args.requests + request_index,
                not args.no_copy,
            )


def pct(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    return ordered[max(int(len(ordered) * fraction) - 1, 0)] * 1e6


def summarise(name: str, runs: Runs, elapsed: float, worker_bytes: int) -> dict:
    chunks = runs.read_chunks + runs.write_chunks
    return {
        "mode": name,
        "elapsed_s": round(elapsed, 3),
        "requests": len(runs.write),
        "read_chunks": runs.read_chunks,
        "write_chunks": runs.write_chunks,
        "read_chunks_per_s": round(runs.read_chunks / elapsed),
        "write_chunks_per_s": round(runs.write_chunks / elapsed),
        "chunks_per_s": round(chunks / elapsed),
        "gib_per_s": round(chunks * worker_bytes / elapsed / 2**30, 3),
        "tokens_per_s": round(chunks * 16 / elapsed),
        "hits": runs.hits,
        "misses": runs.misses,
        "evicted": runs.evicted,
        "store_failures": runs.store_failures,
        "degraded": runs.degraded,
        "page_outs": runs.page_outs,
        "page_out_ms": round(runs.page_out_s * 1e3, 1),
        "read_p50_us": round(statistics.median(runs.read) * 1e6, 1) if runs.read else 0,
        "read_p99_us": round(pct(runs.read, 0.99), 1),
        "write_p50_us": round(statistics.median(runs.write) * 1e6, 1)
        if runs.write
        else 0,
        "write_p99_us": round(pct(runs.write, 0.99), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulated KV offload workload for the Rust cache service"
    )
    parser.add_argument("--mode", choices=["rust", "cpu", "both"], default="both")
    parser.add_argument("--worker-bytes", type=int, default=PAGE)
    parser.add_argument(
        "--chunks", type=int, default=1024, help="cache chunks per engine"
    )
    parser.add_argument("--requests", type=int, default=4, help="requests per round")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--prefix-chunks", type=int, default=192)
    parser.add_argument("--tail-chunks", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1, help="engines for rust mode")
    parser.add_argument("--rswap", action="store_true", help="attach rswap, unpinned")
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="skip the byte copies, to report the control plane alone",
    )
    parser.add_argument(
        "--socket",
        help="use a service already listening on this socket instead of starting one",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.rswap and args.mode == "cpu":
        print("--rswap needs --mode rust", file=sys.stderr)
        return 2
    if args.mode in ("rust", "both") and not BINARY.is_file():
        print(f"missing {BINARY}; build it first", file=sys.stderr)
        return 2

    RS_WORKLOAD.rswap = args.rswap
    results = []
    service = None
    if args.mode in ("rust", "both") and args.socket is None:
        service = Service(
            args.chunks * args.worker_bytes * max(args.threads, 1) * 2, args.rswap
        )
        args.socket = service.path
    device = None
    if args.rswap:
        device = RswapDevice()
    try:
        if args.mode in ("rust", "both"):
            runs = Runs()
            started = time.perf_counter()
            if args.threads == 1:
                run_rust(args, runs, 0)
            else:
                threads = [
                    threading.Thread(target=run_rust, args=(args, runs, index))
                    for index in range(args.threads)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            label = "rust" if args.threads == 1 else f"rust x{args.threads}"
            if args.no_copy:
                label += " (control only)"
            if args.rswap:
                label += " rswap"
            results.append(
                summarise(label, runs, time.perf_counter() - started, args.worker_bytes)
            )
        if args.mode in ("cpu", "both"):
            runs = Runs()
            started = time.perf_counter()
            run_cpu(args, runs)
            cpu_label = "cpu" + (" (control only)" if args.no_copy else "")
            results.append(
                summarise(
                    cpu_label, runs, time.perf_counter() - started, args.worker_bytes
                )
            )
    finally:
        if service is not None:
            service.close()

    if device is not None:
        counters = device.stats()
        device.close()
        for result in results:
            result.update(counters)

    for result in results:
        print(json.dumps(result) if args.json else json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
