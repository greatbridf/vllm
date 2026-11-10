# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded constant-arrival rswap comparison with real service and host copies."""

import argparse
import concurrent.futures
import ctypes
import errno
import gc
import json
import os
import resource
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from benchmarks.kv_offload import bench_service as B
from tests.v1.kv_offload.test_service_rswap import _page_digest, _redis_params
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext

GIB = 2**30
ROW = 5 * 2**20


def delete_remote(keys):
    params = _redis_params()
    if params is None:
        raise RuntimeError("cleanup requires the configured Redis backend")
    with socket.create_connection(
        (params["host"], int(params["port"])), timeout=30
    ) as connection:
        reader = connection.makefile("rb")

        def command(parts):
            frame = b"*%d\r\n" % len(parts)
            frame += b"".join(b"$%d\r\n%s\r\n" % (len(part), part) for part in parts)
            connection.sendall(frame)
            reply = reader.readline()
            if not reply.startswith((b"+", b":")):
                raise RuntimeError(f"Redis cleanup rejected: {reply!r}")

        command(
            [
                b"AUTH",
                params.get("redis_user", "default").encode(),
                params["password"].encode(),
            ]
        )
        batch = []
        for key in keys:
            for page in range(ROW // B.PAGE):
                batch.append(_page_digest(key, page).encode())
                if len(batch) == 1024:
                    command([b"DEL", *batch])
                    batch.clear()
        if batch:
            command([b"DEL", *batch])


def memory(cgroup):
    host = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return {
        "host_nonfree": host["MemTotal"] - host["MemFree"],
        "host_unavailable": host["MemTotal"] - host["MemAvailable"],
        **{
            name: int((cgroup / name).read_text())
            for name in ("memory.current", "memory.peak", "memory.swap.current")
        },
    }


def cold_pages(engine):
    region = engine.regions[0]
    vec = (ctypes.c_ubyte * (region.total_size_bytes // B.PAGE))()
    if B.LIBC.mincore(
        ctypes.c_void_p(region.base_tensor.data_ptr()),
        ctypes.c_size_t(region.total_size_bytes),
        vec,
    ):
        raise OSError(ctypes.get_errno(), "mincore")
    return (
        np.frombuffer(vec, dtype=np.uint8).reshape(engine.spec.num_chunks, -1) & 1
    ) == 0


def delta(before, after):
    return {
        name: after[name] - before[name]
        for name in (
            "read_bytes",
            "read_pages",
            "read_batches",
            "read_errors",
            "written_bytes",
            "written_pages",
            "written_errors",
            "written_no_digest",
        )
    }


def distribution(values):
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def run(args, mode):
    output = args.output / f"{mode}.json"
    original = Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip()
    parent = Path("/sys/fs/cgroup") / original.lstrip("/")
    cg = Path("/sys/fs/cgroup") / f"kv-arrivals-{mode}-{os.getpid()}"
    cg.mkdir()
    for name, value in (
        ("memory.max", 4 * GIB),
        ("memory.high", int(2.5 * GIB)),
        ("memory.swap.max", 8 * GIB),
        ("memory.zswap.max", 0),
    ):
        (cg / name).write_text(str(value))
    result = {
        "mode": mode,
        "kernel": os.uname().release,
        "cache_capacity_bytes": 8 * GIB,
        "row_bytes": ROW,
        "populated_rows": args.populated,
        "context_chunks": args.context_chunks,
        "cached_tokens": args.context_chunks * 16,
        "tail_chunks": 4,
        "arrival_rate": args.rate,
        "concurrency_limit": 8,
        "memory_limit_bytes": 4 * GIB,
        "requests": [],
        "stores": [],
        "samples": [],
        "reclaims": [],
        "errors": [],
        "service_timeout_s": args.service_timeout,
    }
    engine = service = None
    stop = threading.Event()
    unsafe = threading.Event()
    rpc = threading.Lock()
    active_lock = threading.Lock()
    active = 0
    peak_active = 0
    monitor = None
    remote_keys = []

    def watch():
        while not stop.is_set():
            sample = {
                "time": time.monotonic(),
                "memory": memory(cg),
                "backend": B.rswap_stats(),
            }
            result["samples"].append(sample)
            # Non-free includes reclaimable filesystem cache: use the stricter
            # interpretation of the user's 12 GiB whole-host limit.
            if sample["memory"]["host_nonfree"] > 11.5 * GIB:
                unsafe.set()
            stop.wait(0.2)

    def check():
        if unsafe.is_set():
            raise RuntimeError("11.5 GiB host RAM safety threshold reached")

    def reclaim(amount):
        before = B.rswap_stats()
        begin = time.monotonic()
        try:
            (cg / "memory.reclaim").write_text(str(amount))
        except OSError as error:
            if error.errno != errno.EAGAIN:
                raise
        result["reclaims"].append(
            {
                "seconds": time.monotonic() - begin,
                "backend": delta(before, B.rswap_stats()),
                "memory": memory(cg),
            }
        )
        check()

    try:
        (cg / "cgroup.procs").write_text(str(os.getpid()))
        B.RS_WORKLOAD.rswap = True
        extra = ["--eviction-policy", "none", "--pin-lease-bytes", "0"]
        if mode == "fault":
            extra += ["--prefetch-max-chunks", "0"]
        service = B.Service(8 * GIB, True, extra)
        engine = B.Engine(
            service.path,
            8 * GIB // ROW,
            ROW // 8,
            uuid.uuid4().hex,
            8,
            extra_config={"service_timeout": args.service_timeout},
        )
        engine.manager.prefetch = False
        engine.manager.prefetch_on_lookup = mode == "prefetch"
        token = uuid.uuid4().bytes
        result["key_prefix_hex"] = token.hex()
        keys = [
            OffloadKey(token + index.to_bytes(4, "big"))
            for index in range(args.populated)
        ]
        remote_keys.extend(keys)
        requests = [ReqContext(f"request-{i}") for i in range(args.requests)]
        groups = [
            list(range(i * args.context_chunks, (i + 1) * args.context_chunks))
            for i in range(args.requests)
        ]
        buffer = B.DeviceBuffer(ROW // 8)
        monitor = threading.Thread(target=watch)
        monitor.start()
        fill_before = B.rswap_stats()
        fill_begin = time.monotonic()
        for first in range(0, args.populated, 8):
            check()
            if memory(cg)["memory.current"] > 2.25 * GIB:
                reclaim(256 * 2**20)
            wanted = keys[first : first + 8]
            owner = ReqContext(f"fill-{first}")
            begin = time.monotonic()
            plan = engine.manager.prepare_store(wanted, owner)
            if plan is None or len(plan.keys_to_store) != len(wanted):
                raise RuntimeError("fill store rejected")
            for index, slot in zip(
                range(first, first + len(wanted)), plan.store_spec.block_ids
            ):
                buffer.buffer[:] = index % 127 + 1
                for view in engine.views:
                    buffer.write_into(view, int(slot))
            engine.manager.complete_store(wanted, owner)
            engine.manager.on_request_finished(owner)
            engine.manager.on_schedule_end(None)
            # Acquire an independent real lease while each row is still warm.
            # Leases are unpinned, allowing reclaim before its scheduled load.
            for i, group in enumerate(groups):
                leased = [keys[j] for j in group if first <= j < first + len(wanted)]
                if leased and not all(
                    status is LookupResult.HIT
                    for status in engine.manager.lookup_many(leased, requests[i])
                ):
                    raise RuntimeError("warm lease acquisition failed")
            engine.manager.client.call("spill", keys=[key.hex() for key in wanted])
            result["stores"].append(time.monotonic() - begin)
            if (first + len(wanted)) % 128 == 0:
                print(
                    "FILL",
                    mode,
                    first + len(wanted),
                    json.dumps(memory(cg)),
                    flush=True,
                )
        result["fill_seconds"] = time.monotonic() - fill_begin
        result["fill_backend"] = delta(fill_before, B.rswap_stats())
        # Python/copy bookkeeping remains outside the KV cgroup during loads.
        parent.joinpath("cgroup.procs").write_text(str(os.getpid()))
        result["memory_high_bytes"] = int(2.5 * GIB)
        reclaim(512 * 2**20)
        counts = cold_pages(engine).sum(axis=1)
        result["initial_cold_bytes"] = int(counts[: args.populated].sum()) * B.PAGE
        result["selected_cold_bytes"] = (
            int(counts[: args.requests * args.context_chunks].sum()) * B.PAGE
        )
        print("READY", mode, result["selected_cold_bytes"], flush=True)
        phase_before = B.rswap_stats()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        epoch = time.monotonic()

        def job(index, due):
            nonlocal active, peak_active
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
            begin = time.monotonic()
            req = requests[index]
            indices = groups[index]
            wanted = [keys[i] for i in indices]
            local = B.DeviceBuffer(ROW // 8)
            polls = 0
            try:
                check()
                cold = int(cold_pages(engine)[indices].sum()) * B.PAGE
                if mode == "prefetch":
                    while True:
                        check()
                        with rpc:
                            statuses = engine.manager.lookup_many(wanted, req)
                            engine.manager.on_schedule_end(None)
                        if all(status is LookupResult.HIT for status in statuses):
                            break
                        if LookupResult.MISS in statuses:
                            raise RuntimeError("KV key lost during prefetch")
                        if time.monotonic() - begin > 300:
                            raise RuntimeError("prefetch timeout")
                        polls += 1
                        time.sleep(0.02)
                ready = time.monotonic()
                with rpc:
                    spec = engine.manager.prepare_load(wanted, req)
                cold_at_copy = int(cold_pages(engine)[indices].sum()) * B.PAGE
                copy_begin = time.monotonic()
                for key_index, slot in zip(indices, spec.block_ids):
                    check()
                    for view in engine.views:
                        local.read_from(view, int(slot))
                        if not bool((local.buffer == key_index % 127 + 1).all()):
                            raise RuntimeError("load bytes differ")
                loaded = time.monotonic()
                with rpc:
                    engine.manager.complete_load(wanted, req)
                tail = [
                    OffloadKey(token + b"tail" + index.to_bytes(4, "big") + bytes([i]))
                    for i in range(4)
                ]
                remote_keys.extend(tail)
                store_begin = time.monotonic()
                with rpc:
                    plan = engine.manager.prepare_store(tail, req)
                if plan is None or len(plan.keys_to_store) != 4:
                    raise RuntimeError("tail store rejected")
                local.buffer[:] = index % 127 + 1
                for slot in plan.store_spec.block_ids:
                    for view in engine.views:
                        local.write_into(view, int(slot))
                with rpc:
                    engine.manager.complete_store(tail, req)
                    engine.manager.on_request_finished(req)
                    engine.manager.on_schedule_end(None)
                end = time.monotonic()
                event = {
                    "index": index,
                    "arrival_s": due - epoch,
                    "completion_s": end - epoch,
                    "queue_s": begin - due,
                    "ttft_proxy_s": loaded - due,
                    "latency_s": end - due,
                    "service_s": end - begin,
                    "wait_s": ready - begin,
                    "copy_verify_s": loaded - copy_begin,
                    "store_s": end - store_begin,
                    "cold_before_bytes": cold,
                    "cold_at_copy_bytes": cold_at_copy,
                    "requested_bytes": len(wanted) * ROW,
                    "polls": polls,
                }
                result["requests"].append(event)
                print("REQUEST", mode, json.dumps(event), flush=True)
            except Exception as error:
                result["errors"].append({"request": index, "error": repr(error)})
                raise
            finally:
                with active_lock:
                    active -= 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for index in range(args.requests):
                check()
                due = epoch + index / args.rate
                time.sleep(max(0, due - time.monotonic()))
                futures.append(pool.submit(job, index, due))
            for future in futures:
                future.result()
        elapsed = time.monotonic() - epoch
        result["phase_seconds"] = elapsed
        result["backend"] = delta(phase_before, B.rswap_stats())
        result["peak_active_requests"] = peak_active
        result["cpu_seconds"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime
            - usage.ru_utime
            - usage.ru_stime
        )
        result["summary"] = {
            name: distribution([event[name] for event in result["requests"]])
            for name in (
                "ttft_proxy_s",
                "latency_s",
                "queue_s",
                "store_s",
                "copy_verify_s",
            )
        }
        result["summary"].update(
            requests_per_second=len(result["requests"]) / elapsed,
            remote_read_MB_per_second=result["backend"]["read_bytes"] / elapsed / 1e6,
            logical_load_MiB_per_second=args.requests
            * args.context_chunks
            * ROW
            / elapsed
            / 2**20,
        )
        result["service_stats"] = engine.manager.client.call("stats")["stats"]
        result["memory_events"] = (cg / "memory.events").read_text()
        result["store_summary_s"] = distribution(result["stores"])
        if result["backend"]["read_errors"] or result["service_stats"]["lost_chunks"]:
            raise RuntimeError("unexpected remote data loss")
        result["success"] = True
        print("DONE", mode, json.dumps(result["summary"]), flush=True)
    except Exception as error:
        result["success"] = False
        result["error"] = repr(error)
        print("ERROR", mode, repr(error), flush=True)
    finally:
        stop.set()
        if monitor:
            monitor.join()
        output.write_text(json.dumps(result, indent=2))
        try:
            if engine:
                view = None
                engine.views.clear()
                gc.collect()
                # This namespace is test-owned and all jobs have drained.
                # Discard backing storage before device teardown even if an
                # exported array still retains a mapping.
                region = engine.regions[0]
                rc = B.LIBC.madvise(
                    ctypes.c_void_p(region.base_tensor.data_ptr()),
                    ctypes.c_size_t(region.total_size_bytes),
                    ctypes.c_int(9),  # MADV_REMOVE punches out shared storage.
                )
                if rc:
                    raise OSError(ctypes.get_errno(), "discard test KV mapping")
                engine.close()
        finally:
            if service:
                try:
                    service.close()
                except Exception as error:
                    result["shutdown_error"] = repr(error)
                    service.process.kill()
                    service.process.wait(timeout=30)
                    service.dir.cleanup()
            parent.joinpath("cgroup.procs").write_text(str(os.getpid()))
            cg.rmdir()
        try:
            delete_remote(remote_keys)
            result["remote_cleanup"] = True
        except Exception as error:
            result["remote_cleanup"] = False
            result["cleanup_error"] = repr(error)
            result["success"] = False
        output.write_text(json.dumps(result, indent=2))
    return result["success"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--populated", type=int, default=1536)
    parser.add_argument("--context-chunks", type=int, default=40)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rate", type=float, default=2)
    parser.add_argument("--service-timeout", type=float, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("fault", "prefetch"),
        default=["fault", "prefetch"],
    )
    args = parser.parse_args()
    if (
        args.requests <= 0
        or args.context_chunks <= 0
        or args.rate <= 0
        or args.requests * args.context_chunks > args.populated
        or args.populated + 4 * args.requests > 8 * GIB // ROW
    ):
        parser.error(
            "prefix/tail rows must fit the cache; rate and sizes must be positive"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    device = B.RswapDevice()
    try:
        results = []
        for mode in args.modes:
            success = run(args, mode)
            results.append(success)
            if not success:
                break
        return 0 if all(results) else 1
    finally:
        device.close()


if __name__ == "__main__":
    raise SystemExit(main())
