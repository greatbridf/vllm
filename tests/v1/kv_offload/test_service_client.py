# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Burst semantics of the cache service client, on a fake and the real service."""

import contextlib
import json
import mmap
import os
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from vllm.v1.kv_offload.service.client import (
    MAX_FRAME,
    CacheServiceClient,
    CacheServiceError,
    CacheServiceUnavailable,
)

PAGE = mmap.PAGESIZE
BINARY = Path(
    os.environ.get(
        "VLLM_KV_CACHE_SERVICE_BINARY", "rust/target/debug/vllm-kv-cache-service"
    )
).resolve()

# A client that waits for a reply its peer never sends fails the test instead
# of hanging it.
OP_TIMEOUT_S = 2.0
_ACCEPT_POLL_S = 0.05

# A request past the script: answer it with an accepted reply, unlike a scripted
# `None`, which drops the connection.
_ACCEPTED = {"ok": True}


def _read_exact(connection: socket.socket, length: int) -> bytes | None:
    """Reads one field of a frame, or returns None when the peer closed."""
    payload = bytearray()
    while len(payload) < length:
        part = connection.recv(length - len(payload))
        if not part:
            return None
        payload.extend(part)
    return bytes(payload)


def _send(connection: socket.socket, answer: Any) -> bool:
    """Writes one scripted answer; returns False when the connection ends."""
    if answer is None:
        return False
    if isinstance(answer, bytes):
        connection.sendall(answer)
        return False
    payload = json.dumps(answer).encode()
    connection.sendall(struct.pack("!I", len(payload)) + payload)
    return True


class FakeService:
    """A hand-threaded peer that answers frames with scripted replies.

    Args:
        path: socket path to listen on, so the client connects for real.
        answers: one entry per request, in order: a reply object, raw bytes to
            write before closing the connection, or `None` to drop it. Every
            further request is answered `{"ok": True}`.
        hold: number of requests to receive before answering any of them, so a
            client that reads while it writes blocks instead. Only the first
            burst is held.

    """

    def __init__(
        self,
        path: str,
        answers: list[Any] | None = None,
        hold: int = 0,
    ):
        self.path = path
        self.received: list[dict[str, Any]] = []
        self.connections = 0
        self._answers = list(answers or [])
        self._hold = hold
        self._flushed = not hold
        self._stopping = threading.Event()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(path)
        self._listener.listen(4)
        self._listener.settimeout(_ACCEPT_POLL_S)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stopping.set()
        self._listener.close()
        self._thread.join(timeout=OP_TIMEOUT_S)

    def _serve(self) -> None:
        while not self._stopping.is_set():
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            connection.settimeout(OP_TIMEOUT_S)
            with connection:
                try:
                    self._answer_requests(connection)
                except (OSError, ValueError):
                    # The client closed the connection under us, or sent
                    # something that is not a frame.
                    continue

    def _answer_requests(self, connection: socket.socket) -> None:
        queued: list[Any] = []
        while True:
            prefix = _read_exact(connection, 4)
            if prefix is None:
                return
            payload = _read_exact(connection, struct.unpack("!I", prefix)[0])
            if payload is None:
                return
            self.received.append(json.loads(payload))
            index = len(self.received) - 1
            scripted = index < len(self._answers)
            queued.append(self._answers[index] if scripted else _ACCEPTED)
            if not self._flushed and len(queued) < self._hold:
                continue
            self._flushed = True
            for answer in queued:
                if not _send(connection, answer):
                    return
            queued = []


@pytest.fixture
def serve(tmp_path):
    """Starts fake services on `tmp_path` and stops them after the test."""
    services: list[FakeService] = []

    def start(answers: list[Any] | None = None, hold: int = 0) -> FakeService:
        service = FakeService(
            str(tmp_path / f"fake{len(services)}.sock"), answers, hold
        )
        services.append(service)
        return service

    yield start
    for service in services:
        service.close()


def test_pipeline_returns_the_replies_of_a_burst_in_call_order(serve):
    """The burst is written before any reply is read, and stays in order."""
    statuses = ["hit", "miss", "pending"]
    fake = serve([{"ok": True, "status": status} for status in statuses], hold=3)
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        replies = client.pipeline(
            [("lookup", {"key": key, "request": "r"}) for key in ("a", "b", "c")]
        )
    finally:
        client.close()

    assert [reply["status"] for reply in replies] == statuses
    assert [request["key"] for request in fake.received] == ["a", "b", "c"]


def test_call_refuses_to_interleave_with_an_unread_burst(serve):
    """Replies carry no id, so a second request on the stream is a bug."""
    fake = serve(hold=2)
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        client.submit("lookup", key="a", request="r")
        with pytest.raises(RuntimeError, match="collect them first"):
            client.call("info")
        with pytest.raises(RuntimeError, match="collect them first"):
            client.pipeline([("lookup", {"key": "b", "request": "r"})])
        client.submit("lookup", key="b", request="r")
        assert client.collect(2) == [{"ok": True}, {"ok": True}]
        assert client.call("info") == {"ok": True}
    finally:
        client.close()


def test_collect_refuses_more_replies_than_were_written(serve):
    """A count the burst cannot answer would take the next reply for it."""
    fake = serve(hold=1)
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        client.submit("lookup", key="a", request="r")
        with pytest.raises(ValueError, match="1 outstanding"):
            client.collect(2)
        assert client.collect(1) == [{"ok": True}]
        with pytest.raises(ValueError, match="0 outstanding"):
            client.collect(1)
    finally:
        client.close()


def test_collect_raises_a_rejection_after_reading_the_rest_of_the_burst(serve):
    """A rejected reply keeps the connection in sync and names its operation."""
    fake = serve(
        [
            {"ok": True, "status": "hit"},
            {"ok": False, "error": "commit requires pending reservations"},
            {"ok": True, "status": "miss"},
        ]
    )
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        client.submit("lookup", key="a", request="r")
        client.submit("commit", keys=["b"], request="r", success=True)
        client.submit("lookup", key="c", request="r")
        with pytest.raises(CacheServiceError, match="rejected commit: commit requires"):
            client.collect(3)
        assert client.call("info") == {"ok": True}
    finally:
        client.close()

    assert [request["op"] for request in fake.received] == [
        "lookup",
        "commit",
        "lookup",
        "info",
    ]


def test_pipeline_raises_the_first_rejection_after_draining_the_burst(serve):
    fake = serve(
        [
            {"ok": True},
            {"ok": False, "error": "no such request"},
            {"ok": False, "error": "no such lease"},
        ]
    )
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        with pytest.raises(CacheServiceError, match="rejected commit: no such request"):
            client.pipeline(
                [
                    ("commit", {"keys": ["a"], "request": "r", "success": True}),
                    ("commit", {"keys": ["b"], "request": "r", "success": True}),
                    ("release_request", {"request": "r"}),
                ]
            )
        assert client.call("stats") == {"ok": True}
    finally:
        client.close()

    # Both rejections and the accepted reply were read.
    assert [request["op"] for request in fake.received] == [
        "commit",
        "commit",
        "release_request",
        "stats",
    ]


def test_a_truncated_reply_closes_the_socket(serve):
    fake = serve([struct.pack("!I", 64) + b"{"])
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        client.submit("lookup", key="a", request="r")
        with pytest.raises(CacheServiceUnavailable, match="lookup failed"):
            client.collect(1)
        assert client.socket.fileno() == -1
    finally:
        client.close()


def test_a_mid_burst_failure_closes_the_socket(serve):
    """A burst that breaks off part way leaves no reply owed on a dead stream."""
    fake = serve([{"ok": True}, struct.pack("!I", 64) + b"{"])
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        with pytest.raises(CacheServiceUnavailable, match="commit failed"):
            client.pipeline(
                [
                    ("commit", {"keys": ["a"], "request": "r", "success": True}),
                    ("commit", {"keys": ["b"], "request": "r", "success": True}),
                    ("release_request", {"request": "r"}),
                ]
            )
        assert client.socket.fileno() == -1
        # The replies died with the socket, so the next call is not refused as
        # interleaving with an unread burst: it reconnects and is answered.
        assert client.call("stats", retry=True) == {"ok": True}
    finally:
        client.close()

    # The peer read the first two frames and broke off writing the second reply,
    # so the rest of the burst went with the connection.
    assert [request["op"] for request in fake.received] == [
        "commit",
        "commit",
        "stats",
    ]


def test_an_oversized_call_in_a_burst_writes_nothing(serve):
    """Both forms encode the whole burst before the first frame goes out."""
    fake = serve()
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    oversized = "x" * MAX_FRAME
    try:
        with pytest.raises(ValueError, match="exceeds frame limit"):
            client.pipeline(
                [
                    ("lookup", {"key": "a", "request": "r"}),
                    ("store", {"keys": [oversized], "request": "r"}),
                ]
            )
        with pytest.raises(ValueError, match="exceeds frame limit"):
            client.submit("store", keys=[oversized], request="r")
        assert client.call("info") == {"ok": True}
    finally:
        client.close()

    # The next request arrived first, so nothing of the bursts was written.
    assert [request["op"] for request in fake.received] == ["info"]


@pytest.mark.parametrize("op", ["open", "attach", "close"])
def test_a_handshake_cannot_join_a_burst(serve, op):
    """An `attach` reply carries a descriptor that must arrive with its marker."""
    fake = serve()
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        with pytest.raises(ValueError, match="cannot be pipelined"):
            client.submit(op)
        with pytest.raises(ValueError, match="cannot be pipelined"):
            client.pipeline([("lookup", {"key": "a", "request": "r"}), (op, {})])
        assert client.call("info") == {"ok": True}
    finally:
        client.close()

    assert [request["op"] for request in fake.received] == ["info"]


def test_a_mutating_burst_may_not_be_retried(serve):
    """Replaying a published or released batch is not idempotent."""
    fake = serve()
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        with pytest.raises(ValueError, match="read-only burst"):
            client.pipeline(
                [("commit", {"keys": ["a"], "request": "r", "success": True})],
                retry=True,
            )
        assert client.call("info") == {"ok": True}
    finally:
        client.close()

    assert [request["op"] for request in fake.received] == ["info"]


def test_a_failed_reply_is_not_retried_even_for_a_read_only_burst(serve):
    """Only a burst that was never answered is repeated."""
    fake = serve([None])
    client = CacheServiceClient(fake.path, op_timeout=OP_TIMEOUT_S)
    try:
        with pytest.raises(CacheServiceUnavailable):
            client.pipeline([("lookup", {"key": "a", "request": "r"})], retry=True)
    finally:
        client.close()

    assert fake.connections == 1


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


def test_a_burst_of_lookups_answers_one_ordered_reply_per_request(service):
    """A real service answers 64 pipelined lookups in request order."""
    client = CacheServiceClient(service)
    keys = [f"{index:02x}" * 32 for index in range(64)]
    stored = keys[0::2]
    request = "pipelined"
    try:
        client.call(
            "open",
            version=1,
            namespace="pipelined",
            fingerprint="a" * 64,
            worker_bytes=64,
            world_size=1,
            num_chunks=64,
        )
        client.call("store", keys=stored, request=request)
        client.call("commit", keys=stored, request=request, success=True)
        replies = client.pipeline(
            [("lookup", {"key": key, "request": request}) for key in keys]
        )
        stats = client.call("stats")["stats"]
    finally:
        with contextlib.suppress(CacheServiceError):
            client.call("close")
        client.close()

    assert len(replies) == 64
    assert [reply["status"] for reply in replies] == [
        "hit" if index % 2 == 0 else "miss" for index in range(64)
    ]
    assert stats["leased"] == 32
