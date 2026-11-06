# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded JSON control protocol and shared-memory descriptor handshake."""

import array
import json
import os
import socket
import struct
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

MAX_FRAME = 1024 * 1024
MAX_KEYS = 4096

# A starting engine races the service's socket, so the connect retries with
# exponential backoff instead of failing on the first refusal.
_CONNECT_ATTEMPTS = 10
_CONNECT_BACKOFF_S = 0.05
_CONNECT_BACKOFF_MAX_S = 1.0

# The handshake cannot share a burst: an `attach` reply is a frame followed by a
# descriptor message that must be received with its marker byte, and `close`
# ends the session the rest of the burst would still use.
_HANDSHAKE_OPS = frozenset({"open", "attach", "close"})

# Operations a burst may repeat after a transport failure: each answers from the
# cache without changing what a store publishes.
_READ_ONLY_OPS = frozenset({"lookup", "lookup_many", "load", "stats", "info"})


class CacheServiceError(RuntimeError):
    """The service rejected a request or answered outside the protocol."""


class CacheServiceUnavailable(CacheServiceError):
    """A transport failure closed the connection this client was using."""


class CacheServiceClient:
    """One bounded control connection to the cache service.

    A request is one frame: a big-endian length prefix and its JSON payload,
    written in a single call. Any transport failure closes the socket, so a
    read-only call may reopen it once while a mutating one may not, because the
    outcome of a lost reply is unknown. The `open` handshake is remembered and
    replayed on that reconnect, because the service keeps the namespace of an
    owner that disconnected: the new session adopts the same mapping, entries,
    and leases, so the retried call reads the cache it wrote.

    The service answers one connection's requests in order and replies carry no
    request id, so a burst is sent before any of its replies is read: `submit`
    writes one more request, `collect` reads its replies, and `pipeline` sends a
    whole burst and reads all of it. A burst is one round trip instead of one per
    request, and the socket stays synchronised with the replies as long as no
    other request is sent before they are collected.
    """

    def __init__(
        self,
        path: str,
        op_timeout: float = 5.0,
        connect_timeout: float = 5.0,
    ):
        self.path = path
        self.op_timeout = op_timeout
        self.connect_timeout = connect_timeout
        # The fields of the namespace this client opened, if it opened one.
        self._open: dict[str, Any] | None = None
        # One operation name per reply the service still owes, in request
        # order, so a rejected reply can be named and a desynchronised stream
        # refused.
        self._pending: deque[str] = deque()
        self.socket = self._connect()

    def _connect(self) -> socket.socket:
        delay = _CONNECT_BACKOFF_S
        error: OSError | None = None
        for attempt in range(_CONNECT_ATTEMPTS):
            if attempt:
                time.sleep(delay)
                delay = min(delay * 2, _CONNECT_BACKOFF_MAX_S)
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(self.connect_timeout)
            try:
                client.connect(self.path)
            except OSError as exc:
                error = exc
                client.close()
                continue
            client.settimeout(self.op_timeout)
            return client
        raise CacheServiceUnavailable(
            f"cannot connect to the KV cache service at {self.path!r} after "
            f"{_CONNECT_ATTEMPTS} attempts: {error}"
        )

    def reconnect(self) -> None:
        """Replaces the closed socket with a new connection.

        The namespace this client opened is reopened on it, which the service
        answers by adopting the mapping its previous owner left behind. The
        replay may only adopt: a namespace the service dropped is not quietly
        recreated while the workers still map it.

        Raises:
            RuntimeError: replies of a burst are unread, so reconnecting would
                lose them.

        """
        self._check_idle("reconnect")
        self.socket = self._connect()
        if self._open is not None:
            self._send("open", {**self._open, "adopt": True})

    def _read_exact(self, length: int) -> bytes:
        payload = bytearray()
        while len(payload) < length:
            part = self.socket.recv(length - len(payload))
            if not part:
                raise ConnectionError("KV cache service disconnected")
            payload.extend(part)
        return bytes(payload)

    def call(self, op: str, retry: bool = False, **fields: Any) -> dict[str, Any]:
        """Sends one request and returns the accepted reply.

        Args:
            op: wire operation name.
            retry: reopen the connection and repeat the call once after a
                transport failure. Only a read-only operation may set it.
            fields: request fields, sent next to the operation name.

        Returns:
            The reply, whose `ok` field has already been checked.

        Raises:
            CacheServiceUnavailable: a transport failure closed the connection.
            CacheServiceError: the service rejected the request or answered
                outside the protocol.
            RuntimeError: replies of an earlier burst are unread, so this one's
                reply could not be told apart from them.

        """
        self._check_idle("call")
        try:
            return self._send(op, fields)
        except CacheServiceUnavailable:
            if not retry:
                raise
        self.reconnect()
        return self._send(op, fields)

    def submit(self, op: str, **fields: Any) -> None:
        """Writes one request without waiting for its reply.

        Args:
            op: wire operation name. The handshake is refused, because it cannot
                share a stream with a burst.
            fields: request fields, sent next to the operation name.

        Raises:
            ValueError: a handshake operation, or a request above the frame
                limit. Nothing is written in either case.
            CacheServiceUnavailable: a transport failure closed the connection.

        """
        if op in _HANDSHAKE_OPS:
            self._refuse_handshake(op)
        self._write(op, self._encode(op, fields))
        self._pending.append(op)

    def collect(self, count: int) -> list[dict[str, Any]]:
        """Reads `count` replies of the requests submitted before it.

        Args:
            count: number of replies to read, at most the number of requests
                written and not collected yet.

        Returns:
            The accepted replies, in request order. A rejected reply is not in
            the list.

        Raises:
            ValueError: `count` is negative or above the replies this client is
                waiting for.
            CacheServiceUnavailable: a transport failure closed the connection.
            CacheServiceError: the service rejected one of the replies. The rest
                of the burst is read first, so the connection stays usable and
                its later replies are not taken for these. The message names the
                first rejected operation, which is not the only request that may
                have been applied.

        """
        if not 0 <= count <= len(self._pending):
            raise ValueError(
                f"cannot collect {count} KV cache service replies: "
                f"{len(self._pending)} outstanding"
            )
        return self._take(count)

    def pipeline(
        self,
        calls: Sequence[tuple[str, Mapping[str, Any]]],
        retry: bool = False,
    ) -> list[dict[str, Any]]:
        """Writes every call, then reads one reply per call.

        The whole burst is encoded before the first frame is written, so a call
        above the frame limit writes nothing and leaves the connection usable.

        Args:
            calls: `(op, fields)` pairs, in the order the service must apply
                them. A handshake operation is refused.
            retry: reopen the connection and write the whole burst again after a
                write failure. Only a burst of read-only operations may set it,
                and only a burst that was never answered is repeated: a failure
                while reading replies is never retried, because some of them may
                have been applied already.

        Returns:
            The accepted replies, in call order.

        Raises:
            ValueError: a call is a handshake operation, `retry` was set for a
                burst that is not read-only, or a frame is above the frame
                limit. Nothing is written in any of these cases.
            CacheServiceUnavailable: a transport failure closed the connection.
            CacheServiceError: the service rejected one of the replies, after the
                rest of the burst was read (see `collect`).
            RuntimeError: replies of an earlier burst are unread.

        """
        ops = [op for op, _ in calls]
        for op in ops:
            if op in _HANDSHAKE_OPS:
                self._refuse_handshake(op)
        if retry and not _READ_ONLY_OPS.issuperset(ops):
            raise ValueError(
                "only a read-only burst may be retried: "
                + ", ".join(sorted(set(ops) - _READ_ONLY_OPS))
            )
        self._check_idle("pipeline")
        frames = [self._encode(op, fields) for op, fields in calls]
        try:
            # One write for the whole burst: the shorter the write, the fewer
            # thread handoffs it costs.
            self._write(f"burst of {len(frames)}", b"".join(frames))
        except CacheServiceUnavailable:
            if not retry:
                raise
            self.reconnect()
            return self.pipeline(calls)
        self._pending.extend(ops)
        return self._take(len(ops))

    def _send(self, op: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        self._write(op, self._encode(op, fields))
        self._pending.append(op)
        reply = self._take(1)[0]
        if op == "open":
            self._open = dict(fields)
        return reply

    def _take(self, count: int) -> list[dict[str, Any]]:
        replies: list[dict[str, Any]] = []
        rejection: CacheServiceError | None = None
        for _ in range(count):
            op = self._pending.popleft()
            reply = self._read(op)
            if reply.get("ok") is not True:
                # A rejected request leaves the connection and the namespace
                # usable, so the rest of the burst is still read.
                if rejection is None:
                    rejection = CacheServiceError(
                        f"KV cache service rejected {op}: {reply.get('error', reply)}"
                    )
                continue
            replies.append(reply)
        if rejection is not None:
            raise rejection
        return replies

    @staticmethod
    def _encode(op: str, fields: Mapping[str, Any]) -> bytes:
        """Returns the frame the service reads for one request."""
        payload = json.dumps({"op": op, **fields}, separators=(",", ":")).encode()
        if len(payload) > MAX_FRAME:
            raise ValueError("KV cache service request exceeds frame limit")
        return struct.pack("!I", len(payload)) + payload

    def _write(self, what: str, frame: bytes) -> None:
        try:
            # One write per frame or burst: the service reads the prefix and the
            # payload from the same buffer.
            self.socket.sendall(frame)
        except OSError as error:
            # An incomplete frame cannot be retried on the same stream.
            self.close()
            raise CacheServiceUnavailable(
                f"KV cache service {what} failed: {error}"
            ) from error

    def _read(self, op: str) -> dict[str, Any]:
        """Reads one reply frame, whatever its `ok` field says."""
        try:
            length = struct.unpack("!I", self._read_exact(4))[0]
            if not 0 < length <= MAX_FRAME:
                raise ConnectionError("Invalid KV cache service response length")
            reply = self._read_exact(length)
        except (OSError, ValueError) as error:
            self.close()
            raise CacheServiceUnavailable(
                f"KV cache service {op} failed: {error}"
            ) from error
        try:
            response = json.loads(reply)
        except ValueError as error:
            self.close()
            raise CacheServiceError(
                f"KV cache service {op} answered outside the protocol: {error}"
            ) from error
        if not isinstance(response, dict):
            self.close()
            raise CacheServiceError(
                f"KV cache service {op} answered outside the protocol: {reply!r}"
            )
        return response

    def _check_idle(self, what: str) -> None:
        if self._pending:
            raise RuntimeError(
                f"{what} with {len(self._pending)} unread KV cache service "
                "replies; collect them first"
            )

    @staticmethod
    def _refuse_handshake(op: str) -> None:
        raise ValueError(
            f"KV cache service {op} is a handshake and cannot be pipelined"
        )

    def receive_fd(self) -> int:
        """Receives the single descriptor of an attach reply."""
        descriptors = array.array("i")
        try:
            marker, ancillary, flags, _ = self.socket.recvmsg(
                1, socket.CMSG_SPACE(descriptors.itemsize), socket.MSG_CMSG_CLOEXEC
            )
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    descriptors.frombytes(
                        data[: len(data) - len(data) % descriptors.itemsize]
                    )
            if marker != b"F" or flags & socket.MSG_CTRUNC:
                raise ConnectionError("Invalid shared-memory descriptor handshake")
            if len(descriptors) != 1:
                raise ConnectionError(
                    f"KV cache service sent {len(descriptors)} descriptors, not one"
                )
        except (OSError, ValueError) as error:
            # A descriptor this call does not return is closed, never leaked.
            for fd in descriptors:
                os.close(fd)
            self.close()
            raise CacheServiceUnavailable(
                f"KV cache service attach failed: {error}"
            ) from error
        return descriptors[0]

    def close(self) -> None:
        """Closes the connection, abandoning the replies it still owed."""
        self.socket.close()
        self._pending.clear()
