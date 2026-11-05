# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded JSON control protocol and shared-memory descriptor handshake."""

import array
import json
import os
import socket
import struct
import time
from typing import Any

MAX_FRAME = 1024 * 1024
MAX_KEYS = 4096

# A starting engine races the service's socket, so the connect retries with
# exponential backoff instead of failing on the first refusal.
_CONNECT_ATTEMPTS = 10
_CONNECT_BACKOFF_S = 0.05
_CONNECT_BACKOFF_MAX_S = 1.0


class CacheServiceError(RuntimeError):
    """The service rejected a request or answered outside the protocol."""


class CacheServiceUnavailable(CacheServiceError):
    """A transport failure closed the connection this client was using."""


class CacheServiceClient:
    """One bounded control connection to the cache service.

    A request is one frame: a big-endian length prefix and its JSON payload,
    written in a single call. Any transport failure closes the socket, so a
    read-only call may reopen it once while a mutating one may not, because the
    outcome of a lost reply is unknown.
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
        """Replaces the closed socket with a new connection."""
        self.socket = self._connect()

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

        """
        try:
            return self._send(op, fields)
        except CacheServiceUnavailable:
            if not retry:
                raise
        self.reconnect()
        return self._send(op, fields)

    def _send(self, op: str, fields: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"op": op, **fields}, separators=(",", ":")).encode()
        if len(payload) > MAX_FRAME:
            raise ValueError("KV cache service request exceeds frame limit")
        try:
            # One write per frame: the service reads the prefix and the payload
            # from the same buffer.
            self.socket.sendall(struct.pack("!I", len(payload)) + payload)
            length = struct.unpack("!I", self._read_exact(4))[0]
            if not 0 < length <= MAX_FRAME:
                raise ConnectionError("Invalid KV cache service response length")
            reply = self._read_exact(length)
        except (OSError, ValueError) as error:
            # An incomplete frame cannot be retried on the same stream.
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
        if response.get("ok") is not True:
            # A rejected request leaves the connection and the namespace usable.
            raise CacheServiceError(
                f"KV cache service rejected {op}: {response.get('error', response)}"
            )
        return response

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
        self.socket.close()
