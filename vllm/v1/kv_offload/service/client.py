# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded JSON control protocol and shared-memory descriptor handshake."""

import array
import json
import os
import socket
import struct
from typing import Any

MAX_FRAME = 1024 * 1024
MAX_KEYS = 4096


class CacheServiceClient:
    def __init__(self, path: str, timeout: float = 5.0):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        try:
            self.socket.connect(path)
        except Exception:
            self.close()
            raise

    def _read_exact(self, length: int) -> bytes:
        payload = bytearray()
        while len(payload) < length:
            part = self.socket.recv(length - len(payload))
            if not part:
                raise ConnectionError("KV cache service disconnected")
            payload.extend(part)
        return bytes(payload)

    def call(self, op: str, **fields: Any) -> dict[str, Any]:
        payload = json.dumps({"op": op, **fields}, separators=(",", ":")).encode()
        if len(payload) > MAX_FRAME:
            raise ValueError("KV cache service request exceeds frame limit")
        try:
            self.socket.sendall(struct.pack("!I", len(payload)) + payload)
            length = struct.unpack("!I", self._read_exact(4))[0]
            if not 0 < length <= MAX_FRAME:
                raise ConnectionError("Invalid KV cache service response length")
            response = json.loads(self._read_exact(length))
            if not isinstance(response, dict) or response.get("ok") is not True:
                raise RuntimeError(f"KV cache service rejected operation: {response}")
            return response
        except Exception:
            # An incomplete frame cannot be retried on the same stream.
            self.close()
            raise

    def receive_fd(self) -> int:
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
            if marker != b"F" or flags & socket.MSG_CTRUNC or len(descriptors) != 1:
                raise ConnectionError("Invalid shared-memory descriptor handshake")
            return descriptors[0]
        except Exception:
            for fd in descriptors:
                os.close(fd)
            self.close()
            raise

    def close(self) -> None:
        self.socket.close()
