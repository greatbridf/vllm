# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler adapter for Rust-owned cache keys and reservations."""

from collections import defaultdict
from collections.abc import Collection

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.service.client import MAX_KEYS, CacheServiceClient


class ServiceOffloadingManager(OffloadingManager):
    def __init__(self, client: CacheServiceClient):
        self.client = client
        self._loads: dict[str, int] = defaultdict(int)
        self._stores: dict[str, int] = defaultdict(int)
        self._finished: set[str] = set()

    @staticmethod
    def _keys(keys: Collection[OffloadKey]) -> list[str]:
        if len(keys) > MAX_KEYS:
            raise ValueError("KV cache service operation exceeds key limit")
        return [key.hex() for key in keys]

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        status = self.client.call("lookup", key=key.hex(), request=req_context.req_id)[
            "status"
        ]
        return {
            "hit": LookupResult.HIT,
            "pending": LookupResult.HIT_PENDING,
            "miss": LookupResult.MISS,
        }[status]

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> CPULoadStoreSpec:
        slots = self.client.call(
            "load", keys=self._keys(keys), request=req_context.req_id
        )["slots"]
        self._loads[req_context.req_id] += 1
        return CPULoadStoreSpec(slots)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        # Keep lookup leases until the request has no remaining load jobs.
        self._loads[req_context.req_id] -= 1
        self._maybe_release(req_context.req_id)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        response = self.client.call("store", keys=self._keys(keys))
        if response.get("full"):
            return None
        new_keys = [OffloadKey(bytes.fromhex(key)) for key in response["keys"]]
        if new_keys:
            self._stores[req_context.req_id] += 1
        return PrepareStoreOutput(
            new_keys,
            CPULoadStoreSpec(response["slots"]),
            [OffloadKey(bytes.fromhex(key)) for key in response["evicted"]],
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        self.client.call("commit", keys=self._keys(keys), success=success)
        self._stores[req_context.req_id] -= 1
        self._maybe_release(req_context.req_id)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        encoded = self._keys(keys)
        self.client.call("touch", keys=encoded)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._finished.add(req_context.req_id)
        self._maybe_release(req_context.req_id)

    def _maybe_release(self, request: str) -> None:
        if (
            request in self._finished
            and self._loads[request] == 0
            and self._stores[request] == 0
        ):
            self.client.call("release_request", request=request)
            self._finished.remove(request)
            self._loads.pop(request, None)
            self._stores.pop(request, None)

    def reset_cache(self) -> None:
        self.client.call("reset")

    def shutdown(self) -> None:
        self.client.close()
