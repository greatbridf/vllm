# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lease and reset state machine of the scheduler's service adapter."""

import hashlib
from typing import Any

import pytest

from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.service.client import (
    MAX_KEYS,
    CacheServiceError,
    CacheServiceUnavailable,
)
from vllm.v1.kv_offload.service.manager import ServiceOffloadingManager

NUM_CHUNKS = 8


def make_key(index: int = 0) -> OffloadKey:
    return OffloadKey(index.to_bytes(4, "big") * 8)


def wire_id(req_id: str) -> str:
    return hashlib.sha256(req_id.encode()).hexdigest()[:32]


class FakeClient:
    """Records every control RPC and answers from the fields it is given."""

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.status = "hit"
        # Store calls that reserve before the cache reports itself full.
        self.full_after: int | None = None
        # Slots to answer a load with, when they differ from one per key.
        self.load_slots: list[int] | None = None
        # Answer every RPC with a transport failure instead.
        self.unavailable = False
        # Answer every RPC with a rejection instead.
        self.reject = False
        self._store_calls = 0

    def call(self, op: str, retry: bool = False, **fields: Any) -> dict[str, Any]:
        self.calls.append((op, {"retry": retry, **fields}))
        if self.unavailable:
            raise CacheServiceUnavailable(f"KV cache service {op} failed: peer gone")
        if self.reject:
            raise CacheServiceError(f"KV cache service rejected {op}: namespace gone")
        return getattr(self, f"_{op}")(**fields)

    def close(self) -> None:
        pass

    def ops(self, op: str) -> list[dict[str, Any]]:
        return [fields for name, fields in self.calls if name == op]

    def _lookup(self, key: str, request: str) -> dict[str, Any]:
        return {"ok": True, "status": self.status}

    def _load(self, keys: list[str], request: str) -> dict[str, Any]:
        slots = self.load_slots if self.load_slots is not None else range(len(keys))
        return {"ok": True, "slots": list(slots)}

    def _store(self, keys: list[str], request: str) -> dict[str, Any]:
        self._store_calls += 1
        if self.full_after is not None and self._store_calls > self.full_after:
            return {"ok": True, "full": True}
        return {
            "ok": True,
            "keys": list(keys),
            "slots": list(range(len(keys))),
            "evicted": [],
        }

    def _commit(self, keys: list[str], request: str, success: bool) -> dict[str, Any]:
        return {"ok": True}

    def _touch(self, keys: list[str]) -> dict[str, Any]:
        return {"ok": True}

    def _release_request(self, request: str) -> dict[str, Any]:
        return {"ok": True, "released": 1, "unknown": 0}

    def _reset(self, force: bool) -> dict[str, Any]:
        return {"ok": True}


def make_manager(client: FakeClient | None = None) -> ServiceOffloadingManager:
    return ServiceOffloadingManager(client or FakeClient(), NUM_CHUNKS)


def test_lookup_lease_released_only_after_finish_and_all_jobs():
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")
    key = make_key()

    assert manager.lookup(key, request) is LookupResult.HIT
    manager.prepare_load([key], request)
    manager.on_request_finished(request)
    assert client.ops("release_request") == []

    manager.complete_load([key], request)
    assert client.ops("release_request") == [
        {"retry": False, "request": wire_id("request")}
    ]


def test_two_loads_in_one_request_issue_one_release_request():
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")
    keys = [make_key(0), make_key(1)]

    manager.prepare_load(keys, request)
    manager.prepare_load(keys[1:], request)
    manager.on_request_finished(request)
    manager.complete_load(keys, request)
    assert client.ops("release_request") == []

    manager.complete_load(keys[1:], request)
    assert len(client.ops("release_request")) == 1


def test_complete_store_before_on_request_finished_defers_release():
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")
    key = make_key()

    assert manager.lookup(key, request) is LookupResult.HIT
    assert manager.prepare_store([key], request) is not None
    manager.complete_store([key], request)
    assert client.ops("release_request") == []

    manager.on_request_finished(request)
    assert len(client.ops("release_request")) == 1


def test_completion_for_unknown_request_is_ignored_without_negative_counters():
    client = FakeClient()
    manager = make_manager(client)
    unknown = ReqContext("unknown")

    manager.complete_load([make_key()], unknown)
    manager.complete_store([make_key()], unknown)
    manager.on_request_finished(unknown)
    assert client.calls == []

    # A tracked request still accounts for its jobs exactly.
    request = ReqContext("request")
    key = make_key()
    assert manager.lookup(key, request) is LookupResult.HIT
    manager.prepare_load([key], request)
    manager.on_request_finished(request)
    manager.complete_load([key], request)
    assert len(client.ops("release_request")) == 1


def test_prepare_load_rejects_slot_count_mismatch():
    client = FakeClient()
    client.load_slots = [0]
    manager = make_manager(client)

    with pytest.raises(CacheServiceError, match="slots for"):
        manager.prepare_load([make_key(0), make_key(1)], ReqContext("request"))


def test_prepare_load_rejects_slot_out_of_range():
    client = FakeClient()
    client.load_slots = [0, NUM_CHUNKS]
    manager = make_manager(client)

    with pytest.raises(CacheServiceError, match="outside"):
        manager.prepare_load([make_key(0), make_key(1)], ReqContext("request"))


def test_prepare_store_splits_batches_above_max_keys():
    client = FakeClient()
    manager = make_manager(client)
    keys = [make_key(index) for index in range(MAX_KEYS + 5)]

    output = manager.prepare_store(keys, ReqContext("request"))

    assert output is not None
    assert [len(fields["keys"]) for fields in client.ops("store")] == [MAX_KEYS, 5]
    # Reserved keys and their slots keep the order of the request.
    assert output.keys_to_store == keys
    store_spec = output.store_spec
    assert isinstance(store_spec, CPULoadStoreSpec)
    assert [int(slot) for slot in store_spec.block_ids] == [
        *range(MAX_KEYS),
        *range(5),
    ]


def test_prepare_store_keeps_what_a_full_chunk_could_not_reserve():
    client = FakeClient()
    client.full_after = 1
    manager = make_manager(client)
    keys = [make_key(index) for index in range(MAX_KEYS + 5)]

    output = manager.prepare_store(keys, ReqContext("request"))

    assert output is not None
    assert len(output.keys_to_store) == MAX_KEYS
    assert len(client.ops("store")) == 2


def test_request_id_is_digested_for_the_wire():
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("🙂" * 500 + "x" * 500)
    key = make_key()

    assert manager.lookup(key, request) is LookupResult.HIT
    assert manager.prepare_store([key], request) is not None
    manager.complete_store([key], request)
    manager.on_request_finished(request)

    requests = {fields["request"] for _, fields in client.calls if "request" in fields}
    assert requests == {wire_id(request.req_id)}
    assert len(requests.pop()) == 32


def test_reset_is_two_phase_and_aborts_pending_reservations():
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")
    key = make_key()

    assert manager.lookup(key, request) is LookupResult.HIT
    output = manager.prepare_store([key], request)
    assert output is not None
    reserved = [output.keys_to_store[0].hex()]

    manager.reset_cache()
    # The lease ends now, but a worker may still be writing the reserved slot.
    assert len(client.ops("release_request")) == 1
    assert client.ops("reset") == []
    assert manager.has_pending_work()

    manager.on_reset_flushed()
    assert client.ops("commit") == [
        {
            "retry": False,
            "keys": reserved,
            "request": wire_id("request"),
            "success": False,
        }
    ]
    assert client.ops("reset") == [{"retry": False, "force": True}]
    # Aborting the reservation precedes the reset that recycles its slot.
    assert [op for op, _ in client.calls][-2:] == ["commit", "reset"]
    assert not manager.has_pending_work()
    assert not manager._requests


def test_unavailable_service_degrades_instead_of_raising():
    client = FakeClient()
    client.unavailable = True
    manager = make_manager(client)
    request = ReqContext("request")
    key = make_key()

    assert manager.lookup(key, request) is LookupResult.MISS
    assert manager.prepare_store([key], request) is None
    manager.complete_load([key], request)
    manager.complete_store([key], request)
    manager.touch([key], request)
    manager.on_request_finished(request)
    manager.reset_cache()
    manager.on_reset_flushed()

    # One failure disables offloading instead of retrying inside every step.
    assert len(client.calls) == 1
    assert not manager.has_pending_work()


def test_unknown_lookup_status_maps_to_retry():
    client = FakeClient()
    client.status = "evicted"
    manager = make_manager(client)

    assert manager.lookup(make_key(), ReqContext("request")) is LookupResult.RETRY


def test_rejected_lookup_degrades_instead_of_raising():
    client = FakeClient()
    client.reject = True
    manager = make_manager(client)
    request = ReqContext("request")

    # A reconnect reaches a service that already dropped the namespace.
    assert manager.lookup(make_key(), request) is LookupResult.MISS
    assert manager.prepare_store([make_key()], request) is None
    assert len(client.calls) == 1
