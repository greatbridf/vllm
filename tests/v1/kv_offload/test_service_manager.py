# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lease and reset state machine of the scheduler's service adapter."""

import hashlib
from typing import Any

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec, CPUOffloadingInfo
from vllm.v1.kv_offload.service.client import (
    MAX_KEYS,
    CacheServiceError,
    CacheServiceUnavailable,
)
from vllm.v1.kv_offload.service.manager import ServiceOffloadingManager
from vllm.v1.metrics.cache_hit_source import CacheHitSource

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
        # Statuses a batched lookup reports, when they differ from `status`.
        self.statuses: list[str] | None = None
        # Store calls that reserve before the cache reports itself full.
        self.full_after: int | None = None
        # Slots to answer a load with, when they differ from one per key.
        self.load_slots: list[int] | None = None
        # Slots to answer a store with, when they differ from one per key.
        self.store_slots: list[int] | None = None
        # Answer `store` alone with a rejection instead.
        self.reject_store = False
        # Replace the whole `stats` reply.
        self.stats_reply: dict[str, Any] | None = None
        # Answer every RPC with a transport failure instead.
        self.unavailable = False
        # Answer every RPC with a rejection instead.
        self.reject = False
        # Keys the service reports as evicted by the next store.
        self.evicted: list[str] = []
        # Namespace counters the service reports from `stats`.
        self.counters: dict[str, int] = {"entries": 0, "pending": 0, "leased": 0}
        self.closed = False
        self._store_calls = 0

    def call(self, op: str, retry: bool = False, **fields: Any) -> dict[str, Any]:
        self.calls.append((op, {"retry": retry, **fields}))
        if self.unavailable:
            raise CacheServiceUnavailable(f"KV cache service {op} failed: peer gone")
        if self.reject:
            raise CacheServiceError(f"KV cache service rejected {op}: namespace gone")
        return getattr(self, f"_{op}")(**fields)

    def close(self) -> None:
        self.closed = True

    def ops(self, op: str) -> list[dict[str, Any]]:
        return [fields for name, fields in self.calls if name == op]

    def _lookup(self, key: str, request: str) -> dict[str, Any]:
        return {"ok": True, "status": self.status}

    def _lookup_many(self, keys: list[str], request: str) -> dict[str, Any]:
        statuses = self.statuses
        if statuses is None:
            statuses = [self.status] * len(keys)
        return {"ok": True, "statuses": list(statuses)}

    def _load(self, keys: list[str], request: str) -> dict[str, Any]:
        slots = self.load_slots if self.load_slots is not None else self._slots(keys)
        return {"ok": True, "slots": list(slots)}

    @staticmethod
    def _slots(keys: list[str]) -> list[int]:
        # Slots are chunk ids of the namespace, so they repeat modulo its size.
        return [index % NUM_CHUNKS for index in range(len(keys))]

    def _store(self, keys: list[str], request: str) -> dict[str, Any]:
        self._store_calls += 1
        if self.reject_store:
            raise CacheServiceError(
                "KV cache service rejected store: batch too large for one control frame"
            )
        if self.full_after is not None and self._store_calls > self.full_after:
            return {"ok": True, "full": True}
        slots = self.store_slots if self.store_slots is not None else self._slots(keys)
        return {
            "ok": True,
            "keys": list(keys),
            "slots": list(slots),
            "evicted": list(self.evicted),
        }

    def _commit(self, keys: list[str], request: str, success: bool) -> dict[str, Any]:
        return {"ok": True}

    def _touch(self, keys: list[str]) -> dict[str, Any]:
        return {"ok": True}

    def _release_request(self, request: str) -> dict[str, Any]:
        return {"ok": True, "released": 1, "unknown": 0}

    def _reset(self, force: bool) -> dict[str, Any]:
        return {"ok": True}

    def _stats(self) -> dict[str, Any]:
        if self.stats_reply is not None:
            return self.stats_reply
        return {"ok": True, "stats": dict(self.counters)}

    def _close(self) -> dict[str, Any]:
        return {"ok": True}


def make_manager(
    client: FakeClient | None = None,
    info: CPUOffloadingInfo | None = None,
    enable_events: bool = False,
) -> ServiceOffloadingManager:
    return ServiceOffloadingManager(
        client or FakeClient(),
        NUM_CHUNKS,
        info=info,
        enable_events=enable_events,
    )


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


def test_refused_commit_drops_the_reservation_without_raising():
    """A reservation the service reclaimed must not end the scheduler step."""
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")
    key = make_key()

    assert manager.prepare_store([key], request) is not None
    manager.on_request_finished(request)
    # The service freed the reservation after its pending timeout.
    client.reject = True
    manager.complete_store([key], request)

    assert len(client.ops("commit")) == 1
    # The store counts as one that did not publish, so the request can finish.
    assert not manager._requests


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


def test_lookup_many_answers_every_key_with_one_rpc_per_batch():
    """A scan must cost one RPC per MAX_KEYS keys, not one per key."""
    client = FakeClient()
    client.statuses = ["hit", "pending", "miss"]
    manager = make_manager(client)
    request = ReqContext("request")

    assert manager.lookup_many([make_key(0), make_key(1), make_key(2)], request) == [
        LookupResult.HIT,
        LookupResult.HIT_PENDING,
        LookupResult.MISS,
    ]
    assert len(client.ops("lookup_many")) == 1
    assert client.ops("lookup") == []

    # A request longer than one batch is split, and keeps the key order.
    client.calls.clear()
    client.statuses = None
    keys = [make_key(index) for index in range(MAX_KEYS + 2)]
    assert manager.lookup_many(keys, request) == [LookupResult.HIT] * len(keys)
    assert [len(fields["keys"]) for fields in client.ops("lookup_many")] == [
        MAX_KEYS,
        2,
    ]


def test_lookup_many_maps_unknown_statuses_to_retry():
    client = FakeClient()
    client.statuses = ["evicted", "hit"]
    manager = make_manager(client)
    request = ReqContext("request")

    assert manager.lookup_many([make_key(0), make_key(1)], request) == [
        LookupResult.RETRY,
        LookupResult.HIT,
    ]


def test_lookup_many_rejects_a_status_count_mismatch():
    client = FakeClient()
    client.statuses = ["hit"]
    manager = make_manager(client)
    request = ReqContext("request")

    with pytest.raises(CacheServiceError, match="for 2 keys"):
        manager.lookup_many([make_key(0), make_key(1)], request)


def test_lookup_many_takes_a_lease_the_request_releases_once():
    """A hit taken by a batch is released like a per-key hit."""
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")

    assert manager.lookup_many([make_key(0), make_key(1)], request) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]
    manager.on_request_finished(request)
    assert len(client.ops("release_request")) == 1

    # A batch of misses takes no lease, so finishing sends no release.
    client.calls.clear()
    client.status = "miss"
    misses = ReqContext("misses")
    assert manager.lookup_many([make_key(0)], misses) == [LookupResult.MISS]
    manager.on_request_finished(misses)
    assert client.ops("release_request") == []


def test_lookup_many_degrades_to_misses():
    client = FakeClient()
    client.unavailable = True
    manager = make_manager(client)
    request = ReqContext("request")

    assert manager.lookup_many([make_key(0), make_key(1)], request) == [
        LookupResult.MISS,
        LookupResult.MISS,
    ]
    # One failure disables offloading instead of retrying inside every step.
    assert manager.lookup_many([make_key(0)], request) == [LookupResult.MISS]
    assert len(client.calls) == 1


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


def test_prepare_store_rejects_slot_out_of_range():
    """A slot outside the namespace would make a worker write unrelated memory."""
    client = FakeClient()
    client.store_slots = [0, NUM_CHUNKS]
    manager = make_manager(client)

    with pytest.raises(CacheServiceError, match="outside"):
        manager.prepare_store([make_key(0), make_key(1)], ReqContext("request"))


def test_prepare_store_stops_reserving_when_the_service_refuses_it():
    """A refused store reserved nothing, so it must not end the scheduler step."""
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")

    client.reject_store = True
    assert manager.prepare_store([make_key()], request) is None

    # Offloading is off from here, so a later lookup does not touch the service.
    client.calls.clear()
    assert manager.lookup(make_key(), request) is LookupResult.MISS
    assert client.calls == []


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
        index % NUM_CHUNKS for index in range(MAX_KEYS + 5)
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


def test_get_stats_maps_service_counters_to_the_cpu_metrics():
    """One stats RPC per collection fills the gauges and the store histogram."""
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")
    client.counters = {"entries": 4, "pending": 1, "leased": 1}

    manager.prepare_store([make_key(index) for index in range(MAX_KEYS + 5)], request)
    assert manager.lookup(make_key(), request) is LookupResult.HIT
    assert client.ops("stats") == []

    stats = manager.get_stats()

    assert stats is not None
    assert stats.reduce() == {
        "vllm:kv_offload_cpu_cache_usage_perc": 4 / NUM_CHUNKS,
        "vllm:kv_offload_cpu_cache_write_usage_perc": 1 / NUM_CHUNKS,
        "vllm:kv_offload_cpu_cache_read_usage_perc": 3 / NUM_CHUNKS,
        "vllm:kv_offload_cpu_allocation_size_count": 2,
        "vllm:kv_offload_cpu_allocation_size_sum": MAX_KEYS + 5,
    }
    assert len(client.ops("stats")) == 1

    # The allocation sizes belong to one collection only.
    again = manager.get_stats()
    assert again is not None
    assert "vllm:kv_offload_cpu_allocation_size_count" not in again.reduce()


def test_get_stats_is_none_when_the_service_is_degraded():
    client = FakeClient()
    manager = make_manager(client)

    client.unavailable = True
    assert manager.get_stats() is None
    # Offloading stays off, so a later collection sends nothing either.
    assert manager.get_stats() is None
    assert len(client.ops("stats")) == 1

    # A namespace the service dropped reports no metrics, and that must not
    # reach the scheduler step either.
    rejected = FakeClient()
    rejected.reject = True
    manager = make_manager(rejected)
    assert manager.get_stats() is None
    assert len(rejected.ops("stats")) == 1


def test_get_stats_is_none_for_a_reply_without_counters():
    """Metrics never fail a step, not even on a reply this client cannot read."""
    client = FakeClient()
    client.stats_reply = {"ok": True}
    manager = make_manager(client)

    assert manager.get_stats() is None
    assert len(client.ops("stats")) == 1


def test_config_info_reports_the_negotiated_geometry():
    info = CPUOffloadingInfo(
        cpu_num_chunks=NUM_CHUNKS,
        cpu_blocks_per_chunk=1,
        cpu_kv_cache_size_bytes=NUM_CHUNKS * 4096,
        cpu_kv_cache_size_tokens=128,
    )

    assert make_manager(info=info).config_info() == [info.as_config_info()]
    # A manager built without geometry publishes no label values.
    assert make_manager().config_info() == [{}]


def test_get_load_source_is_host():
    manager = make_manager()

    assert (
        manager.get_load_source(make_key(), ReqContext("request"))
        is CacheHitSource.HOST
    )


def test_take_events_reports_stored_evicted_and_failed_stores():
    """Stored, evicted, and abandoned chunks are announced like the CPU tier."""
    client = FakeClient()
    manager = make_manager(client, enable_events=True)
    request = ReqContext("request")
    stored = make_key(0)
    evicted = make_key(1)
    client.evicted = [evicted.hex()]

    output = manager.prepare_store([stored], request)
    assert output is not None
    assert output.evicted_keys == [evicted]
    assert list(manager.take_events()) == [
        OffloadingEvent(
            keys=[evicted],
            medium=Medium.CPU,
            removed=True,
            removal_expected=True,
        )
    ]

    manager.complete_store([stored], request)
    [event] = manager.take_events()
    assert event.keys == [stored]
    assert (event.removed, event.removal_expected, event.metadata_only) == (
        False,
        True,
        False,
    )
    assert event.req_context is request

    failed = make_key(2)
    client.evicted = []
    assert manager.prepare_store([failed], request) is not None
    manager.complete_store([failed], request, success=False)
    [event] = manager.take_events()
    assert event.keys == [failed]
    assert (event.removed, event.metadata_only) == (True, True)

    # The drain detached its batch, so nothing is replayed.
    assert list(manager.take_events()) == []


def test_events_are_off_unless_the_spec_enables_them():
    client = FakeClient()
    manager = make_manager(client)
    request = ReqContext("request")
    client.evicted = [make_key(1).hex()]

    assert manager.prepare_store([make_key(0)], request) is not None
    manager.complete_store([make_key(0)], request)

    assert list(manager.take_events()) == []


def test_partially_consumed_drain_does_not_replay_events():
    client = FakeClient()
    manager = make_manager(client, enable_events=True)
    request = ReqContext("request")
    for index in range(2):
        assert manager.prepare_store([make_key(index)], request) is not None
        manager.complete_store([make_key(index)], request)

    events = iter(manager.take_events())
    assert next(events).keys == [make_key(0)]
    assert list(manager.take_events()) == []


def test_shutdown_closes_the_namespace_then_the_socket():
    """A shutdown retires the namespace instead of leaving it adoptable."""
    client = FakeClient()
    manager = make_manager(client)

    manager.shutdown()

    assert [op for op, _ in client.calls] == ["close"]
    assert client.closed


def test_shutdown_swallows_a_failed_close():
    """A service that already dropped the namespace must not fail shutdown."""
    client = FakeClient()
    client.reject = True
    manager = make_manager(client)

    manager.shutdown()

    assert len(client.ops("close")) == 1
    assert client.closed
