# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler adapter for Rust-owned cache keys and reservations."""

import contextlib
import hashlib
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    ConfigInfoMapping,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingInfo,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.service.client import (
    MAX_KEYS,
    CacheServiceClient,
    CacheServiceError,
    CacheServiceUnavailable,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource

logger = init_logger(__name__)

_LOOKUP_RESULTS = {
    "hit": LookupResult.HIT,
    "pending": LookupResult.HIT_PENDING,
    "miss": LookupResult.MISS,
}

_UNKNOWN_COMPLETION = (
    "KV cache service manager received a completion for a request it does not "
    "track, which only a reset can leave behind; ignoring it instead of driving "
    "lease accounting negative."
)
_UNMATCHED_COMPLETION = (
    "KV cache service manager received a completion for a request with no "
    "outstanding job of that kind; ignoring it."
)
_UNKNOWN_LEASE = (
    "KV cache service knows no lease for a request reported as finished; the "
    "service and this manager disagree on lease accounting."
)
_REFUSED_COMMIT = (
    "KV cache service refused a commit, which a reservation older than its "
    "pending timeout leaves behind; the store did not publish, so the key is a "
    "miss again: %s"
)


@dataclass
class _RequestLeases:
    """Service state one request holds until it finishes."""

    # Outstanding load and store jobs, each counted from `prepare_*`.
    loads: int = 0
    stores: int = 0
    # Keys reserved by a store whose commit has not arrived yet.
    reserved: set[str] = field(default_factory=set)
    # The service holds lookup leases for this request.
    leased: bool = False
    finished: bool = False


def _batches(keys: list[str]) -> Iterator[list[str]]:
    """Splits an encoded key list into batches the protocol accepts."""
    for start in range(0, len(keys), MAX_KEYS):
        yield keys[start : start + MAX_KEYS]


class ServiceOffloadingManager(OffloadingManager):
    """Scheduler view of the cache service's keys, leases, and slots.

    A tracked request holds one service-side lease set, released only once the
    request finished, every load and store job it issued completed, and no
    reservation is still waiting for its commit. `reset_cache()` releases the
    leases at once but leaves the reset itself to `on_reset_flushed()`, because
    a slot may not be recycled while a worker can still write to it.

    A transport failure disables offloading for the rest of the engine instead
    of failing the scheduler step: a lookup becomes a miss, a store reservation
    becomes `None`, and completions, touches, and resets become no-ops. A lookup
    or a store the service rejects counts as the same failure, because it reserves
    nothing and the step must survive it. `prepare_load` raises instead, since a
    lookup that could not reach the service missed first. A commit the service
    refuses, which a reservation older than its pending timeout leaves behind,
    drops that reservation instead of raising. A reply that breaks the slot
    contract is a service bug rather than a transport failure, so it raises
    `CacheServiceError` instead of degrading.

    `get_stats()` reports the CPU-tier metrics from one `stats` RPC per
    collection, `config_info()` the chunk geometry the spec negotiated with the
    service, and `take_events()` the store and eviction events its own store
    calls produced once KV cache events are enabled. `lookup_many()` answers a
    whole batch of a scan with one RPC per `MAX_KEYS` keys instead of one per
    key, and stops at the first miss when the caller asks it to, so a prefix scan
    leases only the chunks it consumes. `shutdown()` closes the namespace before
    the connection, so the service retires it instead of keeping it adoptable.
    """

    def __init__(
        self,
        client: CacheServiceClient,
        num_chunks: int,
        info: CPUOffloadingInfo | None = None,
        enable_events: bool = False,
    ):
        self.client = client
        self.num_chunks = num_chunks
        # Static labels the connector publishes; empty when the spec gave none.
        self._config_info: ConfigInfoMapping = (
            info.as_config_info() if info is not None else {}
        )
        # Keyed by the wire id, so the map lives exactly as long as the request.
        self._requests: dict[str, _RequestLeases] = {}
        self._reset_pending = False
        self._degraded = False
        # Chunks each store call asked the service for since the last drain.
        self._allocation_sizes: list[int] = []
        self._events: list[OffloadingEvent] | None = [] if enable_events else None

    @staticmethod
    def _wire_id(req_id: str) -> str:
        """Digests a request id, which a user may set to any length."""
        return hashlib.sha256(req_id.encode()).hexdigest()[:32]

    @staticmethod
    def _encode(keys: Collection[OffloadKey]) -> list[str]:
        return [key.hex() for key in keys]

    def _request_state(self, request: str) -> _RequestLeases:
        """Returns the state a request holds, tracking it on first use."""
        state = self._requests.get(request)
        if state is None:
            state = _RequestLeases()
            self._requests[request] = state
        return state

    def _degrade(self, op: str, error: CacheServiceError) -> None:
        self._degraded = True
        logger.error_once(
            "KV cache service %s failed; disabling KV offloading for this engine "
            "and reporting every lookup as a miss: %s",
            op,
            error,
        )

    def _call_or_degrade(
        self, op: str, retry: bool = False, **fields: Any
    ) -> dict[str, Any] | None:
        """Calls the service, degrading to `None` when it cannot be reached."""
        if self._degraded:
            return None
        try:
            return self.client.call(op, retry=retry, **fields)
        except CacheServiceUnavailable as error:
            self._degrade(op, error)
            return None

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        request = self._wire_id(req_context.req_id)
        try:
            response = self._call_or_degrade(
                "lookup", retry=True, key=key.hex(), request=request
            )
        except CacheServiceError as error:
            # A reconnect reaches a service that already dropped this namespace
            # and rejects the lookup; offloading is over for this engine either
            # way, and the scheduler step must survive it.
            self._degrade("lookup", error)
            return LookupResult.MISS
        if response is None:
            return LookupResult.MISS
        status = response.get("status")
        result = self._lookup_result(status)
        if result is LookupResult.HIT:
            # The service takes the lease itself and reports a hit only then.
            self._request_state(request).leased = True
        return result

    @staticmethod
    def _lookup_result(status: Any) -> LookupResult:
        """Maps one wire status, deferring on one this client does not know."""
        result = _LOOKUP_RESULTS.get(status) if isinstance(status, str) else None
        if result is None:
            logger.warning_once(
                "KV cache service reported unknown lookup status %r; treating it as "
                "a retry.",
                status,
            )
            return LookupResult.RETRY
        return result

    def lookup_many(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        until_miss: bool = False,
    ) -> list[LookupResult]:
        """Answers one `lookup_many` RPC per batch instead of one per key.

        With `until_miss` the service stops at the first key that missed and
        leases only the hits before it, so the reply, and this result, end there.
        """
        request = self._wire_id(req_context.req_id)
        encoded = self._encode(keys)
        results: list[LookupResult] = []
        for batch in _batches(encoded):
            try:
                response = self._call_or_degrade(
                    "lookup_many",
                    retry=True,
                    keys=batch,
                    request=request,
                    until_miss=until_miss,
                )
            except CacheServiceError as error:
                # Same reasoning as `lookup`: a request the service cannot
                # answer is a miss, and offloading is over for this engine.
                self._degrade("lookup_many", error)
                response = None
            if response is None:
                # A degraded manager reports every key as a miss, so a scan still
                # sees the gap where offloading stopped.
                return [LookupResult.MISS] * len(encoded)
            statuses = self._lookup_statuses(response, len(batch), until_miss)
            results += [self._lookup_result(status) for status in statuses]
            if len(statuses) < len(batch):
                break
        if LookupResult.HIT in results:
            # The service took the leases itself and reports a hit only then.
            self._request_state(request).leased = True
        return results

    @staticmethod
    def _lookup_statuses(
        reply: dict[str, Any], expected: int, until_miss: bool
    ) -> list[Any]:
        """Returns the statuses a batch reply carries, or raises.

        A truncated scan reports fewer statuses than keys, which only a request
        that asked for it may accept, and only when the reply ends at the miss
        that stopped it.
        """
        statuses = reply.get("statuses")
        truncated = (
            until_miss
            and isinstance(statuses, list)
            and 0 < len(statuses) < expected
            and statuses[-1] == "miss"
        )
        if not isinstance(statuses, list) or (
            len(statuses) != expected and not truncated
        ):
            raise CacheServiceError(
                f"KV cache service returned {statuses!r} for {expected} keys"
            )
        return statuses

    def _chunk_slots(self, reply: dict[str, Any], expected: int) -> list[int]:
        """Returns the one chunk slot per key a reply carries, or raises."""
        slots = reply.get("slots")
        if not isinstance(slots, list) or len(slots) != expected:
            raise CacheServiceError(
                f"KV cache service returned {slots!r} slots for {expected} keys"
            )
        if not all(
            isinstance(slot, int) and 0 <= slot < self.num_chunks for slot in slots
        ):
            raise CacheServiceError(
                f"KV cache service returned a slot outside 0..{self.num_chunks}"
            )
        return slots

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> CPULoadStoreSpec:
        request = self._wire_id(req_context.req_id)
        encoded = self._encode(keys)
        slots: list[int] = []
        for batch in _batches(encoded):
            reply = self.client.call("load", keys=batch, request=request)
            slots += self._chunk_slots(reply, len(batch))
        state = self._request_state(request)
        state.loads += 1
        state.leased = True
        return CPULoadStoreSpec(slots)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        request = self._wire_id(req_context.req_id)
        state = self._requests.get(request)
        if state is None:
            logger.warning_once(_UNKNOWN_COMPLETION)
            return
        if not state.loads:
            logger.warning_once(_UNMATCHED_COMPLETION)
            return
        # Keep lookup leases until the request has no remaining load jobs.
        state.loads -= 1
        self._maybe_release(request)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        request = self._wire_id(req_context.req_id)
        stored: list[OffloadKey] = []
        slots: list[int] = []
        evicted_keys: list[OffloadKey] = []
        for batch in _batches(self._encode(keys)):
            self._allocation_sizes.append(len(batch))
            try:
                response = self._call_or_degrade("store", keys=batch, request=request)
            except CacheServiceError as error:
                # A store the service refuses has reserved nothing, so this is
                # not a lost connection: stop reserving and leave those chunks
                # in GPU memory instead of ending the scheduler step.
                self._degrade("store", error)
                break
            if response is None or response.get("full"):
                break
            reserved = response.get("keys")
            evicted = response.get("evicted")
            if not isinstance(reserved, list) or not isinstance(evicted, list):
                raise CacheServiceError(
                    f"KV cache service reserved no key of {len(batch)} asked for"
                )
            slots += self._chunk_slots(response, len(reserved))
            stored += [OffloadKey(bytes.fromhex(key)) for key in reserved]
            evicted_keys += [OffloadKey(bytes.fromhex(key)) for key in evicted]
        if not stored:
            return None
        if self._events is not None and evicted_keys:
            # This store made room, so the removal is one this manager caused.
            self._events.append(
                OffloadingEvent(
                    keys=evicted_keys,
                    medium=Medium.CPU,
                    removed=True,
                    removal_expected=True,
                )
            )
        state = self._request_state(request)
        state.stores += 1
        state.reserved.update(key.hex() for key in stored)
        return PrepareStoreOutput(stored, CPULoadStoreSpec(slots), evicted_keys)

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        request = self._wire_id(req_context.req_id)
        state = self._requests.get(request)
        if state is None:
            logger.warning_once(_UNKNOWN_COMPLETION)
            return
        if not state.stores:
            logger.warning_once(_UNMATCHED_COMPLETION)
            return
        encoded = self._encode(keys)
        for batch in _batches(encoded):
            if not self._commit(batch, request, success):
                return
        state.stores -= 1
        state.reserved.difference_update(encoded)
        if self._events is not None and keys:
            if success:
                self._events.append(
                    OffloadingEvent(
                        keys=list(keys),
                        medium=Medium.CPU,
                        removed=False,
                        removal_expected=True,
                        req_context=req_context,
                    )
                )
            else:
                # The service released the metadata of a store that failed.
                self._events.append(
                    OffloadingEvent(
                        keys=list(keys),
                        medium=Medium.CPU,
                        removed=True,
                        metadata_only=True,
                    )
                )
        self._maybe_release(request)

    def _commit(self, keys: list[str], request: str, success: bool) -> bool:
        """Commits one batch without ending the scheduler step for either failure.

        Returns `False` when the service is unreachable, which ends offloading for
        this engine. A refused commit is not a transport failure: the service has
        reclaimed a reservation that outlived its pending timeout, so the store
        did not publish and the key is a miss again.
        """
        try:
            committed = self._call_or_degrade(
                "commit", keys=keys, request=request, success=success
            )
        except CacheServiceError as error:
            logger.warning_once(_REFUSED_COMMIT, error)
            return True
        return committed is not None

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        for batch in _batches(self._encode(keys)):
            if self._call_or_degrade("touch", keys=batch) is None:
                return

    def on_request_finished(self, req_context: ReqContext) -> None:
        request = self._wire_id(req_context.req_id)
        state = self._requests.get(request)
        if state is None:
            return
        state.finished = True
        self._maybe_release(request)

    def _maybe_release(self, request: str) -> None:
        state = self._requests.get(request)
        if (
            state is None
            or not state.finished
            or state.loads
            or state.stores
            or state.reserved
        ):
            return
        del self._requests[request]
        if not state.leased:
            return
        response = self._call_or_degrade("release_request", request=request)
        if response is not None and response.get("unknown"):
            logger.warning_once(_UNKNOWN_LEASE)

    def reset_cache(self) -> None:
        for request, state in self._requests.items():
            if state.leased:
                self._call_or_degrade("release_request", request=request)
                state.leased = False
        self._reset_pending = True

    def has_pending_work(self) -> bool:
        return self._reset_pending

    def on_reset_flushed(self) -> None:
        """Ends the reset now that every flushed transfer has been reported."""
        if not self._reset_pending:
            return
        for request, state in list(self._requests.items()):
            for batch in _batches(sorted(state.reserved)):
                self._reset_call("commit", keys=batch, request=request, success=False)
        self._reset_call("reset", force=True)
        self._requests.clear()
        self._reset_pending = False

    def _reset_call(self, op: str, **fields: Any) -> None:
        """Sends one step of the reset sequence, which degrades instead of raising."""
        try:
            self._call_or_degrade(op, **fields)
        except CacheServiceError as error:
            self._degrade(op, error)

    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        return CacheHitSource.HOST

    def config_info(self) -> Sequence[ConfigInfoMapping]:
        """Reports the chunk geometry the spec negotiated with the service."""
        return [self._config_info]

    def get_stats(self) -> OffloadingConnectorStats | None:
        """Reports the CPU-tier metrics from one `stats` RPC per collection."""
        allocation_sizes, self._allocation_sizes = self._allocation_sizes, []
        try:
            response = self._call_or_degrade("stats", retry=True)
        except CacheServiceError as error:
            # Metrics never fail a scheduler step, and a namespace the service
            # dropped cannot report them.
            self._degrade("stats", error)
            return None
        if response is None:
            return None
        counters = response.get("stats")
        if not isinstance(counters, dict) or not {"entries", "pending"} <= set(
            counters
        ):
            # Metrics never fail a scheduler step, not even on a reply this
            # client cannot read.
            logger.warning_once(
                "KV cache service reported no namespace counters in %s",
                ", ".join(sorted(response)) or "an empty reply",
            )
            return None
        stats = OffloadingConnectorStats()
        usage = counters["entries"] / self.num_chunks if self.num_chunks else 0.0
        write_usage = counters["pending"] / self.num_chunks if self.num_chunks else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC,
            max(usage - write_usage, 0.0),
        )
        for allocation_size in allocation_sizes:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        # No declared metric carries the service's own counters, so they reach
        # the operator through the log instead of an invented metric name.
        logger.debug("KV cache service namespace counters: %s", counters)
        return stats

    def take_events(self) -> Iterable[OffloadingEvent]:
        """Drains the store and eviction events collected since the last call."""
        if self._events is None:
            return
        # Detach the batch up front so a partially consumed drain cannot replay
        # a key a retry has since re-stored.
        events, self._events = self._events, []
        yield from events

    def shutdown(self) -> None:
        """Retires the namespace, then closes the connection.

        A service that is already gone, or one that dropped this namespace,
        needs no close, and shutdown must not raise either way.
        """
        with contextlib.suppress(CacheServiceError):
            self.client.call("close")
        self.client.close()
