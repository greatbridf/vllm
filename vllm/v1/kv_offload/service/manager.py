# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler adapter for Rust-owned cache keys and reservations."""

import hashlib
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.service.client import (
    MAX_KEYS,
    CacheServiceClient,
    CacheServiceError,
    CacheServiceUnavailable,
)

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
    the service rejects after a reconnect counts as the same failure, because a
    service that dropped the namespace cannot serve it again. `prepare_load`
    raises instead, since a lookup that could not reach the service missed
    first. A reply that breaks the slot contract is a service bug rather than a
    transport failure, so it raises `CacheServiceError` instead of degrading.
    """

    def __init__(self, client: CacheServiceClient, num_chunks: int):
        self.client = client
        self.num_chunks = num_chunks
        # Keyed by the wire id, so the map lives exactly as long as the request.
        self._requests: dict[str, _RequestLeases] = {}
        self._reset_pending = False
        self._degraded = False

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
        result = _LOOKUP_RESULTS.get(status) if isinstance(status, str) else None
        if result is None:
            logger.warning_once(
                "KV cache service reported unknown lookup status %r; treating it as "
                "a retry.",
                status,
            )
            return LookupResult.RETRY
        if result is LookupResult.HIT:
            # The service takes the lease itself and reports a hit only then.
            self._request_state(request).leased = True
        return result

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> CPULoadStoreSpec:
        request = self._wire_id(req_context.req_id)
        encoded = self._encode(keys)
        slots: list[int] = []
        for batch in _batches(encoded):
            slots += self.client.call("load", keys=batch, request=request)["slots"]
        if len(slots) != len(encoded):
            raise CacheServiceError(
                f"KV cache service returned {len(slots)} slots for {len(encoded)} keys"
            )
        if not all(0 <= slot < self.num_chunks for slot in slots):
            raise CacheServiceError(
                f"KV cache service returned a slot outside 0..{self.num_chunks}"
            )
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
        evicted: list[OffloadKey] = []
        for batch in _batches(self._encode(keys)):
            response = self._call_or_degrade("store", keys=batch, request=request)
            if response is None or response.get("full"):
                break
            reserved = response["keys"]
            if len(reserved) != len(response["slots"]):
                raise CacheServiceError(
                    "KV cache service reserved a different number of keys and slots"
                )
            stored += [OffloadKey(bytes.fromhex(key)) for key in reserved]
            slots += response["slots"]
            evicted += [OffloadKey(bytes.fromhex(key)) for key in response["evicted"]]
        if not stored:
            return None
        state = self._request_state(request)
        state.stores += 1
        state.reserved.update(key.hex() for key in stored)
        return PrepareStoreOutput(stored, CPULoadStoreSpec(slots), evicted)

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
            committed = self._call_or_degrade(
                "commit", keys=batch, request=request, success=success
            )
            if committed is None:
                return
        state.stores -= 1
        state.reserved.difference_update(encoded)
        self._maybe_release(request)

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

    def shutdown(self) -> None:
        self.client.close()
