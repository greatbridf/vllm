# Process-Separated KV Offloading

Read this before changing the Rust CPU cache service or its native offloading spec.
The operator-facing side of this backend, including its
`kv_connector_extra_config` keys and the metrics it reports, lives in
[KV Offloading Usage Guide](../features/kv_offloading_usage.md).

- Keep cache keys, reservations, and eviction in the service. Worker-side memory
  allocators are not a substitute for an out-of-process backend.
- Use shared memory descriptors, not process-local addresses, across the process
  boundary. The service owns allocation; workers retain GPU DMA execution.
- Lookup hits hold a lease before returning. Keep leases through all dependent
  loads; publish stores only after every participating worker reports completion.
  A batched lookup is one frame of per-key lookups, so it takes the same lease
  and may name keys past the miss that ends the scan.
- A reservation belongs to the request that made it: reject a `commit` from any
  other request and keep the reservation pending. Bound how long a reservation may
  stay pending, because the connector reports a pending key as a hit that is not
  readable yet and retries it instead of missing. Keep the counters that show a
  leaked reservation or lease in `stats`.
- A forced reset is the only way out of leases and pending reservations, so it
  must belong to a caller that flushed its own transfers first. Never recycle a
  dropped slot while a worker could still write to it. The connector provides
  that flush fence: `reset_cache()` releases what it tracks, and
  `on_reset_flushed()` ends the reset once the workers reported the flush.
- A transport failure must not fail the scheduler step. Degrade to a miss, stop
  reserving, and keep the engine live; a reply that breaks the slot contract is
  a service bug, so raise it rather than degrading.
- A lost worker may still have DMA in flight. Invalidate its namespace rather
  than reclaiming slots into the same mapping. Do not reconnect old workers to a
  newly created namespace.
- An owner's departure is not a lost worker: detach the namespace, keep its
  mapping, entries, and leases, and let the same engine adopt it with a matching
  fingerprint and geometry, because one reconnect must not cost the engine its
  whole cache. Refuse adoption while the previous owner is still connected, and
  bound how long a detached namespace may stay adoptable so a crashed scheduler
  cannot pin memory behind its workers. A reconnect may adopt only, never create a
  second mapping while the workers still hold the first.
- A rejected request is not a lost worker. Answer it with `{"ok": false, ...}` and
  keep the session and namespace alive; only frame-level failures and I/O errors
  end a connection. Never hold a lock across a socket write or a descriptor send.
- Preserve complete packed physical rows for DeepSeek, including indexer data
  and padding. Follow the model's existing replay behavior for non-cacheable
  scratch and bounded-replay state.
- Keep the socket private and authenticate local peers by UID. Bound control
  frames, keys, geometry, allocation sizes, connections, and idle time; seal
  shared-memory file sizes. Stop cleanly on `SIGTERM`/`SIGINT` and reclaim the
  socket a killed predecessor left, so a supervisor restarts the service.
- Keep unsupported topology, events, and secondary-tier combinations explicit.
  The initial backend is single-node TP-only and does not persist across engine
  sessions. Model names alone do not certify byte-layout compatibility. Reject a
  `kv_connector_extra_config` key the backend cannot honour instead of ignoring
  it, and fit the requested geometry to the budget the service reports before
  asking for it, so a bad configuration fails at startup with a named key or a
  size, not with an opaque rejection from `open`.
- Pin only the bytes a rank owns. A shared region holds every rank's slice, so
  register this rank's slices instead of the whole mapping, and treat a failed
  unregistration at shutdown as a warning, not an error.
- Validate the actual Rust subprocess and Python descriptor handshake, not only
  mocked RPC calls. Run byte-level storage and lease tests on CPU; GPU DMA and
  cold-versus-reload model output require separate validation on GPU hardware.
- Follow the Rust subtree instructions for Cargo dependencies and testing. Use
  the repository virtual environment for Python tests. Report unrun GPU checks
  separately from passing CPU tests.

## Review checklist

Each invariant below is what a change to this backend must keep, and the test
that pins it:

- **Reservation ownership.** A `store` reservation belongs to the request that
  made it, and a `commit` from another request is rejected with the reservation
  still pending (`cache.rs::commit`,
  `commit_by_another_request_leaves_the_reservation_pending`).
- **Publish only after every worker reported.** `store` reserves, `commit` with
  `success` publishes, `complete_store` is the only caller, and a failed commit
  releases the slots and clears their hashes
  (`failed_publish_keeps_entry_pending_and_returns_error`).
- **Lease before hit.** `lookup` and `lookup_many` take the lease before
  reporting a hit, and `load` refuses a key this request does not lease
  (`lookup_lease_prevents_eviction_before_load`). A lease is released only when
  the request finished and every job it issued completed
  (`test_lookup_lease_released_only_after_finish_and_all_jobs`).
- **Namespace retirement.** A disconnected worker never hands its mapping to a
  new namespace: the namespace is retired and stays charged to capacity until its
  last session leaves (`main.rs::Registry::retire`,
  `worker_disconnect_still_invalidates_the_namespace`).
- **Namespace adoption.** A disconnected owner only detaches: the same fingerprint
  and geometry adopt the mapping with its entries and leases, a different one is
  rejected, and a live owner refuses it
  (`main.rs::Registry::open`,
  `owner_reconnect_adopts_the_namespace_and_keeps_entries`,
  `reopen_with_a_different_fingerprint_is_rejected`). A namespace nobody adopts is
  reclaimed when its grace period ends, and a reconnect can never create one
  (`detached_namespace_expires_after_the_grace_period`, `adopt_never_creates_a_namespace`,
  `test_reconnect_never_recreates_a_namespace_the_service_dropped`). A namespace
  with no sessions left is reclaimed at once
  (`namespace_without_sessions_is_reclaimed`).
- **Explicit shutdown.** `close` retires the namespace before the socket goes, and
  is the only teardown that leaves nothing adoptable (`close_retires_the_namespace_immediately`,
  `test_shutdown_closes_the_namespace_so_the_service_frees_it`).
- **Reset ordering.** `reset` refuses leases and pending reservations; the forced
  reset follows `on_reset_flushed()`, and only after every worker reported the
  transfers it flushed (`forced_reset_discards_leases_and_pending_reservations`,
  `test_reset_is_two_phase_and_aborts_pending_reservations`).
- **Bounded frames.** 1 MiB in either direction, 4096 keys per request, and a
  `store` reply that cannot fit is rejected before any slot is reserved
  (`protocol.rs::store_reply_fits`, `check_keys_is_bounded`).
- **No lock across socket I/O.** The reply is built before the wire is touched,
  the registry lock is released before the cache is used, and the frame and the
  descriptor are written after `handle` returns, so a client that stops reading
  blocks only its own connection
  (`slow_reader_does_not_block_another_namespace`).

## Validating a change

```bash
# Unit and integration tests: cache, protocol, and the service binary.
cargo test --manifest-path rust/Cargo.toml -p vllm-kv-cache-service

# Lints and formatting.
cargo clippy --manifest-path rust/Cargo.toml -p vllm-kv-cache-service \
  --all-targets -- -D warnings
cargo fmt --manifest-path rust/Cargo.toml -p vllm-kv-cache-service -- --check

# The connector, spec, manager, and region tests against the real binary.
cargo build --manifest-path rust/Cargo.toml -p vllm-kv-cache-service
.venv/bin/python -m pytest --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/test_service.py tests/v1/kv_offload/test_service_manager.py \
  tests/v1/kv_offload/test_service_region.py -q
.venv/bin/python -m pytest --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/test_factory.py -k rust -q
```

None of these needs root, an rswap kernel, or a GPU. Two of the suites can pass
while skipping what they cover, so a report says which happened:

- `tests/hash.rs` prints `SKIP: ...` and returns without asserting when the
  kernel has no BPF privileges or `BPF_MAP_TYPE_ARENA`, so a green `cargo test`
  on such a host has not exercised the arena or the program.
- `test_service.py` skips when the binary is missing (`Build
  vllm-kv-cache-service or set VLLM_KV_CACHE_SERVICE_BINARY`).
  `test_service_manager.py` and `test_service_region.py` need neither the binary
  nor a GPU; `test_service.py` needs only the binary.

The opt-in suites need hardware or privileges and stay out of CI:

```bash
# Kernel digest path: root, CONFIG_RSWAP, an idle swap setup, and a readable
# rswap backend for the digest audit.
sudo VLLM_TEST_KV_SERVICE_RSWAP=1 .venv/bin/python -m pytest \
  --confcutdir=tests/v1/kv_offload tests/v1/kv_offload/test_service_rswap.py -v

# GPU DMA and model output, against a service you started yourself.
VLLM_TEST_DEEPSEEK_V41_MODEL=/path/to/DeepSeek-V4.1-Flash \
VLLM_TEST_DEEPSEEK_V41_TP=8 \
VLLM_TEST_KV_CACHE_SERVICE_SOCKET=/run/vllm-kv-service/cache.sock \
.venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_offloading_connector.py \
  -k 'deepseek_v41_mmap and rust' -v
```

A PR touching this backend reports:

- the commands above it ran, each with its result, and every `SKIP` with the
  reason the host gave;
- the GPU or model-output evidence separately from the CPU results, or a
  statement that it was not run;
- the measured delta from the two benchmark commands below when the change
  touches a lookup, eviction, lease, reservation, or frame path, and the
  re-measured numbers when the README tables move;
- any behavior change in the same commit's `rust/src/kv-cache-service/README.md`
  and, if an operator sees it, `docs/features/kv_offloading_usage.md`.

## Benchmarks

Measure a change on CPU before arguing it is cheap. Two layers, neither needing a
GPU, root, or rswap:

```bash
cargo bench --manifest-path rust/Cargo.toml -p vllm-kv-cache-service \
  --bench cache -- --warm-up-time 0.2 --measurement-time 0.5
cargo run --release --manifest-path rust/Cargo.toml -p vllm-kv-cache-service \
  -- bench --json --assert-max rpc_serial_lookup=200
```

`cargo bench` covers the cache metadata alone: promotion from the LRU tail, victim
selection, lease release, and key validation. The `bench` subcommand runs the real
dispatch loop in-process and reports one JSON object per case, so it also covers
frame encoding, decoding, and the per-step RPC mix. A change that affects a lookup,
eviction, lease, or validation path must state the measured delta in its PR, and
the numbers in
[`rust/src/kv-cache-service/README.md`](../../rust/src/kv-cache-service/README.md)
must be re-measured when they move. Set `--assert-max` bounds 2-3x above the local
number: they exist to catch an O(n) path or a dropped batch, not jitter.
