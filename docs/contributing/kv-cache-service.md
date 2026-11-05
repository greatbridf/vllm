# Process-Separated KV Offloading

Read this before changing the Rust CPU cache service or its native offloading spec.

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
