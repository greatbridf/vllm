# Process-Separated KV Offloading

Read this before changing the Rust CPU cache service or its native offloading spec.

- Keep cache keys, reservations, and eviction in the service. Worker-side memory
  allocators are not a substitute for an out-of-process backend.
- Use shared memory descriptors, not process-local addresses, across the process
  boundary. The service owns allocation; workers retain GPU DMA execution.
- Lookup hits hold a lease before returning. Keep leases through all dependent
  loads; publish stores only after every participating worker reports completion.
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
  frames, keys, geometry, and allocation sizes; seal shared-memory file sizes.
- Keep unsupported topology, events, and secondary-tier combinations explicit.
  The initial backend is single-node TP-only and does not persist across engine
  sessions. Model names alone do not certify byte-layout compatibility.
- Validate the actual Rust subprocess and Python descriptor handshake, not only
  mocked RPC calls. Run byte-level storage and lease tests on CPU; GPU DMA and
  cold-versus-reload model output require separate validation on GPU hardware.
- Follow the Rust subtree instructions for Cargo dependencies and testing. Use
  the repository virtual environment for Python tests. Report unrun GPU checks
  separately from passing CPU tests.
