# Rust KV Cache Service

A Linux, single-node DRAM cache backend for vLLM's `OffloadingConnector`.
The Rust process owns sealed `memfd` mappings, slot reservations, content keys,
lookup leases, and LRU eviction. Workers map the descriptors and execute the
existing GPU DMA transfers. No CUDA code runs in the service.

## Start

```bash
cargo build --manifest-path rust/Cargo.toml -p vllm-kv-cache-service
install -d -m 700 /tmp/vllm-kv-service
rust/target/debug/vllm-kv-cache-service \
  --socket /tmp/vllm-kv-service/cache.sock \
  --capacity-bytes 68719476736
```

Hash reporting is disabled by default; add `--rswap-attach` (or the visible
alias `--hash-attach`) to the command above to enable it. The flag accepts an
optional sysfs path and defaults to
`/sys/kernel/mm/xswap/rswap_attach`.

An existing socket is not overwritten by a running service: a start refuses
when another process still accepts connections on the path, and reclaims the
socket a killed process left behind. Use a private directory owned by the
service user; only same-UID clients are accepted.

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 8 --enable-prefix-caching --enforce-eager \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "RustServiceOffloadingSpec",
      "service_socket": "/tmp/vllm-kv-service/cache.sock",
      "cpu_bytes_to_use": 68719476736,
      "blocks_per_chunk": 1
    }
  }'
```

The service's capacity is an aggregate limit across engine namespaces.
`cpu_bytes_to_use` reserves an engine's share, including page padding and a
separate slice for every TP worker. Every rank stores its own copy.
Workers attempt host pinning through the existing DMA worker; pageable fallback
does not guarantee that Linux keeps all pages resident in DRAM.

## Ownership

The scheduler opens a unique engine namespace; workers attach with matching
layout fingerprints and receive a descriptor through `SCM_RIGHTS`. Each chunk
has consecutive worker slices. Prefix keys include the cache-group identifier.
The service reserves missing keys atomically and publishes them only when the
scheduler receives completion from all workers.

Lookup acquires a request lease before reporting a hit, so a later allocation
cannot evict the load source. Request leases are released after finish and all
pending transfers. Active leases and pending stores cannot be reset or evicted.

A reservation belongs to the request that made it: `store` records the owner, and
`commit` rejects a batch reserved by another request with
`commit does not own the reservation` and leaves those reservations pending. A
reservation nobody commits would stay pending for the life of the namespace, and
the connector reports a pending key as a hit that is not readable yet, so
`store`, `stats`, and `reset` first reclaim pending reservations older than
`--pending-timeout-secs` (default 60; 0 disables). Each reclaimed request is
logged as `reservations_reclaimed` with the namespace, key count, and age. A
`commit` that arrives after its reservation was reclaimed is rejected like any
other commit without a pending reservation, so keep the timeout above the slowest
store a worker performs.

`release` and `release_request` still succeed on unknown input, but report what
they ended: `{"ok": true, "released": N, "unknown": M}` counts the leases
dropped and the keys or request ids that held none.

`stats` reports the namespace state (`entries`, `pending`, `leased`,
`free_chunks`, `bytes`, `num_chunks`, `requests` for requests currently holding
a lease) plus counters that are monotonic since the namespace opened and survive
`reset`: `hits`, `misses`, `pending_hits`, `evictions`, `stores`, `commits`,
`aborted`, `reclaimed` in keys, and the rejected or failed `commit` calls in
`failed_commits`. A leaked lease or reservation shows up there instead of only as
a stalled cache.

A `reset` refuses a cache with a pending reservation or a lease. `reset` with
`"force": true` instead drops every lease and reservation, clears the slots, and
empties the cache; a caller may only send one after flushing every transfer whose
slot it abandons. The connector resets in two phases: `reset_cache()` releases
the leases it tracks, and the forced `reset` follows from `on_reset_flushed()`
once every worker reported the transfers the reset flushed. It keeps the engine
stepping until then, because a dropped slot may not be recycled while a worker
could still write to it.

Eviction takes the least recently used key: a `lookup` hit and a `touch` both
promote a key, so a shared prefix that is read again outlives keys that nothing
has looked up since they were stored. A `store` batch never evicts a key it
contains, and a reservation is not a victim until it is committed.

A rejection (`{"ok": false, "error": ...}`) keeps the session and its namespace
alive: an invalid key or request id, an unknown namespace, a commit without a
matching reservation, a `store` batch too large for one control frame, and an
unknown handshake field all leave the caller free to retry or give up without
losing the cache. Only frame-level failures (bad length, non-JSON, unknown op)
and I/O errors close a connection. Attached workers may read `stats`; every
other command mutates scheduler-owned metadata and is rejected for them.
`store` and `commit` both carry the request that owns the reservation, so a
request id must be the same one across the whole lifetime of that request.
No reply is written while the namespace registry is locked, so a client that
stops reading blocks only its own connection, for at most the 5 s write timeout.

Disconnecting an attached client still invalidates its namespace. Old mappings
are never reused by a new namespace, and remain charged to capacity while
attached clients retain their sessions. A request id is digested before it goes
on the wire, so any user-supplied id fits the 256-byte limit; the service sees
one stable 32-character id per request.

A control RPC that fails at the transport level closes the client. A read-only
`lookup` reconnects and repeats itself once, and every other operation stops
offloading for that engine instead of failing the scheduler step: a lookup
becomes a miss and reservations, completions, touches, and resets become
no-ops. A rejection after such a reconnect disables offloading the same way,
because a namespace the service dropped cannot serve the request again. A reply
that breaks the slot contract raises instead, since it means a service bug and
not a lost connection. No lookup ever reports a hit the service did not take a
lease for.

## Lifecycle

`SIGTERM` and `SIGINT` stop the accept loop, unlink the socket path, and exit
with status 0, so a supervisor can restart the service without cleaning up
first. After a process was killed without that chance, the next start probes the
path instead of failing: a service that still accepts connections is refused
with `another service is listening on <path>`, while a socket that refuses
connections is unlinked and rebound. Any other probe error refuses the start
with the operating-system error.

`--max-connections` (default 128) bounds simultaneous clients. A connection over
the limit is accepted and closed at once, so the excess client fails fast
instead of waiting in the listen backlog. `--idle-timeout-secs` (default 900; 0
disables it) closes a connection that sends no request for that long, which
reclaims the session of a client that died without closing its socket. A closed
idle connection is indistinguishable from any other dropped connection on the
client side, so keep the timeout above the engine's longest idle period or
disable it with 0. One connection is one thread and one descriptor, so both
limits bound the process. Transient accept failures (`EMFILE`, `ENFILE`,
`ENOMEM`) are logged and retried after a short backoff rather than ending the
service. `--pending-timeout-secs` (default 60; 0 disables) bounds how long a
reservation may stay pending; see Ownership.
Every event writes one line to stderr, starting with `kv-cache-service: start`
(socket, capacity, pid, and both limits) and ending with `shutdown`: namespace
open and attach, retirement with the sessions left, the reclaimed socket, and
rejected or failed requests with the operation name. Per-lookup paths log
nothing.

A request loop runs under `catch_unwind`, so a build with unwinding answers
`{"ok": false, "error": "internal error"}` and closes only that connection, and
a poisoned registry or cache lock is recovered instead of wedging every
namespace. This workspace builds its binaries with `panic = "abort"`, where a
panic ends the process instead; the socket path it leaves behind is reclaimed by
the next start.

## Hash reporting

Hash reporting is optional and off by default. Pass `--rswap-attach` (or
`--hash-attach`), optionally with a sysfs path that defaults to
`/sys/kernel/mm/xswap/rswap_attach`, to enable it. It requires an
rswap-capable kernel (`CONFIG_RSWAP`) with `BPF_PROG_TYPE_SYSCALL` support.
Cargo builds `bpf/rswap_hash.bpf.c` with clang when available; without clang,
`--rswap-attach` is unavailable at runtime.

At namespace open, the service loads the embedded BPF object with CAP_BPF and
attaches it to the memfd VMA with CAP_SYS_ADMIN by writing
`"<vaddr-hex> <prog-fd>"` to the sysfs path, for example:

```bash
echo "<vaddr-hex> <prog-fd>" > /sys/kernel/mm/xswap/rswap_attach
```

The program receives `{vaddr, hash[48]}` and returns non-zero only when it has
a digest; the kernel copies the 48-byte hash into the swap cluster before the
mapping disappears. The program runs under the page table lock, so it must not
sleep or fault, and it calls no helpers: it reads arena memory directly.

The service owns one `BPF_MAP_TYPE_ARENA` per namespace, mapped at a
4 GiB-aligned VA. Arena page 0 holds the geometry header and each following 64-byte
entry describes one base page: a single `AtomicBool` ready flag, the 48-byte
hash, and the absolute page index. Userspace publishes by writing the hash
while the flag is false and then storing the flag with `Release`; the program
acquires `BPF_F_ARENA_SCALAR`, so the arena address is a scalar and its loads
are rewritten to `BPF_PROBE_MEM32` arena accesses. The arena VA is baked into
the program at load time, and the arena fd is bound through `fd_array`.

This is the same layout and publication protocol the eBPF arena POC uses. Only
touched entries fault arena pages in, so arena memory grows with published
pages. The 48-byte hash is a 36-byte canonical chunk key, a 4-byte
little-endian original key length whose high bit marks malformed public input,
and an 8-byte little-endian chunk-relative page offset in pages at the end.
Shorter keys are zero-padded to 36 bytes and longer keys use a SHA-256
fingerprint; the length field disambiguates them. Publishing happens on
successful commit; failed commits, evictions, resets, and namespace disconnect
clear the ready flags. Namespace disconnect writes `"<vaddr-hex> 0"` to
detach.

Attachment is per-VMA in the writing task's mm: each process that maps the
memfd needs its own attachment, and workers are a TODO. This metadata-only
option changes no wire responses, geometry fields, or worker mappings, even
when enabled, and model output is unaffected.

## Scope

- Linux, trusted same-UID local clients, single-node TP-only.
- Complete physical packed rows, including DeepSeek MLA/indexer bytes and padding.
- Existing model replay for non-cacheable compressor and bounded-replay state.
- No secondary tiers, persistent cache, cross-engine cache reuse, KV events,
  context parallelism, pipeline parallelism, or shared-copy TP deduplication.
- Bounded synchronous control RPCs are a correctness-first MVP. Batch/pipelined
  lookups, background liveness checks and CUDA-IPC service-side DMA are future work.
- A `store` batch whose reply cannot fit in one 1 MiB frame is rejected with
  `batch too large for one control frame` before any slot is reserved.

## Tests

```bash
cargo test --manifest-path rust/Cargo.toml -p vllm-kv-cache-service
.venv/bin/python -m pytest --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/test_service.py -v
```

On a machine booted into an rswap-capable kernel (`CONFIG_RSWAP`) with an rswap
backend configured, `test_service_rswap.py` also covers the kernel side: it
pushes pages through the service, reclaims exactly those pages into a private
rswap device, and checks that each one is stored under its published digest and
reloads unchanged. It needs root and an idle swap setup, so it is opt-in:

```bash
sudo VLLM_TEST_KV_SERVICE_RSWAP=1 .venv/bin/python -m pytest \
  --confcutdir=tests/v1/kv_offload tests/v1/kv_offload/test_service_rswap.py -v

# Larger footprint (chunks x 4 pages of the host page size):
sudo VLLM_TEST_KV_SERVICE_RSWAP=1 VLLM_TEST_KV_SERVICE_RSWAP_CHUNKS=256 \
  .venv/bin/python -m pytest --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/test_service_rswap.py -v
```

With the service running and CUDA model weights available:

```bash
VLLM_TEST_DEEPSEEK_V41_MODEL=/path/to/DeepSeek-V4.1-Flash \
VLLM_TEST_DEEPSEEK_V41_TP=8 \
VLLM_TEST_KV_CACHE_SERVICE_SOCKET=/tmp/vllm-kv-service/cache.sock \
.venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_offloading_connector.py \
  -k 'deepseek_v41_mmap and rust' -v
```

CPU subprocess tests verify the actual protocol and shared memory, not GPU DMA
or model-output correctness.
