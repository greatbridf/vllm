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
An engine asks the `info` op for `--capacity-bytes` while it starts and trims its
own `num_chunks` to that budget, so a `cpu_bytes_to_use` the service cannot hold
costs a warning and a smaller cache instead of a failed `open`; a budget too
small for one row stops the engine with the minimum to reserve. The realized
geometry is logged once at INFO, so the engine log matches what the service
allocated.
Workers attempt host pinning through the existing DMA worker and register only
their own slice of every chunk, so pinned memory stays one slice per worker
instead of the whole multi-rank region; pageable fallback does not guarantee
that Linux keeps all pages resident in DRAM. Sizing has the arithmetic behind
`cpu_bytes_to_use`.

Pinning is the one setting that decides whether kernel-managed offload can run.
`cudaHostRegister` page-locks the range, and the kernel excludes locked ranges
from reclaim: `MADV_PAGEOUT` answers `EINVAL` and the rswap digest program never
sees those pages, so the remote tier stays empty. Set
`"pin_host_memory": false` in `kv_connector_extra_config` when the cache must
stay migratable for rswap; transfers become pageable, which is slower but
correct, and `pin_host_memory` defaults to `true`, preserving the faster pinned
copy. The two behaviours are pinned by `test_service_rswap_offload.py`
(`test_locked_region_is_not_reclaimable`, `test_released_pinning_skips_host_registration`,
and a store/swap-out/load round trip through a mock engine).

The service locks the same way for its own leases when `--pin-lease-bytes` is
set, so a deployment that wants kernel-managed offload keeps that flag at its
default `0`: a row locked by a lease is not reclaimable either.

## Deployment

Run the service under a supervisor. A `systemd` unit:

```ini
# /etc/systemd/system/vllm-kv-cache.service
[Unit]
Description=vLLM Rust KV cache service
After=local-fs.target

[Service]
Type=simple
User=vllm
RuntimeDirectory=vllm-kv-service
RuntimeDirectoryMode=0700
ExecStart=/usr/local/bin/vllm-kv-cache-service \
    --socket /run/vllm-kv-service/cache.sock \
    --capacity-bytes 68719476736
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
```

`RuntimeDirectory` creates `/run/vllm-kv-service` owned by `User` with mode
`0700` and removes it when the unit stops, which is the layout the service
requires: it refuses to start unless the socket's parent is a directory owned by
its own UID with no group or other access, and it creates the socket with mode
`0600`. The start fails with `socket directory must be private (mode 0700)` when
the directory grants group or other access. Peers are authenticated by UID, so
the engine must run as `User`.

The unit needs no cleanup step. `SIGTERM` from `systemctl stop` unlinks the
socket and exits 0, so `Restart=on-failure` restarts the service only after a
failure; a process that was killed leaves its socket behind and the next start
reclaims it (see Lifecycle). Changing `--capacity-bytes` needs a restart of the
service, which drops every namespace. Every event of the service is one line in
the journal (`journalctl -u vllm-kv-cache.service`).

`--rswap-attach` needs BPF privileges, which the unit grants explicitly; leave
both lines out when hash reporting is off, because the service needs no
capability otherwise.

```ini
CapabilityBoundingSet=CAP_BPF CAP_SYS_ADMIN
AmbientCapabilities=CAP_BPF CAP_SYS_ADMIN
```

Build the binary with

```bash
cargo build --release --manifest-path rust/Cargo.toml -p vllm-kv-cache-service
```

which writes `rust/target/release/vllm-kv-cache-service`; install it where
`ExecStart` points. One service can serve every engine on the node: each engine
opens its own namespace, so an engine needs a distinct `engine_id` and its share
of `--capacity-bytes`.

## Sizing

One engine reserves one row per chunk:

```text
worker_bytes    = blocks_per_chunk x worker_kv_bytes_per_block
row_bytes       = round_up(worker_bytes x world_size, page_size)
num_chunks      = cpu_bytes_to_use // row_bytes
namespace_bytes = row_bytes x num_chunks
```

`worker_bytes` is one rank's slice of a chunk and every rank stores its own copy,
so a row holds `world_size` slices rounded up to the host page size (`4096` bytes
on most hosts, `16384` on a 16 KiB-page aarch64 kernel). The padding is bounded
by one page per row, so a chunk whose row is smaller than a page wastes most of
its reservation: a 100-byte per-rank chunk over 8 ranks is one 800-byte row,
charged as a whole page.

The reservation is charged for as long as the namespace exists, filled or not,
and `cpu_bytes_to_use` must hold at least one row. For example 64 GiB over 8
ranks with 2 MiB of KV per rank and chunk gives `row_bytes = 16 MiB` and
`num_chunks = 4096`.

`--capacity-bytes` must cover every engine's reservation plus one more:

```text
sum over engines of row_bytes x num_chunks, plus one engine's full reservation
```

The extra reservation covers an engine that starts while its predecessor still
holds sessions: a namespace whose worker disconnected is retired and stays
charged until its last session leaves, and a namespace whose scheduler
disconnected stays detached and adoptable until `--namespace-grace-secs` runs
out (see Ownership and Failure modes).

A size that does not fit shows up in one of three ways:

- `cpu_bytes_to_use` below one row fails the engine at startup with the minimum
  to reserve, and an aggregate budget below one row fails it with the
  `--capacity-bytes` to raise.
- A `cpu_bytes_to_use` above what `--capacity-bytes` holds is trimmed with a
  warning to the chunks the budget holds. That cap compares against the whole
  budget rather than against what other namespaces already reserved, so a second
  engine can still be rejected by `open` with `cache exceeds service capacity`,
  which fails its startup.
- Inside a namespace a full cache evicts. Only when no chunk can be freed,
  because every one is leased, reserved, or part of the store batch itself, does
  `store` answer `{"ok": true, "full": true}` and the engine keeps those chunks
  in GPU memory only.

`--pin-lease-bytes` is sized by the working set, not by the cache. A locked row
cannot be reclaimed, so a `memory.max` close to the cache size pushes the pressure
onto the other memory charged to the same cgroup and can OOM the writer instead.
Size the flag to roughly the bytes the engines keep leased at once, and leave
`memory.max` headroom above the cache size for the rest of the process.

## Protocol

Every message is one frame: a 4-byte big-endian length and that many bytes of
JSON. Both directions are bounded by 1 MiB. A request is an object whose `op`
field names one of the operations below; an unknown operation, an unknown field,
a non-JSON payload, a length of 0, and a length above the limit are frame-level
failures. Requests carry no ids, so replies are ordered: the service answers one
connection's requests in arrival order, exactly one reply per request. A client
may therefore write a burst of frames before reading any reply and then read one
reply per request in the same order; `CacheServiceClient.pipeline` does exactly
that, and a burst costs one round trip instead of one per request.

The connector batches the calls whose reply nothing in its step depends on: a
completed store queues one `commit` per 4096-key batch, a finished request queues
its `release_request`, and `touch` queues its batches, and the manager writes that
queue as one burst from `on_schedule_end`, once per scheduler step. A step therefore
costs one write burst and one read burst for its completions and releases instead
of two or three round trips per request. A step whose lookups answered `pending`
writes one more burst after it, the `prefetch` of those keys, so the readahead of
a chunk the kernel moved out starts at the step boundary instead of inside the
deferred load that reads it. The calls whose answer the step needs --
`lookup`, `lookup_many`, `load`, `store` -- stay one request each, so a key stored
earlier in a step is published at the end of that step and a `lookup` for it in the
same step still reports `pending`. The manager flushes the queue before a `store`,
a `stats`, a `reset`, and a `close`, so a call that depends on what the queue still
owes never overtakes it. A `load`, a `commit`, and a `touch` whose chunks split
above 4096 keys write one burst for that operation; a `lookup_many` asks one batch
at a time, because a scan that stopped at a miss decides whether the next batch is
asked for at all, and a `store` stops at the first batch the cache cannot fit.

A burst holds no handshake. An `attach` reply is a frame followed by one
descriptor message, which must be received with its marker byte, so the handshake
cannot be followed by other requests on that stream; `open` and `close` are
excluded with it, because a burst is a set of operations on the namespace the
session already holds. A client that sends one more request while a burst is
unread would read the next reply as that request's, so a single request may be
sent only once the burst is collected.

| `op` | Fields | Sent by | Reply |
| --- | --- | --- | --- |
| `open` | `version`, `namespace`, `fingerprint`, `worker_bytes`, `world_size`, `num_chunks`, `adopt` (default `false`) | scheduler | the namespace geometry |
| `attach` | `version`, `namespace`, `fingerprint`, `rank` | worker | the namespace geometry, then one shared-memory descriptor |
| `lookup` | `key`, `request` | scheduler | `status` |
| `lookup_many` | `keys`, `request`, `until_miss` (default `false`) | scheduler | `statuses`, one per key, ending at the first `miss` when `until_miss` is set |
| `load` | `keys`, `request`, `prefetch` (default `true`) | scheduler | `slots`, one per key, and `prefetched`, the pages the kernel was asked to read ahead |
| `prefetch` | `keys` | scheduler | `pages` the kernel was asked to read ahead |
| `release` | `keys`, `request` | scheduler | `released`, `unknown` |
| `release_request` | `request` | scheduler | `released`, `unknown` |
| `store` | `keys`, `request` | scheduler | `keys`, `slots`, `evicted`, or `full` |
| `commit` | `keys`, `request`, `success` | scheduler | `ok` only |
| `spill` | `keys` | scheduler | `pages` written out |
| `reset` | `force`, default `false` | scheduler | `ok` only |
| `stats` | none | scheduler or worker | `stats`, `sessions`, `detached` |
| `close` | none | scheduler | `ok` only |
| `info` | none | anyone, before any handshake | service state |

`load` prefetches by default: after it resolves the slots it asks the kernel to
read those chunks back in with one `MADV_WILLNEED` per contiguous row, on its own
thread so the reply is never delayed. A swap-in is submitted per fault and the
faulting task then blocks on the folio, so a task that faults page after page
keeps exactly one request in flight; the prefetch is what lets the kernel overlap
the remote round trips (measured: `read_peak_inflight` 1 before, 8 after).

`prefetch` asks for that readahead earlier: it resolves the committed chunks
`keys` name and hands every whole row to the kernel on a thread of its own, so a
client can start the remote reads at the end of one schedule step instead of
inside the load of the next. It takes no lease, so it pins nothing: a `store` in
between may still evict one of its keys, and the readahead of a slot that has
since been recycled reads bytes nobody needs. It rejects a key that is missing or
not committed, which is what makes the keys a lookup answered `pending` for safe
to send. The reply counts the pages that were advised, not the pages the kernel
fetched, exactly as `load`'s `prefetched` does.

`spill` asks the kernel to write committed chunks out through the swap path
without waiting for memory pressure: it faults each page of the chunk into this
process's page tables first, because the digest is resolved from the VMA that
carries the rswap program and the reverse-mapping walk stops as soon as a folio
is no longer mapped, and then calls `madvise(MADV_PAGEOUT)` on the chunk.
It is the service's own mapping that both holds those entries and carries the
attach, so the service has to be the caller; a worker's mapping cannot do it.
`MADV_PAGEOUT` is not a usable trigger for this mapping: measured on a 7.3 rswap
kernel it reports success while nothing is reclaimed (`/proc/swaps` `xswap<N>`
`Used` stays 0) and no digest program run is recorded, whereas an anonymous
mapping of the same size writes every page out. So the response `pages` is what
was advised, not what reached the backend. A row a lease holds locked under
`--pin-lease-bytes` is excluded from reclaim the same way, so `MADV_PAGEOUT`
answers `EINVAL` for it: spill only the chunks no request leases.

Cgroup v2 reclaim is the trigger that works: charge the KV pages to a cgroup of
its own and write `memory.reclaim`. At the DeepSeek-V4.1-Flash geometry
(4.92 MiB per chunk across TP=8) one run of 7 chunks moved 2 909 pages / 46.5 MiB
into the backend at ~22.8 ms per 32-page batch. Read back is what fails on that
kernel: `rswap: kernel_connect ... failed: -4` (EINTR) followed by
`rswap: read type 1 failed: -4` and `Read-error on swap-device`, which kills the
task that touched the page, and `read_errors`/`redis_retries` climb. Do not use
this path in production until the read side retries and reports instead of
faulting.

`open` and `attach` are the only handshake operations, and a session holds at
most one namespace. `open` creates a namespace under a caller-chosen name, with
the caller's 64-hex layout fingerprint and the geometry it will reserve;
`attach` joins that namespace at one rank and receives the descriptor of its
mapping. After the handshake the session that opened the namespace owns every
mutating operation, while an attached worker may send `stats` and nothing else.
An `attach` reply is the frame followed by one 1-byte message that carries one
`SCM_RIGHTS` descriptor, so a client must receive it before it sends another
request. `info` needs no handshake and is the only operation that reports the
service rather than one namespace.

`close` retires the namespace the caller owns, which is how the connector stops
the engine without depending on what its last disconnect happened to leave
behind (see Ownership). `open` with `"adopt": true` may only adopt a detached
namespace and never creates one: a client that reconnects replays its handshake
that way, so a namespace the service already dropped degrades the connector
instead of pointing it at a second mapping while the workers still hold the
first.

A `stats` reply, one counter set per namespace, plus the sessions using it and
whether its owner left it adoptable:

```json
{"ok": true, "stats": {"entries": 0, "pending": 0, "leased": 0, "resident": 0, "remote": 0, "free_chunks": 2, "requests": 0, "pinned_slots": 0, "pinned_bytes": 0, "pin_skipped": 0, "pin_failures": 0, "bytes": 32768, "num_chunks": 2, "hits": 0, "misses": 0, "pending_hits": 0, "evictions": 0, "stores": 0, "commits": 0, "aborted": 0, "failed_commits": 0, "reclaimed": 0}, "sessions": 2, "detached": false}
```

An `info` reply, which an operator or a probe can read before any engine opened a
namespace, where `detached` counts the namespaces no owner holds:

```json
{"ok": true, "version": "0.1.0", "pid": 81470, "uptime_ms": 2710, "capacity_bytes": 68719476736, "used_bytes": 0, "retired_bytes": 0, "namespaces": 0, "detached": 0, "sessions": 0, "max_connections": 128}
```

A frame-level failure closes the connection without a reply, which ends that
session as any disconnect does (see Ownership): an owner's namespace stays
adoptable until its grace period ends, and a worker's is retired. A request-level
failure answers
`{"ok": false, "error": "..."}` and leaves the session and its namespace alive;
Ownership lists what a rejection covers and what it leaves to the caller.

## Ownership

The scheduler opens a unique engine namespace; workers attach with matching
layout fingerprints and receive a descriptor through `SCM_RIGHTS`. Each chunk
has consecutive worker slices. Prefix keys include the cache-group identifier.
The service reserves missing keys atomically and publishes them only when the
scheduler receives completion from all workers.

Lookup acquires a request lease before reporting a hit, so a later allocation
cannot evict the load source. Request leases are released after finish and all
pending transfers. Active leases and pending stores cannot be reset or evicted.

A hit also means readable now. A committed chunk whose row is not entirely in
DRAM answers `pending` instead of `hit`, because reading it would take a fault
per page, and that answer takes no lease: a lease on a chunk that cannot be read
yet is what would stop a later allocation from replacing it with a chunk that can.
The check is one `mincore(2)` over the chunk's row, so `pending` covers both
"reserved and not published" and "published but not resident".

A hit also stays readable. The residency check answers for the moment it runs, so
`--pin-lease-bytes` has the service `mlock(2)` the row of a fresh hit before that
hit is reported, which makes the kernel keep it: a `HIT` is then readable from
DRAM until the last request leasing it releases, whatever the cgroup's
`memory.max` does in between. `load` locks the rows it names again, which costs
nothing for a row already locked. Locking a row is one `mlock(2)` over its pages
(4096 of them for a 16 MiB DeepSeek row), so an opting-in deployment pays that
once per chunk it starts reading and the budget is what bounds it. A lock ends
with the last reader's `release` or `release_request`, with a forced `reset`, and
with the slot's teardown, so no lock outlives the chunk it belongs to.

The flag is a ceiling and never an eviction policy: a pin that would pass it is
skipped, the hit stands without the guarantee, and no other request's lease is
ever dropped to make room. `0`, the default, pins nothing and leaves a `pending`
answer as the only protection. A lock or unlock the kernel refuses (no
`CAP_IPC_LOCK`, a `RLIMIT_MEMLOCK` below one row) is logged once as `pin_failed`
and counted, never fatal.

`lookup` reports the status of one key. `lookup_many` reports the same for a
batch of up to 4096 keys, as `{"ok": true, "statuses": [...]}` with one status
per key in the order they were sent, and takes the same lease and LRU promotion
per hit. With `until_miss` the scan ends at the first `miss`: the reply is then a
prefix of the batch whose last status is that `miss`, and only the hits before it
hold a lease, so a chunk behind the gap stays evictable and is not carried as a
lease until the request finishes. The connector asks a full-attention prefix that
way, in batches of 4096 keys, so a prefix scan leases exactly the chunks it
consumes. A sliding-window tail scan asks without the flag, because it needs the
misses and the hits behind them, and holds those leases until the request ends.

A reservation belongs to the request that made it: `store` records the owner, and
`commit` rejects a batch reserved by another request with
`commit does not own the reservation` and leaves those reservations pending. A
reservation nobody commits would stay pending for the life of the namespace, and
the connector reports a pending key as a hit that is not readable yet, so
`store`, `stats`, and `reset` first reclaim pending reservations older than
`--pending-timeout-secs` (default 60; 0 disables). Each reclaimed request is
logged as `reservations_reclaimed` with the namespace, key count, and age. A
`commit` that arrives after its reservation was reclaimed is rejected like any
other commit without a pending reservation; the connector logs one warning and
treats that store as one that did not publish, so keep the timeout above the
slowest store a worker performs.

`release` and `release_request` still succeed on unknown input, but report what
they ended: `{"ok": true, "released": N, "unknown": M}` counts the leases
dropped and the keys or request ids that held none. A lease has no timeout of
its own: it ends with `release`, `release_request`, a forced `reset`, or with the
namespace, so a request the connector never finishes keeps its chunks out of the
eviction order until the engine stops.

`stats` reports the namespace state (`entries`, `pending`, `leased`,
`free_chunks`, `bytes`, `num_chunks`, `requests` for requests currently holding
a lease, and the `pinned_slots` and `pinned_bytes` a lock holds) plus counters
that are monotonic since the namespace opened and survive `reset`: `hits`,
`misses`, `pending_hits`, `evictions`, `stores`, `commits`, `aborted`, `reclaimed`
in keys, the rejected or failed `commit` calls in `failed_commits`, and the pins
the budget skipped or the kernel refused in `pin_skipped` and `pin_failures`.
Where the kernel holds the entries is reported beside them:
`resident` counts the chunks whose whole row is in DRAM and `remote` the rest
(`entries - resident`), so an operator can see how much of the cache a memory
limit has moved out. Beside that object it reports the `sessions` using the
namespace and whether its owner left it `detached`. A leaked lease or reservation
shows up there instead of only as a stalled cache. `stats` probes the residency of
every entry, so it costs one `mincore(2)` over each row on top of the metadata it
counts.

`info` needs no handshake and reports the service rather than one namespace:
`version`, `pid`, `uptime_ms`, `capacity_bytes`, `used_bytes` (live plus retired
mappings), `retired_bytes`, `namespaces` (live ones), `detached` (live ones no
owner holds), `sessions` (live and retired), and `max_connections`. A probe or an
operator can call it before any engine opened a namespace; like every other
command it still requires a same-UID peer.

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
alive: an invalid key or request id, an unknown namespace, incompatible handshake
fields, a commit without a matching reservation, and a `store` batch too large
for one control frame all leave the caller free to retry or give up without
losing the cache. Only frame-level failures (bad length, non-JSON, unknown op or
field, a short frame) and I/O errors close a connection. Attached workers may
read `stats`; every other command mutates scheduler-owned metadata and is
rejected for them.
`store` and `commit` both carry the request that owns the reservation, so a
request id must be the same one across the whole lifetime of that request. The
connector digests an id before it goes on the wire, so any user-supplied id fits
the 256-byte limit and the service sees one stable 32-character id per request.
No reply is written while the namespace registry is locked, so a client that
stops reading blocks only its own connection, for at most the 5 s write timeout.

A session is one of three kinds, and its kind decides what its departure costs
the namespace:

- an **owner**, the session that sent `open`,
- a **worker**, a session that sent `attach` for one rank,
- a **probe**, a session that has done neither and may only send `info`.

A worker's departure retires its namespace: that rank may still have DMA in
flight into slots nobody may recycle, so every owner call answers
`cache namespace invalidated` from then on, and the connector degrades. A
retired mapping is never adopted, never handed to a new namespace, and stays
charged to capacity until its last remaining session leaves.

An owner's departure alone only detaches the namespace. The mapping, every
published entry, and every lease stay, and a later `open` with the same name,
the same fingerprint, and the same geometry adopts them: it binds the new session
as owner, answers with the geometry of the mapping it adopted, and leaves the
entries and leases intact, so a worker keeps `attach`ing to the mapping it already
has. A re-`open` with another fingerprint or another geometry is rejected with
`namespace geometry mismatch` instead of reusing a mapping it does not describe,
and adoption is refused with `namespace already owned by another scheduler`
while the previous owner session is still connected.

A detached namespace keeps its workers and its memory, so it cannot stay forever:
`--namespace-grace-secs` (default 300; 0 disables) reclaims it that long after it
detached, which frees the mapping of a scheduler that crashed and never came back
even while its workers are still attached. Every transition is one log line:
`namespace_detached` when the owner leaves, `namespace_adopted` when another
session takes the namespace over, `namespace_retired` when a worker's departure
keeps the mapping for its remaining sessions, `namespace_invalidated` when the
mapping is freed instead because no session is left to hold it, and
`namespace_reclaimed` when the grace period ends. `stats` reports the `sessions` of a namespace and whether it is
`detached`, and `info` counts the `detached` namespaces of the service.

`close` retires the namespace at once, which is what a scheduler shutting down
sends: `ServiceOffloadingManager.shutdown()` sends `close` before it closes the
socket, so a stopped engine leaves nothing for a later engine to adopt, and a
worker keeps its own disconnect, which retires the namespace once its DMA is
drained.

A control RPC that fails at the transport level closes the client. A read-only
`lookup`, `lookup_many`, or `stats` reconnects and repeats itself once: the new
connection replays the `open` handshake as an adoption, so the retried call reads
the cache the engine already stored instead of failing on a namespace only its
old session knew. A replay that has nothing to adopt, because the namespace was
retired or its grace period ended, is rejected and disables offloading, since a
second mapping the workers do not hold would be worse than no offloading at all.
Every other operation stops offloading for that engine instead of failing the
scheduler step: a lookup becomes a miss, a store reservation becomes `None`, and
completions, touches, and resets become no-ops. A rejection disables offloading
the same way, including one that reaches the store path, which has reserved
nothing. `prepare_load` is the one operation that raises, because the connector
has already committed to the tokens its lookup reported and loading from a source
the service cannot name is worse than a failed step. A reply that breaks the slot
contract raises instead, since it means a service bug and not a lost connection.
No lookup ever reports a hit the service did not take a lease for.

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
reservation may stay pending; see Ownership. `--namespace-grace-secs` (default
300; 0 disables) bounds how long a namespace whose owner left stays adoptable;
the accept loop's own poll sweeps it, so no thread or timer exists for that.
`--pin-lease-bytes` (default 0; 0 disables pinning) bounds the bytes of leased
rows the service may lock, per namespace; see Ownership and Sizing.
Every event writes one line to stderr, starting with `kv-cache-service: start`
(socket, capacity, pid, and every limit) and ending with `shutdown`: namespace
open, attach, detach, adoption, retirement, and reclaim, the reclaimed socket,
the first refused `mlock` or `munlock` as `pin_failed`, and rejected or failed
requests with the operation name. Per-lookup paths log nothing.

A request loop runs under `catch_unwind`, so a build with unwinding answers
`{"ok": false, "error": "internal error"}` and closes only that connection, and
a poisoned registry or cache lock is recovered instead of wedging every
namespace. This workspace builds its binaries with `panic = "abort"`, where a
panic ends the process instead; the socket path it leaves behind is reclaimed by
the next start.

## Monitoring

The service has no metrics endpoint. It writes one line per event to stderr (see
Lifecycle) and answers two read-only operations, `info` for the service and
`stats` for one namespace. `info` needs no handshake, so any same-UID process can
probe it, while `stats` needs a session in the namespace.

```bash
.venv/bin/python - <<'PY'
from vllm.v1.kv_offload.service.client import CacheServiceClient

client = CacheServiceClient("/run/vllm-kv-service/cache.sock")
print(client.call("info"))
client.close()
PY
```

`info` reports `capacity_bytes`, `used_bytes` (live plus retired namespaces),
`retired_bytes`, `namespaces`, `detached`, and `sessions`. A `used_bytes` that
stays at its value after every engine of a restart is gone, with `retired_bytes`
equal to it, means a worker of the old engine still holds its session. A
`detached` above 0 means a scheduler left a namespace adoptable, and the grace
period is what eventually frees it. The service has no aggregate `stats`: one
namespace is one `stats` call.

`stats` reports one namespace, and its counters show a leak before the cache
stalls:

| Reading | Meaning |
| --- | --- |
| `leased`, `requests` | Chunks with a live request lease, and the requests holding them. A leased chunk cannot be evicted, and no lease has a timeout, so a value that does not fall to 0 once the engine is idle is a lease the engine never released. |
| `pending` | Chunks reserved by a `store` and not committed. Sustained non-zero means a worker reserved slots and never committed them; `--pending-timeout-secs` reclaims them and logs `reservations_reclaimed`. |
| `free_chunks` | Chunks no key holds. At 0 with `pending` at 0, a `store` can only evict. |
| `resident`, `remote` | Chunks whose whole row is in DRAM, and the entries whose pages the kernel no longer holds. A `remote` above 0 is a cache a memory limit or a remote tier has pushed out, where a lookup of those keys answers `pending` until they are resident again. |
| `hits`, `misses`, `pending_hits` | Lookup outcomes since `open`, unchanged across `reset`. `pending_hits` counts the answers that were `pending`, both from a reservation and from a row that was not resident. |
| `evictions`, `stores`, `commits`, `aborted` | Chunks evicted, reserved, published, and released without publishing. |
| `failed_commits`, `reclaimed` | Commits rejected (ownership, missing reservation, failed publish) and keys reclaimed by the pending timeout. |
| `pinned_slots`, `pinned_bytes` | Rows a lease holds locked under `--pin-lease-bytes`, and the bytes they charge. They cannot be reclaimed, so a value that does not fall to 0 once the engine is idle is a lock its leases kept; see Sizing before raising the flag. |
| `pin_skipped`, `pin_failures` | Pins the budget refused, and locks or unlocks the kernel refused. A `pin_skipped` that climbs means the flag is below the working set and later hits go without the guarantee; a `pin_failures` that climbs means the host cannot lock memory at all, which `pin_failed` logged once. |
| `sessions`, `detached` | Sessions using the namespace, and whether an owner left it adoptable. `detached` with `sessions` above 0 is a scheduler that is gone while its workers still hold the mapping; `--namespace-grace-secs` is what frees it. |

Engine side, the connector samples one `stats` RPC per collection and reports
`vllm:kv_offload_cpu_cache_usage_perc` (`entries / num_chunks`),
`vllm:kv_offload_cpu_cache_write_usage_perc` (`pending / num_chunks`),
`vllm:kv_offload_cpu_cache_read_usage_perc` (the difference of the two), and one
`vllm:kv_offload_cpu_allocation_size` observation per store batch.
`vllm:kv_offload_config_info` carries the negotiated geometry in
`cpu_num_chunks`, `cpu_blocks_per_chunk`, `cpu_kv_cache_size_bytes`, and
`cpu_kv_cache_size_tokens`, with no `tier` label because this backend reports one
configuration. `vllm:kv_offload_stores_skipped` never appears here, since it is
declared only for a `store_threshold` of 2 or more, which the spec rejects. The
counters that no declared metric carries, such as `evictions` and `reclaimed`,
are logged at debug level.

## Failure modes and recovery

| Symptom | Cause | Recovery |
| --- | --- | --- |
| `socket directory must be private (mode 0700)` at start | the socket's parent grants group or other access, or belongs to another user | use a `RuntimeDirectory` as Deployment shows |
| `another service is listening on <path>` at start | a live service owns the path | stop it, or point `service_socket` at this one; the path was not unlinked |
| `capacity must be positive`, `connection limit must be positive` at start | `--capacity-bytes 0` or `--max-connections 0` | pass a positive bound |
| `service was built without the rswap BPF object` on `open` | `--rswap-attach` on a build without clang | rebuild with clang or drop the flag; the service itself starts |
| `namespace already owned by another scheduler` | another live engine uses the same `engine_id` | give each engine a distinct `engine_id`, or stop the old engine |
| `namespace geometry mismatch` | a re-`open` of an adoptable namespace with another fingerprint or another geometry | use the `engine_id`, layout, and sizing the running namespace was opened with, or wait for its grace period to end |
| `no detached namespace to adopt` | a reconnect replayed its `open` after the namespace was retired or its grace period expired | the engine degraded on purpose; restart it, and size `--namespace-grace-secs` above the longest connection outage it should survive |
| `cache exceeds service capacity` at engine start | `--capacity-bytes` is smaller than the reservations, or a restarting engine's predecessor still holds sessions | raise it and restart the service, or stop the stale engine; see Sizing |
| `cache namespace invalidated` | a worker of that namespace disconnected, the owner sent `close`, or the grace period expired | restart the engine; its cache is cold |
| `scheduler namespace not available` on `attach` | the worker attached after the namespace was invalidated | restart the whole engine, not only the worker |
| `incompatible or duplicate worker` on `attach` | a second attach of the same rank, or a rank that does not match the namespace fingerprint | restart the engine; every rank must attach once, with one layout |
| `workers cannot mutate cache metadata` | an attached worker sent an operation other than `stats` | a client bug; the worker's session and the namespace still live |
| `cannot reset a cache with active transfers or leases` | a `reset` without `force` while a lease or reservation is open | release the leases, or send `{"force": true}` once every transfer whose slot it abandons is flushed |
| `cannot connect to the KV cache service at ...` | the socket is missing, the service is down, or the path or UID is wrong | check the unit, `ls -l` the socket, and `info` |
| `invalid_frame` in the journal, then a closed connection | a bad frame length, a non-JSON payload, or an unknown `op` or field | a client bug or a version mismatch; the session is gone, and an owner's namespace is adoptable only until its grace period ends, so restart the engine |
| one error line, then every lookup misses | a transport failure, or a rejection after the reconnect | offloading is off for that engine until it restarts |
| `internal error` | a panic under `catch_unwind` | that connection closed; restart the engine if it recurs |

Nothing reconnects a mutating operation: a read-only `lookup`, `lookup_many`, or
`stats` repeats itself once on a new connection, which adopts the namespace its
previous session opened, and every other operation degrades as Ownership
describes. A namespace whose ownership was lost for longer than
`--namespace-grace-secs` cannot be adopted again, so its restarted engine starts
cold. Retired mappings are never reused, and a namespace stays charged until the
last of its sessions leaves, so a restart of one engine can fail on capacity
while the previous process is still exiting.

## Hash reporting

Hash reporting is optional and off by default. Pass `--rswap-attach` (or
`--hash-attach`), optionally with a sysfs path that defaults to
`/sys/kernel/mm/xswap/rswap_attach`, to enable it. It requires an
rswap-capable kernel (`CONFIG_RSWAP`) with `BPF_PROG_TYPE_SYSCALL` support.
Cargo builds `bpf/rswap_hash.bpf.c` with clang when available; without clang the
build still succeeds, the flag is still accepted, and every namespace `open`
fails with `service was built without the rswap BPF object`.

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
- No secondary tiers, persistent cache, cross-engine cache reuse, context
  parallelism, pipeline parallelism, or shared-copy TP deduplication.
- Bounded synchronous control RPCs are a correctness-first MVP. Lookups are
  batched through `lookup_many`; the connector does not pipeline its requests,
  and background liveness checks and CUDA-IPC service-side DMA are future work.
- A `store` batch whose reply cannot fit in one 1 MiB frame, counting the keys
  it echoes as reserved and the evicted keys it reports, is rejected with
  `batch too large for one control frame` before any slot is reserved or
  evicted. A `lookup_many` reply carries statuses only, so a batch bounded by
  `MAX_KEYS` always fits.

Numbers the service enforces:

- `world_size` from 1 to 64 per namespace; `worker_bytes` and `num_chunks` of at
  least 1, with `row_bytes x num_chunks` fitting the remaining capacity.
- `num_chunks` at most 2^31 - 1 in the connector, and at most 2^32 - 2 on the
  wire, because `u32::MAX` is the absent-slot marker.
- At most 4096 keys per request. A key is 1 to 512 hexadecimal characters of even
  length and unique within its batch.
- A request id is 1 to 256 bytes on the wire; the connector sends a 32-character
  SHA-256 prefix of the vLLM request id, so any user id fits.
- A namespace name is 1 to 256 bytes, and a layout fingerprint is exactly 64
  hexadecimal characters.
- One frame is at most 1 MiB in either direction.
- One namespace per session, because a session `open`s or `attach`es once, and
  one in-flight request per client: the connector never pipelines.
- `--max-connections` 128, `--idle-timeout-secs` 900,
  `--pending-timeout-secs` 60, `--namespace-grace-secs` 300, and
  `--pin-lease-bytes` 0 by default.

## Performance

Both layers below are CPU-only: no GPU, no root, no rswap. The numbers are one
idle run on an aarch64 Linux VM (Apple silicon host, 11 vCPU, 16 GiB DRAM,
16 KiB pages), so re-measure on the machine you deploy instead of comparing
across machines. Repeated runs spread by about 10 %, except for the groups that
rebuild their fixture at 2^15 entries, where ten samples put the spread at 30 %.

Cache metadata (`cargo bench`):

| bench | shape | cost |
| --- | --- | --- |
| `lookup_hit` | 2^11 / 2^13 / 2^15 entries, rows not faulted in | 209 / 226 / 234 ns |
| `touch_least_recent` | same, promoted entry at the LRU tail | 85 / 86 / 113 ns |
| `store_evict` | 2^13 entries, 1 / 16 / 64 victims | 414 ns / 3.95 µs / 13.9 µs |
| `store_evict` | 2^15 entries, 1 / 16 / 64 victims | 428 ns / 4.36 µs / 14.5 µs |
| `release_request` | 2^13 entries, 1 / 64 leases | 64 ns / 266 ns |
| `release_request` | 2^15 entries, 1 / 64 leases | 66 ns / 378 ns |
| `check_keys` | 1 / 256 / 4096 keys | 20 ns / 10.7 µs / 183 µs |

Promotion, victim selection, and lease release are flat in the entry count: a
2^15-entry cache evicts 64 keys in 14.5 µs. Only `check_keys` grows with its
batch, at 45 ns per key of duplicate detection. The groups that rebuild their
fixture per iteration (`touch_least_recent`, `store_evict`, `release_request`)
promote or evict the true LRU tail; at 2^15 entries that setup dominates each
iteration, so criterion reports the fewer samples it warns about.

`lookup_hit` is the residency check: the fixture's pages are never faulted in,
which is what a cache at a memory limit looks like, so the case measures the
`mincore(2)` over the chunk's row plus the metadata walk that answer `pending`.
That is 170-190 ns of the 209-234 ns, against 38-46 ns before the check existed;
a resident hit adds one lease and one promotion, which `touch_least_recent`
measures. The probe grows with the row: 64 KiB rows here, 4096 pages for a
16 MiB DeepSeek row.

Control plane (`bench` subcommand, 4096 keys, 1 connection):

| case | p50 | p99 | per op |
| --- | --- | --- | --- |
| `rpc_serial_lookup` | 27.7 µs | 30.9 µs | 27.9 µs |
| `rpc_pipelined_lookup` | 0.71 µs | 9.5 µs | 1.30 µs |
| `rpc_lookup_many` | 1656 µs | 2026 µs | 1706 µs (0.42 µs per key) |
| `mixed_schedule_step` | 74.3 µs | 1685 µs | 494 µs |
| `deferred_flush` | 42.9 µs | 86.5 µs | 39.9 µs |
| `stats` | 750 µs | 828 µs | 755 µs |

One lookup round trip costs 28 µs, so a per-chunk lookup costs the scheduler
28 µs per chunk of the prefix. Pipelining the same lookups costs 0.7 µs per op
in the median and about 1.3 µs on average, because the residency probe is a
syscall per chunk and no longer amortizes: a 4096-key `lookup_many` covers a
64 k-token prefix in 1.7 ms, against 0.9 ms before the probe (0.42 µs per key).
`stats` pays the same probe per entry, 750 µs for 8192 entries against 34 µs, so
it is one collection per interval and never a per-lookup cost. Pipelined ops are
charged the time since the previous reply, so the first op of a window also pays
for the window to drain.

`mixed_schedule_step` is a synthetic schedule step of `--requests` requests: each
one scans the prefix, loads its hits, and stores one 64-chunk batch, then the
step's commits and releases are written as one burst, as the manager writes them
at the end of a step. `deferred_flush` is that deferrable tail on its own: one
store per request, then the step's commits and releases. `--flush-immediately`
writes each of those calls as its own round trip instead, which is the shape the
connector had before the burst. Both cases report the `requests` of a step and
the `round_trips` it costs, so the control plane's shape is in the output rather
than in the reader's head:

| requests per step | round trips, one call at a time | round trips, one burst | step cost, one at a time | step cost, burst |
| --- | --- | --- | --- | --- |
| 1 | 5 | 4 | 2.52 ms | 2.52 ms |
| 8 | 40 | 25 | 20.5 ms | 20.1 ms |
| 32 | 160 | 97 | 84.5 ms | 82.6 ms |

The two columns are the same binary on the same cache, so only the wire shape
differs, and a step cost is `us_per_op` times the ops of one step. The deferrable
tail is about 35 % cheaper (`deferred_flush`: 1.19 ms → 0.73 ms for 8 requests,
4.63 ms → 2.89 ms for 32), but a step that also scans a 4096-key prefix is
scan-bound, so the whole step only moves about 2 %, and the residency probe the
scan now makes moved both columns up by about 45 %. What falls everywhere is
the round trips, by `2N - 1` per step: one write and one read per step instead of
two more round trips per request. Per-round-trip cost on this host drifts by up to
2x between runs, so compare the columns of one table instead of two tables.

```bash
cargo bench --manifest-path rust/Cargo.toml -p vllm-kv-cache-service \
  --bench cache -- --warm-up-time 0.2 --measurement-time 0.5
cargo run --release --manifest-path rust/Cargo.toml -p vllm-kv-cache-service \
  -- bench --json --keys 4096 --conns 1 --iterations 500 \
  --assert-max rpc_serial_lookup=200
```

`bench --keys`, `--conns`, `--iterations`, `--requests`, and `--cache-bytes` shape
the run.
It serves real protocol frames from the real dispatch loop on a private socket in
the same process, one namespace per connection, and prints one JSON object per
case (an aligned table without `--json`). Every `--assert-max <case>=<us>` bound
that fails makes the process exit non-zero, so an accidental O(n) path (a 400 µs
eviction instead of 14 µs) or a lost lookup batch fails CI instead of passing as
noise.

## Tests

```bash
cargo test --manifest-path rust/Cargo.toml -p vllm-kv-cache-service
cargo clippy --manifest-path rust/Cargo.toml -p vllm-kv-cache-service \
  --all-targets -- -D warnings
cargo fmt --manifest-path rust/Cargo.toml -p vllm-kv-cache-service -- --check
.venv/bin/python -m pytest --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/test_service.py tests/v1/kv_offload/test_service_manager.py \
  tests/v1/kv_offload/test_service_region.py \
  tests/v1/kv_offload/test_service_client.py -q
.venv/bin/python -m pytest --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/test_factory.py -k rust -q
```

`tests/server.rs` starts the built binary on a private socket, so those runs need
no Python environment, GPU, or root. The arena and program tests in
`tests/hash.rs` print `SKIP: ...` and return without asserting when the kernel
lacks BPF privileges or `BPF_MAP_TYPE_ARENA`, so a green `cargo test` does not by
itself prove the BPF path ran. The Python suites need
`rust/target/debug/vllm-kv-cache-service` (or `VLLM_KV_CACHE_SERVICE_BINARY`) and
skip when it is missing; `test_service_manager.py` and `test_service_region.py`
need neither the binary nor a GPU, and `test_service_client.py` needs it for its
one real-service case and runs the rest of the client against a fake peer.

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

The digest audit compares what the service published with what the rswap backend
stored, so it needs a readable rswap backend; without one the test prints `no
readable rswap backend: N pages stored unaudited` and asserts only the kernel's
page counters and the round trip.

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

## Further reading

- [KV Offloading Usage Guide](../../../docs/features/kv_offloading_usage.md) has the
  `kv_connector_extra_config` reference and what the engine does with this
  backend: metrics, cache-hit source, KV events, and the rejected combinations.
- [Process-Separated KV Offloading](../../../docs/contributing/kv-cache-service.md)
  has the invariants a change must keep and how to validate one.
