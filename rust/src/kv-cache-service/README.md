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

An existing socket is not overwritten. After stopping the service, remove its
stale socket before restarting. Use a private directory owned by the service
user; only same-UID clients are accepted.

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

A rejection (`{"ok": false, "error": ...}`) keeps the session and its namespace
alive: an invalid key or request id, an unknown namespace, a commit without a
matching reservation, a `store` batch too large for one control frame, and an
unknown handshake field all leave the caller free to retry or give up without
losing the cache. Only frame-level failures (bad length, non-JSON, unknown op)
and I/O errors close a connection. Attached workers may read `stats`; every
other command mutates scheduler-owned metadata and is rejected for them.
No reply is written while the namespace registry is locked, so a client that
stops reading blocks only its own connection, for at most the 5 s write timeout.

Disconnecting an attached client still invalidates its namespace. Old mappings
are never reused by a new namespace, and remain charged to capacity while
attached clients retain their sessions. Service errors fail the connector
closed; this version does not reconnect or silently treat a failed receive as a
hit.

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
