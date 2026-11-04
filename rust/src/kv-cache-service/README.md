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

Disconnecting any attached client invalidates the namespace. Old mappings are
never reused by a new namespace, and remain charged to capacity while attached
clients retain their sessions. Service errors fail the connector closed; this
version does not reconnect or silently treat a failed receive as a hit.

## Scope

- Linux, trusted same-UID local clients, single-node TP-only.
- Complete physical packed rows, including DeepSeek MLA/indexer bytes and padding.
- Existing model replay for non-cacheable compressor and bounded-replay state.
- No secondary tiers, persistent cache, cross-engine cache reuse, KV events,
  context parallelism, pipeline parallelism, or shared-copy TP deduplication.
- Bounded synchronous control RPCs are a correctness-first MVP. Batch/pipelined
  lookups, background liveness checks and CUDA-IPC service-side DMA are future work.

## Tests

```bash
cargo test --manifest-path rust/Cargo.toml -p vllm-kv-cache-service
.venv/bin/python -m pytest --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/test_service.py -v
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
