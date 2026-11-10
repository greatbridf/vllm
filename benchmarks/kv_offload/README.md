# Simulated KV Offloading

Executed 8 GiB-cache results, including prefetch failures and safety caveats:
[November 10, 2026 report](RSWAP_CONSTANT_RESULTS_20261110.md).

`bench_service.py` compares CPU and process-separated Rust KV paths.
`bench_rswap_constant.py` compares fault-driven reload with service prefetch
under constant arrivals, using an actual rswap kernel and remote Redis backend.
Neither runs an LLM or GPU; NumPy copies replace GPU DMA.

## Constant-Arrival Rswap Comparison

Run from the repository root, after building the Rust service:

```bash
cargo build --manifest-path rust/Cargo.toml -p vllm-kv-cache-service
sudo -n .venv/bin/python -m benchmarks.kv_offload.bench_rswap_constant \
  --output /path/to/new-results-directory
```

Requires root, cgroup v2 with memory control enabled, an rswap/BPF-capable kernel,
and access to the configured Redis backend, including permission to delete
test-owned objects. No other rswap device may be active. The harness creates
and destroys its own device without changing zram.

Default workload:

- 8 GiB service capacity; 1,638 complete 5 MiB rows (7.998 GiB namespace).
- Eight simulated TP rank mappings; 640 KiB per rank per row.
- 1,536 populated rows: 7.5 GiB of real KV bytes, leaving room for tail stores.
- 24 requests at exactly two scheduled arrivals per second; eight copy threads.
  Excess arrivals queue, and all submitted requests drain before completion.
- Disjoint 40-row prefixes per request: 200 MiB, labeled as 640 cached tokens
  under the harness's 16-token block convention, versus the older 512-token test.
  This is synthetic geometry, not a named model's layout or a long-context model.
- Each completed load stores four new rows (20 MiB) and releases its lease.
- Kernel-managed residency, service eviction policy `none`, no lease mlock,
  no GPU memory registration and no artificial pressure allocation.
- 4 GiB cgroup RAM and 8 GiB swap limits. Bounded cgroup reclaim during fill
  starts above 2.25 GiB, with `memory.high=2.5 GiB`, followed by a 512 MiB
  reclaim before arrivals. The lower watermark provides host RAM margin.
  The 200 ms monitor aborts further work at 11.5 GiB whole-host non-free RAM,
  including filesystem cache, leaving margin below a 12 GiB ceiling.
  This polling guard is not a hard global memory cap.

The harness acquires a separate real request lease while each requested row is
warm, before reclaim. Both arms use this ordering, with no pinned pages.
It is necessary for a fault-driven load to satisfy the service's lease contract:
a cold lookup normally reports pending rather than granting a new load lease.
This is therefore a controlled leased-KV reload experiment, not a faithful
simulation of fresh production scheduler admissions.

Fault mode disables Rust lookup prefetch and Python lookup/load prefetch, then
copies through those retained leases. Prefetch mode enables Rust lookup readahead
and Python schedule-boundary prefetch, polls until resident, then copies.
Load-time prefetch is disabled in both arms to isolate lookup-time read-ahead.
The shared manager's scheduler RPCs are serialized; copies execute concurrently.

## Metrics

Each arm writes JSON with all request events, 200 ms backend/memory samples,
fill stores, reclaim events, service counters and cleanup status.

- `ttft_proxy_s`: scheduled arrival to complete KV load and byte verification.
  It includes queue delay but excludes model prefill, decode and first-token work.
- `latency_s`: scheduled arrival through load plus tail store and commit.
- `queue_s`, `wait_s`, `copy_verify_s`, `store_s`: separated request stages.
- Request throughput: completed requests divided by first-arrival-to-last-finish
  wall time, including queue drain. Offered rate is reported separately.
- Logical load throughput and backend read MB/s; no output-token throughput.
- Initial and selected nonresident bytes, bytes cold at dispatch/copy,
  remote errors, peak active jobs and memory counters.
- `cpu_seconds` measures Python process user/system CPU during arrivals;
  it excludes the Rust service and kernel background workers.

All loaded bytes are checked against a row-specific pattern. Unexpected remote
read errors or lost chunks fail the arm. Kernel write errors for anonymous
Python pages lacking digest metadata can occur during cgroup reclaim; report
these separately from KV data errors. Arms run sequentially (fault then prefetch)
and cold residency is measured rather than assumed to match. Do not attribute
raw latency differences solely to prefetch if cold-byte totals differ.

The default `service_timeout` is five seconds. A diagnostic replay can set
`--service-timeout 60 --modes prefetch`; report this configuration difference
and the default-timeout failure, not only the replay's completed requests.
Output is saved before cleanup so shutdown errors do not erase workload evidence.

After jobs drain, test-owned backing storage is hole-punched and exported views
are released before device teardown. Redis objects are deleted only for the run's
UUID keys. This ordering prevents swapoff from reloading a retained mapping
after its remote objects have been deleted.
The harness does not restart Redis or change its eviction configuration.

For a small plumbing smoke test:

```bash
sudo -n .venv/bin/python -m benchmarks.kv_offload.bench_rswap_constant \
  --output /path/to/new-smoke-directory \
  --populated 32 --requests 8 --context-chunks 4 --rate 8
```

The same 8 GiB logical capacity remains reserved in this smoke test, but only
160 MiB is populated. It is not evidence of multi-GiB workload performance.
