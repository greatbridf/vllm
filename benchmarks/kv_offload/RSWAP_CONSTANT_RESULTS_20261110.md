# 8 GiB Constant-Arrival Rswap Results

November 10, 2026. Branch `feat/rust-kv-cache-service`, base
`cea80fc750fa5e69fa313de1c4e3bcce4bf0b3bf`, with the uncommitted remote-loss
recovery changes. Debug Rust binary rebuilt before testing.
Kernel `7.3.0-rust-17091-gd5d231c8975e`, aarch64, 16 KiB pages.
Redis endpoint `21.6.122.137:6379`.

## Workload

Real Rust subprocess, real Python spec/manager/region, eight TP rank views,
eight concurrent host-copy jobs and one serialized scheduler RPC connection.
No model, GPU DMA, simulated compute delay or generated token ran.

- Service capacity 8 GiB; namespace 1,638 rows / 7.998 GiB.
- 1,536 populated 5 MiB rows: 7.5 GiB of actual KV data.
- 24 scheduled arrivals, one every 500 ms (2 requests/s), at most eight jobs
  active. All arrivals are submitted over 11.5 seconds; excess jobs queue.
- Each request loads 40 disjoint rows / 200 MiB, representing 640 cached tokens
  under the synthetic 16-token block convention. Older comparison: 512 tokens.
- Each request then stores four new rows / 20 MiB. Total requested prefix data
  is 4,800 MiB / 4.6875 GiB; total tail data is 480 MiB.
- Service eviction policy `none`, lease pin budget zero, pageable worker mappings.
- 4 GiB cgroup RAM, 8 GiB swap, memory.high 2.5 GiB, bounded 256 MiB reclaim
  above 2.25 GiB during fill, then 512 MiB reclaim before requests.
  No artificial pressure buffer was allocated.
- Each independent request acquires its real lease while its rows are warm,
  before reclaim. This enables a valid fault-driven load and keeps lease ordering
  matched between arms. It is not fresh production scheduler admission.
- Fault arm disables Rust lookup and Python lookup/load prefetch.
  Prefetch arms enable Rust lookup and Python schedule-boundary prefetch;
  load-time prefetch remains disabled.

TTFT proxy includes arrival queueing and finishes when KV loading/byte verification
finishes. Request latency additionally includes tail store and commit. Neither is
model TTFT. All completed loads were verified against row-specific patterns.

## Results

| Metric | Fault Driven, 5 s RPC Timeout | Prefetch, 5 s | Prefetch, 60 s Diagnostic |
| --- | ---: | --- | ---: |
| Successful requests | 24/24 | Failed | 11/24 |
| Selected cold data | 4.685 GiB | 4.686 GiB | 4.682 GiB |
| Median TTFT proxy | 161.51 s | Not available | 173.10 s, completed subset only |
| p95 TTFT proxy | 273.41 s | Not available | 183.49 s, completed subset only |
| Median total latency | 161.52 s | Not available | 190.34 s, completed subset only |
| Median queue delay | 69.34 s | Not available | 1.02 ms, completed subset only |
| Maximum queue delay | 163.36 s | Not available | Not comparable |
| Median copy/verification | 90.73 s | Not available | 49.75 ms, completed subset only |
| Median 20 MiB tail store | 12.18 ms | Not available | 24.78 ms, completed subset only |
| Completed request throughput | 0.0843 req/s | Invalid | Invalid |
| Remote read throughput | 17.68 MB/s | Invalid | Invalid |

Partial-prefetch quantiles are descriptive only, not an apples-to-apples
comparison: thirteen requests failed and omitted requests are not fast successes.
Do not infer a latency benefit or throughput from that completed subset.

Fault phase: 284.73 seconds wall time, 7.21 seconds Python user/system CPU.
CPU excludes Rust and kernel workers. Backend reads 5,034,803,200 bytes,
approximately the requested cold data, plus 2,552,217,600 bytes of background
writes. Read batches 39,398. Peak active jobs and remote reads were eight.
Median fill-store RPC/copy/spill operation was 11.94 ms for eight rows / 40 MiB;
the separately timed fill lasted 290.97 seconds including reclaim.
The complete fault run had no remote read errors, lost chunks, discard failures,
service evictions or cgroup OOM events. It did hit memory.max: 2,517 max events,
and peak cgroup RAM 4 GiB. Anonymous pages without digests caused 8,690 ENODATA
write failures in the request phase; these are not KV read failures.

Default prefetch: a `prefetch` control RPC exceeded the five-second timeout.
The manager disabled offloading and reported misses. The helper's ten-second
service shutdown also timed out; this early harness version lost the final JSON.
The console failure and measured selected cold bytes are retained here.
The failed arm is not a measured prefetch-throughput result.

Diagnostic prefetch: raising `service_timeout` to 60 seconds did not make this
workload reliable. Eleven requests completed, with all copied bytes correct and
zero bytes cold at their copy boundaries. A control-connection failure/reconnect
then encountered "namespace already owned by another scheduler", degrading
lookups to misses. Thirteen requests failed. Remote read-error counters remained
unchanged. Thus "KV key lost during prefetch" in the harness is a degraded-control
miss, not evidence that Redis deleted these objects.
The service again exceeded its ten-second graceful shutdown wait and was killed.
Observed thread inspection showed many detached prefetch threads, including
threads in uninterruptible kernel waits. The exact blocking cause was not proved.

## Interpretation

The offered rate is deliberately overloaded for all-cold prefixes: 2 req/s
requires 419 MB/s of prefix data before overhead, versus the user's approximately
30 MB/s link estimate. At that link rate, the 5.03 GB of cold reads alone would
take about 168 seconds under ideal sustained delivery. The observed fault phase
was 285 seconds, with concurrent writeback and many read round trips; it is not
explained by raw bandwidth alone.

Prefetch did remove fault-time stalls from copies that completed, but did not
reliably complete the constant-arrival workload. Increasing the timeout was
insufficient. This run does not demonstrate an end-to-end prefetch speedup.
Investigate bounded/deduplicated prefetch work, control-lock latency, residency
under reclaim and reconnect ownership before treating this as production-ready.

## Safety And Caveats

The completed fault arm's peak sampled host non-free RAM was 11.28 GiB.
The 60-second diagnostic arm's peak was 5.19 GiB and its peak cgroup RAM 2.50 GiB.
Samples were taken every 200 ms. The guard is not a hard global memory limit.
Filesystem caches were flushed/dropped before the safe retry and once during
the fault request phase to preserve margin. This changes system state and is
part of the conditions, not a clean randomized paired benchmark.

An earlier full attempt aborted during fill at the host guard. Its cleanup
retained exported NumPy views, then deleted remote objects before swapoff.
Swapoff consequently attempted to reload retained mappings, and sampled host
non-free RAM reached 15.25 GiB, exceeding the user's ceiling. No performance
results from that attempt are included. Recovery discarded only test-owned
memfds and removed the test device. The harness now releases exported views
and uses MADV_REMOVE on test-owned storage before deleting remote objects,
saves results before cleanup, and stops after a failed arm.

All final test subprocesses, cgroups and test devices are gone; only the original
unused zram remains. Fault baseline and diagnostic replay report successful
UUID-scoped Redis cleanup. Remote objects from the default-prefetch shutdown
failure may remain: its early harness did not persist the UUID or reach remote
cleanup. Redis was not restarted or reconfigured.

## Reproduction And Artifacts

```bash
sudo -n .venv/bin/python -m benchmarks.kv_offload.bench_rswap_constant \
  --output /home/david.linux/vllm-rswap-8g-constant-safe-20261110

sudo -n .venv/bin/python -m benchmarks.kv_offload.bench_rswap_constant \
  --output /home/david.linux/vllm-rswap-8g-constant-prefetch60-20261110 \
  --modes prefetch --service-timeout 60
```

Paths above describe executed runs; output directories must be new on replay.
Raw successful baseline: first directory's `fault.json`.
Raw failed diagnostic: second directory's `prefetch.json`.
Initial aborted evidence: `/home/david.linux/vllm-rswap-8g-constant-20261110`.
Small matched 160 MiB smoke: both arms passed, median TTFT 8.56 s fault versus
6.85 s prefetch. This is not the 7.5 GiB populated-cache result.

See `README.md` for the reusable workload contract and CLI. Ruff checks,
format checks and git diff --check passed. No GPU/model checks ran.
