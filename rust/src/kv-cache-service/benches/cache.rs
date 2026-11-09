// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Data-structure cost of the cache metadata: LRU promotion, victim selection,
//! lease release, and key validation at the sizes a 64 GiB service reaches.
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_kv_cache_service::cache::{Cache, PrefetchLimits, check_keys};

/// Bounds the samples of a group whose every iteration rebuilds its fixture, so
/// the run stays short: an O(n) path is orders of magnitude away from an O(1)
/// one, and ten samples already separate them.
const REBUILT_SAMPLES: usize = 10;
/// Per-chunk worker bytes. Everything past the metadata is a virtual reservation
/// the benches never touch, so this only has to look like a real row.
const WORKER_BYTES: usize = 64 * 1024;
/// A service `store` batch is bounded by `protocol::MAX_KEYS`; publishing a
/// fixture the same way keeps the benches to shapes the service accepts.
const FIXTURE_BATCH: usize = 1024;
const FIXTURE_REQUEST: &str = "fixture";
const LOOKUP_REQUEST: &str = "lookup";
const LEASE_REQUEST: &str = "lease";
const STORE_REQUEST: &str = "store";
const SIZES: [usize; 3] = [1 << 11, 1 << 13, 1 << 15];
const EVICT_SIZES: [usize; 2] = [1 << 13, 1 << 15];
const EVICT_BATCH: [usize; 3] = [1, 16, 64];
const LEASE_SIZES: [usize; 2] = [1 << 13, 1 << 15];
const LEASE_BATCH: [usize; 2] = [1, 64];
const KEY_BATCH: [usize; 3] = [1, 256, 4096];

/// A cache of `n` published entries, one per chunk, plus the keys in insertion
/// order: `keys[0]` sits at the least recently used end.
struct Fixture {
    cache: Cache,
    keys: Vec<String>,
    /// Slot of every key, as the store that reserved it reported.
    slots: Vec<usize>,
}

fn hex_key(prefix: char, index: usize) -> String {
    format!("{prefix}{index:063x}")
}

/// Writes one chunk's row through the mapping, as a worker's DMA does. A chunk
/// whose pages are not in DRAM answers `pending` rather than a hit, so a case
/// that leases chunks has to publish readable ones.
fn fill_row(cache: &Cache, slot: usize) {
    let base = cache.mapping.as_ptr() as usize + slot * cache.row_bytes;
    for offset in (0..cache.row_bytes).step_by(cache.page_size) {
        // SAFETY: the row lies inside the sealed mapping.
        unsafe {
            let page = std::ptr::with_exposed_provenance_mut::<u8>(base + offset);
            std::ptr::write_volatile(page, std::ptr::read_volatile(page));
        }
    }
}

impl Fixture {
    fn new(n: usize) -> Fixture {
        let keys: Vec<String> = (0..n).map(|index| hex_key('a', index)).collect();
        let mut cache = Cache::new(
            1,
            WORKER_BYTES,
            1,
            n,
            n * WORKER_BYTES,
            0,
            PrefetchLimits::default(),
        )
        .expect("fixture geometry is valid");
        let mut slots = Vec::with_capacity(n);
        for batch in keys.chunks(FIXTURE_BATCH) {
            let (_, reserved, _) =
                cache.store(batch, FIXTURE_REQUEST).expect("room for every fixture key");
            cache
                .commit(batch, FIXTURE_REQUEST, true)
                .expect("a fixture publishes its own reservations");
            slots.extend(reserved);
        }
        Fixture { cache, keys, slots }
    }
}

fn lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_hit");
    for n in SIZES {
        let Fixture {
            mut cache, keys, ..
        } = Fixture::new(n);
        let mut cursor = 0;
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                // Keys are visited in insertion order, so the first pass promotes
                // the least recently used key on every iteration and later passes
                // promote from anywhere in the order. An eviction order that costs
                // time per position therefore cannot hide behind the head case.
                let key = &keys[cursor % n];
                cursor += 1;
                black_box(cache.lookup(black_box(key), LOOKUP_REQUEST))
            })
        });
    }
    group.finish();
}

fn touch_least_recent(c: &mut Criterion) {
    let mut group = c.benchmark_group("touch_least_recent");
    group.sample_size(REBUILT_SAMPLES);
    for n in SIZES {
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    // Rebuilt per iteration so the touched key really is the least
                    // recently used one, which is the cost an order maintained by
                    // shifting the whole map pays in full.
                    let Fixture {
                        mut cache, keys, ..
                    } = Fixture::new(n);
                    let key = std::slice::from_ref(&keys[0]);
                    let start = Instant::now();
                    cache.touch(black_box(key));
                    elapsed += start.elapsed();
                }
                elapsed
            })
        });
    }
    group.finish();
}

fn store_evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_evict");
    group.sample_size(REBUILT_SAMPLES);
    for n in EVICT_SIZES {
        for k in EVICT_BATCH {
            let incoming: Vec<String> = (0..k).map(|index| hex_key('b', index)).collect();
            group.throughput(Throughput::Elements(k as u64));
            group.bench_function(BenchmarkId::new(format!("keys{k}"), n), |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        // The cache holds one chunk per fixture key and none of the
                        // incoming keys, so the store has to evict one victim per key.
                        let Fixture { mut cache, .. } = Fixture::new(n);
                        let start = Instant::now();
                        let plan = cache.store(black_box(&incoming), STORE_REQUEST);
                        elapsed += start.elapsed();
                        assert!(plan.is_ok(), "a full cache must make room");
                    }
                    elapsed
                })
            });
        }
    }
    group.finish();
}

fn release_request(c: &mut Criterion) {
    let mut group = c.benchmark_group("release_request");
    group.sample_size(REBUILT_SAMPLES);
    for n in LEASE_SIZES {
        for leases in LEASE_BATCH {
            group.bench_function(BenchmarkId::new(format!("leases{leases}"), n), |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        let Fixture {
                            mut cache,
                            keys,
                            slots,
                        } = Fixture::new(n);
                        let step = (n / leases).max(1);
                        for index in (0..leases).map(|lease| lease * step) {
                            // A lease needs a chunk a worker wrote.
                            fill_row(&cache, slots[index]);
                            cache.lookup(&keys[index], LEASE_REQUEST);
                        }
                        let start = Instant::now();
                        black_box(cache.release_request(LEASE_REQUEST));
                        elapsed += start.elapsed();
                    }
                    elapsed
                })
            });
        }
    }
    group.finish();
}

fn validate_keys(c: &mut Criterion) {
    let mut group = c.benchmark_group("check_keys");
    for m in KEY_BATCH {
        let keys: Vec<String> = (0..m).map(|index| hex_key('a', index)).collect();
        group.bench_function(BenchmarkId::from_parameter(m), |b| {
            b.iter(|| black_box(check_keys(black_box(&keys))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    lookup_hit,
    touch_least_recent,
    store_evict,
    release_request,
    validate_keys
);
criterion_main!(benches);
