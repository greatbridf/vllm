// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::io::Write;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use serde_json::{Value, json};
use vllm_kv_cache_service::cache::{
    Cache, EvictionPolicy, PREFETCH_STREAMS, PrefetchLimits, Released, Status, StoreRejection,
    check_keys, check_prefetched_row, check_request, namespace_bytes, page_out_rows, prefetch_rows,
    prefetch_rows_parallel,
};
use vllm_kv_cache_service::hash::{HashFactory, RswapConfig};
use vllm_kv_cache_service::protocol::{self, Request};
use vllm_kv_cache_service::remote::{RedisTier, RemoteTier};

#[macro_use]
mod log;
mod bench;

const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
/// Upper bound on how long a shutdown signal waits for the accept loop, which
/// polls the listener instead of blocking in `accept`.
const ACCEPT_POLL: Duration = Duration::from_millis(200);
/// Backoff after a transient accept failure such as descriptor exhaustion.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(50);

/// Set by the SIGTERM and SIGINT handler, read by the accept loop.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// The command line. Serving requires a socket and a capacity, while the `bench`
/// subcommand measures without either, so those two are optional here and still
/// required by the parser whenever no subcommand is given.
#[derive(Parser)]
#[command(subcommand_negates_reqs = true)]
struct Args {
    /// Socket in a directory accessible only to the service owner.
    #[arg(long, required = true)]
    socket: Option<PathBuf>,
    /// Aggregate upper bound on bytes allocated across cache namespaces.
    #[arg(long, required = true)]
    capacity_bytes: Option<usize>,
    /// Replace least-recently-used keys, or retain keys and leave page reclaim
    /// to Linux. With `none`, stores needing more slots return `full`.
    #[arg(long, value_enum, default_value = "lru")]
    eviction_policy: EvictionPolicy,
    /// Simultaneous client connections served at once.
    #[arg(long, default_value_t = 128)]
    max_connections: usize,
    /// Close a connection that sends no request for this many seconds; 0 keeps
    /// idle connections open.
    #[arg(long, default_value_t = 900)]
    idle_timeout_secs: u64,
    /// Free a `store` reservation this many seconds after it was made; 0 keeps
    /// reservations until their request commits or releases them.
    #[arg(long, default_value_t = 60)]
    pending_timeout_secs: u64,
    /// Reclaim a namespace whose owner disconnected this many seconds after it
    /// detached, unless an owner adopted it again; 0 keeps it until it has no
    /// sessions left.
    #[arg(long, default_value_t = 300)]
    namespace_grace_secs: u64,
    /// Lock the rows of leased chunks with `mlock(2)` so a `hit` stays readable,
    /// up to this many bytes across the namespace; 0 leaves leases unpinned.
    #[arg(long, default_value_t = 0)]
    pin_lease_bytes: usize,
    /// Read the row of a `lookup` that is not resident ahead, then lock it once it
    /// is, so a later load reads it without a fault; a slot whose readahead
    /// started less than this many milliseconds ago is left alone, and 0
    /// prefetches on every lookup. Locking needs `--pin-lease-bytes`.
    #[arg(long, default_value_t = 2000)]
    prefetch_cooldown_ms: u64,
    /// Rows one lookup-time readahead may start; a wider batch leaves the rest
    /// for a later lookup.
    #[arg(long, default_value_t = 32)]
    prefetch_max_chunks: usize,
    /// After a commit, keep this service's cgroup at or below this percentage of
    /// its `memory.max` by asking the kernel to reclaim through
    /// `memory.reclaim`; 0 disables it. `madvise(MADV_PAGEOUT)` does not collect
    /// this mapping's pages, so cgroup reclaim is the only trigger that works.
    #[arg(long, default_value_t = 0)]
    spill_watermark_percent: u64,
    /// Attach the rswap digest BPF program through sysfs; disabled by default.
    /// Defaults to /sys/kernel/mm/xswap/rswap_attach when no path is given.
    #[arg(
        long,
        visible_alias = "hash-attach",
        num_args = 0..=1,
        default_missing_value = "/sys/kernel/mm/xswap/rswap_attach"
    )]
    rswap_attach: Option<PathBuf>,
    /// Read missing rows from an rswap Redis tier; never write remote objects.
    #[arg(long, env = "VLLM_KV_REMOTE_REDIS_URL", hide_env_values = true)]
    remote_redis_url: Option<String>,
    /// Total remote-read budget per lookup RPC; keep below service_timeout.
    #[arg(long, default_value_t = 1000, value_parser = clap::value_parser!(u64).range(1..))]
    remote_timeout_ms: u64,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Measure the control plane in-process and report the cost of each case.
    Bench(bench::BenchArgs),
}

/// Per-connection bounds taken from the command line.
#[derive(Clone, Copy)]
struct Limits {
    max_connections: usize,
    idle_timeout: Option<Duration>,
}

/// What a connected session is to its namespace.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SessionKind {
    /// The session that sent `open`.
    Owner,
    /// A session that sent `attach` for one rank.
    Worker,
}

/// One engine namespace: the shared cache mapping plus the sessions using it.
struct Namespace {
    cache: Arc<Mutex<Cache>>,
    fingerprint: String,
    attached_ranks: HashSet<usize>,
    sessions: HashMap<u64, SessionKind>,
    /// Set while no owner is bound. The namespace keeps its mapping, entries, and
    /// leases so the same engine can adopt it, until the grace period ends or an
    /// attached worker disconnects.
    detached_since: Option<Instant>,
}

impl Namespace {
    fn detached(&self) -> bool {
        self.detached_since.is_some()
    }
}

/// The geometry an `open` reserves, checked against a detached namespace before
/// it is adopted.
#[derive(Clone, Copy)]
struct Geometry {
    worker_bytes: usize,
    world_size: usize,
    num_chunks: usize,
}

/// Namespaces reachable by name. Its lock is only held to clone a cache handle or
/// to insert and remove an entry, never across socket I/O.
/// Chunks one watermark spill pages out.
const SPILL_CHUNKS: usize = 32;

struct Registry {
    namespaces: HashMap<String, Namespace>,
    retired: Vec<Namespace>,
    budget: usize,
    rswap_attach: Option<PathBuf>,
    remote: Option<Arc<dyn RemoteTier>>,
    remote_timeout: Duration,
    pin_lease_bytes: usize,
    eviction_policy: EvictionPolicy,
    /// `(memory.reclaim, watermark percent)` of this service's cgroup.
    spill: Option<(PathBuf, u64)>,
    prefetch: PrefetchLimits,
    pending_timeout: Option<Duration>,
    started: Instant,
}

/// A reply is built before the wire is touched, so a failed write cannot add a
/// second frame to the stream.
enum Reply {
    Frame(Vec<u8>),
    Attach { frame: Vec<u8>, fd: OwnedFd },
}

/// A rejected request leaves the session and its namespace alive; `Closed` ends
/// the connection.
enum Failure {
    Rejected(String),
    Closed(io::Error),
}

impl From<io::Error> for Failure {
    fn from(error: io::Error) -> Self {
        Failure::Closed(error)
    }
}

/// Removes the socket path however the process stops serving. Only an abrupt
/// kill bypasses it; the next start reclaims that path.
struct SocketPath<'a>(&'a Path);

impl Drop for SocketPath<'_> {
    fn drop(&mut self) {
        match fs::remove_file(self.0) {
            Ok(()) => log_event!("socket_removed", "path" = self.0.display()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                log_event!(
                    "socket_remove_failed",
                    "path" = self.0.display(),
                    "error" = error
                );
            }
        }
    }
}

/// Holds one slot of the connection bound and frees it when its session ends.
struct ConnectionSlot(Arc<AtomicUsize>);

impl ConnectionSlot {
    fn acquire(active: &Arc<AtomicUsize>) -> ConnectionSlot {
        active.fetch_add(1, Ordering::Relaxed);
        ConnectionSlot(Arc::clone(active))
    }
}

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

fn invalid(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::other(message)
}

fn reject(message: impl std::fmt::Display) -> Failure {
    Failure::Rejected(format!("{message}"))
}

/// Recovers a poisoned lock instead of failing: one panicking session must not
/// wedge every other namespace. The first recovery is logged once.
fn lock_or_recover<'a, T>(lock: &'a Mutex<T>, name: &str) -> MutexGuard<'a, T> {
    match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            static REPORTED: AtomicBool = AtomicBool::new(false);
            if !REPORTED.swap(true, Ordering::Relaxed) {
                log_event!("lock_poisoned", "lock" = name);
            }
            poisoned.into_inner()
        }
    }
}

fn geometry_reply(cache: &Cache) -> Value {
    json!({
        "ok": true,
        "version": 1,
        "worker_bytes": cache.worker_bytes,
        "world_size": cache.world_size,
        "row_bytes": cache.row_bytes,
        "num_chunks": cache.num_chunks,
        "total_bytes": cache.mapping.len(),
    })
}

/// Service-wide state for an operator or a probe, reported without reading any
/// slot.
fn info(registry: &Registry, limits: &Limits) -> Value {
    json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "pid": std::process::id(),
        "uptime_ms": registry.started.elapsed().as_millis() as u64,
        "capacity_bytes": registry.budget,
        "used_bytes": registry.used_bytes(),
        "retired_bytes": registry.retired_bytes(),
        "namespaces": registry.namespaces.len(),
        "detached": registry.detached(),
        "sessions": registry.sessions(),
        "max_connections": limits.max_connections,
    })
}

fn handle(
    request: Request,
    registry: &Arc<Mutex<Registry>>,
    limits: &Limits,
    session: u64,
    namespace: &mut Option<String>,
) -> Result<Reply, Failure> {
    match request {
        Request::Open {
            version,
            namespace: name,
            fingerprint,
            worker_bytes,
            world_size,
            num_chunks,
            adopt,
        } => {
            if version != 1
                || namespace.is_some()
                || name.is_empty()
                || name.len() > 256
                || fingerprint.len() != 64
                || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(reject("invalid open handshake"));
            }
            let size = namespace_bytes(worker_bytes, world_size, num_chunks).map_err(reject)?;
            let geometry = Geometry {
                worker_bytes,
                world_size,
                num_chunks,
            };
            let frame = {
                let mut registry = lock_or_recover(registry, "registry");
                registry.open(session, &name, &fingerprint, geometry, size, adopt)?
            };
            *namespace = Some(name);
            Ok(Reply::Frame(frame))
        }
        Request::Attach {
            version,
            namespace: name,
            fingerprint,
            rank,
        } => {
            if version != 1 || namespace.is_some() {
                return Err(reject("invalid attach handshake"));
            }
            let reply = {
                let mut registry = lock_or_recover(registry, "registry");
                registry.attach(session, &name, &fingerprint, rank)?
            };
            *namespace = Some(name);
            Ok(reply)
        }
        Request::Close => {
            let Some(name) = namespace.clone() else {
                return Err(reject("handshake required"));
            };
            {
                let mut registry = lock_or_recover(registry, "registry");
                registry.close(session, &name)?;
            }
            *namespace = None;
            Ok(Reply::Frame(protocol::encode_response(
                &json!({"ok": true}),
            )?))
        }
        Request::Info => {
            let registry = lock_or_recover(registry, "registry");
            let frame = protocol::encode_response(&info(&registry, limits))?;
            Ok(Reply::Frame(frame))
        }
        command => {
            let name = namespace.as_deref().ok_or_else(|| reject("handshake required"))?;
            let (handle, pending_timeout, sessions, detached, remote, remote_timeout) = {
                let registry = lock_or_recover(registry, "registry");
                let entry = registry
                    .namespaces
                    .get(name)
                    .ok_or_else(|| reject("cache namespace invalidated"))?;
                (
                    Arc::clone(&entry.cache),
                    registry.pending_timeout,
                    entry.sessions.len(),
                    entry.detached(),
                    registry.remote.clone(),
                    registry.remote_timeout,
                )
            };
            let mut cache = lock_or_recover(&handle, "cache");
            // Read-only stats stay available to attached workers; every other
            // command mutates scheduler-owned cache metadata.
            if cache.owner != session && !matches!(command, Request::Stats) {
                return Err(reject("workers cannot mutate cache metadata"));
            }
            let response = match command {
                Request::Lookup { key, request } => {
                    check_keys(std::slice::from_ref(&key)).map_err(reject)?;
                    check_request(&request).map_err(reject)?;
                    let mut status = cache.lookup(&key, &request);
                    if status == Status::Miss
                        && let Some(remote) = &remote
                    {
                        let geometry = (cache.row_bytes, cache.page_size);
                        drop(cache);
                        let row = remote.load(
                            &key,
                            geometry.0,
                            geometry.1,
                            Instant::now() + remote_timeout,
                        );
                        cache = remote_cache(registry, &handle, name, session)?;
                        status = promote_remote(&mut cache, &key, &request, row);
                    }
                    prefetch_pending(
                        &mut cache,
                        &handle,
                        std::slice::from_ref(&key),
                        &[status],
                        &request,
                    );
                    json!({"ok": true, "status": status})
                }
                Request::LookupMany {
                    keys,
                    request,
                    until_miss,
                } => {
                    check_keys(&keys).map_err(reject)?;
                    check_request(&request).map_err(reject)?;
                    // A truncated scan reports fewer statuses than keys, which
                    // is smaller than the batch and so still fits one frame.
                    let statuses = if let Some(remote) = &remote {
                        let deadline = Instant::now() + remote_timeout;
                        let mut statuses = Vec::with_capacity(keys.len());
                        for key in &keys {
                            let mut status = cache.lookup(key, &request);
                            if status == Status::Miss && Instant::now() < deadline {
                                let geometry = (cache.row_bytes, cache.page_size);
                                drop(cache);
                                let row = remote.load(key, geometry.0, geometry.1, deadline);
                                cache = remote_cache(registry, &handle, name, session)?;
                                status = promote_remote(&mut cache, key, &request, row);
                            }
                            statuses.push(status);
                            if until_miss && status == Status::Miss {
                                break;
                            }
                        }
                        statuses
                    } else {
                        cache.lookup_many(&keys, &request, until_miss)
                    };
                    prefetch_pending(&mut cache, &handle, &keys, &statuses, &request);
                    json!({"ok": true, "statuses": statuses})
                }
                Request::Load {
                    keys,
                    request,
                    prefetch,
                } => {
                    check_keys(&keys).map_err(reject)?;
                    check_request(&request).map_err(reject)?;
                    let slots = cache.load(&keys, &request).map_err(reject)?;
                    let pages = slots.len() * cache.row_bytes / cache.page_size;
                    let prefetched = if prefetch {
                        prefetch_detached(&mut cache, &handle, slots.clone());
                        pages
                    } else {
                        0
                    };
                    json!({"ok": true, "slots": slots, "prefetched": prefetched})
                }
                Request::Prefetch { keys } => {
                    check_keys(&keys).map_err(reject)?;
                    let slots = cache.committed_slots(&keys).map_err(reject)?;
                    let pages = slots.len() * cache.row_bytes / cache.page_size;
                    prefetch_detached(&mut cache, &handle, slots);
                    json!({"ok": true, "pages": pages})
                }
                Request::Release { keys, request } => {
                    check_keys(&keys).map_err(reject)?;
                    check_request(&request).map_err(reject)?;
                    released_reply(cache.release(&keys, &request))
                }
                Request::ReleaseAll { request } => {
                    check_request(&request).map_err(reject)?;
                    released_reply(cache.release_request(&request))
                }
                Request::Store { keys, request } => {
                    check_keys(&keys).map_err(reject)?;
                    check_request(&request).map_err(reject)?;
                    reclaim_expired(&mut cache, name, pending_timeout);
                    match cache.store(&keys, &request) {
                        Ok((keys, slots, evicted)) => {
                            json!({
                                "ok": true,
                                "keys": keys,
                                "slots": slots,
                                "evicted": evicted
                            })
                        }
                        Err(StoreRejection::Full) => json!({"ok": true, "full": true}),
                        // Nothing was reserved, so the caller may retry the batch
                        // in smaller pieces on this session.
                        Err(StoreRejection::Unanswerable) => {
                            return Err(reject("batch too large for one control frame"));
                        }
                    }
                }
                Request::Commit {
                    keys,
                    request,
                    success,
                } => {
                    check_keys(&keys).map_err(reject)?;
                    check_request(&request).map_err(reject)?;
                    cache.commit(&keys, &request, success).map_err(reject)?;
                    if success && let Some(spill) = registry_spill(registry) {
                        spawn_watermark_spill(registry, spill, name);
                    }
                    json!({"ok": true})
                }
                Request::Touch { keys } => {
                    check_keys(&keys).map_err(reject)?;
                    cache.touch(&keys);
                    json!({"ok": true})
                }
                Request::Spill { keys } => {
                    check_keys(&keys).map_err(reject)?;
                    json!({"ok": true, "pages": cache.spill(&keys).map_err(reject)?})
                }
                Request::Reset { force } => {
                    reclaim_expired(&mut cache, name, pending_timeout);
                    cache.reset(force).map_err(reject)?;
                    json!({"ok": true})
                }
                Request::Stats => {
                    reclaim_expired(&mut cache, name, pending_timeout);
                    json!({
                        "ok": true,
                        "stats": cache.stats(),
                        "sessions": sessions,
                        "detached": detached,
                    })
                }
                Request::Open { .. } | Request::Attach { .. } | Request::Close | Request::Info => {
                    unreachable!()
                }
            };
            Ok(Reply::Frame(protocol::encode_response(&response)?))
        }
    }
}

fn released_reply(released: Released) -> Value {
    json!({"ok": true, "released": released.leases, "unknown": released.unknown})
}

/// Revalidate the namespace after remote I/O, locking registry before cache so
/// retirement cannot clear its hashes and then race a late promotion.
fn remote_cache<'a>(
    registry: &Arc<Mutex<Registry>>,
    handle: &'a Arc<Mutex<Cache>>,
    name: &str,
    session: u64,
) -> Result<MutexGuard<'a, Cache>, Failure> {
    let registry = lock_or_recover(registry, "registry");
    let entry = registry
        .namespaces
        .get(name)
        .filter(|entry| Arc::ptr_eq(&entry.cache, handle))
        .ok_or_else(|| reject("cache namespace invalidated"))?;
    if entry.sessions.get(&session) != Some(&SessionKind::Owner) {
        return Err(reject("namespace owner changed during remote lookup"));
    }
    Ok(lock_or_recover(handle, "cache"))
}

fn promote_remote(
    cache: &mut Cache,
    key: &str,
    request: &str,
    row: io::Result<Option<Vec<u8>>>,
) -> Status {
    match row {
        Ok(Some(row)) => match cache.promote_remote(key, request, &row) {
            Ok(status) => status,
            Err(error) => {
                log_event!("remote_promotion_failed", "error" = error);
                Status::Miss
            }
        },
        Ok(None) => Status::Miss,
        Err(error) => {
            log_event!("remote_lookup_failed", "error" = error);
            Status::Miss
        }
    }
}

/// Asks the kernel to read `slots` back in, on a thread of its own.
///
/// The readahead must not delay a reply: at a memory limit the readahead itself
/// reclaims, which takes longer than a client waits. The mapping is shared, so
/// the work runs with the cache lock released.
fn prefetch_detached(cache: &mut Cache, handle: &Arc<Mutex<Cache>>, slots: Vec<usize>) {
    let slots: Vec<usize> = slots.into_iter().filter(|&slot| !cache.is_resident(slot)).collect();
    if slots.is_empty() {
        return;
    }
    let (mapping, row_bytes, page_size) = cache.prefetch_parts();
    let Ok((file, checks)) = cache.prefetch_checks(&slots) else {
        return;
    };
    let handle = Arc::clone(handle);
    std::thread::spawn(move || {
        let _ = prefetch_rows(&mapping, row_bytes, page_size, &slots);
        for (slot, generation) in checks {
            let result = check_prefetched_row(&file, row_bytes, page_size, slot);
            let mut cache = lock_or_recover(&handle, "cache");
            if let Err(error) = result
                && error.raw_os_error() == Some(libc::EIO)
            {
                cache.lose_chunk(slot, generation);
            }
            cache.finish_check(slot, generation);
        }
    });
}

/// Starts the readahead a lookup's `pending` answers asked for.
///
/// Only a committed row that is not resident has anything to read ahead, and the
/// readahead runs on a thread of its own so the reply never waits. Each row it
/// makes resident is locked for `request` once the read finishes, which is what
/// turns the later lookup from another `pending` into a hit the load can copy
/// without faulting.
fn prefetch_pending(
    cache: &mut Cache,
    handle: &Arc<Mutex<Cache>>,
    keys: &[String],
    statuses: &[Status],
    request: &str,
) {
    let pending: Vec<&str> = keys
        .iter()
        .zip(statuses)
        .filter(|(_, status)| **status == Status::Pending)
        .map(|(key, _)| key.as_str())
        .collect();
    if pending.is_empty() {
        return;
    }
    let slots = cache.prefetch_slots(&pending);
    let accepted = cache.begin_prefetch(&slots, request, Instant::now());
    if accepted.is_empty() {
        return;
    }
    let Ok((file, checks)) = cache.prefetch_checks(&accepted) else {
        return;
    };
    let handle = Arc::clone(handle);
    let request = request.to_owned();
    thread::spawn(move || {
        let (mapping, row_bytes, page_size) = lock_or_recover(&handle, "cache").prefetch_parts();
        let _ = prefetch_rows_parallel(&mapping, row_bytes, page_size, &accepted, PREFETCH_STREAMS);
        for (slot, generation) in checks {
            let checked = check_prefetched_row(&file, row_bytes, page_size, slot);
            let mut cache = lock_or_recover(&handle, "cache");
            if checked.as_ref().is_err_and(|error| error.raw_os_error() == Some(libc::EIO)) {
                cache.lose_chunk(slot, generation);
            } else if checked.is_ok() {
                cache.finish_checked_prefetch(slot, generation, &request);
            }
            cache.finish_check(slot, generation);
        }
    });
}

/// Frees pending reservations that outlived `--pending-timeout-secs` before a
/// command that could otherwise see them, and logs each reclaimed request. A
/// reservation nobody commits would otherwise keep its keys pending forever,
/// which the connector reports as a hit that never becomes readable.
fn reclaim_expired(cache: &mut Cache, namespace: &str, pending_timeout: Option<Duration>) {
    let Some(timeout) = pending_timeout else {
        return;
    };
    for reclaimed in cache.reclaim_expired(Instant::now(), timeout) {
        log_event!(
            "reservations_reclaimed",
            "namespace" = namespace,
            "request" = reclaimed.request,
            "keys" = reclaimed.keys,
            "age_secs" = reclaimed.age.as_secs(),
        );
    }
}

/// This service's cgroup v2 `memory.reclaim`, if it has one.
///
/// `madvise(MADV_PAGEOUT)` does not collect the pages of this mapping (measured:
/// it reports success and pages nothing out), while cgroup reclaim does, so the
/// watermark spill goes through the cgroup.
fn cgroup_reclaim_path() -> Option<PathBuf> {
    let membership = fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = membership.lines().find_map(|line| line.strip_prefix("0::"))?.trim();
    let reclaim = PathBuf::from(format!("/sys/fs/cgroup{path}/memory.reclaim"));
    reclaim.is_file().then_some(reclaim)
}

/// The watermark spill, if this service has one.
fn registry_spill(registry: &Mutex<Registry>) -> Option<(PathBuf, u64)> {
    lock_or_recover(registry, "registry").spill.clone()
}

/// Pages the coldest chunks out, or falls back to cgroup reclaim.
///
/// The page-out goes through `MADV_PAGEOUT` on the rows the cache would evict
/// next; that only works for ranges that are not locked, which is why a commit
/// drops the reservation's lock. Runs on its own thread: both triggers reclaim
/// synchronously and must never delay a reply. If the rows are still resident
/// afterwards (pageout collected nothing), the cgroup's `memory.reclaim` is the
/// fallback, because it reclaims whatever the kernel finds reclaimable.
fn spawn_watermark_spill(
    registry: &Arc<Mutex<Registry>>,
    (reclaim, watermark): (PathBuf, u64),
    namespace: &str,
) {
    let prepared = {
        let registry = lock_or_recover(registry, "registry");
        registry.namespaces.get(namespace).map(|entry| {
            let cache = Arc::clone(&entry.cache);
            let (mapping, row_bytes, page_size) = lock_or_recover(&cache, "cache").prefetch_parts();
            (cache, mapping, row_bytes, page_size)
        })
    };
    let Some((cache, mapping, row_bytes, page_size)) = prepared else {
        return;
    };
    let reclaim = reclaim.clone();
    std::thread::spawn(move || {
        let under_watermark = |current: Option<u64>, limit: Option<u64>| match (current, limit) {
            (Some(current), Some(limit)) if limit > 0 => current <= limit * watermark / 100,
            _ => true,
        };
        let current = read_cgroup_number(reclaim.with_file_name("memory.current"));
        let limit = read_cgroup_number(reclaim.with_file_name("memory.max"));
        if under_watermark(current, limit) {
            return;
        }
        let candidates = lock_or_recover(&cache, "cache").spill_candidates(SPILL_CHUNKS);
        if !candidates.is_empty() {
            match page_out_rows(&mapping, row_bytes, page_size, &candidates) {
                Ok(pages) => log_event!(
                    "spill_pageout",
                    "chunks" = candidates.len(),
                    "pages" = pages
                ),
                Err(error) => log_event!("spill_pageout_failed", "error" = error),
            }
            lock_or_recover(&cache, "cache")
                .add_reclaimed_bytes((candidates.len() * row_bytes) as u64);
        }
        let current = read_cgroup_number(reclaim.with_file_name("memory.current"));
        if under_watermark(current, limit) {
            return;
        }
        let want = current.unwrap_or(0).saturating_sub(limit.unwrap_or(0) * watermark / 100);
        let outcome = fs::OpenOptions::new()
            .write(true)
            .open(&reclaim)
            .and_then(|mut file| file.write_all(want.to_string().as_bytes()));
        match outcome {
            Ok(()) => log_event!("spill_reclaim", "bytes" = want),
            Err(error) => log_event!("spill_failed", "bytes" = want, "error" = error),
        }
    });
}

fn read_cgroup_number(path: PathBuf) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn serve(
    stream: &mut UnixStream,
    registry: &Arc<Mutex<Registry>>,
    session: u64,
    namespace: &mut Option<String>,
    limits: &Limits,
) {
    let idle_timeout = limits.idle_timeout;
    let timeouts = stream
        .set_write_timeout(Some(WRITE_TIMEOUT))
        .and_then(|()| idle_timeout.map_or(Ok(()), |idle| stream.set_read_timeout(Some(idle))));
    if let Err(error) = timeouts {
        log_event!("timeout_failed", "session" = session, "error" = error);
        return;
    }
    loop {
        let request = match protocol::read_request(stream) {
            Ok(request) => request,
            Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                log_event!("idle_timeout", "session" = session);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => {
                log_event!("invalid_frame", "session" = session, "error" = error);
                break;
            }
        };
        let op = request.op();
        let written = match handle(request, registry, limits, session, namespace) {
            Ok(Reply::Frame(frame)) => protocol::write_frame(stream, &frame),
            Ok(Reply::Attach { frame, fd }) => protocol::write_frame(stream, &frame)
                .and_then(|()| protocol::send_fd(stream, fd.as_fd())),
            Err(Failure::Rejected(message)) => {
                log_event!(
                    "request_rejected",
                    "session" = session,
                    "op" = op,
                    "error" = message
                );
                protocol::encode_response(&json!({"ok": false, "error": message}))
                    .and_then(|frame| protocol::write_frame(stream, &frame))
            }
            Err(Failure::Closed(error)) => {
                log_event!(
                    "request_failed",
                    "session" = session,
                    "op" = op,
                    "error" = error
                );
                break;
            }
        };
        if let Err(error) = written {
            // A partially written frame cannot be repaired on this stream.
            log_event!(
                "write_failed",
                "session" = session,
                "op" = op,
                "error" = error
            );
            break;
        }
    }
}

/// Serves one connection and always releases its namespace, including when the
/// session panics.
fn serve_connection(
    mut stream: UnixStream,
    registry: Arc<Mutex<Registry>>,
    session: u64,
    limits: Limits,
) {
    let mut namespace = None;
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        serve(&mut stream, &registry, session, &mut namespace, &limits);
    }));
    if let Err(payload) = outcome {
        log_event!(
            "panic",
            "session" = session,
            "panic" = panic_message(payload.as_ref())
        );
        // Best effort: the stream may hold a partially written frame.
        let reply = json!({"ok": false, "error": "internal error"});
        let _ = protocol::encode_response(&reply)
            .and_then(|frame| protocol::write_frame(&mut stream, &frame));
    }
    release_session(&registry, namespace.as_deref(), session);
    log_event!("disconnect", "session" = session);
}

/// Renders a panic payload for the log line.
fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// Ends one departed session; `Registry::release_session` decides what that costs
/// its namespace.
fn release_session(registry: &Mutex<Registry>, name: Option<&str>, session: u64) {
    lock_or_recover(registry, "registry").release_session(session, name);
}

/// Reclaims namespaces their owner left behind longer than `grace` ago.
fn sweep_namespaces(registry: &Mutex<Registry>, grace: Duration) {
    lock_or_recover(registry, "registry").sweep(Instant::now(), grace);
}

fn invalidate_hashes(namespace: &Namespace) {
    lock_or_recover(&namespace.cache, "cache").invalidate_hashes();
}

impl Registry {
    /// Opens `name` for `session`, or adopts the namespace its previous owner left
    /// behind. Adoption keeps the mapping, every entry, and every lease, and only
    /// succeeds for the fingerprint and geometry the mapping was created with, so
    /// a reconnecting scheduler never reuses a cache it would not have opened.
    /// `adopt` forbids creating a namespace, because a client that reconnects must
    /// not allocate a second mapping while the workers still hold the first.
    fn open(
        &mut self,
        session: u64,
        name: &str,
        fingerprint: &str,
        geometry: Geometry,
        size: usize,
        adopt: bool,
    ) -> Result<Vec<u8>, Failure> {
        if let Some(entry) = self.namespaces.get_mut(name) {
            if !entry.detached() {
                return Err(reject("namespace already owned by another scheduler"));
            }
            let (frame, bytes) = {
                let mut cache = lock_or_recover(&entry.cache, "cache");
                if fingerprint != entry.fingerprint
                    || geometry.worker_bytes != cache.worker_bytes
                    || geometry.world_size != cache.world_size
                    || geometry.num_chunks != cache.num_chunks
                {
                    return Err(reject("namespace geometry mismatch"));
                }
                // The cache reports the adopter as its owner, so its sessions may
                // mutate what the previous owner reserved and published.
                cache.owner = session;
                (
                    protocol::encode_response(&geometry_reply(&cache))?,
                    cache.mapping.len(),
                )
            };
            entry.sessions.insert(session, SessionKind::Owner);
            entry.detached_since = None;
            log_event!(
                "namespace_adopted",
                "session" = session,
                "namespace" = name,
                "sessions" = entry.sessions.len(),
                "bytes" = bytes,
            );
            return Ok(frame);
        }
        if adopt {
            return Err(reject("no detached namespace to adopt"));
        }
        // The budget is charged for the whole namespace lifetime, whatever the
        // sessions do with it, and a retired mapping is never a candidate.
        let remaining = self.budget.saturating_sub(self.used_bytes());
        if size > remaining {
            return Err(reject("cache exceeds service capacity"));
        }
        let factory = self.rswap_attach.as_ref().map(|path| RswapConfig {
            attach_path: path.clone(),
        });
        let cache = Cache::with_hashes(
            session,
            geometry.worker_bytes,
            geometry.world_size,
            geometry.num_chunks,
            remaining,
            self.pin_lease_bytes,
            self.prefetch,
            self.eviction_policy,
            factory.as_ref().map(|factory| factory as &dyn HashFactory),
        )?;
        let frame = protocol::encode_response(&geometry_reply(&cache))?;
        log_event!(
            "namespace_opened",
            "session" = session,
            "namespace" = name,
            "num_chunks" = cache.num_chunks,
            "row_bytes" = cache.row_bytes,
            "bytes" = cache.mapping.len(),
            "budget_left_bytes" = remaining - size,
        );
        self.namespaces.insert(
            name.to_owned(),
            Namespace {
                cache: Arc::new(Mutex::new(cache)),
                fingerprint: fingerprint.to_owned(),
                attached_ranks: HashSet::new(),
                sessions: HashMap::from([(session, SessionKind::Owner)]),
                detached_since: None,
            },
        );
        Ok(frame)
    }

    /// Binds `session` to rank `rank` of a namespace and hands back its geometry
    /// and a duplicate of its mapping.
    fn attach(
        &mut self,
        session: u64,
        name: &str,
        fingerprint: &str,
        rank: usize,
    ) -> Result<Reply, Failure> {
        let entry = self
            .namespaces
            .get_mut(name)
            .ok_or_else(|| reject("scheduler namespace not available"))?;
        let (frame, fd) = {
            let cache = lock_or_recover(&entry.cache, "cache");
            if fingerprint != entry.fingerprint
                || rank >= cache.world_size
                || entry.attached_ranks.contains(&rank)
            {
                return Err(reject("incompatible or duplicate worker"));
            }
            let frame = protocol::encode_response(&geometry_reply(&cache))?;
            (frame, protocol::duplicate_fd(cache.file.as_fd())?)
        };
        entry.attached_ranks.insert(rank);
        entry.sessions.insert(session, SessionKind::Worker);
        log_event!(
            "worker_attached",
            "session" = session,
            "namespace" = name,
            "rank" = rank
        );
        Ok(Reply::Attach { frame, fd })
    }

    /// Retires a namespace its owner closed: a shutdown never leaves a mapping a
    /// later engine could adopt.
    fn close(&mut self, session: u64, name: &str) -> Result<(), Failure> {
        let entry =
            self.namespaces.get(name).ok_or_else(|| reject("cache namespace invalidated"))?;
        if entry.sessions.get(&session) != Some(&SessionKind::Owner) {
            return Err(reject("workers cannot mutate cache metadata"));
        }
        if let Some(mut entry) = self.namespaces.remove(name) {
            // The closing session leaves with its namespace, so only the sessions
            // that stay behind keep the retired mapping charged.
            entry.sessions.remove(&session);
            self.retire(name, entry, "close", session);
        }
        Ok(())
    }

    /// Ends one session. A worker's departure retires its namespace, because that
    /// rank may still have DMA in flight into slots nobody may recycle; an owner's
    /// departure only detaches it, so the same engine can adopt its entries.
    fn release_session(&mut self, session: u64, name: Option<&str>) {
        if let Some(name) = name
            && let Some(kind) = self
                .namespaces
                .get(name)
                .and_then(|entry| entry.sessions.get(&session).copied())
            && let Some(mut entry) = self.namespaces.remove(name)
        {
            entry.sessions.remove(&session);
            match kind {
                SessionKind::Worker => self.retire(name, entry, "worker_disconnect", session),
                SessionKind::Owner => {
                    log_event!(
                        "namespace_detached",
                        "session" = session,
                        "namespace" = name,
                        "sessions_left" = entry.sessions.len(),
                    );
                    entry.detached_since = Some(Instant::now());
                    self.namespaces.insert(name.to_owned(), entry);
                }
            }
        }
        for entry in &mut self.retired {
            entry.sessions.remove(&session);
            invalidate_hashes(entry);
        }
        self.retired.retain(|entry| !entry.sessions.is_empty());
    }

    /// Frees a namespace's mapping, or holds it charged while its remaining
    /// sessions keep it, and logs the transition.
    fn retire(&mut self, name: &str, entry: Namespace, reason: &'static str, session: u64) {
        invalidate_hashes(&entry);
        if entry.sessions.is_empty() {
            log_event!(
                "namespace_invalidated",
                "session" = session,
                "namespace" = name,
                "reason" = reason,
            );
            return;
        }
        log_event!(
            "namespace_retired",
            "session" = session,
            "namespace" = name,
            "reason" = reason,
            "sessions_left" = entry.sessions.len(),
        );
        self.retired.push(entry);
    }

    /// Reclaims namespaces no owner adopted for `grace`, so a crashed scheduler
    /// cannot pin memory behind the workers it left attached.
    fn sweep(&mut self, now: Instant, grace: Duration) {
        let expired: Vec<String> = self
            .namespaces
            .iter()
            .filter(|(_, entry)| {
                entry
                    .detached_since
                    .is_some_and(|since| now.saturating_duration_since(since) >= grace)
            })
            .map(|(name, _)| name.to_owned())
            .collect();
        for name in expired {
            if let Some(entry) = self.namespaces.remove(&name) {
                invalidate_hashes(&entry);
                log_event!(
                    "namespace_reclaimed",
                    "namespace" = name,
                    "reason" = "grace_expired",
                    "sessions_lost" = entry.sessions.len(),
                );
            }
        }
    }

    fn used_bytes(&self) -> usize {
        self.namespaces.values().chain(&self.retired).fold(0, |used, entry| {
            used.saturating_add(lock_or_recover(&entry.cache, "cache").mapping.len())
        })
    }

    /// Bytes still held by namespaces whose scheduler is gone but whose workers
    /// keep the mapping.
    fn retired_bytes(&self) -> usize {
        self.retired
            .iter()
            .map(|entry| lock_or_recover(&entry.cache, "cache").mapping.len())
            .sum()
    }

    /// Sessions of live and retired namespaces, which is what the connection
    /// limit bounds.
    fn sessions(&self) -> usize {
        self.namespaces
            .values()
            .chain(&self.retired)
            .map(|entry| entry.sessions.len())
            .sum()
    }

    /// Live namespaces no owner holds, which stay adoptable until their grace
    /// period ends.
    fn detached(&self) -> usize {
        self.namespaces.values().filter(|entry| entry.detached()).count()
    }
}

fn same_uid(stream: &UnixStream) -> io::Result<bool> {
    // SAFETY: credentials points to a suitably sized live buffer.
    unsafe {
        let mut credentials: libc::ucred = std::mem::zeroed();
        let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        if libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut credentials as *mut libc::ucred).cast(),
            &mut length,
        ) < 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(credentials.uid == libc::geteuid())
    }
}

extern "C" fn request_shutdown(_signal: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// Makes `SIGTERM` and `SIGINT` ask the accept loop to stop.
fn install_shutdown_signal_handlers() -> io::Result<()> {
    for signal in [libc::SIGTERM, libc::SIGINT] {
        // SAFETY: the handler stores one atomic flag and returns, which is
        // async-signal-safe.
        let previous =
            unsafe { libc::signal(signal, request_shutdown as *const () as libc::sighandler_t) };
        if previous == libc::SIG_ERR {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Binds the listener, reclaiming the socket of a service that died without
/// unlinking it. A path that still accepts connections belongs to a live service
/// and is never unlinked.
fn bind_listener(path: &Path) -> io::Result<UnixListener> {
    match UnixStream::connect(path) {
        Ok(_) => Err(invalid(format!(
            "another service is listening on {}",
            path.display()
        ))),
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => UnixListener::bind(path),
            io::ErrorKind::ConnectionRefused => {
                match fs::remove_file(path) {
                    Ok(()) => log_event!("socket_reclaimed", "path" = path.display()),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
                UnixListener::bind(path)
            }
            _ => Err(io::Error::new(
                error.kind(),
                format!("cannot probe existing socket {}: {error}", path.display()),
            )),
        },
    }
}

/// Accepts connections until a shutdown signal arrives or the listener fails.
///
/// The listener is polled rather than blocking in `accept`, so an arriving
/// signal is not restarted away and shutdown stays bounded.
fn accept_connections(
    listener: &UnixListener,
    registry: &Arc<Mutex<Registry>>,
    limits: &Limits,
    grace: Option<Duration>,
) -> io::Result<()> {
    let active = Arc::new(AtomicUsize::new(0));
    let mut session = 0u64;
    while !SHUTDOWN.load(Ordering::SeqCst) {
        // The poll below is the only timer this process has, so a namespace whose
        // owner left is reclaimed from here instead of from a thread of its own.
        if let Some(grace) = grace {
            sweep_namespaces(registry, grace);
        }
        let mut pollfd = libc::pollfd {
            fd: listener.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pollfd describes one live descriptor for the duration of the
        // call; poll only reads and writes it.
        let ready = unsafe { libc::poll(&mut pollfd, 1, ACCEPT_POLL.as_millis() as libc::c_int) };
        if ready < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            log_event!("accept_failed", "error" = error);
            return Err(error);
        }
        if ready == 0 {
            continue;
        }
        let stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(error) => match error.raw_os_error() {
                // The peer gave up between poll and accept; nothing to serve.
                Some(libc::EINTR | libc::ECONNABORTED) => continue,
                Some(libc::EMFILE | libc::ENFILE | libc::ENOMEM) => {
                    log_event!("accept_failed", "error" = error);
                    thread::sleep(ACCEPT_BACKOFF);
                    continue;
                }
                _ => {
                    log_event!("accept_failed", "error" = error);
                    return Err(error);
                }
            },
        };
        match same_uid(&stream) {
            Ok(true) => {}
            Ok(false) => {
                log_event!("connection_rejected", "reason" = "peer_uid");
                continue;
            }
            Err(error) => {
                log_event!(
                    "connection_rejected",
                    "reason" = "peer_credentials",
                    "error" = error
                );
                continue;
            }
        }
        let live = active.load(Ordering::Relaxed);
        if live >= limits.max_connections {
            // The excess client is accepted and closed at once, so it fails
            // immediately instead of waiting in the listen backlog.
            log_event!(
                "connection_rejected",
                "reason" = "max_connections",
                "active" = live,
                "limit" = limits.max_connections,
            );
            continue;
        }
        session = session.checked_add(1).ok_or_else(|| invalid("session ID exhausted"))?;
        log_event!(
            "connection_accepted",
            "session" = session,
            "active" = live + 1
        );
        let slot = ConnectionSlot::acquire(&active);
        let registry = Arc::clone(registry);
        let limits = *limits;
        thread::spawn(move || {
            serve_connection(stream, registry, session, limits);
            drop(slot);
        });
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let Some(command) = args.command else {
        return run_service(args);
    };
    match command {
        Command::Bench(bench) => bench::run(bench),
    }
}

fn run_service(args: Args) -> io::Result<()> {
    let socket = args.socket.ok_or_else(|| invalid("--socket is required"))?;
    let capacity_bytes =
        args.capacity_bytes.ok_or_else(|| invalid("--capacity-bytes is required"))?;
    let parent = socket.parent().ok_or_else(|| invalid("socket directory required"))?;
    let metadata = fs::metadata(parent)?;
    // SAFETY: geteuid takes no arguments and returns the calling process UID.
    let uid = unsafe { libc::geteuid() };
    if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 || metadata.uid() != uid {
        return Err(invalid("socket directory must be private (mode 0700)"));
    }
    if capacity_bytes == 0 {
        return Err(invalid("capacity must be positive"));
    }
    if args.max_connections == 0 {
        return Err(invalid("connection limit must be positive"));
    }
    install_shutdown_signal_handlers()?;
    let listener = bind_listener(&socket)?;
    let _socket_path = SocketPath(&socket);
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let limits = Limits {
        max_connections: args.max_connections,
        idle_timeout: (args.idle_timeout_secs > 0)
            .then(|| Duration::from_secs(args.idle_timeout_secs)),
    };
    log_event!(
        "start",
        "socket" = socket.display(),
        "capacity_bytes" = capacity_bytes,
        "pid" = std::process::id(),
        "max_connections" = limits.max_connections,
        "idle_timeout_secs" = args.idle_timeout_secs,
        "pending_timeout_secs" = args.pending_timeout_secs,
        "namespace_grace_secs" = args.namespace_grace_secs,
        "pin_lease_bytes" = args.pin_lease_bytes,
        "eviction_policy" = format!("{:?}", args.eviction_policy),
        "spill_watermark_percent" = args.spill_watermark_percent,
        "prefetch_cooldown_ms" = args.prefetch_cooldown_ms,
        "prefetch_max_chunks" = args.prefetch_max_chunks,
    );
    let registry = Arc::new(Mutex::new(Registry {
        namespaces: HashMap::new(),
        retired: Vec::new(),
        budget: capacity_bytes,
        rswap_attach: args.rswap_attach,
        remote: args
            .remote_redis_url
            .as_deref()
            .map(RedisTier::new)
            .transpose()?
            .map(|tier| Arc::new(tier) as Arc<dyn RemoteTier>),
        remote_timeout: Duration::from_millis(args.remote_timeout_ms),
        pin_lease_bytes: args.pin_lease_bytes,
        eviction_policy: args.eviction_policy,
        spill: cgroup_reclaim_path()
            .filter(|_| args.spill_watermark_percent > 0 && args.spill_watermark_percent < 100)
            .map(|path| (path, args.spill_watermark_percent)),
        prefetch: PrefetchLimits {
            cooldown: Duration::from_millis(args.prefetch_cooldown_ms),
            max_chunks: args.prefetch_max_chunks,
        },
        pending_timeout: (args.pending_timeout_secs > 0)
            .then(|| Duration::from_secs(args.pending_timeout_secs)),
        started: Instant::now(),
    }));
    let grace =
        (args.namespace_grace_secs > 0).then(|| Duration::from_secs(args.namespace_grace_secs));
    let result = accept_connections(&listener, &registry, &limits, grace);
    let reason = if SHUTDOWN.load(Ordering::SeqCst) {
        "signal"
    } else {
        "listener_error"
    };
    log_event!("shutdown", "reason" = reason);
    result
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::Parser;

    use super::*;

    #[test]
    fn rswap_attach_defaults_to_sysfs_path() {
        let args = Args::try_parse_from([
            "vllm-kv-cache-service",
            "--socket",
            "/tmp/cache.sock",
            "--capacity-bytes",
            "4096",
            "--rswap-attach",
        ])
        .unwrap();
        assert_eq!(
            args.rswap_attach.as_deref(),
            Some(Path::new("/sys/kernel/mm/xswap/rswap_attach"))
        );
    }

    #[test]
    fn connection_and_idle_limits_have_defaults() {
        let args = Args::try_parse_from([
            "vllm-kv-cache-service",
            "--socket",
            "/tmp/cache.sock",
            "--capacity-bytes",
            "4096",
        ])
        .unwrap();
        assert_eq!(args.max_connections, 128);
        assert_eq!(args.idle_timeout_secs, 900);
        assert_eq!(args.pending_timeout_secs, 60);
        assert_eq!(args.namespace_grace_secs, 300);
        assert_eq!(args.pin_lease_bytes, 0);
        assert_eq!(args.prefetch_cooldown_ms, 2000);
        assert_eq!(args.prefetch_max_chunks, 32);
    }
}
