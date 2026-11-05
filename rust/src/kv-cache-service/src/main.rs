// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
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
use vllm_kv_cache_service::cache::{Cache, Released, check_keys, check_request, namespace_bytes};
use vllm_kv_cache_service::hash::{HashFactory, RswapConfig};
use vllm_kv_cache_service::protocol::{self, Request};

#[macro_use]
mod log;

const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
/// Upper bound on how long a shutdown signal waits for the accept loop, which
/// polls the listener instead of blocking in `accept`.
const ACCEPT_POLL: Duration = Duration::from_millis(200);
/// Backoff after a transient accept failure such as descriptor exhaustion.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(50);

/// Set by the SIGTERM and SIGINT handler, read by the accept loop.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[derive(Parser)]
struct Args {
    /// Socket in a directory accessible only to the service owner.
    #[arg(long)]
    socket: PathBuf,
    /// Aggregate upper bound on bytes allocated across cache namespaces.
    #[arg(long)]
    capacity_bytes: usize,
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
    /// Attach the rswap digest BPF program through sysfs; disabled by default.
    /// Defaults to /sys/kernel/mm/xswap/rswap_attach when no path is given.
    #[arg(
        long,
        visible_alias = "hash-attach",
        num_args = 0..=1,
        default_missing_value = "/sys/kernel/mm/xswap/rswap_attach"
    )]
    rswap_attach: Option<PathBuf>,
}

/// Per-connection bounds taken from the command line.
#[derive(Clone, Copy)]
struct Limits {
    max_connections: usize,
    idle_timeout: Option<Duration>,
}

/// One engine namespace: the shared cache mapping plus the sessions using it.
struct Namespace {
    cache: Arc<Mutex<Cache>>,
    fingerprint: String,
    attached_ranks: HashSet<usize>,
    sessions: HashSet<u64>,
}

/// Namespaces reachable by name. Its lock is only held to clone a cache handle or
/// to insert and remove an entry, never across socket I/O.
struct Registry {
    namespaces: HashMap<String, Namespace>,
    retired: Vec<Namespace>,
    budget: usize,
    rswap_attach: Option<PathBuf>,
    pending_timeout: Option<Duration>,
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

fn geometry(cache: &Cache) -> Value {
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

fn handle(
    request: Request,
    registry: &Mutex<Registry>,
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
            let mut registry = lock_or_recover(registry, "registry");
            if registry.namespaces.contains_key(&name) {
                return Err(reject("namespace already owned by another scheduler"));
            }
            let remaining = registry.budget.saturating_sub(registry.used_bytes());
            if size > remaining {
                return Err(reject("cache exceeds service capacity"));
            }
            let factory = registry.rswap_attach.as_ref().map(|path| RswapConfig {
                attach_path: path.clone(),
            });
            let cache = Cache::with_hashes(
                session,
                worker_bytes,
                world_size,
                num_chunks,
                remaining,
                factory.as_ref().map(|factory| factory as &dyn HashFactory),
            )?;
            let frame = protocol::encode_response(&geometry(&cache))?;
            log_event!(
                "namespace_opened",
                "session" = session,
                "namespace" = name,
                "num_chunks" = cache.num_chunks,
                "row_bytes" = cache.row_bytes,
                "bytes" = cache.mapping.len(),
                "budget_left_bytes" = remaining - size,
            );
            registry.namespaces.insert(
                name.clone(),
                Namespace {
                    cache: Arc::new(Mutex::new(cache)),
                    fingerprint,
                    attached_ranks: HashSet::new(),
                    sessions: HashSet::from([session]),
                },
            );
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
            let mut registry = lock_or_recover(registry, "registry");
            let entry = registry
                .namespaces
                .get_mut(&name)
                .ok_or_else(|| reject("scheduler namespace not available"))?;
            let (frame, fd) = {
                let cache = lock_or_recover(&entry.cache, "cache");
                if fingerprint != entry.fingerprint
                    || rank >= cache.world_size
                    || entry.attached_ranks.contains(&rank)
                {
                    return Err(reject("incompatible or duplicate worker"));
                }
                let frame = protocol::encode_response(&geometry(&cache))?;
                (frame, protocol::duplicate_fd(cache.file.as_fd())?)
            };
            entry.attached_ranks.insert(rank);
            entry.sessions.insert(session);
            log_event!(
                "worker_attached",
                "session" = session,
                "namespace" = name,
                "rank" = rank
            );
            *namespace = Some(name);
            Ok(Reply::Attach { frame, fd })
        }
        command => {
            let name = namespace.as_deref().ok_or_else(|| reject("handshake required"))?;
            let (cache, pending_timeout) = {
                let registry = lock_or_recover(registry, "registry");
                let entry = registry
                    .namespaces
                    .get(name)
                    .ok_or_else(|| reject("cache namespace invalidated"))?;
                (Arc::clone(&entry.cache), registry.pending_timeout)
            };
            let mut cache = lock_or_recover(&cache, "cache");
            // Read-only stats stay available to attached workers; every other
            // command mutates scheduler-owned cache metadata.
            if cache.owner != session && !matches!(command, Request::Stats) {
                return Err(reject("workers cannot mutate cache metadata"));
            }
            let response = match command {
                Request::Lookup { key, request } => {
                    check_keys(std::slice::from_ref(&key)).map_err(reject)?;
                    check_request(&request).map_err(reject)?;
                    json!({"ok": true, "status": cache.lookup(&key, &request)})
                }
                Request::Load { keys, request } => {
                    check_keys(&keys).map_err(reject)?;
                    check_request(&request).map_err(reject)?;
                    json!({"ok": true, "slots": cache.load(&keys, &request).map_err(reject)?})
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
                    // The reply echoes every key; reject before reserving slots.
                    if !protocol::store_reply_fits(&keys) {
                        return Err(reject("batch too large for one control frame"));
                    }
                    reclaim_expired(&mut cache, name, pending_timeout);
                    match cache.store(&keys, &request) {
                        Some((keys, slots, evicted)) => {
                            json!({
                                "ok": true,
                                "keys": keys,
                                "slots": slots,
                                "evicted": evicted
                            })
                        }
                        None => json!({"ok": true, "full": true}),
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
                    json!({"ok": true})
                }
                Request::Touch { keys } => {
                    check_keys(&keys).map_err(reject)?;
                    cache.touch(&keys);
                    json!({"ok": true})
                }
                Request::Reset { force } => {
                    reclaim_expired(&mut cache, name, pending_timeout);
                    cache.reset(force).map_err(reject)?;
                    json!({"ok": true})
                }
                Request::Stats => {
                    reclaim_expired(&mut cache, name, pending_timeout);
                    json!({"ok": true, "stats": cache.stats()})
                }
                Request::Open { .. } | Request::Attach { .. } => unreachable!(),
            };
            Ok(Reply::Frame(protocol::encode_response(&response)?))
        }
    }
}

fn released_reply(released: Released) -> Value {
    json!({"ok": true, "released": released.leases, "unknown": released.unknown})
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

fn serve(
    stream: &mut UnixStream,
    registry: &Mutex<Registry>,
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
        let written = match handle(request, registry, session, namespace) {
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

/// Invalidates the departed session's namespace and frees retired memory once no
/// session holds it. An abandoned in-flight DMA destination is never recycled, so
/// the namespace is retired rather than handed to a new client.
fn release_session(registry: &Mutex<Registry>, name: Option<&str>, session: u64) {
    let mut registry = lock_or_recover(registry, "registry");
    if let Some(name) = name
        && registry
            .namespaces
            .get(name)
            .is_some_and(|entry| entry.sessions.contains(&session))
        && let Some(mut entry) = registry.namespaces.remove(name)
    {
        entry.sessions.remove(&session);
        invalidate_hashes(&entry);
        if entry.sessions.is_empty() {
            log_event!(
                "namespace_invalidated",
                "session" = session,
                "namespace" = name
            );
        } else {
            log_event!(
                "namespace_retired",
                "session" = session,
                "namespace" = name,
                "reason" = "session_disconnect",
                "sessions_left" = entry.sessions.len(),
            );
            registry.retired.push(entry);
        }
    }
    for entry in &mut registry.retired {
        entry.sessions.remove(&session);
        invalidate_hashes(entry);
    }
    registry.retired.retain(|entry| !entry.sessions.is_empty());
}

fn invalidate_hashes(namespace: &Namespace) {
    lock_or_recover(&namespace.cache, "cache").invalidate_hashes();
}

impl Registry {
    fn used_bytes(&self) -> usize {
        self.namespaces.values().chain(&self.retired).fold(0, |used, entry| {
            used.saturating_add(lock_or_recover(&entry.cache, "cache").mapping.len())
        })
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
) -> io::Result<()> {
    let active = Arc::new(AtomicUsize::new(0));
    let mut session = 0u64;
    while !SHUTDOWN.load(Ordering::SeqCst) {
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
    let parent = args.socket.parent().ok_or_else(|| invalid("socket directory required"))?;
    let metadata = fs::metadata(parent)?;
    // SAFETY: geteuid takes no arguments and returns the calling process UID.
    let uid = unsafe { libc::geteuid() };
    if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 || metadata.uid() != uid {
        return Err(invalid("socket directory must be private (mode 0700)"));
    }
    if args.capacity_bytes == 0 {
        return Err(invalid("capacity must be positive"));
    }
    if args.max_connections == 0 {
        return Err(invalid("connection limit must be positive"));
    }
    install_shutdown_signal_handlers()?;
    let listener = bind_listener(&args.socket)?;
    let _socket_path = SocketPath(&args.socket);
    fs::set_permissions(&args.socket, fs::Permissions::from_mode(0o600))?;
    let limits = Limits {
        max_connections: args.max_connections,
        idle_timeout: (args.idle_timeout_secs > 0)
            .then(|| Duration::from_secs(args.idle_timeout_secs)),
    };
    log_event!(
        "start",
        "socket" = args.socket.display(),
        "capacity_bytes" = args.capacity_bytes,
        "pid" = std::process::id(),
        "max_connections" = limits.max_connections,
        "idle_timeout_secs" = args.idle_timeout_secs,
        "pending_timeout_secs" = args.pending_timeout_secs,
    );
    let registry = Arc::new(Mutex::new(Registry {
        namespaces: HashMap::new(),
        retired: Vec::new(),
        budget: args.capacity_bytes,
        rswap_attach: args.rswap_attach,
        pending_timeout: (args.pending_timeout_secs > 0)
            .then(|| Duration::from_secs(args.pending_timeout_secs)),
    }));
    let result = accept_connections(&listener, &registry, &limits);
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
    }
}
