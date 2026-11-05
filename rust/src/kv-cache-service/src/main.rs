// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use clap::Parser;
use serde_json::{Value, json};
use vllm_kv_cache_service::cache::{Cache, check_keys, check_request, namespace_bytes};
use vllm_kv_cache_service::hash::{HashFactory, RswapConfig};
use vllm_kv_cache_service::protocol::{self, Request};

const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Parser)]
struct Args {
    /// Socket in a directory accessible only to the service owner.
    #[arg(long)]
    socket: PathBuf,
    /// Aggregate upper bound on bytes allocated across cache namespaces.
    #[arg(long)]
    capacity_bytes: usize,
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

fn invalid(message: &str) -> io::Error {
    io::Error::other(message)
}

fn reject(message: impl std::fmt::Display) -> Failure {
    Failure::Rejected(format!("{message}"))
}

fn lock_registry(registry: &Mutex<Registry>) -> io::Result<MutexGuard<'_, Registry>> {
    registry.lock().map_err(|_| invalid("registry lock is poisoned"))
}

fn lock_cache(cache: &Mutex<Cache>) -> io::Result<MutexGuard<'_, Cache>> {
    cache.lock().map_err(|_| invalid("cache lock is poisoned"))
}

fn log(session: u64, event: &str, detail: impl std::fmt::Display) {
    eprintln!("vllm-kv-cache-service: session {session}: {event}: {detail}");
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
            let mut registry = lock_registry(registry)?;
            if registry.namespaces.contains_key(&name) {
                return Err(reject("namespace already owned by another scheduler"));
            }
            let used = registry.used_bytes()?;
            let remaining = registry.budget.saturating_sub(used);
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
            let mut registry = lock_registry(registry)?;
            let entry = registry
                .namespaces
                .get_mut(&name)
                .ok_or_else(|| reject("scheduler namespace not available"))?;
            let (frame, fd) = {
                let cache = lock_cache(&entry.cache)?;
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
            *namespace = Some(name);
            Ok(Reply::Attach { frame, fd })
        }
        command => {
            let name = namespace.as_deref().ok_or_else(|| reject("handshake required"))?;
            let cache = {
                let registry = lock_registry(registry)?;
                let entry = registry
                    .namespaces
                    .get(name)
                    .ok_or_else(|| reject("cache namespace invalidated"))?;
                Arc::clone(&entry.cache)
            };
            let mut cache = lock_cache(&cache)?;
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
                    cache.release(&keys, &request);
                    json!({"ok": true})
                }
                Request::ReleaseAll { request } => {
                    check_request(&request).map_err(reject)?;
                    cache.release_request(&request);
                    json!({"ok": true})
                }
                Request::Store { keys } => {
                    check_keys(&keys).map_err(reject)?;
                    // The reply echoes every key; reject before reserving slots.
                    if !protocol::store_reply_fits(&keys) {
                        return Err(reject("batch too large for one control frame"));
                    }
                    match cache.store(&keys) {
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
                Request::Commit { keys, success } => {
                    check_keys(&keys).map_err(reject)?;
                    cache.commit(&keys, success).map_err(reject)?;
                    json!({"ok": true})
                }
                Request::Touch { keys } => {
                    check_keys(&keys).map_err(reject)?;
                    cache.touch(&keys);
                    json!({"ok": true})
                }
                Request::Reset => {
                    cache.reset().map_err(reject)?;
                    json!({"ok": true})
                }
                Request::Stats => json!({"ok": true, "stats": cache.stats()}),
                Request::Open { .. } | Request::Attach { .. } => unreachable!(),
            };
            Ok(Reply::Frame(protocol::encode_response(&response)?))
        }
    }
}

fn serve(mut stream: UnixStream, registry: Arc<Mutex<Registry>>, session: u64) {
    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
    let mut namespace = None;
    loop {
        let request = match protocol::read_request(&mut stream) {
            Ok(request) => request,
            Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                log(session, "idle timeout", &error);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => {
                log(session, "invalid frame", &error);
                break;
            }
        };
        let written = match handle(request, &registry, session, &mut namespace) {
            Ok(Reply::Frame(frame)) => protocol::write_frame(&mut stream, &frame),
            Ok(Reply::Attach { frame, fd }) => protocol::write_frame(&mut stream, &frame)
                .and_then(|()| protocol::send_fd(&stream, fd.as_fd())),
            Err(Failure::Rejected(message)) => {
                log(session, "rejected request", &message);
                protocol::encode_response(&json!({"ok": false, "error": message}))
                    .and_then(|frame| protocol::write_frame(&mut stream, &frame))
            }
            Err(Failure::Closed(error)) => {
                log(session, "closing connection", &error);
                break;
            }
        };
        if let Err(error) = written {
            // A partially written frame cannot be repaired on this stream.
            log(session, "write failed", &error);
            break;
        }
    }
    release_session(&registry, namespace.as_deref(), session);
}

/// Invalidates the departed session's namespace and frees retired memory once no
/// session holds it. An abandoned in-flight DMA destination is never recycled, so
/// the namespace is retired rather than handed to a new client.
fn release_session(registry: &Mutex<Registry>, name: Option<&str>, session: u64) {
    let Ok(mut registry) = lock_registry(registry) else {
        return;
    };
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
            log(session, "namespace invalidated", name);
        } else {
            log(session, "namespace retired", name);
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
    if let Ok(mut cache) = lock_cache(&namespace.cache) {
        cache.invalidate_hashes();
    }
}

impl Registry {
    fn used_bytes(&self) -> io::Result<usize> {
        let mut used: usize = 0;
        for entry in self.namespaces.values().chain(&self.retired) {
            used = used.saturating_add(lock_cache(&entry.cache)?.mapping.len());
        }
        Ok(used)
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
    // Refuse to unlink an existing socket: it may belong to a live service.
    let listener = UnixListener::bind(&args.socket)?;
    fs::set_permissions(&args.socket, fs::Permissions::from_mode(0o600))?;
    let registry = Arc::new(Mutex::new(Registry {
        namespaces: HashMap::new(),
        retired: Vec::new(),
        budget: args.capacity_bytes,
        rswap_attach: args.rswap_attach,
    }));
    let mut session = 0u64;
    for connection in listener.incoming() {
        let stream = connection?;
        if !same_uid(&stream)? {
            continue;
        }
        session = session.checked_add(1).ok_or_else(|| invalid("session ID exhausted"))?;
        let registry = Arc::clone(&registry);
        thread::spawn(move || serve(stream, registry, session));
    }
    Ok(())
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
}
