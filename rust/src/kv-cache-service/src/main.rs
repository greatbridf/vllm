// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
mod cache;
mod protocol;

use std::collections::HashMap;
use std::fs;
use std::io;
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use clap::Parser;
use serde_json::{Value, json};

use cache::{Cache, check_keys, check_request};
use protocol::Request;

#[derive(Parser)]
struct Args {
    /// Socket in a directory accessible only to the service owner.
    #[arg(long)]
    socket: PathBuf,
    /// Aggregate upper bound on bytes allocated across cache namespaces.
    #[arg(long)]
    capacity_bytes: usize,
}

struct State {
    caches: HashMap<String, Cache>,
    retired: Vec<Cache>,
    budget: usize,
}

fn invalid(message: &str) -> io::Error {
    io::Error::other(message)
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
    stream: &mut UnixStream,
    state: &mut State,
    session: u64,
    namespace: &mut Option<String>,
) -> io::Result<()> {
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
                return Err(invalid("invalid open handshake"));
            }
            if state.caches.contains_key(&name) {
                return Err(invalid("namespace already owned by another scheduler"));
            }
            let used: usize = state
                .caches
                .values()
                .chain(state.retired.iter())
                .map(|cache| cache.mapping.len())
                .sum();
            let cache = Cache::new(
                session,
                fingerprint,
                worker_bytes,
                world_size,
                num_chunks,
                state.budget.saturating_sub(used),
            )?;
            protocol::write_response(stream, &geometry(&cache))?;
            state.caches.insert(name.clone(), cache);
            *namespace = Some(name);
        }
        Request::Attach {
            version,
            namespace: name,
            fingerprint,
            rank,
        } => {
            if version != 1 || namespace.is_some() {
                return Err(invalid("invalid attach handshake"));
            }
            let cache = state
                .caches
                .get_mut(&name)
                .ok_or_else(|| invalid("scheduler namespace not available"))?;
            if fingerprint != cache.fingerprint
                || rank >= cache.world_size
                || cache.attached_ranks.contains(&rank)
            {
                return Err(invalid("incompatible or duplicate worker"));
            }
            protocol::write_response(stream, &geometry(cache))?;
            protocol::send_fd(stream, cache.file.as_fd())?;
            cache.attached_ranks.insert(rank);
            cache.sessions.insert(session);
            *namespace = Some(name);
        }
        command => {
            let name = namespace.as_ref().ok_or_else(|| invalid("handshake required"))?;
            let cache = state
                .caches
                .get_mut(name)
                .ok_or_else(|| invalid("cache namespace invalidated"))?;
            if cache.owner != session {
                return Err(invalid("workers cannot mutate cache metadata"));
            }
            let response = match command {
                Request::Lookup { key, request } => {
                    check_keys(std::slice::from_ref(&key))?;
                    check_request(&request)?;
                    json!({"ok": true, "status": cache.lookup(&key, &request)})
                }
                Request::Load { keys, request } => {
                    check_keys(&keys)?;
                    check_request(&request)?;
                    json!({"ok": true, "slots": cache.load(&keys, &request)?})
                }
                Request::Release { keys, request } => {
                    check_keys(&keys)?;
                    check_request(&request)?;
                    cache.release(&keys, &request);
                    json!({"ok": true})
                }
                Request::ReleaseAll { request } => {
                    check_request(&request)?;
                    cache.release_request(&request);
                    json!({"ok": true})
                }
                Request::Store { keys } => {
                    check_keys(&keys)?;
                    match cache.store(&keys) {
                        Some((keys, slots, evicted)) => {
                            json!({"ok": true, "keys": keys, "slots": slots, "evicted": evicted})
                        }
                        None => json!({"ok": true, "full": true}),
                    }
                }
                Request::Commit { keys, success } => {
                    check_keys(&keys)?;
                    cache.commit(&keys, success)?;
                    json!({"ok": true})
                }
                Request::Touch { keys } => {
                    check_keys(&keys)?;
                    cache.touch(&keys);
                    json!({"ok": true})
                }
                Request::Reset => {
                    cache.reset()?;
                    json!({"ok": true})
                }
                Request::Stats => json!({"ok": true, "stats": cache.stats()}),
                Request::Open { .. } | Request::Attach { .. } => unreachable!(),
            };
            protocol::write_response(stream, &response)?;
        }
    }
    Ok(())
}

fn serve(mut stream: UnixStream, state: Arc<Mutex<State>>, session: u64) {
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
    let mut namespace = None;
    while let Ok(request) = protocol::read_request(&mut stream) {
        let Ok(mut state) = state.lock() else { break };
        if let Err(error) = handle(request, &mut stream, &mut state, session, &mut namespace) {
            let _ = protocol::write_response(
                &mut stream,
                &json!({"ok": false, "error": format!("{error}")}),
            );
            break;
        }
    }
    if let Some(name) = namespace
        && let Ok(mut state) = state.lock()
    {
        // Never recycle an abandoned in-flight DMA destination. Remove the
        // entire namespace; new clients receive a different memfd allocation.
        if state.caches.get(&name).is_some_and(|cache| cache.sessions.contains(&session))
            && let Some(mut cache) = state.caches.remove(&name)
        {
            cache.sessions.remove(&session);
            if !cache.sessions.is_empty() {
                state.retired.push(cache);
            }
        }
        for cache in &mut state.retired {
            cache.sessions.remove(&session);
        }
        state.retired.retain(|cache| !cache.sessions.is_empty());
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
    let state = Arc::new(Mutex::new(State {
        caches: HashMap::new(),
        retired: Vec::new(),
        budget: args.capacity_bytes,
    }));
    let mut session = 0u64;
    for connection in listener.incoming() {
        let stream = connection?;
        if !same_uid(&stream)? {
            continue;
        }
        session = session.checked_add(1).ok_or_else(|| invalid("session ID exhausted"))?;
        let state = Arc::clone(&state);
        thread::spawn(move || serve(stream, state, session));
    }
    Ok(())
}
