// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Process-level tests: spawn the real service and speak raw control frames.
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tempfile::TempDir;

const CAPACITY: usize = 64 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(5);
const STARTUP: Duration = Duration::from_secs(10);

struct Service {
    child: Child,
    path: PathBuf,
    dir: Arc<TempDir>,
}

impl Service {
    fn start() -> Service {
        Service::start_with(&[])
    }

    fn start_with(extra_args: &[&str]) -> Service {
        let dir = tempfile::tempdir().unwrap();
        fs_permissions(dir.path(), 0o700);
        Service::start_at(Arc::new(dir), extra_args)
    }

    /// Starts a service on an existing temporary directory's socket path.
    fn start_at(dir: Arc<TempDir>, extra_args: &[&str]) -> Service {
        let path = dir.path().join("cache.sock");
        let child = Command::new(env!("CARGO_BIN_EXE_vllm-kv-cache-service"))
            .arg("--socket")
            .arg(&path)
            .arg("--capacity-bytes")
            .arg(CAPACITY.to_string())
            .args(extra_args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let service = Service { child, path, dir };
        service.wait_until_listening();
        service
    }

    fn wait_until_listening(&self) {
        let deadline = Instant::now() + STARTUP;
        while !self.path.exists() {
            assert!(Instant::now() < deadline, "service startup timed out");
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// Waits until the process accepts a new connection, which a left-over
    /// socket path alone cannot prove.
    fn wait_until_serving(&mut self) {
        let deadline = Instant::now() + STARTUP;
        loop {
            if UnixStream::connect(&self.path).is_ok() {
                return;
            }
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!(
                    "service exited before serving ({status}): {}",
                    self.stderr()
                );
            }
            assert!(
                Instant::now() < deadline,
                "service never accepted a connection"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Kills the service without a chance to clean up, then hands back its
    /// directory so a replacement can start on the socket path it left behind.
    fn kill_hard(mut self) -> Arc<TempDir> {
        self.child.kill().unwrap();
        self.child.wait().unwrap();
        assert!(
            self.path.exists(),
            "SIGKILL must leave the socket path behind"
        );
        Arc::clone(&self.dir)
    }

    fn signal_and_wait(&mut self, signal: libc::c_int) -> (ExitStatus, String) {
        // SAFETY: the pid names the child this test spawned.
        assert_eq!(
            unsafe { libc::kill(self.child.id() as libc::pid_t, signal) },
            0
        );
        let deadline = Instant::now() + STARTUP;
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "service ignored signal {signal}");
            thread::sleep(Duration::from_millis(10));
        };
        (status, self.stderr())
    }

    /// Everything the service wrote to stderr; only valid once it exited.
    fn stderr(&mut self) -> String {
        let mut text = String::new();
        if let Some(stderr) = self.child.stderr.as_mut() {
            let _ = stderr.read_to_string(&mut text);
        }
        text
    }

    fn connect(&self) -> Client {
        Client::connect(&self.path)
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn fs_permissions(path: &Path, mode: u32) {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

struct Client {
    stream: UnixStream,
}

impl Client {
    fn connect(path: &Path) -> Client {
        let stream = UnixStream::connect(path).unwrap();
        stream.set_read_timeout(Some(TIMEOUT)).unwrap();
        stream.set_write_timeout(Some(TIMEOUT)).unwrap();
        Client { stream }
    }

    fn call(&mut self, request: Value) -> Value {
        self.send(&request);
        self.receive()
    }

    fn send(&mut self, request: &Value) {
        self.stream.write_all(&encode(request)).unwrap();
    }

    fn receive(&mut self) -> Value {
        let mut header = [0; 4];
        self.stream
            .read_exact(&mut header)
            .unwrap_or_else(|error| panic!("no response frame: {error}"));
        let mut payload = vec![0; u32::from_be_bytes(header) as usize];
        self.stream.read_exact(&mut payload).unwrap();
        serde_json::from_slice(&payload).unwrap()
    }
}

fn encode(request: &Value) -> Vec<u8> {
    let payload = serde_json::to_vec(request).unwrap();
    let mut frame = (payload.len() as u32).to_be_bytes().to_vec();
    frame.extend_from_slice(&payload);
    frame
}

fn open(client: &mut Client, namespace: &str) -> Value {
    client.call(json!({
        "op": "open",
        "version": 1,
        "namespace": namespace,
        "fingerprint": "a".repeat(64),
        "worker_bytes": 64,
        "world_size": 2,
        "num_chunks": 2,
    }))
}

fn attach(client: &mut Client, namespace: &str, rank: usize) -> Value {
    client.call(json!({
        "op": "attach",
        "version": 1,
        "namespace": namespace,
        "fingerprint": "a".repeat(64),
        "rank": rank,
    }))
}

/// Receives the descriptor frame the service sends after the attach geometry.
fn receive_descriptor(stream: &UnixStream) -> Vec<RawFd> {
    let mut marker = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: marker.as_mut_ptr().cast(),
        iov_len: 1,
    };
    // usize storage provides the alignment required by cmsghdr.
    let mut control = [0usize; 8];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = control.len() * size_of::<usize>();
    // SAFETY: all pointers refer to live, aligned buffers for the call.
    let received = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
    assert_eq!(received, 1, "expected a one-byte descriptor frame");
    assert_eq!(marker[0], b'F', "descriptor frame must carry the F marker");

    let mut descriptors = Vec::new();
    // SAFETY: CMSG_FIRSTHDR/CMSG_NXTHDR walk the control buffer recvmsg filled.
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(&msg);
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let data = libc::CMSG_DATA(header).cast::<libc::c_int>();
                let length = (*header).cmsg_len - libc::CMSG_LEN(0) as usize;
                let count = length / size_of::<libc::c_int>();
                descriptors.extend(std::slice::from_raw_parts(data, count).iter().copied());
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
    }
    descriptors
}

#[test]
fn rejected_request_keeps_the_namespace_alive() {
    let service = Service::start();
    let mut client = service.connect();
    assert_eq!(open(&mut client, "engine-a")["ok"], true);

    let rejection = client.call(json!({"op": "lookup", "key": "not-hex", "request": "r"}));
    assert_eq!(rejection["ok"], false);
    assert!(rejection["error"].as_str().unwrap().contains("invalid or duplicate cache keys"));

    let stats = client.call(json!({"op": "stats"}));
    assert_eq!(stats["ok"], true);
    assert_eq!(stats["stats"]["entries"], 0);
}

#[test]
fn worker_mutation_is_rejected_without_invalidating_the_namespace() {
    let service = Service::start();
    let mut scheduler = service.connect();
    assert_eq!(open(&mut scheduler, "engine-b")["ok"], true);

    let mut worker = service.connect();
    let geometry = attach(&mut worker, "engine-b", 0);
    assert_eq!(geometry["ok"], true);
    let descriptors = receive_descriptor(&worker.stream);
    assert_eq!(descriptors.len(), 1);
    // SAFETY: the descriptor is owned by this test and no longer used.
    assert_eq!(unsafe { libc::close(descriptors[0]) }, 0);

    let rejection = worker.call(json!({"op": "store", "keys": ["aabb"]}));
    assert_eq!(rejection["ok"], false);
    assert!(rejection["error"].as_str().unwrap().contains("workers"));

    // The worker session is still open, and the scheduler namespace survived.
    assert_eq!(worker.call(json!({"op": "stats"}))["ok"], true);
    assert_eq!(scheduler.call(json!({"op": "stats"}))["ok"], true);
}

#[test]
fn slow_reader_does_not_block_another_namespace() {
    let service = Service::start();
    let mut stalled = service.connect();
    assert_eq!(open(&mut stalled, "engine-slow")["ok"], true);
    let mut other = service.connect();
    assert_eq!(open(&mut other, "engine-other")["ok"], true);

    // The stalled client never reads, so the service blocks writing its replies
    // once this socket's receive queue is full. Two waits of 10 ms that each find
    // no room prove the service is stalled rather than between requests.
    let request = encode(&json!({"op": "lookup", "key": "aa", "request": "r"}));
    stalled.stream.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + TIMEOUT;
    let mut stalls = 0;
    while stalls < 2 {
        match stalled.stream.write_all(&request) {
            Ok(()) => stalls = 0,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                stalls += 1;
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("stalled client failed: {error}"),
        }
        assert!(
            Instant::now() < deadline,
            "the stalled client never filled its socket"
        );
    }

    let start = Instant::now();
    let stats = other.call(json!({"op": "stats"}));
    let elapsed = start.elapsed();
    assert_eq!(stats["ok"], true);
    assert!(
        elapsed < Duration::from_secs(1),
        "stats took {elapsed:?} behind a stalled writer"
    );
}

#[test]
fn attach_reply_carries_exactly_one_descriptor() {
    let service = Service::start();
    let mut scheduler = service.connect();
    assert_eq!(open(&mut scheduler, "engine-fd")["ok"], true);

    let mut worker = service.connect();
    let geometry = attach(&mut worker, "engine-fd", 0);
    assert_eq!(geometry["ok"], true);
    assert_eq!(geometry["version"], 1);
    assert_eq!(geometry["num_chunks"], 2);

    let descriptors = receive_descriptor(&worker.stream);
    assert_eq!(
        descriptors.len(),
        1,
        "attach must send exactly one descriptor"
    );
    // SAFETY: recvmsg returned a fresh descriptor owned by this test.
    assert!(unsafe { libc::fcntl(descriptors[0], libc::F_GETFD) } >= 0);
    // SAFETY: the descriptor is owned by this test and no longer used.
    assert_eq!(unsafe { libc::close(descriptors[0]) }, 0);
}

#[test]
fn store_batch_that_cannot_be_answered_reserves_no_slot() {
    let service = Service::start();
    let mut client = service.connect();
    // Enough chunks that the cache would accept the whole batch.
    let opened = client.call(json!({
        "op": "open",
        "version": 1,
        "namespace": "engine-big",
        "fingerprint": "a".repeat(64),
        "worker_bytes": 64,
        "world_size": 2,
        "num_chunks": 4096,
    }));
    assert_eq!(opened["ok"], true);

    // Longest keys that still fit one request frame: echoing them back, twice
    // with a slot per key, exceeds one control frame.
    let keys: Vec<_> = (0..4096).map(|index| format!("{index:0250x}")).collect();
    let rejection = client.call(json!({"op": "store", "keys": keys}));
    assert_eq!(rejection["ok"], false);
    assert!(
        rejection["error"]
            .as_str()
            .unwrap()
            .contains("batch too large for one control frame")
    );

    let stats = client.call(json!({"op": "stats"}));
    assert_eq!(stats["ok"], true);
    assert_eq!(
        stats["stats"]["entries"], 0,
        "a rejected batch must not reserve"
    );
    assert_eq!(stats["stats"]["pending"], 0);
}

#[test]
fn starts_after_reclaiming_a_stale_socket() {
    let first = Service::start();
    let dir = first.kill_hard();

    let mut service = Service::start_at(Arc::clone(&dir), &[]);
    service.wait_until_serving();
    let mut client = service.connect();
    assert_eq!(open(&mut client, "engine-reclaimed")["ok"], true);
    assert_eq!(client.call(json!({"op": "stats"}))["ok"], true);
}

#[test]
fn refuses_to_start_when_another_service_is_listening() {
    let service = Service::start();

    let second = Command::new(env!("CARGO_BIN_EXE_vllm-kv-cache-service"))
        .arg("--socket")
        .arg(&service.path)
        .arg("--capacity-bytes")
        .arg(CAPACITY.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap();

    assert!(
        !second.status.success(),
        "a second service took over a live socket"
    );
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(
        stderr.contains("another service is listening on"),
        "unexpected refusal: {stderr}"
    );

    let mut client = service.connect();
    assert_eq!(open(&mut client, "engine-live")["ok"], true);
}

#[test]
fn sigterm_unlinks_the_socket_and_exits_zero() {
    let mut service = Service::start();
    let mut client = service.connect();
    assert_eq!(open(&mut client, "engine-term")["ok"], true);

    let (status, stderr) = service.signal_and_wait(libc::SIGTERM);

    assert!(status.success(), "exit status {status}: {stderr}");
    assert!(
        !service.path.exists(),
        "a clean stop must unlink the socket path"
    );
    assert!(
        stderr.contains("kv-cache-service: shutdown reason=signal"),
        "missing shutdown event: {stderr}"
    );
}

#[test]
fn connection_cap_rejects_excess_clients() {
    let service = Service::start_with(&["--max-connections", "2"]);
    let mut first = service.connect();
    assert_eq!(open(&mut first, "engine-cap-a")["ok"], true);
    let mut second = service.connect();
    assert_eq!(open(&mut second, "engine-cap-b")["ok"], true);

    let mut excess = service.connect();
    let mut buffer = [0u8; 4];
    let read = excess
        .stream
        .read(&mut buffer)
        .expect("the excess client must be closed, not reset");
    assert_eq!(read, 0, "the excess client must observe EOF");

    assert_eq!(first.call(json!({"op": "stats"}))["ok"], true);
    assert_eq!(second.call(json!({"op": "stats"}))["ok"], true);
}

#[test]
fn idle_connection_is_closed_after_the_timeout() {
    let service = Service::start_with(&["--idle-timeout-secs", "1"]);
    let mut client = service.connect();

    let started = Instant::now();
    let mut buffer = [0u8; 4];
    let read = client
        .stream
        .read(&mut buffer)
        .expect("the idle connection must be closed, not left open");
    let elapsed = started.elapsed();

    assert_eq!(read, 0, "the idle peer must observe EOF");
    assert!(
        elapsed >= Duration::from_millis(500),
        "closed before the idle timeout: {elapsed:?}"
    );
}
