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
    open_with(client, namespace, &"a".repeat(64), 2)
}

fn open_with(client: &mut Client, namespace: &str, fingerprint: &str, num_chunks: usize) -> Value {
    client.call(json!({
        "op": "open",
        "version": 1,
        "namespace": namespace,
        "fingerprint": fingerprint,
        "worker_bytes": 64,
        "world_size": 2,
        "num_chunks": num_chunks,
    }))
}

#[test]
fn no_eviction_retains_keys_when_the_namespace_is_full() {
    let service = Service::start_with(&["--eviction-policy", "none"]);
    let mut client = service.connect();
    assert_eq!(open(&mut client, "no-eviction")["ok"], true);
    assert_eq!(
        client.call(json!({"op": "store", "keys": ["aa", "bb"], "request": "writer"}))["ok"],
        true
    );
    assert_eq!(
        client.call(json!({
            "op": "commit", "keys": ["aa", "bb"], "request": "writer", "success": true
        }))["ok"],
        true
    );
    assert_eq!(
        client.call(json!({"op": "store", "keys": ["cc"], "request": "writer"})),
        json!({"ok": true, "full": true})
    );
    let retained = client.call(json!({
        "op": "store", "keys": ["aa", "bb"], "request": "writer"
    }));
    assert_eq!(retained["keys"], json!([]));
    assert_eq!(retained["evicted"], json!([]));
    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    assert_eq!(stats["evictions"], 0);
    assert_eq!(stats["entries"], 2);
    assert_eq!(stats["pending"], 0);
}

/// The replay a reconnecting client sends: adopt what a departed owner left, or
/// nothing.
fn adopt(client: &mut Client, namespace: &str) -> Value {
    client.call(json!({
        "op": "open",
        "version": 1,
        "namespace": namespace,
        "fingerprint": "a".repeat(64),
        "worker_bytes": 64,
        "world_size": 2,
        "num_chunks": 2,
        "adopt": true,
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

/// Attaches one rank and discards the descriptor, which these tests do not read.
fn attach_worker(service: &Service, namespace: &str, rank: usize) -> Client {
    let mut worker = service.connect();
    assert_eq!(attach(&mut worker, namespace, rank)["ok"], true);
    for descriptor in receive_descriptor(&worker.stream) {
        // SAFETY: the descriptor is owned by this test and never used again.
        assert_eq!(unsafe { libc::close(descriptor) }, 0);
    }
    worker
}

fn page_size() -> usize {
    // SAFETY: sysconf takes no pointer arguments.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(page > 0, "cannot determine the page size");
    page as usize
}

/// A test-owned writable mapping of a namespace, taken through an attached worker.
///
/// A chunk is only readable once a worker wrote its row, so a test that expects a
/// `hit` faults the row in through this mapping first. The session keeps the
/// namespace alive for as long as the mapping does: a worker's departure retires
/// it.
struct WorkerMapping {
    /// The session the descriptor arrived on.
    worker: Client,
    base: *mut u8,
    size: usize,
    page_size: usize,
}

impl WorkerMapping {
    fn attach(service: &Service, namespace: &str, rank: usize) -> WorkerMapping {
        let mut worker = service.connect();
        assert_eq!(attach(&mut worker, namespace, rank)["ok"], true);
        let descriptors = receive_descriptor(&worker.stream);
        assert_eq!(descriptors.len(), 1, "attach must send one descriptor");
        let fd = descriptors[0];
        let size = file_size(fd);
        // SAFETY: fd is the sealed mapping the service handed this process, and
        // `Drop` unmaps exactly this range.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED, "cannot map the service mapping");
        // SAFETY: the descriptor is owned by this test and the mapping holds it.
        assert_eq!(unsafe { libc::close(fd) }, 0);
        WorkerMapping {
            worker,
            base: base.cast::<u8>(),
            size,
            page_size: page_size(),
        }
    }

    /// Attaches at `rank` and faults every page in, for a test that only needs
    /// the chunks of the namespace to be readable.
    fn faulted(service: &Service, namespace: &str, rank: usize) -> WorkerMapping {
        let mapping = WorkerMapping::attach(service, namespace, rank);
        mapping.touch(0, mapping.size);
        mapping
    }

    /// Faults the bytes in as a worker's DMA does, one page at a time, writing
    /// back the byte each page already holds.
    fn touch(&self, offset: usize, len: usize) {
        for position in (offset..offset + len).step_by(self.page_size) {
            // SAFETY: the page lies inside this mapping.
            unsafe {
                let page = self.base.add(position);
                std::ptr::write_volatile(page, std::ptr::read_volatile(page));
            }
        }
    }
}

impl Drop for WorkerMapping {
    fn drop(&mut self) {
        // SAFETY: base and size are the mapping this struct created.
        assert_eq!(unsafe { libc::munmap(self.base.cast(), self.size) }, 0);
    }
}

/// Size of the sealed mapping behind `fd`.
fn file_size(fd: RawFd) -> usize {
    // SAFETY: status is a live buffer of the size fstat writes.
    let mut status: libc::stat = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::fstat(fd, &mut status) }, 0, "fstat failed");
    status.st_size as usize
}

fn info(probe: &mut Client) -> Value {
    probe.call(json!({"op": "info"}))
}

/// Retries `attempt` until it holds, which is how a test observes the release a
/// disconnect or a grace period triggers asynchronously.
fn wait_until(what: &str, mut attempt: impl FnMut() -> bool) {
    let deadline = Instant::now() + TIMEOUT;
    while !attempt() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(10));
    }
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

    let rejection = worker.call(json!({"op": "store", "keys": ["aabb"], "request": "r"}));
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
fn lookup_many_reports_one_status_per_key_and_leases_the_hits() {
    let service = Service::start();
    let mut client = service.connect();
    let opened = client.call(json!({
        "op": "open",
        "version": 1,
        "namespace": "engine-batch",
        "fingerprint": "a".repeat(64),
        "worker_bytes": 64,
        "world_size": 2,
        "num_chunks": 4,
    }));
    assert_eq!(opened["ok"], true);

    // `aa` is published, `bb` is still reserved, and `dd` was never stored.
    assert_eq!(
        client.call(json!({"op": "store", "keys": ["aa", "bb", "cc"], "request": "a"}))["ok"],
        true
    );
    assert_eq!(
        client.call(json!({"op": "commit", "keys": ["aa"], "request": "a", "success": true}))["ok"],
        true
    );
    let _mapping = WorkerMapping::faulted(&service, "engine-batch", 0);

    let reply = client.call(json!({
        "op": "lookup_many",
        "keys": ["aa", "bb", "dd", "cc"],
        "request": "b",
    }));
    assert_eq!(reply["ok"], true);
    assert_eq!(
        reply["statuses"],
        json!(["hit", "pending", "miss", "pending"])
    );

    // The batch took the lease a per-key hit takes, for the request that asked.
    let loaded = client.call(json!({"op": "load", "keys": ["aa"], "request": "b"}));
    assert_eq!(loaded["ok"], true);
    assert_eq!(loaded["slots"].as_array().unwrap().len(), 1);
    let foreign = client.call(json!({"op": "load", "keys": ["aa"], "request": "c"}));
    assert_eq!(foreign["ok"], false);
    assert!(foreign["error"].as_str().unwrap().contains("ready leased entry"));

    // One key per counter, not one per request.
    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    assert_eq!(stats["hits"], 1);
    assert_eq!(stats["pending_hits"], 2);
    assert_eq!(stats["misses"], 1);
    assert_eq!(stats["leased"], 1);
}

#[test]
fn until_miss_scan_ends_at_the_miss_and_leases_nothing_past_it() {
    let service = Service::start();
    let mut client = service.connect();
    let opened = client.call(json!({
        "op": "open",
        "version": 1,
        "namespace": "engine-prefix",
        "fingerprint": "a".repeat(64),
        "worker_bytes": 64,
        "world_size": 2,
        "num_chunks": 4,
    }));
    assert_eq!(opened["ok"], true);

    let stored = client.call(json!({
        "op": "store",
        "keys": ["aa", "bb", "cc"],
        "request": "a",
    }));
    assert_eq!(stored["ok"], true);
    let committed = client.call(json!({
        "op": "commit",
        "keys": ["aa", "bb", "cc"],
        "request": "a",
        "success": true,
    }));
    assert_eq!(committed["ok"], true);
    let _mapping = WorkerMapping::faulted(&service, "engine-prefix", 0);

    // `dd` is missing, so the scan stops there and `cc` is never looked up.
    let reply = client.call(json!({
        "op": "lookup_many",
        "keys": ["aa", "dd", "cc"],
        "request": "b",
        "until_miss": true,
    }));
    assert_eq!(reply["ok"], true);
    assert_eq!(reply["statuses"], json!(["hit", "miss"]));

    // `cc` is published but was not leased, so the scan cannot load it.
    let past_miss = client.call(json!({"op": "load", "keys": ["cc"], "request": "b"}));
    assert_eq!(past_miss["ok"], false);
    assert!(past_miss["error"].as_str().unwrap().contains("ready leased entry"));
    let hit = client.call(json!({"op": "load", "keys": ["aa"], "request": "b"}));
    assert_eq!(hit["ok"], true);

    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    assert_eq!(stats["hits"], 1);
    assert_eq!(stats["misses"], 1);
    assert_eq!(stats["leased"], 1);
}

#[test]
fn prefetch_asks_for_a_whole_row_before_the_load_that_reads_it() {
    let service = Service::start();
    let mut scheduler = service.connect();
    let opened = open_with(&mut scheduler, "engine-prefetch", &"a".repeat(64), 4);
    assert_eq!(opened["ok"], true);
    let row_bytes = opened["row_bytes"].as_u64().expect("geometry reports row_bytes") as usize;

    // `aa` is committed but no worker has written its row yet, so it is not
    // readable, while `bb` is still only reserved.
    let stored = scheduler.call(json!({"op": "store", "keys": ["aa", "bb"], "request": "a"}));
    assert_eq!(stored["ok"], true);
    let slot = stored["slots"][0].as_u64().expect("store reports the slot") as usize;
    assert_eq!(
        scheduler.call(json!({"op": "commit", "keys": ["aa"], "request": "a", "success": true}))["ok"],
        true
    );
    assert_eq!(
        scheduler.call(json!({"op": "lookup", "key": "aa", "request": "a"}))["status"],
        "pending"
    );

    let mapping = WorkerMapping::attach(&service, "engine-prefetch", 0);
    let prefetched = scheduler.call(json!({"op": "prefetch", "keys": ["aa"]}));
    assert_eq!(prefetched["ok"], true);
    assert_eq!(prefetched["pages"], json!(row_bytes / mapping.page_size));

    // The reply does not wait for the readahead, so let it finish and then take
    // the fault a worker's DMA would have taken before the chunk was read.
    thread::sleep(Duration::from_millis(50));
    mapping.touch(slot * row_bytes, row_bytes);
    assert_eq!(
        scheduler.call(json!({"op": "lookup", "key": "aa", "request": "a"}))["status"],
        "hit"
    );

    // A key the cache does not hold and one it has only reserved are both
    // refused, and a rejection leaves the session that sent it alive.
    for (key, error) in [
        ("cc", "prefetch requires a stored chunk"),
        ("bb", "prefetch requires a committed chunk"),
    ] {
        let rejected = scheduler.call(json!({"op": "prefetch", "keys": [key]}));
        assert_eq!(rejected["ok"], false);
        assert_eq!(rejected["error"], error);
    }
    assert_eq!(scheduler.call(json!({"op": "stats"}))["ok"], true);
}

#[test]
fn a_pending_lookup_many_starts_the_readahead_of_its_committed_rows() {
    let service = Service::start();
    let mut client = service.connect();
    let opened = open_with(&mut client, "engine-readahead", &"a".repeat(64), 4);
    assert_eq!(opened["ok"], true);

    // Committed but never written by a worker, so both rows are still remote and
    // both lookups report them as not readable yet.
    assert_eq!(
        client.call(json!({"op": "store", "keys": ["aa", "bb"], "request": "a"}))["ok"],
        true
    );
    assert_eq!(
        client.call(json!({
            "op": "commit",
            "keys": ["aa", "bb"],
            "request": "a",
            "success": true,
        }))["ok"],
        true
    );
    let reply = client.call(json!({
        "op": "lookup_many",
        "keys": ["aa", "bb"],
        "request": "b",
    }));
    assert_eq!(reply["statuses"], json!(["pending", "pending"]));
    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    assert_eq!(stats["prefetch_calls"], 1);
    assert_eq!(stats["prefetch_chunks"], 2);

    // The rows are inside their cooldown, so the second scan starts nothing and
    // counts the two rows it dropped.
    let reply = client.call(json!({
        "op": "lookup_many",
        "keys": ["aa", "bb"],
        "request": "b",
    }));
    assert_eq!(reply["statuses"], json!(["pending", "pending"]));
    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    assert_eq!(stats["prefetch_calls"], 1);
    assert_eq!(stats["prefetch_skipped"], 2);
}

#[test]
fn a_zero_prefetch_cooldown_reads_a_pending_row_ahead_again() {
    let service = Service::start_with(&["--prefetch-cooldown-ms", "0"]);
    let mut client = service.connect();
    let opened = open(&mut client, "engine-readahead-again");
    assert_eq!(opened["ok"], true);
    assert_eq!(
        client.call(json!({"op": "store", "keys": ["aa"], "request": "a"}))["ok"],
        true
    );
    assert_eq!(
        client.call(json!({"op": "commit", "keys": ["aa"], "request": "a", "success": true}))["ok"],
        true
    );

    for calls in 1..=2 {
        let reply = client.call(json!({"op": "lookup", "key": "aa", "request": "b"}));
        assert_eq!(reply["status"], "pending");
        assert_eq!(
            client.call(json!({"op": "stats"}))["stats"]["prefetch_calls"],
            calls
        );
    }
}

#[test]
fn prefetch_max_chunks_leaves_the_rest_of_a_batch_to_a_later_lookup() {
    let service = Service::start_with(&["--prefetch-max-chunks", "1"]);
    let mut client = service.connect();
    let opened = open_with(&mut client, "engine-readahead-cap", &"a".repeat(64), 4);
    assert_eq!(opened["ok"], true);
    assert_eq!(
        client.call(json!({"op": "store", "keys": ["aa", "bb"], "request": "a"}))["ok"],
        true
    );
    assert_eq!(
        client.call(json!({
            "op": "commit",
            "keys": ["aa", "bb"],
            "request": "a",
            "success": true,
        }))["ok"],
        true
    );

    let reply = client.call(json!({
        "op": "lookup_many",
        "keys": ["aa", "bb"],
        "request": "b",
    }));
    assert_eq!(reply["statuses"], json!(["pending", "pending"]));
    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    assert_eq!(stats["prefetch_calls"], 1);
    assert_eq!(stats["prefetch_chunks"], 1);
    assert_eq!(stats["prefetch_skipped"], 1);
}

#[test]
fn a_leased_chunk_is_pinned_until_its_request_releases_it() {
    // The namespace reserves two rows, which this budget covers, so a `hit` that
    // pinned nothing here was refused by the kernel rather than by the budget.
    let budget = (64 * 1024).to_string();
    let service = Service::start_with(&["--pin-lease-bytes", budget.as_str()]);
    let mut client = service.connect();
    let opened = open(&mut client, "engine-pin");
    assert_eq!(opened["ok"], true);
    let row_bytes = opened["row_bytes"].as_u64().expect("geometry reports row_bytes");

    assert_eq!(
        client.call(json!({"op": "store", "keys": ["aa"], "request": "a"}))["ok"],
        true
    );
    assert_eq!(
        client.call(json!({"op": "commit", "keys": ["aa"], "request": "a", "success": true}))["ok"],
        true
    );
    // A hit means readable now, so a worker writes the row before the lookup.
    let _mapping = WorkerMapping::faulted(&service, "engine-pin", 0);
    assert_eq!(
        client.call(json!({"op": "lookup", "key": "aa", "request": "a"}))["status"],
        "hit"
    );

    // A host that cannot `mlock` (no `CAP_IPC_LOCK`, or a `RLIMIT_MEMLOCK` below
    // one row) counts the refusal instead of offering the guarantee.
    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    let refused = stats["pin_failures"].as_u64().expect("stats reports pin_failures");
    if refused == 0 {
        assert_eq!(stats["pinned_slots"], 1);
        assert!(stats["pinned_bytes"].as_u64().unwrap() >= row_bytes);
    }

    let released = client.call(json!({"op": "release_request", "request": "a"}));
    assert_eq!(released["released"], 1);
    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    if refused == 0 {
        assert_eq!(stats["pinned_slots"], 0);
        assert_eq!(stats["pinned_bytes"], 0);
    }
}

#[test]
fn store_from_one_request_cannot_be_published_by_another() {
    let service = Service::start();
    let mut client = service.connect();
    assert_eq!(open(&mut client, "engine-owner")["ok"], true);

    let stored = client.call(json!({"op": "store", "keys": ["aa"], "request": "a"}));
    assert_eq!(stored["ok"], true);
    assert_eq!(stored["keys"], json!(["aa"]));

    let rejected = client.call(json!({
        "op": "commit",
        "keys": ["aa"],
        "request": "b",
        "success": true,
    }));
    assert_eq!(rejected["ok"], false);
    assert!(rejected["error"].as_str().unwrap().contains("does not own the reservation"));
    // The rejected commit left the reservation pending, not published.
    let pending = client.call(json!({"op": "lookup", "key": "aa", "request": "a"}));
    assert_eq!(pending["status"], "pending");

    let committed = client.call(json!({
        "op": "commit",
        "keys": ["aa"],
        "request": "a",
        "success": true,
    }));
    assert_eq!(committed["ok"], true);
    let _mapping = WorkerMapping::faulted(&service, "engine-owner", 0);
    let hit = client.call(json!({"op": "lookup", "key": "aa", "request": "a"}));
    assert_eq!(hit["status"], "hit");

    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    assert_eq!(stats["commits"], 1);
    assert_eq!(stats["failed_commits"], 1);
    assert_eq!(stats["pending"], 0);
}

#[test]
fn forced_reset_discards_leases_and_pending_reservations() {
    let service = Service::start();
    let mut client = service.connect();
    assert_eq!(open(&mut client, "engine-reset")["ok"], true);

    let both = json!(["aa", "bb"]);
    assert_eq!(
        client.call(json!({"op": "store", "keys": both, "request": "a"}))["ok"],
        true
    );
    assert_eq!(
        client.call(json!({"op": "commit", "keys": ["aa"], "request": "a", "success": true}))["ok"],
        true
    );
    let _mapping = WorkerMapping::faulted(&service, "engine-reset", 0);
    let hit = client.call(json!({"op": "lookup", "key": "aa", "request": "a"}));
    assert_eq!(hit["status"], "hit");

    // The lookup lease and the `bb` reservation keep the ordinary reset out.
    let refused = client.call(json!({"op": "reset"}));
    assert_eq!(refused["ok"], false);
    assert!(refused["error"].as_str().unwrap().contains("cannot reset"));

    let forced = client.call(json!({"op": "reset", "force": true}));
    assert_eq!(forced["ok"], true);
    let stats = client.call(json!({"op": "stats"}))["stats"].clone();
    assert_eq!(stats["entries"], 0);
    assert_eq!(stats["pending"], 0);
    assert_eq!(stats["leased"], 0);
    assert_eq!(stats["free_chunks"], 2);
    let missed = client.call(json!({"op": "lookup", "key": "aa", "request": "a"}));
    assert_eq!(missed["status"], "miss");
}

#[test]
fn expired_reservations_are_reclaimed_instead_of_staying_pending() {
    let service = Service::start_with(&["--pending-timeout-secs", "1"]);
    let mut client = service.connect();
    assert_eq!(open(&mut client, "engine-reclaim")["ok"], true);
    assert_eq!(
        client.call(json!({"op": "store", "keys": ["aa"], "request": "a"}))["ok"],
        true
    );

    // Every `stats` call also reclaims, so polling it is the observable event.
    let deadline = Instant::now() + TIMEOUT;
    loop {
        let stats = client.call(json!({"op": "stats"}))["stats"].clone();
        if stats["entries"] == 0 {
            assert_eq!(stats["reclaimed"], 1);
            break;
        }
        assert!(Instant::now() < deadline, "the reservation never expired");
        thread::sleep(Duration::from_millis(50));
    }
    let missed = client.call(json!({"op": "lookup", "key": "aa", "request": "a"}));
    assert_eq!(missed["status"], "miss");
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
    let rejection = client.call(json!({"op": "store", "keys": keys, "request": "r"}));
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
fn info_reports_service_capacity_without_a_handshake() {
    let service = Service::start();
    let mut client = service.connect();

    let info = client.call(json!({"op": "info"}));
    assert_eq!(info["ok"], true);
    assert_eq!(info["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(info["capacity_bytes"].as_u64(), Some(CAPACITY as u64));
    assert_eq!(info["used_bytes"].as_u64(), Some(0));
    assert_eq!(info["retired_bytes"].as_u64(), Some(0));
    assert_eq!(info["namespaces"].as_u64(), Some(0));
    assert_eq!(info["detached"].as_u64(), Some(0));
    assert_eq!(info["sessions"].as_u64(), Some(0));
    assert_eq!(info["max_connections"].as_u64(), Some(128));
    assert!(info["pid"].as_u64().is_some_and(|pid| pid > 0));

    // An open namespace is charged to the service, and its session is counted.
    let opened = open(&mut client, "engine-info");
    assert_eq!(opened["ok"], true);
    let info = client.call(json!({"op": "info"}));
    assert_eq!(info["used_bytes"], opened["total_bytes"]);
    assert_eq!(info["namespaces"].as_u64(), Some(1));
    assert_eq!(info["sessions"].as_u64(), Some(1));
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

#[test]
fn owner_reconnect_adopts_the_namespace_and_keeps_entries() {
    let service = Service::start();
    let mut owner = service.connect();
    let opened = open(&mut owner, "engine-adopt");
    assert_eq!(opened["ok"], true);
    assert_eq!(
        owner.call(json!({"op": "store", "keys": ["aa"], "request": "a"}))["ok"],
        true
    );
    assert_eq!(
        owner.call(json!({"op": "commit", "keys": ["aa"], "request": "a", "success": true}))["ok"],
        true
    );

    // A transport hiccup on the scheduler's side leaves the mapping behind.
    drop(owner);
    let mut probe = service.connect();
    wait_until("the namespace to detach", || {
        info(&mut probe)["detached"].as_u64() == Some(1)
    });

    let mut adopted = service.connect();
    let reopened = open(&mut adopted, "engine-adopt");
    assert_eq!(reopened["ok"], true);
    assert_eq!(reopened["total_bytes"], opened["total_bytes"]);
    let stats = adopted.call(json!({"op": "stats"}));
    assert_eq!(stats["stats"]["entries"], 1);
    assert_eq!(stats["sessions"].as_u64(), Some(1));
    assert_eq!(stats["detached"], false);
    // Adopted, not recreated: the published entry survives the handshake and is
    // readable once a worker writes its row.
    let _mapping = WorkerMapping::faulted(&service, "engine-adopt", 0);
    assert_eq!(
        adopted.call(json!({"op": "lookup", "key": "aa", "request": "b"}))["status"],
        "hit"
    );
    // The adopting session owns the mapping, so it can still mutate it.
    assert_eq!(
        adopted.call(json!({"op": "store", "keys": ["bb"], "request": "c"}))["ok"],
        true
    );
    assert_eq!(
        adopted.call(json!({"op": "commit", "keys": ["bb"], "request": "c", "success": true}))["ok"],
        true
    );
    let reported = info(&mut probe);
    assert_eq!(reported["namespaces"].as_u64(), Some(1));
    assert_eq!(reported["detached"].as_u64(), Some(0));
    assert_eq!(reported["used_bytes"], opened["total_bytes"]);
}

#[test]
fn reopen_with_a_different_fingerprint_is_rejected() {
    let service = Service::start();
    let mut owner = service.connect();
    assert_eq!(open(&mut owner, "engine-mismatch")["ok"], true);
    drop(owner);
    let mut probe = service.connect();
    wait_until("the namespace to detach", || {
        info(&mut probe)["detached"].as_u64() == Some(1)
    });

    // Another layout, and another geometry under the same fingerprint, are both
    // refused instead of reusing a mapping they do not describe.
    let mut client = service.connect();
    for rejected in [
        open_with(&mut client, "engine-mismatch", &"b".repeat(64), 2),
        open_with(&mut client, "engine-mismatch", &"a".repeat(64), 1),
    ] {
        assert_eq!(rejected["ok"], false);
        assert_eq!(rejected["error"], "namespace geometry mismatch");
    }

    // The mapping is still there for the layout that created it.
    let adopted = open(&mut client, "engine-mismatch");
    assert_eq!(adopted["ok"], true);
    assert_eq!(
        client.call(json!({"op": "stats"}))["stats"]["num_chunks"],
        2
    );
}

#[test]
fn adoption_is_refused_while_the_previous_owner_is_connected() {
    let service = Service::start();
    let mut owner = service.connect();
    assert_eq!(open(&mut owner, "engine-live-owner")["ok"], true);

    let mut second = service.connect();
    let refused = open(&mut second, "engine-live-owner");
    assert_eq!(refused["ok"], false);
    assert_eq!(
        refused["error"],
        "namespace already owned by another scheduler"
    );

    // The owner keeps its namespace, and a worker attach still finds it.
    assert_eq!(owner.call(json!({"op": "stats"}))["ok"], true);
    let mut worker = attach_worker(&service, "engine-live-owner", 0);
    assert_eq!(worker.call(json!({"op": "stats"}))["ok"], true);
}

#[test]
fn worker_disconnect_still_invalidates_the_namespace() {
    let service = Service::start();
    let mut owner = service.connect();
    assert_eq!(open(&mut owner, "engine-worker-loss")["ok"], true);
    let worker = attach_worker(&service, "engine-worker-loss", 0);
    drop(worker);

    // That rank may still have DMA in flight, so the owner's calls stop working.
    wait_until("the namespace to be invalidated", || {
        owner.call(json!({"op": "stats"}))["error"] == "cache namespace invalidated"
    });
    let mut probe = service.connect();
    let reported = info(&mut probe);
    assert_eq!(reported["namespaces"].as_u64(), Some(0));
    assert_eq!(reported["detached"].as_u64(), Some(0));

    // The mapping stays charged while the owner's session keeps it, and a new
    // namespace of the same name is a fresh mapping rather than the old one.
    assert_eq!(reported["sessions"].as_u64(), Some(1));
    let mut replacement = service.connect();
    assert_eq!(open(&mut replacement, "engine-worker-loss")["ok"], true);
    assert_eq!(
        replacement.call(json!({"op": "lookup", "key": "aa", "request": "r"}))["status"],
        "miss"
    );
}

#[test]
fn close_retires_the_namespace_immediately() {
    let service = Service::start();
    let mut owner = service.connect();
    let opened = open(&mut owner, "engine-close");
    assert_eq!(opened["ok"], true);
    let mut worker = attach_worker(&service, "engine-close", 0);

    assert_eq!(owner.call(json!({"op": "close"}))["ok"], true);
    // Closed, not detached: neither session has a namespace any more.
    assert_eq!(
        worker.call(json!({"op": "stats"}))["error"],
        "cache namespace invalidated"
    );
    assert_eq!(
        owner.call(json!({"op": "stats"}))["error"],
        "handshake required"
    );

    let mut probe = service.connect();
    let retired = info(&mut probe);
    assert_eq!(retired["namespaces"].as_u64(), Some(0));
    assert_eq!(retired["detached"].as_u64(), Some(0));
    // The worker's session keeps the retired mapping charged, never adoptable.
    assert_eq!(retired["retired_bytes"], opened["total_bytes"]);
    let mut replaced = service.connect();
    assert_eq!(open(&mut replaced, "engine-close")["ok"], true);

    // The last session leaving frees the closed mapping.
    drop(worker);
    wait_until("the closed mapping to be freed", || {
        info(&mut probe)["retired_bytes"].as_u64() == Some(0)
    });
}

#[test]
fn namespace_without_sessions_is_reclaimed() {
    let service = Service::start();
    let mut owner = service.connect();
    let opened = open(&mut owner, "engine-orphan");
    assert_eq!(opened["ok"], true);
    let worker = attach_worker(&service, "engine-orphan", 0);

    // A detached namespace with a worker still keeps its mapping charged...
    drop(owner);
    let mut probe = service.connect();
    wait_until("the namespace to detach", || {
        info(&mut probe)["detached"].as_u64() == Some(1)
    });
    assert_eq!(info(&mut probe)["used_bytes"], opened["total_bytes"]);

    // ...and the last session leaving reclaims it at once, not after the grace
    // period the default configuration sets to 300 seconds.
    drop(worker);
    wait_until("the namespace to be reclaimed", || {
        info(&mut probe)["used_bytes"].as_u64() == Some(0)
    });
    let reported = info(&mut probe);
    assert_eq!(reported["namespaces"].as_u64(), Some(0));
    assert_eq!(reported["sessions"].as_u64(), Some(0));
}

#[test]
fn detached_namespace_expires_after_the_grace_period() {
    let mut service = Service::start_with(&["--namespace-grace-secs", "1"]);
    let mut owner = service.connect();
    let opened = open(&mut owner, "engine-grace");
    assert_eq!(opened["ok"], true);
    let mut worker = attach_worker(&service, "engine-grace", 0);
    drop(owner);

    // A restart inside the grace period still finds the mapping adoptable.
    let mut probe = service.connect();
    wait_until("the namespace to detach", || {
        info(&mut probe)["detached"].as_u64() == Some(1)
    });
    assert_eq!(info(&mut probe)["used_bytes"], opened["total_bytes"]);

    // Nobody adopted it, so the mapping a crashed scheduler left behind is
    // reclaimed even though its worker is still attached.
    wait_until("the grace period to reclaim the namespace", || {
        info(&mut probe)["used_bytes"].as_u64() == Some(0)
    });
    assert_eq!(info(&mut probe)["namespaces"].as_u64(), Some(0));
    assert_eq!(
        worker.call(json!({"op": "stats"}))["error"],
        "cache namespace invalidated"
    );

    let (status, stderr) = service.signal_and_wait(libc::SIGTERM);
    assert!(status.success(), "exit status {status}: {stderr}");
    assert!(
        stderr.contains("reason=grace_expired"),
        "missing reclaim event: {stderr}"
    );
}

#[test]
fn adopt_never_creates_a_namespace() {
    let service = Service::start();
    let mut client = service.connect();
    let refused = adopt(&mut client, "engine-absent");
    assert_eq!(refused["ok"], false);
    assert_eq!(refused["error"], "no detached namespace to adopt");

    // Nothing was allocated behind the client's back, and the session is usable
    // for the namespace it creates explicitly.
    let mut probe = service.connect();
    let reported = info(&mut probe);
    assert_eq!(reported["namespaces"].as_u64(), Some(0));
    assert_eq!(reported["used_bytes"].as_u64(), Some(0));
    assert_eq!(open(&mut client, "engine-absent")["ok"], true);

    // A detached namespace is what the replay is for.
    drop(client);
    wait_until("the namespace to detach", || {
        info(&mut probe)["detached"].as_u64() == Some(1)
    });
    let mut reconnected = service.connect();
    assert_eq!(adopt(&mut reconnected, "engine-absent")["ok"], true);
}

#[test]
fn adoption_keeps_the_mapping_an_attached_worker_holds() {
    let service = Service::start();
    let mut owner = service.connect();
    assert_eq!(open(&mut owner, "engine-adopt-worker")["ok"], true);
    let mut worker = WorkerMapping::faulted(&service, "engine-adopt-worker", 0);
    assert_eq!(
        owner.call(json!({"op": "store", "keys": ["aa"], "request": "a"}))["ok"],
        true
    );
    assert_eq!(
        owner.call(json!({"op": "commit", "keys": ["aa"], "request": "a", "success": true}))["ok"],
        true
    );
    drop(owner);

    let mut probe = service.connect();
    wait_until("the namespace to detach", || {
        info(&mut probe)["detached"].as_u64() == Some(1)
    });
    let mut adopted = service.connect();
    assert_eq!(adopt(&mut adopted, "engine-adopt-worker")["ok"], true);

    // The worker keeps the session and the mapping it already had, and the
    // adopted owner serves the entry the previous owner published.
    let stats = worker.worker.call(json!({"op": "stats"}));
    assert_eq!(stats["ok"], true);
    assert_eq!(stats["sessions"].as_u64(), Some(2));
    assert_eq!(stats["detached"], false);
    assert_eq!(
        adopted.call(json!({"op": "lookup", "key": "aa", "request": "b"}))["status"],
        "hit"
    );
}
