// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Control-plane measurement: the real service on a private socket, driven by
//! real client frames, reported as one line per case.
//!
//! The cases cover what the connector does per schedule step, so an accidental
//! O(n) regression in the cache or a dropped batch shows up as a per-op cost
//! that `--assert-max` can gate on. Every number is per operation, and with more
//! than one connection it is the aggregate over them. The step cases also report
//! the requests and the round trips of one step, so writing their deferrable
//! calls as one burst can be compared with writing each of them on its own.
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use memmap2::MmapMut;
use serde_json::{Value, json};
use vllm_kv_cache_service::cache::namespace_bytes;
use vllm_kv_cache_service::protocol::{self, MAX_KEYS};

use crate::{Limits, Registry, SHUTDOWN, accept_connections, invalid, lock_or_recover};

/// Per-chunk worker bytes of a realistic service geometry, so the rows a case
/// faults in are shaped like a deployed cache's.
const WORKER_BYTES: usize = 64 * 1024;
/// Chunks per namespace for every key of the population, so a case can populate
/// its prefix and still leave the store path room to evict.
const CHUNKS_PER_KEY: usize = 2;
/// Requests one pipelined connection writes before it reads a reply, sized to
/// stay well inside the socket buffer at any key width.
const PIPELINE_WINDOW: u64 = 256;
/// Keys one synthetic schedule step stores: the width of one 64-chunk store.
const STEP_STORE_KEYS: usize = 64;
const DEFAULT_KEYS: usize = MAX_KEYS;
const DEFAULT_REQUESTS: usize = 1;
const DEFAULT_ITERATIONS: u64 = 500;
const DEFAULT_CACHE_BYTES: usize = 4 << 30;
const RPC_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the bench waits for the service to free a finished case's
/// namespaces before it reuses the budget.
const RELEASE_TIMEOUT: Duration = Duration::from_secs(10);
const POPULATE_REQUEST: &str = "populate";
const LOOKUP_REQUEST: &str = "lookup";
const STEP_REQUEST: &str = "step";

#[derive(clap::Args)]
pub struct BenchArgs {
    /// Prefix keys per case: the batch width and the namespace population.
    #[arg(long, default_value_t = DEFAULT_KEYS)]
    keys: usize,
    /// Concurrent client connections, each with its own namespace.
    #[arg(long, default_value_t = 1)]
    conns: usize,
    /// Operations per case and connection.
    #[arg(long, default_value_t = DEFAULT_ITERATIONS)]
    iterations: u64,
    /// Concurrent requests per synthetic step: each scans the prefix, loads its
    /// hits, and stores one batch, then the step's commits and releases go out
    /// as one burst.
    #[arg(long, default_value_t = DEFAULT_REQUESTS)]
    requests: usize,
    /// Aggregate cache budget the bench namespaces must fit.
    #[arg(long, default_value_t = DEFAULT_CACHE_BYTES)]
    cache_bytes: usize,
    /// Print one JSON object per case instead of an aligned table.
    #[arg(long)]
    json: bool,
    /// Send each request's commit and release as its own round trip instead of
    /// one burst per step, which is the shape the connector had before the
    /// burst, and what the step cases are compared against.
    #[arg(long)]
    flush_immediately: bool,
    /// Fail the run when a case exceeds this many microseconds per operation,
    /// as `<case>=<us>`.
    #[arg(long = "assert-max", value_name = "CASE=US", value_parser = bound)]
    assert_max: Vec<Bound>,
}

/// An upper bound on one case's per-op cost.
#[derive(Clone)]
struct Bound {
    case: String,
    max_us: f64,
}

fn bound(value: &str) -> Result<Bound, String> {
    let (case, max_us) = value
        .split_once('=')
        .ok_or_else(|| format!("expected <case>=<us>, not {value}"))?;
    let max_us = max_us
        .parse::<f64>()
        .map_err(|error| format!("invalid bound for {case}: {error}"))?;
    if !max_us.is_finite() || max_us <= 0.0 {
        return Err(format!("the bound for {case} must be a positive number"));
    }
    Ok(Bound {
        case: case.to_owned(),
        max_us,
    })
}

/// One case's per-op cost and the shape it was measured at.
struct Measured {
    case: &'static str,
    keys: usize,
    conns: usize,
    ops: usize,
    p50_us: f64,
    p99_us: f64,
    us_per_op: f64,
    /// Set on the cases that model a whole schedule step.
    step: Option<StepShape>,
}

/// What one iteration of a step case models.
#[derive(Clone, Copy)]
struct StepShape {
    requests: usize,
    /// Transport round trips one step costs, one per burst it writes.
    round_trips: usize,
}

impl Measured {
    fn new(
        case: &'static str,
        keys: usize,
        conns: usize,
        mut samples: Vec<u64>,
    ) -> io::Result<Measured> {
        if samples.is_empty() {
            return Err(invalid("a bench case recorded no operation"));
        }
        samples.sort_unstable();
        let total: u64 = samples.iter().sum();
        Ok(Measured {
            case,
            keys,
            conns,
            ops: samples.len(),
            p50_us: micros(samples[rank(&samples, 50)] as f64),
            p99_us: micros(samples[rank(&samples, 99)] as f64),
            us_per_op: micros(total as f64 / samples.len() as f64),
            step: None,
        })
    }

    /// Reports the request count and the round trips one step of the case costs,
    /// so a burst's saving is visible without deriving it from the op count.
    fn step(mut self, requests: usize, round_trips: usize) -> Measured {
        self.step = Some(StepShape {
            requests,
            round_trips,
        });
        self
    }

    fn json(&self) -> Value {
        let mut object = json!({
            "case": self.case,
            "keys": self.keys,
            "conns": self.conns,
            "ops": self.ops,
            "p50_us": round(self.p50_us),
            "p99_us": round(self.p99_us),
            "us_per_op": round(self.us_per_op),
        });
        if let Some(step) = self.step {
            object["requests"] = json!(step.requests);
            object["round_trips"] = json!(step.round_trips);
        }
        object
    }

    fn line(&self) -> String {
        let round_trips = match self.step {
            Some(step) => step.round_trips.to_string(),
            None => "-".to_owned(),
        };
        format!(
            "{:<22}{:>6}{:>6}{:>10}{:>10.2}{:>10.2}{:>11.2}{:>13}",
            self.case,
            self.keys,
            self.conns,
            self.ops,
            self.p50_us,
            self.p99_us,
            self.us_per_op,
            round_trips
        )
    }
}

/// Nearest-rank percentile of sorted samples.
fn rank(samples: &[u64], percent: usize) -> usize {
    (percent * samples.len()).div_ceil(100).max(1) - 1
}

fn micros(nanos: f64) -> f64 {
    nanos / 1000.0
}

/// Rounds a reported microsecond value to what a tenth of a nanosecond holds.
fn round(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos() as u64
}

pub fn run(args: BenchArgs) -> io::Result<()> {
    if args.keys == 0 || args.conns == 0 || args.iterations == 0 || args.requests == 0 {
        return Err(invalid(
            "--keys, --conns, --iterations, and --requests must be positive",
        ));
    }
    let chunks = args
        .keys
        .checked_mul(CHUNKS_PER_KEY)
        .ok_or_else(|| invalid("--keys is too large"))?;
    let worker_bytes = worker_bytes(args.conns, chunks, args.cache_bytes)?;
    let mut bench = Bench {
        service: Service::start(args.cache_bytes)?,
        keys: (0..args.keys).map(|index| hex_key('a', index)).collect(),
        conns: args.conns,
        iterations: args.iterations,
        requests: args.requests,
        immediate: args.flush_immediately,
        worker_bytes,
        chunks,
        opened: 0,
    };
    let cases = vec![
        bench.serial_lookup()?,
        bench.pipelined_lookup()?,
        bench.lookup_many()?,
        bench.mixed_schedule_step()?,
        bench.deferred_flush()?,
        bench.stats()?,
    ];
    if args.json {
        for case in &cases {
            println!("{}", serde_json::to_string(&case.json())?);
        }
    } else {
        println!(
            "{:<22}{:>6}{:>6}{:>10}{:>10}{:>10}{:>11}{:>13}",
            "case", "keys", "conns", "ops", "p50_us", "p99_us", "us_per_op", "round_trips"
        );
        for case in &cases {
            println!("{}", case.line());
        }
    }
    drop(bench);
    for bound in &args.assert_max {
        let case = cases
            .iter()
            .find(|case| case.case == bound.case)
            .ok_or_else(|| invalid(format!("no bench case named {}", bound.case)))?;
        if case.us_per_op > bound.max_us {
            return Err(invalid(format!(
                "{} costs {:.2} us/op, over --assert-max {}={}",
                case.case, case.us_per_op, bound.case, bound.max_us
            )));
        }
    }
    Ok(())
}

/// The largest per-chunk size that fits one namespace per connection in the
/// budget, preferring a realistic row.
fn worker_bytes(conns: usize, chunks: usize, budget: usize) -> io::Result<usize> {
    const GRANULE: usize = 4096;
    let namespace_budget = budget / conns;
    let mut bytes = (namespace_budget / chunks).min(WORKER_BYTES) / GRANULE * GRANULE;
    while bytes >= GRANULE && namespace_bytes(bytes, 1, chunks)? > namespace_budget {
        bytes /= 2;
    }
    if bytes < GRANULE {
        return Err(invalid(format!(
            "--cache-bytes {budget} cannot hold {conns} namespaces of {chunks} chunks"
        )));
    }
    Ok(bytes)
}

fn hex_key(prefix: char, index: usize) -> String {
    format!("{prefix}{index:063x}")
}

/// The service under test: the real accept loop and request dispatch, on a
/// private socket in a directory only this process may use.
struct Service {
    /// Removed after the listener is gone, in that order.
    _dir: PrivateDir,
    socket: PathBuf,
    registry: Arc<Mutex<Registry>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Service {
    fn start(budget: usize) -> io::Result<Service> {
        let dir = PrivateDir::new()?;
        let socket = dir.path().join("bench.sock");
        let listener = UnixListener::bind(&socket)?;
        fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
        let registry = Arc::new(Mutex::new(Registry {
            namespaces: HashMap::new(),
            retired: Vec::new(),
            budget,
            rswap_attach: None,
            pending_timeout: None,
            started: Instant::now(),
        }));
        let limits = Limits {
            max_connections: 128,
            idle_timeout: None,
        };
        let accept = Arc::clone(&registry);
        let thread = thread::spawn(move || {
            // Only a listener failure ends the loop early, and the next client
            // call then reports it.
            let _ = accept_connections(&listener, &accept, &limits, None);
        });
        Ok(Service {
            _dir: dir,
            socket,
            registry,
            thread: Some(thread),
        })
    }

    /// The rows a case fills, taken from the namespace the service allocated. The
    /// bench is the process that serves it too, so it writes through the cache's
    /// own mapping instead of a worker's descriptor.
    fn rows(&self, namespace: &str) -> io::Result<Rows> {
        let registry = lock_or_recover(&self.registry, "registry");
        let entry = registry
            .namespaces
            .get(namespace)
            .ok_or_else(|| invalid(format!("the bench namespace {namespace} is not open")))?;
        let cache = lock_or_recover(&entry.cache, "cache");
        Ok(Rows {
            mapping: Arc::clone(&cache.mapping),
            row_bytes: cache.row_bytes,
            page_size: cache.page_size,
        })
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        // The accept loop polls the listener, so it stops within one poll.
        SHUTDOWN.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A private temporary directory, removed when the bench ends.
struct PrivateDir(PathBuf);

impl PrivateDir {
    fn new() -> io::Result<PrivateDir> {
        let path = std::env::temp_dir().join(format!("vllm-kv-bench-{}", std::process::id()));
        match fs::remove_dir_all(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        fs::create_dir(&path)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
        Ok(PrivateDir(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for PrivateDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A same-UID client that speaks the service protocol on one connection.
struct Client {
    stream: UnixStream,
}

impl Client {
    fn connect(socket: &Path) -> io::Result<Client> {
        let stream = UnixStream::connect(socket)?;
        stream.set_read_timeout(Some(RPC_TIMEOUT))?;
        stream.set_write_timeout(Some(RPC_TIMEOUT))?;
        Ok(Client { stream })
    }

    /// Writes one length-prefixed frame; requests and replies share the envelope.
    fn send(&mut self, request: &Value) -> io::Result<()> {
        self.stream.write_all(&protocol::encode_response(request)?)
    }

    fn receive(&mut self) -> io::Result<Value> {
        let mut header = [0; 4];
        self.stream.read_exact(&mut header)?;
        let length = u32::from_be_bytes(header) as usize;
        if length == 0 || length > protocol::MAX_FRAME {
            return Err(invalid("invalid frame length from the bench service"));
        }
        let mut payload = vec![0; length];
        self.stream.read_exact(&mut payload)?;
        serde_json::from_slice(&payload).map_err(io::Error::other)
    }

    fn call(&mut self, request: &Value) -> io::Result<Value> {
        self.send(request)?;
        self.receive()
    }

    /// Writes a whole burst in one write before reading any reply, which is what
    /// a pipelining client does.
    fn send_burst(&mut self, requests: &[Value]) -> io::Result<()> {
        let mut frames = Vec::new();
        for request in requests {
            frames.extend_from_slice(&protocol::encode_response(request)?);
        }
        self.stream.write_all(&frames)
    }
}

/// Reports the service's rejection instead of counting it as a measurement.
fn accepted(reply: &Value) -> io::Result<&Value> {
    match reply["ok"].as_bool() {
        Some(true) => Ok(reply),
        _ => Err(invalid(format!(
            "the bench service rejected a request: {reply}"
        ))),
    }
}

/// The rows a case stored, written through the cache's mapping as a worker's DMA
/// would: the service reports a chunk `pending` until its pages are in DRAM, so a
/// case that measures lookups has to fill what it populated.
struct Rows {
    mapping: Arc<MmapMut>,
    row_bytes: usize,
    page_size: usize,
}

impl Rows {
    /// Faults the rows of `slots` in, one page at a time, writing back the byte
    /// each page already holds.
    fn fill(&self, slots: &[usize]) {
        for slot in slots {
            let base = self.mapping.as_ptr() as usize + slot * self.row_bytes;
            for offset in (0..self.row_bytes).step_by(self.page_size) {
                // SAFETY: the row lies inside the sealed mapping.
                unsafe {
                    let page = std::ptr::with_exposed_provenance_mut::<u8>(base + offset);
                    std::ptr::write_volatile(page, std::ptr::read_volatile(page));
                }
            }
        }
    }
}

/// One bench connection and the namespace it opened.
struct Session {
    client: Client,
    rows: Rows,
}

impl Session {
    fn open(
        service: &Service,
        namespace: String,
        fingerprint: u64,
        worker_bytes: usize,
        chunks: usize,
    ) -> io::Result<Session> {
        let mut client = Client::connect(&service.socket)?;
        let reply = client.call(&json!({
            "op": "open",
            "version": 1,
            "namespace": namespace,
            "fingerprint": format!("{fingerprint:064x}"),
            "worker_bytes": worker_bytes,
            "world_size": 1,
            "num_chunks": chunks,
        }))?;
        accepted(&reply)?;
        let rows = service.rows(&namespace)?;
        Ok(Session { client, rows })
    }

    /// Publishes `keys`, so every case starts from a populated cache.
    fn populate(&mut self, keys: &[String]) -> io::Result<()> {
        for batch in keys.chunks(MAX_KEYS) {
            let stored = self.client.call(&json!({
                "op": "store",
                "keys": batch,
                "request": POPULATE_REQUEST,
            }))?;
            let reserved = accepted(&stored)?["slots"]
                .as_array()
                .ok_or_else(|| invalid("store reply without slots"))?;
            let slots: Vec<usize> = reserved
                .iter()
                .map(|slot| {
                    slot.as_u64()
                        .map(|slot| slot as usize)
                        .ok_or_else(|| invalid("store reply with a non-numeric slot"))
                })
                .collect::<io::Result<_>>()?;
            let committed = self.client.call(&json!({
                "op": "commit",
                "keys": batch,
                "request": POPULATE_REQUEST,
                "success": true,
            }))?;
            accepted(&committed)?;
            self.rows.fill(&slots);
        }
        Ok(())
    }
}

/// Runs `work` on every session in its own thread and returns the per-op latency
/// samples all of them recorded, in nanoseconds.
fn measure<F>(sessions: &mut [Session], work: F) -> io::Result<Vec<u64>>
where
    F: Fn(&mut Session, &mut Vec<u64>) -> io::Result<()> + Sync,
{
    let recorded = thread::scope(|scope| -> io::Result<Vec<Vec<u64>>> {
        let handles: Vec<_> = sessions
            .iter_mut()
            .map(|session| {
                let work = &work;
                scope.spawn(move || {
                    let mut samples = Vec::new();
                    work(session, &mut samples)?;
                    Ok(samples)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| invalid("a bench client panicked"))?)
            .collect()
    })?;
    Ok(recorded.into_iter().flatten().collect())
}

/// Sends one request and records its round trip; the reply must be accepted.
fn timed(client: &mut Client, samples: &mut Vec<u64>, request: &Value) -> io::Result<Value> {
    let start = Instant::now();
    let reply = client.call(request)?;
    samples.push(nanos(start));
    accepted(&reply)?;
    Ok(reply)
}

/// Round trips a step's deferrable calls cost: the one burst, or one per call
/// when they are written as the caller makes them.
fn flush_round_trips(requests: usize, immediate: bool) -> usize {
    if immediate { requests * 2 } else { 1 }
}

/// Writes a step's deferrable calls, as one burst or one round trip each, and
/// records one sample per reply either way.
fn deferred(
    client: &mut Client,
    samples: &mut Vec<u64>,
    requests: &[Value],
    immediate: bool,
) -> io::Result<()> {
    if immediate {
        for request in requests {
            timed(client, samples, request)?;
        }
        return Ok(());
    }
    burst(client, samples, requests)
}

/// Writes a whole burst, then charges each reply the time since the previous
/// one, so the burst's samples add up to its one round trip.
fn burst(client: &mut Client, samples: &mut Vec<u64>, requests: &[Value]) -> io::Result<()> {
    let mut start = Instant::now();
    client.send_burst(requests)?;
    for _ in requests {
        let reply = client.receive()?;
        samples.push(nanos(start));
        start = Instant::now();
        accepted(&reply)?;
    }
    Ok(())
}

/// The step's stored keys, one batch per request, taken from a pool of their own.
fn step_keys(base: usize, request: usize, store_keys: usize) -> Vec<String> {
    (0..store_keys)
        .map(|offset| hex_key('c', base + request * store_keys + offset))
        .collect()
}

/// The commit and the release of one request, in the order the manager enqueues
/// them: a commit is queued when the store completes, the release when the
/// request finishes or its last job does.
fn deferred_calls(request: String, fresh: &[String]) -> [Value; 2] {
    [
        json!({
            "op": "commit",
            "keys": fresh,
            "request": request,
            "success": true,
        }),
        json!({"op": "release_request", "request": request}),
    ]
}

struct Bench {
    service: Service,
    keys: Vec<String>,
    conns: usize,
    iterations: u64,
    requests: usize,
    immediate: bool,
    worker_bytes: usize,
    chunks: usize,
    opened: u64,
}

impl Bench {
    /// Opens one namespace per connection and populates it.
    fn sessions(&mut self, case: &str) -> io::Result<Vec<Session>> {
        let mut sessions = Vec::with_capacity(self.conns);
        for index in 0..self.conns {
            self.opened += 1;
            let namespace = format!("bench-{case}-{index}-{}", self.opened);
            let mut session = Session::open(
                &self.service,
                namespace,
                self.opened,
                self.worker_bytes,
                self.chunks,
            )?;
            session.populate(&self.keys)?;
            sessions.push(session);
        }
        Ok(sessions)
    }

    /// Drops a finished case's connections, closing their namespaces so the
    /// service frees them, so the next case opens against the whole budget.
    fn finish(&self, mut sessions: Vec<Session>) -> io::Result<()> {
        for session in &mut sessions {
            let closed = session.client.call(&json!({"op": "close"}))?;
            accepted(&closed)?;
        }
        drop(sessions);
        let mut probe = Client::connect(&self.service.socket)?;
        let deadline = Instant::now() + RELEASE_TIMEOUT;
        loop {
            let response = probe.call(&json!({"op": "info"}))?;
            if accepted(&response)?["used_bytes"].as_u64() == Some(0) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(invalid("the service kept a bench namespace allocated"));
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// One `lookup` round trip per operation, which is what the connector pays
    /// once per chunk unless it batches.
    fn serial_lookup(&mut self) -> io::Result<Measured> {
        let mut sessions = self.sessions("serial")?;
        let keys = &self.keys;
        let iterations = self.iterations;
        let samples = measure(&mut sessions, |session, samples| {
            for iteration in 0..iterations {
                let key = &keys[iteration as usize % keys.len()];
                let request = json!({"op": "lookup", "key": key, "request": LOOKUP_REQUEST});
                let start = Instant::now();
                let reply = session.client.call(&request)?;
                samples.push(nanos(start));
                if accepted(&reply)?["status"] != "hit" {
                    return Err(invalid(format!("serial lookup missed: {reply}")));
                }
            }
            Ok(())
        })?;
        self.finish(sessions)?;
        Measured::new("rpc_serial_lookup", self.keys.len(), self.conns, samples)
    }

    /// The same lookups written ahead in windows of `PIPELINE_WINDOW`, so the
    /// cost is the transport's marginal per-op cost rather than a round trip.
    fn pipelined_lookup(&mut self) -> io::Result<Measured> {
        let mut sessions = self.sessions("pipelined")?;
        let keys = &self.keys;
        let iterations = self.iterations;
        let samples = measure(&mut sessions, |session, samples| {
            let mut done = 0;
            while done < iterations {
                let window = (iterations - done).min(PIPELINE_WINDOW);
                for step in 0..window {
                    let key = &keys[(done + step) as usize % keys.len()];
                    let request = json!({"op": "lookup", "key": key, "request": LOOKUP_REQUEST});
                    session.client.send(&request)?;
                }
                // Each reply is charged the time since the previous one, so an op
                // costs the service time it adds to an already full pipeline; the
                // first one also pays for the window to drain.
                let mut start = Instant::now();
                for _ in 0..window {
                    accepted(&session.client.receive()?)?;
                    samples.push(nanos(start));
                    start = Instant::now();
                }
                done += window;
            }
            Ok(())
        })?;
        self.finish(sessions)?;
        Measured::new("rpc_pipelined_lookup", self.keys.len(), self.conns, samples)
    }

    /// The whole prefix in one frame per `MAX_KEYS` keys, as the connector asks
    /// for a full-attention prefix.
    fn lookup_many(&mut self) -> io::Result<Measured> {
        let mut sessions = self.sessions("many")?;
        let keys = &self.keys;
        let iterations = self.iterations;
        let samples = measure(&mut sessions, |session, samples| {
            for _ in 0..iterations {
                for batch in keys.chunks(MAX_KEYS) {
                    let request = json!({
                        "op": "lookup_many",
                        "keys": batch,
                        "request": LOOKUP_REQUEST,
                    });
                    let start = Instant::now();
                    let reply = session.client.call(&request)?;
                    samples.push(nanos(start));
                    accepted(&reply)?;
                }
            }
            Ok(())
        })?;
        self.finish(sessions)?;
        Measured::new("rpc_lookup_many", self.keys.len(), self.conns, samples)
    }

    /// A synthetic schedule step of `--requests` requests: each scans the
    /// prefix, loads its hits, and stores one 64-chunk batch, then the commits
    /// and releases of the whole step go out as one burst, as the manager writes
    /// them at the end of the step.
    fn mixed_schedule_step(&mut self) -> io::Result<Measured> {
        let mut sessions = self.sessions("step")?;
        let keys = &self.keys;
        let iterations = self.iterations;
        let requests = self.requests;
        let immediate = self.immediate;
        let store_keys = STEP_STORE_KEYS.min(keys.len()).max(1);
        let batches = keys.chunks(MAX_KEYS).count();
        let samples = measure(&mut sessions, |session, samples| {
            for iteration in 0..iterations {
                let base = iteration as usize * store_keys * requests;
                let mut queued = Vec::with_capacity(requests * 2);
                for index in 0..requests {
                    let request = format!("{STEP_REQUEST}-{index}");
                    let mut hits: Vec<&String> = Vec::new();
                    for batch in keys.chunks(MAX_KEYS) {
                        let reply = timed(
                            &mut session.client,
                            samples,
                            &json!({
                                "op": "lookup_many",
                                "keys": batch,
                                "request": request,
                            }),
                        )?;
                        let statuses = reply["statuses"]
                            .as_array()
                            .ok_or_else(|| invalid("lookup_many reply without statuses"))?;
                        for (key, status) in batch.iter().zip(statuses) {
                            if status == "hit" {
                                hits.push(key);
                            }
                        }
                    }
                    if hits.len() != keys.len() {
                        return Err(invalid("a step scan missed the prefix it populated"));
                    }
                    for batch in hits.chunks(MAX_KEYS) {
                        timed(
                            &mut session.client,
                            samples,
                            &json!({"op": "load", "keys": batch, "request": request}),
                        )?;
                    }
                    let fresh = step_keys(base, index, store_keys);
                    let reply = timed(
                        &mut session.client,
                        samples,
                        &json!({"op": "store", "keys": fresh, "request": request}),
                    )?;
                    if reply["full"] == true {
                        return Err(invalid("a step store found no evictable slot"));
                    }
                    queued.extend(deferred_calls(request, &fresh));
                }
                deferred(&mut session.client, samples, &queued, immediate)?;
            }
            Ok(())
        })?;
        self.finish(sessions)?;
        // One scan and one load batch per request, then its store, and every
        // request's commits and releases in one burst instead of two each.
        let round_trips = requests * (2 * batches + 1) + flush_round_trips(requests, immediate);
        let measured = Measured::new("mixed_schedule_step", self.keys.len(), self.conns, samples)?;
        Ok(measured.step(requests, round_trips))
    }

    /// The deferrable tail of a step on its own: each request's store round trip,
    /// then one burst carrying every request's commit and release.
    fn deferred_flush(&mut self) -> io::Result<Measured> {
        let mut sessions = self.sessions("flush")?;
        let keys = &self.keys;
        let iterations = self.iterations;
        let requests = self.requests;
        let immediate = self.immediate;
        let store_keys = STEP_STORE_KEYS.min(keys.len()).max(1);
        let samples = measure(&mut sessions, |session, samples| {
            for iteration in 0..iterations {
                let base = iteration as usize * store_keys * requests;
                let mut queued = Vec::with_capacity(requests * 2);
                for index in 0..requests {
                    let request = format!("{STEP_REQUEST}-{index}");
                    let fresh = step_keys(base, index, store_keys);
                    let reply = timed(
                        &mut session.client,
                        samples,
                        &json!({"op": "store", "keys": fresh, "request": request}),
                    )?;
                    if reply["full"] == true {
                        return Err(invalid("a flush store found no evictable slot"));
                    }
                    queued.extend(deferred_calls(request, &fresh));
                }
                deferred(&mut session.client, samples, &queued, immediate)?;
            }
            Ok(())
        })?;
        self.finish(sessions)?;
        let round_trips = requests + flush_round_trips(requests, immediate);
        let measured = Measured::new("deferred_flush", self.keys.len(), self.conns, samples)?;
        Ok(measured.step(requests, round_trips))
    }

    /// A namespace poll, which walks the metadata of a populated cache.
    fn stats(&mut self) -> io::Result<Measured> {
        let mut sessions = self.sessions("stats")?;
        let iterations = self.iterations;
        let samples = measure(&mut sessions, |session, samples| {
            for _ in 0..iterations {
                let start = Instant::now();
                let reply = session.client.call(&json!({"op": "stats"}))?;
                samples.push(nanos(start));
                accepted(&reply)?;
            }
            Ok(())
        })?;
        self.finish(sessions)?;
        Measured::new("stats", self.keys.len(), self.conns, samples)
    }
}
