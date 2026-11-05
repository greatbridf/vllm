// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Control-plane measurement: the real service on a private socket, driven by
//! real client frames, reported as one line per case.
//!
//! The cases cover what the connector does per schedule step, so an accidental
//! O(n) regression in the cache or a dropped batch shows up as a per-op cost
//! that `--assert-max` can gate on. Every number is per operation, and with more
//! than one connection it is the aggregate over them.
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

use serde_json::{Value, json};
use vllm_kv_cache_service::cache::namespace_bytes;
use vllm_kv_cache_service::protocol::{self, MAX_KEYS};

use crate::{Limits, Registry, SHUTDOWN, accept_connections, invalid};

/// Per-chunk worker bytes of a realistic service geometry; the benches never
/// touch the mapping, so this only has to be shaped like a deployed one.
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
    /// Aggregate cache budget the bench namespaces must fit.
    #[arg(long, default_value_t = DEFAULT_CACHE_BYTES)]
    cache_bytes: usize,
    /// Print one JSON object per case instead of an aligned table.
    #[arg(long)]
    json: bool,
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
        })
    }

    fn json(&self) -> Value {
        json!({
            "case": self.case,
            "keys": self.keys,
            "conns": self.conns,
            "ops": self.ops,
            "p50_us": round(self.p50_us),
            "p99_us": round(self.p99_us),
            "us_per_op": round(self.us_per_op),
        })
    }

    fn line(&self) -> String {
        format!(
            "{:<22}{:>6}{:>6}{:>10}{:>10.2}{:>10.2}{:>11.2}",
            self.case, self.keys, self.conns, self.ops, self.p50_us, self.p99_us, self.us_per_op
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
    if args.keys == 0 || args.conns == 0 || args.iterations == 0 {
        return Err(invalid(
            "--keys, --conns, and --iterations must be positive",
        ));
    }
    let chunks = args
        .keys
        .checked_mul(CHUNKS_PER_KEY)
        .ok_or_else(|| invalid("--keys is too large"))?;
    let worker_bytes = worker_bytes(args.conns, chunks, args.cache_bytes)?;
    let service = Service::start(args.cache_bytes)?;
    let mut bench = Bench {
        socket: service.socket.clone(),
        keys: (0..args.keys).map(|index| hex_key('a', index)).collect(),
        conns: args.conns,
        iterations: args.iterations,
        worker_bytes,
        chunks,
        opened: 0,
    };
    let cases = vec![
        bench.serial_lookup()?,
        bench.pipelined_lookup()?,
        bench.lookup_many()?,
        bench.mixed_schedule_step()?,
        bench.stats()?,
    ];
    if args.json {
        for case in &cases {
            println!("{}", serde_json::to_string(&case.json())?);
        }
    } else {
        println!(
            "{:<22}{:>6}{:>6}{:>10}{:>10}{:>10}{:>11}",
            "case", "keys", "conns", "ops", "p50_us", "p99_us", "us_per_op"
        );
        for case in &cases {
            println!("{}", case.line());
        }
    }
    drop(bench);
    drop(service);
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
        let thread = thread::spawn(move || {
            // Only a listener failure ends the loop early, and the next client
            // call then reports it.
            let _ = accept_connections(&listener, &registry, &limits);
        });
        Ok(Service {
            _dir: dir,
            socket,
            thread: Some(thread),
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

/// One bench connection and the namespace it opened.
struct Session {
    client: Client,
}

impl Session {
    fn open(
        socket: &Path,
        namespace: String,
        fingerprint: u64,
        worker_bytes: usize,
        chunks: usize,
    ) -> io::Result<Session> {
        let mut client = Client::connect(socket)?;
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
        Ok(Session { client })
    }

    /// Publishes `keys`, so every case starts from a populated cache.
    fn populate(&mut self, keys: &[String]) -> io::Result<()> {
        for batch in keys.chunks(MAX_KEYS) {
            let stored = self.client.call(&json!({
                "op": "store",
                "keys": batch,
                "request": POPULATE_REQUEST,
            }))?;
            accepted(&stored)?;
            let committed = self.client.call(&json!({
                "op": "commit",
                "keys": batch,
                "request": POPULATE_REQUEST,
                "success": true,
            }))?;
            accepted(&committed)?;
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

struct Bench {
    socket: PathBuf,
    keys: Vec<String>,
    conns: usize,
    iterations: u64,
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
                &self.socket,
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

    /// Drops a finished case's connections and waits until the service released
    /// their namespaces, so the next case opens against the whole budget.
    fn finish(&self, sessions: Vec<Session>) -> io::Result<()> {
        drop(sessions);
        let mut probe = Client::connect(&self.socket)?;
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

    /// A synthetic schedule step: the prefix scan, a load of its hits, a store of
    /// one 64-chunk batch and its commit, then the release of the step's leases.
    fn mixed_schedule_step(&mut self) -> io::Result<Measured> {
        let mut sessions = self.sessions("step")?;
        let keys = &self.keys;
        let iterations = self.iterations;
        let step_keys = STEP_STORE_KEYS.min(keys.len()).max(1);
        let samples = measure(&mut sessions, |session, samples| {
            for iteration in 0..iterations {
                let mut hits: Vec<&String> = Vec::new();
                for batch in keys.chunks(MAX_KEYS) {
                    let request = json!({
                        "op": "lookup_many",
                        "keys": batch,
                        "request": STEP_REQUEST,
                    });
                    let start = Instant::now();
                    let reply = session.client.call(&request)?;
                    samples.push(nanos(start));
                    let statuses = accepted(&reply)?["statuses"]
                        .as_array()
                        .ok_or_else(|| invalid("lookup_many reply without statuses"))?;
                    for (key, status) in batch.iter().zip(statuses) {
                        if status == "hit" {
                            hits.push(key);
                        }
                    }
                }
                if !hits.is_empty() {
                    for batch in hits.chunks(MAX_KEYS) {
                        let request = json!({"op": "load", "keys": batch, "request": STEP_REQUEST});
                        let start = Instant::now();
                        let reply = session.client.call(&request)?;
                        samples.push(nanos(start));
                        accepted(&reply)?;
                    }
                }
                let base = iteration as usize * step_keys;
                let fresh: Vec<String> =
                    (0..step_keys).map(|index| hex_key('c', base + index)).collect();
                let start = Instant::now();
                let reply = session
                    .client
                    .call(&json!({"op": "store", "keys": fresh, "request": STEP_REQUEST}))?;
                samples.push(nanos(start));
                if accepted(&reply)?["full"] == true {
                    return Err(invalid("a step store found no evictable slot"));
                }
                let start = Instant::now();
                let reply = session.client.call(&json!({
                    "op": "commit",
                    "keys": fresh,
                    "request": STEP_REQUEST,
                    "success": true,
                }))?;
                samples.push(nanos(start));
                accepted(&reply)?;
                let start = Instant::now();
                let reply = session
                    .client
                    .call(&json!({"op": "release_request", "request": STEP_REQUEST}))?;
                samples.push(nanos(start));
                accepted(&reply)?;
            }
            Ok(())
        })?;
        self.finish(sessions)?;
        Measured::new("mixed_schedule_step", self.keys.len(), self.conns, samples)
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
