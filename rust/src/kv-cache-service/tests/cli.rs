// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::process::{Command, Output};

#[test]
fn help_documents_rswap_attach_and_alias() {
    let output = Command::new(env!("CARGO_BIN_EXE_vllm-kv-cache-service"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("--rswap-attach"));
    assert!(stdout.contains("--hash-attach"));
    assert!(stdout.contains("/sys/kernel/mm/xswap/rswap_attach"));
    assert!(stdout.contains("bench"));
    assert!(stdout.contains("--eviction-policy"));
}

fn bench(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vllm-kv-cache-service"))
        .arg("bench")
        .arg("--json")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn bench_reports_one_json_object_per_case() {
    let output = bench(&[
        "--keys",
        "256",
        "--iterations",
        "5",
        "--requests",
        "2",
        "--assert-max",
        "rpc_serial_lookup=100000",
    ]);
    assert!(
        output.status.success(),
        "bench failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let cases: Vec<serde_json::Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("bench prints JSON lines"))
        .collect();
    let named: Vec<&str> = cases.iter().map(|case| case["case"].as_str().unwrap()).collect();
    assert_eq!(
        named,
        [
            "rpc_serial_lookup",
            "rpc_pipelined_lookup",
            "rpc_lookup_many",
            "mixed_schedule_step",
            "deferred_flush",
            "stats",
        ]
    );
    for case in &cases {
        assert_eq!(case["keys"], 256);
        assert_eq!(case["conns"], 1);
        assert!(case["ops"].as_u64().unwrap() > 0);
        assert!(case["p50_us"].as_f64().unwrap() > 0.0);
        assert!(case["p99_us"].as_f64().unwrap() >= case["p50_us"].as_f64().unwrap());
        assert!(case["us_per_op"].as_f64().unwrap() > 0.0);
        let is_step =
            ["mixed_schedule_step", "deferred_flush"].contains(&case["case"].as_str().unwrap());
        assert_eq!(case.get("requests").is_some(), is_step, "{case}");
        assert_eq!(case.get("round_trips").is_some(), is_step, "{case}");
    }
    let step = |name: &str| {
        cases
            .iter()
            .find(|case| case["case"] == name)
            .expect("the step cases are printed")
            .clone()
    };
    // Two requests of one scan and load batch each: their stores, then the one
    // burst that carries both commits and both releases.
    let mixed = step("mixed_schedule_step");
    assert_eq!(mixed["requests"], 2);
    assert_eq!(mixed["round_trips"], 7);
    let flush = step("deferred_flush");
    assert_eq!(flush["requests"], 2);
    assert_eq!(flush["round_trips"], 3);
}

#[test]
fn bench_reports_the_immediate_path_round_trips() {
    let output = bench(&[
        "--keys",
        "256",
        "--iterations",
        "2",
        "--requests",
        "2",
        "--flush-immediately",
    ]);
    assert!(
        output.status.success(),
        "bench failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let cases: Vec<serde_json::Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("bench prints JSON lines"))
        .collect();
    let round_trips = |name: &str| {
        cases
            .iter()
            .find(|case| case["case"] == name)
            .expect("the step cases are printed")["round_trips"]
            .as_u64()
            .unwrap()
    };
    // One round trip per commit and per release, so a burst that carries all of
    // them never appears in the count: two requests of a scan and a load batch
    // each, their stores, and 4 deferrable calls.
    assert_eq!(round_trips("mixed_schedule_step"), 10);
    assert_eq!(round_trips("deferred_flush"), 6);
}

#[test]
fn bench_assert_max_fails_the_run() {
    let output = bench(&["--iterations", "5", "--assert-max", "stats=0.000001"]);
    assert!(
        !output.status.success(),
        "a case over its --assert-max bound must fail the run"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("stats costs") && stderr.contains("over --assert-max stats=0.000001"),
        "unexpected failure: {stderr}"
    );
}
