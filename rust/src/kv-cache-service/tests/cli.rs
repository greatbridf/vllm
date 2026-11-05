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
        "64",
        "--iterations",
        "5",
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
            "stats",
        ]
    );
    for case in &cases {
        assert_eq!(case["keys"], 64);
        assert_eq!(case["conns"], 1);
        assert!(case["ops"].as_u64().unwrap() > 0);
        assert!(case["p50_us"].as_f64().unwrap() > 0.0);
        assert!(case["p99_us"].as_f64().unwrap() >= case["p50_us"].as_f64().unwrap());
        assert!(case["us_per_op"].as_f64().unwrap() > 0.0);
    }
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
