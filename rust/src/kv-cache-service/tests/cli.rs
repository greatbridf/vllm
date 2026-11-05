// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::process::Command;

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
}
