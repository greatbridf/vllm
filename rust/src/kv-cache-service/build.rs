// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(rswap_bpf_object)");
    println!("cargo:rerun-if-changed=bpf/rswap_hash.bpf.c");
    println!("cargo:rerun-if-changed=bpf/rswap_hash.h");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let object = out_dir.join("rswap_hash.bpf.o");
    let status = Command::new("clang")
        .args(["-target", "bpf", "-O2", "-g", "-c"])
        .arg("bpf/rswap_hash.bpf.c")
        .args(["-I", "bpf", "-o"])
        .arg(&object)
        .current_dir(&manifest)
        .status();

    match status {
        Ok(status) if status.success() => {
            println!("cargo:rustc-cfg=rswap_bpf_object");
        }
        Ok(status) => println!(
            "cargo:warning=clang failed to build bpf/rswap_hash.bpf.c ({status}); \
             --rswap-attach will be unavailable"
        ),
        Err(error) => println!(
            "cargo:warning=clang is unavailable ({error}); \
             --rswap-attach will be unavailable"
        ),
    }
}
