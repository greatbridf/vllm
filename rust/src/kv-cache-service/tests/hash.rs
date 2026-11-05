// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#[cfg(rswap_bpf_object)]
use std::io;

#[cfg(rswap_bpf_object)]
use vllm_kv_cache_service::bpf;
#[cfg(rswap_bpf_object)]
use vllm_kv_cache_service::bpf::{Arena, Mapping, Program};
#[cfg(rswap_bpf_object)]
use vllm_kv_cache_service::hash::{ArenaMetadata, arena_pages};
use vllm_kv_cache_service::hash::{CANONICAL_KEY_LEN, HASH_LEN, LENGTH_LEN, LENGTH_OFF};
use vllm_kv_cache_service::hash::{OFFSET_LEN, OFFSET_OFF};
#[cfg(rswap_bpf_object)]
use vllm_kv_cache_service::hash::{PageHash, RswapConfig};

#[cfg(rswap_bpf_object)]
const OBJECT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rswap_hash.bpf.o"));

#[test]
fn public_page_hash_layout_is_stable() {
    let hash = PageHash::new("aabbcc", 2);
    let decoded = PageHash::from_bytes(&[0xaa, 0xbb, 0xcc], 2);
    assert_eq!(hash.hash(), decoded.hash());
    assert_eq!(hash.hash().len(), HASH_LEN);
    assert_eq!(CANONICAL_KEY_LEN, 36);
    assert_eq!(LENGTH_OFF, CANONICAL_KEY_LEN);
    assert_eq!(LENGTH_OFF + LENGTH_LEN, OFFSET_OFF);
    assert_eq!(OFFSET_OFF + OFFSET_LEN, HASH_LEN);
    assert_eq!(hash.canonical_len(), 3);
    assert_eq!(hash.key_len(), 3);
    assert_eq!(hash.page_offset(), 2);
    assert_eq!(hash.canonical_key()[..3], [0xaa, 0xbb, 0xcc]);
    assert_eq!(
        &hash.hash()[LENGTH_OFF..OFFSET_OFF],
        &3u32.to_le_bytes()[..]
    );
    assert_eq!(&hash.hash()[OFFSET_OFF..HASH_LEN], &2u64.to_le_bytes()[..]);
}

#[cfg(rswap_bpf_object)]
#[test]
fn program_patches_one_arena_marker_and_writes_only_the_digest() {
    const LD_IMM64: u8 = 0x18;
    const CALL: u8 = 0x85;
    const STACK_REG: u8 = 10;
    const CTX_SIZE: usize = 56;
    let arena_va = 0x3000_0000u64;

    let insns = bpf::program_insns(OBJECT, arena_va).expect("map-free program");
    assert!(insns.len() >= 16);
    let mut markers = 0;
    let mut helper_calls = 0;
    let mut max_ctx_write_end = 0usize;
    for (index, insn) in insns.iter().enumerate() {
        if insn[0] == 0 {
            assert!(index > 0 && insns[index - 1][0] == LD_IMM64);
            continue;
        }
        if insn[0] == LD_IMM64 {
            let tail = &insns[index + 1];
            let low = u32::from_le_bytes([insn[4], insn[5], insn[6], insn[7]]);
            let high = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]);
            if (u64::from(high) << 32) | u64::from(low) == arena_va {
                markers += 1;
            }
        }
        if insn[0] == CALL {
            helper_calls += 1;
        }
        let Some(width) = store_width(insn[0]) else {
            continue;
        };
        let dst_reg = insn[1] & 0x0f;
        let off = i16::from_le_bytes([insn[2], insn[3]]);
        if dst_reg == STACK_REG {
            continue;
        }
        assert!(off >= 0, "unexpected negative non-stack store");
        let end = off as usize + width;
        assert!(end <= CTX_SIZE, "ctx write ends at {end}");
        max_ctx_write_end = max_ctx_write_end.max(end);
    }
    assert_eq!(markers, 1, "arena VA must be patched exactly once");
    assert_eq!(helper_calls, 0, "the program must not call helpers");
    assert_eq!(max_ctx_write_end, CTX_SIZE);
}

#[cfg(rswap_bpf_object)]
fn store_width(code: u8) -> Option<usize> {
    match code {
        0x62 | 0x63 => Some(4),
        0x6a | 0x6b => Some(2),
        0x72 | 0x73 => Some(1),
        0x7a | 0x7b => Some(8),
        _ => None,
    }
}

#[cfg(rswap_bpf_object)]
fn skip_without_bpf(error: &io::Error, what: &str) -> bool {
    if matches!(
        error.kind(),
        io::ErrorKind::PermissionDenied | io::ErrorKind::Unsupported
    ) {
        eprintln!("SKIP: {what} requires BPF privileges or support: {error}");
        return true;
    }
    false
}

#[cfg(rswap_bpf_object)]
fn skip_without_arena(error: &io::Error, what: &str) -> bool {
    if skip_without_bpf(error, what) {
        return true;
    }
    if error.kind() == io::ErrorKind::InvalidInput {
        eprintln!("SKIP: {what} needs a kernel with BPF_MAP_TYPE_ARENA: {error}");
        return true;
    }
    false
}

#[cfg(rswap_bpf_object)]
fn test_run_or_skip(prog_fd: i32, ctx: &mut [u8; 56]) -> Option<u32> {
    match bpf::test_run(prog_fd, ctx) {
        Ok(retval) => Some(retval),
        Err(error) if skip_without_bpf(&error, "BPF_PROG_TEST_RUN") => None,
        Err(error) => panic!("BPF_PROG_TEST_RUN failed: {error}"),
    }
}

#[cfg(rswap_bpf_object)]
#[test]
fn privileged_program_reads_published_arena_metadata() {
    let num_entries = 2;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let pages = arena_pages(num_entries, page_size).unwrap();
    let len = pages * page_size;
    let user_va = match bpf::next_arena_va(len) {
        Ok(user_va) => user_va,
        Err(error) => panic!("arena VA: {error}"),
    };
    let arena = match Arena::create(pages as u32, user_va) {
        Ok(arena) => arena,
        Err(error) if skip_without_arena(&error, "BPF_MAP_CREATE arena") => return,
        Err(error) => panic!("BPF_MAP_CREATE arena failed: {error}"),
    };
    let mapping = Mapping::arena(&arena, len).expect("map arena at its VA");
    let data_base_va = 0x3000_0000u64;
    let metadata =
        ArenaMetadata::new(mapping, data_base_va, 4096, num_entries, num_entries).unwrap();
    let prog = match Program::load(OBJECT, &arena, user_va, "rswap_hash") {
        Ok(prog) => prog,
        Err(error) if skip_without_bpf(&error, "BPF_PROG_LOAD") => return,
        Err(error) => panic!("BPF_PROG_LOAD failed: {error}"),
    };

    metadata.prepare_slot(0, "aabbcc");
    metadata.publish_slot(0);
    let expected = PageHash::new("aabbcc", 0);
    let mut ctx = [0u8; 56];
    ctx[..8].copy_from_slice(&data_base_va.to_ne_bytes());
    let Some(retval) = test_run_or_skip(prog.raw_fd(), &mut ctx) else {
        return;
    };
    assert_eq!(retval, 1);
    assert_eq!(&ctx[8..56], expected.hash());

    metadata.clear_slot(0);
    ctx[8..].fill(0);
    let Some(retval) = test_run_or_skip(prog.raw_fd(), &mut ctx) else {
        return;
    };
    assert_eq!(retval, 0);
    assert!(ctx[8..].iter().all(|&byte| byte == 0));
}

#[cfg(rswap_bpf_object)]
#[test]
fn missing_attach_path_returns_error_from_open() {
    use vllm_kv_cache_service::hash::{HashFactory, HashGeometry};

    let config = RswapConfig {
        attach_path: std::path::PathBuf::from("/definitely/missing/rswap_attach"),
    };
    let geometry = HashGeometry {
        data_base_va: 0x3000_0000,
        page_size: 4096,
        num_entries: 2,
        pages_per_chunk: 2,
    };
    let error = config.open(geometry).err().expect("open should fail");
    if skip_without_arena(&error, "RswapConfig::open") {
        return;
    }
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
}
