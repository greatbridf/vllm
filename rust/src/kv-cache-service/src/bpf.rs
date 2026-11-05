// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Raw BPF syscalls, the arena mapping, and the ELF loader for
//! `rswap_hash.bpf.o`.
//!
//! The program has no maps and no relocations: it reads arena metadata through
//! a scalar arena address. The loader replaces the object's
//! `RSWAP_ARENA_MARKER` immediate with the arena's user VA and binds the arena
//! to the program through `fd_array`, which is what makes the verifier set
//! `prog->aux->arena`.

use std::ffi::CStr;
use std::io;
use std::mem::{offset_of, size_of};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicU64, Ordering};

const BPF_MAP_CREATE: u32 = 0;
const BPF_PROG_LOAD: u32 = 5;
const BPF_PROG_TEST_RUN: u32 = 10;
const BPF_MAP_TYPE_ARENA: u32 = 33;
const BPF_F_MMAPABLE: u32 = 1 << 10;
const BPF_F_SLEEPABLE: u32 = 1 << 4;
const BPF_F_ARENA_SCALAR: u32 = 1 << 8;
const BPF_PROG_TYPE_SYSCALL: u32 = 31;

const BPF_OBJ_NAME_LEN: usize = 16;
const LOG_SIZE: usize = 1 << 16;
const BPF_LD_IMM64: u8 = 0x18;
const EHDR_SIZE: usize = 64;
const SHDR_SIZE: usize = 64;
const ET_REL: u16 = 1;
const EM_BPF: u16 = 247;
const SHT_PROGBITS: u32 = 1;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_REL: u32 = 9;

/// Marker the loader replaces with the arena user VA.
const RSWAP_ARENA_MARKER: u64 = 0x5253_5741_5052_4f42;

/// First arena VA hint. Arenas are 4 GiB aligned so one arena never crosses a
/// 32-bit boundary, which is what `BPF_F_ARENA_SCALAR` rebasing requires.
const ARENA_VA_BASE: u64 = 0x20_0000_0000;
const ARENA_VA_STRIDE: u64 = 1 << 32;

static NEXT_ARENA_VA: AtomicU64 = AtomicU64::new(ARENA_VA_BASE);

// Byte-exact prefixes of `union bpf_attr`; the kernel reads fields Rust does not.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)]
struct MapCreateAttr {
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_flags: u32,
    inner_map_fd: u32,
    numa_node: u32,
    map_name: [u8; BPF_OBJ_NAME_LEN],
    map_ifindex: u32,
    btf_fd: u32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    map_extra: u64,
}

const _: () = assert!(offset_of!(MapCreateAttr, map_type) == 0);
const _: () = assert!(offset_of!(MapCreateAttr, key_size) == 4);
const _: () = assert!(offset_of!(MapCreateAttr, value_size) == 8);
const _: () = assert!(offset_of!(MapCreateAttr, max_entries) == 12);
const _: () = assert!(offset_of!(MapCreateAttr, map_flags) == 16);
const _: () = assert!(offset_of!(MapCreateAttr, map_name) == 28);
const _: () = assert!(offset_of!(MapCreateAttr, map_extra) == 64);
const _: () = assert!(size_of::<MapCreateAttr>() == 72);

#[repr(C)]
#[derive(Default)]
#[allow(dead_code)]
struct ProgLoadAttr {
    prog_type: u32,
    insn_cnt: u32,
    insns: u64,
    license: u64,
    log_level: u32,
    log_size: u32,
    log_buf: u64,
    kern_version: u32,
    prog_flags: u32,
    prog_name: [u8; BPF_OBJ_NAME_LEN],
    prog_ifindex: u32,
    expected_attach_type: u32,
    prog_btf_fd: u32,
    func_info_rec_size: u32,
    func_info: u64,
    func_info_cnt: u32,
    line_info_rec_size: u32,
    line_info: u64,
    line_info_cnt: u32,
    attach_btf_id: u32,
    attach_prog_fd: u32,
    core_relo_cnt: u32,
    fd_array: u64,
    core_relos: u64,
    core_relo_rec_size: u32,
    log_true_size: u32,
    prog_token_fd: i32,
    fd_array_cnt: u32,
}

const _: () = assert!(offset_of!(ProgLoadAttr, insns) == 8);
const _: () = assert!(offset_of!(ProgLoadAttr, license) == 16);
const _: () = assert!(offset_of!(ProgLoadAttr, prog_flags) == 44);
const _: () = assert!(offset_of!(ProgLoadAttr, prog_name) == 48);
const _: () = assert!(offset_of!(ProgLoadAttr, fd_array) == 120);
const _: () = assert!(offset_of!(ProgLoadAttr, fd_array_cnt) == 148);
const _: () = assert!(size_of::<ProgLoadAttr>() == 152);

#[repr(C)]
#[derive(Default)]
#[allow(dead_code)]
struct TestRunAttr {
    prog_fd: u32,
    retval: u32,
    data_size_in: u32,
    data_size_out: u32,
    data_in: u64,
    data_out: u64,
    repeat: u32,
    duration: u32,
    ctx_size_in: u32,
    ctx_size_out: u32,
    ctx_in: u64,
    ctx_out: u64,
    flags: u32,
}

const _: () = assert!(offset_of!(TestRunAttr, prog_fd) == 0);
const _: () = assert!(offset_of!(TestRunAttr, retval) == 4);
const _: () = assert!(offset_of!(TestRunAttr, ctx_size_in) == 40);
const _: () = assert!(offset_of!(TestRunAttr, ctx_size_out) == 44);
const _: () = assert!(offset_of!(TestRunAttr, ctx_in) == 48);
const _: () = assert!(offset_of!(TestRunAttr, ctx_out) == 56);
const _: () = assert!(offset_of!(TestRunAttr, flags) == 64);
const _: () = assert!(size_of::<TestRunAttr>() == 72);

fn bpf_syscall<T>(cmd: u32, attr: &mut T) -> io::Result<i32> {
    // SAFETY: `attr` is a live writable object and the kernel is given exactly
    // its byte size. BPF returns -1 on failure and a non-negative fd otherwise.
    let result = unsafe {
        libc::syscall(
            libc::SYS_bpf,
            cmd as libc::c_ulong,
            attr as *mut T,
            size_of::<T>() as libc::c_ulong,
        )
    };
    if result < 0 {
        return Err(io::Error::last_os_error());
    }
    i32::try_from(result).map_err(|_| io::Error::other("BPF syscall fd overflow"))
}

/// Returns the next 4 GiB-aligned arena VA for a mapping of `len` bytes.
pub fn next_arena_va(len: usize) -> io::Result<u64> {
    let len = u64::try_from(len).map_err(|_| io::Error::other("arena size overflow"))?;
    let stride = len
        .checked_add(ARENA_VA_STRIDE)
        .ok_or_else(|| io::Error::other("arena size overflow"))?;
    let stride = stride.div_ceil(ARENA_VA_STRIDE) * ARENA_VA_STRIDE;
    NEXT_ARENA_VA
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(stride)
        })
        .map_err(|_| io::Error::other("arena address space exhausted"))
}

/// A `BPF_MAP_TYPE_ARENA` map created for one cache namespace.
pub struct Arena {
    fd: OwnedFd,
    user_va: u64,
}

impl Arena {
    /// Creates an arena map of `pages` 4 KiB pages at `user_va`.
    ///
    /// Args:
    ///     pages: Arena size in 4 KiB pages.
    ///     user_va: Fixed user VA the arena must be mapped at.
    ///
    /// Raises:
    ///     An `io::Error` if the kernel rejects the arena geometry.
    pub fn create(pages: u32, user_va: u64) -> io::Result<Self> {
        let mut attr = MapCreateAttr {
            map_type: BPF_MAP_TYPE_ARENA,
            max_entries: pages,
            map_flags: BPF_F_MMAPABLE,
            map_name: obj_name("rswap_hash"),
            map_extra: user_va,
            ..Default::default()
        };
        let fd = bpf_syscall(BPF_MAP_CREATE, &mut attr)?;
        // SAFETY: BPF_MAP_CREATE returned a new descriptor owned only here.
        Ok(Self {
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
            user_va,
        })
    }

    pub fn raw_fd(&self) -> i32 {
        self.fd.as_raw_fd()
    }

    pub fn user_va(&self) -> u64 {
        self.user_va
    }
}

/// An `mmap`ed region that is unmapped on drop.
pub struct Mapping {
    // Addresses are usize rather than raw pointers so the arena stays Send.
    addr: usize,
    len: usize,
}

impl Mapping {
    /// Maps `len` zeroed bytes privately.
    pub fn anonymous(len: usize) -> io::Result<Self> {
        Self::new(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
        )
    }

    /// Maps `len` bytes of `fd` at exactly `addr`.
    ///
    /// Args:
    ///     arena: Arena owning the map descriptor.
    ///     len: Mapping length in bytes.
    ///
    /// Raises:
    ///     An `io::Error` if the mapping is not placed at the arena VA.
    pub fn arena(arena: &Arena, len: usize) -> io::Result<Self> {
        let map = Self::new(
            arena.user_va as *mut libc::c_void,
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_FIXED,
            arena.raw_fd(),
        )?;
        if map.addr() as u64 != arena.user_va {
            return Err(io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                format!(
                    "mapped at {:#x} instead of {:#x}",
                    map.addr(),
                    arena.user_va
                ),
            ));
        }
        Ok(map)
    }

    fn new(
        addr: *mut libc::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
    ) -> io::Result<Self> {
        // SAFETY: the caller supplies valid mmap arguments; MAP_FAILED is
        // checked before the returned address is stored.
        let ptr = unsafe { libc::mmap(addr, len, prot, flags, fd, 0) };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            addr: ptr as usize,
            len,
        })
    }

    pub fn addr(&self) -> usize {
        self.addr
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the region is mapped read/write for `len` bytes for as long
        // as `self` is alive.
        unsafe { std::slice::from_raw_parts(self.addr as *const u8, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.addr as *mut u8, self.len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the region was returned by mmap and is still owned here.
        unsafe {
            libc::munmap(self.addr as *mut libc::c_void, self.len);
        }
    }
}

/// Returns the `syscall` section with the arena VA patched in.
///
/// Args:
///     object: Bytes of a BPF ELF relocatable object.
///     arena_va: Arena user VA the program must read.
///
/// Returns:
///     Eight-byte BPF instructions ready for `BPF_PROG_LOAD`.
///
/// Raises:
///     A string error for a malformed object, an unexpected relocation, or a
///     missing/duplicate arena marker.
pub fn program_insns(object: &[u8], arena_va: u64) -> Result<Vec<[u8; 8]>, String> {
    let (sections, shstrndx) = sections(object)?;
    let names = sections.get(shstrndx).ok_or("section-name table index is out of bounds")?;
    if names.kind != SHT_STRTAB {
        return Err("section-name table is not a string table".into());
    }
    let names_offset = to_usize(names.offset, "section-name table offset")?;
    let mut syscall = None;
    for (index, header) in sections.iter().enumerate() {
        if header.kind != SHT_PROGBITS {
            continue;
        }
        let at = checked_add(names_offset, header.name as usize, "section name")?;
        if string_at(object, at, "section name")? == "syscall" {
            if syscall.is_some() {
                return Err("object has multiple 'syscall' sections".into());
            }
            syscall = Some((index, *header));
        }
    }
    let (syscall_index, header) =
        syscall.ok_or_else(|| "no 'syscall' PROGBITS section in object".to_string())?;
    for section in &sections {
        if (section.kind == SHT_REL || section.kind == SHT_RELA)
            && section.info as usize == syscall_index
        {
            return Err("'syscall' has relocations, but the program must be map-free".into());
        }
    }
    let start = to_usize(header.offset, "syscall section offset")?;
    let size = to_usize(header.size, "syscall section size")?;
    if size % 8 != 0 {
        return Err(format!(
            "'syscall' section is {size} bytes, not a multiple of 8"
        ));
    }
    let end = checked_add(start, size, "syscall section")?;
    let bytes = object
        .get(start..end)
        .ok_or_else(|| "'syscall' section is outside the file".to_string())?;
    let mut insns: Vec<[u8; 8]> = bytes.as_chunks::<8>().0.to_vec();

    let mut patched = 0usize;
    for index in 0..insns.len() {
        if insns[index][0] != BPF_LD_IMM64 {
            continue;
        }
        let Some(tail) = insns.get(index + 1) else {
            return Err("ld_imm64 at the end of the section".into());
        };
        if tail[0] != 0 {
            return Err(format!(
                "ld_imm64 at instruction {index} has a non-zero tail"
            ));
        }
        let low = u32::from_le_bytes([
            insns[index][4],
            insns[index][5],
            insns[index][6],
            insns[index][7],
        ]);
        let high = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]);
        if (u64::from(high) << 32) | u64::from(low) != RSWAP_ARENA_MARKER {
            continue;
        }
        let low = arena_va as u32;
        let high = (arena_va >> 32) as u32;
        insns[index][4..8].copy_from_slice(&low.to_le_bytes());
        insns[index + 1][4..8].copy_from_slice(&high.to_le_bytes());
        patched += 1;
    }
    match patched {
        1 => {}
        0 => return Err("arena marker is missing from the program".into()),
        count => return Err(format!("arena marker occurs {count} times")),
    }
    for (index, insn) in insns.iter().enumerate() {
        if insn[0] == 0 && (index == 0 || insns[index - 1][0] != BPF_LD_IMM64) {
            return Err(format!("invalid zero opcode at instruction {index}"));
        }
    }
    Ok(insns)
}

/// An owning loaded BPF program descriptor and its instruction count.
pub struct Program(OwnedFd, usize);

impl Program {
    /// Loads the `syscall` section as a sleepable syscall program bound to an
    /// arena.
    ///
    /// Args:
    ///     object: Bytes of a BPF ELF relocatable object.
    ///     arena: Arena whose fd the verifier associates with the program.
    ///     arena_va: Arena user VA patched into the program.
    ///     name: Kernel-visible program name.
    ///
    /// Raises:
    ///     An `io::Error` including the verifier log on load failure.
    pub fn load(object: &[u8], arena: &Arena, arena_va: u64, name: &str) -> io::Result<Self> {
        let insns = program_insns(object, arena_va)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let fd_array = [arena.raw_fd()];
        let mut log_buf = vec![0u8; LOG_SIZE];
        let mut attr = ProgLoadAttr {
            prog_type: BPF_PROG_TYPE_SYSCALL,
            insn_cnt: insns.len() as u32,
            insns: ptr_value(insns.as_ptr()),
            license: ptr_value(c"GPL".as_ptr()),
            log_level: 1,
            log_size: log_buf.len() as u32,
            log_buf: ptr_value(log_buf.as_mut_ptr()),
            prog_flags: BPF_F_SLEEPABLE | BPF_F_ARENA_SCALAR,
            prog_name: obj_name(name),
            fd_array: ptr_value(fd_array.as_ptr()),
            fd_array_cnt: fd_array.len() as u32,
            ..Default::default()
        };
        let result = bpf_syscall(BPF_PROG_LOAD, &mut attr);
        let log = CStr::from_bytes_until_nul(&log_buf)
            .map(|log| log.to_string_lossy().into_owned())
            .unwrap_or_default();
        match result {
            Ok(fd) => {
                // SAFETY: BPF_PROG_LOAD returned a new descriptor owned only
                // here; load errors were handled before constructing the fd.
                Ok(Self(unsafe { OwnedFd::from_raw_fd(fd) }, insns.len()))
            }
            Err(error) => Err(io::Error::new(
                error.kind(),
                format!("BPF_PROG_LOAD failed: {error}; verifier log:\n{log}"),
            )),
        }
    }

    pub fn raw_fd(&self) -> i32 {
        self.0.as_raw_fd()
    }

    /// Returns the number of instructions submitted to the verifier.
    pub fn insn_count(&self) -> usize {
        self.1
    }
}

/// Runs a syscall program once as a test or diagnostic helper.
///
/// The kernel copies the mutated context back through `ctx_in` for syscall
/// programs, so `ctx` is both input and output.
///
/// Args:
///     prog_fd: Loaded syscall program descriptor.
///     ctx: Fifty-six-byte `{vaddr, hash[48]}` context.
///
/// Returns:
///     The program's return value.
///
/// Raises:
///     An `io::Error` if the test-run command fails.
pub fn test_run(prog_fd: i32, ctx: &mut [u8; 56]) -> io::Result<u32> {
    let mut attr = TestRunAttr {
        prog_fd: prog_fd as u32,
        ctx_size_in: size_of_val(ctx) as u32,
        ctx_in: ptr_value(ctx.as_mut_ptr()),
        ..Default::default()
    };
    // Syscall test runs reject ctx_out, repeat, data, flags, and duration.
    bpf_syscall(BPF_PROG_TEST_RUN, &mut attr)?;
    Ok(attr.retval)
}

#[derive(Clone, Copy)]
struct SectionHeader {
    name: u32,
    kind: u32,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
}

impl SectionHeader {
    fn parse(elf: &[u8], at: usize) -> Result<Self, String> {
        let bytes = slice(elf, at, SHDR_SIZE)?;
        Ok(Self {
            name: read_u32(bytes, 0)?,
            kind: read_u32(bytes, 4)?,
            offset: read_u64(bytes, 24)?,
            size: read_u64(bytes, 32)?,
            link: read_u32(bytes, 40)?,
            info: read_u32(bytes, 44)?,
        })
    }
}

fn sections(elf: &[u8]) -> Result<(Vec<SectionHeader>, usize), String> {
    check_header(elf)?;
    let shoff = to_usize(read_u64(elf, 0x28)?, "section header offset")?;
    let shentsize = read_u16(elf, 0x3a)? as usize;
    let mut shnum = read_u16(elf, 0x3c)? as usize;
    let mut shstrndx = read_u16(elf, 0x3e)? as usize;
    if shentsize < SHDR_SIZE {
        return Err(format!(
            "e_shentsize is {shentsize}, expected >= {SHDR_SIZE}"
        ));
    }
    let header = |index: usize| -> Result<SectionHeader, String> {
        let stride = index.checked_mul(shentsize).ok_or("section header offset overflow")?;
        SectionHeader::parse(elf, checked_add(shoff, stride, "section header")?)
    };
    if shnum == 0 {
        shnum = to_usize(header(0)?.size, "extended section count")?;
    }
    if shstrndx == 0xffff {
        shstrndx = header(0)?.link as usize;
    }
    let mut sections = Vec::with_capacity(shnum);
    for index in 0..shnum {
        sections.push(header(index)?);
    }
    if shstrndx >= sections.len() {
        return Err("section-name table index is out of bounds".into());
    }
    Ok((sections, shstrndx))
}

fn check_header(elf: &[u8]) -> Result<(), String> {
    if elf.len() < EHDR_SIZE || &elf[..4] != b"\x7fELF" {
        return Err("not an ELF file".into());
    }
    if elf[4] != 2 {
        return Err("not an ELF64 file".into());
    }
    if elf[5] != 1 {
        return Err("not a little-endian ELF file".into());
    }
    if read_u16(elf, 16)? != ET_REL {
        return Err("not a relocatable (ET_REL) object".into());
    }
    let machine = read_u16(elf, 18)?;
    if machine != EM_BPF {
        return Err(format!(
            "e_machine is {machine}, expected EM_BPF ({EM_BPF})"
        ));
    }
    Ok(())
}

fn string_at<'a>(elf: &'a [u8], at: usize, what: &str) -> Result<&'a str, String> {
    let rest = elf.get(at..).ok_or_else(|| format!("truncated {what} at {at:#x}"))?;
    let end = rest
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(|| format!("unterminated {what}"))?;
    std::str::from_utf8(&rest[..end]).map_err(|error| format!("{what} is not UTF-8: {error}"))
}

fn slice(bytes: &[u8], at: usize, len: usize) -> Result<&[u8], String> {
    let end = checked_add(at, len, "ELF field")?;
    bytes.get(at..end).ok_or_else(|| format!("truncated ELF at {at:#x}"))
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16, String> {
    let bytes = slice(bytes, at, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, String> {
    let bytes = slice(bytes, at, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(bytes: &[u8], at: usize) -> Result<u64, String> {
    let bytes = slice(bytes, at, 8)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn checked_add(base: usize, offset: usize, what: &str) -> Result<usize, String> {
    base.checked_add(offset).ok_or_else(|| format!("{what} offset overflow"))
}

fn to_usize(value: u64, what: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{what} exceeds address space"))
}

fn ptr_value<T>(ptr: *const T) -> u64 {
    ptr as usize as u64
}

fn obj_name(name: &str) -> [u8; BPF_OBJ_NAME_LEN] {
    let mut out = [0u8; BPF_OBJ_NAME_LEN];
    let len = name.len().min(BPF_OBJ_NAME_LEN - 1);
    out[..len].copy_from_slice(&name.as_bytes()[..len]);
    out
}
