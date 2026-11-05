// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Arena metadata for per-page hashes, and optional kernel hash reporting.
//!
//! Userspace owns the arena: a header page holding the geometry, then one
//! 64-byte entry per base page of the service's shared mapping. The kernel
//! never writes the arena; the rswap digest program only reads it. Each entry
//! has a single `AtomicBool` ready flag, published with `Release` after the
//! hash is written, so a reader never observes a hash before the flag.
//!
//! The 48-byte hash has a fixed little-endian layout: 36 canonical chunk-key
//! bytes, 4 bytes for the original decoded key length (high bit marks malformed
//! public input), and 8 bytes for the chunk-relative page offset in pages at
//! the end.

use std::cell::UnsafeCell;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::mem::{align_of, offset_of, size_of};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};

use crate::bpf::{self, Arena, Mapping, Program};

pub const CANONICAL_KEY_LEN: usize = 36;
pub const LENGTH_LEN: usize = 4;
pub const OFFSET_LEN: usize = 8;
pub const LENGTH_OFF: usize = CANONICAL_KEY_LEN;
pub const OFFSET_OFF: usize = CANONICAL_KEY_LEN + LENGTH_LEN;
pub const HASH_LEN: usize = 48;

/// Smallest host page size the arena ABI supports; the header is 4 KiB wide.
pub const ARENA_PAGE_SIZE: usize = 4096;
pub const ARENA_MAGIC: u64 = 0x5253_5741_5041_5248;
pub const ARENA_VERSION: u64 = 1;
pub const ENTRIES_OFF: usize = 4096;
pub const ENTRY_SIZE: usize = 64;

const SHA256_LEN: usize = 32;
const FINGERPRINT_LABEL: &[u8] = b"vllm-kv-cache-chunk-fingerprint";
const MALFORMED_KEY_LABEL: &[u8] = b"vllm-kv-cache-raw-key:";
const MALFORMED_FLAG: u32 = 1 << 31;
const MAX_ARENA_BYTES: usize = 1 << 32;

const _: () = assert!(LENGTH_LEN == size_of::<u32>());
const _: () = assert!(OFFSET_LEN == size_of::<u64>());
const _: () = assert!(LENGTH_OFF == CANONICAL_KEY_LEN);
const _: () = assert!(OFFSET_OFF == CANONICAL_KEY_LEN + LENGTH_LEN);
const _: () = assert!(CANONICAL_KEY_LEN + LENGTH_LEN + OFFSET_LEN == HASH_LEN);
const _: () = assert!(SHA256_LEN + LENGTH_LEN == CANONICAL_KEY_LEN);

#[cfg(rswap_bpf_object)]
const OBJECT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rswap_hash.bpf.o"));

/// A deterministic hash for one base page of a cache chunk.
pub struct PageHash {
    hash: [u8; HASH_LEN],
    key_len: usize,
    canonical_len: usize,
    page_offset: u64,
}

impl PageHash {
    /// Builds a per-page hash from a validated cache key.
    ///
    /// Args:
    ///     key: Even-length hexadecimal cache key; malformed input is hashed
    ///         as domain-separated raw string bytes and marked in the length
    ///         field because the service rejects it upstream.
    ///     page_offset: Zero-based page within the chunk, not a byte offset.
    ///
    /// Returns:
    ///     A deterministic page hash.
    pub fn new(key: &str, page_offset: u64) -> Self {
        match decode_hex(key) {
            Some(key_bytes) => Self::from_bytes(&key_bytes, page_offset),
            None => Self::from_malformed_key(key, page_offset),
        }
    }

    /// Builds a per-page hash from decoded cache-key bytes.
    ///
    /// Args:
    ///     key_bytes: Decoded cache key bytes.
    ///     page_offset: Zero-based page within the chunk, not a byte offset.
    ///
    /// Returns:
    ///     A deterministic page hash.
    pub fn from_bytes(key_bytes: &[u8], page_offset: u64) -> Self {
        let canonical = canonical_key(key_bytes);
        let canonical_len = key_bytes.len().min(CANONICAL_KEY_LEN);
        Self::from_canonical(
            canonical,
            key_bytes.len(),
            canonical_len,
            page_offset,
            false,
        )
    }

    fn from_malformed_key(key: &str, page_offset: u64) -> Self {
        let mut labeled = Vec::with_capacity(MALFORMED_KEY_LABEL.len() + key.len());
        labeled.extend_from_slice(MALFORMED_KEY_LABEL);
        labeled.extend_from_slice(key.as_bytes());
        Self::from_canonical(
            fingerprint(&labeled),
            key.len(),
            CANONICAL_KEY_LEN,
            page_offset,
            true,
        )
    }

    fn from_canonical(
        canonical: [u8; CANONICAL_KEY_LEN],
        key_len: usize,
        canonical_len: usize,
        page_offset: u64,
        malformed: bool,
    ) -> Self {
        let mut encoded_key_len = u32::try_from(key_len).expect("key length fits u32");
        if malformed {
            encoded_key_len |= MALFORMED_FLAG;
        } else {
            debug_assert_eq!(
                encoded_key_len & MALFORMED_FLAG,
                0,
                "valid key length overlaps malformed-key flag"
            );
        }
        let mut hash = [0u8; HASH_LEN];
        hash[..CANONICAL_KEY_LEN].copy_from_slice(&canonical);
        hash[LENGTH_OFF..OFFSET_OFF].copy_from_slice(&encoded_key_len.to_le_bytes());
        hash[OFFSET_OFF..HASH_LEN].copy_from_slice(&page_offset.to_le_bytes());
        Self {
            hash,
            key_len,
            canonical_len,
            page_offset,
        }
    }

    /// Returns the 48-byte per-page hash.
    pub fn hash(&self) -> &[u8; HASH_LEN] {
        &self.hash
    }

    /// Returns the original decoded cache-key length.
    pub fn key_len(&self) -> usize {
        self.key_len
    }

    /// Returns the number of canonical preimage bytes copied into the field.
    pub fn canonical_len(&self) -> usize {
        self.canonical_len
    }

    /// Returns the first 36 bytes of the per-page hash.
    pub fn canonical_key(&self) -> [u8; CANONICAL_KEY_LEN] {
        self.hash[..CANONICAL_KEY_LEN].try_into().expect("canonical key length")
    }

    /// Returns the chunk-relative page offset in pages.
    pub fn page_offset(&self) -> u64 {
        self.page_offset
    }

    /// Returns the encoded original key length stored in the hash.
    pub fn encoded_key_len(&self) -> u32 {
        let bytes: [u8; LENGTH_LEN] =
            self.hash[LENGTH_OFF..OFFSET_OFF].try_into().expect("length field");
        u32::from_le_bytes(bytes)
    }
}

fn canonical_key(key_bytes: &[u8]) -> [u8; CANONICAL_KEY_LEN] {
    let mut canonical = [0u8; CANONICAL_KEY_LEN];
    if key_bytes.len() <= CANONICAL_KEY_LEN {
        canonical[..key_bytes.len()].copy_from_slice(key_bytes);
        return canonical;
    }
    fingerprint(key_bytes)
}

fn fingerprint(key_bytes: &[u8]) -> [u8; CANONICAL_KEY_LEN] {
    let mut labeled_hasher = Sha256::new();
    labeled_hasher.update(FINGERPRINT_LABEL);
    labeled_hasher.update(key_bytes);
    let labeled = labeled_hasher.finalize();
    let raw = Sha256::digest(key_bytes);
    let mut canonical = [0u8; CANONICAL_KEY_LEN];
    canonical[..SHA256_LEN].copy_from_slice(&labeled[..SHA256_LEN]);
    canonical[SHA256_LEN..].copy_from_slice(&raw[..LENGTH_LEN]);
    canonical
}

fn decode_hex(key: &str) -> Option<Vec<u8>> {
    if !key.len().is_multiple_of(2) {
        return None;
    }
    key.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Some((hex_value(pair[0])? << 4) | hex_value(pair[1])?))
        .collect()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// One entry per base page of the service's shared mapping.
///
/// Zeroed arena memory means "not ready". The single `AtomicBool` is the only
/// atomic; userspace writes the hash through `UnsafeCell` while the flag is
/// false, then publishes with `Release`.
#[repr(C, align(8))]
pub struct PageMeta {
    ready: AtomicBool,
    _pad: [u8; 7],
    hash: UnsafeCell<[u8; HASH_LEN]>,
    page_index: UnsafeCell<u64>,
}

const _: () = assert!(size_of::<PageMeta>() == ENTRY_SIZE);
const _: () = assert!(align_of::<PageMeta>() == 8);
const _: () = assert!(offset_of!(PageMeta, hash) == 8);
const _: () = assert!(offset_of!(PageMeta, page_index) == 8 + HASH_LEN);

impl PageMeta {
    fn zeroed() -> Self {
        Self {
            ready: AtomicBool::new(false),
            _pad: [0; 7],
            hash: UnsafeCell::new([0; HASH_LEN]),
            page_index: UnsafeCell::new(0),
        }
    }

    fn clear(&self) {
        self.ready.store(false, Ordering::Release);
        // SAFETY: the service mutex serializes writers, and `ready` was cleared
        // with Release before these field writes.
        unsafe {
            *self.hash.get() = [0; HASH_LEN];
            *self.page_index.get() = 0;
        }
    }

    fn write(&self, page_hash: &PageHash, page_index: u64) {
        // SAFETY: this entry's `ready` flag is false, so no reader uses these
        // fields while they are rewritten; the service mutex serializes writers.
        unsafe {
            *self.hash.get() = *page_hash.hash();
            *self.page_index.get() = page_index;
        }
    }

    fn publish(&self) {
        self.ready.store(true, Ordering::Release);
    }

    /// Returns whether the hash is published.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Returns the 48-byte per-page hash.
    pub fn page_hash(&self) -> [u8; HASH_LEN] {
        // SAFETY: the service mutex serializes writers; copying avoids
        // exposing a reference into the UnsafeCell.
        unsafe { *self.hash.get() }
    }

    /// Returns the absolute base-page index in the shared mapping.
    pub fn page_index(&self) -> u64 {
        // SAFETY: the service mutex serializes writers and `u64` is Copy.
        unsafe { *self.page_index.get() }
    }

    /// Returns the chunk-relative page offset encoded in the hash.
    pub fn page_offset(&self) -> u64 {
        let hash = self.page_hash();
        let bytes: [u8; OFFSET_LEN] = hash[OFFSET_OFF..HASH_LEN].try_into().expect("offset field");
        u64::from_le_bytes(bytes)
    }
}

/// Arena page 0: geometry and the location of the entry table.
#[repr(C)]
pub struct ArenaHeader {
    pub magic: u64,
    pub version: u64,
    pub page_shift: u64,
    pub data_base_va: u64,
    pub entries_off: u64,
    pub entry_size: u64,
    pub num_entries: u64,
    pub pages_per_chunk: u64,
    pub reserved: [u64; 8],
}

const _: () = assert!(size_of::<ArenaHeader>() == 128);
const _: () = assert!(offset_of!(ArenaHeader, num_entries) == 48);
const _: () = assert!(size_of::<ArenaHeader>() <= ENTRIES_OFF);

/// A `&mut T` view of `T` at `off` in `bytes`.
///
/// # Panics
/// If `off` is not aligned for `T` or the view would leave `bytes`; both are
/// static properties of the arena layout.
fn view_mut<T>(bytes: &mut [u8], off: usize) -> &mut T {
    assert_eq!(off % align_of::<T>(), 0, "unaligned arena view at {off:#x}");
    let end = off + size_of::<T>();
    assert!(end <= bytes.len(), "arena view at {off:#x} out of bounds");
    // SAFETY: the offset is aligned and in bounds, and the arena view types are
    // plain old data with no invalid bit patterns; the borrow of `bytes` keeps
    // the view alive no longer than the slice itself.
    unsafe { &mut *bytes[off..end].as_mut_ptr().cast::<T>() }
}

/// The block metadata manager over one arena mapping.
///
/// This is the only place that knows the on-arena layout; the rswap digest
/// program mirrors it with volatile reads.
pub struct ArenaMetadata {
    _mapping: Mapping,
    header_addr: usize,
    entries_addr: usize,
    num_entries: usize,
    pages_per_chunk: usize,
}

impl ArenaMetadata {
    /// Lays out the header over an already mapped arena.
    ///
    /// Args:
    ///     mut mapping: Arena mapping; zeroed bytes mean "not ready".
    ///     data_base_va: Base VA of the service's shared mapping.
    ///     page_size: Host page size of the shared mapping.
    ///     num_entries: Base pages in the shared mapping.
    ///     pages_per_chunk: Base pages in one cache chunk.
    ///
    /// Raises:
    ///     An `io::Error` for invalid geometry or a mapping that is too small.
    pub fn new(
        mut mapping: Mapping,
        data_base_va: u64,
        page_size: usize,
        num_entries: usize,
        pages_per_chunk: usize,
    ) -> io::Result<Self> {
        if page_size == 0
            || !page_size.is_power_of_two()
            || num_entries == 0
            || pages_per_chunk == 0
            || !num_entries.is_multiple_of(pages_per_chunk)
        {
            return Err(invalid("invalid arena metadata geometry"));
        }
        let entries_end = entries_end(num_entries)?;
        if entries_end > mapping.as_slice().len() {
            return Err(invalid("arena mapping is too small for its metadata"));
        }
        let page_shift = page_size.trailing_zeros();
        let header = ArenaHeader {
            magic: ARENA_MAGIC,
            version: ARENA_VERSION,
            page_shift: u64::from(page_shift),
            data_base_va,
            entries_off: ENTRIES_OFF as u64,
            entry_size: ENTRY_SIZE as u64,
            num_entries: num_entries as u64,
            pages_per_chunk: pages_per_chunk as u64,
            reserved: [0; 8],
        };
        *view_mut::<ArenaHeader>(mapping.as_mut_slice(), 0) = header;
        let header_addr = mapping.addr();
        let entries_addr = header_addr
            .checked_add(ENTRIES_OFF)
            .ok_or_else(|| invalid("arena entry offset overflow"))?;
        // SAFETY: `entries_addr` is aligned for PageMeta and the mapping holds
        // `num_entries` zeroed slots; zeroed memory is valid for every field.
        unsafe {
            let entries =
                std::slice::from_raw_parts_mut(entries_addr as *mut PageMeta, num_entries);
            for entry in entries {
                std::ptr::write(entry, PageMeta::zeroed());
            }
        }
        Ok(Self {
            _mapping: mapping,
            header_addr,
            entries_addr,
            num_entries,
            pages_per_chunk,
        })
    }

    /// Returns the arena header.
    pub fn header(&self) -> &ArenaHeader {
        // SAFETY: `header_addr` points at the initialized header inside the
        // live mapping, and the borrow cannot outlive `self`.
        unsafe { &*(self.header_addr as *const ArenaHeader) }
    }

    /// Returns the metadata entry at `index`.
    pub fn entry(&self, index: usize) -> &PageMeta {
        assert!(index < self.num_entries, "arena entry out of bounds");
        let addr = self.entries_addr + index * ENTRY_SIZE;
        // SAFETY: the table holds `num_entries` initialized entries, `index` is
        // in bounds, and the borrow cannot outlive `self`.
        unsafe { &*(addr as *const PageMeta) }
    }

    /// Returns the number of metadata entries.
    pub fn num_entries(&self) -> usize {
        self.num_entries
    }

    /// Returns the number of base pages in one cache chunk.
    pub fn pages_per_chunk(&self) -> usize {
        self.pages_per_chunk
    }

    /// Returns the number of entries currently published as ready.
    pub fn ready_count(&self) -> usize {
        (0..self.num_entries).filter(|&index| self.entry(index).is_ready()).count()
    }

    /// Writes every hash of `slot` while keeping it unpublished.
    pub fn prepare_slot(&self, slot: usize, key: &str) {
        let range = self.slot_range(slot);
        for index in range.clone() {
            self.entry(index).clear();
        }
        for (page_offset, index) in range.enumerate() {
            let page_hash = PageHash::new(key, page_offset as u64);
            self.entry(index).write(&page_hash, index as u64);
        }
    }

    /// Publishes every page of `slot` with a Release store.
    pub fn publish_slot(&self, slot: usize) {
        for index in self.slot_range(slot) {
            self.entry(index).publish();
        }
    }

    /// Clears the ready flags and hashes of `slot`.
    pub fn clear_slot(&self, slot: usize) {
        for index in self.slot_range(slot) {
            self.entry(index).clear();
        }
    }

    /// Clears every entry.
    pub fn clear_all(&self) {
        for index in 0..self.num_entries {
            self.entry(index).clear();
        }
    }

    fn slot_range(&self, slot: usize) -> Range<usize> {
        let start = slot.checked_mul(self.pages_per_chunk).expect("slot overflow");
        let end = start.checked_add(self.pages_per_chunk).expect("slot overflow");
        assert!(end <= self.num_entries, "arena slot out of bounds");
        start..end
    }
}

impl Drop for ArenaMetadata {
    fn drop(&mut self) {
        self.clear_all();
    }
}

/// Geometry of the shared mapping the hashes describe.
#[derive(Clone, Copy, Debug)]
pub struct HashGeometry {
    pub data_base_va: u64,
    pub page_size: usize,
    pub num_entries: usize,
    pub pages_per_chunk: usize,
}

/// Publishes and clears per-page hashes for cache slots.
pub trait PageHashes: Send {
    fn publish(&mut self, slot: usize, key: &str) -> io::Result<()>;
    fn clear(&mut self, slot: usize);
    fn reset(&mut self);
}

/// Opens a hash-reporting backend for one cache mapping.
pub trait HashFactory: Send {
    fn open(&self, geometry: HashGeometry) -> io::Result<Box<dyn PageHashes>>;
}

/// Kernel sysfs path used to attach the rswap digest program.
#[derive(Clone, Debug)]
pub struct RswapConfig {
    pub attach_path: PathBuf,
}

impl HashFactory for RswapConfig {
    fn open(&self, geometry: HashGeometry) -> io::Result<Box<dyn PageHashes>> {
        #[cfg(not(rswap_bpf_object))]
        {
            let _ = geometry;
            return Err(io::Error::other(
                "service was built without the rswap BPF object (clang unavailable)",
            ));
        }
        #[cfg(rswap_bpf_object)]
        {
            let pages = arena_pages(geometry.num_entries, geometry.page_size)?;
            let len = pages
                .checked_mul(geometry.page_size)
                .ok_or_else(|| invalid("arena size overflow"))?;
            let user_va = bpf::next_arena_va(len)?;
            let arena = Arena::create(pages as u32, user_va)?;
            let mapping = Mapping::arena(&arena, len)?;
            let metadata = ArenaMetadata::new(
                mapping,
                geometry.data_base_va,
                geometry.page_size,
                geometry.num_entries,
                geometry.pages_per_chunk,
            )?;
            let prog = Program::load(OBJECT, &arena, user_va, "rswap_hash")?;
            write_attach(&self.attach_path, geometry.data_base_va, prog.raw_fd())?;
            Ok(Box::new(RswapHashReports {
                metadata,
                _arena: arena,
                _prog: prog,
                attach_path: self.attach_path.clone(),
                attach_va: geometry.data_base_va,
            }))
        }
    }
}

/// Hash reporting backed by arena metadata read by the rswap digest program.
///
/// The kernel copies the published 48-byte hash into the swap cluster before
/// the mapping disappears; the program runs under the page table lock and must
/// not sleep. Arena pages are faulted only for entries that are published.
///
/// Attachment is per-VMA in the writing task's mm: workers that map the shared
/// mapping in their own processes need their own attachment.
/// TODO: attach the program in worker processes too.
pub struct RswapHashReports {
    metadata: ArenaMetadata,
    // Kept alive for as long as the metadata is reachable from the program.
    _arena: Arena,
    _prog: Program,
    attach_path: PathBuf,
    attach_va: u64,
}

impl PageHashes for RswapHashReports {
    fn publish(&mut self, slot: usize, key: &str) -> io::Result<()> {
        self.metadata.prepare_slot(slot, key);
        self.metadata.publish_slot(slot);
        Ok(())
    }

    fn clear(&mut self, slot: usize) {
        self.metadata.clear_slot(slot);
    }

    fn reset(&mut self) {
        self.metadata.clear_all();
    }
}

impl Drop for RswapHashReports {
    fn drop(&mut self) {
        let _ = write_attach(&self.attach_path, self.attach_va, 0);
    }
}

/// Writes one `<vaddr-hex> <prog-fd>` sysfs attach command.
///
/// The command must reach the attribute as a single write: sysfs stores run
/// once per `write(2)`, so a formatted multi-part write parses as several
/// truncated commands.
///
/// Args:
///     path: Kernel sysfs attach attribute, or a test file.
///     vaddr: Base virtual address of the shared mapping's VMA.
///     prog_fd: Program fd to attach; zero detaches.
///
/// Raises:
///     An `io::Error` if the attribute cannot be opened or written.
pub fn write_attach(path: &Path, vaddr: u64, prog_fd: i32) -> io::Result<()> {
    let command = format!("{vaddr:#x} {prog_fd}\n");
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.write_all(command.as_bytes())
}

fn entries_end(num_entries: usize) -> io::Result<usize> {
    let bytes = num_entries
        .checked_mul(ENTRY_SIZE)
        .ok_or_else(|| invalid("arena entry table overflow"))?;
    ENTRIES_OFF
        .checked_add(bytes)
        .ok_or_else(|| invalid("arena entry table overflow"))
}

/// Returns the arena page count needed for `num_entries` entries.
///
/// Args:
///     num_entries: Entries the arena must hold.
///     page_size: Host page size; `BPF_MAP_TYPE_ARENA` sizes and maps in host
///         pages, and the mapping length must match the map exactly.
///
/// Returns:
///     The number of `page_size` pages that cover the header and entries.
///
/// Raises:
///     An `io::Error` for overflow or an arena larger than 4 GiB.
pub fn arena_pages(num_entries: usize, page_size: usize) -> io::Result<usize> {
    if page_size < ARENA_PAGE_SIZE || !page_size.is_power_of_two() {
        return Err(invalid(
            "arena page size must be a power of two of at least 4 KiB",
        ));
    }
    let pages = entries_end(num_entries)?.div_ceil(page_size);
    let bytes = pages.checked_mul(page_size).ok_or_else(|| invalid("arena size overflow"))?;
    if pages == 0 || bytes > MAX_ARENA_BYTES {
        return Err(invalid("arena metadata exceeds the 4 GiB arena limit"));
    }
    Ok(pages)
}

fn invalid(message: &str) -> io::Error {
    io::Error::other(message)
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;
    use std::fs;

    use super::*;

    #[test]
    fn page_hash_golden_vector_uses_fixed_layout() {
        let page_hash = PageHash::new("aa", 1);
        let mut expected = [0u8; HASH_LEN];
        expected[0] = 0xaa;
        expected[LENGTH_OFF..OFFSET_OFF].copy_from_slice(&1u32.to_le_bytes());
        expected[OFFSET_OFF..HASH_LEN].copy_from_slice(&1u64.to_le_bytes());
        assert!(expected[1..LENGTH_OFF].iter().all(|&byte| byte == 0));
        assert_eq!(page_hash.hash(), &expected);
        assert_eq!(page_hash.canonical_len(), 1);
        assert_eq!(page_hash.key_len(), 1);
        assert_eq!(page_hash.encoded_key_len(), 1);
        assert_eq!(page_hash.page_offset(), 1);
    }

    #[test]
    fn canonical_len_tracks_copied_preimage_bytes() {
        assert_eq!(PageHash::from_bytes(&[0xaa], 0).canonical_len(), 1);
        assert_eq!(
            PageHash::from_bytes(&[0; CANONICAL_KEY_LEN], 0).canonical_len(),
            CANONICAL_KEY_LEN
        );
        assert_eq!(
            PageHash::from_bytes(&[0; CANONICAL_KEY_LEN + 1], 0).canonical_len(),
            CANONICAL_KEY_LEN
        );
        assert_eq!(
            PageHash::new("not-hex", 0).canonical_len(),
            CANONICAL_KEY_LEN
        );
    }

    #[test]
    fn page_hash_disambiguates_padding_from_key_and_offset_bytes() {
        let short_with_offset = PageHash::new("aa", 1);
        let longer_without_offset = PageHash::new("aa01", 0);
        assert_ne!(short_with_offset.hash(), longer_without_offset.hash());
        assert_eq!(short_with_offset.key_len(), 1);
        assert_eq!(longer_without_offset.key_len(), 2);
        assert_eq!(
            &short_with_offset.hash()[LENGTH_OFF..OFFSET_OFF],
            &1u32.to_le_bytes()[..]
        );
        assert_eq!(
            &longer_without_offset.hash()[LENGTH_OFF..OFFSET_OFF],
            &2u32.to_le_bytes()[..]
        );
        assert_eq!(
            &short_with_offset.hash()[OFFSET_OFF..HASH_LEN],
            &1u64.to_le_bytes()[..]
        );
        assert_eq!(
            &longer_without_offset.hash()[OFFSET_OFF..HASH_LEN],
            &0u64.to_le_bytes()[..]
        );
    }

    #[test]
    fn page_hash_length_disambiguates_fingerprint_from_short_key() {
        let long_key = "ab".repeat(CANONICAL_KEY_LEN + 1);
        let long = PageHash::new(&long_key, 0);
        let decoded = decode_hex(&long_key).unwrap();
        let mut labeled_hasher = Sha256::new();
        labeled_hasher.update(FINGERPRINT_LABEL);
        labeled_hasher.update(&decoded);
        let labeled = labeled_hasher.finalize();
        let raw = Sha256::digest(&decoded);
        let mut expected_canonical = [0u8; CANONICAL_KEY_LEN];
        expected_canonical[..SHA256_LEN].copy_from_slice(&labeled[..SHA256_LEN]);
        expected_canonical[SHA256_LEN..].copy_from_slice(&raw[..LENGTH_LEN]);
        assert_eq!(long.canonical_key(), expected_canonical);

        let mut short_key = String::with_capacity(CANONICAL_KEY_LEN * 2);
        for byte in expected_canonical {
            write!(short_key, "{byte:02x}").unwrap();
        }
        let short = PageHash::new(&short_key, 0);
        assert_eq!(short.canonical_key(), long.canonical_key());
        assert_eq!(short.key_len(), CANONICAL_KEY_LEN);
        assert_eq!(long.key_len(), CANONICAL_KEY_LEN + 1);
        assert_ne!(short.hash(), long.hash());
    }

    #[test]
    fn malformed_key_does_not_alias_hex_decoded_bytes() {
        let malformed = PageHash::new("not-hex", 0);
        let valid = PageHash::new("6e6f742d686578", 0);
        assert_eq!(malformed.key_len(), valid.key_len());
        assert_ne!(malformed.canonical_key(), valid.canonical_key());
        assert_ne!(malformed.hash(), valid.hash());
    }

    #[test]
    fn malformed_fallback_cannot_alias_its_fingerprint_as_a_valid_key() {
        let malformed = "z".repeat(CANONICAL_KEY_LEN);
        let raw = PageHash::new(&malformed, 0);
        let mut valid_key = String::with_capacity(CANONICAL_KEY_LEN * 2);
        for byte in raw.canonical_key() {
            write!(valid_key, "{byte:02x}").unwrap();
        }
        let valid = PageHash::new(&valid_key, 0);
        assert_eq!(raw.canonical_key(), valid.canonical_key());
        assert_ne!(raw.hash(), valid.hash());
        assert_eq!(raw.encoded_key_len() & MALFORMED_FLAG, MALFORMED_FLAG);
        assert_eq!(valid.encoded_key_len() & MALFORMED_FLAG, 0);
    }

    #[test]
    fn page_offset_is_little_endian_at_end_of_hash() {
        let offset = 0x0102_0304_0506_0708u64;
        let page_hash = PageHash::new("aa", offset);
        assert_eq!(
            &page_hash.hash()[OFFSET_OFF..HASH_LEN],
            &offset.to_le_bytes()[..]
        );
        assert_eq!(page_hash.page_offset(), offset);
    }

    #[test]
    fn metadata_publishes_and_clears_arena_entries() {
        let mapping = Mapping::anonymous(entries_end(4).unwrap()).unwrap();
        let arena = ArenaMetadata::new(mapping, 0x3000_0000, 4096, 4, 2).unwrap();
        let header = arena.header();
        assert_eq!(header.magic, ARENA_MAGIC);
        assert_eq!(header.version, ARENA_VERSION);
        assert_eq!(header.page_shift, 12);
        assert_eq!(header.entries_off, ENTRIES_OFF as u64);
        assert_eq!(header.entry_size, ENTRY_SIZE as u64);
        assert_eq!(header.num_entries, 4);
        assert_eq!(header.pages_per_chunk, 2);
        assert_eq!(arena.num_entries(), 4);

        arena.prepare_slot(1, "aabbcc");
        let expected = PageHash::new("aabbcc", 0);
        assert_eq!(arena.ready_count(), 0);
        assert!(!arena.entry(2).is_ready());
        assert_eq!(arena.entry(2).page_hash(), *expected.hash());
        assert_eq!(arena.entry(2).page_index(), 2);
        assert_eq!(arena.entry(2).page_offset(), 0);
        assert_eq!(arena.entry(3).page_offset(), 1);

        arena.publish_slot(1);
        assert_eq!(arena.ready_count(), 2);
        assert!(arena.entry(2).is_ready() && arena.entry(3).is_ready());
        arena.clear_slot(1);
        assert_eq!(arena.ready_count(), 0);
        assert_eq!(arena.entry(2).page_hash(), [0; HASH_LEN]);
        assert_eq!(arena.entry(2).page_index(), 0);

        arena.prepare_slot(0, "ddeeff");
        arena.publish_slot(0);
        arena.clear_all();
        assert_eq!(arena.ready_count(), 0);
        assert!(!arena.entry(0).is_ready());
    }

    #[test]
    fn arena_pages_cover_header_and_entries() {
        assert_eq!(arena_pages(1, ARENA_PAGE_SIZE).unwrap(), 2);
        assert_eq!(arena_pages(64, ARENA_PAGE_SIZE).unwrap(), 2);
        assert_eq!(arena_pages(65, ARENA_PAGE_SIZE).unwrap(), 3);
        assert_eq!(arena_pages(1, 16384).unwrap(), 1);
        assert_eq!(arena_pages(4096, 16384).unwrap(), 17);
        assert!(arena_pages(usize::MAX, ARENA_PAGE_SIZE).is_err());
        assert!(arena_pages(1, 2048).is_err());
        assert!(arena_pages(1, 5000).is_err());
    }

    #[test]
    fn write_attach_formats_attach_and_detach() {
        let file = tempfile::NamedTempFile::new().unwrap();
        write_attach(file.path(), 0x3000_0000, 7).unwrap();
        assert_eq!(fs::read_to_string(file.path()).unwrap(), "0x30000000 7\n");
        write_attach(file.path(), 0x3000_0000, 0).unwrap();
        assert_eq!(fs::read_to_string(file.path()).unwrap(), "0x30000000 0\n");
    }
}
