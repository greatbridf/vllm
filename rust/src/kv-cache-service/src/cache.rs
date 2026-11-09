// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs;
use std::fs::File;
use std::io;
use std::os::fd::FromRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use memmap2::{MmapMut, MmapOptions};
use serde::Serialize;

use crate::hash::{HashFactory, HashGeometry, PageHashes};
use crate::protocol::{MAX_KEYS, store_reply_fits};

/// Absent slot, at either end of the LRU order. Chunk ids stay below it.
const NO_SLOT: u32 = u32::MAX;

/// One chunk slot. `prev`/`next` thread the LRU order, which holds every slot a
/// key is indexed by, most recently used first; `readers` holds interned request
/// ids so a lease costs no allocation per hit.
struct Slot {
    key: Box<str>,
    prev: u32,
    next: u32,
    readers: Vec<u32>,
    ready: bool,
    /// Whether the slot's row is locked against reclaim for its readers.
    pinned: bool,
    reservation: Option<Reservation>,
}

impl Slot {
    fn idle() -> Slot {
        Slot {
            key: Box::default(),
            prev: NO_SLOT,
            next: NO_SLOT,
            readers: Vec::new(),
            ready: false,
            pinned: false,
            reservation: None,
        }
    }
}

/// The request a pending slot is reserved for and when `store` made it, so only
/// that request can publish it and a reservation nobody commits can be reclaimed.
struct Reservation {
    owner: u32,
    since: Instant,
}

/// A batch of expired reservations the service dropped, one entry per owning
/// request, for the namespace log line.
#[derive(Debug, PartialEq, Eq)]
pub struct Reclaimed {
    pub request: Box<str>,
    pub keys: usize,
    /// Age of the oldest reclaimed reservation of that request.
    pub age: Duration,
}

/// What `Cache::lookup` found for one key, spelled as the wire reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Hit,
    Pending,
    Miss,
}

/// How far a lookup-time readahead reaches.
#[derive(Clone, Copy, Debug)]
pub struct PrefetchLimits {
    /// How long a slot stays out of the readahead after one started for it, so a
    /// lookup the readahead has not satisfied yet does not queue the same pages
    /// again; 0 prefetches on every lookup.
    pub cooldown: Duration,
    /// Rows one readahead may start; a wider batch counts the rest as skipped and
    /// leaves them for a later lookup.
    pub max_chunks: usize,
}

impl Default for PrefetchLimits {
    fn default() -> Self {
        Self {
            cooldown: Duration::from_secs(2),
            max_chunks: 32,
        }
    }
}

pub struct Cache {
    pub owner: u64,
    pub worker_bytes: usize,
    pub world_size: usize,
    pub row_bytes: usize,
    pub page_size: usize,
    pub num_chunks: usize,
    pub file: File,
    /// Shared so a prefetch can run after the cache lock is released.
    pub mapping: Arc<MmapMut>,
    slots: Vec<Slot>,
    /// Key to slot for every reserved or published chunk.
    index: HashMap<Box<str>, u32>,
    /// The slots `index` does not own; together they cover `slots`.
    free: Vec<u32>,
    /// Most recently used slot, and the eviction end of the same order.
    lru_head: u32,
    lru_tail: u32,
    leases: LeaseIndex,
    owners: OwnerIndex,
    counters: Counters,
    /// Bytes of leased rows the service may lock, a ceiling that a pin it cannot
    /// fit is skipped against; 0 leaves every lease unpinned.
    pin_budget: usize,
    /// Bytes the service currently holds locked.
    pinned_bytes: usize,
    prefetch: PrefetchLimits,
    /// The readahead a lookup started for a slot, by slot.
    readaheads: HashMap<u32, ReadAhead>,
    /// Slots a readahead locked for the request that asked for them.
    prefetch_pins: HashMap<Box<str>, Vec<u32>>,
    hashes: Option<Box<dyn PageHashes>>,
}

/// A readahead a lookup started for one slot: when it started, which the cooldown
/// measures, and the request whose load it keeps the row for while it is in
/// flight. The worker that reads the row clears the request, so a row released
/// meanwhile is not locked for a request that can no longer load it.
struct ReadAhead {
    since: Instant,
    request: Option<Box<str>>,
}

/// Monotonic per-namespace counters since `open`. `reset` leaves them alone, so a
/// reservation or lease leak stays visible after the cache itself is emptied.
#[derive(Default)]
struct Counters {
    hits: usize,
    misses: usize,
    pending_hits: usize,
    evictions: usize,
    stores: usize,
    commits: usize,
    aborted: usize,
    failed_commits: usize,
    reclaimed: usize,
    pin_skipped: usize,
    pin_skipped_headroom: usize,
    pin_failures: usize,
    prefetch_calls: usize,
    prefetch_chunks: usize,
    prefetch_skipped: usize,
}

/// Leases ended by `release` or `release_request`, and the keys or request id
/// that held none. Unknown input stays non-fatal but is reported, so a client
/// bug cannot pass as success.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Released {
    pub leases: usize,
    pub unknown: usize,
}

/// Interned names of the requests that own pending reservations, so a slot names
/// its owner in four bytes. An id keeps its name until the last reservation it
/// owns ends, so a recycled id never inherits another request's reservation.
#[derive(Default)]
struct OwnerIndex {
    ids: HashMap<Box<str>, u32>,
    names: Vec<Box<str>>,
    /// Pending reservations per id, indexed by id.
    pending: Vec<usize>,
    /// Ids whose last reservation ended, free for reuse.
    free: Vec<u32>,
}

impl OwnerIndex {
    /// Returns the id of `name`, counting one reservation against it.
    fn acquire(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            self.pending[id as usize] += 1;
            return id;
        }
        let id = match self.free.pop() {
            Some(id) => {
                self.names[id as usize] = Box::from(name);
                id
            }
            None => {
                self.names.push(Box::from(name));
                self.pending.push(0);
                u32::try_from(self.names.len() - 1).expect("interned owners fit in u32")
            }
        };
        self.pending[id as usize] = 1;
        self.ids.insert(Box::from(name), id);
        id
    }

    /// Returns the id of `name` while it owns any pending reservation.
    fn get(&self, name: &str) -> Option<u32> {
        self.ids.get(name).copied()
    }

    fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    /// Drops one reservation reference, recycling the id with the last one.
    fn release(&mut self, id: u32) {
        self.pending[id as usize] -= 1;
        if self.pending[id as usize] == 0 {
            self.ids.remove(&*self.names[id as usize]);
            self.free.push(id);
        }
    }
}

/// Leases held by in-flight requests, addressed by interned request ids so a
/// slot's reader list stores `u32` ids instead of a request name per hit.
#[derive(Default)]
struct LeaseIndex {
    ids: HashMap<Box<str>, u32>,
    /// Slots each id leases, indexed by id.
    slots: Vec<Vec<u32>>,
    /// Ids of finished requests, free for reuse.
    released: Vec<u32>,
}

impl LeaseIndex {
    /// Returns the interned id of `name`, allocating one on first use.
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = match self.released.pop() {
            Some(id) => {
                self.slots[id as usize].clear();
                id
            }
            None => {
                self.slots.push(Vec::new());
                u32::try_from(self.slots.len() - 1).expect("interned request ids fit in u32")
            }
        };
        self.ids.insert(Box::from(name), id);
        id
    }

    fn get(&self, name: &str) -> Option<u32> {
        self.ids.get(name).copied()
    }

    fn lease(&mut self, id: u32, slot: u32) {
        self.slots[id as usize].push(slot);
    }

    /// Drops one slot's lease, keeping the request's other leases.
    fn unlease(&mut self, id: u32, slot: u32) {
        self.slots[id as usize].retain(|leased| *leased != slot);
    }

    /// Ends a request's leases, returning its id and the slots it held.
    fn release(&mut self, name: &str) -> Option<(u32, Vec<u32>)> {
        let id = self.ids.remove(name)?;
        self.released.push(id);
        Some((id, std::mem::take(&mut self.slots[id as usize])))
    }

    /// Number of requests that currently hold a lease. A request that released
    /// every key keeps its interned id until `release_request`.
    fn lease_holders(&self) -> usize {
        self.slots.iter().filter(|slots| !slots.is_empty()).count()
    }
}

/// Keys, slots, and evicted keys returned by `Cache::store`.
pub type StorePlan = (Vec<String>, Vec<usize>, Vec<String>);

/// Why `Cache::store` reserved nothing for a batch.
#[derive(Debug, PartialEq, Eq)]
pub enum StoreRejection {
    /// No chunk could be freed: every one is leased, reserved, or in the batch.
    Full,
    /// The reply that echoes the reserved and evicted keys exceeds one frame.
    Unanswerable,
}

fn invalid(message: &str) -> io::Error {
    io::Error::other(message)
}

pub fn check_keys(keys: &[String]) -> io::Result<()> {
    if keys.len() > MAX_KEYS || keys.iter().any(|key| !is_valid_key(key)) {
        return Err(invalid("invalid or duplicate cache keys"));
    }
    // A single key cannot repeat, so only a batch needs a duplicate check.
    if keys.len() > 1 {
        let mut seen = HashSet::with_capacity(keys.len());
        if keys.iter().any(|key| !seen.insert(key.as_str())) {
            return Err(invalid("invalid or duplicate cache keys"));
        }
    }
    Ok(())
}

fn is_valid_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 512
        && key.len().is_multiple_of(2)
        && key.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn check_request(request: &str) -> io::Result<()> {
    if request.is_empty() || request.len() > 256 {
        return Err(invalid("invalid request identifier"));
    }
    Ok(())
}

/// Bytes one namespace with this geometry reserves: one padded row per chunk.
pub fn namespace_bytes(
    worker_bytes: usize,
    world_size: usize,
    num_chunks: usize,
) -> io::Result<usize> {
    if worker_bytes == 0
        || world_size == 0
        || world_size > 64
        || num_chunks == 0
        || num_chunks > NO_SLOT as usize
    {
        return Err(invalid("invalid cache geometry"));
    }
    padded_row_bytes(worker_bytes, world_size, page_size()?)?
        .checked_mul(num_chunks)
        .ok_or_else(|| invalid("cache geometry overflow"))
}

fn page_size() -> io::Result<usize> {
    // SAFETY: sysconf has no pointer arguments; the result is validated.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return Err(invalid("cannot determine page size"));
    }
    Ok(page as usize)
}

fn padded_row_bytes(worker_bytes: usize, world_size: usize, page: usize) -> io::Result<usize> {
    worker_bytes
        .checked_mul(world_size)
        .and_then(|bytes| bytes.checked_add(page - 1))
        .map(|bytes| bytes / page * page)
        .ok_or_else(|| invalid("cache geometry overflow"))
}

impl Cache {
    pub fn new(
        owner: u64,
        worker_bytes: usize,
        world_size: usize,
        num_chunks: usize,
        budget: usize,
        pin_budget: usize,
        prefetch: PrefetchLimits,
    ) -> io::Result<Self> {
        Self::with_hashes(
            owner,
            worker_bytes,
            world_size,
            num_chunks,
            budget,
            pin_budget,
            prefetch,
            None,
        )
    }

    pub fn with_hashes(
        owner: u64,
        worker_bytes: usize,
        world_size: usize,
        num_chunks: usize,
        budget: usize,
        pin_budget: usize,
        prefetch: PrefetchLimits,
        factory: Option<&dyn HashFactory>,
    ) -> io::Result<Self> {
        let size = namespace_bytes(worker_bytes, world_size, num_chunks)?;
        if size > budget || size > isize::MAX as usize {
            return Err(invalid("cache exceeds service capacity"));
        }
        let page = page_size()?;
        let row_bytes = padded_row_bytes(worker_bytes, world_size, page)?;
        let name = CString::new("vllm-kv-cache").expect("literal contains no NUL");
        let flags = libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING;
        // SAFETY: name is NUL-terminated; the returned descriptor is owned below.
        let fd = unsafe { libc::memfd_create(name.as_ptr(), flags) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: memfd_create returned a new descriptor, owned only here.
        let file = unsafe { File::from_raw_fd(fd) };
        file.set_len(size as u64)?;
        // Prevent clients from resizing the mapping beneath active DMA.
        // SAFETY: fd remains live and the seal argument is a valid bitmask.
        if unsafe {
            libc::fcntl(
                fd,
                libc::F_ADD_SEALS,
                libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL,
            )
        } < 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the sealed file length covers the mapping. The service never
        // reads/writes these bytes; leased workers access disjoint rank slices.
        let mapping = unsafe { MmapOptions::new().len(size).map_mut(&file)? };
        let hashes = if let Some(factory) = factory {
            let pages_per_chunk =
                row_bytes.checked_div(page).ok_or_else(|| invalid("cache geometry overflow"))?;
            let num_entries = num_chunks
                .checked_mul(pages_per_chunk)
                .ok_or_else(|| invalid("cache geometry overflow"))?;
            Some(factory.open(HashGeometry {
                data_base_va: mapping.as_ptr() as u64,
                page_size: page,
                num_entries,
                pages_per_chunk,
            })?)
        } else {
            None
        };
        Ok(Self {
            owner,
            worker_bytes,
            world_size,
            row_bytes,
            page_size: page,
            num_chunks,
            file,
            mapping: Arc::new(mapping),
            slots: (0..num_chunks).map(|_| Slot::idle()).collect(),
            index: HashMap::new(),
            free: (0..num_chunks as u32).rev().collect(),
            lru_head: NO_SLOT,
            lru_tail: NO_SLOT,
            leases: LeaseIndex::default(),
            owners: OwnerIndex::default(),
            counters: Counters::default(),
            pin_budget,
            pinned_bytes: 0,
            prefetch,
            readaheads: HashMap::new(),
            prefetch_pins: HashMap::new(),
            hashes,
        })
    }

    /// Clears all published per-page hashes.
    pub fn invalidate_hashes(&mut self) {
        if let Some(hashes) = &mut self.hashes {
            hashes.reset();
        }
    }

    /// Looks up one key, leasing it and promoting it on a hit.
    ///
    /// `pending` covers both a chunk that is reserved but not published and one
    /// whose pages the kernel no longer holds in DRAM: neither is readable now,
    /// so neither takes a lease the caller could load through. A chunk becomes
    /// readable once a worker wrote its row and the kernel has it back in memory.
    pub fn lookup(&mut self, key: &str, request: &str) -> Status {
        let Some(&slot) = self.index.get(key) else {
            self.counters.misses += 1;
            return Status::Miss;
        };
        if !self.slots[slot as usize].ready || !self.is_resident(slot as usize) {
            self.counters.pending_hits += 1;
            return Status::Pending;
        }
        self.counters.hits += 1;
        // Lookup itself takes the read lease, closing the lookup/load race.
        let request = self.leases.intern(request);
        if !self.slots[slot as usize].readers.contains(&request) {
            self.slots[slot as usize].readers.push(request);
            self.leases.lease(request, slot);
        }
        // The row was just found resident, so locking it faults nothing back in
        // and keeps a later reclaim from taking it away from this lease.
        self.pin_slot(slot as usize);
        self.promote(slot);
        Status::Hit
    }

    /// Looks up one request's keys in order, with the status `lookup` reports
    /// for each and the same lease and LRU effect.
    ///
    /// With `until_miss` the scan ends at the first key that is not a hit, so it
    /// leases only the chunks it consumes and reports one status per key up to
    /// and including that miss.
    pub fn lookup_many(&mut self, keys: &[String], request: &str, until_miss: bool) -> Vec<Status> {
        let mut statuses = Vec::with_capacity(keys.len());
        for key in keys {
            let status = self.lookup(key, request);
            statuses.push(status);
            if until_miss && status == Status::Miss {
                break;
            }
        }
        statuses
    }

    /// Writes committed chunks out through the kernel's swap path.
    ///
    /// The digest is resolved from the VMA that carries the rswap program, and
    /// the reverse-mapping walk stops as soon as the folio is no longer mapped
    /// (`try_to_unmap()` uses `rwc.done = folio_not_mapped`), so an attached VMA
    /// without a page table entry of its own is never visited. Faulting each
    /// page in first keeps the folio mapped until this VMA is walked, which is
    /// what lets the kernel store the page under the digest the service
    /// published. Only a committed chunk may be spilled, because a page with no
    /// published digest cannot be written out at all.
    ///
    /// Args:
    ///     keys: Cache keys of committed chunks.
    ///
    /// Returns:
    ///     The number of base pages handed to the kernel.
    pub fn spill(&self, keys: &[String]) -> io::Result<usize> {
        let mut pages = 0;
        for key in keys {
            let slot = *self
                .index
                .get(key.as_str())
                .ok_or_else(|| invalid("spill requires a stored chunk"))?;
            if !self.slots[slot as usize].ready {
                return Err(invalid("spill requires a committed chunk"));
            }
            let base = self.mapping.as_ptr() as usize + slot as usize * self.row_bytes;
            for offset in (0..self.row_bytes).step_by(self.page_size) {
                // SAFETY: the slot lies inside the sealed mapping and the offset
                // is page-aligned within its row; a volatile read installs a
                // page table entry and changes no data.
                unsafe {
                    let page = std::ptr::with_exposed_provenance::<u8>(base + offset);
                    let _ = std::ptr::read_volatile(page);
                }
            }
            // SAFETY: the range is the slot's whole row inside the live mapping.
            let advised = unsafe {
                libc::madvise(
                    base as *mut libc::c_void,
                    self.row_bytes,
                    libc::MADV_PAGEOUT,
                )
            };
            if advised != 0 {
                return Err(io::Error::last_os_error());
            }
            pages += self.row_bytes / self.page_size;
        }
        Ok(pages)
    }

    /// Asks the kernel to read the chunks of `slots` back in, in one go.
    ///
    /// A swap-in is submitted per fault, and the submitting task blocks on the
    /// folio until its read finishes, so a task that faults page after page
    /// keeps exactly one request in flight. `MADV_WILLNEED` over the whole row
    /// submits the range at once, which is what lets the kernel overlap the
    /// remote round trips; adjacent chunks share one call.
    ///
    /// Args:
    ///     slots: Slots of the chunks a load is about to read.
    ///
    /// Returns:
    ///     The number of pages handed to the kernel.
    pub fn prefetch(&self, slots: &[usize]) -> io::Result<usize> {
        prefetch_rows_parallel(
            &self.mapping,
            self.row_bytes,
            self.page_size,
            slots,
            PREFETCH_STREAMS,
        )
    }

    /// The mapping and geometry a detached prefetch needs.
    pub fn prefetch_parts(&self) -> (Arc<MmapMut>, usize, usize) {
        (Arc::clone(&self.mapping), self.row_bytes, self.page_size)
    }

    /// Number of the slot's row pages the kernel holds in DRAM, as `mincore(2)`
    /// reports them. Asking touches no page, so it cannot change the answer.
    pub fn resident_pages(&self, slot: usize) -> io::Result<usize> {
        let pages = self.row_bytes / self.page_size;
        let mut residency = vec![0u8; pages];
        let base = self.row_base(slot);
        // SAFETY: the row lies inside the sealed mapping and is page-aligned, and
        // `residency` holds the one byte per page that mincore(2) writes.
        let queried = unsafe {
            libc::mincore(
                base as *mut libc::c_void,
                self.row_bytes,
                residency.as_mut_ptr(),
            )
        };
        if queried != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(residency.iter().filter(|resident| **resident & 1 != 0).count())
    }

    /// Whether the whole row is in DRAM, and so readable without waiting for a
    /// fault to fetch a page back. A range `mincore(2)` refuses to describe is
    /// not resident, because a chunk the service cannot vouch for is not a hit.
    pub fn is_resident(&self, slot: usize) -> bool {
        self.resident_pages(slot)
            .is_ok_and(|pages| pages == self.row_bytes / self.page_size)
    }

    /// Slots of the committed chunks `keys` name, skipping a key the cache does
    /// not hold or has only reserved: a reservation has no row to read ahead yet.
    pub fn prefetch_slots(&self, keys: &[&str]) -> Vec<usize> {
        keys.iter()
            .filter_map(|key| {
                let slot = *self.index.get(*key)?;
                self.slots[slot as usize].ready.then_some(slot as usize)
            })
            .collect()
    }

    /// Starts the readahead a lookup's `pending` answers asked for.
    ///
    /// A slot already locked stays as it is, and so does one whose readahead
    /// started less than the cooldown ago: the kernel may still be fetching it,
    /// and another lookup would only queue the same pages again. One call accepts
    /// at most `max_chunks` rows and counts the rest, so a wide scan cannot turn
    /// into thousands of outstanding reads. The accepted slots stay marked until
    /// [`Cache::finish_prefetch`] ends their readahead.
    ///
    /// Args:
    ///     slots: Slots of committed rows that are not resident.
    ///     request: Request whose load the readahead keeps the rows for.
    ///     now: When the readahead starts.
    ///
    /// Returns:
    ///     The slots to read ahead, in the order given.
    pub fn begin_prefetch(&mut self, slots: &[usize], request: &str, now: Instant) -> Vec<usize> {
        let mut accepted = Vec::new();
        for &slot in slots {
            if self.slots[slot].pinned {
                continue;
            }
            let cooled = self.readaheads.get(&(slot as u32)).is_some_and(|readahead| {
                now.saturating_duration_since(readahead.since) < self.prefetch.cooldown
            });
            if cooled || accepted.len() >= self.prefetch.max_chunks {
                self.counters.prefetch_skipped += 1;
                continue;
            }
            accepted.push(slot);
        }
        if accepted.is_empty() {
            return accepted;
        }
        self.counters.prefetch_calls += 1;
        self.counters.prefetch_chunks += accepted.len();
        let request = Some(Box::from(request));
        for &slot in &accepted {
            self.readaheads.insert(
                slot as u32,
                ReadAhead {
                    since: now,
                    request: request.clone(),
                },
            );
        }
        accepted
    }

    /// Ends one readahead: clears the mark this request set, and locks the row for
    /// that request once it is resident, so the load that follows reads it without
    /// a fault. A mark a release or a reset already dropped is left alone.
    ///
    /// Args:
    ///     slot: A slot `begin_prefetch` accepted.
    ///     request: The request that asked for the readahead.
    ///
    /// Returns:
    ///     Whether the row was left locked for the request.
    pub fn finish_prefetch(&mut self, slot: usize, request: &str) -> bool {
        let Some(readahead) = self.readaheads.get_mut(&(slot as u32)) else {
            return false;
        };
        if readahead.request.as_deref() != Some(request) {
            return false;
        }
        readahead.request = None;
        if !self.is_resident(slot) {
            return false;
        }
        self.pin_prefetched(slot, request)
    }

    #[allow(dead_code)]
    fn prefetch_locked(&self, slots: &[usize]) -> io::Result<usize> {
        let mut sorted = slots.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut pages = 0;
        let mut index = 0;
        while index < sorted.len() {
            let first = sorted[index];
            let mut last = first;
            while index + 1 < sorted.len() && sorted[index + 1] == last + 1 {
                index += 1;
                last = sorted[index];
            }
            let base = self.mapping.as_ptr() as usize + first * self.row_bytes;
            let len = (last - first + 1) * self.row_bytes;
            // SAFETY: the slots lie inside the sealed mapping and the range is
            // whole rows, so it is mapped and page-aligned.
            let advised =
                unsafe { libc::madvise(base as *mut libc::c_void, len, libc::MADV_WILLNEED) };
            if advised != 0 {
                return Err(io::Error::last_os_error());
            }
            pages += len / self.page_size;
            index += 1;
        }
        Ok(pages)
    }

    /// Slots of the committed chunks `keys` name, without taking a lease.
    ///
    /// A prefetch runs before the load that needs the slots, so it must not hold
    /// the chunks a request leases on the lookup that produced the keys.
    pub fn committed_slots(&self, keys: &[String]) -> io::Result<Vec<usize>> {
        keys.iter()
            .map(|key| {
                let slot = *self
                    .index
                    .get(key.as_str())
                    .ok_or_else(|| invalid("prefetch requires a stored chunk"))?;
                if !self.slots[slot as usize].ready {
                    return Err(invalid("prefetch requires a committed chunk"));
                }
                Ok(slot as usize)
            })
            .collect()
    }

    /// Slots of the chunks `request` leases, and the row of each is locked for
    /// the copy that reads it. Pinning again is free, so a repeated load of the
    /// same chunk neither counts nor locks twice.
    pub fn load(&mut self, keys: &[String], request: &str) -> io::Result<Vec<usize>> {
        let request = self.leases.get(request);
        let mut slots = Vec::with_capacity(keys.len());
        for key in keys {
            let slot = *self
                .index
                .get(key.as_str())
                .ok_or_else(|| invalid("load requires a ready leased entry"))?;
            let entry = &self.slots[slot as usize];
            let leased = request.is_some_and(|request| entry.readers.contains(&request));
            if !entry.ready || !leased {
                return Err(invalid("load requires a ready leased entry"));
            }
            slots.push(slot as usize);
        }
        // Every key is checked before any row is locked, so a rejected batch
        // leaves the locks where they were.
        for &slot in &slots {
            self.pin_slot(slot);
        }
        Ok(slots)
    }

    pub fn release(&mut self, keys: &[String], request: &str) -> Released {
        let mut released = Released::default();
        let Some(request) = self.leases.get(request) else {
            released.unknown = keys.len();
            return released;
        };
        for key in keys {
            let Some(&slot) = self.index.get(key.as_str()) else {
                released.unknown += 1;
                continue;
            };
            let entry = &mut self.slots[slot as usize];
            match entry.readers.iter().position(|reader| *reader == request) {
                Some(position) => {
                    entry.readers.swap_remove(position);
                    let unleased = entry.readers.is_empty();
                    self.leases.unlease(request, slot);
                    if unleased {
                        self.unpin_slot(slot as usize);
                    }
                    released.leases += 1;
                }
                None => released.unknown += 1,
            }
        }
        released
    }

    /// Ends one request's leases, then its prefetch pins.
    ///
    /// A readahead the request started is dropped with them, so a row it makes
    /// resident is not locked for a request that can no longer load it.
    pub fn release_request(&mut self, request: &str) -> Released {
        let mut released = Released::default();
        let mut held = false;
        if let Some((id, slots)) = self.leases.release(request) {
            held = true;
            for slot in &slots {
                let entry = &mut self.slots[*slot as usize];
                entry.readers.retain(|reader| *reader != id);
                if entry.readers.is_empty() {
                    self.unpin_slot(*slot as usize);
                }
            }
            released.leases = slots.len();
        }
        if let Some(pinned) = self.prefetch_pins.remove(request) {
            held = true;
            for slot in pinned {
                if self.slots[slot as usize].readers.is_empty() {
                    self.unpin_slot(slot as usize);
                }
            }
        }
        let readaheads = self.readaheads.len();
        self.readaheads
            .retain(|_, readahead| readahead.request.as_deref() != Some(request));
        if self.readaheads.len() != readaheads {
            held = true;
        }
        if !held {
            released.unknown = 1;
        }
        released
    }

    pub fn touch(&mut self, keys: &[String]) {
        for key in keys {
            if let Some(&slot) = self.index.get(key.as_str()) {
                self.promote(slot);
            }
        }
    }

    /// Reserves a slot for every key the cache does not already hold, evicting
    /// the least recently used chunks that make room.
    ///
    /// A cache that cannot make room ([`StoreRejection::Full`]) and a reply that
    /// would not fit one control frame ([`StoreRejection::Unanswerable`]) both
    /// leave every slot, key, and counter as they were, so a caller that cannot be
    /// answered keeps the cache it had.
    pub fn store(&mut self, keys: &[String], request: &str) -> Result<StorePlan, StoreRejection> {
        let incoming: HashSet<&str> = keys.iter().map(String::as_str).collect();
        let missing: Vec<&str> = keys
            .iter()
            .map(String::as_str)
            .filter(|key| !self.index.contains_key(*key))
            .collect();
        let needed = missing.len().saturating_sub(self.free.len());
        let victims = self.eviction_victims(needed, &incoming).ok_or(StoreRejection::Full)?;
        // The reply echoes the evicted keys too, and they were stored by an
        // earlier batch, so they may be longer than any key in this one.
        let echoed: Vec<&str> =
            victims.iter().map(|slot| &*self.slots[*slot as usize].key).collect();
        if !store_reply_fits(&missing, &echoed) {
            return Err(StoreRejection::Unanswerable);
        }
        let mut evicted = Vec::with_capacity(victims.len());
        for slot in victims {
            let key = self.slots[slot as usize].key.to_string();
            if let Some(hashes) = &mut self.hashes {
                hashes.clear(slot as usize);
            }
            self.release_slot(slot);
            evicted.push(key);
        }
        self.counters.evictions += evicted.len();
        let mut reserved = Vec::with_capacity(missing.len());
        let mut slots = Vec::with_capacity(missing.len());
        let since = Instant::now();
        for key in missing {
            let slot = self.free.pop().expect("capacity checked");
            let owner = self.owners.acquire(request);
            self.slots[slot as usize].key = Box::from(key);
            self.slots[slot as usize].ready = false;
            self.slots[slot as usize].reservation = Some(Reservation { owner, since });
            self.push_front(slot);
            self.index.insert(Box::from(key), slot);
            reserved.push(key.to_owned());
            slots.push(slot as usize);
        }
        self.counters.stores += slots.len();
        Ok((reserved, slots, evicted))
    }

    /// Publishes the reservations `keys` hold for `request`, or aborts them.
    ///
    /// A reservation belongs to the request that made it: a mismatched or failed
    /// commit leaves every reservation pending instead of freeing or publishing
    /// bytes this request never wrote.
    pub fn commit(&mut self, keys: &[String], request: &str, success: bool) -> io::Result<()> {
        let slots = match self.reserved_slots(keys, request) {
            Ok(slots) => slots,
            Err(error) => {
                self.counters.failed_commits += 1;
                return Err(error);
            }
        };
        let committed = slots.len();
        if success {
            // Publish before any entry becomes ready: a failed publish leaves
            // every reservation pending and the caller fails closed.
            if let Some(hashes) = &mut self.hashes {
                for (key, &slot) in keys.iter().zip(&slots) {
                    if let Err(error) = hashes.publish(slot as usize, key) {
                        for &slot in &slots {
                            hashes.clear(slot as usize);
                        }
                        self.counters.failed_commits += 1;
                        return Err(error);
                    }
                }
            }
            for slot in slots {
                self.publish_slot(slot);
            }
            self.counters.commits += committed;
        } else {
            for slot in slots {
                if let Some(hashes) = &mut self.hashes {
                    hashes.clear(slot as usize);
                }
                self.release_slot(slot);
            }
            self.counters.aborted += committed;
        }
        Ok(())
    }

    /// Slots `keys` are reserved for `request`, or the error that rejects the
    /// batch. Every key is checked before any state changes.
    fn reserved_slots(&self, keys: &[String], request: &str) -> io::Result<Vec<u32>> {
        let owner = self.owners.get(request);
        let mut slots = Vec::with_capacity(keys.len());
        for key in keys {
            let slot = *self
                .index
                .get(key.as_str())
                .filter(|slot| !self.slots[**slot as usize].ready)
                .ok_or_else(|| invalid("commit requires pending reservations"))?;
            let reservation = self.slots[slot as usize]
                .reservation
                .as_ref()
                .ok_or_else(|| invalid("commit does not own the reservation"))?;
            if Some(reservation.owner) != owner {
                return Err(invalid("commit does not own the reservation"));
            }
            slots.push(slot);
        }
        Ok(slots)
    }

    /// Frees pending reservations older than `timeout` as of `now`, clearing each
    /// hash before its slot. Returns one record per owning request, for logging.
    pub fn reclaim_expired(&mut self, now: Instant, timeout: Duration) -> Vec<Reclaimed> {
        let mut expired = Vec::new();
        for &slot in self.index.values() {
            let Some(reservation) = &self.slots[slot as usize].reservation else {
                continue;
            };
            let age = now.saturating_duration_since(reservation.since);
            if age >= timeout {
                expired.push((reservation.owner, slot, age));
            }
        }
        let mut reclaimed: Vec<Reclaimed> = Vec::with_capacity(expired.len());
        for (owner, slot, age) in expired {
            let request = Box::from(self.owners.name(owner));
            match reclaimed.iter_mut().find(|entry| entry.request == request) {
                Some(entry) => {
                    entry.keys += 1;
                    entry.age = entry.age.max(age);
                }
                None => reclaimed.push(Reclaimed {
                    request,
                    keys: 1,
                    age,
                }),
            }
            if let Some(hashes) = &mut self.hashes {
                hashes.clear(slot as usize);
            }
            self.release_slot(slot);
        }
        self.counters.reclaimed += reclaimed.iter().map(|entry| entry.keys).sum::<usize>();
        reclaimed
    }

    /// Empties the cache. Without `force` any pending reservation or lease keeps
    /// it until that transfer ends; with `force` every lease and reservation is
    /// dropped, which a client may only do after its own transfers are flushed.
    pub fn reset(&mut self, force: bool) -> io::Result<()> {
        if !force
            && self.index.values().any(|slot| {
                let entry = &self.slots[*slot as usize];
                !entry.ready || !entry.readers.is_empty()
            })
        {
            return Err(invalid(
                "cannot reset a cache with active transfers or leases",
            ));
        }
        if let Some(hashes) = &mut self.hashes {
            hashes.reset();
        }
        // Every reader is gone, so no row may stay locked. A slot keeps its lock
        // only if the kernel refused the unlock, and the retry here is free.
        for slot in 0..self.num_chunks {
            self.unpin_slot(slot);
        }
        self.index.clear();
        self.leases = LeaseIndex::default();
        self.owners = OwnerIndex::default();
        self.readaheads.clear();
        self.prefetch_pins.clear();
        self.lru_head = NO_SLOT;
        self.lru_tail = NO_SLOT;
        for entry in &mut self.slots {
            *entry = Slot::idle();
        }
        self.free = (0..self.num_chunks as u32).rev().collect();
        Ok(())
    }

    pub fn stats(&self) -> HashMap<&'static str, usize> {
        let mut pending = 0;
        let mut leased = 0;
        let mut resident = 0;
        for &slot in self.index.values() {
            let entry = &self.slots[slot as usize];
            pending += usize::from(!entry.ready);
            leased += usize::from(!entry.readers.is_empty());
            resident += usize::from(self.is_resident(slot as usize));
        }
        HashMap::from([
            ("entries", self.index.len()),
            ("pending", pending),
            ("leased", leased),
            ("resident", resident),
            // Entries whose pages are not all in DRAM, so reading them costs a
            // fault that fetches the page back.
            ("remote", self.index.len() - resident),
            ("free_chunks", self.free.len()),
            ("hits", self.counters.hits),
            ("misses", self.counters.misses),
            ("pending_hits", self.counters.pending_hits),
            ("evictions", self.counters.evictions),
            ("stores", self.counters.stores),
            ("commits", self.counters.commits),
            ("aborted", self.counters.aborted),
            ("failed_commits", self.counters.failed_commits),
            ("reclaimed", self.counters.reclaimed),
            ("requests", self.leases.lease_holders()),
            ("pinned_slots", self.pinned_slots()),
            ("pinned_bytes", self.pinned_bytes),
            ("pin_skipped", self.counters.pin_skipped),
            ("pin_skipped_headroom", self.counters.pin_skipped_headroom),
            ("pin_failures", self.counters.pin_failures),
            ("prefetch_calls", self.counters.prefetch_calls),
            ("prefetch_chunks", self.counters.prefetch_chunks),
            ("prefetch_skipped", self.counters.prefetch_skipped),
            ("bytes", self.mapping.len()),
            ("num_chunks", self.num_chunks),
        ])
    }

    /// Moves a slot to the head of the LRU order.
    fn promote(&mut self, slot: u32) {
        if self.lru_head != slot {
            self.unlink(slot);
            self.push_front(slot);
        }
    }

    fn push_front(&mut self, slot: u32) {
        let head = self.lru_head;
        self.slots[slot as usize].prev = NO_SLOT;
        self.slots[slot as usize].next = head;
        if head == NO_SLOT {
            self.lru_tail = slot;
        } else {
            self.slots[head as usize].prev = slot;
        }
        self.lru_head = slot;
    }

    fn unlink(&mut self, slot: u32) {
        let (prev, next) = (
            self.slots[slot as usize].prev,
            self.slots[slot as usize].next,
        );
        if prev == NO_SLOT {
            self.lru_head = next;
        } else {
            self.slots[prev as usize].next = next;
        }
        if next == NO_SLOT {
            self.lru_tail = prev;
        } else {
            self.slots[next as usize].prev = prev;
        }
        self.slots[slot as usize].prev = NO_SLOT;
        self.slots[slot as usize].next = NO_SLOT;
    }

    /// Drops a slot's key and returns it to the free list. Only a slot without
    /// leases is released, so no request's lease index needs rewriting.
    fn release_slot(&mut self, slot: u32) {
        // Eviction never picks a leased slot, but a lock must not outlive the
        // slot it is charged to, so the unlock is unconditional.
        self.unpin_slot(slot as usize);
        self.forget_prefetch(slot);
        self.unlink(slot);
        let key = std::mem::take(&mut self.slots[slot as usize].key);
        self.slots[slot as usize].ready = false;
        if let Some(reservation) = self.slots[slot as usize].reservation.take() {
            self.owners.release(reservation.owner);
        }
        debug_assert!(
            self.slots[slot as usize].readers.is_empty(),
            "a leased slot cannot be released"
        );
        self.index.remove(key.as_ref());
        self.free.push(slot);
    }

    /// Turns a pending reservation into a published entry and ends its owner's
    /// reference to the reservation.
    fn publish_slot(&mut self, slot: u32) {
        self.slots[slot as usize].ready = true;
        if let Some(reservation) = self.slots[slot as usize].reservation.take() {
            self.owners.release(reservation.owner);
        }
    }

    /// Address of the slot's row in the mapping, which is the range residency and
    /// locking cover.
    fn row_base(&self, slot: usize) -> usize {
        self.mapping.as_ptr() as usize + slot * self.row_bytes
    }

    /// Rows currently locked against reclaim.
    fn pinned_slots(&self) -> usize {
        self.slots.iter().filter(|entry| entry.pinned).count()
    }

    /// Drops what a readahead left on the slot, so a recycled slot neither stays
    /// in a request's pin list nor inside the cooldown of the row it held.
    fn forget_prefetch(&mut self, slot: u32) {
        self.readaheads.remove(&slot);
        for pinned in self.prefetch_pins.values_mut() {
            pinned.retain(|pinned_slot| *pinned_slot != slot);
        }
    }

    /// Whether one more row fits the pin budget.
    fn pin_fits_budget(&self) -> bool {
        self.pin_budget != 0 && self.pinned_bytes.saturating_add(self.row_bytes) <= self.pin_budget
    }

    /// Locks a row the readahead just made resident and charges the pin to
    /// `request`, so the release that ends the request unlocks it too.
    ///
    /// A row another request already holds locked is left to that request, and a
    /// pin the budget or the cgroup's headroom refuses leaves the row readable
    /// without the guarantee.
    fn pin_prefetched(&mut self, slot: usize, request: &str) -> bool {
        if self.slots[slot].pinned {
            return false;
        }
        if !self.pin_fits_budget() {
            if self.pin_budget != 0 {
                self.counters.pin_skipped += 1;
            }
            return false;
        }
        if !self.pin_headroom() {
            self.counters.pin_skipped_headroom += 1;
            return false;
        }
        self.pin_slot(slot);
        if !self.slots[slot].pinned {
            return false;
        }
        self.prefetch_pins.entry(Box::from(request)).or_default().push(slot as u32);
        true
    }

    /// Whether this process's cgroup v2 has room for one more locked row.
    ///
    /// A lock makes the row unreclaimable, so pinning just under `memory.max`
    /// would push the reclaim onto the rest of the cgroup and could OOM the writer
    /// instead of the cache. A tenth of the limit stays free; a cgroup without a
    /// limit, or one whose files this process cannot read, leaves the budget as the
    /// only ceiling.
    fn pin_headroom(&self) -> bool {
        let Some((current, limit)) = cgroup_memory() else {
            return true;
        };
        current.saturating_add(self.row_bytes as u64) <= limit.saturating_sub(limit / 10)
    }

    /// Locks the slot's row in DRAM, so the lease that reports it readable keeps
    /// it that way: the kernel excludes a locked range from reclaim.
    ///
    /// The budget is a ceiling, never an eviction policy: a pin that does not fit
    /// it is skipped and the caller keeps its hit without the guarantee, because
    /// unlocking a row another request still reads to make room would break that
    /// request instead. A refused `mlock` is counted and logged once, never fatal.
    fn pin_slot(&mut self, slot: usize) {
        if self.slots[slot].pinned || self.pin_budget == 0 {
            return;
        }
        if !self.pin_fits_budget() {
            self.counters.pin_skipped += 1;
            return;
        }
        // SAFETY: the row lies inside the sealed mapping, so the range is mapped
        // and page-aligned.
        let row = self.row_base(slot) as *const libc::c_void;
        let locked = unsafe { libc::mlock(row, self.row_bytes) };
        if locked != 0 {
            self.counters.pin_failures += 1;
            log_pin_failure(io::Error::last_os_error());
            return;
        }
        self.slots[slot].pinned = true;
        self.pinned_bytes += self.row_bytes;
    }

    /// Unlocks a row no request leases any more. A refused `munlock` keeps the
    /// flag and the bytes charged, because the range stays locked; a slot reused
    /// with that flag is already locked and is not charged twice.
    fn unpin_slot(&mut self, slot: usize) {
        if !self.slots[slot].pinned {
            return;
        }
        // SAFETY: the row is the range `pin_slot` locked inside the mapping.
        let row = self.row_base(slot) as *const libc::c_void;
        let unlocked = unsafe { libc::munlock(row, self.row_bytes) };
        if unlocked != 0 {
            self.counters.pin_failures += 1;
            log_pin_failure(io::Error::last_os_error());
            return;
        }
        self.slots[slot].pinned = false;
        self.pinned_bytes -= self.row_bytes;
    }

    /// Collects `needed` eviction victims from the least recently used end,
    /// skipping pending slots, leased slots, and slots the caller retains. Walks
    /// past those only as far as it must, and returns `None` when the cache
    /// cannot make room.
    fn eviction_victims(&self, needed: usize, incoming: &HashSet<&str>) -> Option<Vec<u32>> {
        let mut victims = Vec::with_capacity(needed);
        let mut slot = self.lru_tail;
        while victims.len() < needed && slot != NO_SLOT {
            let entry = &self.slots[slot as usize];
            if entry.ready && entry.readers.is_empty() && !incoming.contains(&*entry.key) {
                victims.push(slot);
            }
            slot = entry.prev;
        }
        (victims.len() == needed).then_some(victims)
    }

    /// Panics unless every chunk is either free or indexed by exactly one key and
    /// the LRU order holds every indexed slot.
    #[cfg(test)]
    fn assert_slot_accounting(&self) {
        assert_eq!(
            self.free.len() + self.index.len(),
            self.num_chunks,
            "a free or indexed slot is missing"
        );
        let mut owners = vec![0u8; self.num_chunks];
        for slot in self.free.iter().chain(self.index.values()) {
            owners[*slot as usize] += 1;
        }
        assert!(
            owners.iter().all(|count| *count == 1),
            "a slot has two owners"
        );
        for (key, slot) in &self.index {
            assert_eq!(
                &*self.slots[*slot as usize].key, &**key,
                "index and slot key differ"
            );
        }
        for &slot in self.index.values() {
            let entry = &self.slots[slot as usize];
            assert_eq!(
                entry.ready,
                entry.reservation.is_none(),
                "a pending slot has no owner or a published slot still has one"
            );
        }
        let mut listed = 0;
        let mut slot = self.lru_head;
        while slot != NO_SLOT {
            listed += 1;
            assert!(listed <= self.num_chunks, "the LRU order loops");
            slot = self.slots[slot as usize].next;
        }
        assert_eq!(
            listed,
            self.index.len(),
            "an indexed slot is missing from the LRU order"
        );
    }
}

/// Reports a refused `mlock` or `munlock` once per process: a host without the
/// privilege to lock memory refuses every row, and one line names the cause while
/// `stats` counts the rest.
fn log_pin_failure(error: io::Error) {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if !REPORTED.swap(true, Ordering::Relaxed) {
        log_event!("pin_failed", "error" = error);
    }
}

/// Bytes the process's own cgroup v2 charges to `memory.current` against
/// `memory.max`.
///
/// Returns:
///     `None` when the process is in no v2 cgroup, either file is unreadable, or
///     the limit is `max`; the caller then has no headroom to check.
fn cgroup_memory() -> Option<(u64, u64)> {
    let cgroup = fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?;
    let path = path.trim_end_matches('/');
    let read = |name: &str| -> Option<u64> {
        fs::read_to_string(format!("/sys/fs/cgroup{path}/{name}"))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    Some((read("memory.current")?, read("memory.max")?))
}

/// Asks the kernel to read whole rows back in with one `MADV_WILLNEED` each.
pub fn prefetch_rows(
    mapping: &MmapMut,
    row_bytes: usize,
    page_size: usize,
    slots: &[usize],
) -> io::Result<usize> {
    prefetch_rows_parallel(mapping, row_bytes, page_size, slots, 1)
}

/// How many `MADV_WILLNEED` streams one prefetch drives at once.
///
/// `madvise(MADV_WILLNEED)` on a shmem mapping ends in `shmem_swapin_range()`,
/// which stages the whole range into one `swap_io_ctx` and submits it once
/// ("do not wait for completion"). One call is therefore one stream, and a
/// thread that issues the row ranges one after another keeps a single stream in
/// flight however many batches the kernel would allow; measured against a bound
/// Redis peer that is one 120 KiB request per 20-35 ms, about 7 MB/s. Driving
/// several rows from several threads is what fills the kernel's read slots.
pub const PREFETCH_STREAMS: usize = 4;

/// Same as [`prefetch_rows`], with `streams` rows in flight at once.
pub fn prefetch_rows_parallel(
    mapping: &MmapMut,
    row_bytes: usize,
    page_size: usize,
    slots: &[usize],
    streams: usize,
) -> io::Result<usize> {
    let mut sorted = slots.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    // Merge adjacent slots: one call covers a contiguous run of rows.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for slot in sorted {
        match runs.last_mut() {
            Some((_, last)) if *last + 1 == slot => *last = slot,
            _ => runs.push((slot, slot)),
        }
    }
    let streams = streams.clamp(1, PREFETCH_STREAMS).min(runs.len().max(1));
    let mut pages = 0;
    let mut first_error: Option<io::Error> = None;
    if streams <= 1 {
        for (first, last) in runs {
            match advise_rows(mapping, row_bytes, page_size, first, last) {
                Ok(advised) => pages += advised,
                Err(error) => {
                    first_error = Some(error);
                    break;
                }
            }
        }
    } else {
        // Every stream takes every `streams`-th run, so the calls of different
        // streams never describe the same range.
        let mapping_ref = mapping;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(streams);
            for lane in 0..streams {
                let lane_runs: Vec<(usize, usize)> =
                    runs.iter().copied().skip(lane).step_by(streams).collect();
                if lane_runs.is_empty() {
                    continue;
                }
                handles.push(scope.spawn(move || {
                    let mut lane_pages = 0;
                    for (first, last) in lane_runs {
                        match advise_rows(mapping_ref, row_bytes, page_size, first, last) {
                            Ok(advised) => lane_pages += advised,
                            Err(error) => return (lane_pages, Some(error)),
                        }
                    }
                    (lane_pages, None)
                }));
            }
            let mut lane_pages_total = 0;
            let mut lane_error: Option<io::Error> = None;
            for handle in handles {
                match handle.join() {
                    Ok((lane_pages, error)) => {
                        lane_pages_total += lane_pages;
                        if lane_error.is_none() {
                            lane_error = error;
                        }
                    }
                    Err(_) => {
                        if lane_error.is_none() {
                            lane_error = Some(invalid("prefetch panicked"));
                        }
                    }
                }
            }
            pages += lane_pages_total;
            if lane_error.is_some() {
                first_error = lane_error;
            }
        });
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    Ok(pages)
}

/// One `MADV_WILLNEED` over rows `first..=last`.
fn advise_rows(
    mapping: &MmapMut,
    row_bytes: usize,
    page_size: usize,
    first: usize,
    last: usize,
) -> io::Result<usize> {
    let base = mapping.as_ptr() as usize + first * row_bytes;
    let len = (last - first + 1) * row_bytes;
    // SAFETY: the slots lie inside the sealed mapping and the range is whole
    // rows, so it is mapped and page-aligned.
    let advised = unsafe { libc::madvise(base as *mut libc::c_void, len, libc::MADV_WILLNEED) };
    if advised != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(len / page_size)
}

impl Drop for Cache {
    fn drop(&mut self) {
        self.invalidate_hashes();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Op {
        Publish(usize, String),
        Clear(usize),
        Reset,
    }

    struct RecordingFactory {
        ops: Arc<Mutex<Vec<Op>>>,
        fail_publish: bool,
    }

    impl HashFactory for RecordingFactory {
        fn open(&self, _geometry: HashGeometry) -> io::Result<Box<dyn PageHashes>> {
            Ok(Box::new(RecordingHashes {
                ops: Arc::clone(&self.ops),
                fail_publish: self.fail_publish,
            }))
        }
    }

    struct RecordingHashes {
        ops: Arc<Mutex<Vec<Op>>>,
        fail_publish: bool,
    }

    impl PageHashes for RecordingHashes {
        fn publish(&mut self, slot: usize, key: &str) -> io::Result<()> {
            self.ops.lock().unwrap().push(Op::Publish(slot, key.to_owned()));
            if self.fail_publish {
                return Err(io::Error::other("publish failed"));
            }
            Ok(())
        }

        fn clear(&mut self, slot: usize) {
            self.ops.lock().unwrap().push(Op::Clear(slot));
        }

        fn reset(&mut self) {
            self.ops.lock().unwrap().push(Op::Reset);
        }
    }

    fn cache() -> Cache {
        cache_with(2)
    }

    fn cache_with(num_chunks: usize) -> Cache {
        cache_with_pins(num_chunks, 0)
    }

    fn cache_with_pins(num_chunks: usize, pin_budget: usize) -> Cache {
        let limits = PrefetchLimits::default();
        Cache::new(1, 64, 1, num_chunks, 16 * 1024 * 1024, pin_budget, limits).unwrap()
    }

    fn cache_with_prefetch(num_chunks: usize, prefetch: PrefetchLimits) -> Cache {
        Cache::new(1, 64, 1, num_chunks, 16 * 1024 * 1024, 0, prefetch).unwrap()
    }

    fn keys(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }

    fn slot(cache: &Cache, key: &str) -> usize {
        cache.index.get(key).map(|slot| *slot as usize).expect("key is indexed")
    }

    fn ready(cache: &Cache, key: &str) -> bool {
        cache.slots[slot(cache, key)].ready
    }

    fn recording_cache(fail_publish: bool) -> (Cache, Arc<Mutex<Vec<Op>>>) {
        let ops = Arc::new(Mutex::new(Vec::new()));
        let factory = RecordingFactory {
            ops: Arc::clone(&ops),
            fail_publish,
        };
        let cache = Cache::with_hashes(
            1,
            64,
            1,
            2,
            2 * 1024 * 1024,
            0,
            PrefetchLimits::default(),
            Some(&factory as &dyn HashFactory),
        )
        .unwrap();
        (cache, ops)
    }

    fn recorded(ops: &Arc<Mutex<Vec<Op>>>) -> Vec<Op> {
        ops.lock().unwrap().clone()
    }

    #[test]
    fn prefetch_covers_every_page_of_the_given_slots() {
        let mut cache = cache_with(4);
        let first = keys(&["aa", "bb", "cc"]);
        cache.store(&first, "r").unwrap();
        cache.commit(&first, "r", true).unwrap();
        let slots: Vec<usize> = first
            .iter()
            .map(|key| cache.index.get(key.as_str()).copied().unwrap() as usize)
            .collect();
        // Adjacent slots are merged into one call, so two runs here.
        assert_eq!(
            cache.prefetch(&slots).unwrap(),
            3 * cache.row_bytes / cache.page_size
        );
        assert_eq!(cache.prefetch(&[]).unwrap(), 0);
    }

    #[test]
    fn spill_requires_a_committed_chunk() {
        let mut cache = cache();
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        // Reserved but not published: a page with no digest cannot be written out.
        assert!(cache.spill(&first).is_err());
        cache.commit(&first, "r", true).unwrap();
        // One page per row of this geometry, and a missing key is refused.
        assert_eq!(
            cache.spill(&first).unwrap(),
            cache.row_bytes / cache.page_size
        );
        assert!(cache.spill(&keys(&["bb"])).is_err());
    }

    /// Faults every page of `key`'s row in, as a worker's DMA does before the
    /// service accepts the chunk as readable. Writing back the byte the page
    /// already holds installs a page table entry and changes no data.
    fn touch(cache: &Cache, key: &str) {
        let base = cache.mapping.as_ptr() as usize + slot(cache, key) * cache.row_bytes;
        for offset in (0..cache.row_bytes).step_by(cache.page_size) {
            // SAFETY: the page lies inside the sealed mapping.
            unsafe {
                let page = std::ptr::with_exposed_provenance_mut::<u8>(base + offset);
                std::ptr::write_volatile(page, std::ptr::read_volatile(page));
            }
        }
    }

    fn touch_all(cache: &Cache, values: &[String]) {
        for value in values {
            touch(cache, value);
        }
    }

    #[test]
    fn a_committed_chunk_is_pending_until_its_pages_are_in_dram() {
        let mut cache = cache();
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        cache.commit(&first, "r", true).unwrap();

        // A fresh mapping holds no page, so the committed chunk is not readable.
        let first_slot = slot(&cache, "aa");
        assert_eq!(cache.resident_pages(first_slot).unwrap(), 0);
        assert!(!cache.is_resident(first_slot));
        assert_eq!(cache.lookup("aa", "r"), Status::Pending);
        assert_eq!(cache.stats()["pending_hits"], 1);
        assert_eq!(cache.stats()["hits"], 0);
        // A chunk that cannot be read yet is not a hit, so it holds no lease.
        assert_eq!(cache.stats()["leased"], 0);
        assert!(cache.load(&first, "r").is_err());
        assert_eq!(cache.stats()["resident"], 0);
        assert_eq!(cache.stats()["remote"], 1);

        touch(&cache, "aa");
        assert_eq!(
            cache.resident_pages(first_slot).unwrap(),
            cache.row_bytes / cache.page_size
        );
        assert!(cache.is_resident(first_slot));
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        assert_eq!(cache.stats()["hits"], 1);
        assert_eq!(cache.stats()["leased"], 1);
        assert_eq!(cache.load(&first, "r").unwrap(), vec![first_slot]);
        assert_eq!(cache.stats()["resident"], 1);
        assert_eq!(cache.stats()["remote"], 0);
    }

    /// Stores, publishes and faults in one key, so a lookup of it is a hit.
    fn readable(cache: &mut Cache, key: &str) {
        let batch = keys(&[key]);
        cache.store(&batch, "writer").unwrap();
        cache.commit(&batch, "writer", true).unwrap();
        touch(cache, key);
    }

    #[test]
    fn a_hit_pins_its_row_until_the_lease_ends() {
        let mut cache = cache_with_pins(2, 8 * page_size().unwrap());
        readable(&mut cache, "aa");
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        // A host that refuses `mlock` (no `CAP_IPC_LOCK`, or a `RLIMIT_MEMLOCK`
        // smaller than one row) pins nothing and counts the refusal, so the two
        // account for the one pin the hit asked for either way.
        let stats = cache.stats();
        assert_eq!(stats["pinned_slots"] + stats["pin_failures"], 1);
        assert_eq!(
            stats["pinned_bytes"],
            stats["pinned_slots"] * cache.row_bytes
        );
        // A second load of the same chunk is not a second lock.
        assert_eq!(cache.load(&keys(&["aa"]), "r").unwrap().len(), 1);
        assert_eq!(cache.stats()["pinned_bytes"], stats["pinned_bytes"]);

        cache.release_request("r");
        assert_eq!(cache.stats()["pinned_slots"], 0);
        assert_eq!(cache.stats()["pinned_bytes"], 0);
    }

    #[test]
    fn a_pin_budget_of_one_row_leaves_the_second_lease_unpinned() {
        let mut cache = cache_with_pins(2, page_size().unwrap());
        let both = keys(&["aa", "bb"]);
        cache.store(&both, "writer").unwrap();
        cache.commit(&both, "writer", true).unwrap();
        touch_all(&cache, &both);

        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        let pinned_one = cache.stats()["pinned_slots"] == 1;
        assert_eq!(cache.lookup("bb", "r"), Status::Hit);
        let stats = cache.stats();
        // The budget changes the guarantee, not the answer: both keys are hits
        // and both keep their lease without the second row being unlocked.
        assert_eq!(stats["hits"], 2);
        assert_eq!(stats["leased"], 2);
        if pinned_one {
            assert_eq!(stats["pinned_slots"], 1);
            assert_eq!(stats["pinned_bytes"], cache.row_bytes);
            assert_eq!(stats["pin_skipped"], 1);
        } else {
            // `mlock` refused on this host, so both pins failed instead.
            assert_eq!(stats["pin_failures"], 2);
            assert_eq!(stats["pin_skipped"], 0);
            assert_eq!(stats["pinned_bytes"], 0);
        }
    }

    #[test]
    fn begin_prefetch_caps_a_lookup_at_the_configured_chunks() {
        let prefetch = PrefetchLimits {
            max_chunks: 2,
            ..Default::default()
        };
        let mut cache = cache_with_prefetch(4, prefetch);
        let stored = keys(&["aa", "bb", "cc"]);
        cache.store(&stored, "r").unwrap();
        cache.commit(&stored, "r", true).unwrap();

        // A key the cache does not hold has no row to read ahead, and the cap
        // keeps the third committed row for a later lookup.
        assert_eq!(
            cache.prefetch_slots(&["aa", "zz"]),
            vec![slot(&cache, "aa")]
        );
        let slots = cache.prefetch_slots(&["aa", "bb", "cc"]);
        assert_eq!(cache.begin_prefetch(&slots, "r", Instant::now()).len(), 2);
        let stats = cache.stats();
        assert_eq!(stats["prefetch_calls"], 1);
        assert_eq!(stats["prefetch_chunks"], 2);
        assert_eq!(stats["prefetch_skipped"], 1);

        // The two accepted slots are inside their cooldown, so the next lookup
        // takes the row the cap left behind and nothing else.
        assert_eq!(cache.begin_prefetch(&slots, "r", Instant::now()).len(), 1);
        assert_eq!(cache.stats()["prefetch_chunks"], 3);
        assert_eq!(cache.stats()["prefetch_skipped"], 3);
    }

    #[test]
    fn a_readahead_inside_the_cooldown_is_not_started_again() {
        let mut cache = cache_with(2);
        let stored = keys(&["aa"]);
        cache.store(&stored, "r").unwrap();
        cache.commit(&stored, "r", true).unwrap();
        let slots = cache.prefetch_slots(&["aa"]);
        let now = Instant::now();

        assert_eq!(cache.begin_prefetch(&slots, "r", now), slots);
        assert!(cache.begin_prefetch(&slots, "r", now).is_empty());
        assert!(cache.begin_prefetch(&slots, "r", now + Duration::from_millis(500)).is_empty());
        assert_eq!(cache.stats()["prefetch_skipped"], 2);
        assert_eq!(cache.stats()["prefetch_calls"], 1);

        let later = now + Duration::from_secs(5);
        assert_eq!(cache.begin_prefetch(&slots, "r", later), slots);
        assert_eq!(cache.stats()["prefetch_calls"], 2);
        assert_eq!(cache.stats()["prefetch_chunks"], 2);
    }

    #[test]
    fn a_pinned_row_is_not_prefetched_again() {
        let mut cache = cache_with_pins(2, 8 * page_size().unwrap());
        readable(&mut cache, "aa");
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        if cache.stats()["pinned_slots"] == 0 {
            // The host refused `mlock`, so no row is locked and every row stays a
            // readahead candidate.
            return;
        }
        assert!(cache.begin_prefetch(&[slot(&cache, "aa")], "r", Instant::now()).is_empty());
        assert_eq!(cache.stats()["prefetch_calls"], 0);
    }

    #[test]
    fn release_request_drops_the_prefetch_pins_of_its_rows() {
        let mut cache = cache_with_pins(2, page_size().unwrap());
        readable(&mut cache, "aa");
        let slots = cache.prefetch_slots(&["aa"]);
        assert_eq!(cache.begin_prefetch(&slots, "r", Instant::now()), slots);
        assert!(cache.finish_prefetch(slots[0], "r"));
        // The budget covers the row, so the readahead either locked it or the
        // host or the cgroup's headroom refused that one pin.
        let stats = cache.stats();
        assert_eq!(
            stats["pinned_slots"] + stats["pin_failures"] + stats["pin_skipped_headroom"],
            1
        );

        assert_eq!(
            cache.release_request("r"),
            Released {
                leases: 0,
                unknown: 0,
            }
        );
        assert_eq!(cache.stats()["pinned_slots"], 0);
        assert_eq!(cache.stats()["pinned_bytes"], 0);
        assert!(cache.prefetch_pins.is_empty());
    }

    #[test]
    fn release_request_drops_a_readahead_that_is_still_in_flight() {
        let mut cache = cache_with_pins(2, page_size().unwrap());
        readable(&mut cache, "aa");
        let slots = cache.prefetch_slots(&["aa"]);
        assert_eq!(cache.begin_prefetch(&slots, "r", Instant::now()), slots);

        // The request ends before the worker read or locked the row, so nothing
        // is charged to a request that can no longer load it.
        assert_eq!(
            cache.release_request("r"),
            Released {
                leases: 0,
                unknown: 0,
            }
        );
        assert!(!cache.finish_prefetch(slots[0], "r"));
        assert_eq!(cache.stats()["pinned_slots"], 0);
    }

    #[test]
    fn a_row_stays_pinned_while_another_request_leases_it() {
        let mut cache = cache_with_pins(2, 8 * page_size().unwrap());
        readable(&mut cache, "aa");
        assert_eq!(cache.lookup("aa", "a"), Status::Hit);
        assert_eq!(cache.lookup("aa", "b"), Status::Hit);
        let pinned = cache.stats()["pinned_bytes"];
        // The first release ends one lease, not the lock the other still reads.
        assert_eq!(cache.release(&keys(&["aa"]), "a").leases, 1);
        assert_eq!(cache.stats()["pinned_bytes"], pinned);
        assert_eq!(cache.release(&keys(&["aa"]), "b").leases, 1);
        assert_eq!(cache.stats()["pinned_bytes"], 0);
    }

    #[test]
    fn forced_reset_unpins_every_row() {
        let mut cache = cache_with_pins(2, 8 * page_size().unwrap());
        let both = keys(&["aa", "bb"]);
        cache.store(&both, "writer").unwrap();
        cache.commit(&both, "writer", true).unwrap();
        touch_all(&cache, &both);
        assert_eq!(cache.lookup("aa", "a"), Status::Hit);
        assert_eq!(cache.lookup("bb", "a"), Status::Hit);
        assert!(cache.reset(false).is_err());

        cache.reset(true).unwrap();
        assert_eq!(cache.stats()["entries"], 0);
        assert_eq!(cache.stats()["pinned_slots"], 0);
        assert_eq!(cache.stats()["pinned_bytes"], 0);
    }

    #[test]
    fn lookup_lease_prevents_eviction_before_load() {
        let mut cache = cache();
        let first = keys(&["aa", "bb"]);
        cache.store(&first, "r").unwrap();
        assert_eq!(cache.lookup("aa", "r"), Status::Pending);
        cache.commit(&first, "r", true).unwrap();
        touch(&cache, "aa");
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        let (_, _, evicted) = cache.store(&keys(&["cc"]), "r").unwrap();
        assert_eq!(evicted, keys(&["bb"]));
        assert_eq!(cache.load(&keys(&["aa"]), "r").unwrap(), vec![0]);
        assert_eq!(cache.store(&keys(&["dd"]), "r"), Err(StoreRejection::Full));
        cache.release_request("r");
        assert!(cache.store(&keys(&["dd"]), "r").is_ok());
    }

    #[test]
    fn failed_store_is_never_published() {
        let mut cache = cache();
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        cache.commit(&first, "r", false).unwrap();
        assert_eq!(cache.lookup("aa", "r"), Status::Miss);
        assert!(cache.commit(&first, "r", true).is_err());
    }

    #[test]
    fn reservation_is_atomic_and_cannot_evict_itself() {
        let mut cache = cache();
        let first = keys(&["aa", "bb"]);
        cache.store(&first, "r").unwrap();
        cache.commit(&first, "r", true).unwrap();
        assert_eq!(
            cache.store(&keys(&["aa", "bb", "cc"]), "r"),
            Err(StoreRejection::Full)
        );
        touch(&cache, "aa");
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        assert!(cache.reset(false).is_err());
    }

    #[test]
    fn geometry_and_keys_are_bounded() {
        assert!(Cache::new(1, usize::MAX, 2, 1, 8192, 0, PrefetchLimits::default()).is_err());
        assert!(check_keys(&keys(&["aa", "aa"])).is_err());
        assert!(check_keys(&keys(&["not-hex"])).is_err());
    }

    #[test]
    fn namespace_bytes_pads_every_row_to_a_page() {
        let page = page_size().unwrap();
        assert_eq!(namespace_bytes(1, 1, 1).unwrap(), page);
        assert_eq!(namespace_bytes(page, 2, 3).unwrap(), 6 * page);
        assert!(namespace_bytes(0, 1, 1).is_err());
        assert!(namespace_bytes(1, 65, 1).is_err());
        assert!(namespace_bytes(1, 1, 0).is_err());
        assert_eq!(namespace_bytes(64, 1, 2).unwrap(), cache().mapping.len());
    }

    #[test]
    fn default_cache_stats_have_no_hash_keys() {
        let stats = cache().stats();
        assert!(!stats.contains_key("hash_entries"));
        assert!(!stats.contains_key("hash_ready"));
        assert!(!stats.contains_key("metadata_entries"));
        assert!(!stats.contains_key("metadata_ready"));
    }

    #[test]
    fn store_does_not_publish_and_commit_publishes_once() {
        let (mut cache, ops) = recording_cache(false);
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        assert!(recorded(&ops).is_empty());
        let slot = slot(&cache, "aa");
        assert!(!ready(&cache, "aa"));
        cache.commit(&first, "r", true).unwrap();
        assert_eq!(recorded(&ops), vec![Op::Publish(slot, "aa".into())]);
        assert!(ready(&cache, "aa"));
    }

    #[test]
    fn failed_publish_keeps_entry_pending_and_returns_error() {
        let (mut cache, ops) = recording_cache(true);
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        let slot = slot(&cache, "aa");
        let error = cache.commit(&first, "r", true).unwrap_err();
        assert!(error.to_string().contains("publish failed"));
        assert_eq!(
            recorded(&ops),
            vec![Op::Publish(slot, "aa".into()), Op::Clear(slot)]
        );
        assert!(!ready(&cache, "aa"));
        assert_eq!(cache.lookup("aa", "r"), Status::Pending);
    }

    #[test]
    fn failed_publish_clears_every_slot_in_the_batch() {
        let (mut cache, ops) = recording_cache(true);
        let first = keys(&["aa", "bb"]);
        cache.store(&first, "r").unwrap();
        let slots: Vec<_> = first.iter().map(|key| slot(&cache, key)).collect();
        assert!(cache.commit(&first, "r", true).is_err());
        assert_eq!(
            recorded(&ops),
            vec![
                Op::Publish(slots[0], "aa".into()),
                Op::Clear(slots[0]),
                Op::Clear(slots[1]),
            ]
        );
        assert!(first.iter().all(|key| !ready(&cache, key)));
    }

    #[test]
    fn failed_commit_and_eviction_clear_before_releasing_slots() {
        let (mut cache, ops) = recording_cache(false);
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        cache.commit(&first, "r", false).unwrap();
        assert_eq!(recorded(&ops), vec![Op::Clear(0)]);

        let both = keys(&["aa", "bb"]);
        cache.store(&both, "r").unwrap();
        cache.commit(&both, "r", true).unwrap();
        let victim_slot = slot(&cache, "aa");
        ops.lock().unwrap().clear();
        let (_, _, evicted) = cache.store(&keys(&["cc"]), "r").unwrap();
        assert_eq!(evicted, keys(&["aa"]));
        assert_eq!(recorded(&ops), vec![Op::Clear(victim_slot)]);
    }

    #[test]
    fn reset_and_invalidate_hashes_call_reset() {
        let (mut cache, ops) = recording_cache(false);
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        cache.commit(&first, "r", true).unwrap();
        ops.lock().unwrap().clear();
        cache.reset(false).unwrap();
        cache.invalidate_hashes();
        assert_eq!(recorded(&ops), vec![Op::Reset, Op::Reset]);
    }

    #[test]
    fn touch_lookup_and_load_do_not_publish_hashes() {
        let (mut cache, ops) = recording_cache(false);
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        cache.commit(&first, "r", true).unwrap();
        let before = recorded(&ops);
        let slot = slot(&cache, "aa");
        touch(&cache, "aa");
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        assert_eq!(cache.load(&keys(&["aa"]), "r").unwrap(), vec![slot]);
        cache.touch(&first);
        assert_eq!(recorded(&ops), before);
    }

    #[test]
    fn eviction_follows_recency_and_lookup_promotes() {
        let mut cache = cache_with(3);
        let stored = keys(&["aa", "bb", "cc"]);
        cache.store(&stored, "r").unwrap();
        cache.commit(&stored, "r", true).unwrap();
        touch(&cache, "aa");
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        // `bb` is the least recently used key, not the oldest inserted one.
        assert_eq!(cache.store(&keys(&["dd"]), "r").unwrap().2, keys(&["bb"]));
        cache.release_request("r");
        cache.touch(&keys(&["cc"]));
        assert_eq!(cache.store(&keys(&["ee"]), "r").unwrap().2, keys(&["aa"]));
        cache.assert_slot_accounting();
    }

    /// Keys in LRU order, most recently used first.
    fn lru_order(cache: &Cache) -> Vec<String> {
        let mut order = Vec::new();
        let mut slot = cache.lru_head;
        while slot != NO_SLOT {
            order.push(cache.slots[slot as usize].key.to_string());
            slot = cache.slots[slot as usize].next;
        }
        order
    }

    #[test]
    fn lookup_many_matches_one_lookup_per_key() {
        // `aa` and `ee` are published, `bb` and `cc` are reserved but
        // uncommitted, and `dd` was never stored.
        let stored = keys(&["aa", "cc", "ee"]);
        let mixed = keys(&["ee", "dd", "aa", "bb"]);
        let mut batched = cache_with(4);
        let mut per_key = cache_with(4);
        for cache in [&mut batched, &mut per_key] {
            cache.store(&stored, "r").unwrap();
            cache.commit(&keys(&["aa", "ee"]), "r", true).unwrap();
            cache.store(&keys(&["bb"]), "r").unwrap();
            touch(cache, "aa");
            touch(cache, "ee");
        }

        let statuses = batched.lookup_many(&mixed, "r", false);
        let expected: Vec<Status> = mixed.iter().map(|key| per_key.lookup(key, "r")).collect();

        assert_eq!(statuses, expected);
        assert_eq!(
            statuses,
            vec![Status::Hit, Status::Miss, Status::Hit, Status::Pending]
        );
        // Same leases: the request that batched the lookups loads the same
        // hits, and a request that took no lease loads nothing.
        assert_eq!(
            batched.load(&keys(&["aa", "ee"]), "r").unwrap(),
            per_key.load(&keys(&["aa", "ee"]), "r").unwrap()
        );
        assert!(batched.load(&keys(&["aa"]), "other").is_err());
        // Same counters and the same promotion order.
        assert_eq!(batched.stats(), per_key.stats());
        assert_eq!(lru_order(&batched), lru_order(&per_key));
        assert_eq!(batched.stats()["hits"], 2);
        assert_eq!(batched.stats()["misses"], 1);
        assert_eq!(batched.stats()["pending_hits"], 1);
        batched.assert_slot_accounting();
    }

    #[test]
    fn an_until_miss_scan_leases_only_the_hits_before_the_miss() {
        // `aa`, `bb`, `cc` and `dd` are published; `zz` was never stored.
        let stored = keys(&["aa", "bb", "cc", "dd"]);
        let scan = keys(&["aa", "bb", "zz", "cc"]);
        let mut truncated = cache_with(4);
        let mut whole = cache_with(4);
        for cache in [&mut truncated, &mut whole] {
            cache.store(&stored, "r").unwrap();
            cache.commit(&stored, "r", true).unwrap();
            touch_all(cache, &stored);
        }

        assert_eq!(
            truncated.lookup_many(&scan, "r", true),
            vec![Status::Hit, Status::Hit, Status::Miss]
        );
        assert_eq!(truncated.stats()["leased"], 2);
        assert_eq!(
            whole.lookup_many(&scan, "r", false),
            vec![Status::Hit, Status::Hit, Status::Miss, Status::Hit]
        );
        assert_eq!(whole.stats()["leased"], 3);
        // One miss and one hit per key either way.
        assert_eq!(truncated.stats()["misses"], whole.stats()["misses"]);
        assert_eq!(truncated.stats()["hits"], 2);

        // `cc`, past the miss, is the least recently used entry of both caches
        // and only the full scan leased it, so only the full scan walks past it.
        assert_eq!(
            truncated.store(&keys(&["ee"]), "r").unwrap().2,
            keys(&["cc"])
        );
        assert_eq!(whole.store(&keys(&["ee"]), "r").unwrap().2, keys(&["dd"]));
        truncated.assert_slot_accounting();
        whole.assert_slot_accounting();
    }

    #[test]
    fn an_until_miss_scan_reports_the_pending_status_it_walks_past() {
        // Only a miss ends the scan: a reservation another request is still
        // writing is reported, and it is not the caller's to hold.
        let mut cache = cache_with(4);
        let stored = keys(&["aa", "bb"]);
        cache.store(&stored, "writer").unwrap();
        cache.commit(&keys(&["aa"]), "writer", true).unwrap();
        touch(&cache, "aa");

        let scan = keys(&["aa", "bb", "zz", "cc"]);
        assert_eq!(
            cache.lookup_many(&scan, "r", true),
            vec![Status::Hit, Status::Pending, Status::Miss]
        );
        assert_eq!(cache.stats()["leased"], 1);
        assert_eq!(cache.stats()["pending_hits"], 1);
        cache.assert_slot_accounting();
    }

    #[test]
    fn release_request_uses_the_lease_index() {
        let mut cache = cache_with(3);
        let stored = keys(&["aa", "bb", "cc"]);
        cache.store(&stored, "r1").unwrap();
        cache.commit(&stored, "r1", true).unwrap();
        touch_all(&cache, &stored);
        assert_eq!(cache.lookup("aa", "r1"), Status::Hit);
        assert_eq!(cache.lookup("bb", "r1"), Status::Hit);
        assert_eq!(cache.lookup("cc", "r2"), Status::Hit);
        cache.release_request("r1");
        assert!(cache.load(&keys(&["aa"]), "r1").is_err());
        assert_eq!(
            cache.load(&keys(&["cc"]), "r2").unwrap(),
            vec![slot(&cache, "cc")]
        );
        // Both keys only `r1` leased are evictable again; `r2` still pins `cc`.
        let stored = |cache: &mut Cache, key: &str| {
            cache.store(&keys(&[key]), "r1").unwrap();
            cache.commit(&keys(&[key]), "r1", true).unwrap();
        };
        stored(&mut cache, "dd");
        assert_eq!(cache.store(&keys(&["ee"]), "r1").unwrap().2, keys(&["bb"]));
        cache.commit(&keys(&["ee"]), "r1", true).unwrap();
        assert_eq!(cache.store(&keys(&["ff"]), "r1").unwrap().2, keys(&["dd"]));
        cache.commit(&keys(&["ff"]), "r1", true).unwrap();
        cache.release_request("r2");
        assert_eq!(cache.store(&keys(&["gg"]), "r1").unwrap().2, keys(&["cc"]));
        cache.assert_slot_accounting();
    }

    #[test]
    fn store_of_mixed_present_and_missing_keys_reserves_only_missing() {
        let mut cache = cache_with(3);
        let stored = keys(&["aa", "bb", "cc"]);
        cache.store(&stored, "r").unwrap();
        cache.commit(&stored, "r", true).unwrap();
        let (reserved, slots, evicted) = cache.store(&keys(&["aa", "dd"]), "r").unwrap();
        assert_eq!(reserved, keys(&["dd"]));
        assert_eq!(slots.len(), 1);
        // `aa` is in the batch, so the next oldest key is the victim.
        assert_eq!(evicted, keys(&["bb"]));
        assert_eq!(cache.stats()["entries"], 3);
        assert_eq!(cache.stats()["pending"], 1);
        cache.assert_slot_accounting();
    }

    #[test]
    fn pending_entries_are_not_evictable_and_can_exhaust_capacity() {
        let mut cache = cache_with(2);
        let first = keys(&["aa", "bb"]);
        cache.store(&first, "r").unwrap();
        assert_eq!(cache.stats()["pending"], 2);
        assert_eq!(cache.store(&keys(&["cc"]), "r"), Err(StoreRejection::Full));
        cache.commit(&first, "r", false).unwrap();
        assert_eq!(cache.stats()["entries"], 0);
        cache.assert_slot_accounting();
        assert!(cache.store(&keys(&["cc"]), "r").is_ok());
    }

    #[test]
    fn store_that_cannot_echo_its_evictions_reserves_nothing() {
        // Long stored keys and short incoming ones: only the evicted keys make
        // the reply exceed one frame, which the incoming batch alone cannot show.
        let mut cache = Cache::new(
            1,
            64,
            1,
            2048,
            64 * 1024 * 1024,
            0,
            PrefetchLimits::default(),
        )
        .unwrap();
        let long: Vec<String> = (0..2048).map(|index| format!("{index:0510x}")).collect();
        for batch in long.chunks(512) {
            cache.store(batch, "r").unwrap();
            cache.commit(batch, "r", true).unwrap();
        }
        let before = cache.stats();
        let short: Vec<String> = (0..2048).map(|index| format!("{index:04x}")).collect();
        assert_eq!(cache.store(&short, "r"), Err(StoreRejection::Unanswerable));
        // Nothing was evicted or reserved: the caller keeps the cache it had.
        assert_eq!(cache.stats(), before);
        cache.assert_slot_accounting();
        touch(&cache, &long[0]);
        assert_eq!(cache.lookup(&long[0], "r"), Status::Hit);
    }

    #[test]
    fn check_keys_is_bounded() {
        assert!(check_keys(&keys(&["aa"])).is_ok());
        assert!(check_keys(&keys(&["aa", "aa"])).is_err());
        assert!(check_keys(&keys(&["not-hex"])).is_err());
        let batch: Vec<String> = (0..MAX_KEYS).map(|index| format!("{index:04x}")).collect();
        assert_eq!(batch.len(), MAX_KEYS);
        assert!(check_keys(&batch).is_ok());
        let mut oversized = batch.clone();
        oversized.push("ffff0".to_string());
        assert!(check_keys(&oversized).is_err());
        let mut duplicated = batch.clone();
        duplicated[MAX_KEYS - 1].clone_from(&batch[0]);
        assert!(check_keys(&duplicated).is_err());
        let mut too_long = batch.clone();
        too_long[3] = "a".repeat(514);
        assert!(check_keys(&too_long).is_err());
        let mut non_hex = batch.clone();
        non_hex[5] = "zz".to_string();
        assert!(check_keys(&non_hex).is_err());
    }

    /// Deterministic generator so a failing operation sequence is reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % bound
        }
    }

    #[test]
    fn no_slot_is_lost_or_duplicated_under_a_randomized_op_mix() {
        let mut random = Lcg(0x5eed);
        let mut cache = cache_with(4);
        let all: Vec<String> = (0..12u8).map(|index| format!("{index:02x}")).collect();
        let requests = ["r0", "r1"];
        let mut reserved: Vec<String> = Vec::new();
        let mut owner = String::new();
        for _ in 0..512 {
            let request = requests[random.below(requests.len())];
            let start = random.below(all.len());
            let mut batch: Vec<String> = Vec::new();
            for offset in 0..=random.below(3) {
                let key = all[(start + offset) % all.len()].clone();
                if !batch.contains(&key) {
                    batch.push(key);
                }
            }
            match random.below(10) {
                0 => {
                    if let Ok((keys, slots, evicted)) = cache.store(&batch, request) {
                        assert_eq!(keys.len(), slots.len());
                        assert!(slots.iter().all(|slot| *slot < cache.num_chunks));
                        assert!(evicted.iter().all(|key| !keys.contains(key)));
                        reserved = keys;
                        owner = request.to_owned();
                        // Commit most reservations at once so the mix spends more
                        // of its time on published chunks than on pending ones.
                        if random.below(4) != 0 {
                            let success = random.below(4) != 0;
                            assert!(cache.commit(&reserved, &owner, success).is_ok());
                            if success {
                                // A worker's DMA is what makes a chunk readable.
                                touch_all(&cache, &reserved);
                            }
                            reserved.clear();
                        }
                    }
                }
                1 => {
                    if cache.commit(&reserved, &owner, true).is_ok() {
                        touch_all(&cache, &reserved);
                        reserved.clear();
                    }
                }
                2 => {
                    if cache.commit(&reserved, &owner, false).is_ok() {
                        reserved.clear();
                    }
                }
                3 => {
                    // A request that did not reserve these keys cannot commit them.
                    if request != owner && !reserved.is_empty() {
                        assert!(cache.commit(&reserved, request, true).is_err());
                    }
                }
                4 => {
                    assert_eq!(cache.lookup_many(&batch, request, false).len(), batch.len());
                    // A truncated scan only reports a prefix of the batch.
                    assert!(cache.lookup_many(&batch, request, true).len() <= batch.len());
                }
                5 => {
                    if let Ok(slots) = cache.load(&batch, request) {
                        assert!(slots.iter().all(|slot| *slot < cache.num_chunks));
                    }
                }
                6 => {
                    cache.release(&batch, request);
                }
                7 => {
                    cache.release_request(request);
                }
                8 => {
                    if random.below(4) == 0 {
                        // Reclaim every reservation, as an idle namespace's timeout would.
                        cache.reclaim_expired(
                            Instant::now() + Duration::from_secs(3600),
                            Duration::from_secs(60),
                        );
                        reserved.clear();
                    } else {
                        let _ = cache.reset(random.below(2) == 0);
                        reserved.clear();
                    }
                }
                _ => cache.touch(&batch),
            }
            cache.assert_slot_accounting();
            let stats = cache.stats();
            assert!(stats["pending"] <= stats["entries"]);
            assert!(stats["leased"] <= stats["entries"]);
            assert!(stats["entries"] <= cache.num_chunks);
            assert_eq!(stats["free_chunks"] + stats["entries"], cache.num_chunks);
        }
    }

    #[test]
    fn commit_by_another_request_leaves_the_reservation_pending() {
        let mut cache = cache();
        let first = keys(&["aa"]);
        cache.store(&first, "a").unwrap();
        let error = cache.commit(&first, "b", true).unwrap_err();
        assert_eq!(error.to_string(), "commit does not own the reservation");
        assert!(!ready(&cache, "aa"));
        assert_eq!(cache.lookup("aa", "b"), Status::Pending);
        assert_eq!(cache.stats()["failed_commits"], 1);
        assert_eq!(cache.stats()["pending"], 1);
        cache.commit(&first, "a", true).unwrap();
        touch(&cache, "aa");
        assert_eq!(cache.lookup("aa", "a"), Status::Hit);
        assert_eq!(cache.stats()["commits"], 1);
        cache.assert_slot_accounting();
    }

    #[test]
    fn owner_keeps_its_name_while_any_reservation_it_made_is_pending() {
        let mut cache = cache();
        let both = keys(&["aa", "bb"]);
        cache.store(&both, "a").unwrap();
        cache.commit(&keys(&["aa"]), "a", false).unwrap();
        // `a` still owns the `bb` reservation, so another request cannot take it.
        assert!(cache.commit(&keys(&["bb"]), "b", true).is_err());
        cache.commit(&keys(&["bb"]), "a", true).unwrap();
        assert!(ready(&cache, "bb"));
        cache.assert_slot_accounting();
    }

    #[test]
    fn reclaim_expired_frees_only_expired_reservations_and_clears_hashes() {
        let (mut cache, ops) = recording_cache(false);
        let both = keys(&["aa", "bb"]);
        cache.store(&both, "a").unwrap();
        let expired_slot = slot(&cache, "aa");
        // Age one reservation past the timeout without sleeping.
        cache.slots[expired_slot].reservation.as_mut().unwrap().since = Instant::now()
            .checked_sub(Duration::from_secs(120))
            .expect("the monotonic clock is older than the timeout");
        let reclaimed = cache.reclaim_expired(Instant::now(), Duration::from_secs(60));
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(&*reclaimed[0].request, "a");
        assert_eq!(reclaimed[0].keys, 1);
        assert!(reclaimed[0].age >= Duration::from_secs(60));
        assert_eq!(recorded(&ops), vec![Op::Clear(expired_slot)]);
        assert_eq!(cache.lookup("aa", "a"), Status::Miss);
        assert_eq!(cache.lookup("bb", "a"), Status::Pending);
        assert_eq!(cache.stats()["reclaimed"], 1);
        assert_eq!(cache.stats()["pending"], 1);
        assert_eq!(cache.stats()["free_chunks"], 1);
        cache.assert_slot_accounting();
        // The surviving reservation is still its owner's to publish.
        cache.commit(&keys(&["bb"]), "a", true).unwrap();
        assert!(ready(&cache, "bb"));
    }

    #[test]
    fn forced_reset_discards_leases_and_pending_reservations() {
        let mut cache = cache_with(3);
        let stored = keys(&["aa", "bb"]);
        cache.store(&stored, "r").unwrap();
        cache.store(&keys(&["cc"]), "r").unwrap();
        cache.commit(&keys(&["aa"]), "r", true).unwrap();
        touch(&cache, "aa");
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        // A lease and a pending reservation both keep the ordinary reset out.
        assert!(cache.reset(false).is_err());
        cache.reset(true).unwrap();
        assert_eq!(cache.stats()["entries"], 0);
        assert_eq!(cache.stats()["pending"], 0);
        assert_eq!(cache.stats()["leased"], 0);
        assert_eq!(cache.stats()["free_chunks"], 3);
        assert_eq!(cache.stats()["requests"], 0);
        assert_eq!(cache.lookup("aa", "r"), Status::Miss);
        assert!(cache.load(&keys(&["aa"]), "r").is_err());
        cache.assert_slot_accounting();
        assert!(cache.store(&stored, "r").is_ok());
    }

    #[test]
    fn stats_counters_track_store_commit_abort_and_eviction() {
        let mut cache = cache_with(2);
        let first = keys(&["aa"]);
        cache.store(&first, "r").unwrap();
        cache.commit(&first, "r", true).unwrap();
        // Committing a published key changes nothing and is counted as a failure.
        assert!(cache.commit(&first, "r", true).is_err());
        let second = keys(&["bb"]);
        cache.store(&second, "r").unwrap();
        assert_eq!(cache.lookup("bb", "r"), Status::Pending);
        cache.commit(&second, "r", false).unwrap();
        let mut stats = cache.stats();
        assert_eq!(stats["stores"], 2);
        assert_eq!(stats["commits"], 1);
        assert_eq!(stats["aborted"], 1);
        assert_eq!(stats["failed_commits"], 1);
        assert_eq!(stats["pending_hits"], 1);
        assert_eq!(stats["evictions"], 0);
        assert_eq!(stats["hits"], 0);
        assert_eq!(stats["misses"], 0);
        assert_eq!(stats["requests"], 0);
        assert_eq!(stats["entries"], 1);

        // The aborted reservation left a free slot; filling it makes the next
        // store evict the least recently used key.
        cache.store(&second, "r").unwrap();
        cache.commit(&second, "r", true).unwrap();
        let (_, _, evicted) = cache.store(&keys(&["cc"]), "r").unwrap();
        assert_eq!(evicted, keys(&["aa"]));
        cache.commit(&keys(&["cc"]), "r", true).unwrap();
        touch(&cache, "cc");
        assert_eq!(cache.lookup("cc", "r"), Status::Hit);
        assert_eq!(cache.lookup("dd", "r"), Status::Miss);
        stats = cache.stats();
        assert_eq!(stats["stores"], 4);
        assert_eq!(stats["commits"], 3);
        assert_eq!(stats["aborted"], 1);
        assert_eq!(stats["failed_commits"], 1);
        assert_eq!(stats["evictions"], 1);
        assert_eq!(stats["hits"], 1);
        assert_eq!(stats["misses"], 1);
        assert_eq!(stats["requests"], 1);
        assert_eq!(stats["entries"], 2);
        assert_eq!(stats["free_chunks"], 0);
        assert_eq!(stats["num_chunks"], 2);
        assert_eq!(stats["bytes"], cache.mapping.len());
        // Counters describe the namespace, not one epoch, so `reset` keeps them.
        cache.reset(true).unwrap();
        assert_eq!(cache.stats()["hits"], 1);
    }

    #[test]
    fn release_reports_keys_and_requests_without_a_lease() {
        let mut cache = cache_with(2);
        let stored = keys(&["aa", "bb"]);
        cache.store(&stored, "r").unwrap();
        cache.commit(&stored, "r", true).unwrap();
        touch_all(&cache, &stored);
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        assert_eq!(
            cache.release(&keys(&["aa", "bb", "cc"]), "r"),
            Released {
                leases: 1,
                unknown: 2,
            }
        );
        assert_eq!(
            cache.release_request("unknown"),
            Released {
                leases: 0,
                unknown: 1,
            }
        );
        assert_eq!(cache.lookup("bb", "r"), Status::Hit);
        assert_eq!(
            cache.release_request("r"),
            Released {
                leases: 1,
                unknown: 0,
            }
        );
        assert_eq!(cache.stats()["requests"], 0);
    }
}
