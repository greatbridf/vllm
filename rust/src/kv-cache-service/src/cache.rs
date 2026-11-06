// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::FromRawFd;
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

pub struct Cache {
    pub owner: u64,
    pub worker_bytes: usize,
    pub world_size: usize,
    pub row_bytes: usize,
    pub num_chunks: usize,
    pub file: File,
    pub mapping: MmapMut,
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
    hashes: Option<Box<dyn PageHashes>>,
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
    ) -> io::Result<Self> {
        Self::with_hashes(owner, worker_bytes, world_size, num_chunks, budget, None)
    }

    pub fn with_hashes(
        owner: u64,
        worker_bytes: usize,
        world_size: usize,
        num_chunks: usize,
        budget: usize,
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
            num_chunks,
            file,
            mapping,
            slots: (0..num_chunks).map(|_| Slot::idle()).collect(),
            index: HashMap::new(),
            free: (0..num_chunks as u32).rev().collect(),
            lru_head: NO_SLOT,
            lru_tail: NO_SLOT,
            leases: LeaseIndex::default(),
            owners: OwnerIndex::default(),
            counters: Counters::default(),
            hashes,
        })
    }

    /// Clears all published per-page hashes.
    pub fn invalidate_hashes(&mut self) {
        if let Some(hashes) = &mut self.hashes {
            hashes.reset();
        }
    }

    pub fn lookup(&mut self, key: &str, request: &str) -> Status {
        let Some(&slot) = self.index.get(key) else {
            self.counters.misses += 1;
            return Status::Miss;
        };
        if !self.slots[slot as usize].ready {
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

    pub fn load(&self, keys: &[String], request: &str) -> io::Result<Vec<usize>> {
        let request = self.leases.get(request);
        keys.iter()
            .map(|key| {
                let slot = *self
                    .index
                    .get(key.as_str())
                    .ok_or_else(|| invalid("load requires a ready leased entry"))?;
                let entry = &self.slots[slot as usize];
                let leased = request.is_some_and(|request| entry.readers.contains(&request));
                if !entry.ready || !leased {
                    return Err(invalid("load requires a ready leased entry"));
                }
                Ok(slot as usize)
            })
            .collect()
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
            let readers = &mut self.slots[slot as usize].readers;
            match readers.iter().position(|reader| *reader == request) {
                Some(position) => {
                    readers.swap_remove(position);
                    self.leases.unlease(request, slot);
                    released.leases += 1;
                }
                None => released.unknown += 1,
            }
        }
        released
    }

    pub fn release_request(&mut self, request: &str) -> Released {
        let Some((request, slots)) = self.leases.release(request) else {
            return Released {
                leases: 0,
                unknown: 1,
            };
        };
        for slot in &slots {
            self.slots[*slot as usize].readers.retain(|reader| *reader != request);
        }
        Released {
            leases: slots.len(),
            unknown: 0,
        }
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
        self.index.clear();
        self.leases = LeaseIndex::default();
        self.owners = OwnerIndex::default();
        self.lru_head = NO_SLOT;
        self.lru_tail = NO_SLOT;
        for entry in &mut self.slots {
            *entry = Slot::idle();
        }
        self.free = (0..self.num_chunks as u32).rev().collect();
        Ok(())
    }

    pub fn stats(&self) -> HashMap<&'static str, usize> {
        let entries = self.index.values().map(|slot| &self.slots[*slot as usize]);
        let mut pending = 0;
        let mut leased = 0;
        for entry in entries {
            pending += usize::from(!entry.ready);
            leased += usize::from(!entry.readers.is_empty());
        }
        HashMap::from([
            ("entries", self.index.len()),
            ("pending", pending),
            ("leased", leased),
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
        Cache::new(1, 64, 1, num_chunks, 16 * 1024 * 1024).unwrap()
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
            Some(&factory as &dyn HashFactory),
        )
        .unwrap();
        (cache, ops)
    }

    fn recorded(ops: &Arc<Mutex<Vec<Op>>>) -> Vec<Op> {
        ops.lock().unwrap().clone()
    }

    #[test]
    fn lookup_lease_prevents_eviction_before_load() {
        let mut cache = cache();
        let first = keys(&["aa", "bb"]);
        cache.store(&first, "r").unwrap();
        assert_eq!(cache.lookup("aa", "r"), Status::Pending);
        cache.commit(&first, "r", true).unwrap();
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
        assert_eq!(cache.lookup("aa", "r"), Status::Hit);
        assert!(cache.reset(false).is_err());
    }

    #[test]
    fn geometry_and_keys_are_bounded() {
        assert!(Cache::new(1, usize::MAX, 2, 1, 8192).is_err());
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
        let mut cache = Cache::new(1, 64, 1, 2048, 64 * 1024 * 1024).unwrap();
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
                            assert!(cache.commit(&reserved, &owner, random.below(4) != 0).is_ok());
                            reserved.clear();
                        }
                    }
                }
                1 => {
                    if cache.commit(&reserved, &owner, true).is_ok() {
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
