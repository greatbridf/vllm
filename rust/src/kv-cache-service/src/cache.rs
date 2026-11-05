// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::FromRawFd;

use memmap2::{MmapMut, MmapOptions};

use crate::hash::{HashFactory, HashGeometry, PageHashes};
use crate::protocol::MAX_KEYS;

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
}

impl Slot {
    fn idle() -> Slot {
        Slot {
            key: Box::default(),
            prev: NO_SLOT,
            next: NO_SLOT,
            readers: Vec::new(),
            ready: false,
        }
    }
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
    hashes: Option<Box<dyn PageHashes>>,
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
}

/// Keys, slots, and evicted keys returned by `Cache::store`.
pub type StorePlan = (Vec<String>, Vec<usize>, Vec<String>);

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
            hashes,
        })
    }

    /// Clears all published per-page hashes.
    pub fn invalidate_hashes(&mut self) {
        if let Some(hashes) = &mut self.hashes {
            hashes.reset();
        }
    }

    pub fn lookup(&mut self, key: &str, request: &str) -> &'static str {
        let Some(&slot) = self.index.get(key) else {
            return "miss";
        };
        if !self.slots[slot as usize].ready {
            return "pending";
        }
        // Lookup itself takes the read lease, closing the lookup/load race.
        let request = self.leases.intern(request);
        if !self.slots[slot as usize].readers.contains(&request) {
            self.slots[slot as usize].readers.push(request);
            self.leases.lease(request, slot);
        }
        self.promote(slot);
        "hit"
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

    pub fn release(&mut self, keys: &[String], request: &str) {
        let Some(request) = self.leases.get(request) else {
            return;
        };
        for key in keys {
            let Some(&slot) = self.index.get(key.as_str()) else {
                continue;
            };
            let readers = &mut self.slots[slot as usize].readers;
            if let Some(position) = readers.iter().position(|reader| *reader == request) {
                readers.swap_remove(position);
                self.leases.unlease(request, slot);
            }
        }
    }

    pub fn release_request(&mut self, request: &str) {
        let Some((request, slots)) = self.leases.release(request) else {
            return;
        };
        for slot in slots {
            self.slots[slot as usize].readers.retain(|reader| *reader != request);
        }
    }

    pub fn touch(&mut self, keys: &[String]) {
        for key in keys {
            if let Some(&slot) = self.index.get(key.as_str()) {
                self.promote(slot);
            }
        }
    }

    pub fn store(&mut self, keys: &[String]) -> Option<StorePlan> {
        let incoming: HashSet<&str> = keys.iter().map(String::as_str).collect();
        let missing: Vec<&str> = keys
            .iter()
            .map(String::as_str)
            .filter(|key| !self.index.contains_key(*key))
            .collect();
        let victims =
            self.eviction_victims(missing.len().saturating_sub(self.free.len()), &incoming)?;
        let mut evicted = Vec::with_capacity(victims.len());
        for slot in victims {
            let key = self.slots[slot as usize].key.to_string();
            if let Some(hashes) = &mut self.hashes {
                hashes.clear(slot as usize);
            }
            self.release_slot(slot);
            evicted.push(key);
        }
        let mut reserved = Vec::with_capacity(missing.len());
        let mut slots = Vec::with_capacity(missing.len());
        for key in missing {
            let slot = self.free.pop().expect("capacity checked");
            self.slots[slot as usize].key = Box::from(key);
            self.slots[slot as usize].ready = false;
            self.push_front(slot);
            self.index.insert(Box::from(key), slot);
            reserved.push(key.to_owned());
            slots.push(slot as usize);
        }
        Some((reserved, slots, evicted))
    }

    pub fn commit(&mut self, keys: &[String], success: bool) -> io::Result<()> {
        let mut slots = Vec::with_capacity(keys.len());
        for key in keys {
            let slot = *self
                .index
                .get(key.as_str())
                .filter(|slot| !self.slots[**slot as usize].ready)
                .ok_or_else(|| invalid("commit requires pending reservations"))?;
            slots.push(slot);
        }
        if success {
            // Publish before any entry becomes ready: a failed publish leaves
            // every reservation pending and the caller fails closed.
            if let Some(hashes) = &mut self.hashes {
                for (key, &slot) in keys.iter().zip(&slots) {
                    if let Err(error) = hashes.publish(slot as usize, key) {
                        for &slot in &slots {
                            hashes.clear(slot as usize);
                        }
                        return Err(error);
                    }
                }
            }
            for slot in slots {
                self.slots[slot as usize].ready = true;
            }
        } else {
            for slot in slots {
                if let Some(hashes) = &mut self.hashes {
                    hashes.clear(slot as usize);
                }
                self.release_slot(slot);
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) -> io::Result<()> {
        if self.index.values().any(|slot| {
            let entry = &self.slots[*slot as usize];
            !entry.ready || !entry.readers.is_empty()
        }) {
            return Err(invalid(
                "cannot reset a cache with active transfers or leases",
            ));
        }
        if let Some(hashes) = &mut self.hashes {
            hashes.reset();
        }
        self.index.clear();
        self.leases = LeaseIndex::default();
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
            ("bytes", self.mapping.len()),
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
        debug_assert!(
            self.slots[slot as usize].readers.is_empty(),
            "a leased slot cannot be released"
        );
        self.index.remove(key.as_ref());
        self.free.push(slot);
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
        cache.store(&first).unwrap();
        assert_eq!(cache.lookup("aa", "r"), "pending");
        cache.commit(&first, true).unwrap();
        assert_eq!(cache.lookup("aa", "r"), "hit");
        let (_, _, evicted) = cache.store(&keys(&["cc"])).unwrap();
        assert_eq!(evicted, keys(&["bb"]));
        assert_eq!(cache.load(&keys(&["aa"]), "r").unwrap(), vec![0]);
        assert!(cache.store(&keys(&["dd"])).is_none());
        cache.release_request("r");
        assert!(cache.store(&keys(&["dd"])).is_some());
    }

    #[test]
    fn failed_store_is_never_published() {
        let mut cache = cache();
        let first = keys(&["aa"]);
        cache.store(&first).unwrap();
        cache.commit(&first, false).unwrap();
        assert_eq!(cache.lookup("aa", "r"), "miss");
        assert!(cache.commit(&first, true).is_err());
    }

    #[test]
    fn reservation_is_atomic_and_cannot_evict_itself() {
        let mut cache = cache();
        let first = keys(&["aa", "bb"]);
        cache.store(&first).unwrap();
        cache.commit(&first, true).unwrap();
        assert!(cache.store(&keys(&["aa", "bb", "cc"])).is_none());
        assert_eq!(cache.lookup("aa", "r"), "hit");
        assert!(cache.reset().is_err());
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
        cache.store(&first).unwrap();
        assert!(recorded(&ops).is_empty());
        let slot = slot(&cache, "aa");
        assert!(!ready(&cache, "aa"));
        cache.commit(&first, true).unwrap();
        assert_eq!(recorded(&ops), vec![Op::Publish(slot, "aa".into())]);
        assert!(ready(&cache, "aa"));
    }

    #[test]
    fn failed_publish_keeps_entry_pending_and_returns_error() {
        let (mut cache, ops) = recording_cache(true);
        let first = keys(&["aa"]);
        cache.store(&first).unwrap();
        let slot = slot(&cache, "aa");
        let error = cache.commit(&first, true).unwrap_err();
        assert!(error.to_string().contains("publish failed"));
        assert_eq!(
            recorded(&ops),
            vec![Op::Publish(slot, "aa".into()), Op::Clear(slot)]
        );
        assert!(!ready(&cache, "aa"));
        assert_eq!(cache.lookup("aa", "r"), "pending");
    }

    #[test]
    fn failed_publish_clears_every_slot_in_the_batch() {
        let (mut cache, ops) = recording_cache(true);
        let first = keys(&["aa", "bb"]);
        cache.store(&first).unwrap();
        let slots: Vec<_> = first.iter().map(|key| slot(&cache, key)).collect();
        assert!(cache.commit(&first, true).is_err());
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
        cache.store(&first).unwrap();
        cache.commit(&first, false).unwrap();
        assert_eq!(recorded(&ops), vec![Op::Clear(0)]);

        let both = keys(&["aa", "bb"]);
        cache.store(&both).unwrap();
        cache.commit(&both, true).unwrap();
        let victim_slot = slot(&cache, "aa");
        ops.lock().unwrap().clear();
        let (_, _, evicted) = cache.store(&keys(&["cc"])).unwrap();
        assert_eq!(evicted, keys(&["aa"]));
        assert_eq!(recorded(&ops), vec![Op::Clear(victim_slot)]);
    }

    #[test]
    fn reset_and_invalidate_hashes_call_reset() {
        let (mut cache, ops) = recording_cache(false);
        let first = keys(&["aa"]);
        cache.store(&first).unwrap();
        cache.commit(&first, true).unwrap();
        ops.lock().unwrap().clear();
        cache.reset().unwrap();
        cache.invalidate_hashes();
        assert_eq!(recorded(&ops), vec![Op::Reset, Op::Reset]);
    }

    #[test]
    fn touch_lookup_and_load_do_not_publish_hashes() {
        let (mut cache, ops) = recording_cache(false);
        let first = keys(&["aa"]);
        cache.store(&first).unwrap();
        cache.commit(&first, true).unwrap();
        let before = recorded(&ops);
        let slot = slot(&cache, "aa");
        assert_eq!(cache.lookup("aa", "r"), "hit");
        assert_eq!(cache.load(&keys(&["aa"]), "r").unwrap(), vec![slot]);
        cache.touch(&first);
        assert_eq!(recorded(&ops), before);
    }

    #[test]
    fn eviction_follows_recency_and_lookup_promotes() {
        let mut cache = cache_with(3);
        let stored = keys(&["aa", "bb", "cc"]);
        cache.store(&stored).unwrap();
        cache.commit(&stored, true).unwrap();
        assert_eq!(cache.lookup("aa", "r"), "hit");
        // `bb` is the least recently used key, not the oldest inserted one.
        assert_eq!(cache.store(&keys(&["dd"])).unwrap().2, keys(&["bb"]));
        cache.release_request("r");
        cache.touch(&keys(&["cc"]));
        assert_eq!(cache.store(&keys(&["ee"])).unwrap().2, keys(&["aa"]));
        cache.assert_slot_accounting();
    }

    #[test]
    fn release_request_uses_the_lease_index() {
        let mut cache = cache_with(3);
        let stored = keys(&["aa", "bb", "cc"]);
        cache.store(&stored).unwrap();
        cache.commit(&stored, true).unwrap();
        assert_eq!(cache.lookup("aa", "r1"), "hit");
        assert_eq!(cache.lookup("bb", "r1"), "hit");
        assert_eq!(cache.lookup("cc", "r2"), "hit");
        cache.release_request("r1");
        assert!(cache.load(&keys(&["aa"]), "r1").is_err());
        assert_eq!(
            cache.load(&keys(&["cc"]), "r2").unwrap(),
            vec![slot(&cache, "cc")]
        );
        // Both keys only `r1` leased are evictable again; `r2` still pins `cc`.
        let stored = |cache: &mut Cache, key: &str| {
            cache.store(&keys(&[key])).unwrap();
            cache.commit(&keys(&[key]), true).unwrap();
        };
        stored(&mut cache, "dd");
        assert_eq!(cache.store(&keys(&["ee"])).unwrap().2, keys(&["bb"]));
        cache.commit(&keys(&["ee"]), true).unwrap();
        assert_eq!(cache.store(&keys(&["ff"])).unwrap().2, keys(&["dd"]));
        cache.commit(&keys(&["ff"]), true).unwrap();
        cache.release_request("r2");
        assert_eq!(cache.store(&keys(&["gg"])).unwrap().2, keys(&["cc"]));
        cache.assert_slot_accounting();
    }

    #[test]
    fn store_of_mixed_present_and_missing_keys_reserves_only_missing() {
        let mut cache = cache_with(3);
        let stored = keys(&["aa", "bb", "cc"]);
        cache.store(&stored).unwrap();
        cache.commit(&stored, true).unwrap();
        let (reserved, slots, evicted) = cache.store(&keys(&["aa", "dd"])).unwrap();
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
        cache.store(&first).unwrap();
        assert_eq!(cache.stats()["pending"], 2);
        assert!(cache.store(&keys(&["cc"])).is_none());
        cache.commit(&first, false).unwrap();
        assert_eq!(cache.stats()["entries"], 0);
        cache.assert_slot_accounting();
        assert!(cache.store(&keys(&["cc"])).is_some());
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
            match random.below(9) {
                0 => {
                    if let Some((keys, slots, evicted)) = cache.store(&batch) {
                        assert_eq!(keys.len(), slots.len());
                        assert!(slots.iter().all(|slot| *slot < cache.num_chunks));
                        assert!(evicted.iter().all(|key| !keys.contains(key)));
                        reserved = keys;
                        // Commit most reservations at once so the mix spends more
                        // of its time on published chunks than on pending ones.
                        if random.below(4) != 0 {
                            assert!(cache.commit(&reserved, random.below(4) != 0).is_ok());
                            reserved.clear();
                        }
                    }
                }
                1 => {
                    if cache.commit(&reserved, true).is_ok() {
                        reserved.clear();
                    }
                }
                2 => {
                    if cache.commit(&reserved, false).is_ok() {
                        reserved.clear();
                    }
                }
                3 => {
                    let status = cache.lookup(&batch[0], request);
                    assert!(matches!(status, "hit" | "miss" | "pending"));
                }
                4 => {
                    if let Ok(slots) = cache.load(&batch, request) {
                        assert!(slots.iter().all(|slot| *slot < cache.num_chunks));
                    }
                }
                5 => cache.release(&batch, request),
                6 => cache.release_request(request),
                7 => {
                    let _ = cache.reset();
                }
                _ => cache.touch(&batch),
            }
            cache.assert_slot_accounting();
            let stats = cache.stats();
            assert!(stats["pending"] <= stats["entries"]);
            assert!(stats["leased"] <= stats["entries"]);
            assert!(stats["entries"] <= cache.num_chunks);
        }
    }
}
