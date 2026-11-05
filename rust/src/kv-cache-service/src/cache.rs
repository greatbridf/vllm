// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::FromRawFd;

use indexmap::IndexMap;
use memmap2::{MmapMut, MmapOptions};

use crate::hash::{HashFactory, HashGeometry, PageHashes};
use crate::protocol::MAX_KEYS;

pub struct Entry {
    pub slot: usize,
    pub ready: bool,
    readers: HashSet<String>,
}

pub struct Cache {
    pub owner: u64,
    pub worker_bytes: usize,
    pub world_size: usize,
    pub row_bytes: usize,
    pub num_chunks: usize,
    pub file: File,
    pub mapping: MmapMut,
    entries: IndexMap<String, Entry>,
    free: Vec<usize>,
    hashes: Option<Box<dyn PageHashes>>,
}

/// Keys, slots, and evicted keys returned by `Cache::store`.
pub type StorePlan = (Vec<String>, Vec<usize>, Vec<String>);

fn invalid(message: &str) -> io::Error {
    io::Error::other(message)
}

pub fn check_keys(keys: &[String]) -> io::Result<()> {
    if keys.len() > MAX_KEYS
        || keys.iter().any(|key| {
            key.is_empty()
                || key.len() > 512
                || key.len() % 2 != 0
                || !key.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        || keys.iter().collect::<HashSet<_>>().len() != keys.len()
    {
        return Err(invalid("invalid or duplicate cache keys"));
    }
    Ok(())
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
    if worker_bytes == 0 || world_size == 0 || world_size > 64 || num_chunks == 0 {
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
            entries: IndexMap::new(),
            free: (0..num_chunks).rev().collect(),
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
        let Some(entry) = self.entries.get_mut(key) else {
            return "miss";
        };
        if !entry.ready {
            return "pending";
        }
        // Lookup itself takes the read lease, closing the lookup/load race.
        entry.readers.insert(request.to_owned());
        "hit"
    }

    pub fn load(&self, keys: &[String], request: &str) -> io::Result<Vec<usize>> {
        keys.iter()
            .map(|key| {
                self.entries
                    .get(key)
                    .filter(|entry| entry.ready && entry.readers.contains(request))
                    .map(|entry| entry.slot)
                    .ok_or_else(|| invalid("load requires a ready leased entry"))
            })
            .collect()
    }

    pub fn release(&mut self, keys: &[String], request: &str) {
        for key in keys {
            if let Some(entry) = self.entries.get_mut(key) {
                entry.readers.remove(request);
            }
        }
    }

    pub fn release_request(&mut self, request: &str) {
        for entry in self.entries.values_mut() {
            entry.readers.remove(request);
        }
    }

    pub fn touch(&mut self, keys: &[String]) {
        for key in keys {
            if let Some(entry) = self.entries.shift_remove(key) {
                self.entries.insert(key.clone(), entry);
            }
        }
    }

    pub fn store(&mut self, keys: &[String]) -> Option<StorePlan> {
        let missing: Vec<_> =
            keys.iter().filter(|key| !self.entries.contains_key(*key)).cloned().collect();
        let victims: Vec<_> = self
            .entries
            .iter()
            .filter(|(key, entry)| {
                let retained = keys.contains(key);
                entry.ready && entry.readers.is_empty() && !retained
            })
            .map(|(key, _)| key.clone())
            .take(missing.len().saturating_sub(self.free.len()))
            .collect();
        if missing.len() > self.free.len() + victims.len() {
            return None;
        }
        for victim in &victims {
            let entry = self.entries.shift_remove(victim).expect("selected entry");
            if let Some(hashes) = &mut self.hashes {
                hashes.clear(entry.slot);
            }
            self.free.push(entry.slot);
        }
        let mut slots = Vec::with_capacity(missing.len());
        for key in &missing {
            let slot = self.free.pop().expect("capacity checked");
            self.entries.insert(
                key.clone(),
                Entry {
                    slot,
                    ready: false,
                    readers: HashSet::new(),
                },
            );
            slots.push(slot);
        }
        Some((missing, slots, victims))
    }

    pub fn commit(&mut self, keys: &[String], success: bool) -> io::Result<()> {
        if keys.iter().any(|key| self.entries.get(key).is_none_or(|entry| entry.ready)) {
            return Err(invalid("commit requires pending reservations"));
        }
        let slots: Vec<usize> =
            keys.iter().map(|key| self.entries.get(key).expect("validated").slot).collect();
        if success {
            // Publish before any entry becomes ready: a failed publish leaves
            // every reservation pending and the caller fails closed.
            if let Some(hashes) = &mut self.hashes {
                for (key, &slot) in keys.iter().zip(&slots) {
                    if let Err(error) = hashes.publish(slot, key) {
                        for &slot in &slots {
                            hashes.clear(slot);
                        }
                        return Err(error);
                    }
                }
            }
            for key in keys {
                self.entries.get_mut(key).expect("validated").ready = true;
            }
        } else {
            for (key, &slot) in keys.iter().zip(&slots) {
                if let Some(hashes) = &mut self.hashes {
                    hashes.clear(slot);
                }
                let entry = self.entries.shift_remove(key).expect("validated");
                self.free.push(entry.slot);
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) -> io::Result<()> {
        if self.entries.values().any(|entry| !entry.ready || !entry.readers.is_empty()) {
            return Err(invalid(
                "cannot reset a cache with active transfers or leases",
            ));
        }
        if let Some(hashes) = &mut self.hashes {
            hashes.reset();
        }
        self.entries.clear();
        self.free = (0..self.num_chunks).rev().collect();
        Ok(())
    }

    pub fn stats(&self) -> HashMap<&'static str, usize> {
        HashMap::from([
            ("entries", self.entries.len()),
            (
                "pending",
                self.entries.values().filter(|entry| !entry.ready).count(),
            ),
            (
                "leased",
                self.entries.values().filter(|entry| !entry.readers.is_empty()).count(),
            ),
            ("bytes", self.mapping.len()),
        ])
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
        Cache::new(1, 64, 1, 2, 2 * 1024 * 1024).unwrap()
    }

    fn keys(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
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
        let slot = cache.entries.get("aa").unwrap().slot;
        assert!(!cache.entries.get("aa").unwrap().ready);
        cache.commit(&first, true).unwrap();
        assert_eq!(recorded(&ops), vec![Op::Publish(slot, "aa".into())]);
        assert!(cache.entries.get("aa").unwrap().ready);
    }

    #[test]
    fn failed_publish_keeps_entry_pending_and_returns_error() {
        let (mut cache, ops) = recording_cache(true);
        let first = keys(&["aa"]);
        cache.store(&first).unwrap();
        let slot = cache.entries.get("aa").unwrap().slot;
        let error = cache.commit(&first, true).unwrap_err();
        assert!(error.to_string().contains("publish failed"));
        assert_eq!(
            recorded(&ops),
            vec![Op::Publish(slot, "aa".into()), Op::Clear(slot)]
        );
        assert!(!cache.entries.get("aa").unwrap().ready);
        assert_eq!(cache.lookup("aa", "r"), "pending");
    }

    #[test]
    fn failed_publish_clears_every_slot_in_the_batch() {
        let (mut cache, ops) = recording_cache(true);
        let first = keys(&["aa", "bb"]);
        cache.store(&first).unwrap();
        let slots: Vec<_> = first.iter().map(|key| cache.entries.get(key).unwrap().slot).collect();
        assert!(cache.commit(&first, true).is_err());
        assert_eq!(
            recorded(&ops),
            vec![
                Op::Publish(slots[0], "aa".into()),
                Op::Clear(slots[0]),
                Op::Clear(slots[1]),
            ]
        );
        assert!(first.iter().all(|key| !cache.entries.get(key).unwrap().ready));
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
        let victim_slot = cache.entries.get("aa").unwrap().slot;
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
        let slot = cache.entries.get("aa").unwrap().slot;
        assert_eq!(cache.lookup("aa", "r"), "hit");
        assert_eq!(cache.load(&keys(&["aa"]), "r").unwrap(), vec![slot]);
        cache.touch(&first);
        assert_eq!(recorded(&ops), before);
    }
}
