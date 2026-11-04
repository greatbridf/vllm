// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::FromRawFd;

use indexmap::IndexMap;
use memmap2::{MmapMut, MmapOptions};

use crate::protocol::MAX_KEYS;

pub struct Entry {
    pub slot: usize,
    pub ready: bool,
    readers: HashSet<String>,
}

pub struct Cache {
    pub owner: u64,
    pub fingerprint: String,
    pub worker_bytes: usize,
    pub world_size: usize,
    pub row_bytes: usize,
    pub num_chunks: usize,
    pub file: File,
    pub mapping: MmapMut,
    pub attached_ranks: HashSet<usize>,
    pub sessions: HashSet<u64>,
    entries: IndexMap<String, Entry>,
    free: Vec<usize>,
}

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

impl Cache {
    pub fn new(
        owner: u64,
        fingerprint: String,
        worker_bytes: usize,
        world_size: usize,
        num_chunks: usize,
        budget: usize,
    ) -> io::Result<Self> {
        if worker_bytes == 0 || world_size == 0 || world_size > 64 || num_chunks == 0 {
            return Err(invalid("invalid cache geometry"));
        }
        // SAFETY: sysconf has no pointer arguments; the result is validated.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return Err(invalid("cannot determine page size"));
        }
        let page = page as usize;
        let row_bytes = worker_bytes
            .checked_mul(world_size)
            .and_then(|bytes| bytes.checked_add(page - 1))
            .map(|bytes| bytes / page * page)
            .ok_or_else(|| invalid("cache geometry overflow"))?;
        let size = row_bytes
            .checked_mul(num_chunks)
            .filter(|size| *size <= budget && *size <= isize::MAX as usize)
            .ok_or_else(|| invalid("cache exceeds service capacity"))?;
        let name = CString::new("vllm-kv-cache").expect("literal contains no NUL");
        // SAFETY: name is NUL-terminated; the returned descriptor is owned below.
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
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
        Ok(Self {
            owner,
            fingerprint,
            worker_bytes,
            world_size,
            row_bytes,
            num_chunks,
            file,
            mapping,
            attached_ranks: HashSet::new(),
            sessions: HashSet::from([owner]),
            entries: IndexMap::new(),
            free: (0..num_chunks).rev().collect(),
        })
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

    pub fn store(&mut self, keys: &[String]) -> Option<(Vec<String>, Vec<usize>, Vec<String>)> {
        let missing: Vec<_> =
            keys.iter().filter(|key| !self.entries.contains_key(*key)).cloned().collect();
        let victims: Vec<_> = self
            .entries
            .iter()
            .filter(|(key, entry)| entry.ready && entry.readers.is_empty() && !keys.contains(key))
            .map(|(key, _)| key.clone())
            .take(missing.len().saturating_sub(self.free.len()))
            .collect();
        if missing.len() > self.free.len() + victims.len() {
            return None;
        }
        for victim in &victims {
            let entry = self.entries.shift_remove(victim).expect("selected entry");
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
        for key in keys {
            if success {
                self.entries.get_mut(key).expect("validated").ready = true;
            } else {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> Cache {
        Cache::new(1, "test".into(), 64, 1, 2, 2 * 1024 * 1024).unwrap()
    }

    fn keys(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
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
        assert!(Cache::new(1, "test".into(), usize::MAX, 2, 1, 8192).is_err());
        assert!(check_keys(&keys(&["aa", "aa"])).is_err());
        assert!(check_keys(&keys(&["not-hex"])).is_err());
    }
}
