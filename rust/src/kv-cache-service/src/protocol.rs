// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;

use serde::Deserialize;
use serde_json::Value;

pub const MAX_FRAME: usize = 1024 * 1024;

/// Default for a boolean the wire leaved enabled, e.g. prefetching.
fn enabled() -> bool {
    true
}
pub const MAX_KEYS: usize = 4096;

/// Longest decimal slot id (20 digits) plus its separating comma.
const SLOT_BYTES: usize = 21;
/// A reply envelope with empty key, slot, and eviction arrays.
const ENVELOPE_BYTES: usize = 45;

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Open {
        version: u32,
        namespace: String,
        fingerprint: String,
        worker_bytes: usize,
        world_size: usize,
        num_chunks: usize,
        /// Adopt a namespace a departed owner left behind instead of creating
        /// one, which is what a reconnect may do and a first start may not.
        #[serde(default)]
        adopt: bool,
    },
    Attach {
        version: u32,
        namespace: String,
        fingerprint: String,
        rank: usize,
    },
    Lookup {
        key: String,
        request: String,
    },
    /// One `lookup` per key of `keys`, answered with one status per key.
    ///
    /// With `until_miss` the scan ends at the first `miss`, so the reply may be
    /// shorter than `keys`: its last status is then that `miss`, and only the
    /// hits before it hold a lease. The reply never echoes a key, so a batch
    /// bounded by `MAX_KEYS` always fits one frame.
    LookupMany {
        keys: Vec<String>,
        request: String,
        #[serde(default)]
        until_miss: bool,
    },
    Load {
        keys: Vec<String>,
        request: String,
        /// Ask the kernel to read the chunks back in as one range.
        #[serde(default = "enabled")]
        prefetch: bool,
    },
    /// Ask the kernel to read committed chunks back in before a `load` needs
    /// them, which is how the readahead overlaps scheduling instead of a fault.
    Prefetch {
        keys: Vec<String>,
    },
    Release {
        keys: Vec<String>,
        request: String,
    },
    #[serde(rename = "release_request")]
    ReleaseAll {
        request: String,
    },
    Store {
        keys: Vec<String>,
        request: String,
    },
    Commit {
        keys: Vec<String>,
        request: String,
        success: bool,
    },
    Touch {
        keys: Vec<String>,
    },
    /// Test/diagnostic: page committed chunks out through the kernel's swap path.
    Spill {
        keys: Vec<String>,
    },
    Reset {
        /// Drop leases and pending reservations instead of refusing the reset.
        #[serde(default)]
        force: bool,
    },
    Stats,
    /// Owner-only: retires the namespace now instead of leaving it adoptable.
    Close,
    /// Service-wide state, available before any handshake.
    Info,
}

impl Request {
    /// Wire name of the operation, for logs.
    pub fn op(&self) -> &'static str {
        match self {
            Request::Open { .. } => "open",
            Request::Attach { .. } => "attach",
            Request::Lookup { .. } => "lookup",
            Request::LookupMany { .. } => "lookup_many",
            Request::Load { .. } => "load",
            Request::Prefetch { .. } => "prefetch",
            Request::Release { .. } => "release",
            Request::ReleaseAll { .. } => "release_request",
            Request::Store { .. } => "store",
            Request::Commit { .. } => "commit",
            Request::Touch { .. } => "touch",
            Request::Spill { .. } => "spill",
            Request::Reset { .. } => "reset",
            Request::Stats => "stats",
            Request::Close => "close",
            Request::Info => "info",
        }
    }
}

/// Whether a `store` reply that echoes every reserved and evicted key plus one
/// slot id per reserved key still fits in a single frame.
///
/// `reserved` and `evicted` must already be validated by `cache::check_keys`, so
/// JSON escaping never expands a hex key. Both count: an evicted key was stored
/// by an earlier batch, so it may be longer than any key this batch carries. A
/// reply that cannot fit is rejected before the cache mutates, so the caller can
/// retry with a smaller batch.
pub fn store_reply_fits(reserved: &[&str], evicted: &[&str]) -> bool {
    let echoed = |keys: &[&str]| -> usize { keys.iter().map(|key| key.len() + 3).sum() };
    echoed(reserved) + echoed(evicted) + reserved.len() * SLOT_BYTES + ENVELOPE_BYTES <= MAX_FRAME
}

pub fn read_request(stream: &mut UnixStream) -> io::Result<Request> {
    let mut header = [0; 4];
    read_into(stream, &mut header)?;
    let length = u32::from_be_bytes(header) as usize;
    if length == 0 || length > MAX_FRAME {
        return Err(io::Error::other("invalid frame length"));
    }
    let mut payload = vec![0; length];
    read_into(stream, &mut payload)?;
    serde_json::from_slice(&payload).map_err(io::Error::other)
}

/// Fills `buffer`, reporting an expired receive timeout as `TimedOut` so the
/// caller can tell an idle peer from a malformed frame.
fn read_into(stream: &mut UnixStream, buffer: &mut [u8]) -> io::Result<()> {
    match stream.read_exact(buffer) {
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) =>
        {
            Err(io::Error::new(io::ErrorKind::TimedOut, error))
        }
        result => result,
    }
}

/// Encodes `response` as one length-prefixed frame.
pub fn encode_response(response: &Value) -> io::Result<Vec<u8>> {
    let payload = serde_json::to_vec(response)?;
    if payload.len() > MAX_FRAME {
        return Err(io::Error::other("response exceeds frame limit"));
    }
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Writes one encoded frame without splitting the header from its payload.
pub fn write_frame(stream: &mut UnixStream, frame: &[u8]) -> io::Result<()> {
    stream.write_all(frame)
}

/// Duplicates `fd` so it stays valid after the cache lock is released.
pub fn duplicate_fd(fd: BorrowedFd<'_>) -> io::Result<OwnedFd> {
    // SAFETY: fcntl duplicates a live borrowed descriptor into a new one.
    let duplicated = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
    if duplicated < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fcntl returned a descriptor owned only here.
    Ok(unsafe { OwnedFd::from_raw_fd(duplicated) })
}

pub fn send_fd(stream: &UnixStream, fd: BorrowedFd<'_>) -> io::Result<()> {
    let mut marker = *b"F";
    let mut iov = libc::iovec {
        iov_base: marker.as_mut_ptr().cast(),
        iov_len: 1,
    };
    // usize storage provides the alignment required by cmsghdr.
    let mut control = [0usize; 8];
    // SAFETY: all pointers refer to live, aligned buffers for the duration of
    // sendmsg; SCM_RIGHTS copies the borrowed descriptor into the peer process.
    unsafe {
        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = libc::CMSG_SPACE(std::mem::size_of::<libc::c_int>() as u32) as usize;
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as usize;
        std::ptr::write(libc::CMSG_DATA(cmsg).cast::<libc::c_int>(), fd.as_raw_fd());
        loop {
            let sent = libc::sendmsg(stream.as_raw_fd(), &msg, libc::MSG_NOSIGNAL);
            if sent == 1 {
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            // A zero-length sendmsg leaves errno untouched; the peer is gone.
            return Err(if sent == 0 {
                io::Error::from(io::ErrorKind::BrokenPipe)
            } else {
                error
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_keys(length: usize, count: usize) -> Vec<String> {
        (0..count).map(|index| format!("{index:0length$x}")).collect()
    }

    #[test]
    fn store_reply_bound_covers_reserved_and_evicted_keys() {
        let digest_keys = hex_keys(64, MAX_KEYS);
        let reserved: Vec<&str> = digest_keys.iter().map(String::as_str).collect();
        let reply = serde_json::json!({
            "ok": true,
            "keys": digest_keys,
            "slots": vec![MAX_KEYS - 1; MAX_KEYS],
            "evicted": digest_keys,
        });
        assert!(store_reply_fits(&reserved, &reserved));
        assert!(encode_response(&reply).unwrap().len() <= 4 + MAX_FRAME);

        // A batch of the longest keys that still fits in one request frame can
        // need a reply twice that size.
        let long_keys = hex_keys(250, MAX_KEYS);
        let request = serde_json::json!({"op": "store", "keys": long_keys});
        assert!(serde_json::to_vec(&request).unwrap().len() <= MAX_FRAME);
        let long: Vec<&str> = long_keys.iter().map(String::as_str).collect();
        assert!(!store_reply_fits(&long, &[]));

        // Short keys evicting long ones: the evicted keys alone fill the frame.
        let short = hex_keys(4, MAX_KEYS);
        let short: Vec<&str> = short.iter().map(String::as_str).collect();
        assert!(store_reply_fits(&short, &[]));
        assert!(!store_reply_fits(&short, &long));
    }

    #[test]
    fn lookup_many_reply_at_max_keys_fits_one_frame() {
        let statuses = vec![crate::cache::Status::Pending; MAX_KEYS];
        let reply = serde_json::json!({"ok": true, "statuses": statuses});

        assert!(encode_response(&reply).unwrap().len() <= 4 + MAX_FRAME);
    }
}
