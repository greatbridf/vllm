// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, BorrowedFd};
use std::os::unix::net::UnixStream;

use serde::Deserialize;
use serde_json::Value;

pub const MAX_FRAME: usize = 1024 * 1024;
pub const MAX_KEYS: usize = 4096;

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
    Load {
        keys: Vec<String>,
        request: String,
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
    },
    Commit {
        keys: Vec<String>,
        success: bool,
    },
    Touch {
        keys: Vec<String>,
    },
    Reset,
    Stats,
}

pub fn read_request(stream: &mut UnixStream) -> io::Result<Request> {
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let length = u32::from_be_bytes(header) as usize;
    if length == 0 || length > MAX_FRAME {
        return Err(io::Error::other("invalid frame length"));
    }
    let mut payload = vec![0; length];
    stream.read_exact(&mut payload)?;
    serde_json::from_slice(&payload).map_err(io::Error::other)
}

pub fn write_response(stream: &mut UnixStream, response: &Value) -> io::Result<()> {
    let payload = serde_json::to_vec(response)?;
    if payload.len() > MAX_FRAME {
        return Err(io::Error::other("response exceeds frame limit"));
    }
    stream.write_all(&(payload.len() as u32).to_be_bytes())?;
    stream.write_all(&payload)
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
        let sent = libc::sendmsg(stream.as_raw_fd(), &msg, libc::MSG_NOSIGNAL);
        if sent != 1 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}
