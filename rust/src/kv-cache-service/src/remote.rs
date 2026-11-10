// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Read-only remote rows, using the same page keys as the rswap kernel.

use std::io;
use std::sync::Mutex;
use std::time::Instant;

use crate::hash::PageHash;

/// A remote tier returns a complete packed physical row or a miss.
pub trait RemoteTier: Send + Sync {
    fn load(
        &self,
        key: &str,
        row_bytes: usize,
        page_size: usize,
        deadline: Instant,
    ) -> io::Result<Option<Vec<u8>>>;
}

/// Redis rswap objects: lowercase 96-character page digests and raw base pages.
pub struct RedisTier {
    client: redis::Client,
    state: Mutex<RedisState>,
}

struct RedisState {
    runtime: tokio::runtime::Runtime,
    connection: Option<redis::aio::MultiplexedConnection>,
}

impl RedisTier {
    /// Parses a Redis URL without connecting or logging its credentials.
    pub fn new(url: &str) -> io::Result<Self> {
        let client = redis::Client::open(url).map_err(io::Error::other)?;
        Ok(Self {
            client,
            state: Mutex::new(RedisState {
                runtime: tokio::runtime::Builder::new_current_thread().enable_all().build()?,
                connection: None,
            }),
        })
    }
}

impl RemoteTier for RedisTier {
    fn load(
        &self,
        key: &str,
        row_bytes: usize,
        page_size: usize,
        deadline: Instant,
    ) -> io::Result<Option<Vec<u8>>> {
        if page_size == 0 || row_bytes == 0 || !row_bytes.is_multiple_of(page_size) {
            return Err(io::Error::other("invalid remote row geometry"));
        }
        // Never queue behind another namespace's network I/O. A busy tier is a
        // miss, not an unbounded wait in the scheduler's control RPC.
        let Ok(mut state) = self.state.try_lock() else {
            return Ok(None);
        };
        let RedisState {
            runtime,
            connection,
        } = &mut *state;
        let read = async {
            if connection.is_none() {
                *connection = Some(
                    self.client
                        .get_multiplexed_async_connection()
                        .await
                        .map_err(io::Error::other)?,
                );
            }
            let connection = connection.as_mut().expect("connection opened");
            let mut row = Vec::with_capacity(row_bytes);
            // Match rswap's bounded pipelined GET batches; do not issue a
            // multi-megabyte row as thousands of serial round trips.
            for first in (0..row_bytes / page_size).step_by(32) {
                let mut pipeline = redis::pipe();
                let end = (first + 32).min(row_bytes / page_size);
                for page in first..end {
                    pipeline.cmd("GET").arg(hex::encode(PageHash::new(key, page as u64).hash()));
                }
                let pages: Vec<Option<Vec<u8>>> =
                    pipeline.query_async(connection).await.map_err(io::Error::other)?;
                if pages.len() != end - first {
                    return Err(io::Error::other("Redis returned an incomplete page batch"));
                }
                for page in pages {
                    let Some(page) = page else {
                        return Ok(None);
                    };
                    if page.len() != page_size {
                        return Err(io::Error::other("Redis page size does not match the host"));
                    }
                    row.extend_from_slice(&page);
                }
            }
            Ok(Some(row))
        };
        // One deadline includes DNS, authentication, writes, and every reply,
        // even when a peer continuously trickles bytes into the socket.
        let result = runtime.block_on(async {
            tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), read)
                .await
                .map_err(|_| {
                    io::Error::new(io::ErrorKind::TimedOut, "remote lookup deadline expired")
                })?
        });
        if result.is_err() {
            // A failed pipeline may leave unread replies. Never reuse it.
            *connection = None;
        }
        result
    }
}
