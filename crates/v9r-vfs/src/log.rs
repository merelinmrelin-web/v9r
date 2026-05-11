use std::sync::Mutex;

use async_trait::async_trait;
use bytes::Bytes;

use v9r_core::VfsResult;

use crate::node::{SyntheticFile, VfsCtx, WriteOutcome};

pub const DEFAULT_LOG_BUFFER_BYTES: usize = 128 * 1024;

#[derive(Debug)]
pub struct LogBuffer {
    inner: Mutex<LogBufferInner>,
}

#[derive(Debug)]
struct LogBufferInner {
    buf: Vec<u8>,
    start: usize,
    len: usize,
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_LOG_BUFFER_BYTES)
    }
}

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LogBufferInner {
                buf: vec![0; capacity],
                start: 0,
                len: 0,
            }),
        }
    }

    pub fn append(&self, data: &[u8]) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        inner.append(data);
    }

    pub fn snapshot(&self) -> Bytes {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Bytes::from(inner.snapshot())
    }
}

impl LogBufferInner {
    fn append(&mut self, data: &[u8]) {
        let capacity = self.buf.len();
        if capacity == 0 || data.is_empty() {
            return;
        }

        if data.len() >= capacity {
            self.buf.copy_from_slice(&data[data.len() - capacity..]);
            self.start = 0;
            self.len = capacity;
            return;
        }

        for &byte in data {
            if self.len < capacity {
                let idx = (self.start + self.len) % capacity;
                self.buf[idx] = byte;
                self.len += 1;
            } else {
                self.buf[self.start] = byte;
                self.start = (self.start + 1) % capacity;
            }
        }
    }

    fn snapshot(&self) -> Vec<u8> {
        if self.len == 0 {
            return Vec::new();
        }

        let capacity = self.buf.len();
        let end = (self.start + self.len) % capacity;
        if self.start < end {
            return self.buf[self.start..end].to_vec();
        }

        let mut out = Vec::with_capacity(self.len);
        out.extend_from_slice(&self.buf[self.start..]);
        out.extend_from_slice(&self.buf[..end]);
        out
    }
}

#[async_trait]
impl SyntheticFile for LogBuffer {
    async fn read(&self, _ctx: &VfsCtx) -> VfsResult<Bytes> {
        Ok(self.snapshot())
    }

    async fn write(&self, _ctx: &VfsCtx, data: Bytes) -> VfsResult<WriteOutcome> {
        self.append(&data);
        Ok(WriteOutcome::Stored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use v9r_core::{CapabilityId, VfsPath};

    #[test]
    fn snapshot_returns_current_buffer() {
        let log = LogBuffer::new(16);
        log.append(b"hello");
        log.append(b" world");

        assert_eq!(&log.snapshot()[..], b"hello world");
    }

    #[test]
    fn append_wraps_as_ring_buffer() {
        let log = LogBuffer::new(8);
        log.append(b"abcdef");
        log.append(b"ghij");

        assert_eq!(&log.snapshot()[..], b"cdefghij");
    }

    #[test]
    fn oversized_append_keeps_tail() {
        let log = LogBuffer::new(5);
        log.append(b"abcdefgh");

        assert_eq!(&log.snapshot()[..], b"defgh");
    }

    #[tokio::test]
    async fn synthetic_write_appends() {
        let log = LogBuffer::new(16);
        let ctx = VfsCtx {
            path: VfsPath::parse("/log").unwrap(),
            writer: CapabilityId(0),
        };

        log.write(&ctx, Bytes::from_static(b"line\n"))
            .await
            .unwrap();

        assert_eq!(&log.read(&ctx).await.unwrap()[..], b"line\n");
    }
}
