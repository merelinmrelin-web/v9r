use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use bytes::Bytes;

use v9r_core::{CapabilityId, NodeId, VfsPath, VfsResult};

use crate::bus::WriteEvent;

/// One node in the VFS arena.
pub struct Node {
    pub name: String,
    pub parent: Option<NodeId>,
    pub mtime: SystemTime,
    pub kind: NodeKind,
}

pub enum NodeKind {
    Dir { children: HashMap<String, NodeId> },
    File { content: Vec<u8> },
    Synthetic(Arc<dyn SyntheticFile>),
}

impl Node {
    pub fn dir_root() -> Self {
        Self {
            name: String::new(),
            parent: None,
            mtime: SystemTime::now(),
            kind: NodeKind::Dir {
                children: HashMap::new(),
            },
        }
    }

    pub fn dir(name: String, parent: Option<NodeId>) -> Self {
        Self {
            name,
            parent,
            mtime: SystemTime::now(),
            kind: NodeKind::Dir {
                children: HashMap::new(),
            },
        }
    }

    pub fn file(name: String, parent: Option<NodeId>, content: Vec<u8>) -> Self {
        Self {
            name,
            parent,
            mtime: SystemTime::now(),
            kind: NodeKind::File { content },
        }
    }
}

/// Context passed to synthetic file callbacks. Lets a synthetic node know who
/// is calling and at what path it's mounted (a single SyntheticFile impl can
/// be mounted at multiple locations in principle).
#[derive(Clone, Debug)]
pub struct VfsCtx {
    pub path: VfsPath,
    pub writer: CapabilityId,
}

/// What a synthetic write produced. `Stored` = handled internally;
/// `Emit` = please broadcast this on the write bus.
pub enum WriteOutcome {
    Stored,
    Emit(WriteEvent),
}

/// A file-like node whose semantics are defined by code rather than bytes.
/// Used for /input, /output, /ctl, /state, etc.
#[async_trait]
pub trait SyntheticFile: Send + Sync {
    async fn read(&self, ctx: &VfsCtx) -> VfsResult<Bytes>;
    async fn write(&self, ctx: &VfsCtx, data: Bytes) -> VfsResult<WriteOutcome>;
}
