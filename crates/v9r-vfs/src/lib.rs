//! v9r-vfs: in-memory virtual file system + write-event bus.
//!
//! Design notes
//! - One `RwLock` over a `SlotMap` arena. Sufficient for the prototype;
//!   shard later if contention shows up.
//! - The VFS knows nothing about Wasm. It emits `WriteEvent`s; the
//!   orchestrator decides what to do with them.
//! - Drop-Lock-Before-Await: synthetic write/read callbacks are async, so
//!   we always release the arena guard before invoking them.

mod bus;
mod log;
mod node;

pub use bus::{WriteBus, WriteEvent};
pub use log::{LogBuffer, DEFAULT_LOG_BUFFER_BYTES};
pub use node::{Node, NodeKind, SyntheticFile, VfsCtx, WriteOutcome};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use bytes::Bytes;
use slotmap::SlotMap;
use tokio::sync::RwLock;

use v9r_core::{Capability, NodeId, VfsError, VfsPath, VfsResult};

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub kind: EntryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    Dir,
    File,
    Synthetic,
}

pub struct Vfs {
    arena: RwLock<SlotMap<NodeId, Node>>,
    host_mounts: RwLock<Vec<HostMount>>,
    root: NodeId,
    bus: WriteBus,
}

#[derive(Clone, Debug)]
pub struct HostMount {
    pub virtual_path: VfsPath,
    pub host_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct VfsStat {
    pub kind: EntryKind,
    pub len: u64,
}

impl Vfs {
    pub fn new() -> Arc<Self> {
        Self::with_capacity(64)
    }

    pub fn with_capacity(cap: usize) -> Arc<Self> {
        let mut arena: SlotMap<NodeId, Node> = SlotMap::with_capacity_and_key(cap);
        let root = arena.insert(Node::dir_root());
        Arc::new(Self {
            arena: RwLock::new(arena),
            host_mounts: RwLock::new(Vec::new()),
            root,
            bus: WriteBus::new(1024),
        })
    }

    pub fn bus(&self) -> &WriteBus {
        &self.bus
    }

    pub fn root_id(&self) -> NodeId {
        self.root
    }

    pub async fn mount_host(&self, virtual_path: &VfsPath, host_path: &Path) -> VfsResult<()> {
        let host_path = std::fs::canonicalize(host_path)?;
        self.mkdir_p(virtual_path).await?;
        let mut mounts = self.host_mounts.write().await;
        mounts.retain(|m| m.virtual_path != *virtual_path);
        mounts.push(HostMount {
            virtual_path: virtual_path.clone(),
            host_path,
        });
        mounts.sort_by_key(|m| std::cmp::Reverse(m.virtual_path.segments().len()));
        Ok(())
    }

    pub async fn host_path_for(&self, path: &VfsPath) -> Option<PathBuf> {
        let mounts = self.host_mounts.read().await;
        let path_segments = path.segments();
        for mount in mounts.iter() {
            let mount_segments = mount.virtual_path.segments();
            if path_segments.len() < mount_segments.len()
                || !path_segments.starts_with(mount_segments)
            {
                continue;
            }
            let mut host = mount.host_path.clone();
            for seg in &path_segments[mount_segments.len()..] {
                host.push(seg);
            }
            return match std::fs::canonicalize(&host) {
                Ok(canonical) => {
                    if canonical.starts_with(&mount.host_path) {
                        tracing::debug!(
                            "VFS Access: [{}] -> Resolved to Host: [{}]",
                            path,
                            canonical.display()
                        );
                        Some(canonical)
                    } else {
                        None
                    }
                }
                Err(_) => None,
            };
        }
        None
    }

    fn resolve(&self, arena: &SlotMap<NodeId, Node>, path: &VfsPath) -> VfsResult<NodeId> {
        let mut cur = self.root;
        for seg in path.segments() {
            match &arena[cur].kind {
                NodeKind::Dir { children } => {
                    cur = *children.get(seg).ok_or(VfsError::NotFound)?;
                }
                _ => return Err(VfsError::NotADir),
            }
        }
        Ok(cur)
    }

    pub async fn mkdir_p(&self, path: &VfsPath) -> VfsResult<NodeId> {
        let mut arena = self.arena.write().await;
        let mut cur = self.root;
        for seg in path.segments() {
            let existing = match &arena[cur].kind {
                NodeKind::Dir { children } => children.get(seg).copied(),
                _ => return Err(VfsError::NotADir),
            };
            cur = match existing {
                Some(id) => id,
                None => {
                    let id = arena.insert(Node::dir(seg.clone(), Some(cur)));
                    if let NodeKind::Dir { children } = &mut arena[cur].kind {
                        children.insert(seg.clone(), id);
                    }
                    id
                }
            };
        }
        Ok(cur)
    }

    /// Create a new file at `path`. If `content` is non-empty, a `WriteEvent`
    /// is emitted so handlers can react to "file appeared with content"
    /// the same way they react to writes. Empty creation is silent — useful
    /// when a handler is bootstrapping placeholder nodes.
    pub async fn create_file(&self, path: &VfsPath, content: Bytes) -> VfsResult<NodeId> {
        let parent_path = path
            .parent()
            .ok_or_else(|| VfsError::InvalidPath("/".into()))?;
        let name = path
            .last()
            .ok_or_else(|| VfsError::InvalidPath("/".into()))?
            .to_string();

        let id = {
            let mut arena = self.arena.write().await;
            let parent_id = self.resolve(&arena, &parent_path)?;
            match &arena[parent_id].kind {
                NodeKind::Dir { children } => {
                    if children.contains_key(&name) {
                        return Err(VfsError::AlreadyExists);
                    }
                }
                _ => return Err(VfsError::NotADir),
            }
            let id = arena.insert(Node::file(name.clone(), Some(parent_id), content.to_vec()));
            if let NodeKind::Dir { children } = &mut arena[parent_id].kind {
                children.insert(name, id);
            }
            id
        };

        if !content.is_empty() {
            self.bus.publish(WriteEvent {
                path: path.clone(),
                node: id,
                bytes_len: content.len(),
                writer: v9r_core::CapabilityId(0),
            });
        }

        Ok(id)
    }

    pub async fn create_synthetic(
        &self,
        path: &VfsPath,
        syn: Arc<dyn SyntheticFile>,
    ) -> VfsResult<NodeId> {
        let parent_path = path
            .parent()
            .ok_or_else(|| VfsError::InvalidPath("/".into()))?;
        let name = path
            .last()
            .ok_or_else(|| VfsError::InvalidPath("/".into()))?
            .to_string();

        let mut arena = self.arena.write().await;
        let parent_id = self.resolve(&arena, &parent_path)?;
        match &arena[parent_id].kind {
            NodeKind::Dir { children } => {
                if children.contains_key(&name) {
                    return Err(VfsError::AlreadyExists);
                }
            }
            _ => return Err(VfsError::NotADir),
        }
        let id = arena.insert(Node {
            name: name.clone(),
            parent: Some(parent_id),
            mtime: SystemTime::now(),
            kind: NodeKind::Synthetic(syn),
        });
        if let NodeKind::Dir { children } = &mut arena[parent_id].kind {
            children.insert(name, id);
        }
        Ok(id)
    }

    pub async fn upsert_synthetic(
        &self,
        path: &VfsPath,
        syn: Arc<dyn SyntheticFile>,
    ) -> VfsResult<NodeId> {
        let parent_path = path
            .parent()
            .ok_or_else(|| VfsError::InvalidPath("/".into()))?;
        let name = path
            .last()
            .ok_or_else(|| VfsError::InvalidPath("/".into()))?
            .to_string();

        let mut arena = self.arena.write().await;
        let parent_id = self.resolve(&arena, &parent_path)?;
        let existing = match &arena[parent_id].kind {
            NodeKind::Dir { children } => children.get(&name).copied(),
            _ => return Err(VfsError::NotADir),
        };

        if let Some(id) = existing {
            if matches!(arena[id].kind, NodeKind::Dir { .. }) {
                return Err(VfsError::IsDir);
            }
            arena[id].kind = NodeKind::Synthetic(syn);
            arena[id].mtime = SystemTime::now();
            return Ok(id);
        }

        let id = arena.insert(Node {
            name: name.clone(),
            parent: Some(parent_id),
            mtime: SystemTime::now(),
            kind: NodeKind::Synthetic(syn),
        });
        if let NodeKind::Dir { children } = &mut arena[parent_id].kind {
            children.insert(name, id);
        }
        Ok(id)
    }

    pub async fn list(&self, path: &VfsPath) -> VfsResult<Vec<DirEntry>> {
        let arena = self.arena.read().await;
        let mut out = match self.resolve(&arena, path) {
            Ok(id) => match &arena[id].kind {
                NodeKind::Dir { children } => children
                    .iter()
                    .map(|(name, &cid)| DirEntry {
                        name: name.clone(),
                        kind: match &arena[cid].kind {
                            NodeKind::Dir { .. } => EntryKind::Dir,
                            NodeKind::File { .. } => EntryKind::File,
                            NodeKind::Synthetic(_) => EntryKind::Synthetic,
                        },
                    })
                    .collect(),
                _ => return Err(VfsError::NotADir),
            },
            Err(VfsError::NotFound) => Vec::new(),
            Err(e) => return Err(e),
        };
        drop(arena);

        if let Some(host_path) = self.host_path_for(path).await {
            let metadata = std::fs::metadata(&host_path)?;
            if !metadata.is_dir() {
                return if out.is_empty() {
                    Err(VfsError::NotADir)
                } else {
                    Ok(out)
                };
            }
            for entry in std::fs::read_dir(&host_path)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().to_string();
                if out.iter().any(|e| e.name == name) {
                    continue;
                }
                let ty = entry.file_type()?;
                out.push(DirEntry {
                    name,
                    kind: if ty.is_dir() {
                        EntryKind::Dir
                    } else {
                        EntryKind::File
                    },
                });
            }
        }

        if out.is_empty() && self.host_path_for(path).await.is_none() {
            return Err(VfsError::NotFound);
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    pub async fn read(&self, cap: &Capability, path: &VfsPath) -> VfsResult<Bytes> {
        // Drop-Lock-Before-Await: the synthetic read is async, so we snapshot
        // (or copy bytes) under the guard, drop it, then await outside.
        enum What {
            File(Bytes),
            Synthetic(Arc<dyn SyntheticFile>),
        }

        let what = {
            let arena = self.arena.read().await;
            let id = match self.resolve(&arena, path) {
                Ok(id) => id,
                Err(VfsError::NotFound) => {
                    drop(arena);
                    return self.read_host(path).await;
                }
                Err(e) => return Err(e),
            };
            match &arena[id].kind {
                NodeKind::File { content } => What::File(Bytes::copy_from_slice(content)),
                NodeKind::Synthetic(syn) => What::Synthetic(syn.clone()),
                NodeKind::Dir { .. } => return Err(VfsError::IsDir),
            }
        };

        match what {
            What::File(b) => Ok(b),
            What::Synthetic(syn) => {
                let ctx = VfsCtx {
                    path: path.clone(),
                    writer: cap.id(),
                };
                syn.read(&ctx).await
            }
        }
    }

    async fn read_host(&self, path: &VfsPath) -> VfsResult<Bytes> {
        let host_path = self.host_path_for(path).await.ok_or(VfsError::NotFound)?;
        let metadata = std::fs::metadata(&host_path)?;
        if metadata.is_dir() {
            return Err(VfsError::IsDir);
        }
        Ok(Bytes::from(std::fs::read(host_path)?))
    }

    pub async fn stat(&self, path: &VfsPath) -> VfsResult<VfsStat> {
        let arena = self.arena.read().await;
        match self.resolve(&arena, path) {
            Ok(id) => match &arena[id].kind {
                NodeKind::Dir { .. } => Ok(VfsStat {
                    kind: EntryKind::Dir,
                    len: 0,
                }),
                NodeKind::File { content } => Ok(VfsStat {
                    kind: EntryKind::File,
                    len: content.len() as u64,
                }),
                NodeKind::Synthetic(_) => Ok(VfsStat {
                    kind: EntryKind::Synthetic,
                    len: 0,
                }),
            },
            Err(VfsError::NotFound) => {
                drop(arena);
                let host_path = self.host_path_for(path).await.ok_or(VfsError::NotFound)?;
                let metadata = std::fs::metadata(host_path)?;
                Ok(VfsStat {
                    kind: if metadata.is_dir() {
                        EntryKind::Dir
                    } else {
                        EntryKind::File
                    },
                    len: metadata.len(),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Write `data` to `path`. Plain Files store the bytes and emit a
    /// `WriteEvent`; Synthetic nodes are dispatched via their trait, and
    /// the bus is only fired if they ask for it.
    pub async fn write(&self, cap: &Capability, path: &VfsPath, data: Bytes) -> VfsResult<()> {
        // Decide what to do under the lock; do the awaiting work after.
        enum Step {
            Emit(WriteEvent),
            CallSynthetic(Arc<dyn SyntheticFile>),
        }

        let step = {
            let mut arena = self.arena.write().await;
            let id = self.resolve(&arena, path)?;
            match &mut arena[id].kind {
                NodeKind::File { content } => {
                    let bytes_len = data.len();
                    *content = data.to_vec();
                    arena[id].mtime = SystemTime::now();
                    Step::Emit(WriteEvent {
                        path: path.clone(),
                        node: id,
                        bytes_len,
                        writer: cap.id(),
                    })
                }
                NodeKind::Synthetic(syn) => Step::CallSynthetic(syn.clone()),
                NodeKind::Dir { .. } => return Err(VfsError::IsDir),
            }
        }; // arena guard dropped here

        match step {
            Step::Emit(ev) => self.bus.publish(ev),
            Step::CallSynthetic(syn) => {
                let ctx = VfsCtx {
                    path: path.clone(),
                    writer: cap.id(),
                };
                match syn.write(&ctx, data).await? {
                    WriteOutcome::Stored => {}
                    WriteOutcome::Emit(ev) => self.bus.publish(ev),
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mkdir_and_list() {
        let vfs = Vfs::new();
        vfs.mkdir_p(&VfsPath::parse("/agents/echo").unwrap())
            .await
            .unwrap();
        let entries = vfs.list(&VfsPath::parse("/agents").unwrap()).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "echo");
        assert_eq!(entries[0].kind, EntryKind::Dir);
    }

    #[tokio::test]
    async fn write_emits_event() {
        let vfs = Vfs::new();
        let cap = Capability::root();
        vfs.mkdir_p(&VfsPath::parse("/a").unwrap()).await.unwrap();
        vfs.create_file(&VfsPath::parse("/a/f").unwrap(), Bytes::new())
            .await
            .unwrap();

        let mut rx = vfs.bus().subscribe();
        vfs.write(
            &cap,
            &VfsPath::parse("/a/f").unwrap(),
            Bytes::from_static(b"hi"),
        )
        .await
        .unwrap();

        let ev = rx.recv().await.unwrap();
        assert_eq!(ev.path, VfsPath::parse("/a/f").unwrap());
        assert_eq!(ev.bytes_len, 2);
    }

    #[tokio::test]
    async fn synthetic_drop_lock_before_await() {
        // A synthetic that re-enters the VFS during its write callback would
        // deadlock if we held the arena lock across .await. This exercises
        // that path: the synthetic awaits, then the host's write returns.
        struct EchoSyn {
            seen: tokio::sync::Mutex<Vec<Bytes>>,
        }
        #[async_trait::async_trait]
        impl SyntheticFile for EchoSyn {
            async fn read(&self, _ctx: &VfsCtx) -> VfsResult<Bytes> {
                Ok(Bytes::new())
            }
            async fn write(&self, _ctx: &VfsCtx, data: Bytes) -> VfsResult<WriteOutcome> {
                tokio::task::yield_now().await;
                self.seen.lock().await.push(data);
                Ok(WriteOutcome::Stored)
            }
        }

        let vfs = Vfs::new();
        let cap = Capability::root();
        vfs.mkdir_p(&VfsPath::parse("/x").unwrap()).await.unwrap();
        let syn = Arc::new(EchoSyn {
            seen: Default::default(),
        });
        vfs.create_synthetic(&VfsPath::parse("/x/sink").unwrap(), syn.clone())
            .await
            .unwrap();
        vfs.write(
            &cap,
            &VfsPath::parse("/x/sink").unwrap(),
            Bytes::from_static(b"hi"),
        )
        .await
        .unwrap();
        assert_eq!(syn.seen.lock().await.len(), 1);
    }
}
