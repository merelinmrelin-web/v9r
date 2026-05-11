//! v9r-orchestrator: subscribes to VFS write events and dispatches
//! registered handlers.
//!
//! The orchestrator owns three pieces of shared state:
//! - `Arc<Vfs>` — the namespace it watches
//! - `Arc<dyn AgentRuntime>` — the runtime invocations are routed to
//! - `Arc<RwLock<WatchTable>>` — routes from path patterns to handlers
//!
//! Handlers are looked up under a brief read lock; mutation (e.g. when
//! `ManifestHandler` registers a new agent's trigger routes) takes the
//! write lock from inside a spawned task. The main loop never holds the
//! lock across `.await`.

pub mod handlers;
pub mod manifest;
pub mod watch;

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{
    broadcast::{error::RecvError, error::TryRecvError, Receiver},
    RwLock,
};

use v9r_core::{CapabilityId, VfsPath};
use v9r_runtime::AgentRuntime;
use v9r_vfs::{Vfs, WriteEvent};

pub use handlers::{AgentTriggerHandler, ManifestHandler};
pub use manifest::{
    AgentRecord, AgentRegistry, AgentSpec, Manifest, ManifestError, MountModeSpec, MountSpec,
    TriggerSpec, TrustLevel,
};
pub use watch::{PathPattern, Route, WatchTable};

#[async_trait]
pub trait TriggerHandler: Send + Sync {
    async fn handle(
        &self,
        ev: WriteEvent,
        vfs: Arc<Vfs>,
        rt: Arc<dyn AgentRuntime>,
    ) -> anyhow::Result<()>;
}

pub struct Orchestrator {
    vfs: Arc<Vfs>,
    rt: Arc<dyn AgentRuntime>,
    watch: Arc<RwLock<WatchTable>>,
    /// Pre-subscribed at construction so events that fire before the
    /// run() task is first polled aren't dropped on the floor —
    /// `broadcast::Sender::send` discards messages when there are zero
    /// live receivers.
    rx: Receiver<WriteEvent>,
}

impl Orchestrator {
    pub fn new(vfs: Arc<Vfs>, rt: Arc<dyn AgentRuntime>, watch: Arc<RwLock<WatchTable>>) -> Self {
        let rx = vfs.bus().subscribe();
        Self { vfs, rt, watch, rx }
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        loop {
            match self.rx.recv().await {
                Ok(ev) => {
                    self.spawn_dispatch(ev).await;
                }
                Err(RecvError::Lagged(n)) => {
                    tracing::warn!(missed = n, "orchestrator lagged");
                }
                Err(RecvError::Closed) => break,
            }
        }
        Ok(())
    }

    pub async fn step(&mut self) -> anyhow::Result<bool> {
        match self.rx.try_recv() {
            Ok(ev) => {
                tracing::debug!(path = %ev.path, "orchestrator.step: received write event");
                self.dispatch(ev).await?;
                Ok(true)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Lagged(n)) => {
                tracing::warn!(missed = n, "orchestrator lagged");
                Ok(true)
            }
            Err(TryRecvError::Closed) => Ok(false),
        }
    }

    pub async fn load_all_from_vfs(&self, root: &str) -> anyhow::Result<usize> {
        let root = VfsPath::parse(root)?;
        let Some(host_root) = self.vfs.host_path_for(&root).await else {
            tracing::warn!(root = %root, "manifest auto-load skipped: no host mount for VFS path");
            return Ok(0);
        };
        let mut stack = vec![host_root.clone()];
        let mut loaded = 0;

        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(e) => {
                    tracing::warn!(host_path = %dir.display(), error = ?e, "manifest auto-load scan failed");
                    continue;
                }
            };
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(e) => {
                        tracing::warn!(host_path = %dir.display(), error = ?e, "manifest auto-load entry failed");
                        continue;
                    }
                };
                let host_path = match std::fs::canonicalize(entry.path()) {
                    Ok(path) => path,
                    Err(e) => {
                        tracing::warn!(host_path = %entry.path().display(), error = ?e, "manifest auto-load canonicalize failed");
                        continue;
                    }
                };
                if host_path.is_dir() {
                    stack.push(host_path);
                    continue;
                }
                if host_path.file_name().and_then(|s| s.to_str()) == Some("manifest.toml") {
                    let path = host_to_vfs_path(&root, &host_root, &host_path)?;
                    if let Some(parent) = path.parent() {
                        self.vfs.mkdir_p(&parent).await?;
                    }
                    let bytes = self.vfs.read(&v9r_core::Capability::root(), &path).await?;
                    let ev = WriteEvent {
                        path: path.clone(),
                        node: self.vfs.root_id(),
                        bytes_len: bytes.len(),
                        writer: CapabilityId(0),
                    };
                    match self.dispatch(ev).await {
                        Ok(()) => loaded += 1,
                        Err(e) => {
                            tracing::warn!(path = %path, host_path = %host_path.display(), error = ?e, "manifest auto-load failed")
                        }
                    }
                }
            }
        }

        Ok(loaded)
    }

    async fn spawn_dispatch(&self, ev: WriteEvent) {
        let handler = {
            let guard = self.watch.read().await;
            guard.match_path(&ev.path)
        };
        if let Some(h) = handler {
            let vfs = self.vfs.clone();
            let rt = self.rt.clone();
            tokio::spawn(async move {
                if let Err(e) = h.handle(ev, vfs, rt).await {
                    tracing::warn!(error = ?e, "trigger handler failed");
                }
            });
        }
    }

    async fn dispatch(&self, ev: WriteEvent) -> anyhow::Result<()> {
        let handler = {
            let guard = self.watch.read().await;
            guard.match_path(&ev.path)
        };
        if let Some(h) = handler {
            let export = if ev.path.last() == Some("input") {
                "on_input"
            } else {
                "unknown"
            };
            tracing::debug!(
                path = %ev.path,
                export,
                "orchestrator.step: dispatching trigger handler"
            );
            h.handle(ev, self.vfs.clone(), self.rt.clone()).await?;
            tracing::debug!(export, "orchestrator.step: trigger handler completed");
        } else {
            tracing::debug!(path = %ev.path, "orchestrator.step: no trigger handler matched");
        }
        Ok(())
    }
}

fn host_to_vfs_path(root: &VfsPath, host_root: &Path, host_path: &Path) -> anyhow::Result<VfsPath> {
    let rel = host_path.strip_prefix(host_root)?;
    let mut segments: Vec<String> = root.segments().to_vec();
    for part in rel.components() {
        let std::path::Component::Normal(seg) = part else {
            continue;
        };
        segments.push(seg.to_string_lossy().to_string());
    }
    Ok(VfsPath::from_segments(segments)?)
}
