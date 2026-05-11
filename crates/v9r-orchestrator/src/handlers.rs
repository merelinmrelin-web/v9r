//! Built-in trigger handlers: `ManifestHandler` keeps the registry and
//! watch table in sync; `AgentTriggerHandler` invokes the agent runtime.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::RwLock;

use v9r_cap::{HostCapability, NamespaceView};
use v9r_core::{Capability, VfsError, VfsPath};
use v9r_runtime::{AgentId, AgentRuntime};
use v9r_vfs::{LogBuffer, SyntheticFile, Vfs, WriteEvent};

use crate::manifest::{AgentRecord, AgentRegistry, Manifest, TrustLevel};
use crate::watch::{PathPattern, WatchTable};
use crate::TriggerHandler;

/// Watches `/agents/*/manifest.toml`. On every write:
/// 1. parses & validates the manifest
/// 2. pre-creates trigger files (empty, so no event fires) and `output`
/// 3. upserts the registry
/// 4. registers/replaces per-trigger routes in the watch table
pub struct ManifestHandler {
    registry: Arc<AgentRegistry>,
    watch: Arc<RwLock<WatchTable>>,
}

impl ManifestHandler {
    pub fn new(registry: Arc<AgentRegistry>, watch: Arc<RwLock<WatchTable>>) -> Self {
        Self { registry, watch }
    }
}

#[async_trait]
impl TriggerHandler for ManifestHandler {
    async fn handle(
        &self,
        ev: WriteEvent,
        vfs: Arc<Vfs>,
        rt: Arc<dyn AgentRuntime>,
    ) -> anyhow::Result<()> {
        let agent_dir = ev
            .path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("manifest has no parent dir"))?;
        let agent_name = agent_dir
            .last()
            .ok_or_else(|| anyhow::anyhow!("agent dir has no name"))?
            .to_string();

        let bytes = vfs.read(&Capability::root(), &ev.path).await?;
        let s =
            std::str::from_utf8(&bytes).map_err(|e| anyhow::anyhow!("manifest not utf-8: {e}"))?;
        let manifest = Manifest::parse(s)?;
        manifest.validate(&agent_name)?;

        // Bootstrap trigger files + output. Empty creations are
        // silent (see Vfs::create_file), so no spurious events fire.
        // AlreadyExists is fine: the user (or a previous manifest write)
        // may have created them.
        for t in &manifest.triggers {
            let p = agent_dir.join(&t.path);
            match vfs.create_file(&p, Bytes::new()).await {
                Ok(_) | Err(VfsError::AlreadyExists) => {}
                Err(e) => return Err(e.into()),
            }
        }
        let output_path = agent_dir.join("output");
        match vfs.create_file(&output_path, Bytes::new()).await {
            Ok(_) | Err(VfsError::AlreadyExists) => {}
            Err(e) => return Err(e.into()),
        }

        let log_buffer = Arc::new(LogBuffer::default());
        let log_node: Arc<dyn SyntheticFile> = log_buffer.clone();
        vfs.upsert_synthetic(&agent_dir.join("log"), log_node)
            .await?;

        // If the manifest references a wasm module, read its bytes from
        // the VFS (relative to the agent dir) and hand them to the
        // runtime. The runtime caches by AgentId so this is also the
        // hot-reload path: rewriting manifest.toml swaps the module.
        let id = AgentId::new(agent_name.clone());
        if let Some(module_path) = &manifest.agent.module {
            // module_path is relative to agent_dir.
            let rel = VfsPath::parse(module_path)?;
            let mut full = agent_dir.clone();
            for seg in rel.segments() {
                full = full.join(seg);
            }
            let wasm = vfs.read(&Capability::root(), &full).await?;
            rt.load(&id, wasm).await?;
        } else {
            // No module declared — make sure any previously-cached one
            // is gone so stale code can't be invoked.
            rt.unload(&id).await;
        }

        self.registry
            .upsert(
                id.clone(),
                AgentRecord {
                    manifest: manifest.clone(),
                    agent_dir: agent_dir.clone(),
                    log_buffer: log_buffer.clone(),
                },
            )
            .await;

        // Replace this agent's routes atomically. Old routes (if any) are
        // removed by label prefix; new ones go in fresh.
        let label_prefix = format!("agent:{agent_name}:");
        let mut watch = self.watch.write().await;
        watch.unregister_prefix(&label_prefix);
        for t in &manifest.triggers {
            let trigger_path = agent_dir.join(&t.path);
            let pattern = PathPattern::exact(&trigger_path);
            let label = format!("{label_prefix}{}", t.path);
            let handler: Arc<dyn TriggerHandler> = Arc::new(AgentTriggerHandler {
                registry: self.registry.clone(),
                export: t.export.clone(),
            });
            watch.register(label, pattern, handler);
        }

        tracing::info!(
            agent = %agent_name,
            mounts = manifest.mounts.len(),
            triggers = manifest.triggers.len(),
            "agent registered"
        );
        Ok(())
    }
}

/// Invokes an agent's WASM export when a trigger file is written.
/// Built per-trigger; the manifest may produce several (one per
/// `[[triggers]]` entry) bound to the same registry but different exports.
pub struct AgentTriggerHandler {
    pub registry: Arc<AgentRegistry>,
    pub export: String,
}

#[async_trait]
impl TriggerHandler for AgentTriggerHandler {
    async fn handle(
        &self,
        ev: WriteEvent,
        vfs: Arc<Vfs>,
        rt: Arc<dyn AgentRuntime>,
    ) -> anyhow::Result<()> {
        let agent_dir = ev
            .path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("trigger has no parent dir"))?;
        let agent_name = agent_dir
            .last()
            .ok_or_else(|| anyhow::anyhow!("agent dir has no name"))?
            .to_string();
        let id = AgentId::new(agent_name.clone());

        let record = self
            .registry
            .get(&id)
            .await
            .ok_or_else(|| anyhow::anyhow!("agent {agent_name} not in registry"))?;

        let cap = Capability::root();
        let input = vfs.read(&cap, &ev.path).await?;

        let view = build_view_from_manifest(vfs.clone(), &record.manifest)?;
        let out = rt
            .invoke(
                &id,
                &self.export,
                input,
                view,
                Some(record.log_buffer.clone()),
            )
            .await?;

        let output_path = agent_dir.join("output");
        match vfs.write(&cap, &output_path, out.clone()).await {
            Ok(()) => Ok(()),
            Err(VfsError::NotFound) => {
                vfs.create_file(&output_path, out).await?;
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }
}

fn build_view_from_manifest(vfs: Arc<Vfs>, manifest: &Manifest) -> anyhow::Result<NamespaceView> {
    let mut builder = NamespaceView::builder(vfs, Capability::root());
    for m in &manifest.mounts {
        let virt = VfsPath::parse(&m.virtual_path)?;
        let real = VfsPath::parse(&m.real_path)?;
        builder = builder.mount(virt, real, m.mode.into());
    }
    if matches!(manifest.agent.trust, TrustLevel::System) || manifest.agent.name == "llm_gateway" {
        builder = builder.host_capability(HostCapability::CanAccessNetwork);
    }
    Ok(builder.build())
}
