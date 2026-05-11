//! v9r-runtime: agent runtime trait + a mock + a real Wasmtime backend.
//!
//! The trait now has `load` / `unload` so the orchestrator can hand wasm
//! bytes to the runtime when an agent's manifest is processed; `invoke`
//! then fires by name.

pub mod mock;
pub mod wasm;

use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;

use v9r_cap::NamespaceView;
use v9r_core::VfsResult;
use v9r_vfs::LogBuffer;

pub use mock::MockRuntime;
pub use wasm::WasmtimeRuntime;

/// Stable identity for an agent in the namespace, e.g. the directory name
/// under `/agents/`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[async_trait]
pub trait AgentRuntime: Send + Sync {
    /// Compile and cache an agent's wasm module. Idempotent — calling again
    /// with the same id replaces the previous module.
    async fn load(&self, id: &AgentId, wasm: Bytes) -> VfsResult<()>;

    /// Forget an agent's module.
    async fn unload(&self, id: &AgentId);

    /// Invoke `export` on the agent identified by `id`. The agent only
    /// touches the VFS through `view` — never directly.
    async fn invoke(
        &self,
        id: &AgentId,
        export: &str,
        input: Bytes,
        view: NamespaceView,
        log: Option<Arc<LogBuffer>>,
    ) -> VfsResult<Bytes>;
}
