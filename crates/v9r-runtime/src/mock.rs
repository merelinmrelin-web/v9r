//! Print-only runtime. Use in tests and during bring-up.

use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;

use v9r_cap::NamespaceView;
use v9r_core::VfsResult;
use v9r_vfs::LogBuffer;

use crate::{AgentId, AgentRuntime};

pub struct MockRuntime;

impl MockRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentRuntime for MockRuntime {
    async fn load(&self, _id: &AgentId, _wasm: Bytes) -> VfsResult<()> {
        Ok(())
    }

    async fn unload(&self, _id: &AgentId) {}

    async fn invoke(
        &self,
        id: &AgentId,
        export: &str,
        input: Bytes,
        view: NamespaceView,
        _log: Option<Arc<LogBuffer>>,
    ) -> VfsResult<Bytes> {
        let preview = String::from_utf8_lossy(&input);
        println!(
            "Wasm Invoke: agent={} export={} bytes={} data={:?}",
            id.as_str(),
            export,
            input.len(),
            preview
        );
        for m in view.mounts() {
            println!(
                "  view: {} -> {} ({:?})",
                m.virtual_path, m.real_path, m.mode
            );
        }
        Ok(Bytes::from(format!(
            "[mock:{}::{}] echo: {}",
            id.as_str(),
            export,
            preview
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use v9r_cap::{MountMode, NamespaceView};
    use v9r_core::{Capability, VfsPath};
    use v9r_vfs::Vfs;

    #[tokio::test]
    async fn mock_echoes_with_view() {
        let vfs = Vfs::new();
        vfs.mkdir_p(&VfsPath::parse("/agents/foo").unwrap())
            .await
            .unwrap();
        let view = NamespaceView::builder(Arc::clone(&vfs), Capability::root())
            .mount(
                VfsPath::root(),
                VfsPath::parse("/agents/foo").unwrap(),
                MountMode::Rw,
            )
            .build();

        let rt = MockRuntime::new();
        let out = rt
            .invoke(
                &AgentId::new("foo"),
                "on_input",
                Bytes::from_static(b"hi"),
                view,
                None,
            )
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&out).contains("echo: hi"));
    }
}
