//! v9r-cap: capability-bounded namespace views for agents.
//!
//! A `NamespaceView` is the only handle agents (and the runtime that hosts
//! them) ever get to the VFS. It owns a list of `MountPoint`s — Plan-9-style
//! bind mounts mapping virtual paths (as the agent sees them) to real paths
//! in the underlying VFS, plus an access mode.
//!
//! Resolution rule: longest virtual-prefix wins. With
//!   /        -> /agents/echo  (Rw)
//!   /tools   -> /tools        (Ro)
//! the agent's `/tools/grep` resolves through the more specific mount.
//!
//! Escape protection: `VfsPath::parse` already canonicalizes `.` / `..` and
//! rejects root-escape, and `VfsPath::from_segments` (the only other path
//! constructor) rejects literal `..` / `.` / empty / `/`-bearing segments.
//! `resolve` only ever appends remainder segments to a real prefix, so even
//! a hypothetical bypass path can't reach above its mount's real_path.

use std::sync::Arc;

use bytes::Bytes;

use v9r_core::{Capability, VfsError, VfsPath, VfsResult};
use v9r_vfs::{DirEntry, Vfs};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostCapability {
    CanAccessNetwork,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MountMode {
    /// Read-only.
    Ro,
    /// Read-write.
    Rw,
}

impl MountMode {
    pub fn allows_read(&self) -> bool {
        matches!(self, MountMode::Ro | MountMode::Rw)
    }
    pub fn allows_write(&self) -> bool {
        matches!(self, MountMode::Rw)
    }
}

#[derive(Clone, Debug)]
pub struct MountPoint {
    pub virtual_path: VfsPath,
    pub real_path: VfsPath,
    pub mode: MountMode,
}

/// An agent's view of the VFS. Cheap to clone — wraps `Arc<Vfs>` and a
/// `Vec<MountPoint>`.
#[derive(Clone)]
pub struct NamespaceView {
    vfs: Arc<Vfs>,
    cap: Capability,
    mounts: Arc<Vec<MountPoint>>,
    host_capabilities: Arc<Vec<HostCapability>>,
}

pub struct NamespaceViewBuilder {
    vfs: Arc<Vfs>,
    cap: Capability,
    mounts: Vec<MountPoint>,
    host_capabilities: Vec<HostCapability>,
}

impl NamespaceView {
    pub fn builder(vfs: Arc<Vfs>, cap: Capability) -> NamespaceViewBuilder {
        NamespaceViewBuilder {
            vfs,
            cap,
            mounts: Vec::new(),
            host_capabilities: Vec::new(),
        }
    }

    pub fn capability(&self) -> &Capability {
        &self.cap
    }

    pub fn mounts(&self) -> &[MountPoint] {
        &self.mounts
    }

    pub fn has_host_capability(&self, capability: HostCapability) -> bool {
        self.host_capabilities.contains(&capability)
    }

    pub fn can_access_network(&self) -> bool {
        self.has_host_capability(HostCapability::CanAccessNetwork)
    }

    /// Translate a virtual path to a real path. Returns `PermissionDenied`
    /// if no mount covers the virtual path — agents only see what's mounted.
    ///
    /// One allocation: a `Vec<String>` of exact capacity holding cloned
    /// segments. True zero-copy would require either an `Arc<str>`-based
    /// `VfsPath` or `Vfs` methods that take segment slices; both are
    /// follow-up refactors.
    pub fn resolve(&self, virtual_path: &VfsPath) -> VfsResult<(VfsPath, MountMode)> {
        let v_segs = virtual_path.segments();

        // Longest-prefix match. `Vec` scan is fine while mount tables are
        // tiny (~5 entries per agent); swap for a trie if that changes.
        let best = self
            .mounts
            .iter()
            .filter(|m| {
                let m_segs = m.virtual_path.segments();
                m_segs.len() <= v_segs.len() && v_segs.starts_with(m_segs)
            })
            .max_by_key(|m| m.virtual_path.segments().len())
            .ok_or(VfsError::PermissionDenied)?;

        let prefix_len = best.virtual_path.segments().len();
        let remainder = &v_segs[prefix_len..];
        let real_segs = best.real_path.segments();

        let mut combined: Vec<String> = Vec::with_capacity(real_segs.len() + remainder.len());
        combined.extend(real_segs.iter().cloned());
        combined.extend(remainder.iter().cloned());

        // Segments came from already-validated VfsPaths, so this can't fail
        // — but go through the validating constructor anyway to keep the
        // "no back door for .." invariant explicit.
        let real_path = VfsPath::from_segments(combined)?;
        Ok((real_path, best.mode))
    }

    pub async fn read(&self, virtual_path: &VfsPath) -> VfsResult<Bytes> {
        let (real, mode) = self.resolve(virtual_path)?;
        if !mode.allows_read() {
            return Err(VfsError::PermissionDenied);
        }
        self.vfs.read(&self.cap, &real).await
    }

    pub async fn write(&self, virtual_path: &VfsPath, data: Bytes) -> VfsResult<()> {
        let (real, mode) = self.resolve(virtual_path)?;
        if !mode.allows_write() {
            return Err(VfsError::PermissionDenied);
        }
        self.vfs.write(&self.cap, &real, data).await
    }

    pub async fn list(&self, virtual_path: &VfsPath) -> VfsResult<Vec<DirEntry>> {
        let (real, mode) = self.resolve(virtual_path)?;
        if !mode.allows_read() {
            return Err(VfsError::PermissionDenied);
        }
        self.vfs.list(&real).await
    }
}

impl NamespaceViewBuilder {
    pub fn mount(mut self, virtual_path: VfsPath, real_path: VfsPath, mode: MountMode) -> Self {
        self.mounts.push(MountPoint {
            virtual_path,
            real_path,
            mode,
        });
        self
    }

    pub fn host_capability(mut self, capability: HostCapability) -> Self {
        if !self.host_capabilities.contains(&capability) {
            self.host_capabilities.push(capability);
        }
        self
    }

    pub fn build(self) -> NamespaceView {
        NamespaceView {
            vfs: self.vfs,
            cap: self.cap,
            mounts: Arc::new(self.mounts),
            host_capabilities: Arc::new(self.host_capabilities),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    async fn fixture() -> Arc<Vfs> {
        let vfs = Vfs::new();
        for d in ["/agents/echo", "/agents/other", "/tools", "/secret"] {
            vfs.mkdir_p(&VfsPath::parse(d).unwrap()).await.unwrap();
        }
        vfs.create_file(&VfsPath::parse("/agents/echo/input").unwrap(), Bytes::new())
            .await
            .unwrap();
        vfs.create_file(
            &VfsPath::parse("/agents/echo/output").unwrap(),
            Bytes::new(),
        )
        .await
        .unwrap();
        vfs.create_file(
            &VfsPath::parse("/tools/grep").unwrap(),
            Bytes::from_static(b"grep-binary"),
        )
        .await
        .unwrap();
        vfs.create_file(
            &VfsPath::parse("/secret/key").unwrap(),
            Bytes::from_static(b"shh"),
        )
        .await
        .unwrap();
        vfs
    }

    #[test]
    fn host_capabilities_are_explicit() {
        let vfs = Vfs::new();
        let user = NamespaceView::builder(Arc::clone(&vfs), Capability::root()).build();
        let system = NamespaceView::builder(vfs, Capability::root())
            .host_capability(HostCapability::CanAccessNetwork)
            .build();

        assert!(!user.can_access_network());
        assert!(system.can_access_network());
    }

    fn echo_view(vfs: Arc<Vfs>) -> NamespaceView {
        NamespaceView::builder(vfs, Capability::root())
            .mount(
                VfsPath::root(),
                VfsPath::parse("/agents/echo").unwrap(),
                MountMode::Rw,
            )
            .mount(
                VfsPath::parse("/tools").unwrap(),
                VfsPath::parse("/tools").unwrap(),
                MountMode::Ro,
            )
            .build()
    }

    #[tokio::test]
    async fn resolve_root_mount() {
        let view = echo_view(fixture().await);
        let (real, mode) = view.resolve(&VfsPath::parse("/input").unwrap()).unwrap();
        assert_eq!(real, VfsPath::parse("/agents/echo/input").unwrap());
        assert_eq!(mode, MountMode::Rw);
    }

    #[tokio::test]
    async fn resolve_longest_prefix() {
        let view = echo_view(fixture().await);
        // /tools/grep should bind through the /tools mount, not /.
        let (real, mode) = view
            .resolve(&VfsPath::parse("/tools/grep").unwrap())
            .unwrap();
        assert_eq!(real, VfsPath::parse("/tools/grep").unwrap());
        assert_eq!(mode, MountMode::Ro);
    }

    #[tokio::test]
    async fn resolve_root_itself() {
        let view = echo_view(fixture().await);
        let (real, _) = view.resolve(&VfsPath::root()).unwrap();
        assert_eq!(real, VfsPath::parse("/agents/echo").unwrap());
    }

    #[tokio::test]
    async fn unmounted_path_denied() {
        // No / mount, only /tools — anything outside /tools is invisible.
        let vfs = fixture().await;
        let view = NamespaceView::builder(vfs, Capability::root())
            .mount(
                VfsPath::parse("/tools").unwrap(),
                VfsPath::parse("/tools").unwrap(),
                MountMode::Ro,
            )
            .build();
        let err = view
            .resolve(&VfsPath::parse("/secret/key").unwrap())
            .unwrap_err();
        assert!(matches!(err, VfsError::PermissionDenied));
    }

    #[tokio::test]
    async fn write_to_ro_mount_denied() {
        let view = echo_view(fixture().await);
        let err = view
            .write(
                &VfsPath::parse("/tools/grep").unwrap(),
                Bytes::from_static(b"hax"),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, VfsError::PermissionDenied));
    }

    #[tokio::test]
    async fn read_through_view_round_trips() {
        let vfs = fixture().await;
        // Pre-populate the agent's input under its real path.
        vfs.write(
            &Capability::root(),
            &VfsPath::parse("/agents/echo/input").unwrap(),
            Bytes::from_static(b"hello"),
        )
        .await
        .unwrap();

        let view = echo_view(vfs);
        let bytes = view.read(&VfsPath::parse("/input").unwrap()).await.unwrap();
        assert_eq!(&bytes[..], b"hello");
    }

    #[tokio::test]
    async fn list_through_view() {
        let view = echo_view(fixture().await);
        let entries = view.list(&VfsPath::root()).await.unwrap();
        let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        // Agent only sees its own dir contents.
        assert_eq!(names, vec!["input", "output"]);
    }

    /// Escape attempts via `..` are blocked at parse time. The cap layer
    /// has a second-line defense: even a path constructed via segments
    /// can't escape because `from_segments` rejects "..".
    #[test]
    fn dotdot_cannot_escape_at_parse() {
        assert!(VfsPath::parse("/../escape").is_err());
        assert!(VfsPath::parse("/input/../../escape").is_err());
        // /a/b/../c is fine — that resolves within the same root.
        let p = VfsPath::parse("/a/b/../c").unwrap();
        assert_eq!(p.segments(), &["a", "c"]);
    }

    #[test]
    fn dotdot_cannot_escape_via_segments() {
        let r = VfsPath::from_segments([".."]);
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn agent_cannot_reach_sibling() {
        // Echo's view must not reach /agents/other or /secret.
        let view = echo_view(fixture().await);
        // Even if the agent constructs the literal real path, the view's
        // root mount only covers that path's prefix when it starts with /.
        // The segments after the virtual / are appended to /agents/echo,
        // so /agents/other gets translated to /agents/echo/agents/other —
        // which doesn't exist, hence NotFound, not a sibling read.
        let err = view
            .read(&VfsPath::parse("/agents/other").unwrap())
            .await
            .unwrap_err();
        assert!(matches!(err, VfsError::NotFound));
    }
}
