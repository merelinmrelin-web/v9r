//! v9r-core: shared types for the v9r workspace.

pub mod adapter;
pub mod bundle;
pub mod context;
pub mod execution;
pub mod manifest;
pub mod task;
pub mod trace;
pub mod vfs;

use std::fmt;

use slotmap::new_key_type;

new_key_type! {
    /// Stable arena key for VFS nodes.
    pub struct NodeId;
}

#[derive(Debug, thiserror::Error)]
pub enum VfsError {
    #[error("path not found")]
    NotFound,
    #[error("is a directory")]
    IsDir,
    #[error("not a directory")]
    NotADir,
    #[error("already exists")]
    AlreadyExists,
    #[error("permission denied")]
    PermissionDenied,
    #[error("invalid path: {0}")]
    InvalidPath(String),
    #[error("synthetic node refused operation: {0}")]
    Synthetic(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type VfsResult<T> = Result<T, VfsError>;

/// Canonicalized absolute path: a sequence of non-empty segments, no `.` / `..`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct VfsPath(Vec<String>);

impl VfsPath {
    pub fn root() -> Self {
        Self(Vec::new())
    }

    /// Parse a slash-delimited path. Resolves `.` and `..`; returns
    /// `InvalidPath` if `..` would escape the root.
    pub fn parse(s: &str) -> VfsResult<Self> {
        let mut out: Vec<String> = Vec::new();
        for seg in s.split('/') {
            match seg {
                "" | "." => {}
                ".." => {
                    if out.pop().is_none() {
                        return Err(VfsError::InvalidPath(format!("path escapes root: {s}")));
                    }
                }
                other => out.push(other.to_string()),
            }
        }
        Ok(Self(out))
    }

    pub fn segments(&self) -> &[String] {
        &self.0
    }

    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    pub fn parent(&self) -> Option<Self> {
        if self.0.is_empty() {
            None
        } else {
            let mut s = self.0.clone();
            s.pop();
            Some(Self(s))
        }
    }

    pub fn last(&self) -> Option<&str> {
        self.0.last().map(|s| s.as_str())
    }

    pub fn join(&self, seg: &str) -> Self {
        let mut s = self.0.clone();
        s.push(seg.to_string());
        Self(s)
    }

    /// Build a path from already-split segments. Each segment is validated:
    /// no empty strings, no `.`/`..`, no embedded `/`. Use this when you've
    /// constructed segments programmatically (e.g. by combining a real
    /// prefix with a remainder during namespace translation) — it's the
    /// only way into `VfsPath` outside of `parse`, so there's no back door
    /// for `..` escape via segment injection.
    pub fn from_segments<I, S>(segments: I) -> VfsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let iter = segments.into_iter();
        let (lo, hi) = iter.size_hint();
        let mut out = Vec::with_capacity(hi.unwrap_or(lo));
        for seg in iter {
            let s: String = seg.into();
            if s.is_empty() || s == "." || s == ".." || s.contains('/') {
                return Err(VfsError::InvalidPath(format!("bad segment: {s:?}")));
            }
            out.push(s);
        }
        Ok(Self(out))
    }
}

impl fmt::Display for VfsPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "/");
        }
        for seg in &self.0 {
            write!(f, "/{}", seg)?;
        }
        Ok(())
    }
}

/// Opaque capability identifier. The capability layer maps these to
/// per-agent NamespaceViews; the VFS itself only carries the ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityId(pub u64);

#[derive(Clone, Debug)]
pub struct Capability {
    id: CapabilityId,
}

impl Capability {
    pub fn new(id: u64) -> Self {
        Self {
            id: CapabilityId(id),
        }
    }
    pub fn root() -> Self {
        Self::new(0)
    }
    pub fn id(&self) -> CapabilityId {
        self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_root() {
        assert!(VfsPath::parse("/").unwrap().is_root());
        assert!(VfsPath::parse("").unwrap().is_root());
    }

    #[test]
    fn parse_segments() {
        let p = VfsPath::parse("/agents/echo/input").unwrap();
        assert_eq!(p.segments(), &["agents", "echo", "input"]);
        assert_eq!(p.last(), Some("input"));
        assert_eq!(p.parent().unwrap().segments(), &["agents", "echo"]);
    }

    #[test]
    fn parse_dotdot() {
        let p = VfsPath::parse("/a/b/../c").unwrap();
        assert_eq!(p.segments(), &["a", "c"]);
        assert!(VfsPath::parse("/..").is_err());
    }

    #[test]
    fn display_round_trip() {
        let p = VfsPath::parse("/a/b/c").unwrap();
        assert_eq!(format!("{p}"), "/a/b/c");
        assert_eq!(format!("{}", VfsPath::root()), "/");
    }

    #[test]
    fn from_segments_rejects_dotdot() {
        assert!(VfsPath::from_segments(["..".to_string()]).is_err());
        assert!(VfsPath::from_segments([".".to_string()]).is_err());
        assert!(VfsPath::from_segments(["a/b".to_string()]).is_err());
        assert!(VfsPath::from_segments([String::new()]).is_err());
        let ok = VfsPath::from_segments(["a", "b"]).unwrap();
        assert_eq!(ok.segments(), &["a", "b"]);
    }
}
