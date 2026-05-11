//! Manifest model + agent registry.
//!
//! A manifest describes one agent: its name, trust level, optional WASM
//! module, the mounts it asks for, and the trigger files it wants to be
//! invoked on. It's the *source of truth* for an agent's namespace policy
//! — the orchestrator no longer hardcodes anything.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use tokio::sync::RwLock;

use v9r_cap::MountMode;
use v9r_core::{VfsError, VfsPath};
use v9r_runtime::AgentId;
use v9r_vfs::LogBuffer;

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("toml parse error: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("manifest is not utf-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("invalid path: {0}")]
    Vfs(#[from] VfsError),
    #[error("name in manifest ({manifest:?}) does not match agent dir ({dir:?})")]
    NameMismatch { manifest: String, dir: String },
    #[error("invalid trigger path {path:?}: must be a single non-empty segment")]
    BadTriggerPath { path: String },
    #[error("agent {agent:?} ({trust:?}) is not allowed to mount real path {real:?}")]
    UntrustedMount {
        agent: String,
        trust: TrustLevel,
        real: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrustLevel {
    #[default]
    User,
    System,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MountModeSpec {
    Ro,
    #[default]
    Rw,
}

impl From<MountModeSpec> for MountMode {
    fn from(m: MountModeSpec) -> Self {
        match m {
            MountModeSpec::Ro => MountMode::Ro,
            MountModeSpec::Rw => MountMode::Rw,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AgentSpec {
    pub name: String,
    #[serde(default)]
    pub trust: TrustLevel,
    pub module: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MountSpec {
    /// `virtual` is a Rust keyword; renamed in the TOML.
    #[serde(rename = "virtual")]
    pub virtual_path: String,
    #[serde(rename = "real")]
    pub real_path: String,
    #[serde(default)]
    pub mode: MountModeSpec,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TriggerSpec {
    /// File name relative to the agent's directory. Single segment only.
    pub path: String,
    /// WASM export to invoke when this file is written.
    pub export: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    pub agent: AgentSpec,
    #[serde(default)]
    pub mounts: Vec<MountSpec>,
    #[serde(default)]
    pub triggers: Vec<TriggerSpec>,
}

impl Manifest {
    pub fn parse(toml_str: &str) -> Result<Self, ManifestError> {
        Ok(toml::from_str(toml_str)?)
    }

    /// Validate that:
    /// - the manifest's name matches `dir_name`
    /// - every trigger path is a single non-empty segment
    /// - every mount's real path is reachable for this trust level
    pub fn validate(&self, dir_name: &str) -> Result<(), ManifestError> {
        if self.agent.name != dir_name {
            return Err(ManifestError::NameMismatch {
                manifest: self.agent.name.clone(),
                dir: dir_name.to_string(),
            });
        }

        for t in &self.triggers {
            if t.path.is_empty() || t.path.contains('/') || t.path == "." || t.path == ".." {
                return Err(ManifestError::BadTriggerPath {
                    path: t.path.clone(),
                });
            }
        }

        validate_mounts(&self.agent.name, self.agent.trust, &self.mounts)?;
        Ok(())
    }
}

/// User agents may only mount real paths under their own dir, `/tools`, or
/// `/shared`. System agents may mount anything.
pub fn validate_mounts(
    agent_name: &str,
    trust: TrustLevel,
    mounts: &[MountSpec],
) -> Result<(), ManifestError> {
    if matches!(trust, TrustLevel::System) {
        return Ok(());
    }
    let allowed: Vec<VfsPath> = vec![
        VfsPath::parse(&format!("/agents/{agent_name}"))?,
        VfsPath::parse("/tools")?,
        VfsPath::parse("/shared")?,
    ];
    for m in mounts {
        let real = VfsPath::parse(&m.real_path)?;
        let ok = allowed.iter().any(|p| {
            let ps = p.segments();
            let rs = real.segments();
            rs.len() >= ps.len() && &rs[..ps.len()] == ps
        });
        if !ok {
            return Err(ManifestError::UntrustedMount {
                agent: agent_name.to_string(),
                trust,
                real: m.real_path.clone(),
            });
        }
    }
    Ok(())
}

/// Source of truth for "what agents exist." Shared between the manifest
/// handler (writer) and the trigger handlers (readers).
#[derive(Default)]
pub struct AgentRegistry {
    inner: RwLock<HashMap<AgentId, AgentRecord>>,
}

#[derive(Clone, Debug)]
pub struct AgentRecord {
    pub manifest: Manifest,
    pub agent_dir: VfsPath,
    pub log_buffer: Arc<LogBuffer>,
}

impl AgentRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub async fn upsert(&self, id: AgentId, record: AgentRecord) {
        self.inner.write().await.insert(id, record);
    }

    pub async fn get(&self, id: &AgentId) -> Option<AgentRecord> {
        self.inner.read().await.get(id).cloned()
    }

    pub async fn remove(&self, id: &AgentId) -> Option<AgentRecord> {
        self.inner.write().await.remove(id)
    }

    pub async fn names(&self) -> Vec<String> {
        self.inner
            .read()
            .await
            .keys()
            .map(|i| i.0.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ECHO_MANIFEST: &str = r#"
[agent]
name = "echo"
trust = "user"

[[mounts]]
virtual = "/"
real = "/agents/echo"
mode = "rw"

[[mounts]]
virtual = "/tools"
real = "/tools"
mode = "ro"

[[triggers]]
path = "input"
export = "on_input"
"#;

    #[test]
    fn parse_minimal() {
        let m = Manifest::parse(ECHO_MANIFEST).unwrap();
        assert_eq!(m.agent.name, "echo");
        assert_eq!(m.agent.trust, TrustLevel::User);
        assert_eq!(m.mounts.len(), 2);
        assert_eq!(m.triggers.len(), 1);
        assert_eq!(m.triggers[0].export, "on_input");
    }

    #[test]
    fn validate_user_cant_mount_sys() {
        let toml = r#"
[agent]
name = "evil"
trust = "user"

[[mounts]]
virtual = "/sys"
real = "/sys"
mode = "rw"
"#;
        let m = Manifest::parse(toml).unwrap();
        let err = m.validate("evil").unwrap_err();
        assert!(matches!(err, ManifestError::UntrustedMount { .. }));
    }

    #[test]
    fn validate_user_cant_mount_other_agents_dir() {
        let toml = r#"
[agent]
name = "alice"
trust = "user"

[[mounts]]
virtual = "/peek"
real = "/agents/bob"
mode = "ro"
"#;
        let m = Manifest::parse(toml).unwrap();
        let err = m.validate("alice").unwrap_err();
        assert!(matches!(err, ManifestError::UntrustedMount { .. }));
    }

    #[test]
    fn validate_system_can_mount_sys() {
        let toml = r#"
[agent]
name = "kern"
trust = "system"

[[mounts]]
virtual = "/sys"
real = "/sys"
mode = "rw"
"#;
        let m = Manifest::parse(toml).unwrap();
        m.validate("kern").unwrap();
    }

    #[test]
    fn validate_name_mismatch() {
        let m = Manifest::parse(ECHO_MANIFEST).unwrap();
        let err = m.validate("not-echo").unwrap_err();
        assert!(matches!(err, ManifestError::NameMismatch { .. }));
    }

    #[test]
    fn validate_bad_trigger_path() {
        let toml = r#"
[agent]
name = "x"
trust = "user"

[[triggers]]
path = "sub/dir"
export = "on_x"
"#;
        let m = Manifest::parse(toml).unwrap();
        let err = m.validate("x").unwrap_err();
        assert!(matches!(err, ManifestError::BadTriggerPath { .. }));
    }
}
