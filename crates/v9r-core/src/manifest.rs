use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessType {
    Read,
    Write,
    Exec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub allow_read: Vec<PathBuf>,
    pub allow_write: Vec<PathBuf>,
    pub allow_exec: Vec<String>,
    pub token_limit: u32,
    pub max_steps: usize,
    pub timeout_ms: u64,
    pub mandatory_artifacts: Vec<PathBuf>,
    pub test_commands: Vec<String>,
}

impl Manifest {
    pub fn is_allowed(&self, path: &Path, access: AccessType) -> bool {
        match access {
            AccessType::Read => is_path_allowed(path, &self.allow_read),
            AccessType::Write => is_path_allowed(path, &self.allow_write),
            AccessType::Exec => is_exec_allowed(path, &self.allow_exec),
        }
    }
}

fn is_path_allowed(path: &Path, allowed_roots: &[PathBuf]) -> bool {
    let path = normalize_path(path);
    allowed_roots
        .iter()
        .map(|root| normalize_path(root))
        .any(|root| path.starts_with(root))
}

fn is_exec_allowed(path: &Path, allowed: &[String]) -> bool {
    let command = path.to_string_lossy();
    let file_name = path.file_name().map(|name| name.to_string_lossy());
    allowed.iter().any(|candidate| {
        candidate == command.as_ref()
            || file_name
                .as_ref()
                .is_some_and(|name| candidate == name.as_ref())
    })
}

pub fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(part) => out.push(part),
            Component::RootDir | Component::Prefix(_) => out.push(component.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_access_is_component_bounded() {
        let manifest = Manifest {
            allow_read: Vec::new(),
            allow_write: vec![PathBuf::from("/tmp/work")],
            allow_exec: Vec::new(),
            token_limit: 100,
            max_steps: 8,
            timeout_ms: 30_000,
            mandatory_artifacts: Vec::new(),
            test_commands: Vec::new(),
        };

        assert!(manifest.is_allowed(Path::new("/tmp/work/file.txt"), AccessType::Write));
        assert!(!manifest.is_allowed(Path::new("/tmp/work-escape/file.txt"), AccessType::Write));
    }
}
