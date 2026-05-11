use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::manifest::{normalize_path, Manifest};
use crate::task::{Task, TaskStatus};
use crate::trace::{TaskEvent, TraceLogger};
use crate::vfs;

const TRACE_FILE_NAME: &str = "trace.jsonl";
const BUNDLE_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskBundle {
    pub version: u32,
    pub source_task_id: Uuid,
    pub source_status: TaskStatus,
    pub files: Vec<BundleFile>,
    pub artifact_hashes: Vec<ArtifactHash>,
    pub trace_jsonl: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleFile {
    pub relative_path: PathBuf,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactHash {
    pub path: PathBuf,
    pub bytes: u64,
    pub fnv64: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("unsupported bundle version: {0}")]
    UnsupportedVersion(u32),
    #[error("unsafe bundle path: {0}")]
    UnsafePath(PathBuf),
    #[error("mandatory artifact missing or empty: {0}")]
    MissingMandatoryArtifact(PathBuf),
    #[error("io at {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("bundle serialization: {0}")]
    Serialize(#[from] Box<bincode::ErrorKind>),
    #[error("trace serialization: {0}")]
    TraceSerialize(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, BundleError>;

pub fn export_bundle(task: &Task, trace_path: Option<&Path>) -> Result<Vec<u8>> {
    let workdir = normalize_path(&task.workdir);
    let mut files = Vec::new();
    collect_files(&workdir, &workdir, &mut files)?;
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

    let trace_jsonl = match trace_path {
        Some(path) => read_optional(path)?,
        None => read_optional(&workdir.join(TRACE_FILE_NAME))?,
    };

    let bundle = TaskBundle {
        version: BUNDLE_VERSION,
        source_task_id: task.id,
        source_status: task.status,
        files,
        artifact_hashes: compute_mandatory_artifact_hashes(task)?,
        trace_jsonl,
    };
    Ok(bincode::serialize(&bundle)?)
}

pub fn import_bundle(data: &[u8], new_manifest: Manifest, new_workdir: PathBuf) -> Result<Task> {
    let bundle: TaskBundle = bincode::deserialize(data)?;
    if bundle.version != BUNDLE_VERSION {
        return Err(BundleError::UnsupportedVersion(bundle.version));
    }

    let new_workdir = normalize_path(&new_workdir);
    fs::create_dir_all(&new_workdir).map_err(|source| BundleError::Io {
        path: new_workdir.clone(),
        source,
    })?;

    for file in bundle.files {
        let relative_path = safe_relative_path(&file.relative_path)?;
        let path = new_workdir.join(relative_path);
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|source| BundleError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        fs::write(&path, file.bytes).map_err(|source| BundleError::Io { path, source })?;
    }

    let trace_path = new_workdir.join(TRACE_FILE_NAME);
    write_trace(&trace_path, &bundle.trace_jsonl, bundle.source_task_id)?;

    let task = Task::new(new_manifest, new_workdir.clone());
    vfs::register_task(task.id, new_workdir);
    Ok(task)
}

pub fn export_bundle_with_trace(task: &Task, trace: &TraceLogger) -> Result<Vec<u8>> {
    export_bundle(task, Some(trace.path()))
}

pub fn compute_mandatory_artifact_hashes(task: &Task) -> Result<Vec<ArtifactHash>> {
    let mut hashes = Vec::with_capacity(task.manifest.mandatory_artifacts.len());
    for artifact in &task.manifest.mandatory_artifacts {
        let path = resolve_task_path(&task.workdir, artifact);
        let bytes = fs::read(&path).map_err(|source| match source.kind() {
            io::ErrorKind::NotFound => BundleError::MissingMandatoryArtifact(path.clone()),
            _ => BundleError::Io {
                path: path.clone(),
                source,
            },
        })?;
        if bytes.is_empty() {
            return Err(BundleError::MissingMandatoryArtifact(path));
        }
        hashes.push(ArtifactHash {
            path,
            bytes: bytes.len() as u64,
            fnv64: fnv64(&bytes),
        });
    }
    hashes.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(hashes)
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<BundleFile>) -> Result<()> {
    for entry in fs::read_dir(dir).map_err(|source| BundleError::Io {
        path: dir.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| BundleError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|source| BundleError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            collect_files(root, &path, out)?;
            continue;
        }
        if metadata.is_file() {
            if path.file_name().is_some_and(|name| name == TRACE_FILE_NAME) {
                continue;
            }
            let relative_path = path
                .strip_prefix(root)
                .map_err(|_| BundleError::UnsafePath(path.clone()))?
                .to_path_buf();
            let bytes = fs::read(&path).map_err(|source| BundleError::Io { path, source })?;
            out.push(BundleFile {
                relative_path,
                bytes,
            });
        }
    }
    Ok(())
}

fn read_optional(path: &Path) -> Result<Vec<u8>> {
    match fs::read(path) {
        Ok(bytes) => Ok(bytes),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(source) => Err(BundleError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn write_trace(path: &Path, trace_jsonl: &[u8], source_task_id: Uuid) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| BundleError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let mut trace = trace_jsonl.to_vec();
    if !trace.is_empty() && !trace.ends_with(b"\n") {
        trace.push(b'\n');
    }
    let mut event = serde_json::to_vec(&TaskEvent::task_handover_received(source_task_id))?;
    event.push(b'\n');
    trace.extend_from_slice(&event);
    fs::write(path, trace).map_err(|source| BundleError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn safe_relative_path(path: &Path) -> Result<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(BundleError::UnsafePath(path.to_path_buf()));
            }
        }
    }
    if out.as_os_str().is_empty() {
        return Err(BundleError::UnsafePath(path.to_path_buf()));
    }
    Ok(out)
}

fn resolve_task_path(workdir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        normalize_path(path)
    } else {
        normalize_path(&workdir.join(path))
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_parent_dir_paths() {
        assert!(safe_relative_path(Path::new("../escape.txt")).is_err());
        assert!(safe_relative_path(Path::new("ok/file.txt")).is_ok());
    }
}
