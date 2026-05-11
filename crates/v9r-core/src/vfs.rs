use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::trace::{TaskEvent, TraceLogger};

const V9R_DIR: &str = ".v9r";
const BACKUPS_DIR: &str = "backups";
const SNAPSHOT_MANIFEST: &str = "manifest.json";
const SAFETY_ERROR: &str = "Safety Error: Cannot use a project root as a mutable workdir. Please use a subfolder or a temporary directory.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointId(pub Uuid);

#[derive(Debug, thiserror::Error)]
pub enum TransactionError {
    #[error("unknown task: {0}")]
    UnknownTask(Uuid),
    #[error("task filesystem is blocked: {0}")]
    FilesystemBlocked(Uuid),
    #[error("symbolic links are not supported in task snapshots: {0}")]
    SymlinkUnsupported(PathBuf),
    #[error("{0}")]
    Safety(String),
    #[error("io at {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("checkpoint worker failed: {0}")]
    Join(String),
    #[error("trace: {0}")]
    Trace(#[from] crate::trace::TraceError),
}

pub type Result<T> = std::result::Result<T, TransactionError>;

#[derive(Clone, Debug)]
struct TaskFsState {
    workdir: PathBuf,
    blocked: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SnapshotManifest {
    files: Vec<SnapshotFile>,
    dirs: Vec<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SnapshotFile {
    relative_path: PathBuf,
    bytes: u64,
    fnv64: u64,
}

static TASKS: OnceLock<Mutex<HashMap<Uuid, TaskFsState>>> = OnceLock::new();

pub fn register_task(task_id: Uuid, workdir: PathBuf) {
    let mut tasks = tasks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    tasks
        .entry(task_id)
        .and_modify(|state| state.workdir = workdir.clone())
        .or_insert(TaskFsState {
            workdir,
            blocked: false,
        });
}

pub fn is_safe_directory(workdir: &Path) -> bool {
    !(workdir.join(".git").is_dir() || workdir.join("Cargo.toml").is_file())
}

pub fn ensure_safe_directory(workdir: &Path) -> Result<()> {
    if is_safe_directory(workdir) {
        Ok(())
    } else {
        Err(TransactionError::Safety(SAFETY_ERROR.to_string()))
    }
}

pub fn block_task_fs(task_id: Uuid) {
    let mut tasks = tasks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(state) = tasks.get_mut(&task_id) {
        state.blocked = true;
    }
}

pub fn ensure_task_fs_unblocked(task_id: Uuid) -> Result<()> {
    let tasks = tasks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let state = tasks
        .get(&task_id)
        .ok_or(TransactionError::UnknownTask(task_id))?;
    if state.blocked {
        return Err(TransactionError::FilesystemBlocked(task_id));
    }
    Ok(())
}

pub async fn checkpoint(task_id: Uuid, trace: &TraceLogger) -> Result<CheckpointId> {
    let checkpoint = tokio::task::spawn_blocking(move || checkpoint_untraced(task_id))
        .await
        .map_err(|err| TransactionError::Join(err.to_string()))??;
    trace
        .log_event(TaskEvent::CheckpointCreated { id: checkpoint })
        .await?;
    Ok(checkpoint)
}

pub async fn rollback(task_id: Uuid, checkpoint: CheckpointId, trace: &TraceLogger) -> Result<()> {
    tokio::task::spawn_blocking(move || rollback_untraced(task_id, checkpoint))
        .await
        .map_err(|err| TransactionError::Join(err.to_string()))??;
    trace
        .log_event(TaskEvent::RollbackPerformed { id: checkpoint })
        .await?;
    Ok(())
}

fn checkpoint_untraced(task_id: Uuid) -> Result<CheckpointId> {
    let workdir = registered_workdir(task_id, true)?;
    ensure_safe_directory(&workdir)?;
    let checkpoint = CheckpointId(Uuid::new_v4());
    let backup_dir = backup_dir(&workdir, task_id, checkpoint);
    let tmp_path = backup_dir.with_file_name(format!(".tmp-{}", checkpoint.0));

    remove_dir_if_exists(&tmp_path)?;
    create_dir_all(&tmp_path)?;
    let manifest = backup_workdir(&workdir, &tmp_path)?;
    write_snapshot_manifest(&tmp_path, &manifest)?;
    if let Some(parent) = backup_dir.parent() {
        create_dir_all(parent)?;
    }
    rename(&tmp_path, &backup_dir)?;

    Ok(checkpoint)
}

fn rollback_untraced(task_id: Uuid, checkpoint: CheckpointId) -> Result<()> {
    let workdir = registered_workdir(task_id, false)?;
    let backup_dir = backup_dir(&workdir, task_id, checkpoint);
    if !backup_dir.is_dir() {
        return Err(TransactionError::Io {
            path: backup_dir,
            source: io::Error::new(io::ErrorKind::NotFound, "checkpoint not found"),
        });
    }
    let manifest = read_snapshot_manifest(&backup_dir)?;
    selective_rollback(&workdir, &backup_dir, &manifest)?;
    Ok(())
}

fn tasks() -> &'static Mutex<HashMap<Uuid, TaskFsState>> {
    TASKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn registered_workdir(task_id: Uuid, require_unblocked: bool) -> Result<PathBuf> {
    let tasks = tasks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let state = tasks
        .get(&task_id)
        .ok_or(TransactionError::UnknownTask(task_id))?;
    if require_unblocked && state.blocked {
        return Err(TransactionError::FilesystemBlocked(task_id));
    }
    Ok(state.workdir.clone())
}

fn backup_dir(workdir: &Path, task_id: Uuid, checkpoint: CheckpointId) -> PathBuf {
    workdir
        .join(V9R_DIR)
        .join(BACKUPS_DIR)
        .join(task_id.to_string())
        .join(checkpoint.0.to_string())
}

fn backup_workdir(workdir: &Path, backup_dir: &Path) -> Result<SnapshotManifest> {
    let mut manifest = SnapshotManifest {
        files: Vec::new(),
        dirs: Vec::new(),
    };
    backup_dir_entries(workdir, workdir, backup_dir, &mut manifest)?;
    manifest
        .files
        .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    manifest.dirs.sort();
    Ok(manifest)
}

fn backup_dir_entries(
    root: &Path,
    dir: &Path,
    backup_dir: &Path,
    manifest: &mut SnapshotManifest,
) -> Result<()> {
    for entry in fs::read_dir(dir).map_err(|source| TransactionError::Io {
        path: dir.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| TransactionError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let src_path = entry.path();
        let metadata = fs::symlink_metadata(&src_path).map_err(|source| TransactionError::Io {
            path: src_path.clone(),
            source,
        })?;

        if metadata.file_type().is_symlink() {
            return Err(TransactionError::SymlinkUnsupported(src_path));
        }
        let relative_path = src_path
            .strip_prefix(root)
            .map_err(|_| {
                TransactionError::Safety(format!(
                    "path escaped snapshot root: {}",
                    src_path.display()
                ))
            })?
            .to_path_buf();
        if relative_path
            .components()
            .next()
            .is_some_and(|component| component.as_os_str() == V9R_DIR)
        {
            continue;
        }
        if metadata.is_dir() {
            manifest.dirs.push(relative_path);
            backup_dir_entries(root, &src_path, backup_dir, manifest)?;
        } else if metadata.is_file() {
            let bytes = fs::read(&src_path).map_err(|source| TransactionError::Io {
                path: src_path.clone(),
                source,
            })?;
            let dst_path = backup_dir.join(&relative_path);
            if let Some(parent) = dst_path.parent() {
                create_dir_all(parent)?;
            }
            fs::write(&dst_path, &bytes).map_err(|source| TransactionError::Io {
                path: dst_path,
                source,
            })?;
            manifest.files.push(SnapshotFile {
                relative_path,
                bytes: bytes.len() as u64,
                fnv64: fnv64(&bytes),
            });
        }
    }
    Ok(())
}

fn selective_rollback(
    workdir: &Path,
    backup_dir: &Path,
    manifest: &SnapshotManifest,
) -> Result<()> {
    let original_files: HashSet<PathBuf> = manifest
        .files
        .iter()
        .map(|file| file.relative_path.clone())
        .collect();
    let original_dirs: HashSet<PathBuf> = manifest.dirs.iter().cloned().collect();
    let mut current_files = Vec::new();
    let mut current_dirs = Vec::new();
    collect_current_entries(workdir, workdir, &mut current_files, &mut current_dirs)?;

    for relative_path in &current_files {
        let path = workdir.join(relative_path);
        if !original_files.contains(relative_path) {
            remove_file_if_exists(&path)?;
            continue;
        }
        let bytes = fs::read(&path).map_err(|source| TransactionError::Io {
            path: path.clone(),
            source,
        })?;
        let Some(snapshot) = manifest
            .files
            .iter()
            .find(|file| &file.relative_path == relative_path)
        else {
            continue;
        };
        if bytes.len() as u64 != snapshot.bytes || fnv64(&bytes) != snapshot.fnv64 {
            restore_file(workdir, backup_dir, relative_path)?;
        }
    }

    for snapshot in &manifest.files {
        let path = workdir.join(&snapshot.relative_path);
        if !path.exists() {
            restore_file(workdir, backup_dir, &snapshot.relative_path)?;
        }
    }

    current_dirs.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for relative_path in current_dirs {
        if !original_dirs.contains(&relative_path) {
            let path = workdir.join(relative_path);
            let _ = fs::remove_dir(&path);
        }
    }
    Ok(())
}

fn collect_current_entries(
    root: &Path,
    dir: &Path,
    files: &mut Vec<PathBuf>,
    dirs: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(dir).map_err(|source| TransactionError::Io {
        path: dir.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| TransactionError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|source| TransactionError::Io {
            path: path.clone(),
            source,
        })?;
        let relative_path = path
            .strip_prefix(root)
            .map_err(|_| {
                TransactionError::Safety(format!("path escaped rollback root: {}", path.display()))
            })?
            .to_path_buf();
        if relative_path
            .components()
            .next()
            .is_some_and(|component| component.as_os_str() == V9R_DIR)
        {
            continue;
        }
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            dirs.push(relative_path);
            collect_current_entries(root, &path, files, dirs)?;
        } else if metadata.is_file() {
            files.push(relative_path);
        }
    }
    Ok(())
}

fn restore_file(workdir: &Path, backup_dir: &Path, relative_path: &Path) -> Result<()> {
    let src = backup_dir.join(relative_path);
    let dst = workdir.join(relative_path);
    if let Some(parent) = dst.parent() {
        create_dir_all(parent)?;
    }
    fs::copy(&src, &dst).map_err(|source| TransactionError::Io { path: src, source })?;
    Ok(())
}

fn write_snapshot_manifest(backup_dir: &Path, manifest: &SnapshotManifest) -> Result<()> {
    let path = backup_dir.join(SNAPSHOT_MANIFEST);
    let bytes = serde_json::to_vec(manifest).map_err(|source| {
        TransactionError::Safety(format!("snapshot manifest encode failed: {source}"))
    })?;
    fs::write(&path, bytes).map_err(|source| TransactionError::Io { path, source })
}

fn read_snapshot_manifest(backup_dir: &Path) -> Result<SnapshotManifest> {
    let path = backup_dir.join(SNAPSHOT_MANIFEST);
    let bytes = fs::read(&path).map_err(|source| TransactionError::Io {
        path: path.clone(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| {
        TransactionError::Safety(format!("snapshot manifest decode failed: {source}"))
    })
}

fn create_dir_all(path: &Path) -> Result<()> {
    fs::create_dir_all(path).map_err(|source| TransactionError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn rename(from: &Path, to: &Path) -> Result<()> {
    fs::rename(from, to).map_err(|source| TransactionError::Io {
        path: from.to_path_buf(),
        source,
    })
}

fn remove_dir_if_exists(path: &Path) -> Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(TransactionError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(TransactionError::Io {
            path: path.to_path_buf(),
            source,
        }),
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

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("v9r-vfs-test-{name}-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rejects_project_roots_as_workdirs() {
        let dir = temp_dir("safety");
        fs::write(dir.join("Cargo.toml"), "[package]\nname = \"unsafe\"\n").unwrap();

        assert!(!is_safe_directory(&dir));
        let err = ensure_safe_directory(&dir).unwrap_err().to_string();
        assert!(err.contains("Safety Error: Cannot use a project root"));
    }

    #[test]
    fn rollback_restores_modified_files_and_deletes_new_files_only() {
        let dir = temp_dir("rollback");
        let task_id = Uuid::new_v4();
        register_task(task_id, dir.clone());
        fs::write(dir.join("keep.txt"), "original\n").unwrap();
        fs::create_dir_all(dir.join("stable-dir")).unwrap();

        let checkpoint = checkpoint_untraced(task_id).unwrap();
        fs::write(dir.join("keep.txt"), "changed\n").unwrap();
        fs::write(dir.join("new.txt"), "new\n").unwrap();
        fs::create_dir_all(dir.join("new-dir")).unwrap();
        fs::write(dir.join("new-dir/file.txt"), "new nested\n").unwrap();

        rollback_untraced(task_id, checkpoint).unwrap();

        assert_eq!(
            fs::read_to_string(dir.join("keep.txt")).unwrap(),
            "original\n"
        );
        assert!(!dir.join("new.txt").exists());
        assert!(!dir.join("new-dir/file.txt").exists());
        assert!(dir.join("stable-dir").exists());
        assert!(dir.exists());
    }
}
