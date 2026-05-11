use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::trace::{TaskEvent, TraceLogger};

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
    let checkpoint = CheckpointId(Uuid::new_v4());
    let task_root = checkpoint_root().join(task_id.to_string());
    let final_path = task_root.join(checkpoint.0.to_string());
    let tmp_path = task_root.join(format!(".tmp-{}", checkpoint.0));

    remove_dir_if_exists(&tmp_path)?;
    create_dir_all(&task_root)?;
    copy_dir(&workdir, &tmp_path)?;
    rename(&tmp_path, &final_path)?;

    Ok(checkpoint)
}

fn rollback_untraced(task_id: Uuid, checkpoint: CheckpointId) -> Result<()> {
    let workdir = registered_workdir(task_id, false)?;
    let snapshot = checkpoint_root()
        .join(task_id.to_string())
        .join(checkpoint.0.to_string());
    if !snapshot.is_dir() {
        return Err(TransactionError::Io {
            path: snapshot,
            source: io::Error::new(io::ErrorKind::NotFound, "checkpoint not found"),
        });
    }

    let parent = workdir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or(
            std::env::current_dir().map_err(|source| TransactionError::Io {
                path: PathBuf::from("."),
                source,
            })?,
        );
    create_dir_all(&parent)?;
    let restore_tmp = parent.join(format!(".v9r-restore-{}", checkpoint.0));
    let backup = parent.join(format!(".v9r-backup-{}", checkpoint.0));

    remove_dir_if_exists(&restore_tmp)?;
    remove_dir_if_exists(&backup)?;
    copy_dir(&snapshot, &restore_tmp)?;

    if workdir.exists() {
        rename(&workdir, &backup)?;
    }

    if let Err(err) = fs::rename(&restore_tmp, &workdir) {
        if backup.exists() {
            let _ = fs::rename(&backup, &workdir);
        }
        return Err(TransactionError::Io {
            path: workdir,
            source: err,
        });
    }

    remove_dir_if_exists(&backup)?;
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

fn checkpoint_root() -> PathBuf {
    std::env::var_os("V9R_CHECKPOINT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("v9r-checkpoints"))
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    create_dir_all(dst)?;
    for entry in fs::read_dir(src).map_err(|source| TransactionError::Io {
        path: src.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| TransactionError::Io {
            path: src.to_path_buf(),
            source,
        })?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let metadata = fs::symlink_metadata(&src_path).map_err(|source| TransactionError::Io {
            path: src_path.clone(),
            source,
        })?;

        if metadata.file_type().is_symlink() {
            return Err(TransactionError::SymlinkUnsupported(src_path));
        }
        if metadata.is_dir() {
            copy_dir(&src_path, &dst_path)?;
        } else if metadata.is_file() {
            fs::copy(&src_path, &dst_path).map_err(|source| TransactionError::Io {
                path: src_path,
                source,
            })?;
        }
    }
    Ok(())
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
