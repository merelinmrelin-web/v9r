use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::manifest::{AccessType, Manifest};
use crate::task::TaskStatus;
use crate::vfs::CheckpointId;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskEvent {
    TaskStarted {
        timestamp: DateTime<Utc>,
        manifest: Manifest,
    },
    CheckpointCreated {
        id: CheckpointId,
    },
    RollbackPerformed {
        id: CheckpointId,
    },
    CommandExecuted {
        command: String,
        exit_code: i32,
    },
    FileAccess {
        path: PathBuf,
        access: AccessType,
        allowed: bool,
    },
    ViolationOccurred {
        reason: String,
    },
    TaskFinished {
        status: TaskStatus,
    },
    TaskHandoverReceived {
        timestamp: DateTime<Utc>,
        source_task_id: Uuid,
    },
}

impl TaskEvent {
    pub fn task_started(manifest: Manifest) -> Self {
        Self::TaskStarted {
            timestamp: Utc::now(),
            manifest,
        }
    }

    pub fn task_handover_received(source_task_id: Uuid) -> Self {
        Self::TaskHandoverReceived {
            timestamp: Utc::now(),
            source_task_id,
        }
    }
}

#[derive(Clone)]
pub struct TraceLogger {
    path: PathBuf,
    file: Arc<Mutex<File>>,
}

#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("trace io at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("trace serialization: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TraceError>;

impl TraceLogger {
    pub async fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| TraceError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|source| TraceError::Io {
                path: path.clone(),
                source,
            })?;

        Ok(Self {
            path,
            file: Arc::new(Mutex::new(file)),
        })
    }

    pub async fn log_event(&self, event: TaskEvent) -> Result<()> {
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');

        let mut file = self.file.lock().await;
        file.write_all(&line)
            .await
            .map_err(|source| TraceError::Io {
                path: self.path.clone(),
                source,
            })?;
        file.flush().await.map_err(|source| TraceError::Io {
            path: self.path.clone(),
            source,
        })?;
        Ok(())
    }

    pub async fn recent_events(&self, limit: usize) -> Result<Vec<String>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let text = tokio::fs::read_to_string(&self.path)
            .await
            .map_err(|source| TraceError::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut lines: Vec<String> = text.lines().rev().take(limit).map(str::to_string).collect();
        lines.reverse();
        Ok(lines)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
