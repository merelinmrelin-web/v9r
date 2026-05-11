use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::manifest::{normalize_path, AccessType};
use crate::task::{Task, TaskStatus};
use crate::trace::{TaskEvent, TraceLogger};
use crate::vfs;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub reads: Vec<PathBuf>,
    pub writes: Vec<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepOutput {
    pub status_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("max_steps exceeded: {0}")]
    MaxStepsExceeded(usize),
    #[error("permission violation: {0}")]
    Violation(String),
    #[error("task filesystem: {0}")]
    Transaction(#[from] vfs::TransactionError),
    #[error("command worker failed: {0}")]
    Join(String),
    #[error("io while executing {program}: {source}")]
    Io { program: String, source: io::Error },
    #[error("trace: {0}")]
    Trace(#[from] crate::trace::TraceError),
}

pub type Result<T> = std::result::Result<T, ExecutionError>;

pub async fn run_task_step(
    task: &mut Task,
    command: CommandSpec,
    trace: &TraceLogger,
) -> Result<StepOutput> {
    vfs::register_task(task.id, task.workdir.clone());
    task.record_tool_call()
        .map_err(|_| ExecutionError::MaxStepsExceeded(task.manifest.max_steps))?;
    if let Err(err) = vfs::ensure_task_fs_unblocked(task.id) {
        task.status = TaskStatus::Violation;
        trace
            .log_event(TaskEvent::ViolationOccurred {
                reason: err.to_string(),
            })
            .await?;
        return Err(err.into());
    }

    if let Some(reason) = check_manifest(task, &command, trace).await? {
        return deny_task(task, trace, reason).await;
    }

    task.status = TaskStatus::Running;
    let cwd = command_cwd(task, &command);
    let program = command.program.clone();
    let args = command.args.clone();

    let output = Command::new(&program)
        .args(args)
        .current_dir(cwd)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|source| ExecutionError::Io { program, source })?;

    task.status = TaskStatus::Verifying;
    let step_output = StepOutput {
        status_code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    };
    let rendered_command = command_line(&command);
    let exit_code = step_output.status_code.unwrap_or(-1);
    task.record_test_command(&rendered_command, exit_code);
    trace
        .log_event(TaskEvent::CommandExecuted {
            command: rendered_command,
            exit_code,
        })
        .await?;
    task.status = if step_output.status_code == Some(0) {
        TaskStatus::Success
    } else {
        TaskStatus::Failed
    };
    trace
        .log_event(TaskEvent::TaskFinished {
            status: task.status,
        })
        .await?;

    Ok(step_output)
}

async fn check_manifest(
    task: &Task,
    command: &CommandSpec,
    trace: &TraceLogger,
) -> Result<Option<String>> {
    let exec_path = PathBuf::from(&command.program);
    let allowed = task.manifest.is_allowed(&exec_path, AccessType::Exec);
    trace
        .log_event(TaskEvent::FileAccess {
            path: exec_path,
            access: AccessType::Exec,
            allowed,
        })
        .await?;
    if !allowed {
        return Ok(Some(format!("exec denied: {}", command.program)));
    }

    let workdir = normalize_path(&task.workdir);
    let cwd = command_cwd(task, command);
    if !cwd.starts_with(&workdir) {
        return Ok(Some(format!("cwd escapes task workdir: {}", cwd.display())));
    }

    for read in &command.reads {
        let path = resolve_task_path(task, read);
        let allowed = task.manifest.is_allowed(&path, AccessType::Read);
        trace
            .log_event(TaskEvent::FileAccess {
                path: path.clone(),
                access: AccessType::Read,
                allowed,
            })
            .await?;
        if !allowed {
            return Ok(Some(format!("read denied: {}", path.display())));
        }
    }

    for write in &command.writes {
        let path = resolve_task_path(task, write);
        let allowed = task.manifest.is_allowed(&path, AccessType::Write);
        trace
            .log_event(TaskEvent::FileAccess {
                path: path.clone(),
                access: AccessType::Write,
                allowed,
            })
            .await?;
        if !allowed {
            return Ok(Some(format!("write denied: {}", path.display())));
        }
    }

    Ok(None)
}

async fn deny_task(task: &mut Task, trace: &TraceLogger, reason: String) -> Result<StepOutput> {
    task.status = TaskStatus::Violation;
    trace
        .log_event(TaskEvent::ViolationOccurred {
            reason: reason.clone(),
        })
        .await?;
    vfs::block_task_fs(task.id);
    trace
        .log_event(TaskEvent::TaskFinished {
            status: TaskStatus::Violation,
        })
        .await?;
    Err(ExecutionError::Violation(reason))
}

fn command_cwd(task: &Task, command: &CommandSpec) -> PathBuf {
    command
        .cwd
        .as_ref()
        .map(|cwd| resolve_task_path(task, cwd))
        .unwrap_or_else(|| normalize_path(&task.workdir))
}

fn resolve_task_path(task: &Task, path: &Path) -> PathBuf {
    if path.is_absolute() {
        normalize_path(path)
    } else {
        normalize_path(&task.workdir.join(path))
    }
}

fn command_line(command: &CommandSpec) -> String {
    std::iter::once(command.program.as_str())
        .chain(command.args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}
