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

    // Runtime-level structural fence — runs BEFORE the per-task manifest
    // allowlist so the manifest cannot opt back into things the runtime
    // refuses globally (shell wrappers, absolute paths, parent traversal,
    // control chars). This is what prevents `sh -c rm -rf .` even when
    // the manifest accidentally allowlists `sh`.
    if let Err(reason) = validate_command_spec(&command) {
        return deny_task(task, trace, reason).await;
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

/// Shell interpreters are denied unconditionally. They are a generic
/// arbitrary-code-execution channel disguised as one entry in an
/// exec allowlist; once `sh` is allowlisted, *every* shell payload
/// runs. The manifest cannot opt back into this list.
const SHELL_DENYLIST: &[&str] = &[
    "sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "ash",
    "powershell", "pwsh", "cmd",
];

fn has_forbidden_control_char(s: &str) -> bool {
    s.chars().any(|c| matches!(c, '\n' | '\r' | '\0'))
}

/// Returns Some(reason) if `arg` looks like an absolute or
/// outside-workdir path. Flag-style args (`--release`,
/// `--flag=value`) pass through; only path-shaped tokens are fenced.
fn arg_path_violation(arg: &str) -> Option<&'static str> {
    if arg.starts_with('/') || arg.starts_with('\\') {
        return Some("absolute path");
    }
    if arg.starts_with('~') {
        return Some("home-relative path");
    }
    // `--config=/etc/foo`, `--out=~/file`, `--out=\\share\foo`
    if arg.contains("=/") || arg.contains("=\\") || arg.contains("=~") {
        return Some("flag with absolute or home path value");
    }
    if arg == ".."
        || arg.starts_with("../")
        || arg.starts_with("..\\")
        || arg.contains("/../")
        || arg.contains("\\..\\")
        || arg.ends_with("/..")
        || arg.ends_with("\\..")
    {
        return Some("parent-dir traversal");
    }
    None
}

/// Run on every spawn payload before any manifest check. Returns
/// Err(reason) if the command must be refused outright.
pub(crate) fn validate_command_spec(spec: &CommandSpec) -> std::result::Result<(), String> {
    // -- program --
    if spec.program.is_empty() {
        return Err("empty program".to_string());
    }
    if spec.program.contains('/') || spec.program.contains('\\') {
        return Err(format!(
            "program must be a bare binary name (resolved via PATH), got: {:?}",
            spec.program
        ));
    }
    if spec.program.starts_with('-') {
        return Err(format!("program looks like a flag: {:?}", spec.program));
    }
    if has_forbidden_control_char(&spec.program) {
        return Err(format!(
            "program contains control character: {:?}",
            spec.program
        ));
    }
    let lc = spec.program.to_ascii_lowercase();
    let bare = lc.strip_suffix(".exe").unwrap_or(&lc);
    if SHELL_DENYLIST.contains(&bare) {
        return Err(format!(
            "shell interpreters are denied (this is the `sh -c …` class): {:?}",
            spec.program
        ));
    }

    // -- args --
    for (i, arg) in spec.args.iter().enumerate() {
        if has_forbidden_control_char(arg) {
            return Err(format!("arg {i} contains a control character: {arg:?}"));
        }
        if let Some(reason) = arg_path_violation(arg) {
            return Err(format!("arg {i} {reason}: {arg:?}"));
        }
    }

    // -- declared reads / writes --
    for (label, paths) in [("read", &spec.reads), ("write", &spec.writes)] {
        for p in paths {
            let s = p.to_string_lossy();
            if let Some(reason) = arg_path_violation(&s) {
                return Err(format!("declared {label} path {reason}: {s}"));
            }
        }
    }

    // -- cwd override --
    if let Some(cwd) = &spec.cwd {
        let s = cwd.to_string_lossy();
        if let Some(reason) = arg_path_violation(&s) {
            return Err(format!("cwd {reason}: {s}"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(program: &str, args: &[&str]) -> CommandSpec {
        CommandSpec {
            program: program.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            cwd: None,
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    #[test]
    fn rejects_shells_even_with_exe_suffix() {
        for sh in [
            "sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "ash",
            "pwsh", "powershell", "cmd",
            "Bash", "ZSH", "Bash.exe", "pwsh.exe",
        ] {
            let err = validate_command_spec(&spec(sh, &["-c", "echo hi"])).unwrap_err();
            assert!(
                err.contains("shell interpreters"),
                "{sh} should be denied, got: {err}"
            );
        }
    }

    #[test]
    fn rejects_program_paths() {
        for bad in ["/bin/rm", "./rm", "../rm", "C:\\Windows\\System32\\cmd"] {
            let err = validate_command_spec(&spec(bad, &[])).unwrap_err();
            assert!(
                err.contains("bare binary name"),
                "{bad} should be denied, got: {err}"
            );
        }
    }

    #[test]
    fn rejects_program_starting_with_dash() {
        let err = validate_command_spec(&spec("-c", &["echo"])).unwrap_err();
        assert!(err.contains("flag"));
    }

    #[test]
    fn rejects_abs_path_args() {
        for bad in ["/etc/passwd", "/", "~/secret", "\\\\share\\x"] {
            let err = validate_command_spec(&spec("cat", &[bad])).unwrap_err();
            assert!(
                err.contains("absolute") || err.contains("home"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn rejects_parent_traversal() {
        for bad in [
            "..", "../foo", "foo/..", "foo/../bar", "..\\foo", "foo\\..\\bar",
        ] {
            let err = validate_command_spec(&spec("cat", &[bad])).unwrap_err();
            assert!(
                err.contains("parent-dir traversal"),
                "{bad} should be denied, got: {err}"
            );
        }
    }

    #[test]
    fn rejects_flag_value_with_abs_path() {
        let err = validate_command_spec(&spec("cargo", &["test", "--config=/etc/cargo"]))
            .unwrap_err();
        assert!(err.contains("absolute or home path value"));
        let err =
            validate_command_spec(&spec("cargo", &["test", "--out=~/secrets"])).unwrap_err();
        assert!(err.contains("absolute or home path value"));
    }

    #[test]
    fn rejects_control_chars_in_args() {
        let err = validate_command_spec(&spec("echo", &["hi\nrm -rf ."])).unwrap_err();
        assert!(err.contains("control character"));
    }

    #[test]
    fn accepts_ordinary_commands() {
        validate_command_spec(&spec("cargo", &["test", "--release"])).unwrap();
        validate_command_spec(&spec(
            "git",
            &["commit", "-m", "fix: bug", "src/main.rs"],
        ))
        .unwrap();
        validate_command_spec(&spec("rustc", &["src/main.rs", "-o", "target/out"])).unwrap();
        validate_command_spec(&spec("ls", &["-la", "tests"])).unwrap();
    }

    #[test]
    fn fences_apply_to_declared_paths_too() {
        let mut s = spec("cat", &["file.txt"]);
        s.reads = vec![PathBuf::from("/etc/passwd")];
        let err = validate_command_spec(&s).unwrap_err();
        assert!(err.contains("declared read path absolute path"));

        let mut s = spec("cat", &["file.txt"]);
        s.writes = vec![PathBuf::from("../escape")];
        let err = validate_command_spec(&s).unwrap_err();
        assert!(err.contains("declared write path parent-dir traversal"));
    }
}
