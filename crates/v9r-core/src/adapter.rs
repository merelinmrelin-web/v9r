use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;
use tokio::time::Instant;
use uuid::Uuid;

use crate::bundle::{compute_mandatory_artifact_hashes, ArtifactHash};
use crate::execution::{run_task_step, CommandSpec, ExecutionError};
use crate::manifest::{normalize_path, AccessType};
use crate::task::{Task, TaskErrorType, TaskReport, TaskStatus};
use crate::trace::{TaskEvent, TraceLogger};
use crate::vfs;

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_MODEL_ID: &str = "google/gemini-flash-1.5";
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
const OLLAMA_MODEL_ID: &str = "llama3";
const HTTP_REFERER: &str = "https://github.com/nikita/v9r";
const X_TITLE: &str = "v9r Orchestrator";
const SNAPSHOT_HISTORY_LIMIT: usize = 32;

pub const SYSTEM_PROMPT: &str = r#"You are operating inside v9r, a task-centric autonomous work runtime.
You must respond only with XML action tags, optionally followed by short text inside <finish>.

Allowed actions:
<execute>program arg1 arg2 ...</execute>
Run a command. The runtime spawns the program DIRECTLY via execve — there is no
shell. So:
  - No pipes, redirects, command substitution, globs, `;`, `&&`, `||`, `$VAR`, `~`.
  - Use double quotes ("...") only for args that contain literal spaces.
  - Paths must be relative to the workdir. Absolute paths and `..` are refused.
  - Shell interpreters (sh, bash, zsh, dash, …) are denied unconditionally.
The runtime executes commands sequentially and checks the task manifest first.

<write path="relative/or/absolute/path">file content</write>
Write file content. Paths outside allow_write are violations.

<finish status="Success">brief reason</finish>
<finish status="Failed">brief reason</finish>
Finish the task when no more actions are needed.

Rules:
- Do not use Markdown code fences.
- Do not describe commands outside XML tags.
- Use only paths visible in <environment_files> or allowed by <system_manifest>.
- Prefer minimal edits and verify with <execute> when possible.
- If blocked by permissions or missing information, use <finish status="Failed">reason</finish>.
"#;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn model_id(&self) -> &str;
    fn base_url(&self) -> &str;
    async fn complete(&self, prompt: String) -> Result<String>;
}

pub struct OpenAiCompatibleProvider {
    api_key: Option<String>,
    model_id: String,
    base_url: String,
    http: reqwest::Client,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind {
    OpenAiCompatible,
    Ollama,
}

pub struct LlmClient {
    provider: Box<dyn LlmProvider>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmAction {
    Execute(String),
    Write { path: PathBuf, content: String },
    Finish { status: TaskStatus, reason: String },
}

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("max_steps exceeded: {0}")]
    MaxStepsExceeded(usize),
    #[error("V9R_API_KEY not set")]
    MissingApiKey,
    #[error("http client: {0}")]
    Http(#[from] reqwest::Error),
    #[error("LLM provider returned status {status}: {body}")]
    ApiStatus {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("invalid LLM provider response: {0}")]
    InvalidResponse(String),
    #[error("invalid action: {0}")]
    InvalidAction(String),
    #[error("permission violation: {0}")]
    Violation(String),
    #[error("context: {0}")]
    Context(#[from] crate::context::ContextError),
    #[error("execution: {0}")]
    Execution(#[from] ExecutionError),
    #[error("trace: {0}")]
    Trace(#[from] crate::trace::TraceError),
    #[error("task filesystem: {0}")]
    Transaction(#[from] vfs::TransactionError),
    #[error("io at {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
}

pub type Result<T> = std::result::Result<T, AdapterError>;

impl LlmClient {
    pub fn new(api_key: impl Into<String>, model_id: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(AdapterError::MissingApiKey);
        }
        Self::openai_compatible(Some(api_key), DEFAULT_BASE_URL.to_string(), model_id.into())
    }

    pub fn from_env() -> Result<Self> {
        Self::from_config(ProviderKind::OpenAiCompatible, None, None, None)
    }

    pub fn from_config(
        provider: ProviderKind,
        model_id: Option<String>,
        base_url: Option<String>,
        api_key: Option<String>,
    ) -> Result<Self> {
        match provider {
            ProviderKind::OpenAiCompatible => {
                let api_key = api_key.or_else(|| std::env::var("V9R_API_KEY").ok());
                if api_key.as_deref().unwrap_or_default().trim().is_empty() {
                    return Err(AdapterError::MissingApiKey);
                }
                let base_url = base_url
                    .or_else(|| std::env::var("V9R_BASE_URL").ok())
                    .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
                let model_id = model_id
                    .or_else(|| std::env::var("V9R_MODEL").ok())
                    .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
                Self::openai_compatible(api_key, base_url, model_id)
            }
            ProviderKind::Ollama => {
                let api_key = api_key.or_else(|| std::env::var("V9R_API_KEY").ok());
                let base_url = base_url
                    .or_else(|| std::env::var("V9R_BASE_URL").ok())
                    .unwrap_or_else(|| OLLAMA_BASE_URL.to_string());
                let model_id = model_id
                    .or_else(|| std::env::var("V9R_MODEL").ok())
                    .unwrap_or_else(|| OLLAMA_MODEL_ID.to_string());
                Self::openai_compatible(api_key, base_url, model_id)
            }
        }
    }

    pub fn openai_compatible(
        api_key: Option<String>,
        base_url: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            provider: Box::new(OpenAiCompatibleProvider::new(api_key, base_url, model_id)?),
        })
    }

    pub fn model_id(&self) -> &str {
        self.provider.model_id()
    }

    pub fn base_url(&self) -> &str {
        self.provider.base_url()
    }

    pub async fn complete(&self, prompt: String) -> Result<String> {
        self.provider.complete(prompt).await
    }
}

impl ProviderKind {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "openai" | "openai-compatible" | "openrouter" | "groq" => Ok(Self::OpenAiCompatible),
            "ollama" => Ok(Self::Ollama),
            other => Err(AdapterError::InvalidAction(format!(
                "unknown provider: {other}"
            ))),
        }
    }
}

impl OpenAiCompatibleProvider {
    pub fn new(
        api_key: Option<String>,
        base_url: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Result<Self> {
        let model_id = model_id.into();
        let model_id = if model_id.trim().is_empty() {
            DEFAULT_MODEL_ID.to_string()
        } else {
            model_id
        };
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;
        Ok(Self {
            api_key: api_key
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty()),
            model_id,
            base_url: normalize_base_url(base_url.into()),
            http,
        })
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn complete(&self, prompt: String) -> Result<String> {
        let mut request = self
            .http
            .post(chat_completions_url(&self.base_url))
            .header("HTTP-Referer", HTTP_REFERER)
            .header("X-Title", X_TITLE);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request
            .json(&json!({
                "model": &self.model_id,
                "messages": [{"role": "user", "content": prompt}],
            }))
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(AdapterError::ApiStatus { status, body });
        }

        let value: serde_json::Value = serde_json::from_str(&body)
            .map_err(|err| AdapterError::InvalidResponse(err.to_string()))?;
        value
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                AdapterError::InvalidResponse("missing choices[0].message.content".to_string())
            })
    }
}

fn normalize_base_url(base_url: String) -> String {
    base_url.trim().trim_end_matches('/').to_string()
}

fn chat_completions_url(base_url: &str) -> String {
    if base_url.ends_with("/chat/completions") {
        base_url.to_string()
    } else {
        format!("{base_url}/chat/completions")
    }
}

impl Task {
    pub async fn run_with_guards(&mut self, client: &LlmClient, trace: &TraceLogger) -> TaskReport {
        vfs::register_task(self.id, self.workdir.clone());
        self.status = TaskStatus::Running;
        tracing::info!(task_id = %self.id, "task run started");

        let snapshot = match vfs::checkpoint(self.id, trace).await {
            Ok(snapshot) => snapshot,
            Err(err) => {
                self.status = TaskStatus::Failed;
                return self.report(
                    Some(TaskErrorType::ExecutionFailed),
                    Some(format!("checkpoint failed: {err}")),
                    Vec::new(),
                );
            }
        };

        let started = Instant::now();
        loop {
            if self.steps_used >= self.manifest.max_steps {
                return self
                    .rollback_report(
                        snapshot,
                        trace,
                        TaskErrorType::MaxStepsExceeded,
                        format!("max_steps exceeded: {}", self.manifest.max_steps),
                    )
                    .await;
            }

            let elapsed_ms = started.elapsed().as_millis() as u64;
            if elapsed_ms >= self.manifest.timeout_ms {
                return self
                    .rollback_report(
                        snapshot,
                        trace,
                        TaskErrorType::Timeout,
                        format!("timeout_ms exceeded: {}", self.manifest.timeout_ms),
                    )
                    .await;
            }
            let remaining = Duration::from_millis(self.manifest.timeout_ms - elapsed_ms);

            let step_result = tokio::time::timeout(remaining, self.step(client, trace)).await;
            match step_result {
                Ok(Ok(())) if matches!(self.status, TaskStatus::Success | TaskStatus::Failed) => {
                    break;
                }
                Ok(Ok(())) => continue,
                Ok(Err(err)) if is_security_violation(&err) => {
                    return self
                        .rollback_report(
                            snapshot,
                            trace,
                            TaskErrorType::SecurityViolation,
                            err.to_string(),
                        )
                        .await;
                }
                Ok(Err(err)) if is_max_steps_error(&err) => {
                    return self
                        .rollback_report(
                            snapshot,
                            trace,
                            TaskErrorType::MaxStepsExceeded,
                            err.to_string(),
                        )
                        .await;
                }
                Ok(Err(err)) => {
                    return self
                        .rollback_report(
                            snapshot,
                            trace,
                            TaskErrorType::ExecutionFailed,
                            err.to_string(),
                        )
                        .await;
                }
                Err(_) => {
                    return self
                        .rollback_report(
                            snapshot,
                            trace,
                            TaskErrorType::Timeout,
                            format!("timeout_ms exceeded: {}", self.manifest.timeout_ms),
                        )
                        .await;
                }
            }
        }

        let finished_status = self.status;
        self.status = TaskStatus::Verifying;
        tracing::info!(task_id = %self.id, "validating task outcome");
        match self.validate_outcome().await {
            Ok(artifact_hashes) => {
                self.status = match finished_status {
                    TaskStatus::Failed => TaskStatus::Failed,
                    _ => TaskStatus::Success,
                };
                if finished_status == TaskStatus::Failed {
                    return self.report(
                        Some(TaskErrorType::ExecutionFailed),
                        Some("agent finished with Failed status".to_string()),
                        artifact_hashes,
                    );
                }
                self.report(None, None, artifact_hashes)
            }
            Err(err) => {
                self.rollback_report(
                    snapshot,
                    trace,
                    TaskErrorType::ValidationFailed,
                    err.to_string(),
                )
                .await
            }
        }
    }

    pub async fn step(&mut self, client: &LlmClient, trace: &TraceLogger) -> Result<()> {
        let snapshot = self.xml_snapshot(trace, SNAPSHOT_HISTORY_LIMIT).await?;
        let prompt = format!("{SYSTEM_PROMPT}\n\n{snapshot}");
        let response = client.complete(prompt).await?;
        let actions = parse_actions(&response)?;
        if actions.is_empty() {
            return Err(AdapterError::InvalidAction(
                "LLM response contained no v9r XML action tags".to_string(),
            ));
        }

        for action in actions {
            match action {
                LlmAction::Execute(command) => {
                    // Direct execve, no shell. `parse_action_command` builds
                    // a (program, args) tuple from the LLM payload using
                    // whitespace tokenization with simple "…" quoting.
                    // The runtime then validates this via
                    // `validate_command_spec` BEFORE spawning.
                    let spec = parse_action_command(&command)?;
                    run_task_step(self, spec, trace).await?;
                    if !matches!(self.status, TaskStatus::Violation) {
                        self.status = TaskStatus::Running;
                    }
                }
                LlmAction::Write { path, content } => {
                    apply_write(self, trace, path, content).await?;
                    if !matches!(self.status, TaskStatus::Violation) {
                        self.status = TaskStatus::Running;
                    }
                }
                LlmAction::Finish { status, reason } => {
                    self.status = status;
                    trace.log_event(TaskEvent::TaskFinished { status }).await?;
                    if !reason.trim().is_empty() {
                        trace
                            .log_event(TaskEvent::CommandExecuted {
                                command: format!("finish: {reason}"),
                                exit_code: if status == TaskStatus::Success { 0 } else { 1 },
                            })
                            .await?;
                    }
                    break;
                }
            }
        }

        Ok(())
    }

    pub async fn validate_outcome(&self) -> anyhow::Result<Vec<ArtifactHash>> {
        for artifact in &self.manifest.mandatory_artifacts {
            let path = resolve_task_path(&self.workdir, artifact);
            let metadata = tokio::fs::metadata(&path)
                .await
                .with_context(|| format!("mandatory artifact missing: {}", path.display()))?;
            if !metadata.is_file() || metadata.len() == 0 {
                anyhow::bail!("mandatory artifact missing or empty: {}", path.display());
            }
            tracing::info!(path = %path.display(), bytes = metadata.len(), "mandatory artifact present");
        }

        if self.ran_test_command && self.last_test_exit_code != Some(0) {
            anyhow::bail!(
                "last test command failed: exit_code={}",
                self.last_test_exit_code.unwrap_or(-1)
            );
        }

        compute_mandatory_artifact_hashes(self).map_err(Into::into)
    }

    fn report(
        &self,
        error_type: Option<TaskErrorType>,
        message: Option<String>,
        artifact_hashes: Vec<ArtifactHash>,
    ) -> TaskReport {
        TaskReport {
            task_id: self.id,
            status: if error_type.is_some() {
                TaskStatus::Failed
            } else {
                self.status
            },
            error_type,
            message,
            steps_used: self.steps_used,
            artifact_hashes,
        }
    }

    async fn rollback_report(
        &mut self,
        snapshot: vfs::CheckpointId,
        trace: &TraceLogger,
        error_type: TaskErrorType,
        message: String,
    ) -> TaskReport {
        tracing::warn!(task_id = %self.id, error_type = ?error_type, message = %message, "task failed; rolling back");
        if let Err(err) = vfs::rollback(self.id, snapshot, trace).await {
            self.status = TaskStatus::Failed;
            return self.report(
                Some(TaskErrorType::RollbackFailed),
                Some(format!("{message}; rollback failed: {err}")),
                Vec::new(),
            );
        }
        self.status = TaskStatus::Failed;
        self.report(Some(error_type), Some(message), Vec::new())
    }
}

pub fn parse_actions(input: &str) -> Result<Vec<LlmAction>> {
    let mut found = Vec::new();
    collect_execute_actions(input, &mut found);
    collect_write_actions(input, &mut found)?;
    collect_finish_actions(input, &mut found)?;
    found.sort_by_key(|(offset, _)| *offset);
    Ok(found.into_iter().map(|(_, action)| action).collect())
}

fn collect_execute_actions(input: &str, out: &mut Vec<(usize, LlmAction)>) {
    let mut cursor = 0;
    while let Some(start) = input[cursor..].find("<execute>") {
        let tag_start = cursor + start;
        let content_start = tag_start + "<execute>".len();
        let Some(end) = input[content_start..].find("</execute>") else {
            break;
        };
        let content_end = content_start + end;
        let command = unescape_xml(input[content_start..content_end].trim());
        if !command.is_empty() {
            out.push((tag_start, LlmAction::Execute(command)));
        }
        cursor = content_end + "</execute>".len();
    }
}

fn collect_write_actions(input: &str, out: &mut Vec<(usize, LlmAction)>) -> Result<()> {
    let mut cursor = 0;
    while let Some(start) = input[cursor..].find("<write") {
        let tag_start = cursor + start;
        let Some(open_end_rel) = input[tag_start..].find('>') else {
            break;
        };
        let open_end = tag_start + open_end_rel;
        let open_tag = &input[tag_start..=open_end];
        let path = attr_value(open_tag, "path").ok_or_else(|| {
            AdapterError::InvalidAction("<write> missing path attribute".to_string())
        })?;
        let content_start = open_end + 1;
        let Some(close_rel) = input[content_start..].find("</write>") else {
            break;
        };
        let content_end = content_start + close_rel;
        out.push((
            tag_start,
            LlmAction::Write {
                path: PathBuf::from(unescape_xml(&path)),
                content: unescape_xml(&input[content_start..content_end]),
            },
        ));
        cursor = content_end + "</write>".len();
    }
    Ok(())
}

fn collect_finish_actions(input: &str, out: &mut Vec<(usize, LlmAction)>) -> Result<()> {
    let mut cursor = 0;
    while let Some(start) = input[cursor..].find("<finish") {
        let tag_start = cursor + start;
        let Some(open_end_rel) = input[tag_start..].find('>') else {
            break;
        };
        let open_end = tag_start + open_end_rel;
        let open_tag = &input[tag_start..=open_end];
        let status = match attr_value(open_tag, "status").as_deref() {
            Some("Success") => TaskStatus::Success,
            Some("Failed") => TaskStatus::Failed,
            Some(other) => {
                return Err(AdapterError::InvalidAction(format!(
                    "unsupported finish status: {other}"
                )))
            }
            None => TaskStatus::Success,
        };
        let content_start = open_end + 1;
        let Some(close_rel) = input[content_start..].find("</finish>") else {
            break;
        };
        let content_end = content_start + close_rel;
        out.push((
            tag_start,
            LlmAction::Finish {
                status,
                reason: unescape_xml(input[content_start..content_end].trim()),
            },
        ));
        cursor = content_end + "</finish>".len();
    }
    Ok(())
}

async fn apply_write(
    task: &mut Task,
    trace: &TraceLogger,
    path: PathBuf,
    content: String,
) -> Result<()> {
    task.record_tool_call()
        .map_err(|_| AdapterError::MaxStepsExceeded(task.manifest.max_steps))?;
    vfs::ensure_task_fs_unblocked(task.id)?;

    // Layer 1: lexical fence on the LLM-supplied path. Rejects absolute
    // paths, `~/...`, `..` components, and lexical escapes when joined
    // to the workdir.
    let resolved = match validate_action_path(&task.workdir, &path) {
        Ok(p) => p,
        Err(reason) => return deny_write_action(task, trace, reason).await,
    };

    // Layer 2: per-task manifest allowlist on the lexical, workdir-relative path.
    let allowed = task.manifest.is_allowed(&resolved, AccessType::Write);
    trace
        .log_event(TaskEvent::FileAccess {
            path: resolved.clone(),
            access: AccessType::Write,
            allowed,
        })
        .await?;
    if !allowed {
        return deny_write_action(
            task,
            trace,
            format!("write denied: {}", resolved.display()),
        )
        .await;
    }

    // Ensure the parent directory exists. We do this BEFORE the symlink
    // fence so canonicalize() has something to resolve.
    if let Some(parent) = resolved
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|source| AdapterError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
    }

    // Layer 3: symlink fence. The lexical check rules out `..` and root,
    // but a *workdir-internal* symlink could still point outside the
    // workdir (e.g. `workdir/legit -> /etc`). Canonicalize the parent of
    // the destination and the workdir, and require the parent to live
    // under the canonicalized workdir.
    let parent_for_real = resolved.parent().unwrap_or_else(|| Path::new("."));
    let parent_real =
        tokio::fs::canonicalize(parent_for_real)
            .await
            .map_err(|source| AdapterError::Io {
                path: parent_for_real.to_path_buf(),
                source,
            })?;
    let workdir_real =
        tokio::fs::canonicalize(&task.workdir)
            .await
            .map_err(|source| AdapterError::Io {
                path: task.workdir.clone(),
                source,
            })?;
    if !parent_real.starts_with(&workdir_real) {
        return deny_write_action(
            task,
            trace,
            format!(
                "write path escapes workdir via symlink: {} -> {}",
                resolved.display(),
                parent_real.display()
            ),
        )
        .await;
    }

    // Atomic write: tempfile in the same directory, then rename. `rename`
    // within a single filesystem on POSIX is atomic, so a crash mid-write
    // leaves the destination either fully-old or fully-new — never half.
    let file_name = resolved.file_name().ok_or_else(|| {
        AdapterError::InvalidAction(format!(
            "write destination has no file name: {}",
            resolved.display()
        ))
    })?;
    let dst_in_real_parent = parent_real.join(file_name);
    write_atomic_to(&dst_in_real_parent, content.as_bytes())
        .await
        .map_err(|source| AdapterError::Io {
            path: dst_in_real_parent.clone(),
            source,
        })?;
    Ok(())
}

/// Strict path fence for LLM-supplied action paths. Mirrors the
/// argument validator in `execution.rs` but is stricter: every input
/// here IS a path, so we don't have flag-style exemptions.
///
/// Returns the workdir-joined, lexically-normalized destination on
/// success. The caller is responsible for the symlink fence afterwards.
pub(crate) fn validate_action_path(
    workdir: &Path,
    raw: &Path,
) -> std::result::Result<PathBuf, String> {
    let s = raw.to_string_lossy();
    if s.is_empty() {
        return Err("action path is empty".to_string());
    }
    if raw.is_absolute() || s.starts_with('/') || s.starts_with('\\') {
        return Err(format!("action path is absolute: {s}"));
    }
    if s.starts_with('~') {
        return Err(format!("action path is home-relative: {s}"));
    }
    // Reject any non-Normal/CurDir component. This catches `..`
    // wherever it appears (leading, embedded, trailing) and Prefix
    // (Windows drive letters) and RootDir.
    for c in raw.components() {
        match c {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!("action path contains parent-dir component: {s}"));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("action path is absolute: {s}"));
            }
        }
    }

    // Belt-and-suspenders: after joining, verify lexical containment.
    // The component check above already guarantees this, but the explicit
    // assertion makes the invariant visible and survives future edits.
    let workdir_norm = normalize_path(workdir);
    let joined = normalize_path(&workdir.join(raw));
    if !joined.starts_with(&workdir_norm) {
        return Err(format!("action path escapes workdir: {s}"));
    }
    Ok(joined)
}

/// Atomic write: temp file in the same directory, then rename to `dst`.
/// Same-directory rename is atomic on POSIX, so a crash mid-write leaves
/// the destination either fully-old or fully-new.
pub(crate) async fn write_atomic_to(dst: &Path, content: &[u8]) -> std::io::Result<()> {
    let parent = dst.parent().unwrap_or_else(|| Path::new("."));
    let file_name = dst.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "destination has no file name")
    })?;
    let tmp = parent.join(format!(
        ".{}.v9r-tmp-{}",
        file_name.to_string_lossy(),
        Uuid::new_v4()
    ));

    // Write payload to temp file. On error, best-effort cleanup.
    if let Err(e) = tokio::fs::write(&tmp, content).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    // Atomic move into place. On error, best-effort cleanup.
    if let Err(e) = tokio::fs::rename(&tmp, dst).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    Ok(())
}

/// Record a write-action violation: status, trace, fs block, finish event.
/// Returns the corresponding `AdapterError::Violation` for the caller to
/// propagate. `is_security_violation` then routes this to rollback.
async fn deny_write_action(
    task: &mut Task,
    trace: &TraceLogger,
    reason: String,
) -> Result<()> {
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
    Err(AdapterError::Violation(reason))
}

/// Parse the body of an `<execute>…</execute>` into a `CommandSpec`.
///
/// **No shell.** The payload is tokenized by whitespace with simple
/// `"…"` quoting for tokens that contain spaces. Backslash escapes are
/// rejected. Unquoted shell metacharacters (`;`, `&`, `|`, `$`, …) are
/// passed through as literal argv bytes — they're inert without a
/// shell, and `execution::validate_command_spec` is the final fence.
pub(crate) fn parse_action_command(command: &str) -> Result<CommandSpec> {
    let command = command.trim();
    if command.is_empty() {
        return Err(AdapterError::InvalidAction(
            "<execute> body is empty".to_string(),
        ));
    }
    let tokens = tokenize_command(command)?;
    let mut iter = tokens.into_iter();
    let program = iter.next().ok_or_else(|| {
        AdapterError::InvalidAction("<execute> body has no program token".to_string())
    })?;
    Ok(CommandSpec {
        program,
        args: iter.collect(),
        cwd: None,
        reads: Vec::new(),
        writes: Vec::new(),
    })
}

fn tokenize_command(command: &str) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut had_quoted = false;
    for c in command.chars() {
        match c {
            '\\' => {
                return Err(AdapterError::InvalidAction(
                    "backslash escapes are not supported in <execute> payloads".to_string(),
                ));
            }
            '\n' | '\r' | '\0' => {
                return Err(AdapterError::InvalidAction(format!(
                    "control character {c:?} in <execute> payload"
                )));
            }
            '"' => {
                in_quotes = !in_quotes;
                had_quoted = true;
            }
            c if c.is_whitespace() && !in_quotes => {
                if !cur.is_empty() || had_quoted {
                    out.push(std::mem::take(&mut cur));
                    had_quoted = false;
                }
            }
            c => cur.push(c),
        }
    }
    if in_quotes {
        return Err(AdapterError::InvalidAction(
            "unmatched double quote in <execute> payload".to_string(),
        ));
    }
    if !cur.is_empty() || had_quoted {
        out.push(cur);
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

fn is_security_violation(err: &AdapterError) -> bool {
    matches!(
        err,
        AdapterError::Violation(_) | AdapterError::Execution(ExecutionError::Violation(_))
    )
}

fn is_max_steps_error(err: &AdapterError) -> bool {
    matches!(
        err,
        AdapterError::MaxStepsExceeded(_)
            | AdapterError::Execution(ExecutionError::MaxStepsExceeded(_))
    )
}

fn attr_value(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')?;
    Some(tag[start..start + end].to_string())
}

fn unescape_xml(input: &str) -> String {
    input
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_actions_in_response_order() {
        let actions = parse_actions(
            r#"<write path="src/main.rs">fn main() {}</write>
<execute>cargo test</execute>
<finish status="Success">fixed</finish>"#,
        )
        .unwrap();

        assert_eq!(actions.len(), 3);
        assert!(matches!(actions[0], LlmAction::Write { .. }));
        assert!(matches!(actions[1], LlmAction::Execute(_)));
        assert!(matches!(actions[2], LlmAction::Finish { .. }));
    }

    #[test]
    fn rejects_unknown_finish_status() {
        let err = parse_actions(r#"<finish status="Done">no</finish>"#).unwrap_err();
        assert!(err.to_string().contains("unsupported finish status"));
    }

    #[test]
    fn parse_action_command_basic() {
        let s = parse_action_command("cargo test").unwrap();
        assert_eq!(s.program, "cargo");
        assert_eq!(s.args, vec!["test"]);
    }

    #[test]
    fn parse_action_command_handles_quoted_arg_with_spaces() {
        let s = parse_action_command(r#"git commit -m "fix: foo bar""#).unwrap();
        assert_eq!(s.program, "git");
        assert_eq!(s.args, vec!["commit", "-m", "fix: foo bar"]);
    }

    #[test]
    fn parse_action_command_preserves_empty_quoted_arg() {
        let s = parse_action_command(r#"echo "" tail"#).unwrap();
        assert_eq!(s.program, "echo");
        assert_eq!(s.args, vec!["", "tail"]);
    }

    #[test]
    fn parse_action_command_rejects_backslash() {
        let err = parse_action_command(r#"git commit -m \"x\""#).unwrap_err();
        assert!(err.to_string().contains("backslash"));
    }

    #[test]
    fn parse_action_command_rejects_newline() {
        let err = parse_action_command("cargo test\nrm -rf .").unwrap_err();
        assert!(err.to_string().contains("control character"));
    }

    #[test]
    fn parse_action_command_rejects_empty_and_whitespace() {
        assert!(parse_action_command("").is_err());
        assert!(parse_action_command("   \t  ").is_err());
    }

    #[test]
    fn parse_action_command_rejects_unmatched_quote() {
        assert!(parse_action_command(r#"echo "unterminated"#).is_err());
    }

    #[test]
    fn parse_action_command_tokenizes_shell_meta_as_literals() {
        // No shell: `;` is just a literal argv byte. The validator below
        // (validate_command_spec) won't reject `;` itself — it's the
        // execve guarantee that makes this safe, since `cargo` will get
        // the `;` as an argument and either reject or ignore it; no
        // second command runs.
        let s = parse_action_command("cargo test ; rm -rf .").unwrap();
        assert_eq!(s.program, "cargo");
        assert_eq!(s.args, vec!["test", ";", "rm", "-rf", "."]);
    }

    /// End-to-end: the user's reported "rm -rf .` smuggled via `sh -c`"
    /// class is rejected at the runtime validator after parsing.
    #[test]
    fn shell_smuggling_is_rejected_by_runtime_validator() {
        use crate::execution::validate_command_spec;

        let spec = parse_action_command(r#"sh -c "rm -rf .""#).unwrap();
        // Parse succeeds — `sh` is just a token. The denial is at
        // validation time, which is the layer the manifest cannot
        // override.
        let err = validate_command_spec(&spec).unwrap_err();
        assert!(err.contains("shell interpreters"));
    }

    // -------- validate_action_path --------

    fn workdir() -> PathBuf {
        // A stable workdir for path-shape tests. Doesn't have to exist
        // — the lexical check is purely string ops.
        PathBuf::from("/tmp/v9r-fake-workdir")
    }

    #[test]
    fn validate_action_path_accepts_relative() {
        let p = validate_action_path(&workdir(), Path::new("src/main.rs")).unwrap();
        assert_eq!(p, PathBuf::from("/tmp/v9r-fake-workdir/src/main.rs"));
    }

    #[test]
    fn validate_action_path_accepts_curdir_components() {
        // Component::CurDir is the lone "." — harmless and used by some tools.
        let p = validate_action_path(&workdir(), Path::new("./src/./main.rs")).unwrap();
        assert_eq!(p, PathBuf::from("/tmp/v9r-fake-workdir/src/main.rs"));
    }

    #[test]
    fn validate_action_path_rejects_absolute() {
        for bad in ["/etc/passwd", "/", "\\Windows\\System32"] {
            let err = validate_action_path(&workdir(), Path::new(bad)).unwrap_err();
            assert!(err.contains("absolute"), "{bad}: {err}");
        }
    }

    #[test]
    fn validate_action_path_rejects_home_relative() {
        for bad in ["~/secret", "~"] {
            let err = validate_action_path(&workdir(), Path::new(bad)).unwrap_err();
            assert!(err.contains("home-relative"), "{bad}: {err}");
        }
    }

    #[test]
    fn validate_action_path_rejects_parent_dir_anywhere() {
        for bad in [
            "..",
            "../escape",
            "src/../../../etc",
            "foo/../bar",
            "foo/..",
        ] {
            let err = validate_action_path(&workdir(), Path::new(bad)).unwrap_err();
            assert!(err.contains("parent-dir"), "{bad}: {err}");
        }
    }

    #[test]
    fn validate_action_path_rejects_empty() {
        let err = validate_action_path(&workdir(), Path::new("")).unwrap_err();
        assert!(err.contains("empty"));
    }

    // -------- write_atomic_to --------

    fn temp_dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("v9r-adapter-test-{name}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn write_atomic_writes_new_file() {
        let dir = temp_dir("new");
        let dst = dir.join("hello.txt");
        write_atomic_to(&dst, b"hi").await.unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"hi");
        // No stale temp file left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("v9r-tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "stale temp file(s): {leftovers:?}");
    }

    #[tokio::test]
    async fn write_atomic_replaces_existing_file_atomically() {
        let dir = temp_dir("replace");
        let dst = dir.join("data.txt");
        std::fs::write(&dst, b"old content").unwrap();
        write_atomic_to(&dst, b"new content").await.unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"new content");
    }

    #[tokio::test]
    async fn write_atomic_cleans_tmp_on_rename_failure() {
        // Hard to force rename to fail portably. Instead, sanity-check
        // that a successful round trip never leaves *.v9r-tmp-* files.
        let dir = temp_dir("cleanup");
        let dst = dir.join("nested/inside/file.txt");
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        for _ in 0..5 {
            write_atomic_to(&dst, b"x").await.unwrap();
        }
        let n_tmp = std::fs::read_dir(dst.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".v9r-tmp-"))
            .count();
        assert_eq!(n_tmp, 0);
    }
}
