use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;
use tokio::time::Instant;

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
<execute>shell command</execute>
Run a command. The runtime executes commands sequentially and checks the task manifest first.

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
                    run_task_step(self, shell_command(command), trace).await?;
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
    let path = resolve_task_path(&task.workdir, &path);
    let allowed = task.manifest.is_allowed(&path, AccessType::Write);
    trace
        .log_event(TaskEvent::FileAccess {
            path: path.clone(),
            access: AccessType::Write,
            allowed,
        })
        .await?;
    if !allowed {
        task.status = TaskStatus::Violation;
        let reason = format!("write denied: {}", path.display());
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
        return Err(AdapterError::Violation(reason));
    }

    if let Some(parent) = path
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
    tokio::fs::write(&path, content)
        .await
        .map_err(|source| AdapterError::Io { path, source })?;
    Ok(())
}

fn shell_command(command: String) -> CommandSpec {
    CommandSpec {
        program: "sh".to_string(),
        args: vec!["-c".to_string(), command],
        cwd: None,
        reads: Vec::new(),
        writes: Vec::new(),
    }
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
}
