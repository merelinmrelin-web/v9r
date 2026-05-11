use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

use crate::manifest::{normalize_path, Manifest};
use crate::task::{Task, TaskStatus};
use crate::trace::TraceLogger;

pub const DEFAULT_MAX_FILE_BYTES: u64 = 64 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSnapshot {
    pub manifest: Manifest,
    pub status: TaskStatus,
    pub files: Vec<FileSnapshot>,
    pub execution_history: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSnapshot {
    pub path: PathBuf,
    pub content: String,
    pub truncated: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("context io at {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("trace: {0}")]
    Trace(#[from] crate::trace::TraceError),
}

pub type Result<T> = std::result::Result<T, ContextError>;

impl TaskSnapshot {
    pub async fn collect(task: &Task, trace: &TraceLogger, history_limit: usize) -> Result<Self> {
        let files = collect_allowed_files(task, DEFAULT_MAX_FILE_BYTES).await?;
        let execution_history = trace.recent_events(history_limit).await?;
        Ok(Self {
            manifest: task.manifest.clone(),
            status: task.status,
            files,
            execution_history,
        })
    }

    pub fn to_xml(&self) -> String {
        let mut out = String::new();
        out.push_str("<task_context>\n");
        out.push_str("  <system_manifest>\n");
        out.push_str(&format!(
            "    <status>{}</status>\n",
            escape_xml(&format!("{:?}", self.status))
        ));
        out.push_str(&format!(
            "    <token_limit>{}</token_limit>\n",
            self.manifest.token_limit
        ));
        out.push_str(&format!(
            "    <max_steps>{}</max_steps>\n",
            self.manifest.max_steps
        ));
        out.push_str(&format!(
            "    <timeout_ms>{}</timeout_ms>\n",
            self.manifest.timeout_ms
        ));
        write_path_list(&mut out, "allow_read", &self.manifest.allow_read);
        write_path_list(&mut out, "allow_write", &self.manifest.allow_write);
        write_path_list(
            &mut out,
            "mandatory_artifacts",
            &self.manifest.mandatory_artifacts,
        );
        out.push_str("    <allow_exec>\n");
        for command in &self.manifest.allow_exec {
            out.push_str(&format!(
                "      <command>{}</command>\n",
                escape_xml(command)
            ));
        }
        out.push_str("    </allow_exec>\n");
        out.push_str("    <test_commands>\n");
        for command in &self.manifest.test_commands {
            out.push_str(&format!(
                "      <command>{}</command>\n",
                escape_xml(command)
            ));
        }
        out.push_str("    </test_commands>\n");
        out.push_str("  </system_manifest>\n");

        out.push_str("  <environment_files>\n");
        for file in &self.files {
            let truncated = if file.truncated {
                " truncated=\"true\""
            } else {
                ""
            };
            out.push_str(&format!(
                "    <file path=\"{}\"{}>",
                escape_xml(&file.path.display().to_string()),
                truncated
            ));
            out.push_str(&escape_xml(&file.content));
            if file.truncated {
                out.push_str("\n[TRUNCATED]");
            }
            out.push_str("</file>\n");
        }
        out.push_str("  </environment_files>\n");

        out.push_str("  <execution_history>\n");
        for event in &self.execution_history {
            out.push_str(&format!("    <event>{}</event>\n", escape_xml(event)));
        }
        out.push_str("  </execution_history>\n");
        out.push_str("</task_context>\n");
        out
    }
}

async fn collect_allowed_files(task: &Task, max_file_bytes: u64) -> Result<Vec<FileSnapshot>> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = task
        .manifest
        .allow_read
        .iter()
        .map(|path| resolve_task_path(&task.workdir, path))
        .collect::<Vec<_>>();

    while let Some(path) = stack.pop() {
        let path = normalize_path(&path);
        if !seen.insert(path.clone()) {
            continue;
        }

        let metadata = match tokio::fs::symlink_metadata(&path).await {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(source) => return Err(io_error(path, source)),
        };

        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            let mut entries = tokio::fs::read_dir(&path)
                .await
                .map_err(|source| io_error(path.clone(), source))?;
            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|source| io_error(path.clone(), source))?
            {
                stack.push(entry.path());
            }
            continue;
        }
        if !metadata.is_file() {
            continue;
        }
        if path.file_name().is_some_and(|name| name == "trace.jsonl") {
            continue;
        }

        if let Some(file) = read_text_file(path, metadata.len(), max_file_bytes).await? {
            files.push(file);
        }
    }

    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(files)
}

async fn read_text_file(
    path: PathBuf,
    len: u64,
    max_file_bytes: u64,
) -> Result<Option<FileSnapshot>> {
    let read_limit = len.min(max_file_bytes) as usize;
    let file = tokio::fs::File::open(&path)
        .await
        .map_err(|source| io_error(path.clone(), source))?;
    let mut bytes = Vec::with_capacity(read_limit.saturating_add(1));
    file.take(max_file_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .await
        .map_err(|source| io_error(path.clone(), source))?;
    let truncated = bytes.len() > read_limit;
    bytes.truncate(read_limit);

    if bytes.contains(&0) {
        return Ok(None);
    }
    let content = match String::from_utf8(bytes) {
        Ok(content) => content,
        Err(_) => return Ok(None),
    };

    Ok(Some(FileSnapshot {
        path,
        content,
        truncated,
    }))
}

fn resolve_task_path(workdir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        normalize_path(path)
    } else {
        normalize_path(&workdir.join(path))
    }
}

fn write_path_list(out: &mut String, tag: &str, paths: &[PathBuf]) {
    out.push_str(&format!("    <{tag}>\n"));
    for path in paths {
        out.push_str(&format!(
            "      <path>{}</path>\n",
            escape_xml(&path.display().to_string())
        ));
    }
    out.push_str(&format!("    </{tag}>\n"));
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn io_error(path: PathBuf, source: io::Error) -> ContextError {
    ContextError::Io { path, source }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_escapes_content() {
        let snapshot = TaskSnapshot {
            manifest: Manifest {
                allow_read: vec![PathBuf::from("main.rs")],
                allow_write: Vec::new(),
                allow_exec: vec!["cargo".to_string()],
                token_limit: 100,
                max_steps: 8,
                timeout_ms: 30_000,
                mandatory_artifacts: Vec::new(),
                test_commands: Vec::new(),
            },
            status: TaskStatus::Idle,
            files: vec![FileSnapshot {
                path: PathBuf::from("main.rs"),
                content: "if a < b && c > d { }".to_string(),
                truncated: false,
            }],
            execution_history: vec!["{\"event\":\"ok\"}".to_string()],
        };

        let xml = snapshot.to_xml();
        assert!(xml.contains("if a &lt; b &amp;&amp; c &gt; d"));
    }
}
