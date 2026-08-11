use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::bundle::{self, ArtifactHash, BundleError};
use crate::context::{ContextError, TaskSnapshot};
use crate::manifest::Manifest;
use crate::trace::TraceLogger;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Idle,
    Running,
    Verifying,
    Success,
    Failed,
    Violation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub manifest: Manifest,
    pub workdir: PathBuf,
    pub status: TaskStatus,
    pub steps_used: usize,
    #[serde(default)]
    pub tokens_used: u64,
    pub ran_test_command: bool,
    pub last_test_exit_code: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskErrorType {
    ValidationFailed,
    SecurityViolation,
    MaxStepsExceeded,
    TokenLimitExceeded,
    Timeout,
    ExecutionFailed,
    RollbackFailed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskReport {
    pub task_id: Uuid,
    pub status: TaskStatus,
    pub error_type: Option<TaskErrorType>,
    pub message: Option<String>,
    pub steps_used: usize,
    pub artifact_hashes: Vec<ArtifactHash>,
}

impl Task {
    pub fn new(manifest: Manifest, workdir: PathBuf) -> Self {
        Self {
            id: Uuid::new_v4(),
            manifest,
            workdir,
            status: TaskStatus::Idle,
            steps_used: 0,
            tokens_used: 0,
            ran_test_command: false,
            last_test_exit_code: None,
        }
    }

    pub async fn xml_snapshot(
        &self,
        trace: &TraceLogger,
        history_limit: usize,
    ) -> Result<String, ContextError> {
        Ok(TaskSnapshot::collect(self, trace, history_limit)
            .await?
            .to_xml())
    }

    pub fn export_bundle(&self) -> Vec<u8> {
        self.try_export_bundle()
            .expect("failed to export v9r task bundle")
    }

    pub fn try_export_bundle(&self) -> Result<Vec<u8>, BundleError> {
        bundle::export_bundle(self, None)
    }

    pub fn export_bundle_with_trace(&self, trace: &TraceLogger) -> Result<Vec<u8>, BundleError> {
        bundle::export_bundle_with_trace(self, trace)
    }

    pub fn from_bundle(
        data: &[u8],
        new_manifest: Manifest,
        new_workdir: PathBuf,
    ) -> Result<Self, BundleError> {
        bundle::import_bundle(data, new_manifest, new_workdir)
    }

    pub fn record_tool_call(&mut self) -> anyhow::Result<()> {
        if self.steps_used >= self.manifest.max_steps {
            anyhow::bail!("max_steps exceeded: {}", self.manifest.max_steps);
        }
        self.steps_used += 1;
        Ok(())
    }

    pub fn record_test_command(&mut self, command: &str, exit_code: i32) {
        if self.manifest.test_commands.iter().any(|test| {
            let test = test.trim();
            !test.is_empty() && (command == test || command.starts_with(test))
        }) {
            self.ran_test_command = true;
            self.last_test_exit_code = Some(exit_code);
        }
    }
}
