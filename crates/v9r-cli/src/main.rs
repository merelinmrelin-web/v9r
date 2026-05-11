use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use tracing_subscriber::EnvFilter;
use v9r_core::adapter::{LlmClient, ProviderKind};
use v9r_core::bundle::TaskBundle;
use v9r_core::execution::{run_task_step, CommandSpec};
use v9r_core::manifest::{normalize_path, Manifest};
use v9r_core::task::{Task, TaskReport};
use v9r_core::trace::{TaskEvent, TraceLogger};
use v9r_core::vfs::{checkpoint, ensure_safe_directory, register_task, rollback};

mod repl;

#[derive(Debug)]
enum Command {
    Run(RunArgs),
    Inspect(InspectArgs),
    Repl,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RunArgs {
    pub(crate) task: Option<String>,
    pub(crate) manifest: Option<PathBuf>,
    pub(crate) workdir: Option<PathBuf>,
    pub(crate) script: Option<PathBuf>,
    pub(crate) provider: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) debug_xml: bool,
}

#[derive(Debug, Default)]
struct InspectArgs {
    bundle: Option<PathBuf>,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()))
        .without_time()
        .try_init()
        .ok();

    match parse_args(env::args().skip(1).collect())? {
        Command::Run(args) => run(args).await,
        Command::Inspect(args) => inspect(args),
        Command::Repl => repl::start().await,
    }
}

pub(crate) async fn run(args: RunArgs) -> Result<()> {
    let stdin_is_pipe = !io::stdin().is_terminal();
    let stdout_is_pipe = !io::stdout().is_terminal();
    let task_text = args.task.clone().context("run: missing --task")?;
    let manifest_path = args.manifest.clone().context("run: missing --manifest")?;
    let workdir = normalize_path(&args.workdir.clone().unwrap_or(env::current_dir()?));
    let manifest = normalize_manifest_paths(load_manifest(&manifest_path)?, &workdir);

    ensure_safe_directory(&workdir)?;

    log_info(&format!(
        "manifest: allow_read={}, allow_write={}, allow_exec={}",
        format_paths(&manifest.allow_read),
        format_paths(&manifest.allow_write),
        format_strings(&manifest.allow_exec)
    ));

    let mut imported_bundle = None;
    if stdin_is_pipe {
        log_info("stdin is a pipe: importing task bundle...");
        let mut data = Vec::new();
        io::stdin().read_to_end(&mut data)?;
        if data.is_empty() {
            return Err(anyhow!("stdin pipe contained no task bundle"));
        }
        let bundle = decode_bundle(&data)?;
        log_info(&format!(
            "bundle loaded: source_task_id={}, files={}",
            bundle.source_task_id,
            bundle.files.len()
        ));
        imported_bundle = Some(data);
    }

    fs::create_dir_all(&workdir)
        .with_context(|| format!("create workdir: {}", workdir.display()))?;
    log_info(&format!("task: {}", task_text.replace('\n', " ")));
    fs::write(workdir.join("task.txt"), task_text)
        .with_context(|| format!("write task file: {}", workdir.join("task.txt").display()))?;

    let mut task = if let Some(data) = imported_bundle {
        Task::from_bundle(&data, manifest, workdir.clone())?
    } else {
        let task = Task::new(manifest, workdir.clone());
        register_task(task.id, workdir.clone());
        task
    };
    log_info(&format!("task started: task_id={}", task.id));

    let trace = TraceLogger::new(workdir.join("trace.jsonl")).await?;
    if !stdin_is_pipe {
        trace
            .log_event(TaskEvent::task_started(task.manifest.clone()))
            .await?;
    }

    if args.debug_xml {
        emit_debug_xml(&task.xml_snapshot(&trace, 32).await?);
    }

    let consistent = if let Some(script) = args.script {
        let checkpoint_id = checkpoint(task.id, &trace).await?;
        log_info(&format!("checkpoint created: id={}", checkpoint_id.0));
        let run_result = match run_script(&mut task, &trace, &script).await {
            Ok(()) => task
                .validate_outcome()
                .await
                .map(|_| ())
                .map_err(Into::into),
            Err(err) => Err(err),
        };
        if let Err(err) = run_result {
            log_warn(&format!("run failed: {err:#}"));
            log_warn(&format!(
                "initiating rollback to checkpoint: {}",
                checkpoint_id.0
            ));
            rollback(task.id, checkpoint_id, &trace).await?;
            log_warn("rollback completed: original files restored; unrelated pre-existing files preserved");
            task.status = v9r_core::task::TaskStatus::Failed;
            false
        } else {
            true
        }
    } else {
        let report = run_llm_step(&mut task, &trace, &args).await?;
        if let Some(error_type) = report.error_type {
            log_warn(&format!(
                "task report: error_type={error_type:?} message={}",
                report.message.unwrap_or_default()
            ));
            false
        } else {
            true
        }
    };

    emit_trace_observability(&trace).await?;
    log_info(&format!("task finished: status={:?}", task.status));

    if !consistent {
        if !stdout_is_pipe {
            println!(
                "task_id={} status={:?} workdir={} bundle_exported=false",
                task.id,
                task.status,
                workdir.display()
            );
        }
        return Ok(());
    }

    let bundle = task.export_bundle_with_trace(&trace)?;
    if stdout_is_pipe {
        io::stdout().write_all(&bundle)?;
    } else {
        println!(
            "task_id={} status={:?} workdir={} bundle_bytes={}",
            task.id,
            task.status,
            workdir.display(),
            bundle.len()
        );
    }

    Ok(())
}

fn inspect(args: InspectArgs) -> Result<()> {
    let data = if let Some(path) = args.bundle {
        fs::read(&path).with_context(|| format!("read bundle: {}", path.display()))?
    } else if !io::stdin().is_terminal() {
        log_info("stdin is a pipe: importing task bundle...");
        let mut data = Vec::new();
        io::stdin().read_to_end(&mut data)?;
        data
    } else {
        return Err(anyhow!(
            "inspect: provide --bundle or pipe bundle into stdin"
        ));
    };

    let bundle = decode_bundle(&data)?;
    println!("bundle.version={}", bundle.version);
    println!("bundle.source_task_id={}", bundle.source_task_id);
    println!("bundle.source_status={:?}", bundle.source_status);
    println!("bundle.files={}", bundle.files.len());
    println!("bundle.artifact_hashes={}", bundle.artifact_hashes.len());
    for file in &bundle.files {
        println!(
            "file path={} bytes={} hash=fnv64:{:016x}",
            file.relative_path.display(),
            file.bytes.len(),
            fnv64(&file.bytes)
        );
    }
    for artifact in &bundle.artifact_hashes {
        println!(
            "artifact path={} bytes={} hash=fnv64:{:016x}",
            artifact.path.display(),
            artifact.bytes,
            artifact.fnv64
        );
    }
    println!("trace.begin");
    for line in String::from_utf8_lossy(&bundle.trace_jsonl).lines() {
        println!("{line}");
    }
    println!("trace.end");
    Ok(())
}

async fn run_script(task: &mut Task, trace: &TraceLogger, script: &Path) -> Result<()> {
    let script = normalize_path(script);
    let content = fs::read_to_string(&script)
        .with_context(|| format!("read script: {}", script.display()))?;
    log_exec(&format!("script: {}", script.display()));
    let output = run_task_step(
        task,
        CommandSpec {
            program: "sh".to_string(),
            args: vec!["-c".to_string(), content],
            cwd: Some(task.workdir.clone()),
            reads: Vec::new(),
            writes: Vec::new(),
        },
        trace,
    )
    .await?;
    emit_exec_output("stdout", &output.stdout);
    emit_exec_output("stderr", &output.stderr);
    Ok(())
}

async fn run_llm_step(task: &mut Task, trace: &TraceLogger, args: &RunArgs) -> Result<TaskReport> {
    let provider = ProviderKind::parse(args.provider.as_deref().unwrap_or("openai-compatible"))?;
    let client = LlmClient::from_config(provider, args.model.clone(), args.base_url.clone(), None)?;
    log_llm(&format!(
        "requesting inference (provider: {:?}, model: {}, base_url: {})...",
        provider,
        client.model_id(),
        client.base_url()
    ));
    let report = task.run_with_guards(&client, trace).await;
    log_llm("inference completed");
    Ok(report)
}

fn parse_args(args: Vec<String>) -> Result<Command> {
    let Some(command) = args.first().map(String::as_str) else {
        return Ok(Command::Repl);
    };
    match command {
        "run" => Ok(Command::Run(parse_run_args(&args[1..])?)),
        "inspect" => Ok(Command::Inspect(parse_inspect_args(&args[1..])?)),
        "repl" => Ok(Command::Repl),
        "help" | "--help" | "-h" => Err(anyhow!(usage())),
        other if other.starts_with('-') => Err(anyhow!("unknown argument: {other}\n{}", usage())),
        _ => Ok(Command::Run(shorthand_run_args(&args))),
    }
}

fn shorthand_run_args(args: &[String]) -> RunArgs {
    RunArgs {
        task: Some(args.join(" ")),
        manifest: Some(PathBuf::from("task.toml")),
        workdir: None,
        script: None,
        provider: Some("ollama".to_string()),
        model: Some("llama3".to_string()),
        base_url: None,
        debug_xml: false,
    }
}

fn parse_run_args(args: &[String]) -> Result<RunArgs> {
    let mut out = RunArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--task" => {
                i += 1;
                out.task = Some(value(args, i, "--task")?.to_string());
            }
            "--manifest" => {
                i += 1;
                out.manifest = Some(PathBuf::from(value(args, i, "--manifest")?));
            }
            "--workdir" => {
                i += 1;
                out.workdir = Some(PathBuf::from(value(args, i, "--workdir")?));
            }
            "--provider" => {
                i += 1;
                out.provider = Some(value(args, i, "--provider")?.to_string());
            }
            "--model" => {
                i += 1;
                out.model = Some(value(args, i, "--model")?.to_string());
            }
            "--base-url" => {
                i += 1;
                out.base_url = Some(value(args, i, "--base-url")?.to_string());
            }
            "-f" => {
                i += 1;
                out.script = Some(PathBuf::from(value(args, i, "-f")?));
            }
            "--debug-xml" => out.debug_xml = true,
            other => return Err(anyhow!("run: unknown argument: {other}")),
        }
        i += 1;
    }
    Ok(out)
}

fn parse_inspect_args(args: &[String]) -> Result<InspectArgs> {
    let mut out = InspectArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bundle" => {
                i += 1;
                out.bundle = Some(PathBuf::from(value(args, i, "--bundle")?));
            }
            other => return Err(anyhow!("inspect: unknown argument: {other}")),
        }
        i += 1;
    }
    Ok(out)
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| anyhow!("missing value for {flag}"))
}

fn load_manifest(path: &Path) -> Result<Manifest> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read manifest: {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse manifest: {}", path.display()))
}

fn normalize_manifest_paths(mut manifest: Manifest, workdir: &Path) -> Manifest {
    manifest.allow_read = manifest
        .allow_read
        .into_iter()
        .map(|path| resolve_path(workdir, &path))
        .collect();
    manifest.allow_write = manifest
        .allow_write
        .into_iter()
        .map(|path| resolve_path(workdir, &path))
        .collect();
    manifest.mandatory_artifacts = manifest
        .mandatory_artifacts
        .into_iter()
        .map(|path| resolve_path(workdir, &path))
        .collect();
    manifest
}

fn resolve_path(workdir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        normalize_path(path)
    } else {
        normalize_path(&workdir.join(path))
    }
}

fn decode_bundle(data: &[u8]) -> Result<TaskBundle> {
    bincode::deserialize(data).context("decode task bundle")
}

async fn emit_trace_observability(trace: &TraceLogger) -> Result<()> {
    for line in trace.recent_events(64).await? {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if let Some(event) = value.get("FileAccess") {
            let path = event
                .get("path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let access = event
                .get("access")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let allowed = event
                .get("allowed")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            log_step(&format!(
                "file_access: {} {} ({})",
                access.to_ascii_lowercase(),
                path,
                if allowed { "allowed" } else { "denied" }
            ));
        } else if let Some(event) = value.get("ViolationOccurred") {
            let reason = event
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            log_warn(&format!("violation: {reason}"));
        } else if let Some(event) = value.get("CommandExecuted") {
            let command = event
                .get("command")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let exit_code = event
                .get("exit_code")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(-1);
            log_exec(&format!("command: {command} exit_code={exit_code}"));
        }
    }
    Ok(())
}

fn emit_exec_output(stream: &str, bytes: &[u8]) {
    for line in String::from_utf8_lossy(bytes).lines() {
        log_exec(&format!("{stream}: {line}"));
    }
}

fn emit_debug_xml(xml: &str) {
    log_llm("xml_context_begin");
    for line in xml.lines() {
        log_llm(&format!("xml: {line}"));
    }
    log_llm("xml_context_end");
}

fn format_paths(paths: &[PathBuf]) -> String {
    format!(
        "[{}]",
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn format_strings(values: &[String]) -> String {
    format!("[{}]", values.join(","))
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn log_info(message: &str) {
    eprintln!("[INFO] {message}");
}

fn log_step(message: &str) {
    eprintln!("[STEP] {message}");
}

fn log_warn(message: &str) {
    eprintln!("[WARN] {message}");
}

fn log_exec(message: &str) {
    eprintln!("[EXEC] {message}");
}

fn log_llm(message: &str) {
    eprintln!("[LLM] {message}");
}

fn usage() -> &'static str {
    "usage:\n  v9r repl\n  v9r \"task description\"\n  v9r run --task <text> --manifest <path> [--workdir <path>] [--provider <name>] [--model <id>] [--base-url <url>] [-f <script>] [--debug-xml]\n  v9r inspect [--bundle <path>]"
}
