use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};

use crate::{run, RunArgs};

#[derive(Clone, Debug)]
pub struct ReplSession {
    pub current_model: String,
    pub current_provider: String,
    pub manifest_path: Option<PathBuf>,
    pub workdir: PathBuf,
    pub base_url: Option<String>,
    pub debug_xml: bool,
}

#[derive(Clone, Default)]
struct ReplHelper;

impl Helper for ReplHelper {}
impl Validator for ReplHelper {}
impl Hinter for ReplHelper {
    type Hint = String;
}

impl Highlighter for ReplHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        Cow::Borrowed(line)
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let before = &line[..pos];
        let start = before.rfind(' ').map(|index| index + 1).unwrap_or(0);
        let word = &before[start..];
        let candidates = [
            "set",
            "load",
            "run",
            "status",
            "exit",
            "model",
            "provider",
            "manifest",
            "workdir",
            "base-url",
            "debug-xml",
            "ollama",
            "openai-compatible",
            "llama3",
        ];
        let pairs = candidates
            .iter()
            .filter(|candidate| candidate.starts_with(word))
            .map(|candidate| Pair {
                display: candidate.to_string(),
                replacement: candidate.to_string(),
            })
            .collect();
        Ok((start, pairs))
    }
}

impl ReplSession {
    pub fn new() -> Result<Self> {
        Ok(Self {
            current_model: "llama3".to_string(),
            current_provider: "ollama".to_string(),
            manifest_path: default_manifest(),
            workdir: std::env::current_dir()?,
            base_url: None,
            debug_xml: false,
        })
    }

    async fn execute(&mut self, line: &str) -> Result<bool> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(true);
        }
        if matches!(line, "exit" | "quit") {
            return Ok(false);
        }
        if line == "status" {
            self.print_status();
            return Ok(true);
        }
        if let Some(rest) = line.strip_prefix("set ") {
            self.set(rest)?;
            return Ok(true);
        }
        if let Some(path) = line.strip_prefix("load ") {
            self.load(path.trim())?;
            return Ok(true);
        }
        if let Some(task) = line.strip_prefix("run ") {
            self.run_task(task.trim()).await?;
            return Ok(true);
        }

        Err(anyhow!("unknown command: {line}"))
    }

    fn set(&mut self, input: &str) -> Result<()> {
        let mut parts = input
            .splitn(2, char::is_whitespace)
            .filter(|part| !part.is_empty());
        let key = parts.next().ok_or_else(|| anyhow!("set: missing key"))?;
        let value = parts
            .next()
            .ok_or_else(|| anyhow!("set: missing value"))?
            .trim();
        match key {
            "model" => self.current_model = value.to_string(),
            "provider" => self.current_provider = value.to_string(),
            "manifest" => self.manifest_path = Some(PathBuf::from(value)),
            "workdir" => self.workdir = PathBuf::from(value),
            "base-url" => self.base_url = Some(value.to_string()),
            "debug-xml" => self.debug_xml = parse_bool(value)?,
            other => return Err(anyhow!("set: unknown key: {other}")),
        }
        eprintln!("[INFO] set: {key}={value}");
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        if path.is_empty() {
            return Err(anyhow!("load: missing manifest path"));
        }
        let path = PathBuf::from(path);
        if !path.exists() {
            return Err(anyhow!("load: manifest not found: {}", path.display()));
        }
        self.manifest_path = Some(path.clone());
        eprintln!("[INFO] manifest loaded: {}", path.display());
        Ok(())
    }

    async fn run_task(&self, task: &str) -> Result<()> {
        let task = parse_task_text(task)?;
        let manifest = self
            .manifest_path
            .clone()
            .ok_or_else(|| anyhow!("run: no manifest loaded"))?;
        eprintln!(
            "[INFO] rollback policy: restore modified files from .v9r/backups; delete only files created during this transaction"
        );
        run(RunArgs {
            task: Some(task),
            manifest: Some(manifest),
            workdir: Some(self.workdir.clone()),
            script: None,
            provider: Some(self.current_provider.clone()),
            model: Some(self.current_model.clone()),
            base_url: self.base_url.clone(),
            debug_xml: self.debug_xml,
        })
        .await?;
        eprintln!("[INFO] rollback policy: original pre-existing files preserved");
        Ok(())
    }

    fn print_status(&self) {
        println!("provider={}", self.current_provider);
        println!("model={}", self.current_model);
        println!(
            "manifest={}",
            self.manifest_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unset>".to_string())
        );
        println!("workdir={}", self.workdir.display());
        println!(
            "base_url={}",
            self.base_url.as_deref().unwrap_or("<provider-default>")
        );
        println!("debug_xml={}", self.debug_xml);
    }
}

pub async fn start() -> Result<()> {
    let mut session = ReplSession::new()?;
    let mut rl: Editor<ReplHelper, DefaultHistory> = Editor::new()?;
    rl.set_helper(Some(ReplHelper));
    let history = history_path();
    let mut persisted_history = load_history(&mut rl, &history);

    eprintln!("[INFO] repl started");
    loop {
        match rl.readline("v9r> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                if !line.trim().is_empty() {
                    persisted_history.push(line.clone());
                }
                match session.execute(&line).await {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(err) => eprintln!("[WARN] {err:#}"),
                }
            }
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }

    save_history(&history, &persisted_history)?;
    eprintln!("[INFO] repl stopped");
    Ok(())
}

fn parse_task_text(input: &str) -> Result<String> {
    let input = input.trim();
    if input.is_empty() {
        return Err(anyhow!("run: missing task"));
    }
    if input.len() >= 2 && input.starts_with('"') && input.ends_with('"') {
        Ok(input[1..input.len() - 1].to_string())
    } else {
        Ok(input.to_string())
    }
}

fn parse_bool(value: &str) -> Result<bool> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(anyhow!("invalid bool: {other}")),
    }
}

fn default_manifest() -> Option<PathBuf> {
    let path = PathBuf::from("task.toml");
    path.exists().then_some(path)
}

fn history_path() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".v9r_repl_history")
}

fn load_history(rl: &mut Editor<ReplHelper, DefaultHistory>, path: &PathBuf) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let _ = rl.add_history_entry(line);
        lines.push(line.to_string());
    }
    lines
}

fn save_history(path: &PathBuf, lines: &[String]) -> Result<()> {
    let mut deduped = Vec::new();
    for line in lines.iter().rev() {
        if !deduped.contains(line) {
            deduped.push(line.clone());
        }
        if deduped.len() >= 500 {
            break;
        }
    }
    deduped.reverse();
    std::fs::write(path, deduped.join("\n") + "\n")?;
    Ok(())
}
