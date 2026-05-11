# v9r

**The Transactional Shell for AI Tasks.**

**Snapshot** the workspace before any work starts.  
**Execute** agent actions inside an explicit manifest.  
**Validate** required artifacts and test commands.  
**Rollback** on failure, timeout, or violation.  
**Export** only consistent state as a portable bundle.

The goal is not to make agents more charming. The goal is to make them accountable.

## What It Is

v9r is a bounded runtime for AI-driven work.

It treats every task as a transaction over a filesystem:

1. declare what the task may read, write, and execute
2. checkpoint the workdir
3. run the task with step and timeout limits
4. validate outputs
5. commit by exporting a bundle, or rollback to the checkpoint

No partial success. No silent state drift. No unbounded execution.

## Installation

To use `v9r` as a global command, install it via Cargo:

```bash
git clone [https://github.com/merelinmrelin-web/v9r](https://github.com/merelinmrelin-web/v9r)
cd v9r
cargo install --path crates/v9r-cli

## Quick Start

Build:

```sh
cargo build --workspace
```

Create `task.toml`:

```toml
allow_read = ["."]
allow_write = ["."]
allow_exec = ["sh", "cargo"]

token_limit = 4096
max_steps = 16
timeout_ms = 30000

mandatory_artifacts = ["result.txt"]
test_commands = ["cargo test"]
```

Start the transactional shell:

```sh
v9r repl
```

REPL session:

```text
v9r> set provider ollama
v9r> set model llama3
v9r> load task.toml
v9r> status
v9r> run "refactor main.rs"
```

No arguments also starts the REPL:

```sh
v9r
```

## Usage

### REPL

The REPL stores session state: provider, model, manifest path, workdir, base URL, and debug XML mode.

Commands:

- `set <key> <value>`: set `model`, `provider`, `manifest`, `workdir`, `base-url`, or `debug-xml`
- `load <path>`: load a manifest into the session
- `run "<task>"`: execute a transaction using the current session
- `status`: print session settings
- `exit`: close the shell

Example with local Ollama:

```text
v9r> set provider ollama
v9r> set model llama3
v9r> load task.toml
v9r> set workdir /tmp/v9r-task
v9r> run "produce result.txt"
```

### Shorthand CLI

For one-off local tasks, a bare task description is shorthand for `run` using default values:

```sh
v9r "refactor main.rs"
```

Defaults:

- provider: `ollama`
- model: `llama3`
- manifest: `task.toml`
- workdir: current directory

### Explicit CLI

```sh
v9r run \
  --provider ollama \
  --model llama3 \
  --task "refactor main.rs" \
  --manifest task.toml \
  --workdir /tmp/v9r-task
```

Run a deterministic script instead of an LLM:

```sh
v9r run \
  --task "produce result.txt" \
  --manifest task.toml \
  --workdir /tmp/v9r-task \
  -f ./task.sh
```

Inspect a bundle:

```sh
v9r inspect --bundle task.bundle
```

## Boring Guarantees

### Atomic Rollback

v9r checkpoints the workdir before execution.

If validation fails, a security violation occurs, max steps are exceeded, or the task times out, the workdir is restored and no final bundle is written.

### Verifiable Outcomes

A task is not successful because a model says it is done.

v9r validates:

- mandatory artifacts exist
- mandatory artifacts are non-empty
- configured test commands exit with `0`
- execution stayed within step and wall-clock bounds

### Portability

Successful state is exported as a `.bundle` containing:

- workdir files
- task trace
- final status
- mandatory artifact hashes

Bundles are designed for pipes and handoff between tasks.

## Constraints

Every task is governed by a manifest:

```toml
allow_read = ["src", "tests", "Cargo.toml"]
allow_write = ["src", "tests"]
allow_exec = ["cargo", "sh"]

token_limit = 4096
max_steps = 12
timeout_ms = 60000

mandatory_artifacts = ["target/report.json"]
test_commands = ["cargo test"]
```

The manifest is the contract. The runtime enforces it.

## Canonical Demos

| Use case | Command shape | Guarantee |
| --- | --- | --- |
| Safe Patching | `v9r repl`, then `run "fix bug"` | rollback on test failure |
| Security Audit | read-only manifest, then `run "audit"` | read-only constraints and isolated writes |
| Artifact Pipeline | `v9r run ... > audit.bundle` then `v9r run ... < audit.bundle > fix.bundle` | portable handoff with trace and hashes |

## Pipe Model

v9r follows Unix stream semantics.

```sh
v9r run --task "audit" --manifest audit.toml --workdir /tmp/audit > audit.bundle
v9r inspect --bundle audit.bundle
v9r run --task "fix" --manifest fix.toml --workdir /tmp/fix < audit.bundle > fix.bundle
```

If `stdin` is a pipe, `v9r run` imports a bundle. If `stdout` is a pipe, it writes a binary bundle. If `stdout` is a terminal, it prints a compact summary.

## Providers

Local-first:

```text
v9r> set provider ollama
v9r> set model llama3
v9r> run "refactor main.rs"
```

OpenAI-compatible endpoints:

```sh
export V9R_API_KEY="..."
export V9R_BASE_URL="https://openrouter.ai/api/v1"
```

```text
v9r> set provider openai-compatible
v9r> set model google/gemini-flash-1.5
v9r> run "fix main.rs"
```

Any `/v1/chat/completions` compatible service can be used, including OpenRouter, OpenAI, Groq, and Ollama.

## Observability

Runtime logs are written to stderr with stable prefixes:

```text
[INFO] task started: task_id=...
[LLM] requesting inference (provider: Ollama, model: llama3, base_url: http://localhost:11434/v1)
[STEP] file_access: write /src/main.rs (allowed)
[EXEC] command: cargo test exit_code=0
[WARN] violation: write denied: /etc/passwd
```

Task history is also stored as JSON Lines in `trace.jsonl` and preserved in bundles.

Communication with models uses a structured, verifiable protocol internally. The CLI surface remains files, manifests, logs, and bundles.

## Project Layout

```text
crates/v9r-core          manifests, tasks, validation, tracing, bundles, LLM providers
crates/v9r-cli           transactional shell and pipe-friendly CLI
crates/v9r-runtime       WASM runtime host integration
crates/v9r-vfs           virtual filesystem primitives
crates/v9r-cap           capability namespace layer
crates/v9r-orchestrator  agent loading and watch integration
examples/task-runtime    minimal task runtime demo
examples/llm-gateway     WASM LLM gateway example
```

## Status

v9r is experimental systems infrastructure.

The current focus is deterministic execution, explicit constraints, atomic rollback, verifiable output, and portable task state.
