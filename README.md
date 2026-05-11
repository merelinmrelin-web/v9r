# v9r

**The Atomic, Verifiable Runtime for Autonomous Agents.**

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

Run locally with Ollama:

```sh
ollama run llama3

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
| Safe Patching | `v9r run --task "fix bug" --manifest patch.toml --workdir /tmp/patch` | rollback on test failure |
| Security Audit | `v9r run --task "audit" --manifest readonly.toml --workdir /tmp/audit` | read-only constraints and isolated writes |
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

```sh
v9r run --provider ollama --model llama3 --task "refactor main.rs" --manifest task.toml
```

OpenAI-compatible endpoints:

```sh
export V9R_API_KEY="..."
export V9R_BASE_URL="https://openrouter.ai/api/v1"

v9r run \
  --provider openai-compatible \
  --model google/gemini-flash-1.5 \
  --task "fix main.rs" \
  --manifest task.toml
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
crates/v9r-cli           pipe-friendly command line interface
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
