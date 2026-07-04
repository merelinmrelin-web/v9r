use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use v9r_core::execution::{run_task_step, run_trusted_script, CommandSpec};
use v9r_core::manifest::Manifest;
use v9r_core::task::Task;
use v9r_core::trace::{TaskEvent, TraceLogger};
use v9r_core::vfs::{checkpoint, register_task, rollback};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let import_bundle = args.iter().any(|arg| arg == "--import-bundle");
    let export_bundle = args.iter().any(|arg| arg == "--export-bundle");
    let root = std::env::temp_dir().join("v9r-task-runtime-demo");
    let workdir = root.join("work");
    if !import_bundle {
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&workdir)?;
        fs::write(workdir.join("state.txt"), b"clean\n")?;
    }

    let manifest = Manifest {
        allow_read: vec![workdir.clone()],
        allow_write: vec![workdir.clone()],
        allow_exec: vec!["sh".to_string()],
        token_limit: 4096,
        max_steps: 16,
        timeout_ms: 30_000,
        mandatory_artifacts: vec![PathBuf::from("state.txt")],
        test_commands: Vec::new(),
    };

    let mut task = if import_bundle {
        let mut data = Vec::new();
        std::io::stdin().read_to_end(&mut data)?;
        Task::from_bundle(&data, manifest, workdir.clone())?
    } else {
        Task::new(manifest, workdir.clone())
    };
    register_task(task.id, workdir.clone());
    let trace_path = if import_bundle {
        workdir.join("trace.jsonl")
    } else {
        root.join("trace.jsonl")
    };
    let trace = TraceLogger::new(&trace_path).await?;
    if !import_bundle {
        trace
            .log_event(TaskEvent::task_started(task.manifest.clone()))
            .await?;
    }

    let checkpoint = checkpoint(task.id, &trace).await?;
    report(export_bundle, &format!("checkpoint: {}", checkpoint.0));

    // Operator-supplied scripts are the one sanctioned shell entry point
    // (`sh` must be in allow_exec). Model-issued commands can never reach
    // a shell — see the denied step below.
    let script = root.join("make-dirty.sh");
    fs::write(&script, "printf 'dirty\\n' > state.txt\n")?;
    run_trusted_script(&mut task, &script, &trace).await?;
    report(
        export_bundle,
        &format!("after allowed write: {:?}", task.status),
    );
    report(
        export_bundle,
        &format!("state: {}", fs::read_to_string(workdir.join("state.txt"))?),
    );

    // A direct command (no shell) trying to escape the workdir: the
    // structural fence rejects the parent-traversal argument before the
    // manifest is even consulted, and the task is marked as a violation.
    let bad_write = run_task_step(
        &mut task,
        CommandSpec {
            program: "cp".to_string(),
            args: vec!["state.txt".to_string(), "../escape.txt".to_string()],
            cwd: Some(workdir.clone()),
            reads: vec![PathBuf::from("state.txt")],
            writes: vec![PathBuf::from("../escape.txt")],
        },
        &trace,
    )
    .await;
    report(export_bundle, &format!("bad write result: {bad_write:?}"));
    report(
        export_bundle,
        &format!("status after violation: {:?}", task.status),
    );

    rollback(task.id, checkpoint, &trace).await?;
    report(
        export_bundle,
        &format!(
            "after rollback: {}",
            fs::read_to_string(workdir.join("state.txt"))?
        ),
    );
    report(
        export_bundle,
        &format!("escape exists: {}", root.join("escape.txt").exists()),
    );
    report(export_bundle, &format!("trace: {}", trace_path.display()));
    report(
        export_bundle,
        &format!("snapshot:\n{}", task.xml_snapshot(&trace, 8).await?),
    );

    if export_bundle {
        let bundle = task.export_bundle_with_trace(&trace)?;
        std::io::stdout().write_all(&bundle)?;
    }

    Ok(())
}

fn report(binary_stdout: bool, message: &str) {
    if binary_stdout {
        eprintln!("{message}");
    } else {
        println!("{message}");
    }
}
