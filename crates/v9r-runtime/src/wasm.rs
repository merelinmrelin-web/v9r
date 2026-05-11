//! Real wasmtime-backed `AgentRuntime`.
//!
//! ## Execution model
//! - One shared `Engine` (with fuel metering enabled) lives for the
//!   process. Modules are compiled once on `load`, cached by `AgentId`.
//! - Each `invoke` runs on `tokio::task::spawn_blocking` so wasmtime can
//!   stay sync. Host functions that need the async `NamespaceView` use
//!   `Handle::block_on` from inside the blocking thread — fine because
//!   spawn_blocking pulls from a dedicated pool.
//! - Per-invoke we build a fresh `Store` with: a `WasiP1Ctx` whose
//!   stdout/stderr are captured into in-memory pipes, the agent's
//!   `NamespaceView`, the runtime handle, and `StoreLimits` for memory.
//!
//! ## Guest ABI
//! The agent module must export:
//! - `memory` — its linear memory
//! - `alloc(size: u32) -> u32` — allocator the host uses to place input
//! - `dealloc(ptr: u32, size: u32)` — paired with alloc
//! - `<export>(in_ptr: u32, in_len: u32) -> u64` — the trigger entry,
//!   returning a packed `(out_ptr << 32) | out_len`
//!
//! Host imports under module name `v9r`:
//! - `vfs_read(path_ptr, path_len, buf_ptr, buf_max) -> i32`
//!     >= 0: bytes written to buf; -1 not found; -2 perm denied; -3 other; -4 buf too small
//! - `vfs_write(path_ptr, path_len, data_ptr, data_len) -> i32`
//!     0 ok; -1 not found; -2 perm denied; -3 other
//! - `vfs_http_request(url_ptr, url_len, body_ptr, body_len, res_ptr, res_max) -> i32`
//!     >= 0: response bytes written; -2 perm denied; -3 invalid input; -5 network; -6 too large; -7 empty; -8 missing auth/config
//!
//! ## Logs
//! WASI stdout + stderr are backed by the agent's mounted `LogBuffer`.
//! Writes append to the synthetic `/log` node as they happen, so partial
//! output is visible even if the guest later exhausts fuel or traps.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::runtime::Handle;
use tokio::sync::RwLock;
use wasmtime::{
    Caller, Config, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc,
};
use wasmtime_wasi::preview1::{add_to_linker_sync, WasiP1Ctx};
use wasmtime_wasi::{
    DirPerms, FilePerms, HostOutputStream, SocketAddrUse, StdoutStream, StreamResult, Subscribe,
    WasiCtxBuilder,
};

use v9r_cap::NamespaceView;
use v9r_core::{VfsError, VfsPath, VfsResult};
use v9r_vfs::LogBuffer;

use crate::{AgentId, AgentRuntime};

pub const DEFAULT_FUEL: u64 = 10_000_000;
pub const DEFAULT_MEMORY_LIMIT: usize = 16 * 1024 * 1024;
const HTTP_TIMEOUT: Duration = Duration::from_secs(180);
const OPENROUTER_CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const OPENROUTER_MODEL_ENV: &str = "OPENROUTER_MODEL";
const OPENROUTER_DEFAULT_MODEL: &str = "google/gemma-4-31b-it:free";
const OPENROUTER_HTTP_REFERER: &str = "https://github.com/nikita/v9r";
const OPENROUTER_TITLE: &str = "v9r Orchestrator";
static HTTP_CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();

struct PreparedHttpRequest {
    url: String,
    body: Vec<u8>,
    openrouter_api_key: Option<String>,
}

pub struct WasmtimeRuntime {
    engine: Engine,
    modules: RwLock<HashMap<AgentId, Module>>,
    fuel_limit: u64,
    memory_limit: usize,
}

/// Per-store state. Lives for the duration of one invoke.
struct AgentState {
    wasi: WasiP1Ctx,
    view: NamespaceView,
    handle: Handle,
    limits: StoreLimits,
    log: Option<Arc<LogBuffer>>,
}

#[derive(Clone)]
struct LogOutput {
    log: Arc<LogBuffer>,
}

struct LogOutputStream {
    log: Arc<LogBuffer>,
}

impl WasmtimeRuntime {
    pub fn new() -> anyhow::Result<Self> {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config)?;
        Ok(Self {
            engine,
            modules: RwLock::new(HashMap::new()),
            fuel_limit: DEFAULT_FUEL,
            memory_limit: DEFAULT_MEMORY_LIMIT,
        })
    }

    pub fn with_fuel(mut self, fuel: u64) -> Self {
        self.fuel_limit = fuel;
        self
    }

    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = bytes;
        self
    }
}

#[async_trait]
impl AgentRuntime for WasmtimeRuntime {
    async fn load(&self, id: &AgentId, wasm: Bytes) -> VfsResult<()> {
        let module = Module::new(&self.engine, &wasm[..])
            .map_err(|e| VfsError::Synthetic(format!("wasm compile failed: {e}")))?;
        self.modules.write().await.insert(id.clone(), module);
        tracing::info!(agent = %id.as_str(), bytes = wasm.len(), "wasm module loaded");
        Ok(())
    }

    async fn unload(&self, id: &AgentId) {
        self.modules.write().await.remove(id);
    }

    async fn invoke(
        &self,
        id: &AgentId,
        export: &str,
        input: Bytes,
        view: NamespaceView,
        log: Option<Arc<LogBuffer>>,
    ) -> VfsResult<Bytes> {
        let module = self.modules.read().await.get(id).cloned().ok_or_else(|| {
            VfsError::Synthetic(format!("module not loaded for agent: {}", id.as_str()))
        })?;

        let engine = self.engine.clone();
        let export = export.to_string();
        let handle = Handle::current();
        let fuel = self.fuel_limit;
        let mem_limit = self.memory_limit;
        let view_for_wasm = view.clone();

        let output = tokio::task::spawn_blocking(move || {
            run_wasm(
                engine,
                module,
                export,
                view_for_wasm,
                handle,
                input,
                log,
                fuel,
                mem_limit,
            )
        })
        .await
        .map_err(|e| VfsError::Synthetic(format!("wasm task panicked: {e}")))?
        .map_err(|e| VfsError::Synthetic(format!("wasm: {e:#}")))?;

        Ok(output)
    }
}

impl StdoutStream for LogOutput {
    fn stream(&self) -> Box<dyn HostOutputStream> {
        Box::new(LogOutputStream {
            log: Arc::clone(&self.log),
        })
    }

    fn isatty(&self) -> bool {
        false
    }
}

#[async_trait]
impl Subscribe for LogOutputStream {
    async fn ready(&mut self) {}
}

impl HostOutputStream for LogOutputStream {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        self.log.append(&bytes);
        Ok(())
    }

    fn flush(&mut self) -> StreamResult<()> {
        Ok(())
    }

    fn check_write(&mut self) -> StreamResult<usize> {
        Ok(1024 * 1024)
    }
}

fn host_http_log(log: Option<&Arc<LogBuffer>>, message: &str) {
    eprintln!("{message}");
    if let Some(log) = log {
        let mut line = Vec::with_capacity(message.len() + 1);
        line.extend_from_slice(message.as_bytes());
        line.push(b'\n');
        log.append(&line);
    }
}

fn http_client() -> &'static reqwest::blocking::Client {
    HTTP_CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .expect("failed to build HTTP client")
    })
}

fn map_localhost_to_loopback(url: &str) -> String {
    for prefix in ["http://localhost", "https://localhost"] {
        if let Some(rest) = url.strip_prefix(prefix) {
            if rest.is_empty() || rest.starts_with(':') || rest.starts_with('/') {
                let scheme = prefix
                    .split_once("://")
                    .map(|(scheme, _)| scheme)
                    .unwrap_or("http");
                return format!("{scheme}://127.0.0.1{rest}");
            }
        }
    }
    url.to_string()
}

fn openrouter_model() -> String {
    std::env::var(OPENROUTER_MODEL_ENV)
        .ok()
        .map(|model| model.trim().to_string())
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| OPENROUTER_DEFAULT_MODEL.to_string())
}

fn prepare_http_request(url: String, body: Vec<u8>) -> Result<PreparedHttpRequest, String> {
    if url != OPENROUTER_CHAT_COMPLETIONS_URL {
        return Ok(PreparedHttpRequest {
            url,
            body,
            openrouter_api_key: None,
        });
    }

    let openrouter_api_key = std::env::var(OPENROUTER_API_KEY_ENV)
        .map(|key| key.trim().to_string())
        .ok()
        .filter(|key| !key.is_empty())
        .ok_or_else(|| "OPENROUTER_API_KEY not set".to_string())?;

    let mut payload: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|err| format!("invalid OpenRouter request JSON: {err}"))?;
    let payload = payload
        .as_object_mut()
        .ok_or_else(|| "OpenRouter request JSON must be an object".to_string())?;
    payload.insert(
        "model".to_string(),
        serde_json::Value::String(openrouter_model()),
    );
    let body = serde_json::to_vec(&payload)
        .map_err(|err| format!("failed to encode OpenRouter request JSON: {err}"))?;

    Ok(PreparedHttpRequest {
        url,
        body,
        openrouter_api_key: Some(openrouter_api_key),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_wasm(
    engine: Engine,
    module: Module,
    export: String,
    view: NamespaceView,
    handle: Handle,
    input: Bytes,
    log: Option<Arc<LogBuffer>>,
    fuel: u64,
    memory_limit: usize,
) -> anyhow::Result<Bytes> {
    let mut wasi_builder = WasiCtxBuilder::new();
    let cwd = std::env::current_dir()?;
    wasi_builder.preopened_dir(&cwd, "/", DirPerms::all(), FilePerms::all())?;

    let allow_network = view.can_access_network();
    wasi_builder
        .allow_ip_name_lookup(allow_network)
        .allow_tcp(allow_network)
        .allow_udp(false)
        .socket_addr_check(move |addr, use_| {
            Box::pin(async move {
                allow_network
                    && matches!(use_, SocketAddrUse::TcpConnect)
                    && addr.ip() == IpAddr::V4(Ipv4Addr::LOCALHOST)
                    && addr.port() == 1234
            })
        });
    if let Some(log) = &log {
        let stdout = LogOutput {
            log: Arc::clone(&log),
        };
        let stderr = LogOutput {
            log: Arc::clone(&log),
        };
        wasi_builder.stdout(stdout).stderr(stderr);
    }
    let wasi = wasi_builder.build_p1();

    let limits = StoreLimitsBuilder::new().memory_size(memory_limit).build();

    let state = AgentState {
        wasi,
        view,
        handle,
        limits,
        log,
    };

    let mut store = Store::new(&engine, state);
    store.limiter(|s: &mut AgentState| &mut s.limits);
    store.set_fuel(fuel)?;

    let mut linker: Linker<AgentState> = Linker::new(&engine);
    add_to_linker_sync(&mut linker, |s: &mut AgentState| &mut s.wasi)?;
    add_v9r_imports(&mut linker)?;

    let instance = linker.instantiate(&mut store, &module)?;

    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| anyhow::anyhow!("guest module has no `memory` export"))?;
    let alloc_fn: TypedFunc<u32, u32> = instance.get_typed_func(&mut store, "alloc")?;
    let dealloc_fn: TypedFunc<(u32, u32), ()> = instance.get_typed_func(&mut store, "dealloc")?;
    let entry: TypedFunc<(u32, u32), u64> = instance.get_typed_func(&mut store, &export)?;

    let in_len = input.len() as u32;
    let in_ptr = alloc_fn.call(&mut store, in_len)?;
    memory.write(&mut store, in_ptr as usize, &input)?;

    let result = entry.call(&mut store, (in_ptr, in_len))?;
    let out_ptr = (result >> 32) as u32;
    let out_len = result as u32;

    let mut out = vec![0u8; out_len as usize];
    if out_len > 0 {
        memory.read(&store, out_ptr as usize, &mut out)?;
    }

    // Free guest-side allocations. If the guest's `dealloc` is a no-op
    // (e.g. for tiny demo agents), that's fine — the whole linear memory
    // dies with the Store at the end of this function anyway.
    dealloc_fn.call(&mut store, (in_ptr, in_len))?;
    if out_len > 0 {
        dealloc_fn.call(&mut store, (out_ptr, out_len))?;
    }

    Ok(Bytes::from(out))
}

fn add_v9r_imports(linker: &mut Linker<AgentState>) -> anyhow::Result<()> {
    linker.func_wrap(
        "v9r",
        "vfs_read",
        |mut caller: Caller<'_, AgentState>,
         path_ptr: u32,
         path_len: u32,
         buf_ptr: u32,
         buf_max: u32|
         -> i32 {
            let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return -3,
            };
            let mut path_buf = vec![0u8; path_len as usize];
            if memory
                .read(&caller, path_ptr as usize, &mut path_buf)
                .is_err()
            {
                return -3;
            }
            let path_str = match std::str::from_utf8(&path_buf) {
                Ok(s) => s,
                Err(_) => return -3,
            };
            let path = match VfsPath::parse(path_str) {
                Ok(p) => p,
                Err(_) => return -3,
            };

            let (view, handle) = {
                let d = caller.data();
                (d.view.clone(), d.handle.clone())
            };
            let bytes = match handle.block_on(view.read(&path)) {
                Ok(b) => b,
                Err(VfsError::NotFound) => return -1,
                Err(VfsError::PermissionDenied) => return -2,
                Err(_) => return -3,
            };
            if bytes.len() > buf_max as usize {
                return -4;
            }
            if memory.write(&mut caller, buf_ptr as usize, &bytes).is_err() {
                return -3;
            }
            bytes.len() as i32
        },
    )?;

    linker.func_wrap(
        "v9r",
        "vfs_write",
        |mut caller: Caller<'_, AgentState>,
         path_ptr: u32,
         path_len: u32,
         data_ptr: u32,
         data_len: u32|
         -> i32 {
            let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return -3,
            };
            let mut path_buf = vec![0u8; path_len as usize];
            if memory
                .read(&caller, path_ptr as usize, &mut path_buf)
                .is_err()
            {
                return -3;
            }
            let path_str = match std::str::from_utf8(&path_buf) {
                Ok(s) => s,
                Err(_) => return -3,
            };
            let path = match VfsPath::parse(path_str) {
                Ok(p) => p,
                Err(_) => return -3,
            };
            let mut data_buf = vec![0u8; data_len as usize];
            if memory
                .read(&caller, data_ptr as usize, &mut data_buf)
                .is_err()
            {
                return -3;
            }
            let _ = &mut caller; // silence unused-mut warning; we wrote earlier in vfs_read

            let (view, handle) = {
                let d = caller.data();
                (d.view.clone(), d.handle.clone())
            };
            match handle.block_on(view.write(&path, Bytes::from(data_buf))) {
                Ok(()) => 0,
                Err(VfsError::NotFound) => -1,
                Err(VfsError::PermissionDenied) => -2,
                Err(_) => -3,
            }
        },
    )?;

    linker.func_wrap(
        "v9r",
        "vfs_http_request",
        |mut caller: Caller<'_, AgentState>,
         url_ptr: u32,
         url_len: u32,
         body_ptr: u32,
         body_len: u32,
         res_ptr: u32,
         res_max: u32|
         -> i32 {
            if !caller.data().view.can_access_network() {
                return -2;
            }

            let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return -3,
            };
            let mut url_buf = vec![0u8; url_len as usize];
            if memory
                .read(&caller, url_ptr as usize, &mut url_buf)
                .is_err()
            {
                return -3;
            }
            let url = match String::from_utf8(url_buf) {
                Ok(s) if s.starts_with("http://") || s.starts_with("https://") => s,
                _ => return -3,
            };
            let url = map_localhost_to_loopback(&url);

            let mut body_buf = vec![0u8; body_len as usize];
            if memory
                .read(&caller, body_ptr as usize, &mut body_buf)
                .is_err()
            {
                return -3;
            }

            let log = {
                let data = caller.data();
                data.log.clone()
            };
            let request = match prepare_http_request(url, body_buf) {
                Ok(request) => request,
                Err(err) => {
                    host_http_log(log.as_ref(), &format!("[host] HTTP Error: {err}"));
                    return if err == "OPENROUTER_API_KEY not set" {
                        -8
                    } else {
                        -3
                    };
                }
            };

            host_http_log(
                log.as_ref(),
                &format!("[host] HTTP: Sending request to {}...", request.url),
            );
            let mut request_builder = http_client()
                .post(&request.url)
                .header(reqwest::header::CONTENT_TYPE, "application/json");
            if let Some(api_key) = request.openrouter_api_key {
                request_builder = request_builder
                    .header(reqwest::header::AUTHORIZATION, format!("Bearer {api_key}"))
                    .header("HTTP-Referer", OPENROUTER_HTTP_REFERER)
                    .header("X-Title", OPENROUTER_TITLE);
            }
            let response = match request_builder.body(request.body).send() {
                Ok(response) => response,
                Err(err) => {
                    host_http_log(log.as_ref(), &format!("[host] HTTP Error: {err:?}"));
                    return -5;
                }
            };
            let status = response.status();
            host_http_log(
                log.as_ref(),
                &format!("[host] HTTP: Received response with status {status}."),
            );
            if !status.is_success() {
                host_http_log(
                    log.as_ref(),
                    &format!("[host] HTTP Error: non-success status {status}"),
                );
                return -5;
            }
            let response = match response.bytes() {
                Ok(bytes) => bytes,
                Err(err) => {
                    host_http_log(log.as_ref(), &format!("[host] HTTP Error: {err:?}"));
                    return -5;
                }
            };

            if response.is_empty() {
                return -7;
            }

            if response.len() > res_max as usize {
                return -6;
            }
            if memory
                .write(&mut caller, res_ptr as usize, response.as_ref())
                .is_err()
            {
                return -3;
            }
            response.len() as i32
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use v9r_cap::{HostCapability, MountMode, NamespaceView};
    use v9r_core::{Capability, VfsPath};
    use v9r_vfs::{LogBuffer, SyntheticFile, Vfs};

    /// Hand-written WAT exporting the v9r ABI: `alloc`, `dealloc`,
    /// `on_input(in_ptr, in_len) -> u64`. The `on_input` here just echoes
    /// the input bytes back unchanged. No host calls — this only exercises
    /// the orchestration plumbing (compile → instantiate → call → memory).
    const ECHO_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (global $bump (mut i32) (i32.const 1024))

  (func $alloc (export "alloc") (param $size i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $bump))
    (global.set $bump (i32.add (global.get $bump) (local.get $size)))
    (local.get $p))

  (func (export "dealloc") (param $ptr i32) (param $size i32))

  ;; on_input: copy `len` bytes from `in_ptr` to a fresh buffer, return packed (ptr<<32 | len)
  (func (export "on_input") (param $in_ptr i32) (param $in_len i32) (result i64)
    (local $out_ptr i32)
    (local.set $out_ptr (call $alloc (local.get $in_len)))
    (memory.copy (local.get $out_ptr) (local.get $in_ptr) (local.get $in_len))
    (i64.or
      (i64.shl (i64.extend_i32_u (local.get $out_ptr)) (i64.const 32))
      (i64.extend_i32_u (local.get $in_len)))))
"#;

    async fn make_view() -> (Arc<Vfs>, NamespaceView, Arc<LogBuffer>) {
        let vfs = Vfs::new();
        vfs.mkdir_p(&VfsPath::parse("/agents/echo").unwrap())
            .await
            .unwrap();
        let log = Arc::new(LogBuffer::default());
        let syn: Arc<dyn SyntheticFile> = log.clone();
        vfs.create_synthetic(&VfsPath::parse("/agents/echo/log").unwrap(), syn)
            .await
            .unwrap();
        let view = NamespaceView::builder(Arc::clone(&vfs), Capability::root())
            .mount(
                VfsPath::root(),
                VfsPath::parse("/agents/echo").unwrap(),
                MountMode::Rw,
            )
            .build();
        (vfs, view, log)
    }

    fn system_view(vfs: Arc<Vfs>) -> NamespaceView {
        NamespaceView::builder(vfs, Capability::root())
            .mount(
                VfsPath::root(),
                VfsPath::parse("/agents/echo").unwrap(),
                MountMode::Rw,
            )
            .host_capability(HostCapability::CanAccessNetwork)
            .build()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn echo_wat_round_trip() {
        let wasm = wat::parse_str(ECHO_WAT).unwrap();
        let rt = WasmtimeRuntime::new().unwrap();
        let id = AgentId::new("echo");
        rt.load(&id, Bytes::from(wasm)).await.unwrap();

        let (_vfs, view, log) = make_view().await;
        let out = rt
            .invoke(
                &id,
                "on_input",
                Bytes::from_static(b"plan9 lives"),
                view,
                Some(log),
            )
            .await
            .unwrap();
        assert_eq!(&out[..], b"plan9 lives");
    }

    /// Infinite loop — should be killed by fuel exhaustion, not hang the test.
    const SPIN_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (func (export "alloc") (param i32) (result i32) (i32.const 0))
  (func (export "dealloc") (param i32 i32))
  (func (export "on_input") (param i32 i32) (result i64)
    (loop $l (br $l))
    (i64.const 0)))
"#;

    #[tokio::test(flavor = "multi_thread")]
    async fn fuel_metering_kills_infinite_loop() {
        let wasm = wat::parse_str(SPIN_WAT).unwrap();
        let rt = WasmtimeRuntime::new().unwrap().with_fuel(100_000);
        let id = AgentId::new("spin");
        rt.load(&id, Bytes::from(wasm)).await.unwrap();

        let (_vfs, view, log) = make_view().await;
        let err = rt
            .invoke(&id, "on_input", Bytes::new(), view, Some(log))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("fuel") || msg.contains("trap"),
            "expected fuel/trap error, got: {msg}"
        );
    }

    const WRITE_THEN_SPIN_WAT: &str = r#"
(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 16) "before spin\n")

  (func (export "alloc") (param i32) (result i32) (i32.const 64))
  (func (export "dealloc") (param i32 i32))

  (func (export "on_input") (param i32 i32) (result i64)
    (i32.store (i32.const 0) (i32.const 16))
    (i32.store (i32.const 4) (i32.const 12))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))
    (loop $l (br $l))
    (i64.const 0)))
"#;

    #[tokio::test(flavor = "multi_thread")]
    async fn stdout_streams_before_fuel_trap() {
        let wasm = wat::parse_str(WRITE_THEN_SPIN_WAT).unwrap();
        let rt = WasmtimeRuntime::new().unwrap().with_fuel(1_000_000);
        let id = AgentId::new("echo");
        rt.load(&id, Bytes::from(wasm)).await.unwrap();

        let (vfs, view, log) = make_view().await;
        let err = rt
            .invoke(&id, "on_input", Bytes::new(), view, Some(log))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("fuel") || msg.contains("trap"));

        let bytes = vfs
            .read(
                &Capability::root(),
                &VfsPath::parse("/agents/echo/log").unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(&bytes[..], b"before spin\n");
    }

    const WASI_READ_CARGO_TOML_WAT: &str = r#"
(module
  (import "wasi_snapshot_preview1" "path_open"
    (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_read"
    (func $fd_read (param i32 i32 i32 i32) (result i32)))

  (memory (export "memory") 1)
  (data (i32.const 64) "Cargo.toml")

  (func (export "alloc") (param i32) (result i32) (i32.const 4096))
  (func (export "dealloc") (param i32 i32))

  (func (export "on_input") (param i32 i32) (result i64)
    (local $errno i32)
    (local $fd i32)

    ;; Open Cargo.toml through the root preopen. Without the / preopen,
    ;; path_open returns an errno instead of a readable file descriptor.
    (local.set $errno
      (call $path_open
        (i32.const 3)   ;; first preopened fd after stdin/stdout/stderr
        (i32.const 0)
        (i32.const 64) (i32.const 10)
        (i32.const 0)
        (i64.const 2) (i64.const 0) ;; FD_READ, no inheriting rights
        (i32.const 0)
        (i32.const 16)))
    (if (local.get $errno)
      (then
        (i32.store (i32.const 128) (local.get $errno))
        (return (i64.or (i64.shl (i64.const 128) (i64.const 32)) (i64.const 4)))))

    (local.set $fd (i32.load (i32.const 16)))
    (i32.store (i32.const 0) (i32.const 128))
    (i32.store (i32.const 4) (i32.const 4))

    (local.set $errno
      (call $fd_read (local.get $fd) (i32.const 0) (i32.const 1) (i32.const 20)))
    (if (local.get $errno)
      (then
        (i32.store (i32.const 128) (local.get $errno))
        (return (i64.or (i64.shl (i64.const 128) (i64.const 32)) (i64.const 4)))))

    (i64.or
      (i64.shl (i64.const 128) (i64.const 32))
      (i64.extend_i32_u (i32.load (i32.const 20)))))
)
"#;

    #[tokio::test(flavor = "multi_thread")]
    async fn wasi_preopens_cwd_as_guest_root() {
        let wasm = wat::parse_str(WASI_READ_CARGO_TOML_WAT).unwrap();
        let rt = WasmtimeRuntime::new().unwrap();
        let id = AgentId::new("echo");
        rt.load(&id, Bytes::from(wasm)).await.unwrap();

        let (_vfs, view, log) = make_view().await;
        let out = rt
            .invoke(&id, "on_input", Bytes::new(), view, Some(log))
            .await
            .unwrap();

        assert_eq!(&out[..1], b"[");
    }

    fn wat_string(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }

    fn start_mock_server(response_body: &'static str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            let body_start = loop {
                let n = stream.read(&mut buf).unwrap();
                assert!(n > 0, "client closed before headers");
                request.extend_from_slice(&buf[..n]);
                if let Some(pos) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request[..body_start]);
            let content_len = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while request.len() < body_start + content_len {
                let n = stream.read(&mut buf).unwrap();
                assert!(n > 0, "client closed before body");
                request.extend_from_slice(&buf[..n]);
            }
            let body = &request[body_start..body_start + content_len];
            tx.send(String::from_utf8(body.to_vec()).unwrap()).unwrap();

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        (format!("http://{addr}/v1/chat/completions"), rx)
    }

    fn llm_gateway_mock_wat(url: &str, response_body: &str, answer: &str) -> String {
        let answer_offset = response_body.find(answer).unwrap();
        format!(
            r#"
(module
  (import "v9r" "vfs_http_request" (func $http (param i32 i32 i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 16) "{url}")

  (func (export "alloc") (param i32) (result i32) (i32.const 4096))
  (func (export "dealloc") (param i32 i32))

  (func (export "on_input") (param $in_ptr i32) (param $in_len i32) (result i64)
    (local $n i32)
    (local.set $n
      (call $http
        (i32.const 16) (i32.const {url_len})
        (local.get $in_ptr) (local.get $in_len)
        (i32.const 2048) (i32.const 1024)))
    (if (i32.lt_s (local.get $n) (i32.const 0))
      (then
        (i32.store (i32.const 128) (local.get $n))
        (return (i64.or (i64.shl (i64.const 128) (i64.const 32)) (i64.const 4)))))
    (i64.or
      (i64.shl
        (i64.extend_i32_u (i32.add (i32.const 2048) (i32.const {answer_offset})))
        (i64.const 32))
      (i64.const {answer_len})))
)
"#,
            url = wat_string(url),
            url_len = url.len(),
            answer_offset = answer_offset,
            answer_len = answer.len(),
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn http_request_requires_network_capability() {
        let response = r#"{"choices":[{"message":{"content":"mock answer"}}]}"#;
        let wat = llm_gateway_mock_wat(
            "http://127.0.0.1:9/v1/chat/completions",
            response,
            "mock answer",
        );
        let wasm = wat::parse_str(&wat).unwrap();
        let rt = WasmtimeRuntime::new().unwrap();
        let id = AgentId::new("echo");
        rt.load(&id, Bytes::from(wasm)).await.unwrap();

        let (_vfs, view, log) = make_view().await;
        let out = rt
            .invoke(
                &id,
                "on_input",
                Bytes::from_static(br#"{"model":"runtime-test-model","messages":[]}"#),
                view,
                Some(log),
            )
            .await
            .unwrap();

        assert_eq!(i32::from_le_bytes(out[..4].try_into().unwrap()), -2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_llm_gateway_mock() {
        let response = r#"{"choices":[{"message":{"content":"mock answer"}}]}"#;
        let (url, requests) = start_mock_server(response);
        let wat = llm_gateway_mock_wat(&url, response, "mock answer");
        let wasm = wat::parse_str(&wat).unwrap();
        let rt = WasmtimeRuntime::new().unwrap();
        let id = AgentId::new("echo");
        rt.load(&id, Bytes::from(wasm)).await.unwrap();
        let request = serde_json::json!({
            "model": "runtime-test-model",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "vfs_read",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": "auto"
        });
        let request = serde_json::to_vec(&request).unwrap();

        let (vfs, _view, log) = make_view().await;
        let out = rt
            .invoke(
                &id,
                "on_input",
                Bytes::from(request),
                system_view(vfs),
                Some(log),
            )
            .await
            .unwrap();

        assert_eq!(&out[..], b"mock answer");

        let body = requests.recv_timeout(Duration::from_secs(2)).unwrap();
        let actual: serde_json::Value = serde_json::from_str(&body).unwrap();
        let expected = serde_json::json!({
            "model": "runtime-test-model",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "vfs_read",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": "auto"
        });
        assert_eq!(actual, expected);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn missing_module_errors() {
        let rt = WasmtimeRuntime::new().unwrap();
        let (_vfs, view, _log) = make_view().await;
        let err = rt
            .invoke(&AgentId::new("ghost"), "on_input", Bytes::new(), view, None)
            .await
            .unwrap_err();
        assert!(matches!(err, VfsError::Synthetic(_)));
    }
}
