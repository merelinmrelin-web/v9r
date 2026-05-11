//! llm-gateway: system agent that proxies prompts to an OpenAI-compatible API.
//!
//! Build:
//!   rustup target add wasm32-wasip1
//!   (cd examples/llm-gateway && cargo build --release --target wasm32-wasip1)
//!   copy target/wasm32-wasip1/release/llm_gateway.wasm into /agents/llm_gateway.

use serde_json::{json, Value};

const ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_MODEL: &str = "google/gemma-4-31b-it:free";
const RESPONSE_MAX: usize = 256 * 1024;
const MAX_TOOL_ROUNDS: usize = 5;

#[link(wasm_import_module = "v9r")]
#[cfg(target_arch = "wasm32")]
extern "C" {
    fn vfs_read(path_ptr: u32, path_len: u32, buf_ptr: u32, buf_max: u32) -> i32;
    fn vfs_write(path_ptr: u32, path_len: u32, data_ptr: u32, data_len: u32) -> i32;
    fn vfs_http_request(
        url_ptr: u32,
        url_len: u32,
        body_ptr: u32,
        body_len: u32,
        res_ptr: u32,
        res_max: u32,
    ) -> i32;
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn vfs_read(_path_ptr: u32, _path_len: u32, _buf_ptr: u32, _buf_max: u32) -> i32 {
    -3
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn vfs_write(_path_ptr: u32, _path_len: u32, _data_ptr: u32, _data_len: u32) -> i32 {
    -3
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn vfs_http_request(
    _url_ptr: u32,
    _url_len: u32,
    _body_ptr: u32,
    _body_len: u32,
    _res_ptr: u32,
    _res_max: u32,
) -> i32 {
    -3
}

#[no_mangle]
pub extern "C" fn alloc(size: u32) -> u32 {
    let mut v: Vec<u8> = Vec::with_capacity(size as usize);
    let p = v.as_mut_ptr() as u32;
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` must come from a previous `alloc(size)` call and not be reused.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: u32, size: u32) {
    let _ = Vec::from_raw_parts(ptr as *mut u8, 0, size as usize);
}

fn host_write(path: &str, data: &[u8]) {
    unsafe {
        let _ = vfs_write(
            path.as_ptr() as u32,
            path.len() as u32,
            data.as_ptr() as u32,
            data.len() as u32,
        );
    }
}

fn host_read(path: &str, max: usize) -> Result<String, String> {
    let mut buf = vec![0u8; max];
    let n = unsafe {
        vfs_read(
            path.as_ptr() as u32,
            path.len() as u32,
            buf.as_mut_ptr() as u32,
            max as u32,
        )
    };
    if n < 0 {
        return Err(format!("vfs_read failed for {path}: {n}"));
    }
    buf.truncate(n as usize);
    String::from_utf8(buf).map_err(|e| format!("{path} is not utf-8: {e}"))
}

fn vfs_http_request_string(url: &str, method: &str, body: &str) -> Result<String, String> {
    if method != "POST" {
        return Err(format!("unsupported method: {method}"));
    }
    let mut response = vec![0u8; RESPONSE_MAX];
    let n = unsafe {
        vfs_http_request(
            url.as_ptr() as u32,
            url.len() as u32,
            body.as_ptr() as u32,
            body.len() as u32,
            response.as_mut_ptr() as u32,
            response.len() as u32,
        )
    };
    if n < 0 {
        if n == -8 {
            return Err("OPENROUTER_API_KEY not set".to_string());
        }
        return Err(format!("vfs_http_request failed: {n}"));
    }
    response.truncate(n as usize);

    String::from_utf8(response).map_err(|e| format!("llm response is not utf-8: {e}"))
}

fn tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "vfs_read",
                "description": "Read file content from the VFS",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"]
                }
            }
        }
    ])
}

fn read_tool_path(path: &str) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(fs_err) => {
            let normalized = if path.starts_with('/') {
                path.to_string()
            } else {
                format!("/{path}")
            };
            host_read(&normalized, RESPONSE_MAX)
                .unwrap_or_else(|host_err| format!("failed to read {path}: {fs_err}; {host_err}"))
        }
    }
}

fn root_file_listing() -> String {
    match std::fs::read_dir("/") {
        Ok(entries) => {
            let mut names: Vec<String> = entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                .collect();
            names.sort();
            if names.is_empty() {
                "(root directory is empty)".to_string()
            } else {
                names.join("\n")
            }
        }
        Err(e) => format!("failed to list root directory: {e}"),
    }
}

fn append_trace(line: &str) {
    use std::io::Write;

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("trace.log")
    {
        let _ = writeln!(file, "{line}");
    }
}

fn tool_result(call: &Value) -> Value {
    let id = call
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("missing_tool_call_id");
    let function = call.get("function").unwrap_or(&Value::Null);
    let name = function.get("name").and_then(Value::as_str).unwrap_or("");
    let args = function
        .get("arguments")
        .and_then(Value::as_str)
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or(Value::Null);

    append_trace(&format!("tool_call id={id} name={name} args={args}"));

    let content = match name {
        "vfs_read" => args
            .get("path")
            .and_then(Value::as_str)
            .map(read_tool_path)
            .unwrap_or_else(|| "vfs_read error: missing path".to_string()),
        other => format!("unsupported tool: {other}"),
    };

    append_trace(&format!("tool_result id={id} content={content}"));

    json!({
        "role": "tool",
        "tool_call_id": id,
        "content": content,
    })
}

fn final_content(value: &Value) -> Option<String> {
    value
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .map(|s| s.to_string())
}

fn llm_model() -> String {
    std::env::var("OPENROUTER_MODEL")
        .ok()
        .map(|model| model.trim().to_string())
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

fn request_llm(prompt: &str) -> Result<String, String> {
    let root_files = root_file_listing();
    let system_prompt = format!(
        "You are llm_gateway inside v9r, a local WASM multi-agent orchestrator. \
You can inspect files in your sandbox by calling the vfs_read tool. \
When a user asks about files, data, manifests, logs, configuration, or local state, use vfs_read instead of saying you cannot access files. \
Paths may be absolute from / or relative to this agent root. \
Return a concise final natural-language answer after using tools.\n\nFiles currently visible in /:\n{root_files}"
    );
    let tools = tools();
    let mut messages = vec![
        json!({
            "role": "system",
            "content": system_prompt
        }),
        json!({"role": "user", "content": prompt}),
    ];

    for _ in 0..MAX_TOOL_ROUNDS {
        let payload = json!({
            "model": llm_model(),
            "messages": messages.clone(),
            "tools": tools.clone(),
            "tool_choice": "auto",
        });
        let payload = serde_json::to_string(&payload).unwrap();

        println!("[llm-gateway] Requesting LLM (timeout: 180s)...");
        println!("[llm-gateway] POST JSON: {payload}");
        let raw = vfs_http_request_string(ENDPOINT, "POST", &payload)?;
        let value: Value =
            serde_json::from_str(&raw).map_err(|e| format!("invalid llm response json: {e}"))?;
        let message = value
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .cloned()
            .ok_or_else(|| "llm response missing choices[0].message".to_string())?;

        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if tool_calls.is_empty() {
            return final_content(&value)
                .ok_or_else(|| "llm response missing choices[0].message.content".to_string());
        }

        messages.push(message);
        for call in tool_calls {
            messages.push(tool_result(&call));
        }
    }

    Err("tool loop exceeded maximum rounds".to_string())
}

fn return_bytes(s: String) -> u64 {
    let bytes = s.into_bytes();
    let len = bytes.len() as u32;
    let ptr = bytes.as_ptr() as u32;
    std::mem::forget(bytes);
    ((ptr as u64) << 32) | (len as u64)
}

/// # Safety
/// `in_ptr..in_ptr+in_len` must be a valid live byte range in our memory.
#[no_mangle]
pub unsafe extern "C" fn on_input(in_ptr: u32, in_len: u32) -> u64 {
    let input = std::slice::from_raw_parts(in_ptr as *const u8, in_len as usize);
    let fallback_prompt = std::str::from_utf8(input).unwrap_or("").trim().to_string();
    let prompt = std::fs::read_to_string("input")
        .or_else(|_| host_read("/input", RESPONSE_MAX))
        .unwrap_or(fallback_prompt)
        .trim()
        .to_string();

    let output = match request_llm(&prompt) {
        Ok(text) => text,
        Err(e) => format!("llm_gateway error: {e}"),
    };

    let _ = std::fs::write("output", output.as_bytes());
    host_write("/output", output.as_bytes());
    return_bytes(output)
}
