//! hello-agent: a minimal v9r guest module.
//!
//! Exports the v9r ABI:
//!   - `memory`           (default linear memory)
//!   - `alloc(size) -> u32`
//!   - `dealloc(ptr, size)`
//!   - `on_input(in_ptr, in_len) -> u64`   packed (out_ptr<<32 | out_len)
//!
//! On invoke it:
//!   1. reads `/tools/greeting` from its namespace via the host import
//!      `v9r::vfs_read` (Ro mount).
//!   2. composes "<greeting>, <input>!" as the response.
//!   3. prints a status line via WASI stdout — the host pipes that into
//!      the agent's `/log` automatically.
//!
//! Build:
//!   rustup target add wasm32-wasip1
//!   (cd examples/hello-agent && cargo build --release --target wasm32-wasip1)
//!   the .wasm lands at examples/hello-agent/target/wasm32-wasip1/release/hello_agent.wasm

#[link(wasm_import_module = "v9r")]
extern "C" {
    fn vfs_read(path_ptr: u32, path_len: u32, buf_ptr: u32, buf_max: u32) -> i32;
    #[allow(dead_code)]
    fn vfs_write(path_ptr: u32, path_len: u32, data_ptr: u32, data_len: u32) -> i32;
}

#[no_mangle]
pub extern "C" fn alloc(size: u32) -> u32 {
    let mut v: Vec<u8> = Vec::with_capacity(size as usize);
    let p = v.as_mut_ptr() as u32;
    // SAFETY: ownership is transferred to the host, which will pair this with `dealloc`.
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` must come from a previous `alloc(size)` call and not be reused.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: u32, size: u32) {
    let _ = Vec::from_raw_parts(ptr as *mut u8, 0, size as usize);
}

fn host_read(path: &str, max: usize) -> Option<String> {
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
        return None;
    }
    buf.truncate(n as usize);
    String::from_utf8(buf).ok()
}

/// # Safety
/// `in_ptr..in_ptr+in_len` must be a valid live byte range in our memory.
#[no_mangle]
pub unsafe extern "C" fn on_input(in_ptr: u32, in_len: u32) -> u64 {
    let input = std::slice::from_raw_parts(in_ptr as *const u8, in_len as usize);
    let name = std::str::from_utf8(input).unwrap_or("(non-utf8)").trim();

    // Default greeting if /tools/greeting isn't mounted or is empty.
    let greeting = host_read("/tools/greeting", 256)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "hello".to_string());

    println!(
        "[hello-agent] greeting={:?} input={:?}",
        greeting, name
    );
    eprintln!("[hello-agent] one line on stderr too");

    let response = format!("{greeting}, {name}!");

    let bytes = response.into_bytes();
    let len = bytes.len() as u32;
    let ptr = bytes.as_ptr() as u32;
    std::mem::forget(bytes);
    ((ptr as u64) << 32) | (len as u64)
}
