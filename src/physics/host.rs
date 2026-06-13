//! Host-side state and WASM import implementations.
//!
//! The physics module is compiled with Emscripten targeting **wasm32**.
//! All imports live under the `"a"` module namespace with single-character
//! names (Emscripten symbol minification).
//!
//! | Export | C / C++ symbol                  | Description                    |
//! |--------|---------------------------------|--------------------------------|
//! | `"i"`  | `__assert_fail`                 | Assertion failure handler      |
//! | `"a"`  | `__cxa_throw`                   | C++ exception throw            |
//! | `"e"`  | `abort`                         | Unconditional abort            |
//! | `"f"`  | `emscripten_resize_heap`        | Heap growth shim               |
//! | `"g"`  | `fd_write` (WASI stub)          | I/O no-op                      |
//! | `"c"`  | `emscripten_notify_memory_growth` | Memory-growth notification   |
//! | `"b"`  | `exit`                          | Graceful exit no-op            |
//! | `"h"`  | `emscripten_get_now`            | Monotonic clock stub → 0       |
//! | `"d"`  | `emscripten_set_timeout`        | Timer stub → 0                 |

use wasmtime::{Caller, Extern, Linker, bail};

/// Per-store state threaded through every host callback via `Store<HostState>`.
///
/// When the WASM module calls `abort()` or throws an unhandled C++ exception,
/// we cannot stop execution mid-trap cleanly, so we record the exit and check
/// after every WASM call returns.
#[derive(Default)]
pub(super) struct HostState {
    /// Whether the module has signalled that it wants to exit.
    pub exited: bool,
    /// The exit code reported by the module (134 for `SIGABRT`/`abort()`).
    pub exit_code: i32,
}

impl HostState {
    /// Returns `Some(exit_code)` if the module has exited, `None` otherwise.
    #[inline]
    pub fn check_exit(&self) -> Option<i32> {
        self.exited.then_some(self.exit_code)
    }
}

/// Mark the store as exited with `code`.  Called from host callbacks.
pub(super) fn mark_exited(caller: &mut Caller<'_, HostState>, code: i32) {
    let st = caller.data_mut();
    st.exited = true;
    st.exit_code = code;
}

/// SIGABRT exit code, matching the value a native process would report.
const EXIT_CODE_ABORT: i32 = 134;

/// Register every import the physics WASM module requires into `linker`.
///
/// Must be called before [`wasmtime::Linker::instantiate`].
pub(super) fn register(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    // "i" — __assert_fail(msg, file, line, func)
    linker.func_wrap(
        "a",
        "i",
        |mut caller: Caller<'_, HostState>,
         msg: i32,
         file: i32,
         line: i32,
         func: i32|
         -> Result<(), wasmtime::Error> {
            let msg_str = read_cstr(&mut caller, msg as u32);
            let file_str = read_cstr(&mut caller, file as u32);
            let func_str = read_cstr(&mut caller, func as u32);
            eprintln!("assertion failed: {msg_str}  at {file_str}:{line} ({func_str})");
            mark_exited(&mut caller, EXIT_CODE_ABORT);
            bail!("assertion failed")
        },
    )?;

    // "a" — __cxa_throw(exc_ptr, typeinfo_ptr, destructor)
    linker.func_wrap(
        "a",
        "a",
        |mut caller: Caller<'_, HostState>,
         exc: i32,
         ty: i32,
         _dtor: i32|
         -> Result<(), wasmtime::Error> {
            eprintln!("C++ exception thrown (exc_ptr={exc:#x}, typeinfo={ty:#x})");

            let memory = match caller.get_export("j") {
                Some(Extern::Memory(m)) => m,
                _ => {
                    eprintln!("wasm memory export missing while handling C++ exception");
                    mark_exited(&mut caller, 1);
                    bail!("C++ exception: <no memory>");
                }
            };

            let message = decode_cpp_exception(memory.data(&caller), exc as usize);
            eprintln!("C++ exception message: {message}");
            mark_exited(&mut caller, 1);
            bail!("C++ exception: {message}");
        },
    )?;

    // "e" — abort()
    linker.func_wrap(
        "a",
        "e",
        |mut caller: Caller<'_, HostState>| -> Result<(), wasmtime::Error> {
            eprintln!("abort()");
            mark_exited(&mut caller, EXIT_CODE_ABORT);
            bail!("abort()")
        },
    )?;

    // "f" — emscripten_resize_heap(desired_bytes) → 1 on success, 0 on failure
    linker.func_wrap(
        "a",
        "f",
        |mut caller: Caller<'_, HostState>, desired: i32| -> i32 {
            let mem = match caller.get_export("j") {
                Some(Extern::Memory(m)) => m,
                _ => return 0,
            };

            let desired = desired as usize;
            let current = mem.data_size(&caller);

            if desired <= current {
                return 1; // already large enough
            }

            let pages = (desired - current).div_ceil(65536) as u64;
            match mem.grow(&mut caller, pages) {
                Ok(_) => 1,
                Err(_) => 0,
            }
        },
    )?;

    // No-op stubs — the physics module calls these but we don't need their
    // side effects in a headless simulation context.
    linker.func_wrap(
        "a",
        "g",
        |_: Caller<'_, HostState>, _fd: i32, _iov: i32, _iovcnt: i32, _pnum: i32| -> i32 { 1 },
    )?;
    linker.func_wrap("a", "c", |_: Caller<'_, HostState>| {})?;
    linker.func_wrap("a", "b", |_: Caller<'_, HostState>, _code: i32| {})?;
    linker.func_wrap("a", "h", |_: Caller<'_, HostState>| -> f64 { 0.0 })?;
    linker.func_wrap(
        "a",
        "d",
        |_: Caller<'_, HostState>, _id: i32, _ms: f64| -> i32 { 0 },
    )?;

    Ok(())
}

/// Reads a null-terminated C string from WASM linear memory at `ptr`.
///
/// Returns a placeholder string if the export is missing or `ptr` is out of
/// bounds — never panics, since this is called from error-handling paths.
fn read_cstr(caller: &mut Caller<'_, HostState>, ptr: u32) -> String {
    let mem = match caller.get_export("j") {
        Some(Extern::Memory(m)) => m,
        _ => return "<no memory>".into(),
    };

    let data = mem.data(caller);
    let start = ptr as usize;

    if start >= data.len() {
        return "<invalid ptr>".into();
    }

    let end = data[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|i| start + i)
        .unwrap_or(data.len());

    String::from_utf8_lossy(&data[start..end]).into_owned()
}

/// Extracts the message string from a C++ exception object in wasm32/Emscripten
/// linear memory.
///
/// # Memory layout (wasm32 + Emscripten libc++, no `_LIBCPP_ABI_ALTERNATE_STRING_LAYOUT`)
///
/// The pointer `exc` points to the thrown object.  For `std::runtime_error`
/// and similar types the layout is:
///
/// ```text
/// exc + 0 .. +4  : vtable pointer  (u32 LE)
/// exc + 4 .. +16 : std::string     (12 bytes on wasm32)
/// ```
///
/// ## `std::string` on wasm32 (`sizeof` = 12, `__min_cap` = 11)
///
/// **Short (SSO) mode** — string fits inline (length ≤ 10):
/// ```text
/// byte  0     : __size_ (u8)  — stores `len << 1`; low bit **0** = short
/// bytes 1..11 : inline character data (null-terminated)
/// ```
///
/// **Long (heap) mode** — string is heap-allocated:
/// ```text
/// bytes 0..4  : __cap_  (u32 LE) — capacity; low bit **1** = long
/// bytes 4..8  : __size_ (u32 LE) — string length
/// bytes 8..12 : __data_ (u32 LE) — pointer to heap buffer
/// ```
///
/// Detection: `data[str_base] & 0x01 == 0` → short; `== 1` → long.
///
/// # Correctness note
/// The original game code checked `data[string_ptr + 11]` (the 64-bit libc++
/// flag byte position) and had the SSO/long sense inverted.  This
/// implementation uses the correct wasm32 offsets and flag semantics.
pub(super) fn decode_cpp_exception(data: &[u8], exc: usize) -> String {
    // std::string starts immediately after the 4-byte vtable pointer.
    let str_base = exc + 4;

    if str_base + 12 > data.len() {
        return "<exception object out of bounds>".into();
    }

    let flag = data[str_base];

    if flag & 0x01 == 0 {
        // Short (SSO) mode
        // Length is stored in the upper 7 bits of the flag byte.
        let len = (flag >> 1) as usize;
        let char_start = str_base + 1;

        if char_start + len > data.len() {
            return "<sso string out of bounds>".into();
        }

        String::from_utf8_lossy(&data[char_start..char_start + len]).into_owned()
    } else {
        // Long (heap) mode
        // __cap_  at str_base + 0  (already read for the flag — ignore it)
        // __size_ at str_base + 4
        // __data_ at str_base + 8
        let len = u32::from_le_bytes(data[str_base + 4..str_base + 8].try_into().unwrap()) as usize;

        let heap_ptr =
            u32::from_le_bytes(data[str_base + 8..str_base + 12].try_into().unwrap()) as usize;

        if heap_ptr == 0 || heap_ptr.checked_add(len).is_none_or(|end| end > data.len()) {
            return "<heap string out of bounds>".into();
        }

        String::from_utf8_lossy(&data[heap_ptr..heap_ptr + len]).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a 12-byte wasm32 `std::string` in short (SSO) mode.
    ///
    /// `s` must be ≤ 10 bytes (the maximum for wasm32 SSO).
    fn make_short_string(s: &str) -> Vec<u8> {
        assert!(s.len() <= 10, "too long for SSO on wasm32");
        let mut buf = vec![0u8; 12];
        buf[0] = (s.len() as u8) << 1; // low bit 0 = short
        buf[1..1 + s.len()].copy_from_slice(s.as_bytes());
        buf
    }

    /// Builds a 12-byte wasm32 `std::string` in long (heap) mode.
    ///
    /// `heap_offset` is the byte offset within the fake heap where the string
    /// data will be stored by the caller.
    fn make_long_string(s: &str, heap_offset: u32) -> Vec<u8> {
        let cap: u32 = (s.len() as u32 + 15) & !15;
        let cap_field = cap | 1; // low bit 1 = long
        let mut buf = vec![0u8; 12];
        buf[0..4].copy_from_slice(&cap_field.to_le_bytes());
        buf[4..8].copy_from_slice(&(s.len() as u32).to_le_bytes());
        buf[8..12].copy_from_slice(&heap_offset.to_le_bytes());
        buf
    }

    #[test]
    fn short_string_decoded_correctly() {
        let string_bytes = make_short_string("hello");
        let mut heap = vec![0u8; 64];
        heap[4..16].copy_from_slice(&string_bytes); // vtable(4) + string(12)
        assert_eq!(decode_cpp_exception(&heap, 0), "hello");
    }

    #[test]
    fn empty_short_string() {
        let string_bytes = make_short_string("");
        let mut heap = vec![0u8; 64];
        heap[4..16].copy_from_slice(&string_bytes);
        assert_eq!(decode_cpp_exception(&heap, 0), "");
    }

    #[test]
    fn max_length_sso_string() {
        // 10 chars is the maximum inline length on wasm32.
        let string_bytes = make_short_string("0123456789");
        let mut heap = vec![0u8; 64];
        heap[4..16].copy_from_slice(&string_bytes);
        assert_eq!(decode_cpp_exception(&heap, 0), "0123456789");
    }

    #[test]
    fn long_string_decoded_correctly() {
        let msg = "this is a longer message";
        let data_offset: u32 = 32;
        let string_bytes = make_long_string(msg, data_offset);

        let mut heap = vec![0u8; 64];
        heap[4..16].copy_from_slice(&string_bytes);
        heap[data_offset as usize..data_offset as usize + msg.len()]
            .copy_from_slice(msg.as_bytes());

        assert_eq!(decode_cpp_exception(&heap, 0), msg);
    }

    #[test]
    fn long_string_null_heap_ptr_returns_sentinel() {
        // heap_ptr == 0 must not be dereferenced.
        let mut heap = vec![0u8; 64];
        let cap_field: u32 = 16 | 1;
        heap[4..8].copy_from_slice(&cap_field.to_le_bytes()); // __cap_ | 1
        heap[8..12].copy_from_slice(&5u32.to_le_bytes()); // __size_ = 5
        // heap[12..16] is 0 → __data_ = 0 (null)
        assert_eq!(
            decode_cpp_exception(&heap, 0),
            "<heap string out of bounds>"
        );
    }

    #[test]
    fn exception_object_out_of_bounds() {
        // Heap too small to contain exc + 4 + 12 bytes.
        let heap = vec![0u8; 8];
        assert_eq!(
            decode_cpp_exception(&heap, 0),
            "<exception object out of bounds>"
        );
    }

    #[test]
    fn sso_length_exceeds_heap_returns_sentinel() {
        // Encode a length that would read past the end of the heap.
        let mut heap = vec![0u8; 16];
        // str_base = 4; flag byte = len=15 << 1 = 30 (short mode, low bit 0)
        heap[4] = 30;
        // char_start = 5; 5 + 15 = 20 > 16 → out of bounds
        assert_eq!(decode_cpp_exception(&heap, 0), "<sso string out of bounds>");
    }
}
