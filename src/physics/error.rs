//! Error type for WASM physics operations.

/// All errors that can arise from interacting with the physics WASM module.
#[derive(Debug, thiserror::Error)]
pub enum PhysicsError {
    /// The WASM module called `abort()` or triggered an assertion failure.
    /// The exit code mirrors the process exit code the module would have
    /// produced natively (134 for `abort()`/`SIGABRT`).
    #[error("wasm module exited with code {0}")]
    WasmExited(i32),

    /// A host-side memory operation targeted a byte range that falls outside
    /// the current WASM linear memory.
    #[error("out-of-bounds wasm memory access: offset {offset} + len {len} > heap size {heap}")]
    OutOfBounds {
        /// Start byte of the attempted access.
        offset: usize,
        /// Number of bytes requested.
        len: usize,
        /// Current size of the WASM heap in bytes.
        heap: usize,
    },

    /// A wasmtime-level error (trap, type mismatch, instantiation failure, …).
    #[error("wasm error: {0}")]
    Wasm(#[from] wasmtime::Error),
}
