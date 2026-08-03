//! Wasmtime engine construction.

use wasmtime::{Config, Engine, OptLevel, WasmBacktraceDetails};

/// Creates a [`wasmtime::Engine`] configured for maximum physics throughput.
///
/// Settings applied:
/// - **Cranelift `Speed` optimisation level** - maximises JIT quality at the
///   cost of slightly longer first-compile time (acceptable; modules are
///   compiled once and reused).
/// - **NaN canonicalisation stays *enabled*.** This is a correctness
///   requirement, not a tunable: without it, floating-point NaN bit patterns
///   produced by the physics module can differ across host CPU
///   architectures, which breaks bit-for-bit determinism when a replay or
///   search result recorded on one machine is verified on another. Do not
///   disable this for a speed win.
/// - **Wasm backtrace details disabled.** Traps are already surfaced as
///   [`PhysicsError`](super::error::PhysicsError) with our own context, so we
///   don't need Wasmtime's own backtrace bookkeeping on every call boundary.
///
/// # Panics
/// Panics if wasmtime cannot initialise with the given config, which should
/// never happen on a supported platform.
pub fn create_engine() -> Engine {
    let mut config = Config::new();
    config.cranelift_opt_level(OptLevel::Speed);
    config.cranelift_nan_canonicalization(true);
    config.wasm_backtrace_details(WasmBacktraceDetails::Disable);
    Engine::new(&config).unwrap()
}
