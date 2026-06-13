//! Wasmtime engine construction.

use wasmtime::{Config, Engine, OptLevel};

/// Creates a [`wasmtime::Engine`] configured for maximum physics throughput.
///
/// Settings applied:
/// - **Cranelift `Speed` optimisation level** — maximises JIT quality at the
///   cost of slightly longer first-compile time (acceptable; modules are
///   compiled once and reused).
///
/// # Panics
/// Panics if wasmtime cannot initialise with the given config, which should
/// never happen on a supported platform.
pub fn create_engine() -> Engine {
    let mut config = Config::new();
    config.cranelift_opt_level(OptLevel::Speed);
    Engine::new(&config).unwrap()
}
