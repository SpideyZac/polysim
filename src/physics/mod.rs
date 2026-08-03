//! WASM physics engine wrapper.
//!
//! This module provides a safe Rust interface over the PolyTrack physics WASM
//! module compiled with Emscripten for wasm32.
//!
//! # Quick start
//!
//! ```rust
//! use polysim::physics::{PolyTrackPhysics, create_engine};
//!
//! let engine = create_engine();
//! let (mut physics, module) = PolyTrackPhysics::from_file(&engine, "physics.wasm")?;
//! // For additional workers, reuse the compiled module:
//! let physics2 = PolyTrackPhysics::from_module(&engine, &module)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod engine;
mod error;
mod exports;
mod host;
mod instance;

pub use engine::create_engine;
pub use error::PhysicsError;
pub use exports::{
    AddTrackPartConfigArgs, CreateCarModelArgs, Exports, InitCarCollisionShapeArgs,
    UpdateCarModelArgs,
};
pub use host::HostState;
pub use instance::{PhysicsLinker, PolyTrackPhysics, create_linker};
