//! High-level simulation layer.
//!
//! This module wraps the low-level WASM physics calls in an ergonomic Rust API
//! suitable for AI training loops and replay systems.
//!
//! # Typical usage
//!
//! ```rust
//! use polysim::{
//!     physics::{PolyTrackPhysics, create_engine},
//!     simulation::{PlayerController, PreparedTrack, SimulationWorker},
//! };
//!
//! const TRACK_EXPORT_STRING: &'static str = "PolyTrack24pdDBvuCABDFAAeL1Ubf5YuEOpZOu9TlLJTI1x80z3XHIY5z83DTiAX0mbmB19KPgF4gN8CODrsPaLf4eUljkI1OO9rFj7CghGZJ5rAKbd2zpCrp4VCzlarogH9tp1fgB";
//!
//! // 1. Compile the WASM module once.
//! let engine = create_engine();
//! let (physics, _module) = PolyTrackPhysics::from_file(&engine, "physics.wasm")?;
//!
//! // 2. Prepare static track data (decode, pack, build mountain, etc.).
//! let prepared = PreparedTrack::from_export_string(TRACK_EXPORT_STRING)?;
//!
//! // 3. Create a worker and initialise it.
//! let mut worker = SimulationWorker::new(physics, prepared)?;
//! worker.init()?;
//!
//! // 4. Spawn a car and run the simulation loop.
//! worker.create_car(0)?;
//! loop {
//!     worker.set_car_controls(
//!         0,
//!         PlayerController {
//!             up: true,
//!             ..Default::default()
//!         },
//!     )?;
//!     let state = worker.update_car(0)?;
//!     if state.finish_frames.is_some() {
//!         break;
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod car_state;
mod prepared;
mod track;
mod types;
mod worker;

pub use car_state::{CarState, CollisionImpulses, WheelContact};
pub use prepared::PreparedTrack;
pub use types::{PlayerController, StartTransform};
pub use worker::SimulationWorker;
