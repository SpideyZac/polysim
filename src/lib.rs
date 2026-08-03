//! `polysim` - headless PolyTrack physics simulation.
//!
//! Wraps the PolyTrack WASM physics engine in a safe, ergonomic Rust API
//! for deterministic, headless car simulation. Useful for TAS tooling, AI
//! training loops, replay systems, and any other use case requiring fast
//! offline simulation.
//!
//! # Architecture
//!
//! There are two layers:
//!
//! - **[`physics`]** - low-level WASM wrapper. Owns the wasmtime [`Store`],
//!   linear memory, and typed export handles. You rarely need to touch this
//!   directly.
//! - **[`simulation`]** - high-level API. [`PreparedTrack`] decodes a track
//!   export string once; [`SimulationWorker`] drives one WASM instance with
//!   one or more simultaneous cars.
//!
//! [`Store`]: wasmtime::Store
//! [`PreparedTrack`]: simulation::PreparedTrack
//! [`SimulationWorker`]: simulation::SimulationWorker
//!
//! # Quick start
//!
//! ```rust
//! use polysim::{
//!     physics::{PolyTrackPhysics, create_engine},
//!     simulation::{PlayerController, PreparedTrack, SimulationWorker},
//! };
//!
//! // 1. Compile the WASM module once - expensive, share across workers.
//! let engine = create_engine();
//! let (physics, module) = PolyTrackPhysics::from_file(&engine, "physics.wasm")?;
//!
//! // 2. Decode static track data - also done once per track.
//! let prepared = PreparedTrack::from_export_string("PolyTrack24...")?;
//!
//! // 3. Create a worker, initialise it, and spawn a car.
//! let mut worker = SimulationWorker::new(physics, prepared)?;
//! worker.init()?;
//! worker.create_car(0)?;
//!
//! // 4. Step the simulation.
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
//!
//! // 5. Spawn more workers cheaply from the compiled module.
//! let physics2 = PolyTrackPhysics::from_module(&engine, &module)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Multiple simultaneous cars
//!
//! A single [`SimulationWorker`] can run many cars at once. Each car gets its
//! own WASM-heap state buffer so reads never clobber each other:
//!
//! ```rust
//! # use polysim::{physics::{PolyTrackPhysics, create_engine}, simulation::{PlayerController, PreparedTrack, SimulationWorker}};
//! # let engine = create_engine();
//! # let (physics, _) = PolyTrackPhysics::from_file(&engine, "physics.wasm")?;
//! # let prepared = PreparedTrack::from_export_string("PolyTrack24...")?;
//! # let mut worker = SimulationWorker::new(physics, prepared)?;
//! # worker.init()?;
//! worker.create_car(0)?;
//! worker.create_car(1)?;
//! worker.create_car(2)?;
//!
//! worker.set_car_controls(0, PlayerController { up: true,  ..Default::default() })?;
//! worker.set_car_controls(1, PlayerController { up: true, right: true, ..Default::default() })?;
//! worker.set_car_controls(2, PlayerController { up: true, left:  true, ..Default::default() })?;
//!
//! let state0 = worker.update_car(0)?;
//! let state1 = worker.update_car(1)?;
//! let state2 = worker.update_car(2)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Multiple parallel workers
//!
//! Compile the WASM module once, then instantiate one [`PolyTrackPhysics`] per
//! thread via [`PolyTrackPhysics::from_module`]. Each worker owns its own WASM
//! heap - no locking required between workers:
//!
//! ```rust
//! # use polysim::physics::{PolyTrackPhysics, create_engine};
//! # use wasmtime::Module;
//! let engine = create_engine();
//! let module = Module::from_file(&engine, "physics.wasm")?;
//!
//! // Each thread gets its own physics instance from the shared module.
//! let worker_a = PolyTrackPhysics::from_module(&engine, &module)?;
//! let worker_b = PolyTrackPhysics::from_module(&engine, &module)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod data_reader;
pub mod physics;
pub mod simulation;
