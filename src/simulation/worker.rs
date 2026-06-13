//! The [`SimulationWorker`] drives one WASM physics instance.

use anyhow::{Context, anyhow};
use bytemuck::cast_slice;

use super::{car_state::CarState, prepared::PreparedTrack, types::PlayerController};
use crate::{
    data_reader::{Detector, assets},
    physics::{Exports, PolyTrackPhysics},
};

/// Per-car state tracked on the Rust side.
struct Car {
    id: u32,
    controls: PlayerController,
    /// Pointer to this car's dedicated 227-byte state read-back buffer in WASM heap.
    state_buffer_ptr: i32,
}

/// Drives one WASM physics instance for one or more cars on a single track.
///
/// # Lifecycle
///
/// ```text
/// PreparedTrack::from_export_string(export_str)?
///     │
///     ▼
/// SimulationWorker::new(physics, prepared)?   // allocates WASM memory
///     │
///     ▼
/// worker.init()?                               // uploads part configs
///     │
///     ▼
/// worker.create_car(id)?                       // repeat per car
///     │
///     ▼
/// loop {
///     worker.set_car_controls(id, controls)?
///     let state = worker.update_car(id)?
/// }
/// ```
///
/// # Sharing a compiled module
///
/// Compile the WASM module once with [`PolyTrackPhysics::from_file`], then
/// pass the returned [`Module`] to [`PolyTrackPhysics::from_module`] for each
/// additional worker to avoid redundant JIT compilation.
///
/// [`Module`]: wasmtime::Module
pub struct SimulationWorker {
    physics: PolyTrackPhysics,
    exports: Exports,
    cars: Vec<Car>,
    /// Pointer to the packed track data blob in WASM heap.
    track_ptr: i32,
    /// Number of 19-byte block records in the track blob.
    part_count: i32,
    /// Pointer to the mountain vertex buffer in WASM heap, or `0` if the
    /// track is too large for a mountain.
    mountain_ptr: i32,
    /// Number of `f32` values in the mountain vertex buffer (`0` = no mountain).
    mountain_vertices_len: i32,
    /// World-space XYZ translation of the mountain mesh.
    mountain_offset: [f32; 3],
    /// Prepared track data shared across all cars on this worker.
    prepared: PreparedTrack,
}

impl SimulationWorker {
    /// Allocate WASM memory for the track and mountain data and return a new
    /// worker.
    ///
    /// Does **not** call [`init`] — that must be called separately before
    /// creating any cars, so that callers can choose when to pay the
    /// initialisation cost.
    ///
    /// # Errors
    /// Propagates any [`PhysicsError`] from WASM memory allocation.
    ///
    /// [`init`]: Self::init
    /// [`PhysicsError`]: crate::physics::PhysicsError
    pub fn new(mut physics: PolyTrackPhysics, prepared: PreparedTrack) -> anyhow::Result<Self> {
        let exports = physics.exports();

        // When the mountain is empty (track too large), pass ptr=0 and len=0
        // rather than calling malloc(0) whose return value is implementation-
        // defined and may be NULL in Emscripten's allocator.
        let (mountain_ptr, mountain_vertices_len) = if prepared.mountain.vertices.is_empty() {
            (0, 0)
        } else {
            let ptr = physics
                .alloc_bytes(cast_slice(&prepared.mountain.vertices))
                .context("allocate mountain vertex buffer")?;
            (ptr, prepared.mountain.vertices.len() as i32)
        };

        let track_ptr = physics
            .alloc_bytes(&prepared.track_bytes)
            .context("allocate track data")?;

        let part_count = (prepared.track_bytes.len() / 19) as i32;
        let mountain_offset = prepared.mountain.offset;

        Ok(Self {
            physics,
            exports,
            cars: Vec::new(),
            track_ptr,
            part_count,
            mountain_ptr,
            mountain_vertices_len,
            mountain_offset,
            prepared,
        })
    }

    /// Upload per-part collision geometry and detector configuration to the
    /// WASM module.
    ///
    /// Iterates **every known asset part type** exactly once, matching the
    /// original game's initialisation order and coverage.  Part types not
    /// present on the current track are registered anyway; the WASM module
    /// expects the full catalogue.
    ///
    /// Must be called exactly once after [`new`], before any cars are created.
    ///
    /// [`new`]: Self::new
    pub fn init(&mut self) -> anyhow::Result<()> {
        let asset_data = assets();

        let verts_ptr = self
            .physics
            .alloc_bytes(cast_slice(&asset_data.car_collision_vertices))
            .context("allocate car collision vertices")?;

        self.physics.call(
            &self.exports.init_car_collision_shape,
            (
                asset_data.car_mass_offset,
                verts_ptr,
                asset_data.car_collision_vertices.len() as i32,
            ),
        )?;
        self.physics.free_wasm(verts_ptr)?;

        for part_data in &asset_data.track_parts {
            let part_verts_ptr = self
                .physics
                .alloc_bytes(cast_slice(&part_data.vertices))
                .context("allocate part vertices")?;

            let det_default = Detector::default();
            let det = part_data.detector.as_ref().unwrap_or(&det_default);
            let start_offset = part_data.start_offset.unwrap_or([0.0; 3]);

            self.physics.call(
                &self.exports.add_track_part_config,
                (
                    part_data.id as i32,
                    part_verts_ptr,
                    part_data.vertices.len() as i32,
                    det.detector_type,
                    det.center[0],
                    det.center[1],
                    det.center[2],
                    det.size[0],
                    det.size[1],
                    det.size[2],
                    part_data.start_offset.is_some() as i32,
                    start_offset[0],
                    start_offset[1],
                    start_offset[2],
                ),
            )?;

            self.physics.free_wasm(part_verts_ptr)?;
        }

        Ok(())
    }

    /// Spawn a new car in the simulation at the track's start position.
    ///
    /// `car_id` must be unique among all cars currently in this worker.
    /// Each car gets its own dedicated WASM state buffer so multiple cars
    /// can be updated independently without overwriting each other's output.
    pub fn create_car(&mut self, car_id: u32) -> anyhow::Result<()> {
        let s = &self.prepared.start;

        // Allocate a dedicated 227-byte read-back buffer for this car.
        let state_buffer_ptr = self
            .physics
            .alloc_bytes(&[0u8; 227])
            .with_context(|| format!("allocate state buffer for car {car_id}"))?;

        self.physics.call(
            &self.exports.create_car_model,
            (
                car_id as i32,
                self.mountain_ptr,
                self.mountain_vertices_len,
                self.mountain_offset[0],
                self.mountain_offset[1],
                self.mountain_offset[2],
                self.track_ptr,
                self.part_count,
                s.position[0],
                s.position[1],
                s.position[2],
                s.quaternion[0],
                s.quaternion[1],
                s.quaternion[2],
                s.quaternion[3],
            ),
        )?;

        self.cars.push(Car {
            id: car_id,
            controls: PlayerController::default(),
            state_buffer_ptr,
        });
        Ok(())
    }

    /// Remove a car from the simulation and free its WASM-side resources.
    pub fn delete_car(&mut self, car_id: u32) -> anyhow::Result<()> {
        self.physics
            .call(&self.exports.delete_car_model, car_id as i32)?;
        if let Some(pos) = self.cars.iter().position(|c| c.id == car_id) {
            let car = self.cars.remove(pos);
            self.physics.free_wasm(car.state_buffer_ptr)?;
        }
        Ok(())
    }

    /// Replace the input state for `car_id`.
    ///
    /// The new controls take effect on the next call to [`update_car`].
    ///
    /// [`update_car`]: Self::update_car
    pub fn set_car_controls(
        &mut self,
        car_id: u32,
        controls: PlayerController,
    ) -> anyhow::Result<()> {
        self.car_mut(car_id)?.controls = controls;
        Ok(())
    }

    /// Advance the physics simulation by one tick for `car_id` and return the
    /// resulting [`CarState`].
    ///
    /// Each car writes into its own dedicated WASM buffer, so multiple cars
    /// can be updated in sequence without clobbering each other's output.
    pub fn update_car(&mut self, car_id: u32) -> anyhow::Result<CarState> {
        let (up, right, down, left, reset, state_buffer_ptr) = {
            let c = self.car_mut(car_id)?;
            (
                c.controls.up as i32,
                c.controls.right as i32,
                c.controls.down as i32,
                c.controls.left as i32,
                c.controls.reset as i32,
                c.state_buffer_ptr,
            )
        };

        self.physics.call(
            &self.exports.update_car_model,
            (
                car_id as i32,
                up,
                right,
                down,
                left,
                reset,
                state_buffer_ptr,
            ),
        )?;

        let raw = self
            .physics
            .wasm_slice(state_buffer_ptr, 227)
            .context("read car state buffer")?;

        CarState::deserialize(
            &raw[4..],
            self.prepared.max_checkpoint,
            &self.prepared.checkpoint_positions,
            &self.prepared.finish_positions,
        )
        .context("deserialize car state")
    }

    /// Run the WASM module's built-in determinism self-test.
    ///
    /// Returns `true` if the test passes.  Should be called once after
    /// [`init`] to verify that the physics module is operating deterministically
    /// on this platform.
    ///
    /// [`init`]: Self::init
    pub fn determinism_test(&mut self) -> anyhow::Result<bool> {
        Ok(self.physics.call(&self.exports.test_determinism, ())? != 0)
    }

    /// Return a mutable reference to the car with `car_id`, or an error.
    fn car_mut(&mut self, car_id: u32) -> anyhow::Result<&mut Car> {
        self.cars
            .iter_mut()
            .find(|c| c.id == car_id)
            .ok_or_else(|| anyhow!("car {car_id} not found"))
    }
}
