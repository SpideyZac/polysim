//! The [`SimulationWorker`] drives one WASM physics instance.

use anyhow::{Context, anyhow, bail};
use bytemuck::cast_slice;

use super::{car_state::CarState, prepared::PreparedTrack, types::PlayerController};
use crate::{
    data_reader::{Detector, assets},
    physics::{Exports, PolyTrackPhysics},
};

/// Upper bound on `car_id`, since ids are used to directly index a dense
/// `Vec`. Far beyond any realistic simultaneous-car count; exists only to
/// stop a stray huge id from forcing a huge allocation.
const MAX_CAR_ID: u32 = 1 << 16;

/// Per-car state tracked on the Rust side.
struct Car {
    controls: PlayerController,
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
    /// Dense, id-indexed car storage: `cars[id]` is `Some` iff a car with
    /// that id currently exists. Car ids are small caller-chosen integers
    /// (see `MAX_CAR_ID`), so direct indexing gives O(1) lookups in
    /// `set_car_controls`/`update_car` instead of the linear scan a
    /// `Vec<Car>` + `.find()` would need on every tick.
    cars: Vec<Option<Car>>,
    /// The official worker allocates one shared 227-byte output buffer as soon
    /// as the WASM runtime is ready, before processing its Init message.
    state_buffer_ptr: i32,
    /// Prepared track data shared across all cars on this worker.
    prepared: PreparedTrack,
}

impl SimulationWorker {
    /// Allocate the official worker's shared state buffer and return a new worker.
    ///
    /// Does **not** call [`init`] - that must be called separately before
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
        let state_buffer_ptr = physics
            .alloc_bytes(&[0u8; 227])
            .context("allocate shared car state buffer")?;

        Ok(Self {
            physics,
            exports,
            cars: Vec::new(),
            state_buffer_ptr,
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
    /// `car_id` must be unique among all cars currently in this worker and
    /// below `MAX_CAR_ID` (car ids directly index internal storage).
    /// Output is read from the same shared buffer as the official worker and
    /// deserialised before the next update can overwrite it.
    pub fn create_car(&mut self, car_id: u32) -> anyhow::Result<()> {
        let idx = self.car_index(car_id)?;
        if matches!(self.cars.get(idx), Some(Some(_))) {
            bail!("car {car_id} already exists");
        }

        self.spawn_car_model(car_id)
            .with_context(|| format!("spawn car model for car {car_id}"))?;

        if idx >= self.cars.len() {
            self.cars.resize_with(idx + 1, || None);
        }
        self.cars[idx] = Some(Car {
            controls: PlayerController::default(),
        });
        Ok(())
    }

    /// Remove a car from the simulation and free its WASM-side resources.
    pub fn delete_car(&mut self, car_id: u32) -> anyhow::Result<()> {
        self.physics
            .call(&self.exports.delete_car_model, car_id as i32)?;
        if let Some(slot) = self.cars.get_mut(car_id as usize) {
            slot.take();
        }
        Ok(())
    }

    /// Respawn `car_id` at the track's start position without touching its
    /// WASM-heap state buffer.
    ///
    /// Equivalent to `delete_car` followed by `create_car`, but skips the
    /// `malloc`/`free` round-trip through the WASM allocator by reusing the
    /// buffer the car already has. Prefer this for reset-heavy workloads
    /// (brute-force search, TAS tooling) that repeatedly restart a car from
    /// scratch.
    pub fn reset_car(&mut self, car_id: u32) -> anyhow::Result<()> {
        if !matches!(self.cars.get(car_id as usize), Some(Some(_))) {
            bail!("car {car_id} not found");
        }

        self.physics
            .call(&self.exports.delete_car_model, car_id as i32)?;
        self.spawn_car_model(car_id)
            .with_context(|| format!("respawn car model for car {car_id}"))?;

        // Controls reset to defaults, matching a from-scratch `create_car`.
        if let Some(Some(car)) = self.cars.get_mut(car_id as usize) {
            car.controls = PlayerController::default();
        }
        Ok(())
    }

    /// Call the WASM `create_car_model` export for `car_id` at the track's
    /// start position. Shared by [`create_car`] and [`reset_car`].
    ///
    /// [`create_car`]: Self::create_car
    /// [`reset_car`]: Self::reset_car
    fn spawn_car_model(&mut self, car_id: u32) -> anyhow::Result<()> {
        // Match simulation_worker.bundle.js exactly: mountain and track are
        // allocated immediately before createCarModel and freed immediately
        // afterwards. Allocation order can affect Bullet's internal pointer-
        // keyed containers, so persistent buffers are not equivalent here.
        let mountain_ptr = self
            .physics
            .alloc_bytes(cast_slice(&self.prepared.mountain.vertices))
            .context("allocate mountain vertex buffer")?;
        let track_ptr = self
            .physics
            .alloc_bytes(&self.prepared.track_bytes)
            .context("allocate track data")?;

        let s = &self.prepared.start;
        let result = self.physics.call(
            &self.exports.create_car_model,
            (
                car_id as i32,
                mountain_ptr,
                self.prepared.mountain.vertices.len() as i32,
                self.prepared.mountain.offset[0],
                self.prepared.mountain.offset[1],
                self.prepared.mountain.offset[2],
                track_ptr,
                (self.prepared.track_bytes.len() / 19) as i32,
                s.position[0],
                s.position[1],
                s.position[2],
                s.quaternion[0],
                s.quaternion[1],
                s.quaternion[2],
                s.quaternion[3],
            ),
        );

        // Preserve the browser worker's successful-path free order. Also try
        // to release both temporary buffers if the WASM call reports an error.
        let free_mountain = self.physics.free_wasm(mountain_ptr);
        let free_track = self.physics.free_wasm(track_ptr);
        result?;
        free_mountain?;
        free_track?;
        Ok(())
    }

    /// Validate `car_id` and return its index into `cars`.
    fn car_index(&self, car_id: u32) -> anyhow::Result<usize> {
        if car_id >= MAX_CAR_ID {
            bail!("car id {car_id} exceeds maximum of {MAX_CAR_ID}");
        }
        Ok(car_id as usize)
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
    /// The result is deserialised from the official worker's shared buffer
    /// before a later car update can overwrite it.
    pub fn update_car(&mut self, car_id: u32) -> anyhow::Result<CarState> {
        self.advance_car(car_id)?;

        let raw = self
            .physics
            .wasm_slice(self.state_buffer_ptr, 227)
            .context("read car state buffer")?;

        CarState::deserialize(
            &raw[4..],
            self.prepared.max_checkpoint,
            &self.prepared.checkpoint_positions,
            &self.prepared.finish_positions,
        )
        .context("deserialize car state")
    }

    /// Advance one tick and return the exact 227 bytes emitted by the
    /// original physics module (including its four-byte car-id header).
    ///
    /// This is primarily useful for differential testing against the browser
    /// worker and for byte-exact replay validation. Prefer [`Self::update_car`] when
    /// the parsed [`CarState`] is sufficient.
    pub fn update_car_raw(&mut self, car_id: u32) -> anyhow::Result<[u8; 227]> {
        self.advance_car(car_id)?;
        let raw = self
            .physics
            .wasm_slice(self.state_buffer_ptr, 227)
            .context("read car state buffer")?;
        Ok(raw.try_into().expect("slice length is fixed at 227 bytes"))
    }

    /// Invoke the WASM update export using the controls currently stored for a car.
    fn advance_car(&mut self, car_id: u32) -> anyhow::Result<()> {
        let (up, right, down, left, reset) = {
            let c = self.car_mut(car_id)?;
            (
                c.controls.up as i32,
                c.controls.right as i32,
                c.controls.down as i32,
                c.controls.left as i32,
                c.controls.reset as i32,
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
                self.state_buffer_ptr,
            ),
        )?;
        Ok(())
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
    ///
    /// O(1): car ids directly index `cars`, so this is a bounds check plus an
    /// array access rather than the linear scan a `Vec<Car>` + `.find()`
    /// would require.
    fn car_mut(&mut self, car_id: u32) -> anyhow::Result<&mut Car> {
        self.cars
            .get_mut(car_id as usize)
            .and_then(Option::as_mut)
            .ok_or_else(|| anyhow!("car {car_id} not found"))
    }
}
