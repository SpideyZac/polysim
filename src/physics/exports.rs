//! Typed handles to every function exported by the physics WASM module.
//!
//! Export names are single ASCII characters - the result of Emscripten's
//! symbol minification.  They must match the compiled `.wasm` binary exactly.

use wasmtime::TypedFunc;

/// Arguments to `init_car_collision_shape` (export `"n"`).
///
/// `(mass_offset: f32, vertices_ptr: i32, vertices_len: i32)`
pub type InitCarCollisionShapeArgs = (f32, i32, i32);

/// Arguments to `add_track_part_config` (export `"o"`).
///
/// ```text
/// part_id:          i32
/// vertices_ptr:     i32
/// vertices_len:     i32
/// detector_type:    i32
/// detector_cx:      f32   ─╮
/// detector_cy:      f32    │ detector centre XYZ
/// detector_cz:      f32   ─╯
/// detector_sx:      f32   ─╮
/// detector_sy:      f32    │ detector half-extents XYZ
/// detector_sz:      f32   ─╯
/// has_start_offset: i32   (0 or 1)
/// start_offset_x:   f32   ─╮
/// start_offset_y:   f32    │ local spawn offset XYZ
/// start_offset_z:   f32   ─╯
/// ```
pub type AddTrackPartConfigArgs = (
    i32,
    i32,
    i32,
    i32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    i32,
    f32,
    f32,
    f32,
);

/// Arguments to `create_car_model` (export `"p"`).
///
/// ```text
/// car_id:                i32
/// mountain_ptr:          i32   (0 if no mountain)
/// mountain_vertices_len: i32   (0 if no mountain)
/// mountain_offset_x:     f32   ─╮
/// mountain_offset_y:     f32    │ world-space mountain origin
/// mountain_offset_z:     f32   ─╯
/// track_ptr:             i32
/// part_count:            i32
/// start_x:               f32   ─╮
/// start_y:               f32    │ car spawn position
/// start_z:               f32   ─╯
/// start_qx:              f32   ─╮
/// start_qy:              f32    │ car spawn quaternion (xyzw)
/// start_qz:              f32    │
/// start_qw:              f32   ─╯
/// ```
pub type CreateCarModelArgs = (
    i32,
    i32,
    i32,
    f32,
    f32,
    f32,
    i32,
    i32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
);

/// Arguments to `update_car_model` (export `"r"`).
///
/// `(car_id, up, right, down, left, reset, state_buf_ptr)` - all `i32`,
/// controls are `0` or `1`.
pub type UpdateCarModelArgs = (i32, i32, i32, i32, i32, i32, i32);

/// Cloneable handles to every typed function exported by the physics WASM
/// module.
///
/// Obtained via [`PolyTrackPhysics::exports`].  All handles share the same
/// underlying wasmtime store; they are only valid while that store lives.
#[derive(Clone)]
pub struct Exports {
    /// WASM `malloc`: allocate `n` bytes, return pointer.  Export `"l"`.
    pub malloc: TypedFunc<i32, i32>,
    /// WASM `free`: release a pointer returned by `malloc`.  Export `"m"`.
    pub free: TypedFunc<i32, ()>,
    /// Upload the car's convex-hull collision shape.  Export `"n"`.
    pub init_car_collision_shape: TypedFunc<InitCarCollisionShapeArgs, ()>,
    /// Register one track-part type with the physics engine.  Export `"o"`.
    pub add_track_part_config: TypedFunc<AddTrackPartConfigArgs, ()>,
    /// Spawn a new car in the simulation.  Export `"p"`.
    pub create_car_model: TypedFunc<CreateCarModelArgs, ()>,
    /// Remove a car from the simulation.  Export `"q"`.
    pub delete_car_model: TypedFunc<i32, ()>,
    /// Advance one physics tick for a car and write its state.  Export `"r"`.
    pub update_car_model: TypedFunc<UpdateCarModelArgs, ()>,
    /// Run the built-in determinism self-test; returns non-zero on success.  Export `"s"`.
    pub test_determinism: TypedFunc<(), i32>,
}
