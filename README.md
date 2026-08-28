# PolySim

Headless [PolyTrack](https://kodub.itch.io/polytrack) physics simulation in Rust.

Wraps the PolyTrack WASM physics engine in a safe, ergonomic API for
deterministic, offline car simulation: no browser, no renderer, no game loop.
Useful for TAS tooling, AI training, replay systems, and brute-force search.

## Requirements

- **Rust 1.85+** (edition 2024 required by `polytrack-codes` and `wasmtime 47.0.3`)
- The `polytrack_physics.wasm` file matching the PolyTrack version you want to reproduce

## Exact browser parity

PolySim runs PolyTrack's original WASM physics rather than reimplementing
Bullet physics in native Rust. The surrounding Rust code mirrors the 0.6.2
simulation worker's allocation order, JavaScript-number-to-`f32` conversion,
start transform, mountain generation, track packing, and one-millisecond tick.

For PolyTrack 0.6.2, use
[`polytrack_physics.wasm`](https://app-polytrack.kodub.com/0.6.2/polytrack_physics.wasm).
Its SHA-256 is
`d4ef02676973d41afc34b23b5248f6950b35dc4cc7e3047e3a9c6bd88e4c180e`.
Mixing assets, the JS worker, and WASM from different game versions is not
expected to reproduce a browser replay exactly.

`SimulationWorker::update_car_raw` returns the complete 227-byte WASM output
for differential tests. A reference run of 1,000 forward-only frames on the
included benchmark track matches the official 0.6.2 Chromium worker byte for
byte (227,000 compared bytes).

## Usage

```toml
[dependencies]
polysim = { git = "https://github.com/SpideyZac/polysim.git" }
wasmtime = "47.0.3"  # for Module when sharing across workers
```

### Single car

```rust
use polysim::{
    physics::{PolyTrackPhysics, create_engine},
    simulation::{PlayerController, PreparedTrack, SimulationWorker},
};

let engine = create_engine();
let (physics, _module) = PolyTrackPhysics::from_file(&engine, "physics.wasm")?;

let prepared = PreparedTrack::from_export_string("PolyTrack24...")?;
let mut worker = SimulationWorker::new(physics, prepared)?;
worker.init()?;
worker.create_car(0)?;

loop {
    worker.set_car_controls(0, PlayerController { up: true, ..Default::default() })?;
    let state = worker.update_car(0)?;
    println!("frame {} speed {:.1} km/h", state.frames, state.speed_kmh);
    if state.finish_frames.is_some() {
        println!("finished at frame {}", state.finish_frames.unwrap());
        break;
    }
}
```

### Multiple cars in one worker

The worker uses the same shared output buffer as PolyTrack. Each call parses
its result before the next update, so sequential multi-car updates are safe:

```rust
worker.create_car(0)?;
worker.create_car(1)?;

worker.set_car_controls(0, PlayerController { up: true, ..Default::default() })?;
worker.set_car_controls(1, PlayerController { up: true, right: true, ..Default::default() })?;

let state0 = worker.update_car(0)?;
let state1 = worker.update_car(1)?;
```

### Multiple parallel workers

Compile once, instantiate one `PolyTrackPhysics` per thread from the shared
`Module`. Each worker owns its own WASM heap - no locking required:

```rust
use wasmtime::Module;

let engine = create_engine();
let module = Module::from_file(&engine, "physics.wasm")?;

std::thread::spawn({
    let engine = engine.clone();
    let module = module.clone();
    move || {
        let phys = PolyTrackPhysics::from_module(&engine, &module).unwrap();
        // ... create worker, run sim
    }
});
```

## Key types

| Type | Description |
| --- | --- |
| [`PreparedTrack`] | Decoded track - construct once, share across workers |
| [`SimulationWorker`] | Drives one WASM instance; supports N simultaneous cars |
| [`CarState`] | Full per-frame snapshot: position, speed, checkpoints, wheels |
| [`PlayerController`] | Five binary inputs: up, right, down, left, reset |
| [`PolyTrackPhysics`] | Low-level WASM wrapper; rarely needed directly |

## `CarState` fields

| Field | Description |
| --- | --- |
| `frames` | Physics tick counter |
| `speed_kmh` | Current speed in km/h |
| `has_started` | Whether the timer is running |
| `finish_frames` | Tick at which the car finished, if at all |
| `next_checkpoint_index` | Index of the next checkpoint to hit |
| `is_finishline_cp` | Whether the next target is the finish line |
| `next_checkpoint_position` | World-space position of the next target |
| `position` | Car world-space XYZ |
| `quaternion` | Car orientation [x, y, z, w] |
| `collision_impulses` | Magnitude of collisions this tick (up to 4) |
| `wheel_contacts` | Per-wheel contact point and normal |
| `wheel_suspension_lengths` | Current suspension compression |
| `steering` | Steering angle in radians |
