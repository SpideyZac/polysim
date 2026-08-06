//! The live WASM physics instance.

use wasmtime::{
    Engine, Linker, Memory, Module, Store, TypedFunc, WasmParams, WasmResults, format_err,
};

use super::{
    error::PhysicsError,
    exports::Exports,
    host::{HostState, register},
};

/// A [`Linker`] with every physics-module host import already registered.
///
/// Building one of these and reusing it via
/// [`PolyTrackPhysics::from_module_with_linker`] avoids re-registering the
/// same handful of imports for every worker you spawn. On its own this is a
/// small saving (a handful of hashmap inserts), but it adds up if you're
/// creating many short-lived instances, e.g. one per branch of a brute-force
/// search.
pub type PhysicsLinker = Linker<HostState>;

/// Build a [`PhysicsLinker`] with all physics-module imports registered.
///
/// Share the result across multiple [`PolyTrackPhysics::from_module_with_linker`]
/// calls instead of letting [`PolyTrackPhysics::from_module`] build (and
/// register) a fresh one every time.
pub fn create_linker(engine: &Engine) -> Result<PhysicsLinker, PhysicsError> {
    let mut linker = Linker::<HostState>::new(engine);
    register(&mut linker)?;
    Ok(linker)
}

/// A live instance of the PolyTrack physics WASM module.
///
/// Owns the wasmtime [`Store`], linear [`Memory`], and all typed export
/// handles.  All physics calls go through [`call`], which checks for
/// graceful-exit conditions before and after every invocation.
///
/// # Instantiation
///
/// For a single worker:
/// ```rust
/// # use polysim::physics::{create_engine, PolyTrackPhysics};
/// let engine = create_engine();
/// let (physics, _module) = PolyTrackPhysics::from_file(&engine, "physics.wasm")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// For multiple workers sharing one compiled module (cheaper):
/// ```rust
/// # use polysim::physics::{create_engine, PolyTrackPhysics};
/// # use wasmtime::Module;
/// let engine = create_engine();
/// let module = Module::from_file(&engine, "physics.wasm")?;
/// let worker_a = PolyTrackPhysics::from_module(&engine, &module)?;
/// let worker_b = PolyTrackPhysics::from_module(&engine, &module)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// [`call`]: PolyTrackPhysics::call
pub struct PolyTrackPhysics {
    store: Store<HostState>,
    memory: Memory,
    exports: Exports,
}

impl PolyTrackPhysics {
    /// Load a `.wasm` file from disk, compile it, and instantiate it.
    ///
    /// Also returns the compiled [`Module`] so it can be reused with
    /// [`from_module`] for additional workers without recompiling.
    ///
    /// [`from_module`]: Self::from_module
    pub fn from_file(engine: &Engine, wasm_path: &str) -> Result<(Self, Module), PhysicsError> {
        let module = Module::from_file(engine, wasm_path)?;
        let physics = Self::from_module(engine, &module)?;
        Ok((physics, module))
    }

    /// Instantiate from a pre-compiled [`Module`].
    ///
    /// Prefer this when spawning multiple [`SimulationWorker`]s - compile the
    /// module once with [`from_file`] or [`Module::from_file`], then call this
    /// for each additional worker.
    ///
    /// Builds a fresh [`PhysicsLinker`] internally. If you're spawning many
    /// workers in a hot loop, build one with [`create_linker`] once and use
    /// [`from_module_with_linker`] instead to skip the repeated import
    /// registration.
    ///
    /// [`SimulationWorker`]: crate::simulation::SimulationWorker
    /// [`from_file`]: Self::from_file
    /// [`from_module_with_linker`]: Self::from_module_with_linker
    pub fn from_module(engine: &Engine, module: &Module) -> Result<Self, PhysicsError> {
        let linker = create_linker(engine)?;
        Self::from_module_with_linker(engine, module, &linker)
    }

    /// Instantiate from a pre-compiled [`Module`] using an already-built
    /// [`PhysicsLinker`].
    ///
    /// Prefer this over [`from_module`] when spawning many workers: build the
    /// linker once with [`create_linker`] and reuse it for every
    /// instantiation instead of re-registering the same imports each time.
    ///
    /// [`from_module`]: Self::from_module
    pub fn from_module_with_linker(
        engine: &Engine,
        module: &Module,
        linker: &PhysicsLinker,
    ) -> Result<Self, PhysicsError> {
        let mut store = Store::new(engine, HostState::default());

        let instance = linker.instantiate(&mut store, module)?;

        // Run the optional Emscripten global constructor (`__wasm_call_ctors`)
        // if the module exports it.  Missing export is not an error.
        if let Ok(init_fn) = instance.get_typed_func::<(), ()>(&mut store, "k") {
            init_fn.call(&mut store, ())?;
        }

        let exports = Exports {
            malloc: instance.get_typed_func(&mut store, "l")?,
            free: instance.get_typed_func(&mut store, "m")?,
            init_car_collision_shape: instance.get_typed_func(&mut store, "n")?,
            add_track_part_config: instance.get_typed_func(&mut store, "o")?,
            create_car_model: instance.get_typed_func(&mut store, "p")?,
            delete_car_model: instance.get_typed_func(&mut store, "q")?,
            update_car_model: instance.get_typed_func(&mut store, "r")?,
            test_determinism: instance.get_typed_func(&mut store, "s")?,
        };

        let memory = instance
            .get_memory(&mut store, "j")
            .ok_or_else(|| format_err!("wasm memory export \"j\" missing"))?;

        Ok(Self {
            store,
            exports,
            memory,
        })
    }

    /// Allocate `data.len()` bytes in WASM heap, copy `data` into them, and
    /// return the wasm32 pointer.
    ///
    /// The caller is responsible for freeing with [`free_wasm`].
    ///
    /// # Errors
    /// - [`PhysicsError::WasmExited`] if the module has already exited.
    /// - [`PhysicsError::Wasm`] if `malloc` traps.
    /// - [`PhysicsError::Wasm`] (via `format_err`) if `malloc` returns null.
    /// - [`PhysicsError::OutOfBounds`] if the returned pointer is outside
    ///   linear memory (should not happen with a correct WASM module).
    ///
    /// [`free_wasm`]: Self::free_wasm
    pub fn alloc_bytes(&mut self, data: &[u8]) -> Result<i32, PhysicsError> {
        self.check_exited()?;

        let ptr = self
            .exports
            .malloc
            .call(&mut self.store, data.len() as i32)?;
        if ptr == 0 {
            return Err(format_err!("wasm malloc returned null").into());
        }

        self.write_bytes(ptr, data)?;
        Ok(ptr)
    }

    /// Free a wasm32 heap pointer previously returned by [`Self::alloc_bytes`].
    ///
    /// Passing `0` is a no-op, mirroring `free(NULL)` in C.
    pub fn free_wasm(&mut self, ptr: i32) -> Result<(), PhysicsError> {
        self.check_exited()?;
        if ptr != 0 {
            self.exports.free.call(&mut self.store, ptr)?;
        }
        self.check_exited()
    }

    /// Return a read-only byte slice into WASM linear memory at `[ptr, ptr+len)`.
    ///
    /// The slice borrows from `self` and is only valid until the next mutable
    /// call (e.g. [`call`], [`alloc_bytes`]).
    ///
    /// # Errors
    /// Returns [`PhysicsError::OutOfBounds`] if the range falls outside the
    /// current heap.
    ///
    /// [`call`]: Self::call
    /// [`alloc_bytes`]: Self::alloc_bytes
    pub fn wasm_slice(&self, ptr: i32, len: usize) -> Result<&[u8], PhysicsError> {
        let offset = ptr as usize;
        let heap = self.memory.data(&self.store).len();

        if offset.checked_add(len).is_none_or(|end| end > heap) {
            return Err(PhysicsError::OutOfBounds { offset, len, heap });
        }

        Ok(&self.memory.data(&self.store)[offset..offset + len])
    }

    /// Call a typed WASM export, checking for a graceful exit before and after.
    ///
    /// Any trap or `abort()` during the call is surfaced as a
    /// [`PhysicsError::Wasm`] or [`PhysicsError::WasmExited`] respectively.
    pub fn call<T, R>(&mut self, f: &TypedFunc<T, R>, args: T) -> Result<R, PhysicsError>
    where
        T: WasmParams,
        R: WasmResults,
    {
        self.check_exited()?;
        let result = f.call(&mut self.store, args)?;
        self.check_exited()?;
        Ok(result)
    }

    /// Clone the [`Exports`] handle set.
    ///
    /// Cloning is cheap - all handles are reference-counted internally by
    /// wasmtime.  The cloned handles share the same store and are only valid
    /// while `self` lives.
    pub fn exports(&self) -> Exports {
        self.exports.clone()
    }

    /// Return an error if the WASM module has called `abort()` or exited.
    fn check_exited(&self) -> Result<(), PhysicsError> {
        if let Some(code) = self.store.data().check_exit() {
            return Err(PhysicsError::WasmExited(code));
        }
        Ok(())
    }

    /// Copy `data` into WASM linear memory at `ptr`, with bounds checking.
    fn write_bytes(&mut self, ptr: i32, data: &[u8]) -> Result<(), PhysicsError> {
        let offset = ptr as usize;
        let heap = self.memory.data(&self.store).len();

        if offset.checked_add(data.len()).is_none_or(|end| end > heap) {
            return Err(PhysicsError::OutOfBounds {
                offset,
                len: data.len(),
                heap,
            });
        }

        self.memory.data_mut(&mut self.store)[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }
}
