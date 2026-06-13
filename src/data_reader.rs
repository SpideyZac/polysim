//! Static simulation asset loading.
//!
//! Assets are compiled into the binary via [`include_bytes!`] and decoded once
//! on first access via a [`OnceLock`].  The binary format is a simple
//! little-endian packed struct written by the asset pipeline.
//!
//! # Binary format
//!
//! ```text
//! car_collision_vertices : u32 (count) + count × f32
//! car_mass_offset        : f32
//! part_count             : u32
//! for each part:
//!   id                   : u32
//!   vertices             : u32 (count) + count × f32
//!   has_detector         : u8  (0 or 1)
//!   if has_detector:
//!     detector_type      : i32
//!     center             : 3 × f32
//!     size               : 3 × f32
//!   has_start_offset     : u8  (0 or 1)
//!   if has_start_offset:
//!     start_offset       : 3 × f32
//! ```

use std::{
    io::{Cursor, Read},
    sync::OnceLock,
};

/// The simulation asset bundle, compiled into the binary at build time.
pub const SIM_DATA: &[u8] = include_bytes!("../simulation_assets.bin");

/// Global singleton for the decoded assets.
static ASSETS: OnceLock<SimulationAssets> = OnceLock::new();

/// Returns a reference to the globally-loaded simulation assets.
///
/// Decoded exactly once on first call using [`OnceLock`]; subsequent calls
/// return the cached value.  Safe to call from multiple threads.
pub fn assets() -> &'static SimulationAssets {
    ASSETS.get_or_init(|| load_assets(SIM_DATA))
}

/// Trigger volume attached to a track part, used by the physics engine to
/// detect checkpoint crossings, finish lines, and respawn surfaces.
#[derive(Debug, Clone)]
pub struct Detector {
    /// Physics-engine-defined detector category.  `-1` means "no detector".
    pub detector_type: i32,
    /// Centre of the trigger box in the part's local coordinate space.
    pub center: [f32; 3],
    /// Half-extents of the trigger box along each local axis.
    pub size: [f32; 3],
}

impl Default for Detector {
    /// Returns a disabled detector (`detector_type = -1`, zero centre and size).
    fn default() -> Self {
        Self {
            detector_type: -1,
            center: [0.0; 3],
            size: [0.0; 3],
        }
    }
}

/// Collision geometry and metadata for one track part type.
#[derive(Debug, Clone)]
pub struct TrackPart {
    /// Numeric part ID, matching the IDs used in encoded track data.
    pub id: u32,
    /// Flat XYZ triangle soup in local space (`len` is always a multiple of 9).
    pub vertices: Vec<f32>,
    /// Optional trigger volume; `None` if this part has no detector.
    pub detector: Option<Detector>,
    /// Local-space offset from the block origin to the car spawn point.
    /// `None` if this part is not a start/finish piece.
    pub start_offset: Option<[f32; 3]>,
}

/// All static data required to initialise the physics simulation.
#[derive(Debug, Clone)]
pub struct SimulationAssets {
    /// Convex-hull vertices for the car collision shape (flat XYZ soup).
    pub car_collision_vertices: Vec<f32>,
    /// Vertical offset from the car origin to its centre of mass.
    pub car_mass_offset: f32,
    /// All known track part types, in asset-database order.
    pub track_parts: Vec<TrackPart>,
}

/// Decode a [`SimulationAssets`] from a raw byte slice produced by the asset
/// pipeline.
///
/// # Panics
/// Panics with a descriptive message if the byte slice is truncated or
/// otherwise malformed.  This should never happen with the shipped binary; the
/// panic is preferable to silent data corruption.
pub fn load_assets(data: &[u8]) -> SimulationAssets {
    let mut r = Reader::new(data);

    let car_collision_vertices = r.vec_f32();
    let car_mass_offset = r.f32();
    let part_count = r.u32() as usize;

    let mut track_parts = Vec::with_capacity(part_count);

    for _ in 0..part_count {
        let id = r.u32();
        let vertices = r.vec_f32();

        let detector = if r.u8() == 1 {
            Some(Detector {
                detector_type: r.i32(),
                center: r.vec3(),
                size: r.vec3(),
            })
        } else {
            None
        };

        let start_offset = if r.u8() == 1 { Some(r.vec3()) } else { None };

        track_parts.push(TrackPart {
            id,
            vertices,
            detector,
            start_offset,
        });
    }

    SimulationAssets {
        car_collision_vertices,
        car_mass_offset,
        track_parts,
    }
}

/// Cursor-based little-endian binary reader used only during asset loading.
struct Reader<'a> {
    cur: Cursor<&'a [u8]>,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            cur: Cursor::new(data),
        }
    }

    fn u8(&mut self) -> u8 {
        let mut b = [0u8; 1];
        self.cur
            .read_exact(&mut b)
            .expect("unexpected EOF reading u8");
        b[0]
    }

    fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.cur
            .read_exact(&mut b)
            .expect("unexpected EOF reading u32");
        u32::from_le_bytes(b)
    }

    fn i32(&mut self) -> i32 {
        let mut b = [0u8; 4];
        self.cur
            .read_exact(&mut b)
            .expect("unexpected EOF reading i32");
        i32::from_le_bytes(b)
    }

    fn f32(&mut self) -> f32 {
        let mut b = [0u8; 4];
        self.cur
            .read_exact(&mut b)
            .expect("unexpected EOF reading f32");
        f32::from_le_bytes(b)
    }

    /// Read three consecutive `f32` values as an XYZ array.
    fn vec3(&mut self) -> [f32; 3] {
        [self.f32(), self.f32(), self.f32()]
    }

    /// Read a length-prefixed `f32` array (`u32` count then `count` × `f32`).
    fn vec_f32(&mut self) -> Vec<f32> {
        let len = self.u32() as usize;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.f32());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fluent builder for constructing well-formed asset binary blobs in tests.
    struct Bin(Vec<u8>);

    impl Bin {
        fn new() -> Self {
            Self(Vec::new())
        }

        fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }

        fn u32_le(mut self, v: u32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn i32_le(mut self, v: i32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn f32_le(mut self, v: f32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }

        /// Write a length-prefixed `f32` array.
        fn f32_vec(mut self, vals: &[f32]) -> Self {
            self = self.u32_le(vals.len() as u32);
            for &v in vals {
                self = self.f32_le(v);
            }
            self
        }

        /// Write three consecutive `f32` values.
        fn xyz(self, x: f32, y: f32, z: f32) -> Self {
            self.f32_le(x).f32_le(y).f32_le(z)
        }

        fn build(self) -> Vec<u8> {
            self.0
        }
    }

    #[test]
    fn minimal_no_parts() {
        let data = Bin::new()
            .f32_vec(&[1.0, 2.0, 3.0])
            .f32_le(0.5)
            .u32_le(0)
            .build();

        let a = load_assets(&data);
        assert_eq!(a.car_collision_vertices, [1.0, 2.0, 3.0]);
        assert!((a.car_mass_offset - 0.5).abs() < 1e-6);
        assert!(a.track_parts.is_empty());
    }

    #[test]
    fn part_no_detector_no_start() {
        let data = Bin::new()
            .f32_vec(&[])
            .f32_le(1.0)
            .u32_le(1)
            .u32_le(42)
            .f32_vec(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .u8(0)
            .u8(0)
            .build();

        let a = load_assets(&data);
        let p = &a.track_parts[0];
        assert_eq!(p.id, 42);
        assert_eq!(p.vertices.len(), 6);
        assert!((p.vertices[0] - 0.1).abs() < 1e-6);
        assert!(p.detector.is_none());
        assert!(p.start_offset.is_none());
    }

    #[test]
    fn part_with_detector() {
        let data = Bin::new()
            .f32_vec(&[])
            .f32_le(0.0)
            .u32_le(1)
            .u32_le(7)
            .f32_vec(&[])
            .u8(1)
            .i32_le(2)
            .xyz(1.0, 2.0, 3.0)
            .xyz(4.0, 5.0, 6.0)
            .u8(0)
            .build();

        let d = load_assets(&data);
        let d = d.track_parts[0].detector.as_ref().unwrap();
        assert_eq!(d.detector_type, 2);
        assert_eq!(d.center, [1.0, 2.0, 3.0]);
        assert_eq!(d.size, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn part_with_start_offset() {
        let data = Bin::new()
            .f32_vec(&[])
            .f32_le(0.0)
            .u32_le(1)
            .u32_le(3)
            .f32_vec(&[])
            .u8(0)
            .u8(1)
            .xyz(10.0, 20.0, 30.0)
            .build();

        assert_eq!(
            load_assets(&data).track_parts[0].start_offset.unwrap(),
            [10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn part_with_detector_and_start_offset() {
        let data = Bin::new()
            .f32_vec(&[])
            .f32_le(0.0)
            .u32_le(1)
            .u32_le(99)
            .f32_vec(&[1.0, 2.0])
            .u8(1)
            .i32_le(-1)
            .xyz(0.0, 0.0, 0.0)
            .xyz(1.0, 1.0, 1.0)
            .u8(1)
            .xyz(5.0, 6.0, 7.0)
            .build();

        let p = &load_assets(&data).track_parts[0];
        assert!(p.detector.is_some());
        assert_eq!(p.start_offset.unwrap(), [5.0, 6.0, 7.0]);
    }

    #[test]
    fn multiple_parts_preserve_order() {
        let data = Bin::new()
            .f32_vec(&[])
            .f32_le(0.0)
            .u32_le(3)
            .u32_le(1)
            .f32_vec(&[])
            .u8(0)
            .u8(0)
            .u32_le(2)
            .f32_vec(&[9.0])
            .u8(0)
            .u8(0)
            .u32_le(3)
            .f32_vec(&[])
            .u8(0)
            .u8(0)
            .build();

        let a = load_assets(&data);
        assert_eq!(a.track_parts.len(), 3);
        assert_eq!(a.track_parts[0].id, 1);
        assert_eq!(a.track_parts[1].id, 2);
        assert_eq!(a.track_parts[1].vertices.len(), 1);
        assert_eq!(a.track_parts[2].id, 3);
    }

    #[test]
    fn detector_default_is_disabled() {
        let d = Detector::default();
        assert_eq!(d.detector_type, -1);
        assert_eq!(d.center, [0.0; 3]);
        assert_eq!(d.size, [0.0; 3]);
    }

    #[test]
    #[should_panic(expected = "unexpected EOF")]
    fn truncated_data_panics_with_message() {
        let data = Bin::new()
            .f32_vec(&[])
            .f32_le(0.0)
            .u32_le(1) // claims 1 part, provides nothing
            .build();
        load_assets(&data);
    }
}
