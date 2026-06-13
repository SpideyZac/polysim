//! Track decoding, binary packing, and procedural mountain-mesh generation.
//!
//! # Mountain mesh
//! The background mountain ring is purely cosmetic in the original game but
//! the WASM physics module also uses it as a collision boundary, so it must be
//! generated identically.  Generation is driven by a deterministic table-based
//! RNG ([`TableRng`]) so that the output is identical across platforms.

use std::f32::consts::{PI, SQRT_2};

use anyhow::anyhow;
use polytrack_codes::v6::{TrackInfo, decode_track_code, decode_track_data};

use super::types::PART_SIZE;

/// Vertical scale applied to normalised RNG heights when building the mountain.
const HEIGHT_SCALE: f32 = 100.0;

/// Radial distance between successive mountain rings, in world units.
const MOUNTAIN_RING_STEP: f32 = 100.0;

/// Number of angular segments per mountain ring.
const MOUNTAIN_RING_SEGS: usize = 8;

/// Minimum mountain radius regardless of track size, in world units.
const MOUNTAIN_MIN_RADIUS: f32 = 200.0;

/// Base radius added to the scaled track half-diagonal.
const MOUNTAIN_RADIUS_BASE: f32 = 160.0;

/// Tracks whose required mountain radius exceeds this value receive no mountain.
/// Avoids generating a mesh too large to fit in WASM memory.
const MOUNTAIN_MAX_RADIUS: f32 = 4500.0;

/// Procedurally generated background mountain mesh.
#[derive(Debug, Clone)]
pub struct MountainMesh {
    /// Flat XYZ vertex buffer for an unindexed triangle soup.
    /// Length is always a multiple of 9 (3 vertices × 3 floats per triangle).
    /// Empty when the track is too large for a mountain ([`MOUNTAIN_MAX_RADIUS`]).
    pub vertices: Vec<f32>,
    /// World-space translation applied to all vertices when passed to the WASM
    /// module.  Equals the track centre in XZ, Y = 0.
    pub offset: [f32; 3],
}

/// Decode a track export string into a [`TrackInfo`].
///
/// # Errors
/// Returns an error if the string is not a valid PolyTrack v6 export code or
/// if the embedded track data cannot be decoded.
pub fn decode_track(export_string: &str) -> anyhow::Result<TrackInfo> {
    let track =
        decode_track_code(export_string).ok_or_else(|| anyhow!("failed to decode track code"))?;
    decode_track_data(&track.track_data).ok_or_else(|| anyhow!("failed to decode track data"))
}

/// Serialise track data into the compact binary format consumed by the WASM
/// physics module.
///
/// # Wire format (19 bytes per block, little-endian)
///
/// | Offset | Size | Field                                    |
/// |--------|------|------------------------------------------|
/// | 0      | 1    | part ID (`u8`)                           |
/// | 1      | 4    | world X (`i32`)                          |
/// | 5      | 4    | world Y (`i32`)                          |
/// | 9      | 4    | world Z (`i32`)                          |
/// | 13     | 1    | rotation index (`u8`)                    |
/// | 14     | 1    | direction index (`u8`)                   |
/// | 15     | 4    | checkpoint order (`i32`, −1 if absent)   |
///
/// World coordinates are computed as `block.{x,y,z} + track_info.min_{x,y,z}`.
pub fn pack_track_data(track_info: &TrackInfo) -> Vec<u8> {
    let block_count: usize = track_info.parts.iter().map(|p| p.blocks.len()).sum();
    let mut buf = vec![0u8; 19 * block_count];
    let mut off = 0;

    for part in &track_info.parts {
        for block in &part.blocks {
            buf[off] = part.id;
            off += 1;

            let wx = block.x as i32 + track_info.min_x;
            let wy = block.y as i32 + track_info.min_y;
            let wz = block.z as i32 + track_info.min_z;

            buf[off..off + 4].copy_from_slice(&wx.to_le_bytes());
            off += 4;
            buf[off..off + 4].copy_from_slice(&wy.to_le_bytes());
            off += 4;
            buf[off..off + 4].copy_from_slice(&wz.to_le_bytes());
            off += 4;

            buf[off] = block.rotation;
            off += 1;
            buf[off] = block.dir as u8;
            off += 1;

            let cp = block.cp_order.map(|x| x as i32).unwrap_or(-1);
            buf[off..off + 4].copy_from_slice(&cp.to_le_bytes());
            off += 4;
        }
    }

    buf
}

/// Build the procedural mountain collision/background mesh for `track_info`.
///
/// Returns a [`MountainMesh`] with an empty vertex buffer when the computed
/// radius exceeds [`MOUNTAIN_MAX_RADIUS`]; in that case the WASM module
/// receives `ptr = 0, len = 0`.
pub fn build_mountain_mesh(track_info: &TrackInfo) -> MountainMesh {
    let (max_x, max_z) = track_bounds(track_info);
    let radius = mountain_radius(track_info, max_x, max_z);

    if radius > MOUNTAIN_MAX_RADIUS {
        return MountainMesh {
            vertices: vec![],
            offset: [0.0; 3],
        };
    }

    let center = mountain_center(track_info, max_x, max_z);
    let ring_count = (radius / 10.0).floor() as usize;
    let rings = generate_rings(ring_count);
    let vertices = triangulate_rings(&rings, radius);

    MountainMesh {
        vertices,
        offset: [center[0], 0.0, center[1]],
    }
}

/// Compute the axis-aligned max block coordinates in world space.
fn track_bounds(track_info: &TrackInfo) -> (i32, i32) {
    let mut max_x = i32::MIN;
    let mut max_z = i32::MIN;
    for part in &track_info.parts {
        for block in &part.blocks {
            max_x = max_x.max(block.x as i32 + track_info.min_x);
            max_z = max_z.max(block.z as i32 + track_info.min_z);
        }
    }
    (max_x, max_z)
}

/// Compute the mountain radius from the track's bounding box.
fn mountain_radius(track_info: &TrackInfo, max_x: i32, max_z: i32) -> f32 {
    let width = (max_x - track_info.min_x).abs() as f32 * PART_SIZE / 2.0;
    let height = (max_z - track_info.min_z).abs() as f32 * PART_SIZE / 2.0;
    f32::max(
        MOUNTAIN_MIN_RADIUS,
        MOUNTAIN_RADIUS_BASE + f32::max(width, height) * SQRT_2,
    )
}

/// Compute the world-space XZ centre of the track's bounding box.
fn mountain_center(track_info: &TrackInfo, max_x: i32, max_z: i32) -> [f32; 2] {
    [
        (track_info.min_x as f32 + (max_x - track_info.min_x) as f32 / 2.0) * PART_SIZE,
        (track_info.min_z as f32 + (max_z - track_info.min_z) as f32 / 2.0) * PART_SIZE,
    ]
}

/// Generate normalised height values for each ring and segment using the
/// deterministic [`TableRng`].
///
/// Segments 0 and 7 are always 0 (ground level) to create natural openings.
/// Segment 1 has a 50 % chance of being 0, consuming one RNG value regardless.
fn generate_rings(ring_count: usize) -> Vec<Vec<f32>> {
    let mut rng = TableRng::default();
    (0..ring_count)
        .map(|_| {
            (0..MOUNTAIN_RING_SEGS)
                .map(|n| {
                    if n == 0 || n == 7 || (n == 1 && rng.next() < 0.5) {
                        0.0
                    } else {
                        rng.next()
                    }
                })
                .collect()
        })
        .collect()
}

/// Convert ring height data into an unindexed triangle-soup vertex buffer.
///
/// Each ring-segment quad is split into two triangles.  The vertex buffer is
/// pre-allocated to the exact required size to avoid reallocations.
fn triangulate_rings(rings: &[Vec<f32>], radius: f32) -> Vec<f32> {
    let ring_count = rings.len();
    // Each ring × each segment gap produces 2 triangles × 3 vertices × 3 floats.
    let mut verts = Vec::with_capacity(ring_count * (MOUNTAIN_RING_SEGS - 1) * 18);

    for e in 0..ring_count {
        let t = (e as f32 / ring_count as f32) * PI * 2.0;
        let i = ((e + 1) as f32 / ring_count as f32) * PI * 2.0;

        let current = &rings[e];
        let next = if e + 1 < ring_count {
            &rings[e + 1]
        } else {
            &rings[0]
        };

        for seg in 0..current.len() - 1 {
            let inner = radius + MOUNTAIN_RING_STEP * seg as f32;
            let outer = radius + MOUNTAIN_RING_STEP * (seg + 1) as f32;

            // Triangle 1: inner-current, inner-next, outer-next
            verts.extend_from_slice(&[
                t.cos() * inner,
                current[seg] * HEIGHT_SCALE,
                t.sin() * inner,
                i.cos() * inner,
                next[seg] * HEIGHT_SCALE,
                i.sin() * inner,
                i.cos() * outer,
                next[seg + 1] * HEIGHT_SCALE,
                i.sin() * outer,
            ]);
            // Triangle 2: inner-current, outer-next, outer-current
            verts.extend_from_slice(&[
                t.cos() * inner,
                current[seg] * HEIGHT_SCALE,
                t.sin() * inner,
                i.cos() * outer,
                next[seg + 1] * HEIGHT_SCALE,
                i.sin() * outer,
                t.cos() * outer,
                current[seg + 1] * HEIGHT_SCALE,
                t.sin() * outer,
            ]);
        }
    }

    verts
}

/// Deterministic table-based pseudo-RNG used exclusively for mountain
/// generation.
///
/// Advances through [`RNG_TABLE`] circularly.  Starts at index 0; the first
/// call to [`next`] returns `RNG_TABLE[1]`, matching the original game's
/// `TableRng::new(None)` behaviour where seed `None` sets `index = 0`.
///
/// [`next`]: TableRng::next
#[derive(Default)]
struct TableRng {
    index: usize,
}

impl TableRng {
    /// Advance the index and return the next value from the table.
    #[inline]
    fn next(&mut self) -> f32 {
        self.index = (self.index + 1) % RNG_TABLE.len();
        RNG_TABLE[self.index]
    }
}

/// Pre-computed random values for mountain ring generation.
///
/// Values are fixed at compile time to guarantee identical mountain geometry
/// across all platforms and runs.
#[allow(clippy::excessive_precision)]
const RNG_TABLE: &[f32] = &[
    0.12047764760664692,
    0.19645762332790628,
    0.5525629082262744,
    0.41272626379209965,
    0.7795036003541387,
    0.13367266027110114,
    0.7999601557377349,
    0.9519714253374205,
    0.1735048382917752,
    0.7513367084489158,
    0.6531386724839523,
    0.9026427867068505,
    0.8543272738216994,
    0.11176849958868162,
    0.6705698284858437,
    0.26628732081296946,
    0.31140322993719605,
    0.45170300835470933,
    0.12615515120247944,
    0.0610638094525735,
    0.291990923385425,
    0.4613983868623317,
    0.6615759832726253,
    0.4373182881232056,
    0.7432890501246443,
    0.39316710322388837,
    0.49444122821563297,
    0.5994296685114344,
    0.060050119050233386,
    0.4165885432422003,
    0.43974364800990084,
    0.1628314496954224,
    0.05787972729968116,
    0.225388541259955,
    0.6075775236386991,
    0.8908354370882479,
    0.47072983115144584,
    0.7662003453186828,
    0.20651036895645647,
    0.03724062137286044,
    0.17110277274376795,
    0.7626426077793496,
    0.8372112804261309,
    0.8761690804447455,
    0.13887024930406633,
    0.8287513367412203,
    0.9794446290917873,
    0.807658524448803,
    0.8465629116398186,
    0.5187285629536083,
    0.33962953580139277,
    0.9798419666114342,
    0.6777071959103609,
    0.5388899884934379,
    0.7863389168762325,
    0.4274591420924474,
    0.25631366937500566,
    0.5695289062505289,
    0.026841382754547727,
    0.18267938207996903,
    0.9853642975717878,
    0.24428485895234409,
    0.5322028747608949,
    0.9655065842019517,
    0.043810183244384016,
    0.541216190236913,
    0.05897981610006209,
    0.2849168541804703,
    0.5349823008832073,
    0.9655676144971486,
    0.22831812764497283,
    0.7698701658704175,
    0.4103995069939841,
    0.25782763124411856,
    0.8490222628872495,
    0.39280879489916987,
    0.31999467883347554,
    0.2860820872456349,
    0.9684928577493004,
    0.9973831481899462,
    0.2930912094664657,
    0.4847128131859766,
    0.7218400909709828,
    0.40407009594106236,
    0.7059298060123587,
    0.45362146566562744,
    0.4640974655488792,
    0.16076769483252273,
    0.5989453525750241,
    0.585759299589679,
    0.9417035568973537,
    0.20117930667657413,
    0.5777873180244959,
    0.1991854396549344,
    0.8743781441651348,
    0.624666386634513,
    0.38720573630932886,
    0.9967931526923675,
    0.49817894572849486,
    0.24585267823751833,
    0.8639168275132305,
    0.2865624029759799,
    0.6163605496913385,
    0.5864748073339972,
    0.8781049154377354,
    0.7497547608938613,
    0.7864098057445887,
    0.0334170452332867,
    0.4875588105294657,
    0.6737395339380896,
    0.21851121231639659,
    0.2923739650597854,
    0.6073797612662293,
    0.41823228947229896,
    0.8531029420136382,
    0.3260916332061783,
    0.6306262204574675,
    0.5268576689601923,
    0.3516570914484707,
    0.8659366375222706,
    0.8447448461834428,
    0.3794548980890986,
    0.9832775904115916,
    0.8442256760399809,
    0.3006550591973338,
    0.9718660619781394,
    0.5103245035851833,
    0.794319831388071,
];

#[cfg(test)]
mod tests {
    use polytrack_codes::v6::{Block, Direction, Environment, Part, TrackInfo};

    use super::*;

    fn block(x: u32, y: u32, z: u32, cp: Option<u16>) -> Block {
        Block {
            x,
            y,
            z,
            dir: Direction::YPos,
            rotation: 0,
            cp_order: cp,
            start_order: None,
            color: 0,
        }
    }

    fn track(min_x: i32, min_y: i32, min_z: i32, parts: Vec<Part>) -> TrackInfo {
        TrackInfo {
            min_x,
            min_y,
            min_z,
            parts,
            env: Environment::Summer,
            sun_dir: 0,
            data_bytes: 0,
        }
    }

    #[test]
    fn pack_empty_track() {
        assert!(pack_track_data(&track(0, 0, 0, vec![])).is_empty());
    }

    #[test]
    fn pack_single_block_byte_layout() {
        // part id=7, block local (3,5,2), min=(10,20,30), rotation=1, no cp.
        let b = Block {
            x: 3,
            y: 5,
            z: 2,
            dir: Direction::YPos,
            rotation: 1,
            cp_order: None,
            start_order: None,
            color: 0,
        };
        let buf = pack_track_data(&track(
            10,
            20,
            30,
            vec![Part {
                id: 7,
                blocks: vec![b],
                amount: 1,
            }],
        ));

        assert_eq!(buf.len(), 19);
        assert_eq!(buf[0], 7); // part id
        assert_eq!(i32::from_le_bytes(buf[1..5].try_into().unwrap()), 13); // wx
        assert_eq!(i32::from_le_bytes(buf[5..9].try_into().unwrap()), 25); // wy
        assert_eq!(i32::from_le_bytes(buf[9..13].try_into().unwrap()), 32); // wz
        assert_eq!(buf[13], 1); // rotation
        assert_eq!(buf[14], Direction::YPos as u8); // direction
        assert_eq!(i32::from_le_bytes(buf[15..19].try_into().unwrap()), -1); // no cp
    }

    #[test]
    fn pack_block_with_checkpoint() {
        let b = block(0, 0, 0, Some(3));
        let buf = pack_track_data(&track(
            0,
            0,
            0,
            vec![Part {
                id: 1,
                blocks: vec![b],
                amount: 1,
            }],
        ));
        assert_eq!(i32::from_le_bytes(buf[15..19].try_into().unwrap()), 3);
    }

    #[test]
    fn pack_negative_world_coords() {
        let b = block(0, 0, 0, None);
        let buf = pack_track_data(&track(
            -5,
            -10,
            -15,
            vec![Part {
                id: 2,
                blocks: vec![b],
                amount: 1,
            }],
        ));
        assert_eq!(i32::from_le_bytes(buf[1..5].try_into().unwrap()), -5);
        assert_eq!(i32::from_le_bytes(buf[5..9].try_into().unwrap()), -10);
        assert_eq!(i32::from_le_bytes(buf[9..13].try_into().unwrap()), -15);
    }

    #[test]
    fn pack_total_length_two_parts() {
        let p0 = Part {
            id: 1,
            blocks: vec![block(0, 0, 0, None), block(1, 0, 0, None)],
            amount: 1,
        };
        let p1 = Part {
            id: 2,
            blocks: vec![block(2, 0, 0, Some(1))],
            amount: 1,
        };
        assert_eq!(pack_track_data(&track(0, 0, 0, vec![p0, p1])).len(), 19 * 3);
    }

    #[test]
    fn rng_first_value_is_table_index_1() {
        // Default index=0; first next() increments to 1.
        let mut rng = TableRng::default();
        assert_eq!(rng.next(), RNG_TABLE[1]);
    }

    #[test]
    fn rng_deterministic_across_instances() {
        let mut a = TableRng::default();
        let mut b = TableRng::default();
        for _ in 0..200 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn rng_wraps_without_panic() {
        let mut rng = TableRng::default();
        for _ in 0..RNG_TABLE.len() * 3 {
            let v = rng.next();
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn mountain_vertices_multiple_of_9() {
        let t = track(
            0,
            0,
            0,
            vec![Part {
                id: 1,
                blocks: vec![block(0, 0, 0, None)],
                amount: 1,
            }],
        );
        let m = build_mountain_mesh(&t);
        if !m.vertices.is_empty() {
            assert_eq!(m.vertices.len() % 9, 0);
        }
    }

    #[test]
    fn generate_rings_shape() {
        let rings = generate_rings(3);
        assert_eq!(rings.len(), 3);
        for ring in &rings {
            assert_eq!(ring.len(), MOUNTAIN_RING_SEGS);
            assert_eq!(ring[0], 0.0); // segment 0 always flat
            assert_eq!(ring[7], 0.0); // segment 7 always flat
        }
    }
}
