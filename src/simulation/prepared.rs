//! Pre-computed, WASM-independent track data.
//!
//! Build a [`PreparedTrack`] once from a track export string, then hand it to
//! as many [`SimulationWorker`]s as needed - e.g. one per AI agent, or one for
//! replay alongside one for live simulation.  All expensive one-time work
//! (track decoding, mountain mesh generation, checkpoint map construction) is
//! done here, not in the worker.
//!
//! [`SimulationWorker`]: super::worker::SimulationWorker

use std::collections::HashMap;

use anyhow::anyhow;
use polytrack_codes::v6::{Block, TrackInfo};

use super::{
    track::{MountainMesh, build_mountain_mesh, decode_track, pack_track_data},
    types::{FINISH_LINE_IDS, PART_SIZE, StartTransform, face_rotation},
};
use crate::data_reader::{TrackPart, assets};

/// All data derived from a track export string that can be computed without a
/// WASM instance.
///
/// Construct with [`PreparedTrack::from_export_string`] and pass to
/// [`SimulationWorker::new`].
///
/// [`SimulationWorker::new`]: super::worker::SimulationWorker::new
#[derive(Debug, Clone)]
pub struct PreparedTrack {
    /// Binary track blob in the 19-bytes-per-block format expected by WASM.
    pub(super) track_bytes: Vec<u8>,
    /// Procedurally generated mountain collision mesh.
    pub(super) mountain: MountainMesh,
    /// Car spawn pose at the start of the race.
    pub(super) start: StartTransform,
    /// Highest checkpoint index present on the track.
    /// `0` means the track has no numbered checkpoints.
    pub(super) max_checkpoint: u16,
    /// O(1) direct-index lookup: checkpoint index → world-space block centre.
    /// Built once; used every physics tick to resolve `next_checkpoint_position`.
    ///
    /// Checkpoint indices are dense small integers (`0..=max_checkpoint`), so
    /// a `Vec` indexed directly by checkpoint number avoids hashing on the
    /// hottest per-frame lookup in the crate. `None` means no block declared
    /// that checkpoint index (shouldn't happen for a valid track, but kept
    /// optional to mirror the old map's fallibility).
    pub(super) checkpoint_positions: Vec<Option<[f32; 3]>>,
    /// World-space positions of every finish-line block.
    /// Used to find the nearest finish-line block when `is_finishline_cp` is true.
    pub(super) finish_positions: Vec<[f32; 3]>,
}

impl PreparedTrack {
    /// Decode and prepare all static track data from a PolyTrack v6 export
    /// string.
    ///
    /// # Errors
    /// Returns an error if the export string is invalid or the track has no
    /// start position.
    pub fn from_export_string(export_string: &str) -> anyhow::Result<Self> {
        let track_info = decode_track(export_string)?;

        let max_checkpoint = track_info
            .parts
            .iter()
            .flat_map(|p| p.blocks.iter())
            .filter_map(|b| b.cp_order)
            .max()
            .unwrap_or(0);

        // Index assets by part ID once; shared between find_start_block
        // and the checkpoint position builder below.
        let asset_data = assets();
        let asset_by_id: HashMap<u8, &TrackPart> = asset_data
            .track_parts
            .iter()
            .map(|p| (p.id as u8, p))
            .collect();

        let start_block = find_start_block(&track_info, &asset_by_id)?;
        let start = calculate_start_transform(&start_block, &track_info);
        let track_bytes = pack_track_data(&track_info);
        let mountain = build_mountain_mesh(&track_info);

        // Build the checkpoint position table (computed once, used every
        // frame). The first block encountered for each checkpoint index wins,
        // matching the original game's `next()` iterator behaviour.
        let mut checkpoint_positions = vec![None; max_checkpoint as usize + 1];
        for part in &track_info.parts {
            for block in &part.blocks {
                if let Some(cp) = block.cp_order {
                    let slot = &mut checkpoint_positions[cp as usize];
                    if slot.is_none() {
                        *slot = Some(block_world_pos(block, &track_info));
                    }
                }
            }
        }

        // Collect world-space positions of every finish-line block.
        let finish_positions: Vec<[f32; 3]> = track_info
            .parts
            .iter()
            .filter(|p| FINISH_LINE_IDS.contains(&p.id))
            .flat_map(|p| p.blocks.iter())
            .map(|b| block_world_pos(b, &track_info))
            .collect();

        Ok(Self {
            track_bytes,
            mountain,
            start,
            max_checkpoint,
            checkpoint_positions,
            finish_positions,
        })
    }

    /// Returns the highest checkpoint index present on the track (0 = no checkpoints).
    #[must_use]
    pub fn max_checkpoint(&self) -> u16 {
        self.max_checkpoint
    }

    /// Returns the number of finish-line blocks on the track.
    #[must_use]
    pub fn finish_block_count(&self) -> usize {
        self.finish_positions.len()
    }

    /// Returns the car's world-space spawn position.
    #[must_use]
    pub fn start_position(&self) -> [f32; 3] {
        self.start.position
    }
}

/// A block that carries a spawn offset, used to position a car at the start.
struct StartBlock<'a> {
    block: &'a Block,
    /// Local-space offset from the block origin to the spawn point.
    start_offset: [f32; 3],
}

/// Find the authoritative start block by selecting the block with the highest
/// `start_order` among all parts that have a `start_offset` in the asset
/// database.
///
/// Parts whose ID is not present in `asset_by_id` are silently skipped -
/// they cannot be start parts, and this matches the original game's behaviour
/// where an unknown part just fails the `.find()` lookup and `continue`s.
///
/// # Errors
/// Returns an error if a start-part block is missing its `start_order` field,
/// or if the track has no start position at all.
fn find_start_block<'a>(
    track_info: &'a TrackInfo,
    asset_by_id: &HashMap<u8, &TrackPart>,
) -> anyhow::Result<StartBlock<'a>> {
    let mut best: Option<(u32, StartBlock)> = None;

    for part in &track_info.parts {
        // Unknown part IDs are silently skipped - they cannot be start parts.
        let Some(part_data) = asset_by_id.get(&part.id) else {
            continue;
        };
        let Some(start_offset) = part_data.start_offset else {
            continue;
        };

        for block in &part.blocks {
            let start_order = block
                .start_order
                .ok_or_else(|| anyhow!("start part block is missing start_order"))?;

            // `>=` keeps the last-seen highest order, matching the original.
            if best
                .as_ref()
                .is_none_or(|(best_order, _)| start_order >= *best_order)
            {
                best = Some((
                    start_order,
                    StartBlock {
                        block,
                        start_offset,
                    },
                ));
            }
        }
    }

    best.map(|(_, sb)| sb)
        .ok_or_else(|| anyhow!("track has no start position"))
}

/// Compute the world-space spawn pose from a start block.
///
/// Replicates the original game's transform exactly:
/// 1. Look up the block's face rotation quaternion.
/// 2. Apply a 180° Y-axis flip (cars face away from the starting direction).
/// 3. Rotate the local spawn offset into world space.
/// 4. Add the block's world-space origin.
fn calculate_start_transform(start: &StartBlock, track_info: &TrackInfo) -> StartTransform {
    let block_quat = face_rotation(start.block.dir, start.block.rotation);

    // The simulation worker replaces Math.sin/cos with its 1-degree lookup
    // table before TrackData::getStartTransform runs.  For Euler(0, PI, 0),
    // Three.js therefore produces [0, 1, 0, table[180]], not glam's f32
    // approximation. Keep every intermediate as a JS Number (f64) and only
    // narrow at the WASM call boundary.
    const SIN_PI: f64 = 1.224_646_799_147_353_2e-16;
    let y_flip = [0.0, 1.0, 0.0, SIN_PI];
    let quaternion = multiply_quaternions(block_quat, y_flip);
    let offset = apply_quaternion(
        [
            start.start_offset[0] as f64,
            start.start_offset[1] as f64,
            start.start_offset[2] as f64,
        ],
        quaternion,
    );

    let position = [
        ((start.block.x as i32 + track_info.min_x) as f64 * PART_SIZE as f64 + offset[0]) as f32,
        ((start.block.y as i32 + track_info.min_y) as f64 * PART_SIZE as f64 + offset[1]) as f32,
        ((start.block.z as i32 + track_info.min_z) as f64 * PART_SIZE as f64 + offset[2]) as f32,
    ];

    StartTransform {
        position,
        quaternion: quaternion.map(|v| v as f32),
    }
}

/// Three.js' `Quaternion.multiplyQuaternions`, preserving its operation order.
#[inline]
fn multiply_quaternions(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [x, y, z, w] = a;
    let [bx, by, bz, bw] = b;
    [
        x * bw + w * bx + y * bz - z * by,
        y * bw + w * by + z * bx - x * bz,
        z * bw + w * bz + x * by - y * bx,
        w * bw - x * bx - y * by - z * bz,
    ]
}

/// Three.js' `Vector3.applyQuaternion`, preserving its operation order.
#[inline]
fn apply_quaternion(v: [f64; 3], q: [f64; 4]) -> [f64; 3] {
    let [x, y, z] = v;
    let [qx, qy, qz, qw] = q;
    let tx = 2.0 * (qy * z - qz * y);
    let ty = 2.0 * (qz * x - qx * z);
    let tz = 2.0 * (qx * y - qy * x);
    [
        x + qw * tx + qy * tz - qz * ty,
        y + qw * ty + qz * tx - qx * tz,
        z + qw * tz + qx * ty - qy * tx,
    ]
}

/// Returns the world-space XYZ centre of `block` on `track_info`.
///
/// World position = `(local + min) * PART_SIZE`.
#[inline]
pub(crate) fn block_world_pos(block: &Block, track_info: &TrackInfo) -> [f32; 3] {
    [
        (block.x as i32 + track_info.min_x) as f32 * PART_SIZE,
        (block.y as i32 + track_info.min_y) as f32 * PART_SIZE,
        (block.z as i32 + track_info.min_z) as f32 * PART_SIZE,
    ]
}

/// Returns the position from `candidates` closest to `origin`, using squared
/// distance (no `sqrt` required).
///
/// Returns `None` if `candidates` is empty.
pub(crate) fn nearest_position(origin: &[f32; 3], candidates: &[[f32; 3]]) -> Option<[f32; 3]> {
    candidates
        .iter()
        .map(|p| {
            let dx = p[0] - origin[0];
            let dy = p[1] - origin[1];
            let dz = p[2] - origin[2];
            (dx * dx + dy * dy + dz * dz, *p)
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .map(|(_, p)| p)
}

#[cfg(test)]
mod tests {
    use polytrack_codes::v6::{Block, Direction, Environment, TrackInfo};

    use super::*;

    fn blk(x: u32, y: u32, z: u32) -> Block {
        Block {
            x,
            y,
            z,
            dir: Direction::YPos,
            rotation: 0,
            cp_order: None,
            start_order: None,
            color: 0,
        }
    }

    fn ti(min_x: i32, min_y: i32, min_z: i32) -> TrackInfo {
        TrackInfo {
            min_x,
            min_y,
            min_z,
            parts: vec![],
            env: Environment::Summer,
            sun_dir: 0,
            data_bytes: 0,
        }
    }

    #[test]
    fn world_pos_zero_min() {
        assert_eq!(
            block_world_pos(&blk(2, 3, 4), &ti(0, 0, 0)),
            [2.0 * PART_SIZE, 3.0 * PART_SIZE, 4.0 * PART_SIZE]
        );
    }

    #[test]
    fn world_pos_positive_min() {
        assert_eq!(
            block_world_pos(&blk(1, 1, 1), &ti(10, 20, 30)),
            [11.0 * PART_SIZE, 21.0 * PART_SIZE, 31.0 * PART_SIZE]
        );
    }

    #[test]
    fn world_pos_negative_min() {
        assert_eq!(
            block_world_pos(&blk(0, 0, 0), &ti(-5, -10, -3)),
            [-5.0 * PART_SIZE, -10.0 * PART_SIZE, -3.0 * PART_SIZE]
        );
    }

    #[test]
    fn world_pos_cancels_to_zero() {
        assert_eq!(
            block_world_pos(&blk(5, 10, 3), &ti(-5, -10, -3)),
            [0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn worker_y_flip_matches_javascript_float_bits() {
        // Direction XPositive, rotation 0. The tiny Z/W components are the
        // clearest regression guard: a native f32 Euler conversion produces
        // values around 1e-8 instead of the worker table's values around 1e-16.
        let q = multiply_quaternions(
            face_rotation(Direction::XPos, 0),
            [0.0, 1.0, 0.0, 1.224_646_799_147_353_2e-16],
        );
        let bits = q.map(|v| (v as f32).to_bits());
        assert_eq!(bits, [0x3f35_04f3, 0x3f35_04f3, 0xa4c7_ad06, 0x24c7_ad06]);
    }

    #[test]
    fn nearest_empty_is_none() {
        assert!(nearest_position(&[0.0; 3], &[]).is_none());
    }

    #[test]
    fn nearest_single_candidate() {
        assert_eq!(
            nearest_position(&[0.0; 3], &[[3.0, 4.0, 0.0]]),
            Some([3.0, 4.0, 0.0])
        );
    }

    #[test]
    fn nearest_picks_closest_of_three() {
        let candidates = [[100.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0]];
        assert_eq!(
            nearest_position(&[0.0; 3], &candidates),
            Some([1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn nearest_exact_match_wins() {
        let candidates = [[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]];
        assert_eq!(
            nearest_position(&[1.0, 1.0, 1.0], &candidates),
            Some([1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn nearest_3d_distance() {
        // (3,4,0) → dist=5;  (1,1,1) → dist≈1.73
        let candidates = [[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]];
        assert_eq!(
            nearest_position(&[0.0; 3], &candidates),
            Some([1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn nearest_negative_coords() {
        let candidates = [[-1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        assert_eq!(
            nearest_position(&[0.0; 3], &candidates),
            Some([-1.0, 0.0, 0.0])
        );
    }
}
