//! Shared constants, lookup tables, and primitive types used across the
//! simulation module.

use parry3d::glamx::Quat;
use polytrack_codes::v6::Direction;

/// World-space size of one track block, in metres.
pub const PART_SIZE: f32 = 5.0;

/// Part IDs that represent finish-line pieces.  Used when determining the
/// target position for a car that has completed all checkpoints.
pub const FINISH_LINE_IDS: [u8; 4] = [74, 6, 78, 76];

/// Pre-computed rotation quaternions indexed by `[direction][rotation]`.
///
/// Stored as a `const` - zero runtime construction cost.  Values are the
/// exact floating-point literals used by the original game; do **not** round
/// or normalise them, as that would break determinism.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
pub const FACE_ROTATION_QUATS: [[Quat; 4]; 6] = [
    [
        Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
        Quat::from_xyzw(0.0, 0.7071067811865475, 0.0, 0.7071067811865476),
        Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
        Quat::from_xyzw(0.0, 0.7071067811865476, 0.0, -0.7071067811865475),
    ],
    [
        Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
        Quat::from_xyzw(0.7071067811865475, 0.0, 0.7071067811865476, 0.0),
        Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(0.7071067811865476, 0.0, -0.7071067811865475, 0.0),
    ],
    [
        Quat::from_xyzw(0.0, 0.0, -0.7071067811865477, 0.7071067811865475),
        Quat::from_xyzw(0.5, 0.5, -0.5, 0.5),
        Quat::from_xyzw(0.7071067811865475, 0.7071067811865477, 0.0, 0.0),
        Quat::from_xyzw(0.5, 0.5, 0.5, -0.5),
    ],
    [
        Quat::from_xyzw(0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
        Quat::from_xyzw(0.5, -0.5, 0.5, 0.5),
        Quat::from_xyzw(0.7071067811865476, -0.7071067811865475, 0.0, 0.0),
        Quat::from_xyzw(0.5, -0.5, -0.5, -0.5),
    ],
    [
        Quat::from_xyzw(0.7071067811865475, 0.0, 0.0, 0.7071067811865476),
        Quat::from_xyzw(0.5, 0.5, 0.5, 0.5),
        Quat::from_xyzw(0.0, 0.7071067811865476, 0.7071067811865475, 0.0),
        Quat::from_xyzw(-0.5, 0.5, 0.5, -0.5),
    ],
    [
        Quat::from_xyzw(-0.7071067811865477, 0.0, 0.0, 0.7071067811865475),
        Quat::from_xyzw(-0.5, -0.5, 0.5, 0.5),
        Quat::from_xyzw(0.0, -0.7071067811865475, 0.7071067811865477, 0.0),
        Quat::from_xyzw(0.5, -0.5, 0.5, -0.5),
    ],
];

/// Return the rotation quaternion for a block with the given `dir` and
/// `rotation` index.  Panics in debug if indices are out of range; the game
/// never produces out-of-range values from valid track data.
#[inline]
pub fn face_rotation(dir: Direction, rotation: u8) -> Quat {
    FACE_ROTATION_QUATS[dir as usize][rotation as usize]
}

/// World-space spawn pose for a car at the start of a race.
#[derive(Debug, Clone)]
pub struct StartTransform {
    /// World-space XYZ position.
    pub position: [f32; 3],
    /// Orientation quaternion in `[x, y, z, w]` order.
    pub quaternion: [f32; 4],
}

/// Snapshot of player (or AI) input for one physics tick.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerController {
    /// Accelerate / throttle.
    pub up: bool,
    /// Steer right.
    pub right: bool,
    /// Brake / reverse.
    pub down: bool,
    /// Steer left.
    pub left: bool,
    /// Instant respawn at last checkpoint.
    pub reset: bool,
}
