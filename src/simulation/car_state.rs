//! Car state deserialisation from the WASM physics output buffer.
//!
//! Each call to [`SimulationWorker::update_car`] causes the WASM module to
//! write 227 bytes into a shared buffer.  The first 4 bytes are a
//! version/padding header; [`CarState::deserialize`] receives the remaining
//! 223 bytes as a `&[u8]` slice directly from WASM linear memory - no copy.
//!
//! [`SimulationWorker::update_car`]: super::worker::SimulationWorker::update_car

use anyhow::{Context, anyhow, bail};

use super::{prepared::nearest_position, types::PlayerController};

/// Contact point between one wheel and the track surface.
#[derive(Debug, Clone)]
pub struct WheelContact {
    /// World-space contact position.
    pub position: [f32; 3],
    /// Surface normal at the contact point, pointing away from the track.
    pub normal: [f32; 3],
}

/// Complete state snapshot for one car after one physics tick.
#[derive(Debug, Clone)]
pub struct CarState {
    /// Physics tick counter since the race began.
    pub frames: u32,
    /// Current speed in km/h.
    pub speed_kmh: f32,
    /// Whether the car has left the start area and the timer is running.
    pub has_started: bool,
    /// Tick at which the car crossed the finish line; `None` until then.
    pub finish_frames: Option<u32>,
    /// Index of the next checkpoint the car must pass through.
    pub next_checkpoint_index: u16,
    /// Whether the car has passed at least one checkpoint and can respawn there.
    pub has_checkpoint_to_respawn_at: bool,
    /// World-space car position.
    pub position: [f32; 3],
    /// Car orientation as a quaternion `[x, y, z, w]`.
    pub quaternion: [f32; 4],
    /// Magnitudes of collision impulses applied this tick (0–4 entries).
    /// Stored inline; no heap allocation.
    pub collision_impulses: CollisionImpulses,
    /// Per-wheel contact data; `None` if that wheel is airborne.
    pub wheel_contacts: [Option<WheelContact>; 4],
    /// Current suspension compression length for each wheel.
    pub wheel_suspension_lengths: [f32; 4],
    /// Rate of change of suspension length for each wheel.
    pub wheel_suspension_velocities: [f32; 4],
    /// Angular delta of each wheel this tick (used for rolling animation).
    pub wheel_delta_rotations: [f32; 4],
    /// Skid intensity per wheel (0 = no skid, 1 = full skid).
    pub wheel_skid_info: [f32; 4],
    /// Steering angle in radians; negative = left, positive = right.
    pub steering: f32,
    /// Whether the brake lights should be illuminated.
    pub brake_light_enabled: bool,
    /// The input controls that produced this state.
    pub controls: PlayerController,
    /// `true` when the car's next target is the finish line rather than a
    /// numbered checkpoint.
    pub is_finishline_cp: bool,
    /// World-space position of the next checkpoint or finish-line block.
    pub next_checkpoint_position: [f32; 3],
}

/// Up to 4 collision impulse magnitudes, stored inline without heap allocation.
///
/// The WASM physics module never produces more than 4 impulses per tick, so a
/// fixed-size array with a length field is sufficient and avoids the per-frame
/// `Vec` allocation that would otherwise occur.
#[derive(Debug, Clone)]
pub struct CollisionImpulses {
    data: [f32; 4],
    len: u8,
}

impl CollisionImpulses {
    /// Creates an empty impulse set.
    #[inline]
    fn new() -> Self {
        Self {
            data: [0.0; 4],
            len: 0,
        }
    }

    /// Appends one impulse value.  The caller must ensure `len < 4`.
    #[inline]
    fn push(&mut self, v: f32) {
        self.data[self.len as usize] = v;
        self.len += 1;
    }

    /// Returns the impulse values as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.len as usize]
    }

    /// Returns the number of impulses recorded this tick.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns `true` if no collision impulses occurred this tick.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Binary layout of the 223-byte payload (after the 4-byte header):
///
/// | Bytes     | Type    | Field                                              |
/// |-----------|---------|-----------------------------------------------------|
/// | 0–2       | u24 LE  | `frames`                                           |
/// | 3–6       | f32 LE  | `speed_kmh`                                        |
/// | 7         | u8      | flag byte (see below)                              |
/// | [8–10]    | u24 LE  | `finish_frames` (only if flag bit 1 set)           |
/// | next 2    | u16 LE  | `next_checkpoint_index`                            |
/// | next 12   | 3×f32   | `position`                                         |
/// | next 16   | 4×f32   | `quaternion`                                       |
/// | next 1    | u8      | impulse count (0–4)                                |
/// | next N×4  | N×f32   | impulse values                                     |
/// | next M×24 | M×6×f32 | wheel contacts (position + normal, present only)   |
/// | next 16   | 4×f32   | `wheel_suspension_lengths`                         |
/// | next 16   | 4×f32   | `wheel_suspension_velocities`                      |
/// | next 16   | 4×f32   | `wheel_delta_rotations`                            |
/// | next 16   | 4×f32   | `wheel_skid_info`                                  |
/// | next 4    | f32 LE  | `steering`                                         |
/// | next 1    | u8      | control byte (see below)                           |
///
/// **Flag byte bits:**
/// - bit 0: `has_started`
/// - bit 1: `finish_frames` present
/// - bit 2: `has_checkpoint_to_respawn_at`
/// - bits 3–6: wheel contact present (one per wheel)
///
/// **Control byte bits:**
/// - bit 0: `up`, 1: `right`, 2: `down`, 3: `left`, 4: `reset`, 5: `brake_light`
impl CarState {
    /// Deserialise a [`CarState`] from the 223-byte payload written by the WASM
    /// module.
    ///
    /// `buf` must be a slice of at least 223 bytes starting after the 4-byte
    /// header.  Reads directly from WASM linear memory - no copy.
    ///
    /// # Errors
    /// Returns an error if the buffer is too short, if the impulse count
    /// exceeds 4, or if the next checkpoint index cannot be resolved.
    pub(super) fn deserialize(
        buf: &[u8],
        max_checkpoint: u16,
        checkpoint_positions: &[Option<[f32; 3]>],
        finish_positions: &[[f32; 3]],
    ) -> anyhow::Result<Self> {
        let mut r = Cursor::new(buf);

        let frames = r.u24_le().context("frames")?;
        let speed_kmh = r.f32_le().context("speed_kmh")?;
        let flags = r.u8().context("flag byte")?;

        let has_started = flags & 0x01 != 0;
        let has_finish_frame = flags & 0x02 != 0;
        let has_checkpoint = flags & 0x04 != 0;
        let wheel_present = [
            flags & 0x08 != 0,
            flags & 0x10 != 0,
            flags & 0x20 != 0,
            flags & 0x40 != 0,
        ];

        let finish_frames = if has_finish_frame {
            Some(r.u24_le().context("finish_frames")?)
        } else {
            None
        };

        let next_checkpoint_index = r.u16_le().context("next_checkpoint_index")?;
        let position = r.vec3().context("position")?;
        let quaternion = r.vec4().context("quaternion")?;

        let impulse_count = r.u8().context("impulse count")?;
        if impulse_count > 4 {
            bail!("collision impulse count {impulse_count} exceeds maximum of 4");
        }
        let mut collision_impulses = CollisionImpulses::new();
        for i in 0..impulse_count {
            collision_impulses.push(
                r.f32_le()
                    .with_context(|| format!("collision_impulse[{i}]"))?,
            );
        }

        let mut wheel_contacts = [None, None, None, None];
        for i in 0..4 {
            if wheel_present[i] {
                wheel_contacts[i] = Some(WheelContact {
                    position: r
                        .vec3()
                        .with_context(|| format!("wheel_contact[{i}].position"))?,
                    normal: r
                        .vec3()
                        .with_context(|| format!("wheel_contact[{i}].normal"))?,
                });
            }
        }

        let wheel_suspension_lengths = r.vec4().context("wheel_suspension_lengths")?;
        let wheel_suspension_velocities = r.vec4().context("wheel_suspension_velocities")?;
        let wheel_delta_rotations = r.vec4().context("wheel_delta_rotations")?;
        let wheel_skid_info = r.vec4().context("wheel_skid_info")?;
        let steering = r.f32_le().context("steering")?;

        let control_byte = r.u8().context("control byte")?;
        let controls = PlayerController {
            up: control_byte & 0x01 != 0,
            right: control_byte & 0x02 != 0,
            down: control_byte & 0x04 != 0,
            left: control_byte & 0x08 != 0,
            reset: control_byte & 0x10 != 0,
        };
        let brake_light_enabled = control_byte & 0x20 != 0;

        // Matches the original game's exact condition.
        // `wrapping_add` handles the u16::MAX edge case without overflow.
        // When max_checkpoint == 0 the track has no checkpoints, so the car
        // always targets the finish line.
        let is_finishline_cp =
            next_checkpoint_index == max_checkpoint.wrapping_add(1) || max_checkpoint == 0;

        let next_checkpoint_position = if is_finishline_cp {
            nearest_position(&position, finish_positions)
                .ok_or_else(|| anyhow!("no finish-line blocks found on track"))?
        } else {
            checkpoint_positions
                .get(next_checkpoint_index as usize)
                .and_then(|p| *p)
                .ok_or_else(|| anyhow!("checkpoint {next_checkpoint_index} not in table"))?
        };

        Ok(Self {
            frames,
            speed_kmh,
            has_started,
            finish_frames,
            next_checkpoint_index,
            has_checkpoint_to_respawn_at: has_checkpoint,
            position,
            quaternion,
            collision_impulses,
            wheel_contacts,
            wheel_suspension_lengths,
            wheel_suspension_velocities,
            wheel_delta_rotations,
            wheel_skid_info,
            steering,
            brake_light_enabled,
            controls,
            is_finishline_cp,
            next_checkpoint_position,
        })
    }
}

/// Zero-copy cursor over the WASM car-state buffer.
///
/// All reads are little-endian.  Bounds are checked lazily on each read; the
/// error message includes the current position for easy debugging.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    fn require(&self, n: usize) -> anyhow::Result<()> {
        if self.pos + n > self.buf.len() {
            bail!(
                "car state buffer too short at offset {} \
                 (need {n} more bytes, have {})",
                self.pos,
                self.buf.len().saturating_sub(self.pos),
            );
        }
        Ok(())
    }

    fn u8(&mut self) -> anyhow::Result<u8> {
        self.require(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn u16_le(&mut self) -> anyhow::Result<u16> {
        self.require(2)?;
        let v = u16::from_le_bytes(self.buf[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn u24_le(&mut self) -> anyhow::Result<u32> {
        self.require(3)?;
        let v = self.buf[self.pos] as u32
            | (self.buf[self.pos + 1] as u32) << 8
            | (self.buf[self.pos + 2] as u32) << 16;
        self.pos += 3;
        Ok(v)
    }

    fn f32_le(&mut self) -> anyhow::Result<f32> {
        self.require(4)?;
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn vec3(&mut self) -> anyhow::Result<[f32; 3]> {
        Ok([self.f32_le()?, self.f32_le()?, self.f32_le()?])
    }

    fn vec4(&mut self) -> anyhow::Result<[f32; 4]> {
        Ok([
            self.f32_le()?,
            self.f32_le()?,
            self.f32_le()?,
            self.f32_le()?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impulses_start_empty() {
        let ci = CollisionImpulses::new();
        assert!(ci.is_empty());
        assert_eq!(ci.len(), 0);
        assert_eq!(ci.as_slice(), &[] as &[f32]);
    }

    #[test]
    fn impulses_push_and_read() {
        let mut ci = CollisionImpulses::new();
        ci.push(1.5);
        ci.push(2.5);
        ci.push(3.5);
        assert_eq!(ci.len(), 3);
        assert!(!ci.is_empty());
        assert_eq!(ci.as_slice(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn impulses_full_capacity() {
        let mut ci = CollisionImpulses::new();
        ci.push(1.0);
        ci.push(2.0);
        ci.push(3.0);
        ci.push(4.0);
        assert_eq!(ci.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    struct Buf(Vec<u8>);

    impl Buf {
        fn new() -> Self {
            Self(Vec::new())
        }

        fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }

        fn u16(mut self, v: u16) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn u24(mut self, v: u32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes()[..3]);
            self
        }

        fn f32(mut self, v: f32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn vec3(self, x: f32, y: f32, z: f32) -> Self {
            self.f32(x).f32(y).f32(z)
        }

        fn vec4(self, x: f32, y: f32, z: f32, w: f32) -> Self {
            self.f32(x).f32(y).f32(z).f32(w)
        }

        /// Append four zero vec4s (suspension lengths/velocities/delta/skid).
        fn zero_wheel_data(self) -> Self {
            self.vec4(0., 0., 0., 0.)
                .vec4(0., 0., 0., 0.)
                .vec4(0., 0., 0., 0.)
                .vec4(0., 0., 0., 0.)
        }

        fn build(self) -> Vec<u8> {
            self.0
        }
    }

    /// Build a minimal valid buffer with no finish frames, no wheels, no impulses.
    fn minimal(
        flags: u8,
        next_cp: u16,
        pos: [f32; 3],
        quat: [f32; 4],
        steering: f32,
        ctrl: u8,
    ) -> Vec<u8> {
        Buf::new()
            .u24(0)
            .f32(0.0)
            .u8(flags)
            .u16(next_cp)
            .vec3(pos[0], pos[1], pos[2])
            .vec4(quat[0], quat[1], quat[2], quat[3])
            .u8(0) // no impulses
            .zero_wheel_data()
            .f32(steering)
            .u8(ctrl)
            .build()
    }

    fn finish_pos() -> Vec<[f32; 3]> {
        vec![[10., 0., 0.], [20., 0., 0.]]
    }
    /// Build a dense checkpoint-position table (index = checkpoint number)
    /// from sparse `(index, position)` entries, mirroring how
    /// `PreparedTrack` builds it in production.
    fn cp_map(entries: &[(u16, [f32; 3])]) -> Vec<Option<[f32; 3]>> {
        let max = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);
        let mut v = vec![None; max as usize + 1];
        for &(k, pos) in entries {
            v[k as usize] = Some(pos);
        }
        v
    }

    #[test]
    fn basic_fields() {
        let buf = Buf::new()
            .u24(1234)
            .f32(88.5)
            .u8(0x01) // has_started
            .u16(2)
            .vec3(1., 2., 3.)
            .vec4(0., 0., 0., 1.)
            .u8(0)
            .zero_wheel_data()
            .f32(0.25)
            .u8(0x00)
            .build();

        let s =
            CarState::deserialize(&buf, 5, &cp_map(&[(2, [5., 5., 5.])]), &finish_pos()).unwrap();
        assert_eq!(s.frames, 1234);
        assert!((s.speed_kmh - 88.5).abs() < 1e-5);
        assert!(s.has_started);
        assert_eq!(s.finish_frames, None);
        assert_eq!(s.next_checkpoint_index, 2);
        assert_eq!(s.position, [1., 2., 3.]);
        assert!(s.collision_impulses.is_empty());
        assert!((s.steering - 0.25).abs() < 1e-5);
        assert!(!s.is_finishline_cp);
        assert_eq!(s.next_checkpoint_position, [5., 5., 5.]);
    }

    #[test]
    fn finish_frames_parsed() {
        let buf = Buf::new()
            .u24(0)
            .f32(0.)
            .u8(0x02) // bit 1 = finish_frames present
            .u24(9999)
            .u16(6)
            .vec3(0., 0., 0.)
            .vec4(0., 0., 0., 1.)
            .u8(0)
            .zero_wheel_data()
            .f32(0.)
            .u8(0x00)
            .build();

        let s = CarState::deserialize(&buf, 5, &[], &finish_pos()).unwrap();
        assert_eq!(s.finish_frames, Some(9999));
        assert!(s.is_finishline_cp);
    }

    #[test]
    fn all_controls_set() {
        let buf = minimal(0x00, 1, [0.; 3], [0., 0., 0., 1.], 0., 0x3F);
        let s = CarState::deserialize(&buf, 5, &cp_map(&[(1, [0.; 3])]), &[]).unwrap();
        assert!(s.controls.up);
        assert!(s.controls.right);
        assert!(s.controls.down);
        assert!(s.controls.left);
        assert!(s.controls.reset);
        assert!(s.brake_light_enabled);
    }

    #[test]
    fn no_controls_set() {
        let buf = minimal(0x00, 1, [0.; 3], [0., 0., 0., 1.], 0., 0x00);
        let s = CarState::deserialize(&buf, 5, &cp_map(&[(1, [0.; 3])]), &[]).unwrap();
        assert!(!s.controls.up);
        assert!(!s.controls.right);
        assert!(!s.controls.down);
        assert!(!s.controls.left);
        assert!(!s.controls.reset);
        assert!(!s.brake_light_enabled);
    }

    #[test]
    fn three_impulses_parsed() {
        let buf = Buf::new()
            .u24(0)
            .f32(0.)
            .u8(0x00)
            .u16(1)
            .vec3(0., 0., 0.)
            .vec4(0., 0., 0., 1.)
            .u8(3)
            .f32(1.1)
            .f32(2.2)
            .f32(3.3)
            .zero_wheel_data()
            .f32(0.)
            .u8(0x00)
            .build();

        let s = CarState::deserialize(&buf, 5, &cp_map(&[(1, [0.; 3])]), &[]).unwrap();
        assert_eq!(s.collision_impulses.len(), 3);
        let sl = s.collision_impulses.as_slice();
        assert!((sl[0] - 1.1).abs() < 1e-5);
        assert!((sl[1] - 2.2).abs() < 1e-5);
        assert!((sl[2] - 3.3).abs() < 1e-5);
    }

    #[test]
    fn two_wheel_contacts_parsed() {
        let buf = Buf::new()
            .u24(0)
            .f32(0.)
            .u8(0x18) // bits 3+4 = wheels 0+1 present
            .u16(1)
            .vec3(0., 0., 0.)
            .vec4(0., 0., 0., 1.)
            .u8(0)
            .vec3(1., 2., 3.)
            .vec3(0., 1., 0.) // wheel 0
            .vec3(4., 5., 6.)
            .vec3(0., -1., 0.) // wheel 1
            .zero_wheel_data()
            .f32(0.)
            .u8(0x00)
            .build();

        let s = CarState::deserialize(&buf, 5, &cp_map(&[(1, [0.; 3])]), &[]).unwrap();
        let w0 = s.wheel_contacts[0].as_ref().unwrap();
        assert_eq!(w0.position, [1., 2., 3.]);
        assert_eq!(w0.normal, [0., 1., 0.]);
        let w1 = s.wheel_contacts[1].as_ref().unwrap();
        assert_eq!(w1.position, [4., 5., 6.]);
        assert!(s.wheel_contacts[2].is_none());
        assert!(s.wheel_contacts[3].is_none());
    }

    #[test]
    fn finishline_when_index_equals_max_plus_one() {
        let buf = minimal(0x00, 5, [0.; 3], [0., 0., 0., 1.], 0., 0x00);
        let s = CarState::deserialize(&buf, 4, &[], &finish_pos()).unwrap();
        assert!(s.is_finishline_cp);
        assert_eq!(s.next_checkpoint_position, [10., 0., 0.]);
    }

    #[test]
    fn not_finishline_when_below_max() {
        let buf = minimal(0x00, 2, [0.; 3], [0., 0., 0., 1.], 0., 0x00);
        let s =
            CarState::deserialize(&buf, 4, &cp_map(&[(2, [7., 8., 9.])]), &finish_pos()).unwrap();
        assert!(!s.is_finishline_cp);
        assert_eq!(s.next_checkpoint_position, [7., 8., 9.]);
    }

    #[test]
    fn finishline_when_max_checkpoint_zero() {
        let buf = minimal(0x00, 0, [0.; 3], [0., 0., 0., 1.], 0., 0x00);
        let s = CarState::deserialize(&buf, 0, &[], &finish_pos()).unwrap();
        assert!(s.is_finishline_cp);
    }

    #[test]
    fn finishline_wrapping_add_at_u16_max() {
        // max=u16::MAX → wrapping_add(1)=0 → next_cp=0 triggers finish line
        let buf = minimal(0x00, 0, [0.; 3], [0., 0., 0., 1.], 0., 0x00);
        let s = CarState::deserialize(&buf, u16::MAX, &[], &finish_pos()).unwrap();
        assert!(s.is_finishline_cp);
    }

    #[test]
    fn truncated_buffer_errors() {
        assert!(CarState::deserialize(&[0u8; 4], 0, &[], &finish_pos()).is_err());
    }

    #[test]
    fn impulse_count_over_4_errors() {
        let buf = Buf::new()
            .u24(0)
            .f32(0.)
            .u8(0x00)
            .u16(0)
            .vec3(0., 0., 0.)
            .vec4(0., 0., 0., 1.)
            .u8(5) // invalid
            .build();
        let err = CarState::deserialize(&buf, 0, &[], &finish_pos()).unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"));
    }

    #[test]
    fn missing_checkpoint_errors() {
        let buf = minimal(0x00, 3, [0.; 3], [0., 0., 0., 1.], 0., 0x00);
        assert!(CarState::deserialize(&buf, 5, &[], &finish_pos()).is_err());
    }

    #[test]
    fn empty_finish_positions_errors() {
        let buf = minimal(0x00, 6, [0.; 3], [0., 0., 0., 1.], 0., 0x00);
        assert!(CarState::deserialize(&buf, 5, &[], &[]).is_err());
    }

    #[test]
    fn suspension_data_parsed() {
        let buf = Buf::new()
            .u24(0)
            .f32(0.)
            .u8(0x00)
            .u16(1)
            .vec3(0., 0., 0.)
            .vec4(0., 0., 0., 1.)
            .u8(0)
            .vec4(0.1, 0.2, 0.3, 0.4)
            .vec4(1.1, 1.2, 1.3, 1.4)
            .vec4(2.1, 2.2, 2.3, 2.4)
            .vec4(3.1, 3.2, 3.3, 3.4)
            .f32(0.75)
            .u8(0x00)
            .build();

        let s = CarState::deserialize(&buf, 5, &cp_map(&[(1, [0.; 3])]), &[]).unwrap();
        assert!((s.wheel_suspension_lengths[0] - 0.1).abs() < 1e-5);
        assert!((s.wheel_suspension_lengths[3] - 0.4).abs() < 1e-5);
        assert!((s.wheel_suspension_velocities[0] - 1.1).abs() < 1e-5);
        assert!((s.wheel_delta_rotations[2] - 2.3).abs() < 1e-5);
        assert!((s.wheel_skid_info[3] - 3.4).abs() < 1e-5);
        assert!((s.steering - 0.75).abs() < 1e-5);
    }
}
