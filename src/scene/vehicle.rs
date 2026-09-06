// scene/vehicle.rs — car rendering.
// +X = car forward,  +Y = up,  ±Z = right/left.
//
// Physics anchor: car_matrix() origin = chassis rigid-body centre Y.
// When settled on flat ground, chassis Y ≈ terrain_H + 0.68 m.

use cgmath::{Matrix4, Vector3, Vector4};
use crate::physics::PhysicsWorld;
use crate::scene::transform::{uniform_from_matrix, scale_matrix, TransformUniform};

pub struct DrawCmd {
    pub mesh_id: usize,
    pub uniform: TransformUniform,
}

pub const MESH_CUBE:        usize = 0;
pub const MESH_WHEEL:       usize = 1;
pub const MESH_TERRAIN:     usize = 2;
pub const MESH_WATER:       usize = 3;
pub const MESH_ROAD:        usize = 4;
pub const MESH_CONE:        usize = 5;
pub const MESH_SPHERE:      usize = 6;
pub const MESH_TERRAIN_BASE: usize = 10;
pub const MESH_WATER_BASE:   usize = 10 + crate::terrain::MAX_CHUNKS;

// ── Palette ───────────────────────────────────────────────────────────────────
const BODY:   [f32; 4] = [0.72, 0.10, 0.10, 0.95]; // deep red
const BODY2:  [f32; 4] = [0.62, 0.08, 0.08, 0.95]; // shadow red (underside)
const GLASS:  [f32; 4] = [0.06, 0.09, 0.14, 0.97];
const DARK:   [f32; 4] = [0.08, 0.08, 0.09, 0.91];
const HEADLT: [f32; 4] = [1.00, 0.98, 0.88, 0.93];
const TAILLT: [f32; 4] = [0.90, 0.04, 0.04, 0.93];
const RUBBER: [f32; 4] = [0.10, 0.10, 0.10, 0.91];
const RIM:    [f32; 4] = [0.82, 0.82, 0.85, 0.97];
const CHROME: [f32; 4] = [0.94, 0.94, 0.96, 0.97];
const UNDER:  [f32; 4] = [0.14, 0.14, 0.15, 0.91]; // underbody / sill

// ── Rotation helpers ──────────────────────────────────────────────────────────

/// Rotate around car's lateral axis (Z). +deg tilts top toward rear.
fn rot_z(deg: f32) -> Matrix4<f32> {
    let a = deg.to_radians();
    let (s, c) = (a.sin(), a.cos());
    Matrix4::from_cols(
        Vector4::new( c,  s, 0.0, 0.0),
        Vector4::new(-s,  c, 0.0, 0.0),
        Vector4::new(0.0, 0.0, 1.0, 0.0),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
    )
}

/// Rotate around car's forward axis (X). +deg tilts top toward right (−Z).
fn rot_x(deg: f32) -> Matrix4<f32> {
    let a = deg.to_radians();
    let (s, c) = (a.sin(), a.cos());
    Matrix4::from_cols(
        Vector4::new(1.0, 0.0, 0.0, 0.0),
        Vector4::new(0.0,  c,  s, 0.0),
        Vector4::new(0.0, -s,  c, 0.0),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
    )
}

pub fn car_draw_cmds(physics: &PhysicsWorld) -> Vec<DrawCmd> {
    let ch = physics.car_matrix();
    let mut cmds = Vec::with_capacity(140);
    type V3 = Vector3<f32>;
    use crate::physics::{WHEEL_WIDTH, WHEEL_RADIUS};

    // Axis-aligned box at offset, full-size (sx,sy,sz).
    let flat = |off: V3, sx: f32, sy: f32, sz: f32, col: [f32; 4]| -> DrawCmd {
        let world = scale_matrix(ch * Matrix4::from_translation(off), sx, sy, sz);
        DrawCmd { mesh_id: MESH_CUBE, uniform: uniform_from_matrix(world, col) }
    };

    // Box rotated around Z (slope along car length — for hood, windshield, etc.)
    let rz = |off: V3, deg: f32, sx: f32, sy: f32, sz: f32, col: [f32; 4]| -> DrawCmd {
        let world = ch * Matrix4::from_translation(off) * rot_z(deg)
            * Matrix4::from_nonuniform_scale(sx, sy, sz);
        DrawCmd { mesh_id: MESH_CUBE, uniform: uniform_from_matrix(world, col) }
    };

    // Box rotated around X (slope across car width — for fender curves, roof arch).
    let rx = |off: V3, deg: f32, sx: f32, sy: f32, sz: f32, col: [f32; 4]| -> DrawCmd {
        let world = ch * Matrix4::from_translation(off) * rot_x(deg)
            * Matrix4::from_nonuniform_scale(sx, sy, sz);
        DrawCmd { mesh_id: MESH_CUBE, uniform: uniform_from_matrix(world, col) }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // UNDERBODY / FLOOR PAN
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 0.00, -0.56, 0.00), 3.80, 0.06, 1.70, UNDER));

    // ═══════════════════════════════════════════════════════════════════════
    // LOWER BODY SLAB  (sill to beltline, −0.50 → 0.00)
    // Three lateral segments: door centre + two fender blocks
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 0.00, -0.25, 0.00), 2.60, 0.50, 1.80, BODY));  // door region
    cmds.push(flat(V3::new( 1.55, -0.25, 0.00), 0.72, 0.50, 1.74, BODY));  // front fender slab
    cmds.push(flat(V3::new(-1.55, -0.25, 0.00), 0.72, 0.50, 1.74, BODY));  // rear fender slab

    // Sill strips
    cmds.push(flat(V3::new( 0.00, -0.52,  0.91), 3.20, 0.06, 0.05, DARK));
    cmds.push(flat(V3::new( 0.00, -0.52, -0.91), 3.20, 0.06, 0.05, DARK));

    // ═══════════════════════════════════════════════════════════════════════
    // FENDER CROWNS — angled top faces to fake a curved cross-section.
    // Each fender top is two angled slabs: inner slope + outer slope.
    // rot_x(+deg) tilts +Y toward −Z, rot_x(−deg) toward +Z.
    // ═══════════════════════════════════════════════════════════════════════

    // Front fender crown (outer panels slope down from centre)
    cmds.push(rx(V3::new( 1.55, 0.08,  0.62), -28.0, 0.68, 0.10, 0.60, BODY)); // outer L
    cmds.push(rx(V3::new( 1.55, 0.08, -0.62),  28.0, 0.68, 0.10, 0.60, BODY)); // outer R
    cmds.push(flat(V3::new( 1.55,  0.10, 0.00), 0.68, 0.06, 0.54, BODY));       // crown top

    // Rear fender crown
    cmds.push(rx(V3::new(-1.55, 0.04,  0.62), -26.0, 0.68, 0.10, 0.60, BODY));
    cmds.push(rx(V3::new(-1.55, 0.04, -0.62),  26.0, 0.68, 0.10, 0.60, BODY));
    cmds.push(flat(V3::new(-1.55,  0.06, 0.00), 0.68, 0.06, 0.54, BODY));

    // Door shoulder — slight outward crown between fenders
    cmds.push(rx(V3::new( 0.00, 0.02,  0.84), -20.0, 2.40, 0.08, 0.28, BODY));
    cmds.push(rx(V3::new( 0.00, 0.02, -0.84),  20.0, 2.40, 0.08, 0.28, BODY));
    cmds.push(flat(V3::new( 0.00,  0.04, 0.00), 2.40, 0.04, 0.72, BODY));

    // ═══════════════════════════════════════════════════════════════════════
    // FRONT BUMPER  / NOSE
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 2.12, -0.10,  0.00), 0.30, 0.48, 1.68, BODY));  // upper fascia
    cmds.push(flat(V3::new( 2.22, -0.42,  0.00), 0.20, 0.24, 1.50, DARK));  // lower apron
    cmds.push(flat(V3::new( 2.30, -0.52,  0.00), 0.08, 0.06, 1.34, DARK));  // chin splitter
    // Corner wraps
    cmds.push(flat(V3::new( 2.06, -0.10,  0.86), 0.44, 0.48, 0.14, BODY));
    cmds.push(flat(V3::new( 2.06, -0.10, -0.86), 0.44, 0.48, 0.14, BODY));
    // Front intake grilles
    cmds.push(flat(V3::new( 2.28, -0.38,  0.34), 0.06, 0.16, 0.36, DARK));
    cmds.push(flat(V3::new( 2.28, -0.38, -0.34), 0.06, 0.16, 0.36, DARK));
    cmds.push(flat(V3::new( 2.28, -0.38,  0.00), 0.06, 0.16, 0.26, DARK));

    // ═══════════════════════════════════════════════════════════════════════
    // HOOD — 3-segment curved profile (front dips, rises at cowl)
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(rz(V3::new( 1.82,  0.08, 0.00),  -8.0, 0.72, 0.05, 1.60, BODY)); // front dip
    cmds.push(rz(V3::new( 1.22,  0.14, 0.00),  -3.0, 0.66, 0.05, 1.58, BODY)); // mid flat
    cmds.push(rz(V3::new( 0.64,  0.18, 0.00),   4.0, 0.52, 0.05, 1.54, BODY)); // cowl rise
    // Hood crease ridges
    cmds.push(flat(V3::new( 1.30,  0.17,  0.55), 1.48, 0.03, 0.06, BODY));
    cmds.push(flat(V3::new( 1.30,  0.17, -0.55), 1.48, 0.03, 0.06, BODY));
    // Hood front drop (joins hood front edge to bumper face)
    cmds.push(flat(V3::new( 2.18,  0.01, 0.00), 0.06, 0.10, 1.58, BODY));

    // ═══════════════════════════════════════════════════════════════════════
    // WINDSHIELD — raked glass at 52°
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(rz(V3::new( 0.26, 0.38, 0.00), 52.0, 0.04, 0.62, 1.48, GLASS));
    // A-pillars
    cmds.push(rz(V3::new( 0.26, 0.38,  0.76), 52.0, 0.14, 0.62, 0.08, DARK));
    cmds.push(rz(V3::new( 0.26, 0.38, -0.76), 52.0, 0.14, 0.62, 0.08, DARK));

    // ═══════════════════════════════════════════════════════════════════════
    // ROOF — 3-segment arch (centre highest, edges angled down)
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new(-0.44, 0.60, 0.00), 1.28, 0.05, 1.18, BODY));           // centre flat
    cmds.push(rx(V3::new(-0.44, 0.56,  0.66), -30.0, 1.28, 0.05, 0.46, BODY));     // left slope
    cmds.push(rx(V3::new(-0.44, 0.56, -0.66),  30.0, 1.28, 0.05, 0.46, BODY));     // right slope

    // ═══════════════════════════════════════════════════════════════════════
    // UPPER CABIN STRUCTURE
    // ═══════════════════════════════════════════════════════════════════════
    // B-pillars
    cmds.push(flat(V3::new(-0.06,  0.32,  0.87), 0.10, 0.62, 0.06, DARK));
    cmds.push(flat(V3::new(-0.06,  0.32, -0.87), 0.10, 0.62, 0.06, DARK));
    // C-pillars (thicker rear pillars)
    cmds.push(flat(V3::new(-1.10,  0.30,  0.85), 0.42, 0.60, 0.10, DARK));
    cmds.push(flat(V3::new(-1.10,  0.30, -0.85), 0.42, 0.60, 0.10, DARK));

    // ═══════════════════════════════════════════════════════════════════════
    // SIDE GLASS
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 0.24,  0.30,  0.88), 1.04, 0.52, 0.04, GLASS)); // front door L
    cmds.push(flat(V3::new( 0.24,  0.30, -0.88), 1.04, 0.52, 0.04, GLASS)); // front door R
    cmds.push(flat(V3::new(-0.70,  0.28,  0.88), 0.72, 0.48, 0.04, GLASS)); // rear door L
    cmds.push(flat(V3::new(-0.70,  0.28, -0.88), 0.72, 0.48, 0.04, GLASS)); // rear door R

    // ═══════════════════════════════════════════════════════════════════════
    // REAR WINDOW — raked glass at −50°
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(rz(V3::new(-1.18, 0.38, 0.00), -50.0, 0.04, 0.52, 1.42, GLASS));
    // D-pillars
    cmds.push(rz(V3::new(-1.18, 0.38,  0.74), -50.0, 0.14, 0.52, 0.08, DARK));
    cmds.push(rz(V3::new(-1.18, 0.38, -0.74), -50.0, 0.14, 0.52, 0.08, DARK));

    // ═══════════════════════════════════════════════════════════════════════
    // TRUNK LID — 3-segment: slight slope, then drops to bumper
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(rz(V3::new(-1.50,  0.12, 0.00),  2.0, 0.44, 0.04, 1.64, BODY)); // upper trunk
    cmds.push(rz(V3::new(-1.84,  0.05, 0.00), -5.0, 0.28, 0.04, 1.62, BODY)); // rear drop
    // Boot lip / spoiler
    cmds.push(flat(V3::new(-1.92,  0.08, 0.00), 0.04, 0.10, 1.56, DARK));

    // ═══════════════════════════════════════════════════════════════════════
    // REAR BUMPER
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new(-2.10, -0.10,  0.00), 0.28, 0.44, 1.70, BODY));
    cmds.push(flat(V3::new(-2.18, -0.40,  0.00), 0.18, 0.24, 1.52, DARK));
    cmds.push(flat(V3::new(-2.08, -0.10,  0.84), 0.44, 0.44, 0.14, BODY)); // corner L
    cmds.push(flat(V3::new(-2.08, -0.10, -0.84), 0.44, 0.44, 0.14, BODY)); // corner R
    // Exhaust pipes
    cmds.push(flat(V3::new(-2.34, -0.48,  0.40), 0.04, 0.08, 0.14, DARK));
    cmds.push(flat(V3::new(-2.34, -0.48, -0.40), 0.04, 0.08, 0.14, DARK));
    cmds.push(flat(V3::new(-2.38, -0.48,  0.40), 0.02, 0.06, 0.10, CHROME));
    cmds.push(flat(V3::new(-2.38, -0.48, -0.40), 0.02, 0.06, 0.10, CHROME));

    // ═══════════════════════════════════════════════════════════════════════
    // WHEEL ARCH LIPS — raised beads above each tyre opening
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 1.55, -0.08,  0.90), 0.84, 0.05, 0.06, DARK));
    cmds.push(flat(V3::new( 1.55, -0.08, -0.90), 0.84, 0.05, 0.06, DARK));
    cmds.push(flat(V3::new(-1.55, -0.12,  0.90), 0.80, 0.05, 0.06, DARK));
    cmds.push(flat(V3::new(-1.55, -0.12, -0.90), 0.80, 0.05, 0.06, DARK));

    // ═══════════════════════════════════════════════════════════════════════
    // HEADLIGHTS — full-width LED bar + recessed housing
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 2.38,  0.00,  0.00), 0.04, 0.04, 1.62, HEADLT)); // LED bar
    cmds.push(flat(V3::new( 2.32, -0.08,  0.54), 0.16, 0.22, 0.38, DARK));   // housing L
    cmds.push(flat(V3::new( 2.32, -0.08, -0.54), 0.16, 0.22, 0.38, DARK));   // housing R
    cmds.push(flat(V3::new( 2.36,  0.02,  0.20), 0.04, 0.04, 0.28, HEADLT)); // DRL L
    cmds.push(flat(V3::new( 2.36,  0.02, -0.20), 0.04, 0.04, 0.28, HEADLT)); // DRL R

    // ═══════════════════════════════════════════════════════════════════════
    // TAIL LIGHTS
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new(-2.34, -0.08,  0.00), 0.04, 0.04, 1.60, TAILLT)); // bar
    cmds.push(flat(V3::new(-2.28, -0.14,  0.52), 0.14, 0.22, 0.38, DARK));
    cmds.push(flat(V3::new(-2.28, -0.14, -0.52), 0.14, 0.22, 0.38, DARK));
    cmds.push(flat(V3::new(-2.32, -0.06,  0.20), 0.04, 0.04, 0.28, TAILLT));
    cmds.push(flat(V3::new(-2.32, -0.06, -0.20), 0.04, 0.04, 0.28, TAILLT));

    // ═══════════════════════════════════════════════════════════════════════
    // DOOR HANDLES
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 0.38, -0.32,  0.92), 0.30, 0.04, 0.04, CHROME));
    cmds.push(flat(V3::new( 0.38, -0.32, -0.92), 0.30, 0.04, 0.04, CHROME));
    cmds.push(flat(V3::new(-0.52, -0.32,  0.92), 0.30, 0.04, 0.04, CHROME));
    cmds.push(flat(V3::new(-0.52, -0.32, -0.92), 0.30, 0.04, 0.04, CHROME));

    // ═══════════════════════════════════════════════════════════════════════
    // WING MIRRORS
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 0.48, -0.06,  0.93), 0.06, 0.14, 0.04, DARK)); // stalk L
    cmds.push(flat(V3::new( 0.48, -0.06, -0.93), 0.06, 0.14, 0.04, DARK)); // stalk R
    cmds.push(flat(V3::new( 0.38, 0.00,  1.06), 0.22, 0.10, 0.18, DARK)); // head L
    cmds.push(flat(V3::new( 0.38, 0.00, -1.06), 0.22, 0.10, 0.18, DARK)); // head R

    // ═══════════════════════════════════════════════════════════════════════
    // WHEELS — tyre + chrome bead + rim + barrel + centre cap
    // ═══════════════════════════════════════════════════════════════════════
    for i in 0..4 {
        let wm = physics.wheel_matrix(i);

        let ws = scale_matrix(wm, WHEEL_WIDTH,           WHEEL_RADIUS,        WHEEL_RADIUS);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(ws, RUBBER) });

        let ob = scale_matrix(wm, WHEEL_WIDTH * 0.06,    WHEEL_RADIUS * 0.97, WHEEL_RADIUS * 0.97);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(ob, CHROME) });

        let rs = scale_matrix(wm, WHEEL_WIDTH * 0.38,    WHEEL_RADIUS * 0.87, WHEEL_RADIUS * 0.87);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(rs, RIM) });

        let ib = scale_matrix(wm, WHEEL_WIDTH * 0.22,    WHEEL_RADIUS * 0.70, WHEEL_RADIUS * 0.70);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(ib, DARK) });

        let cs = scale_matrix(wm, WHEEL_WIDTH * 0.14,    WHEEL_RADIUS * 0.22, WHEEL_RADIUS * 0.22);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(cs, CHROME) });
    }

    cmds
}
