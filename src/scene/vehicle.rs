// scene/vehicle.rs — car rendering: lofted smooth body + box details.
// +X = car forward,  +Y = up,  ±Z = right/left.
//
// Physics anchor: car_matrix() origin = chassis rigid-body centre Y.
// When settled on flat ground, chassis Y ≈ terrain_H + 0.68 m
// (chassis_half_h 0.30 + wheel_radius 0.38 with suspension compressed).
// wheel_matrix() visual position uses VISUAL_SUSP_LEN < SUSPENSION_REST
// so wheels appear to sit under the arches rather than hanging below.

use cgmath::{Matrix4, Vector3};
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
const BODY:   [f32; 4] = [0.58, 0.60, 0.62, 0.95]; // Quicksilver metallic
const GLASS:  [f32; 4] = [0.04, 0.06, 0.10, 0.97]; // dark-tinted glass
const DARK:   [f32; 4] = [0.07, 0.07, 0.08, 0.91]; // matte black trim
const HEADLT: [f32; 4] = [1.00, 0.98, 0.88, 0.93]; // LED white
const TAILLT: [f32; 4] = [0.90, 0.04, 0.04, 0.93]; // tail-light red
const RUBBER: [f32; 4] = [0.10, 0.10, 0.10, 0.91]; // tyre rubber
const RIM:    [f32; 4] = [0.82, 0.82, 0.85, 0.97]; // aero-rim silver
const CHROME: [f32; 4] = [0.94, 0.94, 0.96, 0.97]; // bright chrome

pub fn car_draw_cmds(physics: &PhysicsWorld) -> Vec<DrawCmd> {
    let ch = physics.car_matrix();
    let mut cmds = Vec::with_capacity(32);
    type V3 = Vector3<f32>;
    use crate::physics::{WHEEL_WIDTH, WHEEL_RADIUS};

    // Flat box panel helper
    let flat = |off: V3, sx: f32, sy: f32, sz: f32, col: [f32; 4]| -> DrawCmd {
        let world = scale_matrix(ch * Matrix4::from_translation(off), sx, sy, sz);
        DrawCmd { mesh_id: MESH_CUBE, uniform: uniform_from_matrix(world, col) }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // BOX BODY — door slab + fenders + cabin + roof
    // y=−0.80: wheel centre.  y=−0.58: floor/sill bottom.  y=0: belt.  y=+0.48: roof.
    // ═══════════════════════════════════════════════════════════════════════
    cmds.push(flat(V3::new( 0.00, -0.29,  0.00), 4.40, 0.58, 1.86, BODY)); // door slab
    cmds.push(flat(V3::new( 1.44, -0.27,  0.00), 1.18, 0.62, 1.76, BODY)); // front fender
    cmds.push(flat(V3::new(-1.44, -0.27,  0.00), 1.18, 0.62, 1.76, BODY)); // rear fender
    cmds.push(flat(V3::new(-0.14,  0.10,  0.00), 2.70, 0.68, 1.76, BODY)); // cabin
    cmds.push(flat(V3::new(-0.36,  0.47,  0.00), 2.50, 0.06, 1.70, BODY)); // roof

    // Rocker / sill strips
    cmds.push(flat(V3::new( 0.00, -0.55,  0.93), 3.20, 0.06, 0.06, DARK));
    cmds.push(flat(V3::new( 0.00, -0.55, -0.93), 3.20, 0.06, 0.06, DARK));

    // Spoiler lip on trunk
    cmds.push(flat(V3::new(-2.14, -0.08,  0.00), 0.56, 0.04, 1.78, DARK));

    // Chin skirt on front bumper
    cmds.push(flat(V3::new( 2.40, -0.56,  0.00), 0.20, 0.06, 1.68, DARK));

    // ═══════════════════════════════════════════════════════════════════════
    // GLASS — windshield, panoramic roof, rear hatch, side windows
    // ═══════════════════════════════════════════════════════════════════════

    // Windshield (front face of cabin box)
    cmds.push(flat(V3::new( 0.72,  0.10,  0.00), 0.05, 0.62, 1.72, GLASS));
    // Panoramic glass roof
    cmds.push(flat(V3::new(-0.36,  0.50,  0.00), 2.28, 0.04, 1.62, GLASS));
    // Rear hatch glass
    cmds.push(flat(V3::new(-1.80,  0.05,  0.00), 0.05, 0.48, 1.62, GLASS));
    // Front side windows
    cmds.push(flat(V3::new( 0.28,  0.15,  0.91), 1.18, 0.46, 0.05, GLASS));
    cmds.push(flat(V3::new( 0.28,  0.15, -0.91), 1.18, 0.46, 0.05, GLASS));
    // Rear side windows
    cmds.push(flat(V3::new(-0.66,  0.13,  0.91), 0.96, 0.42, 0.05, GLASS));
    cmds.push(flat(V3::new(-0.66,  0.13, -0.91), 0.96, 0.42, 0.05, GLASS));
    // Quarter windows
    cmds.push(flat(V3::new(-1.34,  0.09,  0.91), 0.30, 0.28, 0.05, GLASS));
    cmds.push(flat(V3::new(-1.34,  0.09, -0.91), 0.30, 0.28, 0.05, GLASS));

    // ═══════════════════════════════════════════════════════════════════════
    // LIGHT BARS — full-width LED signature
    // ═══════════════════════════════════════════════════════════════════════

    cmds.push(flat(V3::new( 2.47, -0.14,  0.00), 0.04, 0.07, 1.78, HEADLT));
    cmds.push(flat(V3::new(-2.47, -0.20,  0.00), 0.04, 0.07, 1.76, TAILLT));

    // ═══════════════════════════════════════════════════════════════════════
    // CAMERA MIRRORS
    // ═══════════════════════════════════════════════════════════════════════

    cmds.push(flat(V3::new( 0.50, -0.17,  0.96), 0.06, 0.18, 0.04, DARK));
    cmds.push(flat(V3::new( 0.50, -0.17, -0.96), 0.06, 0.18, 0.04, DARK));
    cmds.push(flat(V3::new( 0.44, -0.07,  1.10), 0.28, 0.10, 0.18, DARK));
    cmds.push(flat(V3::new( 0.44, -0.07, -1.10), 0.28, 0.10, 0.18, DARK));
    cmds.push(flat(V3::new( 0.34, -0.07,  1.17), 0.03, 0.05, 0.04, CHROME));
    cmds.push(flat(V3::new( 0.34, -0.07, -1.17), 0.03, 0.05, 0.04, CHROME));

    // ═══════════════════════════════════════════════════════════════════════
    // DOOR HANDLES
    // ═══════════════════════════════════════════════════════════════════════

    cmds.push(flat(V3::new( 0.28, -0.45,  0.95), 0.34, 0.04, 0.04, CHROME));
    cmds.push(flat(V3::new( 0.28, -0.45, -0.95), 0.34, 0.04, 0.04, CHROME));
    cmds.push(flat(V3::new(-0.62, -0.45,  0.95), 0.34, 0.04, 0.04, CHROME));
    cmds.push(flat(V3::new(-0.62, -0.45, -0.95), 0.34, 0.04, 0.04, CHROME));

    // ═══════════════════════════════════════════════════════════════════════
    // WHEELS — tyre + bead + rim + barrel + centre cap
    // ═══════════════════════════════════════════════════════════════════════

    for i in 0..4 {
        let wm = physics.wheel_matrix(i);

        let ws = scale_matrix(wm, WHEEL_WIDTH,            WHEEL_RADIUS,        WHEEL_RADIUS);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(ws, RUBBER) });

        let ob = scale_matrix(wm, WHEEL_WIDTH * 0.06,     WHEEL_RADIUS * 0.97, WHEEL_RADIUS * 0.97);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(ob, CHROME) });

        let rs = scale_matrix(wm, WHEEL_WIDTH * 0.38,     WHEEL_RADIUS * 0.87, WHEEL_RADIUS * 0.87);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(rs, RIM) });

        let ib = scale_matrix(wm, WHEEL_WIDTH * 0.22,     WHEEL_RADIUS * 0.70, WHEEL_RADIUS * 0.70);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(ib, DARK) });

        let cs = scale_matrix(wm, WHEEL_WIDTH * 0.14,     WHEEL_RADIUS * 0.22, WHEEL_RADIUS * 0.22);
        cmds.push(DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(cs, CHROME) });
    }

    cmds
}
