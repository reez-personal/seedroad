// scene/environment.rs — mountain world: terrain, winding road, trees, water stream.
//
// Trees use recursive branch generation with phyllotactic (137.5°) rotation,
// per-tree deterministic PRNG seeded from world position.
// Conifer crowns = MESH_CONE tiers.  Deciduous foliage = MESH_SPHERE clusters.

use std::f32::consts::TAU;
use cgmath::{Matrix4, Vector3, Vector4};
use crate::scene::transform::{uniform_from_matrix, scale_matrix};
use crate::scene::vehicle::{DrawCmd, MESH_CUBE, MESH_WHEEL, MESH_ROAD, MESH_CONE, MESH_SPHERE,
                             MESH_TERRAIN_BASE, MESH_WATER_BASE};
use crate::terrain::{TerrainManager, WATER_Y};

// ── Palette ───────────────────────────────────────────────────────────────────
// alpha = 0.0  → terrain material (height-based color in shader)
// alpha = 0.65 → water material  (animated transparent in shader)
// alpha = 1.0  → solid color
const TERRAIN_MAT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const WATER_MAT:   [f32; 4] = [0.04, 0.26, 0.50, 0.65];
#[allow(dead_code)]
const _UNUSED_MAT: [f32; 4] = [0.0; 4]; // suppress unused-import warning
const ROAD:        [f32; 4] = [0.07, 0.07, 0.08, 0.91]; // dark asphalt, matte
const BARK:        [f32; 4] = [0.26, 0.14, 0.05, 0.91]; // conifer bark, matte
const BARK_D:      [f32; 4] = [0.32, 0.18, 0.07, 0.91]; // deciduous bark, matte

// ── Primitive helpers ─────────────────────────────────────────────────────────

fn cube(tx: f32, ty: f32, tz: f32, sx: f32, sy: f32, sz: f32, col: [f32; 4]) -> DrawCmd {
    let m = scale_matrix(Matrix4::from_translation(Vector3::new(tx, ty, tz)), sx, sy, sz);
    DrawCmd { mesh_id: MESH_CUBE, uniform: uniform_from_matrix(m, col) }
}

/// Vertical cylinder (local Y axis) at world (tx, ty, tz), radius r, height h.
fn vcyl(tx: f32, ty: f32, tz: f32, r: f32, h: f32, col: [f32; 4]) -> DrawCmd {
    let rz90 = Matrix4::from_cols(
        Vector4::new( 0.0, 1.0, 0.0, 0.0),
        Vector4::new(-1.0, 0.0, 0.0, 0.0),
        Vector4::new( 0.0, 0.0, 1.0, 0.0),
        Vector4::new( 0.0, 0.0, 0.0, 1.0),
    );
    let m = scale_matrix(
        Matrix4::from_translation(Vector3::new(tx, ty, tz)) * rz90,
        h, r, r,
    );
    DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(m, col) }
}

/// Vertical upward cone centred at (cx, cy, cz), base-radius r, height h.
fn vcone(cx: f32, cy: f32, cz: f32, r: f32, h: f32, col: [f32; 4]) -> DrawCmd {
    let m = scale_matrix(Matrix4::from_translation(Vector3::new(cx, cy, cz)), r, h, r);
    DrawCmd { mesh_id: MESH_CONE, uniform: uniform_from_matrix(m, col) }
}

/// Sphere centred at (cx, cy, cz) with radius r.
fn vsphere(cx: f32, cy: f32, cz: f32, r: f32, col: [f32; 4]) -> DrawCmd {
    let m = scale_matrix(Matrix4::from_translation(Vector3::new(cx, cy, cz)), r, r, r);
    DrawCmd { mesh_id: MESH_SPHERE, uniform: uniform_from_matrix(m, col) }
}

/// Cylinder from world-point `from` to `to` using Rodrigues rotation on +X axis.
fn oriented_cyl(from: Vector3<f32>, to: Vector3<f32>, radius: f32, col: [f32; 4]) -> DrawCmd {
    use cgmath::InnerSpace;
    let diff = to - from;
    let len  = diff.magnitude();
    if len < 0.01 {
        return cube(from.x, from.y, from.z, radius*2.0, radius*2.0, radius*2.0, col);
    }
    let dir = diff / len;
    let mid = (from + to) * 0.5;
    let rot = rotation_x_to(dir);
    let m = Matrix4::from_translation(mid) * rot
        * Matrix4::from_nonuniform_scale(len, radius, radius);
    DrawCmd { mesh_id: MESH_WHEEL, uniform: uniform_from_matrix(m, col) }
}

fn rotation_x_to(target: Vector3<f32>) -> Matrix4<f32> {
    use cgmath::InnerSpace;
    let x    = Vector3::new(1.0_f32, 0.0, 0.0);
    let dot  = x.dot(target).clamp(-1.0, 1.0);
    let cross = x.cross(target);
    let sin_a = cross.magnitude();
    if sin_a < 1e-6 {
        if dot > 0.0 { return Matrix4::from_scale(1.0); }
        return Matrix4::from_cols(
            Vector4::new(-1.0, 0.0,  0.0, 0.0),
            Vector4::new( 0.0, 1.0,  0.0, 0.0),
            Vector4::new( 0.0, 0.0, -1.0, 0.0),
            Vector4::new( 0.0, 0.0,  0.0, 1.0),
        );
    }
    let k = cross / sin_a;
    let (c, s, t) = (dot, sin_a, 1.0 - dot);
    let (kx, ky, kz) = (k.x, k.y, k.z);
    Matrix4::from_cols(
        Vector4::new(t*kx*kx+c,    t*kx*ky+s*kz, t*kx*kz-s*ky, 0.0),
        Vector4::new(t*kx*ky-s*kz, t*ky*ky+c,    t*ky*kz+s*kx, 0.0),
        Vector4::new(t*kx*kz+s*ky, t*ky*kz-s*kx, t*kz*kz+c,    0.0),
        Vector4::new(0.0,           0.0,           0.0,          1.0),
    )
}

// ── Per-tree deterministic PRNG (xorshift32, no deps) ────────────────────────

fn tree_rng(wx: f32, wz: f32) -> impl FnMut() -> f32 {
    // Seed from world position using bit manipulation for good distribution
    let ix = (wx * 73.1 + 1000.0) as u32;
    let iz = (wz * 91.7 + 1000.0) as u32;
    let mut s = ix.wrapping_mul(2654435761).wrapping_add(iz.wrapping_mul(2246822519));
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    if s == 0 { s = 1; }
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s as f32) / u32::MAX as f32
    }
}

// ── Leaf colour with per-cluster jitter ──────────────────────────────────────

fn leaf_col(rng: &mut impl FnMut() -> f32, base_brightness: f32) -> [f32; 4] {
    // Rich forest greens — matte (alpha 0.91) so leaves don't reflect the sky.
    // Range: deep shadow-green (0.14) up to sunlit leaf-green (0.48).
    let r  = rng() * 0.06;                                          // nearly no red
    let g  = (0.22 + base_brightness * 0.26 + (rng() - 0.5) * 0.10)
              .clamp(0.14, 0.50);
    let b  = 0.04 + rng() * 0.05;                                   // hint of blue
    [r, g, b, 0.91]  // 0.91 = matte, no sky env reflection
}

// ── Top-level ─────────────────────────────────────────────────────────────────

pub fn environment_draw_cmds(terrain: &TerrainManager, car_pos: Vector3<f32>) -> Vec<DrawCmd> {
    let mut c: Vec<DrawCmd> = Vec::with_capacity(8192);

    // Terrain chunks (one draw call per active chunk)
    for (_, cd) in terrain.chunks() {
        c.push(DrawCmd {
            mesh_id: MESH_TERRAIN_BASE + cd.mesh_slot,
            uniform: uniform_from_matrix(Matrix4::from_scale(1.0), TERRAIN_MAT),
        });
        if let Some(ws) = cd.water_slot {
            c.push(DrawCmd {
                mesh_id: MESH_WATER_BASE + ws,
                uniform: uniform_from_matrix(Matrix4::from_scale(1.0), WATER_MAT),
            });
        }
    }

    // Road surface (single pre-built mesh from spline)
    c.push(DrawCmd {
        mesh_id: MESH_ROAD,
        uniform: uniform_from_matrix(Matrix4::from_scale(1.0), ROAD),
    });

    // Forest (follows car position)
    scatter_trees(&mut c, terrain, car_pos);

    c
}

// ── Dynamic road-ahead generation ────────────────────────────────────────────
//
// Called every frame with the car's current world position.
// Finds where the car is on the road parametric curve, then emits:
//   • white centre-line dashes for the next ~200 m ahead
//   • yellow edge markers every 20 m (like real mountain road posts)
// These DrawCmds appear in the frame cmd list alongside the static road mesh.
//
// The "generates ahead" effect: only road within the forward window is submitted.

const LINE_WHITE: [f32; 4] = [0.92, 0.92, 0.90, 0.92]; // road marking, low-gloss
const MARKER_YLW: [f32; 4] = [0.82, 0.74, 0.06, 0.92]; // yellow post, low-gloss

pub fn road_ahead_cmds(car_pos: cgmath::Vector3<f32>, terrain: &TerrainManager) -> Vec<DrawCmd> {
    let road = &terrain.road;

    // Find nearest road point to car
    let (_, _, best_dist, _) = road.nearest_to(car_pos.x, car_pos.z);

    // Constants for stable world-space patterns.
    // All phases use absolute arc-length modulo so the pattern is fixed in
    // world-space and never slides as the car moves.
    const STRIDE:      f32 = 4.0;   // metres per iteration step
    const LOOK_AHEAD:  f32 = 600.0; // metres of road to draw ahead
    const DASH_PERIOD: f32 = 40.0;  // full on+off cycle length (20 m on, 20 m off)
    const DASH_ON:     f32 = 20.0;  // "on" portion of cycle
    const POST_EVERY:  f32 = 200.0; // metres between yellow edge posts

    // Quantise start to the absolute grid so the loop always hits the same
    // world-arc-length values regardless of where the car is.
    let start = (best_dist / STRIDE).floor() * STRIDE;
    let end   = (start + LOOK_AHEAD).min(road.total_dist());

    let mut out: Vec<DrawCmd> = Vec::with_capacity(200);
    let mut d0 = start;
    while d0 < end {
        let d1  = (d0 + STRIDE).min(road.total_dist());
        let (x0, z0) = road.position_at(d0);
        let (x1, z1) = road.position_at(d1);
        let y0 = terrain.height_at(x0, z0) + 0.16; // above road mesh (+0.12) + margin
        let y1 = terrain.height_at(x1, z1) + 0.16;

        let dx  = x1 - x0;
        let dz  = z1 - z0;
        let len = (dx * dx + dz * dz).sqrt().max(0.001);
        let rx  = -dz / len;
        let rz  =  dx / len;

        // Centre-line dashes: stable pattern based on absolute distance.
        let dash_phase = d0 % DASH_PERIOD;
        if dash_phase < DASH_ON {
            let mx = (x0 + x1) * 0.5;
            let mz = (z0 + z1) * 0.5;
            let my = (y0 + y1) * 0.5;
            out.push(oriented_cyl(
                Vector3::new(mx, my, mz),
                Vector3::new(x1, y1, z1),
                0.15, LINE_WHITE,
            ));
        }

        // Yellow edge posts at fixed 200 m world intervals (stable in world space).
        let post_phase = d0 % POST_EVERY;
        if post_phase < STRIDE {
            let post_h = 0.8_f32;
            let ox = rx * 6.8;
            let oz = rz * 6.8;
            let py  = terrain.height_at(x0 + ox, z0 + oz);
            let py2 = terrain.height_at(x0 - ox, z0 - oz);
            out.push(vcyl(x0 + ox,  py  + post_h * 0.5, z0 + oz,  0.06, post_h, MARKER_YLW));
            out.push(vcyl(x0 - ox,  py2 + post_h * 0.5, z0 - oz,  0.06, post_h, MARKER_YLW));
        }

        d0 += STRIDE;
    }

    out
}

// ── Road proximity check ──────────────────────────────────────────────────────

fn near_road(wx: f32, wz: f32, road: &crate::terrain::RoadSpline) -> bool {
    let (_, _, _, dist) = road.nearest_to(wx, wz);
    dist < 18.0 // 18 m exclusion around road centreline
}

// ── Jittered-grid tree scatter ────────────────────────────────────────────────

fn scatter_trees(c: &mut Vec<DrawCmd>, terrain: &TerrainManager, car_pos: Vector3<f32>) {
    // 20×20 grid, 24 m cells → 480 m coverage centred on the car.
    // Grid snaps to world-cell boundaries so trees never pop as the car moves.
    const HALF_RANGE: f32 = 240.0;
    const CELLS: usize    = 20;
    const CELL_SIZE: f32  = HALF_RANGE * 2.0 / CELLS as f32; // 24 m

    // Snap grid origin to world cell grid
    let grid_ox = ((car_pos.x - HALF_RANGE) / CELL_SIZE).floor() * CELL_SIZE;
    let grid_oz = ((car_pos.z - HALF_RANGE) / CELL_SIZE).floor() * CELL_SIZE;

    for ci in 0..CELLS {
        for cj in 0..CELLS {
            // World-space origin of this cell
            let cell_wx = grid_ox + ci as f32 * CELL_SIZE;
            let cell_wz = grid_oz + cj as f32 * CELL_SIZE;

            // Deterministic per-cell jitter derived from world grid index (not loop index)
            let ix = (cell_wx / CELL_SIZE).round() as i32;
            let iz = (cell_wz / CELL_SIZE).round() as i32;
            let mut h = (ix as u32).wrapping_mul(2654435761_u32)
                .wrapping_add((iz as u32).wrapping_mul(1013904223_u32));
            h ^= h << 13; h ^= h >> 17; h ^= h << 5;
            let jx = (h as f32 / u32::MAX as f32) * CELL_SIZE;
            h ^= h << 13; h ^= h >> 17; h ^= h << 5;
            let jz = (h as f32 / u32::MAX as f32) * CELL_SIZE;

            let wx = cell_wx + jx;
            let wz = cell_wz + jz;

            let base_y = terrain.height_at(wx, wz);
            if base_y == 0.0 { continue; } // chunk not loaded or lake floor

            // Cull: underwater or too shallow (lake shore buffer)
            if base_y < WATER_Y + 1.2 { continue; }

            // Cull: above treeline (rock/snow above 44 m)
            if base_y > 44.0 { continue; }

            // Cull: steep slope
            let n = terrain.normal_at(wx, wz);
            let slope = 1.0 - n[1]; // 0 = flat, 1 = vertical
            if slope > 0.55 { continue; }

            // Cull: near road
            if near_road(wx, wz, &terrain.road) { continue; }

            // Per-tree decisions use position-based rng (stable across frames)
            let mut rng = tree_rng(wx, wz);

            // Species: altitude-based mix
            let is_conifer = if base_y < 10.0 {
                rng() < 0.15 // mostly deciduous near water
            } else if base_y < 22.0 {
                rng() < 0.55 // mixed forest
            } else {
                rng() < 0.90 // mostly conifer on upper slopes
            };

            let scale = 0.85 + rng() * 0.35;

            // LOD: simplified geometry for trees > 180 m from car
            let dist_sq = (wx - car_pos.x).powi(2) + (wz - car_pos.z).powi(2);
            let full_detail = dist_sq < 180.0 * 180.0;

            if is_conifer {
                draw_conifer(c, wx, base_y, wz, scale, full_detail);
            } else {
                draw_deciduous(c, wx, base_y, wz, scale, full_detail);
            }
        }
    }
}

// ── Conifer ───────────────────────────────────────────────────────────────────
//
// Structure:
//   root-flare vcyl + 1–2 trunk segments (slight lean)
//   5–6 MESH_CONE tiers (bottom → top, narrowing)
//   3 branches per lower whorl: oriented_cyl bark + small vcone tip

fn draw_conifer(c: &mut Vec<DrawCmd>, wx: f32, base_y: f32, wz: f32, s: f32, full: bool) {
    let mut rng = tree_rng(wx, wz);

    let trunk_h = (5.5 + rng() * 3.5) * s;
    let lean_x  = (rng() - 0.5) * 0.18;
    let lean_z  = (rng() - 0.5) * 0.18;

    let top_x   = wx + lean_x * trunk_h;
    let top_z   = wz + lean_z * trunk_h;
    let top_y   = base_y + trunk_h;

    // Root flare
    c.push(vcyl(wx, base_y + 0.45 * s, wz, 0.30 * s, 0.90 * s, BARK));
    // Trunk
    c.push(oriented_cyl(
        Vector3::new(wx, base_y + 0.9 * s, wz),
        Vector3::new(top_x, top_y, top_z),
        0.16 * s, BARK,
    ));

    let n_tiers: usize = if full { 5 + (rng() * 2.0) as usize } else { 3 };
    // Phyllotactic offset per tier (~137.5° = golden angle)
    let golden = 2.399_f32;
    let mut branch_angle_accum = rng() * TAU;

    for tier in 0..n_tiers {
        let frac = tier as f32 / (n_tiers - 1).max(1) as f32;
        let tier_y = base_y + trunk_h * (0.22 + frac * 0.72);
        let tier_cx = wx + lean_x * trunk_h * (0.22 + frac * 0.72);
        let tier_cz = wz + lean_z * trunk_h * (0.22 + frac * 0.72);

        let tier_r = (1.90 - frac * 1.50) * s;
        let tier_h = (1.60 - frac * 0.80) * s;

        let col = leaf_col(&mut rng, 1.0 - frac * 0.5);
        c.push(vcone(tier_cx, tier_y, tier_cz, tier_r.max(0.25 * s), tier_h.max(0.4 * s), col));

        if !full || frac > 0.65 { continue; } // lower whorls only get branches

        let n_branches: usize = 3;
        for _ in 0..n_branches {
            branch_angle_accum += golden;
            let angle = branch_angle_accum;
            let reach = tier_r * (0.55 + rng() * 0.35);
            let bx = tier_cx + angle.cos() * reach;
            let bz = tier_cz + angle.sin() * reach;
            let by = tier_y - reach * 0.22 - rng() * reach * 0.08; // slight droop

            let base_pt = Vector3::new(tier_cx, tier_y, tier_cz);
            let tip_pt  = Vector3::new(bx, by, bz);
            c.push(oriented_cyl(base_pt, tip_pt, 0.055 * s, BARK));

            let bb = 0.5 + rng() * 0.4;
            let bcol = leaf_col(&mut rng, bb);
            c.push(vcone(bx, by + reach * 0.15, bz, reach * 0.32, reach * 0.55, bcol));
        }
    }
}

// ── Deciduous ─────────────────────────────────────────────────────────────────
//
// Structure:
//   2 trunk segments with slight lean + root flare
//   3 primary branches (120° apart + jitter) with gravity droop
//     2 secondary branches each: oriented_cyl + sphere cluster
//     1 sphere at primary tip

fn draw_deciduous(c: &mut Vec<DrawCmd>, wx: f32, base_y: f32, wz: f32, s: f32, full: bool) {
    let mut rng = tree_rng(wx + 1000.0, wz + 1000.0); // different seed than conifer

    let trunk_h  = (3.2 + rng() * 2.0) * s;
    let lean_x   = (rng() - 0.5) * 0.25;
    let lean_z   = (rng() - 0.5) * 0.25;

    let mid = Vector3::new(wx + lean_x * trunk_h * 0.4, base_y + trunk_h * 0.4, wz + lean_z * trunk_h * 0.4);
    let top = Vector3::new(wx + lean_x * trunk_h,       base_y + trunk_h,        wz + lean_z * trunk_h);

    // Root flare
    c.push(vcyl(wx, base_y + 0.35 * s, wz, 0.28 * s, 0.70 * s, BARK_D));
    // Two trunk segments
    c.push(oriented_cyl(Vector3::new(wx, base_y + 0.7 * s, wz), mid, 0.22 * s, BARK_D));
    c.push(oriented_cyl(mid, top, 0.17 * s, BARK_D));

    // LOD: far trees get just one central canopy ball
    if !full {
        let r = (1.8 + rng() * 0.8) * s;
        let bright = 0.6 + rng() * 0.3;
        let col = leaf_col(&mut rng, bright);
        c.push(vsphere(top.x, top.y + r * 0.5, top.z, r, col));
        return;
    }

    // 3 primary branches at ~120° phyllotactic spacing
    let golden = 2.399_f32;
    let mut base_angle = rng() * TAU;

    for pi in 0..3usize {
        base_angle += golden;
        let _ = pi;

        let pitch_sin = 0.55 + rng() * 0.30;
        let pitch_cos = (1.0 - pitch_sin * pitch_sin).sqrt();
        let dir1 = Vector3::new(
            base_angle.cos() * pitch_sin,
            pitch_cos,
            base_angle.sin() * pitch_sin,
        );
        let reach1 = (2.2 + rng() * 1.4) * s;
        // Gravity droop: tip sags slightly
        let droop = Vector3::new(0.0, -reach1 * 0.08, 0.0);
        let tip1 = top + dir1 * reach1 + droop;

        c.push(oriented_cyl(top, tip1, 0.12 * s, BARK_D));

        // Foliage sphere at primary tip
        let r1 = (0.9 + rng() * 0.5) * s;
        let b1 = 0.4 + rng() * 0.5;
        c.push(vsphere(tip1.x, tip1.y, tip1.z, r1, leaf_col(&mut rng, b1)));

        // 2 secondary branches
        for si in 0..2usize {
            let a2 = base_angle + (si as f32 - 0.5) * 1.35 + rng() * 0.5;
            let p2  = 0.58 + rng() * 0.25;
            let dir2 = Vector3::new(
                a2.cos() * p2,
                (1.0 - p2 * p2).sqrt() * (if si == 0 { 1.0 } else { 0.85 }),
                a2.sin() * p2,
            );
            let reach2 = reach1 * (0.45 + rng() * 0.25);
            let droop2 = Vector3::new(0.0, -reach2 * 0.10, 0.0);
            let tip2 = tip1 + dir2 * reach2 + droop2;

            c.push(oriented_cyl(tip1, tip2, 0.07 * s, BARK_D));

            let r2 = (0.65 + rng() * 0.35) * s;
            let b2 = 0.5 + rng() * 0.4;
            c.push(vsphere(tip2.x, tip2.y, tip2.z, r2, leaf_col(&mut rng, b2)));
        }
    }

    // Central canopy sphere to fill the silhouette
    let cr = (1.2 + rng() * 0.4) * s;
    c.push(vsphere(top.x, top.y + cr * 0.4, top.z, cr, leaf_col(&mut rng, 0.6)));
}
