// scene/grass.rs — GPU-instanced static grass with noise-driven clustering.
//
// CRITICAL: vnoise() and hash2() here are exact CPU ports of the WGSL functions
// in shaders/basic.wgsl.  Both systems use the same hash, frequencies, and
// thresholds, so grass blades appear precisely where the terrain shader draws
// green and are absent where it draws bare dirt.
//
// Density uses two noise layers:
//   Layer 1 — lushness (~71 m scale): large patches of meadow vs bare ground.
//   Layer 2 — cluster  (~9.5 m scale): tight bunches within lush areas.
// Per-cluster height/tint variation comes from a third layer at ~31 m.

use bytemuck::{Pod, Zeroable};
use std::f32::consts::TAU;
use crate::terrain::{TerrainManager, WATER_Y};

// ── Tunable constants ─────────────────────────────────────────────────────────

pub const GRASS_SPACING:       f32 = 0.80;
pub const GRASS_DENSE_RADIUS:  f32 = 35.0;
pub const GRASS_FADE_RADIUS:   f32 = 90.0;
pub const BLADE_HEIGHT_MIN:    f32 = 0.22;
pub const BLADE_HEIGHT_MAX:    f32 = 0.55;
pub const GRASS_MAX_SLOPE_COS: f32 = 0.62;
pub const GRASS_MIN_Y:         f32 = 3.0;
pub const GRASS_MAX_Y:         f32 = 44.0;

// These match the smoothstep(lower, upper, noise) bounds used in basic.wgsl's
// terrain branch.  Raising LUSH_GATE thins grass overall; raising CLUST_GATE
// widens the gaps between clusters.
const LUSH_GATE:  f32 = 0.28;   // = basic.wgsl smoothstep lower bound for lush_n
const CLUST_GATE: f32 = 0.28;   // = basic.wgsl smoothstep lower bound for clust_n

/// Metres from road centreline inside which no grass spawns.
const ROAD_GRASS_EXCL: f32 = 10.0;

// ── GrassInstance (32 bytes) ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GrassInstance {
    pub world_pos: [f32; 3],
    pub height:    f32,
    pub rotation:  f32,
    pub tint:      f32,
    pub _pad:      [f32; 2],
}

// ── Noise — exact CPU port of WGSL hash2 / vnoise ────────────────────────────
//
// WGSL `fract(x)` is defined as `x - floor(x)`, always in [0, 1).
// Rust `f32::fract()` keeps the sign, so we replicate the WGSL behaviour.

#[inline(always)]
fn wfract(x: f32) -> f32 { x - x.floor() }

#[inline(always)]
fn hash2(px: f32, pz: f32) -> f32 {
    // Matches WGSL: fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123)
    let dot = px * 127.1 + pz * 311.7;
    wfract(dot.sin() * 43758.5453123)
}

/// Value noise, [0, 1].  Identical to `vnoise()` in basic.wgsl.
pub fn vnoise(px: f32, pz: f32) -> f32 {
    let ix = px.floor();
    let iz = pz.floor();
    let fx = px - ix;   // wfract(px) — same result since floor is already subtracted
    let fz = pz - iz;
    // Smoothstep blend weights
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uz = fz * fz * (3.0 - 2.0 * fz);
    let a = hash2(ix,       iz      );
    let b = hash2(ix + 1.0, iz      );
    let c = hash2(ix,       iz + 1.0);
    let d = hash2(ix + 1.0, iz + 1.0);
    let x0 = a + (b - a) * ux;
    let x1 = c + (d - c) * ux;
    x0 + (x1 - x0) * uz
}

/// Inline smoothstep (same formula as WGSL built-in).
#[inline(always)]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Grass instance builder ────────────────────────────────────────────────────

pub fn build_grass_instances(
    terrain: &TerrainManager,
    car_pos: cgmath::Vector3<f32>,
) -> Vec<GrassInstance> {
    // Pre-filter road spline samples to the rebuild area.
    let search_r  = GRASS_FADE_RADIUS + ROAD_GRASS_EXCL + 4.0;
    let search_r2 = search_r * search_r;
    let road_nearby: Vec<[f32; 2]> = terrain.road.positions.iter()
        .filter(|&&[rx, rz]| {
            let dx = rx - car_pos.x;
            let dz = rz - car_pos.z;
            dx * dx + dz * dz < search_r2
        })
        .copied()
        .collect();
    let excl2 = ROAD_GRASS_EXCL * ROAD_GRASS_EXCL;

    let mut instances = Vec::new();

    let fade  = GRASS_FADE_RADIUS;
    let dense = GRASS_DENSE_RADIUS;
    let step  = GRASS_SPACING;

    let min_ix = ((car_pos.x - fade) / step).floor() as i32;
    let max_ix = ((car_pos.x + fade) / step).ceil()  as i32;
    let min_iz = ((car_pos.z - fade) / step).floor() as i32;
    let max_iz = ((car_pos.z + fade) / step).ceil()  as i32;

    for ix in min_ix..=max_ix {
        for iz in min_iz..=max_iz {
            let gx = ix as f32 * step;
            let gz = iz as f32 * step;

            let dx   = gx - car_pos.x;
            let dz   = gz - car_pos.z;
            let dist = (dx * dx + dz * dz).sqrt();
            if dist > fade { continue; }

            let dist_density = if dist < dense {
                1.0_f32
            } else {
                1.0 - (dist - dense) / (fade - dense)
            };

            // Per-cell deterministic RNG (xorshift32, seeded from grid position).
            let seed = (ix as u32)
                .wrapping_mul(2_654_435_761_u32)
                .wrapping_add((iz as u32).wrapping_mul(1_013_904_223_u32));
            let mut h = seed;
            let mut rng = move || -> f32 {
                h ^= h << 13;
                h ^= h >> 17;
                h ^= h << 5;
                h as f32 / u32::MAX as f32
            };

            // Jitter first so all sampling is at the actual blade world position.
            let wx = gx + (rng() - 0.5) * step * 0.9;
            let wz = gz + (rng() - 0.5) * step * 0.9;

            // ── Lushness gate (~71 m features) ───────────────────────────────
            // Same formula as basic.wgsl: lush_n = vnoise(xz*0.014)*0.60 + vnoise(xz*0.028)*0.40
            let lush_n = vnoise(wx * 0.014, wz * 0.014) * 0.60
                       + vnoise(wx * 0.028, wz * 0.028) * 0.40;
            if lush_n < LUSH_GATE { continue; }

            // ── Cluster gate (~9.5 m features) ───────────────────────────────
            // Same formula as basic.wgsl: clust_n = vnoise(xz*0.105)
            let clust_n = vnoise(wx * 0.105, wz * 0.105);
            if clust_n < CLUST_GATE { continue; }

            // ── Terrain lookup (after cheap noise gates) ──────────────────────
            let wy = terrain.height_at(wx, wz);
            if wy < GRASS_MIN_Y || wy > GRASS_MAX_Y { continue; }

            // ── Slope check ───────────────────────────────────────────────────
            let normal = terrain.normal_at(wx, wz);
            if normal[1] < GRASS_MAX_SLOPE_COS { continue; }

            // ── Road exclusion ────────────────────────────────────────────────
            let on_road = road_nearby.iter().any(|&[rx, rz]| {
                let ddx = wx - rx;
                let ddz = wz - rz;
                ddx * ddx + ddz * ddz < excl2
            });
            if on_road { continue; }

            // ── Final density roll ────────────────────────────────────────────
            // Weights mirror the WGSL grass_cover = smoothstep(L)*smoothstep(C).
            let lush_w  = smoothstep(LUSH_GATE,  0.68, lush_n);
            let clust_w = smoothstep(CLUST_GATE, 0.64, clust_n);

            let alt_fade    = 1.0 - ((wy - (GRASS_MAX_Y - 9.0)).max(0.0) / 9.0).min(1.0);
            let above_water = (wy - WATER_Y).max(0.0);
            let water_bonus = (1.0 - (above_water / 5.0).min(1.0)) * 0.35;

            let total_density =
                (dist_density * lush_w * clust_w * alt_fade + water_bonus).min(1.0);
            if rng() > total_density { continue; }

            // ── Per-cluster variation (~31 m features) ────────────────────────
            // Offset of +73.0 keeps this independent of the lush/cluster fields
            // that also sample near x*0.028.
            let height_n = vnoise(wx * 0.032,        wz * 0.032       );
            let tint_n   = vnoise(wx * 0.028 + 73.0, wz * 0.028 + 41.0);

            let ht_scale = 0.65 + height_n * 0.70;   // 0.65× to 1.35×
            let base_h   = BLADE_HEIGHT_MIN + rng() * (BLADE_HEIGHT_MAX - BLADE_HEIGHT_MIN);
            let final_h  = (base_h * ht_scale)
                .clamp(BLADE_HEIGHT_MIN * 0.5, BLADE_HEIGHT_MAX * 1.45);

            // 55% cluster tint bias, 45% per-blade randomness.
            let tint = rng() * 0.45 + tint_n * 0.55;

            instances.push(GrassInstance {
                world_pos: [wx, wy, wz],
                height:    final_h,
                rotation:  rng() * TAU,
                tint,
                _pad:      [0.0; 2],
            });
        }
    }

    instances
}
