// terrain/mod.rs — chunk-based streaming terrain + infinite road spline.
//
// World is divided into CHUNK_CELLS×CHUNK_CELLS quad chunks, each covering
// CHUNK_WORLD × CHUNK_WORLD metres.  TerrainManager maintains a 5×5 window of
// active chunks centred on the player; new chunks are generated on demand using
// absolute-coordinate Perlin noise so boundaries are always seamless.
//
// The road is a noise-driven spline (heading changes from Perlin curvature),
// not a closed loop, so driving forward always reveals new terrain.

use crate::renderer::buffer::Vertex;
use std::collections::HashMap;

// ── Chunk constants ───────────────────────────────────────────────────────────

pub const CHUNK_CELLS: usize = 64;
pub const CHUNK_VERTS: usize = CHUNK_CELLS + 1;           // 65 vertices per edge
pub const CHUNK_WORLD: f32   = 256.0;                     // metres per chunk
pub const CHUNK_STEP:  f32   = CHUNK_WORLD / CHUNK_CELLS as f32; // 4 m/cell

pub const WATER_Y: f32 = 2.5;
pub const MAX_H:   f32 = 80.0;

pub const VIEW_RADIUS: i32   = 2;
pub const VIEW_DIAM:   i32   = VIEW_RADIUS * 2 + 1; // 5
pub const MAX_CHUNKS:  usize = (VIEW_DIAM * VIEW_DIAM) as usize; // 25

/// Backwards-compat alias for physics (per-chunk heightfield scale).
pub const WORLD: f32 = CHUNK_WORLD;

// ── Road spline constants ─────────────────────────────────────────────────────

const ROAD_HALF:    f32   = 7.5;        // half-width of paved surface (m)
const BLEND_OUT:    f32   = ROAD_HALF * 5.0; // outer edge of shoulder blend
pub const SPLINE_STEP:  f32   = 4.0;   // metres between spline samples
const SPLINE_TOTAL: f32   = 10_000.0;  // 10 km total
pub const N_SPLINE: usize = (SPLINE_TOTAL / SPLINE_STEP) as usize + 1; // 2501

// ── Road spline ────────────────────────────────────────────────────────────────

pub struct RoadSpline {
    pub positions: Vec<[f32; 2]>, // [x, z] every SPLINE_STEP metres
    pub heights:   Vec<f32>,       // smoothed road height at each sample
}

impl RoadSpline {
    pub fn generate(noise: &Perlin) -> Self {
        let mut positions: Vec<[f32; 2]> = Vec::with_capacity(N_SPLINE);

        // Start south-west of origin, heading roughly east-north-east.
        let mut x: f32   = -100.0;
        let mut z: f32   = -180.0;
        let mut heading: f32 = 0.10; // radians from +X

        for _ in 0..N_SPLINE {
            positions.push([x, z]);
            let curv = noise.noise(x * 0.0015, z * 0.0015) * 0.13;
            heading += curv;
            x += heading.cos() * SPLINE_STEP;
            z += heading.sin() * SPLINE_STEP;
        }

        // Raw heights at each spline position (base noise, no road carving)
        let raw_h: Vec<f32> = positions.iter()
            .map(|&[px, pz]| base_height(noise, px, pz))
            .collect();

        // Wide moving-average smooth for gentle grade
        let win = 80usize;
        let mut heights: Vec<f32> = (0..N_SPLINE).map(|i| {
            let lo = i.saturating_sub(win / 2);
            let hi = (i + win / 2).min(N_SPLINE - 1);
            let cnt = hi - lo + 1;
            raw_h[lo..=hi].iter().sum::<f32>() / cnt as f32
        }).collect();

        // Grade limiter: max 12% slope per segment
        let max_dh = SPLINE_STEP * 0.12;
        for _ in 0..4 {
            for i in 1..N_SPLINE {
                let dh = heights[i] - heights[i - 1];
                if dh.abs() > max_dh {
                    heights[i] = heights[i - 1] + dh.signum() * max_dh;
                }
            }
            for i in (0..N_SPLINE - 1).rev() {
                let dh = heights[i] - heights[i + 1];
                if dh.abs() > max_dh {
                    heights[i] = heights[i + 1] + dh.signum() * max_dh;
                }
            }
        }

        Self { positions, heights }
    }

    /// World XZ position at distance `d` metres along the spline.
    pub fn position_at(&self, d: f32) -> (f32, f32) {
        let idx = ((d / SPLINE_STEP) as usize).min(self.positions.len() - 2);
        let t   = (d / SPLINE_STEP) - idx as f32;
        let [x0, z0] = self.positions[idx];
        let [x1, z1] = self.positions[idx + 1];
        (x0 + (x1 - x0) * t, z0 + (z1 - z0) * t)
    }

    /// Road surface height at distance `d`.
    pub fn height_at(&self, d: f32) -> f32 {
        let idx = ((d / SPLINE_STEP) as usize).min(self.heights.len() - 2);
        let t   = (d / SPLINE_STEP) - idx as f32;
        self.heights[idx] * (1.0 - t) + self.heights[idx + 1] * t
    }

    /// Unit forward vector in XZ at distance `d`.
    pub fn direction_at(&self, d: f32) -> (f32, f32) {
        let idx = ((d / SPLINE_STEP) as usize).min(self.positions.len() - 2);
        let [x0, z0] = self.positions[idx];
        let [x1, z1] = self.positions[idx + 1];
        let dx = x1 - x0;
        let dz = z1 - z0;
        let len = (dx * dx + dz * dz).sqrt().max(1e-6);
        (dx / len, dz / len)
    }

    pub fn total_dist(&self) -> f32 { (N_SPLINE - 1) as f32 * SPLINE_STEP }

    /// Find nearest spline index to world point (wx, wz).
    /// Returns (road_x, road_z, dist_along, dist_perpendicular).
    pub fn nearest_to(&self, wx: f32, wz: f32) -> (f32, f32, f32, f32) {
        // Coarse pass every 8 samples
        let mut best_d2 = f32::MAX;
        let mut best_i  = 0usize;
        for i in (0..self.positions.len()).step_by(8) {
            let [rx, rz] = self.positions[i];
            let d2 = (wx - rx) * (wx - rx) + (wz - rz) * (wz - rz);
            if d2 < best_d2 { best_d2 = d2; best_i = i; }
        }
        // Fine pass ±16 samples around best
        let lo = best_i.saturating_sub(16);
        let hi = (best_i + 16).min(self.positions.len() - 1);
        for i in lo..=hi {
            let [rx, rz] = self.positions[i];
            let d2 = (wx - rx) * (wx - rx) + (wz - rz) * (wz - rz);
            if d2 < best_d2 { best_d2 = d2; best_i = i; }
        }
        let [rx, rz] = self.positions[best_i];
        (rx, rz, best_i as f32 * SPLINE_STEP, best_d2.sqrt())
    }

    /// Build a terrain-conforming road surface mesh from the spline.
    pub fn build_road_mesh(&self) -> (Vec<Vertex>, Vec<u16>) {
        const N_SEG:   usize = 2499; // must be < N_SPLINE-1
        const N_CROSS: usize = 9;    // cross-section vertices (8 quads)
        const HALF_W:  f32   = 6.0;  // half road width

        let n_seg = N_SEG.min(self.positions.len().saturating_sub(2));
        let mut verts: Vec<Vertex> = Vec::with_capacity((n_seg + 1) * N_CROSS);
        let mut idx:   Vec<u16>   = Vec::with_capacity(n_seg * (N_CROSS - 1) * 6);

        for i in 0..=n_seg {
            let d = i as f32 * SPLINE_STEP;
            let (cx, cz) = self.position_at(d);
            let (fw, fz) = self.direction_at(d);
            let rx = -fz; // right perpendicular
            let rz =  fw;

            for k in 0..N_CROSS {
                let s    = k as f32 / (N_CROSS - 1) as f32;
                let side = (-1.0 + s * 2.0) * HALF_W;
                let wx   = cx + rx * side;
                let wz   = cz + rz * side;
                let wy   = self.height_at(d) + 0.12;
                verts.push(Vertex {
                    position: [wx, wy, wz],
                    normal:   [0.0, 1.0, 0.0],
                    uv:       [s, d * 0.025],
                });
            }
        }

        for i in 0..n_seg {
            for k in 0..(N_CROSS - 1) {
                let a = (i       * N_CROSS + k)     as u16;
                let b = (i       * N_CROSS + k + 1) as u16;
                let c = ((i + 1) * N_CROSS + k)     as u16;
                let d = ((i + 1) * N_CROSS + k + 1) as u16;
                idx.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }

        (verts, idx)
    }
}

// ── Base terrain height (no road carving) ─────────────────────────────────────

/// Ridged multifractal: produces sharp mountain ridges and peaks.
/// Output range ≈ [0, 1].
fn ridged_fbm(noise: &Perlin, x: f32, z: f32, octaves: u32) -> f32 {
    let (mut val, mut amp, mut freq, mut total) = (0.0_f32, 0.5, 1.0, 0.0_f32);
    for _ in 0..octaves {
        let n = 1.0 - noise.noise(x * freq, z * freq).abs();
        val   += amp * n * n; // squared for sharper ridges
        total += amp;
        amp   *= 0.5;
        freq  *= 2.0;
    }
    val / total
}

fn base_height(noise: &Perlin, wx: f32, wz: f32) -> f32 {
    // Domain warp: displace sample coordinates by a low-freq noise field (±70 m).
    // Breaks up the regularity of FBM, creating organic flowing valley/ridge shapes.
    let wx_w = wx + noise.fbm(wx / 380.0 + 1.7, wz / 380.0 + 9.2, 2) * 70.0;
    let wz_w = wz + noise.fbm(wx / 380.0 + 8.3, wz / 380.0 + 2.8, 2) * 70.0;

    // Raw coarse FBM in [-0.5, 0.5]: negative → valley/lake, positive → upland.
    let coarse = noise.fbm(wx_w / 200.0 + 3.1, wz_w / 200.0 + 0.7, 5);

    // Ridged multifractal: sharp ridges and peaks [0, 1], evaluated in warped space.
    let ridge = ridged_fbm(noise, wx_w / 150.0 + 5.5, wz_w / 150.0 + 2.3, 5);

    // Fine detail: small bumps that don't depend on the warp (avoid swimming on chunk boundaries).
    let detail = noise.fbm(wx / 66.7 + 14.2, wz / 66.7 + 7.9, 3) * 0.15;

    // Ridges only appear on uplands (coarse > 0); valleys stay smooth.
    let upland = coarse.max(0.0) * 2.0; // 0 in valley → up to 1 on plateau
    let shape  = coarse + ridge * upland * 0.60 + detail;

    // Map to metres:
    //   coarse ≈ -0.5 → shape ≈ -0.5 → height = -22.5 + 6 = -16.5 → 0 m (lake)
    //   coarse ≈  0.0 → shape ≈  0.0 → height =   0.0 + 6 =  6 m   (grassy plain)
    //   coarse ≈  0.4, ridge 0.9 → shape ≈ 1.0 → height = 55 + 6 = 61 m (peak)
    (shape * 55.0 + 6.0).max(0.0)
}

// ── Hydraulic erosion ─────────────────────────────────────────────────────────

/// Simple single-pass particle erosion: 400 droplets × 30 steps each.
/// Erodes steep slopes and deposits sediment in flatter areas, softening
/// sharp edges and carving small gullies.
fn hydraulic_erosion(data: &mut [f32], seed: u32) {
    let mut rng = seed ^ 0xDEAD_BEEF;

    for _ in 0..400 {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        let mut c = (rng as usize) % CHUNK_CELLS;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        let mut r = (rng as usize) % CHUNK_CELLS;

        let mut sediment = 0.0_f32;

        for _ in 0..30 {
            let idx = r * CHUNK_VERTS + c;
            let h   = data[idx];

            // Cardinal neighbor heights (clamped to grid edge)
            let hl = if c > 0           { data[r * CHUNK_VERTS + c - 1]     } else { h };
            let hr = if c < CHUNK_CELLS { data[r * CHUNK_VERTS + c + 1]     } else { h };
            let hd = if r > 0           { data[(r - 1) * CHUNK_VERTS + c]   } else { h };
            let hu = if r < CHUNK_CELLS { data[(r + 1) * CHUNK_VERTS + c]   } else { h };

            // Steepest descent
            let mut best_drop = 0.0_f32;
            let mut dc: i32   = 0;
            let mut dr: i32   = 0;
            if h - hl > best_drop { best_drop = h - hl; dc = -1; dr =  0; }
            if h - hr > best_drop { best_drop = h - hr; dc =  1; dr =  0; }
            if h - hd > best_drop { best_drop = h - hd; dc =  0; dr = -1; }
            if h - hu > best_drop { best_drop = h - hu; dc =  0; dr =  1; }

            if best_drop <= 0.0 {
                data[idx] += sediment * 0.5;
                sediment  *= 0.5;
                break;
            }

            let erode   = (best_drop * 0.12).min(0.10);
            data[idx]  -= erode;
            sediment   += erode;

            // Advance droplet
            let nc = (c as i32 + dc) as usize;
            let nr = (r as i32 + dr) as usize;
            if nc >= CHUNK_VERTS || nr >= CHUNK_VERTS { break; }
            c = nc; r = nr;

            // Deposit a fraction at new cell
            let deposit    = sediment * 0.06;
            sediment      -= deposit;
            data[r * CHUNK_VERTS + c] += deposit;
        }

        // Drop remaining sediment
        data[r * CHUNK_VERTS + c] += sediment;
    }
}

// ── Chunk heightmap generation ────────────────────────────────────────────────

/// Generate heights for the chunk at chunk-grid coords (cx, cz).
/// Returns a flat Vec<f32> of length CHUNK_VERTS × CHUNK_VERTS,
/// row-major (index = row * CHUNK_VERTS + col, row→Z col→X).
pub fn generate_chunk_heights(cx: i32, cz: i32, noise: &Perlin, road: &RoadSpline) -> Vec<f32> {
    let ox = cx as f32 * CHUNK_WORLD;
    let oz = cz as f32 * CHUNK_WORLD;

    let mut data = vec![0.0_f32; CHUNK_VERTS * CHUNK_VERTS];

    for row in 0..CHUNK_VERTS {
        for col in 0..CHUNK_VERTS {
            let wx = ox + col as f32 * CHUNK_STEP;
            let wz = oz + row as f32 * CHUNK_STEP;
            data[row * CHUNK_VERTS + col] = base_height(noise, wx, wz);
        }
    }

    // Hydraulic erosion: deterministic per-chunk, shapes gullies and softens ridges.
    // Applied before road carving so the carved road remains pristine.
    let erosion_seed = (cx.wrapping_mul(1_000_003_i32)
        .wrapping_add(cz.wrapping_mul(997_i32))) as u32;
    hydraulic_erosion(&mut data, erosion_seed);

    // Road carving: only consider road samples near this chunk
    let margin   = BLEND_OUT + CHUNK_STEP;
    let lo_x = ox - margin;
    let hi_x = ox + CHUNK_WORLD + margin;
    let lo_z = oz - margin;
    let hi_z = oz + CHUNK_WORLD + margin;

    let nearby: Vec<(usize, [f32; 2])> = road.positions.iter()
        .enumerate()
        .filter(|(_, &[rx, rz])| rx >= lo_x && rx <= hi_x && rz >= lo_z && rz <= hi_z)
        .map(|(i, &p)| (i, p))
        .collect();

    if nearby.is_empty() { return data; }

    for row in 0..CHUNK_VERTS {
        for col in 0..CHUNK_VERTS {
            let wx = ox + col as f32 * CHUNK_STEP;
            let wz = oz + row as f32 * CHUNK_STEP;

            let mut min_d2 = f32::MAX;
            let mut near_i = 0usize;
            for &(i, [rx, rz]) in &nearby {
                let d2 = (wx - rx) * (wx - rx) + (wz - rz) * (wz - rz);
                if d2 < min_d2 { min_d2 = d2; near_i = i; }
            }
            let min_d  = min_d2.sqrt();
            let road_h = road.heights[near_i];

            if min_d < ROAD_HALF {
                data[row * CHUNK_VERTS + col] = road_h; // perfectly flat
            } else if min_d < BLEND_OUT {
                let t      = (min_d - ROAD_HALF) / (BLEND_OUT - ROAD_HALF);
                let smooth = t * t * (3.0 - 2.0 * t); // smoothstep
                data[row * CHUNK_VERTS + col] =
                    road_h * (1.0 - smooth) + data[row * CHUNK_VERTS + col] * smooth;
            }
        }
    }

    data
}

// ── Chunk mesh builders ───────────────────────────────────────────────────────

pub fn build_chunk_mesh(cx: i32, cz: i32, heights: &[f32]) -> (Vec<Vertex>, Vec<u16>) {
    let ox = cx as f32 * CHUNK_WORLD;
    let oz = cz as f32 * CHUNK_WORLD;

    let mut verts: Vec<Vertex> = Vec::with_capacity(CHUNK_VERTS * CHUNK_VERTS);
    let mut idx:   Vec<u16>   = Vec::with_capacity(CHUNK_CELLS * CHUNK_CELLS * 6);

    for row in 0..CHUNK_VERTS {
        for col in 0..CHUNK_VERTS {
            let wx = ox + col as f32 * CHUNK_STEP;
            let wz = oz + row as f32 * CHUNK_STEP;
            let wy = heights[row * CHUNK_VERTS + col];

            let hl = if col > 0             { heights[row * CHUNK_VERTS + col - 1] } else { wy };
            let hr = if col < CHUNK_VERTS-1 { heights[row * CHUNK_VERTS + col + 1] } else { wy };
            let hd = if row > 0             { heights[(row-1) * CHUNK_VERTS + col] } else { wy };
            let hu = if row < CHUNK_VERTS-1 { heights[(row+1) * CHUNK_VERTS + col] } else { wy };

            let nx = (hl - hr) / (2.0 * CHUNK_STEP);
            let nz = (hd - hu) / (2.0 * CHUNK_STEP);
            let ny = 1.0_f32;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();

            verts.push(Vertex {
                position: [wx, wy, wz],
                normal:   [nx / len, ny / len, nz / len],
                uv:       [col as f32 / CHUNK_CELLS as f32, row as f32 / CHUNK_CELLS as f32],
            });
        }
    }

    for row in 0..CHUNK_CELLS {
        for col in 0..CHUNK_CELLS {
            let a = (row       * CHUNK_VERTS + col)     as u16;
            let b = (row       * CHUNK_VERTS + col + 1) as u16;
            let c = ((row + 1) * CHUNK_VERTS + col)     as u16;
            let d = ((row + 1) * CHUNK_VERTS + col + 1) as u16;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    (verts, idx)
}

pub fn build_chunk_water_mesh(cx: i32, cz: i32, heights: &[f32]) -> (Vec<Vertex>, Vec<u16>) {
    let ox = cx as f32 * CHUNK_WORLD;
    let oz = cz as f32 * CHUNK_WORLD;

    let mut verts: Vec<Vertex> = Vec::new();
    let mut idx:   Vec<u16>   = Vec::new();

    for row in 0..CHUNK_CELLS {
        for col in 0..CHUNK_CELLS {
            let h00 = heights[ row      * CHUNK_VERTS + col    ];
            let h10 = heights[ row      * CHUNK_VERTS + col + 1];
            let h01 = heights[(row + 1) * CHUNK_VERTS + col    ];
            let h11 = heights[(row + 1) * CHUNK_VERTS + col + 1];
            if (h00 + h10 + h01 + h11) * 0.25 >= WATER_Y { continue; }
            // Guard: buffer is sized for CHUNK_CELLS² quads × 4 verts each
            if verts.len() + 4 > CHUNK_CELLS * CHUNK_CELLS * 4 { break; }

            let base = verts.len() as u16;
            for &(dc, dr) in &[(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
                let wx = ox + (col + dc) as f32 * CHUNK_STEP;
                let wz = oz + (row + dr) as f32 * CHUNK_STEP;
                // Pack terrain height below water into uv.x so the fragment
                // shader can compute water depth = WATER_Y - terrain_h.
                let terrain_h = heights[(row + dr) * CHUNK_VERTS + (col + dc)];
                verts.push(Vertex {
                    position: [wx, WATER_Y, wz],
                    normal:   [0.0, 1.0, 0.0],
                    uv: [terrain_h, 0.0],
                });
            }
            idx.extend_from_slice(&[base, base+2, base+1, base+1, base+2, base+3]);
        }
    }

    (verts, idx)
}

// ── TerrainManager ────────────────────────────────────────────────────────────

pub struct ChunkData {
    pub heights:    Vec<f32>,
    pub mesh_slot:  usize,         // terrain_chunks[mesh_slot] in renderer
    pub water_slot: Option<usize>, // water_chunks[water_slot] in renderer (if has water)
}

pub struct TerrainManager {
    pub noise:  Perlin,
    pub road:   RoadSpline,
    chunks:     HashMap<(i32, i32), ChunkData>,
    center:     (i32, i32),
    free_terrain: Vec<usize>,
    free_water:   Vec<usize>,
}

impl TerrainManager {
    pub fn new() -> Self {
        let noise = Perlin::new(42);
        let road  = RoadSpline::generate(&noise);

        let free_terrain: Vec<usize> = (0..MAX_CHUNKS).rev().collect();
        let free_water:   Vec<usize> = (0..MAX_CHUNKS).rev().collect();

        let mut mgr = Self {
            noise, road,
            chunks: HashMap::new(),
            center: (i32::MIN, i32::MIN), // force first update
            free_terrain,
            free_water,
        };
        mgr.update_center(0, 0);
        mgr
    }

    pub fn world_to_chunk(wx: f32, wz: f32) -> (i32, i32) {
        ((wx / CHUNK_WORLD).floor() as i32,
         (wz / CHUNK_WORLD).floor() as i32)
    }

    /// Must be called each frame with the car position.
    /// Returns (added_keys, removed_keys) to let the caller update GPU and physics.
    pub fn update(&mut self, wx: f32, wz: f32) -> (Vec<(i32,i32)>, Vec<(i32,i32)>) {
        let (cx, cz) = Self::world_to_chunk(wx, wz);
        if cx == self.center.0 && cz == self.center.1 { return (vec![], vec![]); }
        self.update_center(cx, cz)
    }

    fn update_center(&mut self, cx: i32, cz: i32) -> (Vec<(i32,i32)>, Vec<(i32,i32)>) {
        self.center = (cx, cz);

        let needed: std::collections::HashSet<(i32,i32)> = (-VIEW_RADIUS..=VIEW_RADIUS)
            .flat_map(|dz| (-VIEW_RADIUS..=VIEW_RADIUS).map(move |dx| (cx + dx, cz + dz)))
            .collect();

        let to_remove: Vec<_> = self.chunks.keys()
            .filter(|k| !needed.contains(k))
            .cloned().collect();
        for key in &to_remove {
            if let Some(cd) = self.chunks.remove(key) {
                self.free_terrain.push(cd.mesh_slot);
                if let Some(ws) = cd.water_slot { self.free_water.push(ws); }
            }
        }

        let mut added = Vec::new();
        for key in needed {
            if self.chunks.contains_key(&key) { continue; }
            let Some(slot) = self.free_terrain.pop() else { continue };
            let heights = generate_chunk_heights(key.0, key.1, &self.noise, &self.road);
            let has_water = heights.iter().any(|&h| h < WATER_Y);
            let water_slot = if has_water { self.free_water.pop() } else { None };
            self.chunks.insert(key, ChunkData { heights, mesh_slot: slot, water_slot });
            added.push(key);
        }

        (added, to_remove)
    }

    pub fn chunk(&self, key: &(i32, i32)) -> Option<&ChunkData> { self.chunks.get(key) }
    pub fn chunks(&self) -> &HashMap<(i32,i32), ChunkData> { &self.chunks }

    pub fn height_at(&self, wx: f32, wz: f32) -> f32 {
        let cx = (wx / CHUNK_WORLD).floor() as i32;
        let cz = (wz / CHUNK_WORLD).floor() as i32;
        let Some(cd) = self.chunks.get(&(cx, cz)) else { return 0.0 };
        let lx    = wx - cx as f32 * CHUNK_WORLD;
        let lz    = wz - cz as f32 * CHUNK_WORLD;
        let col_f = (lx / CHUNK_STEP).clamp(0.0, (CHUNK_CELLS - 1) as f32);
        let row_f = (lz / CHUNK_STEP).clamp(0.0, (CHUNK_CELLS - 1) as f32);
        let c0 = (col_f as usize).min(CHUNK_CELLS - 1);
        let r0 = (row_f as usize).min(CHUNK_CELLS - 1);
        let tf = col_f.fract();
        let sf = row_f.fract();
        let h  = &cd.heights;
        h[ r0      * CHUNK_VERTS + c0    ] * (1.0-tf) * (1.0-sf)
      + h[ r0      * CHUNK_VERTS + c0 + 1] * tf       * (1.0-sf)
      + h[(r0 + 1) * CHUNK_VERTS + c0    ] * (1.0-tf) * sf
      + h[(r0 + 1) * CHUNK_VERTS + c0 + 1] * tf       * sf
    }

    pub fn normal_at(&self, wx: f32, wz: f32) -> [f32; 3] {
        let eps = CHUNK_STEP * 2.0;
        let hl = self.height_at(wx - eps, wz);
        let hr = self.height_at(wx + eps, wz);
        let hd = self.height_at(wx, wz - eps);
        let hu = self.height_at(wx, wz + eps);
        let nx = (hl - hr) / (2.0 * eps);
        let nz = (hd - hu) / (2.0 * eps);
        let len = (nx * nx + 1.0_f32 + nz * nz).sqrt();
        [nx / len, 1.0 / len, nz / len]
    }

    /// World (x, z) of the road's starting point.
    pub fn spawn_point(&self) -> (f32, f32) {
        let [x, z] = self.road.positions[0];
        (x, z)
    }
}

// ── Perlin gradient noise ─────────────────────────────────────────────────────

pub struct Perlin { perm: [u8; 512] }

impl Perlin {
    pub fn new(seed: u64) -> Self {
        let mut p: Vec<u8> = (0..=255u8).collect();
        let mut rng = seed;
        for i in (1..256usize).rev() {
            rng = rng.wrapping_mul(6364136223846793005)
                     .wrapping_add(1442695040888963407);
            let j = ((rng >> 33) as usize) % (i + 1);
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..256 { perm[i] = p[i]; perm[i + 256] = p[i]; }
        Self { perm }
    }

    pub fn noise(&self, x: f32, z: f32) -> f32 {
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let lerp  = |a: f32, b: f32, t: f32| a + t * (b - a);
        let grad  = |h: u8, x: f32, z: f32| -> f32 {
            match h & 3 { 0 => x + z, 1 => -x + z, 2 => x - z, _ => -x - z }
        };
        let xi = x.floor() as i32; let zi = z.floor() as i32;
        let xf = x - x.floor();    let zf = z - z.floor();
        let u = fade(xf); let v = fade(zf);
        let p = &self.perm;
        let xi_u  = (xi     & 255) as usize;
        let xi1_u = ((xi+1) & 255) as usize;
        let zi_u  = (zi     & 255) as usize;
        let zi1_u = ((zi+1) & 255) as usize;
        let aa = p[p[xi_u]  as usize + zi_u ];
        let ab = p[p[xi_u]  as usize + zi1_u];
        let ba = p[p[xi1_u] as usize + zi_u ];
        let bb = p[p[xi1_u] as usize + zi1_u];
        lerp(
            lerp(grad(aa, xf,       zf      ), grad(ba, xf-1.0, zf      ), u),
            lerp(grad(ab, xf,       zf-1.0  ), grad(bb, xf-1.0, zf-1.0  ), u),
            v,
        )
    }

    pub fn fbm(&self, x: f32, z: f32, octaves: u32) -> f32 {
        let (mut val, mut amp, mut freq, mut total) = (0.0_f32, 0.5, 1.0, 0.0_f32);
        for _ in 0..octaves {
            val   += amp * self.noise(x * freq, z * freq);
            total += amp;
            amp   *= 0.5;
            freq  *= 2.0;
        }
        val / total
    }
}
