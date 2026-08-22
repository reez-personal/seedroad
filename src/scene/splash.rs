// scene/splash.rs — CPU-simulated water splash particles, GPU-instanced billboards.

use bytemuck::{Pod, Zeroable};
use crate::terrain::{TerrainManager, WATER_Y};

const PARTICLE_LIFE:  f32   = 0.80;
const PARTICLE_SIZE:  f32   = 0.35;
const GRAVITY:        f32   = 9.8;
// Minimum car speed (m/s) before any spray emits.
const MIN_SPEED:      f32   = 0.8;
// Particles/second at speed 10 m/s (quadratic: × (speed/10)²).
const BASE_RATE:      f32   = 18.0;
pub const MAX_SPLASH: usize = 4096;

// ── GPU instance (32 bytes) ───────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SplashInstance {
    pub world_pos: [f32; 3],
    pub size:      f32,
    pub alpha:     f32,
    pub _pad:      [f32; 3],
}

// ── CPU particle ─────────────────────────────────────────────────────────────

struct Particle {
    pos:      [f32; 3],
    vel:      [f32; 3],
    life:     f32,
    max_life: f32,
}

// ── SplashSystem ─────────────────────────────────────────────────────────────

pub struct SplashSystem {
    particles:   Vec<Particle>,
    rng:         u32,
    /// Fractional particle accumulator per wheel [4].
    frac_emit:   [f32; 4],
}

impl SplashSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::with_capacity(MAX_SPLASH),
            rng:       0xDEADBEEF,
            frac_emit: [0.0; 4],
        }
    }

    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        self.rng as f32 / u32::MAX as f32
    }

    pub fn update(
        &mut self,
        dt:         f32,
        wheel_data: &[[f32; 4]],
        terrain:    &TerrainManager,
    ) {
        // Step existing particles.
        self.particles.retain_mut(|p| {
            p.vel[1] -= GRAVITY * dt;
            p.pos[0] += p.vel[0] * dt;
            p.pos[1] += p.vel[1] * dt;
            p.pos[2] += p.vel[2] * dt;
            p.life   -= dt;
            p.life > 0.0
        });

        // Emit from each wheel that is in a water-covered area.
        for (wi, wheel) in wheel_data.iter().enumerate() {
            let wx = wheel[0];
            let wz = wheel[1];

            // Water exists wherever the terrain surface is below WATER_Y.
            let terrain_h = terrain.height_at(wx, wz);
            if terrain_h >= WATER_Y {
                self.frac_emit[wi] = 0.0;
                continue;
            }

            let speed = (wheel[2] * wheel[2] + wheel[3] * wheel[3]).sqrt();
            if speed < MIN_SPEED {
                self.frac_emit[wi] = 0.0;
                continue;
            }

            // Quadratic scaling: BASE_RATE × (speed / 10)²
            let rate = BASE_RATE * (speed / 10.0) * (speed / 10.0);
            self.frac_emit[wi] += rate * dt;

            let count = self.frac_emit[wi] as usize;
            self.frac_emit[wi] -= count as f32;
            let count = count.min(12);   // hard cap per wheel per frame

            for _ in 0..count {
                if self.particles.len() >= MAX_SPLASH { break; }

                let spread = speed * 0.30;
                let r0 = self.rand(); let r1 = self.rand(); let r2 = self.rand();
                let r3 = self.rand(); let r4 = self.rand(); let r5 = self.rand();

                let vx = (r0 - 0.5) * spread - wheel[2] * 0.12;
                let vy =  r1 * speed * 0.40 + 2.0;
                let vz = (r2 - 0.5) * spread - wheel[3] * 0.12;
                let jx = (r3 - 0.5) * 0.5;
                let jz = (r4 - 0.5) * 0.5;
                let life = PARTICLE_LIFE * (0.5 + r5 * 0.5);

                self.particles.push(Particle {
                    pos:      [wx + jx, WATER_Y + 0.05, wz + jz],
                    vel:      [vx, vy, vz],
                    life,
                    max_life: PARTICLE_LIFE,
                });
            }
        }
    }

    /// Build GPU instance list from live particles.
    pub fn instances(&self) -> Vec<SplashInstance> {
        self.particles.iter().map(|p| {
            let t = (p.life / p.max_life).clamp(0.0, 1.0);
            // Size grows then shrinks, peaks around 60% remaining life.
            let size_t = if t > 0.6 { (1.0 - t) / 0.4 } else { t / 0.6 };
            SplashInstance {
                world_pos: p.pos,
                size:      PARTICLE_SIZE * (0.25 + size_t * 0.75),
                alpha:     t * t,
                _pad:      [0.0; 3],
            }
        }).collect()
    }

    pub fn particle_count(&self) -> usize { self.particles.len() }
}
