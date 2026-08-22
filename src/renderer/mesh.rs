// renderer/mesh.rs — procedural mesh builders.

use crate::renderer::buffer::Vertex;
use std::f32::consts::PI;

/// Unit cube: −0.5 to +0.5 on each axis, 24 vertices (4 per face, flat normals).
pub fn build_cube() -> (Vec<Vertex>, Vec<u16>) {
    const H: f32 = 0.5;
    fn v(p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> Vertex { Vertex { position: p, normal: n, uv } }

    let verts = vec![
        // +X
        v([ H,-H,-H],[1.,0.,0.],[0.,1.]), v([ H, H,-H],[1.,0.,0.],[0.,0.]),
        v([ H, H, H],[1.,0.,0.],[1.,0.]), v([ H,-H, H],[1.,0.,0.],[1.,1.]),
        // -X
        v([-H,-H, H],[-1.,0.,0.],[0.,1.]), v([-H, H, H],[-1.,0.,0.],[0.,0.]),
        v([-H, H,-H],[-1.,0.,0.],[1.,0.]), v([-H,-H,-H],[-1.,0.,0.],[1.,1.]),
        // +Y
        v([-H, H,-H],[0.,1.,0.],[0.,1.]), v([-H, H, H],[0.,1.,0.],[0.,0.]),
        v([ H, H, H],[0.,1.,0.],[1.,0.]), v([ H, H,-H],[0.,1.,0.],[1.,1.]),
        // -Y
        v([-H,-H, H],[0.,-1.,0.],[0.,1.]), v([-H,-H,-H],[0.,-1.,0.],[0.,0.]),
        v([ H,-H,-H],[0.,-1.,0.],[1.,0.]), v([ H,-H, H],[0.,-1.,0.],[1.,1.]),
        // +Z
        v([-H,-H, H],[0.,0.,1.],[0.,1.]), v([ H,-H, H],[0.,0.,1.],[1.,1.]),
        v([ H, H, H],[0.,0.,1.],[1.,0.]), v([-H, H, H],[0.,0.,1.],[0.,0.]),
        // -Z
        v([ H,-H,-H],[0.,0.,-1.],[0.,1.]), v([-H,-H,-H],[0.,0.,-1.],[1.,1.]),
        v([-H, H,-H],[0.,0.,-1.],[1.,0.]), v([ H, H,-H],[0.,0.,-1.],[0.,0.]),
    ];

    let mut idx = Vec::with_capacity(36);
    for f in 0..6u16 {
        let b = f * 4;
        idx.extend_from_slice(&[b, b+1, b+2, b, b+2, b+3]);
    }
    (verts, idx)
}

/// Cylinder along the X axis, radius 1, length 1, centred at origin.
/// `sides` controls how round the wheel looks (20 is plenty).
pub fn build_cylinder(sides: u32) -> (Vec<Vertex>, Vec<u16>) {
    let mut verts: Vec<Vertex> = Vec::new();
    let mut idx:   Vec<u16>   = Vec::new();

    let half = 0.5f32; // half-length along X

    for cap in [half, -half] {
        let normal_x = cap.signum();
        let centre_idx = verts.len() as u16;
        verts.push(Vertex {
            position: [cap, 0.0, 0.0],
            normal:   [normal_x, 0.0, 0.0],
            uv:       [0.5, 0.5],
        });
        let ring_start = verts.len() as u16;
        for i in 0..sides {
            let theta = 2.0 * PI * i as f32 / sides as f32;
            let y = theta.cos();
            let z = theta.sin();
            verts.push(Vertex {
                position: [cap, y, z],
                normal:   [normal_x, 0.0, 0.0],
                uv:       [0.5 + y * 0.5, 0.5 + z * 0.5],
            });
        }
        // Cap triangles
        for i in 0..sides {
            let a = ring_start + i as u16;
            let b = ring_start + (i + 1) as u16 % sides as u16;
            if cap > 0.0 {
                idx.extend_from_slice(&[centre_idx, a, b]);
            } else {
                idx.extend_from_slice(&[centre_idx, b, a]);
            }
        }
    }

    // Side quads
    let cap0_ring = 1u16;                      // ring start of +X cap
    let cap1_ring = 1 + sides as u16 + 1;     // ring start of −X cap
    for i in 0..sides {
        let a0 = cap0_ring + i as u16;
        let b0 = cap0_ring + (i + 1) as u16 % sides as u16;
        let a1 = cap1_ring + i as u16;
        let b1 = cap1_ring + (i + 1) as u16 % sides as u16;
        // Side normals: outward radial
        let theta = 2.0 * PI * i as f32 / sides as f32;
        let ny = theta.cos();
        let nz = theta.sin();
        let base = verts.len() as u16;
        // 4 vertices for this quad strip (with proper side normals)
        verts.push(Vertex { position: verts[a0 as usize].position, normal: [0.0, ny, nz], uv: [i as f32 / sides as f32, 0.0] });
        verts.push(Vertex { position: verts[b0 as usize].position, normal: [0.0, ny, nz], uv: [(i+1) as f32 / sides as f32, 0.0] });
        verts.push(Vertex { position: verts[b1 as usize].position, normal: [0.0, ny, nz], uv: [(i+1) as f32 / sides as f32, 1.0] });
        verts.push(Vertex { position: verts[a1 as usize].position, normal: [0.0, ny, nz], uv: [i as f32 / sides as f32, 1.0] });
        idx.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    (verts, idx)
}

/// Cone pointing upward (+Y tip), Y-axis aligned, centred at origin.
/// Base at y=−0.5 (radius 1.0), tip at y=+0.5.
/// Scale non-uniformly (sx=radius, sy=height, sz=radius) when placing.
pub fn build_cone(sides: u32) -> (Vec<Vertex>, Vec<u16>) {
    let mut verts: Vec<Vertex> = Vec::new();
    let mut idx:   Vec<u16>   = Vec::new();

    let base_y = -0.5_f32;
    let tip_y  =  0.5_f32;

    // Side normal slope: base_radius=1, height=1, slant=sqrt(2)
    // ny component (upward) = base_r / slant = 1/sqrt(2)
    // nr component (outward) = height / slant = 1/sqrt(2)
    let s45 = std::f32::consts::FRAC_1_SQRT_2;

    // ── Base cap (normal = (0,−1,0), visible from below) ──────────────────
    let base_ctr = verts.len() as u16;
    verts.push(Vertex { position: [0.0, base_y, 0.0], normal: [0.0, -1.0, 0.0], uv: [0.5, 0.5] });
    let base_ring = verts.len() as u16;
    for i in 0..sides {
        let t = 2.0 * PI * i as f32 / sides as f32;
        verts.push(Vertex {
            position: [t.cos(), base_y, t.sin()],
            normal:   [0.0, -1.0, 0.0],
            uv:       [0.5 + t.cos() * 0.5, 0.5 + t.sin() * 0.5],
        });
    }
    // CCW from below: centre, a, b  (a and b at increasing theta)
    for i in 0..sides {
        let a = base_ring + i as u16;
        let b = base_ring + ((i + 1) % sides) as u16;
        idx.extend_from_slice(&[base_ctr, a, b]);
    }

    // ── Side faces (flat-shaded per face with outward+upward normals) ──────
    // One triangle per sector; each face has 3 private vertices.
    for i in 0..sides {
        let t0 = 2.0 * PI * i as f32 / sides as f32;
        let t1 = 2.0 * PI * (i + 1) as f32 / sides as f32;
        let tm = (t0 + t1) * 0.5;
        let nx = tm.cos() * s45;
        let nz = tm.sin() * s45;

        let base_a = verts.len() as u16;
        verts.push(Vertex {
            position: [t0.cos(), base_y, t0.sin()],
            normal:   [nx, s45, nz],
            uv:       [i as f32 / sides as f32, 0.0],
        });
        verts.push(Vertex {
            position: [t1.cos(), base_y, t1.sin()],
            normal:   [nx, s45, nz],
            uv:       [(i + 1) as f32 / sides as f32, 0.0],
        });
        verts.push(Vertex {
            position: [0.0, tip_y, 0.0],
            normal:   [nx, s45, nz],
            uv:       [(i as f32 + 0.5) / sides as f32, 1.0],
        });
        // CCW from outside: base_a, tip, base_b
        idx.extend_from_slice(&[base_a, base_a + 2, base_a + 1]);
    }

    (verts, idx)
}

/// SUV body shell built by lofting 10 cross-sections along the X axis.

/// Low-poly UV sphere, Y-up, radius 1, centred at origin.
/// Smooth per-vertex normals (outward = position on unit sphere).
pub fn build_sphere(stacks: u32, slices: u32) -> (Vec<Vertex>, Vec<u16>) {
    let mut verts: Vec<Vertex> = Vec::new();
    let mut idx:   Vec<u16>   = Vec::new();

    // (stacks+1) rings × (slices+1) columns — last column wraps to first
    for s in 0..=stacks {
        let phi = PI * s as f32 / stacks as f32;   // 0 (top) → PI (bottom)
        let y   = phi.cos();
        let r   = phi.sin();
        for sl in 0..=slices {
            let theta = 2.0 * PI * sl as f32 / slices as f32;
            let x = r * theta.cos();
            let z = r * theta.sin();
            verts.push(Vertex {
                position: [x, y, z],
                normal:   [x, y, z], // unit sphere: position == outward normal
                uv:       [sl as f32 / slices as f32, s as f32 / stacks as f32],
            });
        }
    }

    // Quads: outward CCW winding matches [a, b, c, b, d, c]
    let w = slices + 1;
    for s in 0..stacks {
        for sl in 0..slices {
            let a = (s       * w + sl)     as u16;
            let b = (s       * w + sl + 1) as u16;
            let c = ((s + 1) * w + sl)     as u16;
            let d = ((s + 1) * w + sl + 1) as u16;
            idx.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }

    (verts, idx)
}

