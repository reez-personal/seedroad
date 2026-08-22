// shaders/droplets.wgsl — screen-space water droplet overlay.
//
// Runs as a fullscreen post-process pass after the main scene.
// Samples the scene color texture and applies lens-water distortion
// where procedural droplets are placed.  Wetness (0-1) controls how
// many droplets are visible; they slowly dry and streak downward.

struct DropletParams {
    wetness: f32,   // 0 = dry, 1 = soaked
    time:    f32,
    _pad:    vec2<f32>,
}

@group(0) @binding(0) var scene_tex:    texture_2d<f32>;
@group(0) @binding(1) var scene_samp:   sampler;
@group(0) @binding(2) var<uniform> dp:  DropletParams;

// ── Fullscreen triangle ───────────────────────────────────────────────────────

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_droplets(@builtin(vertex_index) vid: u32) -> VsOut {
    // Oversized triangle covering NDC [-1,1]².
    // wgpu NDC: Y+ = up.  Texture UV: (0,0) = top-left, Y+ = down.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0,  1.0),   // top-left  → UV (0,0)
        vec2<f32>( 3.0,  1.0),   // top-right → UV (2,0) [off screen]
        vec2<f32>(-1.0, -3.0),   // bot-left  → UV (0,2) [off screen]
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
    out.uv  = uv[vid];
    return out;
}

// ── Noise (value noise, keeps dependency on no imports) ───────────────────────

fn hash2d(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise2(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2d(i),               hash2d(i + vec2(1.0,0.0)), u.x),
               mix(hash2d(i + vec2(0.0,1.0)), hash2d(i + vec2(1.0,1.0)), u.x), u.y);
}

// ── Droplet shape SDF (returns (coverage, distortion_strength)) ───────────────
//
// Droplets are small circles scattered via a grid + hash jitter.
// Inside a droplet: refraction offsets the UV slightly.

fn droplet(uv: vec2<f32>, cell_size: f32, time: f32) -> vec2<f32> {
    let cell = floor(uv / cell_size);
    let local = fract(uv / cell_size);

    // Each cell holds one potential droplet.
    let h  = hash2d(cell + vec2<f32>(17.3, 31.7));
    let h2 = hash2d(cell + vec2<f32>(43.1, 7.9));
    let h3 = hash2d(cell + vec2<f32>(91.3, 53.1));

    // Only show droplet if wetness is above a per-cell threshold.
    if h3 > dp.wetness { return vec2<f32>(0.0); }

    // Jitter centre within cell (avoid edges).
    let centre = vec2<f32>(0.2 + h * 0.6, 0.2 + h2 * 0.6);

    // Droplets slowly drift downward as they dry.
    let drip_speed = (1.0 - dp.wetness) * 0.04;
    let offset = vec2<f32>(0.0, drip_speed * time * h);
    let d = length(local - centre - offset);

    let radius = 0.10 + h * 0.08;
    if d > radius { return vec2<f32>(0.0); }

    // Lens shape: convex in middle, thin at edges.
    let t = 1.0 - d / radius;
    let lens = t * t * (3.0 - 2.0 * t);

    // Normal from lens profile → refraction direction (points toward centre).
    let strength = lens * 0.018;
    return vec2<f32>(strength, lens);
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_droplets(in: VsOut) -> @location(0) vec4<f32> {
    if dp.wetness < 0.005 {
        // Completely dry: pass through unmodified.
        return textureSample(scene_tex, scene_samp, in.uv);
    }

    var total_dist = vec2<f32>(0.0);
    var total_cover = 0.0;

    // Layer 1: large drops (cell 0.12 of screen)
    let d1 = droplet(in.uv, 0.120, dp.time);
    total_dist  += d1.x * normalize(in.uv - vec2<f32>(0.5));
    total_cover  = max(total_cover, d1.y);

    // Layer 2: medium drops (cell 0.065)
    let d2 = droplet(in.uv + vec2<f32>(0.031, 0.019), 0.065, dp.time * 0.9);
    total_dist  += d2.x * 0.6 * normalize(in.uv - vec2<f32>(0.5) + vec2<f32>(0.01));
    total_cover  = max(total_cover, d2.y * 0.7);

    // Sample scene with UV distortion from lens refraction.
    let scene_uv = clamp(in.uv + total_dist, vec2(0.001), vec2(0.999));
    let scene    = textureSample(scene_tex, scene_samp, scene_uv).rgb;

    // Inside a droplet: slightly brighten and tint cool (water lens brightens).
    let droplet_tint = mix(vec3<f32>(1.0), vec3<f32>(0.90, 0.95, 1.05), total_cover);
    let out_col = scene * droplet_tint;

    // Thin film on the glass between droplets: very subtle darkening.
    let film = dp.wetness * (1.0 - total_cover) * 0.04;

    return vec4<f32>(out_col * (1.0 - film), 1.0);
}
