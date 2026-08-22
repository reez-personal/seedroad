// shaders/grass.wgsl — GPU-instanced static grass (no wind, no interaction).

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    time:      f32,
    car_pos:   vec3<f32>,
    _pad2:     f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Blade vertex attributes (locations 0-1) + instance attributes (locations 2-5)
struct VertIn {
    @location(0) lpos:        vec3<f32>,  // blade local position (pre-rotated sub-blade offset)
    @location(1) height_t:    f32,        // normalised height along blade [0,1]
    @location(2) world_pos:   vec3<f32>,  // instance world base position
    @location(3) inst_height: f32,        // instance blade height scale
    @location(4) rotation:    f32,        // instance Y rotation (rotates whole clump)
    @location(5) tint:        f32,        // instance colour tint [0,1]
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos:  vec3<f32>,
    @location(1) height_t:   f32,
    @location(2) frag_tint:  f32,
}

@vertex
fn vs_grass(v: VertIn) -> VsOut {
    // 1. Scale blade by instance height
    let sp = v.lpos * v.inst_height;

    // 2. Rotate whole clump around Y axis by instance rotation
    let sr = sin(v.rotation);
    let cr = cos(v.rotation);
    let rp = vec3<f32>(
        sp.x * cr - sp.z * sr,
        sp.y,
        sp.x * sr + sp.z * cr,
    );

    // 3. Final world position — no wind, no interaction displacement
    let world = v.world_pos + rp;

    var result: VsOut;
    result.clip_pos  = camera.view_proj * vec4<f32>(world, 1.0);
    result.world_pos = world;
    result.height_t  = v.height_t;
    result.frag_tint = v.tint;
    return result;
}

@fragment
fn fs_grass(fragment: VsOut) -> @location(0) vec4<f32> {
    // Colour: mix lush green and dry yellow-green by per-instance tint
    let lush = vec3<f32>(0.12, 0.40, 0.06);
    let dry  = vec3<f32>(0.22, 0.38, 0.08);
    var col  = mix(lush, dry, fragment.frag_tint);

    // Darken at the root
    col = col * (0.4 + 0.6 * fragment.height_t);

    // Simple Lambert lighting (two-sided — abs keeps underside lit)
    let sun_dir = normalize(vec3<f32>(0.6, 1.0, 0.4));
    let ndl     = abs(dot(vec3<f32>(0.0, 1.0, 0.0), sun_dir));
    let sky_amb = vec3<f32>(0.20, 0.32, 0.45) * 0.25;
    col = col * (ndl * vec3<f32>(1.05, 0.95, 0.70) + sky_amb);

    // ACES tone-mapping
    let c = col * 1.10;
    col = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(col, 1.0);
}
