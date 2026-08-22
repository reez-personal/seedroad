// shaders/splash.wgsl — camera-facing billboard particles for water splash.

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    eye_pos:    vec3<f32>,
    time:       f32,
    car_pos:    vec3<f32>,
    _pad2:      f32,
    cam_right:  vec3<f32>,
    _pad3:      f32,
    cam_up:     vec3<f32>,
    _pad4:      f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// ── Vertex inputs ─────────────────────────────────────────────────────────────
// Blade mesh (location 0-1): a simple unit quad in local 2-D space.
struct QuadVert {
    @location(0) local_uv: vec2<f32>,   // (-0.5 … 0.5)
}

// Instance (location 1-4): one per live particle.
struct InstIn {
    @location(1) world_pos: vec3<f32>,
    @location(2) size:      f32,
    @location(3) alpha:     f32,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
    @location(1)       alpha:    f32,
}

@vertex
fn vs_splash(q: QuadVert, inst: InstIn) -> VsOut {
    // Billboard: offset the quad corners along camera right/up axes.
    let world = inst.world_pos
        + camera.cam_right * q.local_uv.x * inst.size
        + camera.cam_up    * q.local_uv.y * inst.size;

    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.uv       = q.local_uv + 0.5;   // remap to [0,1]
    out.alpha    = inst.alpha;
    return out;
}

@fragment
fn fs_splash(in: VsOut) -> @location(0) vec4<f32> {
    // Soft circle: fade to 0 at edges.
    let d = length(in.uv - 0.5) * 2.0;   // 0 at centre, 1 at rim
    if d > 1.0 { discard; }

    let soft = 1.0 - smoothstep(0.55, 1.0, d);

    // Bright white-blue water spray.
    let col = mix(vec3<f32>(0.55, 0.78, 0.95), vec3<f32>(1.0, 1.0, 1.0), soft * 0.4);

    // Gamma-encode so it blends correctly into the already-encoded framebuffer.
    let gc = pow(clamp(col, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
    return vec4<f32>(gc, soft * in.alpha * 0.85);
}
