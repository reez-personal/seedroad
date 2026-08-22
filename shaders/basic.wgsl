// shaders/basic.wgsl — Cook-Torrance GGX PBR, aerial-perspective fog, ACES tone-map.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    time:      f32,
    car_pos:   vec3<f32>,
    _pad2:     f32,
}

// 256-byte aligned: mat4(64) + mat4(64) + vec4(16) + pad(112)
struct TransformUniform {
    model:  mat4x4<f32>,
    normal: mat4x4<f32>,
    color:  vec4<f32>,   // .a: >1.5=sky, <0.05=terrain, 0.05-0.89=water, >=0.9=solid
}

@group(0) @binding(0) var<uniform> camera:    CameraUniform;
@group(0) @binding(1) var<uniform> transform: TransformUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_normal: vec3<f32>,
    @location(1)       world_pos:    vec3<f32>,
    @location(2)       uv:           vec2<f32>,
}

@vertex
fn vs_main(vin: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let wp        = transform.model * vec4<f32>(vin.position, 1.0);
    out.world_pos = wp.xyz;
    out.clip_pos  = camera.view_proj * wp;
    out.world_normal = normalize((transform.normal * vec4<f32>(vin.normal, 0.0)).xyz);
    out.uv = vin.uv;
    // Sky cube is scaled to 850 m but far plane is 500 m — pin sky to far plane in NDC
    // so vertices never get clipped (eliminates the black wedge at the horizon).
    if transform.color.a > 1.5 {
        out.clip_pos.z = out.clip_pos.w;
    }
    return out;
}

// ── Noise helpers ─────────────────────────────────────────────────────────────

fn hash2(p: vec2<f32>) -> f32 {
    let k = vec2<f32>(127.1, 311.7);
    return fract(sin(dot(p, k)) * 43758.5453123);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ── Sun — consistent across all materials ────────────────────────────────────
fn sun_dir() -> vec3<f32>   { return normalize(vec3<f32>(0.65, 0.12, 0.40)); }
fn sun_color() -> vec3<f32> { return vec3<f32>(1.05, 0.82, 0.55); }

// ── Physically-based sky with Mie scattering ─────────────────────────────────
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let d  = normalize(dir);
    let t  = clamp(d.y, -0.1, 1.0);

    let zenith = vec3<f32>(0.04, 0.06, 0.24);
    let mid    = vec3<f32>(0.14, 0.26, 0.62);
    let horiz  = vec3<f32>(0.92, 0.56, 0.16);
    let glow   = vec3<f32>(0.80, 0.22, 0.04);

    let t1 = smoothstep(-0.10, 0.20, t);
    let t2 = smoothstep( 0.18, 0.55, t);
    let t3 = smoothstep( 0.50, 1.00, t);

    var sky = mix(glow, horiz, t1);
    sky = mix(sky, mid,    t2);
    sky = mix(sky, zenith, t3);

    let sd     = max(dot(d, sun_dir()), 0.0);
    let disc   = step(0.9996, sd) * 3.5;
    let limb   = pow(sd, 32.0) * 2.0;
    let corona = pow(sd,  5.0) * 0.90;
    sky += sun_color() * (disc + limb + corona);

    let mie = pow(max(sd, 0.0), 4.0) * 0.55;
    sky += vec3<f32>(0.85, 0.42, 0.10) * mie * smoothstep(-0.05, 0.35, d.y + 0.15);

    return sky;
}

// ── ACES filmic tone-mapping ──────────────────────────────────────────────────
fn aces(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let cc = 2.43; let d = 0.59; let e = 0.14;
    return clamp((c * (a * c + b)) / (c * (cc * c + d) + e), vec3(0.0), vec3(1.0));
}

// ── Height + slope → terrain colour ──────────────────────────────────────────
fn terrain_color(y: f32, slope: f32) -> vec3<f32> {
    let sand   = vec3<f32>(0.62, 0.54, 0.36);
    let grass  = vec3<f32>(0.16, 0.46, 0.08);
    let meadow = vec3<f32>(0.18, 0.52, 0.09);
    let forest = vec3<f32>(0.07, 0.26, 0.05);
    let rock   = vec3<f32>(0.48, 0.44, 0.40);
    let snow   = vec3<f32>(0.94, 0.96, 1.00);

    let t1 = smoothstep( 2.0,  8.0, y);
    let t2 = smoothstep( 8.0, 18.0, y);
    let t3 = smoothstep(18.0, 28.0, y);
    let t4 = smoothstep(36.0, 50.0, y);
    let t5 = smoothstep(52.0, 62.0, y);

    var col = mix(sand,   grass,  t1);
    col      = mix(col,   meadow, t2);
    col      = mix(col,   forest, t3);
    col      = mix(col,   rock,   t4);
    col      = mix(col,   snow,   t5);

    let slope_rock = smoothstep(0.44, 0.76, slope);
    col = mix(col, rock, slope_rock);
    return col;
}

// ── Cook-Torrance GGX PBR ─────────────────────────────────────────────────────

fn ggx_d(ndoth: f32, rough: f32) -> f32 {
    let a  = rough * rough;
    let a2 = a * a;
    let d  = ndoth * ndoth * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d + 0.00001);
}

fn schlick_g1(ndotx: f32, rough: f32) -> f32 {
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    return ndotx / (ndotx * (1.0 - k) + k);
}

fn smith_g(ndotv: f32, ndotl: f32, rough: f32) -> f32 {
    return schlick_g1(ndotv, rough) * schlick_g1(ndotl, rough);
}

fn schlick_f(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

/// Full PBR shading: direct sun + hemisphere ambient + env reflection.
/// metallic=0 → dielectric (car paint, terrain, rubber)
/// metallic=1 → conductor (rims, chrome)
fn shade_pbr(N: vec3<f32>, V: vec3<f32>, albedo: vec3<f32>, roughness: f32, metallic: f32) -> vec3<f32> {
    let L  = sun_dir();
    let H  = normalize(V + L);

    let ndotl = max(dot(N, L), 0.0);
    let ndotv = max(dot(N, V), 0.001);
    let ndoth = max(dot(N, H), 0.0);
    let hdotv = max(dot(H, V), 0.0);

    // Base reflectance: 0.04 for dielectrics, albedo for metals
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Cook-Torrance specular
    let D = ggx_d(ndoth, roughness);
    let G = smith_g(ndotv, ndotl, roughness);
    let F = schlick_f(hdotv, f0);

    let ks       = F;
    let kd       = (vec3<f32>(1.0) - ks) * (1.0 - metallic);
    let specular = D * G * F / (4.0 * ndotv * ndotl + 0.0001);

    // Direct sun contribution
    let sun_rad = sun_color() * 3.6;
    let Lo      = (kd * albedo / 3.14159265 + specular) * sun_rad * ndotl;

    // Hemisphere irradiance (sky up, ground down)
    let sky_irr = vec3<f32>(0.20, 0.30, 0.50);
    let gnd_irr = vec3<f32>(0.15, 0.18, 0.05);
    let hemi    = 0.5 + 0.5 * dot(N, vec3<f32>(0.0, 1.0, 0.0));
    let amb_irr = mix(gnd_irr, sky_irr, hemi);
    let amb_f   = schlick_f(ndotv, f0);
    let amb_ks  = amb_f;
    let amb_kd  = (vec3<f32>(1.0) - amb_ks) * (1.0 - metallic);
    let ambient = (amb_kd * albedo * amb_irr) * 0.14;

    // Subtle fill (blue-purple from opposite direction)
    let fill_L   = normalize(vec3<f32>(-0.55, 0.25, -0.45));
    let fill_ndl = max(dot(N, fill_L), 0.0);
    let fill     = albedo * fill_ndl * vec3<f32>(0.18, 0.22, 0.40) * 0.20 * (1.0 - metallic * 0.6);

    // Specular environment reflection (glossy surfaces pick up sky colour)
    let Rv      = reflect(-V, N);
    let env_col = sky_color(Rv);
    let env_f   = schlick_f(ndotv, f0);
    let env_contrib = env_col * env_f * pow(1.0 - roughness, 2.0) * 0.40;

    return Lo + ambient + fill + env_contrib;
}

// ── Aerial-perspective fog ─────────────────────────────────────────────────────
fn apply_fog(col: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let view_vec = world_pos - camera.eye_pos;
    let dist     = length(view_vec);
    let view_dir = view_vec / max(dist, 0.001);

    // Height-based fog: exponentially denser near sea level
    // Near-cutoff: no fog within 20 m so close objects stay crisp.
    let h_fog   = exp(-max(world_pos.y, 0.0) * 0.07);
    let fog_amt = 1.0 - exp(-max(dist - 20.0, 0.0) * 0.0025 * (0.35 + h_fog * 0.65));

    // Aerial perspective colour: look up actual sky at view direction (clamped near horizon)
    let horiz_dir = normalize(vec3<f32>(view_dir.x, max(view_dir.y * 0.5, -0.04), view_dir.z));
    let aerial    = sky_color(horiz_dir) * 0.62 + vec3<f32>(0.54, 0.46, 0.34) * 0.38;

    // Sun scatter in fog
    let sun_scat = pow(max(dot(view_dir, sun_dir()), 0.0), 5.0) * 0.55;
    let fog_col  = aerial + sun_color() * sun_scat;

    return mix(col, fog_col, fog_amt * 0.48);
}

// ── Warm Forza-ish grade (applied after ACES) ─────────────────────────────────
fn color_grade(c: vec3<f32>) -> vec3<f32> {
    // Slightly warmer shadows + lifted mids + punchy highlights
    let lum  = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    // Warm the shadows gently
    let warm_shadow = vec3<f32>(1.04, 0.98, 0.90);
    let warm        = mix(warm_shadow, vec3<f32>(1.0), smoothstep(0.0, 0.5, lum));
    var gc   = c * warm;
    // Saturation boost (Forza greens pop)
    let lum2 = dot(gc, vec3<f32>(0.299, 0.587, 0.114));
    gc = mix(vec3<f32>(lum2), gc, 1.12);
    return clamp(gc, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N   = normalize(in.world_normal);
    let V   = normalize(camera.eye_pos - in.world_pos);
    let mat = transform.color.a;

    var out_color: vec3<f32>;
    var out_alpha: f32 = 1.0;

    if mat > 1.5 {
        // ── Sky ───────────────────────────────────────────────────────────
        let view_dir = normalize(in.world_pos - camera.eye_pos);
        out_color    = sky_color(view_dir);

        let shaft = pow(max(dot(view_dir, sun_dir()), 0.0), 7.0)
            * smoothstep(-0.05, 0.30, view_dir.y + 0.10) * 0.40;
        out_color += vec3<f32>(1.0, 0.65, 0.20) * shaft;

        out_color = pow(clamp(out_color, vec3(0.0), vec3(1.0)), vec3<f32>(1.0 / 2.2));
        return vec4<f32>(out_color, 1.0);

    } else if mat < 0.05 {
        // ── Terrain ───────────────────────────────────────────────────────────
        let slope = 1.0 - clamp(dot(N, vec3<f32>(0.0, 1.0, 0.0)), 0.0, 1.0);
        let y     = in.world_pos.y;
        let xz    = in.world_pos.xz;

        // ── Shared density noise (same spatial scale as grass.rs placement) ──
        //
        // Lushness (~70 m features): separates meadow from bare ground.
        // vnoise returns [0,1]; in grass.rs fbm returns ~[-0.5,0.5], so
        // thresholds differ numerically but the SCALE of the pattern matches.
        let lush_a = vnoise(xz * 0.014);           // ~71 m features
        let lush_b = vnoise(xz * 0.028);           // ~36 m features (second octave)
        let lush_n = lush_a * 0.60 + lush_b * 0.40;

        // Cluster (~9.5 m features): tight patches within lush areas.
        let clust_n = vnoise(xz * 0.105);

        // grass_cover: [0,1] — how much grass vs bare dirt on this pixel.
        // Mirrors the LUSH_THRESHOLD/CLUST_THRESHOLD logic in grass.rs.
        let grass_cover = smoothstep(0.28, 0.68, lush_n)
                        * smoothstep(0.28, 0.64, clust_n);

        // ── Material palette ──────────────────────────────────────────────────
        let bare_dirt = vec3<f32>(0.60, 0.48, 0.30); // warm tan earth
        let dark_dirt = vec3<f32>(0.46, 0.36, 0.22); // deep earthy brown
        let gravel    = vec3<f32>(0.56, 0.51, 0.40); // gritty grey-brown scree
        let grass_low = vec3<f32>(0.14, 0.39, 0.06); // valley grass
        let grass_up  = vec3<f32>(0.19, 0.36, 0.08); // upland meadow
        let rock_col  = vec3<f32>(0.46, 0.43, 0.40); // mountain rock
        let snow_col  = vec3<f32>(0.92, 0.95, 1.00);

        // ── Altitude transitions ──────────────────────────────────────────────
        let t_soil   = smoothstep( 5.0, 13.0, y);   // dirt → grass zone
        let t_upland = smoothstep(15.0, 26.0, y);   // low grass → upland meadow
        let t_rock   = smoothstep(30.0, 46.0, y);   // vegetation → rock
        let t_snow   = smoothstep(50.0, 62.0, y);   // rock → snow

        // Build albedo from bottom up.  grass_cover blends dirt↔vegetation
        // so bare-looking pixels show dirt and lush pixels show green.
        var albedo = bare_dirt;
        albedo = mix(albedo, mix(bare_dirt, grass_low, grass_cover), t_soil);
        albedo = mix(albedo, mix(dark_dirt, grass_up,  grass_cover), t_upland);
        albedo = mix(albedo, rock_col, t_rock);
        albedo = mix(albedo, snow_col, t_snow);

        // ── Slope overrides ───────────────────────────────────────────────────
        // Moderate slopes: gravel/scree bleeds in before full rock.
        let t_gravel     = smoothstep(0.24, 0.46, slope) * (1.0 - t_rock);
        let t_slope_rock = smoothstep(0.42, 0.72, slope);
        albedo = mix(albedo, gravel,   t_gravel * 0.55);
        albedo = mix(albedo, rock_col, t_slope_rock);

        // ── Multi-scale procedural detail ─────────────────────────────────────
        // Noise-offset UV breaks visible grid tiling at distance.
        let uv_jit = vnoise(xz * 0.19) * 5.5;
        let uv_xz  = xz + vec2<f32>(uv_jit, uv_jit * 0.73);
        let fine_n = vnoise(uv_xz * 3.9);           // ~0.25 m grain
        let med_n  = vnoise(xz * 0.54);              // ~1.85 m variation
        albedo    *= 0.86 + fine_n * 0.18 + med_n * 0.08;

        // ── Pebble detail and drainage streaks on bare ground ─────────────────
        // bare_mask: strong where both slope and altitude put us in dirt zone.
        let bare_mask = (1.0 - grass_cover) * (1.0 - t_slope_rock) * (1.0 - t_snow);

        // Pebble scatter: darker rounded stones embedded in dust.
        let pebble_n    = vnoise(uv_xz * 10.5);
        let scatter_n   = vnoise(xz * 4.1);
        let pebble_mask = smoothstep(0.57, 0.77, pebble_n) * scatter_n * bare_mask;
        albedo = mix(albedo, vec3<f32>(0.31, 0.28, 0.22), pebble_mask * 0.44);

        // Drainage streaks: anisotropic noise suggests dry rills running downhill.
        // x-freq < z-freq so streaks run roughly along the Z axis (downhill visual).
        let streak_n    = vnoise(vec2<f32>(xz.x * 0.055, xz.y * 0.130));
        let streak_mask = smoothstep(0.56, 0.80, streak_n) * bare_mask
                        * (1.0 - t_upland) * (1.0 - t_gravel);
        albedo = mix(albedo, albedo * vec3<f32>(0.80, 0.76, 0.70), streak_mask * 0.38);

        // ── Roughness — material-based ────────────────────────────────────────
        // Bare dirt is very rough and diffuse; grass is moderate; rock middle;
        // snow smooth.  steep slopes match rock roughness.
        var terrain_rough = mix(0.95, 0.80, grass_cover);  // dirt ↔ grass
        terrain_rough = mix(terrain_rough, 0.87, t_rock);
        terrain_rough = mix(terrain_rough, 0.35, t_snow);
        terrain_rough = mix(terrain_rough, 0.88, t_slope_rock);
        terrain_rough = clamp(terrain_rough, 0.30, 0.98);

        out_color = shade_pbr(N, V, albedo, terrain_rough, 0.0);
        out_color = apply_fog(out_color, in.world_pos);
        out_color = aces(out_color * 1.08);
        out_color = color_grade(out_color);
        out_color = pow(out_color, vec3<f32>(1.0 / 2.2));

    } else if mat < 0.90 {
        // ── Water ─────────────────────────────────────────────────────────
        let t  = camera.time;
        let wx = in.world_pos.x;
        let wz = in.world_pos.z;

        // Depth from terrain height packed into uv.x at mesh build time.
        let terrain_h = in.uv.x;
        let water_depth = clamp(2.5 - terrain_h, 0.0, 8.0); // WATER_Y = 2.5
        let depth_t = water_depth / 8.0;   // 0 = shallow shore, 1 = deep

        let wa = sin(wx * 0.26 + wz * 0.12 + t * 0.85) * 0.14;
        let wb = sin(wx * 0.15 - wz * 0.32 + t * 0.60) * 0.10;
        let wc = sin(wx * 0.48 + wz * 0.40 - t * 1.20) * 0.05;
        let wN = normalize(vec3<f32>(wa + wc, 1.0, wb + wc));

        let cos_i = abs(dot(wN, V));
        let fres  = 0.04 + 0.96 * pow(1.0 - cos_i, 4.0);

        // Depth-driven colour: teal shallows → navy deep
        let shallow_col = vec3<f32>(0.06, 0.40, 0.52);
        let deep_col    = vec3<f32>(0.01, 0.10, 0.28);
        let wcol        = mix(shallow_col, deep_col, depth_t * 0.85);

        let HW      = normalize(sun_dir() + V);
        let glitter = pow(max(dot(wN, HW), 0.0), 100.0) * fres * 10.0;

        let R_water = reflect(-V, wN);
        var refl_col: vec3<f32>;
        if R_water.y >= 0.0 {
            refl_col = sky_color(R_water);
        } else {
            let depth_y = clamp(-R_water.y * 10.0, 1.0, 8.0);
            refl_col = terrain_color(depth_y, 0.25) * 0.55;
        }
        let sky_refl = refl_col * fres * 0.70;

        let to_car   = camera.car_pos - in.world_pos;
        let car_dist = length(to_car);
        let car_dir  = normalize(to_car);
        let car_align = max(dot(R_water, car_dir), 0.0);
        let car_refl  = pow(car_align, 10.0) * exp(-car_dist * 0.04) * fres * 2.5;
        let car_col   = vec3<f32>(0.96, 0.96, 0.97);

        let car_xz = camera.car_pos.xz;
        let cdist  = length(in.world_pos.xz - car_xz);
        let ripple = sin(cdist * 2.5 - t * 5.0)
            * exp(-cdist * 0.25) * 0.18
            * (1.0 - smoothstep(5.0, 10.0, cdist));

        out_color = wcol * 0.65
            + sun_color() * (glitter + ripple * 0.3)
            + sky_refl
            + car_col * car_refl;

        // Depth-driven alpha: shallow shores transparent, deep water more opaque.
        // Fresnel still gives extra opacity at grazing angles.
        let base_alpha = mix(0.08, 0.55, depth_t);
        out_alpha = mix(base_alpha, min(base_alpha + 0.28, 0.85), fres);

        out_color = aces(out_color * 1.12);
        out_color = color_grade(out_color);
        out_color = pow(out_color, vec3<f32>(1.0 / 2.2));

    } else {
        // ── Solid (car, road, trees, glass) ──────────────────────────────
        // Alpha encodes surface tier:
        //   0.91      → matte dielectric  (asphalt, rubber, bark)    roughness ≈ 0.90
        //   0.91–0.95 → ramp to semi-gloss car paint                 roughness ≈ 0.20
        //   0.95–0.99 → ramp to mirror (glass, chrome)               roughness ≈ 0.05
        let t_paint  = smoothstep(0.91, 0.95, mat);  // rubber → paint
        let t_mirror = smoothstep(0.95, 0.99, mat);  // paint  → glass / chrome
        // α=0.91 → rough=0.90 (matte), α=0.95 → rough=0.35 (semi-gloss paint),
        // α=0.97 → rough=0.20 (glass/chrome), α=0.99 → rough=0.05 (mirror).
        let roughness = mix(mix(0.90, 0.35, t_paint), 0.05, t_mirror);

        // Detect metallic: alpha >= 0.97 AND bright/neutral colour → conductor
        let lum     = dot(transform.color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        let metallic = smoothstep(0.965, 0.99, mat) * smoothstep(0.40, 0.72, lum);

        var base_col = shade_pbr(N, V, transform.color.rgb, roughness, metallic);

        // Asphalt grain on dark matte surfaces
        if mat < 0.93 && lum < 0.20 {
            let g1 = vnoise(in.world_pos.xz * 5.0)  * 0.08;
            let g2 = vnoise(in.world_pos.xz * 24.0) * 0.04;
            base_col *= 0.95 + g1 + g2;
        }

        out_color = apply_fog(base_col, in.world_pos);
        out_color = aces(out_color * 1.10);
        out_color = color_grade(out_color);
        out_color = pow(out_color, vec3<f32>(1.0 / 2.2));
    }

    return vec4<f32>(out_color, out_alpha);
}
