// renderer/mod.rs — multi-mesh, per-object-colour GPU renderer.
//
// Terrain chunks are stored in two pre-allocated pools (terrain_chunks and
// water_chunks, MAX_CHUNKS slots each).  When a chunk is generated or freed
// the caller uploads new mesh data with upload_terrain_chunk / upload_water_chunk.
//
// DrawCmds with mesh_id >= MESH_WATER_BASE → water_chunks pool
// DrawCmds with mesh_id >= MESH_TERRAIN_BASE → terrain_chunks pool
// All others → static meshes array (cube, cylinder, road, cone, sphere …)

pub mod buffer;
pub mod mesh;
pub mod pipeline;

use std::sync::Arc;
use winit::window::Window;

use crate::scene::camera::{Camera, CameraUniform};
use crate::scene::grass::GrassInstance;
use crate::scene::splash::SplashInstance;
use crate::scene::transform::UNIFORM_ALIGN;
use crate::scene::vehicle::{DrawCmd, MESH_TERRAIN_BASE, MESH_WATER_BASE};
use crate::terrain::{CHUNK_CELLS, CHUNK_VERTS, MAX_CHUNKS, build_chunk_mesh, build_chunk_water_mesh};
use buffer::{create_index_buffer, create_vertex_buffer, Vertex};
use mesh::{build_cube, build_cylinder, build_cone, build_sphere};
use pipeline::{create_bind_group, create_bind_group_layout, create_render_pipeline,
               create_water_pipeline, create_sky_pipeline,
               create_grass_bind_group, create_grass_bind_group_layout, create_grass_pipeline,
               create_splash_bind_group, create_splash_bind_group_layout, create_splash_pipeline,
               create_droplet_bind_group, create_droplet_bind_group_layout, create_droplet_pipeline};
use wgpu::util::DeviceExt;

// ── Constants ────────────────────────────────────────────────────────────────

const MAX_INSTANCES: usize = 8192;

#[cfg(not(target_arch = "wasm32"))]
const MSAA_SAMPLES: u32 = 4;
#[cfg(target_arch = "wasm32")]
const MSAA_SAMPLES: u32 = 1;

// ── GpuMesh ──────────────────────────────────────────────────────────────────

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer:  wgpu::Buffer,
    pub index_count:   u32,
}

// ── Renderer ─────────────────────────────────────────────────────────────────

pub struct Renderer {
    #[cfg(not(target_arch = "wasm32"))]
    _window:        Arc<Window>,
    surface:        wgpu::Surface<'static>,
    device:         wgpu::Device,
    queue:          wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,

    pipeline:       wgpu::RenderPipeline,
    pipeline_water: wgpu::RenderPipeline,
    pipeline_sky:   wgpu::RenderPipeline,
    bind_group:     wgpu::BindGroup,

    /// Static mesh shapes: 0=cube, 1=cylinder, 2=stub, 3=stub, 4=road, 5=cone, 6=sphere
    meshes:         Vec<GpuMesh>,

    /// Per-chunk terrain mesh pool: MAX_CHUNKS pre-allocated slots.
    /// Slot i → terrain_chunks[i].  Updated via upload_terrain_chunk().
    terrain_chunks: Vec<GpuMesh>,
    /// Per-chunk water mesh pool.
    water_chunks:   Vec<GpuMesh>,

    transform_pool: wgpu::Buffer,
    staging:        Vec<u8>,
    camera_buf:     wgpu::Buffer,
    depth_texture:  wgpu::Texture,
    depth_view:     wgpu::TextureView,
    msaa_texture:   Option<wgpu::Texture>,
    msaa_view:      Option<wgpu::TextureView>,

    // Grass rendering
    grass_pipeline:          wgpu::RenderPipeline,
    grass_bind_group:        wgpu::BindGroup,
    grass_blade_vbuf:        wgpu::Buffer,
    grass_blade_ibuf:        wgpu::Buffer,
    grass_blade_index_count: u32,
    grass_instance_buf:      wgpu::Buffer,
    grass_instance_count:    u32,

    // Splash particle rendering
    splash_pipeline:      wgpu::RenderPipeline,
    splash_bind_group:    wgpu::BindGroup,
    splash_quad_vbuf:     wgpu::Buffer,
    splash_quad_ibuf:     wgpu::Buffer,
    splash_instance_buf:  wgpu::Buffer,
    splash_instance_count: u32,

    // Screen droplet post-process
    /// Intermediate RGBA texture: scene renders here, then post-process samples it.
    screen_tex:       wgpu::Texture,
    screen_view:      wgpu::TextureView,
    screen_sampler:   wgpu::Sampler,
    droplet_pipeline: wgpu::RenderPipeline,
    droplet_bgl:      wgpu::BindGroupLayout,
    droplet_bg:       wgpu::BindGroup,
    droplet_params:   wgpu::Buffer,
    /// Accumulated wetness [0,1] — written by platform each frame.
    pub wetness:      f32,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, road: &crate::terrain::RoadSpline) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // On WASM, create the surface from the canvas element directly so that
        // wgpu never calls back into winit's Canvas RefCell (which would cause
        // "RefCell already borrowed" panics from re-entrant borrow_mut calls).
        #[cfg(target_arch = "wasm32")]
        let surface = {
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().expect("no canvas on window");
            instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                .expect("create surface")
        };
        #[cfg(not(target_arch = "wasm32"))]
        let surface = instance.create_surface(window.clone()).expect("create surface");

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference:       wgpu::PowerPreference::HighPerformance,
            compatible_surface:     Some(&surface),
            force_fallback_adapter: false,
        }).await.expect("no adapter");

        log::info!("adapter: {:?}", adapter.get_info());

        #[cfg(not(target_arch = "wasm32"))]
        let required_limits = adapter.limits();
        #[cfg(target_arch = "wasm32")]
        let required_limits = wgpu::Limits::downlevel_webgl2_defaults();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label:             Some("dev"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints:      Default::default(),
            },
            None,
        ).await.unwrap_or_else(|e| panic!("device: {e}"));

        // Surface configuration
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied()
                         .unwrap_or(caps.formats[0]);
        let max_dim = device.limits().max_texture_dimension_2d;
        let surface_config = wgpu::SurfaceConfiguration {
            usage:    wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:    size.width.clamp(1, max_dim),
            height:   size.height.clamp(1, max_dim),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let (depth_texture, depth_view) =
            depth_texture(&device, surface_config.width, surface_config.height, MSAA_SAMPLES);

        let (msaa_texture, msaa_view) = if MSAA_SAMPLES > 1 {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("msaa"),
                size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: MSAA_SAMPLES,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (Some(tex), Some(view))
        } else {
            (None, None)
        };

        // ── Static mesh library ───────────────────────────────────────────
        let (cube_v,  cube_i)  = build_cube();
        let (cyl_v,   cyl_i)   = build_cylinder(20);
        let (cone_v,  cone_i)  = build_cone(16);
        let (sph_v,   sph_i)   = build_sphere(8, 12);
        let (road_v,  road_i)  = road.build_road_mesh();

        fn stub(device: &wgpu::Device) -> GpuMesh {
            let v = vec![Vertex { position: [0.0,0.0,0.0], normal: [0.0,1.0,0.0], uv: [0.0,0.0] }];
            let i: Vec<u16> = vec![0, 0, 0];
            GpuMesh {
                vertex_buffer: create_vertex_buffer(device, &v),
                index_buffer:  create_index_buffer(device, &i),
                index_count:   0,
            }
        }

        let meshes = vec![
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &cube_v),  index_buffer: create_index_buffer(&device, &cube_i),  index_count: cube_i.len()  as u32 }, // 0 CUBE
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &cyl_v),   index_buffer: create_index_buffer(&device, &cyl_i),   index_count: cyl_i.len()   as u32 }, // 1 WHEEL
            stub(&device),  // 2 TERRAIN stub (chunks used instead)
            stub(&device),  // 3 WATER   stub (chunks used instead)
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &road_v),  index_buffer: create_index_buffer(&device, &road_i),  index_count: road_i.len()  as u32 }, // 4 ROAD
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &cone_v),  index_buffer: create_index_buffer(&device, &cone_i),  index_count: cone_i.len()  as u32 }, // 5 CONE
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &sph_v),   index_buffer: create_index_buffer(&device, &sph_i),   index_count: sph_i.len()   as u32 }, // 6 SPHERE
            stub(&device),  // 7 stub (reserved)
        ];

        // ── Terrain chunk pool ────────────────────────────────────────────
        // Terrain mesh: CHUNK_VERTS × CHUNK_VERTS vertices (shared grid).
        let terrain_v_bytes = (CHUNK_VERTS * CHUNK_VERTS * std::mem::size_of::<Vertex>()) as u64;
        let terrain_i_bytes = (CHUNK_CELLS * CHUNK_CELLS * 6 * std::mem::size_of::<u16>()) as u64;

        // Water mesh: worst case every quad below water = CHUNK_CELLS² quads × 4 verts.
        let water_v_bytes = (CHUNK_CELLS * CHUNK_CELLS * 4 * std::mem::size_of::<Vertex>()) as u64;
        let water_i_bytes = (CHUNK_CELLS * CHUNK_CELLS * 6 * std::mem::size_of::<u16>()) as u64;

        let terrain_chunks: Vec<GpuMesh> = (0..MAX_CHUNKS).map(|_| GpuMesh {
            vertex_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: terrain_v_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: terrain_i_bytes,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_count: 0,
        }).collect();

        let water_chunks: Vec<GpuMesh> = (0..MAX_CHUNKS).map(|_| GpuMesh {
            vertex_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: water_v_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: water_i_bytes,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_count: 0,
        }).collect();

        // ── Camera uniform buffer ─────────────────────────────────────────
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cam"),
            size:  std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Transform pool ────────────────────────────────────────────────
        let pool_size = (UNIFORM_ALIGN * MAX_INSTANCES) as u64;
        let transform_pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool"), size: pool_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = vec![0u8; UNIFORM_ALIGN * MAX_INSTANCES];

        let bgl        = create_bind_group_layout(&device);
        let bind_group = create_bind_group(&device, &bgl, &camera_buf, &transform_pool);
        let pipeline       = create_render_pipeline(&device, format, &bgl, MSAA_SAMPLES);
        let pipeline_water = create_water_pipeline(&device, format, &bgl, MSAA_SAMPLES);
        let pipeline_sky   = create_sky_pipeline(&device, format, &bgl, MSAA_SAMPLES);

        // ── Grass pipeline setup ──────────────────────────────────────────────────

        // Grass clump mesh: 3 blades at 0°/120°/240°, each with 5 levels + tip.
        // Each vert: [lpos: vec3, height_t: f32] = 16 bytes.
        // Total: 33 verts, 81 indices.  Instance rotation rotates the whole clump.
        let levels: &[f32] = &[0.0, 0.2, 0.4, 0.6, 0.8];
        let mut blade_verts: Vec<f32> = Vec::new();
        let mut blade_indices: Vec<u16> = Vec::new();

        let sub_angles: [f32; 3] = [
            0.0,
            std::f32::consts::TAU / 3.0,
            std::f32::consts::TAU * 2.0 / 3.0,
        ];

        for (b, &sub_a) in sub_angles.iter().enumerate() {
            let base = (b as u16) * 11;
            let sin_a = sub_a.sin();
            let cos_a = sub_a.cos();

            for &t in levels {
                let half_w = 0.10 * (1.0 - t * 0.85);
                let curve  = 0.15 * t * t;
                // Left vertex rotated by sub_a around Y
                let lx = -half_w * cos_a - curve * sin_a;
                let lz = -half_w * sin_a + curve * cos_a;
                blade_verts.extend_from_slice(&[lx, t, lz, t]);
                // Right vertex rotated by sub_a
                let rx =  half_w * cos_a - curve * sin_a;
                let rz =  half_w * sin_a + curve * cos_a;
                blade_verts.extend_from_slice(&[rx, t, rz, t]);
            }
            // Tip: local (0, 1, 0.15) rotated by sub_a
            let tx = -0.15 * sin_a;
            let tz =  0.15 * cos_a;
            blade_verts.extend_from_slice(&[tx, 1.0, tz, 1.0]);

            // Quad strip + tip triangle
            blade_indices.extend_from_slice(&[
                base+0, base+2, base+1,   base+1, base+2, base+3,
                base+2, base+4, base+3,   base+3, base+4, base+5,
                base+4, base+6, base+5,   base+5, base+6, base+7,
                base+6, base+8, base+7,   base+7, base+8, base+9,
                base+8, base+10, base+9,
            ]);
        }

        let grass_blade_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass_blade_v"),
            contents: bytemuck::cast_slice(&blade_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let grass_blade_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass_blade_i"),
            contents: bytemuck::cast_slice(&blade_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let grass_blade_index_count = blade_indices.len() as u32;

        // Pre-allocated instance buffer (80k clumps × 32 bytes = 2.5 MB)
        const MAX_GRASS: usize = 80_000;
        let grass_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_inst"),
            size: (MAX_GRASS * std::mem::size_of::<GrassInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grass_bgl        = create_grass_bind_group_layout(&device);
        let grass_bind_group = create_grass_bind_group(&device, &grass_bgl, &camera_buf);
        let grass_pipeline   = create_grass_pipeline(&device, format, &grass_bgl, MSAA_SAMPLES);

        // ── Splash particle setup ─────────────────────────────────────────────
        // Unit quad: 4 verts at (±0.5, ±0.5), 6 indices.
        let quad_verts: Vec<f32> = vec![
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5,
        ];
        let quad_idx: Vec<u16> = vec![0, 1, 2, 0, 2, 3];
        let splash_quad_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splash_qv"), contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let splash_quad_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splash_qi"), contents: bytemuck::cast_slice(&quad_idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let splash_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splash_inst"),
            size: (crate::scene::splash::MAX_SPLASH * std::mem::size_of::<SplashInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let splash_bgl        = create_splash_bind_group_layout(&device);
        let splash_bind_group = create_splash_bind_group(&device, &splash_bgl, &camera_buf);
        let splash_pipeline   = create_splash_pipeline(&device, format, &splash_bgl, MSAA_SAMPLES);

        // ── Screen droplet post-process setup ─────────────────────────────────
        let (screen_tex, screen_view) = screen_color_texture(
            &device, surface_config.width, surface_config.height, format);
        let screen_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:         Some("screen_samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:    wgpu::FilterMode::Linear,
            min_filter:    wgpu::FilterMode::Linear,
            ..Default::default()
        });

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct DropletParamsGpu { wetness: f32, time: f32, _pad: [f32; 2] }
        let droplet_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("droplet_params"),
            size: std::mem::size_of::<DropletParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let droplet_bgl = create_droplet_bind_group_layout(&device);
        let droplet_bg  = create_droplet_bind_group(
            &device, &droplet_bgl, &screen_view, &screen_sampler, &droplet_params);
        let droplet_pipeline = create_droplet_pipeline(&device, format, &droplet_bgl);

        Self {
            #[cfg(not(target_arch = "wasm32"))]
            _window: window, surface, device, queue, surface_config,
            pipeline, pipeline_water, pipeline_sky, bind_group,
            meshes, terrain_chunks, water_chunks,
            transform_pool, staging, camera_buf,
            depth_texture, depth_view,
            msaa_texture, msaa_view,
            grass_pipeline,
            grass_bind_group,
            grass_blade_vbuf,
            grass_blade_ibuf,
            grass_blade_index_count,
            grass_instance_buf,
            grass_instance_count: 0,
            splash_pipeline,
            splash_bind_group,
            splash_quad_vbuf,
            splash_quad_ibuf,
            splash_instance_buf,
            splash_instance_count: 0,
            screen_tex,
            screen_view,
            screen_sampler,
            droplet_pipeline,
            droplet_bgl,
            droplet_bg,
            droplet_params,
            wetness: 0.0,
        }
    }

    // ── WASM constructor (bypasses winit entirely) ────────────────────────────

    /// WASM-only constructor: creates the renderer from a raw HtmlCanvasElement
    /// without requiring a winit Window, bypassing winit's buggy web event loop.
    #[cfg(target_arch = "wasm32")]
    pub async fn new_wasm(canvas: web_sys::HtmlCanvasElement, road: &crate::terrain::RoadSpline) -> Self {
        let w = canvas.width().max(1);
        let h = canvas.height().max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .expect("create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no adapter");

        log::info!("adapter: {:?}", adapter.get_info());

        let required_limits = wgpu::Limits::downlevel_webgl2_defaults();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("dev"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints:      Default::default(),
                },
                None,
            )
            .await
            .unwrap_or_else(|e| panic!("device: {e}"));

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied()
                         .unwrap_or(caps.formats[0]);
        let max_dim = device.limits().max_texture_dimension_2d;
        let surface_config = wgpu::SurfaceConfiguration {
            usage:    wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:    w.min(max_dim),
            height:   h.min(max_dim),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let (depth_texture, depth_view) =
            depth_texture(&device, surface_config.width, surface_config.height, MSAA_SAMPLES);

        let (msaa_texture, msaa_view) = if MSAA_SAMPLES > 1 {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("msaa"),
                size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: MSAA_SAMPLES,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (Some(tex), Some(view))
        } else {
            (None, None)
        };

        // ── Static mesh library ───────────────────────────────────────────
        let (cube_v,  cube_i)  = build_cube();
        let (cyl_v,   cyl_i)   = build_cylinder(20);
        let (cone_v,  cone_i)  = build_cone(16);
        let (sph_v,   sph_i)   = build_sphere(8, 12);
        let (road_v,  road_i)  = road.build_road_mesh();

        fn stub(device: &wgpu::Device) -> GpuMesh {
            let v = vec![Vertex { position: [0.0,0.0,0.0], normal: [0.0,1.0,0.0], uv: [0.0,0.0] }];
            let i: Vec<u16> = vec![0, 0, 0];
            GpuMesh {
                vertex_buffer: create_vertex_buffer(device, &v),
                index_buffer:  create_index_buffer(device, &i),
                index_count:   0,
            }
        }

        let meshes = vec![
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &cube_v),  index_buffer: create_index_buffer(&device, &cube_i),  index_count: cube_i.len()  as u32 },
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &cyl_v),   index_buffer: create_index_buffer(&device, &cyl_i),   index_count: cyl_i.len()   as u32 },
            stub(&device),
            stub(&device),
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &road_v),  index_buffer: create_index_buffer(&device, &road_i),  index_count: road_i.len()  as u32 },
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &cone_v),  index_buffer: create_index_buffer(&device, &cone_i),  index_count: cone_i.len()  as u32 },
            GpuMesh { vertex_buffer: create_vertex_buffer(&device, &sph_v),   index_buffer: create_index_buffer(&device, &sph_i),   index_count: sph_i.len()   as u32 },
            stub(&device),
        ];

        // ── Terrain chunk pool ────────────────────────────────────────────
        let terrain_v_bytes = (CHUNK_VERTS * CHUNK_VERTS * std::mem::size_of::<Vertex>()) as u64;
        let terrain_i_bytes = (CHUNK_CELLS * CHUNK_CELLS * 6 * std::mem::size_of::<u16>()) as u64;
        let water_v_bytes = (CHUNK_CELLS * CHUNK_CELLS * 4 * std::mem::size_of::<Vertex>()) as u64;
        let water_i_bytes = (CHUNK_CELLS * CHUNK_CELLS * 6 * std::mem::size_of::<u16>()) as u64;

        let terrain_chunks: Vec<GpuMesh> = (0..MAX_CHUNKS).map(|_| GpuMesh {
            vertex_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: terrain_v_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: terrain_i_bytes,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_count: 0,
        }).collect();

        let water_chunks: Vec<GpuMesh> = (0..MAX_CHUNKS).map(|_| GpuMesh {
            vertex_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: water_v_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: water_i_bytes,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            index_count: 0,
        }).collect();

        // ── Camera uniform buffer ─────────────────────────────────────────
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cam"),
            size:  std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Transform pool ────────────────────────────────────────────────
        let pool_size = (UNIFORM_ALIGN * MAX_INSTANCES) as u64;
        let transform_pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool"), size: pool_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = vec![0u8; UNIFORM_ALIGN * MAX_INSTANCES];

        let bgl        = create_bind_group_layout(&device);
        let bind_group = create_bind_group(&device, &bgl, &camera_buf, &transform_pool);
        let pipeline       = create_render_pipeline(&device, format, &bgl, MSAA_SAMPLES);
        let pipeline_water = create_water_pipeline(&device, format, &bgl, MSAA_SAMPLES);
        let pipeline_sky   = create_sky_pipeline(&device, format, &bgl, MSAA_SAMPLES);

        // ── Grass pipeline setup ──────────────────────────────────────────────────
        let levels: &[f32] = &[0.0, 0.2, 0.4, 0.6, 0.8];
        let mut blade_verts: Vec<f32> = Vec::new();
        let mut blade_indices: Vec<u16> = Vec::new();

        let sub_angles: [f32; 3] = [
            0.0,
            std::f32::consts::TAU / 3.0,
            std::f32::consts::TAU * 2.0 / 3.0,
        ];

        for (b, &sub_a) in sub_angles.iter().enumerate() {
            let base = (b as u16) * 11;
            let sin_a = sub_a.sin();
            let cos_a = sub_a.cos();

            for &t in levels {
                let half_w = 0.10 * (1.0 - t * 0.85);
                let curve  = 0.15 * t * t;
                let lx = -half_w * cos_a - curve * sin_a;
                let lz = -half_w * sin_a + curve * cos_a;
                blade_verts.extend_from_slice(&[lx, t, lz, t]);
                let rx =  half_w * cos_a - curve * sin_a;
                let rz =  half_w * sin_a + curve * cos_a;
                blade_verts.extend_from_slice(&[rx, t, rz, t]);
            }
            let tx = -0.15 * sin_a;
            let tz =  0.15 * cos_a;
            blade_verts.extend_from_slice(&[tx, 1.0, tz, 1.0]);

            blade_indices.extend_from_slice(&[
                base+0, base+2, base+1,   base+1, base+2, base+3,
                base+2, base+4, base+3,   base+3, base+4, base+5,
                base+4, base+6, base+5,   base+5, base+6, base+7,
                base+6, base+8, base+7,   base+7, base+8, base+9,
                base+8, base+10, base+9,
            ]);
        }

        let grass_blade_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass_blade_v"),
            contents: bytemuck::cast_slice(&blade_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let grass_blade_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass_blade_i"),
            contents: bytemuck::cast_slice(&blade_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let grass_blade_index_count = blade_indices.len() as u32;

        const MAX_GRASS: usize = 80_000;
        let grass_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grass_inst"),
            size: (MAX_GRASS * std::mem::size_of::<GrassInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grass_bgl        = create_grass_bind_group_layout(&device);
        let grass_bind_group = create_grass_bind_group(&device, &grass_bgl, &camera_buf);
        let grass_pipeline   = create_grass_pipeline(&device, format, &grass_bgl, MSAA_SAMPLES);

        // ── Splash particle setup ─────────────────────────────────────────────
        let quad_verts: Vec<f32> = vec![
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5,
        ];
        let quad_idx: Vec<u16> = vec![0, 1, 2, 0, 2, 3];
        let splash_quad_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splash_qv"), contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let splash_quad_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splash_qi"), contents: bytemuck::cast_slice(&quad_idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let splash_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splash_inst"),
            size: (crate::scene::splash::MAX_SPLASH * std::mem::size_of::<SplashInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let splash_bgl        = create_splash_bind_group_layout(&device);
        let splash_bind_group = create_splash_bind_group(&device, &splash_bgl, &camera_buf);
        let splash_pipeline   = create_splash_pipeline(&device, format, &splash_bgl, MSAA_SAMPLES);

        // ── Screen droplet post-process setup ─────────────────────────────────
        let (screen_tex, screen_view) = screen_color_texture(
            &device, surface_config.width, surface_config.height, format);
        let screen_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:         Some("screen_samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:    wgpu::FilterMode::Linear,
            min_filter:    wgpu::FilterMode::Linear,
            ..Default::default()
        });

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct DropletParamsGpu { wetness: f32, time: f32, _pad: [f32; 2] }
        let droplet_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("droplet_params"),
            size: std::mem::size_of::<DropletParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let droplet_bgl = create_droplet_bind_group_layout(&device);
        let droplet_bg  = create_droplet_bind_group(
            &device, &droplet_bgl, &screen_view, &screen_sampler, &droplet_params);
        let droplet_pipeline = create_droplet_pipeline(&device, format, &droplet_bgl);

        Self {
            surface, device, queue, surface_config,
            pipeline, pipeline_water, pipeline_sky, bind_group,
            meshes, terrain_chunks, water_chunks,
            transform_pool, staging, camera_buf,
            depth_texture, depth_view,
            msaa_texture, msaa_view,
            grass_pipeline,
            grass_bind_group,
            grass_blade_vbuf,
            grass_blade_ibuf,
            grass_blade_index_count,
            grass_instance_buf,
            grass_instance_count: 0,
            splash_pipeline,
            splash_bind_group,
            splash_quad_vbuf,
            splash_quad_ibuf,
            splash_instance_buf,
            splash_instance_count: 0,
            screen_tex,
            screen_view,
            screen_sampler,
            droplet_pipeline,
            droplet_bgl,
            droplet_bg,
            droplet_params,
            wetness: 0.0,
        }
    }

    // ── Upload chunk mesh data ────────────────────────────────────────────────

    pub fn upload_terrain_chunk(&mut self, slot: usize, cx: i32, cz: i32, heights: &[f32]) {
        let (verts, indices) = build_chunk_mesh(cx, cz, heights);
        self.queue.write_buffer(&self.terrain_chunks[slot].vertex_buffer, 0,
            bytemuck::cast_slice(&verts));
        self.queue.write_buffer(&self.terrain_chunks[slot].index_buffer, 0,
            bytemuck::cast_slice(&indices));
        self.terrain_chunks[slot].index_count = indices.len() as u32;
    }

    pub fn upload_water_chunk(&mut self, slot: usize, cx: i32, cz: i32, heights: &[f32]) {
        let (verts, indices) = build_chunk_water_mesh(cx, cz, heights);
        if verts.is_empty() {
            self.water_chunks[slot].index_count = 0;
            return;
        }
        self.queue.write_buffer(&self.water_chunks[slot].vertex_buffer, 0,
            bytemuck::cast_slice(&verts));
        self.queue.write_buffer(&self.water_chunks[slot].index_buffer, 0,
            bytemuck::cast_slice(&indices));
        self.water_chunks[slot].index_count = indices.len() as u32;
    }

    pub fn clear_water_chunk(&mut self, slot: usize) {
        self.water_chunks[slot].index_count = 0;
    }

    // ── Grass uploads ─────────────────────────────────────────────────────────

    pub fn upload_splash_instances(&mut self, instances: &[SplashInstance]) {
        let count = instances.len().min(crate::scene::splash::MAX_SPLASH);
        if count > 0 {
            self.queue.write_buffer(&self.splash_instance_buf, 0,
                bytemuck::cast_slice(&instances[..count]));
        }
        self.splash_instance_count = count as u32;
    }

    pub fn upload_grass_instances(&mut self, instances: &[GrassInstance]) {
        let count = instances.len().min(80_000);
        if count > 0 {
            self.queue.write_buffer(&self.grass_instance_buf, 0, bytemuck::cast_slice(&instances[..count]));
        }
        self.grass_instance_count = count as u32;
    }

    // ── Resize ───────────────────────────────────────────────────────────────

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        let max_dim = self.device.limits().max_texture_dimension_2d;
        self.surface_config.width  = w.min(max_dim);
        self.surface_config.height = h.min(max_dim);
        self.surface.configure(&self.device, &self.surface_config);
        let (dt, dv) = depth_texture(&self.device, w, h, MSAA_SAMPLES);
        self.depth_texture = dt;
        self.depth_view    = dv;
        if MSAA_SAMPLES > 1 {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("msaa"),
                size: wgpu::Extent3d { width: w.min(max_dim), height: h.min(max_dim), depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: MSAA_SAMPLES,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.msaa_texture = Some(tex);
            self.msaa_view    = Some(view);
        }
        // Recreate screen color texture and droplet bind group at new size.
        let fmt = self.surface_config.format;
        let (st, sv) = screen_color_texture(&self.device, w.min(max_dim), h.min(max_dim), fmt);
        self.screen_tex  = st;
        self.screen_view = sv;
        self.droplet_bg  = create_droplet_bind_group(
            &self.device, &self.droplet_bgl,
            &self.screen_view, &self.screen_sampler, &self.droplet_params);
    }

    pub fn surface_width(&self)  -> u32 { self.surface_config.width }
    pub fn surface_height(&self) -> u32 { self.surface_config.height }

    // ── Render ───────────────────────────────────────────────────────────────

    pub fn render(
        &mut self,
        camera:   &Camera,
        commands: &[DrawCmd],
    ) -> Result<(), wgpu::SurfaceError> {
        self.queue.write_buffer(&self.camera_buf, 0,
            bytemuck::bytes_of(&camera.build_uniform()));

        // Split by material type
        let sky_cmds:    Vec<_> = commands.iter().filter(|c| c.uniform.color[3] > 1.5).collect();
        let opaque_cmds: Vec<_> = commands.iter().filter(|c|
            c.uniform.color[3] < 0.05 || (c.uniform.color[3] >= 0.90 && c.uniform.color[3] <= 1.5)
        ).collect();
        let water_cmds:  Vec<_> = commands.iter().filter(|c|
            c.uniform.color[3] >= 0.05 && c.uniform.color[3] < 0.90
        ).collect();

        let sky_n    = sky_cmds.len().min(MAX_INSTANCES);
        let opaque_n = opaque_cmds.len().min(MAX_INSTANCES - sky_n);
        let water_n  = water_cmds.len().min(MAX_INSTANCES - sky_n - opaque_n);
        let total    = sky_n + opaque_n + water_n;

        // Write uniforms to staging
        let write = |staging: &mut Vec<u8>, cmds: &[&DrawCmd], base: usize| {
            for (i, cmd) in cmds.iter().enumerate() {
                let off = (base + i) * UNIFORM_ALIGN;
                let bytes = bytemuck::bytes_of(&cmd.uniform);
                staging[off..off + bytes.len()].copy_from_slice(bytes);
            }
        };
        write(&mut self.staging, &sky_cmds[..sky_n],       0);
        write(&mut self.staging, &opaque_cmds[..opaque_n], sky_n);
        write(&mut self.staging, &water_cmds[..water_n],   sky_n + opaque_n);

        if total > 0 {
            self.queue.write_buffer(&self.transform_pool, 0,
                &self.staging[..total * UNIFORM_ALIGN]);
        }

        // Update droplet params uniform.
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct DropletParamsGpu { wetness: f32, time: f32, _pad: [f32; 2] }
        self.queue.write_buffer(&self.droplet_params, 0, bytemuck::bytes_of(
            &DropletParamsGpu { wetness: self.wetness, time: camera.time, _pad: [0.0; 2] }
        ));

        let output = self.surface.get_current_texture()?;
        let swapchain_view = output.texture.create_view(&Default::default());
        // On native: scene renders into screen_view, then the droplet post-process
        // reads it and writes to the swapchain.
        // On WASM/WebGL2: there is no glMemoryBarrier, so wgpu's GL backend cannot
        // guarantee the intermediate texture is flushed between passes — sampling it
        // in the droplet pass would read stale (black) data.  Render directly to
        // the swapchain instead and skip the droplet pass entirely.
        #[cfg(not(target_arch = "wasm32"))]
        let scene_target = &self.screen_view;
        #[cfg(target_arch = "wasm32")]
        let scene_target = &swapchain_view;
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("enc") });

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if let Some(mv) = &self.msaa_view { mv } else { scene_target },
                    resolve_target: if self.msaa_view.is_some() { Some(scene_target) } else { None },
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None, occlusion_query_set: None,
            });

            // Sky pass
            pass.set_pipeline(&self.pipeline_sky);
            for (i, cmd) in sky_cmds.iter().take(sky_n).enumerate() {
                let mesh = self.resolve_mesh(cmd.mesh_id);
                if mesh.index_count == 0 { continue; }
                pass.set_bind_group(0, &self.bind_group, &[(i * UNIFORM_ALIGN) as u32]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            // Opaque pass
            pass.set_pipeline(&self.pipeline);
            for (i, cmd) in opaque_cmds.iter().take(opaque_n).enumerate() {
                let mesh = self.resolve_mesh(cmd.mesh_id);
                if mesh.index_count == 0 { continue; }
                let off = (sky_n + i) * UNIFORM_ALIGN;
                pass.set_bind_group(0, &self.bind_group, &[off as u32]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            // Water pass
            pass.set_pipeline(&self.pipeline_water);
            for (i, cmd) in water_cmds.iter().take(water_n).enumerate() {
                let mesh = self.resolve_mesh(cmd.mesh_id);
                if mesh.index_count == 0 { continue; }
                let off = (sky_n + opaque_n + i) * UNIFORM_ALIGN;
                pass.set_bind_group(0, &self.bind_group, &[off as u32]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        // ── Grass pass ────────────────────────────────────────────────────────────
        if self.grass_instance_count > 0 {
            let mut gp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("grass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if let Some(mv) = &self.msaa_view { mv } else { scene_target },
                    resolve_target: if self.msaa_view.is_some() { Some(scene_target) } else { None },
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None, occlusion_query_set: None,
            });
            gp.set_pipeline(&self.grass_pipeline);
            gp.set_bind_group(0, &self.grass_bind_group, &[]);
            gp.set_vertex_buffer(0, self.grass_blade_vbuf.slice(..));
            gp.set_vertex_buffer(1, self.grass_instance_buf.slice(..));
            gp.set_index_buffer(self.grass_blade_ibuf.slice(..), wgpu::IndexFormat::Uint16);
            gp.draw_indexed(0..self.grass_blade_index_count, 0, 0..self.grass_instance_count);
        }

        // ── Splash pass ───────────────────────────────────────────────────────────
        if self.splash_instance_count > 0 {
            let mut sp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splash"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if let Some(mv) = &self.msaa_view { mv } else { scene_target },
                    resolve_target: if self.msaa_view.is_some() { Some(scene_target) } else { None },
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None, occlusion_query_set: None,
            });
            sp.set_pipeline(&self.splash_pipeline);
            sp.set_bind_group(0, &self.splash_bind_group, &[]);
            sp.set_vertex_buffer(0, self.splash_quad_vbuf.slice(..));
            sp.set_vertex_buffer(1, self.splash_instance_buf.slice(..));
            sp.set_index_buffer(self.splash_quad_ibuf.slice(..), wgpu::IndexFormat::Uint16);
            sp.draw_indexed(0..6, 0, 0..self.splash_instance_count);
        }

        // ── Droplet post-process pass ─────────────────────────────────────────────
        // Native only: on WASM the scene renders directly to the swapchain
        // (scene_target == swapchain_view) so screen_view is never written.
        // Running the droplet pass would sample the empty screen_view and
        // overwrite the swapchain with black.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut dp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("droplets"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swapchain_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None, occlusion_query_set: None,
            });
            dp.set_pipeline(&self.droplet_pipeline);
            dp.set_bind_group(0, &self.droplet_bg, &[]);
            dp.draw(0..3, 0..1);  // fullscreen triangle from vertex_index
        }

        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }

    fn resolve_mesh(&self, mesh_id: usize) -> &GpuMesh {
        if mesh_id >= MESH_WATER_BASE {
            &self.water_chunks[mesh_id - MESH_WATER_BASE]
        } else if mesh_id >= MESH_TERRAIN_BASE {
            &self.terrain_chunks[mesh_id - MESH_TERRAIN_BASE]
        } else {
            &self.meshes[mesh_id]
        }
    }
}

// ── Screen colour texture helper ──────────────────────────────────────────────

fn screen_color_texture(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screen_color"),
        size:  wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

// ── Depth texture helper ──────────────────────────────────────────────────────

fn depth_texture(device: &wgpu::Device, w: u32, h: u32, samples: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size:  wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: samples,   // ← was hardcoded 1
        dimension: wgpu::TextureDimension::D2,
        format:    wgpu::TextureFormat::Depth24Plus,
        usage:     wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
