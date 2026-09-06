// platform/mod.rs — game loop and winit ApplicationHandler (native only).
//
// On WASM the winit event loop is not used.  All WASM game-loop code lives in
// platform/web.rs, which is called directly from lib.rs via wasm_bindgen.

use std::{rc::Rc, cell::RefCell, sync::Arc};

use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, DeviceEvent, DeviceId},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use crate::input::InputState;
use crate::physics::PhysicsWorld;
use crate::renderer::Renderer;
use crate::scene::camera::Camera;
use crate::scene::environment::{environment_draw_cmds, road_ahead_cmds};
use crate::scene::grass::{build_grass_instances, GrassInstance};
use crate::scene::splash::SplashSystem;
use crate::scene::vehicle::{car_draw_cmds, DrawCmd, MESH_CUBE};
use crate::scene::transform::uniform_from_matrix;
use crate::terrain::TerrainManager;
use cgmath::Matrix4;

// ── AppState ─────────────────────────────────────────────────────────────────

struct AppState {
    window:        Arc<Window>,
    renderer:      Renderer,
    camera:        Camera,
    physics:       PhysicsWorld,
    input:         InputState,
    env_cmds:      Vec<DrawCmd>,
    terrain:       TerrainManager,
    /// World-space position at which env_cmds was last built.
    /// Rebuilt only when the car moves more than CELL_SIZE / 2 from this point.
    env_built_at:  [f32; 2],
    // Grass
    grass_instances: Vec<GrassInstance>,
    grass_built_at:  [f32; 2],
    // Splash
    splash: SplashSystem,
}

// ── App ──────────────────────────────────────────────────────────────────────

struct App {
    state:  Rc<RefCell<Option<AppState>>>,
    window: Option<Arc<Window>>,

    #[cfg(not(target_arch = "wasm32"))]
    last_frame: std::time::Instant,
}

impl App {
    fn new() -> Self {
        Self {
            state:  Rc::new(RefCell::new(None)),
            window: None,
            #[cfg(not(target_arch = "wasm32"))]
            last_frame: std::time::Instant::now(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn tick_dt(&mut self) -> f32 {
        let dt = self.last_frame.elapsed().as_secs_f32().min(0.05);
        self.last_frame = std::time::Instant::now();
        dt
    }
}

// ── ApplicationHandler ───────────────────────────────────────────────────────

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Guard: resumed() can be called more than once on some platforms.
        // Only initialize on the first call.
        if self.window.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("Seedroad — WASD/Arrows · Space=brake · Q/E=look · R=upright")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));

        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        self.window = Some(window.clone());

        // ── Native: synchronous init ──────────────────────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        {
            // 1. Build terrain + road spline (CPU)
            let mut terrain = TerrainManager::new();

            // 2. Init renderer (uses road spline for road mesh)
            let mut renderer = pollster::block_on(Renderer::new(window.clone(), &terrain.road));
            let (w, h) = (renderer.surface_width(), renderer.surface_height());

            // 3. Upload all initially loaded terrain chunks to GPU
            //    Collect keys first to avoid borrowing terrain during upload.
            let chunk_info: Vec<(i32, i32, usize, Option<usize>)> = terrain.chunks().iter()
                .map(|(&(cx,cz), cd)| (cx, cz, cd.mesh_slot, cd.water_slot))
                .collect();
            for (cx, cz, slot, water_slot) in &chunk_info {
                let heights = terrain.chunk(&(*cx, *cz)).unwrap().heights.clone();
                renderer.upload_terrain_chunk(*slot, *cx, *cz, &heights);
                if let Some(ws) = water_slot {
                    renderer.upload_water_chunk(*ws, *cx, *cz, &heights);
                }
            }

            // 4. Physics world (uses per-chunk colliders from terrain)
            let physics = PhysicsWorld::new(&terrain);

            // 5. Build initial environment draw commands
            let spawn = terrain.spawn_point();
            let spawn_pos = cgmath::Vector3::new(spawn.0, 0.0, spawn.1);
            let env_cmds = environment_draw_cmds(&terrain, spawn_pos);

            // 6. Grass
            let initial_grass = build_grass_instances(&terrain, spawn_pos);
            renderer.upload_grass_instances(&initial_grass);

            *self.state.borrow_mut() = Some(AppState {
                window,
                camera: Camera::new(w, h),
                renderer,
                physics,
                input:         InputState::default(),
                env_cmds,
                env_built_at:  [spawn.0, spawn.1],
                terrain,
                grass_instances: initial_grass,
                grass_built_at:  [spawn.0, spawn.1],
                splash:          SplashSystem::new(),
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id:        WindowId,
        event:      WindowEvent,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        let native_dt = if matches!(event, WindowEvent::RedrawRequested) {
            self.tick_dt()
        } else {
            0.0
        };

        let mut borrow = self.state.borrow_mut();
        let Some(s) = borrow.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                s.renderer.resize(size.width, size.height);
                s.camera.resize(size.width, size.height);
            }

            WindowEvent::KeyboardInput { event: key_ev, .. } => {
                s.input.handle_key_event(&key_ev);
                if s.input.upright {
                    s.physics.upright_in_place(&s.terrain);
                }
            }

            WindowEvent::RedrawRequested => {
                #[cfg(not(target_arch = "wasm32"))]
                let dt = native_dt;
                #[cfg(target_arch = "wasm32")]
                let dt = 1.0_f32 / 60.0;

                let throttle = if s.input.forward   {  1.0 }
                               else if s.input.backward { -0.6 } else { 0.0 };
                let steer    = if s.input.left  {  0.50 }
                               else if s.input.right { -0.50 } else { 0.0 };
                let brake    = if s.input.brake { 1.0 } else { 0.0 };

                s.camera.time += dt;
                s.physics.apply_controls(throttle, steer, brake);
                s.physics.step(dt);

                // ── Chunk streaming ───────────────────────────────────────────
                let car_pos = s.physics.car_position();
                let (added, removed) = s.terrain.update(car_pos.x, car_pos.z);

                for (cx, cz) in &removed {
                    s.physics.remove_chunk_collider(*cx, *cz);
                }
                for &(cx, cz) in &added {
                    // Clone heights out first to avoid simultaneous borrows
                    let (slot, water_slot, heights) = {
                        let cd = s.terrain.chunk(&(cx, cz)).unwrap();
                        (cd.mesh_slot, cd.water_slot, cd.heights.clone())
                    };
                    s.renderer.upload_terrain_chunk(slot, cx, cz, &heights);
                    if let Some(ws) = water_slot {
                        s.renderer.upload_water_chunk(ws, cx, cz, &heights);
                    }
                    s.physics.add_chunk_collider(cx, cz, &heights);
                }
                // Rebuild env_cmds only when the car has moved half a tree-cell (~12 m).
                // The tree grid is world-snapped so it doesn't change until then.
                const ENV_REBUILD_DIST: f32 = 12.0;
                let dx = car_pos.x - s.env_built_at[0];
                let dz = car_pos.z - s.env_built_at[1];
                if !added.is_empty() || dx * dx + dz * dz > ENV_REBUILD_DIST * ENV_REBUILD_DIST {
                    s.env_cmds = environment_draw_cmds(&s.terrain, car_pos);
                    s.env_built_at = [car_pos.x, car_pos.z];
                }

                // ── Grass rebuild ─────────────────────────────────────────────
                const GRASS_REBUILD_DIST: f32 = 8.0;
                {
                    let gx = car_pos.x - s.grass_built_at[0];
                    let gz = car_pos.z - s.grass_built_at[1];
                    if gx * gx + gz * gz > GRASS_REBUILD_DIST * GRASS_REBUILD_DIST {
                        s.grass_instances = build_grass_instances(&s.terrain, car_pos);
                        s.renderer.upload_grass_instances(&s.grass_instances);
                        s.grass_built_at = [car_pos.x, car_pos.z];
                    }
                }

                // ── Camera ────────────────────────────────────────────────────
                let target_yaw = if s.input.look_left { 1.25_f32 }
                                 else if s.input.look_right { -1.25_f32 }
                                 else { 0.0_f32 };
                let yaw_t = 1.0 - (-6.0_f32 * dt).exp();
                s.camera.yaw_offset += (target_yaw - s.camera.yaw_offset) * yaw_t;

                let car_fwd = s.physics.car_forward();
                s.camera.follow(car_pos, car_fwd, dt);
                s.camera.car_pos = car_pos.into();

                // ── Splash particles ──────────────────────────────────────────
                let wheel_data = s.physics.wheel_world_data();
                s.splash.update(dt, &wheel_data, &s.terrain);
                let splash_instances = s.splash.instances();
                // Wetness: ramp up while actively splashing, slow decay when dry.
                let pc = s.splash.particle_count();
                if pc > 0 {
                    let intensity = (pc as f32 / 200.0).min(1.0);
                    s.renderer.wetness = (s.renderer.wetness + dt * 2.0 * intensity).min(1.0);
                } else {
                    s.renderer.wetness = (s.renderer.wetness - dt * 0.04).max(0.0);
                }
                s.renderer.upload_splash_instances(&splash_instances);

                // ── Build draw list ───────────────────────────────────────────
                let sky_m = Matrix4::from_translation(s.camera.eye) * Matrix4::from_scale(850.0_f32);
                let sky_cmd = DrawCmd {
                    mesh_id: MESH_CUBE,
                    uniform: uniform_from_matrix(sky_m, [0.0, 0.0, 0.0, 10.0]),
                };
                let mut cmds = Vec::with_capacity(8192);
                cmds.push(sky_cmd);
                for c in &s.env_cmds {
                    cmds.push(DrawCmd { mesh_id: c.mesh_id, uniform: c.uniform });
                }
                cmds.extend(road_ahead_cmds(car_pos, &s.terrain));
                cmds.extend(car_draw_cmds(&s.physics));

                match s.renderer.render(&s.camera, &cmds) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        let (w, h) = (s.renderer.surface_width(), s.renderer.surface_height());
                        s.renderer.resize(w, h);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => log::warn!("render: {e:?}"),
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(w) = &self.window { w.request_redraw(); }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, _: DeviceEvent) {}
}

// ── Logging / panic hook ──────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
pub fn init_wasm() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Warn).expect("console_log");
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run() {
    let event_loop = EventLoop::new().expect("event loop");

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

// ── WASM game loop (bypasses winit entirely) ──────────────────────────────────

#[cfg(target_arch = "wasm32")]
pub mod web;
