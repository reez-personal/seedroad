// platform/web.rs — WASM-only game loop using raw web APIs.
//
// Bypasses winit's event loop entirely to avoid the "RefCell already
// borrowed" panic in winit 0.30's web Canvas implementation.  All game
// logic is identical to the native path; only the plumbing differs.

use std::{rc::Rc, cell::RefCell};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use cgmath::Matrix4;

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

// ── Game state ────────────────────────────────────────────────────────────────

struct AppState {
    renderer:        Renderer,
    camera:          Camera,
    physics:         PhysicsWorld,
    input:           InputState,
    env_cmds:        Vec<DrawCmd>,
    terrain:         TerrainManager,
    env_built_at:    [f32; 2],
    grass_instances: Vec<GrassInstance>,
    grass_built_at:  [f32; 2],
    splash:          SplashSystem,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run() {
    // ── 1. Canvas ─────────────────────────────────────────────────────────────
    let doc = web_sys::window().unwrap().document().unwrap();
    let canvas: web_sys::HtmlCanvasElement = doc
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();

    let win = web_sys::window().unwrap();
    let vw = win.inner_width().unwrap().as_f64().unwrap() as u32;
    let vh = win.inner_height().unwrap().as_f64().unwrap() as u32;
    canvas.set_width(vw.max(1));
    canvas.set_height(vh.max(1));

    let el: &web_sys::HtmlElement = canvas.unchecked_ref();
    doc.body().unwrap().append_child(el).unwrap();
    el.set_attribute(
        "style",
        "position:fixed;top:0;left:0;width:100%;height:100%;display:block;",
    )
    .unwrap();

    // ── 2. Keyboard input ─────────────────────────────────────────────────────
    let input: Rc<RefCell<InputState>> = Rc::new(RefCell::new(InputState::default()));

    {
        let inp = input.clone();
        let cb = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(
            move |e: web_sys::KeyboardEvent| {
                inp.borrow_mut().handle_web_key(&e.code(), true);
            },
        );
        web_sys::window()
            .unwrap()
            .add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }
    {
        let inp = input.clone();
        let cb = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(
            move |e: web_sys::KeyboardEvent| {
                inp.borrow_mut().handle_web_key(&e.code(), false);
            },
        );
        web_sys::window()
            .unwrap()
            .add_event_listener_with_callback("keyup", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // ── 3. Touch controls (mobile) ────────────────────────────────────────────
    // Each button sends touchstart → set flag true, touchend/cancel → false.
    {
        let doc = web_sys::window().unwrap().document().unwrap();

        // (id, start-setter, end-setter)
        struct Btn { id: &'static str }
        let buttons: &[(&str, fn(&mut InputState, bool))] = &[
            ("ctrl-left",  |s, v| s.left     = v),
            ("ctrl-right", |s, v| s.right    = v),
            ("ctrl-gas",   |s, v| s.forward  = v),
            ("ctrl-brake", |s, v| s.brake    = v),
            ("ctrl-reset", |s, v| s.upright  = v),
        ];

        for &(id, setter) in buttons {
            if let Some(el) = doc.get_element_by_id(id) {
                use wasm_bindgen::JsCast;
                let el: web_sys::HtmlElement = el.dyn_into().unwrap();

                // touchstart → true
                let inp = input.clone();
                let cb = Closure::<dyn FnMut(web_sys::TouchEvent)>::new(
                    move |e: web_sys::TouchEvent| {
                        e.prevent_default();
                        setter(&mut inp.borrow_mut(), true);
                    },
                );
                el.add_event_listener_with_callback("touchstart", cb.as_ref().unchecked_ref()).unwrap();
                cb.forget();

                // touchend → false
                let inp = input.clone();
                let cb = Closure::<dyn FnMut(web_sys::TouchEvent)>::new(
                    move |e: web_sys::TouchEvent| {
                        e.prevent_default();
                        setter(&mut inp.borrow_mut(), false);
                    },
                );
                el.add_event_listener_with_callback("touchend", cb.as_ref().unchecked_ref()).unwrap();
                cb.forget();

                // touchcancel → false
                let inp = input.clone();
                let cb = Closure::<dyn FnMut(web_sys::TouchEvent)>::new(
                    move |_: web_sys::TouchEvent| {
                        setter(&mut inp.borrow_mut(), false);
                    },
                );
                el.add_event_listener_with_callback("touchcancel", cb.as_ref().unchecked_ref()).unwrap();
                cb.forget();
            }
        }
    }

    // ── 4. Game initialisation ────────────────────────────────────────────────
    let mut terrain = TerrainManager::new();
    let mut renderer = Renderer::new_wasm(canvas.clone(), &terrain.road).await;
    let (w, h) = (renderer.surface_width(), renderer.surface_height());

    {
        let chunk_info: Vec<(i32, i32, usize, Option<usize>)> = terrain
            .chunks()
            .iter()
            .map(|(&(cx, cz), cd)| (cx, cz, cd.mesh_slot, cd.water_slot))
            .collect();
        for (cx, cz, slot, water_slot) in &chunk_info {
            let heights = terrain.chunk(&(*cx, *cz)).unwrap().heights.clone();
            renderer.upload_terrain_chunk(*slot, *cx, *cz, &heights);
            if let Some(ws) = water_slot {
                renderer.upload_water_chunk(*ws, *cx, *cz, &heights);
            }
        }
    }

    let physics = PhysicsWorld::new(&terrain);
    let spawn = terrain.spawn_point();
    let spawn_pos = cgmath::Vector3::new(spawn.0, 0.0, spawn.1);
    let env_cmds = environment_draw_cmds(&terrain, spawn_pos);
    let initial_grass = build_grass_instances(&terrain, spawn_pos);
    renderer.upload_grass_instances(&initial_grass);

    let state: Rc<RefCell<AppState>> = Rc::new(RefCell::new(AppState {
        renderer,
        camera:          Camera::new(w, h),
        physics,
        input:           InputState::default(),
        env_cmds,
        env_built_at:    [spawn.0, spawn.1],
        terrain,
        grass_instances: initial_grass,
        grass_built_at:  [spawn.0, spawn.1],
        splash:          SplashSystem::new(),
    }));

    // ── 4. requestAnimationFrame loop ─────────────────────────────────────────
    //
    // Classic Rc/RefCell/Closure self-referential rAF pattern:
    // `raf` owns the Closure; `raf_inner` is cloned into the closure itself
    // so it can re-schedule the next frame.
    let raf: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let raf_inner = raf.clone();
    let last_ts:   Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));
    let state_raf  = state.clone();
    let input_raf  = input.clone();
    let last_ts2   = last_ts.clone();

    *raf.borrow_mut() = Some(Closure::new(move |ts: f64| {
        // ── dt ────────────────────────────────────────────────────────────────
        let mut lt = last_ts2.borrow_mut();
        let dt = if *lt == 0.0 {
            1.0_f32 / 60.0
        } else {
            ((ts - *lt) / 1000.0) as f32
        };
        *lt = ts;
        drop(lt);
        let dt = dt.clamp(1.0 / 240.0, 0.05);

        // ── sync input, tick ──────────────────────────────────────────────────
        {
            let mut s = state_raf.borrow_mut();
            s.input = input_raf.borrow().clone();
            game_tick(&mut s, dt);
        }

        // ── schedule next frame ───────────────────────────────────────────────
        web_sys::window()
            .unwrap()
            .request_animation_frame(
                raf_inner
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .as_ref()
                    .unchecked_ref(),
            )
            .unwrap();
    }));

    // Kick off the loop.
    web_sys::window()
        .unwrap()
        .request_animation_frame(
            raf.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
        )
        .unwrap();
}

// ── Per-frame game tick ───────────────────────────────────────────────────────

fn game_tick(s: &mut AppState, dt: f32) {
    // Upright action: triggered while R is held.
    if s.input.upright {
        s.physics.upright_in_place(&s.terrain);
    }

    let throttle = if s.input.forward { 1.0 } else if s.input.backward { -0.6 } else { 0.0 };
    let steer    = if s.input.left  {  0.50 } else if s.input.right { -0.50 } else { 0.0 };
    let brake    = if s.input.brake { 1.0 } else { 0.0 };

    s.camera.time += dt;
    s.physics.apply_controls(throttle, steer, brake);
    s.physics.step(dt);

    // ── Chunk streaming ───────────────────────────────────────────────────────
    let car_pos = s.physics.car_position();
    let (added, removed) = s.terrain.update(car_pos.x, car_pos.z);

    for (cx, cz) in &removed {
        s.physics.remove_chunk_collider(*cx, *cz);
    }
    for &(cx, cz) in &added {
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

    const ENV_REBUILD_DIST: f32 = 12.0;
    let dx = car_pos.x - s.env_built_at[0];
    let dz = car_pos.z - s.env_built_at[1];
    if !added.is_empty() || dx * dx + dz * dz > ENV_REBUILD_DIST * ENV_REBUILD_DIST {
        s.env_cmds = environment_draw_cmds(&s.terrain, car_pos);
        s.env_built_at = [car_pos.x, car_pos.z];
    }

    // ── Grass rebuild ─────────────────────────────────────────────────────────
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

    // ── Camera ────────────────────────────────────────────────────────────────
    let target_yaw = if s.input.look_left { 1.25_f32 }
                     else if s.input.look_right { -1.25_f32 }
                     else { 0.0_f32 };
    let yaw_t = 1.0 - (-6.0_f32 * dt).exp();
    s.camera.yaw_offset += (target_yaw - s.camera.yaw_offset) * yaw_t;

    let car_fwd = s.physics.car_forward();
    s.camera.follow(car_pos, car_fwd, dt);
    s.camera.car_pos = car_pos.into();

    // ── Splash particles ──────────────────────────────────────────────────────
    let wheel_data = s.physics.wheel_world_data();
    s.splash.update(dt, &wheel_data, &s.terrain);
    let splash_instances = s.splash.instances();
    let pc = s.splash.particle_count();
    if pc > 0 {
        let intensity = (pc as f32 / 200.0).min(1.0);
        s.renderer.wetness = (s.renderer.wetness + dt * 2.0 * intensity).min(1.0);
    } else {
        s.renderer.wetness = (s.renderer.wetness - dt * 0.04).max(0.0);
    }
    s.renderer.upload_splash_instances(&splash_instances);

    // ── Draw list ─────────────────────────────────────────────────────────────
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

    // ── Render ────────────────────────────────────────────────────────────────
    match s.renderer.render(&s.camera, &cmds) {
        Ok(_) => {}
        Err(wgpu::SurfaceError::Lost) => {
            let (w, h) = (s.renderer.surface_width(), s.renderer.surface_height());
            s.renderer.resize(w, h);
        }
        Err(e) => log::warn!("render: {e:?}"),
    }
}
