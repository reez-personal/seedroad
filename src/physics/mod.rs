// physics/mod.rs — rapier3d 0.22 physics world + vehicle controller.
//
// Per-chunk heightfield colliders: each terrain chunk (256m × 256m) has its
// own ColliderHandle inserted at the chunk's world-space offset.
// Colliders are added/removed dynamically as chunks stream in/out.

use rapier3d::prelude::*;
use rapier3d::control::{DynamicRayCastVehicleController, WheelTuning};
use rapier3d::na::UnitQuaternion as NaQuaternion;
use cgmath::{Matrix4, Vector3, Vector4};
use std::collections::HashMap;
use crate::terrain::{TerrainManager, CHUNK_VERTS, CHUNK_WORLD};

// ── Constants ────────────────────────────────────────────────────────────────

pub const WHEEL_RADIUS:    f32 = 0.38;
pub const WHEEL_WIDTH:     f32 = 0.28;
pub const SUSPENSION_REST: f32 = 0.50;
pub const CHASSIS_HALF_H:  f32 = 0.30;

const TRACK:     f32 = 0.92;
const WHEELBASE: f32 = 1.40;

const WHEEL_OFFSETS: [(f32, f32); 4] = [
    ( TRACK, -WHEELBASE),  // front-left
    (-TRACK, -WHEELBASE),  // front-right
    ( TRACK,  WHEELBASE),  // rear-left
    (-TRACK,  WHEELBASE),  // rear-right
];

// ── PhysicsWorld ─────────────────────────────────────────────────────────────

pub struct PhysicsWorld {
    pub rigid_body_set:     RigidBodySet,
    pub collider_set:       ColliderSet,
    pub gravity:            Vector<Real>,
    pub integration_params: IntegrationParameters,
    pub physics_pipeline:   PhysicsPipeline,
    pub island_manager:     IslandManager,
    pub broad_phase:        DefaultBroadPhase,
    pub narrow_phase:       NarrowPhase,
    pub impulse_joints:     ImpulseJointSet,
    pub multibody_joints:   MultibodyJointSet,
    pub ccd_solver:         CCDSolver,
    pub query_pipeline:     QueryPipeline,
    pub chassis_handle:     RigidBodyHandle,
    pub vehicle:            DynamicRayCastVehicleController,
    chunk_colliders:        HashMap<(i32, i32), ColliderHandle>,
    spawn_pt:               [f32; 3],
    spawn_angle:            f32,
}

impl PhysicsWorld {
    pub fn new(terrain: &TerrainManager) -> Self {
        let mut rbs = RigidBodySet::new();
        let mut cls = ColliderSet::new();

        // ── Per-chunk heightfield colliders ──────────────────────────────
        let mut chunk_colliders: HashMap<(i32, i32), ColliderHandle> = HashMap::new();
        for (&(cx, cz), cd) in terrain.chunks() {
            let handle = insert_chunk_collider(&mut cls, cx, cz, &cd.heights);
            chunk_colliders.insert((cx, cz), handle);
        }

        // ── Car chassis ──────────────────────────────────────────────────
        let (spawn_x, spawn_z) = terrain.spawn_point();
        let spawn_y = terrain.height_at(spawn_x, spawn_z) + 1.50;

        let (fwd_x, fwd_z) = terrain.road.direction_at(0.0);
        // R_y(θ) applied to physics -Z gives (-sinθ, 0, -cosθ).
        // For that to equal road forward (fwd_x, fwd_z):
        //   sinθ = -fwd_x,  cosθ = -fwd_z  →  θ = atan2(-fwd_x, -fwd_z).
        let spawn_angle = (-fwd_x).atan2(-fwd_z);
        let spawn_rot = NaQuaternion::from_axis_angle(
            &rapier3d::na::Unit::new_normalize(vector![0.0, 1.0, 0.0]),
            spawn_angle,
        );

        let chassis_rb = RigidBodyBuilder::dynamic()
            .position(Isometry::from_parts(
                Translation::new(spawn_x, spawn_y, spawn_z),
                spawn_rot,
            ))
            .linear_damping(0.12)  // light drag — reduces residual bounce without killing top speed
            .angular_damping(3.0)  // prevents nose-up flips
            .ccd_enabled(true)     // prevents tunnelling through chunk seams
            .build();
        let chassis_handle = rbs.insert(chassis_rb);

        let chassis_col = ColliderBuilder::cuboid(2.1, CHASSIS_HALF_H, 1.0)
            .mass(1200.0)
            .friction(0.15)
            .restitution(0.05)
            .build();
        cls.insert_with_parent(chassis_col, chassis_handle, &mut rbs);

        // ── Vehicle controller ───────────────────────────────────────────
        let mut vehicle = DynamicRayCastVehicleController::new(chassis_handle);

        let tuning = WheelTuning {
            suspension_stiffness:    35.0, // moderately stiff spring
            suspension_compression:   0.83, // default
            suspension_damping:       4.0,  // overdamped vs default 0.88 — suppresses bounce
            max_suspension_travel:    0.45, // keep < SUSPENSION_REST so wheels never lose ground contact
            side_friction_stiffness:  1.0,  // standard lateral grip
            friction_slip:           10.8,  // standard longitudinal grip
            max_suspension_force:  6000.0,  // default
        };

        for &(wx, wz) in &WHEEL_OFFSETS {
            vehicle.add_wheel(
                point![wx, -CHASSIS_HALF_H, wz],
                vector![0.0, -1.0, 0.0],
                vector![1.0,  0.0, 0.0],
                SUSPENSION_REST,
                WHEEL_RADIUS,
                &tuning,
            );
        }

        let mut world = Self {
            rigid_body_set: rbs,
            collider_set:   cls,
            gravity: vector![0.0, -9.81, 0.0],
            integration_params: IntegrationParameters::default(),
            physics_pipeline:   PhysicsPipeline::new(),
            island_manager:     IslandManager::new(),
            broad_phase:        DefaultBroadPhase::new(),
            narrow_phase:       NarrowPhase::new(),
            impulse_joints:     ImpulseJointSet::new(),
            multibody_joints:   MultibodyJointSet::new(),
            ccd_solver:         CCDSolver::new(),
            query_pipeline:     QueryPipeline::new(),
            chassis_handle,
            vehicle,
            chunk_colliders,
            spawn_pt: [spawn_x, spawn_y, spawn_z],
            spawn_angle,
        };

        // Warm up — settle car onto terrain (120 frames × 4 substeps = 8 sim-seconds).
        for _ in 0..120 { world.step(1.0 / 60.0); }
        world
    }

    // ── Dynamic chunk collider management ────────────────────────────────────

    pub fn add_chunk_collider(&mut self, cx: i32, cz: i32, heights: &[f32]) {
        let handle = insert_chunk_collider(&mut self.collider_set, cx, cz, heights);
        self.chunk_colliders.insert((cx, cz), handle);
    }

    pub fn remove_chunk_collider(&mut self, cx: i32, cz: i32) {
        if let Some(handle) = self.chunk_colliders.remove(&(cx, cz)) {
            self.collider_set.remove(
                handle, &mut self.island_manager,
                &mut self.rigid_body_set, false,
            );
        }
    }

    // ── Controls ─────────────────────────────────────────────────────────────

    pub fn apply_controls(&mut self, throttle: f32, steering: f32, brake: f32) {
        let speed = self.rigid_body_set[self.chassis_handle].linvel().magnitude();
        let max_speed   = 50.0_f32; // ~180 km/h cap
        let speed_norm  = (speed / max_speed).min(1.0);
        let torque_f    = (1.0 - speed_norm * speed_norm).max(0.0);
        // AWD: 2400 N per wheel (all 4), balanced front/rear
        let per_wheel   = throttle * 2400.0 * torque_f;

        let wheels = self.vehicle.wheels_mut();
        wheels[0].steering    = steering;
        wheels[1].steering    = steering;
        // All four wheels drive
        for w in wheels.iter_mut() {
            w.engine_force = per_wheel;
            w.brake        = brake * 160.0;
        }
    }

    pub fn step(&mut self, dt: f32) {
        // 4 substeps per frame: each substep has a 4× smaller dt.
        // Smaller dt makes the spring-damper integration numerically stable —
        // the suspension no longer oscillates due to Euler integration error.
        const SUBSTEPS: u32 = 4;
        let sub_dt = (dt / SUBSTEPS as f32).clamp(1.0 / 960.0, 1.0 / 80.0);
        self.integration_params.dt = sub_dt;

        for _ in 0..SUBSTEPS {
            self.vehicle.update_vehicle(
                sub_dt,
                &mut self.rigid_body_set,
                &self.collider_set,
                &self.query_pipeline,
                QueryFilter::default().exclude_rigid_body(self.chassis_handle),
            );
            self.physics_pipeline.step(
                &self.gravity,
                &self.integration_params,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.rigid_body_set,
                &mut self.collider_set,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.ccd_solver,
                Some(&mut self.query_pipeline),
                &(),
                &(),
            );
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    pub fn car_position(&self) -> Vector3<f32> {
        let p = self.rigid_body_set[self.chassis_handle].translation();
        Vector3::new(p.x, p.y, p.z)
    }

    pub fn car_forward(&self) -> Vector3<f32> {
        let rot = self.rigid_body_set[self.chassis_handle].rotation();
        let f   = rot * vector![0.0, 0.0, -1.0];
        Vector3::new(f.x, f.y, f.z)
    }

    pub fn car_matrix(&self) -> Matrix4<f32> {
        let base = isometry_to_mat4(self.rigid_body_set[self.chassis_handle].position());
        let fix = Matrix4::from_cols(
            Vector4::new( 0.0, 0.0, -1.0, 0.0),
            Vector4::new( 0.0, 1.0,  0.0, 0.0),
            Vector4::new( 1.0, 0.0,  0.0, 0.0),
            Vector4::new( 0.0, 0.0,  0.0, 1.0),
        );
        base * fix
    }

    pub fn wheel_matrix(&self, i: usize) -> Matrix4<f32> {
        let chassis_iso = self.rigid_body_set[self.chassis_handle].position();
        let wheel = &self.vehicle.wheels()[i];
        let conn_ws = chassis_iso * point![
            WHEEL_OFFSETS[i].0, -CHASSIS_HALF_H, WHEEL_OFFSETS[i].1
        ];
        let dir_ws  = chassis_iso.rotation * vector![0.0, -1.0, 0.0];
        // Use 40 % of rest_length for the visual offset: the soft rapier suspension
        // compresses significantly when settled, so using the full rest_length
        // renders wheels well below the car body.
        let center  = conn_ws + dir_ws * (wheel.suspension_rest_length * 0.40);
        let steer = NaQuaternion::from_axis_angle(
            &rapier3d::na::Unit::new_normalize(vector![0.0, 1.0, 0.0]),
            wheel.steering,
        );
        let spin = NaQuaternion::from_axis_angle(
            &rapier3d::na::Unit::new_normalize(vector![1.0, 0.0, 0.0]),
            wheel.rotation,
        );
        let world_rot = chassis_iso.rotation * steer * spin;
        isometry_to_mat4(&Isometry::from_parts(
            Translation::new(center.x, center.y, center.z),
            world_rot,
        ))
    }

    /// Returns [[wx, wz, vel_x, vel_z]; 4] for the 4 wheels.
    pub fn wheel_world_data(&self) -> Vec<[f32; 4]> {
        let vel = self.rigid_body_set[self.chassis_handle].linvel();
        (0..4).map(|i| {
            let wm = self.wheel_matrix(i);
            // cgmath Matrix4: m[col][row]. Column 3 = translation.
            let wx = wm[3][0]; // translation X
            let wz = wm[3][2]; // translation Z
            [wx, wz, vel.x, vel.z]
        }).collect()
    }

    // ── Resets ────────────────────────────────────────────────────────────────

    /// Teleport car back to spawn point.
    pub fn reset_car(&mut self) {
        let [sx, sy, sz] = self.spawn_pt;
        let rot = NaQuaternion::from_axis_angle(
            &rapier3d::na::Unit::new_normalize(vector![0.0, 1.0, 0.0]),
            self.spawn_angle,
        );
        let rb = &mut self.rigid_body_set[self.chassis_handle];
        rb.set_translation(vector![sx, sy, sz], true);
        rb.set_rotation(rot, true);
        rb.set_linvel(vector![0.0, 0.0, 0.0], true);
        rb.set_angvel(vector![0.0, 0.0, 0.0], true);
    }

    /// Upright the car at its current XZ position (flip recovery, keeps location).
    pub fn upright_in_place(&mut self, terrain: &TerrainManager) {
        let pos   = self.car_position();
        let fwd   = self.car_forward();
        let yaw   = (-fwd.x).atan2(-fwd.z); // preserve heading direction (same formula as spawn)
        let rot   = NaQuaternion::from_axis_angle(
            &rapier3d::na::Unit::new_normalize(vector![0.0, 1.0, 0.0]),
            yaw,
        );
        let ground_y = terrain.height_at(pos.x, pos.z) + 1.50;
        let rb = &mut self.rigid_body_set[self.chassis_handle];
        rb.set_translation(vector![pos.x, ground_y, pos.z], true);
        rb.set_rotation(rot, true);
        rb.set_linvel(vector![0.0, 0.0, 0.0], true);
        rb.set_angvel(vector![0.0, 0.0, 0.0], true);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn insert_chunk_collider(
    cls: &mut ColliderSet,
    cx: i32, cz: i32,
    heights: &[f32],
) -> ColliderHandle {
    let matrix = rapier3d::na::DMatrix::from_fn(CHUNK_VERTS, CHUNK_VERTS, |r, c| {
        heights[r * CHUNK_VERTS + c]
    });
    // Heightfield is centred at its translation; chunk occupies
    // [cx*CW, (cx+1)*CW] × [cz*CW, (cz+1)*CW], so centre = (cx+0.5)*CW.
    let ox = cx as f32 * CHUNK_WORLD + CHUNK_WORLD * 0.5;
    let oz = cz as f32 * CHUNK_WORLD + CHUNK_WORLD * 0.5;
    let col = ColliderBuilder::heightfield_with_flags(
        matrix,
        vector![CHUNK_WORLD, 1.0, CHUNK_WORLD],
        HeightFieldFlags::FIX_INTERNAL_EDGES,
    )
    .translation(vector![ox, 0.0, oz])
    .friction(0.85)
    .build();
    cls.insert(col)
}

fn isometry_to_mat4(iso: &Isometry<Real>) -> Matrix4<f32> {
    let m = iso.to_homogeneous();
    Matrix4::from_cols(
        Vector4::new(m[0],  m[1],  m[2],  m[3] ),
        Vector4::new(m[4],  m[5],  m[6],  m[7] ),
        Vector4::new(m[8],  m[9],  m[10], m[11]),
        Vector4::new(m[12], m[13], m[14], m[15]),
    )
}
