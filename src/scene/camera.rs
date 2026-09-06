// scene/camera.rs — smooth follow camera with orbital look-left/right.

use cgmath::{
    perspective, Deg, EuclideanSpace, InnerSpace, Matrix4, Point3, Vector3,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj:  [[f32; 4]; 4],
    pub eye_pos:    [f32; 3],
    pub time:       f32,
    pub car_pos:    [f32; 3],
    pub _pad2:      f32,
    /// World-space camera right vector (for billboard particles).
    pub cam_right:  [f32; 3],
    pub _pad3:      f32,
    /// World-space camera up vector (for billboard particles).
    pub cam_up:     [f32; 3],
    pub _pad4:      f32,
}

pub struct Camera {
    pub eye:        Vector3<f32>,
    pub target:     Vector3<f32>,
    aspect:         f32,
    pub time:       f32,
    pub car_pos:    [f32; 3],
    /// Orbital yaw around the car: 0 = behind, +π/2 = left side, −π/2 = right side.
    pub yaw_offset: f32,
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            eye:        Vector3::new(0.0, 4.0, 10.0),
            target:     Vector3::new(0.0, 0.5, 0.0),
            aspect:     width as f32 / height.max(1) as f32,
            time:       0.0,
            car_pos:    [0.0; 3],
            yaw_offset: 0.0,
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        self.aspect = w as f32 / h.max(1) as f32;
    }

    /// Smoothly move camera to follow `car_pos` along `car_fwd`.
    /// `yaw_offset` orbits the eye horizontally around the car (radians).
    pub fn follow(&mut self, car_pos: Vector3<f32>, car_fwd: Vector3<f32>, dt: f32) {
        let back  = -car_fwd.normalize();
        // Right vector in world XZ plane (no Y component to avoid roll)
        let right = Vector3::new(-back.z, 0.0, back.x).normalize();

        // Rotate the behind-offset by yaw_offset around the car's Y axis
        let yc = self.yaw_offset.cos();
        let ys = self.yaw_offset.sin();
        let orbit = back * yc - right * ys;

        let desired_eye    = car_pos + orbit * 9.0 + Vector3::new(0.0, 3.5, 0.0);
        let desired_target = car_pos + Vector3::new(0.0, 0.8, 0.0);

        // XZ tracks quickly so the camera swings around corners well.
        // Y tracks much more slowly — high-frequency vertical bounce (suspension
        // oscillation) is filtered out and never transferred to the view.
        let t_xz     = (1.0 - (-10.0_f32 * dt).exp()).min(1.0);
        let t_eye_y  = (1.0 - ( -3.0_f32 * dt).exp()).min(1.0);
        let t_look_y = (1.0 - ( -3.0_f32 * dt).exp()).min(1.0);

        self.eye.x  += (desired_eye.x - self.eye.x) * t_xz;
        self.eye.z  += (desired_eye.z - self.eye.z) * t_xz;
        self.eye.y  += (desired_eye.y - self.eye.y) * t_eye_y;

        self.target.x += (desired_target.x - self.target.x) * t_xz;
        self.target.z += (desired_target.z - self.target.z) * t_xz;
        self.target.y += (desired_target.y - self.target.y) * t_look_y;
    }

    pub fn build_uniform(&self) -> CameraUniform {
        let view = Matrix4::look_at_rh(
            Point3::from_vec(self.eye),
            Point3::from_vec(self.target),
            Vector3::unit_y(),
        );
        let proj = perspective(Deg(60.0), self.aspect, 0.1, 500.0);

        // Remap Z from cgmath's OpenGL [-1,1] to wgpu's Vulkan-style [0,1].
        #[rustfmt::skip]
        let correction = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );
        let view_proj = correction * proj * view;

        // Extract camera right/up from view matrix rows (view is col-major: m[col][row]).
        let cam_right = [view[0][0], view[1][0], view[2][0]];
        let cam_up    = [view[0][1], view[1][1], view[2][1]];

        CameraUniform {
            view_proj:  view_proj.into(),
            eye_pos:    self.eye.into(),
            time:       self.time,
            car_pos:    self.car_pos,
            _pad2:      0.0,
            cam_right,
            _pad3:      0.0,
            cam_up,
            _pad4:      0.0,
        }
    }
}
