// scene/transform.rs — model matrix + normal matrix + per-object color,
// packed into a 256-byte aligned struct for wgpu dynamic uniform offsets.

use cgmath::{Matrix, Matrix4, Quaternion, Vector3, One, Zero, SquareMatrix};

/// Minimum uniform-buffer-offset alignment required by wgpu (WebGL2 spec).
pub const UNIFORM_ALIGN: usize = 256;

/// GPU-side per-object data.
/// Padded to exactly 256 bytes so it can live in a dynamic-offset uniform buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransformUniform {
    pub model:  [[f32; 4]; 4], //  64 bytes
    pub normal: [[f32; 4]; 4], //  64 bytes  (transpose-inverse of model, as mat4)
    pub color:  [f32; 4],      //  16 bytes  (RGBA, linear)
    pub _pad:   [f32; 28],     // 112 bytes  → total = 256 bytes ✓
}

impl TransformUniform {
    pub fn new(model: Matrix4<f32>, color: [f32; 4]) -> Self {
        let normal = model.invert()
            .unwrap_or(Matrix4::from_scale(1.0))
            .transpose();
        Self {
            model:  model.into(),
            normal: normal.into(),
            color,
            _pad:   [0.0; 28],
        }
    }
}

/// CPU-side transform (position + rotation + scale).
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale:    Vector3<f32>,
    pub color:    [f32; 4],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vector3::zero(),
            rotation: Quaternion::one(),
            scale:    Vector3::new(1.0, 1.0, 1.0),
            color:    [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl Transform {
    pub fn with_color(color: [f32; 4]) -> Self {
        Self { color, ..Default::default() }
    }

    pub fn matrix(&self) -> Matrix4<f32> {
        let t = Matrix4::from_translation(self.position);
        let r = Matrix4::from(self.rotation);
        let s = Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);
        t * r * s
    }

    pub fn build_uniform(&self) -> TransformUniform {
        TransformUniform::new(self.matrix(), self.color)
    }
}

/// Build a TransformUniform directly from a world-space 4×4 matrix + color.
pub fn uniform_from_matrix(m: Matrix4<f32>, color: [f32; 4]) -> TransformUniform {
    TransformUniform::new(m, color)
}

/// Scale a matrix by (sx, sy, sz) without affecting translation.
pub fn scale_matrix(m: Matrix4<f32>, sx: f32, sy: f32, sz: f32) -> Matrix4<f32> {
    m * Matrix4::from_nonuniform_scale(sx, sy, sz)
}
