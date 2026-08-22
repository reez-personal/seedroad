// renderer/buffer.rs — GPU buffer creation helpers.

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub uv:       [f32; 2],
}

impl Vertex {
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, offset: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { shader_location: 1, offset: std::mem::offset_of!(Vertex, normal) as u64, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { shader_location: 2, offset: std::mem::offset_of!(Vertex, uv)     as u64, format: wgpu::VertexFormat::Float32x2 },
            ],
        }
    }
}

pub fn create_vertex_buffer(device: &wgpu::Device, data: &[Vertex]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("vb"),
        contents: bytemuck::cast_slice(data),
        usage:    wgpu::BufferUsages::VERTEX,
    })
}

pub fn create_index_buffer(device: &wgpu::Device, data: &[u16]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("ib"),
        contents: bytemuck::cast_slice(data),
        usage:    wgpu::BufferUsages::INDEX,
    })
}
