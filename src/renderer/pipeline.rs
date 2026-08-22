// renderer/pipeline.rs — render pipeline and bind group layout.
//
// Bind group 0, layout:
//   binding 0 — CameraUniform   (static uniform, no dynamic offset)
//   binding 1 — TransformUniform (dynamic offset — one per draw call)

use crate::renderer::buffer::Vertex;
use crate::scene::transform::UNIFORM_ALIGN;

pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            // Camera — written once per frame, no dynamic offset.
            wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            },
            // Per-object transform — dynamic offset selects the right slice.
            wgpu::BindGroupLayoutEntry {
                binding:    1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size:   wgpu::BufferSize::new(UNIFORM_ALIGN as u64),
                },
                count: None,
            },
        ],
    })
}

pub fn create_bind_group(
    device:         &wgpu::Device,
    layout:         &wgpu::BindGroupLayout,
    camera_buf:     &wgpu::Buffer,
    transform_buf:  &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
            // Binding 1 uses dynamic offset; expose the whole buffer, wgpu
            // will slice it at draw time.
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: transform_buf,
                    offset: 0,
                    size:   wgpu::BufferSize::new(UNIFORM_ALIGN as u64),
                }),
            },
        ],
    })
}

/// Transparent alpha-blend pipeline for water (no depth write, no back-face cull).
pub fn create_water_pipeline(
    device:         &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    bgl:            &wgpu::BindGroupLayout,
    sample_count:   u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("water_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/basic.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("water_pl"),
        bind_group_layouts:   &[bgl],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("water_rp"),
        layout: Some(&layout),

        vertex: wgpu::VertexState {
            module:              &shader,
            entry_point:         "vs_main",
            buffers:             &[crate::renderer::buffer::Vertex::buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },

        fragment: Some(wgpu::FragmentState {
            module:      &shader,
            entry_point: "fs_main",
            targets:     &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend:  Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation:  wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),

        primitive: wgpu::PrimitiveState {
            topology:   wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode:  None, // render both sides of water surface
            ..Default::default()
        },

        depth_stencil: Some(wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: false, // don't occlude terrain below water
            depth_compare:       wgpu::CompareFunction::Less,
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default(),
        }),

        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview:   None,
        cache:       None,
    })
}

/// Sky dome pipeline: renders inside a cube (cull Front), no depth write, Always depth compare.
pub fn create_sky_pipeline(
    device:         &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    bgl:            &wgpu::BindGroupLayout,
    sample_count:   u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("sky_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/basic.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky_pl"), bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky_rp"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: "vs_main",
            buffers: &[crate::renderer::buffer::Vertex::buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format, blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Front), // render inside of the sky cube
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always, // sky is always behind geometry
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None, cache: None,
    })
}

/// Grass bind group layout: only the camera uniform (no wind params, no interaction texture).
pub fn create_grass_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("grass_bgl"),
        entries: &[
            // 0: CameraUniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn create_grass_bind_group(
    device:     &wgpu::Device,
    layout:     &wgpu::BindGroupLayout,
    camera_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grass_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
        ],
    })
}

pub fn create_grass_pipeline(
    device:         &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    bgl:            &wgpu::BindGroupLayout,
    sample_count:   u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grass_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/grass.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("grass_pl"), bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    // Blade vertex buffer layout (16 bytes/vert): lpos: vec3 + height_t: f32
    let blade_layout = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { shader_location: 0, offset: 0,  format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { shader_location: 1, offset: 12, format: wgpu::VertexFormat::Float32   },
        ],
    };
    // Instance buffer layout (32 bytes/instance)
    let inst_layout = wgpu::VertexBufferLayout {
        array_stride: 32,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute { shader_location: 2, offset: 0,  format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { shader_location: 3, offset: 12, format: wgpu::VertexFormat::Float32   },
            wgpu::VertexAttribute { shader_location: 4, offset: 16, format: wgpu::VertexFormat::Float32   },
            wgpu::VertexAttribute { shader_location: 5, offset: 20, format: wgpu::VertexFormat::Float32   },
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("grass_rp"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: "vs_grass",
            buffers: &[blade_layout, inst_layout],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: "fs_grass",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None, // two-sided grass
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None, cache: None,
    })
}

// ── Splash particle pipeline ──────────────────────────────────────────────────

/// Bind group layout for splash: only camera uniform (same layout as grass).
pub fn create_splash_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("splash_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

pub fn create_splash_bind_group(
    device:     &wgpu::Device,
    layout:     &wgpu::BindGroupLayout,
    camera_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("splash_bg"),
        layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
    })
}

pub fn create_splash_pipeline(
    device:         &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    bgl:            &wgpu::BindGroupLayout,
    sample_count:   u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("splash_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/splash.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("splash_pl"), bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    // Per-vertex: local_uv vec2 (8 bytes)
    let quad_layout = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            shader_location: 0, offset: 0, format: wgpu::VertexFormat::Float32x2,
        }],
    };
    // Per-instance: world_pos(12) + size(4) + alpha(4) + _pad(12) = 32 bytes
    let inst_layout = wgpu::VertexBufferLayout {
        array_stride: 32,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute { shader_location: 1, offset: 0,  format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { shader_location: 2, offset: 12, format: wgpu::VertexFormat::Float32   },
            wgpu::VertexAttribute { shader_location: 3, offset: 16, format: wgpu::VertexFormat::Float32   },
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("splash_rp"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: "vs_splash",
            buffers: &[quad_layout, inst_layout],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: "fs_splash",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation:  wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None, cache: None,
    })
}

// ── Droplet post-process pipeline ─────────────────────────────────────────────

pub fn create_droplet_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("droplet_bgl"),
        entries: &[
            // 0: scene texture
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            // 1: sampler
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // 2: DropletParams uniform
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn create_droplet_bind_group(
    device:       &wgpu::Device,
    layout:       &wgpu::BindGroupLayout,
    scene_view:   &wgpu::TextureView,
    sampler:      &wgpu::Sampler,
    params_buf:   &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("droplet_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
    })
}

pub fn create_droplet_pipeline(
    device:         &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    bgl:            &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("droplet_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/droplets.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("droplet_pl"), bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("droplet_rp"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: "vs_droplets",
            buffers: &[],   // fullscreen triangle from vertex_index, no VB
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: "fs_droplets",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend:  None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,   // post-process: no depth test
        multisample: wgpu::MultisampleState::default(),
        multiview: None, cache: None,
    })
}

pub fn create_render_pipeline(
    device:         &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    bgl:            &wgpu::BindGroupLayout,
    sample_count:   u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/basic.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("pl"),
        bind_group_layouts:   &[bgl],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("rp"),
        layout: Some(&layout),

        vertex: wgpu::VertexState {
            module:              &shader,
            entry_point:         "vs_main",
            buffers:             &[Vertex::buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },

        fragment: Some(wgpu::FragmentState {
            module:      &shader,
            entry_point: "fs_main",
            targets:     &[Some(wgpu::ColorTargetState {
                format:     surface_format,
                blend:      None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),

        primitive: wgpu::PrimitiveState {
            topology:           wgpu::PrimitiveTopology::TriangleList,
            front_face:         wgpu::FrontFace::Ccw,
            cull_mode:          Some(wgpu::Face::Back),
            ..Default::default()
        },

        depth_stencil: Some(wgpu::DepthStencilState {
            format:               wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled:  true,
            depth_compare:        wgpu::CompareFunction::Less,
            stencil:              wgpu::StencilState::default(),
            bias:                 wgpu::DepthBiasState::default(),
        }),

        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview:   None,
        cache:       None,
    })
}
