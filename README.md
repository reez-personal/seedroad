# rust-game

A from-scratch 3D renderer in Rust targeting both native desktop and the browser via WebAssembly + WebGPU (with automatic WebGL2 fallback).

## Prerequisites

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### 2. Add the WASM target
```bash
rustup target add wasm32-unknown-unknown
```

### 3. Install Trunk (browser build tool)
```bash
cargo install trunk
```

Trunk handles: compiling to WASM, running `wasm-bindgen`, bundling assets, and serving a local dev server with live-reload.

---

## Running natively (fast iteration)

```bash
cargo run
```

Uses `env_logger`. Set `RUST_LOG=debug` for verbose wgpu output:
```bash
RUST_LOG=debug cargo run
```

---

## Running in the browser

```bash
trunk serve
```

Then open [http://localhost:8080](http://localhost:8080).

Trunk watches `src/`, `shaders/`, and `index.html` and live-reloads on changes.

For a release build (smaller, faster WASM):
```bash
trunk build --release
```
Output lands in `dist/`.

---

## Browser compatibility

| Browser | Backend |
|---------|---------|
| Chrome 113+ / Edge 113+ | WebGPU (native API) |
| Firefox (with `dom.webgpu.enabled`) | WebGPU |
| Safari 18+ | WebGPU |
| Older browsers with WebGL2 | WebGL2 fallback (wgpu auto-selects) |

wgpu picks the best available backend automatically — no code changes needed.

---

## Project layout

```
Cargo.toml          — workspace manifest, all dependencies
index.html          — browser entry point (Trunk injects WASM loader)
Trunk.toml          — trunk build configuration
shaders/
  basic.wgsl        — WGSL vertex + fragment shader (Lambertian lighting)
src/
  main.rs           — native entry point (env_logger + pollster::block_on)
  lib.rs            — library root + WASM entry point (#[wasm_bindgen(start)])
  platform/
    mod.rs          — logging init, canvas DOM injection, shared event loop
  renderer/
    mod.rs          — Renderer struct: wgpu init, resize, render loop
    pipeline.rs     — render pipeline + bind group layout construction
    buffer.rs       — Vertex type, cube mesh data, buffer helpers
  scene/
    mod.rs
    camera.rs       — orbiting perspective camera, CameraUniform
    transform.rs    — position/rotation/scale → model matrix, TransformUniform
```

---

## What's rendered

A unit cube centered at the origin, lit with a single hardcoded directional light (Lambertian diffuse + ambient). The camera orbits around the cube at 0.5 rad/s. Depth testing is enabled.

---

## Suggested next milestones

1. **Camera controller** — keyboard/mouse orbit (drag to rotate, scroll to zoom), replacing the auto-orbit.
2. **Texture support** — load a PNG with the `image` crate, upload as a `wgpu::Texture`, sample in the shader.
3. **glTF model loading** — use the `gltf` crate to load meshes + materials; replace the hardcoded cube.
4. **Shadow mapping** — depth pre-pass from the light's perspective, sample the shadow map in the main pass.
5. **PBR materials** — metallic/roughness workflow, image-based lighting (IBL), environment maps.
6. **Post-processing** — render to a texture, then apply bloom / tone-mapping / FXAA in a full-screen pass.
