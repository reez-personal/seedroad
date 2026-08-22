pub mod input;
pub mod physics;
pub mod terrain;
pub mod platform;
pub mod renderer;
pub mod scene;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    platform::init_wasm();
    platform::run();
}
