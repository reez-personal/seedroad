// main.rs — native desktop entry point only.
// env_logger is not available on wasm32, so gate the call.

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    seedroad::platform::run();
}
