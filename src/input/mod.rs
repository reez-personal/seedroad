// input/mod.rs — tracks which keys are held each frame.

use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{Key, NamedKey};

#[derive(Default, Clone)]
pub struct InputState {
    pub forward:    bool,  // W / ArrowUp
    pub backward:   bool,  // S / ArrowDown
    pub left:       bool,  // A / ArrowLeft
    pub right:      bool,  // D / ArrowRight
    pub brake:      bool,  // Space
    pub upright:    bool,  // R — upright car at current position (flip recovery)
    pub look_left:  bool,  // Q — swing camera to left side of car
    pub look_right: bool,  // E — swing camera to right side of car
}

impl InputState {
    pub fn handle_key_event(&mut self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;
        match &event.logical_key {
            Key::Character(s) => match s.as_str() {
                "w" | "W" => self.forward     = pressed,
                "s" | "S" => self.backward    = pressed,
                "a" | "A" => self.left        = pressed,
                "d" | "D" => self.right       = pressed,
                "r" | "R" => self.upright     = pressed,
                "q" | "Q" => self.look_left   = pressed,
                "e" | "E" => self.look_right  = pressed,
                _ => {}
            },
            Key::Named(NamedKey::ArrowUp)    => self.forward  = pressed,
            Key::Named(NamedKey::ArrowDown)  => self.backward = pressed,
            Key::Named(NamedKey::ArrowLeft)  => self.left     = pressed,
            Key::Named(NamedKey::ArrowRight) => self.right    = pressed,
            Key::Named(NamedKey::Space)      => self.brake    = pressed,
            _ => {}
        }
    }
}
