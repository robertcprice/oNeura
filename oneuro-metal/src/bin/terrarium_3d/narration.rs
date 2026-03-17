//! Contextual narration overlay for education mode.
//!
//! Displays context-sensitive explanatory text based on the current
//! scenario, zoom level, simulation events, and selected entity.

use oneuro_metal::TerrariumWorldSnapshot;
use super::camera::ZoomLevel;
use super::hud::{draw_rect, draw_text};
use super::color::rgb;
use super::mesh::EntityTag;
use super::scenarios::Scenario;
use super::{VIEWPORT_W, TOTAL_W, TOTAL_H};

/// Narration state — manages what text to display and fade timing.
pub struct Narration {
    current_text: String,
    fade_timer: u32,
    last_event: Option<NarrationEvent>,
}

/// Events that trigger narration.
#[derive(Clone, Copy, PartialEq)]
enum NarrationEvent {
    PopulationCrash,
    PopulationBoom,
    DroughtOnset,
    NightFall,
    Dawn,
    PlantDeath,
    Extinction,
}

impl Narration {
    pub fn new() -> Self {
        Self {
            current_text: String::new(),
            fade_timer: 0,
        last_event: None,
        }
    }

    /// Update narration based on current simulation state.
    /// Priority: scenario narration > event detection > zoom context > idle.
    pub fn update(
        &mut self,
        scenario: Scenario,
        frame: usize,
        zoom: &ZoomLevel,
        snap: &TerrariumWorldSnapshot,
        prev_snap: &Option<TerrariumWorldSnapshot>,
        selected: EntityTag,
    ) {
        // 1. Scenario-scripted narration has highest priority
        if let Some(text) = scenario.narration(frame) {
            self.set(text);
            return;
        }

        // 2. Event detection (population crash, drought, night, etc.)
        if let Some(prev) = prev_snap {
            // Population crash: >30% drop in a single frame
            if prev.plants > 3 && snap.plants < prev.plants * 7 / 10 {
                self.set_event(NarrationEvent::PopulationCrash,
                    "Population crash detected! Plants are dying faster than they reproduce.");
                return;
            }
            if prev.flies > 3 && snap.flies < prev.flies * 6 / 10 {
                self.set_event(NarrationEvent::PopulationCrash,
                    "Fly population crashing! Food scarcity is causing mass starvation.");
                return;
            }
            // Boom: >50% increase
            if snap.flies > prev.flies + 3 && snap.flies > prev.flies * 3 / 2 {
                self.set_event(NarrationEvent::PopulationBoom,
                    "Population boom! Abundant resources are fueling rapid fly reproduction.");
                return;
            }
            // Drought: moisture dropping significantly
            if snap.mean_soil_moisture < 0.1 && prev.mean_soil_moisture > 0.15 {
                self.set_event(NarrationEvent::DroughtOnset,
                    "Drought conditions developing. Soil moisture critically low.");
                return;
            }
            // Nightfall
            if prev.light > 0.3 && snap.light < 0.2 {
                self.set_event(NarrationEvent::NightFall,
                    "Night falls. Photosynthesis stops. Organisms switch to respiration-only metabolism.");
                return;
            }
            // Dawn
            if prev.light < 0.2 && snap.light > 0.3 {
                self.set_event(NarrationEvent::Dawn,
                    "Dawn breaks. Photosynthesis resumes. Plants begin fixing CO2 accumulated overnight.");
                return;
            }
            // Extinction
            if snap.plants == 0 && prev.plants > 0 {
                self.set_event(NarrationEvent::Extinction,
                    "All plants have died! The ecosystem has collapsed. No primary producers remain.");
                return;
            }
        }

        // 3. Zoom-level context (only if no active narration)
        if self.fade_timer == 0 {
            let text = match zoom {
                ZoomLevel::Ecosystem => match selected {
                    EntityTag::None => "Ecosystem view: terrain, water, plants, and flies. Click to select an entity.",
                    _ => "Entity selected. Zoom in (+ key) for organism-level detail.",
                },
                ZoomLevel::Organism => match selected {
                    EntityTag::Plant(_) => "Organism view: plant physiology. Vitality bars show tissue health.",
                    EntityTag::Fly(_) => "Organism view: fly body state. Energy, speed, and flight status.",
                    EntityTag::Water(_) => "Organism view: water body volume and dissolved species.",
                    _ => "Zoom in more to see cellular-level detail.",
                },
                ZoomLevel::Cellular => "Cellular view: metabolite pools (ATP, glucose, starch). Each bar = one cell cluster.",
                ZoomLevel::Molecular => "Molecular view: all 14 TerrariumSpecies as density bars. This is the chemical substrate.",
            };
            self.set(text);
        }

        // Tick fade timer
        if self.fade_timer > 0 { self.fade_timer -= 1; }
    }

    fn set(&mut self, text: &str) {
        self.current_text = text.to_string();
        self.fade_timer = 180; // ~3 seconds at 60fps, ~6 at 30fps
    }

    fn set_event(&mut self, event: NarrationEvent, text: &str) {
        if self.last_event == Some(event) && self.fade_timer > 60 {
            return; // Don't repeat the same event while still displaying
        }
        self.last_event = Some(event);
        self.current_text = text.to_string();
        self.fade_timer = 240; // Events display longer
    }

    /// Render the narration overlay at the bottom of the viewport.
    pub fn draw(&self, buffer: &mut [u32]) {
        if self.current_text.is_empty() || self.fade_timer == 0 { return; }

        let alpha = if self.fade_timer < 30 {
            self.fade_timer as f32 / 30.0
        } else {
            1.0
        };

        // Split text into lines that fit the viewport width
        let max_chars = (VIEWPORT_W - 32) / 8;
        let lines = word_wrap(&self.current_text, max_chars);
        let line_count = lines.len();
        let box_h = line_count * 12 + 16;
        let box_y = TOTAL_H - box_h - 8;
        let box_x = 8;
        let box_w = VIEWPORT_W - 16;

        // Semi-transparent background
        let bg = if alpha > 0.5 { rgb(10, 14, 20) } else { rgb(8, 10, 14) };
        draw_rect(buffer, TOTAL_W, TOTAL_H, box_x, box_y, box_w, box_h, bg);
        // Top border
        draw_rect(buffer, TOTAL_W, TOTAL_H, box_x, box_y, box_w, 1, rgb(60, 100, 160));

        // Text
        let text_color = if alpha > 0.5 {
            rgb(200, 220, 240)
        } else {
            rgb(120, 140, 160)
        };
        for (i, line) in lines.iter().enumerate() {
            draw_text(buffer, TOTAL_W, TOTAL_H, box_x + 12, box_y + 8 + i * 12, line, text_color);
        }
    }
}

/// Simple word-wrap: split text into lines of at most max_chars.
fn word_wrap(text: &str, max_chars: usize) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let len = text.len();

    while start < len {
        let end = (start + max_chars).min(len);
        if end >= len {
            lines.push(&text[start..]);
            break;
        }
        // Find last space before end
        let mut split = end;
        while split > start && bytes[split] != b' ' {
            split -= 1;
        }
        if split == start { split = end; } // No space found, hard break
        lines.push(&text[start..split]);
        start = split;
        // Skip the space
        if start < len && bytes[start] == b' ' { start += 1; }
    }
    lines
}
