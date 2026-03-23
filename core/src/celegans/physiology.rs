#[derive(Debug, Clone)]
pub(super) struct CelegansPhysiologyState {
    pub energy_reserve: f32,
    pub gut_content: f32,
    pub local_food_density: f32,
    pub pharyngeal_pumping_hz: f32,
}

impl Default for CelegansPhysiologyState {
    fn default() -> Self {
        Self {
            energy_reserve: 0.7,
            gut_content: 0.2,
            local_food_density: 0.0,
            pharyngeal_pumping_hz: 0.0,
        }
    }
}

impl CelegansPhysiologyState {
    pub fn hunger_drive(&self) -> f32 {
        (1.0 - self.energy_reserve).clamp(0.0, 1.0)
    }

    pub fn locomotor_drive_scale(&self) -> f32 {
        (0.3 + 0.7 * self.energy_reserve).clamp(0.2, 1.0)
    }

    pub fn update(&mut self, dt_ms: f32, pharyngeal_drive: f32, mean_contraction: f32) {
        let dt_s = dt_ms / 1000.0;
        let target_pump_hz =
            (self.local_food_density * (1.0 + pharyngeal_drive * 3.0) * 4.5).clamp(0.0, 5.0);
        let pump_alpha = (dt_s * 8.0).clamp(0.0, 1.0);
        self.pharyngeal_pumping_hz =
            self.pharyngeal_pumping_hz * (1.0 - pump_alpha) + target_pump_hz * pump_alpha;

        let ingestion = self.local_food_density * (self.pharyngeal_pumping_hz / 5.0) * 0.6 * dt_s;
        self.gut_content = (self.gut_content + ingestion).clamp(0.0, 1.0);

        let assimilation = self.gut_content * 0.18 * dt_s;
        self.gut_content = (self.gut_content - assimilation * 0.7).clamp(0.0, 1.0);

        let basal_cost = 0.012 * dt_s;
        let locomotor_cost = mean_contraction * 0.04 * dt_s;
        self.energy_reserve =
            (self.energy_reserve + assimilation - basal_cost - locomotor_cost).clamp(0.0, 1.0);
    }
}
