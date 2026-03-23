//! Rust-native cellular metabolism surface for coarse biological tissues.
//!
//! This mirrors the Python `oneuro.molecular.metabolism.CellularMetabolism`
//! interface closely enough that higher-level plant and terrarium code can
//! swap to a native implementation without rewriting the biology loop first.

use crate::constants::{clamp, michaelis_menten};

const GLYCOLYSIS_VMAX_PER_MS: f32 = 0.5 / 1000.0;
const GLYCOLYSIS_KM: f32 = 0.05;
const OXPHOS_VMAX_PER_MS: f32 = 0.1 / 1000.0;
const OXPHOS_PYRUVATE_KM: f32 = 0.05;
const OXPHOS_O2_KM: f32 = 0.001;
const LDH_VMAX_PER_MS: f32 = 0.3 / 1000.0;
const LDH_KM: f32 = 1.0;
const ATP_MIN_FUNCTIONAL: f32 = 0.5;
const O2_HYPOXIC: f32 = 0.01;
const PROTEIN_SYNTH_PER_MS: f32 = 0.0001;
const ATP_CEILING: f32 = 8.0;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellularMetabolismSim {
    glucose: f32,
    pyruvate: f32,
    lactate: f32,
    oxygen: f32,
    atp: f32,
    adp: f32,
    amp: f32,
    nad_plus: f32,
    nadh: f32,
    total_atp_produced: f32,
    total_atp_consumed: f32,
}

impl CellularMetabolismSim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        glucose: f32,
        pyruvate: f32,
        lactate: f32,
        oxygen: f32,
        atp: f32,
        adp: f32,
        amp: f32,
        nad_plus: f32,
        nadh: f32,
    ) -> Self {
        Self {
            glucose: glucose.max(0.0),
            pyruvate: pyruvate.max(0.0),
            lactate: lactate.max(0.0),
            oxygen: oxygen.max(0.0),
            atp: atp.max(0.0),
            adp: adp.max(0.0),
            amp: amp.max(0.0),
            nad_plus: nad_plus.max(0.0),
            nadh: nadh.max(0.0),
            total_atp_produced: 0.0,
            total_atp_consumed: 0.0,
        }
    }

    pub fn glucose(&self) -> f32 {
        self.glucose
    }

    pub fn pyruvate(&self) -> f32 {
        self.pyruvate
    }

    pub fn lactate(&self) -> f32 {
        self.lactate
    }

    pub fn oxygen(&self) -> f32 {
        self.oxygen
    }

    pub fn atp(&self) -> f32 {
        self.atp
    }

    pub fn adp(&self) -> f32 {
        self.adp
    }

    pub fn amp(&self) -> f32 {
        self.amp
    }

    pub fn nad_plus(&self) -> f32 {
        self.nad_plus
    }

    pub fn nadh(&self) -> f32 {
        self.nadh
    }

    pub fn energy_ratio(&self) -> f32 {
        let total = self.atp + self.adp + self.amp;
        if total <= 0.0 {
            0.0
        } else {
            self.atp / total
        }
    }

    pub fn atp_available(&self) -> bool {
        self.atp > ATP_MIN_FUNCTIONAL
    }

    pub fn is_hypoxic(&self) -> bool {
        self.oxygen < O2_HYPOXIC
    }

    pub fn supply_glucose(&mut self, amount: f32) {
        self.glucose = (self.glucose + amount.max(0.0)).max(0.0);
    }

    pub fn supply_oxygen(&mut self, amount: f32) {
        self.oxygen = (self.oxygen + amount.max(0.0)).max(0.0);
    }

    pub fn supply_lactate(&mut self, amount: f32) {
        self.lactate = (self.lactate + amount.max(0.0)).max(0.0);
    }

    pub fn consume_atp(&mut self, amount_mm: f32) -> bool {
        if amount_mm <= 0.0 {
            return true;
        }
        if self.atp < amount_mm {
            return false;
        }
        self.atp -= amount_mm;
        self.adp += amount_mm * 0.95;
        self.amp += amount_mm * 0.05;
        self.total_atp_consumed += amount_mm;
        true
    }

    pub fn protein_synthesis_cost(&mut self, dt: f32, gene_expression_rate: f32) -> bool {
        let cost = PROTEIN_SYNTH_PER_MS * gene_expression_rate.max(0.0) * dt.max(0.0);
        self.consume_atp(cost)
    }

    fn adenylate_kinase(&mut self, dt: f32) {
        if self.amp <= 0.0 || self.atp <= 0.0 {
            return;
        }
        let forward_flux = (0.5 * self.amp * self.atp * dt).min(self.amp).min(self.atp);
        if forward_flux > 0.0 {
            self.amp -= forward_flux;
            self.atp -= forward_flux;
            self.adp += 2.0 * forward_flux;
        }
    }

    fn lactate_to_pyruvate(&mut self, dt: f32) {
        let rate = michaelis_menten(self.lactate, LDH_VMAX_PER_MS, LDH_KM);
        let nad_factor = if self.nad_plus > 0.0 {
            self.nad_plus / (self.nad_plus + 0.05)
        } else {
            0.0
        };
        let lactate_consumed = (rate * nad_factor * dt).min(self.lactate);
        if lactate_consumed <= 0.0 {
            return;
        }
        self.lactate -= lactate_consumed;
        self.pyruvate += lactate_consumed;
        let nad_converted = self.nad_plus.min(lactate_consumed);
        self.nad_plus -= nad_converted;
        self.nadh += nad_converted;
    }

    fn glycolysis(&mut self, dt: f32) {
        let glucose_rate = michaelis_menten(self.glucose, GLYCOLYSIS_VMAX_PER_MS, GLYCOLYSIS_KM);
        let nad_factor = if self.nad_plus > 0.0 {
            self.nad_plus / (self.nad_plus + 0.05)
        } else {
            0.0
        };
        let glucose_consumed = (glucose_rate * nad_factor * dt).min(self.glucose);
        if glucose_consumed <= 0.0 {
            return;
        }
        self.glucose -= glucose_consumed;
        self.pyruvate += glucose_consumed * 2.0;

        let atp_produced = glucose_consumed * 2.0;
        self.atp += atp_produced;
        let adp_used = self.adp.min(atp_produced);
        self.adp -= adp_used;

        let nadh_produced = glucose_consumed * 2.0;
        let nad_converted = self.nad_plus.min(nadh_produced);
        self.nad_plus -= nad_converted;
        self.nadh += nad_converted;

        self.total_atp_produced += atp_produced;
    }

    fn oxidative_phosphorylation(&mut self, dt: f32) {
        let pyruvate_rate = michaelis_menten(self.pyruvate, OXPHOS_VMAX_PER_MS, OXPHOS_PYRUVATE_KM);
        let o2_factor = if self.oxygen > 0.0 {
            self.oxygen / (OXPHOS_O2_KM + self.oxygen)
        } else {
            0.0
        };
        let nadh_factor = if self.nadh > 0.0 {
            self.nadh / (self.nadh + 0.01)
        } else {
            0.0
        };
        let adp_factor = self.adp / (self.adp + 0.1);
        let pyruvate_consumed =
            (pyruvate_rate * o2_factor * nadh_factor * adp_factor * dt).min(self.pyruvate);
        if pyruvate_consumed <= 0.0 {
            return;
        }

        let o2_consumed = (pyruvate_consumed * 2.5).min(self.oxygen);
        self.pyruvate -= pyruvate_consumed;
        self.oxygen -= o2_consumed;

        let atp_produced = pyruvate_consumed * 34.0;
        self.atp += atp_produced;
        let adp_used = self.adp.min(atp_produced);
        self.adp -= adp_used;

        let nadh_oxidized = (pyruvate_consumed * 10.0).min(self.nadh);
        self.nadh -= nadh_oxidized;
        self.nad_plus += nadh_oxidized;
        self.total_atp_produced += atp_produced;
    }

    fn perfuse(&mut self, dt: f32) {
        self.glucose += (5.0 - self.glucose).max(0.0) * 0.015 * dt;
        self.oxygen += (0.05 - self.oxygen).max(0.0) * 0.020 * dt;
        self.lactate += (1.0 - self.lactate).max(0.0) * 0.010 * dt;
    }

    fn clamp_state(&mut self) {
        self.glucose = clamp(self.glucose, 0.0, 12.0);
        self.pyruvate = clamp(self.pyruvate, 0.0, 6.0);
        self.lactate = clamp(self.lactate, 0.0, 8.0);
        self.oxygen = clamp(self.oxygen, 0.0, 1.0);
        self.atp = clamp(self.atp, 0.0, ATP_CEILING);
        self.adp = clamp(self.adp, 0.0, 8.0);
        self.amp = clamp(self.amp, 0.0, 4.0);
        self.nad_plus = clamp(self.nad_plus, 0.0, 6.0);
        self.nadh = clamp(self.nadh, 0.0, 6.0);
    }

    pub fn step(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        self.adenylate_kinase(dt);
        self.lactate_to_pyruvate(dt);
        self.glycolysis(dt);
        self.oxidative_phosphorylation(dt);
        self.perfuse(dt);
        self.clamp_state();
    }
}

impl Default for CellularMetabolismSim {
    fn default() -> Self {
        Self::new(5.0, 0.1, 1.0, 0.05, 3.0, 0.3, 0.05, 0.5, 0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::CellularMetabolismSim;

    #[test]
    fn metabolism_remains_bounded() {
        let mut sim = CellularMetabolismSim::default();
        for _ in 0..400 {
            let _ = sim.protein_synthesis_cost(1.0, 0.8);
            sim.step(1.0);
        }
        assert!(sim.atp() >= 0.0);
        assert!(sim.glucose() >= 0.0);
        assert!(sim.oxygen() >= 0.0);
        assert!((0.0..=1.0).contains(&sim.energy_ratio()));
    }

    #[test]
    fn supply_increases_substrates() {
        let mut sim = CellularMetabolismSim::default();
        let before = sim.glucose();
        sim.supply_glucose(0.5);
        assert!(sim.glucose() > before);
    }
}
