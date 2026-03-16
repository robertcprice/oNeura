//! Native seed cellular state dynamics for terrarium seed internals.
//!
//! The Rust terrarium now uses this state for seed-internal reserve handling,
//! render projection, and germination readiness instead of one-off reserve
//! scalars.

use crate::cellular_metabolism::CellularMetabolismSim;
use crate::constants::clamp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedTissue {
    Coat,
    Endosperm,
    Radicle,
    Cotyledon,
}

#[derive(Debug, Clone)]
pub struct SeedClusterSnapshot {
    pub cell_count: f32,
    pub vitality: f32,
    pub hydration: f32,
    pub energy_charge: f32,
    pub sugar_pool: f32,
    pub nitrogen_pool: f32,
    pub transcript_stress_response: f32,
    pub transcript_germination_program: f32,
    pub chem_glucose: f32,
    pub chem_oxygen: f32,
    pub chem_atp: f32,
}

#[derive(Debug, Clone)]
struct SeedClusterSim {
    cell_count: f32,
    vitality: f32,
    state_water: f32,
    state_glucose: f32,
    state_starch: f32,
    state_nitrate: f32,
    state_amino_acid: f32,
    transcript_stress_response: f32,
    transcript_germination_program: f32,
    chemistry: CellularMetabolismSim,
}

impl SeedClusterSim {
    fn new(tissue: SeedTissue, seed_mass: f32, reserve_carbon: f32, dormancy_t: f32) -> Self {
        let seed_mass_t = clamp((seed_mass - 0.03) / 0.17, 0.0, 1.0);
        let reserve_t = clamp(reserve_carbon / 0.2, 0.0, 1.0);
        let base_cells = 20.0 + seed_mass_t * 28.0 + reserve_t * 18.0;
        let germination_bias = 0.08 + (1.0 - dormancy_t) * 0.20;
        match tissue {
            SeedTissue::Coat => Self {
                cell_count: base_cells * 0.26,
                vitality: 0.78,
                state_water: 0.55 + reserve_t * 0.12,
                state_glucose: 0.16 + reserve_t * 0.04,
                state_starch: 0.12 + reserve_t * 0.04,
                state_nitrate: 0.05 + seed_mass_t * 0.02,
                state_amino_acid: 0.08 + seed_mass_t * 0.03,
                transcript_stress_response: 0.18 + dormancy_t * 0.16,
                transcript_germination_program: germination_bias * 0.25,
                chemistry: CellularMetabolismSim::new(
                    1.2, 0.08, 0.2, 0.03, 1.8, 0.5, 0.05, 0.42, 0.06,
                ),
            },
            SeedTissue::Endosperm => Self {
                cell_count: base_cells * 0.42,
                vitality: 0.84,
                state_water: 1.10 + reserve_t * 0.28,
                state_glucose: 0.72 + reserve_t * 0.56,
                state_starch: 1.60 + reserve_t * 1.40,
                state_nitrate: 0.14 + seed_mass_t * 0.05,
                state_amino_acid: 0.18 + seed_mass_t * 0.06,
                transcript_stress_response: 0.10 + dormancy_t * 0.10,
                transcript_germination_program: germination_bias * 0.32,
                chemistry: CellularMetabolismSim::new(
                    2.8, 0.12, 0.4, 0.04, 2.2, 0.65, 0.06, 0.56, 0.08,
                ),
            },
            SeedTissue::Radicle => Self {
                cell_count: base_cells * 0.14,
                vitality: 0.80,
                state_water: 0.72 + reserve_t * 0.12,
                state_glucose: 0.28 + reserve_t * 0.16,
                state_starch: 0.06 + reserve_t * 0.03,
                state_nitrate: 0.18 + seed_mass_t * 0.06,
                state_amino_acid: 0.16 + seed_mass_t * 0.05,
                transcript_stress_response: 0.14 + dormancy_t * 0.08,
                transcript_germination_program: germination_bias * 0.55,
                chemistry: CellularMetabolismSim::new(
                    2.0, 0.10, 0.6, 0.05, 2.0, 0.60, 0.07, 0.54, 0.08,
                ),
            },
            SeedTissue::Cotyledon => Self {
                cell_count: base_cells * 0.18,
                vitality: 0.82,
                state_water: 0.68 + reserve_t * 0.14,
                state_glucose: 0.34 + reserve_t * 0.20,
                state_starch: 0.28 + reserve_t * 0.18,
                state_nitrate: 0.12 + seed_mass_t * 0.04,
                state_amino_acid: 0.20 + seed_mass_t * 0.05,
                transcript_stress_response: 0.12 + dormancy_t * 0.08,
                transcript_germination_program: germination_bias * 0.44,
                chemistry: CellularMetabolismSim::new(
                    2.2, 0.10, 0.3, 0.05, 2.1, 0.58, 0.06, 0.52, 0.08,
                ),
            },
        }
    }

    fn energy_charge(&self) -> f32 {
        let sugar_term = self.sugar_pool() / (self.sugar_pool() + 0.8);
        clamp(
            self.chemistry.energy_ratio() * 0.72 + sugar_term * 0.28,
            0.0,
            1.0,
        )
    }

    fn hydration(&self) -> f32 {
        clamp(
            (self.state_water + self.chemistry.oxygen() * 0.8) / (0.6 + self.cell_count * 0.018),
            0.0,
            1.4,
        )
    }

    fn sugar_pool(&self) -> f32 {
        (self.state_glucose + self.state_starch * 0.82 + self.chemistry.glucose() * 0.18).max(0.0)
    }

    fn nitrogen_pool(&self) -> f32 {
        (self.state_nitrate + self.state_amino_acid).max(0.0)
    }

    fn clamp_state(&mut self) {
        self.cell_count = self.cell_count.max(0.0);
        self.vitality = clamp(self.vitality, 0.0, 1.0);
        self.state_water = self.state_water.max(0.0);
        self.state_glucose = self.state_glucose.max(0.0);
        self.state_starch = self.state_starch.max(0.0);
        self.state_nitrate = self.state_nitrate.max(0.0);
        self.state_amino_acid = self.state_amino_acid.max(0.0);
        self.transcript_stress_response = clamp(self.transcript_stress_response, 0.0, 1.6);
        self.transcript_germination_program = clamp(self.transcript_germination_program, 0.0, 1.6);
    }

    fn update_vitality(&mut self, stress_signal: f32) {
        self.vitality = clamp(
            0.14 + self.energy_charge() * 0.34
                + self.hydration().min(1.0) * 0.20
                + (self.nitrogen_pool() / 0.9).min(1.0) * 0.12
                - stress_signal * 0.16,
            0.0,
            1.0,
        );
    }

    fn homeostatic_recovery(&mut self, stress_signal: f32) {
        if self.chemistry.atp() >= 0.28 || self.sugar_pool() <= 0.06 || self.state_water <= 0.18 {
            return;
        }
        let rescue = (0.04 + self.state_amino_acid * 0.015)
            .min(self.sugar_pool() * (0.05 - stress_signal * 0.015).max(0.015));
        if rescue <= 0.0 {
            return;
        }
        let glucose_draw = self.state_glucose.min(rescue * 0.78);
        let starch_draw = self.state_starch.min((rescue - glucose_draw).max(0.0));
        self.state_glucose = (self.state_glucose - glucose_draw).max(0.0);
        self.state_starch = (self.state_starch - starch_draw).max(0.0);
        self.chemistry.supply_glucose(rescue * 0.55);
        self.state_water = (self.state_water - rescue * 0.03).max(0.0);
    }

    fn step_cluster_chemistry(
        &mut self,
        dt: f32,
        glucose_flux: f32,
        oxygen_flux: f32,
        lactate_flux: f32,
        protein_load: f32,
        transport_load: f32,
    ) {
        self.chemistry
            .supply_glucose(glucose_flux.max(0.0) + self.state_glucose.max(0.0) * 0.015);
        self.chemistry.supply_oxygen(oxygen_flux.max(0.0));
        self.chemistry
            .supply_lactate(lactate_flux.max(0.0) + self.state_starch.max(0.0) * 0.008);

        let total_ms = (dt * 4.0).clamp(30.0, 240.0);
        let n_sub = ((total_ms / 90.0).floor() as usize + 1).clamp(1, 4);
        let chem_dt = total_ms / n_sub as f32;
        for _ in 0..n_sub {
            let _ = self
                .chemistry
                .consume_atp(transport_load.max(0.0) * chem_dt * 0.00018);
            let _ = self
                .chemistry
                .protein_synthesis_cost(chem_dt, protein_load.max(0.0) * 0.48);
            self.chemistry.step(chem_dt);
        }

        self.state_glucose = self.state_glucose * 0.90 + self.chemistry.glucose() * 0.10;
        self.state_water = self.state_water * 0.97 + self.chemistry.oxygen() * 0.85;
        self.state_amino_acid = self.state_amino_acid * 0.985 + self.chemistry.pyruvate() * 0.015;
    }

    fn snapshot(&self) -> SeedClusterSnapshot {
        SeedClusterSnapshot {
            cell_count: self.cell_count,
            vitality: self.vitality,
            hydration: self.hydration(),
            energy_charge: self.energy_charge(),
            sugar_pool: self.sugar_pool(),
            nitrogen_pool: self.nitrogen_pool(),
            transcript_stress_response: self.transcript_stress_response,
            transcript_germination_program: self.transcript_germination_program,
            chem_glucose: self.chemistry.glucose(),
            chem_oxygen: self.chemistry.oxygen(),
            chem_atp: self.chemistry.atp(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SeedCellularFeedback {
    pub reserve_carbon: f32,
    pub vitality: f32,
    pub energy_charge: f32,
    pub hydration: f32,
    pub germination_drive: f32,
    pub germination_readiness: f32,
    pub ready_to_germinate: bool,
    pub radicle_extension: f32,
    pub cotyledon_opening: f32,
    pub coat_integrity: f32,
}

#[derive(Debug, Clone)]
pub struct SeedCellularStateSim {
    coat: SeedClusterSim,
    endosperm: SeedClusterSim,
    radicle: SeedClusterSim,
    cotyledon: SeedClusterSim,
    last_feedback: SeedCellularFeedback,
}

impl SeedCellularStateSim {
    pub fn new(seed_mass: f32, reserve_carbon: f32, dormancy_s: f32) -> Self {
        let dormancy_t = clamp(dormancy_s / 26_000.0, 0.0, 1.0);
        Self {
            coat: SeedClusterSim::new(SeedTissue::Coat, seed_mass, reserve_carbon, dormancy_t),
            endosperm: SeedClusterSim::new(
                SeedTissue::Endosperm,
                seed_mass,
                reserve_carbon,
                dormancy_t,
            ),
            radicle: SeedClusterSim::new(
                SeedTissue::Radicle,
                seed_mass,
                reserve_carbon,
                dormancy_t,
            ),
            cotyledon: SeedClusterSim::new(
                SeedTissue::Cotyledon,
                seed_mass,
                reserve_carbon,
                dormancy_t,
            ),
            last_feedback: SeedCellularFeedback::default(),
        }
    }

    pub fn cluster_snapshot(&self, tissue: SeedTissue) -> SeedClusterSnapshot {
        match tissue {
            SeedTissue::Coat => self.coat.snapshot(),
            SeedTissue::Endosperm => self.endosperm.snapshot(),
            SeedTissue::Radicle => self.radicle.snapshot(),
            SeedTissue::Cotyledon => self.cotyledon.snapshot(),
        }
    }

    fn weighted_mean(pairs: &[(f32, f32)]) -> f32 {
        let total_weight: f32 = pairs.iter().map(|(w, _)| *w).sum();
        if total_weight <= 1.0e-9 {
            0.0
        } else {
            pairs.iter().map(|(w, v)| *w * *v).sum::<f32>() / total_weight
        }
    }

    pub fn vitality(&self) -> f32 {
        Self::weighted_mean(&[
            (self.coat.cell_count, self.coat.vitality),
            (self.endosperm.cell_count, self.endosperm.vitality),
            (self.radicle.cell_count, self.radicle.vitality),
            (self.cotyledon.cell_count, self.cotyledon.vitality),
        ])
    }

    pub fn energy_charge(&self) -> f32 {
        Self::weighted_mean(&[
            (self.coat.cell_count, self.coat.energy_charge()),
            (self.endosperm.cell_count, self.endosperm.energy_charge()),
            (self.radicle.cell_count, self.radicle.energy_charge()),
            (self.cotyledon.cell_count, self.cotyledon.energy_charge()),
        ])
    }

    pub fn hydration(&self) -> f32 {
        Self::weighted_mean(&[
            (self.coat.cell_count, self.coat.hydration()),
            (self.endosperm.cell_count, self.endosperm.hydration()),
            (self.radicle.cell_count, self.radicle.hydration()),
            (self.cotyledon.cell_count, self.cotyledon.hydration()),
        ])
    }

    pub fn reserve_carbon_equivalent(&self) -> f32 {
        let reserve_mm = self.endosperm.state_starch * 1.05
            + self.endosperm.state_glucose * 0.55
            + self.cotyledon.state_starch * 0.24
            + self.cotyledon.state_glucose * 0.18;
        clamp(reserve_mm / 18.0, 0.0, 0.2)
    }

    pub fn last_feedback(&self) -> &SeedCellularFeedback {
        &self.last_feedback
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        dt: f32,
        moisture: f32,
        deep_moisture: f32,
        nutrients: f32,
        symbionts: f32,
        canopy: f32,
        litter: f32,
        dormancy_s: f32,
        reserve_carbon: f32,
    ) -> SeedCellularFeedback {
        let dormancy_t = clamp(dormancy_s / 26_000.0, 0.0, 1.0);
        let reserve_t = clamp(reserve_carbon / 0.2, 0.0, 1.0);
        let hydration_input = (moisture * 0.72 + deep_moisture * 0.28).clamp(0.0, 1.6);
        let nutrient_input = (nutrients + symbionts * 0.45).clamp(0.0, 1.8);
        let litter_t = clamp(litter / 0.25, 0.0, 1.0);
        let shelter_t = clamp(canopy / 1.4, 0.0, 1.0);
        let imbibition = hydration_input * (0.38 + (1.0 - dormancy_t) * 0.62);
        let germination_drive = ((1.0 - dormancy_t)
            * (0.20 + hydration_input * 0.80)
            * (0.28 + nutrient_input * 0.52)
            * (0.72 + litter_t * 0.28)
            * (0.86 + reserve_t * 0.14)
            * (0.94 + (1.0 - shelter_t) * 0.06))
            .clamp(0.0, 1.8);
        let stress = ((0.30 - hydration_input).max(0.0) * 1.4
            + (0.18 - nutrient_input).max(0.0) * 0.7
            + shelter_t * 0.04)
            .clamp(0.0, 1.6);
        let water_flux = imbibition * dt * 0.008;
        let nutrient_flux = nutrient_input * dt * 0.0016;
        let reserve_release = reserve_t * (0.00018 + germination_drive * 0.00050) * dt;

        self.coat.state_water += water_flux * 0.24 - self.coat.state_water * 0.0009 * dt;
        self.coat.state_glucose += reserve_release * 0.04;
        self.coat.transcript_stress_response += stress * 0.0040 * dt - imbibition * 0.0014 * dt;
        self.coat.transcript_germination_program +=
            imbibition * 0.0024 * dt - dormancy_t * 0.0020 * dt;

        self.endosperm.state_water += water_flux * 0.34;
        self.endosperm.state_glucose += reserve_release * 0.64;
        self.endosperm.state_starch = (self.endosperm.state_starch
            - reserve_release * (0.88 + germination_drive * 0.24))
            .max(0.0);
        self.endosperm.state_amino_acid += nutrient_flux * 0.08;
        self.endosperm.transcript_stress_response +=
            stress * 0.0028 * dt - hydration_input * 0.0010 * dt;
        self.endosperm.transcript_germination_program +=
            germination_drive * 0.0034 * dt - dormancy_t * 0.0014 * dt;

        self.radicle.state_water += water_flux * 0.18;
        self.radicle.state_glucose += reserve_release * 0.18 + nutrient_flux * 0.04;
        self.radicle.state_nitrate += nutrient_flux * 0.42;
        self.radicle.state_amino_acid += nutrient_flux * 0.18;
        self.radicle.transcript_stress_response +=
            stress * 0.0032 * dt - germination_drive * 0.0011 * dt;
        self.radicle.transcript_germination_program +=
            germination_drive * 0.0060 * dt - stress * 0.0016 * dt;

        self.cotyledon.state_water += water_flux * 0.16;
        self.cotyledon.state_glucose += reserve_release * 0.12 + litter_t * 0.0008 * dt;
        self.cotyledon.state_starch += reserve_release * 0.05;
        self.cotyledon.state_nitrate += nutrient_flux * 0.20;
        self.cotyledon.state_amino_acid += nutrient_flux * 0.14;
        self.cotyledon.transcript_stress_response +=
            stress * 0.0028 * dt - germination_drive * 0.0010 * dt;
        self.cotyledon.transcript_germination_program +=
            germination_drive * 0.0044 * dt - stress * 0.0012 * dt;

        self.coat.step_cluster_chemistry(
            dt,
            reserve_release * 0.02,
            0.002 + hydration_input * 0.001,
            0.001,
            0.08 + stress * 0.12,
            0.05 + imbibition * 0.04,
        );
        self.endosperm.step_cluster_chemistry(
            dt,
            reserve_release * 0.20,
            0.003 + hydration_input * 0.0014,
            0.002,
            0.12 + germination_drive * 0.12,
            0.08 + germination_drive * 0.08,
        );
        self.radicle.step_cluster_chemistry(
            dt,
            reserve_release * 0.14 + nutrient_flux * 0.02,
            0.004 + hydration_input * 0.002 + nutrient_input * 0.001,
            0.004 + symbionts * 0.003,
            0.14 + germination_drive * 0.18,
            0.10 + germination_drive * 0.12,
        );
        self.cotyledon.step_cluster_chemistry(
            dt,
            reserve_release * 0.10,
            0.003 + hydration_input * 0.0012,
            0.003,
            0.12 + germination_drive * 0.12,
            0.08 + germination_drive * 0.10,
        );

        let radicle_growth = (germination_drive
            * self.radicle.energy_charge()
            * self.radicle.cell_count
            * 0.00010
            * dt)
            .max(0.0);
        let cotyledon_growth = (germination_drive
            * self.cotyledon.energy_charge()
            * self.cotyledon.cell_count
            * 0.00006
            * dt)
            .max(0.0);
        self.radicle.cell_count += radicle_growth;
        self.cotyledon.cell_count += cotyledon_growth;
        self.endosperm.cell_count =
            (self.endosperm.cell_count - (radicle_growth + cotyledon_growth) * 0.55).max(4.0);
        self.coat.cell_count = (self.coat.cell_count - imbibition * 0.0008 * dt).max(4.0);

        for cluster in [
            &mut self.coat,
            &mut self.endosperm,
            &mut self.radicle,
            &mut self.cotyledon,
        ] {
            cluster.clamp_state();
            cluster.homeostatic_recovery(stress);
            cluster.update_vitality(stress);
            cluster.clamp_state();
        }

        let reserve_carbon = self.reserve_carbon_equivalent();
        let vitality = self.vitality();
        let energy_charge = self.energy_charge();
        let hydration = self.hydration();
        let radicle_energy = self.radicle.energy_charge();
        let cotyledon_energy = self.cotyledon.energy_charge();
        let endosperm_sugar = self.endosperm.sugar_pool().min(1.5);
        let radicle_extension = clamp(
            self.radicle.transcript_germination_program * 0.62
                + radicle_growth / (self.radicle.cell_count + 1.0),
            0.0,
            1.5,
        );
        let cotyledon_opening = clamp(
            self.cotyledon.transcript_germination_program * 0.55 + cotyledon_energy * 0.45,
            0.0,
            1.5,
        );
        let coat_integrity = clamp(
            self.coat.vitality * 0.68
                + (1.0 - self.coat.transcript_stress_response.min(1.0)) * 0.32,
            0.0,
            1.0,
        );
        let germination_readiness = clamp(
            germination_drive * 0.34
                + radicle_extension * 0.24
                + cotyledon_opening * 0.16
                + radicle_energy * 0.08
                + cotyledon_energy * 0.06
                + vitality * 0.06
                + (hydration.min(1.2) / 1.2) * 0.06
                + (endosperm_sugar / 1.5) * 0.08
                + (1.0 - coat_integrity) * 0.04,
            0.0,
            1.5,
        );
        let ready_to_germinate = dormancy_s <= 0.0
            && germination_readiness > 0.45
            && radicle_extension > 0.18
            && cotyledon_opening > 0.24
            && radicle_energy > 0.16
            && cotyledon_energy > 0.16
            && vitality > 0.22
            && hydration > 0.30
            && endosperm_sugar > 0.10;
        let feedback = SeedCellularFeedback {
            reserve_carbon,
            vitality,
            energy_charge,
            hydration,
            germination_drive: germination_drive.clamp(0.0, 1.5),
            germination_readiness,
            ready_to_germinate,
            radicle_extension,
            cotyledon_opening,
            coat_integrity,
        };
        self.last_feedback = feedback.clone();
        feedback
    }
}

#[cfg(test)]
mod tests {
    use super::{SeedCellularStateSim, SeedTissue};

    #[test]
    fn seed_cellular_state_stays_bounded() {
        let mut state = SeedCellularStateSim::new(0.11, 0.16, 14_000.0);
        for _ in 0..180 {
            let feedback = state.step(30.0, 0.7, 0.5, 0.18, 0.08, 0.3, 0.06, 9_000.0, 0.14);
            assert!((0.0..=0.2).contains(&feedback.reserve_carbon));
            assert!(feedback.hydration >= 0.0);
            assert!(feedback.energy_charge >= 0.0);
        }
        let radicle = state.cluster_snapshot(SeedTissue::Radicle);
        let endosperm = state.cluster_snapshot(SeedTissue::Endosperm);
        assert!(radicle.cell_count > 0.0);
        assert!(endosperm.sugar_pool >= 0.0);
    }
}
