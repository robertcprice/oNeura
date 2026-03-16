//! Coarse plant cellular state dynamics for terrarium ecology.
//!
//! This mirrors the current Python-side `PlantCellularState` evolution path:
//! four tissue clusters (leaf, stem, root, meristem), each with a reduced
//! subset of whole-cell state plus a native metabolism module.

use crate::cellular_metabolism::CellularMetabolismSim;
use crate::constants::clamp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlantTissue {
    Leaf,
    Stem,
    Root,
    Meristem,
}

impl PlantTissue {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "leaf" => Some(Self::Leaf),
            "stem" => Some(Self::Stem),
            "root" => Some(Self::Root),
            "meristem" => Some(Self::Meristem),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlantClusterSnapshot {
    pub cell_count: f32,
    pub vitality: f32,
    pub division_buffer: f32,
    pub state_atp: f32,
    pub state_adp: f32,
    pub state_glucose: f32,
    pub state_starch: f32,
    pub state_water: f32,
    pub state_nitrate: f32,
    pub state_amino_acid: f32,
    pub state_auxin: f32,
    pub cytoplasm_water: f32,
    pub cytoplasm_sucrose: f32,
    pub cytoplasm_nitrate: f32,
    pub apoplast_water: f32,
    pub apoplast_nitrate: f32,
    pub transcript_stress_response: f32,
    pub transcript_cell_cycle: f32,
    pub transcript_transport_program: f32,
    pub chem_glucose: f32,
    pub chem_pyruvate: f32,
    pub chem_lactate: f32,
    pub chem_oxygen: f32,
    pub chem_atp: f32,
    pub chem_adp: f32,
    pub chem_amp: f32,
    pub chem_nad_plus: f32,
    pub chem_nadh: f32,
}

#[derive(Debug, Clone)]
struct PlantClusterSim {
    #[allow(dead_code)]
    tissue: PlantTissue,
    cell_count: f32,
    vitality: f32,
    division_buffer: f32,
    state_atp: f32,
    state_adp: f32,
    state_glucose: f32,
    state_starch: f32,
    state_water: f32,
    state_nitrate: f32,
    state_amino_acid: f32,
    state_auxin: f32,
    cytoplasm_water: f32,
    cytoplasm_sucrose: f32,
    cytoplasm_nitrate: f32,
    apoplast_water: f32,
    apoplast_nitrate: f32,
    transcript_stress_response: f32,
    transcript_cell_cycle: f32,
    transcript_transport_program: f32,
    chemistry: CellularMetabolismSim,
}

impl PlantClusterSim {
    fn new(tissue: PlantTissue, cell_count: f32) -> Self {
        let scale = (cell_count / 100.0).max(0.4);
        match tissue {
            PlantTissue::Leaf => Self {
                tissue,
                cell_count,
                vitality: 1.0,
                division_buffer: 0.0,
                state_atp: 2.6,
                state_adp: 1.0,
                state_glucose: 1.7,
                state_starch: 1.3,
                state_water: 4.8,
                state_nitrate: 0.35,
                state_amino_acid: 0.6,
                state_auxin: 0.06,
                cytoplasm_water: 4.8 * scale,
                cytoplasm_sucrose: 1.7 * 0.8 * scale,
                cytoplasm_nitrate: 0.35 * scale,
                apoplast_water: 4.8 * 0.7 * scale,
                apoplast_nitrate: 0.35 * 0.5 * scale,
                transcript_stress_response: 0.2,
                transcript_cell_cycle: 0.1,
                transcript_transport_program: 0.12,
                chemistry: CellularMetabolismSim::new(
                    5.8, 0.14, 0.8, 0.07, 3.2, 0.35, 0.05, 0.55, 0.08,
                ),
            },
            PlantTissue::Stem => Self {
                tissue,
                cell_count,
                vitality: 1.0,
                division_buffer: 0.0,
                state_atp: 1.8,
                state_adp: 0.9,
                state_glucose: 1.1,
                state_starch: 0.6,
                state_water: 3.8,
                state_nitrate: 0.25,
                state_amino_acid: 0.5,
                state_auxin: 0.10,
                cytoplasm_water: 3.8 * scale,
                cytoplasm_sucrose: 1.1 * 0.8 * scale,
                cytoplasm_nitrate: 0.25 * scale,
                apoplast_water: 3.8 * 0.7 * scale,
                apoplast_nitrate: 0.25 * 0.5 * scale,
                transcript_stress_response: 0.2,
                transcript_cell_cycle: 0.1,
                transcript_transport_program: 0.3,
                chemistry: CellularMetabolismSim::new(
                    4.6, 0.12, 0.9, 0.05, 2.6, 0.40, 0.06, 0.50, 0.07,
                ),
            },
            PlantTissue::Root => Self {
                tissue,
                cell_count,
                vitality: 1.0,
                division_buffer: 0.0,
                state_atp: 1.7,
                state_adp: 1.0,
                state_glucose: 0.9,
                state_starch: 0.3,
                state_water: 4.4,
                state_nitrate: 1.2,
                state_amino_acid: 0.7,
                state_auxin: 0.08,
                cytoplasm_water: 4.4 * scale,
                cytoplasm_sucrose: 0.9 * 0.8 * scale,
                cytoplasm_nitrate: 1.2 * scale,
                apoplast_water: 4.4 * 0.7 * scale,
                apoplast_nitrate: 1.2 * 0.5 * scale,
                transcript_stress_response: 0.2,
                transcript_cell_cycle: 0.1,
                transcript_transport_program: 0.3,
                chemistry: CellularMetabolismSim::new(
                    4.2, 0.10, 1.2, 0.04, 2.4, 0.45, 0.07, 0.55, 0.06,
                ),
            },
            PlantTissue::Meristem => Self {
                tissue,
                cell_count,
                vitality: 1.0,
                division_buffer: 0.0,
                state_atp: 2.0,
                state_adp: 0.9,
                state_glucose: 1.2,
                state_starch: 0.4,
                state_water: 3.5,
                state_nitrate: 0.5,
                state_amino_acid: 0.8,
                state_auxin: 0.16,
                cytoplasm_water: 3.5 * scale,
                cytoplasm_sucrose: 1.2 * 0.8 * scale,
                cytoplasm_nitrate: 0.5 * scale,
                apoplast_water: 3.5 * 0.7 * scale,
                apoplast_nitrate: 0.5 * 0.5 * scale,
                transcript_stress_response: 0.2,
                transcript_cell_cycle: 0.3,
                transcript_transport_program: 0.12,
                chemistry: CellularMetabolismSim::new(
                    4.8, 0.10, 1.0, 0.05, 2.7, 0.38, 0.05, 0.52, 0.07,
                ),
            },
        }
    }

    fn energy_charge(&self) -> f32 {
        let amp = self.state_amp();
        let total = self.state_atp + self.state_adp + amp;
        let nucleotide_ratio = if total <= 1e-9 {
            0.0
        } else {
            self.state_atp / total
        };
        clamp(
            nucleotide_ratio * 0.62 + self.chemistry.energy_ratio() * 0.38,
            0.0,
            1.0,
        )
    }

    fn state_amp(&self) -> f32 {
        self.chemistry.amp()
    }

    fn sugar_pool(&self) -> f32 {
        (self.state_glucose + self.state_starch + self.cytoplasm_sucrose * 0.4).max(0.0)
    }

    fn water_pool(&self) -> f32 {
        (self.state_water + self.cytoplasm_water * 0.08).max(0.0)
    }

    fn nitrogen_pool(&self) -> f32 {
        (self.state_nitrate + self.state_amino_acid).max(0.0)
    }

    fn clamp_state(&mut self) {
        self.cell_count = self.cell_count.max(0.0);
        self.vitality = clamp(self.vitality, 0.0, 1.0);
        self.division_buffer = clamp(self.division_buffer, 0.0, 1.0);
        self.state_atp = self.state_atp.max(0.0);
        self.state_adp = self.state_adp.max(0.0);
        self.state_glucose = self.state_glucose.max(0.0);
        self.state_starch = self.state_starch.max(0.0);
        self.state_water = self.state_water.max(0.0);
        self.state_nitrate = self.state_nitrate.max(0.0);
        self.state_amino_acid = self.state_amino_acid.max(0.0);
        self.state_auxin = self.state_auxin.max(0.0);
        self.cytoplasm_water = self.cytoplasm_water.max(0.0);
        self.cytoplasm_sucrose = self.cytoplasm_sucrose.max(0.0);
        self.cytoplasm_nitrate = self.cytoplasm_nitrate.max(0.0);
        self.apoplast_water = self.apoplast_water.max(0.0);
        self.apoplast_nitrate = self.apoplast_nitrate.max(0.0);
        self.transcript_stress_response = self.transcript_stress_response.max(0.0);
        self.transcript_cell_cycle = self.transcript_cell_cycle.max(0.0);
        self.transcript_transport_program = self.transcript_transport_program.max(0.0);
    }

    fn update_vitality(&mut self, stress_signal: f32) {
        self.vitality = clamp(
            0.10 + self.energy_charge() * 0.28
                + (self.sugar_pool() / 2.8).min(1.0) * 0.18
                + (self.water_pool() / 5.0).min(1.0) * 0.15
                + (self.nitrogen_pool() / 2.0).min(1.0) * 0.11
                - stress_signal * 0.12,
            0.0,
            1.0,
        );
    }

    fn homeostatic_recovery(&mut self, temp_factor: f32, stress_signal: f32) {
        let atp = self.state_atp;
        let glucose = self.state_glucose;
        let starch = self.state_starch;
        let water = self.state_water;
        let amino = self.state_amino_acid;
        let available_carbon = glucose + starch * 0.7;
        if atp >= 0.18 || available_carbon <= 0.05 || water <= 0.35 {
            return;
        }
        let rescue = (0.08 + amino * 0.015).min(
            available_carbon
                * (0.06 + temp_factor * 0.025)
                * (1.0 - stress_signal * 0.18).max(0.30),
        );
        if rescue <= 0.0 {
            return;
        }
        let glucose_draw = glucose.min(rescue * 0.82);
        let starch_draw = starch.min((rescue - glucose_draw).max(0.0));
        self.state_glucose = (glucose - glucose_draw).max(0.0);
        self.state_starch = (starch - starch_draw).max(0.0);
        self.state_atp = atp + rescue;
        self.state_adp = (self.state_adp - rescue * 0.34).max(0.0);
        self.state_water = (water - rescue * 0.04).max(0.0);
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
            .supply_glucose(glucose_flux.max(0.0) + self.state_glucose.max(0.0) * 0.02);
        self.chemistry.supply_oxygen(oxygen_flux.max(0.0));
        self.chemistry
            .supply_lactate(lactate_flux.max(0.0) + self.state_starch.max(0.0) * 0.01);

        let total_ms = (dt * 5.0).clamp(40.0, 360.0);
        let n_sub = ((total_ms / 110.0).floor() as usize + 1).clamp(1, 4);
        let chem_dt = total_ms / n_sub as f32;
        for _ in 0..n_sub {
            let _ = self
                .chemistry
                .consume_atp(transport_load.max(0.0) * chem_dt * 0.00020);
            let _ = self
                .chemistry
                .protein_synthesis_cost(chem_dt, protein_load.max(0.0) * 0.55);
            self.chemistry.step(chem_dt);
        }

        self.state_atp = self.state_atp * 0.84 + self.chemistry.atp() * 0.16;
        self.state_adp = self.state_adp * 0.84 + self.chemistry.adp() * 0.16;
        self.state_glucose = self.state_glucose * 0.92 + self.chemistry.glucose() * 0.08;
        self.state_water = self.state_water * 0.985 + self.chemistry.oxygen() * 1.2;
        self.state_amino_acid = self.state_amino_acid * 0.985 + self.chemistry.pyruvate() * 0.015;
    }

    fn snapshot(&self) -> PlantClusterSnapshot {
        PlantClusterSnapshot {
            cell_count: self.cell_count,
            vitality: self.vitality,
            division_buffer: self.division_buffer,
            state_atp: self.state_atp,
            state_adp: self.state_adp,
            state_glucose: self.state_glucose,
            state_starch: self.state_starch,
            state_water: self.state_water,
            state_nitrate: self.state_nitrate,
            state_amino_acid: self.state_amino_acid,
            state_auxin: self.state_auxin,
            cytoplasm_water: self.cytoplasm_water,
            cytoplasm_sucrose: self.cytoplasm_sucrose,
            cytoplasm_nitrate: self.cytoplasm_nitrate,
            apoplast_water: self.apoplast_water,
            apoplast_nitrate: self.apoplast_nitrate,
            transcript_stress_response: self.transcript_stress_response,
            transcript_cell_cycle: self.transcript_cell_cycle,
            transcript_transport_program: self.transcript_transport_program,
            chem_glucose: self.chemistry.glucose(),
            chem_pyruvate: self.chemistry.pyruvate(),
            chem_lactate: self.chemistry.lactate(),
            chem_oxygen: self.chemistry.oxygen(),
            chem_atp: self.chemistry.atp(),
            chem_adp: self.chemistry.adp(),
            chem_amp: self.chemistry.amp(),
            chem_nad_plus: self.chemistry.nad_plus(),
            chem_nadh: self.chemistry.nadh(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlantCellularFeedback {
    pub photosynthetic_capacity: f32,
    pub maintenance_cost: f32,
    pub storage_exchange: f32,
    pub division_growth: f32,
    pub senescence_mass: f32,
    pub energy_charge: f32,
    pub vitality: f32,
    pub sugar_pool: f32,
    pub water_pool: f32,
    pub nitrogen_pool: f32,
    pub division_signal: f32,
    pub new_cells: f32,
}

#[derive(Debug, Clone)]
pub struct PlantCellularStateSim {
    leaf: PlantClusterSim,
    stem: PlantClusterSim,
    root: PlantClusterSim,
    meristem: PlantClusterSim,
    division_events: f32,
    last_new_cells: f32,
    last_senescence: f32,
}

impl PlantCellularStateSim {
    pub fn new(leaf_cells: f32, stem_cells: f32, root_cells: f32, meristem_cells: f32) -> Self {
        Self {
            leaf: PlantClusterSim::new(PlantTissue::Leaf, leaf_cells),
            stem: PlantClusterSim::new(PlantTissue::Stem, stem_cells),
            root: PlantClusterSim::new(PlantTissue::Root, root_cells),
            meristem: PlantClusterSim::new(PlantTissue::Meristem, meristem_cells),
            division_events: 0.0,
            last_new_cells: 0.0,
            last_senescence: 0.0,
        }
    }

    pub fn cluster_snapshot(&self, tissue: PlantTissue) -> PlantClusterSnapshot {
        match tissue {
            PlantTissue::Leaf => self.leaf.snapshot(),
            PlantTissue::Stem => self.stem.snapshot(),
            PlantTissue::Root => self.root.snapshot(),
            PlantTissue::Meristem => self.meristem.snapshot(),
        }
    }

    pub fn division_events(&self) -> f32 {
        self.division_events
    }

    pub fn last_new_cells(&self) -> f32 {
        self.last_new_cells
    }

    pub fn last_senescence(&self) -> f32 {
        self.last_senescence
    }

    fn weighted_mean(pairs: &[(f32, f32)]) -> f32 {
        let total_weight: f32 = pairs.iter().map(|(w, _)| *w).sum();
        if total_weight <= 1e-9 {
            0.0
        } else {
            pairs.iter().map(|(w, v)| *w * *v).sum::<f32>() / total_weight
        }
    }

    pub fn total_cells(&self) -> f32 {
        self.leaf.cell_count
            + self.stem.cell_count
            + self.root.cell_count
            + self.meristem.cell_count
    }

    pub fn vitality(&self) -> f32 {
        Self::weighted_mean(&[
            (self.leaf.cell_count, self.leaf.vitality),
            (self.stem.cell_count, self.stem.vitality),
            (self.root.cell_count, self.root.vitality),
            (self.meristem.cell_count, self.meristem.vitality),
        ])
    }

    pub fn energy_charge(&self) -> f32 {
        Self::weighted_mean(&[
            (self.leaf.cell_count, self.leaf.energy_charge()),
            (self.stem.cell_count, self.stem.energy_charge()),
            (self.root.cell_count, self.root.energy_charge()),
            (self.meristem.cell_count, self.meristem.energy_charge()),
        ])
    }

    pub fn sugar_pool(&self) -> f32 {
        self.leaf.sugar_pool()
            + self.stem.sugar_pool()
            + self.root.sugar_pool()
            + self.meristem.sugar_pool()
    }

    pub fn water_pool(&self) -> f32 {
        self.leaf.water_pool()
            + self.stem.water_pool()
            + self.root.water_pool()
            + self.meristem.water_pool()
    }

    pub fn nitrogen_pool(&self) -> f32 {
        self.leaf.nitrogen_pool()
            + self.stem.nitrogen_pool()
            + self.root.nitrogen_pool()
            + self.meristem.nitrogen_pool()
    }

    pub fn division_signal(&self) -> f32 {
        self.meristem.division_buffer
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        dt: f32,
        local_light: f32,
        temp_factor: f32,
        water_in: f32,
        nutrient_in: f32,
        water_status: f32,
        nutrient_status: f32,
        symbiosis: f32,
        stress_signal: f32,
        storage_signal: f32,
    ) -> PlantCellularFeedback {
        let stress = stress_signal.max(0.0);
        let leaf_resp =
            self.leaf.cell_count * (0.0000032 + 0.0000026 * temp_factor + stress * 0.0000010) * dt;
        let stem_resp =
            self.stem.cell_count * (0.0000024 + 0.0000018 * temp_factor + stress * 0.0000008) * dt;
        let root_resp =
            self.root.cell_count * (0.0000028 + 0.0000020 * temp_factor + stress * 0.0000010) * dt;
        let meristem_resp = self.meristem.cell_count
            * (0.0000036 + 0.0000024 * temp_factor + stress * 0.0000012)
            * dt;

        let photosynthate = local_light
            * temp_factor
            * water_status.min(1.0)
            * nutrient_status.min(1.0)
            * self.leaf.cell_count
            * 0.000010
            * dt;
        let starch_release = self
            .leaf
            .state_starch
            .min((stress - 0.30).max(0.0) * self.leaf.cell_count * 0.0000018 * dt);
        let sugar_export = (photosynthate + starch_release).max(0.0) * 0.46;

        let root_water_capture = water_in * (0.70 + symbiosis * 3.2);
        let root_nitrogen_capture = nutrient_in * (0.68 + symbiosis * 1.3);
        let xylem_flow = root_water_capture * 0.60;
        let stem_to_leaf_water = xylem_flow * 0.64;
        let stem_to_meristem_water = xylem_flow * 0.12;
        let transpiration = local_light
            * (0.00026 + self.leaf.cell_count * 0.00000014)
            * dt
            * (1.0 + stress * 0.35);

        let nitrate_assim = root_nitrogen_capture * (0.50 + symbiosis * 0.90);
        let amino_flux = nitrate_assim * 0.62;
        let stem_to_root_sugar = sugar_export * 0.20;
        let stem_to_meristem_sugar = sugar_export * 0.28;
        let stem_buffer_sugar = sugar_export - stem_to_root_sugar - stem_to_meristem_sugar;
        let leaf_catabolism = (self.leaf.state_glucose * 0.0011 * dt * temp_factor).min(0.45);
        let stem_catabolism =
            ((self.stem.state_glucose + stem_buffer_sugar) * 0.0015 * dt * temp_factor).min(0.42);
        let root_catabolism =
            ((self.root.state_glucose + stem_to_root_sugar) * 0.0018 * dt * temp_factor).min(0.40);
        let meristem_catabolism =
            ((self.meristem.state_glucose + stem_to_meristem_sugar) * 0.0021 * dt * temp_factor)
                .min(0.46);

        let meristem_auxin_prev = self.meristem.state_auxin;

        self.leaf.cytoplasm_water += stem_to_leaf_water * 0.65 - transpiration * 0.35;
        self.leaf.cytoplasm_sucrose += photosynthate * 0.32 - sugar_export * 0.18;
        self.leaf.apoplast_water += stem_to_leaf_water * 0.35 - transpiration * 0.65;
        self.leaf.state_atp += photosynthate * 0.42 + leaf_catabolism * 0.84 - leaf_resp * 0.55;
        self.leaf.state_adp += leaf_resp * 0.22 - photosynthate * 0.14 - leaf_catabolism * 0.44;
        self.leaf.state_glucose += photosynthate * 0.68 + starch_release * 0.45
            - sugar_export * 0.58
            - leaf_resp * 0.10
            - leaf_catabolism;
        self.leaf.state_starch += photosynthate * 0.20 - starch_release;
        self.leaf.state_water += stem_to_leaf_water + water_in * 0.14 - transpiration;
        self.leaf.state_nitrate += nutrient_in * 0.08 - nitrate_assim * 0.18;
        self.leaf.state_amino_acid += amino_flux * 0.12 - leaf_resp * 0.02;
        self.leaf.state_auxin += local_light * 0.0004 * dt - stress * 0.0003 * dt;
        self.leaf.transcript_stress_response += stress * 0.003 * dt - photosynthate * 0.0008;

        self.stem.cytoplasm_water += xylem_flow - stem_to_leaf_water - stem_to_meristem_water;
        self.stem.cytoplasm_sucrose +=
            sugar_export * 0.40 - stem_to_root_sugar - stem_to_meristem_sugar;
        self.stem.state_atp += stem_buffer_sugar * 0.18 + stem_catabolism * 0.92 - stem_resp * 0.48;
        self.stem.state_adp += stem_resp * 0.18 - stem_buffer_sugar * 0.08 - stem_catabolism * 0.50;
        self.stem.state_glucose += stem_buffer_sugar * 0.54 - stem_resp * 0.10 - stem_catabolism;
        self.stem.state_starch += stem_buffer_sugar * 0.10;
        self.stem.state_water += xylem_flow - stem_to_leaf_water - stem_to_meristem_water;
        self.stem.state_nitrate += root_nitrogen_capture * 0.14 - amino_flux * 0.10;
        self.stem.state_amino_acid += amino_flux * 0.18 - stem_resp * 0.02;
        self.stem.state_auxin += meristem_auxin_prev * 0.01;
        self.stem.transcript_transport_program += stem_buffer_sugar * 0.004 - stress * 0.0015 * dt;

        self.root.apoplast_water += root_water_capture * 0.55;
        self.root.apoplast_nitrate += root_nitrogen_capture * 0.50;
        self.root.cytoplasm_water += root_water_capture * 0.45 - xylem_flow * 0.65;
        self.root.cytoplasm_nitrate += root_nitrogen_capture * 0.52;
        self.root.state_atp +=
            root_nitrogen_capture * 0.18 + stem_to_root_sugar * 0.20 + root_catabolism * 0.96
                - root_resp * 0.56;
        self.root.state_adp +=
            root_resp * 0.20 - stem_to_root_sugar * 0.10 - root_catabolism * 0.54;
        self.root.state_glucose += stem_to_root_sugar * 0.66 - root_resp * 0.14 - root_catabolism;
        self.root.state_starch += stem_to_root_sugar * 0.08;
        self.root.state_water += root_water_capture - xylem_flow;
        self.root.state_nitrate += root_nitrogen_capture - nitrate_assim;
        self.root.state_amino_acid += amino_flux * 0.50 - root_resp * 0.04;
        self.root.state_auxin += 0.0003 * dt;
        self.root.transcript_transport_program +=
            root_nitrogen_capture * 0.008 + root_water_capture * 0.006 - stress * 0.0012 * dt;

        self.meristem.cytoplasm_water += stem_to_meristem_water;
        self.meristem.cytoplasm_sucrose += stem_to_meristem_sugar * 0.70;
        self.meristem.state_atp +=
            stem_to_meristem_sugar * 0.34 + amino_flux * 0.12 + meristem_catabolism * 1.02
                - meristem_resp * 0.58;
        self.meristem.state_adp +=
            meristem_resp * 0.20 - stem_to_meristem_sugar * 0.14 - meristem_catabolism * 0.58;
        self.meristem.state_glucose +=
            stem_to_meristem_sugar * 0.72 - meristem_resp * 0.14 - meristem_catabolism;
        self.meristem.state_starch += stem_to_meristem_sugar * 0.05;
        self.meristem.state_water += stem_to_meristem_water - meristem_resp * 0.04;
        self.meristem.state_nitrate += root_nitrogen_capture * 0.08 - amino_flux * 0.06;
        self.meristem.state_amino_acid += amino_flux * 0.20 - meristem_resp * 0.03;
        self.meristem.state_auxin += stem_to_meristem_sugar * 0.01 + 0.0006 * dt;
        self.meristem.transcript_cell_cycle +=
            stem_to_meristem_sugar * 0.010 + amino_flux * 0.005 - stress * 0.0016 * dt;

        self.leaf.step_cluster_chemistry(
            dt,
            photosynthate * 0.12,
            0.006 + local_light * 0.010,
            0.002,
            0.18 + self.leaf.transcript_stress_response,
            0.10 + self.leaf.transcript_transport_program,
        );
        self.stem.step_cluster_chemistry(
            dt,
            stem_buffer_sugar * 0.18,
            0.004 + local_light * 0.004,
            0.003,
            0.14 + self.stem.transcript_transport_program,
            0.16 + self.stem.transcript_transport_program,
        );
        self.root.step_cluster_chemistry(
            dt,
            stem_to_root_sugar * 0.20,
            0.003 + water_status * 0.004 + symbiosis * 0.010,
            0.006 + symbiosis * 0.008,
            0.12 + self.root.transcript_transport_program,
            0.18 + self.root.transcript_transport_program,
        );
        self.meristem.step_cluster_chemistry(
            dt,
            stem_to_meristem_sugar * 0.24,
            0.004 + local_light * 0.003,
            0.004,
            0.16 + self.meristem.transcript_cell_cycle,
            0.10 + self.meristem.transcript_cell_cycle,
        );

        let resource_ready = 1.0_f32
            .min((self.meristem.state_glucose + stem_to_meristem_sugar * 6.0) / 1.2)
            .min((self.meristem.state_water + stem_to_meristem_water * 18.0) / 2.2)
            .min((self.meristem.state_amino_acid + amino_flux * 0.8) / 0.7)
            .min((self.meristem.energy_charge() + meristem_catabolism * 0.4) / 0.60);
        let division_drive = resource_ready
            * temp_factor
            * (0.32 + storage_signal * 0.12)
            * (1.0 - stress.min(1.4) * 0.52);
        let new_cells = (division_drive * self.meristem.cell_count * 0.00012 * dt).max(0.0);
        self.leaf.cell_count += new_cells * 0.36;
        self.stem.cell_count += new_cells * 0.24;
        self.root.cell_count += new_cells * 0.28;
        self.meristem.cell_count += new_cells * 0.12;
        self.meristem.division_buffer = clamp(division_drive, 0.0, 1.0);
        self.division_events += new_cells;
        self.last_new_cells = new_cells;

        let senescence = (stress - 0.72).max(0.0)
            * (0.00005 * dt)
            * (self.leaf.cell_count + self.root.cell_count);
        let leaf_loss = (self.leaf.cell_count * 0.02).min(senescence * 0.62);
        let root_loss = (self.root.cell_count * 0.015).min(senescence * 0.38);
        self.leaf.cell_count = (self.leaf.cell_count - leaf_loss).max(8.0);
        self.root.cell_count = (self.root.cell_count - root_loss).max(8.0);
        self.last_senescence = leaf_loss + root_loss;

        for cluster in [
            &mut self.leaf,
            &mut self.stem,
            &mut self.root,
            &mut self.meristem,
        ] {
            cluster.clamp_state();
            cluster.homeostatic_recovery(temp_factor, stress);
            cluster.update_vitality(stress);
            cluster.clamp_state();
        }

        let total_resp = leaf_resp + stem_resp + root_resp + meristem_resp;
        let energy_charge = self.energy_charge();
        let vitality = self.vitality();
        PlantCellularFeedback {
            photosynthetic_capacity: clamp(
                0.66 + self.leaf.energy_charge() * 0.36 + vitality * 0.16,
                0.45,
                1.4,
            ),
            maintenance_cost: total_resp * 0.0045,
            storage_exchange: photosynthate * 0.0018 - total_resp * 0.0012,
            division_growth: new_cells * 0.00042,
            senescence_mass: (leaf_loss + root_loss) * 0.00026,
            energy_charge,
            vitality,
            sugar_pool: self.sugar_pool(),
            water_pool: self.water_pool(),
            nitrogen_pool: self.nitrogen_pool(),
            division_signal: self.division_signal(),
            new_cells,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PlantCellularStateSim, PlantTissue};

    #[test]
    fn plant_cellular_state_stays_bounded() {
        let mut state = PlantCellularStateSim::new(150.0, 120.0, 140.0, 70.0);
        for _ in 0..200 {
            let feedback = state.step(30.0, 0.8, 0.95, 0.04, 0.02, 0.9, 0.8, 0.1, 0.1, 0.3);
            assert!(feedback.energy_charge >= 0.0);
        }
        let leaf = state.cluster_snapshot(PlantTissue::Leaf);
        assert!(leaf.cell_count > 0.0);
        assert!((0.0..=1.0).contains(&leaf.vitality));
    }
}
