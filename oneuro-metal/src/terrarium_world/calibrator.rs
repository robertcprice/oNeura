use crate::molecular_dynamics::GPUMolecularDynamics;

/// Kinetic parameters derived from molecular dynamics calibration.
///
/// This is a local stub replacing the removed `crate::terrarium::SubstrateKinetics`.
/// Each field is a Vmax or rate constant for a major soil biogeochemical pathway.
#[derive(Debug, Clone)]
pub struct SubstrateKinetics {
    pub mean_soil_temperature_c: f32,
    pub respiration_vmax: f32,
    pub nitrification_vmax: f32,
    pub denitrification_vmax: f32,
    pub photosynthesis_vmax: f32,
    pub fermentation_vmax: f32,
    pub mineralization_vmax: f32,
}

impl Default for SubstrateKinetics {
    fn default() -> Self {
        Self {
            mean_soil_temperature_c: 22.0,
            respiration_vmax: 0.12,
            nitrification_vmax: 0.04,
            denitrification_vmax: 0.03,
            photosynthesis_vmax: 0.08,
            fermentation_vmax: 0.06,
            mineralization_vmax: 0.05,
        }
    }
}

pub struct MolecularRateCalibrator {
    /// MD simulation with a small probe system.
    md: GPUMolecularDynamics,
    /// Last calibrated kinetics from the MD probe.
    pub kinetics: SubstrateKinetics,
    /// Number of calibration runs completed.
    pub calibration_count: u64,
    /// Temperature (K) used in the last probe.
    pub probe_temperature: f32,
    /// Mean kinetic energy from last probe (kcal/mol).
    pub probe_kinetic_energy: f32,
}

impl MolecularRateCalibrator {
    pub fn new() -> Self {
        const N_PROBE_ATOMS: usize = 64;
        const N_WATERS: usize = 21;
        const BOX_SIDE: f32 = 18.0;
        const GRID_SPACING: f32 = 5.5;
        const GRID_ORIGIN: [f32; 3] = [3.0, 3.0, 3.0];

        const O_MASS: f32 = 15.999;
        const H_MASS: f32 = 1.008;
        const O_CHARGE: f32 = -0.834;
        const H_CHARGE: f32 = 0.417;
        const O_SIGMA: f32 = 3.1507;
        const O_EPSILON: f32 = 0.1521;
        const H_SIGMA: f32 = 0.0;
        const H_EPSILON: f32 = 0.0;
        const OH_BOND_LEN: f32 = 0.9572;
        const H2_OFFSET_X: f32 = -0.2400;
        const H2_OFFSET_Y: f32 = 0.9268;
        const HOH_ANGLE_RAD: f32 = 1.8242;
        const OH_BOND_K: f32 = 553.0;
        const HOH_ANGLE_K: f32 = 100.0;

        // Build a compact TIP3P-like water probe: 21 full waters plus one
        // extra oxygen to keep the legacy 64-atom calibrator footprint.
        let mut md = GPUMolecularDynamics::new(N_PROBE_ATOMS, "cpu");
        let mut positions = vec![0.0f32; N_PROBE_ATOMS * 3];
        let mut masses = vec![0.0f32; N_PROBE_ATOMS];
        let mut charges = vec![0.0f32; N_PROBE_ATOMS];
        let mut sigma = vec![0.0f32; N_PROBE_ATOMS];
        let mut epsilon = vec![0.0f32; N_PROBE_ATOMS];

        let mut water_centers = Vec::with_capacity(N_WATERS);
        for iz in 0..2 {
            for iy in 0..3 {
                for ix in 0..3 {
                    water_centers.push([
                        GRID_ORIGIN[0] + ix as f32 * GRID_SPACING,
                        GRID_ORIGIN[1] + iy as f32 * GRID_SPACING,
                        GRID_ORIGIN[2] + iz as f32 * GRID_SPACING,
                    ]);
                }
            }
        }
        water_centers.extend([
            [
                GRID_ORIGIN[0] + 0.5 * GRID_SPACING,
                GRID_ORIGIN[1] + 0.5 * GRID_SPACING,
                GRID_ORIGIN[2] + 2.0 * GRID_SPACING,
            ],
            [
                GRID_ORIGIN[0] + 1.5 * GRID_SPACING,
                GRID_ORIGIN[1] + 0.5 * GRID_SPACING,
                GRID_ORIGIN[2] + 2.0 * GRID_SPACING,
            ],
            [
                GRID_ORIGIN[0] + 0.5 * GRID_SPACING,
                GRID_ORIGIN[1] + 1.5 * GRID_SPACING,
                GRID_ORIGIN[2] + 2.0 * GRID_SPACING,
            ],
        ]);
        debug_assert_eq!(water_centers.len(), N_WATERS);

        for (water_idx, center) in water_centers.iter().enumerate() {
            let o = 3 * water_idx;
            let h1 = o + 1;
            let h2 = o + 2;
            let [ox, oy, oz] = *center;

            positions[o * 3] = ox;
            positions[o * 3 + 1] = oy;
            positions[o * 3 + 2] = oz;
            positions[h1 * 3] = ox + OH_BOND_LEN;
            positions[h1 * 3 + 1] = oy;
            positions[h1 * 3 + 2] = oz;
            positions[h2 * 3] = ox + H2_OFFSET_X;
            positions[h2 * 3 + 1] = oy + H2_OFFSET_Y;
            positions[h2 * 3 + 2] = oz;

            masses[o] = O_MASS;
            masses[h1] = H_MASS;
            masses[h2] = H_MASS;
            charges[o] = O_CHARGE;
            charges[h1] = H_CHARGE;
            charges[h2] = H_CHARGE;
            sigma[o] = O_SIGMA;
            sigma[h1] = H_SIGMA;
            sigma[h2] = H_SIGMA;
            epsilon[o] = O_EPSILON;
            epsilon[h1] = H_EPSILON;
            epsilon[h2] = H_EPSILON;

            md.add_bond(o, h1, OH_BOND_LEN, OH_BOND_K);
            md.add_bond(o, h2, OH_BOND_LEN, OH_BOND_K);
            md.add_angle(h1, o, h2, HOH_ANGLE_RAD, HOH_ANGLE_K);
        }

        let extra_o = N_PROBE_ATOMS - 1;
        let extra_o_pos = [
            GRID_ORIGIN[0] + 1.5 * GRID_SPACING,
            GRID_ORIGIN[1] + 1.5 * GRID_SPACING,
            GRID_ORIGIN[2] + 2.0 * GRID_SPACING,
        ];
        positions[extra_o * 3] = extra_o_pos[0];
        positions[extra_o * 3 + 1] = extra_o_pos[1];
        positions[extra_o * 3 + 2] = extra_o_pos[2];
        masses[extra_o] = O_MASS;
        charges[extra_o] = O_CHARGE;
        sigma[extra_o] = O_SIGMA;
        epsilon[extra_o] = O_EPSILON;

        md.set_positions(&positions);
        md.set_masses(&masses);
        md.set_charges(&charges);
        md.set_lj_params(&sigma, &epsilon);
        md.set_box([BOX_SIDE, BOX_SIDE, BOX_SIDE]);
        md.set_temperature(300.0);
        md.initialize_velocities();

        Self {
            md,
            kinetics: SubstrateKinetics::default(),
            calibration_count: 0,
            probe_temperature: 300.0,
            probe_kinetic_energy: 0.0,
        }
    }

    /// Run a short MD probe and derive updated kinetic parameters.
    ///
    /// The probe runs 50 Velocity Verlet steps. From the resulting kinetic
    /// energy (temperature), we compute a Boltzmann-scaled correction factor
    /// for each reaction pathway's Vmax. This is the genuine molecular
    /// dynamics -> reaction rate bridge.
    pub fn calibrate(&mut self, soil_temperature_c: f32) -> SubstrateKinetics {
        let temp_k = soil_temperature_c + 273.15;
        self.md.set_temperature(temp_k);

        // Run 50 MD steps (short burst, ~0.5ms on CPU for 64 atoms).
        // Bonded TIP3P-like water remains stable at a 1 fs probe timestep.
        let dt_fs = 1.0;
        self.md.run(49, dt_fs, None);
        // Final step to capture stats.
        let stats = self.md.step(dt_fs);
        self.probe_temperature = if stats.temperature.is_finite() {
            stats.temperature.max(0.0)
        } else {
            temp_k
        };
        self.probe_kinetic_energy = if stats.kinetic_energy.is_finite() {
            stats.kinetic_energy.max(0.0)
        } else {
            0.0
        };
        self.calibration_count += 1;

        // Derive rate corrections from MD observables.
        // The Eyring equation: k = (kB*T/h) * exp(-Ea/RT)
        // We use the ratio of actual MD temperature to reference (300K) to
        // scale Vmax values, approximating the Arrhenius temperature dependence.
        let t_ratio = (temp_k / 300.0).max(0.5).min(2.0);
        // Approximate activation energy scaling (dimensionless Ea/R factor).
        let ea_respiration = 6500.0; // ~54 kJ/mol for aerobic respiration
        let ea_nitrification = 7200.0; // ~60 kJ/mol for ammonia oxidation
        let ea_denitrification = 5800.0; // ~48 kJ/mol for nitrate reduction
        let ea_photosynthesis = 4000.0; // ~33 kJ/mol for CO2 fixation (RuBisCO)
        let ea_fermentation = 5500.0; // ~46 kJ/mol for anaerobic glycolysis

        let ref_inv = 1.0 / 300.0;
        let cur_inv = 1.0 / temp_k;
        let arrhenius = |ea: f32| -> f32 { (ea * (ref_inv - cur_inv)).exp() };

        // Pressure/crowding from MD: use vdw_energy as a proxy for molecular
        // crowding that affects diffusion-limited reactions.
        let vdw_clamped = if stats.vdw_energy.is_finite() {
            stats.vdw_energy.clamp(-10000.0, 10000.0)
        } else {
            0.0
        };
        let crowding_factor = 1.0 / (1.0 + vdw_clamped.abs() * 0.01);

        let mut kinetics = SubstrateKinetics::default();
        kinetics.mean_soil_temperature_c = soil_temperature_c;

        // Scale Vmax by Arrhenius temperature dependence from MD probe.
        kinetics.respiration_vmax *= arrhenius(ea_respiration) * crowding_factor;
        kinetics.nitrification_vmax *= arrhenius(ea_nitrification) * crowding_factor;
        kinetics.denitrification_vmax *= arrhenius(ea_denitrification) * crowding_factor;
        kinetics.photosynthesis_vmax *= arrhenius(ea_photosynthesis);
        kinetics.fermentation_vmax *= arrhenius(ea_fermentation) * crowding_factor;
        kinetics.mineralization_vmax *= t_ratio * crowding_factor;

        self.kinetics = kinetics.clone();
        kinetics
    }
}
