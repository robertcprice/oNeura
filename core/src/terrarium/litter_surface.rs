use super::*;

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct LitterSurfaceState {
    pub cover_fraction: f32,
    pub support_depth_mm: f32,
    pub pore_exposure: f32,
    pub roughness_mm: f32,
    pub collapse_rate: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LitterSurfaceInputs {
    pub cell_size_mm: f32,
    pub litter_carbon: f32,
    pub organic_matter: f32,
    pub surface_moisture: f32,
    pub deep_moisture: f32,
    pub bioturbation_mm2_day: f32,
}

fn target_litter_surface(inputs: LitterSurfaceInputs) -> LitterSurfaceState {
    let litter_t = clamp(inputs.litter_carbon / 0.20, 0.0, 1.6);
    let organic_t = clamp(inputs.organic_matter / 0.80, 0.0, 1.4);
    let support_mass_t = clamp(litter_t * 0.64 + organic_t * 0.36, 0.0, 1.6);
    let surface_wet_t = clamp(inputs.surface_moisture / 0.70, 0.0, 1.4);
    let deep_wet_t = clamp(inputs.deep_moisture / 0.70, 0.0, 1.4);
    let dry_surface_t = clamp((0.42 - inputs.surface_moisture).max(0.0) / 0.30, 0.0, 1.6);
    let drying_gradient_t = clamp(
        (inputs.deep_moisture - inputs.surface_moisture).max(0.0) / 0.26,
        0.0,
        1.6,
    );
    let bioturb_t = clamp(inputs.bioturbation_mm2_day / 5.0, 0.0, 1.6);
    let cover_fraction = clamp(
        0.02 + litter_t * 0.44 + organic_t * 0.16 + surface_wet_t * 0.06 - dry_surface_t * 0.05,
        0.0,
        1.0,
    );
    let support_depth_mm = inputs.cell_size_mm.max(1.0e-3)
        * clamp(
            0.04 + support_mass_t * 0.40 + surface_wet_t * 0.10 + bioturb_t * 0.05
                - dry_surface_t * 0.10,
            0.02,
            0.95,
        );
    let pore_exposure = clamp(
        0.03 + (1.0 - cover_fraction) * 0.22
            + dry_surface_t * 0.20
            + drying_gradient_t * 0.20
            + bioturb_t * 0.14,
        0.0,
        1.0,
    );
    let roughness_mm = inputs.cell_size_mm.max(1.0e-3)
        * clamp(
            0.03 + bioturb_t * 0.14 + dry_surface_t * 0.10 + (1.0 - cover_fraction) * 0.08,
            0.01,
            0.70,
        );
    let collapse_rate = clamp(
        0.04 + dry_surface_t * 0.12
            + (1.0 - support_mass_t.min(1.0)) * 0.18
            + bioturb_t * 0.10
            + deep_wet_t * 0.04,
        0.0,
        1.0,
    );
    LitterSurfaceState {
        cover_fraction,
        support_depth_mm,
        pore_exposure,
        roughness_mm,
        collapse_rate,
    }
}

pub fn initialize_litter_surface(inputs: LitterSurfaceInputs) -> LitterSurfaceState {
    target_litter_surface(inputs)
}

pub fn step_litter_surface(
    state: &mut LitterSurfaceState,
    eco_dt_s: f32,
    inputs: LitterSurfaceInputs,
) {
    let target = target_litter_surface(inputs);
    let adapt = clamp(1.0 - (-eco_dt_s.max(0.0) / 900.0).exp(), 0.04, 1.0);
    state.cover_fraction = clamp(
        state.cover_fraction + (target.cover_fraction - state.cover_fraction) * adapt,
        0.0,
        1.0,
    );
    state.support_depth_mm = clamp(
        state.support_depth_mm + (target.support_depth_mm - state.support_depth_mm) * adapt,
        0.0,
        inputs.cell_size_mm.max(1.0e-3),
    );
    state.pore_exposure = clamp(
        state.pore_exposure + (target.pore_exposure - state.pore_exposure) * adapt,
        0.0,
        1.0,
    );
    state.roughness_mm = clamp(
        state.roughness_mm + (target.roughness_mm - state.roughness_mm) * adapt,
        0.0,
        inputs.cell_size_mm.max(1.0e-3),
    );
    state.collapse_rate = clamp(
        state.collapse_rate + (target.collapse_rate - state.collapse_rate) * adapt,
        0.0,
        1.0,
    );
}

impl TerrariumWorld {
    pub(super) fn step_litter_surface(&mut self, eco_dt: f32) {
        let plane = self.config.width * self.config.height;
        if self.litter_surface.len() != plane {
            self.litter_surface = vec![LitterSurfaceState::default(); plane];
        }
        for flat in 0..plane {
            step_litter_surface(
                &mut self.litter_surface[flat],
                eco_dt,
                LitterSurfaceInputs {
                    cell_size_mm: self.config.cell_size_mm,
                    litter_carbon: self.litter_carbon[flat],
                    organic_matter: self.organic_matter[flat],
                    surface_moisture: self.moisture[flat],
                    deep_moisture: self.deep_moisture[flat],
                    bioturbation_mm2_day: self.earthworm_population.bioturbation_rate[flat],
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{initialize_litter_surface, step_litter_surface, LitterSurfaceInputs};

    #[test]
    fn litter_surface_tracks_dense_wet_cover() {
        let state = initialize_litter_surface(LitterSurfaceInputs {
            cell_size_mm: 0.24,
            litter_carbon: 0.20,
            organic_matter: 0.90,
            surface_moisture: 0.86,
            deep_moisture: 0.74,
            bioturbation_mm2_day: 0.6,
        });
        assert!(state.cover_fraction > 0.4);
        assert!(state.support_depth_mm > 0.05);
        assert!(state.pore_exposure < 0.5);
    }

    #[test]
    fn litter_surface_reacts_to_sparse_drying_cover() {
        let mut state = initialize_litter_surface(LitterSurfaceInputs {
            cell_size_mm: 0.24,
            litter_carbon: 0.16,
            organic_matter: 0.72,
            surface_moisture: 0.80,
            deep_moisture: 0.78,
            bioturbation_mm2_day: 0.4,
        });
        let dense_cover = state.cover_fraction;
        let dense_support = state.support_depth_mm;

        for _ in 0..24 {
            step_litter_surface(
                &mut state,
                120.0,
                LitterSurfaceInputs {
                    cell_size_mm: 0.24,
                    litter_carbon: 0.01,
                    organic_matter: 0.05,
                    surface_moisture: 0.10,
                    deep_moisture: 0.82,
                    bioturbation_mm2_day: 2.8,
                },
            );
        }

        assert!(state.cover_fraction < dense_cover);
        assert!(state.support_depth_mm < dense_support);
        assert!(state.pore_exposure > 0.2);
        assert!(state.collapse_rate > 0.1);
    }
}
