//! Pure named-complex seed, update, and aggregation helpers extracted from
//! `whole_cell.rs`.
//!
//! These reducers keep formula-heavy complex assembly dynamics out of the main
//! simulator implementation while leaving live runtime signal gathering in the
//! runtime itself.

use crate::whole_cell_assembly_projection::{
    accumulate_assembly_inventory_projection, named_complex_effective_projection,
};
use crate::whole_cell_complex_channels::complex_channel_step;
use crate::whole_cell_data::{
    WholeCellComplexAssemblyState, WholeCellComplexSpec, WholeCellNamedComplexState,
};
use crate::whole_cell_process_weights::named_complex_projection_weights;
use crate::whole_cell_scale_reducers::finite_scale;

#[derive(Debug, Clone, Copy)]
pub(crate) struct NamedComplexSeedInputs {
    pub(crate) component_satisfaction: f32,
    pub(crate) structural_support: f32,
    pub(crate) subunit_pool: f32,
    pub(crate) crowding_penalty: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NamedComplexUpdateInputs {
    pub(crate) dt_scale: f32,
    pub(crate) crowding_penalty: f32,
    pub(crate) degradation_pressure: f32,
    pub(crate) component_satisfaction: f32,
    pub(crate) component_supply_signal: f32,
    pub(crate) structural_support: f32,
    pub(crate) subunit_pool_target: f32,
}

pub(crate) fn named_complex_total_stoichiometry(complex: &WholeCellComplexSpec) -> f32 {
    complex
        .components
        .iter()
        .map(|component| component.stoichiometry.max(1) as f32)
        .sum::<f32>()
        .max(1.0)
}

pub(crate) fn seed_named_complex_state(
    complex: &WholeCellComplexSpec,
    inputs: NamedComplexSeedInputs,
) -> WholeCellNamedComplexState {
    let component_satisfaction = inputs.component_satisfaction.clamp(0.0, 1.0);
    let structural_support = inputs.structural_support.clamp(0.0, 2.0);
    let subunit_pool = inputs.subunit_pool.max(0.0);
    let crowding = inputs.crowding_penalty.clamp(0.65, 1.10);
    let total_stoichiometry = named_complex_total_stoichiometry(complex);
    let target_abundance =
        (complex.basal_abundance.max(0.1) * component_satisfaction * structural_support * crowding)
            .clamp(0.0, 512.0);
    let nucleation_intermediate =
        (0.10 * target_abundance * component_satisfaction * total_stoichiometry.sqrt())
            .clamp(0.0, 256.0);
    let elongation_intermediate =
        (0.08 * target_abundance * structural_support * total_stoichiometry.sqrt())
            .clamp(0.0, 256.0);
    let assembly_progress = (0.42 * component_satisfaction
        + 0.30 * structural_support
        + 0.18 * saturating_signal(subunit_pool, 6.0 + 0.5 * total_stoichiometry.max(1.0))
        + 0.10
            * saturating_signal(
                nucleation_intermediate + elongation_intermediate,
                2.0 + 0.4 * total_stoichiometry.max(1.0),
            ))
    .clamp(0.0, 1.0);

    WholeCellNamedComplexState {
        id: complex.id.clone(),
        operon: complex.operon.clone(),
        asset_class: complex.asset_class,
        subsystem_targets: complex.subsystem_targets.clone(),
        subunit_pool,
        nucleation_intermediate,
        elongation_intermediate,
        abundance: target_abundance,
        target_abundance,
        assembly_rate: 0.0,
        degradation_rate: 0.0,
        nucleation_rate: 0.0,
        elongation_rate: 0.0,
        maturation_rate: 0.0,
        component_satisfaction,
        structural_support,
        assembly_progress,
    }
}

pub(crate) fn update_named_complex_state(
    state: &WholeCellNamedComplexState,
    complex: &WholeCellComplexSpec,
    inputs: NamedComplexUpdateInputs,
) -> WholeCellNamedComplexState {
    let component_satisfaction = inputs.component_satisfaction.clamp(0.0, 1.0);
    let component_supply_signal = inputs.component_supply_signal.clamp(0.0, 1.0);
    let structural_support = inputs.structural_support.clamp(0.0, 2.0);
    let crowding = inputs.crowding_penalty.clamp(0.65, 1.10);
    let dt_scale = inputs.dt_scale.clamp(0.5, 6.0);
    let degradation_pressure = inputs.degradation_pressure.clamp(0.60, 1.80);
    let total_stoichiometry = named_complex_total_stoichiometry(complex);
    let complexity_penalty = 1.0 / total_stoichiometry.sqrt().max(1.0);
    let assembly_support = finite_scale(
        0.52 * component_satisfaction + 0.34 * structural_support + 0.14 * crowding,
        1.0,
        0.45,
        1.75,
    );
    let target_abundance =
        (complex.basal_abundance.max(0.1) * component_satisfaction * structural_support * crowding)
            .clamp(0.0, 512.0);
    let subunit_supply_rate = (inputs.subunit_pool_target.max(0.0) - state.subunit_pool).max(0.0)
        * (0.10 + 0.16 * component_supply_signal);
    let subunit_turnover_rate = state.subunit_pool
        * (0.010
            + 0.012 * (1.0 - component_satisfaction).max(0.0)
            + 0.008 * (1.0 - crowding).max(0.0));
    let nucleation_rate = state.subunit_pool
        * (0.012 + 0.020 * component_satisfaction)
        * complexity_penalty
        * assembly_support;
    let nucleation_turnover_rate = state.nucleation_intermediate
        * (0.012
            + 0.012 * (1.0 - structural_support).max(0.0)
            + 0.008 * (1.0 - component_satisfaction).max(0.0));
    let elongation_rate = state.nucleation_intermediate
        * (0.016 + 0.024 * structural_support)
        * (0.82 + 0.18 * component_supply_signal);
    let elongation_turnover_rate = state.elongation_intermediate
        * (0.010 + 0.010 * (1.0 - structural_support).max(0.0) + 0.006 * (1.0 - crowding).max(0.0));
    let maturation_rate = state.elongation_intermediate
        * (0.020 + 0.032 * assembly_support)
        * (0.80 + 0.20 * component_satisfaction);
    let channel_degradation_pressure = (degradation_pressure
        + 0.45 * (1.0 - component_satisfaction).max(0.0)
        + 0.20 * (1.0 - structural_support).max(0.0))
    .clamp(0.60, 2.10);
    let (abundance, assembly_rate, degradation_rate) = complex_channel_step(
        state.abundance,
        target_abundance,
        assembly_support,
        channel_degradation_pressure,
        dt_scale,
        512.0,
    );
    let subunit_pool = (state.subunit_pool
        + dt_scale * (subunit_supply_rate - 0.70 * nucleation_rate - subunit_turnover_rate))
        .clamp(0.0, 2048.0);
    let nucleation_intermediate = (state.nucleation_intermediate
        + dt_scale * (nucleation_rate - 0.65 * elongation_rate - nucleation_turnover_rate))
        .clamp(0.0, 512.0);
    let elongation_intermediate = (state.elongation_intermediate
        + dt_scale * (elongation_rate - 0.75 * maturation_rate - elongation_turnover_rate))
        .clamp(0.0, 512.0);
    let assembly_progress = (0.36 * component_satisfaction
        + 0.26 * structural_support
        + 0.18 * saturating_signal(subunit_pool, 6.0 + 0.5 * total_stoichiometry.max(1.0))
        + 0.20
            * saturating_signal(
                nucleation_intermediate + elongation_intermediate,
                2.0 + 0.35 * total_stoichiometry.max(1.0),
            ))
    .clamp(0.0, 1.0);

    WholeCellNamedComplexState {
        id: state.id.clone(),
        operon: state.operon.clone(),
        asset_class: state.asset_class,
        subsystem_targets: state.subsystem_targets.clone(),
        subunit_pool,
        nucleation_intermediate,
        elongation_intermediate,
        abundance,
        target_abundance,
        assembly_rate,
        degradation_rate,
        nucleation_rate,
        elongation_rate,
        maturation_rate,
        component_satisfaction,
        structural_support,
        assembly_progress,
    }
}

pub(crate) fn aggregate_named_complex_assembly_state(
    states: &[WholeCellNamedComplexState],
    complexes: &[WholeCellComplexSpec],
) -> WholeCellComplexAssemblyState {
    let mut inventory = WholeCellComplexAssemblyState::default();
    for (state, complex) in states.iter().zip(complexes.iter()) {
        let weights = named_complex_projection_weights(complex);
        let (
            effective_abundance,
            effective_target,
            effective_assembly_rate,
            effective_degradation_rate,
        ) = named_complex_effective_projection(state);
        accumulate_assembly_inventory_projection(
            &mut inventory,
            weights,
            effective_abundance,
            effective_target,
            effective_assembly_rate,
            effective_degradation_rate,
        );
    }
    inventory
}

fn saturating_signal(value: f32, half_saturation: f32) -> f32 {
    let value = value.max(0.0);
    let half_saturation = half_saturation.max(1.0e-6);
    (value / (value + half_saturation)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whole_cell_data::{WholeCellAssetClass, WholeCellComplexComponentSpec};

    #[test]
    fn named_complex_total_stoichiometry_sums_component_counts() {
        let complex = WholeCellComplexSpec {
            id: "complex".to_string(),
            name: "Complex".to_string(),
            operon: "operon".to_string(),
            components: vec![
                WholeCellComplexComponentSpec {
                    protein_id: "a".to_string(),
                    stoichiometry: 2,
                },
                WholeCellComplexComponentSpec {
                    protein_id: "b".to_string(),
                    stoichiometry: 3,
                },
            ],
            basal_abundance: 4.0,
            asset_class: WholeCellAssetClass::Generic,
            process_weights: Default::default(),
            subsystem_targets: Vec::new(),
        };

        assert!((named_complex_total_stoichiometry(&complex) - 5.0).abs() < 1.0e-6);
    }

    #[test]
    fn seed_named_complex_state_builds_initial_intermediates() {
        let complex = WholeCellComplexSpec {
            id: "complex".to_string(),
            name: "Complex".to_string(),
            operon: "operon".to_string(),
            components: vec![WholeCellComplexComponentSpec {
                protein_id: "a".to_string(),
                stoichiometry: 2,
            }],
            basal_abundance: 8.0,
            asset_class: WholeCellAssetClass::Translation,
            process_weights: Default::default(),
            subsystem_targets: Vec::new(),
        };

        let state = seed_named_complex_state(
            &complex,
            NamedComplexSeedInputs {
                component_satisfaction: 0.8,
                structural_support: 1.1,
                subunit_pool: 12.0,
                crowding_penalty: 1.0,
            },
        );

        assert!(state.abundance > 0.0);
        assert!((state.abundance - state.target_abundance).abs() < 1.0e-6);
        assert!(state.nucleation_intermediate > 0.0);
        assert!(state.elongation_intermediate > 0.0);
        assert!(state.assembly_progress > 0.0);
    }

    #[test]
    fn update_named_complex_state_grows_toward_supported_target() {
        let complex = WholeCellComplexSpec {
            id: "complex".to_string(),
            name: "Complex".to_string(),
            operon: "operon".to_string(),
            components: vec![WholeCellComplexComponentSpec {
                protein_id: "a".to_string(),
                stoichiometry: 2,
            }],
            basal_abundance: 10.0,
            asset_class: WholeCellAssetClass::Translation,
            process_weights: Default::default(),
            subsystem_targets: Vec::new(),
        };
        let state = WholeCellNamedComplexState {
            id: "complex".to_string(),
            operon: "operon".to_string(),
            asset_class: WholeCellAssetClass::Translation,
            subsystem_targets: Vec::new(),
            subunit_pool: 20.0,
            nucleation_intermediate: 5.0,
            elongation_intermediate: 3.0,
            abundance: 2.0,
            target_abundance: 4.0,
            assembly_rate: 0.0,
            degradation_rate: 0.0,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 0.6,
            structural_support: 0.8,
            assembly_progress: 0.0,
        };

        let updated = update_named_complex_state(
            &state,
            &complex,
            NamedComplexUpdateInputs {
                dt_scale: 1.0,
                crowding_penalty: 1.0,
                degradation_pressure: 0.8,
                component_satisfaction: 0.9,
                component_supply_signal: 0.8,
                structural_support: 1.1,
                subunit_pool_target: 30.0,
            },
        );

        assert!(updated.abundance > state.abundance);
        assert!(updated.subunit_pool >= state.subunit_pool);
        assert!(updated.assembly_rate > 0.0);
        assert!(updated.assembly_progress > 0.0);
    }
}
