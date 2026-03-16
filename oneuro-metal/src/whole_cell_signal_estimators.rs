//! Pure resource and subsystem signal estimators extracted from `whole_cell.rs`.
//!
//! These helpers reduce local chemistry and subsystem state into normalized
//! scalar signals for the rule layer.

use crate::substrate_ir::{
    ScalarBranch, ScalarContext, ScalarFactor, ScalarRule, EMPTY_SCALAR_FACTOR,
};
use crate::whole_cell_submodels::WholeCellSubsystemState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum SubsystemEstimatorSignal {
    HealthMix = 0,
    SupportScale,
    DemandCrowdingMix,
    PenaltyMix,
}

impl SubsystemEstimatorSignal {
    const COUNT: usize = Self::PenaltyMix as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct SubsystemEstimatorContext {
    signals: [f32; SubsystemEstimatorSignal::COUNT],
}

impl Default for SubsystemEstimatorContext {
    fn default() -> Self {
        Self {
            signals: [0.0; SubsystemEstimatorSignal::COUNT],
        }
    }
}

impl SubsystemEstimatorContext {
    fn set(&mut self, signal: SubsystemEstimatorSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ SubsystemEstimatorSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum ResourceEstimatorSignal {
    RawPool = 0,
    LocalMean,
    SupportMix,
    Pressure,
}

impl ResourceEstimatorSignal {
    const COUNT: usize = Self::Pressure as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct ResourceEstimatorContext {
    signals: [f32; ResourceEstimatorSignal::COUNT],
}

impl Default for ResourceEstimatorContext {
    fn default() -> Self {
        Self {
            signals: [0.0; ResourceEstimatorSignal::COUNT],
        }
    }
}

impl ResourceEstimatorContext {
    fn set(&mut self, signal: ResourceEstimatorSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ ResourceEstimatorSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

const fn subsystem_factor(signal: SubsystemEstimatorSignal, bias: f32, scale: f32) -> ScalarFactor {
    ScalarFactor::new(signal as usize, bias, scale, 1.0)
}

const fn resource_factor(signal: ResourceEstimatorSignal, bias: f32, scale: f32) -> ScalarFactor {
    ScalarFactor::new(signal as usize, bias, scale, 1.0)
}

const fn scalar_branch_1(f1: ScalarFactor, coefficient: f32) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        1,
        [
            f1,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const SUBSYSTEM_INVENTORY_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.108,
    4,
    [
        ScalarBranch::new(
            1.0,
            1,
            [
                subsystem_factor(SubsystemEstimatorSignal::HealthMix, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.08,
            1,
            [
                subsystem_factor(SubsystemEstimatorSignal::SupportScale, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            1.0,
            1,
            [
                subsystem_factor(SubsystemEstimatorSignal::DemandCrowdingMix, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            -1.0,
            1,
            [
                subsystem_factor(SubsystemEstimatorSignal::PenaltyMix, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
    ],
    0.15,
    1.60,
);

const GLUCOSE_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.10,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.42,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.20,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

const OXYGEN_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.10,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.50,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.16,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

const AMINO_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.08,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.62,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.08,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

const NUCLEOTIDE_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.12,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.62,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.10,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.16,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

const MEMBRANE_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.10,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            1.00,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.08,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.12,
        ),
    ],
    0.0,
    1.0,
);

const ENERGY_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.08,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.40,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.22,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

fn evaluate_resource_signal(
    rule: ScalarRule,
    raw_pool: f32,
    local_mean: f32,
    support_mix: f32,
    pressure: f32,
) -> f32 {
    let mut ctx = ResourceEstimatorContext::default();
    ctx.set(ResourceEstimatorSignal::RawPool, raw_pool);
    ctx.set(ResourceEstimatorSignal::LocalMean, local_mean);
    ctx.set(ResourceEstimatorSignal::SupportMix, support_mix);
    ctx.set(ResourceEstimatorSignal::Pressure, pressure);
    rule.evaluate(ctx.scalar())
}

pub(crate) fn glucose_signal(
    raw_pool: f32,
    local_mean: f32,
    support_mix: f32,
    pressure: f32,
) -> f32 {
    evaluate_resource_signal(
        GLUCOSE_SIGNAL_RULE,
        raw_pool,
        local_mean,
        support_mix,
        pressure,
    )
}

pub(crate) fn oxygen_signal(
    raw_pool: f32,
    local_mean: f32,
    support_mix: f32,
    pressure: f32,
) -> f32 {
    evaluate_resource_signal(
        OXYGEN_SIGNAL_RULE,
        raw_pool,
        local_mean,
        support_mix,
        pressure,
    )
}

pub(crate) fn amino_signal(raw_pool: f32, local_mean: f32, support_mix: f32, pressure: f32) -> f32 {
    evaluate_resource_signal(
        AMINO_SIGNAL_RULE,
        raw_pool,
        local_mean,
        support_mix,
        pressure,
    )
}

pub(crate) fn nucleotide_signal(
    raw_pool: f32,
    local_mean: f32,
    support_mix: f32,
    pressure: f32,
) -> f32 {
    evaluate_resource_signal(
        NUCLEOTIDE_SIGNAL_RULE,
        raw_pool,
        local_mean,
        support_mix,
        pressure,
    )
}

pub(crate) fn membrane_signal(
    raw_pool: f32,
    local_mean: f32,
    support_mix: f32,
    pressure: f32,
) -> f32 {
    evaluate_resource_signal(
        MEMBRANE_SIGNAL_RULE,
        raw_pool,
        local_mean,
        support_mix,
        pressure,
    )
}

pub(crate) fn energy_signal(
    raw_pool: f32,
    local_mean: f32,
    support_mix: f32,
    pressure: f32,
) -> f32 {
    evaluate_resource_signal(
        ENERGY_SIGNAL_RULE,
        raw_pool,
        local_mean,
        support_mix,
        pressure,
    )
}

pub(crate) fn subsystem_inventory_signal(state: WholeCellSubsystemState, support: f32) -> f32 {
    let mut ctx = SubsystemEstimatorContext::default();
    ctx.set(
        SubsystemEstimatorSignal::HealthMix,
        0.18 * state.structural_order
            + 0.16 * state.assembly_component_availability
            + 0.20 * state.assembly_occupancy
            + 0.16 * state.assembly_stability,
    );
    ctx.set(SubsystemEstimatorSignal::SupportScale, support);
    ctx.set(
        SubsystemEstimatorSignal::DemandCrowdingMix,
        0.118 * state.demand_satisfaction + 0.074 * state.crowding_penalty,
    );
    ctx.set(
        SubsystemEstimatorSignal::PenaltyMix,
        0.14 * state.assembly_turnover + 0.02 * state.byproduct_load,
    );
    SUBSYSTEM_INVENTORY_SIGNAL_RULE.evaluate(ctx.scalar())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whole_cell_submodels::Syn3ASubsystemPreset;

    #[test]
    fn resource_signals_increase_with_raw_pool() {
        let low = glucose_signal(0.1, 0.1, 0.5, 0.0);
        let high = glucose_signal(1.0, 0.1, 0.5, 0.0);
        assert!(high > low);
    }

    #[test]
    fn subsystem_inventory_signal_penalizes_turnover_and_byproducts() {
        let mut healthy = WholeCellSubsystemState::new(Syn3ASubsystemPreset::FtsZSeptumRing);
        healthy.crowding_penalty = 0.5;
        healthy.structural_order = 1.0;
        healthy.assembly_component_availability = 1.0;
        healthy.assembly_occupancy = 1.0;
        healthy.assembly_stability = 1.0;
        healthy.demand_satisfaction = 1.0;
        healthy.assembly_turnover = 0.0;
        healthy.byproduct_load = 0.0;

        let mut stressed = WholeCellSubsystemState::new(Syn3ASubsystemPreset::FtsZSeptumRing);
        stressed.crowding_penalty = 1.0;
        stressed.structural_order = 1.0;
        stressed.assembly_component_availability = 1.0;
        stressed.assembly_occupancy = 1.0;
        stressed.assembly_stability = 1.0;
        stressed.demand_satisfaction = 1.0;
        stressed.assembly_turnover = 1.0;
        stressed.byproduct_load = 1.0;

        let healthy = subsystem_inventory_signal(healthy, 1.0);
        let stressed = subsystem_inventory_signal(stressed, 1.0);
        assert!(healthy > stressed);
    }
}
