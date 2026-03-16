//! Pure assembly-inventory authority and surrogate-diagnostic helpers extracted
//! from `whole_cell.rs`.

use crate::whole_cell_data::WholeCellComplexAssemblyState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WholeCellSurrogateDiagnosticPools {
    pub(crate) active_rnap: f32,
    pub(crate) active_ribosomes: f32,
    pub(crate) dnaa: f32,
    pub(crate) ftsz: f32,
}

pub(crate) fn authoritative_assembly_inventory(
    complex_assembly: WholeCellComplexAssemblyState,
    named_complex_inventory: Option<WholeCellComplexAssemblyState>,
    explicit_runtime_inventory: Option<WholeCellComplexAssemblyState>,
    fallback_inventory: WholeCellComplexAssemblyState,
) -> WholeCellComplexAssemblyState {
    if complex_assembly.total_complexes() > 1.0e-6 {
        complex_assembly
    } else if let Some(inventory) = named_complex_inventory {
        inventory
    } else if let Some(inventory) = explicit_runtime_inventory {
        inventory
    } else {
        fallback_inventory
    }
}

pub(crate) fn assembly_inventory_projection_available(
    inventory: WholeCellComplexAssemblyState,
) -> bool {
    inventory.total_complexes()
        + inventory.atp_band_target
        + inventory.ribosome_target
        + inventory.rnap_target
        + inventory.replisome_target
        + inventory.membrane_target
        + inventory.ftsz_target
        + inventory.dnaa_target
        + inventory.atp_band_assembly_rate
        + inventory.ribosome_assembly_rate
        + inventory.rnap_assembly_rate
        + inventory.replisome_assembly_rate
        + inventory.membrane_assembly_rate
        + inventory.ftsz_assembly_rate
        + inventory.dnaa_assembly_rate
        + inventory.atp_band_degradation_rate
        + inventory.ribosome_degradation_rate
        + inventory.rnap_degradation_rate
        + inventory.replisome_degradation_rate
        + inventory.membrane_degradation_rate
        + inventory.ftsz_degradation_rate
        + inventory.dnaa_degradation_rate
        > 1.0e-6
}

pub(crate) fn surrogate_diagnostic_pools(
    inventory: WholeCellComplexAssemblyState,
) -> WholeCellSurrogateDiagnosticPools {
    WholeCellSurrogateDiagnosticPools {
        active_rnap: inventory.rnap_complexes.clamp(8.0, 256.0),
        active_ribosomes: inventory.ribosome_complexes.clamp(12.0, 320.0),
        dnaa: inventory.dnaa_activity.clamp(8.0, 256.0),
        ftsz: inventory.ftsz_polymer.clamp(12.0, 384.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoritative_assembly_inventory_prefers_current_state() {
        let current = WholeCellComplexAssemblyState {
            ribosome_complexes: 12.0,
            ..WholeCellComplexAssemblyState::default()
        };
        let named = WholeCellComplexAssemblyState {
            ribosome_complexes: 8.0,
            ..WholeCellComplexAssemblyState::default()
        };
        let fallback = WholeCellComplexAssemblyState {
            ribosome_complexes: 4.0,
            ..WholeCellComplexAssemblyState::default()
        };

        let selected = authoritative_assembly_inventory(current, Some(named), None, fallback);

        assert!((selected.ribosome_complexes - 12.0).abs() < 1.0e-6);
    }

    #[test]
    fn authoritative_assembly_inventory_prefers_named_then_explicit_then_fallback() {
        let named = WholeCellComplexAssemblyState {
            rnap_complexes: 9.0,
            ..WholeCellComplexAssemblyState::default()
        };
        let explicit = WholeCellComplexAssemblyState {
            rnap_complexes: 7.0,
            ..WholeCellComplexAssemblyState::default()
        };
        let fallback = WholeCellComplexAssemblyState {
            rnap_complexes: 5.0,
            ..WholeCellComplexAssemblyState::default()
        };

        let named_selected = authoritative_assembly_inventory(
            WholeCellComplexAssemblyState::default(),
            Some(named),
            Some(explicit),
            fallback,
        );
        let explicit_selected = authoritative_assembly_inventory(
            WholeCellComplexAssemblyState::default(),
            None,
            Some(explicit),
            fallback,
        );
        let fallback_selected = authoritative_assembly_inventory(
            WholeCellComplexAssemblyState::default(),
            None,
            None,
            fallback,
        );

        assert!((named_selected.rnap_complexes - 9.0).abs() < 1.0e-6);
        assert!((explicit_selected.rnap_complexes - 7.0).abs() < 1.0e-6);
        assert!((fallback_selected.rnap_complexes - 5.0).abs() < 1.0e-6);
    }

    #[test]
    fn assembly_inventory_projection_available_reads_targets_and_rates() {
        let inventory = WholeCellComplexAssemblyState {
            ribosome_target: 2.0,
            dnaa_assembly_rate: 0.5,
            ..WholeCellComplexAssemblyState::default()
        };

        assert!(assembly_inventory_projection_available(inventory));
        assert!(!assembly_inventory_projection_available(
            WholeCellComplexAssemblyState::default()
        ));
    }

    #[test]
    fn surrogate_diagnostic_pools_clamp_inventory_channels() {
        let inventory = WholeCellComplexAssemblyState {
            rnap_complexes: 2.0,
            ribosome_complexes: 800.0,
            dnaa_activity: 40.0,
            ftsz_polymer: 6.0,
            ..WholeCellComplexAssemblyState::default()
        };

        let pools = surrogate_diagnostic_pools(inventory);

        assert!((pools.active_rnap - 8.0).abs() < 1.0e-6);
        assert!((pools.active_ribosomes - 320.0).abs() < 1.0e-6);
        assert!((pools.dnaa - 40.0).abs() < 1.0e-6);
        assert!((pools.ftsz - 12.0).abs() < 1.0e-6);
    }
}
