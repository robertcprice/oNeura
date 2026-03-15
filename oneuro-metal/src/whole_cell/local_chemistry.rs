use super::*;

impl WholeCellSimulator {
    pub fn enable_local_chemistry(
        &mut self,
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        voxel_size_au: f32,
        use_gpu: bool,
    ) {
        self.chemistry_bridge = Some(WholeCellChemistryBridge::new(
            x_dim,
            y_dim,
            z_dim,
            voxel_size_au,
            use_gpu,
        ));
    }

    /// Disable the local chemistry submodel.
    pub fn disable_local_chemistry(&mut self) {
        self.chemistry_bridge = None;
        self.chemistry_report = LocalChemistryReport::default();
        self.chemistry_site_reports.clear();
        self.last_md_probe = None;
        self.scheduled_subsystem_probes.clear();
        self.md_translation_scale = 1.0;
        self.md_membrane_scale = 1.0;
    }

    /// Latest local chemistry report, if enabled.
    pub fn local_chemistry_report(&self) -> Option<LocalChemistryReport> {
        if self.has_explicit_local_chemistry_state() {
            Some(self.chemistry_report)
        } else {
            None
        }
    }

    /// Latest per-subsystem local chemistry reports, if enabled.
    pub fn local_chemistry_sites(&self) -> Vec<LocalChemistrySiteReport> {
        if self.has_explicit_local_chemistry_state() {
            self.chemistry_site_reports.clone()
        } else {
            Vec::new()
        }
    }

    /// Latest localized MD probe report.
    pub fn last_md_probe(&self) -> Option<LocalMDProbeReport> {
        self.last_md_probe
    }

    /// Current persistent coupling state for each Syn3A subsystem preset.
    pub fn subsystem_states(&self) -> Vec<WholeCellSubsystemState> {
        self.subsystem_states.clone()
    }

    /// Run a localized MD probe through the optional chemistry bridge.
    pub fn run_local_md_probe(
        &mut self,
        request: LocalMDProbeRequest,
    ) -> Option<LocalMDProbeReport> {
        let report = {
            let bridge = self.chemistry_bridge.as_mut()?;
            bridge.run_md_probe(request)
        };
        if let Some(preset) = Self::preset_for_site(report.site) {
            self.apply_probe_to_subsystem(preset, report);
        }
        self.last_md_probe = Some(report);
        self.md_translation_scale = report.recommended_translation_scale;
        self.md_membrane_scale = report.recommended_membrane_scale;
        Some(report)
    }

    /// Run a named Syn3A subsystem probe using the default request.
    pub fn run_syn3a_subsystem_probe(
        &mut self,
        preset: Syn3ASubsystemPreset,
    ) -> Option<LocalMDProbeReport> {
        self.run_local_md_probe(preset.default_probe_request())
    }

    /// Generate lower-scale calibration sweep samples from the active chemistry bridge.
    pub fn derivation_calibration_samples(
        &mut self,
        dt_ms: f32,
        equilibration_steps: usize,
    ) -> Vec<WholeCellDerivationCalibrationSample> {
        match self.chemistry_bridge.as_mut() {
            Some(bridge) => bridge.derivation_calibration_samples(dt_ms, equilibration_steps),
            None => Vec::new(),
        }
    }

    /// Fit descriptor-to-signature derivation gains against bridge sweep outputs.
    pub fn fit_derivation_calibration(
        &mut self,
        dt_ms: f32,
        equilibration_steps: usize,
    ) -> Result<Option<WholeCellDerivationCalibrationFit>, String> {
        match self.chemistry_bridge.as_mut() {
            Some(bridge) => bridge
                .fit_derivation_calibration(dt_ms, equilibration_steps)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Schedule a Syn3A subsystem probe to run periodically.
    pub fn schedule_syn3a_subsystem_probe(
        &mut self,
        preset: Syn3ASubsystemPreset,
        interval_steps: u64,
    ) {
        let interval_steps = interval_steps.max(1);
        if let Some(existing) = self
            .scheduled_subsystem_probes
            .iter_mut()
            .find(|probe| probe.preset == preset)
        {
            existing.interval_steps = interval_steps;
            return;
        }
        self.scheduled_subsystem_probes
            .push(ScheduledSubsystemProbe {
                preset,
                interval_steps,
            });
    }

    /// Clear all scheduled Syn3A subsystem probes.
    pub fn clear_syn3a_subsystem_probes(&mut self) {
        self.scheduled_subsystem_probes.clear();
        for state in &mut self.subsystem_states {
            *state = WholeCellSubsystemState::new(state.preset);
            state.apply_chemistry_report(self.chemistry_report);
        }
        self.md_translation_scale = 1.0;
        self.md_membrane_scale = 1.0;
    }

    /// Return a copy of the scheduled subsystem probes.
    pub fn scheduled_syn3a_subsystem_probes(&self) -> Vec<ScheduledSubsystemProbe> {
        self.scheduled_subsystem_probes.clone()
    }

    /// Enable the default set of Syn3A subsystem probes.
    pub fn enable_default_syn3a_subsystems(&mut self) {
        if self.chemistry_bridge.is_none() {
            self.enable_local_chemistry(12, 12, 6, 0.5, true);
        }
        self.clear_syn3a_subsystem_probes();
        for preset in Syn3ASubsystemPreset::all() {
            self.schedule_syn3a_subsystem_probe(*preset, preset.default_interval_steps());
        }
    }

    pub(super) fn update_local_chemistry(&mut self, dt: f32) {
        let snapshot = self.snapshot();
        let scheduled_probes = self.scheduled_subsystem_probes.clone();
        let Some((chemistry_report, chemistry_site_reports, last_md_report, due_reports)) = ({
            let Some(ref mut bridge) = self.chemistry_bridge else {
                return;
            };
            let chemistry_report = bridge.step_with_snapshot((dt * 2.0).max(0.1), Some(&snapshot));
            let chemistry_site_reports = bridge.site_reports();
            let last_md_report = bridge.last_md_report();
            let mut due_reports = Vec::new();
            for scheduled in &scheduled_probes {
                if self.step_count % scheduled.interval_steps == 0 {
                    let report = bridge.run_md_probe(scheduled.preset.default_probe_request());
                    due_reports.push((scheduled.preset, report));
                }
            }
            Some((
                chemistry_report,
                chemistry_site_reports,
                last_md_report,
                due_reports,
            ))
        }) else {
            return;
        };

        self.chemistry_report = chemistry_report;
        self.chemistry_site_reports = chemistry_site_reports;
        self.refresh_subsystem_chemistry_state();
        if scheduled_probes.is_empty() {
            self.last_md_probe = last_md_report;
            return;
        }

        if due_reports.is_empty() {
            if let Some(report) = last_md_report {
                self.last_md_probe = Some(report);
            }
        } else {
            for (preset, report) in &due_reports {
                self.apply_probe_to_subsystem(*preset, *report);
            }
            let count = due_reports.len() as f32;
            self.md_translation_scale = due_reports
                .iter()
                .map(|(_, report)| report.recommended_translation_scale)
                .sum::<f32>()
                / count;
            self.md_membrane_scale = due_reports
                .iter()
                .map(|(_, report)| report.recommended_membrane_scale)
                .sum::<f32>()
                / count;
            self.last_md_probe = due_reports.last().map(|(_, report)| *report);
        }
    }

    pub(super) fn md_translation_scale(&self) -> f32 {
        self.md_translation_scale
    }

    pub(super) fn md_membrane_scale(&self) -> f32 {
        self.md_membrane_scale
    }

    pub(super) fn preset_for_site(
        site: crate::whole_cell_submodels::WholeCellChemistrySite,
    ) -> Option<Syn3ASubsystemPreset> {
        match site {
            crate::whole_cell_submodels::WholeCellChemistrySite::AtpSynthaseBand => {
                Some(Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::RibosomeCluster => {
                Some(Syn3ASubsystemPreset::RibosomePolysomeCluster)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::ChromosomeTrack => {
                Some(Syn3ASubsystemPreset::ReplisomeTrack)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::SeptumRing => {
                Some(Syn3ASubsystemPreset::FtsZSeptumRing)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::Cytosol => None,
        }
    }

    pub(super) fn subsystem_state(&self, preset: Syn3ASubsystemPreset) -> WholeCellSubsystemState {
        self.subsystem_states
            .iter()
            .copied()
            .find(|state| state.preset == preset)
            .unwrap_or_else(|| WholeCellSubsystemState::new(preset))
    }

    pub(super) fn subsystem_state_mut(
        &mut self,
        preset: Syn3ASubsystemPreset,
    ) -> Option<&mut WholeCellSubsystemState> {
        self.subsystem_states
            .iter_mut()
            .find(|state| state.preset == preset)
    }

    pub(super) fn refresh_subsystem_chemistry_state(&mut self) {
        for state in &mut self.subsystem_states {
            if let Some(report) = self
                .chemistry_site_reports
                .iter()
                .find(|report| report.preset == state.preset)
                .copied()
            {
                state.apply_site_report(report);
            } else {
                state.apply_chemistry_report(self.chemistry_report);
            }
        }
    }

    pub(super) fn apply_probe_to_subsystem(
        &mut self,
        preset: Syn3ASubsystemPreset,
        report: LocalMDProbeReport,
    ) {
        let step_count = self.step_count;
        if let Some(state) = self.subsystem_state_mut(preset) {
            state.apply_probe_report(report, step_count);
        }
    }

    pub(super) fn atp_band_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .atp_scale
    }

    pub(super) fn ribosome_translation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .translation_scale
    }

    pub(super) fn replisome_replication_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::ReplisomeTrack)
            .replication_scale
    }

    pub(super) fn replisome_segregation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::ReplisomeTrack)
            .segregation_scale
    }

    pub(super) fn ftsz_translation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing)
            .translation_scale
    }

    pub(super) fn ftsz_constriction_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing)
            .constriction_scale
    }

    pub(super) fn membrane_assembly_scale(&self) -> f32 {
        let atp_band = self.subsystem_state(Syn3ASubsystemPreset::AtpSynthaseMembraneBand);
        let septum = self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing);
        (0.55 * atp_band.membrane_scale + 0.45 * septum.membrane_scale).clamp(0.70, 1.45)
    }

    pub(super) fn localized_supply_scale(&self) -> f32 {
        if self.chemistry_site_reports.is_empty() {
            return 1.0;
        }
        let mean_satisfaction = self
            .chemistry_site_reports
            .iter()
            .map(|report| report.demand_satisfaction)
            .sum::<f32>()
            / self.chemistry_site_reports.len() as f32;
        Self::finite_scale(mean_satisfaction, 1.0, 0.55, 1.0)
    }

    pub(super) fn localized_resource_pressure(&self) -> f32 {
        if self.chemistry_site_reports.is_empty() {
            return 0.0;
        }
        self.chemistry_site_reports
            .iter()
            .map(|report| {
                0.45 * report.substrate_draw
                    + 0.55 * report.energy_draw
                    + 0.50 * report.biosynthetic_draw
                    + 0.60 * report.byproduct_load
                    + (1.0 - report.demand_satisfaction).max(0.0) * 1.2
            })
            .sum::<f32>()
            / self.chemistry_site_reports.len() as f32
    }

    pub(super) fn effective_metabolic_load(&self) -> f32 {
        let supply_scale = self.localized_supply_scale();
        let pressure = self.localized_resource_pressure();
        let local_multiplier =
            (1.0 + pressure * 0.16 + (1.0 - supply_scale).max(0.0) * 0.35).clamp(1.0, 2.2);
        let organism_multiplier = self
            .organism_expression
            .metabolic_burden_scale
            .clamp(0.85, 1.65);
        self.metabolic_load.max(0.1) * local_multiplier * organism_multiplier
    }
}
