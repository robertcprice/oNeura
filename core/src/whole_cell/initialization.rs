use super::*;

impl WholeCellSimulator {
    pub(crate) fn surface_area_from_radius(radius_nm: f32) -> f32 {
        4.0 * PI * radius_nm * radius_nm
    }

    pub(crate) fn volume_from_radius(radius_nm: f32) -> f32 {
        4.0 / 3.0 * PI * radius_nm.powi(3)
    }

    fn restore_saved_state(&mut self, saved: WholeCellSavedState) -> Result<(), String> {
        let saved_scheduler_state = saved.scheduler_state.clone();
        self.program_name = saved.program_name.clone();
        self.contract = saved.contract.clone();
        self.provenance = saved.provenance.clone();
        self.organism_data_ref = saved.organism_data_ref.clone();
        self.organism_data = saved.organism_data.clone();
        self.organism_assets = saved.organism_assets.clone();
        self.organism_process_registry = saved.organism_process_registry.clone();
        self.organism_expression = saved.organism_expression.clone();
        self.chromosome_state = saved.chromosome_state.clone();
        self.membrane_division_state = saved.membrane_division_state.clone();
        self.organism_species = saved.organism_species.clone();
        self.normalize_runtime_species_bulk_fields();
        self.organism_reactions = saved.organism_reactions.clone();
        self.complex_assembly = saved.complex_assembly;
        self.named_complexes = saved.named_complexes.clone();
        self.lattice
            .set_species(IntracellularSpecies::ATP, &saved.lattice.atp)?;
        self.lattice
            .set_species(IntracellularSpecies::AminoAcids, &saved.lattice.amino_acids)?;
        self.lattice.set_species(
            IntracellularSpecies::Nucleotides,
            &saved.lattice.nucleotides,
        )?;
        self.lattice.set_species(
            IntracellularSpecies::MembranePrecursors,
            &saved.lattice.membrane_precursors,
        )?;
        if let Some(spatial) = saved.spatial_fields {
            self.apply_spatial_field_state(&spatial);
        }
        self.sync_from_lattice();

        let saved_ftsz = saved.core.ftsz.max(0.0);
        let saved_dnaa = saved.core.dnaa.max(0.0);
        let saved_active_ribosomes = saved.core.active_ribosomes.max(0.0);
        let saved_active_rnap = saved.core.active_rnap.max(0.0);
        let saved_genome_bp = saved.core.genome_bp.max(1);
        let saved_replicated_bp = saved.core.replicated_bp.min(saved_genome_bp);
        let saved_chromosome_separation_nm = saved.core.chromosome_separation_nm.max(0.0);
        let saved_radius_nm = saved.core.radius_nm.max(50.0);
        let saved_surface_area_nm2 = saved.core.surface_area_nm2.max(1.0);
        let saved_volume_nm3 = saved.core.volume_nm3.max(1.0);
        let saved_division_progress = saved.core.division_progress.clamp(0.0, 1.0);
        let saved_has_explicit_chromosome_state = self.chromosome_state.chromosome_length_bp > 1
            || self.chromosome_state.replicated_bp > 0
            || !self.chromosome_state.loci.is_empty()
            || !self.chromosome_state.forks.is_empty();
        let saved_has_explicit_membrane_state =
            self.membrane_division_state.preferred_membrane_area_nm2 > 1.0
                || self.membrane_division_state.membrane_area_nm2 > 1.0;
        let saved_has_explicit_complex_assembly = self.complex_assembly.total_complexes() > 1.0e-6;
        let saved_has_explicit_named_complexes = !self.named_complexes.is_empty();

        self.time_ms = saved.core.time_ms.max(0.0);
        self.step_count = saved.core.step_count;
        self.adp_mm = saved.core.adp_mm.max(0.0);
        self.glucose_mm = saved.core.glucose_mm.max(0.0);
        self.oxygen_mm = saved.core.oxygen_mm.max(0.0);
        self.metabolic_load = saved.core.metabolic_load.max(0.1);
        self.quantum_profile = saved.core.quantum_profile.normalized();
        if self.organism_data.is_some() {
            self.genome_bp = self
                .organism_data
                .as_ref()
                .map(|organism| organism.chromosome_length_bp.max(1))
                .unwrap_or(saved_genome_bp);
            self.replicated_bp = saved_replicated_bp.min(self.genome_bp);
            self.chromosome_separation_nm = saved_chromosome_separation_nm;
            self.radius_nm = saved_radius_nm;
            self.surface_area_nm2 = saved_surface_area_nm2;
            self.volume_nm3 = saved_volume_nm3;
            self.division_progress = saved_division_progress;
            if self.organism_expression.transcription_units.is_empty() {
                self.refresh_organism_expression_state();
            }
            if self.organism_species.is_empty() || self.organism_reactions.is_empty() {
                self.initialize_runtime_process_state();
            }
            if self.named_complexes.is_empty() {
                self.initialize_named_complexes_state();
            }
            self.chromosome_state = if self.chromosome_state.chromosome_length_bp > 1
                || !self.chromosome_state.loci.is_empty()
            {
                self.normalize_chromosome_state(self.chromosome_state.clone())
            } else {
                if !saved_has_explicit_chromosome_state {
                    self.replicated_bp = saved_replicated_bp.min(self.genome_bp);
                    self.chromosome_separation_nm = saved_chromosome_separation_nm;
                }
                self.seeded_chromosome_state()
            };
            self.synchronize_chromosome_summary();
            self.membrane_division_state =
                if self.membrane_division_state.preferred_membrane_area_nm2 > 1.0 {
                    self.normalize_membrane_division_state(self.membrane_division_state.clone())
                } else {
                    if !saved_has_explicit_membrane_state {
                        self.radius_nm = saved_radius_nm;
                        self.surface_area_nm2 = saved_surface_area_nm2;
                        self.volume_nm3 = saved_volume_nm3;
                        self.division_progress = saved_division_progress;
                    }
                    self.seeded_membrane_division_state()
                };
            self.synchronize_membrane_division_summary();
            if !self.named_complexes.is_empty() {
                if let Some(assets) = self.organism_assets.as_ref() {
                    self.complex_assembly = self.aggregate_named_complex_assembly_state(assets);
                }
            } else if self.complex_assembly.total_complexes() <= 1.0e-6 {
                self.initialize_complex_assembly_state();
            }
        } else {
            // Bundle-less restore still has to honor explicit persisted biology
            // when the caller stripped bundled organism descriptors. Only fall
            // back to scalar summaries when the richer layer is genuinely absent.
            if !self.has_explicit_expression_state() {
                self.organism_expression = self
                    .synthesize_expression_state_from_runtime_process()
                    .unwrap_or_default();
            }
            self.ftsz = saved_ftsz;
            self.dnaa = saved_dnaa;
            self.active_ribosomes = saved_active_ribosomes;
            self.active_rnap = saved_active_rnap;
            self.genome_bp = saved_genome_bp;
            self.replicated_bp = saved_replicated_bp.min(self.genome_bp);
            self.chromosome_separation_nm = saved_chromosome_separation_nm;
            self.radius_nm = saved_radius_nm;
            self.surface_area_nm2 = saved_surface_area_nm2;
            self.volume_nm3 = saved_volume_nm3;
            self.division_progress = saved_division_progress;
            self.chromosome_state = if saved_has_explicit_chromosome_state {
                self.normalize_chromosome_state(self.chromosome_state.clone())
            } else {
                self.seeded_chromosome_state()
            };
            self.synchronize_chromosome_summary();
            self.membrane_division_state = if saved_has_explicit_membrane_state {
                self.normalize_membrane_division_state(self.membrane_division_state.clone())
            } else {
                self.seeded_membrane_division_state()
            };
            self.synchronize_membrane_division_summary();
            if !self.named_complexes.is_empty() {
                let aggregate = self.aggregate_named_complex_assembly_state_without_assets();
                self.complex_assembly = if self.complex_assembly.total_complexes() > 1.0e-6 {
                    self.merge_bundleless_assembly_channels(aggregate, self.complex_assembly)
                } else {
                    aggregate
                };
            } else if self.complex_assembly.total_complexes() <= 1.0e-6 {
                self.initialize_complex_assembly_state();
            }
            if saved_has_explicit_named_complexes || saved_has_explicit_complex_assembly {
                self.supplement_bundleless_named_complex_carriers_from_assembly();
            }
        }
        self.refresh_spatial_fields();
        self.refresh_rdme_drive_fields();

        self.disable_local_chemistry();
        if let Some(local) = saved.local_chemistry {
            self.enable_local_chemistry(
                local.x_dim,
                local.y_dim,
                local.z_dim,
                local.voxel_size_au,
                local.use_gpu,
            );
            if local.enable_default_syn3a_subsystems {
                self.enable_default_syn3a_subsystems();
            }
            if !local.scheduled_subsystem_probes.is_empty() {
                self.clear_syn3a_subsystem_probes();
                for probe in local.scheduled_subsystem_probes {
                    self.schedule_syn3a_subsystem_probe(probe.preset, probe.interval_steps);
                }
            }
        }

        self.chemistry_report = saved.chemistry_report;
        self.chemistry_site_reports = saved.chemistry_site_reports;
        self.last_md_probe = saved.last_md_probe;
        self.scheduled_subsystem_probes = saved.scheduled_subsystem_probes;
        let subsystem_states_were_missing = saved.subsystem_states.is_empty();
        self.subsystem_states = if subsystem_states_were_missing {
            Syn3ASubsystemPreset::all()
                .iter()
                .copied()
                .map(WholeCellSubsystemState::new)
                .collect()
        } else {
            saved.subsystem_states
        };
        // Saved chemistry sites need to repopulate persistent subsystem coupling
        // state when older payloads only carried site reports. Without this
        // refresh, restore would keep default subsystem scales until the next
        // live chemistry bridge step even though the saved boundary already has
        // explicit localized chemistry detail.
        if subsystem_states_were_missing {
            self.refresh_subsystem_chemistry_state();
        }
        self.md_translation_scale = Self::finite_scale(saved.md_translation_scale, 1.0, 0.70, 1.45);
        self.md_membrane_scale = Self::finite_scale(saved.md_membrane_scale, 1.0, 0.70, 1.45);
        self.stochastic_config = saved.stochastic_config;
        self.stochastic_operon_states = saved.stochastic_operon_states;
        self.stochastic_rng = saved.stochastic_rng;
        self.scheduler_state =
            Self::normalized_scheduler_state(&self.config, saved_scheduler_state);
        if self
            .scheduler_state
            .stage_clocks
            .iter()
            .all(|clock| clock.run_count == 0)
        {
            self.refresh_multirate_scheduler();
        }
        self.initialize_surrogate_pool_diagnostics();
        Ok(())
    }

    /// Create a simulator with JCVI-syn3A-like defaults.
    pub fn new(config: WholeCellConfig) -> Self {
        let scheduler_state = Self::build_scheduler_state(&config);
        let backend = if config.use_gpu && gpu::has_gpu() {
            WholeCellBackend::Metal
        } else {
            WholeCellBackend::Cpu
        };
        let spatial_dims = (config.x_dim, config.y_dim, config.z_dim);

        #[cfg(target_os = "macos")]
        let gpu = if backend == WholeCellBackend::Metal {
            GpuContext::new().ok()
        } else {
            None
        };

        let backend = {
            #[cfg(target_os = "macos")]
            {
                if backend == WholeCellBackend::Metal && gpu.is_some() {
                    WholeCellBackend::Metal
                } else {
                    WholeCellBackend::Cpu
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                WholeCellBackend::Cpu
            }
        };

        let mut lattice = IntracellularLattice::new(
            config.x_dim,
            config.y_dim,
            config.z_dim,
            config.voxel_size_nm,
        );
        lattice.fill_species(IntracellularSpecies::ATP, 1.20);
        lattice.fill_species(IntracellularSpecies::AminoAcids, 0.95);
        lattice.fill_species(IntracellularSpecies::Nucleotides, 0.80);
        lattice.fill_species(IntracellularSpecies::MembranePrecursors, 0.35);

        let radius_nm = 200.0;
        let surface_area_nm2 = Self::surface_area_from_radius(radius_nm);
        let volume_nm3 = Self::volume_from_radius(radius_nm);

        let mut simulator = Self {
            config,
            backend,
            program_name: None,
            contract: WholeCellContractSchema::default(),
            provenance: WholeCellProvenance::default(),
            organism_data_ref: None,
            organism_data: None,
            organism_assets: None,
            organism_process_registry: None,
            organism_expression: WholeCellOrganismExpressionState::default(),
            organism_species: Vec::new(),
            organism_reactions: Vec::new(),
            complex_assembly: WholeCellComplexAssemblyState::default(),
            named_complexes: Vec::new(),
            #[cfg(target_os = "macos")]
            gpu,
            lattice,
            spatial_fields: IntracellularSpatialState::new(
                spatial_dims.0,
                spatial_dims.1,
                spatial_dims.2,
            ),
            rdme_drive_fields: WholeCellRdmeDriveState::new(
                spatial_dims.0,
                spatial_dims.1,
                spatial_dims.2,
            ),
            time_ms: 0.0,
            step_count: 0,
            atp_mm: 1.20,
            amino_acids_mm: 0.95,
            nucleotides_mm: 0.80,
            membrane_precursors_mm: 0.35,
            adp_mm: 0.30,
            glucose_mm: 1.0,
            oxygen_mm: 0.85,
            ftsz: 0.0,
            dnaa: 0.0,
            active_ribosomes: 0.0,
            active_rnap: 0.0,
            genome_bp: 543_000,
            replicated_bp: 0,
            chromosome_separation_nm: 40.0,
            chromosome_state: WholeCellChromosomeState::default(),
            membrane_division_state: WholeCellMembraneDivisionState::default(),
            radius_nm,
            surface_area_nm2,
            volume_nm3,
            division_progress: 0.0,
            metabolic_load: 1.0,
            quantum_profile: WholeCellQuantumProfile::default(),
            discovered_quantum_reactions: Vec::new(),
            chemistry_bridge: None,
            chemistry_report: LocalChemistryReport::default(),
            chemistry_site_reports: Vec::new(),
            last_md_probe: None,
            scheduled_subsystem_probes: Vec::new(),
            subsystem_states: Syn3ASubsystemPreset::all()
                .iter()
                .copied()
                .map(WholeCellSubsystemState::new)
                .collect(),
            scheduler_state,
            md_translation_scale: 1.0,
            md_membrane_scale: 1.0,
            stochastic_config:
                crate::whole_cell::stochastic_expression::StochasticExpressionConfig::default(),
            stochastic_operon_states: Vec::new(),
            stochastic_rng: crate::whole_cell::stochastic_expression::StochasticRng::new(42),
        };
        simulator.sync_from_lattice();
        simulator.chromosome_state = simulator.seeded_chromosome_state();
        simulator.synchronize_chromosome_summary();
        simulator.membrane_division_state = simulator.seeded_membrane_division_state();
        simulator.synchronize_membrane_division_summary();
        simulator.refresh_spatial_fields();
        simulator.refresh_rdme_drive_fields();
        simulator.refresh_organism_expression_state();
        simulator.initialize_complex_assembly_state();
        simulator.initialize_runtime_process_state();
        simulator.refresh_multirate_scheduler();
        simulator.initialize_surrogate_pool_diagnostics();
        simulator.run_quantum_auto_discovery();
        simulator
    }

    /// Create a simulator from a data-driven whole-cell program spec.
    pub fn from_program_spec(spec: WholeCellProgramSpec) -> Self {
        let config = spec.config.clone();
        let local_chemistry = spec.local_chemistry.clone();
        let spec_has_explicit_named_complexes = !spec.named_complexes.is_empty();
        let spec_has_explicit_complex_assembly = spec
            .complex_assembly
            .as_ref()
            .map(|state| state.total_complexes() > 1.0e-6)
            .unwrap_or(false);
        let mut simulator = Self::new(config);
        simulator.program_name = spec.program_name.clone();
        simulator.contract = spec.contract.clone();
        simulator.provenance = spec.provenance.clone();
        simulator.organism_data_ref = spec.organism_data_ref.clone();
        simulator.organism_data = spec.organism_data.clone();
        simulator.organism_assets = spec.organism_assets.clone();
        simulator.organism_process_registry = spec.organism_process_registry.clone();

        simulator
            .lattice
            .fill_species(IntracellularSpecies::ATP, spec.initial_lattice.atp.max(0.0));
        simulator.lattice.fill_species(
            IntracellularSpecies::AminoAcids,
            spec.initial_lattice.amino_acids.max(0.0),
        );
        simulator.lattice.fill_species(
            IntracellularSpecies::Nucleotides,
            spec.initial_lattice.nucleotides.max(0.0),
        );
        simulator.lattice.fill_species(
            IntracellularSpecies::MembranePrecursors,
            spec.initial_lattice.membrane_precursors.max(0.0),
        );
        simulator.sync_from_lattice();

        simulator.adp_mm = spec.initial_state.adp_mm.max(0.0);
        simulator.glucose_mm = spec.initial_state.glucose_mm.max(0.0);
        simulator.oxygen_mm = spec.initial_state.oxygen_mm.max(0.0);
        simulator.genome_bp = spec.initial_state.genome_bp.max(1);
        simulator.replicated_bp = spec.initial_state.replicated_bp.min(simulator.genome_bp);
        simulator.chromosome_separation_nm = spec.initial_state.chromosome_separation_nm.max(0.0);
        simulator.radius_nm = spec.initial_state.radius_nm.max(50.0);
        simulator.surface_area_nm2 = Self::surface_area_from_radius(simulator.radius_nm);
        simulator.volume_nm3 = Self::volume_from_radius(simulator.radius_nm);
        simulator.division_progress = spec.initial_state.division_progress.clamp(0.0, 1.0);
        simulator.metabolic_load = spec.initial_state.metabolic_load.max(0.1);
        simulator.quantum_profile = spec.quantum_profile.normalized();
        simulator.apply_organism_data_initialization();
        simulator.chromosome_state = spec
            .chromosome_state
            .clone()
            .map(|state| simulator.normalize_chromosome_state(state))
            .unwrap_or_else(|| simulator.seeded_chromosome_state());
        simulator.synchronize_chromosome_summary();
        simulator.membrane_division_state = spec
            .membrane_division_state
            .clone()
            .map(|state| simulator.normalize_membrane_division_state(state))
            .unwrap_or_else(|| simulator.seeded_membrane_division_state());
        simulator.synchronize_membrane_division_summary();

        // Preserve explicit spatial fields after chromosome and membrane state have
        // been normalized, but before RDME drives and later chemistry-aware bootstrap
        // stages read locality from those fields. This keeps compiled or persisted
        // locality state authoritative when the caller already has it.
        simulator.refresh_spatial_fields();
        if let Some(spatial) = spec.spatial_fields.as_ref() {
            simulator.apply_spatial_field_state(spatial);
        }
        simulator.refresh_rdme_drive_fields();

        simulator.disable_local_chemistry();
        if let Some(local) = local_chemistry {
            simulator.enable_local_chemistry(
                local.x_dim,
                local.y_dim,
                local.z_dim,
                local.voxel_size_au,
                local.use_gpu,
            );
            if local.enable_default_syn3a_subsystems {
                simulator.enable_default_syn3a_subsystems();
            }
            if !local.scheduled_subsystem_probes.is_empty() {
                simulator.clear_syn3a_subsystem_probes();
                for probe in local.scheduled_subsystem_probes {
                    simulator.schedule_syn3a_subsystem_probe(probe.preset, probe.interval_steps);
                }
            }
        }

        // Preserve explicit local-chemistry runtime state before later bootstrap
        // stages read those signals. Expression refresh, assembly fallback, runtime
        // chemistry initialization, and scheduler adaptation all consume chemistry
        // support signals, so explicit chemistry reports and probe-coupling state
        // need to land before those downstream fallbacks run.
        if !spec.scheduled_subsystem_probes.is_empty() {
            simulator.clear_syn3a_subsystem_probes();
            for probe in spec.scheduled_subsystem_probes {
                simulator.schedule_syn3a_subsystem_probe(probe.preset, probe.interval_steps);
            }
        }
        simulator.chemistry_report = spec.chemistry_report.unwrap_or_default();
        simulator.chemistry_site_reports = spec.chemistry_site_reports.clone();
        simulator.last_md_probe = spec.last_md_probe;
        if !spec.subsystem_states.is_empty() {
            simulator.subsystem_states = spec.subsystem_states.clone();
        } else {
            if simulator.subsystem_states.is_empty() {
                simulator.subsystem_states = Syn3ASubsystemPreset::all()
                    .iter()
                    .copied()
                    .map(WholeCellSubsystemState::new)
                    .collect();
            }
            simulator.refresh_subsystem_chemistry_state();
        }
        simulator.md_translation_scale = Self::finite_scale(
            spec.md_translation_scale
                .unwrap_or(simulator.md_translation_scale),
            1.0,
            0.70,
            1.45,
        );
        simulator.md_membrane_scale = Self::finite_scale(
            spec.md_membrane_scale
                .unwrap_or(simulator.md_membrane_scale),
            1.0,
            0.70,
            1.45,
        );

        // Preserve any explicit expression payload carried by the program spec before
        // regenerating it from the organism descriptor. This lets callers bootstrap
        // with concrete transcription-unit state and cached process scales when they
        // already have them, while still falling back to descriptor-driven rebuilds
        // when only coarse organism metadata is available.
        simulator.organism_expression = spec.organism_expression.unwrap_or_default();
        if simulator.organism_expression.transcription_units.is_empty() {
            simulator.refresh_organism_expression_state();
        }

        // Preserve any explicit assembly payload carried by the program spec before
        // falling back to derived seeding. This happens after expression refresh so
        // descriptor-driven process scales still shape any derived fallback targets.
        // Named complexes are the richest form because they preserve per-complex
        // state; aggregated complex_assembly is the next fallback when only channel
        // totals are available.
        simulator.named_complexes = spec.named_complexes.clone();
        simulator.complex_assembly = spec.complex_assembly.unwrap_or_default();
        if !simulator.named_complexes.is_empty() {
            let needs_named_complex_reset = simulator
                .organism_assets
                .as_ref()
                .map(|assets| simulator.named_complexes.len() != assets.complexes.len())
                .unwrap_or(false);
            if needs_named_complex_reset {
                simulator.initialize_named_complexes_state();
            }
            if let Some(assets) = simulator.organism_assets.as_ref() {
                simulator.complex_assembly =
                    simulator.aggregate_named_complex_assembly_state(assets);
            }
        } else if simulator.complex_assembly.total_complexes() <= 1.0e-6 {
            simulator.initialize_complex_assembly_state();
        }
        if simulator.organism_assets.is_none()
            && (spec_has_explicit_named_complexes || spec_has_explicit_complex_assembly)
        {
            simulator.supplement_bundleless_named_complex_carriers_from_assembly();
        }

        // Preserve explicit runtime chemistry state when the caller already has
        // concrete species and reaction counts. If either side is missing, fall back
        // to registry-driven bootstrap so the pair stays internally consistent.
        simulator.organism_species = spec.organism_species.unwrap_or_default();
        simulator.normalize_runtime_species_bulk_fields();
        simulator.organism_reactions = spec.organism_reactions.unwrap_or_default();
        if simulator.organism_species.is_empty() || simulator.organism_reactions.is_empty() {
            simulator.initialize_runtime_process_state();
        }

        // Preserve explicit multirate clocks when they are supplied with the program
        // spec. Unlike the descriptor-driven default path, these clocks may already
        // encode caller-managed due steps and run counts, so we only recompute them
        // when the program spec omits a scheduler payload entirely.
        if let Some(scheduler_state) = spec.scheduler_state {
            simulator.scheduler_state =
                Self::normalized_scheduler_state(&simulator.config, scheduler_state);
        } else {
            simulator.refresh_multirate_scheduler();
        }
        simulator.initialize_surrogate_pool_diagnostics();
        simulator.run_quantum_auto_discovery();
        simulator
    }

    /// Create a simulator from bundled Syn3A reference data.
    pub fn bundled_syn3a_reference() -> Result<Self, String> {
        bundled_syn3a_program_spec().map(Self::from_program_spec)
    }

    /// Create a simulator from a native organism bundle manifest path.
    pub fn from_bundle_manifest_path(manifest_path: &str) -> Result<Self, String> {
        compile_program_spec_from_bundle_manifest_path(manifest_path).map(Self::from_program_spec)
    }

    /// Create a simulator from a legacy-derived-asset bundle manifest path.
    pub fn from_legacy_bundle_manifest_path(manifest_path: &str) -> Result<Self, String> {
        compile_legacy_program_spec_from_bundle_manifest_path(manifest_path)
            .map(Self::from_program_spec)
    }

    /// Return the bundled Syn3A reference spec JSON.
    pub fn bundled_syn3a_reference_spec_json() -> &'static str {
        bundled_syn3a_program_spec_json()
    }

    /// Return the bundled Syn3A organism descriptor JSON.
    pub fn bundled_syn3a_organism_spec_json() -> &'static str {
        crate::whole_cell_data::bundled_syn3a_organism_spec_json()
    }

    /// Return the bundled Syn3A compiled genome asset package JSON.
    pub fn bundled_syn3a_genome_asset_package_json() -> Result<&'static str, String> {
        bundled_syn3a_genome_asset_package_json()
    }

    /// Return the bundled Syn3A compiled process registry JSON.
    pub fn bundled_syn3a_process_registry_json() -> Result<&'static str, String> {
        crate::whole_cell_data::bundled_syn3a_process_registry_json()
    }

    /// Create a simulator from a JSON-encoded whole-cell program spec.
    pub fn from_program_spec_json(spec_json: &str) -> Result<Self, String> {
        parse_program_spec_json(spec_json).map(Self::from_program_spec)
    }

    /// Create a simulator from a legacy JSON-encoded whole-cell program spec.
    pub fn from_legacy_program_spec_json(spec_json: &str) -> Result<Self, String> {
        parse_legacy_program_spec_json(spec_json).map(Self::from_program_spec)
    }

    fn checkpoint_payload(&self) -> WholeCellSavedState {
        let (ftsz, dnaa, active_ribosomes, active_rnap) = self.current_diagnostic_pool_summaries();
        WholeCellSavedState {
            program_name: self
                .program_name
                .clone()
                .or_else(|| Some("native_runtime".to_string())),
            contract: self.contract.clone(),
            provenance: {
                let mut provenance = self.provenance.clone();
                provenance.backend = Some(self.backend.as_str().to_string());
                provenance
            },
            organism_data_ref: self.organism_data_ref.clone(),
            organism_data: self.organism_data.clone(),
            organism_assets: self.organism_assets.clone(),
            organism_process_registry: self.organism_process_registry.clone(),
            organism_expression: self.organism_expression.clone(),
            chromosome_state: self.chromosome_state.clone(),
            membrane_division_state: self.membrane_division_state.clone(),
            organism_species: self.organism_species.clone(),
            organism_reactions: self.organism_reactions.clone(),
            complex_assembly: self.complex_assembly,
            named_complexes: self.named_complexes.clone(),
            scheduler_state: self.scheduler_state.clone(),
            config: self.config.clone(),
            core: WholeCellSavedCoreState {
                time_ms: self.time_ms,
                step_count: self.step_count,
                adp_mm: self.adp_mm,
                glucose_mm: self.glucose_mm,
                oxygen_mm: self.oxygen_mm,
                ftsz,
                dnaa,
                active_ribosomes,
                active_rnap,
                genome_bp: self.current_genome_bp(),
                replicated_bp: self.current_replicated_bp(),
                chromosome_separation_nm: self.current_chromosome_separation_nm(),
                radius_nm: self.current_radius_nm(),
                surface_area_nm2: self.current_surface_area_nm2(),
                volume_nm3: self.current_volume_nm3(),
                division_progress: self.current_division_progress(),
                metabolic_load: self.metabolic_load,
                quantum_profile: self.quantum_profile,
            },
            lattice: WholeCellLatticeState {
                atp: self.lattice.clone_species(IntracellularSpecies::ATP),
                amino_acids: self.lattice.clone_species(IntracellularSpecies::AminoAcids),
                nucleotides: self
                    .lattice
                    .clone_species(IntracellularSpecies::Nucleotides),
                membrane_precursors: self
                    .lattice
                    .clone_species(IntracellularSpecies::MembranePrecursors),
            },
            spatial_fields: Some(WholeCellSpatialFieldState {
                membrane_adjacency: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::MembraneAdjacency),
                septum_zone: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::SeptumZone),
                nucleoid_occupancy: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::NucleoidOccupancy),
                membrane_band_zone: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::MembraneBandZone),
                pole_zone: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::PoleZone),
            }),
            local_chemistry: self.chemistry_bridge.as_ref().map(|bridge| {
                let (x_dim, y_dim, z_dim) = bridge.lattice_shape();
                WholeCellLocalChemistrySpec {
                    x_dim,
                    y_dim,
                    z_dim,
                    voxel_size_au: bridge.voxel_size_au(),
                    use_gpu: bridge.use_gpu_backend(),
                    enable_default_syn3a_subsystems: false,
                    scheduled_subsystem_probes: self.scheduled_subsystem_probes.clone(),
                }
            }),
            chemistry_report: self.chemistry_report,
            chemistry_site_reports: self.chemistry_site_reports.clone(),
            last_md_probe: self.last_md_probe,
            scheduled_subsystem_probes: self.scheduled_subsystem_probes.clone(),
            subsystem_states: self.subsystem_states.clone(),
            md_translation_scale: self.md_translation_scale,
            md_membrane_scale: self.md_membrane_scale,
            stochastic_config: self.stochastic_config.clone(),
            stochastic_operon_states: self.stochastic_operon_states.clone(),
            stochastic_rng: self.stochastic_rng.clone(),
        }
    }

    /// Serialize the current simulator state into a restartable checkpoint payload.
    pub fn checkpoint_state(&self) -> Result<crate::whole_cell_data::WholeCellCheckpoint, String> {
        crate::whole_cell_data::finalize_saved_state(self.checkpoint_payload())
    }

    /// Serialize the current simulator state into a restartable JSON payload.
    pub fn save_state_json(&self) -> Result<String, String> {
        let saved = self.checkpoint_state()?;
        saved_state_to_json(&saved)
    }

    /// Restore a simulator from a structured whole-cell checkpoint payload.
    pub fn from_checkpoint_state(
        checkpoint: crate::whole_cell_data::WholeCellCheckpoint,
    ) -> Result<Self, String> {
        let mut simulator = Self::new(checkpoint.config.clone());
        simulator.restore_saved_state(checkpoint)?;
        Ok(simulator)
    }

    /// Restore a simulator from a JSON-encoded saved state.
    pub fn from_saved_state_json(state_json: &str) -> Result<Self, String> {
        let saved = parse_saved_state_json(state_json)?;
        Self::from_checkpoint_state(saved)
    }

    /// Restore a simulator from a legacy JSON-encoded saved state.
    pub fn from_legacy_saved_state_json(state_json: &str) -> Result<Self, String> {
        let saved = parse_legacy_saved_state_json(state_json)?;
        Self::from_checkpoint_state(saved)
    }
}
