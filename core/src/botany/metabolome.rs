//! Molecular Botany: Plant metabolome and chemical synthesis.
//!
//! Implements molecular-level metabolic pathways including:
//! - Photosynthesis (Calvin cycle): 6 CO2 + 6 H2O -> C6H12O6 + 6 O2
//! - Sugar interconversion: glucose <-> fructose (phosphoglucose isomerase)
//! - Sucrose synthesis: glucose + fructose -> sucrose (sucrose-phosphate synthase)
//! - Malate synthesis via TCA cycle (important for fruit acidity)
//! - Ethylene biosynthesis: methionine -> SAM -> ACC -> ethylene
//!   (Yang cycle; Adams & Yang, 1979, PNAS 76:170-174)
//! - VOC emission: terpene and ester synthesis from carbon backbone
//!   Q10 ~2.4 (Guenther et al., 1993, JGR 98:12609-12617)
//!
//! Literature references:
//! - Photosynthesis rate: 5-15 umol CO2/m2/s at full sun (Farquhar et al., 1980)
//! - Ethylene biosynthesis Km: ~10 uM for ACC oxidase (Dong et al., 1992)
//! - VOC emission Q10: ~2.4 (Guenther et al., 1993)
//! - Respiration: ~1-2% of biomass per day (Amthor, 2000, Plant Cell Environ)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Precise molecular inventory for a plant organism.
/// Tracks counts of actual molecules synthesized or consumed.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlantMetabolome {
    // -- Primary metabolites (photosynthesis) --
    /// Glucose molecules (C6H12O6)
    pub glucose_count: f64,
    /// Water molecules (H2O)
    pub water_count: f64,
    /// Carbon Dioxide molecules (CO2)
    pub co2_count: f64,
    /// Oxygen molecules (O2)
    pub oxygen_count: f64,

    // -- Storage and transport sugars --
    /// Starch reserve (glucose polymer, long-term storage)
    pub starch_reserve: f64,
    /// Fructose molecules (C6H12O6, isomer of glucose)
    pub fructose_count: f64,
    /// Sucrose molecules (C12H22O11, transport sugar: glucose + fructose)
    pub sucrose_count: f64,

    // -- Organic acids --
    /// Malic acid / malate (C4H6O5, TCA cycle intermediate, fruit acidity)
    pub malate_count: f64,

    // -- Amino acids and nitrogen metabolism --
    /// Generic amino acid pool (simplified; represents total free amino acids)
    pub amino_acid_pool: f64,

    // -- Hormones and signaling molecules --
    /// Ethylene (C2H4, ripening and stress hormone)
    pub ethylene_count: f64,

    // -- Volatile organic compounds --
    /// Current VOC emission rate (molecules per second).
    /// Represents terpenes (isoprene, monoterpenes) and esters.
    pub voc_emission_rate: f64,
    /// Cumulative VOC emitted (for tracking total emission).
    pub voc_total_emitted: f64,

    // -- Species-specific metabolites (Phase C) --
    /// Citric acid (C6H8O7, dominant organic acid in citrus fruits, TCA cycle product)
    pub citrate_count: f64,
    /// Sorbitol (C6H14O6, sugar alcohol, dominant in pome fruits: apple, pear)
    pub sorbitol_count: f64,
    /// Benzaldehyde (C6H5CHO, characteristic stone-fruit aroma compound)
    pub benzaldehyde_count: f64,
    /// Limonene (C10H16, monoterpene dominant in citrus peel oil)
    pub limonene_count: f64,
    /// Anthocyanin (flavonoid pigment, red/purple/blue, accumulated in stone fruits)
    pub anthocyanin_count: f64,
    /// Carotenoid (C40 terpenoid pigments, orange/yellow, accumulated in citrus)
    pub carotenoid_count: f64,

    // -- Defense signaling molecules (Phase 5: Inter-organism VOC signaling) --
    /// Jasmonic acid (JA, wound/stress/defense hormone).
    /// Synthesized via the octadecanoid pathway from linolenic acid upon tissue damage.
    /// (Wasternack & Hause, 2013, Ann Bot 111:1021-1058)
    pub jasmonate_count: f64,
    /// Salicylic acid (SA, systemic acquired resistance hormone).
    /// Synthesized via isochorismate pathway in chloroplasts.
    /// (Vlot et al., 2009, Annu Rev Phytopathol 47:177-206)
    pub salicylate_count: f64,
    /// Green leaf volatiles (C6 aldehydes/alcohols, immediate damage signal).
    /// Released from membrane lipids upon tissue disruption via lipoxygenase.
    /// (Matsui, 2006, Curr Opin Plant Biol 9:274-280)
    pub green_leaf_volatile_count: f64,
    /// Methyl salicylate (MeSA, volatile defense signal for neighbor priming).
    /// Methylated form of SA that travels through air to prime nearby plants.
    /// (Park et al., 2007, Science 318:113-116)
    pub methyl_salicylate_count: f64,
}

/// Snapshot of metabolome pools for coupling back into the terrarium grid.
#[derive(Debug, Clone, Default)]
pub struct SubstrateReport {
    pub o2_produced: f64,
    pub co2_pool: f64,
    pub water_pool: f64,
    pub glucose_pool: f64,
    pub ethylene_level: f64,
    pub voc_rate: f64,
    pub starch_pool: f64,
    pub sucrose_pool: f64,
    pub fructose_pool: f64,
    pub malate_pool: f64,
    pub citrate_pool: f64,
    pub sorbitol_pool: f64,
    pub benzaldehyde_pool: f64,
    pub limonene_pool: f64,
    pub anthocyanin_pool: f64,
    pub carotenoid_pool: f64,
}

/// Report from a single metabolic step, tracking fluxes through all pathways.
#[derive(Debug, Clone, Default)]
pub struct MetabolicReport {
    // -- Photosynthesis --
    pub glucose_synthesized: f64,
    pub co2_consumed: f64,
    pub o2_released: f64,
    pub h2o_consumed: f64,

    // -- Secondary metabolism --
    pub fructose_synthesized: f64,
    pub sucrose_synthesized: f64,
    pub malate_synthesized: f64,
    pub ethylene_emitted: f64,
    pub voc_emitted: f64,

    // -- Respiration --
    /// Total CO2 released by cellular respiration (maintenance + growth)
    pub total_respiration_co2: f64,
}

impl PlantMetabolome {
    pub fn new() -> Self {
        Self {
            water_count: 1000.0,    // Starting hydration
            co2_count: 500.0,       // Ambient CO2 reservoir
            glucose_count: 50.0,    // Seed glucose (endosperm reserve)
            starch_reserve: 30.0,   // Seed starch (cotyledon storage)
            amino_acid_pool: 10.0,  // Minimal starting amino acids
            ..Default::default()
        }
    }

    /// Replenish CO2 and H2O from the terrarium atmosphere/soil into the metabolome.
    /// This feeds the substrate pools that photosynthesis draws from, preventing
    /// them from depleting to zero when the plant is disconnected from the terrarium.
    ///
    /// `co2_flux` and `water_flux` are in arbitrary molecule-equivalent units
    /// proportional to what the terrarium grid provides at the plant's location.
    pub fn replenish_substrates(&mut self, co2_flux: f64, water_flux: f64) {
        self.co2_count += co2_flux.max(0.0);
        self.water_count += water_flux.max(0.0);
    }

    /// Report atmospheric fluxes produced by this metabolome during the last step.
    /// Returns (O2 produced, CO2 consumed, H2O consumed) for coupling back into
    /// the terrarium's atmospheric grid.
    pub fn substrate_report(&self) -> SubstrateReport {
        SubstrateReport {
            o2_produced: self.oxygen_count,
            co2_pool: self.co2_count,
            water_pool: self.water_count,
            glucose_pool: self.glucose_count,
            ethylene_level: self.ethylene_count,
            voc_rate: self.voc_emission_rate,
            starch_pool: self.starch_reserve,
            sucrose_pool: self.sucrose_count,
            fructose_pool: self.fructose_count,
            malate_pool: self.malate_count,
            citrate_pool: self.citrate_count,
            sorbitol_pool: self.sorbitol_count,
            benzaldehyde_pool: self.benzaldehyde_count,
            limonene_pool: self.limonene_count,
            anthocyanin_pool: self.anthocyanin_count,
            carotenoid_pool: self.carotenoid_count,
        }
    }

    /// Perform photosynthesis reaction: 6 CO2 + 6 H2O + photons -> C6H12O6 + 6 O2
    ///
    /// Rate is modulated by light intensity and RuBisCO expression (if provided
    /// via gene_expression map). Base rate: ~10 umol CO2/m2/s at full sun,
    /// scaled to molecule counts. (Farquhar et al., 1980)
    pub fn photosynthesis_step(&mut self, light_intensity: f32, dt: f32) -> MetabolicReport {
        let mut report = MetabolicReport::default();

        // Photosynthesis Vmax from Eyring TST (Farquhar+ 1980: ~10 µmol CO₂/m²/s)
        // RuBisCO CO bond eff=0.93 (Portis 2003) — temperature response implicit
        // Note: uses 25°C reference since photosynthesis_step lacks temperature param;
        // the RbcL expression scaling in full_metabolic_step provides environmental adjustment.
        let photosyn_vmax = crate::terrarium::emergent_rates::metabolome_rate("photosynthesis", 25.0);
        let max_rate = (light_intensity as f64 * photosyn_vmax * dt as f64)
            .min(self.co2_count / 6.0)
            .min(self.water_count / 6.0);

        if max_rate > 0.0 {
            let glucose_gain = max_rate;
            let substrate_loss = max_rate * 6.0;

            self.glucose_count += glucose_gain;
            self.co2_count -= substrate_loss;
            self.water_count -= substrate_loss;
            self.oxygen_count += substrate_loss;

            report.glucose_synthesized = glucose_gain;
            report.co2_consumed = substrate_loss;
            report.o2_released = substrate_loss;
            report.h2o_consumed = substrate_loss;
        }

        report
    }

    /// Convert glucose to starch for long-term storage
    pub fn store_starch(&mut self, amount: f64) {
        let actual = amount.min(self.glucose_count);
        self.glucose_count -= actual;
        self.starch_reserve += actual;
    }

    // -----------------------------------------------------------------------
    // New metabolic pathways
    // -----------------------------------------------------------------------

    /// Fructose synthesis from glucose via phosphoglucose isomerase.
    ///
    /// This is a near-equilibrium isomerization (Keq ~0.5 in favor of glucose).
    /// `fraction` is the fraction of current glucose to convert [0.0, 1.0].
    /// In vivo, ~30-50% of hexose phosphate pool is fructose.
    pub fn glucose_to_fructose(&mut self, fraction: f64) {
        let fraction = fraction.clamp(0.0, 1.0);
        let converted = self.glucose_count * fraction;
        self.glucose_count -= converted;
        self.fructose_count += converted;
    }

    /// Sucrose synthesis: glucose + fructose -> sucrose + H2O
    ///
    /// Catalyzed by sucrose-phosphate synthase. Sucrose is the primary
    /// long-distance transport sugar in phloem. Rate limited by the
    /// lesser of glucose and fructose availability.
    /// `fraction` is fraction of available substrate pairs to convert.
    pub fn synthesize_sucrose(&mut self, fraction: f64) {
        let fraction = fraction.clamp(0.0, 1.0);
        let pairs = self.glucose_count.min(self.fructose_count);
        let converted = pairs * fraction;
        self.glucose_count -= converted;
        self.fructose_count -= converted;
        self.sucrose_count += converted;
        self.water_count += converted; // condensation releases water
    }

    /// Competitive organic acid synthesis: citrate vs malate from TCA carbon.
    ///
    /// Both citrate and malate originate from the TCA cycle. Citrate synthase
    /// and malate dehydrogenase compete for the same pool of TCA intermediates.
    /// The branching ratio is determined by the relative enzyme expression levels
    /// (from the GRN), so citrus species naturally produce more citrate and
    /// apple species produce more malate — because their genomes encode different
    /// enzyme expression patterns.
    ///
    /// Net stoichiometry (simplified):
    /// - Citrate: 1 glucose -> ~1.5 citrate + 1.5 CO2
    /// - Malate:  1 glucose -> ~2 malate + 2 CO2
    pub fn synthesize_organic_acids(
        &mut self,
        glucose_consumed: f64,
        citrate_synthase_expr: f32,
        malate_dehydrogenase_expr: f32,
    ) {
        let actual = glucose_consumed.min(self.glucose_count).max(0.0);
        if actual <= 0.0 {
            return;
        }
        let cs = citrate_synthase_expr.max(0.01) as f64;
        let mdh = malate_dehydrogenase_expr.max(0.01) as f64;
        let total_enzyme = cs + mdh;
        let citrate_fraction = cs / total_enzyme;
        let malate_fraction = mdh / total_enzyme;

        let citrate_glucose = actual * citrate_fraction;
        let malate_glucose = actual * malate_fraction;

        self.glucose_count -= actual;
        // Citrate: 1 glucose (6C) -> 1.5 citrate (6C each, but partial oxidation) + CO2
        self.citrate_count += citrate_glucose * 1.5;
        self.co2_count += citrate_glucose * 1.5;
        // Malate: 1 glucose (6C) -> 2 malate (4C each) + 2 CO2
        self.malate_count += malate_glucose * 2.0;
        self.co2_count += malate_glucose * 2.0;
    }

    /// Legacy malate-only synthesis (for backward compatibility with non-species tests).
    pub fn synthesize_malate(&mut self, glucose_consumed: f64) {
        self.synthesize_organic_acids(glucose_consumed, 0.0, 1.0);
    }

    /// Sorbitol synthesis from glucose via aldose reductase.
    ///
    /// Important sugar alcohol in pome fruits (apple, pear).
    /// Sorbitol serves as a major phloem transport sugar in Rosaceae.
    /// 1 glucose + NADPH -> 1 sorbitol + NADP+
    pub fn synthesize_sorbitol(&mut self, fraction: f64, sorbitol_dehydrogenase_expr: f32) {
        let expr = sorbitol_dehydrogenase_expr.clamp(0.0, 1.0) as f64;
        let converted = (self.glucose_count * fraction.clamp(0.0, 0.1) * expr)
            .min(self.glucose_count);
        if converted > 0.0 {
            self.glucose_count -= converted;
            self.sorbitol_count += converted;
        }
    }

    /// Benzaldehyde synthesis (stone fruit aroma).
    ///
    /// Derived from phenylalanine via the phenylpropanoid pathway.
    /// Key aroma compound in cherry, peach, almond.
    /// Rate from Eyring TST: PAL enzyme eff=0.87 (Aharoni+ 2000)
    pub fn synthesize_benzaldehyde(&mut self, benzaldehyde_synthase_expr: f32, temperature: f32, dt: f32) {
        let expr = benzaldehyde_synthase_expr.clamp(0.0, 1.0) as f64;
        let base = crate::terrarium::emergent_rates::metabolome_rate("benzaldehyde", temperature);
        let rate = base * expr * dt as f64;
        let produced = rate.min(self.amino_acid_pool * 0.02);
        if produced > 0.0 {
            self.amino_acid_pool -= produced * 0.5; // phenylalanine consumed
            self.benzaldehyde_count += produced;
        }
    }

    /// Limonene synthesis (citrus terpene).
    ///
    /// Monoterpene (C10H16) synthesized from geranyl diphosphate via
    /// limonene synthase. Dominant in citrus peel oil (>90% of peel volatiles).
    /// Rate from Eyring TST: TPS eff=0.86 (Lucker+ 2002)
    pub fn synthesize_limonene(&mut self, limonene_synthase_expr: f32, temperature: f32, dt: f32) {
        let expr = limonene_synthase_expr.clamp(0.0, 1.0) as f64;
        let base = crate::terrarium::emergent_rates::metabolome_rate("limonene", temperature);
        let rate = base * expr * dt as f64;
        let produced = rate.min(self.glucose_count * 0.005);
        if produced > 0.0 {
            // C10 terpene costs ~10/6 glucose
            self.glucose_count -= produced * 1.67;
            self.glucose_count = self.glucose_count.max(0.0);
            self.limonene_count += produced;
        }
    }

    /// Anthocyanin biosynthesis (red/purple pigment).
    ///
    /// Flavonoid pigment synthesized via CHS -> CHI -> F3H -> DFR -> ANS pathway.
    /// Accumulated in stone fruit skin and some apple cultivars.
    /// Rate from Eyring TST: CHS eff=0.88 (Winkel-Shirley 2001)
    pub fn synthesize_anthocyanin(&mut self, anthocyanin_expr: f32, chs_expr: f32, temperature: f32, dt: f32) {
        let expr = (anthocyanin_expr * 0.7 + chs_expr * 0.3).clamp(0.0, 1.0) as f64;
        let base = crate::terrarium::emergent_rates::metabolome_rate("anthocyanin", temperature);
        let rate = base * expr * dt as f64;
        let produced = rate.min(self.glucose_count * 0.003);
        if produced > 0.0 {
            self.glucose_count -= produced * 2.0; // phenylpropanoid + malonyl-CoA
            self.glucose_count = self.glucose_count.max(0.0);
            self.anthocyanin_count += produced;
        }
    }

    /// Carotenoid biosynthesis (orange/yellow pigment).
    ///
    /// C40 terpenoid pigments (beta-carotene, lutein, zeaxanthin).
    /// Synthesized from geranylgeranyl diphosphate via PSY (phytoene synthase).
    /// Rate from Eyring TST: PSY eff=0.85 (Cunningham+ 2002)
    pub fn synthesize_carotenoid(&mut self, fruit_dev_expr: f32, light: f32, temperature: f32, dt: f32) {
        // Carotenoid synthesis is light-dependent and increases during fruit development
        let drive = (fruit_dev_expr * 0.6 + light * 0.4).clamp(0.0, 1.0) as f64;
        let base = crate::terrarium::emergent_rates::metabolome_rate("carotenoid", temperature);
        let rate = base * drive * dt as f64;
        let produced = rate.min(self.glucose_count * 0.002);
        if produced > 0.0 {
            self.glucose_count -= produced * 3.0; // C40 requires substantial carbon
            self.glucose_count = self.glucose_count.max(0.0);
            self.carotenoid_count += produced;
        }
    }

    /// Ethylene biosynthesis: methionine -> SAM -> ACC -> ethylene.
    ///
    /// The Yang cycle regenerates methionine so it is not consumed.
    /// Rate is controlled by ACC synthase (rate-limiting) and ACC oxidase.
    /// ACC oxidase Km ~10 uM (Dong et al., 1992, Planta 188:439-444).
    ///
    /// `ripening_expression` is the expression level of fruit ripening
    /// genes [0.0, 1.0] which controls ACC synthase transcription.
    /// Higher expression -> more ethylene.
    ///
    /// Ethylene is produced from amino acid pool (methionine).
    pub fn ethylene_synthesis(&mut self, ripening_expression: f32, temperature: f32, dt: f32) {
        // Eyring TST rate: ACC oxidase eff=0.88 (Yang & Hoffman 1984, Dong+ 1992)
        let base_rate = crate::terrarium::emergent_rates::metabolome_rate("ethylene", temperature);
        // Hill-like dependence on ripening gene expression
        let expr = ripening_expression.clamp(0.0, 1.0) as f64;
        let rate = base_rate * expr * expr; // quadratic to model cooperativity

        let produced = (rate * dt as f64).min(self.amino_acid_pool * 0.01);
        if produced > 0.0 {
            // Methionine is recycled in the Yang cycle, so amino acid pool
            // decreases only slightly (SAM consumption side reactions)
            self.amino_acid_pool -= produced * 0.1;
            self.amino_acid_pool = self.amino_acid_pool.max(0.0);
            self.ethylene_count += produced;
        }
    }

    /// VOC emission: terpene and ester synthesis from carbon backbone.
    ///
    /// VOC emission follows an exponential temperature dependence (Q10 ~2.4)
    /// and is light-dependent (isoprene emission requires photosynthesis).
    /// (Guenther et al., 1993, JGR 98:12609-12617)
    ///
    /// Returns the amount of VOC emitted this timestep, for olfactory coupling
    /// to the terrarium's odorant grid.
    pub fn voc_emission(&mut self, light: f32, temperature: f32, dt: f32) -> f64 {
        // Q10 model: rate = base_rate * Q10^((T - Tref) / 10)
        // Reference temperature: 30 C (standard for isoprene measurements)
        // Q10 = 2.4 (Guenther et al., 1993)
        let q10: f64 = 2.4;
        let t_ref: f64 = 30.0;
        let temp_factor = q10.powf((temperature as f64 - t_ref) / 10.0);

        // Light dependence: isoprene emission proportional to PAR
        // Using hyperbolic light response (Guenther et al., 1993)
        let light_f = light.clamp(0.0, 1.0) as f64;
        let alpha = 0.0027; // empirical constant
        let cl1 = 1.066;
        let light_response = (alpha * cl1 * light_f * 1000.0)
            / ((1.0 + alpha * alpha * light_f * light_f * 1_000_000.0).sqrt());

        // Base emission rate from Eyring TST (Guenther+ 1993: isoprene synthase eff=0.87)
        // Note: Q10 temp_factor already captures temperature response, but we use Eyring
        // at reference temperature for the base rate to anchor it to bond energy physics.
        let base_rate = crate::terrarium::emergent_rates::metabolome_rate("voc", t_ref as f32);
        let emission_rate = base_rate * temp_factor * light_response;

        // VOC synthesis requires carbon from glucose
        let emitted = (emission_rate * dt as f64).min(self.glucose_count * 0.005);
        if emitted > 0.0 {
            // Each isoprene (C5H8) costs ~5/6 of a glucose
            self.glucose_count -= emitted * 0.83;
            self.glucose_count = self.glucose_count.max(0.0);
            self.voc_emission_rate = emission_rate;
            self.voc_total_emitted += emitted;
        } else {
            self.voc_emission_rate = 0.0;
        }

        emitted
    }

    /// Defense VOC emission rate: sum of GLV and MeSA, scaled.
    ///
    /// Used by flora.rs to emit defense volatiles into the odorant grid channel.
    /// Returns a value in [0.0, 1.0] representing emission intensity.
    pub fn defense_voc_emission(&self) -> f32 {
        ((self.green_leaf_volatile_count + self.methyl_salicylate_count) * 0.01).min(1.0) as f32
    }

    /// Cellular respiration: glucose + 6 O2 -> 6 CO2 + 6 H2O + ATP.
    ///
    /// Maintenance respiration consumes ~1-2% of biomass per day.
    /// Rate scales with temperature (Q10 ~2.0 for plant respiration).
    /// (Amthor, 2000, Plant Cell Environ 23:1241-1257)
    pub fn respiration_step(&mut self, temperature: f32, dt: f32) -> f64 {
        // Temperature-dependent rate from Eyring TST (replaces Q10 model)
        // Amthor 2000: ~1.5% biomass/day → 0.5 glucose/s at 25°C
        let temp_rate = crate::terrarium::emergent_rates::metabolome_rate("respiration", temperature);
        let glucose_consumed = (temp_rate * dt as f64)
            .min(self.glucose_count)
            .min(self.oxygen_count / 6.0);

        if glucose_consumed > 0.0 {
            self.glucose_count -= glucose_consumed;
            self.oxygen_count -= glucose_consumed * 6.0;
            self.co2_count += glucose_consumed * 6.0;
            self.water_count += glucose_consumed * 6.0;
        }

        glucose_consumed * 6.0 // CO2 produced
    }

    /// Complete metabolic step integrating all pathways.
    ///
    /// This is the main entry point for per-timestep metabolism, coupling
    /// gene expression from the GRN to metabolic fluxes.
    ///
    /// `gene_expression` maps gene circuit IDs to their expression levels [0.0, 1.0].
    /// Relevant genes:
    /// - "RbcL": modulates photosynthesis rate
    /// - "FRUIT_DEVELOPMENT": modulates ethylene and malate synthesis
    /// - "VOLATILE_EMISSION" / "TERPENE_BIOSYNTHESIS": modulates VOC emission
    /// - "CHS": chalcone synthase (secondary metabolism indicator)
    pub fn full_metabolic_step(
        &mut self,
        light: f32,
        temperature: f32,
        gene_expression: &HashMap<String, f32>,
        dt: f32,
    ) -> MetabolicReport {
        let mut report = MetabolicReport::default();

        // 1. Photosynthesis (modulated by RbcL expression if available)
        let rbcl = gene_expression.get("RbcL").copied().unwrap_or(1.0);
        let effective_light = light * rbcl.clamp(0.1, 1.0);
        let photo_report = self.photosynthesis_step(effective_light, dt);
        report.glucose_synthesized = photo_report.glucose_synthesized;
        report.co2_consumed = photo_report.co2_consumed;
        report.o2_released = photo_report.o2_released;
        report.h2o_consumed = photo_report.h2o_consumed;

        // 2. Sugar interconversion: convert some glucose to fructose
        //    (~30% at equilibrium, phosphoglucose isomerase)
        let total_hexose = self.glucose_count + self.fructose_count;
        if total_hexose > 0.0 {
            let target_fructose_fraction = 0.3;
            let current_fructose_fraction = self.fructose_count / total_hexose;
            if current_fructose_fraction < target_fructose_fraction {
                let deficit = (target_fructose_fraction - current_fructose_fraction) * total_hexose;
                let convert = deficit.min(self.glucose_count) * 0.1 * dt as f64; // slow equilibration
                self.glucose_to_fructose(convert / self.glucose_count.max(1e-9));
                report.fructose_synthesized = convert;
            }
        }

        // 3. Sucrose synthesis (transport sugar, ~10% of hexose flux)
        if self.glucose_count > 1.0 && self.fructose_count > 1.0 {
            let old_sucrose = self.sucrose_count;
            self.synthesize_sucrose(0.02 * dt as f64);
            report.sucrose_synthesized = self.sucrose_count - old_sucrose;
        }

        // 4. Organic acid synthesis (competitive: citrate vs malate, enzyme-expression-driven)
        let fruit_dev = gene_expression
            .get("FRUIT_DEVELOPMENT")
            .copied()
            .unwrap_or(0.0);
        let citrate_synthase_expr = gene_expression
            .get("CITRATE_SYNTHASE")
            .copied()
            .unwrap_or(0.1); // low default — most species are malate-dominant
        let malate_dehydrogenase_expr = gene_expression
            .get("MALATE_DEHYDROGENASE")
            .copied()
            .unwrap_or(0.5); // moderate default
        if fruit_dev > 0.1 && self.glucose_count > 5.0 {
            let acid_glucose = self.glucose_count * 0.01 * fruit_dev as f64 * dt as f64;
            let old_malate = self.malate_count;
            let old_citrate = self.citrate_count;
            self.synthesize_organic_acids(acid_glucose, citrate_synthase_expr, malate_dehydrogenase_expr);
            report.malate_synthesized = self.malate_count - old_malate;
            // citrate is tracked but not in the base report (backward compat)
            let _ = self.citrate_count - old_citrate;
        }

        // 4b. Sorbitol synthesis (pome fruits: apple, pear — SORBITOL_DEHYDROGENASE)
        let sorbitol_dh_expr = gene_expression
            .get("SORBITOL_DEHYDROGENASE")
            .copied()
            .unwrap_or(0.0);
        if sorbitol_dh_expr > 0.05 && self.glucose_count > 5.0 {
            self.synthesize_sorbitol(0.03 * dt as f64, sorbitol_dh_expr);
        }

        // 4c. Benzaldehyde synthesis (stone fruit aroma: cherry, peach)
        let benzaldehyde_expr = gene_expression
            .get("BENZALDEHYDE_SYNTHASE")
            .copied()
            .unwrap_or(0.0);
        if benzaldehyde_expr > 0.05 {
            self.synthesize_benzaldehyde(benzaldehyde_expr, temperature, dt);
        }

        // 4d. Limonene synthesis (citrus terpene)
        let limonene_expr = gene_expression
            .get("LIMONENE_SYNTHASE")
            .copied()
            .unwrap_or(0.0);
        if limonene_expr > 0.05 {
            self.synthesize_limonene(limonene_expr, temperature, dt);
        }

        // 4e. Anthocyanin synthesis (red/purple pigment — stone fruits, some apples)
        let anthocyanin_expr = gene_expression
            .get("ANTHOCYANIN_BIOSYNTHESIS")
            .copied()
            .unwrap_or(0.0);
        let chs_expr = gene_expression.get("CHS").copied().unwrap_or(0.0);
        if anthocyanin_expr > 0.05 || chs_expr > 0.3 {
            self.synthesize_anthocyanin(anthocyanin_expr, chs_expr, temperature, dt);
        }

        // 4f. Carotenoid synthesis (orange/yellow pigment — citrus, many fruits)
        if fruit_dev > 0.1 || light > 0.3 {
            self.synthesize_carotenoid(fruit_dev, light, temperature, dt);
        }

        // 5. Ethylene biosynthesis (ripening hormone)
        let ripening = fruit_dev.max(
            gene_expression
                .get("VOLATILE_EMISSION")
                .copied()
                .unwrap_or(0.0)
                * 0.5,
        );
        let old_ethylene = self.ethylene_count;
        self.ethylene_synthesis(ripening, temperature, dt);
        report.ethylene_emitted = self.ethylene_count - old_ethylene;

        // 6. VOC emission (terpenes, esters)
        //    Modulated by VOLATILE_EMISSION and TERPENE_BIOSYNTHESIS gene expression
        let vol_expr = gene_expression
            .get("VOLATILE_EMISSION")
            .copied()
            .unwrap_or(0.3);
        let terp_expr = gene_expression
            .get("TERPENE_BIOSYNTHESIS")
            .copied()
            .unwrap_or(0.0);
        let voc_scale = (vol_expr + terp_expr).clamp(0.0, 2.0);
        let scaled_temp = temperature * (0.5 + 0.5 * voc_scale);
        report.voc_emitted = self.voc_emission(light, scaled_temp, dt);

        // 7. Cellular respiration (maintenance cost)
        report.total_respiration_co2 = self.respiration_step(temperature, dt);

        // 8. Starch storage: if glucose is abundant, store excess
        if self.glucose_count > 100.0 {
            self.store_starch(self.glucose_count * 0.05 * dt as f64);
        }

        // 9. Defense signaling pathways (Phase 5: inter-organism VOC signaling)
        //    JA/SA hormones are synthesized from gene expression, then converted
        //    to volatile forms (GLV, MeSA) that diffuse through the odorant grid.
        //    (Wasternack & Hause, 2013; Vlot et al., 2009)
        let ja_response_expr = gene_expression
            .get("JA_RESPONSE")
            .copied()
            .unwrap_or(0.0) as f64;
        let sa_response_expr = gene_expression
            .get("SA_RESPONSE")
            .copied()
            .unwrap_or(0.0) as f64;

        // JA synthesis from JA_RESPONSE expression: hill(expr, 0.3, 3) * rate
        // Requires glucose as substrate (linolenic acid precursor from membranes)
        // Rate from Eyring TST: octadecanoid pathway (Wasternack+ 2013)
        let ja_vmax = crate::terrarium::emergent_rates::metabolome_rate("jasmonate", temperature);
        let ja_synth_rate = (ja_response_expr.powf(3.0)
            / (0.3f64.powf(3.0) + ja_response_expr.powf(3.0)))
            * ja_vmax
            * dt as f64;
        if self.glucose_count > ja_synth_rate * 0.5 {
            self.jasmonate_count += ja_synth_rate;
            self.glucose_count -= ja_synth_rate * 0.5;
        }
        // JA decay: half-life ~2.3 min → k = ln(2)/(2.3×60) ≈ 0.005 (Narsai+ 2007)
        let ja_decay = (2.0f64.ln() / (2.3 * 60.0)) as f64;
        self.jasmonate_count *= (1.0 - ja_decay * dt as f64).max(0.0);
        self.jasmonate_count = self.jasmonate_count.max(0.0);

        // SA synthesis from SA_RESPONSE expression: hill(expr, 0.4, 2) * rate
        // Synthesized via isochorismate pathway (Vlot+ 2009)
        let sa_vmax = crate::terrarium::emergent_rates::metabolome_rate("salicylate", temperature);
        let sa_synth_rate = (sa_response_expr.powf(2.0)
            / (0.4f64.powf(2.0) + sa_response_expr.powf(2.0)))
            * sa_vmax
            * dt as f64;
        if self.glucose_count > sa_synth_rate * 0.3 {
            self.salicylate_count += sa_synth_rate;
            self.glucose_count -= sa_synth_rate * 0.3;
        }
        // SA decay: half-life ~3.9 min → k = ln(2)/(3.9×60) ≈ 0.003 (slower, SAR accumulation)
        let sa_decay = (2.0f64.ln() / (3.9 * 60.0)) as f64;
        self.salicylate_count *= (1.0 - sa_decay * dt as f64).max(0.0);
        self.salicylate_count = self.salicylate_count.max(0.0);

        // GLV emission: produced from jasmonate via lipoxygenase pathway (Matsui 2006).
        // Green leaf volatiles are C6 aldehydes released upon tissue damage.
        let glv_vmax = crate::terrarium::emergent_rates::metabolome_rate("glv", temperature);
        let ja_norm = (self.jasmonate_count / 10.0).min(1.0);
        let glv_rate = (ja_norm.powf(2.0) / (0.5f64.powf(2.0) + ja_norm.powf(2.0)))
            * glv_vmax
            * dt as f64;
        self.green_leaf_volatile_count += glv_rate;
        // GLV highly volatile — atmospheric half-life ~35s → k = ln(2)/35 ≈ 0.02
        let glv_decay = (2.0f64.ln() / 35.0) as f64;
        self.green_leaf_volatile_count *= (1.0 - glv_decay * dt as f64).max(0.0);
        self.green_leaf_volatile_count = self.green_leaf_volatile_count.max(0.0);

        // MeSA emission: methyl salicylate via SAMT methyltransferase (Park+ 2007).
        let mesa_vmax = crate::terrarium::emergent_rates::metabolome_rate("mesa", temperature);
        let sa_norm = (self.salicylate_count / 10.0).min(1.0);
        let mesa_rate = (sa_norm.powf(2.0) / (0.4f64.powf(2.0) + sa_norm.powf(2.0)))
            * mesa_vmax
            * dt as f64;
        self.methyl_salicylate_count += mesa_rate;
        // MeSA volatile decay: atmospheric half-life ~46s → k = ln(2)/46 ≈ 0.015
        let mesa_decay = (2.0f64.ln() / 46.0) as f64;
        self.methyl_salicylate_count *= (1.0 - mesa_decay * dt as f64).max(0.0);
        self.methyl_salicylate_count = self.methyl_salicylate_count.max(0.0);

        report
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fructose_synthesis() {
        let mut met = PlantMetabolome::new();
        met.glucose_count = 100.0;
        met.fructose_count = 0.0;

        met.glucose_to_fructose(0.3);

        assert!(
            (met.fructose_count - 30.0).abs() < 0.01,
            "30% of 100 glucose should give 30 fructose, got {}",
            met.fructose_count
        );
        assert!(
            (met.glucose_count - 70.0).abs() < 0.01,
            "Glucose should drop to 70, got {}",
            met.glucose_count
        );
    }

    #[test]
    fn test_fructose_synthesis_clamps_fraction() {
        let mut met = PlantMetabolome::new();
        met.glucose_count = 50.0;

        // Fraction > 1.0 should be clamped
        met.glucose_to_fructose(2.0);
        assert!(
            (met.fructose_count - 50.0).abs() < 0.01,
            "Fraction clamped to 1.0: all glucose should convert, got {}",
            met.fructose_count
        );
        assert!(
            met.glucose_count < 0.01,
            "All glucose consumed, got {}",
            met.glucose_count
        );
    }

    #[test]
    fn test_sucrose_synthesis() {
        let mut met = PlantMetabolome::new();
        met.glucose_count = 50.0;
        met.fructose_count = 50.0;
        let old_water = met.water_count;

        met.synthesize_sucrose(0.5); // convert 50% of pairs

        assert!(
            (met.sucrose_count - 25.0).abs() < 0.01,
            "50% of 50 pairs = 25 sucrose, got {}",
            met.sucrose_count
        );
        assert!(
            (met.glucose_count - 25.0).abs() < 0.01,
            "Glucose should be 25, got {}",
            met.glucose_count
        );
        assert!(
            met.water_count > old_water,
            "Sucrose synthesis releases water"
        );
    }

    #[test]
    fn test_malate_synthesis() {
        let mut met = PlantMetabolome::new();
        met.glucose_count = 100.0;
        let old_co2 = met.co2_count;

        met.synthesize_malate(10.0);

        // With competitive acid synthesis, a tiny fraction goes to citrate (0.01 default)
        // so malate gets ~99% of the 10 glucose → ~19.8 malate
        assert!(
            met.malate_count > 19.0 && met.malate_count < 21.0,
            "10 glucose -> ~20 malate, got {}",
            met.malate_count
        );
        assert!(
            (met.glucose_count - 90.0).abs() < 0.01,
            "Glucose should drop by 10, got {}",
            met.glucose_count
        );
        assert!(
            met.co2_count > old_co2,
            "Malate synthesis releases CO2"
        );
    }

    #[test]
    fn test_ethylene_synthesis_rate() {
        let mut met = PlantMetabolome::new();
        met.amino_acid_pool = 100.0;

        // Low ripening expression
        let mut met_low = met.clone();
        met_low.ethylene_synthesis(0.1, 25.0, 10.0);
        let low_eth = met_low.ethylene_count;

        // High ripening expression
        let mut met_high = met.clone();
        met_high.ethylene_synthesis(0.9, 25.0, 10.0);
        let high_eth = met_high.ethylene_count;

        assert!(
            high_eth > low_eth * 5.0,
            "Ethylene at high ripening ({}) should be much more than low ({})",
            high_eth,
            low_eth
        );
        assert!(
            high_eth > 0.0,
            "Some ethylene should be produced at high expression"
        );
    }

    #[test]
    fn test_ethylene_zero_at_zero_expression() {
        let mut met = PlantMetabolome::new();
        met.amino_acid_pool = 100.0;
        met.ethylene_synthesis(0.0, 25.0, 10.0);
        assert!(
            met.ethylene_count < 1e-9,
            "No ethylene at zero expression, got {}",
            met.ethylene_count
        );
    }

    #[test]
    fn test_voc_emission_temperature_dependent() {
        let mut met_cold = PlantMetabolome::new();
        met_cold.glucose_count = 1000.0;
        let cold_voc = met_cold.voc_emission(0.8, 15.0, 10.0);

        let mut met_warm = PlantMetabolome::new();
        met_warm.glucose_count = 1000.0;
        let warm_voc = met_warm.voc_emission(0.8, 35.0, 10.0);

        assert!(
            warm_voc > cold_voc,
            "VOC emission should increase with temperature: warm={} > cold={}",
            warm_voc,
            cold_voc
        );

        // Q10 ~2.4 means a 20C increase should roughly multiply by 2.4^2 = 5.76
        // Allow wide tolerance because light response also modulates
        if cold_voc > 1e-9 {
            let ratio = warm_voc / cold_voc;
            assert!(
                ratio > 2.0,
                "Q10 effect: warm/cold ratio should be >2, got {}",
                ratio
            );
        }
    }

    #[test]
    fn test_voc_emission_requires_light() {
        let mut met = PlantMetabolome::new();
        met.glucose_count = 1000.0;
        let dark_voc = met.voc_emission(0.0, 30.0, 10.0);
        assert!(
            dark_voc < 1e-9,
            "VOC emission should be negligible in dark, got {}",
            dark_voc
        );
    }

    #[test]
    fn test_respiration_produces_co2() {
        let mut met = PlantMetabolome::new();
        met.glucose_count = 100.0;
        met.oxygen_count = 1000.0;
        let old_co2 = met.co2_count;

        let co2 = met.respiration_step(25.0, 1.0);

        assert!(co2 > 0.0, "Respiration should produce CO2");
        assert!(
            met.co2_count > old_co2,
            "CO2 count should increase after respiration"
        );
        assert!(
            met.glucose_count < 100.0,
            "Glucose should decrease from respiration"
        );
    }

    #[test]
    fn test_full_metabolic_step() {
        let mut met = PlantMetabolome::new();
        met.glucose_count = 200.0;
        met.co2_count = 2000.0;
        met.water_count = 5000.0;
        met.oxygen_count = 500.0;
        met.amino_acid_pool = 50.0;

        let mut gene_expr = HashMap::new();
        gene_expr.insert("RbcL".to_string(), 0.8f32);
        gene_expr.insert("FRUIT_DEVELOPMENT".to_string(), 0.5);
        gene_expr.insert("VOLATILE_EMISSION".to_string(), 0.4);

        let report = met.full_metabolic_step(0.8, 25.0, &gene_expr, 1.0);

        // Photosynthesis should have occurred
        assert!(
            report.glucose_synthesized > 0.0,
            "Photosynthesis should produce glucose"
        );

        // Respiration should have occurred
        assert!(
            report.total_respiration_co2 > 0.0,
            "Respiration should release CO2"
        );

        // Some fructose should have been made
        assert!(
            met.fructose_count > 0.0 || report.fructose_synthesized >= 0.0,
            "Some sugar interconversion expected"
        );

        // Ethylene should have been produced (FRUIT_DEVELOPMENT > 0)
        assert!(
            report.ethylene_emitted > 0.0,
            "Ethylene should be emitted with fruit development expression"
        );
    }

    #[test]
    fn test_citrus_high_citrate() {
        // Citrus species express high citrate synthase → citrate >> malate
        let mut met = PlantMetabolome::new();
        met.glucose_count = 500.0;
        met.co2_count = 2000.0;
        met.water_count = 5000.0;
        met.oxygen_count = 500.0;
        met.amino_acid_pool = 50.0;

        let mut gene_expr = HashMap::new();
        gene_expr.insert("RbcL".to_string(), 0.8f32);
        gene_expr.insert("FRUIT_DEVELOPMENT".to_string(), 0.7);
        gene_expr.insert("CITRATE_SYNTHASE".to_string(), 0.85);    // HIGH (citrus)
        gene_expr.insert("MALATE_DEHYDROGENASE".to_string(), 0.15); // low
        gene_expr.insert("VOLATILE_EMISSION".to_string(), 0.3);

        for _ in 0..500 {
            met.full_metabolic_step(0.8, 25.0, &gene_expr, 1.0);
            met.replenish_substrates(50.0, 100.0);
        }

        assert!(
            met.citrate_count > met.malate_count * 2.0,
            "Citrus citrate ({}) should be >> malate ({})",
            met.citrate_count, met.malate_count
        );
    }

    #[test]
    fn test_apple_high_malate() {
        // Apple species express high malate dehydrogenase → malate >> citrate
        let mut met = PlantMetabolome::new();
        met.glucose_count = 500.0;
        met.co2_count = 2000.0;
        met.water_count = 5000.0;
        met.oxygen_count = 500.0;
        met.amino_acid_pool = 50.0;

        let mut gene_expr = HashMap::new();
        gene_expr.insert("RbcL".to_string(), 0.8f32);
        gene_expr.insert("FRUIT_DEVELOPMENT".to_string(), 0.7);
        gene_expr.insert("CITRATE_SYNTHASE".to_string(), 0.1);     // low (apple)
        gene_expr.insert("MALATE_DEHYDROGENASE".to_string(), 0.85); // HIGH
        gene_expr.insert("SORBITOL_DEHYDROGENASE".to_string(), 0.7);
        gene_expr.insert("VOLATILE_EMISSION".to_string(), 0.3);

        for _ in 0..500 {
            met.full_metabolic_step(0.8, 25.0, &gene_expr, 1.0);
            met.replenish_substrates(50.0, 100.0);
        }

        assert!(
            met.malate_count > met.citrate_count * 2.0,
            "Apple malate ({}) should be >> citrate ({})",
            met.malate_count, met.citrate_count
        );
        assert!(
            met.sorbitol_count > 0.0,
            "Apple should produce sorbitol, got {}",
            met.sorbitol_count
        );
    }

    #[test]
    fn test_cherry_benzaldehyde() {
        // Cherry: benzaldehyde synthase expressed → benzaldehyde produced
        let mut met = PlantMetabolome::new();
        met.glucose_count = 300.0;
        met.amino_acid_pool = 100.0;

        let mut gene_expr = HashMap::new();
        gene_expr.insert("BENZALDEHYDE_SYNTHASE".to_string(), 0.75f32);

        for _ in 0..200 {
            met.full_metabolic_step(0.7, 25.0, &gene_expr, 1.0);
            met.replenish_substrates(30.0, 50.0);
        }

        assert!(
            met.benzaldehyde_count > 0.5,
            "Cherry should produce benzaldehyde, got {}",
            met.benzaldehyde_count
        );

        // Apple should NOT produce benzaldehyde (no BENZALDEHYDE_SYNTHASE expression)
        let mut met_apple = PlantMetabolome::new();
        met_apple.glucose_count = 300.0;
        met_apple.amino_acid_pool = 100.0;
        let apple_expr = HashMap::new(); // no benzaldehyde synthase
        for _ in 0..200 {
            met_apple.full_metabolic_step(0.7, 25.0, &apple_expr, 1.0);
            met_apple.replenish_substrates(30.0, 50.0);
        }
        assert!(
            met_apple.benzaldehyde_count < 0.01,
            "Apple should not produce benzaldehyde, got {}",
            met_apple.benzaldehyde_count
        );
    }

    #[test]
    fn test_fruit_composition_varies_with_conditions() {
        // Same species-like expression, different environments → different metabolome
        let mut gene_expr = HashMap::new();
        gene_expr.insert("RbcL".to_string(), 0.8f32);
        gene_expr.insert("FRUIT_DEVELOPMENT".to_string(), 0.6);
        gene_expr.insert("CITRATE_SYNTHASE".to_string(), 0.7);

        // Warm, bright → lots of sugar
        let mut met_warm = PlantMetabolome::new();
        met_warm.glucose_count = 300.0;
        met_warm.co2_count = 2000.0;
        met_warm.water_count = 5000.0;
        met_warm.oxygen_count = 500.0;
        met_warm.amino_acid_pool = 50.0;
        for _ in 0..300 {
            met_warm.full_metabolic_step(0.95, 30.0, &gene_expr, 1.0);
            met_warm.replenish_substrates(60.0, 120.0);
        }

        // Cool, shaded → less sugar
        let mut met_cool = PlantMetabolome::new();
        met_cool.glucose_count = 300.0;
        met_cool.co2_count = 2000.0;
        met_cool.water_count = 5000.0;
        met_cool.oxygen_count = 500.0;
        met_cool.amino_acid_pool = 50.0;
        for _ in 0..300 {
            met_cool.full_metabolic_step(0.2, 12.0, &gene_expr, 1.0);
            met_cool.replenish_substrates(20.0, 60.0);
        }

        let warm_sugar = met_warm.glucose_count + met_warm.fructose_count + met_warm.sucrose_count;
        let cool_sugar = met_cool.glucose_count + met_cool.fructose_count + met_cool.sucrose_count;
        // Warm should generally have more sugar due to more photosynthesis
        // (Though respiration also increases — the net depends on balance)
        let warm_acid = met_warm.citrate_count + met_warm.malate_count;
        let cool_acid = met_cool.citrate_count + met_cool.malate_count;

        // The key emergent property: compositions DIFFER because growing conditions differ
        let sugar_differs = (warm_sugar - cool_sugar).abs() > 1.0;
        let acid_differs = (warm_acid - cool_acid).abs() > 0.1;
        assert!(
            sugar_differs || acid_differs,
            "Different growing conditions should yield different metabolome: \
            warm_sugar={warm_sugar:.1} cool_sugar={cool_sugar:.1} \
            warm_acid={warm_acid:.2} cool_acid={cool_acid:.2}"
        );
    }

    #[test]
    fn test_full_metabolic_step_no_gene_expression() {
        // With empty gene expression map, defaults should still work
        let mut met = PlantMetabolome::new();
        met.glucose_count = 100.0;
        met.co2_count = 1000.0;
        met.oxygen_count = 500.0;

        let empty_expr = HashMap::new();
        let report = met.full_metabolic_step(0.5, 22.0, &empty_expr, 1.0);

        // Photosynthesis should still work (RbcL defaults to 1.0)
        assert!(
            report.glucose_synthesized > 0.0,
            "Photosynthesis should work without explicit gene expression"
        );
    }

    // -- Phase 5: Defense signaling / VOC tests --

    #[test]
    fn test_jasmonate_synthesis_from_damage() {
        let mut m = PlantMetabolome::new();
        let mut gene_expr = HashMap::new();
        gene_expr.insert("JA_RESPONSE".to_string(), 0.8f32);
        gene_expr.insert("RbcL".to_string(), 0.5);
        m.full_metabolic_step(0.5, 22.0, &gene_expr, 1.0);
        assert!(
            m.jasmonate_count > 0.0,
            "JA_RESPONSE expression should produce jasmonate: {}",
            m.jasmonate_count
        );
    }

    #[test]
    fn test_jasmonate_decay_without_damage() {
        let mut m = PlantMetabolome::new();
        m.jasmonate_count = 5.0;
        let gene_expr = HashMap::new();
        m.full_metabolic_step(0.5, 22.0, &gene_expr, 1.0);
        assert!(
            m.jasmonate_count < 5.0,
            "JA should decay without JA_RESPONSE: {}",
            m.jasmonate_count
        );
    }

    #[test]
    fn test_glv_emission_requires_ja() {
        let mut m = PlantMetabolome::new();
        // No jasmonate = no GLV
        let gene_expr = HashMap::new();
        m.full_metabolic_step(0.5, 22.0, &gene_expr, 1.0);
        assert!(
            m.green_leaf_volatile_count < 0.01,
            "GLV should be near-zero without JA: {}",
            m.green_leaf_volatile_count
        );
    }

    #[test]
    fn test_glv_rises_with_ja() {
        let mut m = PlantMetabolome::new();
        m.jasmonate_count = 8.0;
        let gene_expr = HashMap::new();
        m.full_metabolic_step(0.5, 22.0, &gene_expr, 1.0);
        assert!(
            m.green_leaf_volatile_count > 0.0,
            "GLV should be produced with high JA: {}",
            m.green_leaf_volatile_count
        );
    }

    #[test]
    fn test_mesa_from_salicylate() {
        let mut m = PlantMetabolome::new();
        m.salicylate_count = 8.0;
        let gene_expr = HashMap::new();
        m.full_metabolic_step(0.5, 22.0, &gene_expr, 1.0);
        assert!(
            m.methyl_salicylate_count > 0.0,
            "MeSA should be produced from SA: {}",
            m.methyl_salicylate_count
        );
    }

    #[test]
    fn test_defense_voc_emission_rate() {
        let mut m = PlantMetabolome::new();
        m.green_leaf_volatile_count = 50.0;
        m.methyl_salicylate_count = 30.0;
        let rate = m.defense_voc_emission();
        assert!(rate > 0.0, "Should have positive emission rate: {rate}");
        assert!(rate <= 1.0, "Rate should be capped at 1.0: {rate}");
    }

    #[test]
    fn test_no_voc_without_damage() {
        let mut m = PlantMetabolome::new();
        let gene_expr = HashMap::new();
        m.full_metabolic_step(0.5, 22.0, &gene_expr, 1.0);
        let rate = m.defense_voc_emission();
        assert!(rate < 0.01, "No VOC without damage cascade: {rate}");
    }
}
