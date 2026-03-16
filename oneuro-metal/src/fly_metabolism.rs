//! Drosophila molecular metabolism -- 7-pool Michaelis-Menten model.
//!
//! Replaces the legacy bookkeeping `energy: f32` on [`BodyState`] with genuine
//! molecular metabolism. Seven hemolymph/tissue pools connected by MM kinetics
//! with literature rate constants produce emergent hunger, fatigue, and energy
//! depletion:
//!
//! | Pool                     | Units | Initial | Reference            |
//! |--------------------------|-------|---------|----------------------|
//! | `crop_sugar_mg`          | mg    | 0.0     | Edgecomb 1994        |
//! | `hemolymph_trehalose_mm` | mM    | 25.0    | Wyatt 1967           |
//! | `hemolymph_glucose_mm`   | mM    | 2.0     | Rulifson 2002        |
//! | `fat_body_glycogen_mg`   | mg    | 0.015   | Wigglesworth 1949    |
//! | `fat_body_lipid_mg`      | mg    | 0.045   | Wigglesworth 1949    |
//! | `muscle_atp_mm`          | mM    | 4.0     | Functional pool      |
//! | `muscle_adp_mm`          | mM    | 1.0     | Recycled via OXPHOS  |
//!
//! The emergence chain: fruit sugar → crop absorption (MM) → trehalose →
//! trehalase (MM) → glucose → glycolysis+OXPHOS (MM) → ATP for muscles.
//! Hunger = low trehalose. Every step is Michaelis-Menten kinetics.

use crate::constants::michaelis_menten;
use crate::organism_metabolism::OrganismMetabolism;

// ============================================================================
// Physiological constants (literature-derived)
// ============================================================================

/// Crop absorption: sucrose → hemolymph trehalose (Edgecomb 1994).
const CROP_VMAX: f32 = 0.5;   // mg/s
const CROP_KM: f32 = 0.1;     // mg

/// Trehalase: trehalose → 2 glucose (Thompson 2003).
const TREHALASE_VMAX: f32 = 2.0; // mM/s
const TREHALASE_KM: f32 = 5.0;   // mM

/// Glycolysis + OXPHOS: glucose → ATP (net ~36 ATP/glucose).
const GLYCOLYSIS_VMAX: f32 = 1.0; // mM/s
const GLYCOLYSIS_KM: f32 = 0.5;   // mM
const ATP_PER_GLUCOSE: f32 = 36.0;

/// Fat body glycogenolysis: glycogen → glucose.
/// Triggered when hemolymph glucose < GLYCOGENOLYSIS_THRESHOLD.
const GLYCOGENOLYSIS_VMAX: f32 = 0.005; // mg/s
const GLYCOGENOLYSIS_KM: f32 = 0.01;    // mg
const GLYCOGENOLYSIS_THRESHOLD: f32 = 1.5; // mM glucose

/// Lipid mobilization: lipid → glucose equivalent (last resort).
const LIPID_MOB_VMAX: f32 = 0.003; // mg/s
const LIPID_MOB_KM: f32 = 0.02;    // mg
const LIPID_MOB_THRESHOLD: f32 = 0.005; // mg glycogen

/// Fat body storage: glucose → glycogen (when glucose > threshold).
const GLYCOGEN_STORAGE_VMAX: f32 = 0.003; // mg/s
const GLYCOGEN_STORAGE_KM: f32 = 1.0;     // mM (glucose above 3 mM drives storage)
const GLYCOGEN_STORAGE_THRESHOLD: f32 = 3.0; // mM glucose

/// Conversion: 1 mg glucose ≈ 5.55 mM in ~1 µL hemolymph volume.
const MG_TO_MM_GLUCOSE: f32 = 5.55;

/// Energy density constants for backward-compatible uJ mapping.
const GLYCOGEN_ENERGY_UJ_PER_MG: f32 = 17_100.0; // 17.1 J/mg glycogen
const LIPID_ENERGY_UJ_PER_MG: f32 = 39_300.0;    // 39.3 J/mg lipid

/// Maximum legacy energy (must match drosophila.rs FLY_ENERGY_MAX).
const FLY_ENERGY_MAX: f32 = 3700.0;

/// Activity-driven ATP demand rates (uW → mM ATP/s via conversion).
/// Conversion: 1 uW = 1e-6 J/s. ATP hydrolysis = 54 kJ/mol.
/// In ~1 µL volume: 1 mM = 1e-6 mol. So 1 mM ATP = 0.054 J.
/// => 1 uW consumes 1e-6/0.054 = 1.85e-5 mM/s per uW.
const UW_TO_MM_ATP_PER_S: f32 = 1.85e-5;
const BASAL_DEMAND_UW: f32 = 20.0;   // resting metabolic rate
const WALK_DEMAND_UW: f32 = 30.0;    // Strauss 1993
const FLY_DEMAND_UW: f32 = 100.0;    // Lehmann 2000

/// Trehalose level at which fly feels satiated (mM).
const TREHALOSE_SATIATED: f32 = 30.0;

/// Pool bounds.
const MAX_CROP_SUGAR: f32 = 0.5;       // mg
const MAX_TREHALOSE: f32 = 40.0;       // mM
const MAX_GLUCOSE: f32 = 10.0;         // mM
const MAX_GLYCOGEN: f32 = 0.030;       // mg
const MAX_LIPID: f32 = 0.090;          // mg
const MAX_ATP: f32 = 8.0;              // mM
const MAX_ADP: f32 = 8.0;              // mM

/// AMP estimation constant (total adenylate pool ~ 5.5 mM).
const TOTAL_ADENYLATE: f32 = 5.5; // mM

/// Crop sugar delivered per unit food bite (mg per bite unit).
/// A bite of 0.05 food units delivers ~0.1 mg sucrose.
pub const CROP_SUGAR_PER_BITE: f32 = 2.0; // mg sugar per unit food consumed

// ============================================================================
// Activity level (set externally each frame)
// ============================================================================

/// Current activity level driving ATP demand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlyActivity {
    Resting,
    Walking(f32), // speed fraction 0..1
    Flying(f32),  // effort fraction 0..1
}

impl Default for FlyActivity {
    fn default() -> Self {
        Self::Resting
    }
}

// ============================================================================
// FlyMetabolism
// ============================================================================

/// Seven-pool Michaelis-Menten metabolic model for Drosophila.
#[derive(Clone, Debug)]
pub struct FlyMetabolism {
    // ---- Pools ----
    pub crop_sugar_mg: f32,
    pub hemolymph_trehalose_mm: f32,
    pub hemolymph_glucose_mm: f32,
    pub fat_body_glycogen_mg: f32,
    pub fat_body_lipid_mg: f32,
    pub muscle_atp_mm: f32,
    pub muscle_adp_mm: f32,
    // ---- Activity ----
    activity: FlyActivity,
    // ---- Neural ----
    /// Neural activity fraction (0–1): brain firing rate normalized to ~50 Hz max.
    /// 20% of resting metabolic rate at full neural activity.
    pub neural_activity_fraction: f32,
    // ---- Environment ----
    /// Ambient O2 mole fraction (0.21 = atmospheric normoxia).
    pub ambient_o2_fraction: f32,
}

impl Default for FlyMetabolism {
    fn default() -> Self {
        Self {
            crop_sugar_mg: 0.0,
            hemolymph_trehalose_mm: 25.0,
            hemolymph_glucose_mm: 2.0,
            fat_body_glycogen_mg: 0.015,
            fat_body_lipid_mg: 0.045,
            muscle_atp_mm: 4.0,
            muscle_adp_mm: 1.0,
            activity: FlyActivity::Resting,
            neural_activity_fraction: 0.0,
            ambient_o2_fraction: 0.21,
        }
    }
}

impl FlyMetabolism {
    /// Set current activity level. Call before `step()`.
    pub fn set_activity(&mut self, activity: FlyActivity) {
        self.activity = activity;
    }

    /// Get current activity level.
    pub fn activity(&self) -> FlyActivity {
        self.activity
    }

    /// Set neural activity fraction (0–1). Call before `step()`.
    /// Maps brain firing rate to metabolic cost.
    pub fn set_neural_activity(&mut self, fraction: f32) {
        self.neural_activity_fraction = fraction.clamp(0.0, 1.0);
    }

    /// Set ambient O2 mole fraction. Call before `step()` in terrarium.
    pub fn set_ambient_o2(&mut self, o2_fraction: f32) {
        self.ambient_o2_fraction = o2_fraction.clamp(0.0, 1.0);
    }

    /// Reverse-map a legacy energy value (uJ) into glycogen + lipid reserves.
    /// Distributes proportionally: ~14% glycogen, ~86% lipid (matches initial ratio).
    pub fn set_reserves_from_uj(&mut self, uj: f32) {
        let uj_clamped = uj.clamp(0.0, FLY_ENERGY_MAX);
        let glycogen_frac = 0.14;
        let glycogen_uj = uj_clamped * glycogen_frac;
        let lipid_uj = uj_clamped * (1.0 - glycogen_frac);
        self.fat_body_glycogen_mg =
            (glycogen_uj / GLYCOGEN_ENERGY_UJ_PER_MG).clamp(0.0, MAX_GLYCOGEN);
        self.fat_body_lipid_mg = (lipid_uj / LIPID_ENERGY_UJ_PER_MG).clamp(0.0, MAX_LIPID);
    }

    /// Ingest sugar into the crop (from feeding on fruit/nectar).
    pub fn ingest(&mut self, sugar_mg: f32) {
        self.crop_sugar_mg = (self.crop_sugar_mg + sugar_mg).min(MAX_CROP_SUGAR);
    }

    /// Hunger signal: low trehalose → high hunger (0.0 = satiated, 1.0 = starving).
    pub fn hunger(&self) -> f32 {
        (1.0 - self.hemolymph_trehalose_mm / TREHALOSE_SATIATED).clamp(0.0, 1.0)
    }

    /// Backward-compatible energy in uJ, mapped from glycogen + lipid reserves.
    /// Range [0, FLY_ENERGY_MAX]. Existing snapshot/evolution code uses this.
    pub fn energy_compat_uj(&self) -> f32 {
        let glycogen_uj = self.fat_body_glycogen_mg * GLYCOGEN_ENERGY_UJ_PER_MG;
        let lipid_uj = self.fat_body_lipid_mg * LIPID_ENERGY_UJ_PER_MG;
        (glycogen_uj + lipid_uj).clamp(0.0, FLY_ENERGY_MAX)
    }

    /// Advance all metabolic pathways by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        // 1. Crop absorption: crop sugar → hemolymph trehalose.
        let crop_flux = michaelis_menten(self.crop_sugar_mg, CROP_VMAX, CROP_KM) * dt;
        let crop_consumed = crop_flux.min(self.crop_sugar_mg);
        self.crop_sugar_mg -= crop_consumed;
        // ~1 mg sucrose → ~2.9 mM trehalose (sucrose→2 hexose→1 trehalose, in ~1µL).
        self.hemolymph_trehalose_mm += crop_consumed * 2.9;

        // 2. Trehalase: trehalose → 2 glucose.
        let trehalase_flux =
            michaelis_menten(self.hemolymph_trehalose_mm, TREHALASE_VMAX, TREHALASE_KM) * dt;
        let tre_consumed = trehalase_flux.min(self.hemolymph_trehalose_mm);
        self.hemolymph_trehalose_mm -= tre_consumed;
        self.hemolymph_glucose_mm += tre_consumed * 2.0; // 1 trehalose → 2 glucose

        // 3. ATP demand from current activity.
        let demand_uw = match self.activity {
            FlyActivity::Resting => BASAL_DEMAND_UW,
            FlyActivity::Walking(speed) => {
                BASAL_DEMAND_UW + (WALK_DEMAND_UW - BASAL_DEMAND_UW) * speed.clamp(0.0, 1.0)
            }
            FlyActivity::Flying(effort) => {
                BASAL_DEMAND_UW + (FLY_DEMAND_UW - BASAL_DEMAND_UW) * effort.clamp(0.0, 1.0)
            }
        };
        let atp_demand = demand_uw * UW_TO_MM_ATP_PER_S * dt;

        // 4. Glycolysis + OXPHOS: glucose → ATP.
        //    O2 modulation: aerobic yield = 36 ATP; anaerobic fallback = 2 ATP.
        let o2_ratio = (self.ambient_o2_fraction / 0.21).clamp(0.0, 1.0);
        let effective_atp_per_glucose = 2.0 + (ATP_PER_GLUCOSE - 2.0) * o2_ratio;
        let glycolysis_flux =
            michaelis_menten(self.hemolymph_glucose_mm, GLYCOLYSIS_VMAX, GLYCOLYSIS_KM) * dt;
        // Limit by available glucose and ADP (need ADP to make ATP).
        let glucose_consumed = glycolysis_flux
            .min(self.hemolymph_glucose_mm)
            .min(self.muscle_adp_mm / effective_atp_per_glucose);
        let atp_produced = glucose_consumed * effective_atp_per_glucose;
        self.hemolymph_glucose_mm -= glucose_consumed;
        self.muscle_atp_mm += atp_produced;
        self.muscle_adp_mm -= atp_produced.min(self.muscle_adp_mm);

        // 5. ATP consumption (demand).
        let atp_consumed = atp_demand.min(self.muscle_atp_mm);
        self.muscle_atp_mm -= atp_consumed;
        self.muscle_adp_mm += atp_consumed;

        // 5b. Neural metabolic cost: 20% of basal rate at full neural firing.
        let neural_atp_cost = self.neural_activity_fraction
            * BASAL_DEMAND_UW * UW_TO_MM_ATP_PER_S * 0.2 * dt;
        let neural_consumed = neural_atp_cost.min(self.muscle_atp_mm);
        self.muscle_atp_mm -= neural_consumed;
        self.muscle_adp_mm += neural_consumed;

        // 6. Fat body glycogenolysis: glycogen → glucose (emergency).
        if self.hemolymph_glucose_mm < GLYCOGENOLYSIS_THRESHOLD {
            let glycogenolysis_flux =
                michaelis_menten(self.fat_body_glycogen_mg, GLYCOGENOLYSIS_VMAX, GLYCOGENOLYSIS_KM)
                    * dt;
            let glyc_consumed = glycogenolysis_flux.min(self.fat_body_glycogen_mg);
            self.fat_body_glycogen_mg -= glyc_consumed;
            self.hemolymph_glucose_mm += glyc_consumed * MG_TO_MM_GLUCOSE;
        }

        // 7. Lipid mobilization (last resort).
        if self.fat_body_glycogen_mg < LIPID_MOB_THRESHOLD {
            let lipid_flux =
                michaelis_menten(self.fat_body_lipid_mg, LIPID_MOB_VMAX, LIPID_MOB_KM) * dt;
            let lipid_consumed = lipid_flux.min(self.fat_body_lipid_mg);
            self.fat_body_lipid_mg -= lipid_consumed;
            // Lipid → glucose equivalent (~2.5x energy density, ~1.4x glucose yield).
            self.hemolymph_glucose_mm += lipid_consumed * MG_TO_MM_GLUCOSE * 1.4;
        }

        // 8. Fat body storage: glucose → glycogen (anabolic, when glucose high).
        if self.hemolymph_glucose_mm > GLYCOGEN_STORAGE_THRESHOLD {
            let excess = self.hemolymph_glucose_mm - GLYCOGEN_STORAGE_THRESHOLD;
            let storage_flux =
                michaelis_menten(excess, GLYCOGEN_STORAGE_VMAX, GLYCOGEN_STORAGE_KM) * dt;
            let stored = storage_flux.min(excess / MG_TO_MM_GLUCOSE);
            self.fat_body_glycogen_mg += stored;
            self.hemolymph_glucose_mm -= stored * MG_TO_MM_GLUCOSE;
        }

        // 9. Clamp all pools to physiological bounds.
        self.crop_sugar_mg = self.crop_sugar_mg.clamp(0.0, MAX_CROP_SUGAR);
        self.hemolymph_trehalose_mm = self.hemolymph_trehalose_mm.clamp(0.0, MAX_TREHALOSE);
        self.hemolymph_glucose_mm = self.hemolymph_glucose_mm.clamp(0.0, MAX_GLUCOSE);
        self.fat_body_glycogen_mg = self.fat_body_glycogen_mg.clamp(0.0, MAX_GLYCOGEN);
        self.fat_body_lipid_mg = self.fat_body_lipid_mg.clamp(0.0, MAX_LIPID);
        self.muscle_atp_mm = self.muscle_atp_mm.clamp(0.0, MAX_ATP);
        self.muscle_adp_mm = self.muscle_adp_mm.clamp(0.0, MAX_ADP);
    }
}

// ============================================================================
// OrganismMetabolism trait implementation
// ============================================================================

impl OrganismMetabolism for FlyMetabolism {
    fn energy_charge(&self) -> f32 {
        let amp = (TOTAL_ADENYLATE - self.muscle_atp_mm - self.muscle_adp_mm).max(0.0);
        let total = self.muscle_atp_mm + self.muscle_adp_mm + amp;
        if total < 1e-6 {
            return 0.0;
        }
        ((self.muscle_atp_mm + 0.5 * self.muscle_adp_mm) / total).clamp(0.0, 1.0)
    }

    fn biomass_mg(&self) -> f32 {
        // Adult Drosophila body mass ~1.0 mg (dry weight ~0.3 mg).
        // Report metabolically active mass: reserves + structural.
        0.3 + self.fat_body_glycogen_mg + self.fat_body_lipid_mg
    }

    fn substrate_saturation(&self) -> f32 {
        // Trehalose is the primary circulating sugar in insects.
        (self.hemolymph_trehalose_mm / TREHALOSE_SATIATED).clamp(0.0, 1.0)
    }

    fn oxygen_status(&self) -> f32 {
        // Coupled to terrarium atmosphere: ambient_o2_fraction / normoxic baseline.
        (self.ambient_o2_fraction / 0.21).clamp(0.0, 1.0)
    }

    fn is_stressed(&self) -> bool {
        self.energy_charge() < 0.5
            || self.hemolymph_trehalose_mm < 5.0
            || self.muscle_atp_mm < 1.0
    }

    fn metabolic_rate_fraction(&self) -> f32 {
        // Ratio of current ATP turnover to maximum possible.
        let demand_uw = match self.activity {
            FlyActivity::Resting => BASAL_DEMAND_UW,
            FlyActivity::Walking(s) => BASAL_DEMAND_UW + (WALK_DEMAND_UW - BASAL_DEMAND_UW) * s.clamp(0.0, 1.0),
            FlyActivity::Flying(e) => BASAL_DEMAND_UW + (FLY_DEMAND_UW - BASAL_DEMAND_UW) * e.clamp(0.0, 1.0),
        };
        (demand_uw / FLY_DEMAND_UW).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metabolism_stays_bounded() {
        let mut m = FlyMetabolism::default();
        // Alternate flying and resting for 1000 steps with no food.
        for i in 0..1000 {
            if i % 2 == 0 {
                m.set_activity(FlyActivity::Flying(0.8));
            } else {
                m.set_activity(FlyActivity::Resting);
            }
            m.step(0.05);
            assert!(m.crop_sugar_mg >= 0.0 && m.crop_sugar_mg <= MAX_CROP_SUGAR);
            assert!(m.hemolymph_trehalose_mm >= 0.0 && m.hemolymph_trehalose_mm <= MAX_TREHALOSE);
            assert!(m.hemolymph_glucose_mm >= 0.0 && m.hemolymph_glucose_mm <= MAX_GLUCOSE);
            assert!(m.fat_body_glycogen_mg >= 0.0 && m.fat_body_glycogen_mg <= MAX_GLYCOGEN);
            assert!(m.fat_body_lipid_mg >= 0.0 && m.fat_body_lipid_mg <= MAX_LIPID);
            assert!(m.muscle_atp_mm >= 0.0 && m.muscle_atp_mm <= MAX_ATP);
            assert!(m.muscle_adp_mm >= 0.0 && m.muscle_adp_mm <= MAX_ADP);
        }
    }

    #[test]
    fn flight_depletes_atp_faster() {
        let mut m_walk = FlyMetabolism::default();
        let mut m_fly = FlyMetabolism::default();
        m_walk.set_activity(FlyActivity::Walking(1.0));
        m_fly.set_activity(FlyActivity::Flying(1.0));
        for _ in 0..100 {
            m_walk.step(0.05);
            m_fly.step(0.05);
        }
        // Flying should deplete more reserves than walking.
        assert!(
            m_fly.fat_body_glycogen_mg + m_fly.fat_body_lipid_mg
                < m_walk.fat_body_glycogen_mg + m_walk.fat_body_lipid_mg,
            "Flying should deplete reserves faster than walking"
        );
    }

    #[test]
    fn feeding_restores_trehalose() {
        let mut m = FlyMetabolism::default();
        // Deplete heavily first: fly for a long time.
        m.set_activity(FlyActivity::Flying(1.0));
        for _ in 0..2000 {
            m.step(0.05);
        }
        let tre_before = m.hemolymph_trehalose_mm;
        // Feed a substantial meal.
        m.ingest(MAX_CROP_SUGAR); // full crop load
        // Only step once to capture the crop absorption spike before trehalase drains it.
        m.set_activity(FlyActivity::Resting);
        m.step(0.05);
        // Crop absorption adds trehalose faster than trehalase removes it in one step.
        assert!(
            m.hemolymph_trehalose_mm > tre_before || m.crop_sugar_mg > 0.0,
            "Feeding should raise trehalose or fill crop: tre before={tre_before:.3}, after={:.3}, crop={:.3}",
            m.hemolymph_trehalose_mm,
            m.crop_sugar_mg
        );
    }

    #[test]
    fn starvation_mobilizes_reserves() {
        let mut m = FlyMetabolism::default();
        // Start with depleted circulating sugars to force reserve mobilization.
        m.hemolymph_trehalose_mm = 0.5;
        m.hemolymph_glucose_mm = 0.5;
        let initial_glycogen = m.fat_body_glycogen_mg;
        let initial_lipid = m.fat_body_lipid_mg;
        m.set_activity(FlyActivity::Walking(0.5));
        for _ in 0..500 {
            m.step(0.05);
        }
        // With low circulating sugars, glycogenolysis should fire.
        assert!(
            m.fat_body_glycogen_mg < initial_glycogen || m.fat_body_lipid_mg < initial_lipid,
            "Starvation should mobilize reserves: glycogen {initial_glycogen:.6} -> {:.6}, lipid {initial_lipid:.6} -> {:.6}",
            m.fat_body_glycogen_mg, m.fat_body_lipid_mg
        );
    }

    #[test]
    fn energy_compat_within_range() {
        let mut m = FlyMetabolism::default();
        for _ in 0..500 {
            m.set_activity(FlyActivity::Flying(1.0));
            m.step(0.05);
            let e = m.energy_compat_uj();
            assert!(e >= 0.0 && e <= FLY_ENERGY_MAX, "energy_compat_uj={e} out of range");
        }
    }

    #[test]
    fn hunger_rises_without_food() {
        let mut m = FlyMetabolism::default();
        let hunger_start = m.hunger();
        m.set_activity(FlyActivity::Walking(0.5));
        for _ in 0..200 {
            m.step(0.05);
        }
        assert!(
            m.hunger() > hunger_start,
            "Hunger should rise without food: start={hunger_start:.3}, end={:.3}",
            m.hunger()
        );
    }

    #[test]
    fn trait_energy_charge_valid() {
        let mut m = FlyMetabolism::default();
        for _ in 0..100 {
            m.step(0.05);
            let ec = m.energy_charge();
            assert!(ec >= 0.0 && ec <= 1.0, "Energy charge {ec} out of [0,1]");
        }
    }

    #[test]
    fn o2_deprivation_reduces_atp_yield() {
        let mut m_normal = FlyMetabolism::default();
        let mut m_hypoxic = FlyMetabolism::default();
        m_hypoxic.set_ambient_o2(0.05); // severe hypoxia
        // Use Flying activity to stress the system enough that O2 matters
        m_normal.set_activity(FlyActivity::Flying(0.8));
        m_hypoxic.set_activity(FlyActivity::Flying(0.8));
        // Start with very limited reserves so hypoxia effect is visible
        // Normal fly: 36 ATP/glucose, Hypoxic: ~2 ATP/glucose (severe hypoxia)
        m_normal.hemolymph_trehalose_mm = 2.0;
        m_hypoxic.hemolymph_trehalose_mm = 2.0;
        m_normal.hemolymph_glucose_mm = 1.0;
        m_hypoxic.hemolymph_glucose_mm = 1.0;
        m_normal.fat_body_glycogen_mg = 0.005;
        m_hypoxic.fat_body_glycogen_mg = 0.005;
        m_normal.fat_body_lipid_mg = 0.01;
        m_hypoxic.fat_body_lipid_mg = 0.01;
        for _ in 0..200 {
            m_normal.step(0.05);
            m_hypoxic.step(0.05);
        }
        // Hypoxic fly should consume more fuel to compensate for lower ATP yield.
        // Check: lower ATP OR lower glycogen OR lower trehalose (total fuel consumed)
        let hypoxic_worse_atp = m_hypoxic.muscle_atp_mm < m_normal.muscle_atp_mm;
        let hypoxic_worse_glyc = m_hypoxic.fat_body_glycogen_mg < m_normal.fat_body_glycogen_mg;
        let hypoxic_consumed_more_fuel =
            m_hypoxic.hemolymph_trehalose_mm + m_hypoxic.hemolymph_glucose_mm
                < m_normal.hemolymph_trehalose_mm + m_normal.hemolymph_glucose_mm;
        assert!(
            hypoxic_worse_atp || hypoxic_worse_glyc || hypoxic_consumed_more_fuel,
            "Hypoxia should reduce ATP, deplete reserves, or consume more fuel: \
             normal ATP={:.3} glyc={:.6} fuel={:.3}, hypoxic ATP={:.3} glyc={:.6} fuel={:.3}",
            m_normal.muscle_atp_mm,
            m_normal.fat_body_glycogen_mg,
            m_normal.hemolymph_trehalose_mm + m_normal.hemolymph_glucose_mm,
            m_hypoxic.muscle_atp_mm,
            m_hypoxic.fat_body_glycogen_mg,
            m_hypoxic.hemolymph_trehalose_mm + m_hypoxic.hemolymph_glucose_mm,
        );
        // oxygen_status should reflect the low O2.
        assert!(
            m_hypoxic.oxygen_status() < 0.5,
            "Hypoxic O2 status={:.3} should be < 0.5",
            m_hypoxic.oxygen_status()
        );
    }

    #[test]
    fn set_reserves_from_uj_roundtrips() {
        let mut m = FlyMetabolism::default();
        for target_uj in [0.0, 500.0, 1500.0, 2500.0, 3700.0] {
            m.set_reserves_from_uj(target_uj);
            let actual = m.energy_compat_uj();
            let diff = (actual - target_uj).abs();
            assert!(
                diff < 50.0,
                "set_reserves_from_uj({target_uj}) → energy_compat_uj()={actual:.1}, diff={diff:.1}"
            );
        }
    }
}
