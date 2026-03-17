use super::*;

impl TerrariumWorld {
    /// Step latent guild banks (microbial/nitrifier/denitrifier secondary genotype evolution).
    /// Stub: will be connected when guild_latent infrastructure is fully wired.
    pub(super) fn step_latent_strain_banks(&mut self, _eco_dt: f32) -> Result<(), String> {
        Ok(())
    }

    /// Step visual biomechanics (wind effects on plants, seeds, fruits).
    /// Stub: requires pose fields on Plant/Seed/Fruit + integrate_displacement.
    pub(super) fn step_visual_biomechanics(&mut self, _dt: f32) {
        // No-op until pose fields are added to TerrariumPlant, TerrariumSeed, TerrariumFruitPatch
    }
}
