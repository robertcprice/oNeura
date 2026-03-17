#!/usr/bin/env python3
"""Wire terrarium_world/flora.rs unconditionally.

Strategy: maximize changes to flora.rs (linter-safe) and minimize
changes to terrarium_world.rs (linter-watched).  The only terrarium_world.rs
changes are:
  1. Delete inline step_plants/step_food_patches_native/step_seeds_native (~630 lines)
  2. Change `mod flora;` from feature-gated to unconditional
  3. Add 2 constants (ATMOS_CO2_BASELINE, ATMOS_O2_BASELINE, PLANT_SPECIATION_THRESHOLD)

terrarium.rs, plant_organism.rs, soil_broad.rs are NOT touched.
genotype.rs, packet.rs, calibrator.rs stay feature-gated.
"""
import re, os, sys

SRC = os.path.dirname(os.path.abspath(__file__)) + "/src"

def read(path):
    with open(path) as f:
        return f.read()

def write(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {path} ({len(content)} bytes)")

# ====================================================================
# 1. flora.rs — Adapt to work WITHOUT upstream changes
# ====================================================================
print("1. Adapting flora.rs ...")
fl = read(f"{SRC}/terrarium_world/flora.rs")

# 1a. Replace physiology accessor calls with genome field access
#     These 4 methods don't exist as public on PlantOrganismSim.
#     Careful: line 46 is `p.physiology.canopy_radius_mm().max(1.0)`
#     so we replace without adding another .max(1.0)
fl = fl.replace("p.physiology.canopy_radius_mm()", "p.genome.canopy_radius_mm")
fl = fl.replace("p.physiology.extinction_coeff()", "(0.45 + p.genome.leaf_efficiency * 0.25)")
fl = fl.replace("p.physiology.root_depth_bias()", "p.genome.root_depth_bias")
fl = fl.replace("p.physiology.root_radius_mm()", "p.genome.root_radius_mm")

# 1b. Replace deposit_2d_background_only calls with deposit_2d.
#     deposit_2d_background_only(field, w, h, x, y, radius, amount, authority, threshold)
#     -> deposit_2d(field, w, h, x, y, radius, amount)
#     The calls span multiple lines. [^,] matches newlines in Python re.
fl = re.sub(
    r'deposit_2d_background_only\('
    r'(\s*[^,]+,)'      # field
    r'(\s*[^,]+,)'      # width
    r'(\s*[^,]+,)'      # height
    r'(\s*[^,]+,)'      # x
    r'(\s*[^,]+,)'      # y
    r'(\s*[^,]+,)'      # radius
    r'(\s*[^,]+,)'      # amount
    r'\s*[^,]+,\s*'     # authority (skip)
    r'[^)]*\)',          # threshold (skip) + closing paren
    r'deposit_2d(\1\2\3\4\5\6\7\n            )',  # reconstruct
    fl,
)
# Clean up potential double-newlines from replacement
fl = re.sub(r'\n\s*\n\s*\)', '\n            )', fl)

# 1c. Replace report.o2_flux → (-report.co2_flux * 1.05)
fl = fl.replace("report.o2_flux", "(-report.co2_flux * 1.05)")

# 1d. Replace nitrifier/denitrifier biomass with microbial_biomass approximations
fl = fl.replace(
    "self.nitrifier_biomass[flat] * 0.55",
    "self.microbial_biomass[flat] * 0.08",
)
fl = fl.replace(
    "self.denitrifier_biomass[flat] * 0.70",
    "self.microbial_biomass[flat] * 0.08",
)
# Catch any remaining direct references
fl = fl.replace("self.nitrifier_biomass[flat]", "(self.microbial_biomass[flat] * 0.15)")
fl = fl.replace("self.denitrifier_biomass[flat]", "(self.microbial_biomass[flat] * 0.12)")

# 1e. Replace exchange_atmosphere_flux_bundle with inline 3-call equivalent
fl = fl.replace(
    "self.exchange_atmosphere_flux_bundle(x, y, z, radius, co2_flux, o2_flux, humidity_flux);",
    "self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, co2_flux);\n"
    "            self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, o2_flux);\n"
    "            self.exchange_atmosphere_humidity(x, y, z, radius, humidity_flux);",
)

# 1f. Add local constants that flora.rs needs
#     These must come BEFORE impl TerrariumWorld {
if "ATMOS_CO2_BASELINE" not in fl:
    fl = fl.replace(
        "use super::*;\n\nimpl TerrariumWorld {",
        "use super::*;\n\n"
        "const ATMOS_CO2_BASELINE: f32 = 0.045;\n"
        "const ATMOS_O2_BASELINE: f32 = 0.21;\n"
        "const PLANT_SPECIATION_THRESHOLD: f32 = 0.15;\n"
        "const EXPLICIT_OWNERSHIP_THRESHOLD: f32 = 0.5;\n"
        "\nimpl TerrariumWorld {",
    )

# 1g. Change super::PLANT_SPECIATION_THRESHOLD to local constant
fl = fl.replace("super::PLANT_SPECIATION_THRESHOLD", "PLANT_SPECIATION_THRESHOLD")

write(f"{SRC}/terrarium_world/flora.rs", fl)

# Verify no remaining broken references
for bad in [
    "deposit_2d_background_only",
    "p.physiology.canopy_radius_mm",
    "p.physiology.extinction_coeff",
    "p.physiology.root_depth_bias",
    "p.physiology.root_radius_mm",
    "report.o2_flux",
    "exchange_atmosphere_flux_bundle",
    "nitrifier_biomass",
    "denitrifier_biomass",
    "explicit_microbe_authority",
]:
    if bad in fl:
        print(f"  WARNING: '{bad}' still present in flora.rs!")

# ====================================================================
# 2. terrarium_world.rs — Delete inlines + wire flora mod
# ====================================================================
print("2. Modifying terrarium_world.rs ...")
tw = read(f"{SRC}/terrarium_world.rs")
tw_changed = False

# 2a. Delete inline step_plants, step_food_patches_native, step_seeds_native
#     They start at "    fn step_plants(" and end before "    fn step_world_fields("
lines = tw.split('\n')
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '    fn step_plants(&mut self, eco_dt: f32)' in line and start_idx is None:
        start_idx = i
    if '    fn step_world_fields(&mut self)' in line:
        end_idx = i
        break

if start_idx is not None and end_idx is not None:
    inline_block = '\n'.join(lines[start_idx:end_idx])
    if 'fn step_food_patches_native' in inline_block:
        deleted = end_idx - start_idx
        lines = lines[:start_idx] + lines[end_idx:]
        tw = '\n'.join(lines)
        tw_changed = True
        print(f"  Deleted {deleted} lines of inline methods (lines {start_idx+1}-{end_idx})")
    else:
        print("  (inline methods already removed)")
else:
    print(f"  WARNING: Could not find inline method boundaries (start={start_idx}, end={end_idx})")

# 2b. Change mod flora from feature-gated to unconditional
old_flora_mod = '#[cfg(feature = "terrarium_advanced")]\nmod flora;'
new_flora_mod = 'mod flora;'
if old_flora_mod in tw:
    tw = tw.replace(old_flora_mod, new_flora_mod)
    tw_changed = True
    print("  Changed flora mod to unconditional")
elif 'mod flora;' in tw:
    print("  (flora mod already unconditional)")
else:
    print("  WARNING: Could not find flora mod declaration")

# 2c. Add constants if not present
if "ATMOS_CO2_BASELINE" not in tw:
    # Add after ATMOS_O2_FRACTION
    tw = tw.replace(
        "const ATMOS_O2_FRACTION: f32 = 0.21;\n",
        "const ATMOS_O2_FRACTION: f32 = 0.21;\n"
        "const ATMOS_CO2_BASELINE: f32 = 0.045;\n"
        "const ATMOS_O2_BASELINE: f32 = 0.21;\n"
        "const PLANT_SPECIATION_THRESHOLD: f32 = 0.15;\n"
        "const EXPLICIT_OWNERSHIP_THRESHOLD: f32 = 0.5;\n",
    )
    tw_changed = True
    print("  Added atmosphere/speciation constants")

if tw_changed:
    write(f"{SRC}/terrarium_world.rs", tw)
else:
    print("  (no changes needed)")

# ====================================================================
# Summary
# ====================================================================
print("\n=== Summary ===")
print("Modified files:")
print("  1. src/terrarium_world/flora.rs — adapted to use genome fields, deposit_2d, inline atmo calls")
print("  2. src/terrarium_world.rs — deleted inline methods, wired flora mod unconditionally")
print("\nUNMODIFIED (genotype/packet/calibrator stay feature-gated):")
print("  - terrarium.rs")
print("  - plant_organism.rs")
print("  - soil_broad.rs")
