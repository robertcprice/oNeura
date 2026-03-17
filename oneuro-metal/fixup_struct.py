#!/usr/bin/env python3
"""Add missing struct fields to TerrariumWorld definition."""
import os

SRC = os.path.dirname(os.path.abspath(__file__)) + "/src"

def read(path):
    with open(path) as f:
        return f.read()

def write(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {path} ({len(content)} bytes)")

tw = read(f"{SRC}/terrarium_world.rs")

# Find the exact struct closing pattern
old = (
    "    ownership_diagnostics: OwnershipDiagnostics,\n"
    "}\n"
    "\n"
    "impl TerrariumWorld {"
)

new_fields = (
    "    ownership_diagnostics: OwnershipDiagnostics,\n"
    "    // ── Microbial guild fields (decomposers) ──\n"
    "    nitrifier_biomass: Vec<f32>,\n"
    "    microbial_cells: Vec<f32>,\n"
    "    microbial_packets: Vec<f32>,\n"
    "    microbial_copiotroph_fraction: Vec<f32>,\n"
    "    microbial_copiotroph_packets: Vec<f32>,\n"
    "    microbial_dormancy: Vec<f32>,\n"
    "    microbial_vitality: Vec<f32>,\n"
    "    microbial_reserve: Vec<f32>,\n"
    "    microbial_strain_yield: Vec<f32>,\n"
    "    microbial_strain_stress_tolerance: Vec<f32>,\n"
    "    microbial_packet_mutation_flux: Vec<f32>,\n"
    "    microbial_latent_packets: Vec<Vec<f32>>,\n"
    "    microbial_latent_strain_yield: Vec<Vec<f32>>,\n"
    "    microbial_latent_strain_stress_tolerance: Vec<Vec<f32>>,\n"
    "    microbial_secondary: SoilBroadSecondaryBanks,\n"
    "    // ── Nitrifier guild fields ──\n"
    "    nitrifier_cells: Vec<f32>,\n"
    "    nitrifier_packets: Vec<f32>,\n"
    "    nitrifier_aerobic_fraction: Vec<f32>,\n"
    "    nitrifier_aerobic_packets: Vec<f32>,\n"
    "    nitrifier_dormancy: Vec<f32>,\n"
    "    nitrifier_vitality: Vec<f32>,\n"
    "    nitrifier_reserve: Vec<f32>,\n"
    "    nitrifier_strain_oxygen_affinity: Vec<f32>,\n"
    "    nitrifier_strain_ammonium_affinity: Vec<f32>,\n"
    "    nitrifier_packet_mutation_flux: Vec<f32>,\n"
    "    nitrifier_latent_packets: Vec<Vec<f32>>,\n"
    "    nitrifier_latent_strain_oxygen_affinity: Vec<Vec<f32>>,\n"
    "    nitrifier_latent_strain_ammonium_affinity: Vec<Vec<f32>>,\n"
    "    nitrifier_secondary: SoilBroadSecondaryBanks,\n"
    "    // ── Denitrifier guild fields ──\n"
    "    denitrifier_biomass: Vec<f32>,\n"
    "    denitrifier_cells: Vec<f32>,\n"
    "    denitrifier_packets: Vec<f32>,\n"
    "    denitrifier_anoxic_fraction: Vec<f32>,\n"
    "    denitrifier_anoxic_packets: Vec<f32>,\n"
    "    denitrifier_dormancy: Vec<f32>,\n"
    "    denitrifier_vitality: Vec<f32>,\n"
    "    denitrifier_reserve: Vec<f32>,\n"
    "    denitrifier_strain_anoxia_affinity: Vec<f32>,\n"
    "    denitrifier_strain_nitrate_affinity: Vec<f32>,\n"
    "    denitrifier_packet_mutation_flux: Vec<f32>,\n"
    "    denitrifier_latent_packets: Vec<Vec<f32>>,\n"
    "    denitrifier_latent_strain_anoxia_affinity: Vec<Vec<f32>>,\n"
    "    denitrifier_latent_strain_nitrate_affinity: Vec<Vec<f32>>,\n"
    "    denitrifier_secondary: SoilBroadSecondaryBanks,\n"
    "    // ── Soil potential fields ──\n"
    "    nitrification_potential: Vec<f32>,\n"
    "    denitrification_potential: Vec<f32>,\n"
    "    // ── Explicit microbe fields ──\n"
    "    pub explicit_microbes: Vec<TerrariumExplicitMicrobe>,\n"
    "    explicit_microbe_authority: Vec<f32>,\n"
    "    explicit_microbe_activity: Vec<f32>,\n"
    "    next_microbe_idx: usize,\n"
    "    next_species_id: u32,\n"
    "    // ── Atmosphere physics ──\n"
    "    air_pressure_kpa: Vec<f32>,\n"
    "    air_density: Vec<f32>,\n"
    "    // ── Packet populations ──\n"
    "    packet_populations: Vec<()>,\n"
    "    // ── MD calibrator ──\n"
    "    md_calibrator: Option<()>,\n"
    "}\n"
    "\n"
    "impl TerrariumWorld {"
)

if old in tw:
    tw = tw.replace(old, new_fields)
    print("  Added 55+ struct fields to TerrariumWorld definition")
else:
    # Try to find the pattern more flexibly
    lines = tw.split('\n')
    found = False
    for i, line in enumerate(lines):
        if 'ownership_diagnostics: OwnershipDiagnostics,' in line and i + 1 < len(lines) and lines[i+1].strip() == '}':
            # Found the struct end
            insert_lines = new_fields.split('\n')[1:-3]  # Skip first line (already there) and last 3 lines
            lines = lines[:i+1] + insert_lines + lines[i+1:]
            tw = '\n'.join(lines)
            found = True
            print(f"  Added fields after line {i+1} (flexible match)")
            break
    if not found:
        print("  ERROR: Could not find struct field insertion point!")
        import sys
        sys.exit(1)

# Also fix WholeCellSimulator Debug requirement
# TerrariumExplicitMicrobe has #[derive(Debug)] but WholeCellSimulator may not impl Debug
# Change TerrariumExplicitMicrobe to not derive Debug
tw = tw.replace(
    "/// An explicit microbe in the terrarium with whole-cell simulation.\n"
    "#[derive(Debug)]\n"
    "pub struct TerrariumExplicitMicrobe {",
    "/// An explicit microbe in the terrarium with whole-cell simulation.\n"
    "pub struct TerrariumExplicitMicrobe {",
)

# Re-gate explicit_microbe_impl.rs and snapshot.rs (too many missing types)
tw = tw.replace(
    "mod snapshot;\n",
    '#[cfg(feature = "terrarium_advanced")]\nmod snapshot;\n',
)
tw = tw.replace(
    "mod explicit_microbe_impl;\n",
    '#[cfg(feature = "terrarium_advanced")]\nmod explicit_microbe_impl;\n',
)
print("  Re-gated snapshot.rs and explicit_microbe_impl.rs (deep dependency chains)")

write(f"{SRC}/terrarium_world.rs", tw)
