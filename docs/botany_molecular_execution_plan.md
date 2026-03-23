# Botany Molecular Execution Plan

## 1. Vision & Architecture
Transition from parameterized "generic plants" to an empirically grounded, molecular-botany simulation. Plants and fruits will be governed by their specific genetic sequences, regulated by abiotic stress, and constructed from precise atomic graphs (Metabolome) rather than abstract "sugar" or "energy" values.

## 2. Genomic Catalog (Phase 1)
Introduce a `GenomicCatalog` module to house actual species DNA.
- **Model Organisms:**
  - *Arabidopsis thaliana*: For rapid life-cycle modeling and baseline genomic testing.
  - *Malus domestica* (Apple) / *Solanum lycopersicum* (Tomato): For fruiting and complex metabolic synthesis.
- **Gene Regulatory Networks (GRNs):**
  - Implement transcription factors that respond to the terrarium's `moisture`, `temperature`, and `light_intensity`.
  - Growth phases (germination, vegetative, flowering, fruiting) will be triggered strictly by GRN state transitions, removing all hardcoded timers.

## 3. The Molecular Metabolome (Phase 2)
Replace the generic `sugar_content` with actual chemical synthesis using the `atomistic_chemistry` engine.
- **Metabolite Graphs:**
  - Fruits will contain a heterogeneous mix of simulated molecules: Fructose ($C_6H_{12}O_6$), Glucose, Malic Acid, etc.
- **Volatile Organic Compounds (VOCs):**
  - Plants will emit specific VOCs (e.g., Ethylene for ripening, specific esters for scent).
  - The *Drosophila* olfactory system will be updated to bind to these precise molecular geometries, driving emergent chemotaxis behavior.

## 4. Epigenetic Morphology & L-Systems (Phase 3)
Connect the plant's genomic state to its physical 3D manifestation.
- **Dynamic L-Systems:**
  - Branching rules will be read from the active genome.
  - Epigenetic markers triggered by environmental stress (e.g., drought) will alter the L-System parameters, causing realistic stunted growth or leaf shedding to conserve $H_2O$.
- **Rendering:**
  - The 3D renderers (Bevy and WebGL) will consume the L-System graphs to draw morphologically accurate, species-specific models dynamically.

## 5. Atomic Digestion (Phase 4)
Close the loop by simulating digestion at the atomic level.
- **Enzymatic Catalysis:**
  - When a fly consumes a fruit, its digestive enzymes will perform simulated cleaving of the specific molecular bonds.
  - ATP yields will be calculated based on the actual bond energies broken during the simulation, rather than arbitrary energy floats.

## Next Steps
- [ ] Implement `GenomicCatalog` and stub the *Arabidopsis* genome.
- [ ] Connect the terrarium's abiotic factors (temperature, moisture) to the GRN inputs.
- [ ] Integrate the `atomistic_chemistry` engine into the plant's fruiting cycle to synthesize Fructose and VOCs.
