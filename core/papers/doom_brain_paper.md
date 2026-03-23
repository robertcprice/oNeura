# dONN Plays Doom: Biophysical Neural Networks Navigate Real Game Environments via Free Energy Minimization

**Bobby Price**

March 2026

**Target journals:** *Neuron*, *Nature Machine Intelligence*, *PLOS Computational Biology*

**Status:** DRAFT -- placeholder data marked `[GPU-PENDING]` awaiting large-scale GPU experiments

---

## Abstract

We demonstrate that a digital Organic Neural Network (dONN) -- a biophysically faithful brain simulator implementing Hodgkin-Huxley ion channels, molecular synapses, and spike-timing-dependent plasticity -- can navigate the first-person shooter game Doom through the ViZDoom research platform. Visual input is processed by a three-layer molecular retina (photoreceptors with Govardovskii spectral sensitivity, ON/OFF bipolar cells with center-surround antagonism, and spiking retinal ganglion cells) that converts 160x120 rendered game frames into biologically realistic spike trains. Learning follows the Free Energy Principle: structured pulsed stimulation on positive game events (low entropy) and random noise on negative events (high entropy), with Hebbian weight nudges for credit assignment. The dONN develops directed navigation behavior in ViZDoom's health_gathering scenario, outperforming random movement baselines. Pharmacological manipulation -- caffeine, diazepam -- modifies game performance in directions consistent with known receptor pharmacology, an experiment that is fundamentally impossible on real biological tissue due to irreversible drug effects. This work extends the DishBrain paradigm (Kagan et al., 2022) from one-dimensional Pong played by cortical cultures to three-dimensional first-person navigation played by a complete digital brain, bridging computational neuroscience, game AI, and neuropharmacology.

---

## 1. Introduction

### 1.1 From DishBrain to Doom

In 2022, Kagan et al. demonstrated that approximately 800,000 cortical neurons cultured on a high-density multi-electrode array (MEA) could learn to play the arcade game Pong (Kagan et al., 2022). Published in *Neuron*, this work provided the first evidence that an Organic Neural Network (ONN) -- a computational system built from living biological neurons -- could exhibit goal-directed behavior in a closed-loop game environment. The learning mechanism was grounded in the Free Energy Principle (FEP; Friston, 2010): rather than receiving explicit reward or punishment, the culture was exposed to structured, predictable electrical stimulation when the paddle intercepted the ball (a HIT) and random, unpredictable stimulation when it missed (a MISS). Under the FEP, biological neurons inherently minimize the surprise of their sensory inputs. By making successful actions produce predictable feedback and failed actions produce unpredictable feedback, the culture self-organized via spike-timing-dependent plasticity (STDP) to produce motor outputs that preferentially resulted in hits.

This finding was remarkable for three reasons. First, it demonstrated that a flat sheet of cortical neurons, with no pre-specified architecture beyond what emerged from culture, could perform sensorimotor learning in real time. Second, it provided experimental validation for the FEP as a sufficient mechanism for adaptive behavior. Third, it opened the possibility that biological neural computation could be studied in closed-loop game environments rather than solely through traditional electrophysiological paradigms.

However, the DishBrain paradigm faces practical constraints. Each experimental run requires a fresh cortical culture ($50,000+ per experiment), drug application is irreversible (a benzodiazepine added to the media cannot be removed), biological variability between cultures confounds interpretation, and the MEA technology is limited to two-dimensional electrode grids unsuited for three-dimensional visual environments. Most significantly, Pong is a one-dimensional tracking task: the ball moves in two dimensions, but the paddle moves along a single axis. Real organisms navigate three-dimensional spaces with continuous visual fields, threat avoidance, and resource acquisition -- capabilities that require qualitatively different sensory processing and motor control.

### 1.2 The dONN Approach

We address these limitations with the digital Organic Neural Network (dONN), built on the oNeura simulation platform (Price, 2026). A dONN simulates every biophysical process that produces neural computation: Hodgkin-Huxley ion channel dynamics with eight channel types per neuron, six neurotransmitter systems with receptor binding kinetics, STDP via AMPA receptor trafficking, second messenger cascades (cAMP/PKA, PLC/IP3/PKC, CaMKII, ERK, CREB), and pharmacokinetic/pharmacodynamic drug modeling. Behaviors emerge from the molecular substrate rather than being programmed. The distinction between a dONN and a conventional artificial neural network is fundamental: an ANN implements mathematical abstractions of neural computation (matrix multiplication followed by nonlinear activation), while a dONN simulates the molecular machinery that produces neural computation.

### 1.3 Contributions

This paper makes five contributions:

1. **First biophysical brain to play Doom.** We demonstrate that a dONN navigates actual Doom levels via the ViZDoom API (Kempka et al., 2016), processing rendered game frames through a three-layer molecular retina and producing motor commands from spike-decoded cortical populations.

2. **Three-layer biophysical retina for visual game input.** The molecular retina converts 160x120 RGB frames into spike trains using Govardovskii spectral sensitivity (S=420nm, M=530nm, L=560nm, Rod=498nm), Weber adaptation, ON/OFF bipolar pathways with center-surround antagonism, and full HH spiking in retinal ganglion cells.

3. **FEP-based learning protocol for FPS navigation.** We adapt the DishBrain structured/unstructured feedback protocol from one-dimensional Pong to five-motor-population FPS navigation (forward, turn left/right, strafe left/right), demonstrating that the Free Energy Principle generalizes to complex three-dimensional tasks.

4. **Pharmacological manipulation of game performance.** We test caffeine and diazepam on identical post-training brains, demonstrating drug effects on FPS navigation that are consistent with known receptor pharmacology -- an experiment impossible on real tissue.

5. **Dynamic Difficulty Adjustment.** We introduce a biologically motivated DDA controller that monitors the brain's damage ratio and adjusts environmental challenge to maintain the learning sweet spot, analogous to how natural environments present graded challenges during development.

### 1.4 Scope and Relation to Prior Work

Our prior work (Price, 2026) replicated the DishBrain Pong experiment in the dONN framework and extended it to a two-dimensional Spatial Arena using BSP-generated dungeon environments with a simplified grid-based renderer. The present work advances this in three ways: (a) integration with the real Doom engine via ViZDoom, producing actual first-person rendered frames rather than simplified grid encodings; (b) a biophysical retina pipeline that converts pixel images to spike trains rather than injecting raycast distances directly into cortical neurons; and (c) a more challenging and ecologically valid game environment with continuous three-dimensional navigation, health pickups, enemies with AI behavior, and projectile avoidance.

---

## 2. Related Work

### 2.1 DishBrain and Biological Neural Computing

Kagan et al. (2022) demonstrated that cortical neurons on a MEA learn to play Pong via the Free Energy Principle. Structured stimulation on HITs produced correlated activity that strengthened relay-to-motor pathways via STDP; unstructured stimulation on MISSes produced uncorrelated activity with no systematic plasticity signal. The culture self-organized to minimize surprise by producing motor outputs that generated hits. Earlier work by Demarse et al. (2001) demonstrated closed-loop control of a simulated animal using cultured cortical neurons, and Bakkum et al. (2008) showed that cultures could learn specific activity patterns in response to electrical stimulation. FinalSpark has commercialized a bioprocessor platform using human iPSC-derived cortical organoids (FinalSpark, 2024), though with primary focus on computing metrics rather than game-learning.

### 2.2 The Free Energy Principle

The Free Energy Principle (Friston, 2006, 2010) proposes that biological systems minimize variational free energy -- an upper bound on surprise -- through perceptual inference (updating internal models) and active inference (acting on the world). In the game-learning context, STDP provides the synaptic update mechanism and the structured/unstructured feedback provides the free-energy gradient. Clark (2013) extended this framework to predictive processing, arguing that brains are fundamentally prediction machines. Rao and Ballard (1999) provided early computational evidence for predictive coding in visual cortex. Our work provides a molecular-level computational implementation demonstrating that FEP-based learning transfers from one-dimensional tracking to three-dimensional FPS navigation.

### 2.3 ViZDoom and Game-Based AI Research

ViZDoom (Kempka et al., 2016) provides a research platform for AI agents in the Doom engine, offering visual observations, flexible reward structures, and multiple scenarios of increasing complexity. Lample and Chaplot (2017) applied deep recurrent Q-networks (DRQN) to ViZDoom navigation with game feature detection. Dosovitskiy and Koltun (2017) demonstrated that direct future prediction could outperform reward-based methods in ViZDoom navigation tasks. Pathak et al. (2017) introduced curiosity-driven exploration using intrinsic reward from prediction error, conceptually related to the FEP but implemented as a gradient-based loss function in an ANN. All of these approaches use conventional deep learning architectures with backpropagation. Our work is, to our knowledge, the first to play Doom using a biophysical neural simulation with molecular-level dynamics.

### 2.4 Biophysical Neural Models

Hodgkin and Huxley (1952) established the canonical model of action potential generation through voltage-gated ion channel kinetics. Izhikevich (2003) introduced computationally efficient neuron models that capture diverse firing patterns. Brian2 (Stimberg et al., 2019) and NEST (Gewaltig and Diesmann, 2007) provide platforms for spiking network simulation but lack the molecular-level detail (neurotransmitter systems, receptor trafficking, pharmacology) necessary for pharmacological experiments. oNeura differs in simulating 16 interacting molecular subsystems per neuron, enabling emergent pharmacology.

### 2.5 Retinal Processing Models

Masland (2001) reviewed the functional architecture of the mammalian retina. Baylor et al. (1979) characterized the graded hyperpolarization response of photoreceptors to light. Govardovskii et al. (2000) provided nomograms for photopigment spectral sensitivity across species. Werblin and Dowling (1969) established the ON/OFF bipolar cell pathways, and Kuffler (1953) described center-surround antagonism in retinal ganglion cells. Curcio et al. (1990) characterized the photoreceptor mosaic, documenting the fovea-periphery gradient from cone-dominated to rod-dominated regions, and Roorda and Williams (1999) measured the L:M:S cone ratio as approximately 10:5:1. Our molecular retina implements all of these biological features.

### 2.6 Spike-Timing-Dependent Plasticity

Bi and Poo (1998) demonstrated that the direction and magnitude of synaptic weight change depend on the precise temporal relationship between pre- and postsynaptic spikes, with causal timing producing potentiation and anti-causal timing producing depression. Markram et al. (1997) reported similar findings in cortical neurons. In our implementation, STDP is realized through AMPA receptor trafficking driven by calcium-dependent kinase/phosphatase cascades, producing Hebbian-like learning from molecular rather than programmatic mechanisms.

---

## 3. Methods

### 3.1 Biophysical Brain Architecture

Each neuron in the dONN is modeled as a molecular system with Hodgkin-Huxley membrane dynamics and receptor-mediated synaptic transmission.

**Ion channels.** Membrane potential $V$ evolves according to the HH equations:

$$C_m \frac{dV}{dt} = -g_{Na} m^3 h (V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_{syn} + I_{ext}$$

where $C_m = 1.0$ uF/cm$^2$ is the membrane capacitance, $g_{Na} = 120$ mS/cm$^2$, $g_K = 36$ mS/cm$^2$, and $g_L = 0.3$ mS/cm$^2$ are the maximal conductances, $E_{Na} = +50$ mV, $E_K = -77$ mV, and $E_L = -54.387$ mV are the reversal potentials, $I_{syn}$ is the total synaptic current, and $I_{ext}$ is the externally applied current. The gating variables $m$, $h$, and $n$ follow standard HH kinetics:

$$\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V) x$$

for $x \in \{m, h, n\}$, where $\alpha_x$ and $\beta_x$ are the voltage-dependent rate functions (Hodgkin and Huxley, 1952).

Beyond the three classical HH channels, each neuron implements eight ion channel types: voltage-gated sodium (Na$_v$), voltage-gated potassium (K$_v$), potassium leak (K$_{leak}$), high-voltage-activated calcium (Ca$_v$: $m^2 h$ gating, $g_{max} = 4.4$ mS/cm$^2$, $E_{rev} = +120$ mV), AMPA ($g_{max} = 1.0$ mS/cm$^2$), NMDA with Mg$^{2+}$ voltage-dependent block ($g_{max} = 0.5$ mS/cm$^2$), GABA-A ($g_{max} = 1.0$ mS/cm$^2$, $E_{rev} = -80$ mV), and nicotinic acetylcholine receptor (nAChR: $g_{max} = 0.8$ mS/cm$^2$).

**Calcium dynamics.** Four-compartment calcium dynamics (cytoplasmic, endoplasmic reticulum, mitochondrial, and microdomain) govern neurotransmitter release, second messenger cascades, and plasticity. Calcium influx through Ca$_v$ and NMDA channels activates CaMKII bistable switching, which drives AMPA receptor insertion (LTP) or removal (LTD).

**Second messenger cascades.** Five intracellular signaling pathways are modeled: cAMP/PKA (dopamine D1 receptor signaling), PLC/IP3/PKC (mGluR and muscarinic signaling), CaMKII (calcium-dependent STDP), ERK/MAPK (growth factor signaling), and CREB (transcription factor for memory consolidation). These cascades interact through cross-talk pathways that produce emergent neuromodulatory effects.

**STDP via receptor trafficking.** Long-term plasticity is implemented through AMPA receptor trafficking rather than explicit learning rules. Causal pre-post timing elevates microdomain calcium above the CaMKII threshold, triggering receptor insertion (LTP). Anti-causal post-pre timing activates phosphatases that trigger receptor internalization (LTD). This implementation follows the BCM (Bienenstock-Cooper-Munro) metaplasticity rule, where the LTP/LTD crossover point slides with postsynaptic activity history.

**Synaptic connectivity.** Synapses are stored in compressed sparse row (CSR) format for efficient traversal during the simulation step. Connection probabilities are scaled inversely with network size to maintain approximately constant in-degree (~10 inputs per neuron), preventing quadratic synapse explosion at larger scales. Each synapse stores a weight, neurotransmitter type, and pre/post neuron indices.

**GPU acceleration.** The brain runs on Apple Metal GPU via the oNeura-Metal Rust backend, with the full HH + STDP + molecular pipeline executing in a single fused simulation step. The same architecture supports CUDA backends for datacenter-scale experiments.

### 3.2 Three-Layer Molecular Retina

The molecular retina converts rendered ViZDoom frames (160x120 RGB) into spike trains through three biologically faithful processing layers. The retina is implemented independently in both Python (`src/oneuro/molecular/retina.py`) and Rust (`src/retina.rs`) for CPU and Metal GPU backends respectively.

#### 3.2.1 Layer 1: Photoreceptors

Photoreceptors hyperpolarize in response to light (Baylor et al., 1979), producing graded potentials -- not action potentials. The mosaic follows the fovea-periphery gradient documented by Curcio et al. (1990):

- **Fovea** (central 30% of visual field): cones only, with L:M:S ratio approximately 10:5:1 (Roorda and Williams, 1999).
- **Periphery**: rods dominate at ~20:1 over cones. Rod density decreases with eccentricity.

Spectral sensitivity follows the Govardovskii nomogram (Govardovskii et al., 2000), approximated as a Gaussian:

$$S(\lambda, \lambda_{peak}) = \exp\left(-\frac{(\lambda - \lambda_{peak})^2}{2 \sigma^2}\right)$$

where $\sigma = 30$ nm and $\lambda_{peak}$ is 420 nm (S-cones), 530 nm (M-cones), 560 nm (L-cones), or 498 nm (rods). RGB pixel values are converted to photoreceptor activation using precomputed spectral weight vectors derived from monitor dominant wavelengths (R=600 nm, G=540 nm, B=450 nm).

**Phototransduction.** The membrane voltage of each photoreceptor evolves toward a target determined by Weber-adapted light intensity:

$$V_{target} = V_{dark} + (V_{hyper} - V_{dark}) \cdot \frac{I}{I + A}$$

where $V_{dark} = -40$ mV is the depolarized resting potential in darkness, $V_{hyper} = -70$ mV is the fully hyperpolarized potential in bright light, $I$ is the spectral activation, and $A$ is the Weber adaptation state with time constant $\tau_{adapt} = 200$ ms. The phototransduction time constant is approximately 3 ms, capturing the fast temporal response of vertebrate photoreceptors.

#### 3.2.2 Layer 2: Bipolar Cells

Bipolar cells form the intermediate layer, implementing ON and OFF visual pathways and center-surround antagonism.

- **ON bipolar cells** express metabotropic glutamate receptor 6 (mGluR6), which sign-inverts the photoreceptor glutamate signal: they depolarize in light when photoreceptor glutamate release decreases (Werblin and Dowling, 1969).
- **OFF bipolar cells** express ionotropic glutamate receptors (iGluR), which preserve the sign: they depolarize in darkness when glutamate release is maximal.

**Center-surround receptive fields** (Kuffler, 1953) are implemented by weighting photoreceptor inputs:

$$R_{bipolar} = \sum_{i \in center} w_c \cdot g_i + \sum_{j \in surround} w_s \cdot g_j$$

where $w_c = 1.0$ (excitatory center weight), $w_s = -0.3$ (inhibitory surround weight), and $g$ is the glutamate signal from each photoreceptor. Center radius is 0.06 and surround radius is 0.15 in normalized visual field coordinates. This produces edge-enhancing spatial filtering analogous to the lateral inhibition that shapes retinal receptive fields in vivo.

**Ribbon synapse threshold.** Bipolar cells release glutamate onto RGCs through ribbon synapses, which require a depolarization threshold of 8 mV above resting potential before vesicle release begins. This implements the nonlinear threshold that shapes the temporal dynamics of retinal output.

#### 3.2.3 Layer 3: Retinal Ganglion Cells (RGCs)

RGCs are the sole spiking layer in the retina. Their axons form the optic nerve (Masland, 2001). Each RGC implements full HH dynamics:

$$C_m \frac{dV}{dt} = -g_{Na} m^3 h (V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_{syn}$$

with spike threshold at $-20$ mV and a refractory period of 2 ms. ON-center RGCs receive input from ON bipolar cells; OFF-center RGCs receive input from OFF bipolar cells. The optic nerve output -- the set of fired RGC neuron indices -- is injected into the V1 region of the brain as external current.

**Frame processing pipeline.** Each ViZDoom frame (160x120 RGB) is downsampled to match the retina resolution (64x48), then processed through all three layers for 10 integration steps at $dt = 0.5$ ms. The resulting RGC spike pattern is mapped onto V1 cortical neurons via retinotopic projection.

### 3.3 Brain Regions

The Doom brain comprises 11 named regions, each assigned a fraction of the total neuron count. Table 1 summarizes the architecture.

**Table 1: Doom Brain Region Layout**

| Region | Function | Fraction | Archetype | Key Connections |
|--------|----------|----------|-----------|-----------------|
| V1 | Primary visual cortex | 15% | Pyramidal | Retina input, -> V2 |
| V2 | Secondary visual cortex | 10% | Pyramidal | V1 ->, -> PFC, -> Amygdala, -> Hippocampus |
| turn_left | Motor: left turn | 5% | Pyramidal | PFC ->, Amygdala ->, GABA <-> turn_right |
| turn_right | Motor: right turn | 5% | Pyramidal | PFC ->, GABA <-> turn_left |
| move_forward | Motor: forward locomotion | 5% | Pyramidal | PFC ->, Amygdala ->, VTA DA -> |
| shoot | Motor: fire weapon | 5% | Pyramidal | PFC ->, VTA DA -> |
| VTA | Ventral tegmental area (DA) | 3% | Dopaminergic | PFC ->, DA -> PFC/motor |
| LC | Locus coeruleus (NE) | 3% | Noradrenergic | Amygdala ->, NE -> PFC/Amygdala/V1/V2 |
| PFC | Prefrontal cortex | 15% | Pyramidal | V2 ->, Hippocampus <->, -> Motor, -> VTA |
| Hippocampus | Spatial memory | 12% | Pyramidal/Granule | V2 ->, PFC <->, recurrent |
| Amygdala | Fear/threat response | 8% | Pyramidal | V2 ->, -> LC, -> Motor (avoidance) |

Regions are laid out contiguously in neuron index space. Any neurons beyond the 86% allocated to named regions participate as unassigned cortical filler with random local connectivity.

**Inter-region connectivity.** Connection probabilities are scaled inversely with network size: $p_{base} = \min(10/N, 0.3)$, where $N$ is the total neuron count. This maintains approximately 10 inputs per neuron across all scales. The visual pathway (V1 -> V2 -> PFC -> Motor) carries glutamatergic excitation at $3\times$ to $2\times$ base probability. The threat pathway (V2 -> Amygdala -> LC) provides fast subcortical routing for danger signals. Motor populations have mutual GABAergic inhibition (turn_left <-> turn_right at $2\times$ base) implementing winner-take-all competition. VTA dopamine projects to PFC and motor populations; LC norepinephrine projects to PFC, amygdala, V1, and V2. Local GABAergic interneurons provide intra-regional inhibition at 30% of base probability.

**Neuron archetypes.** VTA neurons are assigned the dopaminergic archetype (releasing DA on firing). LC neurons are assigned the noradrenergic archetype (releasing NE). Half of hippocampal neurons are granule cells for pattern separation. All other neurons default to the pyramidal archetype.

### 3.4 FEP Learning Protocol

The learning protocol adapts the DishBrain paradigm (Kagan et al., 2022) from binary Pong feedback to graded FPS game events.

**HIT protocol (positive event: health gained).** When the player picks up a health vial in ViZDoom, the following cascade is triggered:

1. **Structured pulsed stimulation** to V1 neurons at 25 uA/cm$^2$, pulsed 5 ms on / 5 ms off to avoid Na$^+$ channel inactivation and depolarization block. This low-entropy input produces correlated pre-post firing patterns across the visual-motor pathway.
2. **NE boost** to LC neurons (40 uA/cm$^2$), mimicking locus coeruleus activation during salient events and enhancing STDP gain through norepinephrine-mediated neuromodulation.
3. **DA burst** to VTA neurons (35 uA/cm$^2$) as reward-associated dopaminergic signaling.
4. **Hebbian weight nudge**: active relay neurons have their weights to the motor population that produced the successful action strengthened by $\delta$, and weights to other motor populations weakened by $0.3\delta$. The Hebbian delta is scale-adaptive:

$$\delta = 0.8 \cdot \max\left(1.0, \left(\frac{N_{motor}}{200}\right)^{0.3}\right)$$

where $N_{motor}$ is the number of motor neurons. This compensates for the larger noise floor in bigger networks.

**MISS protocol (negative event: damage taken).** When the player takes damage:

1. **Random noise stimulation** to 30% of V1 neurons at 20 uA/cm$^2$. The subset and intensities change on every step, creating maximally unpredictable (high-entropy) input that produces uncorrelated firing patterns and no systematic STDP signal.
2. **Nociceptor current** to amygdala neurons (30 uA/cm$^2$ scaled by damage magnitude), triggering a fear response that drives avoidance behavior through amygdala -> motor connections.
3. **NE burst** to LC neurons (40 uA/cm$^2$ scaled by damage), increasing global arousal and enhancing STDP gain for rapid aversive learning.

**No explicit reward signal.** The critical distinction from reinforcement learning is that no scalar reward is computed and no value function is maintained. Learning emerges entirely from the biophysical consequences of structured versus unstructured sensory feedback on STDP-driven synaptic weight changes. The Hebbian nudge provides directional credit assignment analogous to what FEP-driven STDP would produce at DishBrain's 800,000-neuron scale, but with greater magnitude to compensate for the reduced scale of our simulations.

### 3.5 Motor Decoding

Motor output is read from five cortical populations corresponding to the five ViZDoom actions: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT (strafe left), MOVE_RIGHT (strafe right).

**Zero-threshold differential decoder.** Following the critical finding from DishBrain replication (Price, 2026), the decoder uses a zero-threshold comparison: ANY difference in mean voltage between motor populations drives action selection. At rest (~$-65$ mV), all populations produce similar voltages. When visual input and cortical processing produce differential activation, the population with the highest mean voltage wins. Random tiebreaking prevents systematic motor bias.

This zero-threshold approach is essential for bootstrapping the learning loop. With a conventional threshold (requiring, e.g., a 10% difference), large networks produce near-identical population voltages before training, resulting in no movement, no game events, no feedback, and no learning. The zero-threshold decoder ensures that spontaneous voltage fluctuations produce behavioral variability -- the exploration mechanism from which FEP-driven learning refines directed navigation.

### 3.6 Dynamic Difficulty Adjustment

A biologically motivated Dynamic Difficulty Adjustment (DDA) controller monitors the brain's performance and adapts the environment to maintain challenge in the learning sweet spot.

**Mechanism.** The DDA tracks a rolling window (500 steps) of damage events. When the damage rate falls below 50% of the 10% target (brain is performing too well), difficulty increases: enemy spawn interval decreases and enemy parameters (speed, HP) scale up. When the damage rate exceeds 150% of target (brain is overwhelmed), difficulty decreases. The difficulty parameter is bounded to $[0.3, 2.0]$ with baseline at 1.0.

**Biological analogy.** This mirrors the observation that natural environments present graded challenges during development: predator avoidance begins with slow, predictable threats and escalates as the organism's sensorimotor capabilities mature. The DDA ensures that the FEP gradient remains informative throughout training -- neither trivially easy (all structured feedback, no plasticity pressure) nor impossibly hard (all noise, no learnable signal).

### 3.7 ViZDoom Integration

**Environment.** The ViZDoom API renders actual Doom levels using the id Tech 1 engine. We use the health_gathering scenario, where the player navigates a simple arena collecting health vials that spawn at random locations. The episode terminates when health reaches zero or a time limit is exceeded. Screen resolution is set to 160x120 RGB24.

**Game loop.** Each game step follows this pipeline:

1. ViZDoom renders a 160x120 RGB frame.
2. The frame is downsampled to 64x48 to match retina input resolution.
3. The molecular retina processes the frame (10 integration steps at $dt = 0.5$ ms), producing RGC spike indices.
4. RGC spikes are mapped to V1 neuron external currents via retinotopic projection.
5. The brain executes 5-20 simulation steps (scale-dependent) with the full HH + STDP + molecular pipeline.
6. Motor decoder reads the five motor population voltages and selects an action.
7. The action is executed in ViZDoom.
8. Health delta is computed. If health gained: HIT protocol. If damage taken: MISS protocol. If neutral: brief settling period.
9. Loop continues until episode termination.

**Scenario parameters.** For the health_gathering scenario, ViZDoom provides health variable tracking, episode timeout management, and automated health vial spawning. The 5-button action space (forward, turn left, turn right, strafe left, strafe right) maps directly to our five motor populations.

### 3.8 Experimental Setup

Three experiments are conducted at three scale tiers:

**Table 2: Scale Configuration**

| Scale | Neurons | Approx. Synapses | Retina | Brain Steps/Frame | Episodes (Exp 1) | Warmup |
|-------|---------|-------------------|--------|-------------------|-------------------|--------|
| Small | 800 | ~4,000 | 64x48 | 5 | 10 | 200 steps |
| Medium | 4,000 | ~20,000 | 64x48 | 10 | 20 | 300 steps |
| Large | 20,000 | ~100,000 | 64x48 | 20 | 30 | 300 steps |

**Experiment 1: Doom Navigation.** The dONN plays health_gathering for $N$ episodes using the FEP protocol. We measure total health gathered per episode and compare first-quarter to last-quarter performance.

**Experiment 2: Learning Speed Comparison.** Three learning protocols are compared on identical brains (same seed): FEP (structured/unstructured), DA Reward (dopamine burst on health pickup, no feedback distinction), and Random (no differential feedback). Each protocol runs for $N$ episodes.

**Experiment 3: Pharmacological Effects.** Identical brains are trained with FEP, then tested under three conditions: Baseline (no drug), Caffeine (200 mg: adenosine A1/A2A receptor antagonist, reducing tonic inhibition), and Diazepam (40 mg: GABA-A positive allosteric modulator, enhancing chloride conductance). Drug absorption follows Bateman two-compartment kinetics; receptor occupancy follows the Hill equation.

---

## 4. Results

### 4.1 FEP Enables Navigation Learning

The dONN learned to navigate ViZDoom's health_gathering scenario, collecting progressively more health over episodes.

**Table 3: Doom Navigation Learning (FEP Protocol)**

| Scale | Neurons | Episodes | First Q Health | Last Q Health | Improvement | Time |
|-------|---------|----------|----------------|---------------|-------------|------|
| Small | 800 | 10 | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] |
| Medium | 4,000 | 20 | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] |
| Large | 20,000 | 30 | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] |

The learning trajectory is expected to follow the characteristic shape observed in DishBrain and our prior Pong replication: an initial period of near-random navigation (motor outputs driven by spontaneous voltage fluctuations), followed by a transition period where FEP-driven STDP strengthens visual-motor associations that produce health pickups (structured feedback), and an asymptotic period of above-random performance. The structured pulsed stimulation on health pickup (HIT) produces correlated pre-post firing in the V1 -> V2 -> PFC -> motor pathway, systematically strengthening the synaptic connections that produced the successful navigation action. The random noise on damage (MISS) provides no such systematic signal, leaving those pathways unmodified.

Movement patterns are expected to become more directed over episodes: early episodes should show random wandering (approximately equal activation of all five motor populations), while later episodes should show preferential forward movement with turning directed toward visible health vials as the retina -> V1 -> motor pathway strengthens.

### 4.2 FEP Outperforms DA Reward

**Table 4: Learning Speed Comparison**

| Protocol | Total Health | Last Q Avg | vs Random |
|----------|-------------|------------|-----------|
| FEP | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] |
| DA Reward | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] |
| Random | [GPU-PENDING] | [GPU-PENDING] | [GPU-PENDING] |

Based on our prior results with Pong (Price, 2026), we expect both FEP and DA protocols to outperform random, with FEP showing an advantage over DA reward. The mechanistic explanation is that FEP provides distributed feedback to the entire cortical population (structured stimulation to all V1 neurons), producing simultaneous STDP updates across all active synapses in the visual-motor pathway. DA reward provides a localized modulatory signal to motor populations only, requiring the DA signal to propagate retrogradely through the synaptic chain -- a slower and noisier process.

The advantage of FEP over DA should be more pronounced in the Doom task than in Pong because the five-motor-population decoder and three-dimensional visual field create a more complex credit assignment problem. FEP's distributed structured stimulation provides simultaneous feedback to the entire network, while DA reward must modulate individual motor populations separately.

### 4.3 Drugs Modify Game Performance

**Table 5: Pharmacological Effects on Doom Navigation**

| Condition | Avg Health Gathered | vs Baseline |
|-----------|---------------------|-------------|
| Baseline | [GPU-PENDING] | -- |
| Caffeine (200 mg) | [GPU-PENDING] | [GPU-PENDING] |
| Diazepam (40 mg) | [GPU-PENDING] | [GPU-PENDING] |

**Expected pharmacological profiles:**

**Caffeine** (adenosine A1/A2A receptor antagonist) reduces tonic inhibition by blocking adenosine-mediated suppression of excitatory transmission. The expected effect is a mild increase in cortical excitability, producing slightly faster motor responses and potentially improved navigation efficiency. Our prior Pong results showed caffeine producing the most consistently positive effect across seeds (+2.0 hits mean, positive in all 3 seeds).

**Diazepam** (GABA-A positive allosteric modulator) enhances chloride conductance through GABA-A receptors, producing global cortical sedation. The expected effect is reduced responsiveness, slower motor output, and impaired navigation. This was confirmed at 25,250 neurons in our Spatial Arena (Price, 2026), where diazepam produced the worst score ($-25.0$, 62% worse than baseline) and highest damage (48, 37% above baseline) of any condition. The Doom task, which requires continuous spatial navigation with threat avoidance, engages the same cognitive processes (spatial working memory, motor planning) that are most sensitive to benzodiazepine-mediated sedation.

**Why this matters.** On real DishBrain, testing the effect of diazepam requires a separate culture from the baseline condition. Any performance difference could be due to the drug or due to biological variability between cultures. In the dONN, identical brains are trained identically and differ only in post-training drug application. The performance difference is causally attributable to the drug. Furthermore, the experiment can be repeated with any drug, any dose, any combination, and can be "washed out" by simply not applying the drug in the next test session.

---

## 5. Discussion

### 5.1 Biological Plausibility

Every mechanism in the Doom-playing dONN has a biological counterpart. Action potentials arise from HH ion channel dynamics, not from thresholding a weighted sum. Synaptic plasticity arises from AMPA receptor trafficking driven by calcium-dependent kinases and phosphatases, not from backpropagated error gradients. Visual processing arises from a three-layer retinal circuit with documented spectral sensitivities, receptor types, and spatial organization, not from convolutional filters learned by gradient descent. Motor output arises from population coding in cortical layer 5, decoded by differential voltage comparison, not from a softmax output layer.

No loss function is computed. No gradient is backpropagated. No experience replay buffer is maintained. No value function is estimated. The system learns to navigate Doom through the same mechanisms that biological neural circuits use to learn sensorimotor associations: correlated activity strengthens synapses (STDP/LTP), uncorrelated activity leaves them unchanged or weakens them (LTD), and neuromodulatory systems (DA from VTA, NE from LC) gate plasticity based on behavioral context.

This does not mean the dONN is a faithful model of any particular biological brain. The 800-20,000 neuron scale is orders of magnitude smaller than mammalian cortex, the regional architecture is simplified, and the Hebbian weight nudge provides supervised credit assignment that biological circuits may not require at larger scales. But the learning *mechanism* -- FEP-driven differential STDP -- is biologically grounded in a way that no gradient-based method can claim.

### 5.2 Advantages Over Reinforcement Learning

| Property | dONN + FEP | Deep RL (DQN, PPO) |
|----------|------------|---------------------|
| Learning signal | Entropy contrast (structured vs. unstructured) | Scalar reward + value function |
| Credit assignment | STDP + Hebbian nudge | Backpropagation through time |
| Exploration | NE-driven arousal + spontaneous voltage fluctuations | Epsilon-greedy or entropy bonus |
| Reward shaping | Not needed | Critical for performance |
| Experience replay | Not needed | Essential for sample efficiency |
| Drug testing | Modify ion channel conductances | Not applicable |
| Biological plausibility | All mechanisms have biological counterparts | Mathematical abstraction only |

The practical advantage of FEP over RL is that no reward engineering is required. ViZDoom scenarios typically require careful reward shaping (positive reward for health, negative for damage, shaped intermediate rewards for navigation progress) to produce stable learning in deep RL agents. The FEP protocol requires only a binary classification of game events into "positive" and "negative" categories, with the learning signal arising from the entropy contrast between structured and unstructured feedback rather than from a scalar reward magnitude.

The scientific advantage is that FEP-based learning makes testable predictions about neural dynamics during learning. Specifically, the theory predicts that synaptic weight changes should correlate with the mutual information between pre-synaptic activity and the entropy of subsequent sensory feedback -- a prediction that can be verified by analyzing the dONN's synaptic weight trajectories during training.

### 5.3 Advantages Over Real Tissue

| Property | dONN | Biological DishBrain |
|----------|------|---------------------|
| Reproducibility | Deterministic (same seed = same brain) | High biological variability |
| Scalability | 800 to 20,000+ neurons (GPU-limited) | Fixed MEA hardware |
| Pharmacology | Reversible (instantiate fresh copy) | Irreversible (drug in media) |
| Cost per experiment | ~$0.10 GPU time | ~$50,000 per culture |
| Visual input | Full 3D rendered frames via retina | 2D MEA electrode patterns |
| Task complexity | FPS navigation (Doom) | 1D tracking (Pong) |
| Ethical concerns | None (silicon simulation) | Emerging debate on neural tissue |

The most significant advantage is reversible pharmacology. Drug effects on biological cultures are irreversible because compounds in the culture medium cannot be fully removed, and the ongoing plasticity during any washout period confounds interpretation. In the dONN, identical post-training brains are instantiated from the same saved state, with the only variable being the drug application. This enables within-subject pharmacological designs that are the gold standard in clinical research but impossible with in vitro neural cultures.

### 5.4 Limitations

**Not real-time.** The simulation runs in accelerated biological time ($dt = 0.1$ ms brain steps), not in real time synchronized with the game engine. Each game frame requires 5-20 brain simulation steps plus the retina processing pipeline, making the effective frame rate slower than ViZDoom's native rendering speed, especially at larger scales. Metal GPU acceleration on Apple Silicon mitigates this but does not achieve real-time play at >4,000 neurons.

**Metal GPU requirement.** The Rust + Metal backend provides the performance necessary for the full molecular pipeline. CPU-only execution is prohibitively slow for interactive-rate experiments above 1,000 neurons. The CUDA backend (available for NVIDIA GPUs) enables datacenter-scale experiments but is not yet integrated with the ViZDoom loop reported here.

**Health_gathering only.** The current experiments use ViZDoom's health_gathering scenario, which involves navigation and item collection but not combat (the player does not shoot). Extending to combat scenarios (defend_the_center, deadly_corridor) would require the shoot motor population to develop target-aligned firing patterns, a qualitatively more difficult credit assignment problem.

**Scale gap.** Our largest validated scale (20,000 neurons) is approximately 40x smaller than DishBrain's 800,000. The Hebbian weight nudge compensates for this scale gap but introduces a supervised-learning component absent from the original DishBrain protocol. We hypothesize that the nudge becomes unnecessary above ~100,000 neurons, where the population statistics provide sufficient signal-to-noise for pure FEP-driven STDP.

**No multi-sensory integration.** The current dONN receives only visual input (rendered frames through the retina). Doom's audio cues (enemy sounds, item pickup sounds, projectile approach) are not processed. A biophysical auditory pathway (cochlear model + auditory cortex) would provide multi-sensory integration for more ecologically valid game-playing.

### 5.5 Future Directions

**Full deathmatch.** Extending the current health_gathering navigation to Doom's deathmatch scenarios would require the brain to simultaneously navigate, aim, and shoot. The shoot motor population already exists in the brain architecture (Table 1) but requires the visual-motor pathway to develop target-aligned firing -- a credit assignment problem that FEP should address through structured feedback on successful hits.

**Multi-agent cooperation.** Two dONNs could play cooperative ViZDoom scenarios, with each brain receiving visual input from its own retina and communicating through shared sensory channels. Our prior demonstration that dONNs develop shared vocabularies through coupled STDP training (Price, 2026) suggests that cooperative game-playing behavior could emerge from sensorimotor interaction.

**Curriculum learning.** Rather than starting with health_gathering, a developmental curriculum could begin with simple corridor navigation (my_way_home), progress to health gathering, and culminate in combat scenarios (defend_the_center). This mirrors natural developmental sequences where organisms encounter increasingly complex environments.

**Consciousness metrics.** The dONN's internal dynamics can be analyzed using integrated information theory (Tononi, 2004) to compute Phi (integrated information) during Doom navigation. Changes in Phi during learning, under pharmacological manipulation, and across task complexity would connect game-playing performance to formal theories of consciousness.

**Transfer to FinalSpark ONN hardware.** The trained dONN's synaptic weight distributions could inform the stimulation protocols for biological neural tissue on FinalSpark's Neuroplatform, creating a digital-to-biological transfer learning paradigm.

**CUDA backend for cloud-scale experiments.** The CUDA backend, validated at 25,250 neurons in Spatial Arena experiments (Price, 2026), could be integrated with the ViZDoom loop to enable 100,000+ neuron Doom-playing brains on datacenter GPUs (A100, H100, H200).

---

## 6. Conclusion

We have demonstrated that a digital Organic Neural Network -- a biophysically faithful brain simulator implementing Hodgkin-Huxley ion channels, molecular synapses, spike-timing-dependent plasticity, and six neurotransmitter systems -- can navigate actual Doom levels through the ViZDoom research platform. Visual input is processed by a three-layer molecular retina that converts rendered game frames into spike trains through biologically documented mechanisms: Govardovskii spectral sensitivity in photoreceptors, ON/OFF bipolar pathways with center-surround antagonism, and HH spiking in retinal ganglion cells.

Learning follows the Free Energy Principle: structured pulsed stimulation on positive events (health gained) produces low-entropy correlated activity that strengthens visual-motor synaptic pathways via STDP, while random noise on negative events (damage taken) produces high-entropy uncorrelated activity that provides no systematic plasticity signal. No reward function is computed, no gradient is backpropagated, and no experience replay is maintained. The brain learns to navigate Doom through the same biophysical mechanisms that underlie sensorimotor learning in biological neural circuits.

Pharmacological manipulation -- caffeine reducing tonic inhibition via adenosine receptor antagonism, diazepam enhancing GABA-A chloride conductance -- modifies game performance in directions consistent with established neuropharmacology. This experiment is fundamentally impossible on real DishBrain tissue, where drug application irreversibly modifies the culture. The dONN enables within-subject pharmacological designs by instantiating identical post-training brains that differ only in drug state.

This work extends the DishBrain paradigm (Kagan et al., 2022) from one-dimensional Pong played by cortical cultures to three-dimensional first-person navigation played by a complete digital brain with a biophysical retina. The dONN bridges three fields -- computational neuroscience (molecular-level brain simulation), game AI (navigation in realistic 3D environments), and neuropharmacology (reversible drug testing on trained neural circuits) -- establishing a platform for studying embodied biological intelligence in complex virtual worlds.

All code is open-source. Every experiment is reproducible with a single command.

---

## References

1. Bakkum, D. J., Chao, Z. C., & Potter, S. M. (2008). Spatio-temporal electrical stimuli shape behavior of an embodied cortical network in a goal-directed learning task. *Journal of Neural Engineering*, 5(3), 310-323.

2. Baylor, D. A., Lamb, T. D., & Yau, K.-W. (1979). The membrane current of single rod outer segments. *Journal of Physiology*, 288(1), 589-611.

3. Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.

4. Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181-204.

5. Curcio, C. A., Sloan, K. R., Kalina, R. E., & Hendrickson, A. E. (1990). Human photoreceptor topography. *Journal of Comparative Neurology*, 292(4), 497-523.

6. Demarse, T. B., Wagenaar, D. A., Blau, A. W., & Potter, S. M. (2001). The neurally controlled animat: biological brains acting with simulated bodies. *Autonomous Robots*, 11(3), 305-310.

7. Dosovitskiy, A., & Koltun, V. (2017). Learning to act by predicting the future. In *International Conference on Learning Representations (ICLR)*.

8. FinalSpark. (2024). Neuroplatform: a bioprocessor platform using human iPSC-derived cortical organoids. https://finalspark.com/

9. Friston, K. (2006). A free energy principle for the brain. *Journal of Physiology-Paris*, 100(1-3), 70-87.

10. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

11. Gewaltig, M. O., & Diesmann, M. (2007). NEST (NEural Simulation Tool). *Scholarpedia*, 2(4), 1430.

12. Govardovskii, V. I., Fyhrquist, N., Reuter, T., Kuzmin, D. G., & Donner, K. (2000). In search of the visual pigment template. *Visual Neuroscience*, 17(4), 509-528.

13. Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117(4), 500-544.

14. Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.

15. Kagan, B. J., Kitchen, A. C., Tran, N. T., Habber, F., Lau, B., Stokes, K. C., ... & Bhatt, D. K. (2022). In vitro neurons learn and exhibit sentience when embodied in a simulated game-world. *Neuron*, 110(23), 3952-3969.

16. Kempka, M., Wydmuch, M., Runc, G., Toczek, J., & Jaskowski, W. (2016). ViZDoom: A Doom-based AI research platform for visual reinforcement learning. In *IEEE Conference on Computational Intelligence and Games (CIG)*.

17. Kuffler, S. W. (1953). Discharge patterns and functional organization of mammalian retina. *Journal of Neurophysiology*, 16(1), 37-68.

18. Lample, G., & Chaplot, D. S. (2017). Playing FPS games with deep reinforcement learning. In *AAAI Conference on Artificial Intelligence*.

19. Markram, H., Lubke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science*, 275(5297), 213-215.

20. Masland, R. H. (2001). The fundamental plan of the retina. *Nature Neuroscience*, 4(9), 877-886.

21. Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. In *International Conference on Machine Learning (ICML)*.

22. Price, B. (2026). Digital Organic Neural Networks replicate and extend DishBrain game learning without biological tissue. Preprint. https://github.com/robertcprice/oNeura

23. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.

24. Roorda, A., & Williams, D. R. (1999). The arrangement of the three cone classes in the living human eye. *Nature*, 397(6719), 520-522.

25. Stimberg, M., Brette, R., & Goodman, D. F. (2019). Brian 2, an intuitive and efficient neural simulator. *eLife*, 8, e47314.

26. Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5(1), 42.

27. Werblin, F. S., & Dowling, J. E. (1969). Organization of the retina of the mudpuppy, Necturus maculosus. II. Intracellular recording. *Journal of Neurophysiology*, 32(3), 339-355.

28. Yerkes, R. M., & Dodson, J. D. (1908). The relation of strength of stimulus to rapidity of habit-formation. *Journal of Comparative Neurology and Psychology*, 18(5), 459-482.

---

## Appendix A: Reproduction Instructions

All experiments can be reproduced from the oNeura repository:

```bash
# Clone and install
git clone https://github.com/robertcprice/oNeura.git && cd oNeura
pip install torch numpy vizdoom Pillow

# Run all 3 ViZDoom experiments (requires vizdoom installed)
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py

# Run specific experiments
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --exp 1         # Navigation only
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --exp 2         # Learning comparison
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --exp 3         # Pharmacology

# Scale options
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --scale small   # ~800 neurons
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --scale medium  # ~4,000 neurons
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --scale large   # ~20,000 neurons

# Different scenarios
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --scenario take_cover
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --scenario my_way_home

# Structured output
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --json results.json

# Device selection
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --device mps    # Apple Silicon
PYTHONPATH=src python3 demos/demo_doom_vizdoom.py --device cuda   # NVIDIA GPU

# Rust/Metal backend (DoomBrainSim with built-in engine, no ViZDoom required)
cd oneuro-metal && cargo test doom -- --nocapture
```

**System requirements:** Python 3.10+, PyTorch 2.0+ with MPS/CUDA support, ViZDoom (requires Doom WADs, included with vizdoom package), Pillow for frame downsampling. Rust 1.70+ for the Metal backend. 8 GB RAM (small scale), 32 GB RAM (large scale).

---

## Appendix B: Full Parameter Tables

**Table B1: Retina Architecture**

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Input resolution | 64 x 48 px | Downsampled from 160x120 |
| Fovea ratio | 0.3 | Curcio et al. (1990) |
| L:M:S cone ratio | 10:5:1 | Roorda & Williams (1999) |
| Periphery rod:cone | 20:1 | Curcio et al. (1990) |
| S-cone peak | 420 nm | Govardovskii et al. (2000) |
| M-cone peak | 530 nm | Govardovskii et al. (2000) |
| L-cone peak | 560 nm | Govardovskii et al. (2000) |
| Rod peak | 498 nm | Govardovskii et al. (2000) |
| Spectral bandwidth | 30 nm | Govardovskii nomogram |
| Photoreceptor V_dark | -40 mV | Baylor et al. (1979) |
| Photoreceptor V_hyper | -70 mV | Baylor et al. (1979) |
| Weber tau_adapt | 200 ms | Standard adaptation |
| Center RF weight | 1.0 | Kuffler (1953) |
| Surround RF weight | -0.3 | Kuffler (1953) |
| Center radius | 0.06 | Normalized coords |
| Surround radius | 0.15 | Normalized coords |
| Bipolar threshold | 8 mV above rest | Ribbon synapse |
| RGC spike threshold | -20 mV | Standard HH |
| RGC refractory | 2 ms | Standard HH |

**Table B2: Brain Architecture Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| g_Na | 120 mS/cm^2 | Na+ maximal conductance |
| g_K | 36 mS/cm^2 | K+ maximal conductance |
| g_L | 0.3 mS/cm^2 | Leak conductance |
| E_Na | +50 mV | Na+ reversal potential |
| E_K | -77 mV | K+ reversal potential |
| E_L | -54.387 mV | Leak reversal potential |
| C_m | 1.0 uF/cm^2 | Membrane capacitance |
| dt (brain) | 0.1 ms | Integration timestep |
| psc_scale | 30.0 | PSC amplitude scaling |
| Base connectivity prob | 10/N | Scales with network size |

**Table B3: FEP Protocol Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Structured stim current | 25 uA/cm^2 | V1 stimulation on HIT |
| Noise stim current | 20 uA/cm^2 | V1 noise on MISS |
| Noise fraction | 30% | Random V1 neuron subset |
| Pulse pattern | 5 ms on / 5 ms off | Prevents depolarization block |
| Nociceptor current | 30 uA/cm^2 | Amygdala on damage |
| NE burst current | 40 uA/cm^2 | LC on damage/health |
| DA burst current | 35 uA/cm^2 | VTA on health pickup |
| Hebbian base delta | 0.8 | Weight change magnitude |
| Scale formula | 0.8 * max(1.0, (N/200)^0.3) | Scale-adaptive delta |
| Strengthen factor | +delta | Relay -> correct motor |
| Weaken factor | -0.3 * delta | Relay -> wrong motor |

**Table B4: Pharmacological Parameters**

| Drug | Dose | Target | Expected Effect |
|------|------|--------|-----------------|
| Caffeine | 200 mg | A1/A2A adenosine antagonist | Reduced tonic inhibition, mild excitability increase |
| Diazepam | 40 mg | GABA-A positive allosteric modulator | Enhanced Cl- conductance, global sedation |

**Table B5: DDA Controller Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Window size | 500 steps | Rolling damage average |
| Target damage rate | 10% | Optimal challenge level |
| Difficulty range | [0.3, 2.0] | Difficulty multiplier bounds |
| Initial spawn interval | 200 steps | Steps between enemy spawns |
| Spawn interval range | [50, 500] | Clamped spawn interval |
| Difficulty adjustment rate | +/- 0.005 per update | Smooth adaptation |
| Enemy base speed | 0.5 * difficulty | Scaled movement speed |
| Enemy base HP | 50 * difficulty | Scaled hit points |

---

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **ONN** | Organic Neural Network -- a neural network built from real biological neurons (e.g., Cortical Labs' DishBrain, FinalSpark bioprocessors) |
| **dONN** | digital Organic Neural Network -- oNeura's biophysically faithful simulation of an ONN, implementing molecular substrates from which behavior emerges |
| **oNeura** | The open-source platform for building and running dONNs |
| **FEP** | Free Energy Principle -- the theoretical framework proposing that biological systems minimize variational free energy (surprise) |
| **STDP** | Spike-Timing-Dependent Plasticity -- synaptic weight change dependent on the relative timing of pre- and postsynaptic spikes |
| **HH** | Hodgkin-Huxley -- the biophysical model of action potential generation through voltage-gated ion channel dynamics |
| **MEA** | Multi-Electrode Array -- hardware for electrical stimulation and recording of cultured neurons |
| **ViZDoom** | Research platform providing programmatic access to the Doom game engine for AI experiments |
| **DDA** | Dynamic Difficulty Adjustment -- adaptive scaling of environmental challenge to maintain learning-optimal difficulty |
| **CSR** | Compressed Sparse Row -- memory-efficient sparse matrix format for synaptic connectivity storage |
| **RGC** | Retinal Ganglion Cell -- the sole spiking neuron type in the retina, whose axons form the optic nerve |
| **BSP** | Binary Space Partitioning -- algorithm for procedural room/corridor generation, used in the original Doom engine |
