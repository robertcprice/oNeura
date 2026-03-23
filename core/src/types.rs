//! Core type definitions for the molecular brain simulator.
//!
//! All enums and type aliases used across the crate.

/// Neuron archetype — determines channel complement and receptor profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NeuronArchetype {
    Pyramidal = 0,
    Interneuron = 1,
    Purkinje = 2,
    Granule = 3,
    Stellate = 4,
    MediumSpiny = 5,
    DopaminergicSN = 6,
    Serotonergic = 7,
    Cholinergic = 8,
}

/// Ion channel type — the 8 channel families in a molecular neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IonChannelType {
    Nav = 0,   // Voltage-gated sodium
    Kv = 1,    // Voltage-gated potassium
    Kleak = 2, // Potassium leak
    Cav = 3,   // Voltage-gated calcium
    NMDA = 4,  // NMDA receptor channel
    AMPA = 5,  // AMPA receptor channel
    GabaA = 6, // GABA-A receptor channel
    NAChR = 7, // Nicotinic acetylcholine receptor
}

impl IonChannelType {
    pub const COUNT: usize = 8;

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Neurotransmitter type — the 6 major NTs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NTType {
    Dopamine = 0,
    Serotonin = 1,
    Norepinephrine = 2,
    Acetylcholine = 3,
    GABA = 4,
    Glutamate = 5,
}

impl NTType {
    pub const COUNT: usize = 6;

    pub fn index(self) -> usize {
        self as usize
    }

    /// Resting ambient concentration in nM.
    pub fn resting_conc_nm(self) -> f32 {
        match self {
            NTType::Dopamine => 20.0,
            NTType::Serotonin => 10.0,
            NTType::Norepinephrine => 15.0,
            NTType::Acetylcholine => 50.0,
            NTType::GABA => 200.0,
            NTType::Glutamate => 500.0,
        }
    }

    /// Half-life for cleft clearance in ms.
    pub fn half_life_ms(self) -> f32 {
        match self {
            NTType::Dopamine => 200.0,
            NTType::Serotonin => 350.0,
            NTType::Norepinephrine => 250.0,
            NTType::Acetylcholine => 2.0,
            NTType::GABA => 100.0,
            NTType::Glutamate => 50.0,
        }
    }
}

/// Receptor type — ionotropic and metabotropic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ReceptorType {
    // Ionotropic (fast)
    AMPA = 0,
    NMDA = 1,
    GabaA = 2,
    NAChR = 3,
    // Metabotropic (slow)
    D1 = 4,
    D2 = 5,
    HT1A = 6,
    HT2A = 7,
    MAChRM1 = 8,
    GabaB = 9,
    Alpha1 = 10,
    Alpha2 = 11,
}

impl ReceptorType {
    /// Whether this receptor directly gates an ion channel.
    pub fn is_ionotropic(self) -> bool {
        matches!(
            self,
            ReceptorType::AMPA | ReceptorType::NMDA | ReceptorType::GabaA | ReceptorType::NAChR
        )
    }

    /// The ion channel type gated by this ionotropic receptor (if any).
    pub fn channel_type(self) -> Option<IonChannelType> {
        match self {
            ReceptorType::AMPA => Some(IonChannelType::AMPA),
            ReceptorType::NMDA => Some(IonChannelType::NMDA),
            ReceptorType::GabaA => Some(IonChannelType::GabaA),
            ReceptorType::NAChR => Some(IonChannelType::NAChR),
            _ => None,
        }
    }

    /// EC50 in nM for this receptor.
    pub fn ec50_nm(self) -> f32 {
        match self {
            ReceptorType::AMPA => 480.0,
            ReceptorType::NMDA => 2400.0,
            ReceptorType::GabaA => 200.0,
            ReceptorType::NAChR => 30.0,
            ReceptorType::D1 => 2340.0,
            ReceptorType::D2 => 2.8,
            ReceptorType::HT1A => 3.2,
            ReceptorType::HT2A => 54.0,
            ReceptorType::MAChRM1 => 7900.0,
            ReceptorType::GabaB => 35.0,
            ReceptorType::Alpha1 => 330.0,
            ReceptorType::Alpha2 => 56.0,
        }
    }

    /// Hill coefficient for this receptor.
    pub fn hill_n(self) -> f32 {
        match self {
            ReceptorType::AMPA => 1.3,
            ReceptorType::NMDA => 1.5,
            ReceptorType::GabaA => 2.0,
            ReceptorType::NAChR => 1.8,
            ReceptorType::D1
            | ReceptorType::D2
            | ReceptorType::HT1A
            | ReceptorType::MAChRM1
            | ReceptorType::Alpha1
            | ReceptorType::Alpha2 => 1.0,
            ReceptorType::HT2A => 1.2,
            ReceptorType::GabaB => 1.5,
        }
    }

    /// G-protein cascade effect for metabotropic receptors.
    pub fn cascade_effect(self) -> CascadeEffect {
        match self {
            ReceptorType::D1 => CascadeEffect::CampIncrease,
            ReceptorType::D2 | ReceptorType::HT1A | ReceptorType::Alpha2 => {
                CascadeEffect::CampDecrease
            }
            ReceptorType::HT2A | ReceptorType::MAChRM1 | ReceptorType::Alpha1 => {
                CascadeEffect::IP3DagIncrease
            }
            ReceptorType::GabaB => CascadeEffect::KChannelOpen,
            _ => CascadeEffect::None,
        }
    }
}

/// Second messenger cascade effect from metabotropic receptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeEffect {
    None,
    CampIncrease,
    CampDecrease,
    IP3DagIncrease,
    KChannelOpen,
}

/// Drug type for pharmacology engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum DrugType {
    Fluoxetine = 0,  // SSRI
    Diazepam = 1,    // Benzodiazepine
    Caffeine = 2,    // Adenosine antagonist
    Amphetamine = 3, // DA/NE releaser
    LDOPA = 4,       // DA precursor
    Donepezil = 5,   // AChE inhibitor
    Ketamine = 6,    // NMDA antagonist
}

/// Brain region types for RegionalBrain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BrainRegionType {
    CorticalColumn = 0,
    ThalamicNucleus = 1,
    Hippocampus = 2,
    BasalGanglia = 3,
}

/// Gene IDs for gene expression pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GeneId {
    GRIN1 = 0,
    GRIN2A = 1,
    GRIA1 = 2,
    GRIA2 = 3,
    GABRA1 = 4,
    SCN1A = 5,
    KCNA1 = 6,
    CACNA1A = 7,
    CHRNA4 = 8,
    ACHE = 9,
    GAD1 = 10,
    BDNF = 11,
    FOS = 12,
    ARC = 13,
    ZIF268 = 14,
    NFKB = 15,
}

/// Transcription factor types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TranscriptionFactorType {
    CREB = 0,
    CFos = 1,
    Arc = 2,
    Zif268 = 3,
    NFkB = 4,
}
