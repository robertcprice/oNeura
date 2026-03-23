use openentropy_core::EntropyPool;
use rand::rngs::OsRng;
use rand::RngCore;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TerrariumSeedSource {
    #[default]
    Explicit,
    OsRandom,
    OpenEntropyMixed,
    CheckpointRestore,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumSeedProvenance {
    pub seed: u64,
    pub source: TerrariumSeedSource,
    pub source_label: String,
    pub os_entropy_bytes: usize,
    pub openentropy_total_sources: Option<usize>,
    pub openentropy_healthy_sources: Option<usize>,
    pub openentropy_raw_bytes: Option<u64>,
    pub openentropy_output_bytes: Option<u64>,
    pub openentropy_conditioning: Option<String>,
}

fn fold_seed_material(bytes: &[u8; 32]) -> u64 {
    let mut lanes = [0u64; 4];
    for (lane, chunk) in lanes.iter_mut().zip(bytes.chunks_exact(8)) {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        *lane = u64::from_le_bytes(arr);
    }
    lanes[0] ^ lanes[1].rotate_left(17) ^ lanes[2].rotate_left(33) ^ lanes[3].rotate_left(49)
}

pub fn explicit_seed_provenance(
    seed: u64,
    source_label: impl Into<String>,
) -> TerrariumSeedProvenance {
    TerrariumSeedProvenance {
        seed,
        source: TerrariumSeedSource::Explicit,
        source_label: source_label.into(),
        ..TerrariumSeedProvenance::default()
    }
}

pub fn checkpoint_seed_provenance(
    seed: u64,
    source_label: impl Into<String>,
) -> TerrariumSeedProvenance {
    TerrariumSeedProvenance {
        seed,
        source: TerrariumSeedSource::CheckpointRestore,
        source_label: source_label.into(),
        ..TerrariumSeedProvenance::default()
    }
}

pub fn natural_seed_provenance(source_label: impl Into<String>) -> TerrariumSeedProvenance {
    let mut os_bytes = [0u8; 32];
    OsRng.fill_bytes(&mut os_bytes);
    let mut mixed = os_bytes;
    let mut provenance = TerrariumSeedProvenance {
        source: TerrariumSeedSource::OsRandom,
        source_label: source_label.into(),
        os_entropy_bytes: os_bytes.len(),
        ..TerrariumSeedProvenance::default()
    };

    let pool = EntropyPool::auto();
    if pool.source_count() > 0 {
        let openentropy_bytes = pool.get_random_bytes(32);
        let health = pool.health_report();
        provenance.openentropy_total_sources = Some(health.total);
        provenance.openentropy_healthy_sources = Some(health.healthy);
        provenance.openentropy_raw_bytes = Some(health.raw_bytes);
        provenance.openentropy_output_bytes = Some(health.output_bytes);
        provenance.openentropy_conditioning = Some("sha256".to_string());
        if health.healthy > 0 && openentropy_bytes.len() >= mixed.len() {
            for (dst, src) in mixed.iter_mut().zip(openentropy_bytes.iter().copied()) {
                *dst ^= src;
            }
            provenance.source = TerrariumSeedSource::OpenEntropyMixed;
        }
    }

    provenance.seed = fold_seed_material(&mixed);
    provenance
}

pub fn resolve_seed_provenance(
    explicit_seed: Option<u64>,
    source_label: impl Into<String>,
) -> TerrariumSeedProvenance {
    let source_label = source_label.into();
    explicit_seed
        .map(|seed| explicit_seed_provenance(seed, source_label.clone()))
        .unwrap_or_else(|| natural_seed_provenance(source_label))
}

impl super::TerrariumWorld {
    pub fn set_seed_provenance(&mut self, provenance: TerrariumSeedProvenance) {
        self.seed_provenance = provenance;
    }

    pub fn seed_provenance(&self) -> &TerrariumSeedProvenance {
        &self.seed_provenance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_seed_provenance_preserves_seed() {
        let provenance = explicit_seed_provenance(42, "test");
        assert_eq!(provenance.seed, 42);
        assert_eq!(provenance.source, TerrariumSeedSource::Explicit);
        assert_eq!(provenance.source_label, "test");
    }

    #[test]
    fn resolve_seed_provenance_prefers_explicit_seed() {
        let provenance = resolve_seed_provenance(Some(7), "cli");
        assert_eq!(provenance.seed, 7);
        assert_eq!(provenance.source, TerrariumSeedSource::Explicit);
    }
}
