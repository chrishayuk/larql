//! Disk persistence for the L2 `MemitStore`.
//!
//! `COMPACT MAJOR` accumulates decomposed (k, d) pairs into the
//! session-resident `MemitStore`. Without persistence the next
//! `USE` rebuilds an empty store, so prior compaction cycles
//! disappear. We snapshot the store as a JSON file under the vindex
//! directory; load runs at `USE`/`EXTRACT` and save runs after
//! every successful `add_cycle`.
//!
//! JSON (rather than the binary `bincode`/`bytes::Bytes` shape used
//! by the rest of `larql-vindex`) keeps the format inspectable by
//! hand and avoids pulling another dependency into `larql-vindex`
//! purely to support this round-trip. The expected size for a
//! month's worth of compaction is well under a megabyte.

use std::path::Path;

use larql_inference::ndarray::Array1;
use larql_vindex::{MemitCycle, MemitFact, MemitStore};
use serde::{Deserialize, Serialize};

/// Filename for the JSON snapshot, kept as a constant so callers
/// don't sprinkle the literal across the codebase.
pub(crate) const MEMIT_STORE_JSON: &str = "memit_store.json";

#[derive(Serialize, Deserialize)]
struct MemitStoreDto {
    cycles: Vec<MemitCycleDto>,
}

#[derive(Serialize, Deserialize)]
struct MemitCycleDto {
    cycle_id: u64,
    layer: usize,
    frobenius_norm: f32,
    min_reconstruction_cos: f32,
    max_off_diagonal: f32,
    facts: Vec<MemitFactDto>,
}

#[derive(Serialize, Deserialize)]
struct MemitFactDto {
    entity: String,
    relation: String,
    target: String,
    key: Vec<f32>,
    decomposed_down: Vec<f32>,
    reconstruction_cos: f32,
}

impl From<&MemitFact> for MemitFactDto {
    fn from(fact: &MemitFact) -> Self {
        Self {
            entity: fact.entity.clone(),
            relation: fact.relation.clone(),
            target: fact.target.clone(),
            key: fact.key.to_vec(),
            decomposed_down: fact.decomposed_down.to_vec(),
            reconstruction_cos: fact.reconstruction_cos,
        }
    }
}

impl From<MemitFactDto> for MemitFact {
    fn from(dto: MemitFactDto) -> Self {
        Self {
            entity: dto.entity,
            relation: dto.relation,
            target: dto.target,
            key: Array1::from(dto.key),
            decomposed_down: Array1::from(dto.decomposed_down),
            reconstruction_cos: dto.reconstruction_cos,
        }
    }
}

impl From<&MemitCycle> for MemitCycleDto {
    fn from(cycle: &MemitCycle) -> Self {
        Self {
            cycle_id: cycle.cycle_id,
            layer: cycle.layer,
            frobenius_norm: cycle.frobenius_norm,
            min_reconstruction_cos: cycle.min_reconstruction_cos,
            max_off_diagonal: cycle.max_off_diagonal,
            facts: cycle.facts.iter().map(MemitFactDto::from).collect(),
        }
    }
}

impl From<MemitCycleDto> for MemitCycle {
    fn from(dto: MemitCycleDto) -> Self {
        Self {
            cycle_id: dto.cycle_id,
            layer: dto.layer,
            frobenius_norm: dto.frobenius_norm,
            min_reconstruction_cos: dto.min_reconstruction_cos,
            max_off_diagonal: dto.max_off_diagonal,
            facts: dto.facts.into_iter().map(MemitFact::from).collect(),
        }
    }
}

/// Load a `MemitStore` snapshot from `<vindex_dir>/memit_store.json`.
///
/// Returns `Ok(None)` when the file is absent, signalling "no prior
/// state, start fresh" rather than an error. Malformed JSON or IO
/// errors are surfaced so a corrupt snapshot doesn't silently mask
/// itself as "no state".
pub(crate) fn load_memit_store(vindex_dir: &Path) -> Result<Option<MemitStore>, String> {
    let path = vindex_dir.join(MEMIT_STORE_JSON);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let dto: MemitStoreDto =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let cycles: Vec<MemitCycle> = dto.cycles.into_iter().map(MemitCycle::from).collect();
    Ok(Some(MemitStore::from_cycles(cycles)))
}

/// Persist a `MemitStore` snapshot to `<vindex_dir>/memit_store.json`,
/// using a tmp-file + rename so a crash mid-write can't corrupt an
/// existing snapshot. Empty stores still write a file (one line of
/// JSON) so callers can tell "we ran COMPACT MAJOR and produced no
/// facts" apart from "the file was never written".
///
/// Since Gate B, B6 (ruling D2) COMPACT MAJOR refuses, so no statement
/// writes this file; `USE` still reads one a vindex already carries. Only
/// tests write it, to build those vindexes.
#[cfg(test)]
pub(crate) fn save_memit_store(vindex_dir: &Path, store: &MemitStore) -> Result<(), String> {
    let dto = MemitStoreDto {
        cycles: store.cycles().iter().map(MemitCycleDto::from).collect(),
    };
    let json =
        serde_json::to_vec_pretty(&dto).map_err(|e| format!("serialise memit_store: {e}"))?;

    let final_path = vindex_dir.join(MEMIT_STORE_JSON);
    let tmp_path = vindex_dir.join(format!("{MEMIT_STORE_JSON}.tmp.{}", std::process::id()));
    std::fs::write(&tmp_path, &json).map_err(|e| format!("write {}: {e}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &final_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!(
            "promote {} → {}: {e}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_memit_persist_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    fn synthetic_store() -> MemitStore {
        let mut store = MemitStore::new();
        let fact = MemitFact {
            entity: "Atlantis".into(),
            relation: "capital-of".into(),
            target: "Poseidon".into(),
            key: Array1::from(vec![0.1f32, 0.2, 0.3, 0.4]),
            decomposed_down: Array1::from(vec![0.5f32, 0.6, 0.7, 0.8]),
            reconstruction_cos: 0.987,
        };
        store.add_cycle(7, vec![fact], 0.42, 0.987, 0.001);
        store
    }

    #[test]
    fn load_returns_none_when_file_absent() {
        let dir = unique_dir("absent");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(load_memit_store(&dir).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_then_load_round_trips_cycle_state() {
        let dir = unique_dir("roundtrip");
        std::fs::create_dir_all(&dir).unwrap();

        let original = synthetic_store();
        save_memit_store(&dir, &original).unwrap();

        let loaded = load_memit_store(&dir).unwrap().expect("snapshot present");
        assert_eq!(loaded.num_cycles(), original.num_cycles());
        assert_eq!(loaded.total_facts(), original.total_facts());

        let cycle = &loaded.cycles()[0];
        assert_eq!(cycle.cycle_id, 0);
        assert_eq!(cycle.layer, 7);
        let fact = &cycle.facts[0];
        assert_eq!(fact.entity, "Atlantis");
        assert_eq!(fact.relation, "capital-of");
        assert_eq!(fact.target, "Poseidon");
        assert_eq!(fact.key.to_vec(), vec![0.1f32, 0.2, 0.3, 0.4]);
        assert_eq!(fact.decomposed_down.to_vec(), vec![0.5f32, 0.6, 0.7, 0.8]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_overwrites_atomically() {
        let dir = unique_dir("overwrite");
        std::fs::create_dir_all(&dir).unwrap();

        save_memit_store(&dir, &synthetic_store()).unwrap();
        // Add a second cycle and re-save.
        let mut store = synthetic_store();
        store.add_cycle(
            9,
            vec![MemitFact {
                entity: "x".into(),
                relation: "y".into(),
                target: "z".into(),
                key: Array1::from(vec![0.0f32; 2]),
                decomposed_down: Array1::from(vec![0.0f32; 2]),
                reconstruction_cos: 1.0,
            }],
            0.0,
            1.0,
            0.0,
        );
        save_memit_store(&dir, &store).unwrap();

        let loaded = load_memit_store(&dir).unwrap().unwrap();
        assert_eq!(loaded.num_cycles(), 2);
        // No leftover .tmp file.
        let leftover_tmp = std::fs::read_dir(&dir)
            .unwrap()
            .any(|e| e.unwrap().file_name().to_string_lossy().contains(".tmp."));
        assert!(!leftover_tmp, "tmp file should be removed after rename");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_cycles_preserves_next_cycle_id() {
        // Regression: rehydrating a store with cycles that have ids 0, 5, 7
        // must set next_cycle_id past 7 so the next add_cycle gets id 8.
        let cycles = vec![
            MemitCycle {
                cycle_id: 0,
                layer: 1,
                facts: Vec::new(),
                frobenius_norm: 0.0,
                min_reconstruction_cos: 1.0,
                max_off_diagonal: 0.0,
            },
            MemitCycle {
                cycle_id: 7,
                layer: 2,
                facts: Vec::new(),
                frobenius_norm: 0.0,
                min_reconstruction_cos: 1.0,
                max_off_diagonal: 0.0,
            },
        ];
        let mut store = MemitStore::from_cycles(cycles);
        let new_id = store.add_cycle(3, Vec::new(), 0.0, 1.0, 0.0);
        assert_eq!(new_id, 8);
    }

    #[test]
    fn load_propagates_corrupt_json_error() {
        let dir = unique_dir("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(MEMIT_STORE_JSON), b"{ not valid json").unwrap();
        let err = load_memit_store(&dir).unwrap_err();
        assert!(err.contains("parse"), "unexpected error: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
