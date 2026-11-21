//! Binary `.lknn` save / load for `KnnStore`.
//!
//! Format (little-endian):
//!   magic   = `b"LKNN"`        (4 bytes)
//!   version = 1                (u32)
//!   dim     = key dimension    (u32)
//!   n_layers                   (u32)
//!   per layer:
//!     layer_id                 (u32)
//!     n_entries                (u32)
//!     keys                     (n_entries × dim × f16)
//!     target_ids               (n_entries × u32)
//!     per entry: meta_len (u32) + meta_bytes (JSON)
//!
//! Keys are quantised to f16 — KNN cosine retrieval doesn't need f32
//! precision. Reconstruction goes through `KnnStore::from_entries`.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::path::Path;

use super::knn_store::{KnnEntry, KnnStore};

const MAGIC: &[u8; 4] = b"LKNN";
const VERSION: u32 = 1;
const U32_BYTES: usize = std::mem::size_of::<u32>();
const F16_BYTES: usize = std::mem::size_of::<u16>();

/// Cap a `Vec::with_capacity` request from an untrusted header count by
/// the number of items the remaining input bytes could possibly encode.
/// The subsequent reads still consume exactly `declared` items — this
/// only stops a corrupt count from pre-allocating gigabytes before the
/// first `read_exact` fails.
fn bounded_capacity(declared: usize, remaining_bytes: usize, min_bytes_per_item: usize) -> usize {
    declared.min(remaining_bytes / min_bytes_per_item.max(1))
}

impl KnnStore {
    /// Save to binary format with f16 keys.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());

        // Infer dim from first entry
        let entries = self.entries();
        let dim = entries
            .values()
            .flat_map(|v| v.first())
            .map(|e| e.key.len())
            .next()
            .unwrap_or(0) as u32;
        buf.extend_from_slice(&dim.to_le_bytes());

        let num_layers = entries.len() as u32;
        buf.extend_from_slice(&num_layers.to_le_bytes());

        // Per layer
        for (&layer, entries) in entries {
            buf.extend_from_slice(&(layer as u32).to_le_bytes());
            buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());

            // Keys as f16 (flat: num_entries * dim)
            for entry in entries {
                for &v in &entry.key {
                    let bits = larql_models::quant::half::f32_to_f16(v);
                    buf.extend_from_slice(&bits.to_le_bytes());
                }
            }

            // Target IDs
            for entry in entries {
                buf.extend_from_slice(&entry.target_id.to_le_bytes());
            }

            // Metadata as JSON blob per entry
            for entry in entries {
                let meta = serde_json::json!({
                    "target_token": entry.target_token,
                    "entity": entry.entity,
                    "relation": entry.relation,
                    "confidence": entry.confidence,
                });
                let meta_bytes =
                    serde_json::to_vec(&meta).map_err(|e| format!("json encode: {e}"))?;
                buf.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(&meta_bytes);
            }
        }

        std::fs::write(path, &buf).map_err(|e| format!("write knn_store: {e}"))
    }

    /// Load from binary format.
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read knn_store: {e}"))?;
        let mut cursor = Cursor::new(data.as_slice());

        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .map_err(|e| format!("read magic: {e}"))?;
        if &magic != MAGIC {
            return Err(format!("bad magic: expected LKNN, got {:?}", magic));
        }

        let version = read_u32(&mut cursor)?;
        if version != VERSION {
            return Err(format!("unsupported knn_store version: {version}"));
        }

        let dim = read_u32(&mut cursor)? as usize;
        let num_layers = read_u32(&mut cursor)? as usize;

        let mut entries = BTreeMap::new();
        for _ in 0..num_layers {
            let layer = read_u32(&mut cursor)? as usize;
            let num_entries = read_u32(&mut cursor)? as usize;

            // Header counts are untrusted — cap every capacity hint by
            // what the remaining bytes could actually hold, so a corrupt
            // header yields a read error instead of a multi-GB
            // allocation. Each entry needs at least dim×2 (f16 key)
            // + 4 (target id) + 4 (meta len) bytes.
            let min_entry_bytes = dim
                .checked_mul(F16_BYTES)
                .and_then(|n| n.checked_add(U32_BYTES + U32_BYTES))
                .ok_or_else(|| "knn_store entry size overflow".to_string())?;
            let remaining = data.len().saturating_sub(cursor.position() as usize);
            let entry_cap = bounded_capacity(num_entries, remaining, min_entry_bytes);
            let key_cap = bounded_capacity(dim, remaining, F16_BYTES);

            // Keys (f16 → f32)
            let mut keys = Vec::with_capacity(entry_cap);
            for _ in 0..num_entries {
                let mut key = Vec::with_capacity(key_cap);
                for _ in 0..dim {
                    let bits = read_u16(&mut cursor)?;
                    key.push(larql_models::quant::half::f16_to_f32(bits));
                }
                keys.push(key);
            }

            // Target IDs
            let mut target_ids = Vec::with_capacity(entry_cap);
            for _ in 0..num_entries {
                target_ids.push(read_u32(&mut cursor)?);
            }

            // Metadata JSON blobs
            let mut layer_entries = Vec::with_capacity(entry_cap);
            for i in 0..num_entries {
                let meta_len = read_u32(&mut cursor)? as usize;
                let remaining = data.len().saturating_sub(cursor.position() as usize);
                if meta_len > remaining {
                    return Err(format!(
                        "truncated knn_store: meta blob declares {meta_len} bytes, \
                         {remaining} remain"
                    ));
                }
                let mut meta_bytes = vec![0u8; meta_len];
                cursor
                    .read_exact(&mut meta_bytes)
                    .map_err(|e| format!("read meta: {e}"))?;
                let meta: serde_json::Value =
                    serde_json::from_slice(&meta_bytes).map_err(|e| format!("json decode: {e}"))?;

                layer_entries.push(KnnEntry {
                    key: keys[i].clone(),
                    target_id: target_ids[i],
                    target_token: meta["target_token"].as_str().unwrap_or("").to_string(),
                    entity: meta["entity"].as_str().unwrap_or("").to_string(),
                    relation: meta["relation"].as_str().unwrap_or("").to_string(),
                    confidence: meta["confidence"].as_f64().unwrap_or(1.0) as f32,
                });
            }

            entries.insert(layer, layer_entries);
        }

        Ok(KnnStore::from_entries(entries))
    }
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("read u16: {e}"))?;
    Ok(u16::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIM: u32 = 2;
    const ONE_LAYER: u32 = 1;

    fn header(dim: u32, n_layers: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&dim.to_le_bytes());
        buf.extend_from_slice(&n_layers.to_le_bytes());
        buf
    }

    fn load_bytes(bytes: &[u8]) -> Result<KnnStore, String> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.lknn");
        std::fs::write(&path, bytes).unwrap();
        KnnStore::load(&path)
    }

    #[test]
    fn load_rejects_bad_magic() {
        let mut bytes = header(TEST_DIM, 0);
        bytes[..MAGIC.len()].copy_from_slice(b"NOPE");
        let err = load_bytes(&bytes).unwrap_err();
        assert!(err.contains("bad magic"), "{err}");
    }

    #[test]
    fn load_rejects_unknown_version() {
        let mut bytes = header(TEST_DIM, 0);
        let version_at = MAGIC.len();
        bytes[version_at..version_at + U32_BYTES].copy_from_slice(&99u32.to_le_bytes());
        let err = load_bytes(&bytes).unwrap_err();
        assert!(err.contains("unsupported knn_store version"), "{err}");
    }

    #[test]
    fn load_rejects_truncated_header() {
        let bytes = header(TEST_DIM, ONE_LAYER);
        let err = load_bytes(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(err.contains("read"), "{err}");
    }

    /// A corrupt `n_entries` of u32::MAX must fail with a read error —
    /// not attempt a multi-GB `Vec::with_capacity` first. Same for a
    /// huge `dim`. This is the allocation-bomb class the bounded
    /// capacities exist for.
    #[test]
    fn load_rejects_absurd_entry_count_without_allocating() {
        let mut bytes = header(TEST_DIM, ONE_LAYER);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // layer id
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // n_entries
        let err = load_bytes(&bytes).unwrap_err();
        assert!(err.contains("read"), "{err}");

        let mut bytes = header(u32::MAX, ONE_LAYER);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // layer id
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_entries
        let err = load_bytes(&bytes).unwrap_err();
        assert!(err.contains("read"), "{err}");
    }

    /// A meta blob declaring more bytes than remain in the file is
    /// rejected before the `vec![0u8; meta_len]` allocation.
    #[test]
    fn load_rejects_oversized_meta_len() {
        let dim = 1u32;
        let mut bytes = header(dim, ONE_LAYER);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // layer id
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_entries
        bytes.extend_from_slice(&0u16.to_le_bytes()); // one f16 key value
        bytes.extend_from_slice(&7u32.to_le_bytes()); // target id
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // meta_len
        let err = load_bytes(&bytes).unwrap_err();
        assert!(err.contains("truncated knn_store"), "{err}");
    }

    /// Corrupt JSON in the meta blob errors instead of being dropped.
    #[test]
    fn load_rejects_corrupt_meta_json() {
        let dim = 1u32;
        let garbage = b"not json";
        let mut bytes = header(dim, ONE_LAYER);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // layer id
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_entries
        bytes.extend_from_slice(&0u16.to_le_bytes()); // one f16 key value
        bytes.extend_from_slice(&7u32.to_le_bytes()); // target id
        bytes.extend_from_slice(&(garbage.len() as u32).to_le_bytes());
        bytes.extend_from_slice(garbage);
        let err = load_bytes(&bytes).unwrap_err();
        assert!(err.contains("json decode"), "{err}");
    }

    /// The happy path still round-trips after the bounded-capacity
    /// hardening (full round-trip coverage lives in `knn_store.rs`).
    #[test]
    fn load_round_trips_saved_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.lknn");
        let mut store = KnnStore::default();
        store.add(
            3,
            vec![1.0, 0.0],
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            0.9,
        );
        store.save(&path).unwrap();
        let loaded = KnnStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        let entries = &loaded.entries()[&3];
        assert_eq!(entries[0].target_id, 42);
        assert_eq!(entries[0].entity, "France");
    }
}
