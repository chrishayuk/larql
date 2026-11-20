//! Gate B, B6 (`docs/gate-b-b6.md` r2, §3.5 and D4): no source file of this
//! crate that defines a serialized type uses a map or set whose iteration
//! order varies from run to run.
//!
//! Two such maps made compiled output differ between identical runs: the
//! `index.json` checksum map and the KNN store's per-layer map, which
//! `knn_store.bin` is written from. Both are now `BTreeMap`. This guard
//! fails when a `HashMap` or `HashSet` appears in any file that derives
//! `Serialize`, so the next one is caught when it is added, not when two
//! outputs happen to differ.
//!
//! The check is per file, so it is stricter than needed: a file that
//! serializes one type and uses a `HashMap` privately for another fails
//! too. Such a file is listed in [`ALLOWED`] with the reason its map never
//! reaches serialized bytes.

use std::path::{Path, PathBuf};

/// Files that define a serialized type and use an unordered collection
/// that is never serialized, each with the reason.
const ALLOWED: &[(&str, &str)] = &[(
    "src/format/weights/write_f32.rs",
    "StreamingWeights borrows a tensor-name index to read safetensors; it is not serialized",
)];

const UNORDERED: &[&str] = &["HashMap", "HashSet"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The file's code with `//` comments removed, so prose about a past
/// `HashMap` does not count.
fn code(text: &str) -> String {
    text.lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

fn derives_serialize(code: &str) -> bool {
    code.match_indices("derive(").any(|(at, _)| {
        code[at..]
            .split(')')
            .next()
            .unwrap_or("")
            .contains("Serialize")
    })
}

#[test]
fn no_serializing_file_uses_an_unordered_collection() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    files.sort();

    let mut offenders = Vec::new();
    for file in &files {
        let rel = file
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let code = code(&std::fs::read_to_string(file).unwrap());
        if !derives_serialize(&code) || ALLOWED.iter().any(|(path, _)| *path == rel) {
            continue;
        }
        for name in UNORDERED {
            if code.contains(name) {
                offenders.push(format!("{rel}: {name}"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a file that defines a serialized type uses an unordered collection; use \
         BTreeMap/BTreeSet, or add the file to ALLOWED with the reason its map is never \
         serialized: {offenders:?}"
    );
}

#[test]
fn every_allowed_file_exists_and_still_needs_its_exception() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (path, reason) in ALLOWED {
        let text = std::fs::read_to_string(root.join(path))
            .unwrap_or_else(|e| panic!("{path} ({reason}): {e}"));
        let code = code(&text);
        assert!(
            derives_serialize(&code) && UNORDERED.iter().any(|n| code.contains(n)),
            "{path} no longer needs its exception; remove it from ALLOWED"
        );
    }
}

/// The checksum map serializes the same whatever order its entries were
/// inserted in.
#[test]
fn the_index_checksum_map_serializes_independently_of_insertion_order() {
    let names = [
        "gate_vectors.bin",
        "embeddings.bin",
        "down_meta.bin",
        "norms.bin",
    ];
    let serialize = |order: &[usize]| {
        // Collected into whatever map type the field has, so a change back
        // to an unordered map compiles and fails here.
        let config = larql_vindex::VindexConfig {
            checksums: Some(
                order
                    .iter()
                    .map(|&i| (names[i].to_string(), format!("{i:064x}")))
                    .collect(),
            ),
            ..Default::default()
        };
        serde_json::to_vec_pretty(&config).unwrap()
    };
    let first = serialize(&[0, 1, 2, 3]);
    for order in [[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
        assert_eq!(serialize(&order), first, "order {order:?}");
    }
}

/// The KNN store saves the same bytes whatever order its layers were
/// filled in.
#[test]
fn the_knn_store_saves_independently_of_insertion_order() {
    use larql_vindex::KnnStore;
    const DIM: usize = 4;
    let save = |layers: &[usize]| {
        let mut store = KnnStore::default();
        for &layer in layers {
            store.add(
                layer,
                vec![layer as f32 + 1.0; DIM],
                layer as u32,
                format!("t{layer}"),
                format!("e{layer}"),
                "r".into(),
                1.0,
            );
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("knn_store.bin");
        store.save(&path).unwrap();
        std::fs::read(&path).unwrap()
    };
    let first = save(&[0, 1, 2, 3, 4, 5]);
    for order in [[5, 4, 3, 2, 1, 0], [2, 5, 0, 4, 1, 3]] {
        assert_eq!(save(&order), first, "order {order:?}");
    }
}
