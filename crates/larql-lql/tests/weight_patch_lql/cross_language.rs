//! The cross-language golden: a structured patch compiled by LARQL, and committed with the files the independent
//! Python verifier judges it by (`verifier/conformance/structured_patch/`).
//!
//! This test pins the Rust half: the fixture base, the `.vlp` LQL writes, and the container LQL compiles are
//! byte-for-byte the committed ones. `verifier/tests/test_structured_patch_golden.py` pins the Python half: the
//! verifier judges the committed pair, and every mutation of it, as the golden records. A change on either side that
//! alters the bytes or the verdict fails one of them.
//!
//! Regenerate (only when the compiler's output is meant to change):
//!   LARQL_REGENERATE_GOLDEN=1 cargo test -p larql-lql --test weight_patch_lql cross_language

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::weight_patch::bf16::patched_bits;
use larql_vindex::format::vindex3::weight_patch::read_bf16_tensor;
use sha2::{Digest, Sha256};

use crate::support::*;

const GOLDEN_DIR: &str = "../../verifier/conformance/structured_patch";
const CANDIDATE_FILE: &str = "candidate.json";
const LOCATOR_HINT: &str = "locator_hint";
const GOLDEN_FILE: &str = "golden.json";
const REGENERATE_VAR: &str = "LARQL_REGENERATE_GOLDEN";
const SCHEMA: &str = "larql-structured-patch-golden/1";
const PROFILE: &str = "f64-seq-rne1/v1";
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
const NEAR_CANCEL_COLUMN: usize = 33;
/// `1 + 2^-30`: the removal leaves a tiny nonzero where a delta cannot.
const NEAR_ONE: &str = "1.000000000931322574615478515625";
const PARTIAL_COLUMNS: &str = "40, 41";
const PARTIAL_SCALE: &str = "0.9";
const DELTA_NEIGHBOURS: i64 = 8;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDEN_DIR)
}

/// Every file under `root`, relative path -> sha256, in path order.
fn file_digests(root: &Path) -> BTreeMap<String, String> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(root, &p, out);
            } else {
                // Keys are '/'-joined on every platform, so the committed golden matches Windows too.
                let rel = p
                    .strip_prefix(root)
                    .unwrap()
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                out.insert(rel, hex(&comparable_bytes(&p)));
            }
        }
    }
    let mut out = BTreeMap::new();
    if root.is_file() {
        out.insert(String::new(), hex(&comparable_bytes(root)));
    } else {
        walk(root, root, &mut out);
    }
    out
}

/// A file's bytes as compared. Every file is byte-exact except `candidate.json`, whose `source.locator_hint` is the
/// temporary directory the fixture was built in (a different path every run, and no part of any weight).
fn comparable_bytes(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    if path.file_name().and_then(|n| n.to_str()) != Some(CANDIDATE_FILE) {
        return bytes;
    }
    let mut doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    doc["source"]
        .as_object_mut()
        .unwrap()
        .remove(LOCATOR_HINT)
        .unwrap();
    serde_json::to_vec(&doc).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn operand(fx: &Fixture, name: &str, rows: usize, row: usize) -> String {
    let bits: Vec<String> = (0..rows)
        .map(|r| format!("{:016x}", if r == row { F64_ONE } else { 0 }))
        .collect();
    let path = fx.path(name);
    std::fs::write(&path, serde_json::json!({ "bits": bits }).to_string()).unwrap();
    q(path)
}

fn reachable_by_a_delta(old: u16, new: u16) -> bool {
    let f = |b: u16| f32::from_bits((b as u32) << 16);
    let centre = (f(new) - f(old)).to_bits() as i64;
    (-DELTA_NEIGHBOURS..=DELTA_NEIGHBOURS).any(|s| {
        let d = f32::from_bits((centre + s) as u32);
        d.is_finite() && patched_bits(old, d) == Some(new)
    })
}

#[test]
fn the_committed_golden_is_what_lql_writes_and_compiles() {
    let fx = Fixture::new();
    let ([rows, cols], base) = read_bf16_tensor(&fx.container, TENSOR).unwrap();
    let live =
        |r: usize| (0..rows).find(|&x| x != r && base[x * cols + NEAR_CANCEL_COLUMN] & 0x7fff != 0);
    let near_row = live(usize::MAX).unwrap();
    let partial_row = live(near_row).unwrap();
    let near_operand = operand(&fx, "near.json", rows, near_row);
    let partial_operand = operand(&fx, "partial.json", rows, partial_row);

    let vlp = fx.path("patch.vlp");
    let mut author = session_on(&fx.container);
    run(&mut author, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    for stmt in [
        format!("REMOVE DIRECTION \"{near_operand}\" FROM \"{TENSOR}\" COLUMNS ({NEAR_CANCEL_COLUMN}) SCALE {NEAR_ONE} NUMERICS \"{PROFILE}\";"),
        format!("REMOVE DIRECTION \"{partial_operand}\" FROM \"{TENSOR}\" COLUMNS ({PARTIAL_COLUMNS}) SCALE {PARTIAL_SCALE} NUMERICS \"{PROFILE}\";"),
    ] {
        run(&mut author, &stmt);
    }
    run(&mut author, "SAVE PATCH;");

    let mut fresh = session_on(&fx.container);
    run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    let compiled = fx.path("compiled.vindex3");
    run(
        &mut fresh,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&compiled)),
    );

    // The case is only worth keeping if the near-cancelling element is one no f32 delta can write.
    let ([_, _], after) = read_bf16_tensor(&compiled, TENSOR).unwrap();
    let at = near_row * cols + NEAR_CANCEL_COLUMN;
    assert_ne!(after[at], base[at]);
    assert!(
        !reachable_by_a_delta(base[at], after[at]),
        "a delta could have written the near-cancelling element, so the golden proves nothing"
    );
    let changed: Vec<usize> = (0..base.len()).filter(|&i| base[i] != after[i]).collect();

    let observed = serde_json::json!({
        "schema": SCHEMA,
        "tensor": TENSOR,
        "profile": PROFILE,
        "changed_elements": changed.iter().map(|&i| [i / cols, i % cols]).collect::<Vec<_>>(),
        "near_cancelling_element": [near_row, NEAR_CANCEL_COLUMN],
        "unreachable_by_any_f32_delta": [[near_row, NEAR_CANCEL_COLUMN]],
        "files": {
            "base": file_digests(&fx.container),
            "patch.vlp": file_digests(&vlp)[""],
            "compiled": file_digests(&compiled),
        },
    });

    let dir = golden_dir();
    if std::env::var_os(REGENERATE_VAR).is_some() {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        copy_tree(&fx.container, &dir.join("base"));
        copy_tree(&compiled, &dir.join("compiled"));
        std::fs::copy(&vlp, dir.join("patch.vlp")).unwrap();
        std::fs::write(
            dir.join(GOLDEN_FILE),
            serde_json::to_string_pretty(&observed).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    let committed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(GOLDEN_FILE)).unwrap()).unwrap();
    assert_eq!(
        committed, observed,
        "LQL no longer writes/compiles the committed golden; if that is intended, regenerate with {REGENERATE_VAR}=1"
    );
    // The committed files themselves are the ones the digests describe (the Python half judges these).
    assert_eq!(file_digests(&dir.join("base")), file_digests(&fx.container));
    assert_eq!(file_digests(&dir.join("compiled")), file_digests(&compiled));
    assert_eq!(
        file_digests(&dir.join("patch.vlp"))[""],
        file_digests(&vlp)[""]
    );
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let p = e.unwrap().path();
        let target = to.join(p.file_name().unwrap());
        if p.is_dir() {
            copy_tree(&p, &target);
        } else {
            std::fs::copy(&p, &target).unwrap();
        }
    }
}
