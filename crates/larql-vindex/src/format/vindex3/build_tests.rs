//! Tests for the c9 streaming builder.
//!
//! c8's gate was "one layer's regions survive the round trip unchanged". c9
//! adds two things that only exist once there is more than one layer: every
//! layer must round-trip (not just the first or last), and the container must
//! not *announce itself* as a container until all of them are down.

use super::*;
use crate::format::generation::{detect_generation, ContainerGeneration};
use crate::format::lyrw2::region_format::RegionFormat;
use crate::format::lyrw2::region_layout::RegionLayout;
use crate::format::lyrw2::region_role::RegionRole;
use crate::format::vindex3::import::ExpertScaleStreams;
use crate::format::vindex3::read::Vindex3Container;
use crate::format::vindex3::write::segment_path;
use tempfile::tempdir;

const HIDDEN: u32 = 8;
const STORED_INTER: u32 = 12;
const SEMANTIC_INTER: u32 = 10;
const EXPERTS: usize = 3;
const TOP_K: u32 = 2;
const LAYERS: u32 = 4;
const MODEL_LAYERS: usize = 6;

/// Distinct per (layer, expert) so a bank read from the wrong layer — the
/// failure a loop introduces and a one-layer test cannot see — cannot pass.
///
/// Sized at the **semantic** width, not the stored one: gate_up is never
/// padded, and a fixture that pads both regions equally cannot tell a correct
/// importer from one that describes gate_up with down's width.
fn gate_up_bytes(layer: u32, expert: usize) -> Vec<u8> {
    let n = (SEMANTIC_INTER * 2 * HIDDEN) as usize * 4;
    (0..n)
        .map(|i| (layer as usize * 101 + expert * 31 + i) as u8)
        .collect()
}

fn down_bytes(layer: u32, expert: usize) -> Vec<u8> {
    let n = (HIDDEN * STORED_INTER) as usize * 4;
    (0..n)
        .map(|i| (layer as usize * 53 + expert * 17 + i + 7) as u8)
        .collect()
}

struct Owned {
    gate_up: Vec<Vec<u8>>,
    down: Vec<Vec<u8>>,
}

fn owned(layer: u32) -> Owned {
    Owned {
        gate_up: (0..EXPERTS).map(|e| gate_up_bytes(layer, e)).collect(),
        down: (0..EXPERTS).map(|e| down_bytes(layer, e)).collect(),
    }
}

fn source(layer: u32, o: &Owned) -> MoeLayerSource<'_> {
    MoeLayerSource {
        layer,
        experts_gate_up: o.gate_up.iter().map(|v| v.as_slice()).collect(),
        experts_down: o.down.iter().map(|v| v.as_slice()).collect(),
        format: RegionFormat::F32,
        scales: ExpertScaleStreams::Inline,
        gate_up_layout: RegionLayout::ContiguousHalves,
        hidden_size: HIDDEN,
        gate_up_stored_intermediate: SEMANTIC_INTER,
        down_stored_intermediate: STORED_INTER,
        semantic_intermediate: SEMANTIC_INTER,
        top_k: TOP_K,
    }
}

/// Build a complete `LAYERS`-layer container at `root`.
fn build_all(root: &Path) -> Vec<Owned> {
    let mut builder = ContainerBuilder::create(root).unwrap();
    let mut kept = Vec::new();
    for layer in 0..LAYERS {
        let o = owned(layer);
        builder.add_moe_layer(&source(layer, &o)).unwrap();
        kept.push(o);
    }
    builder
        .finish("gemma-fixture", "gemma", HIDDEN as usize, MODEL_LAYERS)
        .unwrap();
    kept
}

// ── The verbatim guarantee, now across layers ────────────────────────────

#[test]
fn every_layer_round_trips_byte_for_byte() {
    let dir = tempdir().unwrap();
    let kept = build_all(dir.path());

    let container = Vindex3Container::open(dir.path()).unwrap();
    assert!(
        container.verify().is_empty(),
        "structural defects: {:?}",
        container.verify()
    );

    let mut checked = 0usize;
    for layer in 0..LAYERS {
        let reader = container.segment(&routed_storage_key(layer)).unwrap();
        for expert in 0..EXPERTS {
            for (role, expected) in [
                (
                    RegionRole::GateUpFused,
                    kept[layer as usize].gate_up[expert].as_slice(),
                ),
                (
                    RegionRole::Down,
                    kept[layer as usize].down[expert].as_slice(),
                ),
            ] {
                let got = reader
                    .region_bytes(0, expert as u32, role)
                    .unwrap()
                    .unwrap_or_else(|| panic!("layer {layer} expert {expert} {role:?} missing"));
                assert_eq!(
                    got, expected,
                    "layer {layer} expert {expert} {role:?} differs from its source"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, LAYERS as usize * EXPERTS * 2);
}

#[test]
fn the_container_reports_v3_and_declares_one_segment_per_layer() {
    let dir = tempdir().unwrap();
    build_all(dir.path());

    assert_eq!(
        detect_generation(dir.path()).unwrap(),
        ContainerGeneration::V3
    );

    let container = Vindex3Container::open(dir.path()).unwrap();
    for layer in 0..LAYERS {
        assert!(
            container.segment(&routed_storage_key(layer)).is_ok(),
            "layer {layer} is not resolvable from the written container"
        );
    }
}

// ── The crash contract ───────────────────────────────────────────────────

#[test]
fn a_directory_is_not_a_container_until_finish_writes_the_index() {
    // The property that makes a long import safe to interrupt: segments land
    // first, and until `index.json` exists no reader will dispatch here.
    let dir = tempdir().unwrap();
    let mut builder = ContainerBuilder::create(dir.path()).unwrap();
    let o = owned(0);
    builder.add_moe_layer(&source(0, &o)).unwrap();

    assert!(
        segment_path(dir.path(), &routed_storage_key(0)).exists(),
        "the segment should be on disk before the index is"
    );
    assert!(
        !dir.path().join(INDEX_JSON).exists(),
        "index.json must not exist mid-import — a reader would treat a partial \
         directory as a complete container"
    );
    assert!(detect_generation(dir.path()).is_err());

    builder
        .finish("gemma-fixture", "gemma", HIDDEN as usize, MODEL_LAYERS)
        .unwrap();
    assert!(dir.path().join(INDEX_JSON).exists());
}

// ── Refusals ─────────────────────────────────────────────────────────────

#[test]
fn importing_the_same_layer_twice_is_refused() {
    let dir = tempdir().unwrap();
    let mut builder = ContainerBuilder::create(dir.path()).unwrap();
    let o = owned(0);
    builder.add_moe_layer(&source(0, &o)).unwrap();

    let err = builder.add_moe_layer(&source(0, &o)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("already in this container"),
        "refusal should name the collision, got: {msg}"
    );
    assert_eq!(builder.layers_added(), 1);
}

#[test]
fn finishing_with_no_layers_is_refused() {
    let dir = tempdir().unwrap();
    let builder = ContainerBuilder::create(dir.path()).unwrap();
    let err = builder
        .finish("gemma-fixture", "gemma", HIDDEN as usize, MODEL_LAYERS)
        .unwrap_err();
    assert!(err.to_string().contains("at least one segment"));
    assert!(
        !dir.path().join(INDEX_JSON).exists(),
        "a refused finish must not leave an index behind"
    );
}

#[test]
fn a_layer_the_importer_refuses_leaves_the_builder_usable() {
    // `check()` runs before any byte is written, so a bad layer must not
    // consume its storage key or half-write its bank.
    let dir = tempdir().unwrap();
    let mut builder = ContainerBuilder::create(dir.path()).unwrap();
    let o = owned(0);

    let mut bad = source(0, &o);
    bad.top_k = EXPERTS as u32 + 1;
    assert!(builder.add_moe_layer(&bad).is_err());
    assert_eq!(builder.layers_added(), 0);

    builder.add_moe_layer(&source(0, &o)).unwrap();
    assert_eq!(builder.layers_added(), 1);
}

// ── Accounting ───────────────────────────────────────────────────────────

#[test]
fn bytes_written_tracks_the_segments_on_disk() {
    let dir = tempdir().unwrap();
    build_all(dir.path());

    let on_disk: u64 = (0..LAYERS)
        .map(|l| {
            std::fs::metadata(segment_path(dir.path(), &routed_storage_key(l)))
                .unwrap()
                .len()
        })
        .sum();

    // Held for the whole test: a temporary dropped at the end of the
    // statement deletes the root the builder writes into.
    let out = tempdir().unwrap();
    let mut builder = ContainerBuilder::create(out.path()).unwrap();
    let mut total = 0u64;
    for layer in 0..LAYERS {
        let o = owned(layer);
        total += builder.add_moe_layer(&source(layer, &o)).unwrap();
    }
    assert_eq!(total, on_disk);
    assert_eq!(builder.bytes_written(), on_disk);
}
