//! The startup memcheck's ternary-aware estimate: a BitNet container is
//! sized by the bytes its I2_S weights occupy on disk and in memory, not
//! by the dense path's f32 sizes, and is no longer skipped.

use larql_server::memcheck::bitnet::{estimate_bitnet_resident_bytes, i2s_packed_bytes};
use larql_server::memcheck::estimate_model_resident_bytes;
use larql_vindex::format::filenames::{bitnet_tensor_filename, BITNET_SCALES_BIN};
use larql_vindex::{BitnetLayout, BitnetTensorEntry};

use super::common;
use super::common::synthetic_bitnet_vindex::{build, BitnetBuild};

const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;

fn file_len(path: &std::path::Path) -> u64 {
    std::fs::metadata(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .len()
}

#[test]
fn i2s_packs_four_weights_per_byte_and_rounds_a_partial_block_up() {
    assert_eq!(i2s_packed_bytes(2, 8), 4);
    assert_eq!(i2s_packed_bytes(1, 5), 2);
    assert_eq!(i2s_packed_bytes(0, 8), 0);
}

#[test]
fn ternary_term_matches_the_i2s_bytes_the_container_actually_holds() {
    let fixture = build(BitnetBuild::DenseOnly);
    let config = larql_vindex::load_vindex_config(&fixture.dir).expect("index.json");
    let layout = config.bitnet_layout.as_ref().expect("bitnet_layout");

    let declared: u64 = layout
        .tensors
        .iter()
        .map(|t| i2s_packed_bytes(t.rows, t.cols))
        .sum();
    let on_disk: u64 = layout
        .tensors
        .iter()
        .map(|t| file_len(&fixture.dir.join(bitnet_tensor_filename(&t.name))))
        .sum();
    assert_eq!(declared, on_disk, "estimate sizes the real I2_S files");
    assert_eq!(
        layout.total_scale_count as u64 * F32_BYTES,
        file_len(&fixture.dir.join(BITNET_SCALES_BIN)),
    );

    let estimate = estimate_bitnet_resident_bytes(&config).expect("a bitnet estimate");
    let embed = (config.vocab_size * config.hidden_size) as u64 * F32_BYTES;
    assert!(
        estimate >= declared + layout.total_scale_count as u64 * F32_BYTES + 2 * embed,
        "covers ternary weights, scales, embed and lm_head: {estimate}"
    );
    assert_eq!(
        estimate_model_resident_bytes(&config),
        estimate,
        "the pre-flight dispatches a BitNet container to the ternary estimate"
    );
}

/// A layout large enough that the BitLinears dominate: the ternary
/// estimate must sit far below what the same projections cost at f32.
#[test]
fn ternary_estimate_is_a_fraction_of_the_f32_size_of_the_same_projections() {
    const ROWS: usize = 4096;
    const COLS: usize = 4096;
    const TENSORS: usize = 16;
    let mut config = common::test_config();
    config.vocab_size = 1;
    config.hidden_size = 1;
    config.bitnet_layout = Some(BitnetLayout {
        tensors: (0..TENSORS)
            .map(|i| BitnetTensorEntry {
                name: format!("t{i}"),
                rows: ROWS,
                cols: COLS,
                scale_offset: i * ROWS,
            })
            .collect(),
        total_scale_count: TENSORS * ROWS,
        ..Default::default()
    });
    let f32_projections = (TENSORS * ROWS * COLS) as u64 * F32_BYTES;
    let estimate = estimate_model_resident_bytes(&config);
    assert!(estimate > 0);
    assert!(
        estimate < f32_projections / 8,
        "{estimate} vs f32 {f32_projections}"
    );
}

#[test]
fn a_non_bitnet_config_keeps_the_dense_estimate() {
    let config = common::test_config();
    assert_eq!(estimate_bitnet_resident_bytes(&config), None);
    assert_eq!(
        estimate_model_resident_bytes(&config),
        config.estimate_resident_bytes()
    );
}
