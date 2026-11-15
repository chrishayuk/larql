//! Synthetic BitNet b1.58 `--keep-quant` container on disk.
//!
//! The native-ternary arms of `/v1/infer`, the chat SSE stream and the
//! `/v1/completions` stream all sit behind `LoadedModel::is_bitnet()`,
//! which needs a container carrying `bitnet_layout` plus the `bitnet/`
//! I2_S artifacts. This module builds one the way
//! `larql convert gguf-to-vindex --keep-quant` does — a vindex build,
//! then `bitnet_writer::write_bitnet_artifacts`, then `bitnet_layout`
//! stamped into `index.json` — so the routes run the real writer →
//! loader → ternary forward seam. Only the source weights are
//! synthetic (`larql_inference::test_utils::make_test_bitnet_weights`).
//!
//! Two shapes, matching the two convert modes:
//! - [`BitnetBuild::DenseOnly`] (`--dense-only`): no gate vectors, so
//!   walk-mode inference has nothing to walk.
//! - [`BitnetBuild::WithGates`]: a full vindex build with gate vectors,
//!   so the ternary walk arm (residual capture + KNN store) runs.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use larql_inference::test_utils::{make_test_bitnet_weights, BITNET_TEST_VOCAB};
use larql_server::bootstrap::LoadVindexOptions;
use larql_server::state::LoadedModel;
use larql_vindex::extract::bitnet_writer::{write_bitnet_artifacts, BitnetArchMeta};
use larql_vindex::format::filenames::{BITNET_SCALES_BIN, INDEX_JSON};

/// Model name stamped into the fixture's `index.json`.
pub const BITNET_FIXTURE_MODEL: &str = "test/synthetic-bitnet";
/// Token id every out-of-vocabulary word maps to. The last real row of
/// the embedding table, so a prompt of arbitrary words (a rendered chat
/// template, say) still tokenises to ids the forward can embed.
pub const BITNET_FIXTURE_UNK_ID: usize = BITNET_TEST_VOCAB - 1;
/// Per-feature down-projection metadata width for the gated build —
/// the value `larql convert` passes to `build_vindex`.
const DOWN_TOP_K: usize = 10;

/// Which `--keep-quant` container shape to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitnetBuild {
    /// `--keep-quant --dense-only`: no gate layers in `index.json`.
    DenseOnly,
    /// `--keep-quant` over a full build: gate vectors present.
    WithGates,
}

/// On-disk synthetic BitNet container. Keep it alive for the duration
/// of the test; dropping it removes the directory.
pub struct SyntheticBitnetVindex {
    pub dir: PathBuf,
    pub vocab_size: usize,
    pub hidden: usize,
    _tmp: tempfile::TempDir,
}

/// The token spelling of id `i` in the fixture's vocabulary.
pub fn token_text(i: usize) -> String {
    format!("[{i}]")
}

/// WordLevel tokenizer over `[0]..[VOCAB-1]` with a whitespace
/// pre-tokenizer, every unknown word mapping to [`BITNET_FIXTURE_UNK_ID`].
fn fixture_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
    let vocab: serde_json::Map<String, serde_json::Value> = (0..BITNET_TEST_VOCAB)
        .map(|i| (token_text(i), serde_json::json!(i)))
        .collect();
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        // WhitespaceSplit, not Whitespace: the latter splits `[3]` into
        // `[`, `3`, `]`, so no vocabulary entry would ever match.
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": token_text(BITNET_FIXTURE_UNK_ID),
        }
    });
    let bytes = serde_json::to_vec(&json).expect("serialise tokenizer json");
    larql_vindex::tokenizers::Tokenizer::from_bytes(bytes).expect("parse fixture tokenizer")
}

/// Build a synthetic BitNet `--keep-quant` container of the given shape.
pub fn build(kind: BitnetBuild) -> SyntheticBitnetVindex {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let dir = tmp.path().to_path_buf();
    let weights = make_test_bitnet_weights();
    let tokenizer = fixture_tokenizer();
    let mut cb = larql_vindex::SilentBuildCallbacks;

    match kind {
        BitnetBuild::DenseOnly => larql_vindex::build_vindex_dense_only(
            &weights,
            &tokenizer,
            BITNET_FIXTURE_MODEL,
            &dir,
            larql_vindex::StorageDtype::F32,
            &mut cb,
        ),
        BitnetBuild::WithGates => larql_vindex::build_vindex(
            &weights,
            &tokenizer,
            BITNET_FIXTURE_MODEL,
            &dir,
            DOWN_TOP_K,
            larql_vindex::ExtractLevel::All,
            larql_vindex::StorageDtype::F32,
            &mut cb,
        ),
    }
    .expect("build synthetic bitnet vindex");

    let meta = BitnetArchMeta::from_architecture(&*weights.arch).expect("bitnet arch meta");
    let layout = write_bitnet_artifacts(&dir, &weights, meta).expect("write I2_S artifacts");
    let mut config = larql_vindex::load_vindex_config(&dir).expect("load index.json");
    config.bitnet_layout = Some(layout);
    let json = serde_json::to_string_pretty(&config).expect("serialise index.json");
    std::fs::write(dir.join(INDEX_JSON), json).expect("stamp bitnet_layout");

    SyntheticBitnetVindex {
        dir,
        vocab_size: weights.vocab_size,
        hidden: weights.hidden_size,
        _tmp: tmp,
    }
}

/// Load a synthetic BitNet container through the production loader
/// (`bootstrap::load_single_vindex`), the way `larql-server <path>` does.
/// The ternary model is left to lazy-load on the first request, as under
/// `--lazy-weights`.
pub fn loaded_model(kind: BitnetBuild) -> (Arc<LoadedModel>, SyntheticBitnetVindex) {
    let fixture = build(kind);
    let path = fixture.dir.to_str().expect("tempdir path is UTF-8");
    let model = larql_server::bootstrap::load_single_vindex(path, LoadVindexOptions::default())
        .expect("load synthetic bitnet vindex");
    (Arc::new(model), fixture)
}

/// Remove the container's per-row scales so the lazy ternary load fails —
/// drives the load-failure arms of the routes.
pub fn break_ternary_artifacts(fixture: &SyntheticBitnetVindex) {
    std::fs::remove_file(fixture.dir.join(BITNET_SCALES_BIN)).expect("remove bitnet scales");
}
