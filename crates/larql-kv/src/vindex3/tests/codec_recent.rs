//! CONTINUATION-CODEC-3 reconnaissance: `codec-recent/v1`, codec/v1 with
//! the newest `w` K rows of every layer exact.
//!
//! Selection (shipped, KV-only) and configuration (`bits` and
//! `exact_recent_k`, both named, no defaults, a window of at least one);
//! the read contract — at every read, K at age < w is bit-identical to the
//! row appended, every other K and every V bit-identical to codec/v1's
//! read of the same appends; window 0 reads exactly as codec/v1; retention
//! identical to codec/v1's through every phase; storage at exactly the
//! declared bytes, each K row encoded once. Bit-parity with the simulator
//! and live-allocation residency are the MEM-1 harness's.

use larql_vindex::format::vindex3::fixtures::{miniature_glimmer, G_TOKENS};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry, LayerLatentKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::{
    ContinuationAuthority, ContinuationConfig,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    ContinuationFactory, ContinuationRegion, ContinuationRegistryError,
};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, HistoryRange, KvState, LayerKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use super::super::{shipped_continuations, CodecKvState, CodecRecentFactory, CodecRecentKvState};
use super::registry_parity::open;

const RESUME: [u32; 3] = [5, 9, 13];
const DECODE: [u32; 4] = [1, 2, 3, 4];
const HEAD_DIM: usize = 128;
const HEADS: usize = 2;
const KV_DIM: usize = HEADS * HEAD_DIM;
/// Rows appended in the read-contract tests: past every window they use
/// and past the sliding layer's history, so ageing and release interleave.
const ROWS: usize = 24;
const SLIDING: usize = 9;
/// Windows below, at, and above the sliding history, and past every row.
const WINDOWS: [usize; 5] = [1, 4, SLIDING, 16, ROWS + 8];

fn config(pairs: &[&str]) -> ContinuationConfig {
    ContinuationConfig::parse(pairs).unwrap()
}

fn gaussian_rows(seed: u64, rows: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows)
        .map(|_| {
            (0..KV_DIM)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect()
        })
        .collect()
}

fn wide(history: HistoryRange) -> LayerKvGeometry {
    LayerKvGeometry {
        kv_dim: KV_DIM,
        head_dim: HEAD_DIM,
        window: match history {
            HistoryRange::Trailing(w) => Some(w),
            _ => None,
        },
        history,
    }
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn codec_recent_v1_is_shipped_and_holds_only_kv() {
    let registry = shipped_continuations();
    assert!(registry
        .identities()
        .contains(&CodecRecentKvState::identity()));
    let (_container, plan, _store) = open(hybrid_kda_mla_f32_model, "cr-hybrid");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let refused = registry
        .select(
            &CodecRecentKvState::identity(),
            &config(&["bits=4", "exact_recent_k=8"]),
            &geometry,
        )
        .expect_err("a hybrid plan needs regions codec-recent/v1 does not hold");
    assert!(
        matches!(refused, ContinuationRegistryError::Unsupported { .. }),
        "{refused}"
    );
    let factory = CodecRecentFactory;
    assert_eq!(factory.identity().to_string(), "codec-recent/v1");
    assert_eq!(factory.regions(), &[ContinuationRegion::Kv]);
}

#[test]
fn both_keys_are_required_named_and_the_window_is_at_least_one_row() {
    let factory = CodecRecentFactory;
    let no_window = factory.validate_config(&config(&["bits=4"])).unwrap_err();
    assert!(no_window.contains("no default window"), "{no_window}");
    let no_bits = factory
        .validate_config(&config(&["exact_recent_k=8"]))
        .unwrap_err();
    assert!(no_bits.contains("no default width"), "{no_bits}");
    let zero = factory
        .validate_config(&config(&["bits=4", "exact_recent_k=0"]))
        .unwrap_err();
    assert!(zero.contains("codec/v1"), "{zero}");
    for bad in ["-1", "eight", ""] {
        let raw = format!("exact_recent_k={bad}");
        let refused = factory
            .validate_config(&config(&["bits=4", &raw]))
            .unwrap_err();
        assert!(refused.contains("whole number"), "{bad}: {refused}");
    }
    let bad_bits = factory
        .validate_config(&config(&["bits=5", "exact_recent_k=8"]))
        .unwrap_err();
    assert!(bad_bits.contains("must be one of"), "{bad_bits}");
    let extra = factory
        .validate_config(&config(&["bits=4", "exact_recent_k=8", "v=4"]))
        .unwrap_err();
    assert!(extra.contains("`v`"), "{extra}");

    let ok = config(&["bits=3", "exact_recent_k=256"]);
    factory.validate_config(&ok).unwrap();
    assert_eq!(factory.build(&ok).position(), 0);
    let digest = |pairs: &[&str]| {
        ContinuationAuthority::new(CodecRecentKvState::identity(), &config(pairs)).config_digest
    };
    assert_ne!(
        digest(&["bits=4", "exact_recent_k=64"]),
        digest(&["bits=4", "exact_recent_k=256"]),
        "two windows, two authorities"
    );
}

/// The read contract, at every read of every append: K of age < w is the
/// appended row bit for bit; older K and all V are codec/v1's read of the
/// same appends bit for bit; the range is codec/v1's.
#[test]
fn every_read_is_exact_inside_the_window_and_codec_v1_outside_it() {
    let layers = [
        wide(HistoryRange::Trailing(SLIDING)),
        wide(HistoryRange::Full),
    ];
    let keys = gaussian_rows(11, ROWS);
    let values = gaussian_rows(12, ROWS);
    for bits_ in [4u8, 3] {
        for w in WINDOWS {
            let mut mixed = CodecRecentKvState::new(bits_, w);
            let mut codec = CodecKvState::new(bits_);
            mixed.prepare(&layers);
            codec.prepare(&layers);
            for (k, v) in keys.iter().zip(&values) {
                for layer in 0..layers.len() {
                    mixed.append(layer, k.clone(), v.clone());
                    codec.append(layer, k.clone(), v.clone());
                    mixed.prepare_layer(layer);
                    codec.prepare_layer(layer);
                    let (m, c) = (mixed.rows(layer), codec.rows(layer));
                    assert_eq!((m.base(), m.end()), (c.base(), c.end()), "w {w}");
                    for (p, appended) in keys.iter().enumerate().take(m.end()).skip(m.base()) {
                        let age = m.end() - 1 - p;
                        let want = if age < w { &appended[..] } else { c.key(p) };
                        assert_eq!(bits(m.key(p)), bits(want), "w {w} K p {p} age {age}");
                        assert_eq!(bits(m.value(p)), bits(c.value(p)), "w {w} V p {p}");
                    }
                    let held = m.end() - m.base();
                    assert_eq!(mixed.exact_keys(layer).len(), w.min(held), "w {w}");
                    assert_eq!(mixed.exact_start(layer), m.end() - w.min(held));
                    assert_eq!(mixed.rows_base(layer), m.base(), "w {w}");
                }
            }
        }
    }
}

#[test]
fn a_window_of_zero_reads_exactly_as_codec_v1() {
    let layers = [
        wide(HistoryRange::Trailing(SLIDING)),
        wide(HistoryRange::Full),
    ];
    let keys = gaussian_rows(21, ROWS);
    let mut mixed = CodecRecentKvState::new(4, 0);
    let mut codec = CodecKvState::new(4);
    mixed.prepare(&layers);
    codec.prepare(&layers);
    for k in &keys {
        for layer in 0..layers.len() {
            mixed.append(layer, k.clone(), k.clone());
            codec.append(layer, k.clone(), k.clone());
        }
    }
    for layer in 0..layers.len() {
        mixed.prepare_layer(layer);
        codec.prepare_layer(layer);
        let (m, c) = (mixed.rows(layer), codec.rows(layer));
        assert_eq!((m.base(), m.end()), (c.base(), c.end()));
        for p in m.base()..m.end() {
            assert_eq!(bits(m.key(p)), bits(c.key(p)));
            assert_eq!(bits(m.value(p)), bits(c.value(p)));
        }
        assert!(mixed.exact_keys(layer).is_empty());
    }
}

/// Every held allocation is exactly its declared size: V and aged K codes
/// at codec/v1's half-row bytes, exact K at the row's width — never the
/// incoming row's spare capacity.
#[test]
fn storage_is_exactly_the_declared_bytes_and_reading_never_re_encodes() {
    let mut state = CodecRecentKvState::new(4, 5);
    state.prepare(&[wide(HistoryRange::Full)]);
    for k in gaussian_rows(31, ROWS) {
        let mut padded = Vec::with_capacity(KV_DIM * 2);
        padded.extend_from_slice(&k);
        state.append(0, padded, k);
    }
    let half = HEADS * (4 + HEAD_DIM * 4 / 8);
    assert_eq!(state.encoded_half_bytes(0), half);
    let (v, k) = (
        state.encoded_values(0).to_vec(),
        state.encoded_keys(0).to_vec(),
    );
    assert_eq!(
        (v.len(), k.len(), state.exact_keys(0).len()),
        (ROWS, ROWS - 5, 5)
    );
    assert!(v
        .iter()
        .chain(&k)
        .all(|r| r.len() == half && r.capacity() == half));
    assert!(state
        .exact_keys(0)
        .iter()
        .all(|r| r.len() == KV_DIM && r.capacity() == KV_DIM));
    state.prepare_layer(0);
    assert_eq!(state.encoded_values(0), v, "reading never re-encodes");
    assert_eq!(state.encoded_keys(0), k, "reading never re-encodes");
}

#[test]
fn codec_recent_retains_exactly_what_codec_v1_retains_through_every_phase() {
    let (_container, plan, store) = open(miniature_glimmer, "cr-retention");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let registry = shipped_continuations();
    let backend = ReferenceBackend::new();
    let mut held = Vec::new();
    for (identity, cfg) in [
        (CodecKvState::identity(), config(&["bits=4"])),
        (
            CodecRecentKvState::identity(),
            config(&["bits=4", "exact_recent_k=3"]),
        ),
    ] {
        let mut state = registry.select(&identity, &cfg, &geometry).unwrap().build();
        prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut *state).unwrap();
        prefill_plan(&plan, &store, &RESUME, &backend, &mut *state).unwrap();
        let mut session =
            DecodeSession::with_kv_state(&plan, &store, &backend, &mut *state).unwrap();
        for &token in &DECODE {
            let logits = session.step(token).unwrap().logits.unwrap();
            assert!(logits.iter().all(|x| x.is_finite()), "{identity}");
        }
        drop(session);
        let ranges: Vec<(usize, usize)> = (0..geometry.len())
            .map(|layer| {
                state.prepare_layer(layer);
                let v = state.rows(layer);
                (v.base(), v.end())
            })
            .collect();
        held.push(ranges);
    }
    assert_eq!(
        held[0], held[1],
        "the window changes bytes per row, never which rows"
    );
}

#[test]
fn a_head_dim_the_codec_cannot_block_is_refused_by_name_before_any_row() {
    let mut odd = wide(HistoryRange::Full);
    odd.head_dim = 96;
    odd.kv_dim = 192;
    let mut state = CodecRecentKvState::new(4, 8);
    let layers = [
        LayerContinuationGeometry::Kv(wide(HistoryRange::Full)),
        LayerContinuationGeometry::Kv(odd),
    ];
    let refused = state.prepare_continuation(&layers).unwrap_err();
    assert!(
        matches!(
            &refused,
            ContinuationError::GeometryUnsupported { layer: 1, .. }
        ),
        "{refused}"
    );
    assert!(
        refused.to_string().contains("CodecRecentKvState"),
        "{refused}"
    );
}

#[test]
fn recurrent_and_latent_state_are_refused_by_name() {
    let mut state = CodecRecentKvState::new(4, 8);
    state.prepare(&[wide(HistoryRange::Full)]);
    for refused in [
        state.recurrent_state(0).unwrap_err().to_string(),
        state.latent_state(0).unwrap_err().to_string(),
    ] {
        assert!(refused.contains("CodecRecentKvState"), "{refused}");
    }
}

#[test]
#[should_panic(expected = "K row at layer 0 is 3 wide; the plan says 256")]
fn a_misfit_row_is_refused() {
    let mut state = CodecRecentKvState::new(4, 8);
    state.prepare(&[wide(HistoryRange::Full)]);
    state.append(0, vec![0.0; 3], vec![0.0; KV_DIM]);
}

#[test]
#[should_panic(expected = "different program geometry")]
fn a_resumed_state_refuses_another_geometry() {
    let mut state = CodecRecentKvState::new(4, 8);
    state.prepare(&[wide(HistoryRange::Full)]);
    state.prepare(&[wide(HistoryRange::Trailing(4))]);
}

/// The scratch `scratch()` exposes is the one `rows` lends: its first
/// held-range rows are the view's, and it grows exactly to the widest
/// range decoded, never an amortised doubling past it.
#[test]
fn the_inspected_scratch_is_the_lent_one_at_exact_capacity() {
    let mut state = CodecRecentKvState::new(4, 3);
    state.prepare(&[wide(HistoryRange::Full)]);
    for k in gaussian_rows(41, 7) {
        state.append(0, k.clone(), k);
    }
    state.prepare_layer(0);
    let (keys, values) = state.scratch();
    assert_eq!(
        (keys.capacity(), values.capacity()),
        (7 * KV_DIM, 7 * KV_DIM)
    );
    let view = state.rows(0);
    for p in 0..7 {
        assert_eq!(view.key(p), &keys[p * KV_DIM..(p + 1) * KV_DIM]);
        assert_eq!(view.value(p), &values[p * KV_DIM..(p + 1) * KV_DIM]);
    }
    assert_eq!(view.key(0).as_ptr(), keys.as_ptr(), "lent in place");
}

/// A latent layer is refused as latent, by name, before any row.
#[test]
fn a_latent_layer_is_refused_as_latent_before_any_row() {
    let mut state = CodecRecentKvState::new(4, 8);
    let layers = [
        LayerContinuationGeometry::Kv(wide(HistoryRange::Full)),
        LayerContinuationGeometry::LatentKv(LayerLatentKvGeometry { width: KV_DIM }),
    ];
    let refused = state.prepare_continuation(&layers).unwrap_err();
    assert!(
        matches!(
            &refused,
            ContinuationError::LatentUnsupported { layer: 1, .. }
        ),
        "{refused}"
    );
    assert!(
        refused.to_string().contains("CodecRecentKvState"),
        "{refused}"
    );
}
