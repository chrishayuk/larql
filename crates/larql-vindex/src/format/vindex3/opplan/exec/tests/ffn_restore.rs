//! SG-EDIT-1 I5 (A): the two FFN restoration contracts, each checked
//! against its own expected state at the address — never by whether an
//! answer recovered.
//!
//! * FFN-write restoration lands `a_k + δ_c` within the declared f32
//!   rounding contract, is bit-exact against the residual add it performs,
//!   and is NOT `r_c` when the corruption reaches the layer's entering
//!   carrier.
//! * Full-carrier restoration lands `r_c` bit for bit.
//!
//! For both, every write before `(ℓ, Ffn)` — at the restored positions and
//! at every other position, including the `(ℓ, Attention)` writes the
//! driver's captured `a_k` and `δ_k` depend on — is bit-identical to the
//! unintervened corrupted run.

use super::decode::fixture;
use super::golden::{G_LAYERS, G_TOKENS};
use super::i5_support::{run, run_plain, same_bits, Recorder, WriteKey};
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::ffn_restore::{
    capture_addresses, ffn_write, ffn_write_contract_ratio, ffn_write_restoration,
    full_carrier_restoration, Restoration,
};
use crate::format::vindex3::opplan::exec::intervene::{CarrierCapture, InterventionPlan};
use crate::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan;
use crate::format::vindex3::opplan::exec::observe::SublayerSite;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

/// The position the corruption swaps, and the positions restored.
const CORRUPTED: usize = 1;
const RESTORED: [usize; 2] = [1, 2];
/// A token the golden prompt does not use at `CORRUPTED`.
const SWAPPED_TOKEN: u32 = 5;

fn corrupted_tokens() -> Vec<u32> {
    let mut tokens = G_TOKENS.to_vec();
    assert_ne!(tokens[CORRUPTED], SWAPPED_TOKEN);
    tokens[CORRUPTED] = SWAPPED_TOKEN;
    tokens
}

fn all_positions() -> Vec<usize> {
    (0..G_TOKENS.len()).collect()
}

/// Run unintervened, recording every write and capturing the carriers the
/// drivers read at `layer`.
fn captured_run<B: PlanBackend>(
    backend: &B,
    tokens: &[u32],
    layer: usize,
) -> (Vec<Vec<f32>>, Recorder) {
    let (_dir, plan, store) = fixture();
    let mut recorder = Recorder {
        carriers: Some(CarrierCapture::at(capture_addresses(
            layer,
            &all_positions(),
        ))),
        ..Recorder::default()
    };
    let logits = run_plain(&plan, &store, backend, tokens, &mut recorder);
    (logits, recorder)
}

fn intervened_run<B: PlanBackend>(
    backend: &B,
    tokens: &[u32],
    plan_: &InterventionPlan,
) -> Recorder {
    let (_dir, plan, store) = fixture();
    let mut recorder = Recorder::default();
    run(
        &plan,
        &store,
        backend,
        tokens,
        &mut recorder,
        plan_,
        &HeadInterventionPlan::none(),
    );
    recorder
}

/// Every write that precedes `(layer, Ffn)` in execution — any earlier
/// layer, and `(layer, Attention)` — at every position, plus every write at
/// positions before the first restored one.
fn upstream_of(key: &WriteKey, layer: usize) -> bool {
    let (l, site, p) = *key;
    l < layer || (l == layer && site == SublayerSite::Attention) || p < RESTORED[0]
}

fn assert_upstream_untouched(label: &str, baseline: &Recorder, run: &Recorder, layer: usize) {
    for (key, after) in &baseline.writes {
        if upstream_of(key, layer) {
            assert!(
                same_bits(after, &run.writes[key]),
                "{label}: upstream write {key:?} changed"
            );
        }
    }
}

fn ffn_write_contract<B: PlanBackend>(backend: &B, label: &str, layer: usize) {
    let (_, clean) = captured_run(backend, &G_TOKENS, layer);
    let (_, corrupt) = captured_run(backend, &corrupted_tokens(), layer);
    let clean_c = clean.carriers.as_ref().unwrap();
    let corrupt_c = corrupt.carriers.as_ref().unwrap();
    let (plan, record) =
        ffn_write_restoration(clean_c, "clean", corrupt_c, "corrupt", layer, &RESTORED).unwrap();
    assert_eq!(plan.declared(), RESTORED.len());
    assert_eq!(record.restoration, Restoration::FfnWrite);
    assert_eq!(record.corrupt_run.as_deref(), Some("corrupt"));
    assert_eq!(
        record
            .positions
            .iter()
            .map(|p| p.position)
            .collect::<Vec<_>>(),
        RESTORED
    );
    let restored = intervened_run(backend, &corrupted_tokens(), &plan);
    assert_upstream_untouched(label, &corrupt, &restored, layer);

    for &p in &RESTORED {
        let after = &restored.writes[&(layer, SublayerSite::Ffn, p)];
        let ratio =
            ffn_write_contract_ratio(clean_c, "clean", corrupt_c, "corrupt", layer, p, after)
                .unwrap();
        assert!(
            ratio <= 1.0,
            "{label}: layer {layer} position {p} misses the FFN-write contract ({ratio})"
        );
        // Bit-exact against the arithmetic the intervention performs:
        // one residual add of (δ_c − δ_k) onto r_k.
        let d_c = ffn_write(clean_c, "clean", layer, p).unwrap();
        let d_k = ffn_write(corrupt_c, "corrupt", layer, p).unwrap();
        let r_k = corrupt_c.get(layer, SublayerSite::Ffn, p).unwrap();
        let recomputed: Vec<f32> = r_k
            .iter()
            .zip(d_c.iter().zip(&d_k))
            .map(|(r, (c, k))| r + (c - k))
            .collect();
        assert!(same_bits(after, &recomputed), "{label}: position {p}");
    }
    // At the corrupted position the entering carriers differ, so the
    // FFN-write restoration is a different state from the clean carrier.
    let a_c = clean_c
        .get(layer, SublayerSite::Attention, CORRUPTED)
        .unwrap();
    let a_k = corrupt_c
        .get(layer, SublayerSite::Attention, CORRUPTED)
        .unwrap();
    assert!(
        !same_bits(a_c, a_k),
        "{label}: the corruption must reach layer {layer}"
    );
    let r_c = clean_c.get(layer, SublayerSite::Ffn, CORRUPTED).unwrap();
    assert!(
        !same_bits(
            &restored.writes[&(layer, SublayerSite::Ffn, CORRUPTED)],
            r_c
        ),
        "{label}: FFN-write restoration must not be read as full-carrier restoration"
    );
}

#[test]
fn ffn_write_restoration_meets_its_contract_on_every_layer_and_backend() {
    for layer in 0..G_LAYERS {
        ffn_write_contract(&ReferenceBackend::new(), "reference", layer);
        ffn_write_contract(&ProductionBackend::new(), "production", layer);
    }
}

fn full_carrier_contract<B: PlanBackend>(backend: &B, label: &str, layer: usize) {
    let (_, clean) = captured_run(backend, &G_TOKENS, layer);
    let (_, corrupt) = captured_run(backend, &corrupted_tokens(), layer);
    let clean_c = clean.carriers.as_ref().unwrap();
    let (plan, record) = full_carrier_restoration(clean_c, "clean", layer, &RESTORED).unwrap();
    assert_eq!(record.restoration, Restoration::FullCarrier);
    assert_eq!(record.corrupt_run, None);
    let restored = intervened_run(backend, &corrupted_tokens(), &plan);
    assert_upstream_untouched(label, &corrupt, &restored, layer);
    for &p in &RESTORED {
        assert!(
            same_bits(
                &restored.writes[&(layer, SublayerSite::Ffn, p)],
                clean_c.get(layer, SublayerSite::Ffn, p).unwrap()
            ),
            "{label}: layer {layer} position {p} is not r_c"
        );
    }
}

#[test]
fn full_carrier_restoration_lands_the_clean_carrier_bit_for_bit() {
    for layer in 0..G_LAYERS {
        full_carrier_contract(&ReferenceBackend::new(), "reference", layer);
        full_carrier_contract(&ProductionBackend::new(), "production", layer);
    }
}

#[test]
fn restoring_a_run_onto_itself_is_the_no_op() {
    let backend = ProductionBackend::new();
    let layer = 0;
    let (logits, clean) = captured_run(&backend, &G_TOKENS, layer);
    let clean_c = clean.carriers.as_ref().unwrap();
    let (plan, _) =
        ffn_write_restoration(clean_c, "clean", clean_c, "clean", layer, &RESTORED).unwrap();
    let (_dir, op_plan, store) = fixture();
    let mut recorder = Recorder::default();
    let restored_logits = run(
        &op_plan,
        &store,
        &backend,
        &G_TOKENS,
        &mut recorder,
        &plan,
        &HeadInterventionPlan::none(),
    );
    super::i5_support::assert_runs_identical(
        "self-restoration",
        (&logits, &clean.writes),
        (&restored_logits, &recorder.writes),
    );
}

#[test]
fn restoration_refuses_what_it_cannot_derive() {
    let backend = ProductionBackend::new();
    let (_, clean) = captured_run(&backend, &G_TOKENS, 0);
    let c = clean.carriers.as_ref().unwrap();
    let empty = ffn_write_restoration(c, "clean", c, "clean", 0, &[]).unwrap_err();
    assert!(empty.to_string().contains("names no position"), "{empty}");
    assert!(full_carrier_restoration(c, "clean", 0, &[]).is_err());
    // Layer 1 was not captured.
    let missing = ffn_write_restoration(c, "clean", c, "clean", 1, &[0]).unwrap_err();
    assert!(
        missing.to_string().contains("was not captured"),
        "{missing}"
    );
    assert!(full_carrier_restoration(c, "clean", 1, &[0]).is_err());
    let width = ffn_write_contract_ratio(c, "clean", c, "clean", 0, 0, &[0.0]).unwrap_err();
    assert!(width.to_string().contains("width"), "{width}");
}

#[test]
fn the_contract_ratio_flags_a_wrong_state() {
    let backend = ProductionBackend::new();
    let (_, clean) = captured_run(&backend, &G_TOKENS, 0);
    let c = clean.carriers.as_ref().unwrap();
    let r = c.get(0, SublayerSite::Ffn, 0).unwrap();
    // The exact state for a self-restoration is r itself: ratio 0.
    assert_eq!(
        ffn_write_contract_ratio(c, "clean", c, "clean", 0, 0, r).unwrap(),
        0.0
    );
    let off: Vec<f32> = r.iter().map(|v| v + 1.0).collect();
    assert!(ffn_write_contract_ratio(c, "clean", c, "clean", 0, 0, &off).unwrap() > 1.0);
}
