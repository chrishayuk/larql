//! SG-EDIT-1 I5 qualification on a REAL Gemma 3 4B container, on the
//! production CPU backend, over NON-COHORT prompts only (the France pilot,
//! a Germany subject swap, and one generic prompt; France and Germany are
//! GW-0 subjects, excluded from the GW-V2 cohort).
//!
//! Instrument evidence only: every check compares bits or a declared
//! rounding contract; no answer is decoded. Ignored by default:
//!
//! ```sh
//! LARQL_V3_CONTAINER=~/chris-models/gemma3-4b-it.vindex3 \
//!   cargo test --release -p larql-vindex --lib i5_real -- --ignored --nocapture
//! ```
//!
//! G1 geometry: the declared head layout at the three SG-EDIT-1 sites
//!    (L6, L14, L24): 8 query heads on 4 KV heads, `head_dim` 256, a
//!    `[2560, 2048]` `o_proj`, no sinks, and H0/H1 reading identical V rows.
//! G2 (E): a positioned `down_proj`-input capture over the S1 band leaves
//!    every carrier write and logit bit-identical.
//! G3 (B): a natural replay reproduces `ctx_h` bit for bit at every
//!    captured (layer, head, position), and a `Replace` with it leaves the
//!    run bit-identical at L24 H1, L6 H6 and L14 H3.
//! G4 (A): FFN-write restoration meets its contract against `a_k + δ_c`
//!    and the residual add it performs; full-carrier restoration lands
//!    `r_c` bit for bit; every upstream write is bit-identical.

use std::collections::BTreeMap;

use super::i5_support::{assert_runs_identical, same_bits, Recorder, WriteKey};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::down_input_capture::DownInputCapture;
use crate::format::vindex3::opplan::exec::ffn_restore::{
    capture_addresses, ffn_write, ffn_write_contract_ratio, ffn_write_restoration,
    full_carrier_restoration,
};
use crate::format::vindex3::opplan::exec::intervene::{CarrierCapture, InterventionPlan};
use crate::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::SublayerSite;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::source_head::SourceHeadCapture;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// `<bos>The capital of France is`, pinned Gemma 3 tokenizer.
const FRANCE: [u32; 6] = [2, 818, 5279, 529, 7001, 563];
/// The same template with the one-token subject swapped to Germany.
const GERMANY: [u32; 6] = [2, 818, 5279, 529, 9405, 563];
/// `<bos>The weather in the mountains today is`.
const GENERIC: [u32; 8] = [2, 818, 7606, 528, 506, 13254, 3124, 563];
/// The subject position in the capital template.
const SUBJECT: usize = 4;

/// Declared Gemma 3 4B geometry.
const HIDDEN: usize = 2560;
const Q_HEADS: usize = 8;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
/// The SG-EDIT-1 head sites: the read (L24 H1) and the two
/// relation-control heads (B6 H6, B14 H3).
const HEAD_SITES: [(usize, usize); 3] = [(24, 1), (6, 6), (14, 3)];
/// The S1 candidate band, and the layers restored in G4 (both ends and
/// the middle of the band).
const S1_BAND: std::ops::RangeInclusive<usize> = 4..=20;
const RESTORED_LAYERS: [usize; 3] = [4, 12, 20];

struct Image {
    plan: ComponentOpPlan,
    ops: PreparedOperands,
    backend: ProductionBackend,
}

impl Image {
    fn open() -> Self {
        let path = std::env::var("LARQL_V3_CONTAINER")
            .expect("set LARQL_V3_CONTAINER to the Gemma 3 4B container directory");
        let root = std::path::Path::new(&path);
        let inspection = inspect_container(root, false).unwrap();
        let outcome = plan_component_ops(&inspection, root, "target").unwrap();
        assert!(outcome.closed(), "defects: {:?}", outcome.defects);
        let plan = outcome.plan.unwrap();
        let store = OperandStore::open(root, &inspection).unwrap();
        let backend = ProductionBackend::new();
        let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
        Self { plan, ops, backend }
    }

    fn run(
        &self,
        tokens: &[u32],
        recorder: &mut Recorder,
        interventions: &InterventionPlan,
        heads: &HeadInterventionPlan,
    ) -> Vec<Vec<f32>> {
        let mut kv = RowKvState::default();
        let mut session =
            DecodeSession::over_prepared(&self.plan, &self.ops, &self.backend, &mut kv).unwrap();
        tokens
            .iter()
            .map(|&t| {
                session
                    .step_intervened(t, recorder, interventions, heads)
                    .unwrap()
                    .logits
                    .unwrap()
            })
            .collect()
    }

    fn plain(&self, tokens: &[u32], recorder: &mut Recorder) -> Vec<Vec<f32>> {
        self.run(
            tokens,
            recorder,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
        )
    }
}

fn g1_geometry(image: &Image) {
    for &(layer, _) in &HEAD_SITES {
        let op = image.plan.layers[layer]
            .attention
            .softmax()
            .unwrap_or_else(|| panic!("layer {layer} has no softmax attention"));
        assert_eq!(
            (op.num_q_heads, op.num_kv_heads, op.head_dim),
            (Q_HEADS, KV_HEADS, HEAD_DIM),
            "layer {layer} head geometry"
        );
        assert_eq!(
            op.o.shape,
            [HIDDEN, Q_HEADS * HEAD_DIM],
            "layer {layer} o_proj"
        );
        assert!(op.sinks.is_none(), "layer {layer} declares sinks");
        assert!(op.output_gate.is_none() && op.o_bias.is_none());
        println!(
            "G1 layer {layer}: q {} kv {} head_dim {} o {:?} score_scale {} softcap {:?} span \
             {:?} window {:?}",
            op.num_q_heads,
            op.num_kv_heads,
            op.head_dim,
            op.o.shape,
            op.score_scale,
            op.logit_softcapping,
            op.span,
            op.window
        );
    }
}

fn every_head_address(tokens: usize) -> Vec<(usize, usize, usize)> {
    let mut addresses = Vec::new();
    for &(layer, _) in &HEAD_SITES {
        for head in 0..Q_HEADS {
            for position in 0..tokens {
                addresses.push((layer, head, position));
            }
        }
    }
    addresses
}

fn g2_capture(image: &Image, label: &str, tokens: &[u32]) {
    let mut plain = Recorder::default();
    let plain_logits = image.plain(tokens, &mut plain);
    let addresses: Vec<(usize, usize)> = S1_BAND
        .flat_map(|layer| (0..tokens.len()).map(move |p| (layer, p)))
        .collect();
    let mut captured = Recorder {
        down: Some(DownInputCapture::at(addresses.clone())),
        ..Recorder::default()
    };
    let captured_logits = image.plain(tokens, &mut captured);
    assert_runs_identical(
        &format!("G2 {label}"),
        (&plain_logits, &plain.writes),
        (&captured_logits, &captured.writes),
    );
    let down = captured.down.unwrap();
    assert!(
        down.missing().is_empty(),
        "G2 {label}: {:?}",
        down.missing()
    );
    println!(
        "G2 {label}: {} down inputs captured (width {}), run bit-identical",
        down.captured(),
        down.get(*S1_BAND.start(), 0).unwrap().len()
    );
}

fn g3_heads(image: &Image, label: &str, tokens: &[u32]) {
    let mut baseline = Recorder {
        heads: Some(SourceHeadCapture::at(every_head_address(tokens.len()))),
        ..Recorder::default()
    };
    let baseline_logits = image.plain(tokens, &mut baseline);
    let heads = baseline.heads.as_ref().unwrap();
    let mut replays = 0usize;
    for (layer, head, position) in every_head_address(tokens.len()) {
        let source = heads.get(layer, head, position).unwrap();
        let replayed = source.replay(&image.plan).unwrap();
        assert!(
            same_bits(&replayed, &source.ctx),
            "G3 {label}: layer {layer} head {head} position {position}: replay != ctx_h"
        );
        replays += 1;
    }
    // H0 and H1 share KV head 0: identical value rows.
    let last = tokens.len() - 1;
    let (h0, h1) = (
        heads.get(24, 0, last).unwrap(),
        heads.get(24, 1, last).unwrap(),
    );
    assert_eq!(h0.values.len(), h1.values.len());
    for (a, b) in h0.values.iter().zip(&h1.values) {
        assert!(
            same_bits(a, b),
            "G1 {label}: L24 H0 and H1 read different V rows"
        );
    }
    println!("G3 {label}: {replays} natural replays bit-exact; L24 H0/H1 share V rows");
    for &(layer, head) in &HEAD_SITES {
        let source = heads.get(layer, head, last).unwrap();
        let replace = source
            .replace_with(source.replay(&image.plan).unwrap())
            .unwrap();
        let plan = HeadInterventionPlan::none().with(replace).unwrap();
        let mut recorder = Recorder::default();
        let logits = image.run(tokens, &mut recorder, &InterventionPlan::none(), &plan);
        assert_runs_identical(
            &format!("G3 {label} L{layer} H{head}"),
            (&baseline_logits, &baseline.writes),
            (&logits, &recorder.writes),
        );
        println!("G3 {label}: Replace(replay) at L{layer} H{head} leaves the run bit-identical");
    }
}

fn upstream_of(key: &WriteKey, layer: usize) -> bool {
    let (l, site, p) = *key;
    l < layer || (l == layer && site == SublayerSite::Attention) || p < SUBJECT
}

fn assert_upstream(
    label: &str,
    baseline: &BTreeMap<WriteKey, Vec<f32>>,
    run: &Recorder,
    layer: usize,
) {
    for (key, after) in baseline {
        if upstream_of(key, layer) {
            assert!(
                same_bits(after, &run.writes[key]),
                "{label}: upstream {key:?} changed"
            );
        }
    }
}

fn g4_restoration(image: &Image) {
    let positions: Vec<usize> = (0..FRANCE.len()).collect();
    let addresses: Vec<_> = RESTORED_LAYERS
        .iter()
        .flat_map(|&l| capture_addresses(l, &positions))
        .collect();
    let mut clean = Recorder {
        carriers: Some(CarrierCapture::at(addresses.clone())),
        ..Recorder::default()
    };
    image.plain(&FRANCE, &mut clean);
    let mut corrupt = Recorder {
        carriers: Some(CarrierCapture::at(addresses)),
        ..Recorder::default()
    };
    image.plain(&GERMANY, &mut corrupt);
    let (cc, kc) = (
        clean.carriers.as_ref().unwrap(),
        corrupt.carriers.as_ref().unwrap(),
    );
    for &layer in &RESTORED_LAYERS {
        let (plan, _) =
            ffn_write_restoration(cc, "france", kc, "germany", layer, &[SUBJECT]).unwrap();
        let mut restored = Recorder::default();
        image.run(
            &GERMANY,
            &mut restored,
            &plan,
            &HeadInterventionPlan::none(),
        );
        let label = format!("G4 FFN-write L{layer}");
        assert_upstream(&label, &corrupt.writes, &restored, layer);
        let after = &restored.writes[&(layer, SublayerSite::Ffn, SUBJECT)];
        let ratio =
            ffn_write_contract_ratio(cc, "france", kc, "germany", layer, SUBJECT, after).unwrap();
        assert!(ratio <= 1.0, "{label}: contract ratio {ratio}");
        let d_c = ffn_write(cc, "france", layer, SUBJECT).unwrap();
        let d_k = ffn_write(kc, "germany", layer, SUBJECT).unwrap();
        let r_k = kc.get(layer, SublayerSite::Ffn, SUBJECT).unwrap();
        let recomputed: Vec<f32> = r_k
            .iter()
            .zip(d_c.iter().zip(&d_k))
            .map(|(r, (c, k))| r + (c - k))
            .collect();
        assert!(
            same_bits(after, &recomputed),
            "{label}: not the residual add performed"
        );
        let r_c = cc.get(layer, SublayerSite::Ffn, SUBJECT).unwrap();
        let equals_clean = same_bits(after, r_c);
        assert!(
            !equals_clean,
            "{label}: indistinguishable from full-carrier restoration"
        );
        println!("{label}: contract ratio {ratio:.3}, bit-exact add, != r_c, upstream untouched");

        let (plan, _) = full_carrier_restoration(cc, "france", layer, &[SUBJECT]).unwrap();
        let mut full = Recorder::default();
        image.run(&GERMANY, &mut full, &plan, &HeadInterventionPlan::none());
        let label = format!("G4 full-carrier L{layer}");
        assert_upstream(&label, &corrupt.writes, &full, layer);
        assert!(
            same_bits(&full.writes[&(layer, SublayerSite::Ffn, SUBJECT)], r_c),
            "{label}: not r_c"
        );
        println!("{label}: r_c bit for bit, upstream untouched");
    }
}

#[test]
#[ignore = "needs the Gemma 3 4B container in LARQL_V3_CONTAINER"]
fn i5_qualifies_on_real_gemma3_4b_production_cpu() {
    let image = Image::open();
    g1_geometry(&image);
    g2_capture(&image, "france", &FRANCE);
    g2_capture(&image, "generic", &GENERIC);
    g3_heads(&image, "france", &FRANCE);
    g3_heads(&image, "generic", &GENERIC);
    g4_restoration(&image);
    println!("I5 real-model qualification: all gates passed");
}
