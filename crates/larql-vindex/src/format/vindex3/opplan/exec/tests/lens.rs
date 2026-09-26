//! V3-LENS-1 witnesses (`docs/v3-lens-1-logit-lens.md`): LP1 parity with
//! the lens armed everywhere, LP2 the anchor — the last FFN site's lens
//! is the executor's logits bit for bit, including through a layer
//! scale — LP4 a proper distribution at every depth, the site arming and
//! rank semantics, and the price counted.

use super::decode::fixture as golden_fixture;
use super::gemma4;
use crate::format::vindex3::fixtures::{G_LAYERS, G_TOKENS, G_VOCAB};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{NoopObserver, StepObserver, SublayerSite};
use crate::format::vindex3::opplan::exec::observe_lens::{
    readout_of, LensLayers, LensSites, LogitLens, LENS_METHOD,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::ComponentOpPlan;

const TOKENS: [u32; 2] = [3, 17];
const TOP_K: usize = 3;
const SMALL_TOKENS: [u32; 3] = [3, 1, 5];
/// f64 log-softmax over f32 logits: the exponentials sum to one within this.
const DISTRIBUTION_TOLERANCE: f64 = 1e-9;

fn all_sites() -> LensSites {
    LensSites {
        layers: LensLayers::All,
        attention: true,
        ffn: true,
    }
}

fn run<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    observer: &mut dyn StepObserver,
) -> Vec<Vec<f32>> {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    tokens
        .iter()
        .map(|&t| session.step_observed(t, observer).unwrap().logits.unwrap())
        .collect()
}

#[test]
fn lp1_the_lens_armed_everywhere_leaves_the_logits_bit_identical() {
    fn check<B: PlanBackend>(backend: &B) {
        let (_c, plan, store) = golden_fixture();
        let ops = PreparedOperands::load(&plan, &store, backend, ExecutionSlice::Full).unwrap();
        let plain = run(&plan, &ops, backend, &G_TOKENS, &mut NoopObserver);
        let mut lens = LogitLens::new(&ops, backend, all_sites(), TOKENS.to_vec(), TOP_K);
        let observed = run(&plan, &ops, backend, &G_TOKENS, &mut lens);
        assert_eq!(plain, observed, "the lens changed the arithmetic");
        assert!(lens.failure.is_none());
        assert_eq!(
            lens.head_passes,
            2 * G_LAYERS * G_TOKENS.len(),
            "one pass per armed site"
        );
        assert_eq!(lens.readouts.len(), lens.head_passes);
    }
    check(&ReferenceBackend::new());
    check(&ProductionBackend::new());
}

#[test]
fn lp2_the_last_ffn_site_is_the_executors_logits_bit_for_bit() {
    fn check<B: PlanBackend>(backend: &B) {
        let (_c, plan, store) = golden_fixture();
        let ops = PreparedOperands::load(&plan, &store, backend, ExecutionSlice::Full).unwrap();
        let last = G_LAYERS - 1;
        let sites = LensSites {
            layers: LensLayers::List(vec![last]),
            attention: false,
            ffn: true,
        };
        let mut lens =
            LogitLens::new(&ops, backend, sites, TOKENS.to_vec(), TOP_K).retaining_logits();
        let executor = run(&plan, &ops, backend, &G_TOKENS, &mut lens);
        assert_eq!(lens.readouts.len(), G_TOKENS.len());
        for (position, readout) in lens.readouts.iter().enumerate() {
            assert_eq!(
                (readout.layer, readout.site, readout.position),
                (last, SublayerSite::Ffn, position)
            );
            let lensed = readout.logits.as_ref().unwrap();
            assert!(
                lensed
                    .iter()
                    .zip(&executor[position])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "position {position}: the lens at the last FFN site is not the executor's logits"
            );
        }
    }
    check(&ReferenceBackend::new());
    check(&ProductionBackend::new());
}

/// H2: on a component with a layer scalar the lens must read
/// `scale × after`; if it read the raw `after`, the anchor would fail.
#[test]
fn lp2_the_anchor_holds_through_a_layer_scale() {
    let dir = tempfile::tempdir().unwrap();
    gemma4::miniature_gemma4(dir.path(), None);
    let container = gemma4::encoded(dir.path());
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = gemma4::closure(container.path()).plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    assert!(plan.layers.iter().all(|l| l.layer_scale.is_some()));
    let last = plan.layers.len() - 1;
    let sites = LensSites {
        layers: LensLayers::List(vec![last]),
        attention: false,
        ffn: true,
    };
    let mut lens = LogitLens::new(&ops, &backend, sites, vec![0, 1], TOP_K).retaining_logits();
    let executor = run(&plan, &ops, &backend, &SMALL_TOKENS, &mut lens);
    assert_eq!(lens.readouts.len(), SMALL_TOKENS.len());
    for (position, readout) in lens.readouts.iter().enumerate() {
        let lensed = readout.logits.as_ref().unwrap();
        assert!(
            lensed
                .iter()
                .zip(&executor[position])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "position {position}: the scaled carrier did not reproduce the exit"
        );
    }
}

#[test]
fn lp4_every_depth_is_a_proper_distribution_and_ranks_are_consistent() {
    let (_c, plan, store) = golden_fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let all: Vec<u32> = (0..G_VOCAB as u32).collect();
    let mut lens = LogitLens::new(&ops, &backend, all_sites(), all, TOP_K).retaining_logits();
    run(&plan, &ops, &backend, &G_TOKENS, &mut lens);
    assert!(lens.failure.is_none());
    for readout in &lens.readouts {
        let mass: f64 = readout.tokens.iter().map(|t| t.logprob.exp()).sum();
        assert!((mass - 1.0).abs() < DISTRIBUTION_TOLERANCE, "mass {mass}");
        assert!(readout
            .tokens
            .iter()
            .all(|t| t.logprob <= 0.0 && t.logprob.is_finite()));
        let logits = readout.logits.as_ref().unwrap();
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(readout.tokens[best].rank, 1);
        assert_eq!(readout.top[0].0 as usize, best);
        assert_eq!(readout.top.len(), TOP_K);
        assert!(readout.top.windows(2).all(|w| w[0].1 >= w[1].1));
    }
}

#[test]
fn rank_is_one_plus_the_strictly_greater_and_ties_keep_the_first_on_top() {
    let logits = [1.0f32, 3.0, 3.0, 2.0];
    let (tokens, top) = readout_of(&logits, &[0, 1, 2, 3], 4).unwrap();
    assert_eq!(
        tokens.iter().map(|t| t.rank).collect::<Vec<_>>(),
        vec![4, 1, 1, 3]
    );
    assert_eq!(
        top.iter().map(|t| t.0).collect::<Vec<_>>(),
        vec![1, 2, 3, 0]
    );
    let total: f64 = tokens.iter().map(|t| t.logprob.exp()).sum();
    assert!((total - 1.0).abs() < DISTRIBUTION_TOLERANCE);
    assert!(
        readout_of(&logits, &[4], 1).is_err(),
        "outside the vocabulary"
    );
    assert!(readout_of(&[], &[0], 1).is_err(), "no logits");
    assert_eq!(readout_of(&logits, &[], 0).unwrap().1.len(), 0);
    assert_eq!(LENS_METHOD, "head-v1");
}

#[test]
fn sites_arm_exactly_what_they_say_and_parse_their_spellings() {
    assert_eq!(LensSites::parse_layers("all").unwrap(), LensLayers::All);
    assert_eq!(
        LensSites::parse_layers(" every:3 ").unwrap(),
        LensLayers::Every(3)
    );
    assert_eq!(
        LensSites::parse_layers("0, 4,9").unwrap(),
        LensLayers::List(vec![0, 4, 9])
    );
    assert!(LensSites::parse_layers("every:0").is_err());
    assert!(LensSites::parse_layers("every:x").is_err());
    assert!(LensSites::parse_layers("1,b").is_err());
    assert!(LensSites::parse_layers(" , ").is_err());

    let every = LensSites {
        layers: LensLayers::Every(2),
        attention: false,
        ffn: true,
    };
    assert!(every.armed(0, SublayerSite::Ffn));
    assert!(!every.armed(1, SublayerSite::Ffn));
    assert!(every.armed(4, SublayerSite::Ffn));
    assert!(!every.armed(0, SublayerSite::Attention));
    let default = LensSites::every_ffn();
    assert!(default.armed(7, SublayerSite::Ffn) && !default.armed(7, SublayerSite::Attention));

    // Armed sites are the price: one head pass each, nothing else.
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut lens = LogitLens::new(&ops, &backend, LensSites::every_ffn(), TOKENS.to_vec(), 0);
    run(&plan, &ops, &backend, &G_TOKENS, &mut lens);
    assert_eq!(lens.head_passes, G_LAYERS * G_TOKENS.len());
    assert!(lens
        .readouts
        .iter()
        .all(|r| r.site == SublayerSite::Ffn && r.top.is_empty() && r.logits.is_none()));
    assert_eq!(lens.sites(), &LensSites::every_ffn());
    assert_eq!(lens.tokens(), &TOKENS);
}

#[test]
fn a_token_outside_the_vocabulary_is_a_recorded_failure_that_stops_the_lens() {
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut lens = LogitLens::new(&ops, &backend, all_sites(), vec![G_VOCAB as u32], TOP_K);
    let plain = run(&plan, &ops, &backend, &G_TOKENS, &mut NoopObserver);
    let observed = run(&plan, &ops, &backend, &G_TOKENS, &mut lens);
    assert_eq!(
        plain, observed,
        "a failing lens still cannot touch execution"
    );
    assert!(lens.failure.is_some());
    assert!(lens.readouts.is_empty());
    assert_eq!(lens.head_passes, 1, "it stopped after the first refusal");
}

/// A record holds the lens through `LensReader`, not its concrete type:
/// the trait object must read, price, report and name its tokens exactly
/// as the lens does.
#[test]
fn the_reader_trait_object_reads_prices_and_reports_like_the_lens() {
    use crate::format::vindex3::opplan::exec::observe::CarrierWriteRecord;
    use crate::format::vindex3::opplan::exec::observe_lens::LensReader;
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut reader: Box<dyn LensReader> = Box::new(LogitLens::new(
        &ops,
        &backend,
        LensSites::every_ffn(),
        TOKENS.to_vec(),
        1,
    ));
    assert_eq!(reader.tokens(), &TOKENS);
    assert_eq!(reader.head_passes(), 0);
    assert!(reader.failure().is_none());
    let hidden = ops.hidden();
    let after: Vec<f32> = (0..hidden).map(|i| 0.01 * i as f32).collect();
    let delta = vec![0.0; hidden];
    let record = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Ffn,
        position: 0,
        delta: &delta,
        after: &after,
        layer_scale: None,
    };
    let readout = reader.read(record).expect("an armed FFN site reads");
    assert_eq!(
        (readout.layer, readout.site, readout.position),
        (0, SublayerSite::Ffn, 0)
    );
    assert_eq!(readout.tokens.len(), TOKENS.len());
    assert_eq!(readout.top.len(), 1);
    assert_eq!(reader.head_passes(), 1);
    // An unarmed site reads nothing and costs nothing.
    let attention = CarrierWriteRecord {
        site: SublayerSite::Attention,
        ..record
    };
    assert!(reader.read(attention).is_none());
    assert_eq!(reader.head_passes(), 1);
    // The same carrier through the head directly is the same distribution.
    let direct = ops.head_logits(&backend, &after).unwrap().unwrap();
    let (tokens, _) = readout_of(&direct, &TOKENS, 0).unwrap();
    assert_eq!(tokens, readout.tokens);
    assert!(reader.failure().is_none());
}
