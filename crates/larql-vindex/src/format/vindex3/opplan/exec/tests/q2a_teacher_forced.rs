//! **Q2a — the teacher-forced quality runner, 1024 positions.**
//!
//! Two arms through the real mixed Metal stack, loaded from the source
//! VINDEX3 container itself:
//!
//! ```text
//! baseline    every routed layer  -> source expert_bank   (BF16, Table)
//! candidate   layer 1 routed      -> compiled Q6_K bank   (Identity)
//!             everything else     -> the SAME source stores
//! ```
//!
//! The single changed variable is layer 1's routed ExpertWeight
//! representation: BF16 source bytes vs the exact persistent Q6_K bytes
//! the compiler sealed. Router, norms, attention, shared experts, state
//! initialisation and the teacher-forced token prefix are identical by
//! construction — and asserted, not assumed.
//!
//! **This run is NON-PROMOTABLE by design**: 1024 positions is under
//! `kimi-logit-v1`'s `positions_min` of 4096, so the gate must refuse
//! however flattering the metrics are. Its product is the machinery and
//! the closure report, not a verdict; Q2b runs the full 8192.
//!
//! ```text
//! LARQL_KIMI_VINDEX3=~/chris-models/Kimi-Linear-48B-A3B-Instruct.vindex3 \
//! LARQL_KIMI_Q6_CANDIDATE=/tmp/kimi-q6-candidate.vindex3 \
//! LARQL_KIMI_QUALITY_BANK=/tmp/kimi_quality_bank \
//!   cargo test -p larql-vindex --features gpu --release --lib q2a_teacher_forced -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute::backend::ComputeBackend;
use larql_compute_metal::trait_impl::kimi_layer::ExecutionTrace;
use larql_compute_metal::MetalBackend;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::kimi_source::{
    verify_complete, CandidateOverlay, KimiSourceModel,
};
use crate::format::vindex3::opplan::exec::stack::{LayerSpec, LayerState};
use crate::format::vindex3::opplan::exec::stack_metal::{DeviceLayer, HybridStack};
use crate::format::vindex3::represent::bank::{BankBuilder, PositionObservation, TopKChange};
use crate::format::vindex3::represent::physical::{ExpertEncoding, ProjectionAddressing};
use crate::format::vindex3::represent::quality::{
    kimi_logit_v1, kimi_logit_v2, Criterion, QualityEvidence,
};

const SOURCE_ENV: &str = "LARQL_KIMI_VINDEX3";
const CANDIDATE_ENV: &str = "LARQL_KIMI_Q6_CANDIDATE";
const BANK_ENV: &str = "LARQL_KIMI_QUALITY_BANK";

/// Q2a's slice of the exported bank: 32 of the 256 sequences.
///
/// Overridable, because SCOPE SEARCH is now the experiment and a
/// diagnostic that only has to separate "this projection cascades" from
/// "this one does not" does not need the full slice. Q2a's own headline
/// run is 32; a projection or depth probe is 8.
const SEQUENCES_ENV: &str = "LARQL_Q2A_SEQUENCES";
const SEQUENCES: usize = 32;

fn sequences() -> usize {
    std::env::var(SEQUENCES_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SEQUENCES)
}

/// A label for the report, so a sweep's outputs do not overwrite each
/// other and each names the scope it measured.
fn run_label() -> String {
    std::env::var("LARQL_Q2A_LABEL").unwrap_or_else(|_| "q2a".to_string())
}
/// Sequences the null arm re-runs — enough positions for the all-zero
/// claim to cover real routing variety, cheap enough not to double the
/// run.
const NULL_SEQUENCES: usize = 4;
/// Baseline top-N kept per position. KL is exact on the mass these
/// carry; the minimum covered mass is reported so a too-flat position
/// announces itself.
///
/// 128 covered only 31 % of the baseline's mass at the worst position
/// of the first run — a teacher-forced sequence's FIRST position has no
/// context, so its distribution is close to flat over 163,840 ids and a
/// short truncation sees almost none of it. Widening costs nothing
/// measurable (the rank sort already runs for argmax and top-10) and
/// buys a KL that is exact on most of the distribution instead of a
/// third of it.
const TOP_N: usize = 2048;
/// Where the machine-readable closure reports are written, one per
/// labelled run.
fn report_path(label: &str) -> String {
    format!("/tmp/kimi_{label}_report.json")
}

fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from)
}

fn logsumexp(v: &[f32]) -> f32 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are never NaN"))
        .expect("non-empty")
        .0
}

fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .expect("logits are never NaN")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// One position's paired measurement, from both arms' full logit
/// vectors, traced routes, and the router's own scores.
#[allow(clippy::too_many_arguments)]
fn observation(
    seq: usize,
    pos: usize,
    baseline: &[f32],
    base_trace: &ExecutionTrace,
    candidate: &[f32],
    cand_trace: &ExecutionTrace,
) -> PositionObservation {
    let top_ids = top_k_ids(baseline, TOP_N);
    PositionObservation {
        sequence: seq as u32,
        position: pos as u32,
        baseline_logits: top_ids.iter().map(|&i| baseline[i]).collect(),
        candidate_logits: top_ids.iter().map(|&i| candidate[i]).collect(),
        top_ids: top_ids.iter().map(|&i| i as u32).collect(),
        baseline_logsumexp: logsumexp(baseline),
        candidate_logsumexp: logsumexp(candidate),
        baseline_argmax: argmax(baseline) as u32,
        candidate_argmax: argmax(candidate) as u32,
        baseline_top10: top_k_ids(baseline, 10).iter().map(|&i| i as u32).collect(),
        candidate_top10: top_k_ids(candidate, 10).iter().map(|&i| i as u32).collect(),
        // Every changed route WEIGHED — how close the decision was and
        // how much mixture mass moved — from the router's own selection
        // scores, so a near-tie swap and an overturned decision are
        // distinguishable rather than both being "one flip".
        route_changes: PositionObservation::weigh_route_changes(
            &base_trace.routes,
            &cand_trace.routes,
            &base_trace.selection_scores,
            &base_trace.combine_weights,
            &cand_trace.combine_weights,
        ),
        // The top-10 change WEIGHED, recorded only where the ordering
        // actually moved: a position that did not move says nothing
        // about how close the ones that did were.
        top10_change: weigh_top_k(baseline, candidate, 10),
        baseline_routes: base_trace.routes.clone(),
        candidate_routes: cand_trace.routes.clone(),
    }
}

/// **What a top-k reordering actually was**, or `None` if the ordering
/// did not change.
///
/// Four facts, because the count answers none of them: how close the
/// boundary was, what the CANDIDATE did to that same pair, how much
/// probability mass moved, and how far anything travelled in rank.
fn weigh_top_k(baseline: &[f32], candidate: &[f32], k: usize) -> Option<TopKChange> {
    let (b_top, c_top) = (top_k_ids(baseline, k), top_k_ids(candidate, k));
    if b_top == c_top {
        return None;
    }
    // The boundary the baseline drew, and what the candidate did to
    // exactly those two ids. Comparing the two is the measurement a
    // worst-case `max|dlogit|` over the whole vocabulary cannot give.
    let ranked = top_k_ids(baseline, k + 1);
    let (boundary_margin, candidate_margin_same_ids) = if ranked.len() == k + 1 {
        let (lo, hi) = (ranked[k - 1], ranked[k]);
        (baseline[lo] - baseline[hi], candidate[lo] - candidate[hi])
    } else {
        (f32::NAN, f32::NAN)
    };

    // Half the L1 between the arms' top-k mass, each normalised over
    // its own k — the top-k analogue of the routed-mixture distance.
    let softmax_over = |logits: &[f32], ids: &[usize]| -> Vec<(usize, f32)> {
        let m = ids
            .iter()
            .map(|i| logits[*i])
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = ids.iter().map(|i| (logits[*i] - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        ids.iter().zip(exps).map(|(i, e)| (*i, e / sum)).collect()
    };
    let (bp, cp) = (
        softmax_over(baseline, &b_top),
        softmax_over(candidate, &c_top),
    );
    let mass_of = |v: &[(usize, f32)], id: usize| {
        v.iter()
            .find(|(i, _)| *i == id)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    };
    let mut union: Vec<usize> = b_top.iter().chain(&c_top).copied().collect();
    union.sort_unstable();
    union.dedup();
    let mass_displaced = 0.5
        * union
            .iter()
            .map(|id| (mass_of(&bp, *id) - mass_of(&cp, *id)).abs())
            .sum::<f32>();

    // How far anything travelled. An id present in one arm's top-k and
    // absent from the other is located in the other's FULL ordering, so
    // an outsider arriving from rank 400 is not scored as a 1-place
    // move.
    let rank_in = |ordering: &[usize], id: usize, full: &[f32]| -> usize {
        ordering
            .iter()
            .position(|x| *x == id)
            .unwrap_or_else(|| full.iter().filter(|v| **v > full[id]).count())
    };
    let max_rank_displacement = union
        .iter()
        .map(|id| {
            let b = rank_in(&b_top, *id, baseline);
            let c = rank_in(&c_top, *id, candidate);
            b.abs_diff(c) as u32
        })
        .max()
        .unwrap_or(0);

    Some(TopKChange {
        boundary_margin,
        candidate_margin_same_ids,
        mass_displaced,
        max_rank_displacement,
    })
}

/// Per-sequence embedding rows from the exported bank.
fn sequence_embeddings(dir: &Path, seq: usize, positions: usize, hidden: usize) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(dir.join(format!("seq_{seq}.f32")))
        .unwrap_or_else(|e| panic!("seq_{seq}.f32: {e}"));
    assert_eq!(
        bytes.len(),
        positions * hidden * 4,
        "seq_{seq} must hold {positions}x{hidden} f32 rows"
    );
    bytes
        .chunks_exact(hidden * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect()
}

/// Build one arm: every layer on device, the head riding the last
/// epoch. `overlay` present = the candidate arm.
fn build_stack<'a>(
    metal: &MetalBackend,
    model: &KimiSourceModel,
    overlay: Option<&CandidateOverlay>,
) -> HybridStack<'a> {
    let n = model.geometry.num_layers;
    let mut device: Vec<Option<DeviceLayer>> = Vec::with_capacity(n);
    for i in 0..n {
        device.push(Some(model.device_layer(metal, i, overlay).unwrap_or_else(
            |e| panic!("layer {i} must load with zero missing operands: {e}"),
        )));
    }
    // Register the per-stack owned attention banks (the mmap-backed
    // stores are registered once by the caller).
    for d in device.iter().flatten() {
        for bank in d.attention_banks() {
            metal.register_weight_region(bank);
        }
    }
    let host: Vec<Option<(LayerSpec<'a>, LayerState)>> = (0..n).map(|_| None).collect();
    let mut stack = HybridStack::new(device, host);
    assert!(
        stack.attach_head(model.head().expect("the head must load")),
        "the stack ends on a device layer, so the head must attach"
    );
    stack
}

/// Run `positions` teacher-forced steps of one sequence through one
/// arm, returning each position's full logits and its execution trace.
fn run_sequence(
    metal: &MetalBackend,
    stack: &mut HybridStack<'_>,
    rows: &[Vec<f32>],
    hidden: usize,
) -> Vec<(Vec<f32>, ExecutionTrace)> {
    stack
        .reset_states()
        .expect("an all-device stack resets cleanly");
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let mut trace = ExecutionTrace {
            // The router's own selection scores and combine weights, so
            // a routing change can be WEIGHED rather than counted.
            // ~27 KB a token, read after the chain's single wait.
            want_selection_scores: true,
            ..ExecutionTrace::default()
        };
        let (logits, _traces, _t) = stack
            .forward_traced(metal, row, hidden, Some(&mut trace))
            .expect("the teacher-forced step must not refuse");
        out.push((logits, trace));
    }
    out
}

#[test]
fn q2a_teacher_forced_quality_bank_runs_and_the_gate_refuses_on_positions() {
    let (Some(source_dir), Some(candidate_dir), Some(bank_dir)) = (
        env_dir(SOURCE_ENV),
        env_dir(CANDIDATE_ENV),
        env_dir(BANK_ENV),
    ) else {
        eprintln!("skipped: set {SOURCE_ENV}, {CANDIDATE_ENV} and {BANK_ENV}");
        return;
    };
    // The residency SET would try to wire ~94 GB of expert bank, past
    // the wired-collector wall (~45 GB). This run uses registered
    // regions under IMPLICIT residency; refusing is better than
    // silently degrading into a measurement of the collector.
    if std::env::var("LARQL_RESIDENCY_SET").is_ok() {
        panic!("unset LARQL_RESIDENCY_SET: the bank run must use implicit residency");
    }
    let Some(metal) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };

    let manifest: Value =
        serde_json::from_slice(&std::fs::read(bank_dir.join("manifest.json")).expect("manifest"))
            .expect("bank manifest parses");
    let positions_per_seq = manifest["positions"].as_u64().unwrap() as usize;
    let bank_hidden = manifest["hidden"].as_u64().unwrap() as usize;

    let t0 = Instant::now();
    let model = KimiSourceModel::open(&source_dir).expect("source container opens");
    let g = model.geometry.clone();
    assert_eq!(
        g.hidden, bank_hidden,
        "the bank was exported for this model"
    );
    let overlay =
        CandidateOverlay::open(&candidate_dir, &source_dir, &g).expect("candidate overlay opens");
    eprintln!(
        "[q2a] source + candidate opened in {:.1}s; candidate compiles layers {:?}",
        t0.elapsed().as_secs_f64(),
        overlay.compiled_layers()
    );
    let overlay_layer = overlay.compiled_layers()[0] as usize;

    // ── Load both arms; the loader refuses on any missing operand. ──
    let t1 = Instant::now();
    let moe_layers: Vec<u32> = (g.dense_prefix_layers..g.num_layers)
        .map(|l| l as u32)
        .collect();
    let registered = model
        .register_stores(&metal, &moe_layers)
        .expect("stores register");
    overlay.register_store(&metal);
    let mut baseline = build_stack(&metal, &model, None);
    let mut candidate = build_stack(&metal, &model, Some(&overlay));
    metal.seal_weight_regions();
    eprintln!(
        "[q2a] both arms loaded in {:.1}s ({registered} mmap store regions registered, \
         implicit residency)",
        t1.elapsed().as_secs_f64()
    );

    // ── Physical attribution: the intended stores are the bound ones. ──
    {
        let probe_b = model
            .device_layer(&metal, overlay_layer, None)
            .expect("baseline probe layer");
        let probe_c = model
            .device_layer(&metal, overlay_layer, Some(&overlay))
            .expect("candidate probe layer");
        // The BASELINE arm is entirely source-backed, whatever the
        // candidate's scope.
        for (name, proj) in [
            ("gate", &probe_b.bank.gate),
            ("up", &probe_b.bank.up),
            ("down", &probe_b.bank.down),
        ] {
            assert_eq!(
                proj.store_id(),
                "kimi-source-expert-bank",
                "baseline {name}"
            );
            assert_eq!(proj.encoding(), ExpertEncoding::Bf16, "baseline {name}");
        }
        // The CANDIDATE arm substitutes exactly the projections the
        // overlay compiled, and leaves the rest source-backed — the
        // asymmetry a projection-scoped experiment IS.
        let compiled: Vec<&str> = overlay
            .compiled_projections()
            .iter()
            .map(String::as_str)
            .collect();
        assert!(!compiled.is_empty(), "an overlay must compile something");
        for (name, proj, spelling) in [
            ("gate", &probe_c.bank.gate, "w1"),
            ("up", &probe_c.bank.up, "w3"),
            ("down", &probe_c.bank.down, "w2"),
        ] {
            let want_candidate = compiled.contains(&spelling);
            let (store, enc) = if want_candidate {
                ("kimi-q6-candidate", ExpertEncoding::Q6K)
            } else {
                ("kimi-source-expert-bank", ExpertEncoding::Bf16)
            };
            assert_eq!(proj.store_id(), store, "candidate {name}");
            assert_eq!(proj.encoding(), enc, "candidate {name}");
            // A source-backed projection of the candidate arm must be
            // the SAME BYTES as the baseline's — pointer-identical, so
            // the only difference between the arms is the compiled one.
            if !want_candidate {
                let b = match name {
                    "gate" => &probe_b.bank.gate,
                    "up" => &probe_b.bank.up,
                    _ => &probe_b.bank.down,
                };
                assert_eq!(
                    proj.region.region.bytes().as_ptr(),
                    b.region.region.bytes().as_ptr(),
                    "{name} is outside the scope, so both arms must bind one object"
                );
            }
        }
        for (name, proj) in [
            ("gate", &probe_c.bank.gate),
            ("up", &probe_c.bank.up),
            ("down", &probe_c.bank.down),
        ] {
            // Only a COMPILED projection is identity-addressed. A
            // projection-scoped candidate leaves the others
            // table-addressed over the source, and that asymmetry
            // inside one layer is the thing this binding exists to
            // express.
            let is_candidate = proj.store_id() == "kimi-q6-candidate";
            match (&proj.addressing, is_candidate) {
                (ProjectionAddressing::Identity { experts, .. }, true) => assert_eq!(
                    *experts, g.experts,
                    "{name} is compiled, so it addresses EVERY expert by identity — any \
                     route, including one the baseline never took, resolves"
                ),
                (ProjectionAddressing::Table(_), false) => {}
                (a, c) => panic!("{name}: compiled={c} but addressed by {a:?}"),
            }
        }
        let shared = probe_c
            .bank
            .shared
            .as_ref()
            .expect("candidate layer keeps its shared expert");
        assert_eq!(shared.gate.store_id(), "kimi-source-decoder-stack");
        assert_eq!(shared.gate.encoding, ExpertEncoding::Bf16);
        // A layer the overlay does NOT compile is the same physical
        // bytes in both arms — pointer-identical, not merely same-named.
        // A layer the overlay does NOT compile. The one after,
        // normally — but the deepest layer has none after it, and the
        // probe's claim is about a source-precision NEIGHBOUR, not
        // about direction.
        let neighbour = if overlay_layer + 1 < g.num_layers {
            overlay_layer + 1
        } else {
            overlay_layer - 1
        };
        assert!(
            neighbour < g.num_layers
                && neighbour >= g.dense_prefix_layers
                && !overlay.compiled_layers().contains(&(neighbour as u32)),
            "the neighbour probe needs a routed source-precision layer"
        );
        let nb = model.device_layer(&metal, neighbour, None).expect("nb");
        let nc = model
            .device_layer(&metal, neighbour, Some(&overlay))
            .expect("nc");
        assert_eq!(nc.bank.gate.store_id(), "kimi-source-expert-bank");
        assert_eq!(
            nb.bank.gate.region.region.bytes().as_ptr(),
            nc.bank.gate.region.region.bytes().as_ptr(),
            "layer {neighbour} must be the SAME mapped bytes in both arms"
        );
        let checked = overlay
            .verify_reads_match_seals(overlay_layer as u32, 16)
            .expect("the bytes the loader reads must be the bytes the compiler sealed");
        eprintln!(
            "[q2a] attribution: {} — compiled projections {:?} from q6-candidate/Q6_K, the \
             rest source/BF16 and pointer-identical to the baseline; shared from decoder \
             stack in both; layer {neighbour} pointer-identical; {checked} compiled operands \
             hash-match their seals",
            overlay.scope(),
            compiled,
        );
    }

    // ── The null arm: BF16 against itself must be EXACTLY zero. ──
    let t2 = Instant::now();
    {
        let mut null_partner = build_stack(&metal, &model, None);
        let mut builder = BankBuilder::new();
        for seq in 0..NULL_SEQUENCES {
            let rows = sequence_embeddings(&bank_dir, seq, positions_per_seq, g.hidden);
            let a = run_sequence(&metal, &mut baseline, &rows, g.hidden);
            let b = run_sequence(&metal, &mut null_partner, &rows, g.hidden);
            for (pos, ((la, ta), (lb, tb))) in a.into_iter().zip(b).enumerate() {
                builder.observe(&observation(seq, pos, &la, &ta, &lb, &tb));
            }
        }
        let null_bank = builder.finish();
        assert_eq!(
            null_bank.positions,
            (NULL_SEQUENCES * positions_per_seq) as u64
        );
        assert_eq!(
            null_bank.logits.kl_p99, 0.0,
            "null arm KL must be exactly zero"
        );
        assert_eq!(
            null_bank.logits.max_logit_delta, 0.0,
            "null arm logits must be bit-equal"
        );
        assert_eq!(null_bank.logits.top1_flips, 0);
        assert_eq!(null_bank.logits.top10_changes, 0);
        assert_eq!(null_bank.routing.route_flips, 0);
        assert_eq!(null_bank.routing.positions_with_route_change, 0);
        eprintln!(
            "[q2a] null arm: {} positions, everything exactly zero ({:.1}s)",
            null_bank.positions,
            t2.elapsed().as_secs_f64()
        );
    }

    // ── The measurement: 32 sequences x 32 teacher-forced positions. ──
    let t3 = Instant::now();
    let mut builder = BankBuilder::new();
    let mut candidate_l1_routed: std::collections::BTreeSet<u32> =
        std::collections::BTreeSet::new();
    let mut baseline_l1_routed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for seq in 0..sequences() {
        let rows = sequence_embeddings(&bank_dir, seq, positions_per_seq, g.hidden);
        let base = run_sequence(&metal, &mut baseline, &rows, g.hidden);
        let cand = run_sequence(&metal, &mut candidate, &rows, g.hidden);
        for (pos, ((lb, tb), (lc, tc))) in base.into_iter().zip(cand).enumerate() {
            baseline_l1_routed.extend(tb.routes[overlay_layer].iter().copied());
            candidate_l1_routed.extend(tc.routes[overlay_layer].iter().copied());
            builder.observe(&observation(seq, pos, &lb, &tb, &lc, &tc));
        }
    }
    let min_covered = builder.min_covered_mass();
    let bank = builder.finish();
    let run_s = t3.elapsed().as_secs_f64();
    assert_eq!(bank.positions, (sequences() * positions_per_seq) as u64);

    // ── The gate refuses on positions, whatever the numbers say. ──
    //
    // Judged by v2, which additionally requires the bank's KL to have
    // SEEN most of the distribution. Both are reported, because v1 is
    // what every earlier claim in this programme cited and a reader
    // needs to compare like with like.
    let evidence = QualityEvidence {
        gate: kimi_logit_v2(),
        bank: bank.clone(),
    };
    let verdict = evidence.verdict();
    let v1_verdict = kimi_logit_v1().evaluate(&bank);
    assert!(
        !verdict.passed(),
        "a sub-4096-position bank can never pass kimi-logit-v2"
    );
    assert!(
        verdict
            .failures
            .iter()
            .any(|(c, d)| *c == Criterion::Positions && d.contains("< 4096")),
        "the refusal must name the positions criterion: {verdict:?}"
    );
    assert!(
        evidence.proven_by().is_none(),
        "nothing may cite this run as quality proof"
    );
    // The coverage criterion is satisfied here, so the refusal is
    // about sample count and the candidate's own numbers — not about
    // an instrument that could not see.
    assert!(
        !verdict
            .failures
            .iter()
            .any(|(c, _)| *c == Criterion::CoveredMass),
        "this bank's truncation must be wide enough to judge: {verdict:?}"
    );

    // ── Adversarial no-fallback control: remove ONE compiled operand a
    //    candidate route actually consumed; the overlay must refuse. ──
    let touched = *candidate_l1_routed
        .iter()
        .next()
        .expect("the candidate routed at least once");
    // A projection the overlay ACTUALLY compiled — a scoped candidate
    // holds one, and naming a projection it left at source would test
    // nothing but the fixture.
    let scoped_projection = overlay
        .compiled_projections()
        .first()
        .expect("an overlay compiles at least one projection")
        .clone();
    let tensor =
        format!("{overlay_layer}.block_sparse_moe.experts.{touched}.{scoped_projection}.weight");
    let mut mutated = overlay.index.clone();
    let key = mutated
        .ledger
        .sealed
        .iter()
        .find(|(_, seal)| seal.tensor == tensor)
        .map(|(k, _)| k.clone())
        .unwrap_or_else(|| panic!("the routed operand `{tensor}` must have been sealed"));
    mutated.ledger.sealed.remove(&key);
    let refusal = verify_complete(&mutated, &g).expect_err(
        "an incomplete compiled bank must refuse at load, never fall back to source bytes",
    );
    assert!(
        format!("{refusal}").contains(&tensor),
        "the refusal must name the missing operand: {refusal}"
    );
    verify_complete(&overlay.index, &g).expect("control: the untouched ledger verifies");

    // ── The closure report. ──
    let label = run_label();
    let report = serde_json::json!({
        "run": label,
        "gate": evidence.gate,
        "verdict_failures": verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "verdict_failures_v1": v1_verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "candidate_map": overlay.index.map.name,
        "candidate_layers": overlay.compiled_layers(),
        "bank": bank,
        "positions": bank.positions,
        "sequences": sequences(),
        "top_n": TOP_N,
        "min_covered_mass": min_covered,
        "stores": {
            "baseline_layer1_routed": "kimi-source-expert-bank (BF16, Table)",
            "candidate_layer1_routed": "kimi-q6-candidate (Q6_K, Identity)",
            "shared_experts": "kimi-source-decoder-stack (BF16, both arms)",
        },
        "layer1_routed_experts": {
            "baseline_distinct": baseline_l1_routed.len(),
            "candidate_distinct": candidate_l1_routed.len(),
            "candidate_only": candidate_l1_routed.difference(&baseline_l1_routed).count(),
        },
        "wall_seconds": run_s,
    });
    let path = report_path(&label);
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report).expect("serialises"),
    )
    .expect("report writes");

    eprintln!(
        "[q2a] {} positions in {run_s:.1}s ({:.1} ms/position/arm)",
        bank.positions,
        1000.0 * run_s / (2.0 * bank.positions as f64),
    );
    eprintln!(
        "[q2a] kl p50/p95/p99 {:.3e}/{:.3e}/{:.3e}  max|dlogit| {:.3e}  top1 flips {}  \
         top10 changes {}  (min covered mass {min_covered:.6})",
        bank.logits.kl_p50,
        bank.logits.kl_p95,
        bank.logits.kl_p99,
        bank.logits.max_logit_delta,
        bank.logits.top1_flips,
        bank.logits.top10_changes,
    );
    eprintln!(
        "[q2a] routing: {} flips over {} positions in {} layer(s), FIRST at layer {:?} \
         (perturbed layer is {overlay_layer}); its expert union baseline {} vs candidate {} \
         ({} candidate-only)",
        bank.routing.route_flips,
        bank.routing.positions_with_route_change,
        bank.routing.layers_with_route_change,
        bank.routing.first_layer_with_route_change,
        baseline_l1_routed.len(),
        candidate_l1_routed.len(),
        candidate_l1_routed.difference(&baseline_l1_routed).count(),
    );
    // **Severity, not count.** Whether the changed routes were
    // near-ties the perturbation nudged across, or decisions the router
    // held with confidence — the difference the flip count cannot show.
    if let Some(m) = &bank.routing.route_margin {
        eprintln!(
            "[q2a] route margins ({} changes): min {:.3e} p50 {:.3e} p95 {:.3e} max {:.3e}",
            m.count, m.min, m.p50, m.p95, m.max
        );
    }
    if let Some(m) = &bank.routing.route_weight_mass_moved {
        eprintln!(
            "[q2a] mixture mass moved: min {:.4} p50 {:.4} p95 {:.4} max {:.4} \
             (1.0 = the routed mixture replaced outright)",
            m.min, m.p50, m.p95, m.max
        );
    }
    if let (Some(b), Some(c)) = (&bank.top10_margin, &bank.top10_candidate_margin) {
        eprintln!(
            "[q2a] top-10 boundary at the {} changed positions: baseline gap p50 {:.3e} \
             max {:.3e}; the CANDIDATE's gap at the same two ids p50 {:.3e} max {:.3e}",
            b.count, b.p50, b.max, c.p50, c.max
        );
    }
    if let (Some(m), Some(r)) = (&bank.top10_mass_displaced, &bank.top10_rank_displacement) {
        eprintln!(
            "[q2a] top-10 consequence: mass displaced p50 {:.4} p95 {:.4} max {:.4}; \
             furthest rank move p50 {:.0} p95 {:.0} max {:.0}",
            m.p50, m.p95, m.max, r.p50, r.p95, r.max
        );
    }
    // The attribution the shallowest-layer number exists for: a
    // perturbed layer that keeps its OWN routing while later ones move
    // is a cascade through the residual stream, not a different
    // selection at the layer under test.
    if let Some(first) = bank.routing.first_layer_with_route_change {
        eprintln!(
            "[q2a] mechanism: {}",
            if first as usize > overlay_layer {
                "CASCADE — the perturbed layer routes identically; later routers see a \
                 moved hidden state"
            } else {
                "LOCAL — the perturbed layer's own expert selection changed"
            }
        );
    }
    eprintln!(
        "[q2a] verdict: {} (v1 would say: {}) — report at {path}",
        verdict.describe(),
        v1_verdict.describe()
    );
}
