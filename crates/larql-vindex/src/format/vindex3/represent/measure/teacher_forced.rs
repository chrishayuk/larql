//! **The teacher-forced quality runner** — two arms through the real
//! mixed Metal stack, loaded from the source VINDEX3 container itself:
//!
//! ```text
//! baseline    every routed layer   -> source expert_bank   (BF16, Table)
//! candidate   the overlay's layers -> compiled banks       (Identity,
//!             each at its map's encoding — Q8_0, Q6_K, or a composed
//!             map holding several layers at different encodings)
//!             everything else      -> the SAME source stores
//! ```
//!
//! The changed variable is exactly the candidate's compiled scope:
//! source BF16 bytes vs the persistent bytes the compiler sealed.
//! Router, norms, attention, shared experts, state initialisation and
//! the teacher-forced token prefix are identical by construction — and
//! asserted, not assumed, per compiled layer.
//!
//! **The verdict semantics follow scale.** Below `positions_min` (4096)
//! this is a DIAGNOSTIC: the gate must refuse on positions however
//! flattering the numbers, and the harness asserts that refusal. At
//! 8192 (`LARQL_Q2A_SEQUENCES=256`) the verdict IS the product — frozen
//! `kimi-logit-v3` decides, the report is written BEFORE any verdict
//! assertion, and a FAIL is a result, not a harness failure.
//!
//! ```text
//! LARQL_KIMI_VINDEX3=~/chris-models/Kimi-Linear-48B-A3B-Instruct.aligned.vindex3 \
//! LARQL_KIMI_Q6_CANDIDATE=/tmp/kimi-q80-l25.vindex3 \
//! LARQL_KIMI_QUALITY_BANK=/tmp/kimi_quality_bank \
//! LARQL_Q2A_SEQUENCES=8 LARQL_Q2A_LABEL=q80-l25 \
//!   cargo test -p larql-vindex --features gpu --release --lib q2a_teacher_forced -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute::backend::ComputeBackend;
use larql_compute_metal::trait_impl::kimi_layer::ExecutionTrace;
use larql_compute_metal::MetalBackend;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::kimi_source::{CandidateOverlay, KimiSourceModel};
use crate::format::vindex3::opplan::exec::stack::{LayerSpec, LayerState};
use crate::format::vindex3::opplan::exec::stack_metal::{DeviceLayer, HybridStack};
use crate::format::vindex3::represent::bank::{
    BankBuilder, PositionObservation, Top1Change, TopKChange,
};
use crate::format::vindex3::represent::measure::outcome::{
    ExecutionFailure, Inadmissible, MeasurementRefusal, VerifiedFacts,
};
use crate::format::vindex3::represent::measure::TeacherForcedRequest;
use crate::format::vindex3::represent::physical::{ExpertEncoding, ProjectionAddressing};
use crate::format::vindex3::represent::quality::QualityBank;
use crate::format::vindex3::represent::quality::{
    kimi_logit_v1, kimi_logit_v2, QualityEvidence, QualityGate,
};

/// The stores an arm may bind, by the id the loader registers them
/// under. Named, because an attribution refusal that compared string
/// literals would be one typo away from never firing.
const SOURCE_EXPERT_BANK: &str = "kimi-source-expert-bank";
const CANDIDATE_BANK: &str = "kimi-candidate-bank";
const SOURCE_DECODER_STACK: &str = "kimi-source-decoder-stack";

/// **What a completed measurement produced, and what it verified.**
///
/// The verdict travels as a recorded fact rather than as authority:
/// this says what the gate decided, and promotion remains the
/// optimiser's to derive. Execution establishes *experiment X produced
/// observation Y*, never *therefore Y is preferred*.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeasurementReceipt {
    /// The request as performed, so a receipt names its own experiment.
    pub request: TeacherForcedRequest,
    /// The gate actually evaluated — checked against the request.
    pub gate: QualityGate,
    /// Every validity condition this run checked.
    pub verified: VerifiedFacts,
    pub bank: QualityBank,
    pub min_covered_mass: f64,
    pub wall_seconds: f64,
    /// Where the closure report was written, before any verdict was
    /// drawn: a refused run still leaves its evidence on disk.
    pub report_path: String,
    pub verdict_passed: bool,
    pub verdict_failures: Vec<String>,
}

impl MeasurementReceipt {
    /// Whether this reading may enter the evidence system.
    ///
    /// NOT whether it passed. A refused candidate that was measured
    /// correctly is admissible evidence of a refusal; a passing
    /// candidate whose run skipped a validity check is not evidence of
    /// anything.
    pub fn qualifies(&self) -> bool {
        self.verified.complete()
    }
}

pub use crate::format::vindex3::represent::measure::{BANK_ENV, CANDIDATE_ENV, SOURCE_ENV};
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

pub fn env_dir(var: &str) -> Option<PathBuf> {
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
pub fn observation(
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
        // The argmax flip weighed, when the winner changed.
        top1_change: weigh_top_1(baseline, candidate),
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

/// **What an argmax flip actually was**, or `None` if the winner did
/// not change.
///
/// Distinguishes a coin-flip between near-equal candidates from an
/// overturned confident choice — two events the flip count scores
/// identically, and the strictest criterion in the contract.
fn weigh_top_1(baseline: &[f32], candidate: &[f32]) -> Option<Top1Change> {
    let (b_win, c_win) = (argmax(baseline), argmax(candidate));
    if b_win == c_win {
        return None;
    }
    // Exact probabilities over the FULL vocabulary, so the mass is the
    // model's own belief rather than a renormalised truncation.
    let lse = logsumexp(baseline);
    let p = |id: usize| (baseline[id] - lse).exp();
    Some(Top1Change {
        // What the baseline held its winner by, over the pair that
        // actually swapped.
        boundary_margin: baseline[b_win] - baseline[c_win],
        // And what the candidate holds the reversed choice by.
        candidate_margin_same_ids: candidate[c_win] - candidate[b_win],
        // The probability the baseline gave up by switching.
        mass_displaced: p(b_win) - p(c_win),
    })
}

/// Per-sequence embedding rows from the exported bank.
pub fn sequence_embeddings(
    dir: &Path,
    seq: usize,
    positions: usize,
    hidden: usize,
) -> Result<Vec<Vec<f32>>, ExecutionFailure> {
    let path = dir.join(format!("seq_{seq}.f32"));
    let unreadable = |detail: String| ExecutionFailure::ArtifactUnreadable {
        what: format!("corpus sequence {seq}"),
        path: path.display().to_string(),
        detail,
    };
    let bytes = std::fs::read(&path).map_err(|e| unreadable(e.to_string()))?;
    if bytes.len() != positions * hidden * 4 {
        return Err(unreadable(format!(
            "holds {} bytes, not the {positions}x{hidden} f32 rows the manifest declares",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(hidden * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect())
}

/// Build one arm: every layer on device, the head riding the last
/// epoch. `overlay` present = the candidate arm.
pub fn build_stack<'a>(
    metal: &MetalBackend,
    model: &KimiSourceModel,
    overlay: Option<&CandidateOverlay>,
) -> Result<HybridStack<'a>, ExecutionFailure> {
    let n = model.geometry.num_layers;
    let mut device: Vec<Option<DeviceLayer>> = Vec::with_capacity(n);
    for i in 0..n {
        device.push(Some(model.device_layer(metal, i, overlay).map_err(
            |e| ExecutionFailure::LayerIncomplete {
                layer: i,
                detail: e.to_string(),
            },
        )?));
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
    let head = model
        .head()
        .map_err(|e| ExecutionFailure::ArtifactUnreadable {
            what: "output head".into(),
            path: String::new(),
            detail: e.to_string(),
        })?;
    if !stack.attach_head(head) {
        return Err(ExecutionFailure::HeadDidNotAttach);
    }
    Ok(stack)
}

/// Run `positions` teacher-forced steps of one sequence through one
/// arm, returning each position's full logits and its execution trace.
pub fn run_sequence(
    metal: &MetalBackend,
    stack: &mut HybridStack<'_>,
    rows: &[Vec<f32>],
    hidden: usize,
) -> Result<Vec<(Vec<f32>, ExecutionTrace)>, ExecutionFailure> {
    stack
        .reset_states()
        .map_err(|e| ExecutionFailure::StepRefused {
            sequence: 0,
            position: 0,
            detail: format!("the stack did not reset: {e}"),
        })?;
    let mut out = Vec::with_capacity(rows.len());
    for (position, row) in rows.iter().enumerate() {
        let mut trace = ExecutionTrace {
            // The router's own selection scores and combine weights, so
            // a routing change can be WEIGHED rather than counted.
            // ~27 KB a token, read after the chain's single wait.
            want_selection_scores: true,
            ..ExecutionTrace::default()
        };
        let (logits, _traces, _t) = stack
            .forward_traced(metal, row, hidden, Some(&mut trace))
            .map_err(|e| ExecutionFailure::StepRefused {
                sequence: 0,
                position,
                detail: e.to_string(),
            })?;
        out.push((logits, trace));
    }
    Ok(out)
}

/// **Run the teacher-forced two-arm measurement.**
///
/// Every condition that made this trustworthy as a `#[cfg(test)]`
/// harness is preserved as a typed refusal — see
/// [`outcome`](super::outcome) for the conservation inventory naming
/// where each one went. A test runner would have turned a violated
/// `assert!` red; a caller gets `Inadmissible` instead, and can tell it
/// from an `ExecutionFailure` that is worth retrying.
pub fn measure_teacher_forced(
    request: &TeacherForcedRequest,
) -> Result<MeasurementReceipt, MeasurementRefusal> {
    // Refused before anything is loaded: an unknown gate, an unknown
    // procedure, an empty slice, a missing artifact.
    let requested_gate = request.admit().map_err(|e| {
        MeasurementRefusal::Execution(ExecutionFailure::ArtifactUnreadable {
            what: "request".into(),
            path: request.source.display().to_string(),
            detail: e.to_string(),
        })
    })?;
    let (source_dir, candidate_dir, bank_dir) = (
        request.source.clone(),
        request.candidate.clone(),
        request.quality_bank.clone(),
    );
    let sequences = || request.sequences;
    let mut verified = VerifiedFacts::default();
    // The residency SET would try to wire ~94 GB of expert bank, past
    // the wired-collector wall (~45 GB). This run uses registered
    // regions under IMPLICIT residency; refusing is better than
    // silently degrading into a measurement of the collector.
    if std::env::var("LARQL_RESIDENCY_SET").is_ok() {
        return Err(Inadmissible::ResidencyModeWouldBeMeasured {
            detail: "LARQL_RESIDENCY_SET is set: the run would wire ~94 GB of expert bank \
                     past the ~45 GB collector wall and measure the collector"
                .into(),
        }
        .into());
    }
    let Some(metal) = MetalBackend::new() else {
        return Err(ExecutionFailure::BackendUnavailable {
            detail: "MetalBackend::new() returned None — the shader library failed to build".into(),
        }
        .into());
    };

    let unreadable = |what: &str, path: &Path, detail: String| {
        MeasurementRefusal::Execution(ExecutionFailure::ArtifactUnreadable {
            what: what.into(),
            path: path.display().to_string(),
            detail,
        })
    };
    let manifest_path = bank_dir.join("manifest.json");
    let manifest: Value = std::fs::read(&manifest_path)
        .map_err(|e| unreadable("quality bank manifest", &manifest_path, e.to_string()))
        .and_then(|b| {
            serde_json::from_slice(&b)
                .map_err(|e| unreadable("quality bank manifest", &manifest_path, e.to_string()))
        })?;
    let field = |k: &str| {
        manifest[k].as_u64().map(|v| v as usize).ok_or_else(|| {
            unreadable(
                "quality bank manifest",
                &manifest_path,
                format!("`{k}` is missing or not a number"),
            )
        })
    };
    let positions_per_seq = field("positions")?;
    let bank_hidden = field("hidden")?;

    let t0 = Instant::now();
    let model = KimiSourceModel::open(&source_dir)
        .map_err(|e| unreadable("source container", &source_dir, e.to_string()))?;
    let g = model.geometry.clone();
    if g.hidden != bank_hidden {
        return Err(Inadmissible::CorpusNotForThisModel {
            corpus_hidden: bank_hidden,
            model_hidden: g.hidden,
        }
        .into());
    }
    let overlay = CandidateOverlay::open(&candidate_dir, &source_dir, &g)
        .map_err(|e| unreadable("candidate overlay", &candidate_dir, e.to_string()))?;
    if overlay.compiled_layers().is_empty() {
        return Err(Inadmissible::CandidateCompilesNothing.into());
    }
    verified.compiled_layers = overlay.compiled_layers().to_vec();
    verified.compiled_projections = overlay.compiled_projections().to_vec();
    eprintln!(
        "[q2a] source + candidate opened in {:.1}s; candidate compiles layers {:?}",
        t0.elapsed().as_secs_f64(),
        overlay.compiled_layers()
    );
    // The EARLIEST compiled layer (verify_complete sorts): the one
    // layer whose router input is untouched in BOTH arms, so its own
    // routing must be identical — the cascade/local discriminator. A
    // composed map's later compiled layers see moved hidden states and
    // may legitimately route differently.
    let overlay_layer = overlay.compiled_layers()[0] as usize;
    let compiled_set: Vec<usize> = overlay
        .compiled_layers()
        .iter()
        .map(|&l| l as usize)
        .collect();

    // ── Load both arms; the loader refuses on any missing operand. ──
    let t1 = Instant::now();
    let moe_layers: Vec<u32> = (g.dense_prefix_layers..g.num_layers)
        .map(|l| l as u32)
        .collect();
    let registered = model
        .register_stores(&metal, &moe_layers)
        .map_err(|e| unreadable("source stores", &source_dir, e.to_string()))?;
    overlay.register_store(&metal);
    let mut baseline = build_stack(&metal, &model, None)?;
    let mut candidate = build_stack(&metal, &model, Some(&overlay))?;
    metal.seal_weight_regions();
    eprintln!(
        "[q2a] both arms loaded in {:.1}s ({registered} mmap store regions registered, \
         implicit residency)",
        t1.elapsed().as_secs_f64()
    );
    // ── Physical attribution: the intended stores are the bound ones,
    //    at EVERY compiled layer — a composed map earns the check per
    //    layer, at that layer's own encoding. Every violation below is
    //    INADMISSIBLE rather than a failure: the logits would still
    //    compute, and would be a measurement of something else.
    for &probe_layer in &compiled_set {
        let probe_b = model.device_layer(&metal, probe_layer, None).map_err(|e| {
            ExecutionFailure::LayerIncomplete {
                layer: probe_layer,
                detail: e.to_string(),
            }
        })?;
        let probe_c = model
            .device_layer(&metal, probe_layer, Some(&overlay))
            .map_err(|e| ExecutionFailure::LayerIncomplete {
                layer: probe_layer,
                detail: e.to_string(),
            })?;
        let unexpected = |arm: &str, projection: &str, want: &str, got: &str| {
            MeasurementRefusal::Inadmissible(Inadmissible::UnexpectedPhysicalRead {
                arm: arm.into(),
                layer: probe_layer,
                projection: projection.into(),
                expected_store: want.into(),
                actual_store: got.into(),
            })
        };

        // The BASELINE arm is entirely source-backed, whatever the
        // candidate's scope.
        for (name, proj) in [
            ("gate", &probe_b.bank.gate),
            ("up", &probe_b.bank.up),
            ("down", &probe_b.bank.down),
        ] {
            if proj.store_id() != SOURCE_EXPERT_BANK {
                return Err(unexpected(
                    "baseline",
                    name,
                    SOURCE_EXPERT_BANK,
                    proj.store_id(),
                ));
            }
            if proj.encoding() != ExpertEncoding::Bf16 {
                return Err(unexpected(
                    "baseline",
                    name,
                    "BF16",
                    &format!("{:?}", proj.encoding()),
                ));
            }
        }

        // The CANDIDATE arm substitutes exactly the projections the
        // overlay compiled, and leaves the rest source-backed — the
        // asymmetry a projection-scoped experiment IS.
        let compiled: Vec<&str> = overlay
            .compiled_projections()
            .iter()
            .map(String::as_str)
            .collect();
        for (name, proj, spelling) in [
            ("gate", &probe_c.bank.gate, "w1"),
            ("up", &probe_c.bank.up, "w3"),
            ("down", &probe_c.bank.down, "w2"),
        ] {
            let want_candidate = compiled.contains(&spelling);
            let (store, enc) = if want_candidate {
                (
                    CANDIDATE_BANK,
                    overlay.encoding_of(probe_layer as u32).map_err(|e| {
                        MeasurementRefusal::Inadmissible(Inadmissible::AddressingMismatch {
                            layer: probe_layer,
                            projection: name.into(),
                            detail: format!(
                                "the layer is in the compiled set and declares no encoding: {e}"
                            ),
                        })
                    })?,
                )
            } else {
                (SOURCE_EXPERT_BANK, ExpertEncoding::Bf16)
            };
            if proj.store_id() != store {
                return Err(unexpected("candidate", name, store, proj.store_id()));
            }
            if proj.encoding() != enc {
                return Err(unexpected(
                    "candidate",
                    name,
                    &format!("{enc:?}"),
                    &format!("{:?}", proj.encoding()),
                ));
            }
            // A source-backed projection of the candidate arm must be
            // the SAME BYTES as the baseline's — pointer-identical, so
            // the only difference between the arms is the compiled one.
            if !want_candidate {
                let b = match name {
                    "gate" => &probe_b.bank.gate,
                    "up" => &probe_b.bank.up,
                    _ => &probe_b.bank.down,
                };
                if proj.region.region.bytes().as_ptr() != b.region.region.bytes().as_ptr() {
                    return Err(Inadmissible::ProtectedOperandChanged {
                        layer: probe_layer,
                        projection: name.into(),
                    }
                    .into());
                }
            }
        }

        // Only a COMPILED projection is identity-addressed. A
        // projection-scoped candidate leaves the others table-addressed
        // over the source, and that asymmetry inside one layer is the
        // thing this binding exists to express.
        for (name, proj) in [
            ("gate", &probe_c.bank.gate),
            ("up", &probe_c.bank.up),
            ("down", &probe_c.bank.down),
        ] {
            let is_candidate = proj.store_id() == CANDIDATE_BANK;
            let mismatch = |detail: String| {
                MeasurementRefusal::Inadmissible(Inadmissible::AddressingMismatch {
                    layer: probe_layer,
                    projection: name.into(),
                    detail,
                })
            };
            match (&proj.addressing, is_candidate) {
                (ProjectionAddressing::Identity { experts, .. }, true) => {
                    if *experts != g.experts {
                        return Err(mismatch(format!(
                            "compiled, so it must address every one of {} experts by identity \
                             — any route, including one the baseline never took, must resolve \
                             — and it addresses {experts}",
                            g.experts
                        )));
                    }
                }
                (ProjectionAddressing::Table(_), false) => {}
                (a, c) => return Err(mismatch(format!("compiled={c} but addressed by {a:?}"))),
            }
        }

        let shared = probe_c
            .bank
            .shared
            .as_ref()
            .ok_or_else(|| unexpected("candidate", "shared", SOURCE_DECODER_STACK, "absent"))?;
        if shared.gate.store_id() != SOURCE_DECODER_STACK {
            return Err(unexpected(
                "candidate",
                "shared",
                SOURCE_DECODER_STACK,
                shared.gate.store_id(),
            ));
        }
        if shared.gate.encoding != ExpertEncoding::Bf16 {
            return Err(unexpected(
                "candidate",
                "shared",
                "BF16",
                &format!("{:?}", shared.gate.encoding),
            ));
        }

        // A layer the overlay does NOT compile is the same physical
        // bytes in both arms — pointer-identical, not merely same-named.
        let neighbour = (g.dense_prefix_layers..g.num_layers)
            .find(|l| !compiled_set.contains(l))
            .ok_or(MeasurementRefusal::Inadmissible(
                Inadmissible::CandidateCompilesNothing,
            ))?;
        let nb = model.device_layer(&metal, neighbour, None).map_err(|e| {
            ExecutionFailure::LayerIncomplete {
                layer: neighbour,
                detail: e.to_string(),
            }
        })?;
        let nc = model
            .device_layer(&metal, neighbour, Some(&overlay))
            .map_err(|e| ExecutionFailure::LayerIncomplete {
                layer: neighbour,
                detail: e.to_string(),
            })?;
        if nc.bank.gate.store_id() != SOURCE_EXPERT_BANK {
            return Err(unexpected(
                "candidate",
                "neighbour gate",
                SOURCE_EXPERT_BANK,
                nc.bank.gate.store_id(),
            ));
        }
        if nb.bank.gate.region.region.bytes().as_ptr()
            != nc.bank.gate.region.region.bytes().as_ptr()
        {
            return Err(Inadmissible::ProtectedOperandChanged {
                layer: neighbour,
                projection: "neighbour gate".into(),
            }
            .into());
        }
        verified.invariant_neighbour_layer = Some(neighbour);

        // The bytes the loader reads must be the bytes the compiler
        // sealed. Without this, "the candidate" is whatever is on disk
        // under a path the overlay names.
        let checked = overlay
            .verify_reads_match_seals(probe_layer as u32, 16)
            .map_err(|e| Inadmissible::SealMismatch {
                layer: probe_layer as u32,
                detail: e.to_string(),
            })?;
        verified.seal_checked_operands += checked;
        verified.attribution_checked_layers.push(probe_layer);
    }

    // ── The null arm: BF16 against itself must be EXACTLY zero. ──
    let t2 = Instant::now();
    {
        let mut null_partner = build_stack(&metal, &model, None)?;
        let mut builder = BankBuilder::new();
        for seq in 0..NULL_SEQUENCES {
            let rows = sequence_embeddings(&bank_dir, seq, positions_per_seq, g.hidden)?;
            let a = run_sequence(&metal, &mut baseline, &rows, g.hidden)?;
            let b = run_sequence(&metal, &mut null_partner, &rows, g.hidden)?;
            for (pos, ((la, ta), (lb, tb))) in a.into_iter().zip(b).enumerate() {
                builder.observe(&observation(seq, pos, &la, &ta, &lb, &tb));
            }
        }
        let null_bank = builder.finish();
        let want = (NULL_SEQUENCES * positions_per_seq) as u64;
        if null_bank.positions != want {
            return Err(Inadmissible::PositionCountMismatch {
                expected: want,
                measured: null_bank.positions,
            }
            .into());
        }
        // **The determinism control.** BF16 against itself must be
        // exactly zero. If it is not, the device is injecting
        // nondeterminism and every downstream KL, flip and route
        // statistic is artifact — while attribution, seals, pointer
        // identity and position counts all still pass.
        for (statistic, observed) in [
            ("kl_p99", null_bank.logits.kl_p99),
            ("max_logit_delta", null_bank.logits.max_logit_delta),
            ("top1_flips", null_bank.logits.top1_flips as f64),
            ("top10_changes", null_bank.logits.top10_changes as f64),
            ("route_flips", null_bank.routing.route_flips as f64),
            (
                "positions_with_route_change",
                null_bank.routing.positions_with_route_change as f64,
            ),
        ] {
            if observed != 0.0 {
                return Err(Inadmissible::NullArmNotZero {
                    statistic: statistic.into(),
                    observed,
                }
                .into());
            }
        }
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
        let rows = sequence_embeddings(&bank_dir, seq, positions_per_seq, g.hidden)?;
        let base = run_sequence(&metal, &mut baseline, &rows, g.hidden)?;
        let cand = run_sequence(&metal, &mut candidate, &rows, g.hidden)?;
        for (pos, ((lb, tb), (lc, tc))) in base.into_iter().zip(cand).enumerate() {
            baseline_l1_routed.extend(tb.routes[overlay_layer].iter().copied());
            candidate_l1_routed.extend(tc.routes[overlay_layer].iter().copied());
            builder.observe(&observation(seq, pos, &lb, &tb, &lc, &tc));
        }
    }
    let min_covered = builder.min_covered_mass();
    let bank = builder.finish();
    let run_s = t3.elapsed().as_secs_f64();
    let want_positions = (sequences() * positions_per_seq) as u64;
    if bank.positions != want_positions {
        return Err(Inadmissible::PositionCountMismatch {
            expected: want_positions,
            measured: bank.positions,
        }
        .into());
    }
    verified.positions = bank.positions;

    // ── The verdicts. v3 is the frozen authority; v1 and v2 are
    // reported because every earlier claim in this programme cited
    // them and a reader needs to compare like with like. ──
    let evidence = QualityEvidence {
        gate: requested_gate.clone(),
        bank: bank.clone(),
    };
    if evidence.gate.id != request.gate {
        return Err(Inadmissible::GateMismatch {
            requested: request.gate.clone(),
            evaluated: evidence.gate.id.clone(),
        }
        .into());
    }
    verified.gate_evaluated = evidence.gate.id.clone();
    let verdict = evidence.verdict();
    let v1_verdict = kimi_logit_v1().evaluate(&bank);
    let v2_verdict = kimi_logit_v2().evaluate(&bank);

    // ── The closure report — written BEFORE any verdict assertion, so
    // a refused run still leaves its evidence on disk. An 8192-position
    // measurement that panics after the numbers exist and before the
    // write has destroyed twenty minutes of instrument time; that
    // happened once and must not happen again. ──
    //
    // The bank's identity travels with the report: /tmp has proven
    // ephemeral, and a verdict whose evidence names no bank cannot be
    // distinguished from a verdict on a different one.
    let bank_manifest_sha256 = {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(&manifest_path)
            .map_err(|e| unreadable("quality bank manifest", &manifest_path, e.to_string()))?;
        format!("{:x}", Sha256::digest(&bytes))
    };
    let label = request.label.clone();
    let report = serde_json::json!({
        "run": label,
        "gate": evidence.gate,
        "authority_report": evidence.report(),
        "verdict_passed": verdict.passed(),
        "verdict_failures": verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "verdict_failures_v2": v2_verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "verdict_failures_v1": v1_verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "candidate_map": overlay.index.map.name,
        "candidate_layers": overlay.compiled_layers(),
        "bank_manifest_sha256": bank_manifest_sha256,
        "bank": bank,
        "positions": bank.positions,
        "sequences": sequences(),
        "top_n": TOP_N,
        "min_covered_mass": min_covered,
        "stores": {
            "baseline_layer1_routed": "kimi-source-expert-bank (BF16, Table)",
            "candidate_layer1_routed": format!(
                "kimi-candidate-bank ({}, Identity)",
                overlay
                    .encoding_of(overlay_layer as u32)
                    .map(|e| e.name())
                    .unwrap_or("unknown")
            ),
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
    let rendered = serde_json::to_vec_pretty(&report)
        .map_err(|e| unreadable("closure report", Path::new(&path), e.to_string()))?;
    std::fs::write(&path, rendered)
        .map_err(|e| unreadable("closure report", Path::new(&path), e.to_string()))?;

    Ok(MeasurementReceipt {
        request: request.clone(),
        gate: evidence.gate.clone(),
        verified,
        bank,
        min_covered_mass: min_covered,
        wall_seconds: run_s,
        report_path: path,
        verdict_passed: verdict.passed(),
        verdict_failures: verdict
            .failures
            .iter()
            .map(|(c, d)| format!("{}: {d}", c.name()))
            .collect(),
    })
}
