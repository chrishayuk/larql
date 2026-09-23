//! **`teacher-forced-two-arm/plan-v1`**: MEASURE-PLAN-1's procedure
//! (`docs/measure-plan-1.md`).
//!
//! One procedure measures any candidate realization of a VINDEX3 container
//! against a reference realization, for any model the plan executes. It
//! carries the Kimi procedure's five claims:
//!
//! ```text
//! measurement
//! + candidate-scope proof       the changed variable exists
//! + physical-attribution proof  each arm bound what it declared
//! + non-candidate identity      what both arms bound is the same bytes
//! + seal/read consistency       every bound representation, and every
//!                               bank payload, matches its seal
//! ```
//!
//! plus the corpus proof and the null arm. The numbers are computable
//! without any of the proofs and would look the same, which is why a
//! violation is a typed refusal ([`PlanInadmissible`]), never a warning. A
//! machine problem is [`PlanExecutionFailure`] instead: nothing was
//! measured, so it is worth retrying. An inadmissible run is not.
//!
//! The Kimi procedure (`teacher-forced-two-arm/v1`) is untouched.

pub mod arm;
pub mod identity;
pub mod metrics;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::super::token_bank::{container_tokenizer_sha256, TokenBank, TokenBankError};
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::index::Vindex3Index;
use arm::{ArmDescription, TeacherForcedArm};
use metrics::{MetricError, PositionMetrics, Summary};

/// This procedure's name.
pub const PROCEDURE: &str = "teacher-forced-two-arm/plan-v1";

/// Samples the null arm scores twice. The freeze fixes `min(4, sequences)`.
pub const NULL_ARM_SAMPLES: usize = 4;

/// Output files, written into the request's output directory.
pub const REPORT_FILE: &str = "report.json";
pub const POSITIONS_FILE: &str = "positions.jsonl";
pub const RECEIPT_FILE: &str = "receipt.json";

/// UNCERTAINTY-2's sample-size facts for a KL p99, stated in every report
/// so an underpowered reading says so.
pub const P99_BIASED_BELOW_SEQUENCES: usize = 32;
pub const P99_QUARTER_PRECISION_SEQUENCES: usize = 64;

/// What to measure. The arms are built by the caller, which knows the
/// backends; the procedure reads their descriptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanMeasureRequest {
    /// Token bank directory (`teacher-forced-token-bank/v1`).
    pub bank: PathBuf,
    /// Samples to measure, in bank order from `seq-000`.
    pub sequences: usize,
    /// A name for the run, recorded in the receipt.
    pub label: String,
    /// Directory the report, positions and receipt are written to. It must
    /// not already hold a report.
    pub output: PathBuf,
}

/// Nothing was measured. Worth retrying once the cause is fixed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanExecutionFailure {
    /// The request is not runnable as written.
    RequestRefused { detail: String },
    /// A bank, container or index could not be read.
    ArtifactUnreadable { detail: String },
    /// An arm could not score a sample.
    StepRefused { arm: String, detail: String },
    /// The output could not be written.
    OutputUnwritable { detail: String },
}

/// Numbers may exist, and they are NOT evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanInadmissible {
    /// The reference arm, run twice over the same sample, did not produce
    /// bit-identical logits. Any KL it reports could be device noise.
    NullArmNotZero { sample: usize, position: usize },
    /// The arms bound the same bytes through the same arm: the changed
    /// variable does not exist.
    CandidateCompilesNothing,
    /// A representation both arms bound is not the same bytes in the two
    /// containers.
    ProtectedOperandChanged { representation: String },
    /// An arm bound something other than it declared.
    UnexpectedPhysicalRead { arm: String, detail: String },
    /// Bytes read are not the bytes sealed: a bank payload, the bank's own
    /// manifest, or a bound representation.
    SealMismatch {
        what: String,
        expected: String,
        found: String,
    },
    /// The bank was tokenised for another model.
    CorpusNotForThisModel {
        arm: String,
        bank: String,
        model: String,
    },
    /// The two arms scored a different number of positions, or a different
    /// vocabulary, for the same sample.
    PositionCountMismatch { sample: usize, detail: String },
}

/// Why a run produced no evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanRefusal {
    Execution(PlanExecutionFailure),
    Inadmissible(PlanInadmissible),
}

/// What the run checked, recorded rather than asserted.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PlanVerifiedFacts {
    /// Samples the null arm scored twice and found bit-identical.
    pub null_arm_samples: usize,
    /// The representations only the candidate bound.
    pub changed_representations: Vec<String>,
    /// Whether the two arms are different execution arms.
    pub arm_changed: bool,
    /// Arms whose bound objects were checked against their declaration.
    pub attributed_arms: usize,
    /// Representations bound by both arms and checked byte-identical.
    pub protected_representations: usize,
    /// Bound representations whose bytes were re-hashed against their seal.
    pub sealed_representations: usize,
    /// Bank payloads read against their seal.
    pub bank_samples_read: usize,
    /// Arms whose tokenizer was checked against the bank.
    pub tokenizer_checked_arms: usize,
    /// Positions measured.
    pub positions: u64,
}

impl PlanVerifiedFacts {
    /// Whether every proof an admissible reading needs was carried out.
    /// Explicit, not a count.
    pub fn complete(&self, sequences: usize) -> bool {
        self.null_arm_samples > 0
            && self.null_arm_samples == sequences.min(NULL_ARM_SAMPLES)
            && (!self.changed_representations.is_empty() || self.arm_changed)
            && self.attributed_arms == ARMS
            && self.sealed_representations > 0
            && self.bank_samples_read == sequences
            && self.tokenizer_checked_arms == ARMS
            && self.positions > 0
    }
}

/// What a finished run says about itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanReceipt {
    pub procedure: String,
    pub label: String,
    pub bank_id: String,
    pub reference: ArmDescription,
    pub candidate: ArmDescription,
    pub facts: PlanVerifiedFacts,
    pub summary: Summary,
}

/// Run the procedure. On success the report, positions and a receipt are
/// in `request.output`; on refusal a receipt naming the refusal is written
/// there when the directory is writable.
pub fn run(
    request: &PlanMeasureRequest,
    reference: &mut dyn TeacherForcedArm,
    candidate: &mut dyn TeacherForcedArm,
) -> Result<PlanReceipt, PlanRefusal> {
    // An earlier run's record is never overwritten, not even by a receipt.
    for file in [REPORT_FILE, RECEIPT_FILE] {
        if request.output.join(file).exists() {
            return Err(execution(PlanExecutionFailure::RequestRefused {
                detail: format!("{} already holds {file}", request.output.display()),
            }));
        }
    }
    let outcome = measure(request, reference, candidate);
    let receipt = match &outcome {
        Ok(receipt) => serde_json::json!({ "admissible": true, "receipt": receipt }),
        Err(refusal) => serde_json::json!({
            "admissible": false,
            "procedure": PROCEDURE,
            "label": request.label,
            "refusal": refusal,
        }),
    };
    let written = std::fs::create_dir_all(&request.output)
        .and_then(|()| write_json(&request.output.join(RECEIPT_FILE), &receipt));
    match (outcome, written) {
        (Ok(_), Err(e)) => Err(execution(PlanExecutionFailure::OutputUnwritable {
            detail: e.to_string(),
        })),
        (outcome, _) => outcome,
    }
}

/// The two arms every run compares.
const ARMS: usize = 2;

fn execution(failure: PlanExecutionFailure) -> PlanRefusal {
    PlanRefusal::Execution(failure)
}

fn inadmissible(reason: PlanInadmissible) -> PlanRefusal {
    PlanRefusal::Inadmissible(reason)
}

fn unreadable(detail: impl std::fmt::Display) -> PlanRefusal {
    execution(PlanExecutionFailure::ArtifactUnreadable {
        detail: detail.to_string(),
    })
}

/// A bank error as the procedure's refusal: a broken seal is evidence of
/// tampering, anything else is a bank that could not be read.
fn bank_refusal(error: TokenBankError) -> PlanRefusal {
    match error {
        TokenBankError::SealMismatch {
            sample,
            expected,
            found,
        } => inadmissible(PlanInadmissible::SealMismatch {
            what: sample,
            expected,
            found,
        }),
        TokenBankError::BankIdMismatch { recorded, derived } => {
            inadmissible(PlanInadmissible::SealMismatch {
                what: "bank manifest".into(),
                expected: recorded,
                found: derived,
            })
        }
        other => unreadable(other),
    }
}

/// The physical-attribution proof for one arm.
fn attribute(arm: &ArmDescription) -> Result<(), PlanRefusal> {
    let refuse = |detail: String| {
        Err(inadmissible(PlanInadmissible::UnexpectedPhysicalRead {
            arm: arm.arm.clone(),
            detail,
        }))
    };
    if arm.stored_only && arm.runtime_quantised > 0 {
        return refuse(format!(
            "quantised {} tensor(s) at load, under a stored-only request",
            arm.runtime_quantised
        ));
    }
    for (object, bound) in &arm.objects {
        match (&arm.requested_pack, bound.stored) {
            (None, true) => {
                return refuse(format!(
                    "read a compiled {} pack for {object}, and asked for none",
                    bound.encoding
                ))
            }
            (Some(pack), true) if &bound.encoding != pack => {
                return refuse(format!(
                    "read a {} pack for {object}, and asked for {pack}",
                    bound.encoding
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

fn read_index(container: &Path) -> Result<Vindex3Index, PlanRefusal> {
    let path = container.join(INDEX_JSON);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| unreadable(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| unreadable(format!("{}: {e}", path.display())))
}

fn same_container(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Score `ids` on `arm`, as the procedure's error.
fn score(arm: &mut dyn TeacherForcedArm, ids: &[u32]) -> Result<Vec<Vec<f32>>, PlanRefusal> {
    let name = arm.describe().arm.clone();
    arm.score(ids)
        .map_err(|detail| execution(PlanExecutionFailure::StepRefused { arm: name, detail }))
}

/// The first position at which two scorings of one sample differ in any
/// bit, or `None` if they are identical.
fn first_difference(a: &[Vec<f32>], b: &[Vec<f32>]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| {
        x.len() != y.len() || x.iter().zip(y).any(|(p, q)| p.to_bits() != q.to_bits())
    })
}

fn measure(
    request: &PlanMeasureRequest,
    reference: &mut dyn TeacherForcedArm,
    candidate: &mut dyn TeacherForcedArm,
) -> Result<PlanReceipt, PlanRefusal> {
    if request.sequences == 0 {
        return Err(execution(PlanExecutionFailure::RequestRefused {
            detail: "zero sequences measures nothing".into(),
        }));
    }
    let bank = TokenBank::open(&request.bank).map_err(bank_refusal)?;
    if request.sequences > bank.sample_count() {
        return Err(execution(PlanExecutionFailure::RequestRefused {
            detail: format!(
                "{} sequences asked of a bank of {}",
                request.sequences,
                bank.sample_count()
            ),
        }));
    }
    let mut facts = PlanVerifiedFacts::default();
    let reference_desc = reference.describe().clone();
    let candidate_desc = candidate.describe().clone();

    // The corpus belongs to both models.
    for arm in [&reference_desc, &candidate_desc] {
        let model = container_tokenizer_sha256(&arm.container).map_err(unreadable)?;
        bank.check_tokenizer(&model).map_err(|_| {
            inadmissible(PlanInadmissible::CorpusNotForThisModel {
                arm: arm.arm.clone(),
                bank: bank.manifest().tokenizer_sha256.clone(),
                model: model.clone(),
            })
        })?;
        facts.tokenizer_checked_arms += 1;
    }

    // Each arm bound what it declared.
    for arm in [&reference_desc, &candidate_desc] {
        attribute(arm)?;
        facts.attributed_arms += 1;
    }

    // The changed variable exists.
    let reference_bound = identity::bound_representations(&reference_desc);
    let candidate_bound = identity::bound_representations(&candidate_desc);
    let changed = identity::changed_representations(&reference_bound, &candidate_bound);
    facts.arm_changed = reference_desc.arm != candidate_desc.arm;
    if changed.is_empty() && !facts.arm_changed {
        return Err(inadmissible(PlanInadmissible::CandidateCompilesNothing));
    }
    facts.changed_representations = changed.iter().cloned().collect();

    // What both arms bound is the same bytes, and every bound
    // representation matches its seal.
    let reference_index = read_index(&reference_desc.container)?;
    let candidate_index = read_index(&candidate_desc.container)?;
    let reference_digests = identity::recompute_digests(
        &reference_desc.container,
        &reference_index,
        &reference_bound,
    )
    .map_err(unreadable)?;
    let candidate_digests = identity::recompute_digests(
        &candidate_desc.container,
        &candidate_index,
        &candidate_bound,
    )
    .map_err(unreadable)?;
    let shared: BTreeSet<String> = reference_bound
        .intersection(&candidate_bound)
        .cloned()
        .collect();
    if !same_container(&reference_desc.container, &candidate_desc.container) {
        if let Some(representation) =
            identity::protected_changes(&shared, &reference_digests, &candidate_digests)
                .into_iter()
                .next()
        {
            return Err(inadmissible(PlanInadmissible::ProtectedOperandChanged {
                representation,
            }));
        }
    }
    facts.protected_representations = shared.len();
    for (index, digests) in [
        (&reference_index, &reference_digests),
        (&candidate_index, &candidate_digests),
    ] {
        if let Some(broken) = identity::seal_break(index, digests) {
            return Err(inadmissible(PlanInadmissible::SealMismatch {
                what: broken.representation,
                expected: broken.expected,
                found: broken.found,
            }));
        }
    }
    facts.sealed_representations = reference_bound.union(&candidate_bound).count();

    // The null arm: the reference, twice, bit for bit.
    let null_samples = request.sequences.min(NULL_ARM_SAMPLES);
    let mut reference_cache: Vec<Vec<Vec<f32>>> = Vec::with_capacity(null_samples);
    for sample in 0..null_samples {
        let ids = bank.read(sample).map_err(bank_refusal)?;
        let first = score(reference, &ids)?;
        let second = score(reference, &ids)?;
        if let Some(position) = first_difference(&first, &second) {
            return Err(inadmissible(PlanInadmissible::NullArmNotZero {
                sample,
                position,
            }));
        }
        reference_cache.push(first);
    }
    facts.null_arm_samples = null_samples;

    // The measurement.
    let mut positions = Vec::new();
    for sample in 0..request.sequences {
        let ids = bank.read(sample).map_err(bank_refusal)?;
        facts.bank_samples_read += 1;
        let category = bank.manifest().samples[sample].category.clone();
        let reference_logits = match reference_cache.get(sample) {
            Some(cached) => cached.clone(),
            None => score(reference, &ids)?,
        };
        let candidate_logits = score(candidate, &ids)?;
        if reference_logits.len() != ids.len() || candidate_logits.len() != ids.len() {
            return Err(inadmissible(PlanInadmissible::PositionCountMismatch {
                sample,
                detail: format!(
                    "{} ids, reference scored {}, candidate scored {}",
                    ids.len(),
                    reference_logits.len(),
                    candidate_logits.len()
                ),
            }));
        }
        for (position, (r, c)) in reference_logits.iter().zip(&candidate_logits).enumerate() {
            let next = ids.get(position + 1).copied();
            let scored = metrics::position_metrics(r, c, next).map_err(|e| match e {
                MetricError::NonFinite { arm } => execution(PlanExecutionFailure::StepRefused {
                    arm: arm.into(),
                    detail: format!("sample {sample} position {position}: a logit is not finite"),
                }),
                other => inadmissible(PlanInadmissible::PositionCountMismatch {
                    sample,
                    detail: format!("position {position}: {other:?}"),
                }),
            })?;
            positions.push(PositionMetrics {
                sample,
                position,
                category: category.clone(),
                kl: scored.kl,
                top1_agree: scored.top1_agree,
                top5_overlap: scored.top5_overlap,
                delta_nll: scored.delta_nll,
                reference_margin: scored.reference_margin,
                reference_entropy: scored.reference_entropy,
            });
        }
    }
    facts.positions = positions.len() as u64;
    let summary = metrics::summarise(&positions).ok_or_else(|| {
        execution(PlanExecutionFailure::RequestRefused {
            detail: "the requested samples hold no positions".into(),
        })
    })?;

    let receipt = PlanReceipt {
        procedure: PROCEDURE.to_string(),
        label: request.label.clone(),
        bank_id: bank.manifest().bank_id.clone(),
        reference: reference_desc,
        candidate: candidate_desc,
        facts,
        summary,
    };
    write_record(
        request,
        &bank,
        &receipt,
        &positions,
        [
            identity::recorded_digests(&reference_index, &reference_bound),
            identity::recorded_digests(&candidate_index, &candidate_bound),
        ],
    )
    .map_err(|e| {
        execution(PlanExecutionFailure::OutputUnwritable {
            detail: e.to_string(),
        })
    })?;
    Ok(receipt)
}

fn write_json(path: &Path, value: &impl Serialize) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(std::io::Error::other)?;
    std::fs::write(path, bytes)
}

/// The report, then the positions. The receipt is `run`'s.
fn write_record(
    request: &PlanMeasureRequest,
    bank: &TokenBank,
    receipt: &PlanReceipt,
    positions: &[PositionMetrics],
    digests: [BTreeMap<String, String>; ARMS],
) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(&request.output)?;
    let [reference_digests, candidate_digests] = digests;
    let report = serde_json::json!({
        "procedure": PROCEDURE,
        "label": request.label,
        "bank": {
            "dir": request.bank,
            "id": bank.manifest().bank_id,
            "prompts": bank.manifest().prompts,
            "tokenizer_sha256": bank.manifest().tokenizer_sha256,
            "sequences": request.sequences,
            "of": bank.sample_count(),
        },
        "reference": { "arm": receipt.reference, "digests": reference_digests },
        "candidate": { "arm": receipt.candidate, "digests": candidate_digests },
        "facts": receipt.facts,
        "summary": receipt.summary,
        "sample_size": {
            "sequences": request.sequences,
            "p99_biased_low_below": P99_BIASED_BELOW_SEQUENCES,
            "p99_quarter_precision_from": P99_QUARTER_PRECISION_SEQUENCES,
            "adequate_for_p99": request.sequences >= P99_QUARTER_PRECISION_SEQUENCES,
        },
        "units": "nats, full vocabulary",
    });
    write_json(&request.output.join(REPORT_FILE), &report)?;
    let mut lines =
        std::io::BufWriter::new(std::fs::File::create(request.output.join(POSITIONS_FILE))?);
    for position in positions {
        serde_json::to_writer(&mut lines, position).map_err(std::io::Error::other)?;
        lines.write_all(b"\n")?;
    }
    lines.flush()
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
