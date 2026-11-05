//! MEM-002 I2b: which computation an attempt selected and pinned.
//!
//! Verified file identity is not execution identity: the same files can run
//! under a different engine, lowering or encoding. So the record says what was
//! asked for and, separately, what the store selected and the prepared image
//! pinned. The effective half comes from the store, the pins and the process,
//! never from a command-line argument: an arm that fell back to canonical bytes
//! reads as canonical bytes.
//!
//! **This records which computation was selected. It does not show two
//! computations are numerically equivalent, or that one is correct.**
//!
//! A campaign attempt whose execution identity is incomplete, or whose request
//! fell back to something else, stops as invalid evidence. Diagnostic execution
//! records the same facts and carries on.

use std::collections::BTreeMap;
use std::path::PathBuf;

use larql_vindex::format::checksums::sha256_file;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::cpu;
use larql_vindex::format::vindex3::opplan::exec::identity::ExecutionIdentity;
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, SelectedRepresentation};
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::opplan::exec::provenance::ExecutionProvenance;
use larql_vindex::format::vindex3::opplan::exec::realization::{
    RealizationForm, RealizationRecord,
};
use larql_vindex::format::vindex3::represent::map::Precision;
use serde_json::{json, Value};

use super::admission::RunMode;
use super::instance_evidence::EvidenceClass;

/// What this record does not claim, written beside the facts.
const SCOPE: &str = "records which computation was selected and pinned; does not establish \
                     numerical equivalence or correctness. Under a loaded plugin, in-process code \
                     is outside these facts, which is why a campaign refuses one.";

/// What the command line asked to execute. Only ever recorded as the request:
/// nothing here is evidence of what ran.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in super::super) struct ExecutionRequest {
    pub(in super::super) component: String,
    /// `--backend`, as the flag spells it.
    pub(in super::super) backend: String,
    /// The encoding `--backend` or `--representation` asked the store for.
    pub(in super::super) wanted_encoding: Option<String>,
    pub(in super::super) plugins: Vec<PathBuf>,
    pub(in super::super) lowering: Option<String>,
    pub(in super::super) representation: Option<String>,
}

impl ExecutionRequest {
    /// The request as a record's block. Plugin digests are of the file as a
    /// path names it now, and a null digest says it could not be read.
    pub(in super::super) fn block(&self) -> Value {
        let plugins: Vec<Value> = self
            .plugins
            .iter()
            .map(|path| {
                json!({
                    "path": path.display().to_string(),
                    "sha256": sha256_file(path).ok(),
                })
            })
            .collect();
        json!({
            "component": self.component,
            "backend": self.backend,
            "wanted_encoding": self.wanted_encoding,
            "plugins": plugins,
            "lowering": self.lowering,
            "representation": self.representation,
        })
    }
}

/// Every object's selected encoding, from the store.
pub(in super::super) fn selection_block(
    selection: &BTreeMap<String, SelectedRepresentation>,
) -> Value {
    selection
        .iter()
        .map(|(object, selected)| {
            json!({
                "object": object,
                "encoding": selected.encoding,
                "from_pack": selected.stored,
            })
        })
        .collect()
}

/// Why this selection is not what was asked for: seen before anything is bound.
/// Whether the packs that exist cover what they should is judged on the pins.
pub(in super::super) fn selection_problems(
    wanted: Option<&str>,
    selection: &BTreeMap<String, SelectedRepresentation>,
    absent: &[String],
) -> Vec<String> {
    let mut problems = Vec::new();
    if selection.is_empty() {
        problems.push("the store selected no object".to_string());
    }
    if let Some(wanted) = wanted {
        let packs: Vec<&String> = selection
            .iter()
            .filter(|(_, selected)| selected.stored && selected.encoding == wanted)
            .map(|(object, _)| object)
            .collect();
        if packs.is_empty() {
            problems.push(format!(
                "encoding {wanted} was asked for and no selected object has a pack of it: \
                 the run would fall back to canonical bytes"
            ));
        }
        let missing: Vec<&&String> = packs.iter().filter(|o| absent.contains(o)).collect();
        if !missing.is_empty() {
            problems.push(format!(
                "encoding {wanted} was asked for and the pack of {missing:?} is not on disk"
            ));
        }
    }
    problems
}

/// How a pin uses its stored bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum PinUse {
    /// Runs the stored form in place.
    InPlace,
    /// Decodes or requantises rather than running the stored form.
    Converts,
    /// Anything else, such as a gathered embedding table: it touches the stored
    /// bytes and isn't evidence the wanted kernel ran.
    Other,
}

/// One pin, reduced to what a fallback check reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct PinFacts {
    pub(in super::super) object: String,
    pub(in super::super) representation: String,
    pub(in super::super) pin_use: PinUse,
    /// The container's precision program compiles this tensor to the wanted
    /// encoding.
    pub(in super::super) program_compiles: bool,
    /// This pin's object was served from a compiled pack.
    pub(in super::super) object_from_pack: bool,
}

impl PinFacts {
    fn of(record: &RealizationRecord, store: &OperandStore, wanted: Option<&str>) -> Self {
        let operand = &record.planned.operand;
        let role = store.role_of(&operand.object, &operand.tensor, &operand.shape);
        let program_compiles = match (store.program(), wanted) {
            (Some(program), Some(wanted)) => {
                program.resolve(role, &operand.tensor) == Precision::Compiled(wanted)
            }
            _ => false,
        };
        Self {
            object: operand.object.clone(),
            representation: record.representation.clone(),
            pin_use: match record.selection.realization.form {
                RealizationForm::Direct(_) | RealizationForm::MappedStored { .. } => {
                    PinUse::InPlace
                }
                RealizationForm::Decode(_) | RealizationForm::Requantise(_) => PinUse::Converts,
                _ => PinUse::Other,
            },
            program_compiles,
            object_from_pack: store
                .selection()
                .get(&operand.object)
                .is_some_and(|s| s.stored),
        }
    }
}

/// Stored forms a precision program can leave alone: a bank already held in a
/// packed encoding isn't compiled again, whatever roles the program names. Any
/// other label beside a compiling program is treated as a missing pack.
const ALREADY_PACKED: &[&str] = &["MXFP4"];

/// The facts a judgement on the effective execution reads, kept separate from
/// where they come from so each refusal can be pinned on its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct EffectiveFacts {
    /// The identity of the provider that prepared the image.
    pub(in super::super) provider: String,
    /// The distinct lowering identities the pins name.
    pub(in super::super) pinned_lowering: Vec<String>,
    pub(in super::super) pins: Vec<PinFacts>,
    pub(in super::super) overlaid: bool,
    /// Why the process's arithmetic could not be read, if it could not.
    pub(in super::super) process_error: Option<String>,
    /// The encoding the container's precision program compiles to, if it
    /// declares one.
    pub(in super::super) program_encoding: Option<String>,
}

/// Why this execution's identity is incomplete, or is not what was asked for.
pub(in super::super) fn effective_problems(
    wanted: Option<&str>,
    facts: &EffectiveFacts,
) -> Vec<String> {
    let mut problems = Vec::new();
    if facts.pins.is_empty() {
        problems.push("the prepared image pinned no operand".to_string());
    }
    if facts.pinned_lowering != std::slice::from_ref(&facts.provider) {
        problems.push(format!(
            "the pins name lowering {:?} and provider {} prepared the image",
            facts.pinned_lowering, facts.provider
        ));
    }
    if facts.overlaid {
        problems.push(
            "the operand source is overlaid, which has no identity outside this process"
                .to_string(),
        );
    }
    if let Some(error) = &facts.process_error {
        problems.push(format!("the process arithmetic is unreadable: {error}"));
    }
    if let Some(wanted) = wanted {
        problems.extend(wanted_problems(wanted, facts));
    }
    problems
}

/// Whether the wanted encoding is what executes: in place, on the objects the
/// container's own program says were compiled to it.
fn wanted_problems(wanted: &str, facts: &EffectiveFacts) -> Vec<String> {
    let mut problems = Vec::new();
    let from_wanted = || facts.pins.iter().filter(|pin| pin.representation == wanted);
    if !from_wanted().any(|pin| pin.pin_use == PinUse::InPlace) {
        problems.push(format!(
            "encoding {wanted} was asked for and no operand runs it in place: \
             the wanted kernel would not run"
        ));
    }
    let converted = from_wanted()
        .filter(|pin| pin.pin_use == PinUse::Converts)
        .count();
    if converted > 0 {
        problems.push(format!(
            "encoding {wanted} was asked for and {converted} operand(s) pinned from it \
             decode or requantise instead of running it"
        ));
    }
    // A program for another encoding says nothing about this one, so it can't
    // show coverage any more than no program can.
    if facts.program_encoding.as_deref() != Some(wanted) {
        problems.push(format!(
            "encoding {wanted} was asked for and the container declares no precision program \
             for it, so an object without a pack can't be told from one left out on purpose"
        ));
    }
    let mut uncompiled: Vec<&str> = facts
        .pins
        .iter()
        .filter(|pin| {
            pin.program_compiles
                && !pin.object_from_pack
                && !ALREADY_PACKED.contains(&pin.representation.as_str())
        })
        .map(|pin| pin.object.as_str())
        .collect();
    uncompiled.sort_unstable();
    uncompiled.dedup();
    if !uncompiled.is_empty() {
        problems.push(format!(
            "encoding {wanted} was asked for and the container's precision program compiles \
             {uncompiled:?}, which has no pack: it would fall back to canonical bytes"
        ));
    }
    problems
}

/// What the store selected and the image pinned, with its problems.
#[derive(Debug, Clone)]
pub(in super::super) struct EffectiveExecution {
    pub(in super::super) block: Value,
    pub(in super::super) problems: Vec<String>,
    /// The kernels the pins say run the wanted encoding in place, as the
    /// executor's ledger names them.
    pub(in super::super) wanted_plans: Vec<String>,
}

/// Read the effective execution off the backend, the image and the process.
/// It reads no payload byte.
pub(in super::super) fn effective_execution<B: PlanBackend>(
    backend: &B,
    ops: &PreparedOperands,
    store: &OperandStore,
    wanted: Option<&str>,
) -> EffectiveExecution {
    let provider = backend.identity().to_string();
    let identity = ExecutionIdentity::of(ops, None);
    let records = ops.realizations();
    let mut pinned_lowering: Vec<String> = records
        .iter()
        .map(|record| record.lowering_provider.to_string())
        .collect();
    pinned_lowering.sort();
    pinned_lowering.dedup();
    let facts = EffectiveFacts {
        provider: provider.clone(),
        pinned_lowering: pinned_lowering.clone(),
        pins: records
            .iter()
            .map(|record| PinFacts::of(record, store, wanted))
            .collect(),
        overlaid: ops.source_stamp().is_overlaid(),
        process_error: identity.as_ref().err().map(ToString::to_string),
        program_encoding: store.program().map(|program| program.encoding.clone()),
    };
    let problems = effective_problems(wanted, &facts);
    let pinned: Vec<(&str, RealizationForm)> = records
        .iter()
        .map(|record| {
            (
                record.representation.as_str(),
                record.selection.realization.form,
            )
        })
        .collect();
    let wanted_plans = wanted_plans(&pinned, wanted);
    let provenance = ExecutionProvenance::of(ops);
    let block = json!({
        "complete": problems.is_empty(),
        "problems": problems,
        "engine": backend.name(),
        "provider": provider,
        "pinned_lowering": pinned_lowering,
        "objects": selection_block(store.selection()),
        "objects_from_pack": store.selection().values().filter(|s| s.stored).count(),
        "precision_program": store.program().map(|program| program.describe()),
        "realizations": provenance.realizations,
        "process": identity.as_ref().ok().map(|i| &i.process),
        "execution_digest": identity.as_ref().ok().map(ExecutionIdentity::digest),
        "digest_binds_model": false,
        "overlaid": facts.overlaid,
        "scope": SCOPE,
    });
    EffectiveExecution {
        block,
        problems,
        wanted_plans,
    }
}

/// The kernels the pins say run the wanted encoding in place. A form with no
/// named kernel adds none, and neither does a pin from any other encoding: a
/// plan every run counts couldn't show the wanted kernel ran.
pub(in super::super) fn wanted_plans(
    pinned: &[(&str, RealizationForm)],
    wanted: Option<&str>,
) -> Vec<String> {
    let mut plans: Vec<String> = pinned
        .iter()
        .filter(|(representation, _)| Some(*representation) == wanted)
        .filter_map(|(_, form)| match form {
            RealizationForm::Direct(plan) => Some(format!("{plan:?}")),
            _ => None,
        })
        .collect();
    plans.sort();
    plans.dedup();
    plans
}

/// One line of the executor's ledger, without its timings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct PlanRan {
    pub(in super::super) plan: String,
    pub(in super::super) calls: u64,
    pub(in super::super) bytes: u64,
}

/// What the executor's ledger counted.
fn plans_ran() -> Vec<PlanRan> {
    cpu::ledger()
        .all()
        .iter()
        .filter(|(_, traffic)| traffic.calls > 0)
        .map(|(plan, traffic)| PlanRan {
            plan: format!("{plan:?}"),
            calls: traffic.calls,
            bytes: traffic.bytes,
        })
        .collect()
}

/// A kernel pinned for the wanted encoding that the ledger never counted did not
/// run. A form with no named kernel, such as a mapped bank, can't be checked here.
pub(in super::super) fn ran_problems(wanted_plans: &[String], ran: &[PlanRan]) -> Vec<String> {
    if wanted_plans.is_empty() || ran.iter().any(|r| wanted_plans.contains(&r.plan)) {
        return Vec::new();
    }
    vec![format!(
        "the pins name {wanted_plans:?} for the wanted encoding and the executor's ledger counted \
         none of them"
    )]
}

/// What the run went on to do: the projection plans the executor's own ledger
/// counted, without its timings, judged with what was found at admission so the
/// last record doesn't read cleaner than the first.
pub(in super::super) fn after_run_execution(admission: &EffectiveExecution) -> EffectiveAfter {
    let ran = plans_ran();
    let problems = ran_problems(&admission.wanted_plans, &ran);
    let plans: Vec<Value> = ran
        .iter()
        .map(|r| json!({ "plan": r.plan, "calls": r.calls, "bytes": r.bytes }))
        .collect();
    EffectiveAfter {
        block: json!({
            "complete": admission.problems.is_empty() && problems.is_empty(),
            "problems": problems,
            "admission_problems": admission.problems,
            "wanted_plans": admission.wanted_plans,
            "projection_plans_ran": plans,
        }),
        problems,
    }
}

/// [`after_run_execution`]'s result.
#[derive(Debug, Clone)]
pub(in super::super) struct EffectiveAfter {
    pub(in super::super) block: Value,
    pub(in super::super) problems: Vec<String>,
}

/// The class evidence carries once its execution has been judged: execution
/// problems invalidate a campaign attempt and leave a diagnostic run's class
/// alone, because diagnostic execution was never evidence of this.
pub(in super::super) fn class_with_execution(
    mode: RunMode,
    class: EvidenceClass,
    problems: &[String],
) -> EvidenceClass {
    if mode == RunMode::Campaign && !problems.is_empty() {
        EvidenceClass::Invalid
    } else {
        class
    }
}

/// The console's account of the execution judged at `when`, from every problem
/// known by then, so it can't read cleaner than the record.
pub(in super::super) fn execution_line(when: &str, problems: &[String]) -> String {
    if problems.is_empty() {
        format!("execution identity ({when}): complete")
    } else {
        format!("execution identity ({when}): {}", problems.join("; "))
    }
}

/// The console's account of the selection before anything is bound: nothing has
/// been pinned yet, so it can't be called complete.
pub(in super::super) fn selection_line(problems: &[String]) -> String {
    if problems.is_empty() {
        "execution selection (before binding): no problem found; nothing pinned yet".to_string()
    } else {
        format!(
            "execution selection (before binding): {}",
            problems.join("; ")
        )
    }
}
