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
use larql_vindex::format::vindex3::opplan::exec::cpu::physical::{
    kquant_execution, KQuantExecution,
};
use larql_vindex::format::vindex3::opplan::exec::identity::ExecutionIdentity;
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, SelectedRepresentation};
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::opplan::exec::provenance::ExecutionProvenance;
use larql_vindex::format::vindex3::opplan::exec::realization::{
    RealizationForm, RealizationRecord,
};
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

/// Why this selection is not what was asked for.
///
/// Packs normally cover some of a container's objects, so an object without
/// one is not by itself a fallback, and the container doesn't record which
/// objects were left out on purpose. Which objects have packs is fixed by
/// `index.json`, which a manifest binds by digest. An attempt with no pack of
/// the wanted encoding at all ran none of what it asked for.
pub(in super::super) fn selection_problems(
    wanted: Option<&str>,
    selection: &BTreeMap<String, SelectedRepresentation>,
) -> Vec<String> {
    let mut problems = Vec::new();
    if selection.is_empty() {
        problems.push("the store selected no object".to_string());
    }
    if let Some(wanted) = wanted {
        let present = selection
            .values()
            .any(|selected| selected.stored && selected.encoding == wanted);
        if !present {
            problems.push(format!(
                "encoding {wanted} was asked for and no selected object has a pack of it: \
                 the run would fall back to canonical bytes"
            ));
        }
    }
    problems
}

/// One pin, reduced to what a fallback check reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct PinFacts {
    pub(in super::super) representation: String,
    /// The pin decodes or requantises rather than running the stored form.
    pub(in super::super) converts: bool,
}

impl PinFacts {
    fn of(record: &RealizationRecord) -> Self {
        Self {
            representation: record.representation.clone(),
            converts: matches!(
                record.selection.realization.form,
                RealizationForm::Decode(_) | RealizationForm::Requantise(_)
            ),
        }
    }
}

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
    /// A stored K-quant pack is widened to f32 in this process.
    pub(in super::super) kquant_widened: bool,
    pub(in super::super) runtime_quantised: u64,
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
        let from_wanted: Vec<&PinFacts> = facts
            .pins
            .iter()
            .filter(|pin| pin.representation == wanted)
            .collect();
        if from_wanted.is_empty() {
            problems.push(format!(
                "encoding {wanted} was asked for and no operand was pinned from it"
            ));
        }
        let converted = from_wanted.iter().filter(|pin| pin.converts).count();
        if converted > 0 {
            problems.push(format!(
                "encoding {wanted} was asked for and {converted} operand(s) pinned from it \
                 decode or requantise instead of running it"
            ));
        }
        if facts.kquant_widened {
            problems.push(format!(
                "encoding {wanted} was asked for and this process widens stored K-quant packs \
                 to f32"
            ));
        }
        problems.extend(quantisation_problem(facts.runtime_quantised));
    }
    problems
}

/// A representation the runtime made is not the one that was asked for.
pub(in super::super) fn quantisation_problem(runtime_quantised: u64) -> Option<String> {
    (runtime_quantised > 0).then(|| {
        format!("the runtime quantised {runtime_quantised} tensor(s) rather than serving them")
    })
}

/// What the store selected and the image pinned, with its problems.
#[derive(Debug, Clone)]
pub(in super::super) struct EffectiveExecution {
    pub(in super::super) block: Value,
    pub(in super::super) problems: Vec<String>,
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
    let mut pinned_lowering: Vec<String> = ops
        .realizations()
        .iter()
        .map(|record| record.lowering_provider.to_string())
        .collect();
    pinned_lowering.sort();
    pinned_lowering.dedup();
    let facts = EffectiveFacts {
        provider: provider.clone(),
        pinned_lowering: pinned_lowering.clone(),
        pins: ops.realizations().iter().map(PinFacts::of).collect(),
        overlaid: ops.source_stamp().is_overlaid(),
        process_error: identity.as_ref().err().map(ToString::to_string),
        kquant_widened: kquant_execution() == KQuantExecution::Widen,
        runtime_quantised: store.runtime_quantised(),
    };
    let problems = effective_problems(wanted, &facts);
    let provenance = ExecutionProvenance::of(ops);
    let block = json!({
        "complete": problems.is_empty(),
        "problems": problems,
        "engine": backend.name(),
        "provider": provider,
        "pinned_lowering": pinned_lowering,
        "objects": selection_block(store.selection()),
        "objects_from_pack": store.selection().values().filter(|s| s.stored).count(),
        "realizations": provenance.realizations,
        "process": identity.as_ref().ok().map(|i| &i.process),
        "execution_digest": identity.as_ref().ok().map(ExecutionIdentity::digest),
        "digest_binds_model": false,
        "overlaid": facts.overlaid,
        "runtime_quantised": facts.runtime_quantised,
        "scope": SCOPE,
    });
    EffectiveExecution { block, problems }
}

/// What the run went on to do: the projection plans the executor's own ledger
/// counted, without its timings, and anything the runtime manufactured.
pub(in super::super) fn after_run_execution(
    store: &OperandStore,
    wanted: Option<&str>,
) -> EffectiveAfter {
    let plans: Vec<Value> = cpu::ledger()
        .all()
        .iter()
        .filter(|(_, traffic)| traffic.calls > 0)
        .map(|(plan, traffic)| {
            json!({
                "plan": format!("{plan:?}"),
                "calls": traffic.calls,
                "bytes": traffic.bytes,
            })
        })
        .collect();
    let runtime_quantised = store.runtime_quantised();
    let problems: Vec<String> = wanted
        .and_then(|_| quantisation_problem(runtime_quantised))
        .into_iter()
        .collect();
    EffectiveAfter {
        block: json!({
            "projection_plans_ran": plans,
            "runtime_quantised": runtime_quantised,
            "problems": problems,
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

/// The console's account of the execution judged at `when`.
pub(in super::super) fn execution_line(when: &str, problems: &[String]) -> String {
    if problems.is_empty() {
        format!("execution identity ({when}): complete")
    } else {
        format!("execution identity ({when}): {}", problems.join("; "))
    }
}
