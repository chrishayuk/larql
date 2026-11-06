//! The encoder a first run uses: the `larql vindex3 encode` pipeline,
//! gated on the plan first so a refusal is itemised and costs headers.
//!
//! Nothing is forked. The artifact is resolved by
//! `larql_vindex::format::vindex3::artifact::resolve` and encoded by
//! `artifact::encode_from_specs` — the one ingest both `larql vindex3
//! encode` and `vindex encode` call. The plan computed here is the same
//! `plan_system` that ingest runs as its own gate; it is computed first
//! only so the refusal can name its findings instead of counting them.

use std::io::Write;
use std::path::Path;

use larql_vindex::format::vindex3::artifact;
use larql_vindex::format::vindex3::plan::capability::Capability;
use larql_vindex::format::vindex3::plan::plan_system;

use super::refusal::{self, Verdict};
use super::resolve::CheckpointEncoder;

type BoxErr = Box<dyn std::error::Error>;

/// What `larql run` asks of a checkpoint: text in, text out.
pub const RUN_CAPABILITY: Capability = Capability::TextGeneration;

/// Plans, gates and encodes through the shared ingest.
#[derive(Debug, Clone, Copy)]
pub struct PlanGatedEncoder {
    /// The admission scope, as `larql vindex3 encode --capability`
    /// takes it. `None` is whole-model completeness.
    pub scope: Option<Capability>,
}

impl PlanGatedEncoder {
    /// The gate `larql run` uses: the text-generation closure, the bar
    /// for "we can run text generation on it" (the container written is
    /// identical to a whole-model encode; only the gate differs).
    pub fn for_run() -> Self {
        Self {
            scope: Some(RUN_CAPABILITY),
        }
    }

    /// The gate's name in prose: the capability's own snake_case
    /// spelling with spaces, or the whole model.
    pub fn scope_label(&self) -> String {
        match self.scope {
            Some(capability) => format!("{capability:?}")
                .chars()
                .enumerate()
                .flat_map(|(i, c)| {
                    let space = (i > 0 && c.is_ascii_uppercase()).then_some(' ');
                    space
                        .into_iter()
                        .chain(std::iter::once(c.to_ascii_lowercase()))
                })
                .collect(),
            None => WHOLE_MODEL_LABEL.to_string(),
        }
    }
}

/// What a `None` scope is called.
pub const WHOLE_MODEL_LABEL: &str = "the whole model";

/// The line reporting what header staging read.
pub fn staging_line(staged_bytes: u64) -> String {
    format!(
        "  staged {} of headers and metadata; planning…",
        artifact::size(staged_bytes)
    )
}

/// The line reporting what one repo artifact pulled over the wire.
pub fn transfer_line(transfer: &artifact::RemoteTransfer) -> String {
    format!(
        "  fetched {} of {} declared across {} tensor(s)",
        artifact::size(transfer.fetched),
        artifact::size(transfer.declared),
        transfer.tensors,
    )
}

impl CheckpointEncoder for PlanGatedEncoder {
    fn encode(&self, spec: &str, out: &Path, status: &mut dyn Write) -> Result<(), BoxErr> {
        let resolved = artifact::resolve(Path::new(spec))?;
        if let Some(report) = resolved.staging() {
            writeln!(status, "{}", staging_line(report.staged_bytes()))?;
        }

        let named = vec![(resolved.name.clone(), resolved.inventory.clone())];
        let plan = plan_system(&named);
        let (verdict, blockers) = refusal::judge(&plan, self.scope);
        let scope = self.scope_label();
        if verdict != Verdict::Admissible {
            return Err(refusal::render(spec, &scope, verdict, &blockers).into());
        }
        writeln!(
            status,
            "  plan admissible for {scope}; encoding to VINDEX3, fetching tensor payloads…"
        )?;

        let outcome = artifact::encode_from_specs(vec![resolved], out, self.scope)?;
        for transfer in &outcome.transfers {
            writeln!(status, "{}", transfer_line(transfer))?;
        }
        writeln!(
            status,
            "  encoded {} representation(s), {} payload",
            outcome.representations,
            artifact::size(outcome.total_payload_bytes),
        )?;
        Ok(())
    }
}
