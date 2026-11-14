//! `COMPILE ... INTO {MODEL, VINDEX}` — dispatch + shared MEMIT fact
//! collection.

use crate::ast::{CompileConflict, CompileTarget, OutputFormat, UseTarget, VindexRef};
use crate::error::LqlError;
use crate::executor::tuning::canonical_prompt;
use crate::executor::{Backend, Session};

mod atomic;
mod bake;
mod into_model;
mod into_vindex;
mod into_vindex_v3;
mod source_files;
mod unexported;

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_compile(
        &mut self,
        vindex: &VindexRef,
        output: &str,
        format: Option<OutputFormat>,
        target: CompileTarget,
        on_conflict: Option<CompileConflict>,
    ) -> Result<Vec<String>, LqlError> {
        match vindex {
            VindexRef::Current => {
                if matches!(self.backend, Backend::Vindex3 { .. }) {
                    return match target {
                        CompileTarget::Vindex => {
                            let weights = matches!(&self.backend, Backend::Vindex3 { overlay, .. } if overlay.has_weight_ops());
                            if weights {
                                self.exec_compile_weights(output)
                            } else {
                                self.exec_compile_into_vindex_v3(output)
                            }
                        }
                        CompileTarget::Model => Err(crate::executor::vindex3::unsupported(
                            "COMPILE INTO MODEL (container → checkpoint export)",
                        )),
                    };
                }
                let vindex_path = match &self.backend {
                    Backend::Vindex { path, .. } => path.clone(),
                    _ => return Err(LqlError::NoBackend),
                };
                unexported::refuse_unwritten_format(target, format)?;
                match target {
                    CompileTarget::Vindex => self.exec_compile_into_vindex(
                        &vindex_path,
                        output,
                        on_conflict.unwrap_or(CompileConflict::LastWins),
                    ),
                    CompileTarget::Model => self.exec_compile_into_model(&vindex_path, output),
                }
            }
            VindexRef::Path(path) => {
                let mut source_session = Session::new();
                source_session.exec_use(&UseTarget::Vindex(path.clone()))?;
                let source_path = match &source_session.backend {
                    Backend::Vindex { path, .. } => path.clone(),
                    _ => return Err(LqlError::NoBackend),
                };
                unexported::refuse_unwritten_format(target, format)?;
                match target {
                    CompileTarget::Vindex => source_session.exec_compile_into_vindex(
                        &source_path,
                        output,
                        on_conflict.unwrap_or(CompileConflict::LastWins),
                    ),
                    CompileTarget::Model => {
                        source_session.exec_compile_into_model(&source_path, output)
                    }
                }
            }
        }
    }
}

// ── Shared MEMIT fact collection (used by INTO MODEL and INTO VINDEX) ──

/// Result of a MEMIT-fact collection pass.
///
/// `facts` is what the solver should consume; `warnings` is a list of
/// human-readable lines for skipped patch ops that the caller should
/// pass on to the user. Skipping (rather than silently substituting
/// `"relation"` / token-id `0`) prevents COMPILE from baking junk
/// directions into `down_weights.bin`.
pub(crate) struct CollectedMemitFacts {
    pub facts: Vec<larql_inference::MemitFact>,
    pub warnings: Vec<String>,
}

/// Collect MEMIT facts from BOTH applied patches on the PatchedVindex
/// AND the in-memory `patch_recording` of the current session.
/// Live INSERT ops go to `patch_recording` until SAVE PATCH; MEMIT
/// needs to see them for COMPILE to bake the uncommitted edits.
fn collect_memit_facts_with_recording(
    patched: &larql_vindex::PatchedVindex,
    vindex_path: &std::path::Path,
    recording_ops: &[larql_vindex::PatchOp],
) -> Result<CollectedMemitFacts, LqlError> {
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| LqlError::exec("load tokenizer for MEMIT", e))?;

    let mut facts = Vec::new();
    let mut warnings = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let push_fact = |op: &larql_vindex::PatchOp,
                     facts: &mut Vec<larql_inference::MemitFact>,
                     warnings: &mut Vec<String>,
                     seen: &mut std::collections::HashSet<_>|
     -> Result<(), LqlError> {
        if let larql_vindex::PatchOp::Insert {
            layer,
            entity,
            relation,
            target,
            ..
        } = op
        {
            // Skip rather than fabricate: a missing relation makes the
            // canonical MEMIT prompt nonsense, and an unencodable target
            // would bake a direction toward token-id 0 (typically <pad>
            // or <unk>) into `down_weights.bin`.
            let Some(rel_str) = relation.as_deref() else {
                warnings.push(format!(
                    "  skipping MEMIT fact: {entity} → {target} @ L{layer} has no relation"
                ));
                return Ok(());
            };
            let key = (entity.clone(), rel_str.to_string(), target.clone(), *layer);
            if !seen.insert(key) {
                return Ok(());
            }
            let prompt = canonical_prompt(rel_str, entity);
            let encoding = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| LqlError::exec("tokenize MEMIT prompt", e))?;
            let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

            let spaced = format!(" {target}");
            let target_encoding = tokenizer
                .encode(spaced.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize MEMIT target", e))?;
            let Some(target_token_id) = target_encoding.get_ids().first().copied() else {
                warnings.push(format!(
                    "  skipping MEMIT fact: target {target:?} did not tokenise (would have written to token-id 0)"
                ));
                return Ok(());
            };

            facts.push(larql_inference::MemitFact {
                prompt_tokens,
                target_token_id,
                layer: *layer,
                label: format!("{entity} → {target} (L{layer})"),
            });
        }
        Ok(())
    };

    for patch in &patched.patches {
        for op in &patch.operations {
            push_fact(op, &mut facts, &mut warnings, &mut seen)?;
        }
    }
    for op in recording_ops {
        push_fact(op, &mut facts, &mut warnings, &mut seen)?;
    }

    Ok(CollectedMemitFacts { facts, warnings })
}
