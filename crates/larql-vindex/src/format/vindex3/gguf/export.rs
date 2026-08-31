//! The assembled qwen35 export pipeline, as one callable boundary.
//!
//! Everything the gates proved, in execution order:
//!
//! ```text
//! operation plan   roles per tensor — the plan's assignment, never a
//!                  tensor-name match (§5:00's distinction, honoured
//!                  here at last: the walk always said roles arrive as
//!                  input, and this is the input arriving)
//! precision map    which representation of each object leaves — the
//!                  container's own programme, not a caller preference
//! walk             coverage, identity, geometry, transforms
//! metadata         graph facts → target vocabulary
//! vocab            the capability snapshot → tokenizer.ggml.*
//! emit             plans executed, nothing decided
//! verify           the file re-read through the independent reader
//! ```
//!
//! The function refuses rather than narrows: an operator this target
//! has no names for, a routed FFN, a walk that is not ready, a verify
//! mismatch — each is an error naming the fact, never a smaller file.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use larql_models::config::PositionPolicy;

use super::emit::{emit_gguf, metadata_to_gguf, verify_emitted, EmitReport, VerifyReport};
use super::metadata::qwen35_metadata;
use super::plan::Qwen35Lowering;
use super::vocab::qwen35_vocab;
use super::walk::{inventory_from_container, walk_primary_text, Ledger};
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::graph::LayerOperator;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{
    plan_component_ops, ComponentOpPlan, LayerAttention, LayerFfn,
};
use crate::VindexError;

/// What one export did — every count observed, none predicted.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenExport {
    pub out: PathBuf,
    pub bytes: u64,
    pub ledger: Ledger,
    pub emit: EmitReport,
    pub verify: VerifyReport,
    pub vocab_tokens: usize,
    pub vocab_padded: usize,
    pub vocab_merges: usize,
    pub selected_encoding: String,
}

fn err(msg: impl Into<String>) -> VindexError {
    VindexError::Parse(msg.into())
}

/// Role assignments from the operation plan: `(object, tensor)` →
/// `(role, layer)`. Exhaustive over what qwen35 can express, and a
/// refusal for what it cannot.
fn roles_from_plan(
    plan: &ComponentOpPlan,
) -> Result<BTreeMap<(String, String), (String, Option<usize>)>, VindexError> {
    let mut roles = BTreeMap::new();
    let mut put =
        |op: &crate::format::vindex3::opplan::OperandRef, role: &str, layer: Option<usize>| {
            roles.insert(
                (op.object.clone(), op.tensor.clone()),
                (role.to_string(), layer),
            );
        };
    if let Some(e) = &plan.embedding {
        put(&e.table, "embedding", None);
    }
    if let Some(n) = &plan.final_norm {
        put(&n.weight, "final norm", None);
    }
    if let Some(o) = &plan.output {
        put(&o.projection, "output head", None);
    }
    for layer in &plan.layers {
        let l = Some(layer.layer);
        put(&layer.pre_attention_norm.weight, "input layer norm", l);
        // qwen35 has exactly two trunk norms per layer. The plan may
        // spell the second as post-attention or pre-FFN; both land on
        // the same target name, and a layer carrying both has a third
        // norm this target cannot bind.
        match (&layer.post_attention_norm, &layer.pre_ffn_norm) {
            (Some(n), None) | (None, Some(n)) => put(&n.weight, "post-attention layer norm", l),
            (Some(_), Some(_)) => {
                return Err(err(format!(
                    "export: layer {} carries both a post-attention and a pre-FFN norm — \
                     qwen35 binds two trunk norms per layer, not three",
                    layer.layer
                )))
            }
            (None, None) => {
                return Err(err(format!(
                    "export: layer {} carries no second trunk norm — qwen35 expects one",
                    layer.layer
                )))
            }
        }
        if layer.post_ffn_norm.is_some() {
            return Err(err(format!(
                "export: layer {} carries a post-FFN norm, which qwen35 has no name for",
                layer.layer
            )));
        }
        match &layer.attention {
            LayerAttention::Softmax(a) => {
                put(&a.q, "query", l);
                put(&a.k, "key", l);
                put(&a.v, "value", l);
                put(&a.o, "output", l);
                if let Some(qk) = &a.qk_norm {
                    put(&qk.q, "attention q norm", l);
                    put(&qk.k, "attention k norm", l);
                }
                if let Some(gate) = &a.output_gate {
                    // qwen35 fuses the gate into the query projection —
                    // the same physical tensor. A gate with its own
                    // tensor is an operand this target cannot bind.
                    if gate.projection.tensor != a.q.tensor || gate.projection.object != a.q.object
                    {
                        return Err(err(format!(
                            "export: layer {} has a standalone attention output gate \
                             (`{}`) — qwen35 binds the gate only as the second half of \
                             a fused query projection",
                            layer.layer, gate.projection.tensor
                        )));
                    }
                }
            }
            LayerAttention::GatedDelta(g) => {
                put(&g.in_proj_qkv, "fused recurrent q|k|v", l);
                put(&g.in_proj_a, "decay projection", l);
                put(&g.in_proj_b, "write-strength projection", l);
                put(&g.in_proj_z, "output-gate projection", l);
                put(&g.conv1d, "causal conv over q|k|v", l);
                put(&g.a_log, "log decay", l);
                put(&g.dt_bias, "timestep bias", l);
                put(&g.norm, "gated norm", l);
                put(&g.out_proj, "output projection", l);
            }
            other => {
                return Err(err(format!(
                    "export: layer {} binds {} — qwen35 names operands for gated \
                     softmax attention and Gated DeltaNet only",
                    layer.layer,
                    match other {
                        LayerAttention::Kda(_) => "KDA",
                        LayerAttention::Mla(_) => "MLA",
                        LayerAttention::Mamba2(_) => "Mamba2",
                        LayerAttention::ConvQkv(_) => "conv-QKV attention",
                        _ => "an unnamed operator",
                    }
                )))
            }
        }
        match &layer.ffn {
            Some(LayerFfn::Dense(f)) => {
                if let Some(gate) = &f.gate {
                    put(gate, "ffn gate", l);
                }
                put(&f.up, "ffn up", l);
                put(&f.down, "ffn down", l);
            }
            Some(_) => {
                return Err(err(format!(
                    "export: layer {} has a routed FFN — the qwen35 target binds dense \
                     FFN operands only",
                    layer.layer
                )))
            }
            None => {
                return Err(err(format!(
                    "export: layer {} has no FFN — qwen35 expects one per layer",
                    layer.layer
                )))
            }
        }
    }
    Ok(roles)
}

/// Export a container's primary-text component as a qwen35 GGUF, and
/// verify the file through the independent reader before returning.
pub fn export_qwen35(root: &Path, out: &Path) -> Result<QwenExport, VindexError> {
    let inspection = inspect_container(root, false)?;
    let component = inspection
        .graph
        .primary_text_component()
        .map_err(|e| err(format!("export: {e:?}")))?;
    let surface = component.execution.as_ref().ok_or_else(|| {
        err(format!(
            "export: component `{}` carries no execution surface",
            component.id
        ))
    })?;

    // Roles: the operation plan's assignment, or its named defects.
    let outcome = plan_component_ops(&inspection, root, &component.id)?;
    let Some(op_plan) = outcome.plan.as_ref().filter(|_| outcome.closed()) else {
        return Err(err(format!(
            "export: the operation plan does not close: {:?}",
            outcome.defects
        )));
    };
    let roles = roles_from_plan(op_plan)?;

    // Selection: the container's own precision programme. A represented
    // container says which encoding executes; a canonical one has only
    // its canonical bytes.
    let preferred: Option<String> = inspection
        .index
        .precision_map
        .as_ref()
        .map(|m| m.encoding.to_uppercase());
    let selected_encoding = preferred.clone().unwrap_or_else(|| "canonical".into());
    let select = move |_object: &str, ids: &[&str]| -> Option<String> {
        if let Some(enc) = &preferred {
            if let Some(id) = ids.iter().find(|id| id.ends_with(&format!("@{enc}"))) {
                return Some(id.to_string());
            }
        }
        ids.first().map(|s| s.to_string())
    };

    // The component's own objects, no name prefixes.
    let objects: std::collections::BTreeSet<String> = inspection
        .graph
        .objects
        .iter()
        .filter(|o| o.component == component.id)
        .map(|o| o.id.clone())
        .collect();
    let included = {
        let objects = objects.clone();
        move |object: &str| objects.contains(object)
    };

    let (sources, excluded) = inventory_from_container(
        root,
        &inspection.index,
        &|object, name| roles.get(&(object.to_string(), name.to_string())).cloned(),
        &included,
        &|object, ids| select(object, ids),
    )?;

    let lowering = Qwen35Lowering::from_surface(surface, component.hidden_size)
        .map_err(|e| err(format!("export: {e}")))?;
    let head = surface
        .head
        .as_ref()
        .ok_or_else(|| err("export: the surface carries no head — vocabulary unknown"))?;
    let required: Vec<(&str, &'static str)> = {
        let mut r = vec![
            ("token_embd.weight", "every model needs an embedding"),
            ("output_norm.weight", "final norm before the head"),
        ];
        if !head.head_reuses_embedding {
            r.push((
                "output.weight",
                "the graph says head_reuses_embedding = false",
            ));
        }
        r
    };
    let (plans, ledger) = walk_primary_text(&sources, excluded, &required, &lowering);
    if !ledger.ready() {
        return Err(err(format!(
            "export: the walk is not ready — {} of {} accounted, {} geometry-reconciled, \
             errors: {:?}",
            ledger.accounted,
            ledger.source_total,
            ledger.geometry_reconciled,
            ledger.errors.iter().take(5).collect::<Vec<_>>()
        )));
    }

    // Metadata inputs, each from the graph's own declarations.
    let policies = component
        .attention
        .as_ref()
        .ok_or_else(|| err("export: no per-layer attention policy table"))?;
    let attending: Vec<usize> = policies
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.operator, LayerOperator::Softmax))
        .map(|(i, _)| i)
        .collect();
    let position = component
        .position_policy(0)
        .ok_or_else(|| err("export: no layer-0 position policy"))?;
    let PositionPolicy::MRope {
        theta,
        rotary_fraction,
        section,
        ..
    } = position
    else {
        return Err(err(format!(
            "export: qwen35 metadata expects MRoPE; the graph declares {position:?}"
        )));
    };
    let sections: Vec<u32> = section.iter().map(|s| *s as u32).collect();
    let table = qwen35_metadata(
        surface,
        component.num_layers,
        component.hidden_size,
        theta,
        &sections,
        rotary_fraction,
        &attending,
        ledger.generated_scale_tensors > 0,
    )
    .map_err(|e| err(format!("export: metadata: {e:?}")))?;

    let vocab = qwen35_vocab(root, head.vocab_size)?;
    let (vocab_tokens, vocab_padded, vocab_merges) = (vocab.tokens, vocab.padded, vocab.merges);
    let mut metadata = metadata_to_gguf(&table);
    metadata.extend(vocab.entries);

    // Payload spans for the selected representation of each object.
    let mut spans: BTreeMap<String, (PathBuf, u64, u64)> = BTreeMap::new();
    let mut by_object: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for id in inspection.index.representations.keys() {
        let object = id.split('@').next().unwrap_or(id).to_string();
        by_object.entry(object).or_default().push(id.clone());
    }
    for (object, ids) in &by_object {
        if !objects.contains(object) {
            continue;
        }
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let chosen = select(object, &refs)
            .ok_or_else(|| err(format!("export: no representation selected for `{object}`")))?;
        let entry = &inspection.index.representations[&chosen];
        let path = root.join(&entry.segment);
        let (header, data_start) = read_segment_header(&path)?;
        for t in &header.tensors {
            spans.insert(
                format!("{object}/{}", t.name),
                (path.clone(), data_start + t.offset, t.len),
            );
        }
    }
    let mut open = |source: &str| -> std::io::Result<Box<dyn Read>> {
        let (path, at, len) = spans
            .get(source)
            .ok_or_else(|| std::io::Error::other(format!("no source span for `{source}`")))?;
        let mut f = std::fs::File::open(path)?;
        f.seek(SeekFrom::Start(*at))?;
        Ok(Box::new(f.take(*len)))
    };

    let emit = emit_gguf(&metadata, &plans, &mut open, out)?;
    let required_names: Vec<&str> = required.iter().map(|(n, _)| *n).collect();
    let verify = verify_emitted(out, &metadata, &plans, &required_names).map_err(|wrong| {
        err(format!(
            "export: the emitted file does not match its plan — {} mismatches, first: {}",
            wrong.len(),
            wrong.first().cloned().unwrap_or_default()
        ))
    })?;
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);

    Ok(QwenExport {
        out: out.to_path_buf(),
        bytes,
        ledger,
        emit,
        verify,
        vocab_tokens,
        vocab_padded,
        vocab_merges,
        selected_encoding,
    })
}

#[cfg(test)]
mod tests {
    use super::export_qwen35;

    /// The assembled pipeline on a real encoded container: the
    /// operation plan's roles close, the walk passes coverage and
    /// geometry, and the refusal that stops the fixture is the honest
    /// one — its graph declares plain RoPE where qwen35's metadata
    /// needs MRoPE. Everything before that gate worked, which is what
    /// this pins: roles arrive from the plan, not from tensor names,
    /// on a container the test encoded itself.
    #[test]
    fn export_reaches_the_metadata_gate_and_refuses_by_name_on_a_non_mrope_graph() {
        use crate::format::vindex3::fixtures::{encode_fixture_container, hybrid_lllf_f32_model};
        let checkpoint = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(
            hybrid_lllf_f32_model,
            checkpoint.path(),
            container.path(),
            "export-gate",
        );
        let out = tempfile::tempdir().unwrap();
        let err = export_qwen35(container.path(), &out.path().join("x.gguf"))
            .expect_err("the fixture declares plain RoPE");
        let msg = err.to_string();
        assert!(msg.contains("MRoPE"), "{msg}");
        assert!(
            msg.contains("Rope"),
            "the refusal names what the graph declares instead: {msg}"
        );
        // And nothing was written: a refused export leaves no file to
        // mistake for a finished one.
        assert!(!out.path().join("x.gguf").exists());
    }
}
