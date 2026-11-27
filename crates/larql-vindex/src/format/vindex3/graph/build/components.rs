//! Building components and their declared taps from inventories.

use super::super::component::{
    Component, ComponentRole, EncoderGeometry, PerceptionComponent, PerceptionTransform,
    ProjectionGeometry,
};
use super::super::edge::HiddenStateEdge;
use super::super::object::{LogicalObject, ObjectKind};
use super::super::policy::{AttentionSpan, LayerOperator, RecurrenceKind};
use super::super::surface::{
    attach_stack_evidence, gate_evidence, head_from_resolved, surface_from_nested,
    surface_from_resolved,
};
use super::super::{SystemGraph, GRAPH_SCHEMA};
use larql_models::inventory::ArchitectureInventory;
use std::collections::BTreeMap;

#[allow(unused_imports)]
use super::*;

/// Build the system graph over every inventory given.
pub fn build_from_inventories(named: &[(String, ArchitectureInventory)]) -> BuiltGraph {
    let mut components: Vec<Component> = Vec::new();
    let mut objects: BTreeMap<String, LogicalObject> = BTreeMap::new();
    let mut edges: Vec<HiddenStateEdge> = Vec::new();
    let mut unplaced: Vec<UnplacedGroup> = Vec::new();
    let mut unresolved_interfaces: Vec<UnresolvedInterface> = Vec::new();
    let mut nested_by_component: BTreeMap<
        String,
        &larql_models::inventory::components::ComponentTopology,
    > = BTreeMap::new();

    // Pass 1: components. Roles from evidence: a declared tap interface
    // makes a drafter; a nested component with perception geometry makes a
    // perception component; otherwise the artifact carries a primary text
    // model.
    for (artifact, inventory) in named {
        let is_drafter = declared_taps(inventory).is_some();
        let base_id = if is_drafter {
            COMPONENT_ID_DRAFT
        } else {
            COMPONENT_ID_TARGET
        };
        let id = unique_id(base_id, &components);
        components.push(Component {
            id,
            role: if is_drafter {
                ComponentRole::Drafter
            } else {
                ComponentRole::PrimaryText
            },
            source_artifact: artifact.clone(),
            num_layers: inventory.resolved.num_layers,
            hidden_size: inventory.resolved.hidden_size,
            attention: Some(attention_table(inventory)),
            execution: None, // attached in pass 3, once objects are known
            perception: None,
        });
        for nested in &inventory.nested_components {
            let id = unique_id(&nested.name, &components);
            nested_by_component.insert(id.clone(), nested);
            // Species from TENSOR EVIDENCE, never from declared depth.
            // Qwen3.8 and Gemma 4 12B both resolve `num_layers: None` — the
            // first because this build reads `num_hidden_layers` and Qwen
            // declares `depth`, the second because there is no tower to
            // declare. Only the tensors tell those apart: Qwen carries
            // `model.visual.blocks.*`, Gemma 4 12B carries a flat
            // `model.vision_embedder.*` projection.
            let modality = declared_modality(&nested.name);
            let owns_a_stack = inventory.tensors.groups.iter().any(|g| {
                classify_group(&g.prefix) == GroupClass::PerceptionTower
                    && group_modality(&g.prefix) == modality
            });
            let perception = modality.map(|modality| PerceptionComponent {
                modality,
                transform: if owns_a_stack {
                    // Every `None` here means "this build could not read
                    // it", never "it does not apply" — the tower exists.
                    PerceptionTransform::Encoder(EncoderGeometry {
                        depth: nested.num_layers,
                        width: nested.hidden_size,
                        num_heads: nested.num_attention_heads,
                    })
                } else {
                    PerceptionTransform::DirectProjection(ProjectionGeometry {
                        output_width: nested.hidden_size,
                    })
                },
            });
            components.push(Component {
                id,
                role: ComponentRole::Perception,
                source_artifact: artifact.clone(),
                // Legacy fields, kept so containers written now still read
                // on older builds. NOT the source of perception semantics:
                // `perception` is authoritative where present, and the 0s
                // these produce for an encoder-free path are exactly the
                // fabrication it exists to replace.
                num_layers: nested.num_layers.unwrap_or(0),
                hidden_size: nested.hidden_size.unwrap_or(0),
                // From the component's own topology, the same way the
                // text path derives its table from its own resolution.
                // `None` only when the component declares no interleave,
                // or names a span the vocabulary cannot express.
                attention: nested_attention_table(nested),
                execution: None, // attached in pass 3
                perception,
            });
        }
    }

    // Pass 2: objects from tensor groups. For a drafter artifact the
    // projector claim runs FIRST — the fusion tensor is identified by
    // shape evidence, and every group sharing its first path segment joins
    // the projector by structural adjacency, *before* name classification
    // gets a chance to scatter its siblings (a projector's own norm would
    // otherwise name-classify as `final_norm`).
    for (artifact, inventory) in named {
        let text_component = component_for_artifact(&components, artifact);
        let taps = declared_taps(inventory);
        let projector_segment = taps.as_ref().and_then(|taps| {
            find_projector_segment(artifact, inventory, taps, &mut unresolved_interfaces)
        });

        for group in &inventory.tensors.groups {
            if let (Some(segment), Some(consumer)) = (&projector_segment, &text_component) {
                if group.prefix.split('.').next() == Some(segment.as_str()) {
                    merge_binding(
                        &mut objects,
                        consumer,
                        ObjectKind::FeatureProjector,
                        artifact,
                        group,
                        inventory,
                    );
                    continue;
                }
            }
            let class = classify_group(&group.prefix);
            let placement = match class {
                GroupClass::PerceptionTower => {
                    perception_component_for(&components, artifact, group_modality(&group.prefix))
                        .map(|c| (c, ObjectKind::PerceptionTower))
                }
                GroupClass::PerceptionAdapter => {
                    perception_component_for(&components, artifact, group_modality(&group.prefix))
                        .map(|c| (c, ObjectKind::PerceptionAdapter))
                }
                GroupClass::Embedding => text_component.clone().map(|c| (c, ObjectKind::Embedding)),
                GroupClass::PositionEmbedding => text_component
                    .clone()
                    .map(|c| (c, ObjectKind::PositionEmbedding)),
                GroupClass::Head => text_component.clone().map(|c| (c, ObjectKind::OutputHead)),
                GroupClass::Norm => text_component.clone().map(|c| (c, ObjectKind::FinalNorm)),
                GroupClass::Stack => text_component
                    .clone()
                    .map(|c| (c, ObjectKind::DecoderStack)),
                // Ownership follows the DECLARATION, not the name: the
                // head's operands belong to the text component exactly
                // when that component declares the topology they reduce.
                GroupClass::HyperConnectionHead
                    if declares_sinkhorn_hyper_connection(inventory) =>
                {
                    text_component
                        .clone()
                        .map(|c| (c, ObjectKind::HyperConnectionHead))
                }
                // The exit pair follows the same rule as the head's
                // operands, and for the same reason: it is the topology's
                // own reduction, so it belongs to the component exactly
                // when that component declares the topology.
                GroupClass::AttentionResidualExit if declares_attention_residual(inventory) => {
                    text_component
                        .clone()
                        .map(|c| (c, ObjectKind::AttentionResidualExit))
                }
                GroupClass::HyperConnectionHead
                | GroupClass::AttentionResidualExit
                | GroupClass::Unknown => None,
            };
            match placement {
                Some((component, kind)) => {
                    merge_binding(&mut objects, &component, kind, artifact, group, inventory);
                    if kind == ObjectKind::DecoderStack {
                        carve_expert_banks(&mut objects, &component, artifact, group, inventory);
                    }
                }
                None => unplaced.push(UnplacedGroup {
                    artifact: artifact.clone(),
                    prefix: group.prefix.clone(),
                    reason: match class {
                        GroupClass::Unknown => {
                            "no placement rule owns this group — judge it before conversion"
                                .to_string()
                        }
                        // Unguarded on purpose. Pass 1 gives every artifact a
                        // non-perception component, so `text_component` is
                        // always `Some` and a head group lands here for ONE
                        // reason: the placement arm found no declared
                        // topology. Re-asking `declares_sinkhorn_hyper_connection`
                        // here was a second derivation of that fact, and the
                        // CI mutation run on this wave's diff proved it dead —
                        // replacing the guard with `true` changed nothing.
                        GroupClass::HyperConnectionHead => HC_HEAD_UNDECLARED_REASON.to_string(),
                        // Unguarded for the same reason the arm above is:
                        // every artifact gets a non-perception component
                        // in pass 1, so an exit group reaches here for ONE
                        // reason — the placement arm found no declared
                        // period.
                        GroupClass::AttentionResidualExit => {
                            ATTN_RES_EXIT_UNDECLARED_REASON.to_string()
                        }
                        _ => {
                            "classified for a component this artifact does not declare".to_string()
                        }
                    },
                }),
            }
        }

        // The interface edge, once the projector object exists.
        if let (Some(taps), Some(_)) = (&taps, &projector_segment) {
            wire_edge(
                artifact,
                inventory,
                taps,
                &components,
                &mut edges,
                &mut unresolved_interfaces,
            );
        }
    }

    // Pass 3: execution surfaces, now that objects are known (a head
    // surface exists iff the component owns embedding/head objects; a
    // perception FFN's gating is tensor evidence under the tower). A
    // surface that cannot be completed is recorded as missing facts —
    // blocking downstream — never filled with defaults.
    let mut incomplete_surfaces: Vec<IncompleteSurface> = Vec::new();
    for component in &mut components {
        let Some((_, inventory)) = named.iter().find(|(n, _)| *n == component.source_artifact)
        else {
            continue;
        };
        let result = match component.role {
            // A direct projection has no encoder surface to complete, and
            // asking for one produced exactly the nonsense this ontology
            // exists to remove: Gemma 4 12B reported "hidden 0 not
            // divisible by 0 heads" for a path that has neither. It still
            // blocks — this build cannot execute it — but it blocks in its
            // own terms.
            ComponentRole::Perception
                if matches!(
                    component.perception.map(|p| p.transform),
                    Some(PerceptionTransform::DirectProjection(_))
                ) =>
            {
                Err(vec![
                    "direct-projection perception: this build has no execution surface for a \
                     modality transform that owns no internal representation"
                        .to_string(),
                ])
            }
            ComponentRole::Perception => match nested_by_component.get(&component.id) {
                Some(nested) => {
                    let has_gate_tensors = objects.values().any(|object| {
                        object.component == component.id
                            && object.kind == ObjectKind::PerceptionTower
                            && gate_evidence(inventory, object)
                    });
                    surface_from_nested(nested, has_gate_tensors)
                }
                None => Err(vec!["nested component reading".to_string()]),
            },
            ComponentRole::PrimaryText | ComponentRole::Drafter => surface_from_resolved(inventory)
                .and_then(|mut surface| {
                    let owns_head = objects.values().any(|object| {
                        object.component == component.id
                            && matches!(object.kind, ObjectKind::Embedding | ObjectKind::OutputHead)
                    });
                    if owns_head {
                        surface.head = Some(head_from_resolved(inventory)?);
                    }
                    // Facts only the operand estate can state (norm
                    // placement) — evidence outranks family defaults.
                    if let Some(stack) = objects.values().find(|object| {
                        object.component == component.id && object.kind == ObjectKind::DecoderStack
                    }) {
                        attach_stack_evidence(&mut surface, inventory, stack)?;
                    }
                    Ok(surface)
                }),
        };
        match result {
            Ok(surface) => component.execution = Some(surface),
            Err(missing) => incomplete_surfaces.push(IncompleteSurface {
                artifact: component.source_artifact.clone(),
                component: component.id.clone(),
                missing,
            }),
        }
    }

    components.sort_by(|a, b| a.id.cmp(&b.id));
    BuiltGraph {
        graph: SystemGraph {
            schema: GRAPH_SCHEMA,
            components,
            objects: objects.into_values().collect(),
            edges,
        },
        unplaced,
        unresolved_interfaces,
        incomplete_surfaces,
    }
}

/// The artifact's declared tap layers, when it declares any.
pub(super) fn declared_taps(inventory: &ArchitectureInventory) -> Option<Vec<usize>> {
    inventory
        .interfaces
        .iter()
        .find(|i| i.path.ends_with(TARGET_LAYER_IDS_KEY))
        .and_then(|i| {
            i.value.as_array().map(|arr| {
                arr.iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|v| v as usize)
                    .collect()
            })
        })
}

/// Declared drafter block size, read from the config facts.
pub(super) fn declared_block_size(inventory: &ArchitectureInventory) -> Option<usize> {
    inventory
        .config_keys
        .iter()
        .find(|f| f.path == BLOCK_SIZE_KEY || f.path.ends_with(&format!(".{BLOCK_SIZE_KEY}")))
        .and_then(|f| f.value.as_u64())
        .map(|v| v as usize)
}

/// Non-perception component owned by `artifact`.
pub(super) fn component_for_artifact(components: &[Component], artifact: &str) -> Option<String> {
    components
        .iter()
        .find(|c| c.source_artifact == artifact && c.role != ComponentRole::Perception)
        .map(|c| c.id.clone())
}

pub(super) fn unique_id(base: &str, components: &[Component]) -> String {
    if !components.iter().any(|c| c.id == base) {
        return base.to_string();
    }
    let mut n = 2;
    loop {
        let candidate = format!("{base}_{n}");
        if !components.iter().any(|c| c.id == candidate) {
            return candidate;
        }
        n += 1;
    }
}

/// Which recurrence this checkpoint's declared geometry identifies.
///
/// KDA is checked first because the two declarations are disjoint in
/// practice and `linear_attn_config` is the more specific evidence: a
/// checkpoint declaring it has named the KDA block's own geometry, while
/// the `linear_*` keys describe Gated DeltaNet. `None` when neither
/// resolved — a declared recurrence this build cannot name.
pub(super) fn recurrence_kind(inventory: &ArchitectureInventory) -> Option<RecurrenceKind> {
    if inventory.resolved.kda.is_some() {
        return Some(RecurrenceKind::Kda);
    }
    if inventory.resolved.mamba2.is_some() {
        return Some(RecurrenceKind::Mamba2);
    }
    inventory
        .resolved
        .linear_attention
        .is_some()
        .then_some(RecurrenceKind::GatedDelta)
}

/// Operator and span for one canonical declared kind.
///
/// A recurrence gets no span — nothing it retains is indexed by position,
/// so there is no prefix to bound.
///
/// **Which** recurrence comes from `recurrence`, resolved from the
/// checkpoint's declared *geometry*, and never from the declaration's own
/// family. That family is inferred from a key name — Kimi Linear's set is
/// called `kda_layers` — and a key name is not evidence of an operator.
/// Trusting it here would reintroduce exactly the defect the
/// unidentified-recurrence variant exists to prevent, one layer up from
/// where it was fixed.
///
/// `mla` is the same shape of decision one level up: `LayerKind::Full`
/// means "not a recurrence", not "ordinary softmax" — a family that
/// declares Multi-Latent Attention runs it on EVERY non-recurrent layer
/// (MLA compresses the KV cache, orthogonal to which layers are dense vs.
/// routed FFN), so `ModelArchitecture::uses_mla` decides it exactly once
/// per model rather than needing a per-layer flag `layer_types` never
/// carries.
pub(super) fn operator_and_span(
    kind: &larql_models::config::LayerKind,
    recurrence: Option<RecurrenceKind>,
    mla: bool,
    conv_qkv: bool,
) -> (LayerOperator, Option<AttentionSpan>) {
    use larql_models::config::LayerKind;
    match kind {
        LayerKind::Full if mla => (LayerOperator::Mla, Some(AttentionSpan::Full)),
        // Same shape of decision as `mla`, one operator over: on a
        // hybrid that declares the conv-QKV block, every full layer
        // runs it — the lineage has no plain-softmax layer to confuse
        // it with.
        LayerKind::Full if conv_qkv => (LayerOperator::ConvQkvAttention, Some(AttentionSpan::Full)),
        LayerKind::Full => (LayerOperator::Softmax, Some(AttentionSpan::Full)),
        LayerKind::Sliding { .. } => (LayerOperator::Softmax, Some(AttentionSpan::Sliding)),
        LayerKind::Recurrent(_) => (
            match recurrence {
                Some(RecurrenceKind::Kda) => LayerOperator::Kda,
                Some(RecurrenceKind::GatedDelta) => LayerOperator::GatedDelta,
                Some(RecurrenceKind::Mamba2) => LayerOperator::Mamba2,
                None => LayerOperator::Recurrent,
            },
            None,
        ),
        // Handled by the caller, which keeps the layer-blind path so the
        // layer lands in the unexpressed bucket rather than acquiring an
        // operator this build invented for it.
        LayerKind::Unexpressed { .. } => (LayerOperator::Softmax, Some(AttentionSpan::Full)),
    }
}

/// Whether this inventory's resolved execution declares Multi-Latent
/// Attention — `false` for every family before MLA closure existed and
/// for any family that never overrides `ModelArchitecture::uses_mla`, so
/// a container written before this field existed still resolves every
/// `Full` layer to plain softmax exactly as it always did.
pub(super) fn uses_mla(inventory: &ArchitectureInventory) -> bool {
    inventory
        .resolved
        .execution
        .as_ref()
        .is_some_and(|e| e.mla.is_some())
}
