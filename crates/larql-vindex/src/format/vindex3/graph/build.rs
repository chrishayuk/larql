//! Build a [`SystemGraph`] from architecture inventories.
//!
//! Placement is **evidence-driven**: roles come from declared interfaces
//! and component topologies, the feature projector is identified by its
//! shape against the declared taps, and anything the rules cannot place
//! comes back in [`BuiltGraph::unplaced`] / `unresolved_interfaces` as
//! data. The planner treats "the builder placed it" as the definition of
//! representable — there is no separate capability table to drift.

use std::collections::BTreeMap;

use larql_models::config::{PositionPolicy, LAYER_TYPE_LINEAR_ATTENTION};
use larql_models::inventory::{ArchitectureInventory, TensorGroup};

use super::component::{
    Component, ComponentRole, EncoderGeometry, Modality, PerceptionComponent, PerceptionTransform,
    ProjectionGeometry,
};
use super::edge::HiddenStateEdge;
use super::object::{Fidelity, LogicalObject, ObjectKind, Representation, SourceBinding};
use super::policy::{
    resolve_layer_kind, AttentionLayerPolicy, AttentionSpan, HeadGeometry, LayerOperator,
    RecurrenceKind,
};
use super::surface::{
    attach_stack_evidence, gate_evidence, head_from_resolved, surface_from_nested,
    surface_from_resolved,
};
use super::{SystemGraph, GRAPH_SCHEMA};

/// Sliding-layer label in the inventory's resolved table.
const RESOLVED_ATTENTION_SLIDING: &str = "sliding";

/// Role-derived component ids. Stable and conceptual; collisions get a
/// numeric suffix rather than a physical name.
const COMPONENT_ID_TARGET: &str = "target";
const COMPONENT_ID_DRAFT: &str = "draft";

/// The interface-declaring key (mirrors the inventory's registry).
const TARGET_LAYER_IDS_KEY: &str = "target_layer_ids";

/// Config leaf carrying the drafter block protocol.
const BLOCK_SIZE_KEY: &str = "block_size";

/// A tensor group no placement rule could own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnplacedGroup {
    pub artifact: String,
    pub prefix: String,
    pub reason: String,
}

/// A declared interface the builder could not turn into an edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedInterface {
    pub artifact: String,
    pub reason: String,
}

/// A component whose execution surface could not be completed — the
/// missing source facts, as data. Blocking downstream: an executor with a
/// partial surface would have to default, which G5 forbids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncompleteSurface {
    pub artifact: String,
    pub component: String,
    pub missing: Vec<String>,
}

/// The build result: the graph plus everything that did not fit.
pub struct BuiltGraph {
    pub graph: SystemGraph,
    pub unplaced: Vec<UnplacedGroup>,
    pub unresolved_interfaces: Vec<UnresolvedInterface>,
    pub incomplete_surfaces: Vec<IncompleteSurface>,
}

/// Name-fragment vocabulary for classifying tensor groups into object
/// kinds. First match wins; specific fragments precede generic ones (a
/// vision tower has `layers` segments too).
const GROUP_PATTERNS: &[(GroupClass, &[&str])] = &[
    (
        GroupClass::PerceptionTower,
        &[
            "vision_tower",
            "vision_model",
            "visual",
            "audio_tower",
            // Inkling-Small names its audio encoder `model.audio.encoder`.
            "audio.encoder",
        ],
    ),
    (
        GroupClass::PerceptionAdapter,
        &[
            "vision_adapter",
            "vision_projection",
            "mm_projector",
            "multi_modal_projector",
            // Gemma 4's `Gemma4MultimodalEmbedder` (`model.embed_vision.
            // embedding_projection`): soft tokens → language hidden. Listed
            // before the embedding fragments, which `embedding_projection`
            // would otherwise match — and it is the projector, not the
            // text embedding table.
            "embed_vision",
            "embed_audio",
            // An encoder-free modality path (Gemma 4 12B,
            // `gemma4_unified_vision`): patches are projected straight into
            // the language embedding space, so the whole modality lives
            // under one embedder prefix with no tower above it.
            //
            // These MUST be owned here rather than left to the generic
            // fragments below, and the reason is a live defect, not
            // tidiness: `model.vision_embedder.pos_embedding` contains
            // "embedding" and `model.vision_embedder.pos_norm` contains
            // "norm", so without an owning pattern the substring pass filed
            // image tensors into the LANGUAGE model's embedding and norm
            // groups (`target.embedding`), and
            // `model.embed_audio.embedding_projection` went the same way.
            // Silently misplacing a modality is worse than leaving it
            // unplaced — an unplaced group blocks the plan and says so.
            "vision_embedder",
            "audio_embedder",
        ],
    ),
    (
        GroupClass::Embedding,
        &[
            "embed_tokens",
            "wte",
            "wpe",
            "token_embd",
            "embedding",
            // Inkling-Small: `model.llm.embed` / `model.llm.unembed`.
            // Qualified with the namespace rather than matching a bare
            // "embed", which would also swallow `unembed` — Embedding is
            // scanned before Head, so the head would file as an embedding
            // table and the two 1.53 GiB objects would merge.
            "llm.embed",
        ],
    ),
    (
        GroupClass::Head,
        &["lm_head", "output.weight", "llm.unembed"],
    ),
    (GroupClass::Norm, &["norm", "ln_", "layernorm"]),
    (GroupClass::Stack, &["layers", "blocks"]),
];

/// Top-level path segments that name a tensor namespace declared *outside*
/// any component this builder places — evidence the checkpoint carries a
/// distinct sub-model the placement vocabulary has no `GroupClass`/
/// `ObjectKind` for yet. Qwen3.5's multi-token-prediction draft head is the
/// first observed case: `mtp.fc`, `mtp.layers.*`, `mtp.norm`,
/// `mtp.pre_fc_norm_hidden`, `mtp.pre_fc_norm_embedding` all live under a
/// `mtp.` prefix that sits beside — not inside — the primary text model's
/// own `model.language_model.*` tensors.
///
/// This check must run *before* the substring [`GROUP_PATTERNS`] scan, not
/// after: `mtp.layers` contains `"layers"`, `mtp.norm` and
/// `mtp.pre_fc_norm_hidden` contain `"norm"`, and `mtp.pre_fc_norm_embedding`
/// contains `"embedding"`, so each would otherwise silently name-classify as
/// `Stack`/`Norm`/`Embedding` and merge into the primary text component's
/// own `DecoderStack`/`FinalNorm`/`Embedding` object — corrupting that
/// object's tensor accounting with a different sub-model's weights. Only
/// `mtp.fc` matches no existing pattern and already surfaced honestly; every
/// other `mtp.*` group was being lost to this shadowing. A namespace here
/// classifies as [`GroupClass::Unknown`] and surfaces in `unplaced` with the
/// same "no placement rule" reason a truly-unrecognised prefix gets — there
/// is no `ObjectKind` for an MTP draft head yet, so honestly refusing to
/// place it is the correct behaviour, not a stand-in for one.
const COMPONENT_EXTERNAL_NAMESPACES: &[&str] = &["mtp"];

/// Whether `prefix`'s first `.`-separated path segment names a
/// [`COMPONENT_EXTERNAL_NAMESPACES`] entry.
fn is_component_external_namespace(prefix: &str) -> bool {
    let first_segment = prefix.split('.').next().unwrap_or(prefix);
    COMPONENT_EXTERNAL_NAMESPACES.contains(&first_segment)
}

/// Intermediate classification of one tensor-group prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupClass {
    PerceptionTower,
    PerceptionAdapter,
    Embedding,
    Head,
    Norm,
    Stack,
    Unknown,
}

/// Which modality a tensor group belongs to, from the subtree that owns it.
///
/// This is ownership, not naming: everything under `model.vision_embedder.`
/// is the image path whatever its leaf is called. Without it a group can
/// only be classified as "some perception thing", and a checkpoint with two
/// perception components has nowhere correct to put it — Gemma 4 12B bound
/// its image tensors to the AUDIO component because placement took the
/// first perception component it found.
fn group_modality(prefix: &str) -> Option<Modality> {
    // Audio first: `embed_audio` would also satisfy no image fragment, but
    // testing it first keeps the two families from ever racing.
    if prefix.contains("audio") {
        return Some(Modality::Audio);
    }
    if prefix.contains("vision") || prefix.contains("visual") {
        return Some(Modality::Image);
    }
    None
}

/// The modality a nested component declares, from the checkpoint's own key
/// (`vision_config` → the component named `vision`).
///
/// Reading the declaration, not inferring from tensor names: the checkpoint
/// states what the component perceives, and this records it so no consumer
/// has to re-derive it from an id string.
fn declared_modality(name: &str) -> Option<Modality> {
    group_modality(name)
}

/// The perception component a group belongs in.
///
/// Prefers an exact modality match. Falls back to the sole perception
/// component when the group names no modality and there is exactly one
/// candidate — which is every single-tower checkpoint, so their placement
/// is unchanged. With two perception components and no modality on the
/// group, refuses: a wrong modality is corruption, and unplaced at least
/// blocks the plan and says so.
fn perception_component_for(
    components: &[Component],
    artifact: &str,
    modality: Option<Modality>,
) -> Option<String> {
    let candidates: Vec<&Component> = components
        .iter()
        .filter(|c| c.source_artifact == artifact && c.role == ComponentRole::Perception)
        .collect();
    if let Some(modality) = modality {
        if let Some(hit) = candidates
            .iter()
            .find(|c| c.perception.map(|p| p.modality) == Some(modality))
        {
            return Some(hit.id.clone());
        }
    }
    match candidates.as_slice() {
        [only] => Some(only.id.clone()),
        _ => None,
    }
}

fn classify_group(prefix: &str) -> GroupClass {
    if is_component_external_namespace(prefix) {
        return GroupClass::Unknown;
    }
    for (class, patterns) in GROUP_PATTERNS {
        if patterns.iter().any(|p| prefix.contains(p)) {
            return *class;
        }
    }
    GroupClass::Unknown
}

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
                GroupClass::Head => text_component.clone().map(|c| (c, ObjectKind::OutputHead)),
                GroupClass::Norm => text_component.clone().map(|c| (c, ObjectKind::FinalNorm)),
                GroupClass::Stack => text_component
                    .clone()
                    .map(|c| (c, ObjectKind::DecoderStack)),
                GroupClass::Unknown => None,
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
                    reason: if matches!(class, GroupClass::Unknown) {
                        "no placement rule owns this group — judge it before conversion".to_string()
                    } else {
                        "classified for a component this artifact does not declare".to_string()
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
fn declared_taps(inventory: &ArchitectureInventory) -> Option<Vec<usize>> {
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
fn declared_block_size(inventory: &ArchitectureInventory) -> Option<usize> {
    inventory
        .config_keys
        .iter()
        .find(|f| f.path == BLOCK_SIZE_KEY || f.path.ends_with(&format!(".{BLOCK_SIZE_KEY}")))
        .and_then(|f| f.value.as_u64())
        .map(|v| v as usize)
}

/// Non-perception component owned by `artifact`.
fn component_for_artifact(components: &[Component], artifact: &str) -> Option<String> {
    components
        .iter()
        .find(|c| c.source_artifact == artifact && c.role != ComponentRole::Perception)
        .map(|c| c.id.clone())
}

fn unique_id(base: &str, components: &[Component]) -> String {
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
fn recurrence_kind(inventory: &ArchitectureInventory) -> Option<RecurrenceKind> {
    if inventory.resolved.kda.is_some() {
        return Some(RecurrenceKind::Kda);
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
fn operator_and_span(
    kind: &larql_models::config::LayerKind,
    recurrence: Option<RecurrenceKind>,
    mla: bool,
) -> (LayerOperator, Option<AttentionSpan>) {
    use larql_models::config::LayerKind;
    match kind {
        LayerKind::Full if mla => (LayerOperator::Mla, Some(AttentionSpan::Full)),
        LayerKind::Full => (LayerOperator::Softmax, Some(AttentionSpan::Full)),
        LayerKind::Sliding { .. } => (LayerOperator::Softmax, Some(AttentionSpan::Sliding)),
        LayerKind::Recurrent(_) => (
            match recurrence {
                Some(RecurrenceKind::Kda) => LayerOperator::Kda,
                Some(RecurrenceKind::GatedDelta) => LayerOperator::GatedDelta,
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
fn uses_mla(inventory: &ArchitectureInventory) -> bool {
    inventory
        .resolved
        .execution
        .as_ref()
        .is_some_and(|e| e.mla.is_some())
}

fn attention_table(inventory: &ArchitectureInventory) -> Vec<AttentionLayerPolicy> {
    let mla = uses_mla(inventory);
    inventory
        .resolved
        .layers
        .iter()
        .map(|layer| {
            // Operator and span decided together, in the one place that
            // rule lives — a recurrence gets no span rather than a
            // defaulted `Full`.
            // The checkpoint's own canonical declaration is authoritative
            // when it made one. The resolved boolean is a *derivation* —
            // it answers sliding-or-full from whichever key the parser
            // happened to read — and on a family whose interleave it
            // cannot read it answers "full" for every layer. That is how
            // Inkling-Small's 35 sliding layers (window 512, against a
            // 1,048,576-token context) were reported as retaining an
            // unbounded prefix.
            //
            // `plan::compare` keeps grading the declared array against the
            // boolean, so the comparison it makes stays a real one: the
            // authority moves here, not there.
            let (operator, span) = match layer.declared_kind.as_ref() {
                // A spelling with no kind keeps the layer-blind path: the
                // graph records a softmax layer whose declaration it
                // cannot round-trip to, which is exactly `unexpressed`.
                Some(larql_models::config::LayerKind::Unexpressed { .. }) | None => {
                    resolve_layer_kind(
                        layer.declared_span.as_deref(),
                        layer.attention == RESOLVED_ATTENTION_SLIDING,
                        recurrence_kind(inventory),
                        mla,
                    )
                }
                Some(kind) => operator_and_span(kind, recurrence_kind(inventory), mla),
            };
            // The architecture's resolved window stays authoritative — it
            // can apply per-family rules the raw config cannot state. The
            // declared window is the FALLBACK, for a family whose window
            // spelling the parser does not know (Inkling-Small writes
            // `sliding_window_size`, not `sliding_window`), where the
            // architecture yields nothing and a sliding layer would
            // otherwise carry no size at all.
            let declared_window = match layer.declared_kind.as_ref() {
                Some(larql_models::config::LayerKind::Sliding { window }) => *window,
                _ => None,
            };
            AttentionLayerPolicy {
                operator,
                span,
                window: layer.window.or(declared_window),
                position: layer.position,
                geometry: Some(HeadGeometry {
                    head_dim: layer.head_dim,
                    num_kv_heads: layer.num_kv_heads,
                }),
                v_from_k: layer.v_from_k,
                // Carried alongside the boolean-derived `span` verbatim, so a
                // declared spelling the vocabulary cannot express (a hybrid
                // linear-attention interleave) is recorded rather than
                // silently lost behind whatever `span` defaulted to. See
                // `AttentionLayerPolicy::declared_span` and
                // `plan::carriage::probe_layer_types`.
                declared_span: layer.declared_span.clone(),
            }
        })
        .collect()
}

/// The per-layer policy of a **nested** component, from that component's
/// own declared topology.
///
/// The same shape as [`attention_table`] one level down: a component's
/// per-layer policy comes from that component's facts. Nested components
/// previously carried `attention: None` — honest at the time, but it
/// meant a perception tower's declared interleave and rope base were
/// parsed into [`ComponentTopology`] and then dropped before the graph,
/// with nothing reporting the loss.
///
/// A component that declares a layer count but no `layer_types` attends
/// fully on every layer — that is what the absence of an interleave means
/// in the HF configs that omit it (Gemma 4's vision tower: 27 layers, one
/// rope base, no `layer_types`), and recording it lets the tower's rope
/// facts be judged against a table instead of vanishing. `None` when the
/// component declares neither a count nor an interleave, or when it names
/// a span the vocabulary does not contain — refusing rather than
/// resolving an unknown spelling to a default, which is the failure this
/// vocabulary exists to prevent.
fn nested_attention_table(
    topology: &larql_models::inventory::components::ComponentTopology,
) -> Option<Vec<AttentionLayerPolicy>> {
    let uniform_full;
    let layer_types: &Vec<String> = match (&topology.layer_types, topology.num_layers) {
        (Some(declared), _) => declared,
        (None, Some(n)) => {
            uniform_full = vec![AttentionSpan::Full.declared_name().to_string(); n];
            &uniform_full
        }
        (None, None) => return None,
    };
    // A rope base the component declares for itself; absent means the
    // component states no position policy, which is a fact, not a zero.
    let position = match topology.rope_theta {
        Some(theta) => PositionPolicy::from_declared_theta(theta),
        None => PositionPolicy::None,
    };
    layer_types
        .iter()
        .map(|entry| {
            // A nested component's contract is stricter than the text
            // stack's: it has no resolved boolean to fall back on, so an
            // unrecognised spelling refuses the whole table rather than
            // deferring to a comparison downstream. `linear_attention` is
            // named here so that a perception tower which declares a
            // recurrence records one — no judged tower does today, and
            // the alternative is for it to refuse a spelling the schema
            // now has a home for.
            let (operator, span) = if entry.eq_ignore_ascii_case(LAYER_TYPE_LINEAR_ATTENTION) {
                // A nested component has no linear-attention geometry to
                // resolve — the recurrence keys are text-stack keys — so
                // a tower declaring one is recorded as an unidentified
                // recurrence, never as Gated DeltaNet.
                (LayerOperator::Recurrent, None)
            } else {
                (
                    LayerOperator::Softmax,
                    Some(AttentionSpan::from_declared(entry)?),
                )
            };
            Some(AttentionLayerPolicy {
                operator,
                span,
                // No nested component declares a sequence window today;
                // a spatial window's extent is not a position count.
                window: None,
                position,
                // A nested topology states one head geometry for the
                // whole tower; the surface carries it.
                geometry: None,
                v_from_k: false,
                declared_span: Some(entry.clone()),
            })
        })
        .collect()
}

/// Merge one tensor group into the `(component, kind)` object, creating it
/// on first sight.
fn merge_binding(
    objects: &mut BTreeMap<String, LogicalObject>,
    component: &str,
    kind: ObjectKind,
    artifact: &str,
    group: &TensorGroup,
    inventory: &ArchitectureInventory,
) {
    let id = format!("{component}.{}", kind.name());
    let object = objects.entry(id.clone()).or_insert_with(|| LogicalObject {
        id,
        component: component.to_string(),
        kind,
        source_bindings: Vec::new(),
        representations: canonical_representation(inventory, |name| {
            name.starts_with(&group.prefix)
        })
        .into_iter()
        .collect(),
    });
    object.source_bindings.push(SourceBinding {
        artifact: artifact.to_string(),
        tensor_prefix: group.prefix.clone(),
        tensors: group.tensors,
        bytes: group.bytes,
    });
}

/// Carve every routed layer's expert bank out of a just-placed decoder
/// stack into the component's [`ObjectKind::ExpertBank`] object, one
/// binding per layer at the prefix the architecture named
/// (`resolved.layers[L].expert_bank`). Ownership is settled by binding
/// specificity ([`super::object::most_specific_owner`]), so the stack keeps
/// its whole-stack binding and simply stops owning what the bank binds;
/// its recorded counts and representation are re-derived over what it
/// still owns, so neither object describes bytes it does not hold.
fn carve_expert_banks(
    objects: &mut BTreeMap<String, LogicalObject>,
    component: &str,
    artifact: &str,
    group: &TensorGroup,
    inventory: &ArchitectureInventory,
) {
    let bank_prefixes: Vec<&str> = inventory
        .resolved
        .layers
        .iter()
        .filter_map(|l| l.expert_bank.as_deref())
        .filter(|p| {
            p.strip_prefix(&group.prefix)
                .is_some_and(|r| r.starts_with('.'))
        })
        .collect();
    if bank_prefixes.is_empty() {
        return;
    }
    let in_bank = |name: &str| {
        bank_prefixes
            .iter()
            .any(|p| name == *p || name.strip_prefix(p).is_some_and(|r| r.starts_with('.')))
    };
    let bank_id = format!("{component}.{}", ObjectKind::ExpertBank.name());
    let bank = objects
        .entry(bank_id.clone())
        .or_insert_with(|| LogicalObject {
            id: bank_id,
            component: component.to_string(),
            kind: ObjectKind::ExpertBank,
            source_bindings: Vec::new(),
            representations: canonical_representation(inventory, in_bank)
                .into_iter()
                .collect(),
        });
    let mut carved_tensors = 0usize;
    let mut carved_bytes = 0u64;
    for prefix in &bank_prefixes {
        let (tensors, bytes) = inventory
            .tensors
            .tensors
            .iter()
            .filter(|t| {
                t.name == *prefix
                    || t.name
                        .strip_prefix(prefix)
                        .is_some_and(|r| r.starts_with('.'))
            })
            .fold((0usize, 0u64), |(n, b), t| (n + 1, b + t.bytes));
        carved_tensors += tensors;
        carved_bytes += bytes;
        bank.source_bindings.push(SourceBinding {
            artifact: artifact.to_string(),
            tensor_prefix: (*prefix).to_string(),
            tensors,
            bytes,
        });
    }
    // The stack no longer owns those tensors: correct its counts and its
    // representation to what it still holds.
    let stack_id = format!("{component}.{}", ObjectKind::DecoderStack.name());
    if let Some(stack) = objects.get_mut(&stack_id) {
        if let Some(binding) = stack
            .source_bindings
            .iter_mut()
            .find(|b| b.artifact == artifact && b.tensor_prefix == group.prefix)
        {
            binding.tensors = binding.tensors.saturating_sub(carved_tensors);
            binding.bytes = binding.bytes.saturating_sub(carved_bytes);
        }
        stack.representations = canonical_representation(inventory, |name| {
            name.starts_with(&group.prefix) && !in_bank(name)
        })
        .into_iter()
        .collect();
    }
}

/// The MXFP4 pair suffixes HF writes for a block-quantised tensor: the
/// packed e2m1 nibbles and the e8m0 scales, both stored as `U8`.
const MXFP4_BLOCKS_SUFFIX: &str = "_blocks";
const MXFP4_SCALES_SUFFIX: &str = "_scales";
/// The encoding name a declared MXFP4 tensor is placed under, in the same
/// vocabulary as the region formats a container writes.
const MXFP4_ENCODING: &str = "MXFP4";

/// The encoding one tensor is placed under: its shard dtype, unless the
/// checkpoint's declared stored representation says those bytes are
/// something else. A `U8` `*_blocks` / `*_scales` tensor under an `mxfp4`
/// declaration, outside `modules_to_not_convert`, is MXFP4 — placing it as
/// raw bytes would drop the one fact that gives the bytes meaning.
fn tensor_encoding<'a>(
    inventory: &'a ArchitectureInventory,
    name: &str,
    dtype: &'a str,
) -> &'a str {
    let Some(rep) = inventory.stored_representation.as_ref() else {
        return dtype;
    };
    let declared_mxfp4 = rep
        .method
        .eq_ignore_ascii_case(larql_models::inventory::representation::QUANT_METHOD_MXFP4);
    let mxfp4_pair = name.ends_with(MXFP4_BLOCKS_SUFFIX) || name.ends_with(MXFP4_SCALES_SUFFIX);
    if declared_mxfp4 && mxfp4_pair && dtype == "U8" && !rep.excludes(name) {
        MXFP4_ENCODING
    } else {
        dtype
    }
}

/// The canonical representation for an object: encodings actually observed
/// in the shard headers under this prefix — read through the checkpoint's
/// declared stored representation, see [`tensor_encoding`] — falling back
/// to the checkpoint's declared dtype when the per-tensor list was
/// stripped. Never invented.
fn canonical_representation(
    inventory: &ArchitectureInventory,
    is_member: impl Fn(&str) -> bool,
) -> Option<Representation> {
    let mut encodings: Vec<&str> = inventory
        .tensors
        .tensors
        .iter()
        .filter(|t| is_member(&t.name))
        .map(|t| tensor_encoding(inventory, &t.name, t.dtype.as_str()))
        .collect();
    encodings.sort_unstable();
    encodings.dedup();
    let encoding = if encodings.is_empty() {
        inventory.identity.dtype.clone()?
    } else {
        encodings.join("+")
    };
    Some(Representation {
        encoding,
        fidelity: Fidelity::Canonical,
    })
}

/// Identify the tap-fusion projector by shape evidence: a 2-D tensor of
/// shape `len(taps)·hidden × hidden` (either orientation). Returns the
/// tensor's first path segment — the adjacency key that claims its sibling
/// groups — or records why identification failed.
fn find_projector_segment(
    artifact: &str,
    inventory: &ArchitectureInventory,
    taps: &[usize],
    unresolved: &mut Vec<UnresolvedInterface>,
) -> Option<String> {
    let hidden = inventory.resolved.hidden_size;
    let expected = taps.len() * hidden;
    if inventory.tensors.tensors.is_empty() {
        unresolved.push(UnresolvedInterface {
            artifact: artifact.to_string(),
            reason: "per-tensor list absent — projector cannot be identified by shape".to_string(),
        });
        return None;
    }
    let projector_tensor = inventory.tensors.tensors.iter().find(|t| {
        t.shape.len() == 2
            && ((t.shape[0] == expected && t.shape[1] == hidden)
                || (t.shape[0] == hidden && t.shape[1] == expected))
    });
    match projector_tensor {
        Some(tensor) => Some(
            tensor
                .name
                .split('.')
                .next()
                .unwrap_or(&tensor.name)
                .to_string(),
        ),
        None => {
            unresolved.push(UnresolvedInterface {
                artifact: artifact.to_string(),
                reason: format!(
                    "no 2-D tensor of shape {expected}×{hidden} implements the declared \
                     {}-tap interface",
                    taps.len()
                ),
            });
            None
        }
    }
}

/// Wire the interface edge: the producer must be exactly one other
/// component deep enough to own every declared tap.
fn wire_edge(
    artifact: &str,
    inventory: &ArchitectureInventory,
    taps: &[usize],
    components: &[Component],
    edges: &mut Vec<HiddenStateEdge>,
    unresolved: &mut Vec<UnresolvedInterface>,
) {
    let Some(consumer) = component_for_artifact(components, artifact) else {
        unresolved.push(UnresolvedInterface {
            artifact: artifact.to_string(),
            reason: "declaring artifact has no component".to_string(),
        });
        return;
    };
    let max_tap = taps.iter().copied().max().unwrap_or(0);
    let mut producers = components.iter().filter(|c| {
        c.source_artifact != *artifact
            && c.role == ComponentRole::PrimaryText
            && c.num_layers > max_tap
    });
    match (producers.next(), producers.next()) {
        (Some(producer), None) => edges.push(HiddenStateEdge {
            producer_component: producer.id.clone(),
            producer_layers: taps.to_vec(),
            consumer_component: consumer.clone(),
            consumer_object: format!("{consumer}.{}", ObjectKind::FeatureProjector.name()),
            block_size: declared_block_size(inventory),
        }),
        (None, _) => unresolved.push(UnresolvedInterface {
            artifact: artifact.to_string(),
            reason: format!("no component in the set owns layer {max_tap} — producer unresolvable"),
        }),
        (Some(_), Some(_)) => unresolved.push(UnresolvedInterface {
            artifact: artifact.to_string(),
            reason: "multiple candidate producers — refusing to guess".to_string(),
        }),
    }
}
