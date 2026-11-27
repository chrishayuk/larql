//! Build a [`SystemGraph`] from architecture inventories.
//!
//! Placement is **evidence-driven**: roles come from declared interfaces
//! and component topologies, the feature projector is identified by its
//! shape against the declared taps, and anything the rules cannot place
//! comes back in [`BuiltGraph::unplaced`] / `unresolved_interfaces` as
//! data. The planner treats "the builder placed it" as the definition of
//! representable — there is no separate capability table to drift.

use larql_models::config::ResidualTopology;
use larql_models::inventory::ArchitectureInventory;

use super::component::{Component, ComponentRole, Modality};
use super::roles::{is_hyper_connection_head_group, ATTENTION_RESIDUAL_EXIT_LEAVES};
use super::SystemGraph;

mod components;
mod tables;
pub use components::*;
use tables::*;

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
    // A learned absolute position table — before the token-embedding
    // fragments, so GPT-2's `wpe` is never merged into `wte`'s object.
    (GroupClass::PositionEmbedding, &["wpe"]),
    (
        GroupClass::Embedding,
        &[
            "embed_tokens",
            "wte",
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
    // The attention-residual exit's pair, and it MUST precede the
    // generic `norm` fragment below. `output_attn_res_norm` contains
    // "norm", so the substring pass filed it into the text component's
    // FinalNorm object — which then bound two tensors while claiming to
    // be the single final norm, and the `[1, hidden]` projection beside
    // it matched nothing and surfaced as an unplaced group. Ownership of
    // a pair that belongs to ONE operation was decided by substring luck
    // in both directions, and only the op plan's `single(FinalNorm)`
    // check could ever have seen the half that was silent.
    //
    // Recognition is not ownership: the placement arm still asks whether
    // the artifact DECLARES the topology, and refuses by name when it
    // does not.
    (
        GroupClass::AttentionResidualExit,
        ATTENTION_RESIDUAL_EXIT_LEAVES,
    ),
    (GroupClass::Norm, &["norm", "ln_", "layernorm"]),
    (GroupClass::Stack, &["layers", "blocks"]),
];

/// The GPT-2 lineage's decoder block list, matched as a whole segment.
const GPT2_BLOCK_LIST: &str = "h";

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
    PositionEmbedding,
    Head,
    Norm,
    Stack,
    /// One of the Sinkhorn hyper-connection head's three bare tensor
    /// groups (`hc_head_{fn,base,scale}`), recognised by exact name from
    /// the role module's own table. Recognition is not ownership: the
    /// placement arm still asks whether the artifact DECLARES the
    /// topology, and refuses by name when it does not.
    HyperConnectionHead,
    /// One of the attention-residual exit's two tensor groups
    /// (`output_attn_res_{norm,proj}`), recognised by name fragment from
    /// the role module's own table. Recognition is not ownership here
    /// either: the placement arm asks the declaration.
    AttentionResidualExit,
    Unknown,
}

/// Whether this artifact declares the Sinkhorn-split hyper-connection
/// topology — `hc_mult`, `hc_sinkhorn_iters` and `hc_eps` together,
/// resolved once by the architecture and never re-derived here.
///
/// The gate on placing the head's operands. Hy4-preview declares a
/// Sinkhorn-free variant (no iteration count) and resolves to NO
/// topology, so its head stays unplaced whatever it is called; a
/// single-stream checkpoint carrying the three names would be a
/// disagreement between estate and declaration, and is refused as one.
fn declares_sinkhorn_hyper_connection(inventory: &ArchitectureInventory) -> bool {
    matches!(
        inventory
            .resolved
            .execution
            .as_ref()
            .and_then(|e| e.residual_topology),
        Some(ResidualTopology::HyperConnection(_))
    )
}

/// Whether this artifact declares the attention-residual topology —
/// `attn_res_block_size`, resolved once by the architecture and never
/// re-derived here.
///
/// The gate on placing the exit pair. A checkpoint carrying the two
/// names without the period is a disagreement between estate and
/// declaration, and is refused as one rather than acquiring a residual
/// topology from its tensor spellings.
fn declares_attention_residual(inventory: &ArchitectureInventory) -> bool {
    matches!(
        inventory
            .resolved
            .execution
            .as_ref()
            .and_then(|e| e.residual_topology),
        Some(ResidualTopology::AttentionResidual { .. })
    )
}

/// The refusal a recognised attention-residual exit group gets on an
/// artifact that does not declare the topology it belongs to.
const ATTN_RES_EXIT_UNDECLARED_REASON: &str =
    "recognised as an attention-residual exit operand, but the artifact declares no \
     attn_res_block_size — the estate and the declaration disagree, so the operand has no owner";

/// The refusal a recognised hyper-connection head group gets on an
/// artifact that does not declare the topology it belongs to.
const HC_HEAD_UNDECLARED_REASON: &str =
    "recognised as a Sinkhorn hyper-connection head operand, but the artifact declares no \
     Sinkhorn-split hyper-connection topology (hc_mult, hc_sinkhorn_iters and hc_eps \
     together) — the estate and the declaration disagree, so the operand has no owner";

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
    // Exact, and before the substring scan: the head's three groups are
    // bare top-level names, and `mtp.0.hc_head_fn` never reaches here
    // because its group is the external `mtp` namespace above.
    if is_hyper_connection_head_group(prefix) {
        return GroupClass::HyperConnectionHead;
    }
    // GPT-2's lineage names its block list `h` (`transformer.h.{L}`, or a
    // bare `h.{L}` in the original OpenAI conversion). An exact LAST
    // segment, never a substring: every prefix contains an `h`.
    if prefix.rsplit('.').next() == Some(GPT2_BLOCK_LIST) {
        return GroupClass::Stack;
    }
    for (class, patterns) in GROUP_PATTERNS {
        if patterns.iter().any(|p| prefix.contains(p)) {
            return *class;
        }
    }
    GroupClass::Unknown
}
