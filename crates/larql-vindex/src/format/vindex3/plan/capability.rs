//! Execution admissibility, scoped to what a request actually needs.
//!
//! Today's plan answers one question — *is every declared semantic fact of
//! this checkpoint representable?* — and reports it as a single Boolean.
//! That is the right question for **model completeness**, and this module
//! does not weaken it: [`SystemPlan::admissible`] still means
//! `blocking == 0` over the whole system.
//!
//! It is the wrong question for **execution**. A checkpoint is not one
//! program. Qwen3.8-27B carries a vision tower whose execution surface
//! this build cannot resolve, and 12 of its 30 blocking findings are that
//! tower — so ordinary text generation, which never reaches a vision
//! weight, is refused for a reason that cannot affect it. Meanwhile
//! Gemma 4 26B-A4B reports `admissible: true` while shipping **zero**
//! audio tensors against a declared `audio_config` and `audio_token_id`:
//! the Boolean says yes to a capability the container cannot perform.
//!
//! Both failures are the same failure. Representability is a property of
//! the *model*; admissibility is a property of a *capability over* that
//! model. This module adds the second without touching the first.
//!
//! # Membership is semantic, not ownership
//!
//! The tempting implementation is `finding.component == "vision"`. It is
//! wrong, and this checkpoint says so out loud: `vision_start_token_id`
//! and `vision_end_token_id` (Qwen3.8), and `image_token_id` / `boi_token_id`
//! / `eoi_token_id` and the `audio_*` family (Gemma 4), all live at
//! `component: "root"`. They are not the tower. They are the **binding
//! edge** — how an image enters the token stream — and text generation
//! must not depend on them while image-conditioned generation must.
//! A component filter passes that case while being wrong about it, which
//! is why membership is decided on the subject's own semantics here.
//!
//! # Fail closed
//!
//! An unclassified subject is required by **every** capability. A new
//! config key, a new tensor group, a spelling this build has never seen —
//! all block everything until something claims them. The opposite default
//! would let an unrecognised blocker quietly exempt itself from the
//! capability that actually needs it, which is the exact shape of the
//! `layer_types` bug: parsed, never consulted, silently fine.

use serde::{Deserialize, Serialize};

use super::super::graph::{Component, ComponentRole, Modality, SystemGraph};
use super::report::Finding;

/// One thing a caller can ask the model to do.
///
/// Deliberately phrased as capabilities of a *request*, not names of
/// components: `ImageConditioned` is a text generation that additionally
/// consumes an image, so it requires everything [`Self::TextGeneration`]
/// requires and more. Nothing here names a family or an architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Text in, text out. The language model and nothing else.
    TextGeneration,
    /// Text generation additionally conditioned on image input.
    ImageConditioned,
    /// Text generation additionally conditioned on audio input.
    AudioConditioned,
    /// Speculative drafting with a multi-token-prediction head — an
    /// OPTIONAL acceleration path, not a prerequisite for base decode.
    ///
    /// Its own capability because the alternative is worse in both
    /// directions: folding MTP into [`Self::TextGeneration`] makes
    /// ordinary decode wait on a draft head it never runs, and dropping
    /// MTP from the census would hide seven real findings.
    Drafting,
}

impl Capability {
    /// Every capability this build can be asked about.
    pub const ALL: [Self; 4] = [
        Self::TextGeneration,
        Self::ImageConditioned,
        Self::AudioConditioned,
        Self::Drafting,
    ];

    /// The modality this capability conditions on, if any.
    fn modality(self) -> Option<Modality> {
        match self {
            Self::TextGeneration => None,
            Self::ImageConditioned => Some(Modality::Image),
            Self::AudioConditioned => Some(Modality::Audio),
            Self::Drafting => None,
        }
    }
}

/// What a subject is, for the purpose of deciding who depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubjectKind {
    /// The language model: every capability runs it.
    Language,
    /// A modality's transform — encoder, projection, whatever the family
    /// uses — reached only by that modality's capability.
    Transform(Modality),
    /// The edge that splices a modality into the token stream. Lives at
    /// the root of the config, belongs to the modality.
    Binding(Modality),
    /// The multi-token-prediction draft head: its own sub-model, reached
    /// only by [`Capability::Drafting`].
    Drafting,
    /// Not classified. Required by everything (see the module contract).
    Unclassified,
}

/// Root-level config keys that bind a modality into the token stream.
///
/// These are the CAP-1 control: every one of them sits at
/// `component: "root"`, so component ownership cannot reach them, and
/// every one of them is required by exactly one modality.
const IMAGE_BINDING_KEYS: &[&str] = &[
    "vision_start_token_id",
    "vision_end_token_id",
    "image_token_id",
    "boi_token_id",
    "eoi_token_id",
    "video_token_id",
    "vision_soft_tokens_per_image",
];

const AUDIO_BINDING_KEYS: &[&str] = &[
    "audio_token_id",
    "boa_token_id",
    "eoa_token_id",
    "eoa_token_index",
];

/// The modality a component perceives, asked of the graph.
///
/// PERCEPTION-2: the component id is an **identifier and nothing more**.
/// This used to match `"vision"`/`"audio"` as strings, which is the same
/// lexical-fallback family that bound `model.vision_embedder.pos_embedding`
/// to `target.embedding` and routed Gemma 4 12B's image tensors into its
/// AUDIO component. A component states what it perceives; nothing here
/// guesses it from a name.
///
/// `None` when the id names no component in the graph, or names one that
/// is not a perception component, or one that declares no modality —
/// every case falls through to the fail-closed default.
fn component_modality(graph: &SystemGraph, component: &str) -> Option<Modality> {
    graph
        .components
        .iter()
        .find(|c: &&Component| c.id == component)
        .filter(|c| c.role == ComponentRole::Perception)
        .and_then(|c| c.perception)
        .map(|p| p.modality)
}

/// Config-key prefix and tensor-namespace prefix of the MTP draft head.
///
/// Two spellings because the head declares itself twice: config keys
/// under `text_config.mtp_*`, and a tensor namespace `mtp.*` that sits
/// beside the primary text model's own weights (see
/// `graph::build::COMPONENT_EXTERNAL_NAMESPACES`).
const MTP_CONFIG_PREFIX: &str = "mtp_";
const MTP_TENSOR_PREFIX: &str = "mtp.";

fn classify(finding: &Finding, graph: &SystemGraph) -> SubjectKind {
    let subject = finding.subject.as_str();
    // Binding first: these live at root, so any component-based test would
    // reach the wrong answer before this one ran.
    let leaf = subject.rsplit('.').next().unwrap_or(subject);
    if IMAGE_BINDING_KEYS.contains(&leaf) {
        return SubjectKind::Binding(Modality::Image);
    }
    if AUDIO_BINDING_KEYS.contains(&leaf) {
        return SubjectKind::Binding(Modality::Audio);
    }
    // The draft head, by either of its two spellings. Placed with the
    // binding keys and before any component test for the same reason:
    // `mtp.fc` and friends carry no component at all.
    if leaf.starts_with(MTP_CONFIG_PREFIX) || subject.starts_with(MTP_TENSOR_PREFIX) {
        return SubjectKind::Drafting;
    }
    // The owning component's own declaration outranks the subject's
    // spelling: a component knows what it perceives.
    if let Some(modality) = component_modality(graph, &finding.component) {
        return SubjectKind::Transform(modality);
    }
    // A modality's config block, named at root rather than owned by any
    // built component — Gemma 4 26B-A4B declares `audio_config` and builds
    // no audio component at all. This reads the checkpoint's own declared
    // key, not a component id.
    if let Some(modality) = subject.split('.').next().and_then(|head| match head {
        "vision_config" => Some(Modality::Image),
        "audio_config" => Some(Modality::Audio),
        _ => None,
    }) {
        return SubjectKind::Transform(modality);
    }
    match finding.component.as_str() {
        "text" | "target" => SubjectKind::Language,
        _ => SubjectKind::Unclassified,
    }
}

/// Whether `capability`'s execution depends on this finding's subject.
pub fn requires(capability: Capability, finding: &Finding, graph: &SystemGraph) -> bool {
    if outside_closure(capability, finding, graph) {
        return false;
    }
    match classify(finding, graph) {
        SubjectKind::Language | SubjectKind::Unclassified => true,
        SubjectKind::Drafting => capability == Capability::Drafting,
        SubjectKind::Transform(modality) | SubjectKind::Binding(modality) => {
            capability.modality() == Some(modality)
        }
    }
}

/// A subject whose SEMANTICS are unresolved but which this container's own
/// structure proves cannot reach `capability`'s execution.
///
/// **This is not "unknown keys stop mattering".** The default is unchanged
/// and fail-closed: an unclassified subject is required by every
/// capability, and `an_undisposed_unknown_text_key_still_blocks_text`
/// holds that line. An entry here is a narrow claim carrying its own
/// falsifier — a predicate over the BUILT GRAPH, so the exclusion is
/// conditional on the evidence actually being present in the container in
/// front of us rather than on the architecture's name.
///
/// The finding itself is untouched: it keeps its class, its
/// `Unrepresented` carriage and its place in the whole-model census. Only
/// capability relevance is decided here, which is why whole-model
/// admissibility can stay false while text generation runs.
fn outside_closure(capability: Capability, finding: &Finding, graph: &SystemGraph) -> bool {
    // Every disposition below is scoped to text generation; a capability
    // whose closure genuinely includes these has no entry.
    if capability != Capability::TextGeneration {
        return false;
    }
    let leaf = finding
        .subject
        .rsplit('.')
        .next()
        .unwrap_or(&finding.subject);
    match leaf {
        // The gate this key might describe is determined by executable
        // structure, not by the key: a double-width `q_proj`, a per-head
        // interleave, `sigmoid`, and a placement — all judged from the
        // reference implementation and mutation-proven on the shipped
        // path (`opplan::exec::tests::output_gate_fused`). HF reads this
        // key nowhere. So whatever it names, the text operator we execute
        // is already fully determined without it.
        //
        // The evidence is required, not assumed: a container whose text
        // component carries NO represented gate gets no exclusion, and
        // the key blocks as before.
        "output_gate_type" => text_component(graph)
            .and_then(|c| c.execution.as_ref())
            .is_some_and(|e| {
                e.attention
                    .as_ref()
                    .is_some_and(|a| a.output_gate.is_some())
            }),
        // A packaging statement, not an operator one — and unlike the
        // gate key it is not even an HF key: zero references across all
        // of transformers. What makes it excludable is that the graph
        // CORROBORATES it as a composition fact: `false` beside a real
        // perception component says the package is not text-only, which
        // is a claim about what the checkpoint contains, not about what
        // the text stack computes.
        //
        // A checkpoint asserting `true` while shipping a perception
        // component is contradicting itself and keeps blocking.
        "language_model_only" => {
            let declares_text_only = finding.declared.as_ref().and_then(|v| v.as_bool());
            let has_perception = graph
                .components
                .iter()
                .any(|c| c.role == ComponentRole::Perception);
            declares_text_only == Some(!has_perception)
        }
        _ => false,
    }
}

/// The primary text component, when the graph has exactly one.
///
/// Ambiguity resolves to `None`, never to the first match (drill F10):
/// grading capabilities against an arbitrarily chosen text component
/// would be quiet wrongness; absence downgrades visibly instead.
fn text_component(graph: &SystemGraph) -> Option<&Component> {
    graph.primary_text_component().ok()
}

/// Admissibility of one capability: does anything it depends on block?
///
/// Note what this does **not** do: it never inspects
/// [`SystemPlan::admissible`](super::report::SystemPlan). A capability is
/// admissible on its own dependency closure, so a container can be
/// inadmissible as a whole and still execute text perfectly well — which
/// is the entire point.
pub fn admissible_for<'a>(
    capability: Capability,
    findings: impl IntoIterator<Item = &'a Finding>,
    graph: &SystemGraph,
) -> CapabilityStatus {
    let mut blocking = 0usize;
    let mut required = 0usize;
    for finding in findings {
        if !requires(capability, finding, graph) {
            continue;
        }
        required += 1;
        if finding.blocks() {
            blocking += 1;
        }
    }
    CapabilityStatus {
        capability,
        admissible: blocking == 0,
        available: available_for(capability, graph),
        supported: supported(capability),
        required,
        blocking,
    }
}

/// Does this checkpoint actually **contain** what the capability needs?
///
/// A different question from [`admissible_for`], deliberately kept apart.
/// Gemma 4 26B-A4B is the control that forces the distinction: it declares
/// `audio_config` and `audio_token_id`, every audio finding is
/// representable, and it ships **zero** audio tensors. Understanding a
/// modality's semantics and possessing its operands are independent facts,
/// and collapsing them would report the container as *not understanding*
/// audio when what it lacks is the weights.
///
/// Asked of the graph, because presence is an operand question: a modality
/// is available when a perception component declaring it exists AND some
/// object of that component is backed by real tensors. A component built
/// from a config block that ships no weights answers `false`.
pub fn available_for(capability: Capability, graph: &SystemGraph) -> bool {
    if capability == Capability::Drafting {
        // Fail-closed, and for a reason worth stating: availability is an
        // OPERAND question asked of the graph, and the builder places no
        // `mtp.*` group at all — there is no `ObjectKind` for a draft
        // head yet, so every one of them surfaces as unplaced. The
        // tensors are in the checkpoint; the graph cannot say so. This
        // answers `false` because "the graph holds no draft-head object"
        // is what it can honestly check, and it becomes a real question
        // when the head gets a placement rule.
        //
        // Deliberately NOT an invented `ObjectKind::DraftHead` with no
        // builder behind it: that would make the graph assert a placement
        // rule this build does not have.
        let _ = graph;
        return false;
    }
    let Some(modality) = capability.modality() else {
        // Text generation needs the language model itself.
        return graph
            .components
            .iter()
            .any(|c| c.role == ComponentRole::PrimaryText);
    };
    let component = graph.components.iter().find(|c| {
        c.role == ComponentRole::Perception && c.perception.map(|p| p.modality) == Some(modality)
    });
    let Some(component) = component else {
        return false;
    };
    graph
        .objects
        .iter()
        .any(|o| o.component == component.id && !o.source_bindings.is_empty())
}

/// Does this **build** have an execution implementation for this
/// capability?
///
/// A property of the build alone — it asks nothing of any checkpoint. A
/// capability can be unsupported on a container that represents it
/// perfectly and ships every operand, which is exactly Gemma 4 26B-A4B's
/// image path today.
///
/// Kept separate from [`runnable`] because the two answer different
/// questions for different readers: `supported` sends you to this
/// codebase, `available` sends you to the checkpoint, and `admissible`
/// sends you to the parser.
pub fn supported(capability: Capability) -> bool {
    match capability {
        // VINDEX3 executes decoder stacks.
        Capability::TextGeneration => true,
        // No perception executor exists: nothing in `opplan/exec` matches
        // `ComponentRole::Perception`, and the V3 runtime takes tokens and
        // nothing else — there is no image or audio input surface to feed
        // one.
        Capability::ImageConditioned | Capability::AudioConditioned => false,
        // No draft head executor: the op plan has no `ObjectKind` for an
        // MTP head, so `mtp.*` surfaces as unplaced and nothing consumes
        // it. Speculative decode is a separate rung.
        Capability::Drafting => false,
    }
}

/// What one capability's dependency closure and operand estate say about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityStatus {
    pub capability: Capability,
    /// Is the capability's semantics **understood**? True iff nothing in
    /// its dependency closure blocks.
    ///
    /// Says nothing about whether the operands exist — see
    /// [`Self::available`]. The two are reported side by side rather than
    /// folded together, because "we do not understand audio" and "this
    /// checkpoint ships no audio" send a reader to entirely different
    /// places.
    pub admissible: bool,
    /// Does this checkpoint **contain** what the capability needs?
    pub available: bool,
    /// Does this **build** implement execution for it?
    ///
    /// Independent of the other two on purpose. `admissible && available &&
    /// !supported` is a real and useful state — it says the gap is this
    /// build's, not the checkpoint's.
    pub supported: bool,
    /// Findings in this capability's closure.
    pub required: usize,
    /// Of those, how many block.
    pub blocking: usize,
}

impl CapabilityStatus {
    /// Can this capability actually be run: understood, present, AND
    /// implemented.
    ///
    /// The conjunction is the only thing a caller should gate execution on.
    /// The three components are reported alongside it because *which* one
    /// is false is the whole diagnostic — Qwen3.8 text is
    /// `available + supported` but not `admissible`, meaning the weights
    /// are here and this build knows how to run a decoder stack, and doing
    /// so would still not be trustworthy while its semantics are
    /// unresolved.
    pub fn runnable(&self) -> bool {
        self.admissible && self.available && self.supported
    }
}
