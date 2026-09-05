//! Output types for the semantic representability plan (`larql vindex3 plan`).
//!
//! A plan is a statement over one or more inspected artifacts of **exactly
//! why** the current VINDEX3 container schema can or cannot faithfully encode
//! them. Findings are typed twice:
//!
//! - **category** — what kind of statement this is: representable,
//!   value-mismatched between authorities, unrepresented, or a declared
//!   cross-component interface.
//! - **semantic class** — how much losing it would matter. `consumed` in the
//!   inventory is *not* sufficient: a key can be consumed and still resolve
//!   to a different value than the checkpoint declared, which is why
//!   mismatch findings compare values, not statuses.
//!
//! The verdict is fail-closed: any mismatched/unrepresented finding whose
//! class is execution-, tensor-, or interface-semantic — or whose key the
//! registry has never seen (`unknown`) — makes the plan inadmissible.

use serde::{Deserialize, Serialize};

use super::semantics::SemanticCluster;

use super::carriage::Carriage;
use crate::error::VindexError;
use crate::format::vindex3::graph::SystemGraph;

/// Current plan schema. Bump on any breaking change to these types.
///
/// v2: the plan carries the built [`SystemGraph`] — representability is
/// defined as "the graph builder placed it", and the graph is the proof.
/// Interfaces are reported as resolved edges rather than candidate guesses.
///
/// v3: findings about config keys carry a [`Carriage`] stage, and the
/// census reports every declared key rather than only the unconsumed
/// ones — so `unrepresented: N` is a count against a stated denominator
/// instead of a lower bound. Adds the `training_only` and `alias`
/// semantic classes.
///
/// v4: the plan names who judged it and what it judged. `planner` carries
/// the planner's package version and its *semantics* version; every
/// artifact carries its `source` — the argument as given and, for a repo,
/// the immutable commit the facts were read at. A verdict without these
/// is not attributable, and a plan of an earlier schema is refused by
/// [`SystemPlan::parse`] rather than read as unattributed.
pub const PLAN_SCHEMA: u32 = 6;

/// The planner's semantics version.
///
/// Bumped **only** when a verdict can change: a new or corrected rule that
/// makes a checkpoint admissible or blocked where it was not before (the
/// sliding-window normalisation flipped six Qwen3 sizes from three
/// blockers to zero — that is a bump). A fix to the CLI, the report's
/// wording or the JSON layout is not. The package version says which
/// build ran; this says whether its answers are comparable with another
/// build's. Anything caching verdicts keys on (source revision, this).
///
/// `plan/tests/identity.rs` pins fixture verdicts against this value, so
/// a change that flips one fails there until the version is bumped.
///
/// **16** — hyper-connection TRAVERSAL (wave 19). The residual topology's
/// refusal is retired from its one authority
/// (`ResidualTopology::unimplemented_reason`), which the executor's
/// preparation step and this report both read: the decode step (19a)
/// and the batch traversal (19b) carry the bundle, reduce it to one
/// vector at each site and expand the sublayer's output back, and an
/// intermediate-state witness against the reference's oracle fails on
/// every deliberate defect tried. The three topology keys (`hc_mult`,
/// `hc_sinkhorn_iters`, `hc_eps`) leave the unsupported-component table
/// and are carried as execution semantics to the component's residual
/// topology. What remains refused is said by name: a component with the
/// topology and NO `hyper_connection_head` object (GLM-5.3-Flash, `mhc`
/// unexplained) keeps a blocking execution-surface finding, because a
/// whole-stack execution has no declared reduction from the bundle
/// before the final norm — so its count does not move, and a drop there
/// would be capability granted past the head boundary. Forecast before
/// the code (`forecasts/wave19-sinkhorn-traversal.json`, scored per
/// reader): GLM-5.3-Flash 31 -> 31 on this reader and -3 on the key
/// table; DeepSeek-V4 rows -3 each, never reaching this reader.
///
/// **15** — hyper-connection ADDRESSABILITY (wave 18). The six per-layer
/// Sinkhorn site operands (`hc_{attn,ffn}_{fn,base,scale}`) are operand
/// roles, required on every layer of a component that declares the
/// topology, checked against the declared stream count's geometry and
/// bound into the op plan; the head's three bare operands
/// (`hc_head_{fn,base,scale}`) are placed as their own object, and only
/// under the declaration. The op plan therefore no longer refuses on the
/// topology — it runs closure — while the plan report and the executor's
/// preparation step still refuse, through the same
/// `ResidualTopology::unimplemented_reason`, whose text now names the
/// traversal as what is missing rather than the arithmetic (wave 17) or
/// the placement (wave 18). Measured before the code: the two
/// DeepSeek-V4 rows lose their three `hc_head_*` unplaced-group blockers
/// and nothing else moves; GLM-5.3-Flash's execution-surface refusal
/// changes its text and keeps its category; Hy4-preview is untouched
/// because its head is spelled under `model.hc_head` and its topology
/// resolves to none. And a measurement that contradicted the programme's
/// expectation: Kimi-K3's four `*_res_{norm,proj}` operands are
/// `[hidden]` and `[1, hidden]` — not a Sinkhorn site's
/// `[(2 + hc)·hc, hc·hidden]` under any stream count — so they are a
/// different residual topology (AttnRes), not this one's second dialect,
/// and they do not move.
///
/// **14** — hyper-connections are a declared RESIDUAL TOPOLOGY, and
/// explicitly not executable. Read from DeepSeek-V4-Flash's own
/// `inference/model.py`: the state is a bundle of `hc_mult` parallel
/// streams, each sublayer reduces the bundle to one vector and expands
/// its output back, and the weights are computed per token through a
/// projection whose statistics a 20-iteration Sinkhorn split turns into
/// reduce weights, expand weights and a cross-stream combination matrix.
/// `ResidualTopology` states it on the COMPONENT — once the residual
/// means `[.., streams, d]`, the embedding, every branch operator and
/// the head must agree — and a HALF declaration refuses rather than
/// completing itself with one stream. The op plan refuses before reading
/// an operand and the report says so, both through the topology's own
/// `unimplemented_reason`. The component label also stops being
/// family-named: `hyper-connections (GLM-5.x)` appeared verbatim on
/// Tencent and DeepSeek checkpoints, and is now named for the mechanism.
///
/// **13** — LFM2's norm dialect is carried. `operator_norm` and
/// `ffn_norm` are the two-norm PRE-only estate under LFM2's own
/// spelling (`Lfm2DecoderLayer.forward`), and `norm_eps` is its
/// epsilon key. No new execution semantic: the placement is one this
/// build already runs. Registering `lfm2` also stops the identity
/// resolving to `GenericArch`, which was serving Llama-shaped defaults
/// to a stack whose every other layer is a short convolution. Forecast
/// before the code, and deliberately not a GREEN wave: four rows lose
/// three blockers each and NONE clears — the conv mixer's geometry and
/// the `full_attn_idxs` schedule are still absent, and they are
/// execution semantics rather than spellings.
///
/// **12** — three families resolve to their own identities. `olmo2`,
/// `olmo3` and `exaone4` matched no registry entry and fell through to
/// `GenericArch`, which had already chosen PER-HEAD QK norm for OLMo-2 —
/// the wrong reduction for a family whose reference normalises the whole
/// projection. Each entry declares only what its reference establishes:
/// OLMo-2's `QkNormScope::FullProjection` (the operator OLMoE already
/// judges), the 1e-5 `rms_norm_eps` class default both families take, and
/// EXAONE-4's per-head norm applied after the head reshape — its own
/// entry precisely because that one difference is an operator, not a
/// label. Registration resolves a NAME and grants nothing else: a
/// declaration the schema cannot carry still refuses under a registered
/// family. Forecast before the code: three rows clear, four keep the
/// blocker named for each.
///
/// **11** — post-norm placement EXECUTES. Wave 10 could represent it and
/// refused to lower it; the generic executor already applied the wrap
/// norms to each sublayer's OUTPUT before the residual add, and what it
/// could not do was run with NO pre-sublayer norm. Both the batch and the
/// decode path now read the raw residual where the placement says no norm
/// conditions it, and the epsilon QK norm runs at moved off the
/// pre-attention norm's field onto the layer's own `declared_norm_eps` —
/// an epsilon and a placement are unrelated facts, and coupling them is
/// what made this unrepresentable. A post-only stack's single declared
/// epsilon belongs to the post sites, which are the only norm sites it
/// has; a four-norm stack still refuses an unjudged post epsilon, because
/// there the two sites exist and can differ. Forecast before the code:
/// no row clears (identity still blocks all seven), seven rows lose the
/// unsupported-component blocker.
///
/// **10** — a stack may normalise its sublayers' OUTPUT. `NormPlacement`
/// knew two transformer shapes, two-norm and four-norm, and OLMo-2,
/// OLMo-3 and EXAONE-4 declare a third: the sublayer reads the raw
/// residual and its result is normalised before the add
/// (`Olmo2DecoderLayer.forward`, identical in the other two). Their
/// operand estate — both wrap norms, neither pre-norm — matched nothing,
/// so the execution surface refused to build and every probe on those
/// components answered nothing at all. `PostOnly` is recognised from that
/// estate, and the spelling collision is why it is read from which norms
/// EXIST: these families' `post_attention_layernorm` is a true post-norm
/// where a Llama stack's is the pre-FFN norm. The op plan REFUSES it —
/// representable, explicitly not executable, a distinction the closure
/// vocabulary now states in its own defect. Forecast before the code:
/// no row clears, seven surfaces build, sixteen blockers retire because a
/// probe can finally answer, and three answer and still refuse.
///
/// **9** — a wholly-routed family has an FFN, and the always-on shared
/// branch is sized by what the checkpoint declares. The FFN presence rule
/// read only the DENSE width, so `Qwen3_5MoeTextConfig` — which declares
/// no `intermediate_size` at all, because every layer is a routed block —
/// was graded as having no FFN op, and `hidden_act` and
/// `num_experts_per_tok` had nothing to answer to. `FfnSurface`'s dense
/// width becomes optional so that absence is stated rather than written
/// as a zero. Beside it, `shared_expert_intermediate_size` is read (in
/// both declared spellings) and becomes the ONE authority for the shared
/// branch's width: two lineages size it differently and this build was
/// deriving it as `moe_intermediate_size * shared_experts`, which is
/// Kimi's fact and is fourfold wrong on Qwen1.5-MoE. Qwen's gated shared
/// expert — `sigmoid(shared_expert_gate(x)) * shared(x)`, summed with the
/// routed branch — is declared with its own operand, and Qwen3.5-MoE's
/// stacked expert bank is declared as the `PackedBF16` it is. Forecast
/// before the code: exactly five rows clear.
///
/// **8** — two keys read by no implementation, ours or upstream, are
/// read-and-checked rather than graded `Unknown`. Falcon3's
/// `activation: "swiglu"` names the FFN shape (gated, SiLU on the gate) and
/// is judged against the shape the execution surface carries; SmolLM2's
/// `is_llama_config: true` is judged against the family the declared
/// identity resolved to. Same treatment as `use_mrope` / `rope_interleaved`
/// in version 3's wave: never echoed, one value away from a wrong FFN or
/// a wrong family. Forecast before the code: exactly two rows clear.
///
/// **7** — `partial_rotary_factor` is read from inside `rope_parameters`,
/// the transformers-5.x flat form and the only spelling every Qwen3.5
/// checkpoint uses. The parser read the legacy top-level key and Gemma 4's
/// per-layer-type block, so Qwen3.8 (which writes both) resolved while
/// Qwen3.5 lost its fraction: no layer carried one, the partial and
/// multi-axis rotary probes answered nothing, and three leaves refused a
/// family whose text path this build executes. Precedence now mirrors
/// `standardize_rope_params`: top level, then per-type block, then flat
/// block. Forecast before the code: three Qwen3.5 dense rows admissible,
/// three Qwen3.5 MoE rows from six blockers to three, nothing else moves.
///
/// **6** — a declaration a companion switch turns off is inert. Qwen2.5
/// ships `sliding_window: 32768` beside `use_sliding_window: false`; the
/// graph carries no window, which agrees with the checkpoint, and the
/// carriage rule was reporting that agreement as a dropped fact. The
/// companion is read at the same nesting level, so one component's switch
/// cannot silence another's window.
///
/// **5** — MoE routing. A key declared with no value states that its
/// subject does not apply and no longer demands a home (Gemma 4's dense
/// sizes declare `top_k_experts: null`), and Qwen's expert schedule
/// (`decoder_sparse_step`, `mlp_only_layers`) is judged against its
/// value: inert at the uniform all-MoE stack, blocking for any real
/// per-layer topology.
///
/// **4** — Llama-3 wavelength-band rope scaling is represented.
/// `PositionPolicy::Llama3` carries the block, so a checkpoint declaring
/// `rope_type: "llama3"` is admissible where it used to be refused. Not
/// new mathematics: `larql-compute` has always implemented the family,
/// and the gap was that the container had nowhere to say so.
///
/// Deliberately NOT bumped for plan schema 6: findings gained an `id` and
/// a `cluster`, and each capability now names its blockers, but no verdict
/// moved. The schema says what the document contains; this says whether
/// its answers are comparable. An instrumentation change that shifted this
/// number would make every stored verdict falsely incomparable.
///
/// **3** — decoding-policy and dropout defaults stopped blocking. They
/// are preserved as declared facts and classified for what they are
/// ([`SemanticClass::GenerationPolicy`], [`SemanticClass::TrainingOnly`])
/// instead of grading `Unknown`, and `pretraining_tp` is judged against
/// its VALUE, because HF Llama's forward pass reads it above 1.
///
/// **2** — the architecture-identity gate. A `model_type` no registry
/// entry matches, and a container/text pair that resolve to different
/// architectures, now block instead of passing silently into
/// `GenericArch`'s Llama-shaped defaults. Measured on the conformance
/// corpus: 15 of 42 declared `model_type` strings, across 30 checkpoints.
pub const PLANNER_SEMANTICS_VERSION: u32 = 16;

/// Who judged a plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannerIdentity {
    /// The crate that implements the planner.
    pub package: String,
    /// That crate's package version — which build ran.
    pub package_version: String,
    /// [`PLANNER_SEMANTICS_VERSION`] at the time — whether two verdicts
    /// are comparable.
    pub semantics_version: u32,
}

impl PlannerIdentity {
    /// This build's identity.
    pub fn current() -> Self {
        Self {
            package: env!("CARGO_PKG_NAME").to_string(),
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            semantics_version: PLANNER_SEMANTICS_VERSION,
        }
    }
}

/// What an artifact's facts were read from, so the verdict names its
/// subject.
///
/// **Cache authority is pinned-only.** A persisted verdict cache requires
/// an immutable source revision — [`revision`](Self::revision), a commit.
/// [`unpinned_revision`](Self::unpinned_revision) is provenance, not a
/// cache identity: `main` on Monday and `main` on Tuesday can name
/// different facts, so a verdict over it may be shown, visibly marked
/// unpinned, and must never be stored as authority.
/// [`SystemPlan::cache_key`] enforces this for the whole plan.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactSource {
    /// The artifact as the caller gave it: a checkpoint directory, a saved
    /// inventory, or an `hf://org/name[@revision]` spec.
    pub path: String,
    /// The immutable revision the facts were read at, when the source has
    /// one — a repo's commit. A verdict is re-usable exactly as far as
    /// this and the planner's semantics version are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// A revision *name* the source fell back to because the hub named no
    /// commit — provenance that can move, and worth saying so. Never a
    /// cache identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unpinned_revision: Option<String>,
}

impl ArtifactSource {
    /// A local source: the path the inventory recorded, no revision.
    pub fn local(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            revision: None,
            unpinned_revision: None,
        }
    }

    /// The immutable revision a cache may key on, if this source has one.
    /// `None` for a local path and for an unpinned revision name.
    pub fn pinned_revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }
}

/// The identity under which a plan's verdict may be persisted: every
/// artifact's immutable revision, in artifact order, and the semantics
/// version that judged them. Exists only when every artifact is pinned.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerdictCacheKey {
    pub revisions: Vec<String>,
    pub semantics_version: u32,
}

/// The whole-system plan over every artifact given.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPlan {
    /// Always [`PLAN_SCHEMA`].
    pub schema: u32,
    /// Who judged this plan. Not defaulted on deserialisation: a plan
    /// without it is an earlier schema and is refused, not read as
    /// unattributed.
    pub planner: PlannerIdentity,
    /// One entry per inspected artifact, in input order.
    pub artifacts: Vec<ArtifactPlan>,
    /// Cross-component interfaces resolved into graph edges.
    pub interfaces: Vec<InterfacePlan>,
    /// True iff no blocking finding exists anywhere in the system.
    ///
    /// **Model completeness**, not executability. A container can be
    /// inadmissible here and still run text generation perfectly — see
    /// [`capabilities`](Self::capabilities).
    pub admissible: bool,
    /// Per-capability execution admissibility, each judged on its own
    /// dependency closure.
    ///
    /// Defaulted on deserialisation so plans written before capability
    /// scoping existed still read, and skipped when empty so those plans
    /// serialise byte-identically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<super::capability::CapabilityStatus>,
    pub summary: PlanSummary,
    /// The system graph the builder produced — what G3 encodes.
    pub graph: SystemGraph,
}

/// Counts over every finding in the system.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSummary {
    pub representable: usize,
    pub mismatched: usize,
    pub unrepresented: usize,
    pub interfaces: usize,
    /// Findings that make the plan inadmissible (see [`Finding::blocks`]).
    pub blocking: usize,
}

/// The plan for one physical artifact (one inventory).
///
/// Deliberately named *artifact*, not *component*: the Glimmer target
/// checkpoint physically carries both the text model and the vision tower.
/// Logical components live inside findings' `component` field; the physical
/// file boundary is never authoritative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactPlan {
    /// Artifact name (the inspected directory's stem).
    pub name: String,
    /// What this artifact's facts were read from.
    pub source: ArtifactSource,
    pub model_type: String,
    pub findings: Vec<PlannedFinding>,
}

impl SystemPlan {
    /// The identity under which this verdict may be cached, or `None`.
    ///
    /// A verdict is cacheable only when every artifact carries an
    /// immutable revision and the planner's semantics version is known —
    /// the same facts under the same rules. One local path or one
    /// unpinned revision name among the artifacts makes the whole plan
    /// uncacheable: it may still be shown, marked as such, but a cache
    /// that stored it would answer tomorrow's question with today's
    /// facts.
    pub fn cache_key(&self) -> Option<VerdictCacheKey> {
        let revisions = self
            .artifacts
            .iter()
            .map(|a| a.source.pinned_revision().map(str::to_string))
            .collect::<Option<Vec<String>>>()?;
        Some(VerdictCacheKey {
            revisions,
            semantics_version: self.planner.semantics_version,
        })
    }

    /// Read a plan back, refusing one written by another schema by name.
    ///
    /// The schema is checked before the body is parsed, so an older plan
    /// fails as "schema 3, this build reads 4" rather than as a missing
    /// field several lines in — and a plan that predates planner identity
    /// can never be read as a verdict nobody attributed.
    pub fn parse(json: &str) -> Result<Self, VindexError> {
        #[derive(Deserialize)]
        struct SchemaProbe {
            schema: Option<u32>,
        }
        let probe: SchemaProbe = serde_json::from_str(json)
            .map_err(|e| VindexError::Parse(format!("plan is not a JSON object: {e}")))?;
        match probe.schema {
            Some(found) if found == PLAN_SCHEMA => {}
            Some(found) => {
                return Err(VindexError::Parse(format!(
                    "plan schema {found}; this build reads plan schema {PLAN_SCHEMA} — \
                     re-run `vindex plan` so the verdict is attributed"
                )))
            }
            None => {
                return Err(VindexError::Parse(format!(
                    "plan declares no schema; this build reads plan schema {PLAN_SCHEMA}"
                )))
            }
        }
        serde_json::from_str(json).map_err(|e| VindexError::Parse(format!("parse plan: {e}")))
    }
}

/// A finding's identity within one plan document.
///
/// Assigned by the document, not by the rule that raised the finding: a
/// rule states something about a subject and has no view on where that
/// statement sits among the others. Stable for the life of a plan JSON,
/// which is all a capability closure needs to point at its blockers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FindingId(pub usize);

/// A finding as it appears in a plan document: what the rule stated, plus
/// what the document assigns to it.
///
/// The split is deliberate. [`Finding`] is a rule's output and carries
/// only facts the rule established. `id` and `cluster` are *derived*, in
/// one place, at assembly — so a new finding-raising rule cannot forget
/// to classify itself, and two rules cannot disagree about which concept
/// a subject belongs to. Thirty construction sites setting a cluster by
/// hand is thirty chances to drift.
///
/// Serialises flat: a reader sees `id` and `cluster` beside `subject` and
/// `class`, with no nesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedFinding {
    pub id: FindingId,
    /// The model concept this finding is about.
    pub cluster: SemanticCluster,
    #[serde(flatten)]
    pub finding: Finding,
}

impl std::ops::Deref for PlannedFinding {
    type Target = Finding;
    fn deref(&self) -> &Finding {
        &self.finding
    }
}

impl PlannedFinding {
    /// Wrap a rule's finding, deriving everything the document assigns.
    pub fn assign(id: usize, finding: Finding) -> Self {
        Self {
            id: FindingId(id),
            cluster: super::semantics::cluster_for(&finding.subject),
            finding,
        }
    }
}

/// One statement about one subject.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub category: FindingCategory,
    pub class: SemanticClass,
    /// Logical component the subject belongs to (`text`, `vision`, `root`).
    pub component: String,
    /// What the finding is about: a config key path, a tensor group, or a
    /// topology aspect.
    pub subject: String,
    /// Value the checkpoint declares, when the finding compares values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared: Option<serde_json::Value>,
    /// Value this build resolves, when the finding compares values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved: Option<serde_json::Value>,
    /// How far VINDEX3 carries the subject past the parser, for findings
    /// about a config key. `None` for findings about tensors, topology or
    /// interfaces, where carriage is not the question being asked.
    ///
    /// This is the field that keeps `consumed` from being read as
    /// `represented`: a key can be parsed and go no further, and the plan
    /// now says which.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub carriage: Option<Carriage>,
    pub detail: String,
}

impl Finding {
    /// Whether this finding makes the plan inadmissible.
    ///
    /// Representable findings never block. Interface findings always carry
    /// [`SemanticClass::InterfaceSemantic`], so the class test covers them.
    pub fn blocks(&self) -> bool {
        !matches!(self.category, FindingCategory::Representable) && self.class.is_critical()
    }
}

/// What kind of statement a finding makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FindingCategory {
    /// The schema has a faithful home for this subject.
    Representable,
    /// Declared and resolved authorities disagree on the value.
    Mismatched,
    /// The subject has no home in the current schema.
    Unrepresented,
    /// A declared cross-component dependency.
    Interface,
}

/// How much losing or corrupting the subject would matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticClass {
    /// Reviewed and safe to drop (registry-listed, justified per entry).
    IgnoredSafe,
    /// Identity/training facts inert for a forward pass.
    MetadataOnly,
    /// Declared for training and inert at inference — an auxiliary-loss
    /// coefficient, a router-logit output switch. Distinct from
    /// [`Self::MetadataOnly`]: these describe the *forward pass of
    /// training*, so the reason they are safe to drop is that inference
    /// does not run that path, not that they are identity strings.
    TrainingOnly,
    /// A decoding-policy default that ships in `config.json` for
    /// historical reasons — sampling temperature, beam count, banned
    /// tokens, what `generate()` returns.
    ///
    /// Its own class rather than [`Self::TrainingOnly`] or
    /// [`Self::MetadataOnly`] because it is neither: it is read at
    /// *inference*, by the decode loop, and it is not an identity fact.
    /// What makes it inert **for VINDEX3** is narrower and worth stating
    /// exactly — a container represents the model's computation, and none
    /// of these change what a forward pass computes; they select among its
    /// outputs. A caller sets them per request, and a checkpoint's values
    /// are a suggested default rather than a property of the weights.
    ///
    /// Measured on the conformance corpus: 40 such subjects across 8
    /// checkpoints graded `Unknown` and therefore blocked — GPT-2 was
    /// refused partly over
    /// `task_specific_params.text-generation.do_sample`.
    GenerationPolicy,
    /// A redundant spelling of a fact the checkpoint declares elsewhere
    /// and a parser reads. Safe only while the canonical key is present
    /// and agrees — the registry names it and the gate checks both, so
    /// `alias` cannot become a way to silence a key.
    Alias,
    /// Changes what a forward pass computes (norm eps, rope, scales…).
    ExecutionSemantic,
    /// Describes stored operands (shapes, widths, component topology).
    TensorSemantic,
    /// Declares a cross-component contract (taps, token protocol).
    InterfaceSemantic,
    /// A model component has been **positively identified** from the
    /// checkpoint semantics, and this build has no implementation for it.
    ///
    /// Blocking, exactly like [`Self::Unknown`] — the difference is not
    /// severity but knowledge:
    ///
    /// ```text
    /// Unknown               nobody has established what this means yet
    /// UnsupportedComponent  we know what component this configures,
    ///                       and this build does not implement it
    /// ```
    ///
    /// which is the difference between uncertainty and known engineering
    /// work. Nine keys naming one absent component is one job; reporting
    /// them as nine anonymous unknowns says nothing about how much.
    ///
    /// Deliberately distinct from every neighbouring idea, because the
    /// engineering implication differs in each case. This is **not** a
    /// parser chore, a spelling alias, a checked default, or an inactive
    /// declaration. It means *there is machinery missing*.
    ///
    /// **The registration rule is positive evidence of component
    /// ownership, never plausible adjacency** — see
    /// [`UNSUPPORTED_COMPONENT_KEYS`](super::semantics::UNSUPPORTED_COMPONENT_KEYS).
    /// A prefix match or a regex is a discovery tool and must not become
    /// the authority; that is precisely how `indexer_rope_interleave`
    /// ended up filed under general RoPE, where acting on it would have
    /// re-paired the whole model's rotary.
    UnsupportedComponent,
    /// Not in the registry — nobody has judged it, so it blocks.
    Unknown,
}

impl SemanticClass {
    /// Classes that make a non-representable finding blocking.
    pub fn is_critical(self) -> bool {
        matches!(
            self,
            Self::ExecutionSemantic
                | Self::TensorSemantic
                | Self::InterfaceSemantic
                | Self::UnsupportedComponent
                | Self::Unknown
        )
    }
}

/// A declared cross-component interface, resolved into a graph edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfacePlan {
    /// Producing component (graph id, e.g. `target`).
    pub producer_component: String,
    /// Tapped layers, in declaration order.
    pub producer_layers: Vec<usize>,
    /// Consuming component (graph id, e.g. `draft`).
    pub consumer_component: String,
    /// Logical object implementing consumption (`draft.feature_projector`).
    pub consumer_object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representable_findings_never_block() {
        let finding = Finding {
            category: FindingCategory::Representable,
            class: SemanticClass::ExecutionSemantic,
            component: "text".into(),
            subject: "x".into(),
            declared: None,
            resolved: None,
            carriage: None,
            detail: String::new(),
        };
        assert!(!finding.blocks());
    }

    #[test]
    fn critical_classes_block_outside_representable() {
        for class in [
            SemanticClass::ExecutionSemantic,
            SemanticClass::TensorSemantic,
            SemanticClass::InterfaceSemantic,
            SemanticClass::Unknown,
        ] {
            let finding = Finding {
                category: FindingCategory::Unrepresented,
                class,
                component: "text".into(),
                subject: "x".into(),
                declared: None,
                resolved: None,
                carriage: None,
                detail: String::new(),
            };
            assert!(finding.blocks(), "{class:?} must block");
        }
    }

    #[test]
    fn benign_classes_do_not_block() {
        for class in [SemanticClass::IgnoredSafe, SemanticClass::MetadataOnly] {
            let finding = Finding {
                category: FindingCategory::Unrepresented,
                class,
                component: "root".into(),
                subject: "x".into(),
                declared: None,
                resolved: None,
                carriage: None,
                detail: String::new(),
            };
            assert!(!finding.blocks(), "{class:?} must not block");
        }
    }

    #[test]
    fn classes_serialise_snake_case() {
        assert_eq!(
            serde_json::to_string(&SemanticClass::ExecutionSemantic).unwrap(),
            "\"execution_semantic\""
        );
        assert_eq!(
            serde_json::to_string(&FindingCategory::Unrepresented).unwrap(),
            "\"unrepresented\""
        );
    }
}
