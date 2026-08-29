/// LQL Abstract Syntax Tree
///
/// Every LQL statement parses into one `Statement` variant.
/// The executor dispatches each variant to the appropriate backend.

#[derive(Debug, Clone)]
pub enum Statement {
    // ── Lifecycle ──
    Extract {
        model: String,
        output: String,
        components: Option<Vec<Component>>,
        layers: Option<Range>,
        extract_level: ExtractLevel,
        /// `FORMAT VINDEX2 | VINDEX3` — which container generation to
        /// write. `None` = no preference; the executor resolves it
        /// through the vindex crate's single extraction-generation
        /// policy, never a parser- or surface-local default.
        format: Option<ExtractFormat>,
    },
    Compile {
        vindex: VindexRef,
        output: String,
        format: Option<OutputFormat>,
        target: CompileTarget,
        /// COMPILE INTO VINDEX only: how to resolve patches that touch the
        /// same (layer, feature) slot. None → default (LastWins).
        on_conflict: Option<CompileConflict>,
    },
    Diff {
        a: VindexRef,
        b: VindexRef,
        layer: Option<u32>,
        relation: Option<String>,
        limit: Option<u32>,
        into_patch: Option<String>,
        /// `DIFF … PHYSICAL` — subordinate segment-level report
        /// (VINDEX3): hashes and linked/rewritten status instead of
        /// the logical model-fact diff.
        physical: bool,
    },
    Use {
        target: UseTarget,
    },

    // ── Query ──
    Walk {
        prompt: String,
        top: Option<u32>,
        layers: Option<Range>,
        mode: Option<WalkMode>,
        compare: bool,
    },
    /// Full inference with attention — requires model weights.
    Infer {
        prompt: String,
        top: Option<u32>,
        compare: bool,
        /// FR1/FR2 KnnStore router selection (`ROUTE VERIFY [FALLBACK] [TOPK n]`).
        /// `None` = inherit the env default (`KnnRouteMode::from_env`).
        route: Option<InferRoute>,
        /// `GENERATE n` — greedy autoregressive continuation through the
        /// bound runtime (VINDEX3 backends; LQL-1). `None` = classic
        /// single-step top-k prediction.
        generate: Option<u32>,
    },
    Select {
        source: SelectSource,
        fields: Vec<Field>,
        conditions: Vec<Condition>,
        nearest: Option<NearestClause>,
        order: Option<OrderBy>,
        limit: Option<u32>,
    },
    Describe {
        entity: String,
        band: Option<LayerBand>,
        layer: Option<u32>,
        relations_only: bool,
        mode: DescribeMode,
    },
    Explain {
        prompt: String,
        mode: ExplainMode,
        layers: Option<Range>,
        band: Option<LayerBand>,
        verbose: bool,
        top: Option<u32>,
        relations_only: bool,
        with_attention: bool,
    },

    // ── Mutation ──
    Insert {
        entity: String,
        relation: String,
        target: String,
        layer: Option<u32>,
        confidence: Option<f32>,
        /// Per-layer down-vector multiplier for the install
        /// (`d_ref * alpha_mul`). Default 0.1 — matches the validated
        /// Python `install_compiled_slot` pipeline (exp 14). Larger
        /// values push the inserted fact harder but dilute neighbours;
        /// smaller values reduce neighbour degradation at the cost of
        /// new-fact confidence. Validated range ~0.05–0.30.
        /// Only used when `mode = Compose`.
        alpha: Option<f32>,
        /// Install mode. `Knn` (default) stores an independent residual
        /// key in the KnnStore — scales freely, no cross-fact interference.
        /// `Compose` uses `install_compiled_slot` to write gate/up/down
        /// overlays — features participate in the forward pass and can
        /// chain for multi-hop, but has a Hopfield-style cap at
        /// ~N=5-10 per layer under template-shared prompts.
        mode: InsertMode,
    },
    Delete {
        conditions: Vec<Condition>,
    },
    Update {
        set: Vec<Assignment>,
        conditions: Vec<Condition>,
    },
    Merge {
        source: String,
        target: Option<String>,
        conflict: Option<ConflictStrategy>,
    },
    /// Global rebalance pass — iterates every compose-mode installed
    /// fact in the session, INFERs its canonical, adjusts its
    /// down_vector up or down until its target probability lands in
    /// the `[floor, ceiling]` band. Fixed-point loop capped at
    /// `max_iters`. Used at end of batch install or before COMPILE
    /// to push compose retrieval past the per-INSERT greedy ceiling.
    ///
    /// `REBALANCE [UNTIL CONVERGED] [MAX <n>] [FLOOR <p>] [CEILING <p>]`
    Rebalance {
        max_iters: Option<u32>,
        floor: Option<f32>,
        ceiling: Option<f32>,
    },

    // ── Introspection ──
    ShowRelations {
        layer: Option<u32>,
        with_examples: bool,
        mode: DescribeMode,
    },
    ShowLayers {
        range: Option<Range>,
    },
    ShowFeatures {
        layer: u32,
        conditions: Vec<Condition>,
        limit: Option<u32>,
    },
    ShowEntities {
        layer: Option<u32>,
        limit: Option<u32>,
    },
    ShowModels,
    /// `SHOW COMPONENTS` (VINDEX3): the system graph's components, as
    /// the container declares them — never reconstructed from names.
    ShowComponents,
    /// `SHOW REPRESENTATIONS ["object"]` (VINDEX3): the physically
    /// present representation directory, optionally filtered to entries
    /// whose logical object matches the given substring.
    ShowRepresentations {
        object: Option<String>,
    },
    /// `SHOW PROVENANCE ["object"]` (VINDEX3): hashes and lineage per
    /// directory entry — payload and segment digests, what an entry was
    /// compiled from, what model the container derives from.
    ShowProvenance {
        object: Option<String>,
    },
    /// `SHOW AUTHORITY` (VINDEX3): the container's own authority
    /// declaration and the profiles it declares by name.
    ShowAuthority,
    Stats {
        vindex: Option<String>,
    },
    ShowCompactStatus,
    /// `COMPACT INTO VINDEX "out"` (VINDEX3): semantics-preserving
    /// physical reorganisation — dead files dropped, referenced
    /// segments carried byte-identically. DIFF is its proof
    /// instrument: SemanticDiff(input, output) must be empty.
    CompactInto {
        output: String,
    },
    CompactMinor,
    CompactMajor {
        full: bool,
        lambda: Option<f32>,
    },

    // ── Patch ──
    BeginPatch {
        path: String,
    },
    SavePatch,
    ApplyPatch {
        path: String,
    },
    ShowPatches,
    RemovePatch {
        path: String,
    },

    // ── Trace ──
    /// Residual stream trace — decomposed forward pass.
    Trace {
        prompt: String,
        /// Token to track through the trace (the `FOR <token>` clause).
        answer: Option<String>,
        decompose: bool,
        layers: Option<Range>,
        positions: Option<TracePositionMode>,
        save: Option<String>,
    },

    // ── Pipe ──
    Pipe {
        left: Box<Statement>,
        right: Box<Statement>,
    },
}

#[derive(Debug, Clone)]
pub enum VindexRef {
    Path(String),
    Current,
}

#[derive(Debug, Clone)]
pub enum UseTarget {
    Vindex(String),
    Model { id: String, auto_extract: bool },
    Remote(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainMode {
    /// EXPLAIN WALK — pure vindex gate KNN, no attention
    Walk,
    /// EXPLAIN INFER — full inference with feature trace
    Infer,
}

/// INFER `ROUTE` clause — selects the FR1/FR2 KnnStore router for this
/// statement. Presence implies the FR1 verified router (top-k + entity-in-prompt
/// verify + abstain); `fallback` adds the FR2 activation alias fallback.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InferRoute {
    /// `ROUTE VERIFY FALLBACK` → two-tier (verify then activation fallback).
    pub fallback: bool,
    /// `TOPK n` — candidate count the verifier considers (None = default 5).
    pub topk: Option<u32>,
    /// `EXIT` → retrieval-augmented early exit: when the verified router fires,
    /// short-circuit the forward at the resolved layer (skip the remaining
    /// layers + lm_head). Verified-only — ignored when `fallback` is set, since
    /// the FR2 fallback needs the full forward to stay parity-identical.
    pub exit: bool,
}

/// Display mode for DESCRIBE and SHOW RELATIONS output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DescribeMode {
    /// Full detail: relation labels, also-tokens, layer ranges, multi-layer hits.
    Verbose,
    /// Default. Compact: top edges only, primary layer, no also-tokens.
    #[default]
    Brief,
    /// No probe labels — pure model signal.
    Raw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerBand {
    /// L0-13: morphological, syntactic, code structure
    Syntax,
    /// L14-27: semantic/factual features (default for DESCRIBE)
    Knowledge,
    /// L28-33: formatting/output features
    Output,
    /// All layers: syntax + knowledge + output
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractLevel {
    /// Default: gate + embed + down_meta (~3 GB f16)
    Browse,
    /// + attention weights (~6 GB f16), enables INFER
    Inference,
    /// + up, norms, lm_head (~10 GB f16), enables COMPILE
    All,
}

/// `EXTRACT ... FORMAT <generation>` — an explicit container-generation
/// request. Absence means "no preference", which is NOT the same value:
/// the default lives in one policy site in the vindex crate, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractFormat {
    Vindex2,
    Vindex3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileTarget {
    /// COMPILE ... INTO MODEL — produce safetensors/gguf
    Model,
    /// COMPILE ... INTO VINDEX — bake patches into clean vindex
    Vindex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkMode {
    Hybrid,
    Pure,
    Dense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Safetensors,
    Gguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InsertMode {
    /// Architecture B — residual key stored in KnnStore. Independent
    /// per-fact entry, no cross-fact interference, scales freely.
    /// Default: this is what you want for knowledge databases.
    #[default]
    Knn,
    /// FFN-overlay — gate/up/down installed via install_compiled_slot.
    /// Features participate in the forward pass and chain for multi-hop,
    /// but have a cap at ~N=5-10 per layer under template-shared prompts.
    /// Use for research demos and multi-hop composition.
    Compose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    KeepSource,
    KeepTarget,
    HighestConfidence,
}

/// Conflict resolution for COMPILE INTO VINDEX when multiple patches touch
/// the same (layer, feature) slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileConflict {
    /// Last patch in the apply order wins (default).
    LastWins,
    /// The patch with the highest confidence on that slot wins.
    HighestConfidence,
    /// Abort the compile if any slot has conflicting writes.
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    FfnGate,
    FfnDown,
    FfnUp,
    Embeddings,
    AttnOv,
    AttnQk,
}

#[derive(Debug, Clone)]
pub struct Range {
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectSource {
    Edges,
    Features,
    Entities,
}

#[derive(Debug, Clone)]
pub enum Field {
    Star,
    Named(String),
}

#[derive(Debug, Clone)]
pub struct Condition {
    pub field: String,
    pub op: CompareOp,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub enum CompareOp {
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
    Like,
    In,
}

#[derive(Debug, Clone)]
pub enum Value {
    String(String),
    Number(f64),
    Integer(i64),
    List(Vec<Value>),
}

#[derive(Debug, Clone)]
pub struct NearestClause {
    pub entity: String,
    pub layer: u32,
}

#[derive(Debug, Clone)]
pub struct OrderBy {
    pub field: String,
    pub descending: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePositionMode {
    Last,
    All,
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub field: String,
    pub value: Value,
}
