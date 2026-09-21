//! ATTR-1D descriptive support identities and prompt-role mapping.
//!
//! This module implements the identity boundary frozen in
//! `docs/v3-attr-1d-descriptive-support.md`:
//!
//! - [`SupportCoordinate`] names a stable physical location and contains no
//!   execution, backend, measurement, semantic-transition, or causal fields;
//! - [`SupportObservationIdentity`] binds that coordinate to one exact
//!   observed execution; and
//! - [`PromptRoleMap`] assigns source roles from construction-time token spans
//!   only, refusing ambiguous or out-of-range inputs.
//!
//! The numerical contribution transform consumes these identities.  It does
//! not get to redefine them after observing a contribution or intervention.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

/// Frozen schema for a stable physical support coordinate.
pub const SUPPORT_COORDINATE_SCHEMA: &str = "larql.attr1d.support-coordinate.v1";
/// Frozen schema for an execution-specific observation binding.
pub const SUPPORT_OBSERVATION_SCHEMA: &str = "larql.attr1d.support-observation.v1";
/// Schema for the construction-time source-role manifest.
pub const PROMPT_ROLE_MAP_SCHEMA: &str = "larql.attr1d.prompt-role-map.v1";
/// Schema for the lossless same-run source evidence consumed by ATTR-1D.
pub const SOURCE_EVIDENCE_SCHEMA: &str = "larql.attr1d.source-evidence.v1";
/// Frozen ATTR-1D contract identity consumed by descriptive records.
pub const ATTR1D_CONTRACT_IDENTITY: &str =
    "sha256:4f48973807db0695d5f30b83a00c73637afbd0dc77d116b5a1f9eb6ec913431c";

/// Frozen numerical acceptance thresholds.
pub const HEAD_SUM_MAX_RELATIVE_L2: f64 = 1e-4;
pub const SOURCE_SPLIT_MAX_RELATIVE_L2: f64 = 1e-5;
pub const PROJECTION_RECONSTRUCTION_SCALE: f64 = 1e-6;
pub const ATTENTION_PROBABILITY_MAX_ABSOLUTE_ERROR: f64 = 1e-6;
pub const VECTOR_DENOMINATOR_FLOOR: f64 = 1e-12;

/// Errors are refusals: callers must not repair an invalid identity or role
/// map using observed model behaviour.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AttributionIdentityError {
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("expected schema {expected}, found {actual}")]
    WrongSchema {
        expected: &'static str,
        actual: String,
    },
    #[error("token span [{start}, {end}) is empty or reversed")]
    InvalidSpan { start: usize, end: usize },
    #[error("token span [{start}, {end}) exceeds prompt length {prompt_len}")]
    SpanOutsidePrompt {
        start: usize,
        end: usize,
        prompt_len: usize,
    },
    #[error(
        "declared {left_role:?} span [{left_start}, {left_end}) overlaps {right_role:?} span [{right_start}, {right_end})"
    )]
    OverlappingRoleSpans {
        left_role: SourceRole,
        left_start: usize,
        left_end: usize,
        right_role: SourceRole,
        right_start: usize,
        right_end: usize,
    },
    #[error("final prompt position {position} is outside prompt length {prompt_len}")]
    FinalPromptPositionOutsidePrompt { position: usize, prompt_len: usize },
    #[error("source position {position} is outside visible range [{start}, {end})")]
    SourcePositionOutsideVisibleRange {
        position: usize,
        start: usize,
        end: usize,
    },
    #[error("source coordinates must be one-token spans, found [{start}, {end})")]
    SourceCoordinateNotOneToken { start: usize, end: usize },
    #[error("{field} does not match its recomputed identity")]
    IdentityBindingMismatch { field: &'static str },
    #[error(
        "visible source range [{start}, {end}) does not end after observed position {position}"
    )]
    SourceRangePositionMismatch {
        start: usize,
        end: usize,
        position: usize,
    },
    #[error("identity JSON serialization failed: {0}")]
    Serialization(String),
}

/// Refusals raised by the deterministic numerical transform.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DescriptiveTransformError {
    #[error(transparent)]
    Identity(#[from] AttributionIdentityError),
    #[error("{field} width {actual} does not match expected width {expected}")]
    Width {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("expected {expected} query heads, observed {actual}")]
    HeadCoverage { expected: usize, actual: usize },
    #[error("query head {head} was observed more than once")]
    DuplicateHead { head: usize },
    #[error("query-head coverage is not the exact range 0..{expected}: {observed:?}")]
    NonCanonicalHeadCoverage {
        expected: usize,
        observed: Vec<usize>,
    },
    #[error("head {head} source position {position} was observed more than once")]
    DuplicateSource { head: usize, position: usize },
    #[error(
        "head {head} source coverage does not match visible range [{start}, {end}): {observed:?}"
    )]
    SourceCoverage {
        head: usize,
        start: usize,
        end: usize,
        observed: Vec<usize>,
    },
    #[error("{field} contains a non-finite scalar")]
    NonFinite { field: &'static str },
    #[error("post-attention RMS epsilon must be finite and positive, found {epsilon}")]
    InvalidRmsEpsilon { epsilon: f64 },
    #[error("prepared head projection refused: {0}")]
    Projection(String),
}

/// The physical context shared by site, bias, head and source coordinates.
///
/// These identities describe the logical prepared program and its input
/// layout.  In particular, they do not name a backend realization or run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportCoordinateContext {
    pub model_system_identity: String,
    pub logical_plan_identity: String,
    pub component_identity: String,
    pub input_layout_identity: String,
    pub position: usize,
    pub layer: usize,
}

impl SupportCoordinateContext {
    /// Construct and validate the backend-neutral coordinate context.
    pub fn new(
        model_system_identity: impl Into<String>,
        logical_plan_identity: impl Into<String>,
        component_identity: impl Into<String>,
        input_layout_identity: impl Into<String>,
        position: usize,
        layer: usize,
    ) -> Result<Self, AttributionIdentityError> {
        let context = Self {
            model_system_identity: model_system_identity.into(),
            logical_plan_identity: logical_plan_identity.into(),
            component_identity: component_identity.into(),
            input_layout_identity: input_layout_identity.into(),
            position,
            layer,
        };
        context.validate()?;
        Ok(context)
    }

    fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_nonempty("model_system_identity", &self.model_system_identity)?;
        require_nonempty("logical_plan_identity", &self.logical_plan_identity)?;
        require_nonempty("component_identity", &self.component_identity)?;
        require_nonempty("input_layout_identity", &self.input_layout_identity)
    }
}

/// ATTR-1D v1 observes attention support.  FFN support gets its own frozen
/// algebra before this enum is extended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportOperator {
    Attention,
}

/// Physical observation site.  Kept separate from [`SupportOperator`] because
/// later operator contracts may admit more than one site per operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportSite {
    Attention,
}

/// Source roles frozen from prompt construction, never inferred from decoded
/// text or from the model's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SourceRole {
    Bos,
    Relation,
    Entity,
    Last,
    Other,
}

/// The typed physical address below a coordinate context.
///
/// The internally tagged representation serializes the frozen flat
/// `coordinate_kind` shape rather than an implementation-specific nested
/// object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "coordinate_kind", rename_all = "snake_case")]
pub enum SupportAddress {
    Site,
    Bias,
    Head {
        query_head: usize,
        kv_head: usize,
    },
    Source {
        query_head: usize,
        kv_head: usize,
        source_start: usize,
        source_end: usize,
        source_role: SourceRole,
        role_map_identity: String,
    },
}

impl SupportAddress {
    fn validate(&self) -> Result<(), AttributionIdentityError> {
        if let Self::Source {
            source_start,
            source_end,
            role_map_identity,
            ..
        } = self
        {
            if source_start.checked_add(1) != Some(*source_end) {
                return Err(AttributionIdentityError::SourceCoordinateNotOneToken {
                    start: *source_start,
                    end: *source_end,
                });
            }
            require_nonempty("role_map_identity", role_map_identity)?;
        }
        Ok(())
    }
}

/// Stable, backend- and execution-neutral physical support identity.
///
/// Measurements and observation provenance intentionally cannot be expressed
/// by this type.  Deserialised coordinates are validated before hashing, so a
/// wrong schema or malformed source span cannot acquire a lawful ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportCoordinate {
    schema: String,
    #[serde(flatten)]
    pub context: SupportCoordinateContext,
    pub operator: SupportOperator,
    pub site: SupportSite,
    #[serde(flatten)]
    pub address: SupportAddress,
}

impl SupportCoordinate {
    fn new(
        context: SupportCoordinateContext,
        address: SupportAddress,
    ) -> Result<Self, AttributionIdentityError> {
        let coordinate = Self {
            schema: SUPPORT_COORDINATE_SCHEMA.to_string(),
            context,
            operator: SupportOperator::Attention,
            site: SupportSite::Attention,
            address,
        };
        coordinate.validate()?;
        Ok(coordinate)
    }

    /// Construct the whole attention-site coordinate.
    pub fn site(context: SupportCoordinateContext) -> Result<Self, AttributionIdentityError> {
        Self::new(context, SupportAddress::Site)
    }

    /// Construct the once-only output-bias coordinate.
    pub fn bias(context: SupportCoordinateContext) -> Result<Self, AttributionIdentityError> {
        Self::new(context, SupportAddress::Bias)
    }

    /// Construct one query-head coordinate.
    pub fn head(
        context: SupportCoordinateContext,
        query_head: usize,
        kv_head: usize,
    ) -> Result<Self, AttributionIdentityError> {
        Self::new(
            context,
            SupportAddress::Head {
                query_head,
                kv_head,
            },
        )
    }

    /// Construct an exact one-token source coordinate.  The role and role-map
    /// identity are derived here so callers cannot hand-label a source after
    /// looking at its contribution.
    pub fn source(
        context: SupportCoordinateContext,
        query_head: usize,
        kv_head: usize,
        source_position: usize,
        role_map: &PromptRoleMap,
        visible_source_range: TokenSpan,
    ) -> Result<Self, AttributionIdentityError> {
        let source_end = source_position.checked_add(1).ok_or(
            AttributionIdentityError::SourceCoordinateNotOneToken {
                start: source_position,
                end: source_position,
            },
        )?;
        Self::new(
            context,
            SupportAddress::Source {
                query_head,
                kv_head,
                source_start: source_position,
                source_end,
                source_role: role_map.role_at(source_position, visible_source_range)?,
                role_map_identity: role_map.identity()?,
            },
        )
    }

    /// Frozen schema named by this value.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Validate invariants that serde alone cannot express.
    pub fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_schema(SUPPORT_COORDINATE_SCHEMA, &self.schema)?;
        self.context.validate()?;
        self.address.validate()
    }

    /// Canonical JSON bytes used by the frozen coordinate hash.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AttributionIdentityError> {
        self.validate()?;
        canonical_json_bytes(self)
    }

    /// `sha256:` identity over sorted-key, whitespace-free canonical JSON.
    pub fn identity(&self) -> Result<String, AttributionIdentityError> {
        Ok(hash_bytes(&self.canonical_bytes()?))
    }
}

/// One exact observation of a stable support coordinate.
///
/// Reference and production backends intentionally produce different values
/// of this type while sharing the same [`SupportCoordinate`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportObservationIdentity {
    schema: String,
    pub support_coordinate_id: String,
    pub execution_identity: String,
    pub provenance_fingerprint: String,
    pub prepared_image_fingerprint: String,
    pub observation_receipt_digest: String,
    pub event_sequence: u64,
}

impl SupportObservationIdentity {
    /// Bind a coordinate to one execution and one sequenced observation.
    pub fn new(
        coordinate: &SupportCoordinate,
        execution_identity: impl Into<String>,
        provenance_fingerprint: impl Into<String>,
        prepared_image_fingerprint: impl Into<String>,
        observation_receipt_digest: impl Into<String>,
        event_sequence: u64,
    ) -> Result<Self, AttributionIdentityError> {
        let observation = Self {
            schema: SUPPORT_OBSERVATION_SCHEMA.to_string(),
            support_coordinate_id: coordinate.identity()?,
            execution_identity: execution_identity.into(),
            provenance_fingerprint: provenance_fingerprint.into(),
            prepared_image_fingerprint: prepared_image_fingerprint.into(),
            observation_receipt_digest: observation_receipt_digest.into(),
            event_sequence,
        };
        observation.validate()?;
        Ok(observation)
    }

    /// Frozen schema named by this value.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Validate the execution binding before hashing or persistence.
    pub fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_schema(SUPPORT_OBSERVATION_SCHEMA, &self.schema)?;
        require_nonempty("support_coordinate_id", &self.support_coordinate_id)?;
        require_nonempty("execution_identity", &self.execution_identity)?;
        require_nonempty("provenance_fingerprint", &self.provenance_fingerprint)?;
        require_nonempty(
            "prepared_image_fingerprint",
            &self.prepared_image_fingerprint,
        )?;
        require_nonempty(
            "observation_receipt_digest",
            &self.observation_receipt_digest,
        )
    }

    /// Canonical JSON bytes used by the frozen observation hash.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AttributionIdentityError> {
        self.validate()?;
        canonical_json_bytes(self)
    }

    /// `sha256:` identity over this exact execution binding.
    pub fn identity(&self) -> Result<String, AttributionIdentityError> {
        Ok(hash_bytes(&self.canonical_bytes()?))
    }
}

/// One source row borrowed from the canonical attention tap and copied into
/// the lossless ATTR-1D input sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSourceEvidence {
    pub position: usize,
    pub attention_probability: f32,
    pub values: Vec<f32>,
}

/// Complete evidence for one query head at one attention write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionHeadEvidence {
    pub query_head: usize,
    pub kv_head: usize,
    pub sink: f32,
    /// The kernel's `Σ_t a[h,t] · v[g(h),t]`, before its optional gate.
    pub mixed_values: Vec<f32>,
    /// Activated output-gate values for this head, when declared by the plan.
    pub gate: Option<Vec<f32>>,
    pub sources: Vec<AttentionSourceEvidence>,
}

/// Lossless observed inputs for one attention support transform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSiteEvidence {
    pub expected_query_heads: usize,
    pub visible_source_range: TokenSpan,
    pub recorded_delta: Vec<f32>,
    pub heads: Vec<AttentionHeadEvidence>,
}

/// Lossless source evidence bound to the exact canonical observation that
/// produced it.
///
/// This is a sidecar rather than another execution surface: callers populate
/// [`AttentionSiteEvidence`] from the borrowed HEAD tap in the same run, then
/// seal it here with the site coordinate, prepared image, provenance, receipt
/// and event sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSourceEvidenceSidecar {
    schema: String,
    pub support_coordinate_id: String,
    pub support_coordinate: SupportCoordinate,
    pub support_observation_id: String,
    pub support_observation: SupportObservationIdentity,
    pub evidence: AttentionSiteEvidence,
}

impl AttentionSourceEvidenceSidecar {
    /// Seal source evidence from one exact observed site.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: SupportCoordinateContext,
        execution_identity: impl Into<String>,
        provenance_fingerprint: impl Into<String>,
        prepared_image_fingerprint: impl Into<String>,
        observation_receipt_digest: impl Into<String>,
        event_sequence: u64,
        evidence: AttentionSiteEvidence,
    ) -> Result<Self, AttributionIdentityError> {
        let support_coordinate = SupportCoordinate::site(context)?;
        let support_coordinate_id = support_coordinate.identity()?;
        let support_observation = SupportObservationIdentity::new(
            &support_coordinate,
            execution_identity,
            provenance_fingerprint,
            prepared_image_fingerprint,
            observation_receipt_digest,
            event_sequence,
        )?;
        let support_observation_id = support_observation.identity()?;
        let sidecar = Self {
            schema: SOURCE_EVIDENCE_SCHEMA.to_string(),
            support_coordinate_id,
            support_coordinate,
            support_observation_id,
            support_observation,
            evidence,
        };
        sidecar.validate()?;
        Ok(sidecar)
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Validate all identity joins without trusting persisted digest fields.
    pub fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_schema(SOURCE_EVIDENCE_SCHEMA, &self.schema)?;
        if !matches!(&self.support_coordinate.address, SupportAddress::Site) {
            return Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_coordinate_kind",
            });
        }
        if self.support_coordinate.identity()? != self.support_coordinate_id {
            return Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_coordinate_id",
            });
        }
        if self.support_observation.support_coordinate_id != self.support_coordinate_id {
            return Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_observation.support_coordinate_id",
            });
        }
        if self.support_observation.identity()? != self.support_observation_id {
            return Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_observation_id",
            });
        }
        if self.evidence.visible_source_range.start >= self.evidence.visible_source_range.end {
            return Err(AttributionIdentityError::InvalidSpan {
                start: self.evidence.visible_source_range.start,
                end: self.evidence.visible_source_range.end,
            });
        }
        if self.support_coordinate.context.position.checked_add(1)
            != Some(self.evidence.visible_source_range.end)
        {
            return Err(AttributionIdentityError::SourceRangePositionMismatch {
                start: self.evidence.visible_source_range.start,
                end: self.evidence.visible_source_range.end,
                position: self.support_coordinate.context.position,
            });
        }
        Ok(())
    }

    /// Deterministic bytes for replay comparison and content addressing.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AttributionIdentityError> {
        self.validate()?;
        canonical_json_bytes(self)
    }

    /// Content identity of this sealed evidence object.
    pub fn identity(&self) -> Result<String, AttributionIdentityError> {
        Ok(hash_bytes(&self.canonical_bytes()?))
    }
}

/// The linear transform applied to raw output-projection children before the
/// residual write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PostAttentionTransform {
    Identity {
        residual_scale: f32,
    },
    RmsNorm {
        epsilon: f64,
        weight_offset: f32,
        weights: Vec<f32>,
        residual_scale: f32,
    },
}

/// Adapter onto the exact prepared image that produced an observation.
///
/// An implementation must project through that image's effective `W_O` head
/// slice.  Re-loading or widening checkpoint weights is outside this seam.
pub trait PreparedAttentionProjection {
    fn hidden_size(&self) -> usize;

    fn project_head(&self, query_head: usize, values: &[f32]) -> Result<Vec<f32>, String>;

    fn output_bias(&self) -> Result<Option<Vec<f32>>, String>;

    fn post_attention_transform(&self) -> Result<PostAttentionTransform, String>;
}

/// One fixed reader from a basis frozen before observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionReader {
    pub identity: String,
    pub values: Vec<f32>,
}

impl AttributionReader {
    pub fn new(
        identity: impl Into<String>,
        values: Vec<f32>,
    ) -> Result<Self, AttributionIdentityError> {
        let reader = Self {
            identity: identity.into(),
            values,
        };
        require_nonempty("reader_identity", &reader.identity)?;
        Ok(reader)
    }
}

/// A fixed-reader projection with both signed and absolute-mass shares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionMeasurement {
    pub reader_identity: String,
    pub value: f64,
    /// Absent exactly when the frozen signed denominator is unstable.
    pub signed_share: Option<f64>,
    /// Absent exactly when the frozen nonnegative mass denominator is zero.
    pub absolute_mass_share: Option<f64>,
}

/// Descriptive support for one exact source token under one query head.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceContribution {
    pub position: usize,
    pub source_role: SourceRole,
    pub attention_probability: f32,
    pub contribution: Vec<f32>,
    pub contribution_norm: f64,
    pub selected_token_projection: Vec<ProjectionMeasurement>,
}

/// A source-role sum.  This is a measurement over exact source coordinates,
/// never a replacement coordinate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRoleContribution {
    pub source_role: SourceRole,
    pub selected_token_projection: Vec<ProjectionMeasurement>,
}

/// Descriptive support for one query head.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadContribution {
    pub query_head: usize,
    pub kv_head: usize,
    pub sink: f32,
    pub attention_probability_error: f64,
    pub contribution: Vec<f32>,
    pub contribution_norm: f64,
    pub selected_token_projection: Vec<ProjectionMeasurement>,
    pub sources: Vec<SourceContribution>,
    pub source_role_projection: Vec<SourceRoleContribution>,
    pub source_reconstruction_relative_l2: f64,
}

/// Per-reader check that site projection equals bias plus all head
/// projections under the recorded residual write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionReconstruction {
    pub reader_identity: String,
    pub absolute_error: f64,
    pub permitted_error: f64,
}

/// The complete deterministic result for one attention write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionDescriptiveSupport {
    pub attr1d_contract_identity: String,
    pub site_contribution: Vec<f32>,
    pub bias_contribution: Vec<f32>,
    pub reconstructed_delta: Vec<f32>,
    pub head_reconstruction_relative_l2: f64,
    pub site_projection: Vec<ProjectionMeasurement>,
    pub bias_projection: Vec<ProjectionMeasurement>,
    pub heads: Vec<HeadContribution>,
    pub projection_reconstruction: Vec<ProjectionReconstruction>,
}

impl AttentionDescriptiveSupport {
    /// All frozen numerical gates, excluding replay/backend checks which need
    /// two separately produced results.
    pub fn passes_numerical_gates(&self) -> bool {
        self.head_reconstruction_relative_l2 <= HEAD_SUM_MAX_RELATIVE_L2
            && self.heads.iter().all(|head| {
                head.source_reconstruction_relative_l2 <= SOURCE_SPLIT_MAX_RELATIVE_L2
                    && head.attention_probability_error <= ATTENTION_PROBABILITY_MAX_ABSOLUTE_ERROR
            })
            && self
                .projection_reconstruction
                .iter()
                .all(|check| check.absolute_error <= check.permitted_error)
    }
}

/// Apply the frozen ATTR-1D attention algebra.
///
/// Event coverage and dimensions are refused before arithmetic.  Gate values
/// multiply both the independently observed mixed head and every source row;
/// source attention probability is never treated as contribution by itself.
pub fn describe_attention_support(
    evidence: &AttentionSiteEvidence,
    prepared: &dyn PreparedAttentionProjection,
    readers: &[AttributionReader],
    role_map: &PromptRoleMap,
) -> Result<AttentionDescriptiveSupport, DescriptiveTransformError> {
    role_map.validate()?;
    validate_site_evidence(evidence, prepared.hidden_size(), readers)?;

    let hidden = prepared.hidden_size();
    let bias = prepared
        .output_bias()
        .map_err(DescriptiveTransformError::Projection)?
        .unwrap_or_else(|| vec![0.0; hidden]);
    require_width("output_bias", bias.len(), hidden)?;
    require_finite("output_bias", &bias)?;
    let post = prepared
        .post_attention_transform()
        .map_err(DescriptiveTransformError::Projection)?;
    validate_post_transform(&post, hidden)?;

    struct RawHead {
        query_head: usize,
        kv_head: usize,
        sink: f32,
        probability_error: f64,
        contribution: Vec<f32>,
        sources: Vec<(usize, SourceRole, f32, Vec<f32>)>,
    }

    let mut raw_heads = Vec::with_capacity(evidence.heads.len());
    for head in &evidence.heads {
        let gated_mixed = gate_values(&head.mixed_values, head.gate.as_deref());
        let contribution = prepared
            .project_head(head.query_head, &gated_mixed)
            .map_err(DescriptiveTransformError::Projection)?;
        require_width("head_projection", contribution.len(), hidden)?;
        require_finite("head_projection", &contribution)?;

        let mut sources = Vec::with_capacity(head.sources.len());
        for source in &head.sources {
            let weighted: Vec<f32> = source
                .values
                .iter()
                .map(|value| source.attention_probability * value)
                .collect();
            let gated = gate_values(&weighted, head.gate.as_deref());
            let projected = prepared
                .project_head(head.query_head, &gated)
                .map_err(DescriptiveTransformError::Projection)?;
            require_width("source_projection", projected.len(), hidden)?;
            require_finite("source_projection", &projected)?;
            let role = role_map.role_at(source.position, evidence.visible_source_range)?;
            sources.push((
                source.position,
                role,
                source.attention_probability,
                projected,
            ));
        }
        let probability_sum = head
            .sources
            .iter()
            .map(|source| f64::from(source.attention_probability))
            .sum::<f64>()
            + f64::from(head.sink);
        raw_heads.push(RawHead {
            query_head: head.query_head,
            kv_head: head.kv_head,
            sink: head.sink,
            probability_error: (probability_sum - 1.0).abs(),
            contribution,
            sources,
        });
    }

    // The RMS scalar is computed over the independently observed head mixtures
    // plus the once-only bias, exactly as the executor's head reader does.
    let mut raw_output = bias
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    for head in &raw_heads {
        for (sum, value) in raw_output.iter_mut().zip(&head.contribution) {
            *sum += f64::from(*value);
        }
    }
    let applied_bias = apply_post_transform(&bias, &raw_output, &post);

    struct AppliedHead {
        query_head: usize,
        kv_head: usize,
        sink: f32,
        probability_error: f64,
        contribution: Vec<f32>,
        sources: Vec<(usize, SourceRole, f32, Vec<f32>)>,
    }
    let applied_heads = raw_heads
        .into_iter()
        .map(|head| AppliedHead {
            query_head: head.query_head,
            kv_head: head.kv_head,
            sink: head.sink,
            probability_error: head.probability_error,
            contribution: apply_post_transform(&head.contribution, &raw_output, &post),
            sources: head
                .sources
                .into_iter()
                .map(|(position, role, probability, contribution)| {
                    (
                        position,
                        role,
                        probability,
                        apply_post_transform(&contribution, &raw_output, &post),
                    )
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let mut reconstructed = applied_bias
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    for head in &applied_heads {
        for (sum, value) in reconstructed.iter_mut().zip(&head.contribution) {
            *sum += f64::from(*value);
        }
    }
    let reconstructed_delta = reconstructed
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let head_reconstruction_relative_l2 = relative_l2(&reconstructed, &evidence.recorded_delta);

    let site_values = project_readers(readers, &evidence.recorded_delta);
    let bias_values = project_readers(readers, &applied_bias);
    let head_values = applied_heads
        .iter()
        .map(|head| project_readers(readers, &head.contribution))
        .collect::<Vec<_>>();
    let absolute_head_denominators = readers
        .iter()
        .enumerate()
        .map(|(reader, _)| {
            bias_values[reader].abs()
                + head_values
                    .iter()
                    .map(|values| values[reader].abs())
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();

    let mut heads = Vec::with_capacity(applied_heads.len());
    for (head_index, head) in applied_heads.iter().enumerate() {
        let source_values = head
            .sources
            .iter()
            .map(|(_, _, _, contribution)| project_readers(readers, contribution))
            .collect::<Vec<_>>();
        let absolute_source_denominators = readers
            .iter()
            .enumerate()
            .map(|(reader, _)| {
                source_values
                    .iter()
                    .map(|values| values[reader].abs())
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();

        let sources = head
            .sources
            .iter()
            .zip(&source_values)
            .map(
                |((position, role, probability, contribution), values)| SourceContribution {
                    position: *position,
                    source_role: *role,
                    attention_probability: *probability,
                    contribution: contribution.clone(),
                    contribution_norm: l2_norm(contribution),
                    selected_token_projection: readers
                        .iter()
                        .enumerate()
                        .map(|(reader_index, reader)| ProjectionMeasurement {
                            reader_identity: reader.identity.clone(),
                            value: values[reader_index],
                            signed_share: stable_ratio(
                                values[reader_index],
                                head_values[head_index][reader_index],
                            ),
                            absolute_mass_share: stable_ratio(
                                values[reader_index].abs(),
                                absolute_source_denominators[reader_index],
                            ),
                        })
                        .collect(),
                },
            )
            .collect::<Vec<_>>();

        let source_role_projection = SourceRole::ordered()
            .iter()
            .filter_map(|role| {
                let indices = sources
                    .iter()
                    .enumerate()
                    .filter_map(|(index, source)| (source.source_role == *role).then_some(index))
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    return None;
                }
                Some(SourceRoleContribution {
                    source_role: *role,
                    selected_token_projection: readers
                        .iter()
                        .enumerate()
                        .map(|(reader_index, reader)| {
                            let value = indices
                                .iter()
                                .map(|index| source_values[*index][reader_index])
                                .sum::<f64>();
                            ProjectionMeasurement {
                                reader_identity: reader.identity.clone(),
                                value,
                                signed_share: stable_ratio(
                                    value,
                                    head_values[head_index][reader_index],
                                ),
                                absolute_mass_share: stable_ratio(
                                    value.abs(),
                                    absolute_source_denominators[reader_index],
                                ),
                            }
                        })
                        .collect(),
                })
            })
            .collect();

        let mut source_sum = vec![0.0f64; hidden];
        for (_, _, _, contribution) in &head.sources {
            for (sum, value) in source_sum.iter_mut().zip(contribution) {
                *sum += f64::from(*value);
            }
        }
        heads.push(HeadContribution {
            query_head: head.query_head,
            kv_head: head.kv_head,
            sink: head.sink,
            attention_probability_error: head.probability_error,
            contribution: head.contribution.clone(),
            contribution_norm: l2_norm(&head.contribution),
            selected_token_projection: readers
                .iter()
                .enumerate()
                .map(|(reader_index, reader)| ProjectionMeasurement {
                    reader_identity: reader.identity.clone(),
                    value: head_values[head_index][reader_index],
                    signed_share: stable_ratio(
                        head_values[head_index][reader_index],
                        site_values[reader_index],
                    ),
                    absolute_mass_share: stable_ratio(
                        head_values[head_index][reader_index].abs(),
                        absolute_head_denominators[reader_index],
                    ),
                })
                .collect(),
            sources,
            source_role_projection,
            source_reconstruction_relative_l2: relative_l2(&source_sum, &head.contribution),
        });
    }

    let projection_reconstruction = readers
        .iter()
        .enumerate()
        .map(|(reader_index, reader)| {
            let reconstructed = bias_values[reader_index]
                + head_values
                    .iter()
                    .map(|values| values[reader_index])
                    .sum::<f64>();
            let absolute_terms = bias_values[reader_index].abs()
                + head_values
                    .iter()
                    .map(|values| values[reader_index].abs())
                    .sum::<f64>();
            ProjectionReconstruction {
                reader_identity: reader.identity.clone(),
                absolute_error: (reconstructed - site_values[reader_index]).abs(),
                permitted_error: PROJECTION_RECONSTRUCTION_SCALE * absolute_terms.max(1.0),
            }
        })
        .collect();

    Ok(AttentionDescriptiveSupport {
        attr1d_contract_identity: ATTR1D_CONTRACT_IDENTITY.to_string(),
        site_contribution: evidence.recorded_delta.clone(),
        bias_contribution: applied_bias,
        reconstructed_delta,
        head_reconstruction_relative_l2,
        site_projection: readers
            .iter()
            .enumerate()
            .map(|(reader_index, reader)| ProjectionMeasurement {
                reader_identity: reader.identity.clone(),
                value: site_values[reader_index],
                signed_share: stable_ratio(site_values[reader_index], site_values[reader_index]),
                absolute_mass_share: None,
            })
            .collect(),
        bias_projection: readers
            .iter()
            .enumerate()
            .map(|(reader_index, reader)| ProjectionMeasurement {
                reader_identity: reader.identity.clone(),
                value: bias_values[reader_index],
                signed_share: stable_ratio(bias_values[reader_index], site_values[reader_index]),
                absolute_mass_share: stable_ratio(
                    bias_values[reader_index].abs(),
                    absolute_head_denominators[reader_index],
                ),
            })
            .collect(),
        heads,
        projection_reconstruction,
    })
}

/// A half-open token span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

impl TokenSpan {
    /// Construct a non-empty half-open span.
    pub fn new(start: usize, end: usize) -> Result<Self, AttributionIdentityError> {
        if start >= end {
            return Err(AttributionIdentityError::InvalidSpan { start, end });
        }
        Ok(Self { start, end })
    }

    /// Whether this span contains one token position.
    pub fn contains(self, position: usize) -> bool {
        self.start <= position && position < self.end
    }

    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// Construction-time prompt structure used to assign source roles.
///
/// The manifest can be serialized and hashed before executing the model.  Its
/// role assignment does not inspect token text, attention, contribution, or
/// outcomes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptRoleMap {
    schema: String,
    pub prompt_token_count: usize,
    pub bos_spans: Vec<TokenSpan>,
    pub relation_spans: Vec<TokenSpan>,
    pub entity_spans: Vec<TokenSpan>,
    pub final_prompt_position: usize,
}

impl PromptRoleMap {
    /// Construct a role map, refusing malformed, out-of-prompt, or overlapping
    /// labelled spans.
    pub fn new(
        prompt_token_count: usize,
        bos_spans: Vec<TokenSpan>,
        relation_spans: Vec<TokenSpan>,
        entity_spans: Vec<TokenSpan>,
        final_prompt_position: usize,
    ) -> Result<Self, AttributionIdentityError> {
        let role_map = Self {
            schema: PROMPT_ROLE_MAP_SCHEMA.to_string(),
            prompt_token_count,
            bos_spans,
            relation_spans,
            entity_spans,
            final_prompt_position,
        };
        role_map.validate()?;
        Ok(role_map)
    }

    /// Frozen schema named by this value.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Validate construction-time span authority.
    pub fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_schema(PROMPT_ROLE_MAP_SCHEMA, &self.schema)?;
        if self.final_prompt_position >= self.prompt_token_count {
            return Err(AttributionIdentityError::FinalPromptPositionOutsidePrompt {
                position: self.final_prompt_position,
                prompt_len: self.prompt_token_count,
            });
        }

        let mut labelled = Vec::new();
        for (role, spans) in [
            (SourceRole::Bos, self.bos_spans.as_slice()),
            (SourceRole::Relation, self.relation_spans.as_slice()),
            (SourceRole::Entity, self.entity_spans.as_slice()),
        ] {
            for span in spans {
                if span.start >= span.end {
                    return Err(AttributionIdentityError::InvalidSpan {
                        start: span.start,
                        end: span.end,
                    });
                }
                if span.end > self.prompt_token_count {
                    return Err(AttributionIdentityError::SpanOutsidePrompt {
                        start: span.start,
                        end: span.end,
                        prompt_len: self.prompt_token_count,
                    });
                }
                labelled.push((role, *span));
            }
        }

        for left_index in 0..labelled.len() {
            let (left_role, left) = labelled[left_index];
            for &(right_role, right) in &labelled[left_index + 1..] {
                if left.overlaps(right) {
                    return Err(AttributionIdentityError::OverlappingRoleSpans {
                        left_role,
                        left_start: left.start,
                        left_end: left.end,
                        right_role,
                        right_start: right.start,
                        right_end: right.end,
                    });
                }
            }
        }
        Ok(())
    }

    /// Stable identity of the prompt-structure declaration.
    pub fn identity(&self) -> Result<String, AttributionIdentityError> {
        self.validate()?;
        Ok(hash_bytes(&canonical_json_bytes(self)?))
    }

    /// Assign a role under the frozen precedence rule.
    ///
    /// `visible_source_range` comes from the exact observation.  A position
    /// outside it is refused rather than clipped or silently mapped to
    /// [`SourceRole::Other`].  Generated positions inside the visible range
    /// but beyond the prompt are `OTHER`.
    pub fn role_at(
        &self,
        position: usize,
        visible_source_range: TokenSpan,
    ) -> Result<SourceRole, AttributionIdentityError> {
        self.validate()?;
        if visible_source_range.start >= visible_source_range.end {
            return Err(AttributionIdentityError::InvalidSpan {
                start: visible_source_range.start,
                end: visible_source_range.end,
            });
        }
        if !visible_source_range.contains(position) {
            return Err(
                AttributionIdentityError::SourcePositionOutsideVisibleRange {
                    position,
                    start: visible_source_range.start,
                    end: visible_source_range.end,
                },
            );
        }

        if contains_position(&self.bos_spans, position) {
            return Ok(SourceRole::Bos);
        }
        if contains_position(&self.relation_spans, position) {
            return Ok(SourceRole::Relation);
        }
        if contains_position(&self.entity_spans, position) {
            return Ok(SourceRole::Entity);
        }
        if position == self.final_prompt_position {
            return Ok(SourceRole::Last);
        }
        Ok(SourceRole::Other)
    }
}

impl SourceRole {
    fn ordered() -> &'static [Self; 5] {
        &[
            Self::Bos,
            Self::Relation,
            Self::Entity,
            Self::Last,
            Self::Other,
        ]
    }
}

fn validate_site_evidence(
    evidence: &AttentionSiteEvidence,
    hidden: usize,
    readers: &[AttributionReader],
) -> Result<(), DescriptiveTransformError> {
    if evidence.visible_source_range.start >= evidence.visible_source_range.end {
        return Err(AttributionIdentityError::InvalidSpan {
            start: evidence.visible_source_range.start,
            end: evidence.visible_source_range.end,
        }
        .into());
    }
    require_width("recorded_delta", evidence.recorded_delta.len(), hidden)?;
    require_finite("recorded_delta", &evidence.recorded_delta)?;
    if evidence.heads.len() != evidence.expected_query_heads {
        return Err(DescriptiveTransformError::HeadCoverage {
            expected: evidence.expected_query_heads,
            actual: evidence.heads.len(),
        });
    }
    let mut seen_heads = BTreeSet::new();
    let mut observed_heads = Vec::with_capacity(evidence.heads.len());
    for head in &evidence.heads {
        if !seen_heads.insert(head.query_head) {
            return Err(DescriptiveTransformError::DuplicateHead {
                head: head.query_head,
            });
        }
        observed_heads.push(head.query_head);
        require_finite("mixed_values", &head.mixed_values)?;
        if !head.sink.is_finite() {
            return Err(DescriptiveTransformError::NonFinite { field: "sink" });
        }
        if let Some(gate) = &head.gate {
            require_width("head_gate", gate.len(), head.mixed_values.len())?;
            require_finite("head_gate", gate)?;
        }
        let mut seen_positions = BTreeSet::new();
        let mut positions = Vec::with_capacity(head.sources.len());
        for source in &head.sources {
            if !seen_positions.insert(source.position) {
                return Err(DescriptiveTransformError::DuplicateSource {
                    head: head.query_head,
                    position: source.position,
                });
            }
            positions.push(source.position);
            require_width(
                "source_values",
                source.values.len(),
                head.mixed_values.len(),
            )?;
            require_finite("source_values", &source.values)?;
            if !source.attention_probability.is_finite() {
                return Err(DescriptiveTransformError::NonFinite {
                    field: "attention_probability",
                });
            }
        }
        let expected = (evidence.visible_source_range.start..evidence.visible_source_range.end)
            .collect::<Vec<_>>();
        if positions != expected {
            return Err(DescriptiveTransformError::SourceCoverage {
                head: head.query_head,
                start: evidence.visible_source_range.start,
                end: evidence.visible_source_range.end,
                observed: positions,
            });
        }
    }
    let expected = (0..evidence.expected_query_heads).collect::<Vec<_>>();
    if observed_heads != expected {
        return Err(DescriptiveTransformError::NonCanonicalHeadCoverage {
            expected: evidence.expected_query_heads,
            observed: observed_heads,
        });
    }
    for reader in readers {
        require_nonempty("reader_identity", &reader.identity)?;
        require_width("reader", reader.values.len(), hidden)?;
        require_finite("reader", &reader.values)?;
    }
    Ok(())
}

fn validate_post_transform(
    transform: &PostAttentionTransform,
    hidden: usize,
) -> Result<(), DescriptiveTransformError> {
    match transform {
        PostAttentionTransform::Identity { residual_scale } => {
            if !residual_scale.is_finite() {
                return Err(DescriptiveTransformError::NonFinite {
                    field: "residual_scale",
                });
            }
        }
        PostAttentionTransform::RmsNorm {
            epsilon,
            weight_offset,
            weights,
            residual_scale,
        } => {
            if !epsilon.is_finite() || *epsilon <= 0.0 {
                return Err(DescriptiveTransformError::InvalidRmsEpsilon { epsilon: *epsilon });
            }
            if !weight_offset.is_finite() || !residual_scale.is_finite() {
                return Err(DescriptiveTransformError::NonFinite {
                    field: "post_attention_transform",
                });
            }
            require_width("post_attention_norm_weights", weights.len(), hidden)?;
            require_finite("post_attention_norm_weights", weights)?;
        }
    }
    Ok(())
}

fn gate_values(values: &[f32], gate: Option<&[f32]>) -> Vec<f32> {
    match gate {
        Some(gate) => values
            .iter()
            .zip(gate)
            .map(|(value, gate)| value * gate)
            .collect(),
        None => values.to_vec(),
    }
}

fn apply_post_transform(
    values: &[f32],
    raw_output: &[f64],
    transform: &PostAttentionTransform,
) -> Vec<f32> {
    match transform {
        PostAttentionTransform::Identity { residual_scale } => {
            values.iter().map(|value| value * residual_scale).collect()
        }
        PostAttentionTransform::RmsNorm {
            epsilon,
            weight_offset,
            weights,
            residual_scale,
        } => {
            let mean_square =
                raw_output.iter().map(|value| value * value).sum::<f64>() / raw_output.len() as f64;
            let alpha = 1.0 / (mean_square + epsilon).sqrt();
            values
                .iter()
                .zip(weights)
                .map(|(value, weight)| {
                    (f64::from(*value)
                        * alpha
                        * f64::from(*weight_offset + *weight)
                        * f64::from(*residual_scale)) as f32
                })
                .collect()
        }
    }
}

fn project_readers(readers: &[AttributionReader], values: &[f32]) -> Vec<f64> {
    readers
        .iter()
        .map(|reader| {
            reader
                .values
                .iter()
                .zip(values)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum()
        })
        .collect()
}

fn relative_l2(reconstructed: &[f64], recorded: &[f32]) -> f64 {
    let error = reconstructed
        .iter()
        .zip(recorded)
        .map(|(left, right)| (*left - f64::from(*right)).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = recorded
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    error / norm.max(VECTOR_DENOMINATOR_FLOOR)
}

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn stable_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator.abs() > VECTOR_DENOMINATOR_FLOOR).then_some(numerator / denominator)
}

fn require_width(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), DescriptiveTransformError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DescriptiveTransformError::Width {
            field,
            actual,
            expected,
        })
    }
}

fn require_finite(field: &'static str, values: &[f32]) -> Result<(), DescriptiveTransformError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(DescriptiveTransformError::NonFinite { field })
    }
}

fn contains_position(spans: &[TokenSpan], position: usize) -> bool {
    spans.iter().any(|span| span.contains(position))
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), AttributionIdentityError> {
    if value.trim().is_empty() {
        Err(AttributionIdentityError::EmptyField { field })
    } else {
        Ok(())
    }
}

fn require_schema(expected: &'static str, actual: &str) -> Result<(), AttributionIdentityError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AttributionIdentityError::WrongSchema {
            expected,
            actual: actual.to_string(),
        })
    }
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, AttributionIdentityError> {
    let value = serde_json::to_value(value)
        .map_err(|error| AttributionIdentityError::Serialization(error.to_string()))?;
    let mut out = String::new();
    write_canonical_json(&value, &mut out);
    Ok(out.into_bytes())
}

fn write_canonical_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical_json(&map[key], out);
            }
            out.push('}');
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        leaf => out.push_str(&leaf.to_string()),
    }
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct SyntheticProjection {
        bias: Option<Vec<f32>>,
        post: PostAttentionTransform,
    }

    impl PreparedAttentionProjection for SyntheticProjection {
        fn hidden_size(&self) -> usize {
            2
        }

        fn project_head(&self, query_head: usize, values: &[f32]) -> Result<Vec<f32>, String> {
            if query_head != 0 {
                return Err(format!("unexpected head {query_head}"));
            }
            if values.len() != 2 {
                return Err(format!("unexpected width {}", values.len()));
            }
            // Identity is sufficient to test the decomposition law. The
            // production adapter is responsible for using effective W_O.
            Ok(values.to_vec())
        }

        fn output_bias(&self) -> Result<Option<Vec<f32>>, String> {
            Ok(self.bias.clone())
        }

        fn post_attention_transform(&self) -> Result<PostAttentionTransform, String> {
            Ok(self.post.clone())
        }
    }

    fn span(start: usize, end: usize) -> TokenSpan {
        TokenSpan::new(start, end).unwrap()
    }

    fn context() -> SupportCoordinateContext {
        SupportCoordinateContext::new(
            "model-system",
            "logical-plan",
            "text-component",
            "prompt-layout",
            12,
            24,
        )
        .unwrap()
    }

    fn role_map() -> PromptRoleMap {
        PromptRoleMap::new(8, vec![span(0, 1)], vec![span(2, 4)], vec![span(6, 8)], 7).unwrap()
    }

    fn small_role_map() -> PromptRoleMap {
        PromptRoleMap::new(2, vec![span(0, 1)], vec![], vec![span(1, 2)], 1).unwrap()
    }

    fn readers() -> Vec<AttributionReader> {
        vec![
            AttributionReader::new("reader-x", vec![1.0, 0.0]).unwrap(),
            AttributionReader::new("reader-y", vec![0.0, 1.0]).unwrap(),
        ]
    }

    fn one_head_evidence(recorded_delta: Vec<f32>) -> AttentionSiteEvidence {
        AttentionSiteEvidence {
            expected_query_heads: 1,
            visible_source_range: span(0, 2),
            recorded_delta,
            heads: vec![AttentionHeadEvidence {
                query_head: 0,
                kv_head: 0,
                sink: 0.0,
                // .25*[2,4] + .75*[4,0]
                mixed_values: vec![3.5, 1.0],
                gate: Some(vec![2.0, 0.5]),
                sources: vec![
                    AttentionSourceEvidence {
                        position: 0,
                        attention_probability: 0.25,
                        values: vec![2.0, 4.0],
                    },
                    AttentionSourceEvidence {
                        position: 1,
                        attention_probability: 0.75,
                        values: vec![4.0, 0.0],
                    },
                ],
            }],
        }
    }

    #[test]
    fn coordinate_serialization_is_flat_and_excludes_execution_identity() {
        let map = role_map();
        let coordinate = SupportCoordinate::new(
            context(),
            SupportAddress::Source {
                query_head: 3,
                kv_head: 1,
                source_start: 6,
                source_end: 7,
                source_role: SourceRole::Entity,
                role_map_identity: map.identity().unwrap(),
            },
        )
        .unwrap();
        let value = serde_json::to_value(&coordinate).unwrap();
        let object = value.as_object().unwrap();

        assert_eq!(object["coordinate_kind"], "source");
        assert_eq!(object["operator"], "attention");
        assert_eq!(object["source_role"], "ENTITY");
        assert!(!object.contains_key("address"));
        for forbidden in [
            "execution_identity",
            "provenance_fingerprint",
            "prepared_image_fingerprint",
            "observation_receipt_digest",
            "contribution",
            "transition_identity",
            "causal_label",
        ] {
            assert!(!object.contains_key(forbidden), "found {forbidden}");
        }
    }

    #[test]
    fn the_same_coordinate_has_distinct_backend_observations() {
        let coordinate = SupportCoordinate::new(
            context(),
            SupportAddress::Head {
                query_head: 3,
                kv_head: 1,
            },
        )
        .unwrap();
        let reference = SupportObservationIdentity::new(
            &coordinate,
            "reference-execution",
            "reference-provenance",
            "reference-image",
            "reference-receipt",
            42,
        )
        .unwrap();
        let production = SupportObservationIdentity::new(
            &coordinate,
            "production-execution",
            "production-provenance",
            "production-image",
            "production-receipt",
            42,
        )
        .unwrap();

        assert_eq!(
            reference.support_coordinate_id,
            production.support_coordinate_id
        );
        assert_ne!(
            reference.identity().unwrap(),
            production.identity().unwrap()
        );
    }

    #[test]
    fn replay_reproduces_canonical_bytes_and_both_identities() {
        let coordinate = SupportCoordinate::new(
            context(),
            SupportAddress::Head {
                query_head: 3,
                kv_head: 1,
            },
        )
        .unwrap();
        let observation = SupportObservationIdentity::new(
            &coordinate,
            "execution",
            "provenance",
            "prepared-image",
            "receipt",
            19,
        )
        .unwrap();

        let coordinate_replay: SupportCoordinate =
            serde_json::from_slice(&coordinate.canonical_bytes().unwrap()).unwrap();
        let observation_replay: SupportObservationIdentity =
            serde_json::from_slice(&observation.canonical_bytes().unwrap()).unwrap();

        assert_eq!(
            coordinate.identity().unwrap(),
            coordinate_replay.identity().unwrap()
        );
        assert_eq!(
            coordinate.canonical_bytes().unwrap(),
            coordinate_replay.canonical_bytes().unwrap()
        );
        assert_eq!(
            observation.identity().unwrap(),
            observation_replay.identity().unwrap()
        );
        assert_eq!(
            observation.canonical_bytes().unwrap(),
            observation_replay.canonical_bytes().unwrap()
        );
    }

    #[test]
    fn canonical_json_sorts_keys_instead_of_using_struct_order() {
        let coordinate = SupportCoordinate::new(context(), SupportAddress::Site).unwrap();
        let text = String::from_utf8(coordinate.canonical_bytes().unwrap()).unwrap();
        assert!(text.starts_with("{\"component_identity\":"), "{text}");
        assert!(text.find("\"layer\":").unwrap() < text.find("\"operator\":").unwrap());
        assert!(text.find("\"schema\":").unwrap() < text.find("\"site\":").unwrap());
    }

    #[test]
    fn source_coordinates_are_exactly_one_token() {
        let error = SupportCoordinate::new(
            context(),
            SupportAddress::Source {
                query_head: 0,
                kv_head: 0,
                source_start: 3,
                source_end: 5,
                source_role: SourceRole::Other,
                role_map_identity: "role-map".to_string(),
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            AttributionIdentityError::SourceCoordinateNotOneToken { start: 3, end: 5 }
        );
    }

    #[test]
    fn role_map_refuses_any_label_overlap() {
        let error = PromptRoleMap::new(8, vec![span(0, 1)], vec![span(2, 5)], vec![span(4, 7)], 7)
            .unwrap_err();
        assert!(matches!(
            error,
            AttributionIdentityError::OverlappingRoleSpans {
                left_role: SourceRole::Relation,
                right_role: SourceRole::Entity,
                ..
            }
        ));
    }

    #[test]
    fn declared_roles_precede_last_and_unlabelled_positions_are_other() {
        let map = role_map();
        let visible = span(0, 10);
        assert_eq!(map.role_at(0, visible).unwrap(), SourceRole::Bos);
        assert_eq!(map.role_at(2, visible).unwrap(), SourceRole::Relation);
        assert_eq!(map.role_at(7, visible).unwrap(), SourceRole::Entity);
        assert_eq!(map.role_at(5, visible).unwrap(), SourceRole::Other);
        assert_eq!(map.role_at(9, visible).unwrap(), SourceRole::Other);

        let last_map =
            PromptRoleMap::new(8, vec![span(0, 1)], vec![span(2, 4)], vec![span(5, 7)], 7).unwrap();
        assert_eq!(last_map.role_at(7, visible).unwrap(), SourceRole::Last);
    }

    #[test]
    fn source_positions_outside_the_visible_range_refuse() {
        let error = role_map().role_at(1, span(2, 8)).unwrap_err();
        assert_eq!(
            error,
            AttributionIdentityError::SourcePositionOutsideVisibleRange {
                position: 1,
                start: 2,
                end: 8,
            }
        );
    }

    #[test]
    fn deserialized_wrong_schemas_cannot_be_hashed() {
        let coordinate = SupportCoordinate::new(context(), SupportAddress::Site).unwrap();
        let mut value = serde_json::to_value(coordinate).unwrap();
        value["schema"] = serde_json::Value::String("not-the-schema".to_string());
        let malformed: SupportCoordinate = serde_json::from_value(value).unwrap();
        assert!(matches!(
            malformed.identity(),
            Err(AttributionIdentityError::WrongSchema { .. })
        ));
    }

    #[test]
    fn descriptive_transform_reconstructs_gated_sources_bias_and_shares() {
        let prepared = SyntheticProjection {
            bias: Some(vec![0.1, -0.2]),
            post: PostAttentionTransform::Identity {
                residual_scale: 0.5,
            },
        };
        // Gated head [7,.5] and bias [.1,-.2], all scaled by .5.
        let evidence = one_head_evidence(vec![3.55, 0.15]);
        let result =
            describe_attention_support(&evidence, &prepared, &readers(), &small_role_map())
                .unwrap();

        assert!(result.passes_numerical_gates(), "{result:#?}");
        assert!(result.head_reconstruction_relative_l2 < 1e-7);
        assert_eq!(result.heads[0].contribution, vec![3.5, 0.25]);
        assert_eq!(result.bias_contribution, vec![0.05, -0.1]);
        assert_eq!(result.heads[0].sources[0].contribution, vec![0.5, 0.25]);
        assert_eq!(result.heads[0].sources[1].contribution, vec![3.0, 0.0]);
        assert_eq!(result.heads[0].sources[0].source_role, SourceRole::Bos);
        assert_eq!(result.heads[0].sources[1].source_role, SourceRole::Entity);
        assert_eq!(result.heads[0].attention_probability_error, 0.0);
        assert!(result.heads[0].source_reconstruction_relative_l2 < 1e-12);

        // Attention mass is descriptive input, not the contribution: the
        // .25 source carries .5 on reader-x, while the .75 source carries 3.
        assert_eq!(
            result.heads[0].sources[0].selected_token_projection[0].value,
            0.5
        );
        assert_eq!(
            result.heads[0].sources[1].selected_token_projection[0].value,
            3.0
        );
        let signed_share = result.heads[0].selected_token_projection[0]
            .signed_share
            .unwrap();
        assert!((signed_share - 3.5 / 3.55).abs() < 1e-7);
    }

    #[test]
    fn rms_transform_uses_the_once_only_raw_output_scalar() {
        let epsilon = 1e-5;
        let raw_output = [7.1f64, 0.3];
        let alpha = 1.0
            / ((raw_output.iter().map(|value| value * value).sum::<f64>() / 2.0) + epsilon).sqrt();
        let expected = vec![
            (7.1 * alpha * 1.0 * 0.5) as f32,
            (0.3 * alpha * 2.0 * 0.5) as f32,
        ];
        let prepared = SyntheticProjection {
            bias: Some(vec![0.1, -0.2]),
            post: PostAttentionTransform::RmsNorm {
                epsilon,
                weight_offset: 1.0,
                weights: vec![0.0, 1.0],
                residual_scale: 0.5,
            },
        };
        let result = describe_attention_support(
            &one_head_evidence(expected),
            &prepared,
            &readers(),
            &small_role_map(),
        )
        .unwrap();

        assert!(result.passes_numerical_gates(), "{result:#?}");
        assert!(result.head_reconstruction_relative_l2 < 1e-6);
        assert!(result.heads[0].source_reconstruction_relative_l2 < 1e-6);
    }

    #[test]
    fn incomplete_source_surface_refuses_before_projection() {
        let prepared = SyntheticProjection {
            bias: None,
            post: PostAttentionTransform::Identity {
                residual_scale: 1.0,
            },
        };
        let mut evidence = one_head_evidence(vec![7.0, 0.5]);
        evidence.heads[0].sources.pop();
        let error = describe_attention_support(&evidence, &prepared, &readers(), &small_role_map())
            .unwrap_err();
        assert!(matches!(
            error,
            DescriptiveTransformError::SourceCoverage { head: 0, .. }
        ));
    }

    #[test]
    fn source_sidecar_binds_receipt_prepared_image_and_event_sequence() {
        let sidecar = AttentionSourceEvidenceSidecar::new(
            SupportCoordinateContext::new(
                "model-system",
                "logical-plan",
                "text-component",
                "prompt-layout",
                1,
                24,
            )
            .unwrap(),
            "execution",
            "provenance",
            "prepared-image",
            "receipt",
            17,
            one_head_evidence(vec![7.0, 0.5]),
        )
        .unwrap();
        let replay: AttentionSourceEvidenceSidecar =
            serde_json::from_slice(&sidecar.canonical_bytes().unwrap()).unwrap();
        assert_eq!(sidecar.identity().unwrap(), replay.identity().unwrap());
        assert_eq!(
            sidecar.support_coordinate_id,
            sidecar.support_observation.support_coordinate_id
        );

        let mut tampered = replay;
        tampered.support_observation.event_sequence = 18;
        assert!(matches!(
            tampered.validate(),
            Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_observation_id"
            })
        ));
    }
}
