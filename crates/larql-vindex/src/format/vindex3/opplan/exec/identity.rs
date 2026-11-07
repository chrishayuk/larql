//! Which exact computation produced a carrier (RESIDUAL-BUS-2, I1 and I2).
//!
//! A carrier value is only comparable with another, and only acceptable
//! across a process boundary, when both sides can say which computation
//! produced it. The reconnaissance found that no existing identity says
//! so: an `ExecutionSlice` is scope, `LoweringIdentity` excludes provider
//! configuration by design, and four process settings that change values
//! reached no record at all.
//!
//! [`ExecutionIdentity`] is the one answer. It is read off a prepared
//! image, the running process, and the model authority established by
//! whoever opened the container, and it binds the effective MODEL: the
//! same executable realization over different weights is not the same
//! identity. An identity with no model authority, or over an overlaid
//! operand source, is recorded but UNANCHORED, and nothing that needs an
//! exact claim may use it.
//!
//! Every value here is an observed fact of the image or the process,
//! never the environment value that asked for it (the same rule as
//! [`super::provenance`]).

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::cpu::integer::{
    activation_block, activation_code, activation_scaling, bit_identical_only, weight_index_enabled,
};
use super::cpu::physical::{arithmetic_arm, kquant_execution, ArithmeticArm, KQuantExecution};
use super::prepared::{ExecutionSlice, PreparedOperands};
use super::realization::RealizationRecord;
use crate::error::VindexError;

/// Version of the canonical serialisation the digest is taken over. A
/// change to what the identity holds is a new schema, never a silent
/// change of meaning under the old one.
pub const IDENTITY_SCHEMA: u32 = 1;

/// Length of a SHA-256 digest written as lowercase hex.
const DIGEST_HEX_LEN: usize = 64;

/// The semantic authority for the model an image executes: a SHA-256
/// digest over the container's index (which declares every payload's
/// hash), its graph and the exact operation plan.
///
/// It names DECLARED payload hashes. Byte verification is a separate
/// authority, and whether an exact route requires it is BUS-3's decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelAuthority(String);

impl ModelAuthority {
    /// Adopt a digest computed by whoever opened the container. Refuses
    /// anything that is not 64 lowercase-or-uppercase hex digits.
    pub fn new(digest: impl Into<String>) -> Result<Self, VindexError> {
        let digest = digest.into();
        if digest.len() != DIGEST_HEX_LEN || !digest.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(VindexError::Parse(format!(
                "a model authority is a {DIGEST_HEX_LEN}-digit hex digest, not `{digest}`"
            )));
        }
        Ok(Self(digest.to_ascii_lowercase()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The process-global arithmetic this process resolved: every setting
/// that can change a value and lives in no operand record.
///
/// Fields are public so a test can vary each one and show the digest
/// moves (F1): the settings themselves are `OnceLock`s and cannot vary
/// inside one process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProcessArithmetic {
    pub arithmetic_arm: ArithmeticArm,
    /// Tensor or block activation scaling. The `q8xq8` and `q8xq8b`
    /// spellings select the same arm and differ only here.
    pub activation_scaling: String,
    pub activation_block: usize,
    /// Symmetric or asymmetric activation codes.
    pub activation_code: String,
    /// The K2-only switch; K3 reassociates, so this changes values.
    pub bit_identical_only: bool,
    pub weight_index: bool,
    pub kquant_execution: KQuantExecution,
    /// The CPU pool's realised size. No test establishes that results
    /// are independent of it, so it is part of the identity until one
    /// does (I2: included unless proven value-preserving).
    pub cpu_workers: usize,
}

impl ProcessArithmetic {
    /// Read what this process resolved.
    pub fn current() -> Result<Self, VindexError> {
        Ok(Self {
            arithmetic_arm: arithmetic_arm(),
            activation_scaling: format!("{:?}", activation_scaling()),
            activation_block: activation_block(),
            activation_code: format!("{:?}", activation_code()),
            bit_identical_only: bit_identical_only(),
            weight_index: weight_index_enabled(),
            kquant_execution: kquant_execution(),
            cpu_workers: super::cpu::shared()?.workers(),
        })
    }
}

/// One operand's pin, in plan order. Not aggregated: two images that pin
/// the same classes on different operands are different computations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PinnedOperand {
    pub operand: String,
    pub layer: Option<usize>,
    pub representation: String,
    pub codec: Option<String>,
    pub lowering: String,
    /// The executor's spelling of the pinned realization: backend, form
    /// and physical plan. Settings that act through selection
    /// (`LARQL_CPU_MAX_FORMAT`, `LARQL_CPU_Q4_CLASSES`) are captured here.
    pub form: String,
}

/// Which exact computation an image performs, bound to the model it
/// executes (RESIDUAL-BUS-2 I1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExecutionIdentity {
    pub schema: u32,
    /// `None` when no container authority was established.
    pub model: Option<ModelAuthority>,
    /// Whether the image was prepared over an overlaid operand source.
    /// An overlay has only a process-unique stamp, so it cannot be named
    /// across a process (D8).
    pub overlaid: bool,
    /// The slice, as scope inside the identity.
    pub slice: String,
    /// Distinct lowering providers, sorted.
    pub lowering: Vec<String>,
    pub operands: Vec<PinnedOperand>,
    pub process: ProcessArithmetic,
}

impl ExecutionIdentity {
    /// Read the identity of `ops` under the running process.
    pub fn of(ops: &PreparedOperands, model: Option<ModelAuthority>) -> Result<Self, VindexError> {
        Ok(Self::with_process(
            ops,
            model,
            ProcessArithmetic::current()?,
        ))
    }

    /// [`Self::of`] with the process arithmetic supplied, so a test can
    /// show each setting moves the digest.
    pub fn with_process(
        ops: &PreparedOperands,
        model: Option<ModelAuthority>,
        process: ProcessArithmetic,
    ) -> Self {
        Self::from_records(
            ops.realizations(),
            ops.slice(),
            model,
            ops.source_stamp().is_overlaid(),
            process,
        )
    }

    /// The identity a set of pins WOULD have under `process`, without a
    /// prepared image. This is how a coordinator states what it expects of
    /// a remote slice: it selects that slice's pins in its own process, as
    /// it already does to compare realizations, and derives the digest a
    /// worker computing the same thing must present.
    pub fn from_records(
        records: &[RealizationRecord],
        slice: &ExecutionSlice,
        model: Option<ModelAuthority>,
        overlaid: bool,
        process: ProcessArithmetic,
    ) -> Self {
        let operands: Vec<PinnedOperand> = records
            .iter()
            .map(|record| PinnedOperand {
                operand: format!("{:?}", record.planned.operand),
                layer: record.planned.layer,
                representation: record.representation.clone(),
                codec: record
                    .codec_provider
                    .as_ref()
                    .map(|c| format!("{}@{}", c.family, c.revision)),
                lowering: record.lowering_provider.to_string(),
                form: format!("{:?}", record.selection.realization),
            })
            .collect();
        let mut lowering: Vec<String> = operands.iter().map(|o| o.lowering.clone()).collect();
        lowering.sort();
        lowering.dedup();
        Self {
            schema: IDENTITY_SCHEMA,
            model,
            overlaid,
            slice: format!("{slice:?}"),
            lowering,
            operands,
            process,
        }
    }

    /// Whether an exact claim or an exact route may use this identity.
    pub fn is_anchored(&self) -> bool {
        self.model.is_some() && !self.overlaid
    }

    /// Refuse an unanchored identity, naming why.
    pub fn ensure_anchored(&self) -> Result<(), VindexError> {
        let why = match (&self.model, self.overlaid) {
            (_, true) => {
                "its operand source is overlaid, and an overlay has no identity outside \
                          the process that made it"
            }
            (None, false) => "no model authority was established for the container it executes",
            (Some(_), false) => return Ok(()),
        };
        Err(VindexError::Parse(format!(
            "this execution identity is unanchored and cannot support an exact claim: {why}"
        )))
    }

    /// SHA-256 over the canonical serialisation, as lowercase hex.
    pub fn digest(&self) -> String {
        // Serialising plain data (strings, integers, unit enums) cannot fail.
        let bytes = serde_json::to_vec(self).expect("an execution identity serialises");
        format!("{:x}", Sha256::digest(bytes))
    }
}

/// What happens to one environment setting the executor reads (I2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingFate {
    /// Recorded directly in [`ProcessArithmetic`].
    InIdentity,
    /// Acts through realization selection, so it is captured in each
    /// operand's pinned form.
    InRealizationForm,
    /// Proven not to change values, or confined to a path that cannot be
    /// sharded. The reason cites the evidence.
    Excluded(&'static str),
}

/// Every environment setting the executor's production code reads, and
/// its fate. The conformance test in `tests/execution_identity.rs`
/// refuses any setting read in `exec/` that is missing here.
pub const SETTING_FATES: &[(&str, SettingFate)] = &[
    ("LARQL_CPU_ARITHMETIC", SettingFate::InIdentity),
    ("LARQL_CPU_ACT_BLOCK", SettingFate::InIdentity),
    ("LARQL_CPU_ACT_CODE", SettingFate::InIdentity),
    ("LARQL_CPU_BIT_IDENTICAL", SettingFate::InIdentity),
    ("LARQL_CPU_WEIGHT_INDEX", SettingFate::InIdentity),
    ("LARQL_CPU_WORKERS", SettingFate::InIdentity),
    ("LARQL_KQUANT_EXEC", SettingFate::InIdentity),
    ("LARQL_CPU_MAX_FORMAT", SettingFate::InRealizationForm),
    ("LARQL_CPU_Q4_CLASSES", SettingFate::InRealizationForm),
    (
        "LARQL_FFN_MULTI_POSITION",
        SettingFate::Excluded(
            "bit-identical by test: tests/step_many.rs the_two_ffn_shapes_compute_the_same_thing",
        ),
    ),
    (
        "LARQL_CPU_STATIONARY",
        SettingFate::Excluded(
            "bit-identical by test: cpu/tests/stationary.rs \
             every_position_is_bit_identical_to_the_frozen_row",
        ),
    ),
    (
        "LARQL_F32_STAGE",
        SettingFate::Excluded(
            "residency only, bit-identical by test: weights/staged_tests.rs \
             a_mapped_image_reads_back_bit_identically",
        ),
    ),
    (
        "LARQL_F32_STAGE_DIR",
        SettingFate::Excluded("residency only: where staged f32 bytes live"),
    ),
    (
        "LARQL_F32_STAGE_MIN_BYTES",
        SettingFate::Excluded("residency only: which f32 images are staged"),
    ),
    (
        "LARQL_QKV_PACK",
        SettingFate::Excluded("the lowered Metal path, which sharding refuses (CPU only)"),
    ),
    (
        "LARQL_LOWERED_HOST_ARGMAX",
        SettingFate::Excluded("the lowered Metal path, which sharding refuses (CPU only)"),
    ),
    (
        "LARQL_LOWERED_GATHER",
        SettingFate::Excluded("the lowered Metal path, which sharding refuses (CPU only)"),
    ),
    (
        "LARQL_ROUTED_BLOCK",
        SettingFate::Excluded(
            "the lowered Metal VERIFY-N path, which sharding refuses (CPU only); its two \
             arms are bit-identical (compute-metal routed_stack_block_with_the_grouped_\
             expert_pass_matches_the_per_row_block)",
        ),
    ),
    // Operator ablations: they CHANGE the model's output by design, and
    // exist only on the lowered Metal path. Excluded because that path
    // cannot be sharded; a device shard would have to bring them into
    // the identity first (RESIDUAL-BUS-2 §5).
    (
        "LARQL_ABLATE_QUERY_SCALE",
        SettingFate::Excluded("a value-changing ablation on the lowered Metal path only"),
    ),
    (
        "LARQL_ABLATE_ROPE",
        SettingFate::Excluded("a value-changing ablation on the lowered Metal path only"),
    ),
    (
        "LARQL_ABLATE_QK_NORM",
        SettingFate::Excluded("a value-changing ablation on the lowered Metal path only"),
    ),
    (
        "LARQL_ABLATE_GATE",
        SettingFate::Excluded("a value-changing ablation on the lowered Metal path only"),
    ),
    (
        "LARQL_ABLATE_POST_NORMS",
        SettingFate::Excluded("a value-changing ablation on the lowered Metal path only"),
    ),
];
