//! **Structured weight-patch ops**: the numerical interpreter of VLP numerics v1 and v2.
//!
//! Four ops edit one row-major BF16 matrix and produce BF16:
//! - `remove_direction`: `w − (α·u)·c` on columns `S`, where `c = uᵀw` is reduced sequentially in `r`;
//! - `add_low_rank`: `w + Σ_k a[:,k]·b[k,:]`, sequential in `k`, starting from the widened base;
//! - `add_low_rank_columns` (v2): the same update confined to columns `J`, with a compact `b` (`K × |J|`), and
//!   an element whose terms all have a zero factor left bit for bit untouched;
//! - `zero_columns`: `0x0000` on columns `S`.
//!
//! The arithmetic follows a versioned numerical profile ([`Profile`]). The language-neutral contracts are
//! `conformance/vlp-numerics/v1/SPEC.md` and `conformance/vlp-numerics/v2/SPEC.md`, and their frozen golden
//! vectors are the acceptance test for this module. This module never changes to fit a vector: a disagreement is a defect here, or a specification
//! defect that needs a new version of the contract.
//!
//! This is the numerics only. Binding a patch to a container, the `.vlp` encoding, the digest, resource limits
//! and the ledger are layered on top of it.
//!
//! **How the arithmetic is kept exact:**
//! - Each step is one IEEE-754 operation in the working format, through `*`, `+` and `-`. Rust never contracts
//!   these into a fused multiply-add.
//! - Nothing here uses BLAS, iterator `sum()` or `mul_add`, except where a named [`Mutation`] calls for it.
//! - Subnormals must be gradual. [`check_float_environment`] refuses to run under flush-to-zero or
//!   denormals-are-zero.

mod arith;
mod env;
mod ops;
mod profile;
mod refusal;

pub use env::{check_float_environment, Canaries, FloatEnvironmentError};
pub use ops::{run, run_with};
pub use profile::Profile;
pub use refusal::{Refusal, RefusalCode};

/// Why [`run`] produced no bytes: the op was refused under the contract, or this platform cannot run the
/// contract's arithmetic exactly.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StructuredError {
    #[error(transparent)]
    Refused(#[from] Refusal),
    #[error(transparent)]
    Environment(#[from] FloatEnvironmentError),
}

impl StructuredError {
    /// The contract's refusal code, when the op itself was refused.
    pub fn refusal_code(&self) -> Option<RefusalCode> {
        match self {
            StructuredError::Refused(r) => Some(r.code),
            StructuredError::Environment(_) => None,
        }
    }
}

/// One element override, applied after computing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fixup {
    pub row: usize,
    pub col: usize,
    pub bits: u16,
}

/// An operand vector as stored bit patterns. The width must match the profile's working format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandBits {
    F64(Vec<u64>),
    F32(Vec<u32>),
}

impl OperandBits {
    pub fn len(&self) -> usize {
        match self {
            OperandBits::F64(v) => v.len(),
            OperandBits::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A scalar operand as a stored bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBits {
    F64(u64),
    F32(u32),
}

/// What the op computes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpKind {
    RemoveDirection {
        u: OperandBits,
        alpha: ScalarBits,
        columns: Vec<usize>,
    },
    AddLowRank {
        rank: usize,
        /// `rows × rank`, row-major.
        a: OperandBits,
        /// `rank × cols`, row-major.
        b: OperandBits,
    },
    /// VLP numerics v2.
    AddLowRankColumns {
        rank: usize,
        /// `rows × rank`, row-major.
        a: OperandBits,
        /// `rank × columns.len()`, row-major and compact: `b[k, p]` is the factor for column `columns[p]`.
        b: OperandBits,
        /// `J`: strictly increasing column indices (the input axis).
        columns: Vec<usize>,
    },
    ZeroColumns {
        columns: Vec<usize>,
    },
}

/// One structured op, as declared. Nothing in it is trusted until [`run`] has checked it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredOp {
    /// The profile id, such as `f64-seq-rne1/v1`. Required for the arithmetic ops and forbidden for
    /// `zero_columns`.
    pub numerics: Option<String>,
    /// The declared `[rows, cols]`.
    pub shape: [usize; 2],
    pub kind: OpKind,
    /// Sorted by (row, col), with no duplicates.
    pub fixups: Vec<Fixup>,
}

/// What the edited tensor is, where that changes whether an op may touch it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    Ordinary,
    /// The token embedding, which doubles as the tied output head. Refused.
    TokenEmbedding,
}

/// The base tensor as stored: row-major BF16 bits.
#[derive(Debug, Clone, Copy)]
pub struct BaseTensor<'a> {
    pub shape: [usize; 2],
    pub bits: &'a [u16],
    pub role: TensorRole,
}

/// A deliberately wrong semantics, threaded through the production path so that a test can show it changes
/// bytes. Every public entry point except [`run_with`] passes [`Mutation::None`]. The variants are the
/// contracts' named wrong semantics (v1 SPEC §5, v2 SPEC §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    None,
    FmaInReduction,
    ReversedReduction,
    PairwiseReduction,
    AlphaTimesUc,
    F32Accumulation,
    SingleRounding,
    DoubleRounding,
    FtzDaz,
    Bf16OutputFlush,
    LowRankFromZero,
    ZeroKeepsSign,
    /// v2: `add_low_rank_columns` run as v1 `add_low_rank` on a `b` zero-padded to every column.
    ZeroPaddedFull,
    /// v2: an element whose terms are all zero is still computed, so `(−0) + (+0) = +0`.
    ZeroUpdateNotKept,
}
