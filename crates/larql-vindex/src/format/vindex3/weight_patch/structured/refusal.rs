//! Why a structured op is refused. Each code is named in VLP numerics v1 (SPEC §4). An op either produces
//! every output byte or refuses with exactly one code.

/// The contract's refusal codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefusalCode {
    NumericsInvalid,
    NumericsForbidden,
    ShapeInvalid,
    RankInvalid,
    ColumnsInvalid,
    FixupsInvalid,
    FixupsForbidden,
    FixupNoOp,
    NonFiniteOperand,
    NonFiniteIntermediate,
    Overflow,
    TensorRefused,
}

impl RefusalCode {
    /// Every code, in the order SPEC §4 lists them.
    pub const ALL: [RefusalCode; 12] = [
        RefusalCode::NumericsInvalid,
        RefusalCode::NumericsForbidden,
        RefusalCode::ShapeInvalid,
        RefusalCode::RankInvalid,
        RefusalCode::ColumnsInvalid,
        RefusalCode::FixupsInvalid,
        RefusalCode::FixupsForbidden,
        RefusalCode::FixupNoOp,
        RefusalCode::NonFiniteOperand,
        RefusalCode::NonFiniteIntermediate,
        RefusalCode::Overflow,
        RefusalCode::TensorRefused,
    ];

    /// The code as the contract spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            RefusalCode::NumericsInvalid => "numerics_invalid",
            RefusalCode::NumericsForbidden => "numerics_forbidden",
            RefusalCode::ShapeInvalid => "shape_invalid",
            RefusalCode::RankInvalid => "rank_invalid",
            RefusalCode::ColumnsInvalid => "columns_invalid",
            RefusalCode::FixupsInvalid => "fixups_invalid",
            RefusalCode::FixupsForbidden => "fixups_forbidden",
            RefusalCode::FixupNoOp => "fixup_no_op",
            RefusalCode::NonFiniteOperand => "non_finite_operand",
            RefusalCode::NonFiniteIntermediate => "non_finite_intermediate",
            RefusalCode::Overflow => "overflow",
            RefusalCode::TensorRefused => "tensor_refused",
        }
    }
}

/// A refusal: its code, and a detail for people (never compared).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("structured op refused ({}): {detail}", code.as_str())]
pub struct Refusal {
    pub code: RefusalCode,
    pub detail: String,
}

impl Refusal {
    pub(super) fn new(code: RefusalCode, detail: impl Into<String>) -> Self {
        Refusal {
            code,
            detail: detail.into(),
        }
    }
}
