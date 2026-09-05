//! What a codec can be asked for, declared so the planner refuses by name.
//!
//! `row_random_access: bool` was the first cut and the property is not
//! boolean: an entropy-coded stream is sequential, a K-quant is addressable
//! at its 256-element block, a float is addressable anywhere. Walk FFN needs
//! arbitrary rows and a stream-sequential codec cannot serve it — the
//! preflight question is "does the representation provide what the plan
//! requires", asked here, rather than a kernel discovering it.

use super::error::CodecError;

/// How finely stored bytes can be addressed without decoding what
/// precedes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessGranularity {
    /// Only from the start: a decode window advances through the stream.
    Sequential,
    /// Any whole block of `elems` elements.
    BlockRandom { elems: usize },
    /// Any row, because rows are whole blocks by construction.
    RowRandom,
    /// Any element.
    ElementRandom,
}

impl AccessGranularity {
    /// Rank on the ladder; a finer granularity provides every coarser one.
    const fn rank(self) -> u8 {
        match self {
            Self::Sequential => 0,
            Self::BlockRandom { .. } => 1,
            Self::RowRandom => 2,
            Self::ElementRandom => 3,
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::Sequential => "sequential".into(),
            Self::BlockRandom { elems } => format!("block-random ({elems} elements)"),
            Self::RowRandom => "row-random".into(),
            Self::ElementRandom => "element-random".into(),
        }
    }

    /// Whether this granularity serves `required`.
    ///
    /// Block-random does not serve a row requirement on its own: whether a
    /// row is a whole number of blocks is a fact about the tensor, and a
    /// codec whose rows are block-aligned by construction declares
    /// [`Self::RowRandom`] instead.
    pub fn provides(self, required: RequiredAccess) -> bool {
        self.rank() >= required.granularity().rank()
    }
}

/// What an execution plan needs to address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredAccess {
    /// The whole tensor, front to back.
    Sequential,
    /// Arbitrary rows — an expert slot, a Walk FFN feature.
    RowRandom,
    /// Arbitrary elements — a browse gather.
    ElementRandom,
}

impl RequiredAccess {
    const fn granularity(self) -> AccessGranularity {
        match self {
            Self::Sequential => AccessGranularity::Sequential,
            Self::RowRandom => AccessGranularity::RowRandom,
            Self::ElementRandom => AccessGranularity::ElementRandom,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::RowRandom => "row-random",
            Self::ElementRandom => "element-random",
        }
    }
}

/// The access, grouping and alignment one codec declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodecCapabilities {
    pub access: AccessGranularity,
    /// Elements sharing one scale — 1 where nothing is shared.
    pub group_elems: usize,
    /// `k` must be a whole number of these.
    pub row_align_elems: usize,
    /// Byte alignment the widest field in the stored bytes needs.
    pub physical_align_bytes: usize,
}

impl CodecCapabilities {
    /// Refuse, by name, a plan this representation cannot address.
    pub fn require(&self, required: RequiredAccess, label: &str) -> Result<(), CodecError> {
        if self.access.provides(required) {
            return Ok(());
        }
        Err(CodecError::AccessRefused {
            label: label.into(),
            provided: self.access.name(),
            required: required.name().into(),
        })
    }

    /// Whether a row of `k` elements is addressable under this codec.
    pub fn admits_k(&self, k: usize) -> bool {
        self.row_align_elems > 0 && k.is_multiple_of(self.row_align_elems)
    }
}
