//! The versioned numerical profiles (WSP-001 stage-2 amendment 1, A1). A profile fixes every numerical choice;
//! a change to any of them is a new version, never a reinterpretation.

use super::refusal::{Refusal, RefusalCode};

/// A numerical profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// f64 working format; one BF16 round-to-nearest-even at the end. p50c's structured op.
    F64SeqRne1,
    /// f64 working format; then f32 RNE, then BF16 RNE. The published `apply_edit.py` low-rank path (p10c).
    F64SeqRne2,
    /// f32 working format; one BF16 RNE at the end. Report-only for p50c.
    F32SeqRne1,
}

impl Profile {
    pub const ALL: [Profile; 3] = [
        Profile::F64SeqRne1,
        Profile::F64SeqRne2,
        Profile::F32SeqRne1,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Profile::F64SeqRne1 => "f64-seq-rne1/v1",
            Profile::F64SeqRne2 => "f64-seq-rne2/v1",
            Profile::F32SeqRne1 => "f32-seq-rne1/v1",
        }
    }

    /// The profile named `id`, or `numerics_invalid` for any other string. There is no default.
    pub fn parse(id: Option<&str>) -> Result<Profile, Refusal> {
        let id = id.ok_or_else(|| {
            Refusal::new(RefusalCode::NumericsInvalid, "no numerics profile declared")
        })?;
        Profile::ALL
            .into_iter()
            .find(|p| p.id() == id)
            .ok_or_else(|| {
                Refusal::new(
                    RefusalCode::NumericsInvalid,
                    format!("unknown numerics profile `{id}`"),
                )
            })
    }

    /// Whether the working format is f64 (else f32).
    pub fn works_in_f64(self) -> bool {
        !matches!(self, Profile::F32SeqRne1)
    }

    /// Whether the chain to BF16 passes through an f32 rounding first.
    pub fn rounds_through_f32(self) -> bool {
        matches!(self, Profile::F64SeqRne2)
    }
}
