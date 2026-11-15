//! A tensor's name in a weight edit.
//!
//! A tensor is named by its name in its segment's tensor table (`26.mlp.down_proj.weight`). Where that name is
//! held by more than one canonical representation, for example `weight` in both the token embedding and the final
//! norm, it is qualified by the representation that holds it: `target.embedding@BF16::weight`.
//!
//! A `.vlp` op always records the bare name together with its source binding, which names the representation, so a
//! qualified name never reaches a patch file or a ledger. [`qualified`] rebuilds the qualified name from an op
//! wherever the op's tensor has to be found again.

/// Separates the representation from the tensor in a qualified name.
pub const QUALIFIER: &str = "::";

/// A parsed tensor name: the representation it is qualified by, if any, and the name in the segment table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorName<'a> {
    pub representation: Option<&'a str>,
    pub tensor: &'a str,
}

impl<'a> TensorName<'a> {
    /// Split `name` at the first [`QUALIFIER`]. A name without one is bare.
    pub fn parse(name: &'a str) -> Self {
        match name.split_once(QUALIFIER) {
            Some((representation, tensor)) => Self {
                representation: Some(representation),
                tensor,
            },
            None => Self {
                representation: None,
                tensor: name,
            },
        }
    }

    /// Whether `tensor` held by `representation` is the tensor this name names.
    pub fn names(&self, representation: &str, tensor: &str) -> bool {
        self.tensor == tensor && self.representation.is_none_or(|r| r == representation)
    }
}

/// The qualified name of `tensor` held by `representation`.
pub fn qualified(representation: &str, tensor: &str) -> String {
    format!("{representation}{QUALIFIER}{tensor}")
}
