//! Model identity passed to `BoundaryKvEngine` at construction time.
//!
//! Every emitted [`larql_boundary::BoundaryFrame`] carries these identity
//! fields. The engine cannot fabricate them — they are properties of the
//! caller's model + tokenizer state. See `BOUNDARY_REF_PROTOCOL.md` §10.4 for
//! the receiver-side mismatch semantics.

/// Identity fields embedded in every emitted boundary frame.
///
/// `model_revision` is the canonical identity for mismatch checks (it should
/// be a content hash of the weights). `architecture` is a human-readable label
/// only and **must not** be relied upon for matching — two checkpoints of the
/// same architecture can have different residual geometry.
#[derive(Debug, Clone)]
pub struct BoundaryModelIdentity {
    pub model_id: String,
    pub model_revision: String,
    pub tokenizer_revision: String,
    pub architecture: String,
}

impl BoundaryModelIdentity {
    /// Convenience for tests and small examples: build an identity with
    /// minimal placeholder fields. Do not use in production paths — real
    /// callers should pass the live model's actual fingerprint.
    pub fn placeholder(model_id: impl Into<String>) -> Self {
        let id = model_id.into();
        Self {
            model_id: id.clone(),
            model_revision: format!("{id}@unknown"),
            tokenizer_revision: format!("{id}-tokenizer@unknown"),
            architecture: id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_fills_all_fields() {
        let id = BoundaryModelIdentity::placeholder("gemma3-4b");
        assert_eq!(id.model_id, "gemma3-4b");
        assert!(id.model_revision.contains("gemma3-4b"));
        assert!(id.tokenizer_revision.contains("gemma3-4b"));
        assert_eq!(id.architecture, "gemma3-4b");
    }

    #[test]
    fn placeholder_clone_is_independent() {
        let id = BoundaryModelIdentity::placeholder("test");
        let id2 = id.clone();
        assert_eq!(id.model_id, id2.model_id);
        assert_eq!(id.model_revision, id2.model_revision);
    }
}
