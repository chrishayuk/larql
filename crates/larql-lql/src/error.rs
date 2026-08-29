/// LQL error types.

#[derive(Debug, thiserror::Error)]
pub enum LqlError {
    #[error("No backend loaded. Run USE \"path.vindex\" first.")]
    NoBackend,

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Mutation requires a vindex. Run EXTRACT first.")]
    MutationRequiresVindex,

    /// A capability profile declined the statement after parsing,
    /// before execution. Distinct from `Execution` so an embedding
    /// surface can map it faithfully (a server answers 403, not 500):
    /// nothing failed — the session's profile does not serve this.
    #[error(
        "{statement} is not available under the {profile} profile. This profile serves: {served}."
    )]
    Refused {
        profile: String,
        statement: String,
        served: String,
    },
}

impl LqlError {
    /// Build an `Execution` variant with a `"context: cause"` message.
    ///
    /// Accepts `&str` or owned `String` for `ctx` so callers can pass a
    /// formatted message (`format!("connect to {url}")`) without an
    /// intermediate `.as_str()` dance.
    pub fn exec(ctx: impl Into<String>, cause: impl std::fmt::Display) -> Self {
        LqlError::Execution(format!("{}: {cause}", ctx.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_concatenates_context_and_cause() {
        let err = LqlError::exec("loading vindex", "file missing");
        assert_eq!(
            err.to_string(),
            "Execution error: loading vindex: file missing"
        );
    }

    #[test]
    fn exec_accepts_owned_string_context() {
        // Regression: callers need to pass `format!(…)` results directly.
        let url = "http://localhost:8080";
        let err = LqlError::exec(format!("failed to connect to {url}"), "io error");
        assert!(err
            .to_string()
            .contains("failed to connect to http://localhost:8080"));
        assert!(err.to_string().contains("io error"));
    }

    #[test]
    fn no_backend_display_message() {
        assert_eq!(
            LqlError::NoBackend.to_string(),
            "No backend loaded. Run USE \"path.vindex\" first."
        );
    }

    #[test]
    fn mutation_requires_vindex_display() {
        assert_eq!(
            LqlError::MutationRequiresVindex.to_string(),
            "Mutation requires a vindex. Run EXTRACT first.",
        );
    }
}
