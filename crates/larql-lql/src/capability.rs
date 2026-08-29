//! Capability profiles — who may execute what, decided after parsing
//! and before execution.
//!
//! A profile is not a string filter over statement text and not a
//! second grammar: the full language parses, and the profile then
//! judges the **parsed** statement at the head of `Session::execute`,
//! before the remote fork, so no transport bypasses it. A statement a
//! profile refuses dies with a [`LqlError::Refused`] that names the
//! profile and what it serves — a capability statement, not an apology
//! (§4.2's contract).
//!
//! What is deliberately not here: any notion of users, roles, or
//! authentication. A profile constrains a *session*; who gets which
//! profile is the embedding surface's decision (a public server hands
//! out `PublicExplorer`, the REPL keeps `Full`).
//!
//! The policy mirrors the plan capability module's fail-closed rule:
//! the `PublicExplorer` arm is an **exhaustive match** over
//! [`Statement`], so a future variant does not compile until someone
//! decides whether the public surface serves it. An allowlist in a set
//! would let a new statement default to allowed — the one wrong
//! default for a public endpoint.

use crate::ast::Statement;
use crate::error::LqlError;

/// The statements `PUBLIC_EXPLORER` serves — part of the refusal
/// contract, printed whenever the profile declines one.
pub const PUBLIC_EXPLORER_SERVES: &str =
    "SHOW COMPONENTS/REPRESENTATIONS/PROVENANCE/AUTHORITY/RELATIONS/LAYERS/FEATURES/ENTITIES/MODELS, \
     DESCRIBE, WALK, SELECT, EXPLAIN, STATS, INFER [TOP n] [GENERATE <= 32]";

/// What a session is allowed to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CapabilityProfile {
    /// The whole language. The REPL's and the default.
    #[default]
    Full,
    /// The public read surface: introspection, browse queries, and
    /// bounded execution over an already-bound container. Mutation,
    /// lifecycle, and anything that names a filesystem path is refused
    /// after parsing, before execution — the verbs still parse, they
    /// just cannot happen.
    PublicExplorer,
}

impl CapabilityProfile {
    /// The longest `GENERATE` continuation the public profile executes.
    /// Single-step `INFER` is always in; unbounded generation is a cost
    /// lever a public endpoint must not hand out.
    pub const PUBLIC_EXPLORER_MAX_GENERATE: u32 = 32;

    /// The profile's wire name — what a refusal and a server banner print.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Full => "FULL",
            Self::PublicExplorer => "PUBLIC_EXPLORER",
        }
    }

    /// Judge a parsed statement. `Ok(())` means the profile permits it
    /// (execution may still refuse on backend capability grounds);
    /// `Err(Refused)` means it must not begin. Pipe trees are judged
    /// whole: one refused leg refuses the statement before any leg runs.
    pub fn check(&self, stmt: &Statement) -> Result<(), LqlError> {
        match self {
            Self::Full => Ok(()),
            Self::PublicExplorer => public_explorer_check(stmt),
        }
    }
}

/// The `PUBLIC_EXPLORER` judgement. Exhaustive over [`Statement`] on
/// purpose — see the module docs for why a set would be the wrong shape.
fn public_explorer_check(stmt: &Statement) -> Result<(), LqlError> {
    use Statement::*;
    match stmt {
        // ── The container answering for itself: introspection + browse ──
        Describe { .. }
        | Select { .. }
        | Walk { .. }
        | Explain { .. }
        | ShowRelations { .. }
        | ShowLayers { .. }
        | ShowFeatures { .. }
        | ShowEntities { .. }
        | ShowModels
        | ShowComponents
        | ShowRepresentations { .. }
        | ShowProvenance { .. }
        | ShowAuthority
        | Stats { .. } => Ok(()),

        // ── Bounded execution: the model is enterable, not rentable ──
        Infer { generate, .. } => match generate {
            Some(n) if *n > CapabilityProfile::PUBLIC_EXPLORER_MAX_GENERATE => {
                Err(LqlError::Refused {
                    profile: CapabilityProfile::PublicExplorer.name().into(),
                    statement: format!(
                        "INFER GENERATE {n} (the profile's bound is {})",
                        CapabilityProfile::PUBLIC_EXPLORER_MAX_GENERATE
                    ),
                    served: PUBLIC_EXPLORER_SERVES.into(),
                })
            }
            _ => Ok(()),
        },

        // ── A pipe is judged whole ──
        Pipe { left, right } => {
            public_explorer_check(left)?;
            public_explorer_check(right)
        }

        // ── Everything else is refused by construction: mutation,
        // lifecycle, patches, compaction, tracing, and every statement
        // that names a filesystem path or another artifact ──
        Extract { .. }
        | Compile { .. }
        | Diff { .. }
        | Use { .. }
        | Insert { .. }
        | Delete { .. }
        | Update { .. }
        | Merge { .. }
        | Rebalance { .. }
        | ShowCompactStatus
        | CompactInto { .. }
        | CompactMinor
        | CompactMajor { .. }
        | BeginPatch { .. }
        | SavePatch
        | ApplyPatch { .. }
        | ShowPatches
        | RemovePatch { .. }
        | Trace { .. } => Err(LqlError::Refused {
            profile: CapabilityProfile::PublicExplorer.name().into(),
            statement: stmt.verb().into(),
            served: PUBLIC_EXPLORER_SERVES.into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn check_public(stmt: &str) -> Result<(), LqlError> {
        let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
        CapabilityProfile::PublicExplorer.check(&parsed)
    }

    #[test]
    fn the_full_profile_gates_nothing() {
        for stmt in [
            "DELETE FROM EDGES WHERE layer = 0;",
            "USE \"anything.vindex\";",
            "SHOW COMPONENTS;",
        ] {
            let parsed = parse(stmt).unwrap();
            assert!(CapabilityProfile::Full.check(&parsed).is_ok(), "{stmt}");
        }
    }

    #[test]
    fn public_explorer_permits_the_read_surface() {
        for stmt in [
            "SHOW COMPONENTS;",
            "SHOW REPRESENTATIONS;",
            "SHOW PROVENANCE \"gate\";",
            "SHOW AUTHORITY;",
            "SHOW LAYERS;",
            "SHOW MODELS;",
            "DESCRIBE \"paris\";",
            "WALK \"paris\";",
            "SELECT * FROM EDGES LIMIT 5;",
            "EXPLAIN INFER \"paris\";",
            "STATS;",
            "INFER \"paris\" TOP 3;",
            "INFER \"paris\" GENERATE 32;",
        ] {
            assert!(check_public(stmt).is_ok(), "{stmt} must be permitted");
        }
    }

    #[test]
    fn public_explorer_refuses_mutation_lifecycle_and_paths() {
        for stmt in [
            "USE \"secret.vindex\";",
            "EXTRACT MODEL \"m\" INTO \"out.vindex\";",
            "COMPILE CURRENT INTO VINDEX \"out\";",
            "DIFF CURRENT CURRENT;",
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
            "DELETE FROM EDGES WHERE layer = 0;",
            "UPDATE EDGES SET confidence = 0.5 WHERE layer = 0;",
            "MERGE \"other.vindex\";",
            "REBALANCE;",
            "COMPACT MINOR;",
            "COMPACT INTO VINDEX \"out\";",
            "BEGIN PATCH \"p.vlp\";",
            "SAVE PATCH;",
            "APPLY PATCH \"p.vlp\";",
            "SHOW PATCHES;",
            "REMOVE PATCH \"p.vlp\";",
            "SHOW COMPACT STATUS;",
            "TRACE \"paris\";",
        ] {
            let err = check_public(stmt).expect_err(stmt);
            assert!(
                matches!(err, LqlError::Refused { .. }),
                "{stmt}: must be Refused, got {err}"
            );
            let msg = err.to_string();
            assert!(msg.contains("PUBLIC_EXPLORER"), "{stmt}: {msg}");
            assert!(msg.contains("serves"), "{stmt}: {msg}");
        }
    }

    #[test]
    fn generate_is_bounded_not_banned() {
        assert!(check_public("INFER \"x\" GENERATE 32;").is_ok());
        let err = check_public("INFER \"x\" GENERATE 33;").expect_err("beyond the bound");
        let msg = err.to_string();
        assert!(msg.contains("33") && msg.contains("32"), "{msg}");
    }

    #[test]
    fn a_pipe_is_judged_whole() {
        let err = check_public("STATS |> DELETE FROM EDGES WHERE layer = 0;")
            .expect_err("the refused leg refuses the pipe");
        assert!(matches!(err, LqlError::Refused { .. }), "{err}");
        // Both orders: the refused leg's position must not matter.
        let err = check_public("DELETE FROM EDGES WHERE layer = 0 |> STATS;")
            .expect_err("the refused leg refuses the pipe");
        assert!(matches!(err, LqlError::Refused { .. }), "{err}");
    }
}
