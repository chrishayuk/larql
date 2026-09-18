//! **A declared experiment identity, checked before a run may measure.**
//!
//! REAL-EVIDENCE-1's first smoke recorded a stream that was hashed,
//! replayable and internally correct — and about the wrong intervention.
//! The shell carried `LARQL_MLA_Q8_LAYER=23,26` and `LARQL_LMHEAD_Q8=1`
//! from an earlier arm, so the run measured `...-mla23-26-headq8` while
//! the programme believed it was reproducing the historical flagship arm
//! `kda-q8-l20-21-22-24-25-x-kimi-map-l20-26q80`. Every integrity check
//! passed, because integrity checks verify that a recorder faithfully
//! records what it is given — not that it was given the right experiment.
//!
//! **Artifact integrity is not experimental identity.** So the identity
//! is DECLARED, in a file the programme commits, and the run compares
//! what it resolved against that declaration before it spends GPU time.
//! Any difference is a refusal, not a warning.
//!
//! Two stages, because the facts become known at two moments: `run`,
//! `expert_candidate` and `scope` are resolved from the environment and
//! the overlay index before any layer is loaded; the byte footprint of
//! the requant is known only once the candidate arm has been built.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::observation_stream::RuntimeScope;

/// Points at a [`DeclaredIdentity`] JSON file. Unset, the run is
/// unchecked and says so; set, every field must match or the run stops.
pub const EXPECT_IDENTITY_ENV: &str = "LARQL_Q2A_EXPECT_IDENTITY";

/// The candidate arm's re-encoded projections, before and after.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteFootprint {
    pub bf16: usize,
    pub q8_0: usize,
}

/// What the programme says the experiment IS. Every field is one the
/// historical reports carry, so a declaration can be checked against
/// the run it claims to reproduce as well as against the run it gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclaredIdentity {
    /// The report's `run` label, e.g.
    /// `kda-q8-l20-21-22-24-25-x-kimi-map-l20-26q80`.
    pub run: String,
    /// The compiled expert overlay's map name, if one is bound.
    pub expert_candidate: Option<String>,
    /// The runtime requant scope. Order-insensitive; `raw_env` ignored.
    pub scope: RuntimeScope,
    pub bytes: ByteFootprint,
    /// Where the declaration came from — historical report paths, bank
    /// hashes, dates. Never compared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
}

/// Every field that differed, named with both values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityRefusal {
    pub differences: Vec<String>,
}

impl fmt::Display for IdentityRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "the run's resolved identity is not the declared experiment; \
             {} field(s) differ:",
            self.differences.len()
        )?;
        for d in &self.differences {
            writeln!(f, "  {d}")?;
        }
        Ok(())
    }
}

fn differ<T: fmt::Debug>(field: &str, declared: T, resolved: T) -> String {
    format!("{field}: declared {declared:?}, resolved {resolved:?}")
}

impl DeclaredIdentity {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("declared identity {}: {e}", path.display()))?;
        let mut d: Self = serde_json::from_slice(&bytes)
            .map_err(|e| format!("declared identity {}: {e}", path.display()))?;
        d.scope = d.scope.normalised();
        Ok(d)
    }

    /// Stage 1 — before any layer is loaded.
    pub fn check_transition(
        &self,
        run: &str,
        expert_candidate: Option<&str>,
        scope: &RuntimeScope,
    ) -> Result<(), IdentityRefusal> {
        let mut differences = Vec::new();
        if self.run != run {
            differences.push(differ("run", self.run.as_str(), run));
        }
        if self.expert_candidate.as_deref() != expert_candidate {
            differences.push(differ(
                "expert_candidate",
                self.expert_candidate.as_deref(),
                expert_candidate,
            ));
        }
        if !self.scope.same_transition(scope) {
            differences.push(differ(
                "scope",
                self.scope.describe(),
                scope.normalised().describe(),
            ));
        }
        refuse_if_any(differences)
    }

    /// Stage 2 — after the candidate arm is built, before it measures.
    pub fn check_bytes(&self, bf16: usize, q8_0: usize) -> Result<(), IdentityRefusal> {
        let mut differences = Vec::new();
        if self.bytes.bf16 != bf16 {
            differences.push(differ("bytes.bf16", self.bytes.bf16, bf16));
        }
        if self.bytes.q8_0 != q8_0 {
            differences.push(differ("bytes.q8_0", self.bytes.q8_0, q8_0));
        }
        refuse_if_any(differences)
    }
}

fn refuse_if_any(differences: Vec<String>) -> Result<(), IdentityRefusal> {
    if differences.is_empty() {
        Ok(())
    } else {
        Err(IdentityRefusal { differences })
    }
}

#[cfg(test)]
mod tests;
