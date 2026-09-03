//! **What the measurement MEANS, not which binary took it.**
//!
//! The tempting identity is a version string — `"diagnostic-kl-v1"` —
//! and it fails in both directions. It moves when a refactor changes
//! nothing observable, splitting a state's evidence for no reason; and
//! it stays put when someone changes the truncation, silently making two
//! incomparable readings look like a repeat.
//!
//! ```text
//! implementation refactor, same semantics   →  SAME id
//! different truncation                      →  different id
//! different aggregation                     →  different id
//! different token selection                 →  different id
//! different metric definition               →  different id
//! different procedure                       →  different id
//! ```
//!
//! So the identity is a digest of *declared semantics*. Truncation is
//! the concrete case this programme already paid for: the Q2a harness
//! runs at `TOP_N = 2048`, and `QualityBank::min_covered_mass` exists
//! because a KL over a truncation covering a third of the mass is a
//! different measurement from one covering all of it — top-128 covered
//! 0.307 of a first position, top-2048 covered 0.729. Two runs at
//! different `top_n` are not repeats of each other and a key that could
//! not tell them apart would say they were.
//!
//! # The limit, stated rather than papered over
//!
//! A declaration can lie. Change the runner's truncation without
//! changing the [`InstrumentSemantics`] it reports and the id does not
//! move, and nothing here can detect that. The mitigation is
//! construction, not validation: build the declaration FROM the
//! constants the runner uses rather than typing them twice. This module
//! makes that possible and cannot make it mandatory, and a reader should
//! know which of the two they are relying on.

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::surface::{FIELD, SECTION};

/// The canonical-form version every instrument id is computed under.
pub const INSTRUMENT_SEMANTICS_ID_VERSION: &str = "instrument-semantics-id/v1";

/// **What an instrument MEANS**, as a digest of its declared semantics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InstrumentSemanticsId(String);

impl InstrumentSemanticsId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        &self.0[..self.0.len().min(12)]
    }
}

impl std::fmt::Display for InstrumentSemanticsId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The measurement's meaning: every choice that changes what a reading
/// is, and none that do not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentSemantics {
    /// What is computed, e.g. `"kl(baseline || candidate)"`.
    pub metric: String,
    /// Distribution truncation the metric was taken over. `None` is
    /// untruncated and is NOT the same as a large `Some` — one is a
    /// claim about the whole distribution, the other about a window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<u32>,
    /// How per-position readings become one number, e.g.
    /// `"distribution{min,p50,p95,p99,max}"`.
    pub aggregation: String,
    /// Which positions contributed, e.g. `"teacher-forced, all
    /// positions"`.
    pub token_selection: String,
    /// The two-arm procedure the reading was taken under, e.g.
    /// `"q2a-teacher-forced/baseline-vs-overlay"`.
    pub procedure: String,
    /// Free text about the implementation — a commit, a binary, a
    /// machine. **Excluded from the digest**: a refactor that changes
    /// nothing observable must not split a state's evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation_note: Option<String>,
}

impl InstrumentSemantics {
    pub fn new(
        metric: impl Into<String>,
        aggregation: impl Into<String>,
        token_selection: impl Into<String>,
        procedure: impl Into<String>,
    ) -> Self {
        Self {
            metric: metric.into(),
            truncation: None,
            aggregation: aggregation.into(),
            token_selection: token_selection.into(),
            procedure: procedure.into(),
            implementation_note: None,
        }
    }

    /// Declare the truncation. Pass the runner's own constant rather
    /// than a literal, so the declaration cannot drift from the code.
    pub fn truncated_to(mut self, top_n: u32) -> Self {
        self.truncation = Some(top_n);
        self
    }

    pub fn implemented_by(mut self, note: impl Into<String>) -> Self {
        self.implementation_note = Some(note.into());
        self
    }

    /// What this instrument MEANS.
    pub fn id(&self) -> InstrumentSemanticsId {
        let truncation = match self.truncation {
            Some(n) => n.to_string(),
            None => "untruncated".to_string(),
        };
        let input = format!(
            "{INSTRUMENT_SEMANTICS_ID_VERSION}{SECTION}metric={}{FIELD}truncation={truncation}\
             {FIELD}aggregation={}{FIELD}tokens={}{FIELD}procedure={}",
            self.metric, self.aggregation, self.token_selection, self.procedure
        );
        InstrumentSemanticsId(hash_bytes(input.as_bytes()))
    }
}
