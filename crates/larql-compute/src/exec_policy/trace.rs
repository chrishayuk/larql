//! On-disk form of a recorded oracle trace: the bridge between an
//! offline harness that DECIDED which expert groups were safe to delete
//! and a serve-path run that REPLAYS those decisions.
//!
//! # Why a file, and why it carries provenance
//!
//! BW-C5's oracle decisions are produced by a CPU KV-fork harness that
//! takes tens of minutes and needs a real model. The serve path cannot
//! recompute them, so the only way to run the same policy in both places
//! is to write them down. That much is obvious.
//!
//! Less obvious, and the reason this is not just a list of pairs: a
//! trace divorced from what produced it licenses claims it has no right
//! to. `(21, 7)` means "skip layer 21 at decode step 7" — but whether
//! that was chosen by a 6-token lookahead or a 16-token one, against
//! which prompt, at what generation length, decides what a replay
//! result can be compared against. So every writer emits a provenance
//! header and every reader preserves it, and a trace with no provenance
//! is readable but says so.
//!
//! # Format
//!
//! Line-oriented text, because it has to be greppable and diffable by
//! hand when a replay disagrees with its source run:
//!
//! ```text
//! # larql-exec-trace v1
//! # source: bwc5_oracle_repeated_policy lookahead=6 prompts=8 generation_length=32
//! 21 0
//! 21 1
//! 21 3
//! ```
//!
//! `layer step`, whitespace-separated, one per line. `#` starts a
//! comment; blank lines are ignored. Any line that is neither a comment
//! nor a well-formed pair is an ERROR, not a skipped line — a trace that
//! silently drops half its decisions would produce a replay that looks
//! like a weaker policy rather than a broken file.

use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Format marker, written first and checked on read.
const MAGIC: &str = "# larql-exec-trace v1";
/// Prefix of the provenance line.
const SOURCE_PREFIX: &str = "# source:";

/// A recorded set of `(layer, step)` skip decisions plus what produced
/// them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trace {
    /// Free-text description of the run that produced this trace.
    /// `None` when the file carried no provenance line — readable, but
    /// a result quoted from it cannot say what policy it replays.
    pub source: Option<String>,
    /// `(layer, step)` pairs to skip. A `BTreeSet` so the written form
    /// is deterministic and two traces diff cleanly.
    pub skips: BTreeSet<(usize, u64)>,
}

impl Trace {
    /// A trace with declared provenance and no decisions yet.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: Some(source.into()),
            skips: BTreeSet::new(),
        }
    }

    /// Record one skip decision.
    pub fn record(&mut self, layer: usize, step: u64) {
        self.skips.insert((layer, step));
    }

    pub fn len(&self) -> usize {
        self.skips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skips.is_empty()
    }

    /// Render to the on-disk form.
    pub fn render(&self) -> String {
        let mut out = String::from(MAGIC);
        out.push('\n');
        if let Some(src) = &self.source {
            let _ = writeln!(out, "{SOURCE_PREFIX} {src}");
        }
        for (layer, step) in &self.skips {
            let _ = writeln!(out, "{layer} {step}");
        }
        out
    }

    /// Parse the on-disk form.
    ///
    /// Every error names the offending line, because the failure mode
    /// this guards against is a replay that quietly covers fewer
    /// decisions than its source run and reads as a weaker policy.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut trace = Trace::default();
        let mut saw_magic = false;
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('#') {
                if line == MAGIC {
                    saw_magic = true;
                } else if let Some(src) = line.strip_prefix(SOURCE_PREFIX) {
                    trace.source = Some(src.trim().to_string());
                } else {
                    let _ = rest;
                }
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(l), Some(s), None) = (parts.next(), parts.next(), parts.next()) else {
                return Err(format!(
                    "line {}: expected `layer step`, got {line:?}",
                    n + 1
                ));
            };
            let layer: usize = l
                .parse()
                .map_err(|_| format!("line {}: layer {l:?} is not an integer", n + 1))?;
            let step: u64 = s
                .parse()
                .map_err(|_| format!("line {}: step {s:?} is not an integer", n + 1))?;
            trace.record(layer, step);
        }
        if !saw_magic {
            return Err(format!(
                "missing format marker — first line must be {MAGIC:?}"
            ));
        }
        Ok(trace)
    }

    /// Read a trace from disk.
    pub fn read(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read exec trace {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Write a trace to disk.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let path = path.as_ref();
        std::fs::write(path, self.render())
            .map_err(|e| format!("cannot write exec trace {}: {e}", path.display()))
    }
}

#[cfg(test)]
#[path = "tests/trace.rs"]
mod tests;
