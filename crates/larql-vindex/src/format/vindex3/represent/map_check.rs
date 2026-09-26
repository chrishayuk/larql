//! **Every exception in a precision map must decide something.**
//!
//! [`PrecisionMap`] resolves by first match, and that grammar accepts
//! three programs silently that no author meant to write:
//!
//! ```text
//! v-proj  -> source            matches no tensor        (a typo)
//! v_proj  -> source
//! v_proj layers 0-9 -> Q8_0    every match already won  (dead rule)
//! *       -> source
//! q_proj  -> Q8_0              after a catch-all        (dead tail)
//! ```
//!
//! Each compiles, records a map that states a decision nobody gets, and
//! makes the recorded program a worse description of the bytes than the
//! map with that line deleted. A person writing maps by hand trips over
//! this rarely; a search that emits maps will do it routinely, and the
//! refusal is how its bugs surface instead of hiding in provenance.
//!
//! **Overlap is not the defect.** `v_proj layers 0-9 -> source` then
//! `v_proj -> NVFP4` overlap, and the second rule still governs every
//! later `v_proj`. The invariant is that each exception WINS at least one
//! eligible tensor — which is why this is judged against the tensors a
//! compilation actually covers, not by reasoning about selectors: a map
//! alone cannot know that `layers 40-47` names depth a 36-layer model
//! does not have.
//!
//! **The surface must be the whole program's scope.** A caller that
//! checks one shard of a compilation against the full map will refuse
//! exceptions that govern other shards. Check where the complete set of
//! tensors the map is recorded for is known.

use std::fmt;

use super::map::{Exception, PrecisionMap};
use super::policy::Role;

/// One exception that decides nothing on the checked surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapDefect {
    /// No eligible tensor matches the exception's selector at all.
    Unmatched { index: usize, selector: String },
    /// Every eligible tensor it matches was already decided by earlier
    /// exceptions, listed in `shadowed_by`.
    Shadowed {
        index: usize,
        selector: String,
        shadowed_by: Vec<usize>,
    },
    /// An exception that matches every tensor, with exceptions after it
    /// that can therefore never be reached. Structural: refused whatever
    /// the surface.
    CatchAllNotLast { index: usize, later: Vec<usize> },
}

impl fmt::Display for MapDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unmatched { index, selector } => {
                write!(
                    f,
                    "exception #{index} `{selector}` matches no eligible tensor"
                )
            }
            Self::Shadowed {
                index,
                selector,
                shadowed_by,
            } => write!(
                f,
                "exception #{index} `{selector}` decides nothing: every tensor it matches is \
                 decided first by exception(s) {shadowed_by:?}"
            ),
            Self::CatchAllNotLast { index, later } => write!(
                f,
                "exception #{index} matches every tensor, so exception(s) {later:?} after it \
                 can never apply"
            ),
        }
    }
}

/// A map refused against a surface, with every defect found — not only
/// the first, so one run reports the whole repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapRefusal {
    pub map: String,
    pub defects: Vec<MapDefect>,
}

impl fmt::Display for MapRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "precision map `{}` refused:", self.map)?;
        for d in &self.defects {
            write!(f, "\n  {d}")?;
        }
        Ok(())
    }
}

impl std::error::Error for MapRefusal {}

/// What each exception did on the surface that passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionCoverage {
    /// Per exception, in declaration order: eligible tensors its selector
    /// matches, and how many of those it is the first match for.
    pub matched: Vec<usize>,
    pub decided: Vec<usize>,
}

impl PrecisionMap {
    /// Check this map against the `(role, tensor)` surface it is about to
    /// be recorded for.
    ///
    /// Only tensors of a role the map compiles count: [`Self::resolve`]
    /// answers source precision for any other role before consulting an
    /// exception, so an exception matching only those decides nothing.
    pub fn check_against<'a>(
        &self,
        surface: impl IntoIterator<Item = (Role, &'a str)>,
    ) -> Result<ExceptionCoverage, MapRefusal> {
        let n = self.exceptions.len();
        let mut matched = vec![0usize; n];
        let mut decided = vec![0usize; n];
        // For an exception that never wins: which earlier ones took its
        // tensors, so the refusal names the rules that shadow it.
        let mut taken_by: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (role, tensor) in surface {
            if !self.roles.iter().any(|r| r == role.name()) {
                continue;
            }
            let mut winner = None;
            for (i, e) in self.exceptions.iter().enumerate() {
                if !e.matches(tensor) {
                    continue;
                }
                matched[i] += 1;
                match winner {
                    None => {
                        winner = Some(i);
                        decided[i] += 1;
                    }
                    Some(w) if !taken_by[i].contains(&w) => taken_by[i].push(w),
                    Some(_) => {}
                }
            }
        }

        let mut defects = Vec::new();
        for (i, e) in self.exceptions.iter().enumerate() {
            if is_catch_all(e) && i + 1 < n {
                defects.push(MapDefect::CatchAllNotLast {
                    index: i,
                    later: (i + 1..n).collect(),
                });
            }
            if matched[i] == 0 {
                defects.push(MapDefect::Unmatched {
                    index: i,
                    selector: e.describe(),
                });
            } else if decided[i] == 0 {
                let mut shadowed_by = std::mem::take(&mut taken_by[i]);
                shadowed_by.sort_unstable();
                defects.push(MapDefect::Shadowed {
                    index: i,
                    selector: e.describe(),
                    shadowed_by,
                });
            }
        }

        if defects.is_empty() {
            Ok(ExceptionCoverage { matched, decided })
        } else {
            Err(MapRefusal {
                map: self.name.clone(),
                defects,
            })
        }
    }
}

/// An exception with no selector matches every tensor —
/// [`Exception::matches`] documents it as the way to write "compile
/// nothing". Legitimate only as the last rule.
fn is_catch_all(e: &Exception) -> bool {
    e.projection.is_none() && e.layers.is_none()
}

#[cfg(test)]
#[path = "map_check_tests.rs"]
mod tests;
