//! Per-component census of the resolved attention table.
//!
//! One layer, one bucket. The buckets exist so the summary can say what a
//! stack actually resolved to — and, where it resolved to nothing
//! executable, say *which kind* of nothing. Two distinct failures live
//! here and reporting them as one loses the difference:
//!
//! - a **recurrence whose operator this build cannot identify** — the
//!   checkpoint named a linear-attention layer and this build has no
//!   operator for that family (GLM-5.3-Flash's 34 KDA layers, Kimi
//!   Linear's 20);
//! - a **declared spelling with no execution vocabulary at all** —
//!   `deepseek_sparse_attention`, GLM-5.3-Flash's other 11.
//!
//! Both block. Only the first is known to be recurrent, and a KV planner
//! reading the summary needs that distinction: an unidentified recurrence
//! has no per-position prefix to bound, whereas an unknown spelling might.

use super::super::graph::policy::{AttentionLayerPolicy, AttentionSpan};

/// Layer counts by resolved kind. Disjoint by construction, so
/// [`Self::full`] is a real remainder rather than an independent count
/// that could drift.
pub(super) struct AttentionCensus {
    pub sliding: usize,
    pub full: usize,
    /// Gated DeltaNet recurrences.
    pub gated_delta: usize,
    /// Kimi Delta Attention recurrences.
    pub kda: usize,
    /// Mamba2/SSD recurrences.
    pub mamba2: usize,
    /// Recurrences it cannot identify.
    pub unidentified_recurrence: usize,
    /// Declared spellings outside every vocabulary.
    pub unexpressed: usize,
    pub nope: usize,
    /// Layers whose operator this build represents but cannot run.
    ///
    /// Not a subset of any bucket above — it cuts across them, because
    /// representation and executability are independent facts. A stack can
    /// be fully represented and wholly unrunnable.
    pub represented_not_executable: usize,
}

impl AttentionCensus {
    pub fn of(table: &[AttentionLayerPolicy]) -> Self {
        let gated_delta = table.iter().filter(|l| l.operator.is_gated_delta()).count();
        let kda = table.iter().filter(|l| l.operator.is_kda()).count();
        let mamba2 = table.iter().filter(|l| l.operator.is_mamba2()).count();
        let unidentified_recurrence = table
            .iter()
            .filter(|l| l.operator.is_unidentified_recurrence())
            .count();
        // An unidentified recurrence never round-trips, so it would fall in
        // here too; it is excluded because it is already counted as itself
        // and a layer must contribute to exactly one bucket.
        let unexpressed = table
            .iter()
            .filter(|l| !l.matches_declaration() && !l.operator.is_unidentified_recurrence())
            .count();
        let sliding = table
            .iter()
            .filter(|l| l.matches_declaration() && l.span == Some(AttentionSpan::Sliding))
            .count();
        Self {
            sliding,
            full: table.len()
                - sliding
                - gated_delta
                - kda
                - mamba2
                - unidentified_recurrence
                - unexpressed,
            gated_delta,
            kda,
            mamba2,
            unidentified_recurrence,
            unexpressed,
            nope: table
                .iter()
                .filter(|l| l.position == larql_models::config::PositionPolicy::None)
                .count(),
            // Counted over layers whose operator IS identified: an
            // unidentified recurrence is not "represented but not
            // executable", it is not represented at all, and it is
            // reported as itself above.
            represented_not_executable: table
                .iter()
                .filter(|l| !l.operator.is_unidentified_recurrence() && !l.operator.has_executor())
                .count(),
        }
    }

    /// Whether this policy has layers it cannot express — either kind.
    ///
    /// A policy that cannot express some of its own layers is not a
    /// representable policy: this finding was once unconditionally
    /// representable, so a stack could disclose that it had no vocabulary
    /// for a layer *and* grade representable in the same breath.
    pub fn blocks(&self) -> bool {
        self.unexpressed > 0 || self.unidentified_recurrence > 0
    }

    /// The summary sentence.
    ///
    /// Every optional clause appears only when its count is non-zero. A
    /// clause that is always emitted states nothing when it reads zero,
    /// and a gate asserting on such a clause passes without testing
    /// anything — which is exactly what the fixed "declared span(s) …"
    /// wording did once `linear_attention` stopped landing there.
    pub fn describe(&self, component: &str) -> String {
        let Self {
            sliding,
            full,
            gated_delta,
            kda,
            mamba2,
            unidentified_recurrence,
            unexpressed,
            nope,
            represented_not_executable,
        } = *self;
        let mut detail = format!(
            "per-layer policy recorded on component `{component}`: {sliding} sliding / {full} full"
        );
        if gated_delta > 0 {
            detail.push_str(&format!(" / {gated_delta} gated-delta recurrent"));
        }
        if kda > 0 {
            detail.push_str(&format!(" / {kda} KDA recurrent"));
        }
        if mamba2 > 0 {
            detail.push_str(&format!(" / {mamba2} Mamba2 recurrent"));
        }
        if unidentified_recurrence > 0 {
            detail.push_str(&format!(
                " / {unidentified_recurrence} recurrent layer(s) whose operator this build \
                 cannot identify"
            ));
        }
        if unexpressed > 0 {
            detail.push_str(&format!(
                " / {unexpressed} declared span(s) this schema has no execution vocabulary for \
                 (see text_config.layer_types)"
            ));
        }
        detail.push_str(&format!(", {nope} NoPE layer(s)"));
        if represented_not_executable > 0 {
            detail.push_str(&format!(
                "; {represented_not_executable} layer(s) represented but NOT executable — no \
                 executor exists for their operator"
            ));
        }
        detail
    }
}
