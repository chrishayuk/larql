//! Shared constants and formatting helpers for the SELECT verbs.
//!
//! Pulling defaults + row formats here keeps each verb file focused
//! on its own scan logic and forces the magic numbers used across
//! tables (column widths, banner lengths, default limits) through a
//! single audited place.

// ── Default LIMIT values ──────────────────────────────────────────
//
// These are picked to fit a comfortable terminal page (~25 rows of
// content + a header). Where one verb explicitly overrides another
// (e.g. SELECT * FROM EDGES with a feature filter wants to see that
// feature across every layer), the override is documented at its
// call site rather than re-tuned here.

/// Default `LIMIT` for `SELECT * FROM EDGES` when no feature filter
/// is present. With a feature filter we use `num_layers` instead so
/// the user sees the requested feature at every layer.
pub(super) const EDGES_DEFAULT_LIMIT: u32 = 20;

/// Default `LIMIT` for `SELECT NEAREST TO`.
pub(super) const NEAREST_DEFAULT_LIMIT: u32 = 20;

/// Default `LIMIT` for `SELECT * FROM ENTITIES`.
pub(super) const ENTITIES_DEFAULT_LIMIT: u32 = 50;

/// Default `LIMIT` for `SELECT * FROM FEATURES` when no layer or
/// feature filter is present. Sized to fit a terminal page; users
/// who want more pass an explicit `LIMIT N`.
pub(super) const FEATURES_DEFAULT_LIMIT: usize = 34;

/// `top_k` for the entity-anchored FFN walk in `SELECT * FROM EDGES
/// WHERE entity = … AND relation = …`. The raw embedding query has
/// low cosine with deep-layer gate directions because the residual
/// has been transformed by N layers of attention+FFN, so we need a
/// wide scan to find the relation-labelled features that fire.
pub(super) const EDGES_WALK_TOP_K: usize = 500;

/// Number of "also" tokens to surface alongside the top-1 in row
/// outputs. Three is enough to disambiguate without crowding the
/// `Also` column.
pub(super) const ALSO_TOPK: usize = 3;

// ── Table column widths ──────────────────────────────────────────
//
// These are paired with banner widths derived from them. The pairs
// must stay in sync — `with_relation_with_also_banner_len`,
// `with_relation_no_also_banner_len`, etc. — otherwise the dashed
// underline doesn't match the column total.

/// Compute the dashed underline string for a banner of `n` cols.
pub(super) fn banner(n: usize) -> String {
    "-".repeat(n)
}

/// Build the "also = [a, b, c]" sub-string from a top-k slice
/// (skipping the top-1, which is shown in its own column).
pub(super) fn format_also(top_k: &[larql_models::TopKEntry]) -> String {
    top_k
        .iter()
        .skip(1)
        .take(ALSO_TOPK)
        .map(|e| e.token.clone())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Decorate the raw "also" string with bracket markers, or empty
/// string when nothing to display.
pub(super) fn also_display(also: &str) -> String {
    if also.is_empty() {
        String::new()
    } else {
        format!("[{also}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::TopKEntry;

    fn entry(token: &str) -> TopKEntry {
        TopKEntry {
            token: token.into(),
            token_id: 0,
            logit: 0.0,
        }
    }

    #[test]
    fn banner_repeats_dashes_n_times() {
        assert_eq!(banner(0), "");
        assert_eq!(banner(1), "-");
        assert_eq!(banner(5), "-----");
    }

    #[test]
    fn format_also_skips_top_1_and_takes_three() {
        let top = vec![
            entry("Paris"),
            entry("French"),
            entry("Europe"),
            entry("France"),
            entry("Eiffel"),
        ];
        // Top-1 ("Paris") is skipped; next three.
        assert_eq!(format_also(&top), "French, Europe, France");
    }

    #[test]
    fn format_also_handles_short_lists() {
        assert_eq!(format_also(&[entry("only")]), "");
        assert_eq!(format_also(&[entry("a"), entry("b")]), "b");
    }

    #[test]
    fn also_display_brackets_non_empty() {
        assert_eq!(also_display("a, b"), "[a, b]");
        assert_eq!(also_display(""), "");
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn default_limits_are_positive() {
        // Pinned: const-evaluated guards against accidentally setting
        // a default limit to zero, which would silently truncate the
        // query result. The runtime assert is the regression test.
        assert!(EDGES_DEFAULT_LIMIT > 0);
        assert!(NEAREST_DEFAULT_LIMIT > 0);
        assert!(ENTITIES_DEFAULT_LIMIT > 0);
        assert!(FEATURES_DEFAULT_LIMIT > 0);
        assert!(EDGES_WALK_TOP_K >= EDGES_DEFAULT_LIMIT as usize);
    }
}
