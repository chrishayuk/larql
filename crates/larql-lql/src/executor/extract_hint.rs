//! The "Extract first" hint a live-weights session gives when a statement
//! needs a vindex. It is meant to be pasted back into LQL, so the model id
//! is rendered as an LQL string literal and the suggested vindex is named
//! after the id's last path component on every platform.

/// Characters the LQL lexer decodes after a backslash inside a quoted
/// string; each must be escaped to survive a round trip.
const LQL_ESCAPED: [char; 2] = ['\\', '"'];

/// Separators that end a path component: `/` in Hugging Face ids and Unix
/// paths, `\` in Windows paths.
const PATH_SEPARATORS: [char; 2] = ['/', '\\'];

/// Suffix of the suggested output directory.
const VINDEX_SUFFIX: &str = ".vindex";

/// `s` as the body of an LQL quoted string literal.
pub(crate) fn lql_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if LQL_ESCAPED.contains(&c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// The last path component of a model id or checkpoint path.
pub(crate) fn model_leaf(model_id: &str) -> &str {
    model_id
        .trim_end_matches(PATH_SEPARATORS)
        .rsplit(PATH_SEPARATORS)
        .next()
        .filter(|leaf| !leaf.is_empty())
        .unwrap_or(model_id)
}

/// `<need>. Extract first:` followed by a paste-able EXTRACT statement.
pub(crate) fn extract_first(need: &str, model_id: &str) -> String {
    format!(
        "{need}. Extract first:\n  EXTRACT MODEL \"{}\" INTO \"{}{VINDEX_SUFFIX}\"",
        lql_literal(model_id),
        lql_literal(model_leaf(model_id)),
    )
}

#[cfg(test)]
mod tests;
