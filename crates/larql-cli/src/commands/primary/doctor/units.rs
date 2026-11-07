//! Human-readable bytes and paths for the doctor's one-line summaries.

use std::path::Path;

/// Decimal units, as the docs print sizes ("1.50 GB").
const UNITS: [(&str, u64); 4] = [
    ("TB", 1_000_000_000_000),
    ("GB", 1_000_000_000),
    ("MB", 1_000_000),
    ("KB", 1_000),
];

/// `bytes` in the largest decimal unit it reaches, one decimal place.
pub(crate) fn format_bytes(bytes: u64) -> String {
    UNITS
        .iter()
        .find(|(_, scale)| bytes >= *scale)
        .map(|(unit, scale)| format!("{:.1} {unit}", bytes as f64 / *scale as f64))
        .unwrap_or_else(|| format!("{bytes} B"))
}

/// `path` with a leading `home` shown as `~`, so summaries stay short and
/// a shared report does not carry the user name.
pub(crate) fn display_path(path: &Path, home: Option<&Path>) -> String {
    match home.and_then(|h| path.strip_prefix(h).ok()) {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_string(),
        Some(rest) => Path::new("~").join(rest).display().to_string(),
        None => path.display().to_string(),
    }
}
