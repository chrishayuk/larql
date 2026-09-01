//! The one byte formatter both CLIs print through.

/// Bytes per gigabyte, decimal — what a checkpoint's own metadata counts in.
const GB: f64 = 1e9;

/// Bytes per gibibyte, binary — what a filesystem reports.
const GIB: f64 = (1u64 << 30) as f64;

/// Bytes per megabyte, decimal.
const MB: f64 = 1e6;

/// A byte count, unit stated.
///
/// Both units are given at GB scale and only there. That is where the
/// distinction bites — `GB` against `GiB` is a 7% difference, big enough
/// that two correct figures side by side read as a bug in one of them —
/// and it is also where the numbers this feature prints get compared
/// against each other. Below that, one unit stays readable: a staging line
/// quoting four figures in two units each is not clearer, it is noise.
pub fn size(bytes: u64) -> String {
    let value = bytes as f64;
    if value >= GB {
        format!("{:.2} GB ({:.2} GiB)", value / GB, value / GIB)
    } else {
        format!("{:.2} MB", value / MB)
    }
}
