//! Where a generated sequence ends.

use std::collections::HashSet;

/// The generated-ids contract every `--generate` loop keeps: at most `n`
/// ids, ending at the first stop id, which is itself kept. A speculative
/// block may run past a stop; the ids after it were never wanted.
pub fn truncate_at_stop(out: &mut Vec<u32>, stop: &HashSet<u32>, n: usize) {
    if let Some(end) = out.iter().position(|id| stop.contains(id)) {
        out.truncate(end + 1);
    }
    out.truncate(n);
}
