//! The greedy choice every speculative and sequential greedy loop shares.

/// The highest logit's id, the lowest id among ties. `None` for empty logits
/// or any non-finite value: a greedy choice over those is undefined, and
/// two loops must never disagree about one.
pub fn greedy_id(logits: &[f32]) -> Option<u32> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    Some(best as u32)
}
