//! The greedy choice every speculative and sequential greedy loop on the
//! remote path shares.
//!
//! **Rule:** the highest logit, the lowest id among ties (the first
//! maximum, the V3 exec harness's rule). `-inf` is a legitimate score, for
//! a masked or padded entry that must never be chosen. NaN and `+inf` have
//! no order, so a choice over them is refused rather than guessed, and two
//! loops can never disagree because one guessed differently.
//!
//! This is **not** the sampler's greedy rule
//! (`layer_graph::generate::sampling`), which skips every non-finite value
//! and takes the **last** of equal maxima. The two agree whenever the
//! maximum is unique and finite. A comparison between a run of this rule
//! and a run of the sampler must use one rule on both sides.

/// The greedy id of `logits`, or `None` when there is no defined choice:
/// empty logits, any NaN or `+inf`, or every entry `-inf`.
pub fn greedy_id(logits: &[f32]) -> Option<u32> {
    if logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        return None;
    }
    let mut best: Option<usize> = None;
    for (index, value) in logits.iter().enumerate() {
        if *value == f32::NEG_INFINITY {
            continue;
        }
        if best.is_none_or(|b| *value > logits[b]) {
            best = Some(index);
        }
    }
    best.map(|b| b as u32)
}
