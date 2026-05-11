//! Qwen3-Next full-attention layer helpers (Phase C.3 of
//! `inference-qwen35-deltanet`).
//!
//! Qwen 3.6's full-attention layers diverge from the standard
//! transformer layer in two structural ways:
//!
//! 1. **Fused Q + per-head sigmoid gate projection.** `attn_q.weight`
//!    has shape `[hidden, 2 * n_head * head_dim]`. The first
//!    `n_head * head_dim` columns are Q (used in attention); the
//!    second `n_head * head_dim` columns are a per-head gate that
//!    multiplies the attention output before the `attn_output`
//!    projection.
//! 2. **Per-head Q/K RMSNorm at `[head_dim]` shape**, applied after
//!    splitting Q/K into heads but before RoPE. The existing
//!    `rms_norm_heads` already does this.
//!
//! Plus a third less-structural quirk: **MRoPE with multiple
//! sections** instead of single-section RoPE. For text-only
//! inference (no multimodal), all sections use the same position
//! value and MRoPE degenerates to partial RoPE on the first
//! `sum(sections)` head dimensions — same as what
//! `apply_rope_partial_at` produces. Multimodal use cases would need
//! per-section position handling; out of scope here.

use ndarray::Array2;

use super::deltanet_state::sigmoid;

/// Split a fused Q+gate projection output into the Q tensor and the
/// per-head sigmoid'd gate.
///
/// Input shape: `[seq_len, 2 * n_head * head_dim]`.
/// Output:
/// - `q`: `[seq_len, n_head * head_dim]` — the Q tensor; caller
///   applies per-head RMSNorm + RoPE + attention.
/// - `gate_sigmoid`: `[seq_len, n_head * head_dim]` — the gate
///   tensor with sigmoid already applied element-wise. Caller
///   multiplies the attention output (per-head) by this BEFORE the
///   final `attn_output` projection.
///
/// The split convention (Q first, gate second) matches llama.cpp's
/// `qwen35.cpp::build_layer_attn` view of the fused tensor.
pub fn split_q_gate(
    fused: &Array2<f32>,
    n_head: usize,
    head_dim: usize,
) -> (Array2<f32>, Array2<f32>) {
    let seq_len = fused.shape()[0];
    let total = fused.shape()[1];
    let half = n_head * head_dim;
    debug_assert_eq!(
        total,
        2 * half,
        "fused Q+gate tensor must have {} cols (2 * n_head * head_dim), got {}",
        2 * half,
        total
    );

    let mut q = Array2::<f32>::zeros((seq_len, half));
    let mut gate = Array2::<f32>::zeros((seq_len, half));

    for s in 0..seq_len {
        for c in 0..half {
            q[[s, c]] = fused[[s, c]];
            gate[[s, c]] = sigmoid(fused[[s, half + c]]);
        }
    }

    (q, gate)
}

/// Apply the per-head Q-gate to the attention output, in place.
///
/// `attn_out`: `[seq_len, n_head * head_dim]` — the attention output
/// (post-softmax, post-V-weighted-sum), still per-head layout.
/// `gate_sigmoid`: `[seq_len, n_head * head_dim]` — output of
/// `split_q_gate`'s second tensor.
///
/// Effect: `attn_out[s, c] *= gate_sigmoid[s, c]` for every `(s, c)`.
/// This SHALL run AFTER the attention computation but BEFORE the
/// `attn_output` projection.
pub fn apply_q_gate(attn_out: &mut Array2<f32>, gate_sigmoid: &Array2<f32>) {
    debug_assert_eq!(attn_out.shape(), gate_sigmoid.shape());
    let (seq_len, dim) = (attn_out.shape()[0], attn_out.shape()[1]);
    for s in 0..seq_len {
        for c in 0..dim {
            attn_out[[s, c]] *= gate_sigmoid[[s, c]];
        }
    }
}

/// Total rotary dimensions across MRoPE sections. Convenience
/// helper for callers that want to know how many head dimensions
/// get rotated (and how many pass through unchanged).
///
/// For Qwen 3.6 27B with `rope_dimension_sections = [16, 24, 24, 0]`,
/// returns `64` — the first 64 dims per head are rotary, the
/// remaining `head_dim - 64 = 192` pass through.
pub fn mrope_total_rotary(sections: &[usize]) -> usize {
    sections.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn split_q_gate_separates_halves() {
        // n_head = 2, head_dim = 2 → half = 4, total = 8.
        // Row 0: Q = [1, 2, 3, 4], gate_pre = [-1, 0, 1, 2].
        // Expected gate_post = sigmoid([-1, 0, 1, 2]).
        let fused = array![[1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0]];
        let (q, gate) = split_q_gate(&fused, 2, 2);
        assert_eq!(q.shape(), &[1, 4]);
        assert_eq!(gate.shape(), &[1, 4]);
        // Q is the first half verbatim.
        assert_eq!(q[[0, 0]], 1.0);
        assert_eq!(q[[0, 3]], 4.0);
        // Gate is the second half passed through sigmoid.
        assert!((gate[[0, 0]] - sigmoid(-1.0)).abs() < 1e-6);
        assert!((gate[[0, 1]] - 0.5).abs() < 1e-6); // sigmoid(0)
        assert!((gate[[0, 2]] - sigmoid(1.0)).abs() < 1e-6);
        assert!((gate[[0, 3]] - sigmoid(2.0)).abs() < 1e-6);
    }

    #[test]
    fn split_q_gate_multi_row() {
        let fused = array![[1.0_f32, 0.0, 2.0, 0.0], [3.0, 0.0, 4.0, 0.0],];
        // n_head=1, head_dim=2 → half=2.
        let (q, gate) = split_q_gate(&fused, 1, 2);
        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(q.row(0).to_vec(), vec![1.0, 0.0]);
        assert_eq!(q.row(1).to_vec(), vec![3.0, 0.0]);
        // Gate pre-sigmoid = [2.0, 0.0] then [4.0, 0.0].
        assert!((gate[[0, 0]] - sigmoid(2.0)).abs() < 1e-6);
        assert!((gate[[0, 1]] - 0.5).abs() < 1e-6);
        assert!((gate[[1, 0]] - sigmoid(4.0)).abs() < 1e-6);
    }

    #[test]
    fn apply_q_gate_multiplies_in_place() {
        let mut attn = array![[2.0_f32, 4.0], [1.0, 8.0]];
        let gate = array![[0.5_f32, 0.25], [1.0, 0.0]];
        apply_q_gate(&mut attn, &gate);
        assert_eq!(attn.row(0).to_vec(), vec![1.0, 1.0]);
        assert_eq!(attn.row(1).to_vec(), vec![1.0, 0.0]);
    }

    #[test]
    fn mrope_total_rotary_matches_qwen_3_6_27b() {
        // Qwen 3.6 27B rope_dimension_sections = [16, 24, 24, 0]
        // → 64 rotary dims per head. head_dim = 256, so 192 pass
        // through.
        assert_eq!(mrope_total_rotary(&[16, 24, 24, 0]), 64);
    }

    #[test]
    fn mrope_total_rotary_empty_is_zero() {
        assert_eq!(mrope_total_rotary(&[]), 0);
    }

    #[test]
    fn split_q_gate_round_trip_with_apply_q_gate() {
        // End-to-end shape check: split a fused tensor, multiply
        // a dummy attn output by the gate, verify shapes stayed
        // consistent.
        let fused = array![[1.0_f32, 2.0, 0.0, 0.0]]; // n_head=1, head_dim=2
        let (_q, gate) = split_q_gate(&fused, 1, 2);
        let mut attn = array![[7.0_f32, 11.0]];
        apply_q_gate(&mut attn, &gate);
        // gate pre-sig = [0.0, 0.0] → sigmoid = [0.5, 0.5]
        assert!((attn[[0, 0]] - 3.5).abs() < 1e-6);
        assert!((attn[[0, 1]] - 5.5).abs() < 1e-6);
    }
}
