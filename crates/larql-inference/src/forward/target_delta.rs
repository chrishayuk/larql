//! Per-fact target-delta optimisation — the MEMIT Phase-3 primitive.
//!
//! Given a model, a prompt, and a target token id, find the residual
//! delta `δ ∈ R^hidden` such that adding `δ` to the FFN output at
//! `install_layer`'s last-position makes `target_id` the top logit,
//! with a KL regulariser keeping the distribution close to baseline.
//!
//! This is the per-fact pre-compute that the Python reference at
//! `~/chris-source/chris-experiments/compilation/15_v11_model/vindex_compile_rome_v11.py::optimise_target_delta`
//! runs before MEMIT's closed-form W-edit. Without it, MEMIT's V*
//! defaults to `target_alpha × embed(target)` — a rough direction
//! that doesn't account for how downstream layers transform the
//! residual. With it, MEMIT's V* is the exact signal that, added
//! to the L_install FFN output, produces the target at logits.
//!
//! ## Algorithm (Python reference)
//!
//! ```text
//! base_logits = model(x)[-1]             # no-edit baseline
//! base_probs  = softmax(base_logits)
//! δ = zeros(hidden); opt = Adam([δ], lr)
//!
//! for step in 0..60:
//!     h = embed(x) * sqrt(dim)
//!     for layer in 0..n_layers:
//!         h = h + attn(attn_norm(h))
//!         ffn_out = ffn(ffn_norm(h))
//!         if layer == install_layer:
//!             ffn_out[-1] += δ            # the perturbation
//!         h = h + ffn_out
//!     logits = lm_head(norm(h))[-1]
//!
//!     loss = cross_entropy(logits, target_id) +
//!            kl_weight · KL(base_probs, softmax(logits))
//!     loss.backward(); opt.step()
//!
//! return δ
//! ```
//!
//! ## Native port status (WIP)
//!
//! The forward pass already exists in `forward/` (via `WalkFfn` and
//! the dense path). What's missing is the **backward pass**: to
//! compute `∂loss/∂δ` we need gradients through:
//!
//!   lm_head @ final_norm(h) @ layers[install..]  ← every op gets a transpose-multiply
//!
//! Hand-rolled backward implementations for each layer (attention,
//! FFN, RMSNorm) are landing in this module as `backward_*` helpers.
//! Scope for this first drop: infrastructure + cross-entropy gradient
//! + lm_head backward (tied embedding). Per-layer transformer-block
//!   backward is the remaining ~80% of the work — tracked as follow-ups
//!   on each `backward_*` stub.
//!
//! Once complete, `optimise_target_delta` runs 60-80 Adam iters per
//! fact in pure Rust; `run_memit` calls it and feeds the optimised
//! deltas into `rome_batch_update` as V*.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::model::ModelWeights;

/// Hyperparameters for target-delta optimisation. Defaults match the
/// Python reference (`vindex_compile_rome_v11.py::optimise_target_delta`).
#[derive(Debug, Clone, Copy)]
pub struct TargetDeltaOpts {
    pub steps: usize,
    pub lr: f32,
    pub kl_weight: f32,
    /// If true, the returned delta is normalised to unit norm — useful
    /// when the downstream MEMIT solve scales its own magnitude.
    pub normalise: bool,
}

impl Default for TargetDeltaOpts {
    fn default() -> Self {
        Self {
            steps: 60,
            lr: 0.5,
            kl_weight: 0.0625,
            normalise: false,
        }
    }
}

/// Result of a single target-delta optimisation.
#[derive(Debug, Clone)]
pub struct TargetDelta {
    pub layer: usize,
    pub delta: Array1<f32>,
    /// Cross-entropy loss on the final step (lower is better; 0 means
    /// the target is the argmax with very high probability).
    pub final_loss: f32,
    /// Baseline loss on the target under the no-edit forward pass —
    /// useful for diagnostics ("did the optimisation actually move?").
    pub baseline_loss: f32,
}

// ── Autograd tape: minimal reverse-mode for the ops we need ────────
//
// Rather than pull in a full autograd crate (candle/burn/dfdx), we
// implement a focused reverse-mode that supports exactly the ops on
// the critical path from `δ` to `loss`. Each forward op returns both
// an output tensor AND appends a closure to a tape that, given the
// upstream gradient, contributes to the inputs' gradients.
//
// This avoids a full refactor of the model forward and keeps us in
// ndarray throughout.
//
// NOTE: this is a structural sketch. The tape record types and
// closures for each layer's backward are filled in piece-by-piece as
// the backward functions below are implemented.

/// Softmax cross-entropy loss for a 1-D logits vector and a single
/// target id. Returns `(loss, dlogits)` where `dlogits[j] = softmax[j] - onehot[target][j]`.
/// Used at the output end — no tape needed since this is the loss itself.
pub(crate) fn cross_entropy_and_grad(
    logits: ArrayView1<f32>,
    target_id: u32,
) -> (f32, Array1<f32>) {
    // Numerically stable log-softmax
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let shifted: Array1<f32> = logits.map(|&v| v - max);
    let exp_sum: f32 = shifted.iter().map(|v| v.exp()).sum();
    let log_sum = exp_sum.ln();
    let loss = -(shifted[target_id as usize] - log_sum);

    // Gradient: softmax(logits) - onehot(target)
    let mut dlogits = shifted.map(|v| (v - log_sum).exp());
    dlogits[target_id as usize] -= 1.0;
    (loss, dlogits)
}

/// Backward through the tied-embedding lm_head: `logits = embed @ h`
/// so `∂loss/∂h = embed.T @ dlogits`. For tied embeddings
/// `lm_head.weight == embed.weight`, so we use the same matrix.
pub(crate) fn lm_head_backward(
    embed_weight: ArrayView2<f32>, // (vocab, hidden)
    dlogits: ArrayView1<f32>,      // (vocab,)
) -> Array1<f32> {
    // ∂loss/∂h[i] = Σ_v dlogits[v] · embed[v, i]
    // = embed.T @ dlogits  →  shape (hidden,)
    let hidden = embed_weight.ncols();
    let mut dh = Array1::<f32>::zeros(hidden);
    for (v_idx, &dl) in dlogits.iter().enumerate() {
        if dl == 0.0 {
            continue;
        }
        let row = embed_weight.row(v_idx);
        for i in 0..hidden {
            dh[i] += dl * row[i];
        }
    }
    dh
}

/// Backward through the final RMSNorm at the last position:
///
///   y = (x / rms(x)) * weight           where rms(x) = sqrt(mean(x^2) + eps)
///
/// The gradient form for RMSNorm (per-position) is:
///
///   ∂L/∂x = (weight / rms) * [dy - (x · (weight · dy)) / (rms^2 · d)]
///
/// where `·` is dot product and `d = hidden`. Returns `dx`, not `dweight`
/// (we don't update the norm weights during target-delta opt — they
/// aren't in the optimisation path).
pub(crate) fn rmsnorm_backward_pos(
    x: ArrayView1<f32>,
    weight: ArrayView1<f32>,
    dy: ArrayView1<f32>,
    eps: f32,
) -> Array1<f32> {
    let d = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / d;
    let rms = (ms + eps).sqrt();

    // inner = (weight · dy) dotted element-wise
    let wdy: Array1<f32> = weight.iter().zip(dy.iter()).map(|(&w, &g)| w * g).collect();
    // xwdy = x · wdy  (scalar)
    let xwdy: f32 = x.iter().zip(wdy.iter()).map(|(&xi, &w)| xi * w).sum();

    // dx[i] = (1/rms) * (wdy[i] - x[i] * xwdy / (d * rms^2))
    let inv_rms = 1.0 / rms;
    let coef = xwdy / (d * rms * rms);
    let mut dx = Array1::<f32>::zeros(x.len());
    for i in 0..x.len() {
        dx[i] = inv_rms * (wdy[i] - x[i] * coef);
    }
    dx
}

/// Backward through a gated FFN block at one position.
///
/// Forward:
///   g_pre = gate_w @ x         (gate_w: ffn_dim × hidden)
///   g     = silu(g_pre)         silu(z) = z · σ(z)
///   u     = up_w @ x            (up_w:   ffn_dim × hidden)
///   act   = g * u               (ffn_dim)
///   out   = down_w @ act        (down_w: hidden × ffn_dim)
///
/// Backward (given d_out):
///   d_act = down_w.T @ d_out
///   d_g   = d_act * u
///   d_u   = d_act * g
///   silu'(z) = σ(z) · (1 + z · (1 - σ(z)))
///   d_g_pre = d_g * silu'(g_pre)
///   d_x = gate_w.T @ d_g_pre + up_w.T @ d_u
#[allow(dead_code)] // reserved primitive for mid-layer target-delta; FD-tested
pub(crate) fn gated_ffn_backward(
    x: ArrayView1<f32>,
    gate_w: ArrayView2<f32>,
    up_w: ArrayView2<f32>,
    down_w: ArrayView2<f32>,
    d_out: ArrayView1<f32>,
) -> Array1<f32> {
    let hidden = x.len();
    let ffn_dim = gate_w.nrows();
    assert_eq!(gate_w.ncols(), hidden);
    assert_eq!(up_w.nrows(), ffn_dim);
    assert_eq!(up_w.ncols(), hidden);
    assert_eq!(down_w.nrows(), hidden);
    assert_eq!(down_w.ncols(), ffn_dim);
    assert_eq!(d_out.len(), hidden);

    // Forward activations we need again for backward.
    let mut g_pre = Array1::<f32>::zeros(ffn_dim);
    let mut u = Array1::<f32>::zeros(ffn_dim);
    for i in 0..ffn_dim {
        let mut gp = 0.0_f32;
        let mut up = 0.0_f32;
        for j in 0..hidden {
            gp += gate_w[[i, j]] * x[j];
            up += up_w[[i, j]] * x[j];
        }
        g_pre[i] = gp;
        u[i] = up;
    }
    // silu and σ
    let sigma: Array1<f32> = g_pre.map(|&z| 1.0 / (1.0 + (-z).exp()));
    let g: Array1<f32> = g_pre
        .iter()
        .zip(sigma.iter())
        .map(|(&z, &s)| z * s)
        .collect();

    // d_act = down_w.T @ d_out → shape ffn_dim
    let mut d_act = Array1::<f32>::zeros(ffn_dim);
    for i in 0..ffn_dim {
        let mut s = 0.0_f32;
        for k in 0..hidden {
            s += down_w[[k, i]] * d_out[k];
        }
        d_act[i] = s;
    }

    // d_g = d_act * u ; d_u = d_act * g
    let d_g: Array1<f32> = d_act.iter().zip(u.iter()).map(|(&a, &b)| a * b).collect();
    let d_u: Array1<f32> = d_act.iter().zip(g.iter()).map(|(&a, &b)| a * b).collect();

    // silu'(z) = σ(z) * (1 + z * (1 - σ(z)))
    let d_g_pre: Array1<f32> = g_pre
        .iter()
        .zip(sigma.iter())
        .zip(d_g.iter())
        .map(|((&z, &s), &dg)| dg * s * (1.0 + z * (1.0 - s)))
        .collect();

    // d_x = gate_w.T @ d_g_pre + up_w.T @ d_u
    let mut d_x = Array1::<f32>::zeros(hidden);
    for j in 0..hidden {
        let mut s = 0.0_f32;
        for i in 0..ffn_dim {
            s += gate_w[[i, j]] * d_g_pre[i] + up_w[[i, j]] * d_u[i];
        }
        d_x[j] = s;
    }
    d_x
}

/// TODO: backward through a single attention block at one position.
///
/// Gradient path is substantially more complex: Q/K/V projections +
/// RoPE + softmax-scaled-dot-product + O projection. At the target
/// position (last token) the attention output depends on keys/values
/// at ALL previous positions, so the backward gradient flows back to
/// all positions of the input residual. For target-delta opt we only
/// need the gradient at the LAST position, since that's where we're
/// perturbing — but the intermediate computation still needs the full
/// attention matrix.
///
/// Scope for a first drop: cross-attention-less (attention to past
/// positions only, last-position gradient). RoPE is position-index
/// pure so its backward is deterministic-position rotation.
#[allow(dead_code)]
pub(crate) fn attention_backward_last_pos() {
    unimplemented!("attention_backward_last_pos: pending implementation")
}

/// Per-fact target delta optimisation.
///
/// CURRENT SUPPORT: `install_layer = n_layers - 1` (last layer). The
/// perturbation at the last block's output flows through only
/// `final_norm` + `lm_head` to logits, both of which have verified
/// backward primitives in this module. For earlier layers the
/// backward through intermediate transformer blocks is still being
/// ported (see `gated_ffn_backward` and `attention_backward_last_pos`
/// stubs).
///
/// Runs Adam for `opts.steps` iterations on a delta ∈ R^hidden,
/// minimising `CE(logits, target_id) + kl_weight · KL(baseline, current)`.
pub fn optimise_target_delta(
    weights: &ModelWeights,
    tokens: &[u32],
    target_id: u32,
    install_layer: usize,
    opts: TargetDeltaOpts,
) -> Result<TargetDelta, String> {
    validate_target_delta_inputs(weights, tokens, target_id)?;

    let n_layers = weights.arch.config().num_layers;
    if install_layer >= n_layers {
        return Err(format!(
            "install_layer {install_layer} ≥ n_layers {n_layers}"
        ));
    }
    if install_layer != n_layers - 1 {
        return Err(format!(
            "optimise_target_delta: only install_layer = n_layers-1 = {} is \
             supported in this build (got {install_layer}). Mid-layer backward \
             through attention+FFN is pending (target_delta.rs stubs).",
            n_layers - 1
        ));
    }

    let hidden = weights.arch.config().hidden_size;
    let norm_offset = weights.arch.norm_weight_offset();
    let final_norm_key = weights.arch.final_norm_key();
    let norm_weight_vec: Vec<f32> = weights
        .vectors
        .get(final_norm_key)
        .map(|v| {
            let mut w = v.to_vec();
            for x in w.iter_mut() {
                *x += norm_offset;
            }
            w
        })
        .ok_or_else(|| format!("missing final norm weight key: {final_norm_key}"))?;
    let norm_weight = Array1::from(norm_weight_vec);
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    if weights.arch.final_logit_softcapping().is_some() {
        return Err("target-delta opt doesn't yet handle logit softcap — port required".into());
    }

    // Baseline forward (no perturbation) for KL regulariser.
    let baseline = crate::forward::predict::forward_raw_logits(weights, tokens, None);
    let base_probs = softmax_1d(&baseline.logits);
    let baseline_loss = {
        let (l, _) = cross_entropy_and_grad(baseline.logits.view(), target_id);
        l
    };

    // Adam state.
    let mut delta = Array1::<f32>::zeros(hidden);
    let mut m = Array1::<f32>::zeros(hidden);
    let mut v = Array1::<f32>::zeros(hidden);
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const ADAM_EPS: f32 = 1e-8;
    const RMS_EPS: f32 = 1e-6;

    let mut final_loss = f32::NAN;
    for step in 1..=opts.steps {
        let out = crate::forward::predict::forward_raw_logits(
            weights,
            tokens,
            Some((install_layer, delta.view())),
        );

        // Loss: CE(target) + kl_weight · KL(base || current)
        let (ce, mut dlogits) = cross_entropy_and_grad(out.logits.view(), target_id);
        let cur_probs = softmax_1d(&out.logits);

        // KL(p || q) gradient on logits: q - p (where q is current probs, p is baseline).
        // Add to dlogits weighted by kl_weight.
        if opts.kl_weight != 0.0 {
            for i in 0..dlogits.len() {
                dlogits[i] += opts.kl_weight * (cur_probs[i] - base_probs[i]);
            }
        }

        // KL value for diagnostics (not strictly needed for backprop).
        let kl_val: f32 = if opts.kl_weight != 0.0 {
            base_probs
                .iter()
                .zip(cur_probs.iter())
                .map(|(&p, &q)| {
                    if p < 1e-12 {
                        0.0
                    } else {
                        p * (p.max(1e-12).ln() - q.max(1e-12).ln())
                    }
                })
                .sum()
        } else {
            0.0
        };
        final_loss = ce + opts.kl_weight * kl_val;

        // Backward: logits ← lm_head ← h_final ← final_norm ← h_pre_norm.
        // Scale gradient by inv_scale since logits = raw / scale.
        for d in dlogits.iter_mut() {
            *d *= inv_scale;
        }

        let last_final = out.h_final.row(out.h_final.nrows() - 1);
        let _last_pre_norm = out.h_pre_norm.row(out.h_pre_norm.nrows() - 1);
        let _ = last_final;

        // lm_head backward: weights.lm_head shape (vocab, hidden); logits = lm_head @ h_last
        let lm = &weights.lm_head;
        let d_h_final = lm_head_backward(lm.view(), dlogits.view());

        // RMSNorm backward at the last position:
        // h_pre_norm[-1] is input; norm_weight is scale; d_h_final is upstream grad.
        let last_pre = out.h_pre_norm.row(out.h_pre_norm.nrows() - 1).to_owned();
        let d_h_pre_norm = rmsnorm_backward_pos(
            last_pre.view(),
            norm_weight.view(),
            d_h_final.view(),
            RMS_EPS,
        );

        // For install_layer = n_layers - 1, δ is added directly to
        // h[-1] after the last block. So ∂loss/∂δ = d_h_pre_norm.
        let grad = d_h_pre_norm;

        // Adam update.
        let s = step as f32;
        let bc1 = 1.0 - BETA1.powi(step as i32);
        let bc2 = 1.0 - BETA2.powi(step as i32);
        for i in 0..hidden {
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * grad[i];
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * grad[i] * grad[i];
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            delta[i] -= opts.lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
        }
        let _ = s;
    }

    if opts.normalise {
        let norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in delta.iter_mut() {
                *x /= norm;
            }
        }
    }

    Ok(TargetDelta {
        layer: install_layer,
        delta,
        final_loss,
        baseline_loss,
    })
}

fn validate_target_delta_inputs(
    weights: &ModelWeights,
    tokens: &[u32],
    target_id: u32,
) -> Result<(), String> {
    if tokens.is_empty() {
        return Err("optimise_target_delta: prompt tokens may not be empty".into());
    }

    let cfg = weights.arch.config();
    let hidden = cfg.hidden_size;
    let vocab = weights.vocab_size;

    if cfg.num_layers == 0 {
        return Err("optimise_target_delta: model has zero layers".into());
    }
    if hidden == 0 {
        return Err("optimise_target_delta: hidden_size may not be zero".into());
    }
    if vocab == 0 {
        return Err("optimise_target_delta: vocab_size may not be zero".into());
    }
    if weights.hidden_size != hidden {
        return Err(format!(
            "optimise_target_delta: cached hidden_size {} != architecture hidden_size {hidden}",
            weights.hidden_size
        ));
    }
    if let Some(arch_vocab) = cfg.vocab_size {
        if arch_vocab != vocab {
            return Err(format!(
                "optimise_target_delta: cached vocab_size {vocab} != architecture vocab_size {arch_vocab}",
            ));
        }
    }

    if weights.embed.nrows() != vocab || weights.lm_head.nrows() != vocab {
        return Err(format!(
            "optimise_target_delta: vocab_size {vocab} does not match embed/lm_head rows ({}/{})",
            weights.embed.nrows(),
            weights.lm_head.nrows()
        ));
    }

    let target = target_id as usize;
    if target >= vocab {
        return Err(format!(
            "optimise_target_delta: target_id {target_id} outside vocab size {vocab}"
        ));
    }

    if let Some(&bad) = tokens.iter().find(|&&id| id as usize >= vocab) {
        return Err(format!(
            "optimise_target_delta: token id {bad} outside vocab size {vocab}"
        ));
    }

    let embed_shape = weights.embed.shape();
    if embed_shape != [vocab, hidden] {
        return Err(format!(
            "optimise_target_delta: embed shape {:?} != [{vocab}, {hidden}]",
            embed_shape
        ));
    }

    let lm_shape = weights.lm_head.shape();
    if lm_shape != [vocab, hidden] {
        return Err(format!(
            "optimise_target_delta: lm_head shape {:?} != [{vocab}, {hidden}]",
            lm_shape
        ));
    }

    let final_norm_key = weights.arch.final_norm_key();
    let norm_len = weights
        .vectors
        .get(final_norm_key)
        .map(|v| v.len())
        .ok_or_else(|| format!("missing final norm weight key: {final_norm_key}"))?;
    if norm_len != hidden {
        return Err(format!(
            "optimise_target_delta: final norm weight len {norm_len} != hidden_size {hidden}"
        ));
    }

    let logits_scaling = weights.arch.logits_scaling();
    if !logits_scaling.is_finite() || logits_scaling == 0.0 {
        return Err(format!(
            "optimise_target_delta: invalid logits scaling {logits_scaling}"
        ));
    }

    Ok(())
}

/// Softmax over a 1-D vector (numerically stable).
fn softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Array1<f32> = logits.map(|&v| (v - max).exp());
    let sum: f32 = exps.iter().sum();
    exps.map(|&v| v / sum)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use ndarray::arr1;
    use ndarray::arr2;
    use ndarray::Array2;

    fn assert_opt_err_contains(tokens: &[u32], target_id: u32, needle: &str) {
        let weights = make_test_weights();
        let err = optimise_target_delta(
            &weights,
            tokens,
            target_id,
            weights.num_layers - 1,
            TargetDeltaOpts {
                steps: 0,
                ..TargetDeltaOpts::default()
            },
        )
        .expect_err("invalid input should return Err");
        assert!(
            err.contains(needle),
            "expected error to contain {needle:?}, got {err:?}"
        );
    }

    #[test]
    fn optimise_target_delta_rejects_empty_prompt() {
        assert_opt_err_contains(&[], 0, "prompt tokens may not be empty");
    }

    #[test]
    fn optimise_target_delta_rejects_target_outside_vocab() {
        let weights = make_test_weights();
        assert_opt_err_contains(&[0], weights.vocab_size as u32, "target_id");
    }

    #[test]
    fn optimise_target_delta_rejects_token_outside_vocab() {
        let weights = make_test_weights();
        assert_opt_err_contains(&[weights.vocab_size as u32], 0, "token id");
    }

    #[test]
    fn optimise_target_delta_rejects_bad_lm_head_shape() {
        let mut weights = make_test_weights();
        weights.lm_head =
            Array2::<f32>::zeros((weights.vocab_size, weights.hidden_size + 1)).into_shared();

        let err = optimise_target_delta(
            &weights,
            &[0],
            0,
            weights.num_layers - 1,
            TargetDeltaOpts::default(),
        )
        .expect_err("bad lm_head shape should return Err");
        assert!(err.contains("lm_head shape"), "{err}");
    }

    #[test]
    fn optimise_target_delta_rejects_bad_final_norm_shape() {
        let mut weights = make_test_weights();
        let key = weights.arch.final_norm_key().to_string();
        weights
            .vectors
            .insert(key, vec![1.0; weights.hidden_size + 1]);

        let err = optimise_target_delta(
            &weights,
            &[0],
            0,
            weights.num_layers - 1,
            TargetDeltaOpts::default(),
        )
        .expect_err("bad final norm shape should return Err");
        assert!(err.contains("final norm weight len"), "{err}");
    }

    #[test]
    fn cross_entropy_and_grad_matches_numerical() {
        // Reference: with logits [1.0, 2.0, 0.5], target=1
        // softmax(logits) ≈ [0.2312, 0.6285, 0.1402]
        // loss = -log(0.6285) ≈ 0.4644
        // dlogits = softmax - onehot(1) = [0.2312, -0.3715, 0.1402]
        let logits = arr1(&[1.0_f32, 2.0, 0.5]);
        let (loss, dlogits) = cross_entropy_and_grad(logits.view(), 1);
        assert!((loss - 0.4644).abs() < 1e-3, "loss {loss}");
        assert!((dlogits[0] - 0.2312).abs() < 1e-3);
        assert!((dlogits[1] - (-0.3715)).abs() < 1e-3);
        assert!((dlogits[2] - 0.1402).abs() < 1e-3);
    }

    #[test]
    fn lm_head_backward_shape_and_values() {
        // embed shape (vocab=3, hidden=4), dlogits (3,) → dh (4,)
        let embed = arr2(&[
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]);
        let dlogits = arr1(&[0.5_f32, -0.3, 0.2]);
        let dh = lm_head_backward(embed.view(), dlogits.view());
        // dh[i] = Σ_v dlogits[v] * embed[v,i]
        //  dh[0] = 0.5*1 + -0.3*0 + 0.2*1 = 0.7
        //  dh[1] = 0.5*0 + -0.3*1 + 0.2*1 = -0.1
        //  dh[2] = 0.2
        //  dh[3] = 0.2
        assert!((dh[0] - 0.7).abs() < 1e-5);
        assert!((dh[1] - (-0.1)).abs() < 1e-5);
        assert!((dh[2] - 0.2).abs() < 1e-5);
        assert!((dh[3] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn gated_ffn_backward_finite_difference() {
        // Small hand-sized case: hidden=3, ffn_dim=4
        let x = arr1(&[0.5_f32, -0.3, 1.0]);
        let gate_w = arr2(&[
            [0.1_f32, -0.2, 0.3],
            [0.4, 0.5, -0.1],
            [-0.3, 0.2, 0.4],
            [0.1, 0.1, -0.2],
        ]);
        let up_w = arr2(&[
            [0.2_f32, 0.1, -0.3],
            [-0.1, 0.4, 0.2],
            [0.3, -0.2, 0.1],
            [0.0, 0.3, 0.2],
        ]);
        let down_w = arr2(&[
            [0.1_f32, 0.2, -0.1, 0.3],
            [0.4, -0.2, 0.1, 0.1],
            [-0.3, 0.1, 0.2, -0.1],
        ]);
        // Forward helper
        let fwd = |xi: &Array1<f32>| -> Array1<f32> {
            let g_pre: Array1<f32> = (0..gate_w.nrows())
                .map(|i| (0..xi.len()).map(|j| gate_w[[i, j]] * xi[j]).sum())
                .collect();
            let u: Array1<f32> = (0..up_w.nrows())
                .map(|i| (0..xi.len()).map(|j| up_w[[i, j]] * xi[j]).sum())
                .collect();
            let g: Array1<f32> = g_pre.map(|&z| z / (1.0 + (-z).exp()));
            let act: Array1<f32> = g.iter().zip(u.iter()).map(|(&a, &b)| a * b).collect();
            (0..down_w.nrows())
                .map(|k| (0..down_w.ncols()).map(|i| down_w[[k, i]] * act[i]).sum())
                .collect()
        };
        // Loss = sum(out) so d_out = ones
        let d_out = Array1::from_elem(3, 1.0_f32);
        let dx_analytical = gated_ffn_backward(
            x.view(),
            gate_w.view(),
            up_w.view(),
            down_w.view(),
            d_out.view(),
        );
        let h = 1e-4_f32;
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += h;
            let mut xm = x.clone();
            xm[i] -= h;
            let lp: f32 = fwd(&xp).iter().sum();
            let lm: f32 = fwd(&xm).iter().sum();
            let num = (lp - lm) / (2.0 * h);
            let err = (dx_analytical[i] - num).abs();
            assert!(
                err < 1e-2,
                "dx[{i}]: analytical {} vs numerical {num}",
                dx_analytical[i]
            );
        }
    }

    #[test]
    fn target_delta_opts_default_matches_python_reference() {
        let d = TargetDeltaOpts::default();
        assert_eq!(d.steps, 60);
        assert!((d.lr - 0.5).abs() < 1e-6);
        assert!((d.kl_weight - 0.0625).abs() < 1e-6);
        assert!(!d.normalise);
    }

    #[test]
    fn optimise_target_delta_happy_path_returns_finite_delta() {
        // Drive the actual Adam loop with tiny steps so we touch every
        // line of the optimisation body without burning CPU.
        let weights = make_test_weights();
        let opts = TargetDeltaOpts {
            steps: 2,
            lr: 0.1,
            kl_weight: 0.0,
            normalise: false,
        };
        let install_layer = weights.num_layers - 1;
        let result = optimise_target_delta(&weights, &[0u32], 1, install_layer, opts)
            .expect("happy path must succeed");
        assert_eq!(result.layer, install_layer);
        assert_eq!(result.delta.len(), weights.hidden_size);
        assert!(result.delta.iter().all(|v| v.is_finite()));
        assert!(result.final_loss.is_finite());
        assert!(result.baseline_loss.is_finite());
    }

    #[test]
    fn optimise_target_delta_with_kl_weight_runs_kl_branch() {
        // kl_weight != 0.0 → triggers the `dlogits += kl * (cur - base)`
        // and `kl_val` accumulation branches.
        let weights = make_test_weights();
        let opts = TargetDeltaOpts {
            steps: 1,
            lr: 0.1,
            kl_weight: 0.05,
            normalise: false,
        };
        let install_layer = weights.num_layers - 1;
        let result = optimise_target_delta(&weights, &[0u32], 1, install_layer, opts)
            .expect("kl-weighted opt must succeed");
        assert!(result.final_loss.is_finite());
    }

    #[test]
    fn optimise_target_delta_with_normalise_returns_unit_norm_delta() {
        let weights = make_test_weights();
        let opts = TargetDeltaOpts {
            steps: 2,
            lr: 0.5,
            kl_weight: 0.0,
            normalise: true,
        };
        let install_layer = weights.num_layers - 1;
        let result = optimise_target_delta(&weights, &[0u32, 1], 2, install_layer, opts)
            .expect("normalise must succeed");
        let norm: f32 = result.delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Either zero (degenerate optimization) or unit norm.
        assert!(
            norm == 0.0 || (norm - 1.0).abs() < 1e-4,
            "normalised delta should have unit norm or be zero, got {norm}"
        );
    }

    #[test]
    fn optimise_target_delta_rejects_mid_layer() {
        let weights = make_test_weights();
        // num_layers=2 in test fixture, so install_layer=0 is mid-layer.
        let err = optimise_target_delta(&weights, &[0u32], 0, 0, TargetDeltaOpts::default())
            .expect_err("mid-layer install must error");
        assert!(
            err.contains("only install_layer = n_layers-1"),
            "expected mid-layer rejection, got: {err}"
        );
    }

    #[test]
    fn optimise_target_delta_rejects_layer_out_of_range() {
        let weights = make_test_weights();
        let err = optimise_target_delta(
            &weights,
            &[0u32],
            0,
            weights.num_layers + 5,
            TargetDeltaOpts::default(),
        )
        .expect_err("install_layer >= n_layers must error");
        assert!(err.contains("≥ n_layers"), "got: {err}");
    }

    #[test]
    fn softmax_1d_sums_to_one_and_handles_extreme_logits() {
        // Tiny + huge logits exercise the numerical-stability shift.
        let logits = arr1(&[100.0f32, 100.5, 99.0, -1000.0]);
        let probs = softmax_1d(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum: {sum}");
        // The smallest logit should produce ~0 probability.
        assert!(probs[3] < 1e-30 || probs[3] == 0.0);
        // Max-logit index should hold the largest probability.
        assert!(probs[1] > probs[0]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn rmsnorm_backward_finite_difference() {
        // Analytical gradient should match numerical at a random point.
        let x = arr1(&[0.5_f32, 1.0, -0.5, 2.0]);
        let w = arr1(&[1.0_f32, 0.5, 2.0, 1.5]);
        let eps = 1e-5_f32;

        // Forward helper
        let fwd = |xi: &Array1<f32>| -> Array1<f32> {
            let d = xi.len() as f32;
            let ms = xi.iter().map(|v| v * v).sum::<f32>() / d;
            let rms = (ms + eps).sqrt();
            xi.iter()
                .zip(w.iter())
                .map(|(xv, wv)| (xv / rms) * wv)
                .collect()
        };

        // Loss = sum of y (so dy = ones)
        let dy = Array1::from_elem(x.len(), 1.0_f32);
        let dx_analytical = rmsnorm_backward_pos(x.view(), w.view(), dy.view(), eps);

        // Numerical dx via finite difference
        let h = 1e-4_f32;
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += h;
            let mut xm = x.clone();
            xm[i] -= h;
            let loss_p: f32 = fwd(&xp).iter().sum();
            let loss_m: f32 = fwd(&xm).iter().sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let err = (dx_analytical[i] - num).abs();
            assert!(
                err < 1e-2,
                "dx[{i}]: analytical {} vs numerical {num} (err {err})",
                dx_analytical[i]
            );
        }
    }
}
