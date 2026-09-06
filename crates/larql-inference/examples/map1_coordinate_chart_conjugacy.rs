//! MAP-1: coordinate-chart conjugacy.
//!
//! Registered as chuk-experiments programme `map`, experiment `map-1`.
//!
//! Claim under test: a residual-stream coordinate (e.g. "component 417")
//! has no intrinsic meaning. It is only meaningful relative to the FFN
//! weights ("chart") that consume it. Concretely, for an invertible
//! permutation `P` of the hidden coordinates and a gated FFN sublayer
//! `F`, define the re-parametrized map:
//!
//! ```text
//! F^P(z) = P F(P^-1 z)      via   W_gate^P = W_gate P^-1
//!                                 W_up^P   = W_up P^-1
//!                                 W_down^P = P W_down
//! ```
//!
//! Three arms, starting from a genuine captured post-attention residual
//! `x` and its next waypoint `y = x + F(x)`:
//!
//! ```text
//! baseline:                          x  -> F   -> y
//! permuted coords, ORIGINAL chart:   Px -> F   -> (undefined / wrong)
//! permuted coords, PERMUTED chart:   Px -> F^P -> Py
//! ```
//!
//! The decisive, falsifiable prediction: the third arm reproduces `P(y)`
//! near-exactly (floating-point noise only), while the second arm — fed
//! permuted coordinates through the unpermuted chart — does not. That
//! would show the coordinate has no absolute meaning; only the
//! (coordinate, chart) pair does.
//!
//! Two phases:
//! 1. Fixture A — synthetic `TinyModel` weights (f32, no vindex needed).
//! 2. A real dense vindex, if `LARQL_MAP1_VINDEX` points at one on disk
//!    (e.g. a Gemma 3 4B f16 vindex). Skipped with a note otherwise.
//!
//! Usage: `cargo run --release -p larql-inference --example map1_coordinate_chart_conjugacy`

use ndarray::{Array1, Array2, ArrayView2, Axis};

use larql_compute::ffn::{gelu_tanh_gate_up, silu_gate_up};
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::{trace_forward_full_hooked, RecordHook};
use larql_inference::test_utils::make_test_weights;
use larql_inference::FfnBackend;
use larql_models::{ModelWeights, WeightArray};

/// Deterministic derangement (fixed-point-free permutation) of `0..n`,
/// seeded so the run is reproducible. Fisher-Yates shuffle + a repair
/// pass — for the sizes used here (16, and low thousands) the repair
/// pass reliably removes the handful of fixed points a random shuffle
/// leaves behind.
fn derangement(n: usize, seed: u64) -> Vec<usize> {
    assert!(n > 1, "no derangement exists for n <= 1");
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed;
    let mut next_u64 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    for i in (1..n).rev() {
        let j = (next_u64() as usize) % (i + 1);
        perm.swap(i, j);
    }
    for i in 0..n {
        if perm[i] == i {
            let j = (i + 1) % n;
            perm.swap(i, j);
        }
    }
    debug_assert!(
        (0..n).all(|i| perm[i] != i),
        "repair pass left a fixed point"
    );
    perm
}

/// `z[k] = x[perm[k]]` — gather x's coordinates into the permuted frame.
fn permute_vector(x: &Array1<f32>, perm: &[usize]) -> Array1<f32> {
    Array1::from_shape_fn(x.len(), |k| x[perm[k]])
}

/// `new[:, k] = old[:, perm[k]]` — permute the input (column) axis.
/// Used for `W_gate` / `W_up`, whose columns index the hidden/residual
/// coordinate that the permutation acts on.
fn permute_columns(w: &WeightArray, perm: &[usize]) -> Array2<f32> {
    Array2::from_shape_fn((w.nrows(), w.ncols()), |(i, k)| w[[i, perm[k]]])
}

/// `new[k, :] = old[perm[k], :]` — permute the output (row) axis.
/// Used for `W_down`, whose rows index the hidden/residual coordinate
/// the FFN writes back into: `W_down^P = P @ W_down`, i.e. row `k` of the
/// permuted matrix is row `perm[k]` of the original (NOT `inv_perm[k]` —
/// `P`'s own definition here is `(Pv)[k] = v[perm[k]]`, so permuting an
/// output axis by `P` uses `perm` directly; `inv_perm` is only needed to
/// invert an *input* axis, which is what `permute_vector`/`permute_columns`
/// use it for).
fn permute_rows(w: &WeightArray, perm: &[usize]) -> Array2<f32> {
    Array2::from_shape_fn((w.nrows(), w.ncols()), |(k, j)| w[[perm[k], j]])
}

/// One gated-FFN row forward: `down(act(gate(x)) [*] up(x))`. Mirrors
/// `larql_compute::ffn::weight::dense_ffn_forward`'s gated branch, but
/// parametrized directly on explicit gate/up/down matrices so it can be
/// evaluated against permuted weights that don't live in a `ModelWeights`.
fn ffn_row(
    x: &Array1<f32>,
    w_gate: ArrayView2<f32>,
    w_up: ArrayView2<f32>,
    w_down: ArrayView2<f32>,
    gelu_tanh: bool,
) -> Array1<f32> {
    let x2 = x.clone().insert_axis(Axis(0));
    let gate = x2.dot(&w_gate.t());
    let up = x2.dot(&w_up.t());
    let act = if gelu_tanh {
        gelu_tanh_gate_up(&gate, &up)
    } else {
        silu_gate_up(&gate, &up)
    };
    act.dot(&w_down.t()).row(0).to_owned()
}

fn max_abs_diff(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn rel_l2(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|y| y.powi(2)).sum::<f32>().sqrt();
    if den == 0.0 {
        num
    } else {
        num / den
    }
}

fn cosine(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

struct ArmMetrics {
    max_abs_diff: f32,
    rel_l2: f32,
    cosine: f32,
}

impl ArmMetrics {
    fn against(candidate: &Array1<f32>, target: &Array1<f32>) -> Self {
        Self {
            max_abs_diff: max_abs_diff(candidate, target),
            rel_l2: rel_l2(candidate, target),
            cosine: cosine(candidate, target),
        }
    }
}

struct Map1Result {
    hidden: usize,
    wrong: ArmMetrics,
    right: ArmMetrics,
}

/// Run the three-arm MAP-1 test at one (weights, layer, captured residual).
///
/// Panics (test failure) if:
/// - the hand-rolled `ffn_row` doesn't reproduce the library's own
///   `WeightFfn::forward` on the unpermuted input (the experiment would
///   be testing a bug in this file, not the conjugacy claim), or
/// - the permuted-coords + permuted-chart arm fails to reproduce `P(y)`
///   near-exactly (the actual falsifiable claim).
fn run_map1_on(
    label: &str,
    weights: &ModelWeights,
    layer: usize,
    x: &Array1<f32>,
    seed: u64,
) -> Map1Result {
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let w_gate = weights
        .tensors
        .get(&arch.ffn_gate_key(layer))
        .expect("gate tensor present");
    let w_up = weights
        .tensors
        .get(&arch.ffn_up_key(layer))
        .expect("up tensor present");
    let w_down = weights
        .tensors
        .get(&arch.ffn_down_key(layer))
        .expect("down tensor present");
    let gelu_tanh = arch.gate_up_is_gelu_tanh();

    // Sanity gate: our hand-rolled forward must match the library's dense
    // FFN forward exactly (unpermuted) before trusting anything built on it.
    let ffn = WeightFfn { weights };
    let x_row = x.clone().insert_axis(Axis(0));
    let lib_out = ffn.forward(layer, &x_row).row(0).to_owned();
    let custom_out = ffn_row(x, w_gate.view(), w_up.view(), w_down.view(), gelu_tanh);
    let sanity = max_abs_diff(&custom_out, &lib_out);
    println!("[{label}] layer {layer}, hidden={hidden}");
    println!("  sanity: hand-rolled F(x) vs library WeightFfn::forward  max|Δ| = {sanity:.3e}");
    assert!(
        sanity < 1e-3,
        "hand-rolled FFN forward diverged from the library's dense_ffn_forward — \
         fix the reimplementation before trusting the permutation result"
    );

    let perm = derangement(hidden, seed);

    let f_x = custom_out; // = F(x), the FFN's raw contribution (pre-norm, pre-residual-add)
    let f_x_p = permute_vector(&f_x, &perm); // P(F(x)) — what every "correct" arm must reproduce

    // The full next-waypoint `y = x + F(x)` is printed for narrative context,
    // but is NOT what the assertions below are scored against: x typically
    // dominates F(x) in norm, so comparing at the `y` level would mostly
    // measure "does a permutation permute a vector" (trivially yes) and
    // dilute the actual map-conjugacy signal down into the noise floor.
    // Scoring directly against `F(x)` isolates the FFN sublayer's claim.
    let y = x + &f_x;
    let _y_p_for_context = permute_vector(&y, &perm);

    let z = permute_vector(x, &perm); // permuted coordinates: Px

    // Arm B: permuted coordinates fed through the ORIGINAL, unpermuted chart.
    let f_z_wrong = ffn_row(&z, w_gate.view(), w_up.view(), w_down.view(), gelu_tanh);

    // Arm C: permuted coordinates AND permuted chart (W^P built per the
    // conjugacy formula: gate/up columns AND down rows all permuted by
    // `perm` — see `permute_columns`/`permute_rows` doc comments for why
    // both land on `perm` rather than one of them needing the inverse).
    let w_gate_p = permute_columns(w_gate, &perm);
    let w_up_p = permute_columns(w_up, &perm);
    let w_down_p = permute_rows(w_down, &perm);
    let f_z_right = ffn_row(
        &z,
        w_gate_p.view(),
        w_up_p.view(),
        w_down_p.view(),
        gelu_tanh,
    );

    let wrong = ArmMetrics::against(&f_z_wrong, &f_x_p);
    let right = ArmMetrics::against(&f_z_right, &f_x_p);

    println!(
        "  permuted coords + ORIGINAL chart (predicted WRONG):   max|Δ|={:.4e}  rel_L2={:.4}  cos={:.4}",
        wrong.max_abs_diff, wrong.rel_l2, wrong.cosine
    );
    println!(
        "  permuted coords + PERMUTED chart (predicted EXACT):   max|Δ|={:.4e}  rel_L2={:.4}  cos={:.4}",
        right.max_abs_diff, right.rel_l2, right.cosine
    );

    assert!(
        right.rel_l2 < 1e-3,
        "decisive claim FAILED: permuted-coords + permuted-chart should reproduce P(F(x)) \
         near-exactly, got rel_L2={:.4}",
        right.rel_l2
    );
    assert!(
        wrong.rel_l2 > 0.05,
        "permuted-coords + original-chart landed suspiciously close to P(F(x)) \
         (rel_L2={:.4}) — the derangement may be too weak to be decisive",
        wrong.rel_l2
    );

    Map1Result {
        hidden,
        wrong,
        right,
    }
}

fn main() {
    println!("=== MAP-1: coordinate-chart conjugacy ===\n");

    // ── Phase 1: Fixture A — synthetic TinyModel (f32, no vindex) ──────────
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let layer = 0usize;

    let mut record = RecordHook::for_layers([layer]);
    let _ = trace_forward_full_hooked(
        &weights,
        &prompt,
        &[layer],
        false,
        0,
        false,
        &ffn,
        &mut record,
    );
    let post_attn = record
        .post_attention
        .get(&layer)
        .expect("post-attention residual captured");
    let x = post_attn.row(post_attn.nrows() - 1).to_owned();

    let phase1 = run_map1_on(
        "Fixture A: synthetic TinyModel",
        &weights,
        layer,
        &x,
        /*seed=*/ 1,
    );

    // ── Phase 2: a real dense vindex, if one is pointed at ──────────────────
    let phase2 = match std::env::var("LARQL_MAP1_VINDEX") {
        Ok(path) => {
            println!("\n(phase 2) loading real vindex: {path}");
            let dir = std::path::Path::new(&path);
            let mut cb = larql_vindex::SilentLoadCallbacks;
            let weights = larql_vindex::load_model_weights_with_opts(
                dir,
                &mut cb,
                larql_vindex::LoadWeightsOptions::default(),
            )
            .expect("load real vindex weights");
            let tokenizer =
                larql_vindex::load_vindex_tokenizer(dir).expect("load vindex tokenizer");
            let prompt_ids = larql_inference::encode_prompt(
                &tokenizer,
                &*weights.arch,
                "The capital of France is",
            )
            .expect("tokenize prompt");

            let layer = weights.num_layers / 2;
            let ffn = WeightFfn { weights: &weights };
            let mut record = RecordHook::for_layers([layer]);
            let _ = trace_forward_full_hooked(
                &weights,
                &prompt_ids,
                &[layer],
                false,
                0,
                false,
                &ffn,
                &mut record,
            );
            let post_attn = record
                .post_attention
                .get(&layer)
                .expect("post-attention residual captured");
            let x = post_attn.row(post_attn.nrows() - 1).to_owned();

            Some(run_map1_on(
                "Real Gemma vindex",
                &weights,
                layer,
                &x,
                /*seed=*/ 2,
            ))
        }
        Err(_) => {
            println!(
                "\n(phase 2) skipped — set LARQL_MAP1_VINDEX=/path/to/model.vindex \
                 to repeat this against a real captured activation"
            );
            None
        }
    };

    println!("\n=== summary ===");
    println!(
        "Fixture A   (hidden={:>4}): wrong rel_L2={:.4}, cos={:.4}  |  right rel_L2={:.4e}, cos={:.6}",
        phase1.hidden, phase1.wrong.rel_l2, phase1.wrong.cosine, phase1.right.rel_l2, phase1.right.cosine
    );
    if let Some(p2) = &phase2 {
        println!(
            "Real Gemma  (hidden={:>4}): wrong rel_L2={:.4}, cos={:.4}  |  right rel_L2={:.4e}, cos={:.6}",
            p2.hidden, p2.wrong.rel_l2, p2.wrong.cosine, p2.right.rel_l2, p2.right.cosine
        );
    }
    println!(
        "\nBoth arms pass their assertions above ⇒ the coordinate has no intrinsic meaning: \
         only the (coordinate, chart) pair does. A residual permuted without its chart is \
         wrong; permuted together with its chart, the FFN sublayer's next waypoint commutes \
         with the permutation exactly."
    );
}
