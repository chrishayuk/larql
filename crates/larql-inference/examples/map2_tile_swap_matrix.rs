//! MAP-2: the tile-swap matrix.
//!
//! Registered as chuk-experiments programme `map`, experiment `map-2`.
//! Pre-registration design: see the `map-2` write-up in the chuk-experiments
//! registry (recorded 2026-08-02, before this code existed).
//!
//! MAP-1 (`map1_coordinate_chart_conjugacy.rs`) established that a hidden
//! coordinate needs its FFN chart to mean anything — exactly, to
//! floating-point noise, on both a synthetic fixture and real Gemma 3 4B.
//! MAP-2 asks the next question: do different layers' FFN tiles form
//! **locality/compatibility structure** (a graph), or is exact conjugacy
//! the only structure there is?
//!
//! For a captured waypoint `x_l` (real post-attention residual at layer
//! `l`, last token), this measures what happens when its FFN delta is
//! computed by a DIFFERENT layer's tile `F_m` instead of its own `F_l`,
//! then that swapped delta is spliced back into the residual stream and
//! the REST of the model runs normally on it (real attention + FFN +
//! PLE/norm for every layer after `l`) through to final logits.
//!
//! Scope decision, explicit and load-bearing (mirrors MAP-1's "score the
//! FFN delta, not the full waypoint" choice): the splice itself replaces
//! the post-layer residual with `x_l + F_m(x_l)` directly, bypassing
//! whatever PLE / post-feedforward-norm / scalar adjustment the model's
//! own layer body would have applied at layer `l`. That means even the
//! `m == l` ("native-reconstructed") cell is NOT bit-identical to an
//! unmodified forward pass — it is this experiment's own reconstruction
//! floor. Every cross-tile cell is scored AGAINST that same
//! native-reconstructed floor (not against the true unmodified forward),
//! which isolates "cost of swapping the tile" from "cost of this
//! experiment's simplified splice." The one-time gap between the
//! native-reconstructed floor and a real unmodified forward is reported
//! separately as `abstraction_floor_kl` so that isolation is checked, not
//! assumed.
//!
//! Explicit, documented subsample (see module docs above on "no silent
//! caps"): this is NOT an exhaustive all-layers-by-all-layers sweep. It
//! covers 3 "home" layers spanning early/mid/late depth, each against a
//! handful of nearby and distant partners, computed in BOTH directions
//! (so reciprocal asymmetry is directly visible), on a single prompt.
//! Scaling to more homes/partners/prompts is the natural follow-up once
//! this harness is validated.
//!
//! Usage: `LARQL_MAP2_VINDEX=/path/to/model.vindex cargo run --release -p larql-inference --example map2_tile_swap_matrix`
//! Without the env var, only phase 0 (a 2-layer synthetic-fixture harness
//! smoke test) runs — a real cross-layer compatibility matrix needs real
//! depth, which the 2-layer TinyModel fixture cannot provide.

use ndarray::{Array1, Array2, Axis};

use larql_compute::ffn::{gelu_tanh_gate_up, silu_gate_up};
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::{
    logit_lens_topk, trace_forward, trace_forward_full_hooked, LayerHook, RecordHook,
};
use larql_inference::test_utils::make_test_weights;
use larql_inference::FfnBackend;
use larql_models::{ModelWeights, WeightArray};

/// One layer's extracted FFN tile: owned copies of its gate/up/down
/// matrices, so a tile captured at layer `m` can be applied to a waypoint
/// captured at a different layer `l` without holding two live borrows of
/// `weights.tensors`.
struct Tile {
    layer: usize,
    gate: Array2<f32>,
    up: Array2<f32>,
    down: Array2<f32>,
}

fn extract_tile(weights: &ModelWeights, layer: usize) -> Tile {
    let arch = &*weights.arch;
    let get = |key: String| -> Array2<f32> {
        let w: &WeightArray = weights
            .tensors
            .get(&key)
            .unwrap_or_else(|| panic!("missing tensor {key}"));
        w.to_owned()
    };
    Tile {
        layer,
        gate: get(arch.ffn_gate_key(layer)),
        up: get(arch.ffn_up_key(layer)),
        down: get(arch.ffn_down_key(layer)),
    }
}

/// Same gated-FFN math as MAP-1's `ffn_row`: `down(act(gate(x)) [*] up(x))`.
/// Duplicated rather than shared — two ~10-line example files don't
/// justify a crate-level abstraction.
fn ffn_row(x: &Array1<f32>, tile: &Tile, gelu_tanh: bool) -> Array1<f32> {
    let x2 = x.clone().insert_axis(Axis(0));
    let gate = x2.dot(&tile.gate.t());
    let up = x2.dot(&tile.up.t());
    let act = if gelu_tanh {
        gelu_tanh_gate_up(&gate, &up)
    } else {
        silu_gate_up(&gate, &up)
    };
    act.dot(&tile.down.t()).row(0).to_owned()
}

/// Splices `x_l + override_delta` into the residual stream at
/// `target_layer`'s post-layer point, discarding whatever the model's own
/// layer body (FFN + PLE + post-ffn norm/scalar) actually produced there.
/// Captures the pre-splice residual (`x_l`) at `on_post_attention` so the
/// caller can read it back out afterward.
struct SpliceHook {
    target_layer: usize,
    override_delta: Array1<f32>,
    captured_x: Option<Array1<f32>>,
}

impl LayerHook for SpliceHook {
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {
        if layer == self.target_layer {
            self.captured_x = Some(h.row(h.nrows() - 1).to_owned());
        }
    }
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        if layer == self.target_layer {
            if let Some(x) = &self.captured_x {
                let last = h.nrows() - 1;
                let spliced = x + &self.override_delta;
                h.row_mut(last).assign(&spliced);
            }
        }
    }
}

fn max_abs_diff(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The same sanity gate MAP-1 used: a native (`m == l`) tile applied via
/// our hand-rolled `ffn_row` must match the library's own
/// `WeightFfn::forward` before anything built on top of `ffn_row` (every
/// cross-tile cell in this file) can be trusted.
fn sanity_check_ffn_row(
    ffn: &WeightFfn,
    x: &Array1<f32>,
    native_tile: &Tile,
    native_delta: &Array1<f32>,
    label: &str,
) {
    let x_row = x.clone().insert_axis(Axis(0));
    let lib_out = ffn.forward(native_tile.layer, &x_row).row(0).to_owned();
    let diff = max_abs_diff(native_delta, &lib_out);
    println!("  sanity [{label}]: hand-rolled F(x) vs library WeightFfn::forward at L{} max|Δ| = {diff:.3e}", native_tile.layer);
    assert!(
        diff < 1e-3,
        "hand-rolled ffn_row diverged from the library's dense_ffn_forward at layer {}",
        native_tile.layer
    );
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

/// Deterministic seeded pseudo-random unit vector (LCG, same generator
/// style as MAP-1's derangement), scaled to `target_norm`. Used to build
/// the norm-matched random-delta control.
fn random_delta(dim: usize, target_norm: f32, seed: u64) -> Array1<f32> {
    let mut state = seed;
    let mut next_f32 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let raw: Array1<f32> = Array1::from_shape_fn(dim, |_| next_f32());
    let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        raw
    } else {
        raw * (target_norm / norm)
    }
}

/// Full vocab distribution at a residual, SORTED descending by
/// probability — via the model's own logit lens (final norm + lm_head +
/// softmax), reused rather than reimplemented. `logit_lens_topk` with
/// `k = vocab_size` returns every token already sorted, so this is free.
fn full_sorted(weights: &ModelWeights, residual: &[f32]) -> Vec<(u32, f32)> {
    logit_lens_topk(weights, residual, weights.vocab_size)
}

fn dense_from_sorted(sorted: &[(u32, f32)], vocab: usize) -> Vec<f32> {
    let mut dense = vec![0.0f32; vocab];
    for &(id, p) in sorted {
        dense[id as usize] = p;
    }
    dense
}

/// KL(p || q) over dense full-vocab distributions. Kept as SECONDARY
/// context, not the decisive metric: once `q`'s probability at `p`'s peak
/// token drops below the clamp epsilon, this saturates at a ceiling
/// (`p_top * ln(1/EPS)`) set by the arbitrary epsilon choice, not by how
/// much worse one wrong answer is than another. That saturation is real
/// and was observed in an earlier run of this experiment (several cells
/// landed within 1e-4 of each other despite very different candidate
/// distributions) — `rank_of_native_top1` below is the metric that
/// doesn't have this failure mode, and is what the write-up leans on.
fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    const EPS: f32 = 1e-30;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi <= 0.0 {
                0.0
            } else {
                pi * (pi.max(EPS) / qi.max(EPS)).ln()
            }
        })
        .sum()
}

/// Position of `token_id` in a descending-sorted full-vocab distribution
/// (0 = still rank-1). Unlike KL or raw probability, this never saturates
/// — a token can always be ranked further down regardless of floating
/// point precision, so it stays informative exactly where the KL/NLL
/// metrics hit their float32 floor.
fn rank_of(sorted: &[(u32, f32)], token_id: u32) -> usize {
    sorted
        .iter()
        .position(|&(id, _)| id == token_id)
        .unwrap_or(sorted.len())
}

/// Runs the model with `SpliceHook` at `layer` splicing in `delta`, returns
/// the final-layer full distribution, sorted descending by probability.
fn run_spliced(
    weights: &ModelWeights,
    ffn: &WeightFfn,
    prompt: &[u32],
    layer: usize,
    delta: Array1<f32>,
    last_layer: usize,
) -> Vec<(u32, f32)> {
    let mut hook = SpliceHook {
        target_layer: layer,
        override_delta: delta,
        captured_x: None,
    };
    let trace = trace_forward_full_hooked(
        weights,
        prompt,
        &[last_layer],
        false,
        0,
        false,
        ffn,
        &mut hook,
    );
    full_sorted(weights, &trace.residuals[0].1)
}

struct CellResult {
    from_layer: usize,
    tile_layer: usize,
    cosine_vs_native: f32,
    rel_l2_vs_native: f32,
    norm_ratio: f32,
    downstream_kl: f32,
    downstream_kl_random_control: f32,
    top1_preserved: bool,
    top1_preserved_random_control: bool,
    /// Rank (0 = still #1) of the NATIVE reconstruction's top token within
    /// the swapped-tile distribution. The decisive metric: doesn't
    /// saturate the way KL does, so it distinguishes "moderately
    /// dislodged" from "catastrophically dislodged" even when both hit
    /// KL's float32 floor.
    native_top1_rank_under_swap: usize,
    native_top1_rank_under_random: usize,
}

/// The parts of a run that are fixed across every cell of the swap matrix.
///
/// Both call sites pass the same five values to every cell, so they travel
/// together rather than as five of `evaluate_cell`'s arguments.
struct RunCtx<'a> {
    weights: &'a ModelWeights,
    ffn: &'a WeightFfn<'a>,
    prompt: &'a [u32],
    last_layer: usize,
    gelu_tanh: bool,
}

fn evaluate_cell(
    ctx: &RunCtx,
    x_l: &Array1<f32>,
    native_tile: &Tile,
    foreign_tile: &Tile,
    native_delta: &Array1<f32>,
    p_native: &[(u32, f32)],
    random_seed: u64,
) -> CellResult {
    let foreign_delta = ffn_row(x_l, foreign_tile, ctx.gelu_tanh);
    let native_norm = native_delta.iter().map(|v| v * v).sum::<f32>().sqrt();
    let random = random_delta(x_l.len(), native_norm, random_seed);

    let p_swap = run_spliced(
        ctx.weights,
        ctx.ffn,
        ctx.prompt,
        native_tile.layer,
        foreign_delta.clone(),
        ctx.last_layer,
    );
    let p_random = run_spliced(
        ctx.weights,
        ctx.ffn,
        ctx.prompt,
        native_tile.layer,
        random,
        ctx.last_layer,
    );
    let native_top1 = p_native[0].0;
    let vocab = ctx.weights.vocab_size;
    let p_native_dense = dense_from_sorted(p_native, vocab);

    CellResult {
        from_layer: native_tile.layer,
        tile_layer: foreign_tile.layer,
        cosine_vs_native: cosine(&foreign_delta, native_delta),
        rel_l2_vs_native: rel_l2(&foreign_delta, native_delta),
        norm_ratio: foreign_delta.iter().map(|v| v * v).sum::<f32>().sqrt()
            / native_norm.max(1e-12),
        downstream_kl: kl_divergence(&p_native_dense, &dense_from_sorted(&p_swap, vocab)),
        downstream_kl_random_control: kl_divergence(
            &p_native_dense,
            &dense_from_sorted(&p_random, vocab),
        ),
        top1_preserved: native_top1 == p_swap[0].0,
        top1_preserved_random_control: native_top1 == p_random[0].0,
        native_top1_rank_under_swap: rank_of(&p_swap, native_top1),
        native_top1_rank_under_random: rank_of(&p_random, native_top1),
    }
}

fn print_cell(c: &CellResult) {
    println!(
        "  waypoint@L{:>2} through tile@L{:>2}:  cos={:>7.4}  rel_L2={:>8.4}  norm_ratio={:>6.3}  KL={:>9.5} (rand={:>9.5})  top1_ok={:<5} (rand={:<5})  native-top1 rank under swap={:>7} (rand={:>7})",
        c.from_layer, c.tile_layer, c.cosine_vs_native, c.rel_l2_vs_native, c.norm_ratio,
        c.downstream_kl, c.downstream_kl_random_control, c.top1_preserved, c.top1_preserved_random_control,
        c.native_top1_rank_under_swap, c.native_top1_rank_under_random
    );
}

/// Phase 0: harness smoke test on the synthetic 2-layer TinyModel fixture.
/// This is NOT a compatibility-matrix demonstration — 2 layers is too
/// shallow for locality structure to mean anything — it only proves the
/// splice/KL/control machinery runs and the diagonal (m == l) cell scores
/// as a near-zero-KL self-comparison, before spending real compute on
/// Gemma.
fn run_smoke_test() {
    println!("=== phase 0: harness smoke test (synthetic TinyModel, 2 layers) ===");
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let gelu_tanh = weights.arch.gate_up_is_gelu_tanh();
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let last_layer = weights.num_layers - 1;

    let mut record = RecordHook::for_layers([0usize]);
    let _ = trace_forward_full_hooked(&weights, &prompt, &[0], false, 0, false, &ffn, &mut record);
    let x0 = record
        .post_attention
        .get(&0)
        .unwrap()
        .row(prompt.len() - 1)
        .to_owned();

    let tile0 = extract_tile(&weights, 0);
    let tile1 = extract_tile(&weights, 1);
    let native_delta = ffn_row(&x0, &tile0, gelu_tanh);

    sanity_check_ffn_row(&ffn, &x0, &tile0, &native_delta, "phase 0 smoke test");

    let p_native = run_spliced(&weights, &ffn, &prompt, 0, native_delta.clone(), last_layer);
    let ctx = RunCtx {
        weights: &weights,
        ffn: &ffn,
        prompt: &prompt,
        last_layer,
        gelu_tanh,
    };
    let cell = evaluate_cell(&ctx, &x0, &tile0, &tile1, &native_delta, &p_native, 42);
    print_cell(&cell);
    let dense_native = dense_from_sorted(&p_native, weights.vocab_size);
    assert!(
        kl_divergence(&dense_native, &dense_native) < 1e-6,
        "KL of a distribution against itself must be ~0"
    );
    assert_eq!(
        rank_of(&p_native, p_native[0].0),
        0,
        "native's own top token must rank 0 against itself"
    );
    println!("  (smoke test passed — harness is sound)\n");
}

/// Unordered (home, partner) pairs for the real-Gemma sweep. Each is
/// evaluated in BOTH directions (home→partner and partner→home) so
/// reciprocal asymmetry is directly visible, not assumed. See the module
/// doc comment for why this is a deliberate subsample, not a full sweep.
const HOME_LAYERS: [usize; 3] = [6, 17, 28];
const OFFSETS: [i64; 3] = [-4, -1, 2];

fn build_pairs(num_layers: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for &home in &HOME_LAYERS {
        if home >= num_layers {
            continue;
        }
        for &off in &OFFSETS {
            let partner = home as i64 + off;
            if partner < 0 || partner as usize >= num_layers || partner as usize == home {
                continue;
            }
            let pair = (home, partner as usize);
            if !pairs.contains(&pair) && !pairs.contains(&(pair.1, pair.0)) {
                pairs.push(pair);
            }
        }
    }
    pairs
}

fn run_real_gemma_sweep(vindex_path: &str) {
    println!("=== phase 1: real Gemma tile-swap sweep ===");
    println!("loading real vindex: {vindex_path}");
    let dir = std::path::Path::new(vindex_path);
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_with_opts(
        dir,
        &mut cb,
        larql_vindex::LoadWeightsOptions::default(),
    )
    .expect("load real vindex weights");
    let tokenizer = larql_vindex::load_vindex_tokenizer(dir).expect("load vindex tokenizer");
    let prompt =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, "The capital of France is")
            .expect("tokenize prompt");
    let ffn = WeightFfn { weights: &weights };
    let gelu_tanh = weights.arch.gate_up_is_gelu_tanh();
    let last_layer = weights.num_layers - 1;

    let pairs = build_pairs(weights.num_layers);
    let mut touched_layers: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    touched_layers.sort_unstable();
    touched_layers.dedup();
    println!(
        "{} unordered pairs ({} directed cells) over {} distinct layers: {:?}\n",
        pairs.len(),
        pairs.len() * 2,
        touched_layers.len(),
        touched_layers
    );

    // One capture + native-reconstruction pass per touched layer.
    let mut waypoints: std::collections::HashMap<usize, Array1<f32>> =
        std::collections::HashMap::new();
    let mut tiles: std::collections::HashMap<usize, Tile> = std::collections::HashMap::new();
    let mut native_deltas: std::collections::HashMap<usize, Array1<f32>> =
        std::collections::HashMap::new();
    let mut native_probs: std::collections::HashMap<usize, Vec<(u32, f32)>> =
        std::collections::HashMap::new();

    for &l in &touched_layers {
        let mut record = RecordHook::for_layers([l]);
        let _ =
            trace_forward_full_hooked(&weights, &prompt, &[l], false, 0, false, &ffn, &mut record);
        let x_l = record
            .post_attention
            .get(&l)
            .unwrap()
            .row(prompt.len() - 1)
            .to_owned();
        let tile = extract_tile(&weights, l);
        let native_delta = ffn_row(&x_l, &tile, gelu_tanh);
        sanity_check_ffn_row(
            &ffn,
            &x_l,
            &tile,
            &native_delta,
            &format!("real sweep L{l}"),
        );
        let p_native = run_spliced(&weights, &ffn, &prompt, l, native_delta.clone(), last_layer);
        waypoints.insert(l, x_l);
        tiles.insert(l, tile);
        native_deltas.insert(l, native_delta);
        native_probs.insert(l, p_native);
    }

    // Report the abstraction floor once, comparing an m==l reconstruction
    // against a genuinely unmodified forward on the same prompt.
    let true_baseline = trace_forward(&weights, &prompt, &[last_layer], false, 0).residuals[0]
        .1
        .clone();
    let p_true = full_sorted(&weights, &true_baseline);
    let some_layer = touched_layers[0];
    let abstraction_floor_kl = kl_divergence(
        &dense_from_sorted(&p_true, weights.vocab_size),
        &dense_from_sorted(&native_probs[&some_layer], weights.vocab_size),
    );
    let abstraction_floor_rank = rank_of(&native_probs[&some_layer], p_true[0].0);
    println!(
        "abstraction floor (native-reconstructed@L{some_layer} vs true unmodified forward): KL={abstraction_floor_kl:.5}, true-top1's rank under native-reconstruction={abstraction_floor_rank} — every cell below is scored against native-reconstructed, not this true baseline, so this is a floor/context number, not part of the decisive comparison (which holds the same abstraction constant on both the swap and random arms).\n"
    );

    let mut results = Vec::new();
    let mut seed = 1000u64;
    let ctx = RunCtx {
        weights: &weights,
        ffn: &ffn,
        prompt: &prompt,
        last_layer,
        gelu_tanh,
    };
    for &(a, b) in &pairs {
        for &(from, to) in &[(a, b), (b, a)] {
            let cell = evaluate_cell(
                &ctx,
                &waypoints[&from],
                &tiles[&from],
                &tiles[&to],
                &native_deltas[&from],
                &native_probs[&from],
                seed,
            );
            print_cell(&cell);
            results.push(cell);
            seed += 1;
        }
    }

    println!("\n=== summary ===");
    let n = results.len() as f32;
    let mean_kl_swap: f32 = results.iter().map(|c| c.downstream_kl).sum::<f32>() / n;
    let mean_kl_random: f32 = results
        .iter()
        .map(|c| c.downstream_kl_random_control)
        .sum::<f32>()
        / n;
    let top1_rate = results.iter().filter(|c| c.top1_preserved).count() as f32 / n;
    let top1_rate_random = results
        .iter()
        .filter(|c| c.top1_preserved_random_control)
        .count() as f32
        / n;
    let mean_rank_swap: f64 = results
        .iter()
        .map(|c| c.native_top1_rank_under_swap as f64)
        .sum::<f64>()
        / results.len() as f64;
    let mean_rank_random: f64 = results
        .iter()
        .map(|c| c.native_top1_rank_under_random as f64)
        .sum::<f64>()
        / results.len() as f64;
    let swap_beats_random = results
        .iter()
        .filter(|c| c.native_top1_rank_under_swap < c.native_top1_rank_under_random)
        .count();
    let random_beats_swap = results
        .iter()
        .filter(|c| c.native_top1_rank_under_random < c.native_top1_rank_under_swap)
        .count();
    let tied = results.len() - swap_beats_random - random_beats_swap;

    println!("{} directed cells", results.len());
    println!("mean downstream KL, swapped tile:   {mean_kl_swap:.5}  (context metric — see kl_divergence's doc comment on saturation)");
    println!("mean downstream KL, random control: {mean_kl_random:.5}");
    println!(
        "top-1 token preserved, swapped tile:   {:.1}%",
        top1_rate * 100.0
    );
    println!(
        "top-1 token preserved, random control:  {:.1}%",
        top1_rate_random * 100.0
    );
    println!("mean rank of native's top token, swapped tile:  {mean_rank_swap:.1}  (0 = perfectly preserved, {} = vocab size)", weights.vocab_size);
    println!("mean rank of native's top token, random control: {mean_rank_random:.1}");
    println!(
        "per-cell comparison: swap ranks native's top token BETTER than random in {swap_beats_random}/{} cells, \
         WORSE in {random_beats_swap}/{}, tied in {tied}/{}",
        results.len(), results.len(), results.len()
    );
    println!(
        "\nDecisive metric is the per-cell rank comparison, not the mean (a few near-vocab-size ranks can\n\
         dominate a mean). If swap beats random in most cells, that is structured off-diagonal\n\
         compatibility — evidence tiles form a locality graph, not just independent per-layer functions.\n\
         If swap and random split roughly 50/50, a wrong tile is exactly as informative as noise of the\n\
         same magnitude, and MAP-2's positive hypothesis is refuted at this depth/offset/prompt scale — a\n\
         scale limitation to state explicitly, not a general claim about all layers/models/prompts."
    );
}

fn main() {
    println!("=== MAP-2: the tile-swap matrix ===\n");
    run_smoke_test();

    match std::env::var("LARQL_MAP2_VINDEX") {
        Ok(path) => run_real_gemma_sweep(&path),
        Err(_) => {
            println!(
                "(phase 1) skipped — set LARQL_MAP2_VINDEX=/path/to/model.vindex \
                 (e.g. the same dense f16 Gemma vindex used by map1_coordinate_chart_conjugacy) \
                 to run the real cross-layer tile-swap sweep"
            );
        }
    }
}
