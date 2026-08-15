//! Gather-contiguous Q4K kernel for known-pool routes (tasks #24/#25).
//!
//! For a route whose feature set is decided WITHOUT gate scores
//! (precomputed pool or cell router, no within-pool ranking), the
//! sparse walk gathers the selected rows' gate/up/down Q4K **bytes**
//! into contiguous buffers and runs the fused row kernels in one
//! cache-friendly pass — no f32 materialisation. Traced as
//! `sparse:gather_q4k`.

use rayon::prelude::*;

use super::WalkFfn;
use larql_vindex::GateIndex;

/// Output of [`WalkFfn::gather_q4k_accumulate`]: one position's output
/// row plus the per-feature values the fused kernels computed, all
/// aligned with the route's `feats` order. The scores ride into the
/// observation record so the runtime trace reports the EXECUTED
/// gate/up dots, not a re-scored twin (2026-07-30 review, item 17).
pub(super) struct GatherAccumulate {
    /// `[hidden]` output row (down accumulate over the route).
    pub out: Vec<f32>,
    /// Scalar activation per route feature.
    pub acts: Vec<f32>,
    /// Gate row-dot per route feature (recomputed from gathered bytes).
    pub gate_scores: Vec<f32>,
    /// Up row-dot per route feature.
    pub up_scores: Vec<f32>,
}

/// Contiguous Q4K bytes for a fixed feature set's gate/up/down rows —
/// the output of one gather, independent of when it runs.
///
/// [`gather_kquant_rows`] is the ONLY place this project selects and
/// copies these bytes. [`WalkFfn::gather_q4k_accumulate`] (BW-A/R4's
/// kernel — gathers fresh on every call) and
/// [`super::compact_dense::CompactDenseLayer`] (BW-B — gathers once,
/// offline) both build a `GatheredRows` through this same function, so
/// any measured difference between the two arms is the "when", never a
/// hidden difference in "what" gets gathered.
pub(super) struct GatheredRows {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
    pub gate_format: String,
    pub up_format: String,
    pub down_format: String,
    pub gate_bytes_per_row: usize,
    pub up_bytes_per_row: usize,
    pub down_bytes_per_row: usize,
    /// Number of rows gathered — `feats.len()` at gather time.
    pub k: usize,
}

/// Gather `feats`' gate/up/down Q4K rows into fresh contiguous buffers.
/// `None` under the same conditions the caller must fall back on: no
/// Q4K bytes for this layer, no feature-major down sidecar (the
/// interleaved down is stored transposed — a feature's down vector is a
/// strided column there, not a gatherable row, see the module doc), an
/// out-of-range feature index, or an empty route.
pub(super) fn gather_kquant_rows(
    index: &dyn GateIndex,
    layer: usize,
    feats: &[usize],
    hidden: usize,
) -> Option<GatheredRows> {
    let slices = index.interleaved_kquant_layer_data(layer)?;
    let gate_info = larql_vindex::quant::registry::lookup(slices[0].1)?;
    let up_info = larql_vindex::quant::registry::lookup(slices[1].1)?;
    let gbpr = gate_info.bytes_per_row(hidden)?;
    let ubpr = up_info.bytes_per_row(hidden)?;
    let gate_b = slices[0].0;
    let up_b = slices[1].0;
    // Down from the feature-major sidecar (gatherable rows).
    let (down_b, down_fmt, padded_width) = index.down_features_q4k_layer_data(layer)?;
    let down_info = larql_vindex::quant::registry::lookup(down_fmt)?;
    let dbpr = down_info.bytes_per_row(padded_width)?;
    let k = feats.len();
    if k == 0 {
        return None;
    }

    let mut gg = vec![0u8; k * gbpr];
    let mut gu = vec![0u8; k * ubpr];
    let mut gd = vec![0u8; k * dbpr];
    for (i, &feat) in feats.iter().enumerate() {
        let (gs, ge) = (feat * gbpr, feat * gbpr + gbpr);
        let (us, ue) = (feat * ubpr, feat * ubpr + ubpr);
        let (ds, de) = (feat * dbpr, feat * dbpr + dbpr);
        if ge > gate_b.len() || ue > up_b.len() || de > down_b.len() {
            return None; // out-of-range feature — bail to the safe path
        }
        gg[i * gbpr..(i + 1) * gbpr].copy_from_slice(&gate_b[gs..ge]);
        gu[i * ubpr..(i + 1) * ubpr].copy_from_slice(&up_b[us..ue]);
        gd[i * dbpr..(i + 1) * dbpr].copy_from_slice(&down_b[ds..de]);
    }

    Some(GatheredRows {
        gate: gg,
        up: gu,
        down: gd,
        gate_format: slices[0].1.to_string(),
        up_format: slices[1].1.to_string(),
        down_format: down_fmt.to_string(),
        gate_bytes_per_row: gbpr,
        up_bytes_per_row: ubpr,
        down_bytes_per_row: dbpr,
        k,
    })
}

/// Score gate/up and accumulate down over already-gathered rows — the
/// fused-kernel half of the gather path, shared by
/// [`WalkFfn::gather_q4k_accumulate`] (rows gathered THIS call) and
/// [`super::compact_dense::CompactDenseLayer`]'s forward (rows gathered
/// LONG before this call). Neither the kernel calls nor their cost
/// depend on which; only the caller's timing does.
pub(super) fn score_and_accumulate(
    rows: &GatheredRows,
    x_slice: &[f32],
    use_gelu: bool,
    hidden: usize,
    activation_floor: f32,
) -> Option<GatherAccumulate> {
    let gate_rd = larql_vindex::quant::registry::lookup(&rows.gate_format)?.row_dot?;
    let up_rd = larql_vindex::quant::registry::lookup(&rows.up_format)?.row_dot?;
    let down_sa = larql_vindex::quant::registry::lookup(&rows.down_format)?.row_scaled_add?;
    let (gbpr, ubpr, dbpr) = (
        rows.gate_bytes_per_row,
        rows.up_bytes_per_row,
        rows.down_bytes_per_row,
    );
    let k = rows.k;
    if k == 0 {
        return None;
    }

    // gate + up scores: fused row-dot over contiguous rows, parallel.
    let gate_s: Vec<f32> = (0..k)
        .into_par_iter()
        .map(|i| gate_rd(&rows.gate[i * gbpr..(i + 1) * gbpr], x_slice).unwrap_or(0.0))
        .collect();
    let up_s: Vec<f32> = (0..k)
        .into_par_iter()
        .map(|i| up_rd(&rows.up[i * ubpr..(i + 1) * ubpr], x_slice).unwrap_or(0.0))
        .collect();
    let acts: Vec<f32> = gate_s
        .iter()
        .zip(&up_s)
        .map(|(&g, &u)| {
            let ag = if use_gelu {
                crate::ffn::gelu_tanh(g)
            } else {
                g * crate::ffn::sigmoid(g)
            };
            ag * u
        })
        .collect();

    // down accumulate: fused scaled-add over contiguous rows, chunked.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = k.div_ceil(n_threads).max(1);
    let partials: Vec<Vec<f32>> = (0..k)
        .collect::<Vec<_>>()
        .par_chunks(chunk)
        .map(|ch| {
            let mut part = vec![0.0f32; hidden];
            for &i in ch {
                if acts[i].abs() > activation_floor {
                    let _ = down_sa(&rows.down[i * dbpr..(i + 1) * dbpr], acts[i], &mut part);
                }
            }
            part
        })
        .collect();
    let mut out = vec![0.0f32; hidden];
    for p in &partials {
        for (o, v) in out.iter_mut().zip(p) {
            *o += v;
        }
    }
    Some(GatherAccumulate {
        out,
        acts,
        gate_scores: gate_s,
        up_scores: up_s,
    })
}

impl<'a> WalkFfn<'a> {
    /// The known-pool route feature set for the gather fast path:
    /// cell-router pool if a router is attached, else the precomputed
    /// per-layer pool truncated to top-K. `None` when the position has
    /// no gate-free route (the caller then runs the scored paths).
    pub(super) fn gather_route_feats(
        &self,
        layer: usize,
        x_slice: &[f32],
        top_k: usize,
    ) -> Option<Vec<usize>> {
        if let Some(router) = self.config.cell_router.as_ref() {
            router.pool_for(layer, x_slice).map(|p| p.to_vec())
        } else if self.config.precomputed_routing {
            self.config.pool_per_layer.as_ref().and_then(|ppl| {
                ppl.get(layer).map(|p| {
                    let mut v = p.clone();
                    v.truncate(top_k);
                    v
                })
            })
        } else {
            None
        }
    }

    /// Gather-contiguous Q4K accumulate — faithful-K fast-path kernel
    /// (task #24; made production-correct by the feature-major down
    /// sidecar, task #25 — see the down paragraph below).
    ///
    /// The scattered per-feature loop pays ~4× per-row overhead at large K
    /// (cache-unfriendly gather + per-hit dispatch), so it loses to dense
    /// above ~20% density. This gathers the selected rows' up/down Q4K
    /// **bytes** into contiguous buffers and runs the *same* fused NEON
    /// kernels (`row_dot` / `row_scaled_add`) — no f32 materialisation.
    /// Contiguity recovers the cost win: at K=4096 (40%) the kernel runs
    /// ~1.4× faster than dense vs the scattered path's 0.80×
    /// (`examples/walk_ffn_gather_gemm.rs`).
    ///
    /// `up` (`slices[1]`) is feature-major Q4K (gatherable). **`down`** must
    /// come from the **feature-major down sidecar**
    /// (`down_features_kquant.bin` via `down_features_q4k_layer_data`) — the
    /// interleaved down is stored *transposed* `[hidden × intermediate]`, so a
    /// feature's down vector is a strided column there, not a gatherable row.
    /// Reading the transposed slab with row striding was the task-#24
    /// "not yet correct for production down" caveat; task #25 resolved it by
    /// requiring the sidecar here (`?` below) and validating against dense
    /// (`examples/walk_ffn_gather_gemm.rs`, |err|/‖ref‖ ≈ 6e-3 — the Q4K
    /// re-quantisation of the Q6_K down, not a layout error). Returns `None`
    /// (caller falls back to the correct scalar paths) when the sidecar is
    /// absent — pinned by `gather_q4k_accumulate_declines_without_down_sidecar`.
    ///
    /// Returns a [`GatherAccumulate`] aligned with `feats`. **Gate is
    /// recomputed** from gathered gate bytes (not taken from any prior
    /// scattered scoring) so the whole gate/up/down pass is contiguous —
    /// `feats` need only be the route's feature indices.
    pub(super) fn gather_q4k_accumulate(
        &self,
        layer: usize,
        feats: &[usize],
        x_slice: &[f32],
        use_gelu: bool,
        hidden: usize,
    ) -> Option<GatherAccumulate> {
        // Gather gate + up + down bytes for the route's rows into contiguous
        // buffers (sequential layout = cache-friendly fused kernel passes).
        // Paid on EVERY call — this is the cost BW-A/R4 measured; see
        // `compact_dense.rs` for the arm that pays it once instead.
        let rows = gather_kquant_rows(self.index, layer, feats, hidden)?;
        score_and_accumulate(
            &rows,
            x_slice,
            use_gelu,
            hidden,
            self.config.effective_activation_floor(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::observe::Observe;
    use crate::ffn::FfnActivations;
    use crate::test_utils::{
        attach_down_features_q4k_to_test_vindex, make_test_q4k_vindex, make_test_q4k_weights,
    };
    use crate::vindex::{CellRouter, WalkFfn, WalkFfnConfig};
    use ndarray::Array2;

    /// Densify a Record-mode observation for assertions.
    fn obs_dense(obs: Option<FfnActivations>) -> Array2<f32> {
        obs.expect("Record mode observes")
            .into_dense()
            .expect("walk observations materialise densely")
    }

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Safety property (task #25): `gather_q4k_accumulate` must **decline**
    /// (return None) when the feature-major down sidecar is absent — the
    /// interleaved down is transposed and not gatherable, so the caller falls
    /// back to the correct scalar paths. The test fixture ships no sidecar.
    /// (With-sidecar correctness is validated against dense on a real vindex
    /// in `examples/walk_ffn_gather_gemm.rs`.)
    #[test]
    fn gather_q4k_accumulate_declines_without_down_sidecar() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let x1 = x(1, hidden);
        let x_slice = x1.row(0).to_vec();
        assert!(!index.has_down_features_kquant());
        assert!(
            ffn.gather_q4k_accumulate(0, &[0, 1, 2, 3], &x_slice, false, hidden)
                .is_none(),
            "gather must decline without the feature-major down sidecar"
        );
    }

    /// Precomputed-pool walk over `pool` with K = pool.len() — same
    /// shape as the parallel-parity tests: the pool pins the exact
    /// feature set and keeps the full-K gemv rewrite off.
    fn walk_pool<'a>(
        weights: &'a larql_models::ModelWeights,
        index: &'a larql_vindex::VectorIndex,
        pool: Vec<usize>,
    ) -> WalkFfn<'a> {
        let k = pool.len();
        let cfg = WalkFfnConfig::sparse(weights.num_layers, k)
            .with_pool_per_layer(Arc::new(vec![pool; weights.num_layers]))
            .with_precomputed_routing(true);
        WalkFfn::from_config(weights, index, cfg).with_dispatch_trace()
    }

    fn rel_l2(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        let num: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        let den: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        num / den.max(f32::MIN_POSITIVE)
    }

    /// The gather-vs-serial parity test the 2026-07-30 review flagged as
    /// missing (roadmap item 20): on the SAME precomputed route, the
    /// gather-contiguous kernel (down from the feature-major Q4K
    /// sidecar) must reproduce the serial scalar walk (down via the
    /// dequantised interleaved cache).
    ///
    /// Gate and up are read from the same interleaved Q4K bytes on both
    /// paths, so activations agree tightly. The output differs by the
    /// down matrix being quantised **twice independently** (interleaved
    /// `[hidden, inter]` blocks vs sidecar transposed `[inter, hidden]`
    /// blocks — exactly what the production writer does, see
    /// `feature_major_down.rs::append_layer`). On this random-uniform
    /// fixture that measures ≈ 8e-2 rel L2 (a real model measured 6e-3,
    /// `walk_ffn_gather_gemm.rs` — correlated weights quantise far
    /// tighter than uniform noise). The bound 1.5e-1 keeps ~17× margin
    /// to a layout bug, which decorrelates the rows and shows up as
    /// O(1) (≈ √2) error.
    #[test]
    fn gather_q4k_matches_serial_walk_on_same_route() {
        let weights = make_test_q4k_weights();
        let mut with_sidecar = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut with_sidecar);
        assert!(with_sidecar.has_down_features_kquant());
        let without_sidecar = make_test_q4k_vindex(&weights);

        let hidden = weights.hidden_size;
        let input = x(1, hidden);
        // Route = every feature; intermediate (256) equals
        // GATHER_MIN_FEATURES so the whole layer is the smallest
        // gather-eligible route on this fixture.
        let pool: Vec<usize> = (0..weights.intermediate_size).collect();

        let gather_ffn = walk_pool(&weights, &with_sidecar, pool.clone());
        let (out_g, obs_g) = gather_ffn
            .walk_ffn_sparse(0, &input, Observe::Record)
            .expect("gather-eligible walk runs");
        let act_g = obs_dense(obs_g);
        assert!(
            gather_ffn
                .take_dispatch_trace()
                .iter()
                .any(|e| e.path == "sparse:gather_q4k"),
            "sidecar + precomputed route of >= GATHER_MIN_FEATURES must gather"
        );

        let serial_ffn = walk_pool(&weights, &without_sidecar, pool);
        let (out_s, obs_s) = serial_ffn
            .walk_ffn_sparse(0, &input, Observe::Record)
            .expect("serial walk runs");
        let act_s = obs_dense(obs_s);
        assert!(
            serial_ffn
                .take_dispatch_trace()
                .iter()
                .all(|e| e.path != "sparse:gather_q4k"),
            "no sidecar → the same route must run the scalar paths"
        );

        let act_err = rel_l2(&act_g, &act_s);
        assert!(
            act_err < 1e-3,
            "gate/up come from the same Q4K bytes on both paths — \
             activation rel L2 = {act_err} (expected < 1e-3)"
        );
        let out_err = rel_l2(&out_g, &out_s);
        assert!(
            out_err < 1.5e-1,
            "gather vs serial output rel L2 = {out_err} (expected < 1.5e-1: \
             independent Q4K quantisation of down in two orientations, \
             ≈ 8e-2 on this fixture; a layout bug is O(1))"
        );
        assert!(
            out_s.iter().any(|v| v.abs() > 0.0),
            "serial reference must be non-zero for the parity to mean anything"
        );
    }

    /// `gather_route_feats` route selection: a cell router's pool is
    /// used verbatim (never truncated — the cell IS the route), a
    /// precomputed per-layer pool is truncated to top-K, and with
    /// neither configured there is no gate-free route.
    #[test]
    fn gather_route_feats_picks_router_then_pool_then_none() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let nl = weights.num_layers;
        let x_slice = vec![0.01_f32; hidden];

        // Cell router: pool comes back verbatim even though top_k = 2.
        let router = CellRouter {
            centroids: vec![vec![0.0f32; hidden]; nl], // 1 cell/layer at origin
            n_cells: vec![1; nl],
            pools: vec![vec![vec![4usize, 5, 6, 7]]; nl],
            hidden,
        };
        let cfg = WalkFfnConfig::sparse(nl, 2).with_cell_router(Arc::new(router));
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        assert_eq!(
            ffn.gather_route_feats(0, &x_slice, 2),
            Some(vec![4, 5, 6, 7]),
            "router pool is the route — top_k must not truncate it"
        );

        // Precomputed pool: truncated to top-K, in pool order.
        let cfg = WalkFfnConfig::sparse(nl, 2)
            .with_pool_per_layer(Arc::new(vec![vec![9usize, 8, 7, 6]; nl]))
            .with_precomputed_routing(true);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        assert_eq!(ffn.gather_route_feats(0, &x_slice, 2), Some(vec![9, 8]));

        // Pool set but precomputed_routing off → scored path, no
        // gate-free route.
        let cfg =
            WalkFfnConfig::sparse(nl, 2).with_pool_per_layer(Arc::new(vec![vec![1usize, 2]; nl]));
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        assert_eq!(ffn.gather_route_feats(0, &x_slice, 2), None);

        // Neither router nor pool → None.
        let cfg = WalkFfnConfig::sparse(nl, 2);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        assert_eq!(ffn.gather_route_feats(0, &x_slice, 2), None);
    }

    /// Safety rails of `gather_q4k_accumulate` WITH the sidecar loaded:
    /// an out-of-range feature index bails to `None` (the caller falls
    /// back to the scalar paths, which bounds-check per feature), and an
    /// empty route declines outright.
    #[test]
    fn gather_q4k_accumulate_bails_on_bad_route_with_sidecar() {
        let weights = make_test_q4k_weights();
        let mut index = make_test_q4k_vindex(&weights);
        attach_down_features_q4k_to_test_vindex(&weights, &mut index);
        let hidden = weights.hidden_size;
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let x1 = x(1, hidden);
        let x_slice = x1.row(0).to_vec();

        // A feature past the layer's rows overruns every gathered slab.
        let out_of_range = weights.intermediate_size * 10;
        assert!(
            ffn.gather_q4k_accumulate(0, &[0, out_of_range], &x_slice, false, hidden)
                .is_none(),
            "out-of-range feature must bail to the safe path"
        );
        assert!(
            ffn.gather_q4k_accumulate(0, &[], &x_slice, false, hidden)
                .is_none(),
            "empty route must decline"
        );
    }
}
