//! Pure logic for `larql dec-bench window-union` — the BW11-1 same-sequence
//! consecutive-position expert-union ceiling (docs/dec-funnel.md R2/R6/R10
//! conventions apply throughout this file; read those three before quoting
//! any number this module produces).
//!
//! ## What this measures, and what it does not
//!
//! Reads the `LARQL_MOE_ROUTE_TRACE` JSONL sink
//! (`larql_compute::ffn::expert_weight::trace`, one `{"layer":L,"seq":S,
//! "experts":[[...]]}` line per routed forward call — see that module's
//! header for the format) from ONE continuous decode run, and asks: across
//! `K` CONSECUTIVE positions of one layer (a contiguous `seq` run), what
//! fraction of naive per-row expert bytes survives as a logical union?
//!
//! This is the SAME arithmetic as [`super::replay::routed_weight_bytes_per_token`]
//! (a cell's union vs naive expert-id count) applied along a DIFFERENT axis.
//! DEC-0's `routed_denominators_for_point` unions ACROSS independent prompts
//! at a fixed step (a serving-batch question); this unions ACROSS consecutive
//! steps of ONE prompt (a speculative-verification-batch question). **R2
//! forbids treating these as the same ratio** — they are different objects
//! even before the activation fraction differs. GPT-OSS's own activation
//! fraction (`top_k / num_experts`, required flags below — never hardcoded,
//! per the workspace's no-hardcoded-types rule) must be quoted with every
//! number this module prints, and DEC-0's Gemma-4 figure (13.9% union at
//! B64, 6.25% activation) must never be substituted for it.
//!
//! **R6 caveat, stated once here rather than at every call site:** this is a
//! LOGICAL union ceiling, not bytes read. No kernel in this codebase groups
//! multiple TOKENS through one expert's weight read — every expert-GEMV
//! kernel (`shaders::mxfp4_grouped_experts` and siblings) is K=1 token per
//! dispatch. A favourable ratio here licenses *building* a token-grouped
//! expert kernel; it is not itself a bandwidth claim.
//!
//! **R10 caveat:** overlapping windows at stride 1 are NOT independent
//! samples — adjacent windows share `K-1` rows, so the reported spread
//! (p10/p90 across windows) describes shape, not a confidence interval.

use std::collections::BTreeMap;
use std::io::BufRead;

use super::replay::routed_weight_bytes_per_token;

// ── Trace parsing ────────────────────────────────────────────────────────────

/// One decoded JSONL line: a layer's routing at one forward-call position.
/// `seq` is that layer's call index (0-based, per
/// `larql_compute::ffn::expert_weight::trace::TraceWriter::next_seq`) —
/// consecutive `seq` values are consecutive positions of ONE continuous
/// sequence (prefill positions followed by decode steps, in call order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRecord {
    pub layer: usize,
    pub seq: usize,
    /// One forward call's selected expert ids. The trace format allows a
    /// call to cover multiple tokens (`"experts":[[...],[...]]`, the
    /// reference-tier batched form); the decode-path writer
    /// (`moe_route_observe::observe`) always emits exactly one, but this
    /// parser accepts either and the caller decides whether multi-token
    /// calls are in scope for windowing (they are refused below — a
    /// windowed `seq` axis assumes one position per call).
    pub experts: Vec<Vec<u32>>,
}

/// Parse the `LARQL_MOE_ROUTE_TRACE` JSONL format. Hand-rolled to match
/// `TraceWriter::record`, which is itself hand-rolled for the same reason
/// (every field is a non-negative integer — nothing to escape, no decoder
/// dependency worth taking for a diagnostic). A malformed line is a loud
/// error, never a skip: a silently-dropped line would understate a
/// window's naive count without saying so.
pub fn parse_trace<R: BufRead>(r: R) -> Result<Vec<TraceRecord>, String> {
    let mut out = Vec::new();
    for (line_no, line) in r.lines().enumerate() {
        let line = line.map_err(|e| format!("line {}: read error: {e}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(parse_trace_line(&line).map_err(|e| format!("line {}: {e}", line_no + 1))?);
    }
    Ok(out)
}

fn parse_trace_line(line: &str) -> Result<TraceRecord, String> {
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let layer = v
        .get("layer")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| "missing integer field \"layer\"".to_string())?;
    let seq = v
        .get("seq")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| "missing integer field \"seq\"".to_string())?;
    let experts_v = v
        .get("experts")
        .and_then(|x| x.as_array())
        .ok_or_else(|| "missing array field \"experts\"".to_string())?;
    let mut experts = Vec::with_capacity(experts_v.len());
    for (i, row) in experts_v.iter().enumerate() {
        let row = row
            .as_array()
            .ok_or_else(|| format!("experts[{i}] is not an array"))?;
        let mut ids = Vec::with_capacity(row.len());
        for (j, id) in row.iter().enumerate() {
            let id = id
                .as_u64()
                .ok_or_else(|| format!("experts[{i}][{j}] is not a non-negative integer"))?;
            ids.push(id as u32);
        }
        experts.push(ids);
    }
    Ok(TraceRecord {
        layer: layer as usize,
        seq: seq as usize,
        experts,
    })
}

/// Group records by layer, ordered by `seq`, as one expert-id row per
/// position. Refuses (rather than silently windowing across a hole) when:
/// a layer's records are not exactly `0..n` with no gaps or repeats — a
/// killed/concatenated run must not be windowed as if it were continuous;
/// or any record covers more than one position (`experts.len() != 1`) —
/// the windowed axis below assumes one position per `seq`, and a batched
/// reference-tier call is a different measurement (DEC-0's cross-row
/// union, not this module's cross-position union).
fn rows_by_layer(records: &[TraceRecord]) -> Result<BTreeMap<usize, Vec<Vec<u32>>>, String> {
    let mut by_layer: BTreeMap<usize, BTreeMap<usize, &Vec<Vec<u32>>>> = BTreeMap::new();
    for r in records {
        if r.experts.len() != 1 {
            return Err(format!(
                "layer {} seq {}: {} position(s) in one record — window-union needs \
                 exactly one position per record (the decode-path trace writer, \
                 moe_route_observe::observe, always emits exactly one; a reference-tier \
                 batched capture is a different measurement, see routed_weight_bytes_per_token)",
                r.layer,
                r.seq,
                r.experts.len()
            ));
        }
        if by_layer
            .entry(r.layer)
            .or_default()
            .insert(r.seq, &r.experts)
            .is_some()
        {
            return Err(format!(
                "layer {} seq {} appears twice in the trace — refusing rather than \
                 guessing which record is authoritative",
                r.layer, r.seq
            ));
        }
    }
    let mut out = BTreeMap::new();
    for (layer, seqs) in by_layer {
        let n = seqs.len();
        let mut rows = Vec::with_capacity(n);
        for expected in 0..n {
            let row = seqs.get(&expected).ok_or_else(|| {
                format!(
                    "layer {layer}: seq {expected} missing — the trace has a gap \
                     (killed run?), so consecutive positions cannot be assumed \
                     contiguous past it"
                )
            })?;
            rows.push(row[0].clone());
        }
        out.insert(layer, rows);
    }
    Ok(out)
}

// ── Window-union statistics ──────────────────────────────────────────────────

/// One layer's pooled window-union ceiling at a given `K`, aggregated over
/// every valid `K`-position sliding window (stride 1) in that layer's
/// trace. Pooled the same way DEC-0's cross-prompt union is pooled — one
/// call to [`routed_weight_bytes_per_token`] over all windows as cells —
/// so the *form* is directly comparable to DEC-0's number even though R2
/// forbids comparing the *value* (different activation fraction, different
/// axis).
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct LayerWindowUnion {
    pub layer: usize,
    pub k: usize,
    pub n_windows: usize,
    pub weight_bytes_tok_naive: f64,
    pub weight_bytes_tok_union: f64,
    /// `weight_bytes_tok_union / weight_bytes_tok_naive`. 1.0 at K=1 by
    /// construction (a single row cannot share with itself); values below
    /// 1.0 quantify realised sharing across consecutive positions.
    pub union_frac: f64,
}

/// Compute [`LayerWindowUnion`] for every layer present in `records`, at
/// window width `k`. `per_expert_bytes` must come from the model/server,
/// never a literal (no-hardcoded-types).
///
/// Errs if `k == 0`, if any layer has fewer than `k` positions (no window
/// fits), or on the trace-shape problems [`rows_by_layer`] refuses.
pub fn window_union_by_layer(
    records: &[TraceRecord],
    k: usize,
    per_expert_bytes: f64,
) -> Result<Vec<LayerWindowUnion>, String> {
    if k == 0 {
        return Err("k must be >= 1".to_string());
    }
    let by_layer = rows_by_layer(records)?;
    let mut out = Vec::with_capacity(by_layer.len());
    for (layer, rows) in by_layer {
        if rows.len() < k {
            return Err(format!(
                "layer {layer}: only {} position(s) captured, need >= {k} for a K={k} window \
                 (increase --steps on the capturing run)",
                rows.len()
            ));
        }
        let n_windows = rows.len() - k + 1;
        let cells: Vec<Vec<Vec<u32>>> = (0..n_windows)
            .map(|start| rows[start..start + k].to_vec())
            .collect();
        let d = routed_weight_bytes_per_token(&cells, per_expert_bytes, n_windows, k);
        let union_frac = if d.weight_bytes_tok_naive > 0.0 {
            d.weight_bytes_tok_union / d.weight_bytes_tok_naive
        } else {
            1.0
        };
        out.push(LayerWindowUnion {
            layer,
            k,
            n_windows,
            weight_bytes_tok_naive: d.weight_bytes_tok_naive,
            weight_bytes_tok_union: d.weight_bytes_tok_union,
            union_frac,
        });
    }
    Ok(out)
}

// ── Shuffled marginal-preserving null (R9) ───────────────────────────────────
//
// A skewed per-expert popularity distribution alone — with ZERO genuine
// same-sequence structure — pushes union_frac below the naive
// uniform-routing prediction, because popular experts collide across
// unrelated rows too (this is exactly the DEC-0 R9 finding: the uniform law
// "overstates per-session support 3.4x... understates top-C coverage
// ~6.8x" on real, skewed routing). So a favourable `window_union_by_layer`
// number is NOT yet evidence of genuine same-sequence correlation; it could
// be entirely explained by marginal skew. The control: pool a layer's rows
// (breaking sequence order, optionally across multiple files/prompts),
// shuffle them, and re-measure the SAME statistic on disjoint K-chunks of
// the shuffled order. This preserves the exact empirical per-expert
// marginal AND each row's real top-k joint structure — it destroys only
// sequence adjacency. If the shuffled number matches the windowed number,
// the "same-sequence" framing is falsified; if the shuffled number sits
// close to the naive uniform-routing prediction while the windowed number
// sits below it, that is the actual signature of genuine temporal
// structure.

/// Deterministic splitmix64 (Steele/Vigna) — reproducible, not
/// wall-clock-seeded, so results are re-derivable. Hand-rolled rather than
/// a `rand` dependency: nothing here needs cryptographic quality, and the
/// workspace avoids new dependencies for diagnostics (mirrors
/// `TraceWriter`'s own hand-rolled JSON, same file).
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Fisher-Yates shuffle of `items`, in place, using `splitmix64_next`.
fn shuffle_in_place<T>(items: &mut [T], seed: &mut u64) {
    for i in (1..items.len()).rev() {
        let j = (splitmix64_next(seed) % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// Every captured row for `layer`, across however many trace files the
/// caller passes, order-independent (no seq-contiguity requirement — the
/// caller is about to shuffle it anyway). Same per-record shape refusal as
/// [`rows_by_layer`] (exactly one position per record).
fn all_rows_for_layer(files: &[&[TraceRecord]], layer: usize) -> Result<Vec<Vec<u32>>, String> {
    let mut rows = Vec::new();
    for file in files {
        for r in file.iter().filter(|r| r.layer == layer) {
            if r.experts.len() != 1 {
                return Err(format!(
                    "layer {} seq {}: {} position(s) in one record — the shuffled control \
                     needs exactly one position per record, same as window_union_by_layer",
                    r.layer,
                    r.seq,
                    r.experts.len()
                ));
            }
            rows.push(r.experts[0].clone());
        }
    }
    Ok(rows)
}

/// Mean `union_frac` over `trials` shuffles of `rows`, each measured on
/// disjoint (non-overlapping) `K`-chunks of that trial's shuffled order —
/// disjoint rather than sliding, so trials don't need R10's overlapping-
/// window caveat and average cleanly. Drops a trailing partial chunk.
/// Errs under the same conditions as [`window_union_by_layer`] (`k == 0`,
/// or fewer than `k` rows to chunk at all).
pub fn shuffled_control_union_frac(
    rows: &[Vec<u32>],
    k: usize,
    per_expert_bytes: f64,
    trials: usize,
    seed: u64,
) -> Result<f64, String> {
    if k == 0 {
        return Err("k must be >= 1".to_string());
    }
    if rows.len() < k {
        return Err(format!(
            "only {} row(s) available, need >= {k} for a K={k} chunk",
            rows.len()
        ));
    }
    if trials == 0 {
        return Err("trials must be >= 1".to_string());
    }
    let mut state = seed;
    let mut order: Vec<Vec<u32>> = rows.to_vec();
    let mut frac_sum = 0.0;
    for _ in 0..trials {
        shuffle_in_place(&mut order, &mut state);
        let n_chunks = order.len() / k;
        let cells: Vec<Vec<Vec<u32>>> = order[..n_chunks * k]
            .chunks(k)
            .map(|c| c.to_vec())
            .collect();
        let d = routed_weight_bytes_per_token(&cells, per_expert_bytes, n_chunks, k);
        frac_sum += if d.weight_bytes_tok_naive > 0.0 {
            d.weight_bytes_tok_union / d.weight_bytes_tok_naive
        } else {
            1.0
        };
    }
    Ok(frac_sum / trials as f64)
}

/// [`shuffled_control_union_frac`] for every layer present across `files`,
/// shaped as [`LayerWindowUnion`] so it slots into [`summarize_spread`]
/// exactly like the real windowed result (`n_windows` here is the trial
/// count, not a window count — the field is reused for the table renderer,
/// not a claim of equivalence).
pub fn shuffled_control_by_layer(
    files: &[&[TraceRecord]],
    k: usize,
    per_expert_bytes: f64,
    trials: usize,
    seed: u64,
) -> Result<Vec<LayerWindowUnion>, String> {
    let mut layers: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for file in files {
        layers.extend(file.iter().map(|r| r.layer));
    }
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers {
        let rows = all_rows_for_layer(files, layer)?;
        // Layer-derived seed so every layer gets an independent shuffle
        // stream rather than all layers replaying the same permutation.
        let layer_seed = seed ^ (layer as u64).wrapping_mul(0x2545F4914F6CDD1D);
        let union_frac =
            shuffled_control_union_frac(&rows, k, per_expert_bytes, trials, layer_seed)?;
        out.push(LayerWindowUnion {
            layer,
            k,
            n_windows: trials,
            weight_bytes_tok_naive: f64::NAN,
            weight_bytes_tok_union: f64::NAN,
            union_frac,
        });
    }
    Ok(out)
}

/// Cross-layer (and, when the caller concatenates traces from several
/// prompts, cross-prompt) spread of [`LayerWindowUnion::union_frac`] at one
/// `K`. Percentiles are over the layer population, NOT a claim about
/// statistical independence between overlapping windows within a layer —
/// see the module header's R10 note.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct WindowUnionSpread {
    pub k: usize,
    pub n_layers: usize,
    pub union_frac_mean: f64,
    pub union_frac_median: f64,
    pub union_frac_p10: f64,
    pub union_frac_p90: f64,
    /// `1 / union_frac_mean` — the DEC-0 "amortisation" framing
    /// (`~7.2×` at Gemma-4's B64), reported in the same units so the two
    /// are legible side by side even though R2 forbids equating them.
    pub amortisation_mean: f64,
}

/// Reduce a (possibly multi-prompt-concatenated) set of per-layer results
/// at one `K` into [`WindowUnionSpread`]. Errs on an empty slice — an empty
/// spread would silently render as zeros rather than as "no data."
pub fn summarize_spread(layers: &[LayerWindowUnion]) -> Result<WindowUnionSpread, String> {
    let Some(k) = layers.first().map(|l| l.k) else {
        return Err("no layer results to summarize".to_string());
    };
    if layers.iter().any(|l| l.k != k) {
        return Err("summarize_spread got mixed K values — summarize one K at a time".to_string());
    }
    let mut fracs: Vec<f64> = layers.iter().map(|l| l.union_frac).collect();
    fracs.sort_by(|a, b| a.partial_cmp(b).expect("union_frac is never NaN"));
    let n = fracs.len();
    let mean = fracs.iter().sum::<f64>() / n as f64;
    Ok(WindowUnionSpread {
        k,
        n_layers: n,
        union_frac_mean: mean,
        union_frac_median: percentile(&fracs, 0.50),
        union_frac_p10: percentile(&fracs, 0.10),
        union_frac_p90: percentile(&fracs, 0.90),
        amortisation_mean: if mean > 0.0 {
            1.0 / mean
        } else {
            f64::INFINITY
        },
    })
}

/// Nearest-rank percentile over an already-sorted slice, `p` in `[0, 1]`.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Human table ───────────────────────────────────────────────────────────────

/// Render one K's spread as a table row, matching `dec-bench drift`'s
/// convention (values first, verdict/interpretation left to the caller —
/// this module reports an oracle ceiling, not a pass/fail gate).
pub fn format_spread_row(s: &WindowUnionSpread, activation_frac: f64) -> String {
    format!(
        "K={:<3} layers={:<3} activation={:>6.2}%  union_frac mean={:.4} median={:.4} \
         p10={:.4} p90={:.4}  amortisation={:.2}x",
        s.k,
        s.n_layers,
        activation_frac * 100.0,
        s.union_frac_mean,
        s.union_frac_median,
        s.union_frac_p10,
        s.union_frac_p90,
        s.amortisation_mean,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn rec(layer: usize, seq: usize, experts: &[u32]) -> String {
        let ids = experts
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!("{{\"layer\":{layer},\"seq\":{seq},\"experts\":[[{ids}]]}}")
    }

    #[test]
    fn shuffle_in_place_is_a_permutation_and_is_seed_deterministic() {
        let original: Vec<u32> = (0..20).collect();

        let mut a = original.clone();
        let mut seed_a = 42u64;
        shuffle_in_place(&mut a, &mut seed_a);

        let mut b = original.clone();
        let mut seed_b = 42u64;
        shuffle_in_place(&mut b, &mut seed_b);
        assert_eq!(a, b, "same seed must reproduce the same permutation");

        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(
            sorted, original,
            "shuffling must not lose or duplicate items"
        );

        let mut c = original.clone();
        let mut seed_c = 43u64;
        shuffle_in_place(&mut c, &mut seed_c);
        assert_ne!(
            a, c,
            "a different seed should (overwhelmingly likely) differ"
        );
    }

    #[test]
    fn shuffled_control_on_all_identical_rows_is_exactly_the_same_frac_as_windowed() {
        // Every row identical: no permutation can change the union — the
        // control and the real windowed measurement must agree exactly,
        // regardless of trial count or seed. This is the equivalence-point
        // sanity check for the control path.
        let rows: Vec<Vec<u32>> = (0..10).map(|_| vec![1, 2, 3, 4]).collect();
        let frac = shuffled_control_union_frac(&rows, 2, 10.0, 50, 7).unwrap();
        // Every chunk pairs two identical 4-expert rows: union 4 / naive 8 = 0.5.
        assert!((frac - 0.5).abs() < 1e-12, "expected 0.5, got {frac}");
    }

    #[test]
    fn shuffled_control_on_all_disjoint_rows_is_exactly_one_regardless_of_order() {
        // Every row pulls from a disjoint block of expert ids: no
        // permutation can create overlap, so union_frac must stay 1.0
        // exactly under any shuffle.
        let rows: Vec<Vec<u32>> = (0..8).map(|i| vec![i * 10, i * 10 + 1]).collect();
        let frac = shuffled_control_union_frac(&rows, 2, 10.0, 50, 99).unwrap();
        assert_eq!(frac, 1.0);
    }

    #[test]
    fn shuffled_control_errs_on_k_zero_or_too_few_rows_or_zero_trials() {
        let rows = vec![vec![1u32], vec![2u32]];
        assert!(shuffled_control_union_frac(&rows, 0, 10.0, 5, 1).is_err());
        assert!(shuffled_control_union_frac(&rows, 5, 10.0, 5, 1).is_err());
        assert!(shuffled_control_union_frac(&rows, 1, 10.0, 0, 1).is_err());
    }

    #[test]
    fn all_rows_for_layer_pools_across_files_and_filters_by_layer() {
        let file_a = vec![
            TraceRecord {
                layer: 0,
                seq: 0,
                experts: vec![vec![1, 2]],
            },
            TraceRecord {
                layer: 1,
                seq: 0,
                experts: vec![vec![9, 9]],
            },
        ];
        let file_b = vec![TraceRecord {
            layer: 0,
            seq: 0,
            experts: vec![vec![3, 4]],
        }];
        let rows = all_rows_for_layer(&[&file_a, &file_b], 0).unwrap();
        assert_eq!(
            rows,
            vec![vec![1, 2], vec![3, 4]],
            "layer 1 must not leak in"
        );
    }

    #[test]
    fn all_rows_for_layer_rejects_a_multi_position_record() {
        let file = vec![TraceRecord {
            layer: 0,
            seq: 0,
            experts: vec![vec![1], vec![2]],
        }];
        assert!(all_rows_for_layer(&[&file], 0).is_err());
    }

    #[test]
    fn shuffled_control_by_layer_covers_every_layer_seen_across_files() {
        let file_a = vec![
            TraceRecord {
                layer: 0,
                seq: 0,
                experts: vec![vec![1, 2, 3, 4]],
            },
            TraceRecord {
                layer: 0,
                seq: 1,
                experts: vec![vec![1, 2, 3, 4]],
            },
        ];
        let file_b = vec![
            TraceRecord {
                layer: 1,
                seq: 0,
                experts: vec![vec![5, 6]],
            },
            TraceRecord {
                layer: 1,
                seq: 1,
                experts: vec![vec![5, 6]],
            },
        ];
        let out = shuffled_control_by_layer(&[&file_a, &file_b], 2, 10.0, 20, 3).unwrap();
        let layers: Vec<usize> = out.iter().map(|l| l.layer).collect();
        assert_eq!(layers, vec![0, 1]);
        assert!(out.iter().all(|l| (l.union_frac - 0.5).abs() < 1e-12));
    }

    #[test]
    fn parses_the_writer_format_and_round_trips() {
        let text = format!(
            "{}\n{}\n\n{}\n",
            rec(0, 0, &[5, 9, 18, 6]),
            rec(0, 1, &[28, 29, 6, 11]),
            rec(1, 0, &[2, 3])
        );
        let recs = parse_trace(Cursor::new(text)).unwrap();
        assert_eq!(recs.len(), 3, "blank lines must be skipped, not counted");
        assert_eq!(
            recs[0],
            TraceRecord {
                layer: 0,
                seq: 0,
                experts: vec![vec![5, 9, 18, 6]]
            }
        );
    }

    #[test]
    fn malformed_line_is_a_loud_error_not_a_skip() {
        let err = parse_trace(Cursor::new("not json\n")).unwrap_err();
        assert!(
            err.contains("line 1"),
            "must name the offending line: {err}"
        );
    }

    #[test]
    fn rejects_a_gap_in_seq_rather_than_windowing_across_it() {
        let text = format!("{}\n{}\n", rec(0, 0, &[1, 2]), rec(0, 2, &[3, 4]));
        let recs = parse_trace(Cursor::new(text)).unwrap();
        let err = window_union_by_layer(&recs, 1, 100.0).unwrap_err();
        assert!(err.contains("seq 1"), "must name the missing seq: {err}");
    }

    #[test]
    fn rejects_a_duplicate_seq() {
        let text = format!("{}\n{}\n", rec(0, 0, &[1, 2]), rec(0, 0, &[3, 4]));
        let recs = parse_trace(Cursor::new(text)).unwrap();
        let err = window_union_by_layer(&recs, 1, 100.0).unwrap_err();
        assert!(err.contains("twice"), "must name the duplication: {err}");
    }

    #[test]
    fn rejects_a_multi_position_record() {
        let line = "{\"layer\":0,\"seq\":0,\"experts\":[[1,2],[3,4]]}\n";
        let recs = parse_trace(Cursor::new(line)).unwrap();
        let err = window_union_by_layer(&recs, 1, 100.0).unwrap_err();
        assert!(
            err.contains("position(s)"),
            "must name the shape problem: {err}"
        );
    }

    #[test]
    fn k_equals_one_is_always_union_frac_one() {
        // K=1: a single row cannot share with itself — naive == union at
        // every window, by construction, regardless of content. This is
        // the equivalence-point sanity check; k_two_disjoint_vs_shared
        // below is the disagreement-point oracle (R8) that actually
        // exercises the union arithmetic.
        let text = format!(
            "{}\n{}\n{}\n",
            rec(0, 0, &[1, 2, 3]),
            rec(0, 1, &[4, 5, 6]),
            rec(0, 2, &[7, 8, 9])
        );
        let recs = parse_trace(Cursor::new(text)).unwrap();
        let layers = window_union_by_layer(&recs, 1, 10.0).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].n_windows, 3);
        assert_eq!(layers[0].union_frac, 1.0);
    }

    #[test]
    fn k_two_disjoint_vs_shared_moves_the_ratio() {
        // Disjoint neighbours: union == naive (no sharing possible).
        let disjoint = format!("{}\n{}\n", rec(0, 0, &[0, 1]), rec(0, 1, &[2, 3]));
        let recs = parse_trace(Cursor::new(disjoint)).unwrap();
        let layers = window_union_by_layer(&recs, 2, 10.0).unwrap();
        assert_eq!(layers[0].union_frac, 1.0);

        // Identical neighbours: union collapses to one row's cost.
        let shared = format!("{}\n{}\n", rec(0, 0, &[0, 1]), rec(0, 1, &[0, 1]));
        let recs = parse_trace(Cursor::new(shared)).unwrap();
        let layers = window_union_by_layer(&recs, 2, 10.0).unwrap();
        assert_eq!(
            layers[0].union_frac, 0.5,
            "two identical rows halve the ratio"
        );

        // Partial overlap: naive 4, union 3.
        let partial = format!("{}\n{}\n", rec(0, 0, &[1, 2]), rec(0, 1, &[2, 3]));
        let recs = parse_trace(Cursor::new(partial)).unwrap();
        let layers = window_union_by_layer(&recs, 2, 10.0).unwrap();
        assert_eq!(layers[0].union_frac, 0.75);
    }

    #[test]
    fn sliding_windows_pool_across_the_whole_layer() {
        // 4 positions, K=2 -> 3 overlapping windows: (0,1) (1,2) (2,3).
        // naive = 2+2+2 = 6 experts total; unions = |{0,1}|=2, |{1,2}|=2,
        // |{2,3}|=2 -> union total 6 -> union_frac 1.0 (all disjoint pairs).
        let text = format!(
            "{}\n{}\n{}\n{}\n",
            rec(0, 0, &[0]),
            rec(0, 1, &[1]),
            rec(0, 2, &[2]),
            rec(0, 3, &[3])
        );
        let recs = parse_trace(Cursor::new(text)).unwrap();
        let layers = window_union_by_layer(&recs, 2, 100.0).unwrap();
        assert_eq!(layers[0].n_windows, 3);
        assert_eq!(layers[0].union_frac, 1.0);
    }

    #[test]
    fn errs_when_a_layer_is_shorter_than_k() {
        let text = rec(0, 0, &[1, 2]);
        let recs = parse_trace(Cursor::new(format!("{text}\n"))).unwrap();
        let err = window_union_by_layer(&recs, 4, 10.0).unwrap_err();
        assert!(err.contains("--steps"), "must point at the fix: {err}");
    }

    #[test]
    fn errs_on_k_zero() {
        let recs = parse_trace(Cursor::new(format!("{}\n", rec(0, 0, &[1])))).unwrap();
        let err = window_union_by_layer(&recs, 0, 10.0).unwrap_err();
        assert!(err.contains("k must be"));
    }

    #[test]
    fn summarize_spread_matches_hand_computed_percentiles() {
        let layers = vec![
            LayerWindowUnion {
                layer: 0,
                k: 4,
                n_windows: 1,
                weight_bytes_tok_naive: 100.0,
                weight_bytes_tok_union: 20.0,
                union_frac: 0.20,
            },
            LayerWindowUnion {
                layer: 1,
                k: 4,
                n_windows: 1,
                weight_bytes_tok_naive: 100.0,
                weight_bytes_tok_union: 40.0,
                union_frac: 0.40,
            },
            LayerWindowUnion {
                layer: 2,
                k: 4,
                n_windows: 1,
                weight_bytes_tok_naive: 100.0,
                weight_bytes_tok_union: 60.0,
                union_frac: 0.60,
            },
        ];
        let s = summarize_spread(&layers).unwrap();
        assert_eq!(s.n_layers, 3);
        assert!((s.union_frac_mean - 0.40).abs() < 1e-12);
        assert_eq!(s.union_frac_median, 0.40);
        assert_eq!(
            s.union_frac_p10, 0.20,
            "nearest-rank nudges low p toward the min"
        );
        assert_eq!(s.union_frac_p90, 0.60);
        assert!((s.amortisation_mean - 2.5).abs() < 1e-12);
    }

    #[test]
    fn summarize_spread_errs_on_empty_input() {
        assert!(summarize_spread(&[]).is_err());
    }

    #[test]
    fn summarize_spread_errs_on_mixed_k() {
        let a = LayerWindowUnion {
            layer: 0,
            k: 2,
            n_windows: 1,
            weight_bytes_tok_naive: 1.0,
            weight_bytes_tok_union: 1.0,
            union_frac: 1.0,
        };
        let mut b = a;
        b.k = 4;
        let err = summarize_spread(&[a, b]).unwrap_err();
        assert!(err.contains("mixed K"));
    }

    #[test]
    fn format_spread_row_names_every_field() {
        let s = WindowUnionSpread {
            k: 4,
            n_layers: 24,
            union_frac_mean: 0.5,
            union_frac_median: 0.5,
            union_frac_p10: 0.4,
            union_frac_p90: 0.6,
            amortisation_mean: 2.0,
        };
        let row = format_spread_row(&s, 0.125);
        assert!(row.contains("K=4"));
        assert!(row.contains("layers=24"));
        assert!(row.contains("12.50%"));
        assert!(row.contains("2.00x"));
    }
}
