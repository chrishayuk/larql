//! GLM-3 rung: does a compiled/selected MoE routing decision survive real
//! autoregressive decode?
//!
//! BW-B (`docs/diagnoses/bwb-compact-dense-oracle.md`, uncommitted worktree
//! `bw10-movement-ledger`) measured a compiled compact-dense MoE
//! representation beating sparse-gather and dense on a SYNTHETIC 32-position
//! smooth-drift trajectory used as a stand-in for "how much the selected-
//! expert mask changes over a decode run" (mean strict run length 1.06
//! positions, disclosed as an unvalidated proxy — no real `generate()` trace
//! was available at the time). This harness closes that gap: it drives a
//! real vindex through the actual production greedy-decode path
//! (`larql_inference::layer_graph::generate::generate`, the same function
//! `larql run --metal` calls) across several distinct prompts, captures the
//! selected top-k expert SET at every MoE layer for every decode token via
//! the already-committed served-tier route boundary
//! (`larql_compute::moe_route_observe`, `LARQL_MOE_ROUTE_TRACE`), and
//! computes the real run-length distribution BW-B never had.
//!
//! Reused, not reinvented (BW-C precedent,
//! `docs/diagnoses/bwc-expert-skip-oracle.md`):
//! - `moe_route_observe::LayerScope` / `refused()` — the layer-attribution
//!   fix for the "thread-local read on the wrong thread" class of bug.
//! - The standing rule that a new measurement instrument needs a live
//!   positive control through the ACTUAL production dispatch path, not a
//!   unit test on the hook in isolation. See `positive_control` below.
//!
//! Usage:
//!   cargo run --release --features gpu -p larql-inference \
//!       --example glm3_moe_route_stability -- <vindex-dir> [max_tokens] [trace-path]
//!
//! Defaults: max_tokens=320, trace-path=<tmp>/glm3_route_trace.jsonl

#[cfg(all(feature = "gpu", target_os = "macos"))]
extern crate blas_src;

#[cfg(all(feature = "gpu", target_os = "macos"))]
mod imp {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Instant;

    use larql_inference::layer_graph::generate::generate;
    use larql_inference::layer_graph::CachedLayerGraph;

    /// A handful of topically distinct raw completion prompts — enough
    /// variety that one prompt's idiosyncrasy (a repetition loop, a short
    /// factual answer that stalls) can't carry the whole result. Raw
    /// continuation style (no chat wrapping) matches the repo's own
    /// steady-state bench protocol (`bench/prompts/gpt-oss-steady-state.txt`,
    /// [`feedback_bench_steady_state_protocol`]) rather than the
    /// instruct-chat "analysis channel" framing, which is prone to short
    /// stock answers that stop well short of "several hundred tokens".
    const PROMPTS: &[(&str, &str)] = &[
        (
            "steady-state-inference-engines",
            // Verbatim continuation seed from bench/prompts/gpt-oss-steady-state.txt
            // (present on the main worktree's working tree, uncommitted there;
            // copied inline here so this harness has no cross-worktree file
            // dependency at run time).
            "The following is a technical discussion about the design of \
inference engines for large language models, covering memory bandwidth, \
quantisation formats, and the scheduling of GPU command buffers.\n\n\
Modern transformer inference splits into two regimes with very different \
cost structures. Prefill processes the entire prompt at once and is \
compute-bound: the attention and feed-forward matrices are multiplied \
against many token positions simultaneously, so arithmetic units stay busy \
and the cost per token is low. Decode, by contrast, generates one token at \
a time. Each step reads the full set of model weights but multiplies them \
against a single vector, so the arithmetic intensity collapses and the step \
becomes bound by memory bandwidth rather than by floating point throughput. \
On a unified-memory machine this distinction is sharp, because the same \
physical memory serves both the processor and the graphics hardware, and \
the achievable bandwidth is a hard ceiling that no amount of kernel tuning \
can exceed.\n\n\
Quantisation attacks this ceiling directly. If the weights occupy fewer \
bytes, fewer bytes must be read per decoded token, and the decode step gets \
faster in almost exact proportion. A four-bit block format stores a group \
of weights alongside a shared scale, trading a small amount of numerical \
precision for a large reduction in traffic. Mixture-of-experts models \
complicate the picture, because only a subset of the expert matrices \
participate in any given token, so the bytes actually read depend on which \
experts the router selects, and the routing pattern varies from token to \
token in ways that are difficult to predict in advance.\n\n\
Given all of this, describe in detail how you would go about measuring \
where the time actually goes in such a system, what controls you would run \
to make sure the measurement is not fooling you, and how you would decide \
whether a proposed optimisation is worth the additional complexity it \
introduces.",
        ),
        (
            "narrative-lighthouse",
            "The lighthouse keeper had not seen another human being in eleven \
weeks when the boat finally appeared on the horizon, low and slow against \
the grey swell. She watched it through the cracked lens of the old \
telescope, the one her predecessor had left bolted to the railing, and \
tried to remember what she would say to whoever stepped off it. Write the \
rest of this story, in careful prose, following her through the rest of \
that day.",
        ),
        (
            "code-reasoning-cache",
            "Here is a Python function that is supposed to implement an LRU \
cache using only a dictionary and a doubly linked list, without using \
collections.OrderedDict:\n\n\
class Node:\n    def __init__(self, key, value):\n        self.key = key\n        self.value = value\n        self.prev = None\n        self.next = None\n\n\
class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.map = {}\n\n\
Explain, step by step, how you would complete this implementation \
correctly, including the get and put methods, the sentinel head/tail nodes, \
and how eviction should work when the cache is full. Then walk through a \
worked example with a capacity of 2 and a sequence of operations.",
        ),
        (
            "explainer-tidal-power",
            "Tidal power generation converts the kinetic and potential energy \
of ocean tides into electricity, using a mechanism that is in some ways \
simpler than wind or solar because the tides are predictable years in \
advance. Explain, at the level of a technically literate but non-expert \
reader, how a tidal barrage differs from a tidal stream turbine, what \
determines the energy available at a given site, what the major \
engineering and ecological trade-offs are, and why so few large-scale \
tidal projects have actually been built compared to wind and solar despite \
the predictability advantage.",
        ),
    ];

    const DEFAULT_MAX_TOKENS: usize = 320;

    struct PromptRun {
        name: &'static str,
        prefill_len: usize,
        decode_len: usize,
        prefill_ms: f64,
        avg_decode_ms: f64,
        sample_text: String,
    }

    /// One decode-step observation for one layer: the sorted top-k expert
    /// ids the router selected. Sorted so set-equality/intersection doesn't
    /// depend on the router's internal rank order.
    type ExpertSet = Vec<usize>;

    fn changed_count(a: &ExpertSet, b: &ExpertSet) -> usize {
        // a, b both sorted, same cardinality (top_k is constant per model).
        let mut ai = a.iter().peekable();
        let mut bi = b.iter().peekable();
        let mut shared = 0usize;
        while let (Some(&x), Some(&y)) = (ai.peek(), bi.peek()) {
            match x.cmp(y) {
                std::cmp::Ordering::Equal => {
                    shared += 1;
                    ai.next();
                    bi.next();
                }
                std::cmp::Ordering::Less => {
                    ai.next();
                }
                std::cmp::Ordering::Greater => {
                    bi.next();
                }
            }
        }
        a.len().saturating_sub(shared)
    }

    fn jaccard(a: &ExpertSet, b: &ExpertSet) -> f64 {
        let k = a.len();
        let changed = changed_count(a, b);
        let shared = k - changed;
        let union = 2 * k - shared;
        if union == 0 {
            1.0
        } else {
            shared as f64 / union as f64
        }
    }

    /// Per-layer accumulator across every prompt's decode-only sequence.
    #[derive(Default)]
    struct LayerStats {
        transitions: usize,
        // changed_hist[c] = count of transitions where exactly c of top_k
        // experts differ from the previous decode step, c in 0..=top_k.
        changed_hist: HashMap<usize, usize>,
        jaccard_sum: f64,
        run_lengths: Vec<usize>,
    }

    impl LayerStats {
        fn record_sequence(&mut self, seq: &[ExpertSet]) {
            if seq.is_empty() {
                return;
            }
            let mut run_len = 1usize;
            for w in seq.windows(2) {
                let (prev, cur) = (&w[0], &w[1]);
                let c = changed_count(prev, cur);
                *self.changed_hist.entry(c).or_insert(0) += 1;
                self.jaccard_sum += jaccard(prev, cur);
                self.transitions += 1;
                if c == 0 {
                    run_len += 1;
                } else {
                    self.run_lengths.push(run_len);
                    run_len = 1;
                }
            }
            self.run_lengths.push(run_len);
        }

        fn mean_jaccard(&self) -> f64 {
            if self.transitions == 0 {
                1.0
            } else {
                self.jaccard_sum / self.transitions as f64
            }
        }

        fn mean_run_len(&self) -> f64 {
            if self.run_lengths.is_empty() {
                0.0
            } else {
                self.run_lengths.iter().sum::<usize>() as f64 / self.run_lengths.len() as f64
            }
        }

        fn median_run_len(&self) -> f64 {
            if self.run_lengths.is_empty() {
                return 0.0;
            }
            let mut v = self.run_lengths.clone();
            v.sort_unstable();
            let mid = v.len() / 2;
            if v.len().is_multiple_of(2) {
                (v[mid - 1] + v[mid]) as f64 / 2.0
            } else {
                v[mid] as f64
            }
        }

        fn pct(&self, c: usize) -> f64 {
            if self.transitions == 0 {
                return 0.0;
            }
            *self.changed_hist.get(&c).unwrap_or(&0) as f64 / self.transitions as f64 * 100.0
        }

        fn merge(&mut self, other: &LayerStats) {
            self.transitions += other.transitions;
            for (k, v) in &other.changed_hist {
                *self.changed_hist.entry(*k).or_insert(0) += v;
            }
            self.jaccard_sum += other.jaccard_sum;
            self.run_lengths.extend(other.run_lengths.iter().copied());
        }

        /// Run-length histogram as percentages of ALL runs: exact buckets
        /// for 1/2/3/4, then a coarser tail. This is the number the task
        /// actually cares about — "how many consecutive tokens keep the
        /// exact same expert set" — separate from the per-transition
        /// mask-change-magnitude table above.
        fn run_len_histogram_pct(&self) -> Vec<(&'static str, f64)> {
            let n = self.run_lengths.len();
            if n == 0 {
                return vec![];
            }
            let mut buckets = [0usize; 6]; // 1,2,3,4,5-9,10+
            for &l in &self.run_lengths {
                let idx = match l {
                    1 => 0,
                    2 => 1,
                    3 => 2,
                    4 => 3,
                    5..=9 => 4,
                    _ => 5,
                };
                buckets[idx] += 1;
            }
            let labels = ["1", "2", "3", "4", "5-9", "10+"];
            labels
                .iter()
                .zip(buckets.iter())
                .map(|(&label, &count)| (label, count as f64 / n as f64 * 100.0))
                .collect()
        }

        fn max_run_len(&self) -> usize {
            self.run_lengths.iter().copied().max().unwrap_or(0)
        }
    }

    /// One JSONL trace line: `{"layer":L,"seq":S,"experts":[[e0,e1,...]]}`.
    /// Hand-parsed — the writer (`larql_compute::ffn::expert_weight::trace`)
    /// emits only non-negative integers with a fixed field order, so a small
    /// dependency-free scanner is exact for this format and avoids pulling
    /// serde_json into an example crate that has no other use for it.
    fn parse_trace_line(line: &str) -> Option<(usize, ExpertSet)> {
        let layer_key = "\"layer\":";
        let experts_key = "\"experts\":[[";
        let l_start = line.find(layer_key)? + layer_key.len();
        let l_end = line[l_start..].find(',')? + l_start;
        let layer: usize = line[l_start..l_end].trim().parse().ok()?;

        let e_start = line.find(experts_key)? + experts_key.len();
        let e_end = line[e_start..].find(']')? + e_start;
        let mut experts: ExpertSet = line[e_start..e_end]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        experts.sort_unstable();
        Some((layer, experts))
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args().skip(1);
        let vindex_path = PathBuf::from(
            args.next()
                .ok_or("usage: glm3_moe_route_stability <vindex-dir> [max_tokens] [trace-path]")?,
        );
        let max_tokens: usize = args
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_TOKENS);
        let trace_path = args.next().map(PathBuf::from).unwrap_or_else(|| {
            std::env::temp_dir().join(format!("glm3_route_trace_{}.jsonl", std::process::id()))
        });

        if !vindex_path.is_dir() {
            return Err(format!("not a vindex dir: {}", vindex_path.display()).into());
        }

        // Must be set before the first routed forward pass — the sink is a
        // process-wide `OnceLock` resolved on first use
        // (`larql_compute::ffn::expert_weight::trace::sink`).
        std::env::set_var(larql_compute::options::ENV_MOE_ROUTE_TRACE, &trace_path);
        println!("━━━ GLM-3: real-decode MoE route-mask stability ━━━━━━━━━━━━━━━━━━");
        println!("  vindex:      {}", vindex_path.display());
        println!("  max_tokens:  {max_tokens} per prompt");
        println!("  trace file:  {}", trace_path.display());
        println!();

        // ── Load once, reuse across every prompt ────────────────────────
        let load_start = Instant::now();
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let cfg = larql_vindex::load_vindex_config(&vindex_path)?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
        let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
        index.load_attn_kquant(&vindex_path)?;
        index.load_interleaved_kquant(&vindex_path)?;
        let _ = index.load_lm_head_kquant(&vindex_path);
        let mut weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
        let num_layers = weights.num_layers;
        println!(
            "  loaded: family={} layers={} hidden={} ({:.1}s)",
            cfg.family,
            num_layers,
            weights.hidden_size,
            load_start.elapsed().as_secs_f64()
        );

        let bands = cfg.layer_bands.clone().ok_or(
            "vindex has no layer_bands — the early/mid/late depth bucketing this harness \
             reports depends on it (gpt-oss-20b-q4k.vindex ships one; a different vindex might not)",
        )?;
        println!(
            "  depth bands: syntax(early)={:?} knowledge(mid)={:?} output(late)={:?}",
            bands.syntax, bands.knowledge, bands.output
        );
        println!();

        let metal_backend =
            larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable")?;
        let cached = CachedLayerGraph::from_residuals(Vec::new());

        let mut runs: Vec<PromptRun> = Vec::new();
        for (name, text) in PROMPTS {
            let token_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, text)?;
            let prefill_len = token_ids.len();
            let t0 = Instant::now();
            let result = generate(
                &mut weights,
                &tokenizer,
                &token_ids,
                max_tokens,
                &index,
                &metal_backend,
                &cached,
                0..num_layers,
            );
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let decode_len = result.tokens.len();
            let sample_text: String = result
                .tokens
                .iter()
                .map(|(t, _)| t.as_str())
                .collect::<String>()
                .chars()
                .take(160)
                .collect();
            println!(
                "  [{name}] prefill={prefill_len} tok, decode={decode_len} tok, \
                 wall={wall_ms:.0}ms, prefill={:.0}ms, avg_decode={:.1}ms/tok ({:.1} tok/s)",
                result.prefill_ms,
                result.avg_decode_ms(),
                result.decode_tok_s(),
            );
            println!("    sample: {sample_text:?}");
            if let Some(err) = &result.error {
                println!("    NOTE: generate() reported an error: {err:?}");
            }
            runs.push(PromptRun {
                name,
                prefill_len,
                decode_len,
                prefill_ms: result.prefill_ms,
                avg_decode_ms: result.avg_decode_ms(),
                sample_text,
            });
        }
        println!();

        // ── Positive control ─────────────────────────────────────────────
        let refused = larql_compute::moe_route_observe::refused();
        let raw = std::fs::read_to_string(&trace_path)
            .map_err(|e| format!("could not read trace file {}: {e}", trace_path.display()))?;
        let mut by_layer: HashMap<usize, Vec<ExpertSet>> = HashMap::new();
        let mut parse_failures = 0usize;
        for line in raw.lines() {
            match parse_trace_line(line) {
                Some((layer, experts)) => by_layer.entry(layer).or_default().push(experts),
                None => parse_failures += 1,
            }
        }
        // `generate_streaming`'s own docs: "Fires on_token for every
        // generated token... including the first, which comes out of the
        // prefill." So the first emitted token's routing decision is the
        // LAST row of the prefill block, not a fresh decode-step call —
        // total routed calls per prompt is normally
        // prefill_len + (decode_len - 1), not prefill_len + decode_len.
        // (Caught by this harness's own consistency check on the first
        // smoke run: every layer was short by exactly `runs.len()` records,
        // one per prompt boundary.)
        //
        // One further wrinkle, caught the same way on the full 320-token
        // run: when a prompt stops BEFORE `max_tokens` (EOS, or the decode
        // loop's `break` on an empty lm_head / GPU-None result), the FINAL
        // decode-loop iteration can perform a routed forward pass (so it IS
        // recorded to the trace) without its token ever being pushed to
        // `result.tokens` (`decode_loop.rs`: `run_one_decode_step` computes
        // the forward *before* the push/break site). That makes the
        // routed-call count for that one prompt `prefill_len + decode_len`
        // (one MORE than the base formula) instead of `- 1`, and it is not
        // knowable from `GenerateResult` alone which of the two happened.
        // Rather than guess, resolve it from the data: try the base formula
        // for every prompt, then attribute any leftover trace records (bounded
        // by the number of prompts that stopped short of `max_tokens`) to
        // those prompts. A leftover that doesn't reconcile this way is left
        // as a genuine control failure, not silently absorbed.
        let base_calls: Vec<usize> = runs
            .iter()
            .map(|r| r.prefill_len + r.decode_len.saturating_sub(1))
            .collect();
        let base_total: usize = base_calls.iter().sum();
        let early_stopped: Vec<usize> = runs
            .iter()
            .enumerate()
            .filter(|(_, r)| r.decode_len < max_tokens)
            .map(|(i, _)| i)
            .collect();

        let mut control_ok = refused == 0 && parse_failures == 0;
        println!("━━━ Positive control ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  moe_route_observe::refused() = {refused}  (0 = every routed selection had a layer attribution)");
        println!("  JSONL parse failures          = {parse_failures}");
        println!(
            "  base expected records/layer   = {base_total}  (sum of prefill_len+decode_len-1 over {} prompts)",
            runs.len()
        );
        println!(
            "  prompts stopped short of max_tokens={max_tokens}: {:?}",
            early_stopped
                .iter()
                .map(|&i| (runs[i].name, runs[i].decode_len))
                .collect::<Vec<_>>()
        );

        // Verify every layer agrees on a single actual total, then try to
        // reconcile that total against `base_total` via the early-stop
        // wrinkle above.
        let mut layer_totals: Vec<usize> = (0..num_layers)
            .map(|l| by_layer.get(&l).map(|v| v.len()).unwrap_or(0))
            .collect();
        layer_totals.dedup();
        let mut extra_calls: HashMap<usize, usize> = HashMap::new();
        if by_layer.is_empty() {
            control_ok = false;
            println!("  MISMATCH: trace file has zero records — routing was never observed");
        } else if layer_totals.len() != 1 {
            control_ok = false;
            println!(
                "  MISMATCH: layers disagree on total record count ({:?} distinct totals across {num_layers} layers) \
                 — a dispatch branch is bypassing moe_route_observe for only SOME layers",
                layer_totals.len()
            );
        } else {
            let actual_total = layer_totals[0];
            if actual_total == base_total {
                println!("  every layer matches the base formula exactly ({actual_total} records) — no early-stop wrinkle this run.");
            } else if actual_total > base_total && actual_total - base_total <= early_stopped.len()
            {
                let leftover = actual_total - base_total;
                println!(
                    "  every layer over base by {leftover} record(s), attributed to the {leftover} \
                     of {} early-stopped prompt(s) whose final decode step routed but did not emit \
                     a token (see comment above).",
                    early_stopped.len()
                );
                for &i in early_stopped.iter().take(leftover) {
                    extra_calls.insert(i, 1);
                }
            } else {
                control_ok = false;
                println!(
                    "  MISMATCH: actual records/layer={actual_total} vs base formula={base_total} \
                     (delta={}) does not reconcile against {} early-stopped prompt(s) — \
                     a dispatch branch may be bypassing moe_route_observe",
                    actual_total as i64 - base_total as i64,
                    early_stopped.len()
                );
            }
        }
        // Diversity sanity: a genuinely live router should not select the
        // identical set at every single position across the whole run.
        let mut any_layer_all_identical = false;
        for (l, seq) in &by_layer {
            if seq.len() > 4 && seq.iter().all(|s| s == &seq[0]) {
                any_layer_all_identical = true;
                println!("  NOTE layer {l}: every single observed selection (prefill+decode) is identical — check this is not a stuck router.");
            }
        }
        println!(
            "  layers with >=1 distinct expert set observed: {}/{}",
            by_layer
                .values()
                .filter(|v| v.iter().collect::<std::collections::HashSet<_>>().len() > 1)
                .count(),
            num_layers
        );
        println!(
            "  RESULT: positive control {}",
            if control_ok {
                "PASSED"
            } else {
                "FAILED — see MISMATCH/NOTE lines above"
            }
        );
        println!();
        if !control_ok {
            eprintln!(
                "positive control failed — refusing to report run-length numbers from an \
                 unverified trace. Fix the dispatch/attribution gap above, then re-run."
            );
            return Err("positive control failed".into());
        }
        let _ = any_layer_all_identical; // surfaced above; not fatal by itself.

        // ── Slice each layer's continuous record stream into per-prompt
        // decode-only windows, discarding prefill. The trajectory for
        // emitted token 1 is the LAST prefill row (see the routed-call-count
        // note above); tokens 2..decode_len each get their own fresh
        // decode-step row. Reassembling them this way gives a `decode_len`
        // -long sequence that matches what was actually emitted, rather
        // than silently dropping token 1's real expert set from the
        // trajectory. ─────────────────────────────────────────────────
        let mut layer_stats: HashMap<usize, LayerStats> = HashMap::new();
        for l in 0..num_layers {
            let all = &by_layer[&l];
            let mut cursor = 0usize;
            let mut stats = LayerStats::default();
            for (i, run) in runs.iter().enumerate() {
                if run.decode_len == 0 {
                    cursor += run.prefill_len;
                    continue;
                }
                let prefill_end = cursor + run.prefill_len;
                // Reconciled above: base formula, plus one extra routed-but-
                // unpushed trailing call for the early-stopped prompts the
                // leftover was attributed to.
                let new_decode_calls =
                    base_calls[i] - run.prefill_len + extra_calls.get(&i).copied().unwrap_or(0);
                let decode_end = prefill_end + new_decode_calls;
                let mut decode_seq: Vec<ExpertSet> = Vec::with_capacity(run.decode_len);
                decode_seq.push(all[prefill_end - 1].clone()); // token 1 == last prefill row
                                                               // The extra reconciled call (if any) is the unpushed final
                                                               // routed-but-not-emitted step — real routing happened, but no
                                                               // token exists for it, so it is consumed (cursor advances
                                                               // past it) without joining the per-token trajectory.
                let pushed_end = decode_end - extra_calls.get(&i).copied().unwrap_or(0);
                decode_seq.extend_from_slice(&all[prefill_end..pushed_end]);
                stats.record_sequence(&decode_seq);
                cursor = decode_end;
            }
            layer_stats.insert(l, stats);
        }

        // ── Per-layer report ──────────────────────────────────────────────
        println!("━━━ Per-layer run-length / mask-change histogram (decode tokens only) ━━━");
        println!(
            "{:>5} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>6} {:>8}",
            "layer",
            "band",
            "trans",
            "same%",
            "chg1%",
            "chg2%",
            "chg3%",
            "chg4%",
            "meanJacc",
            "meanRun",
            "medRun",
            "nRuns"
        );
        for l in 0..num_layers {
            let band = if l >= bands.syntax.0 && l <= bands.syntax.1 {
                "early"
            } else if l >= bands.knowledge.0 && l <= bands.knowledge.1 {
                "mid"
            } else {
                "late"
            };
            let s = &layer_stats[&l];
            println!(
                "{:>5} {:>6} {:>6} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>10.3} {:>10.2} {:>6.1} {:>8}",
                l,
                band,
                s.transitions,
                s.pct(0),
                s.pct(1),
                s.pct(2),
                s.pct(3),
                s.pct(4),
                s.mean_jaccard(),
                s.mean_run_len(),
                s.median_run_len(),
                s.run_lengths.len(),
            );
        }
        println!();

        // ── Depth-bucket rollup ──────────────────────────────────────────
        let mut early = LayerStats::default();
        let mut mid = LayerStats::default();
        let mut late = LayerStats::default();
        let mut overall = LayerStats::default();
        for l in 0..num_layers {
            let s = &layer_stats[&l];
            overall.merge(s);
            if l >= bands.syntax.0 && l <= bands.syntax.1 {
                early.merge(s);
            } else if l >= bands.knowledge.0 && l <= bands.knowledge.1 {
                mid.merge(s);
            } else {
                late.merge(s);
            }
        }
        println!("━━━ Depth-bucket rollup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for (name, s) in [
            ("early (syntax)", &early),
            ("mid (knowledge)", &mid),
            ("late (output)", &late),
            ("ALL LAYERS", &overall),
        ] {
            println!(
                "  {name:16} trans={:<7} same%={:<6.1} chg1%={:<6.1} chg2%={:<6.1} chg3%={:<6.1} chg4%={:<6.1} meanJacc={:<7.3} meanRun={:<7.2} medRun={:<6.1} maxRun={:<5} nRuns={}",
                s.transitions, s.pct(0), s.pct(1), s.pct(2), s.pct(3), s.pct(4),
                s.mean_jaccard(), s.mean_run_len(), s.median_run_len(), s.max_run_len(), s.run_lengths.len(),
            );
        }
        println!();

        // Run-length histogram proper: what the task actually asked for —
        // "how many consecutive tokens keep the exact same expert set before
        // it changes at all" — as a distribution over run lengths, not just
        // mean/median. Reported per depth bucket, not collapsed to one
        // number.
        println!("━━━ Run-length histogram (% of runs, by depth bucket) ━━━━━━━━━━━━");
        println!(
            "  {:16} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "bucket", "len=1", "len=2", "len=3", "len=4", "5-9", "10+"
        );
        for (name, s) in [
            ("early (syntax)", &early),
            ("mid (knowledge)", &mid),
            ("late (output)", &late),
            ("ALL LAYERS", &overall),
        ] {
            let h = s.run_len_histogram_pct();
            if h.is_empty() {
                continue;
            }
            println!(
                "  {name:16} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}%",
                h[0].1, h[1].1, h[2].1, h[3].1, h[4].1, h[5].1
            );
        }
        println!();

        // ── BW-B1 cross-reference: analytical amortization estimate ───────
        // BW-B1 (`docs/diagnoses/bwb-compact-dense-oracle.md`) found N* —
        // the number of reused calls a materialize-once representation needs
        // to break even against gather/dense — ranging ~0.34-0.39 (K=1024,
        // vs gather) up to ~9.2 (K=2048, vs dense) on qwen3-0.6b kernel
        // timings, and used a SYNTHETIC 32-position smooth-drift trajectory
        // (mean strict run length 1.06 positions) as its only stand-in for
        // real reuse. This is an ANALYTICAL estimate from the histogram
        // above, cross-referenced against those already-measured N*
        // thresholds — not a new wall-clock measurement of a compiled
        // kernel against this trace (that is follow-up work).
        const BWB1_SYNTHETIC_MEAN_RUN_LEN: f64 = 1.06;
        const BWB1_N_STAR_LOW: f64 = 0.34; // K=1024 vs gather, most favorable
        const BWB1_N_STAR_HIGH: f64 = 9.2; // K=2048 vs dense, least favorable
        println!("━━━ BW-B1 cross-reference (analytical, not a new kernel benchmark) ━━━");
        println!(
            "  BW-B1 synthetic-proxy mean strict run length: {BWB1_SYNTHETIC_MEAN_RUN_LEN:.2} positions"
        );
        for (name, s) in [
            ("early", &early),
            ("mid", &mid),
            ("late", &late),
            ("ALL LAYERS", &overall),
        ] {
            let real_mean = s.mean_run_len();
            let ratio = if BWB1_SYNTHETIC_MEAN_RUN_LEN > 0.0 {
                real_mean / BWB1_SYNTHETIC_MEAN_RUN_LEN
            } else {
                f64::NAN
            };
            let clears_low = real_mean >= BWB1_N_STAR_LOW;
            let clears_high = real_mean >= BWB1_N_STAR_HIGH;
            // Each run of length L contributes 1 "compile" token (its first)
            // and (L-1) "reuse" tokens (the rest) — so the fraction of
            // decode tokens that would have reused a materialized mask is
            // exactly the identical-mask transition rate, `pct(0)`.
            // (total_tokens = same_count + num_runs; reuse = same_count.)
            let reuse_fraction = s.pct(0) / 100.0;
            println!(
                "  {name:12} real mean run={:<7.2} ({:.2}x the synthetic proxy) reuse_fraction={:<6.1}% \
                 clears N*_low(0.34)={clears_low} clears N*_high(9.2)={clears_high}",
                real_mean,
                ratio,
                reuse_fraction * 100.0,
            );
        }
        println!();
        println!("(runs summary)");
        for r in &runs {
            println!(
                "  [{}] prefill={} decode={} prefill_ms={:.0} avg_decode_ms={:.1}",
                r.name, r.prefill_len, r.decode_len, r.prefill_ms, r.avg_decode_ms
            );
            let _ = &r.sample_text; // already printed above during generation
        }

        Ok(())
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    imp::main()
}

#[cfg(not(all(feature = "gpu", target_os = "macos")))]
fn main() {
    eprintln!("glm3_moe_route_stability requires `--features gpu` on macOS.");
}
