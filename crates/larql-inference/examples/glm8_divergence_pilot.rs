//! GLM-8 — successful-vs-failed divergence pilot (PILOT scale).
//!
//! Runs entirely on the model already used throughout this codebase's
//! other active research (gpt-oss-20b, served from a local Q4K vindex —
//! no GLM weights or architecture support needed). See project memory
//! `project_glm_capability_ladder` for the full ladder; this is rung
//! GLM-8.
//!
//! ## What this measures
//!
//! For a small set of checkable word-problem prompts, sample several
//! independent temperature rollouts of the SAME prompt. Some rollouts
//! land on the correct final answer, some don't. Find pairs of rollouts
//! (one correct, one incorrect) whose generated token sequences share an
//! identical prefix and then diverge into different next tokens. At that
//! divergence point, the model's forward pass over the shared prefix is
//! identical for both rollouts (deterministic given tokens) — the only
//! thing that differs is which token got *sampled* from that one shared
//! distribution. So the softmax probability the model itself assigned to
//! "the token that turned out to be on the success path" vs "the token
//! that turned out to be on the failure path" is a direct, dispatch-real
//! readout of internal state immediately before the fork: it is the same
//! probability value the production decode loop already computed and
//! used to draw the sample (`sample_and_emit`'s `on_token(id, text,
//! prob)` callback), not a separately re-derived proxy.
//!
//! That is the "predictive units" measure this pilot reports: does
//! `log P(success_token | shared_prefix) > log P(fail_token |
//! shared_prefix)` more often than chance across captured divergence
//! events? No new capture hook is introduced — this reuses the existing,
//! already-tested per-token callback
//! (`layer_graph::generate::gpu::sampling_step::sample_and_emit`, see
//! `generate_streaming_callback_fires_per_token` in
//! `layer_graph::generate::cpu` tests) rather than inventing a new
//! instrumentation surface, and it sidesteps any CPU/Metal dispatch-
//! parity risk because both probabilities come from the one real
//! generation run that produced them — nothing is recomputed.
//!
//! ## Stages
//!
//! 1. **Pre-screen** (`run_prescreen`): sample each candidate task
//!    several times at temperature, grade pass/fail, print the measured
//!    split. Only tasks with a genuine mix (not 0% or 100%) go on to
//!    stage 2.
//! 2. **Divergence-pair capture** (`run_main_harness`): more rollouts per
//!    qualifying task, pairwise divergence detection between every
//!    (pass, fail) rollout pair, deduplicated by (task, prefix length,
//!    diverging token ids) so replays of the same fork aren't double
//!    counted.
//! 3. **Predictive-signal report**: raw per-event arms plus a sign test
//!    and mean log-probability gap.
//!
//! Usage:
//! ```text
//! cargo run --release -p larql-inference --features gpu \
//!   --example glm8_divergence_pilot -- [vindex-dir]
//! ```
//! Defaults to `~/chris-models/gpt-oss-20b-q4k.vindex` (verified present
//! on this machine, same one `bench/prompts/gpt-oss-steady-state.txt`
//! and prior sessions' gpt-oss work in this repo use).
//!
//! `LARQL_GPU_ROUTE=1` is set internally for the harness process. Before
//! this file was written, a manual greedy A/B (route on vs off, same
//! prompt, same seed) confirmed token-for-token identical output on this
//! vindex — see the report for the exact numbers (12.7 vs 28.4 tok/s,
//! same generated text). It's ~2.2x faster and this is a pilot with a
//! real wall-clock budget; `verify_sampling_is_real` below is the
//! in-process control that still runs every time (greedy determinism +
//! seed-divergence under temperature), independent of that one-off check.

#[cfg(all(feature = "gpu", target_os = "macos"))]
extern crate blas_src;

#[cfg(all(feature = "gpu", target_os = "macos"))]
mod pilot {
    use larql_inference::layer_graph::generate::{generate_streaming, EosConfig, SamplingConfig};
    use larql_inference::layer_graph::CachedLayerGraph;
    use larql_inference::wrap_chat_prompt;
    use larql_vindex::VectorIndex;
    use std::path::{Path, PathBuf};

    // ── Tunables (named, not scattered magic numbers) ──────────────────

    /// Sampling temperature for every non-greedy rollout in this pilot.
    const SAMPLING_TEMPERATURE: f32 = 0.9;
    /// Decode budget per rollout. gpt-oss's harmony format spends the
    /// first chunk on an `analysis` channel before the graded `final`
    /// channel; 200 was enough for a one-off greedy probe run on this
    /// vindex (same eggs/baskets question used as the control prompt
    /// below) to reach a stated `Answer:` line comfortably.
    const MAX_TOKENS: usize = 200;
    /// Rollouts per task in the capability pre-screen (stage 1).
    const PRESCREEN_SAMPLES_PER_TASK: usize = 6;
    /// Rollouts per qualifying task in the divergence-pair harness
    /// (stage 2) — larger than the pre-screen so pairwise (pass, fail)
    /// comparisons have a real chance of finding shared-prefix forks.
    const MAIN_SAMPLES_PER_TASK: usize = 14;
    /// A task pre-screens as "genuine mix" when its correct count among
    /// `PRESCREEN_SAMPLES_PER_TASK` rollouts falls strictly between 0
    /// and the sample count — not "too easy" (never fails) and not
    /// "too hard" (never succeeds).
    fn is_genuine_mix(correct: usize, total: usize) -> bool {
        correct > 0 && correct < total
    }
    /// Harmony format's textual seam between the `analysis` and `final`
    /// channels once special tokens (`<|end|><|start|>assistant
    /// <|channel|>`) are decoded to their (empty-string) surface forms —
    /// empirically confirmed by a raw per-token dump of a probe rollout
    /// on this vindex: the token stream decodes to the literal substring
    /// `...assistantfinal...` at that seam. Used to isolate the graded
    /// answer from scratch-work in the `analysis` channel, which
    /// sometimes free-associates numbers that
    /// aren't the model's stated answer.
    const HARMONY_FINAL_CHANNEL_SEAM: &str = "assistantfinal";
    const ANSWER_MARKER: &str = "Answer:";

    // ── Task set ─────────────────────────────────────────────────────

    struct Task {
        name: &'static str,
        question: &'static str,
        expected: i64,
    }

    /// Six short arithmetic word problems spanning a difficulty range —
    /// per the pilot brief, tried up front rather than hand-picked after
    /// seeing results, so the pre-screen numbers are a real measurement
    /// and not survivorship-selected.
    fn candidate_tasks() -> Vec<Task> {
        vec![
            Task {
                name: "eggs_one_step",
                question: "Tom has 4 baskets with 6 eggs in each basket. He sells 9 eggs. \
                            How many eggs does he have left?",
                expected: 15,
            },
            Task {
                name: "train_two_leg",
                question: "A train travels 60 miles per hour for 3 hours, then 40 miles per \
                            hour for 2 hours. What is the total distance traveled, in miles?",
                expected: 260,
            },
            Task {
                name: "marbles_conditional",
                question: "There are 23 red marbles and 17 blue marbles in a jar. 12 marbles \
                            are removed. Exactly 5 of the removed marbles were blue. How many \
                            red marbles remain in the jar?",
                expected: 16,
            },
            Task {
                name: "apples_fractions",
                question: "A store had 144 apples. It sold 3/8 of them in the morning and \
                            1/6 of the REMAINING apples in the afternoon. How many apples \
                            are left after the afternoon?",
                expected: 75,
            },
            Task {
                name: "age_algebra",
                question: "Jack is twice as old as his sister Emma. In 5 years, the sum of \
                            their ages will be 40. How old is Emma right now?",
                expected: 10,
            },
            Task {
                name: "remainder_boxes",
                question: "A bakery made 250 cupcakes and packed them into boxes of 12. \
                            After packing as many full boxes as possible, how many cupcakes \
                            were left over (not in a full box)?",
                expected: 10,
            },
        ]
    }

    fn build_question(q: &str) -> String {
        format!(
            "{q} Give your reasoning, then end with a final line of the exact form \
             'Answer: <number>' (a single integer, nothing else on that line)."
        )
    }

    // ── Answer extraction ───────────────────────────────────────────

    /// Parse the first integer following `Answer:` inside `text`,
    /// restricted to content after the harmony `final`-channel seam
    /// when present (falls back to the whole text otherwise — some
    /// truncated rollouts never reach the seam within `MAX_TOKENS`).
    fn extract_answer(text: &str) -> Option<i64> {
        let scoped = match text.rfind(HARMONY_FINAL_CHANNEL_SEAM) {
            Some(idx) => &text[idx + HARMONY_FINAL_CHANNEL_SEAM.len()..],
            None => text,
        };
        let marker_idx = scoped.find(ANSWER_MARKER)?;
        let after = &scoped[marker_idx + ANSWER_MARKER.len()..];
        parse_leading_integer(after)
    }

    /// Skip leading whitespace/punctuation, then parse an optional sign
    /// and a run of digits. Returns `None` if no digit run is found.
    fn parse_leading_integer(s: &str) -> Option<i64> {
        let s =
            s.trim_start_matches(|c: char| c.is_whitespace() || c == '*' || c == '"' || c == '\'');
        let mut chars = s.char_indices().peekable();
        let mut end = 0usize;
        let mut seen_digit = false;
        if let Some(&(_, c)) = chars.peek() {
            if c == '-' || c == '+' {
                chars.next();
            }
        }
        for (i, c) in chars {
            if c.is_ascii_digit() {
                seen_digit = true;
                end = i + c.len_utf8();
            } else {
                break;
            }
        }
        if !seen_digit {
            return None;
        }
        s[..end].parse::<i64>().ok()
    }

    // ── Rollout capture ─────────────────────────────────────────────

    /// One temperature (or greedy) generation, with the per-token
    /// (id, surface text, model-assigned softmax probability) trace the
    /// production decode loop already computes — this is the "internal
    /// state at each step" readout the whole pilot is built on.
    struct Rollout {
        seed: u64,
        generated_ids: Vec<u32>,
        generated_texts: Vec<String>,
        generated_probs: Vec<f64>,
        answer: Option<i64>,
        pass: bool,
        wall_ms: f64,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rollout(
        weights: &mut larql_inference::model::ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        index: &VectorIndex,
        backend: &dyn larql_compute::ComputeBackend,
        cached: &CachedLayerGraph,
        eos: &EosConfig,
        token_ids: &[u32],
        expected: i64,
        sampling: SamplingConfig,
        seed: u64,
    ) -> Rollout {
        let num_layers = weights.num_layers;
        let mut trace: Vec<(u32, String, f64)> = Vec::with_capacity(MAX_TOKENS);
        let t0 = std::time::Instant::now();
        let result = generate_streaming(
            weights,
            tokenizer,
            token_ids,
            MAX_TOKENS,
            index,
            backend,
            cached,
            0..num_layers,
            sampling,
            eos,
            |id, text, prob| trace.push((id, text.to_string(), prob)),
            None,
        );
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(err) = &result.error {
            eprintln!("  [rollout seed={seed}] generate error: {err}");
        }
        let joined: String = trace.iter().map(|(_, t, _)| t.as_str()).collect();
        let answer = extract_answer(&joined);
        let pass = answer == Some(expected);
        Rollout {
            seed,
            generated_ids: trace.iter().map(|(id, _, _)| *id).collect(),
            generated_texts: trace.iter().map(|(_, t, _)| t.clone()).collect(),
            generated_probs: trace.iter().map(|(_, _, p)| *p).collect(),
            answer,
            pass,
            wall_ms,
        }
    }

    // ── Divergence-pair detection ────────────────────────────────────

    /// A single fork: `prefix_len` generated tokens were identical
    /// between a passing and a failing rollout of the same task, then
    /// they diverged into `token_pass` vs `token_fail` — with the
    /// model's own probability for each, read from the real generation
    /// trace that produced it.
    #[derive(Clone)]
    struct DivergenceEvent {
        task: &'static str,
        prefix_len: usize,
        prefix_preview: String,
        token_pass: u32,
        text_pass: String,
        prob_pass: f64,
        token_fail: u32,
        text_fail: String,
        prob_fail: f64,
        /// Number of (pass, fail) rollout pairs that independently
        /// reproduced this exact fork — replays of the same branch
        /// point are folded in here rather than double-counted as
        /// separate evidence.
        support: usize,
    }

    impl DivergenceEvent {
        fn delta_log_prob(&self) -> f64 {
            self.prob_pass.max(f64::MIN_POSITIVE).ln() - self.prob_fail.max(f64::MIN_POSITIVE).ln()
        }
    }

    /// First index at which two rollouts' generated token ids differ,
    /// bounded by the shorter sequence. `None` when one is a prefix of
    /// the other (no token-level fork observed within the captured
    /// range) — that pair contributes no evidence.
    fn first_divergence(a: &Rollout, b: &Rollout) -> Option<usize> {
        let n = a.generated_ids.len().min(b.generated_ids.len());
        (0..n).find(|&i| a.generated_ids[i] != b.generated_ids[i])
    }

    fn preview(texts: &[String], upto: usize) -> String {
        let joined: String = texts[..upto].iter().map(|s| s.as_str()).collect();
        let trimmed = joined.trim();
        const MAX_PREVIEW_CHARS: usize = 80;
        if trimmed.chars().count() > MAX_PREVIEW_CHARS {
            let head: String = trimmed.chars().rev().take(MAX_PREVIEW_CHARS).collect();
            format!("…{}", head.chars().rev().collect::<String>())
        } else {
            trimmed.to_string()
        }
    }

    /// All (pass, fail) rollout pairs for one task, deduplicated by
    /// (prefix length, diverging token ids) — see module doc for why
    /// this is prefix-consistent by construction (two pairs agreeing on
    /// "first difference at index i" necessarily agree on tokens
    /// `0..i`).
    fn collect_divergence_events(
        task_name: &'static str,
        rollouts: &[Rollout],
    ) -> Vec<DivergenceEvent> {
        let passes: Vec<&Rollout> = rollouts.iter().filter(|r| r.pass).collect();
        let fails: Vec<&Rollout> = rollouts.iter().filter(|r| !r.pass).collect();
        let mut events: Vec<DivergenceEvent> = Vec::new();
        for p in &passes {
            for f in &fails {
                let Some(i) = first_divergence(p, f) else {
                    continue;
                };
                let (tp, tf) = (p.generated_ids[i], f.generated_ids[i]);
                if let Some(existing) = events
                    .iter_mut()
                    .find(|e| e.prefix_len == i && e.token_pass == tp && e.token_fail == tf)
                {
                    existing.support += 1;
                    continue;
                }
                events.push(DivergenceEvent {
                    task: task_name,
                    prefix_len: i,
                    prefix_preview: preview(&p.generated_texts, i),
                    token_pass: tp,
                    text_pass: p.generated_texts[i].clone(),
                    prob_pass: p.generated_probs[i],
                    token_fail: tf,
                    text_fail: f.generated_texts[i].clone(),
                    prob_fail: f.generated_probs[i],
                    support: 1,
                });
            }
        }
        events
    }

    // ── Exact two-sided sign test (small n, hand-rolled — no stats dep) ─

    fn ln_factorial(n: u32) -> f64 {
        (1..=n).map(|i| (i as f64).ln()).sum()
    }

    fn binom_pmf(n: u32, k: u32, p: f64) -> f64 {
        let log_c = ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k);
        (log_c + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp()
    }

    /// Exact two-sided binomial test against p=0.5: sum the pmf of every
    /// outcome at least as extreme (pmf <= observed pmf, with a small
    /// relative slack for float noise) as the observed count `k` of
    /// `n` trials.
    fn two_sided_sign_test_p(n: u32, k: u32) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let pk = binom_pmf(n, k, 0.5);
        const SLACK: f64 = 1.0 + 1e-9;
        (0..=n)
            .map(|j| binom_pmf(n, j, 0.5))
            .filter(|&pj| pj <= pk * SLACK)
            .sum()
    }

    // ── Model loading ────────────────────────────────────────────────

    struct Model {
        weights: larql_inference::model::ModelWeights,
        tokenizer: tokenizers::Tokenizer,
        index: VectorIndex,
        vindex_path: PathBuf,
        model_hint: String,
    }

    fn load_model(vindex_path: &Path) -> Result<Model, Box<dyn std::error::Error>> {
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let cfg = larql_vindex::load_vindex_config(vindex_path)?;
        let mut index = VectorIndex::load_vindex(vindex_path, &mut cb)?;
        index.load_attn_kquant(vindex_path)?;
        index.load_interleaved_kquant(vindex_path)?;
        let _ = index.load_lm_head_kquant(vindex_path);
        let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)?;
        let weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)?;
        Ok(Model {
            weights,
            tokenizer,
            index,
            vindex_path: vindex_path.to_path_buf(),
            model_hint: cfg.model,
        })
    }

    fn encode_question(
        model: &Model,
        question: &str,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let wrap = wrap_chat_prompt(
            &model.vindex_path,
            Some(model.model_hint.as_str()),
            &build_question(question),
        );
        Ok(larql_inference::encode_prompt(
            &model.tokenizer,
            &*model.weights.arch,
            &wrap.prompt,
        )?)
    }

    // ── Positive controls (req #3: verify the real dispatch path before
    //    trusting it — no new hook here, so this checks the sampler /
    //    LARQL_GPU_ROUTE dispatch choices this pilot itself makes) ────

    /// Two checks, printed loudly and asserted before any measurement is
    /// trusted:
    /// 1. Greedy decode is deterministic (same tokens twice) — confirms
    ///    the harness's own plumbing round-trips correctly.
    /// 2. Two different seeds at `SAMPLING_TEMPERATURE` produce different
    ///    token sequences — confirms sampling is genuinely active and
    ///    this pilot isn't silently reading a de-facto-greedy path.
    fn verify_sampling_is_real(
        model: &mut Model,
        backend: &dyn larql_compute::ComputeBackend,
        cached: &CachedLayerGraph,
        eos: &EosConfig,
        token_ids: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        const CONTROL_TOKENS: usize = 24;
        let num_layers = model.weights.num_layers;

        let mut greedy_a = Vec::new();
        generate_streaming(
            &mut model.weights,
            &model.tokenizer,
            token_ids,
            CONTROL_TOKENS,
            &model.index,
            backend,
            cached,
            0..num_layers,
            SamplingConfig::greedy(),
            eos,
            |id, _, _| greedy_a.push(id),
            None,
        );
        let mut greedy_b = Vec::new();
        generate_streaming(
            &mut model.weights,
            &model.tokenizer,
            token_ids,
            CONTROL_TOKENS,
            &model.index,
            backend,
            cached,
            0..num_layers,
            SamplingConfig::greedy(),
            eos,
            |id, _, _| greedy_b.push(id),
            None,
        );
        let greedy_deterministic = greedy_a == greedy_b;

        let mut sample_a = Vec::new();
        generate_streaming(
            &mut model.weights,
            &model.tokenizer,
            token_ids,
            CONTROL_TOKENS,
            &model.index,
            backend,
            cached,
            0..num_layers,
            SamplingConfig::temperature(SAMPLING_TEMPERATURE).with_seed(1),
            eos,
            |id, _, _| sample_a.push(id),
            None,
        );
        let mut sample_b = Vec::new();
        generate_streaming(
            &mut model.weights,
            &model.tokenizer,
            token_ids,
            CONTROL_TOKENS,
            &model.index,
            backend,
            cached,
            0..num_layers,
            SamplingConfig::temperature(SAMPLING_TEMPERATURE).with_seed(2),
            eos,
            |id, _, _| sample_b.push(id),
            None,
        );
        let sampling_is_active = sample_a != sample_b;

        println!("━━━ Positive control: is the sampling dispatch real? ━━━━━━━━━━━━━━");
        println!(
            "  greedy×2 deterministic:      {} ({:?} vs {:?})",
            if greedy_deterministic { "PASS" } else { "FAIL" },
            greedy_a,
            greedy_b
        );
        println!(
            "  T={SAMPLING_TEMPERATURE} seed1≠seed2 diverge: {} ({:?} vs {:?})",
            if sampling_is_active { "PASS" } else { "FAIL" },
            sample_a,
            sample_b
        );
        if !greedy_deterministic || !sampling_is_active {
            return Err(
                "positive control failed — sampling dispatch is not behaving as \
                         expected; refusing to trust downstream pass/fail rates"
                    .into(),
            );
        }
        println!();
        Ok(())
    }

    /// Wall-clock diagnostics for a batch of rollouts — printed once per
    /// stage so a slow run is visible without re-deriving it from raw
    /// logs, and so `Rollout::{seed,wall_ms}` aren't write-only fields.
    fn print_timing_summary(label: &str, rollouts: &[&Rollout]) {
        if rollouts.is_empty() {
            return;
        }
        let total_ms: f64 = rollouts.iter().map(|r| r.wall_ms).sum();
        let avg_ms = total_ms / rollouts.len() as f64;
        let seeds: Vec<u64> = rollouts.iter().map(|r| r.seed).collect();
        println!(
            "  [{label}] {} rollouts, seeds {}..{}, {:.0}ms total, {:.0}ms/rollout avg",
            rollouts.len(),
            seeds.iter().min().unwrap(),
            seeds.iter().max().unwrap(),
            total_ms,
            avg_ms
        );
    }

    // ── Stage 1: pre-screen ─────────────────────────────────────────

    struct PrescreenResult {
        task: &'static str,
        rollouts: Vec<Rollout>,
        correct: usize,
        total: usize,
        qualifies: bool,
    }

    fn run_prescreen(
        model: &mut Model,
        backend: &dyn larql_compute::ComputeBackend,
        cached: &CachedLayerGraph,
        eos: &EosConfig,
    ) -> Result<Vec<PrescreenResult>, Box<dyn std::error::Error>> {
        println!("━━━ Stage 1: capability pre-screen ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        let mut out = Vec::new();
        for task in candidate_tasks() {
            let token_ids = encode_question(model, task.question)?;
            let mut rollouts = Vec::new();
            for seed in 0..PRESCREEN_SAMPLES_PER_TASK as u64 {
                let r = run_rollout(
                    &mut model.weights,
                    &model.tokenizer,
                    &model.index,
                    backend,
                    cached,
                    eos,
                    &token_ids,
                    task.expected,
                    SamplingConfig::temperature(SAMPLING_TEMPERATURE).with_seed(seed),
                    seed,
                );
                print!("{}", if r.pass { '.' } else { 'x' });
                use std::io::Write;
                std::io::stdout().flush().ok();
                rollouts.push(r);
            }
            let correct = rollouts.iter().filter(|r| r.pass).count();
            let total = rollouts.len();
            let qualifies = is_genuine_mix(correct, total);
            println!(
                "\n  {:<20} {}/{} correct  expected={}  answers={:?}  {}",
                task.name,
                correct,
                total,
                task.expected,
                rollouts.iter().map(|r| r.answer).collect::<Vec<_>>(),
                if qualifies {
                    "→ QUALIFIES (genuine mix)"
                } else if correct == 0 {
                    "→ skip (always wrong)"
                } else {
                    "→ skip (always right)"
                }
            );
            print_timing_summary("prescreen", &rollouts.iter().collect::<Vec<_>>());
            out.push(PrescreenResult {
                task: task.name,
                rollouts,
                correct,
                total,
                qualifies,
            });
        }
        println!();
        Ok(out)
    }

    // ── Stage 2: divergence-pair harness ────────────────────────────

    fn run_main_harness(
        model: &mut Model,
        backend: &dyn larql_compute::ComputeBackend,
        cached: &CachedLayerGraph,
        eos: &EosConfig,
        prescreen: &[PrescreenResult],
    ) -> Result<Vec<DivergenceEvent>, Box<dyn std::error::Error>> {
        println!("━━━ Stage 2: divergence-pair capture ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        let qualifying: Vec<Task> = candidate_tasks_static()
            .into_iter()
            .filter(|t| prescreen.iter().any(|p| p.task == t.name && p.qualifies))
            .collect();
        if qualifying.is_empty() {
            println!("  No task produced a genuine mix in the pre-screen — nothing to run.");
            return Ok(Vec::new());
        }
        let mut all_events = Vec::new();
        for task in qualifying {
            let token_ids = encode_question(model, task.question)?;
            let mut rollouts = Vec::new();
            for seed in 0..MAIN_SAMPLES_PER_TASK as u64 {
                // Offset the seed range from the pre-screen so stage 2
                // draws independent samples rather than replaying stage 1.
                let seed = seed + 1000;
                let r = run_rollout(
                    &mut model.weights,
                    &model.tokenizer,
                    &model.index,
                    backend,
                    cached,
                    eos,
                    &token_ids,
                    task.expected,
                    SamplingConfig::temperature(SAMPLING_TEMPERATURE).with_seed(seed),
                    seed,
                );
                print!("{}", if r.pass { '.' } else { 'x' });
                use std::io::Write;
                std::io::stdout().flush().ok();
                rollouts.push(r);
            }
            let correct = rollouts.iter().filter(|r| r.pass).count();
            println!(
                "\n  {:<20} {}/{} correct in main harness",
                task.name,
                correct,
                rollouts.len()
            );
            print_timing_summary("main", &rollouts.iter().collect::<Vec<_>>());
            let events = collect_divergence_events(task.name, &rollouts);
            println!(
                "  {:<20} {} unique divergence events ({} supporting pairs)",
                task.name,
                events.len(),
                events.iter().map(|e| e.support).sum::<usize>()
            );
            all_events.extend(events);
        }
        println!();
        Ok(all_events)
    }

    /// `candidate_tasks()` returns owned data; this variant exists so
    /// stage 2 can filter by name without re-deriving the struct fields
    /// each call site — trivial but keeps `run_main_harness` readable.
    fn candidate_tasks_static() -> Vec<Task> {
        candidate_tasks()
    }

    // ── Report ───────────────────────────────────────────────────────

    fn report(events: &[DivergenceEvent]) {
        println!("━━━ Stage 3: predictive-signal report ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        if events.is_empty() {
            println!("  NULL RESULT: zero divergence events captured. No predictive claim");
            println!("  can be made — either no qualifying task produced matched");
            println!("  pass/fail rollouts sharing a token prefix, or all matched pairs");
            println!("  diverged with no observable fork (one sequence a strict prefix");
            println!("  of the other every time).");
            return;
        }

        println!(
            "  {:<20} {:>4} {:>7} {:<14} {:>10}  {:<14} {:>10}  {:>9}  {:>4}",
            "task",
            "wt",
            "pfx_len",
            "success_tok",
            "P(succ)",
            "fail_tok",
            "P(fail)",
            "Δlogprob",
            "sign"
        );
        for e in events {
            let d = e.delta_log_prob();
            println!(
                "  {:<20} {:>4} {:>7} {:<14} {:>10.4}  {:<14} {:>10.4}  {:>9.3}  {:>4}",
                e.task,
                e.support,
                e.prefix_len,
                format!("{:?}", e.text_pass),
                e.prob_pass,
                format!("{:?}", e.text_fail),
                e.prob_fail,
                d,
                if d > 0.0 {
                    "succ"
                } else if d < 0.0 {
                    "fail"
                } else {
                    "tie"
                }
            );
        }
        println!();
        println!("  Prefix context (truncated) for each event, in the same order:");
        for (i, e) in events.iter().enumerate() {
            println!("    [{i}] {:?}", e.prefix_preview);
        }
        println!();

        let n = events.len() as u32;
        let succ_higher = events.iter().filter(|e| e.delta_log_prob() > 0.0).count();
        let fail_higher = events.iter().filter(|e| e.delta_log_prob() < 0.0).count();
        let ties = events.len() - succ_higher - fail_higher;
        let mean_delta =
            events.iter().map(|e| e.delta_log_prob()).sum::<f64>() / events.len() as f64;
        let var_delta = events
            .iter()
            .map(|e| (e.delta_log_prob() - mean_delta).powi(2))
            .sum::<f64>()
            / events.len().max(1) as f64;
        let sd_delta = var_delta.sqrt();
        let total_support: usize = events.iter().map(|e| e.support).sum();
        let p = two_sided_sign_test_p(n, succ_higher as u32);

        println!("  ── Summary ──────────────────────────────────────────────────────");
        println!(
            "  unique divergence events: {}  (from {} supporting rollout pairs across {} task(s))",
            events.len(),
            total_support,
            events
                .iter()
                .map(|e| e.task)
                .collect::<std::collections::HashSet<_>>()
                .len()
        );
        println!(
            "  P(success_token) > P(fail_token): {succ_higher}/{n}   < : {fail_higher}/{n}   tie: {ties}/{n}"
        );
        println!("  mean Δlogprob (success − fail): {mean_delta:+.4}  (sd {sd_delta:.4})");
        println!("  exact two-sided sign test p-value (H0: P=0.5): {p:.4}");
        println!();
        const ALPHA: f64 = 0.05;
        if p < ALPHA && mean_delta > 0.0 {
            println!(
                "  → POSITIVE SIGNAL: pre-divergence softmax probability predicts which \
                 branch succeeds, at p={p:.4} < {ALPHA} ({succ_higher}/{n} events favouring \
                 the eventual success token)."
            );
        } else if p < ALPHA && mean_delta < 0.0 {
            println!(
                "  → INVERTED SIGNAL: the model was systematically MORE confident in the \
                 token that led to failure (p={p:.4}). Worth a closer look, not the \
                 hypothesised direction."
            );
        } else {
            println!(
                "  → NULL RESULT: no significant relationship between pre-divergence \
                 softmax confidence and eventual success (p={p:.4}, not < {ALPHA}). At this \
                 pilot's scale ({n} events), the model's own next-token probability at the \
                 fork does not reliably predict which branch leads to the correct final \
                 answer."
            );
        }
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        // See module doc: LARQL_GPU_ROUTE=1 is verified against the
        // non-route path for this vindex (greedy parity, checked
        // manually before this harness was written — see the report
        // for the numbers) and used here purely for wall-clock budget.
        if std::env::var_os("LARQL_GPU_ROUTE").is_none() {
            std::env::set_var("LARQL_GPU_ROUTE", "1");
        }

        let vindex_path = std::env::args()
            .nth(1)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/christopherhay".into());
                PathBuf::from(home).join("chris-models/gpt-oss-20b-q4k.vindex")
            });
        if !vindex_path.is_dir() {
            return Err(format!("vindex dir not found: {}", vindex_path.display()).into());
        }

        println!("GLM-8 divergence pilot — vindex: {}", vindex_path.display());
        println!(
            "temperature={SAMPLING_TEMPERATURE} max_tokens={MAX_TOKENS} \
             prescreen_n={PRESCREEN_SAMPLES_PER_TASK} main_n={MAIN_SAMPLES_PER_TASK}"
        );
        println!();

        let mut model = load_model(&vindex_path)?;
        let backend =
            larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable")?;
        let cached = CachedLayerGraph::from_residuals(Vec::new());
        let eos = EosConfig::from_vindex_dir(&vindex_path);

        let control_tokens = encode_question(&model, candidate_tasks()[0].question)?;
        verify_sampling_is_real(&mut model, &backend, &cached, &eos, &control_tokens)?;

        let prescreen = run_prescreen(&mut model, &backend, &cached, &eos)?;
        print_prescreen_recap(&prescreen);
        let events = run_main_harness(&mut model, &backend, &cached, &eos, &prescreen)?;
        report(&events);
        Ok(())
    }

    /// One-line-per-task recap of every pre-screen measurement, kept
    /// separate from the inline printing in `run_prescreen` (which
    /// interleaves with the `.`/`x` progress dots) so the final report
    /// has a clean, scannable table of exactly what was measured before
    /// stage 2 selected on it.
    fn print_prescreen_recap(prescreen: &[PrescreenResult]) {
        println!("━━━ Pre-screen recap ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for p in prescreen {
            println!(
                "  {:<20} {}/{}  seeds={:?}  {}",
                p.task,
                p.correct,
                p.total,
                p.rollouts.iter().map(|r| r.seed).collect::<Vec<_>>(),
                if p.qualifies { "qualifies" } else { "excluded" }
            );
        }
        println!();
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    pilot::main()
}

#[cfg(not(all(feature = "gpu", target_os = "macos")))]
fn main() {
    eprintln!("glm8_divergence_pilot requires --features gpu on macOS (Metal backend).");
}
