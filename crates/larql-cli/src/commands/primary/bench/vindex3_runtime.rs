//! I/O-bound runtime for the VINDEX3 bench arm. Excluded from the
//! per-file coverage gate — every call here opens a real container and
//! executes it. Pure helpers live in `vindex3.rs`.
//!
//! The measured path is the serving path: operands prepared once
//! (`PreparedOperands`, as `larql serve` binds them), a batch prefill
//! into fresh continuation state (`prefill_prepared`, as the server's
//! `prefill_into`), then a `DecodeSession` over the same image stepping
//! one greedy token at a time. Loading and preparation are outside every
//! timer; they are reported on stderr, not in the row.

use std::path::Path;
use std::time::Instant;

use larql_inference::layer_graph::generate::{Detokenizer, EosConfig};
use larql_inference::vindex3::OpenedComponent;
use larql_vindex::format::filenames::TOKENIZER_JSON;
use larql_vindex::format::vindex3::opplan::exec::backend::{DispatchStats, PlanBackend};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
use larql_vindex::format::vindex3::opplan::exec::prefill_prepared;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::tokenizers::Tokenizer;

use super::args::BenchArgs;
use super::local::generation_fingerprint;
use super::row::BenchRow;
#[cfg(all(feature = "gpu", target_os = "macos"))]
use super::vindex3::check_reset_witness;
use super::vindex3::{
    backend_name, is_device_backend, representation_note, row_label, summarise, DeviceWindow,
    TimedRun,
};
use crate::commands::primary::vindex3_cmd::decode::argmax;
use crate::commands::primary::vindex3_cmd::prepare::{
    prepare, with_plan_backend, BackendVisitor, DEFAULT_COMPONENT,
};
use crate::commands::primary::vindex3_cmd::ExecBackend;
#[cfg(all(feature = "gpu", target_os = "macos"))]
use crate::commands::primary::vindex3_cmd::{lowered::LoweredSession, prepare::lowered_formats};
#[cfg(all(feature = "gpu", target_os = "macos"))]
use larql_compute_metal::MetalBackend;
#[cfg(all(feature = "gpu", target_os = "macos"))]
use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormats;

use crate::commands::primary::vindex3_cmd::plugins::Plugins;

type BoxErr = Box<dyn std::error::Error>;

/// Bench one V3 backend: open, prepare, pre-warm (device backends only),
/// then one timed generation.
pub(super) fn run_vindex3(
    container: &Path,
    args: &BenchArgs,
    backend: ExecBackend,
) -> Result<BenchRow, BoxErr> {
    let tokenizer_path = container.join(TOKENIZER_JSON);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("load {}: {e}", tokenizer_path.display()))?;
    let eos = EosConfig::from_vindex_dir(container);

    let opening = Instant::now();
    let plugins = Plugins::none();
    let opened = prepare(
        container,
        DEFAULT_COMPONENT,
        backend,
        RepresentationSource::Auto,
        &plugins,
    )?;
    eprintln!(
        "[bench] {}: opened {} ({}) in {:.1} s",
        row_label(backend),
        opened.model_name,
        opened.family,
        opening.elapsed().as_secs_f64()
    );
    let from_pack = opened
        .store
        .selection()
        .values()
        .filter(|s| s.stored)
        .count();
    let representation = representation_note(
        backend,
        opened.want.as_deref(),
        from_pack,
        opened.store.selection().len(),
    )?;
    let prompt_ids = encode(container, &opened.family, &tokenizer, &args.prompt)?;
    if args.verbose {
        eprintln!(
            "[bench] prompt ids ({}): {:?}",
            prompt_ids.len(),
            prompt_ids
        );
    }
    let timed = Timed {
        which: backend,
        opened: &opened,
        args,
        tokenizer: &tokenizer,
        eos: &eos,
        prompt_ids: &prompt_ids,
        representation: representation.as_deref(),
    };
    // A lowered arm does not execute through the interpreter; it has its
    // own session, timed by the same statistic.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    if let Some((formats, _)) = lowered_formats(backend) {
        return timed.lowered(formats);
    }
    with_plan_backend(backend, &plugins, timed)
}

/// The prompt as the V2 bench sends it: through the container's chat
/// template when one renders, raw otherwise.
///
/// A rendered template already carries its own special tokens (BOS
/// included), so it is encoded without the tokenizer's post-processor —
/// adding them again would double the BOS. A raw prompt gets them.
fn encode(
    container: &Path,
    family: &str,
    tokenizer: &Tokenizer,
    prompt: &str,
) -> Result<Vec<u32>, BoxErr> {
    let rendered = larql_inference::chat::render_user_prompt(container, family, prompt)
        .unwrap_or_else(|_| prompt.to_string());
    let templated = rendered != prompt;
    let encoded = tokenizer
        .encode(rendered.as_str(), !templated)
        .map_err(|e| format!("encode prompt: {e}"))?;
    let ids = encoded.get_ids().to_vec();
    if ids.is_empty() {
        return Err("prompt encodes to no tokens — nothing to condition on".into());
    }
    Ok(ids)
}

struct Timed<'a> {
    which: ExecBackend,
    opened: &'a OpenedComponent,
    args: &'a BenchArgs,
    tokenizer: &'a Tokenizer,
    eos: &'a EosConfig,
    prompt_ids: &'a [u32],
    representation: Option<&'a str>,
}

/// What one generation measured.
struct Generation {
    prefill_ms: f64,
    step_ms: Vec<f64>,
    emitted: Vec<(String, f64)>,
    device: Option<DeviceWindow>,
}

impl BackendVisitor for Timed<'_> {
    type Out = BenchRow;

    fn visit<B: PlanBackend>(self, backend: &B) -> Result<BenchRow, BoxErr> {
        let plan = &self.opened.plan;
        let loading = Instant::now();
        let ops = PreparedOperands::load(plan, &self.opened.store, backend, ExecutionSlice::Full)?;
        eprintln!(
            "[bench] {}: operands prepared in {:.1} s",
            row_label(self.which),
            loading.elapsed().as_secs_f64()
        );

        if is_device_backend(self.which) {
            // One untimed token through a fresh state: device buffers,
            // pipelines and pools, exactly as the V2 Metal pre-warm.
            self.generate(backend, &ops, 1)?;
        }

        let max_tokens = self.args.warmup + self.args.tokens;
        let started = Instant::now();
        let generation = self.generate(backend, &ops, max_tokens)?;
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        if self.args.verbose {
            let text: String = generation.emitted.iter().map(|(t, _)| t.as_str()).collect();
            eprintln!("[bench] {} generated: {text:?}", backend_name(self.which));
        }

        Ok(summarise(&TimedRun {
            backend: self.which,
            prefill_ms: generation.prefill_ms,
            step_ms: &generation.step_ms,
            warmup: self.args.warmup,
            target_tokens: self.args.tokens,
            wall_ms,
            prompt_tokens: self.prompt_ids.len(),
            device: generation.device,
            representation: self.representation,
            fingerprint: &generation_fingerprint(&generation.emitted),
        }))
    }
}

impl Timed<'_> {
    /// Prefill, then up to `max_tokens` greedy decode steps, each timed.
    ///
    /// A step is what the V2 `generate` times per token: advance the
    /// continuation by the last token, read the head, take the argmax,
    /// detokenise. It ends at the container's EOS as V2 does.
    fn generate<B: PlanBackend>(
        &self,
        backend: &B,
        ops: &PreparedOperands,
        max_tokens: usize,
    ) -> Result<Generation, BoxErr> {
        let plan = &self.opened.plan;
        let mut kv = RowKvState::default();

        let prefill_started = Instant::now();
        let out = prefill_prepared(plan, ops, self.prompt_ids, backend, &mut kv)?;
        let logits = out.logits.ok_or(NO_HEAD)?;
        let (mut next, mut value) = argmax(&logits).ok_or(NO_LOGITS)?;
        let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;

        let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv)?;
        let mut detok = Detokenizer::new(self.tokenizer);
        detok.seed(self.prompt_ids);
        let mut emitted = Vec::with_capacity(max_tokens);
        let mut step_ms = Vec::with_capacity(max_tokens);
        // Device counters are cumulative; snapshot them where the measured
        // window starts, so the device figure covers exactly the steps the
        // row's mean does.
        let mut before = None;
        for step in 0..max_tokens {
            let id = u32::try_from(next)?;
            // The logit this id was chosen on, before the step replaces it.
            let chosen = f64::from(value);
            if self.eos.eos_token_ids.contains(&id) {
                break;
            }
            if step == self.args.warmup {
                before = backend.dispatch_stats();
            }
            let started = Instant::now();
            let logits = session.step(id)?.logits.ok_or(NO_HEAD)?;
            (next, value) = argmax(&logits).ok_or(NO_LOGITS)?;
            let text = detok.push(id);
            step_ms.push(started.elapsed().as_secs_f64() * 1e3);
            let stop = self.eos.is_eos_with_tokenizer(id, &text, self.tokenizer);
            emitted.push((text, chosen));
            if stop {
                break;
            }
        }
        let device = device_window(before, backend.dispatch_stats());
        Ok(Generation {
            prefill_ms,
            step_ms,
            emitted,
            device,
        })
    }
}

/// Nanoseconds to milliseconds.
const NANOS_PER_MS: f64 = 1e6;

/// Device time between two cumulative snapshots, the device's own clock
/// included when both snapshots carry it.
fn device_window(
    before: Option<DispatchStats>,
    after: Option<DispatchStats>,
) -> Option<DeviceWindow> {
    let (before, after) = (before?, after?);
    let ms = |a: u64, b: u64| a.saturating_sub(b) as f64 / NANOS_PER_MS;
    let clock = before.device_clock.zip(after.device_clock);
    Some(DeviceWindow {
        call_ms: Some(ms(after.device_nanos, before.device_nanos)),
        submissions: after.submissions.saturating_sub(before.submissions),
        commit_to_done_ms: clock.map(|(b, a)| ms(a.commit_to_done_nanos, b.commit_to_done_nanos)),
        gpu_ms: clock.map(|(b, a)| ms(a.gpu_nanos, b.gpu_nanos)),
        device_submissions: clock.map(|(b, a)| a.submissions.saturating_sub(b.submissions)),
    })
}

/// The chosen-logit slot of a lowered step's fingerprint entry. The
/// lowered decode returns only the device argmax id; reading the logit
/// back would put a host copy on the timed path. So a lowered fingerprint
/// covers the generated text alone. It compares lowered repeats with each
/// other, never with an interpreter row, whose fingerprint carries logits.
#[cfg(all(feature = "gpu", target_os = "macos"))]
const UNREAD_LOGIT: f64 = 0.0;

/// What one lowered generation measured, plus the id sequence the reset
/// witness compares.
#[cfg(all(feature = "gpu", target_os = "macos"))]
struct LoweredGeneration {
    generation: Generation,
    /// Every id emitted, then the id the last step produced.
    ids: Vec<u32>,
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
impl Timed<'_> {
    /// Bench a lowered arm: load resident once, pre-warm with one token
    /// as every device arm is, reset to position 0, then one timed
    /// generation.
    ///
    /// The session is reset rather than rebuilt, so the weights are loaded
    /// once. The reset is checked on every run rather than assumed: the
    /// warm-up and the timed run condition on the same prompt from a fresh
    /// state, so their first ids must agree, or the row is refused.
    fn lowered(&self, formats: WeightFormats) -> Result<BenchRow, BoxErr> {
        let gpu = MetalBackend::new().ok_or("no Metal device available for a lowered arm")?;
        let max_tokens = self.args.warmup + self.args.tokens;
        let capacity = self.prompt_ids.len() + max_tokens;
        let loading = Instant::now();
        let mut keep = Vec::new();
        let mut session = LoweredSession::new(
            &gpu,
            &self.opened.plan,
            &self.opened.store,
            formats,
            capacity,
            &mut keep,
        )?;
        eprintln!(
            "[bench] {}: operands resident in {:.1} s",
            row_label(self.which),
            loading.elapsed().as_secs_f64()
        );

        let warm = self.lowered_generate(&mut session, 1)?;
        session.reset();
        let started = Instant::now();
        let timed = self.lowered_generate(&mut session, max_tokens)?;
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        session.quiesce();

        check_reset_witness(self.which, &warm.ids, &timed.ids)?;

        let generation = timed.generation;
        if self.args.verbose {
            let text: String = generation.emitted.iter().map(|(t, _)| t.as_str()).collect();
            eprintln!("[bench] {} generated: {text:?}", backend_name(self.which));
        }
        Ok(summarise(&TimedRun {
            backend: self.which,
            prefill_ms: generation.prefill_ms,
            step_ms: &generation.step_ms,
            warmup: self.args.warmup,
            target_tokens: self.args.tokens,
            wall_ms,
            prompt_tokens: self.prompt_ids.len(),
            device: generation.device,
            representation: self.representation,
            fingerprint: &generation_fingerprint(&generation.emitted),
        }))
    }

    /// The prompt one position per step, as the lowered path executes
    /// it; then up to `max_tokens` greedy steps chained from the device
    /// argmax, each timed exactly as the interpreter's are.
    fn lowered_generate(
        &self,
        session: &mut LoweredSession<'_>,
        max_tokens: usize,
    ) -> Result<LoweredGeneration, BoxErr> {
        let prefill_started = Instant::now();
        let mut next = None;
        for &token in self.prompt_ids {
            next = session.step(token)?;
        }
        let mut next = next.ok_or(NO_HEAD)?;
        let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;

        session.begin_decode();
        let mut detok = Detokenizer::new(self.tokenizer);
        detok.seed(self.prompt_ids);
        let mut emitted = Vec::with_capacity(max_tokens);
        let mut step_ms = Vec::with_capacity(max_tokens);
        let mut ids = Vec::with_capacity(max_tokens + 1);
        // The window starts at the first measured step, as the
        // interpreter's device snapshot does.
        let mut submissions_before = None;
        let mut gpu_ms = 0.0;
        for step in 0..max_tokens {
            let id = next;
            if self.eos.eos_token_ids.contains(&id) {
                break;
            }
            if step == self.args.warmup {
                submissions_before = Some(session.submissions());
            }
            let started = Instant::now();
            next = session.step(id)?.ok_or(NO_HEAD)?;
            let text = detok.push(id);
            step_ms.push(started.elapsed().as_secs_f64() * 1e3);
            if step >= self.args.warmup {
                gpu_ms += session.last_gpu_ms();
            }
            let stop = self.eos.is_eos_with_tokenizer(id, &text, self.tokenizer);
            emitted.push((text, UNREAD_LOGIT));
            ids.push(id);
            if stop {
                break;
            }
        }
        ids.push(next);
        // The lowered host never blocks inside a device call, so there is
        // no call wall to report; the GPU span is what it measures.
        let device = submissions_before.map(|before| DeviceWindow {
            submissions: session.submissions().saturating_sub(before),
            gpu_ms: Some(gpu_ms),
            ..DeviceWindow::default()
        });
        Ok(LoweredGeneration {
            generation: Generation {
                prefill_ms,
                step_ms,
                emitted,
                device,
            },
            ids,
        })
    }
}

const NO_HEAD: &str = "plan carries no output head — cannot generate";
const NO_LOGITS: &str = "output head produced no logits";
