//! `larql run <container> [prompt]` — a VINDEX3 container executes its
//! own program, text in and text out.
//!
//! The container carries its tokenizer (the capability snapshot puts it
//! there at encode), so this arm has the exact shape of the BitNet one
//! beside it in `run_cmd`: load the tokenizer, encode the prompt, decode
//! greedily, stream the decoded text. Everything between encode and
//! decode belongs to `vindex3_cmd` — the container is prepared by the
//! one authority on preparing containers and executed by the same
//! interpreter `larql vindex3 exec` reports on — so an id produced here
//! is the id that verb would produce for the same ids in. `--emit-ids`
//! prints both sides, which makes a run an oracle for it.
//!
//! What this arm does *not* do is as deliberate as what it does. No chat
//! template: the prompt is encoded raw, exactly as the BitNet arm does
//! (a template is a follow-up on both). No sampler: greedy, so a run
//! doubles as a fixture. And no flag it cannot honour is accepted
//! silently — the dense path's engine, composition, expert and image
//! flags are refused by name rather than dropped.
//!
//! Weights are loaded once, at model lifetime; every prompt — the one on
//! the command line, or each line of the chat loop — gets a brand-new
//! continuation state over them, so nothing from one turn can reach the
//! next.
//!
//! The model is named by the container. `index.model` is the identity
//! authority; the directory name is an explicit fallback for a container
//! encoded nameless, and [`resolved_display_name`] is the only place
//! that fallback is decided — the banner and the verbose report both
//! read it, so neither can grow a second derivation.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

use larql_inference::layer_graph::generate::{Detokenizer, EosConfig};
use larql_inference::vindex3::OpenedComponent;
use larql_vindex::format::filenames::TOKENIZER_JSON;
use larql_vindex::format::generation::{detect_generation, ContainerGeneration};
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;
use larql_vindex::tokenizers::Tokenizer;

use super::run_cmd::{KvCacheKind, RunArgs};
use super::vindex3_cmd::decode::{greedy_decode, DecodeReport, Flow};
use super::vindex3_cmd::prepare::{
    prepare, with_plan_backend, BackendVisitor, DEFAULT_COMPONENT, ENGINE_PREFIX,
};
use super::vindex3_cmd::ExecBackend;

#[cfg(test)]
mod tests;

type BoxErr = Box<dyn std::error::Error>;

/// The argmax alone — `--top`'s default, and the only prediction width
/// this arm produces.
const SINGLE_PREDICTION: usize = 1;

/// The chat loop's prompt, written to the status stream so stdout stays
/// the model's.
const CHAT_PROMPT: &str = "> ";

/// What a container that declares no name is called, when even its
/// directory has no printable name.
const NAMELESS_CONTAINER: &str = "container";

/// The name a run shows for its model.
///
/// The container's own declaration (`index.model`) is the identity
/// authority and wins whenever it is non-empty. The directory name is
/// the explicit fallback for a container encoded nameless — the only
/// path on which filesystem identity may ever be shown as the model's.
pub(super) fn resolved_display_name(declared: &str, container: &Path) -> String {
    if !declared.is_empty() {
        return declared.to_string();
    }
    container
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(NAMELESS_CONTAINER)
        .to_string()
}

/// Whether `dir` is a VINDEX3 container.
///
/// Anything that is not — a VINDEX2 vindex, a checkpoint directory, an
/// unreadable index — answers `false`, so the dense path, which owns
/// those errors, is the one to raise them.
pub(crate) fn is_vindex3_container(dir: &Path) -> bool {
    matches!(detect_generation(dir), Ok(ContainerGeneration::V3))
}

/// Serve a VINDEX3 container: one prompt to stdout, or the chat loop.
pub fn run(container: &Path, args: &RunArgs) -> Result<(), BoxErr> {
    run_to(
        container,
        args,
        &mut io::stdin().lock(),
        &mut io::stdout().lock(),
        &mut io::stderr(),
    )
}

/// [`run`] with its streams injected — the chat loop reads `input`,
/// every generated character goes to `out`, and everything that is
/// *about* the run (banner, prompt, ids, timings, errors) goes to
/// `status`, which is stderr in the binary.
pub(super) fn run_to(
    container: &Path,
    args: &RunArgs,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    status: &mut dyn Write,
) -> Result<(), BoxErr> {
    refuse_inapplicable_flags(args)?;
    let backend = select_backend(args.metal)?;
    let tokenizer_path = container.join(TOKENIZER_JSON);
    if !tokenizer_path.is_file() {
        return Err(format!(
            "{} carries no {TOKENIZER_JSON}: it was encoded from a checkpoint without one, so \
             text cannot be tokenised here (`larql vindex3 exec --tokens` runs it from ids)",
            container.display()
        )
        .into());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("load {}: {e}", tokenizer_path.display()))?;
    let eos = EosConfig::from_vindex_dir(container);
    let prepared = prepare(
        container,
        DEFAULT_COMPONENT,
        backend,
        RepresentationSource::Auto,
    )?;
    with_plan_backend(
        backend,
        Runner {
            container,
            args,
            prepared: &prepared,
            tokenizer: &tokenizer,
            eos: &eos,
            input,
            out,
            status,
        },
    )
}

/// The dense path's flags this arm cannot honour, refused by name.
///
/// The container's own program runs through the VINDEX3 interpreter with
/// its own continuation state; the engine, composition, expert and image
/// flags all describe the dense VINDEX2 engine. Refused together, so one
/// message names every flag that has to go.
fn refuse_inapplicable_flags(args: &RunArgs) -> Result<(), BoxErr> {
    let set: Vec<&str> = [
        ("--top", args.top != SINGLE_PREDICTION),
        ("--kv-cache", args.kv_cache != KvCacheKind::Standard),
        ("--engine", args.engine.is_some()),
        ("--ffn", args.ffn.is_some()),
        ("--routed-from", args.routed_from.is_some()),
        ("--experts", args.experts),
        ("--experts-dir", args.experts_dir.is_some()),
        ("--ops", !args.ops.is_empty()),
        ("--constrained", args.constrained),
        ("--moe-shards", args.moe_shards.is_some()),
        ("--moe-units-manifest", args.moe_units_manifest.is_some()),
        ("--image", !args.image.is_empty()),
        ("--mm-weights", args.mm_weights.is_some()),
    ]
    .into_iter()
    .filter_map(|(flag, given)| given.then_some(flag))
    .collect();
    if set.is_empty() {
        return Ok(());
    }
    Err(format!(
        "a VINDEX3 container runs its own program through the VINDEX3 interpreter; these \
         flags describe the dense VINDEX2 engine and are not honoured here: {}",
        set.join(", ")
    )
    .into())
}

/// `--metal` names the Metal realisation; otherwise the `larql-compute`
/// CPU kernels. Split into two whole definitions rather than a `cfg`
/// block inside one, for the reason `run_cmd::generate_routed_metal`
/// documents: the `gpu` feature compiles everywhere, the Metal crate
/// only on macOS, and Cargo cannot express the conjunction.
#[cfg(all(feature = "gpu", target_os = "macos"))]
fn select_backend(metal: bool) -> Result<ExecBackend, BoxErr> {
    Ok(if metal {
        ExecBackend::Metal
    } else {
        ExecBackend::Production
    })
}

#[cfg(not(all(feature = "gpu", target_os = "macos")))]
fn select_backend(metal: bool) -> Result<ExecBackend, BoxErr> {
    if metal {
        return Err("--metal needs the `gpu` feature on macOS; this build has neither".into());
    }
    Ok(ExecBackend::Production)
}

/// The run, once the backend is a concrete type.
struct Runner<'a> {
    container: &'a Path,
    args: &'a RunArgs,
    prepared: &'a OpenedComponent,
    tokenizer: &'a Tokenizer,
    eos: &'a EosConfig,
    input: &'a mut dyn BufRead,
    out: &'a mut dyn Write,
    status: &'a mut dyn Write,
}

impl BackendVisitor for Runner<'_> {
    type Out = ();

    fn visit<B: PlanBackend>(self, backend: &B) -> Result<(), BoxErr> {
        let loading = Instant::now();
        // The expensive, immutable half — once, for every prompt.
        let ops = PreparedOperands::load(
            &self.prepared.plan,
            &self.prepared.store,
            backend,
            ExecutionSlice::Full,
        )?;
        let engine = format!("{ENGINE_PREFIX}-{}", backend.name());
        let identity = resolved_display_name(&self.prepared.model_name, self.container);
        if self.args.verbose {
            writeln!(
                self.status,
                "[{engine}] {identity} ({}): weights resident in {:.1} s",
                self.prepared.family,
                loading.elapsed().as_secs_f64()
            )?;
        }
        let model = ResidentModel {
            plan: &self.prepared.plan,
            ops: &ops,
            backend,
            tokenizer: self.tokenizer,
            eos: self.eos,
            engine: &engine,
            args: self.args,
        };
        if let Some(prompt) = self.args.prompt.as_deref() {
            return model.generate(prompt, self.out, self.status);
        }
        chat_loop(&identity, &model, self.input, self.out, self.status)
    }
}

/// One loaded model, ready to answer any number of prompts.
struct ResidentModel<'a, B: PlanBackend> {
    plan: &'a ComponentOpPlan,
    ops: &'a PreparedOperands,
    backend: &'a B,
    tokenizer: &'a Tokenizer,
    eos: &'a EosConfig,
    engine: &'a str,
    args: &'a RunArgs,
}

impl<B: PlanBackend> ResidentModel<'_, B> {
    /// Encode `prompt`, decode up to `--max-tokens` greedily, and stream
    /// the text to `out` as it is produced. Ends at the first EOS.
    fn generate(
        &self,
        prompt: &str,
        out: &mut dyn Write,
        status: &mut dyn Write,
    ) -> Result<(), BoxErr> {
        let encoded = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("encode prompt: {e}"))?;
        let ids = encoded.get_ids();
        // A brand-new continuation state per prompt. Not a reset — a
        // replacement, so there is nothing that *could* carry over.
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(self.plan, self.ops, self.backend, &mut kv)?;
        let mut detok = Detokenizer::new(self.tokenizer);
        detok.seed(ids);
        let decoded = greedy_decode(&mut session, ids, self.args.max_tokens, &mut |id, _| {
            // An EOS id halts before it is decoded at all; a stop
            // string halts on its surface form, re-decoded with
            // specials kept when the clean delta is empty.
            if self.eos.eos_token_ids.contains(&id) {
                return Ok(Flow::Halt);
            }
            let delta = detok.push(id);
            if self.eos.is_eos_with_tokenizer(id, &delta, self.tokenizer) {
                return Ok(Flow::Halt);
            }
            out.write_all(delta.as_bytes())?;
            out.flush()?;
            Ok(Flow::Continue)
        })?;
        writeln!(out)?;
        if self.args.emit_ids {
            writeln!(status, "[{}] prompt ids: {:?}", self.engine, ids)?;
            writeln!(
                status,
                "[{}] generated ids: {:?}",
                self.engine, decoded.generated
            )?;
        }
        if self.args.verbose {
            writeln!(
                status,
                "[{}] {} prompt tokens in {:.2} s, {} generated",
                self.engine,
                ids.len(),
                decoded.prompt_seconds,
                decoded.generated.len(),
            )?;
            if let Some(report) = DecodeReport::from_steps(&decoded.step_seconds) {
                writeln!(
                    status,
                    "[{}] decode {:.0} ms/token ({:.2} tok/s), steady {:.0} ms/token",
                    self.engine,
                    report.mean_seconds_per_token * 1e3,
                    report.mean_seconds_per_token.recip(),
                    report.steady_seconds_per_token * 1e3,
                )?;
            }
        }
        Ok(())
    }
}

/// Single-turn chat: one line in, one generation out, until EOF. Each
/// line is its own prompt — no history, no template — over the one
/// resident model.
fn chat_loop<B: PlanBackend>(
    identity: &str,
    model: &ResidentModel<'_, B>,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    status: &mut dyn Write,
) -> Result<(), BoxErr> {
    writeln!(
        status,
        "larql chat ({}) — {identity} (Ctrl-D to exit)",
        model.engine
    )?;
    loop {
        write!(status, "{CHAT_PROMPT}")?;
        status.flush()?;
        let mut line = String::new();
        match input.read_line(&mut line) {
            Ok(0) => {
                writeln!(status)?;
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(Box::new(e)),
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if let Err(e) = model.generate(prompt, out, status) {
            writeln!(status, "Error: {e}")?;
        }
    }
}
