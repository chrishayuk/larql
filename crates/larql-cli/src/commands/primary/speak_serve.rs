//! `larql speak-serve` — the speech model, resident.
//!
//! `run --speak` pays the whole load bill for one utterance: safetensors
//! load, the Q4_K quantisation pass, tokenizer — tens of seconds before a
//! single frame exists, against generation that finishes in a few. This
//! subcommand pays it once and then serves utterances against the hot
//! model, which is what [`MossSpeech`] was shaped for: the loaded model
//! borrowed as one unit, with a fresh [`MossSession`] per utterance.
//!
//! It is the process-level step towards `docs/tts-funnel.md` §5's
//! "keep the speech machine hot" — not the realtime runtime itself
//! (no ring buffer, no audio callback), and the codec stays external
//! (§6). Audio tokens go to a token file exactly as `run --speak`
//! writes them, so the same codec consumes both.
//!
//! ## Protocol
//!
//! One request is a block of `key: value` header lines terminated by a
//! blank line (or EOF). `text:` is the only required key:
//!
//! ```text
//! text: Good evening Sam.
//! voice: /path/to/jarvis.tokens
//! tokens: /path/to/out.tokens.txt
//! stream: /path/to/frames.fifo
//! seed: 5
//! max-frames: 200
//! reply: /path/to/reply.fifo
//!
//! ```
//!
//! Every key except `text:` is optional and falls back to the flag given
//! at startup, though a request needs `tokens:` or `stream:` (or both) to
//! have somewhere to put its frames. `voice: -` forces the unconditioned
//! voice for one request even when a default voice is loaded. The reply is
//! a single line — `ok frames=N audio=Ns gen=Ns first_frame=Nms` or
//! `err <message>` — written to `reply:` when given, else to stdout.
//!
//! `stream:` is the live path: each frame's ids are written to that FIFO
//! and flushed *the moment the frame exists*, so a player can decode and
//! start sounding while the rest of the utterance is still generating.
//! Closing the FIFO is the end-of-utterance signal, so there is no in-band
//! terminator to mistake for a frame. `tokens:` may be given alongside it
//! to keep the file artifact as well.
//!
//! With `--listen <fifo>` the server reads requests from a FIFO, one
//! writer at a time, reopening after each client closes; without it,
//! requests come from stdin. Requests are served strictly in sequence:
//! one model, one utterance at a time. A failing request is reported and
//! the server stays up.
//!
//! ## Portability
//!
//! Requests on stdin and `tokens:` file output work anywhere, so the
//! amortisation this subcommand exists for is available on every
//! platform. The FIFO paths are Unix-only and refuse with a clear
//! message elsewhere rather than failing obscurely: `--listen` needs
//! `mkfifo`, and `reply:` needs a non-blocking write-open to survive a
//! dead client (see [`respond`]). `stream:` opens an existing path for
//! writing and depends on a reader being on the far end, so in practice
//! it is a FIFO too.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Args;
use larql_compute::ffn::q4k_weight::Q4kFfn;
use larql_compute::{KvIndex, PackedAttnIndex};
use larql_inference::ffn::{FfnBackend, WeightFfn};
use larql_inference::speech::moss_prompt::build_prompt;
use larql_inference::speech::moss_realtime::generate_frames_streaming;
use larql_inference::speech::moss_sampling::{DecodeMode, MossSampling};
use larql_inference::speech::stream_timing::SECONDS_PER_FRAME;
use larql_inference::tokenizer::load_tokenizer;
use larql_models::loading::safetensors::load_model_dir;
use larql_models::speech::moss_tts_realtime::{
    depth_transformer_model, load_moss_tts_aux_from_safetensors, MossTtsRealtimeConfig,
};
use ndarray::Array2;

use super::run_cmd_speak::{read_token_rows, write_token_rows};

type BoxErr = Box<dyn std::error::Error>;

/// Sentinel for `voice:` meaning "no voice reference this request".
const UNCONDITIONED: &str = "-";

/// How long to wait for a client to open its reply FIFO before deciding it
/// is gone — see [`respond`]. Short on purpose: a client opens its read end
/// immediately after writing its request, so it is already waiting well
/// before generation finishes. If nobody is there seconds after we have a
/// reply, nobody is coming, and every extra second is one the server spends
/// unavailable to everyone else.
#[cfg(unix)]
const REPLY_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Args, Debug)]
pub struct SpeakServeArgs {
    /// The speech checkpoint's safetensors directory (MOSS-TTS-Realtime).
    /// Vindex residency is TTS funnel step 6, so this is a checkpoint
    /// directory for now, exactly as `run --speak` takes it.
    pub model: String,

    /// FIFO to read requests from. Created if missing. Without it,
    /// requests are read from stdin.
    #[arg(long, value_name = "PATH")]
    pub listen: Option<PathBuf>,

    /// Default voice reference (a token-rows file). Per-request `voice:`
    /// overrides it; `voice: -` drops to the unconditioned voice.
    #[arg(long, value_name = "TOKENS")]
    pub voice: Option<PathBuf>,

    /// Default frame cap (12.5 frames per second of audio).
    #[arg(long, default_value = "1500")]
    pub max_frames: usize,

    /// Default RNG seed for sampled mode.
    #[arg(long, default_value = "0")]
    pub seed: u64,

    /// Greedy decoding (parity/debug). Does not terminate reliably on
    /// novel text — the default sampled mode is what you want.
    #[arg(long)]
    pub greedy: bool,

    /// Quantise FFNs and attention projections to Q4_K at startup. Paid
    /// once for the life of the server rather than once per utterance.
    #[arg(long)]
    pub q4: bool,

    /// Route the backbone through the default backend composition
    /// (Metal + CPU fallback when built with the `gpu` feature).
    #[arg(long)]
    pub metal: bool,
}

/// One parsed request. Every field but `text` falls back to the server
/// default.
#[derive(Default)]
struct Request {
    text: Option<String>,
    voice: Option<String>,
    tokens: Option<PathBuf>,
    stream: Option<PathBuf>,
    seed: Option<u64>,
    max_frames: Option<usize>,
    greedy: Option<bool>,
    reply: Option<PathBuf>,
}

impl Request {
    /// Absorb one `key: value` line. Unknown keys are an error rather
    /// than a silent no-op: a typo'd `voise:` would otherwise synthesise
    /// the wrong voice and look like a model bug.
    fn absorb(&mut self, line: &str) -> Result<(), BoxErr> {
        let (key, value) = line
            .split_once(':')
            .ok_or_else(|| format!("malformed header line (want `key: value`): {line:?}"))?;
        let value = value.trim();
        match key.trim() {
            "text" => self.text = Some(value.to_string()),
            "voice" => self.voice = Some(value.to_string()),
            "tokens" => self.tokens = Some(PathBuf::from(value)),
            "stream" => self.stream = Some(PathBuf::from(value)),
            "seed" => self.seed = Some(value.parse()?),
            "max-frames" => self.max_frames = Some(value.parse()?),
            "greedy" => self.greedy = Some(matches!(value, "1" | "true" | "yes")),
            "reply" => self.reply = Some(PathBuf::from(value)),
            other => return Err(format!("unknown request key {other:?}").into()),
        }
        Ok(())
    }
}

/// Everything loaded once, borrowed by every request.
struct Resident<'a> {
    config: MossTtsRealtimeConfig,
    tokenizer: tokenizers::Tokenizer,
    backend: Box<dyn larql_inference::EngineBackend>,
    weights: &'a larql_models::ModelWeights,
    audio_tables: Vec<Array2<f32>>,
    depth: larql_models::speech::moss_tts_realtime::DepthTransformerModel,
}

pub fn run(args: SpeakServeArgs) -> Result<(), BoxErr> {
    let model_dir = PathBuf::from(&args.model);
    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        return Err(format!(
            "speak-serve expects the speech checkpoint's safetensors directory \
             (no config.json under {model_dir:?}); vindex residency is funnel step 6"
        )
        .into());
    }

    // ── Load: the bill this whole subcommand exists to pay once ──
    let started = Instant::now();
    let weights = load_model_dir(&model_dir)?;
    if weights.arch.family() != "moss_tts_realtime" {
        return Err(format!(
            "speak-serve supports moss_tts_realtime models; this is {:?}",
            weights.arch.family()
        )
        .into());
    }
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let config = MossTtsRealtimeConfig::from_config_json(&config_json)?;
    let aux =
        load_moss_tts_aux_from_safetensors(&model_dir, weights.arch.as_ref(), config.clone())?;
    let depth = depth_transformer_model(aux.local, &config)?;
    let tokenizer = load_tokenizer(&model_dir)?;
    let load_seconds = started.elapsed().as_secs_f64();

    let backend = if args.metal {
        larql_inference::default_engine_backend()
    } else {
        larql_inference::cpu_engine_backend()
    };
    let resident = Resident {
        config,
        tokenizer,
        backend,
        weights: &weights,
        audio_tables: aux.audio_embed_tables,
        depth,
    };

    // Dense and (optionally) Q4 backends live for the whole server, so
    // the quantisation pass is amortised over every utterance served.
    let backbone_dense = WeightFfn {
        weights: resident.weights,
    };
    let depth_dense = WeightFfn {
        weights: &resident.depth.weights,
    };
    let (ffn, depth_ffn): (&dyn FfnBackend, &dyn FfnBackend);
    let (backbone_q4, depth_q4);
    let (backbone_attn_q4, depth_attn_q4);
    let (backbone_attn, depth_attn): (Option<&dyn KvIndex>, Option<&dyn KvIndex>);
    let mut quant_seconds = None;
    if args.q4 {
        let quant_start = Instant::now();
        backbone_q4 = Q4kFfn::quantize_from(resident.weights)?;
        depth_q4 = Q4kFfn::quantize_from(&resident.depth.weights)?;
        backbone_attn_q4 = PackedAttnIndex::quantize_from(resident.weights)?;
        depth_attn_q4 = PackedAttnIndex::quantize_from(&resident.depth.weights)?;
        quant_seconds = Some(quant_start.elapsed().as_secs_f64());
        ffn = &backbone_q4;
        depth_ffn = &depth_q4;
        backbone_attn = Some(&backbone_attn_q4);
        depth_attn = Some(&depth_attn_q4);
    } else {
        ffn = &backbone_dense;
        depth_ffn = &depth_dense;
        backbone_attn = None;
        depth_attn = None;
    }

    eprintln!(
        "speak-serve ready: model load {load_seconds:.1}s{}, {} — paid once",
        match quant_seconds {
            Some(seconds) => format!(", Q4_K quantisation {seconds:.1}s"),
            None => String::new(),
        },
        match &args.voice {
            Some(path) => format!("default voice {}", path.display()),
            None => "unconditioned by default".to_string(),
        },
    );

    let serve = |request: Request| -> Result<String, BoxErr> {
        serve_one(
            &resident,
            &args,
            ffn,
            depth_ffn,
            backbone_attn,
            depth_attn,
            request,
        )
    };

    match &args.listen {
        Some(fifo) => {
            ensure_fifo(fifo)?;
            eprintln!("listening on {}", fifo.display());
            // Each client opens, writes one request block, closes. The
            // read side sees EOF and reopens for the next one; opening a
            // FIFO for reading blocks until a writer arrives, so this
            // loop idles without spinning.
            loop {
                let file = std::fs::File::open(fifo)?;
                drain(BufReader::new(file), &serve);
            }
        }
        None => {
            eprintln!("reading requests from stdin");
            drain(BufReader::new(std::io::stdin()), &serve);
            Ok(())
        }
    }
}

/// Read request blocks until EOF, serving each and reporting failures
/// without tearing the server down.
fn drain<R: BufRead>(reader: R, serve: &dyn Fn(Request) -> Result<String, BoxErr>) {
    let mut request = Request::default();
    let mut pending = false;
    let mut malformed: Option<String> = None;

    let settle = |request: &mut Request, malformed: &mut Option<String>| {
        let taken = std::mem::take(request);
        let reply_to = taken.reply.clone();
        let reply = match malformed.take() {
            Some(error) => format!("err {error}"),
            None => match serve(taken) {
                Ok(line) => line,
                Err(error) => format!("err {error}"),
            },
        };
        if reply.starts_with("err ") {
            eprintln!("speak-serve: {reply}");
        }
        if let Err(error) = respond(reply_to.as_deref(), &reply) {
            eprintln!("speak-serve: could not deliver reply: {error}");
        }
    };

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("speak-serve: read failed: {error}");
                return;
            }
        };
        if line.trim().is_empty() {
            if pending {
                settle(&mut request, &mut malformed);
                pending = false;
            }
            continue;
        }
        pending = true;
        // Keep parsing the block after a bad line so `reply:` is still
        // picked up — the client is waiting on that FIFO either way.
        if let Err(error) = request.absorb(&line) {
            if malformed.is_none() {
                malformed = Some(error.to_string());
            }
        }
    }
    if pending {
        settle(&mut request, &mut malformed);
    }
}

/// Synthesise one utterance against the resident model.
fn serve_one(
    resident: &Resident<'_>,
    args: &SpeakServeArgs,
    ffn: &dyn FfnBackend,
    depth_ffn: &dyn FfnBackend,
    backbone_attn: Option<&dyn KvIndex>,
    depth_attn: Option<&dyn KvIndex>,
    request: Request,
) -> Result<String, BoxErr> {
    let text = request.text.as_deref().unwrap_or_default();
    if text.trim().is_empty() {
        return Err("request has no text:".into());
    }
    let tokens_path = request.tokens.clone();
    if tokens_path.is_none() && request.stream.is_none() {
        return Err("request needs tokens: (a file) or stream: (a live FIFO), or both".into());
    }

    // `voice: -` is an explicit "no reference", distinct from an absent
    // key (which inherits the server default).
    let voice: Option<PathBuf> = match request.voice.as_deref() {
        Some(UNCONDITIONED) => None,
        Some(path) => Some(PathBuf::from(path)),
        None => args.voice.clone(),
    };
    let reference = voice
        .as_deref()
        .map(|path| read_token_rows(path, resident.config.rvq))
        .transpose()?;
    let prompt = build_prompt(
        &resident.tokenizer,
        &resident.config,
        reference.as_ref(),
        text,
    )?;

    let mode = if request.greedy.unwrap_or(args.greedy) {
        DecodeMode::Greedy
    } else {
        DecodeMode::Sampled(MossSampling::default())
    };
    // Live mode: the player is already waiting on the far end of this FIFO,
    // so opening blocks until it is there. Frames then go over as they are
    // produced instead of at end of utterance, which is the whole point —
    // playback can start while the rest is still being generated.
    let mut live = match request.stream.as_deref() {
        Some(path) => Some(std::io::BufWriter::new(
            std::fs::OpenOptions::new().write(true).open(path)?,
        )),
        None => None,
    };
    let mut live_error: Option<std::io::Error> = None;
    let mut first_frame_seconds: Option<f64> = None;

    let generation_start = Instant::now();
    let generation = generate_frames_streaming(
        resident.backend.as_ref(),
        resident.weights,
        ffn,
        &resident.audio_tables,
        &resident.depth,
        depth_ffn,
        &resident.config,
        &prompt.prefill_matrix,
        &prompt.text_queue,
        prompt.text_pad_id,
        request.max_frames.unwrap_or(args.max_frames),
        mode,
        request.seed.unwrap_or(args.seed),
        backbone_attn,
        depth_attn,
        |_index, codes| {
            if first_frame_seconds.is_none() {
                first_frame_seconds = Some(generation_start.elapsed().as_secs_f64());
            }
            if live_error.is_none() {
                if let Some(out) = live.as_mut() {
                    if let Err(error) = write_frame_line(out, codes) {
                        live_error = Some(error);
                    }
                }
            }
        },
    )?;
    let generate_seconds = generation_start.elapsed().as_secs_f64();
    // Closing the writer is the player's end-of-utterance signal: it reads
    // to EOF, so there is no in-band terminator to confuse with a frame.
    drop(live);
    if let Some(error) = live_error {
        return Err(format!("live stream write failed: {error}").into());
    }

    let emitted = generation.emitted();
    if emitted.is_empty() {
        return Err("generation produced no frames before EOS".into());
    }
    if let Some(path) = &tokens_path {
        write_token_rows(path, emitted)?;
    }

    // Steady-state rate is the number that decides whether streaming
    // playback could keep up, and it is not the end-to-end rate: prefill
    // is a fixed cost that a short utterance amortises badly. Report both,
    // plus the per-frame split, so the resident path stays as measurable
    // as `run --speak`'s benchmark table.
    let audio_seconds = emitted.len() as f64 * SECONDS_PER_FRAME;
    let steady_seconds = (generate_seconds - generation.prefill_seconds).max(f64::MIN_POSITIVE);
    let steady_frames = generation.frames.len().saturating_sub(1);
    eprintln!(
        "spoke {} frames ({audio_seconds:.2}s audio) in {generate_seconds:.2}s \
         ({:.2}x realtime end-to-end): {}",
        emitted.len(),
        audio_seconds / generate_seconds,
        truncate(text, 60),
    );
    eprintln!(
        "  first frame {:.0} ms | prefill {:.2}s ({} rows) | steady {:.0} ms/frame \
         ({:.2}x realtime){}",
        first_frame_seconds.unwrap_or(0.0) * 1000.0,
        generation.prefill_seconds,
        prompt.prefill_matrix.nrows(),
        steady_seconds / steady_frames.max(1) as f64 * 1000.0,
        steady_frames as f64 * SECONDS_PER_FRAME / steady_seconds,
        stage_split(&generation.stage_timings),
    );
    Ok(format!(
        "ok frames={} audio={audio_seconds:.2}s gen={generate_seconds:.2}s \
         first_frame={:.0}ms{}",
        emitted.len(),
        first_frame_seconds.unwrap_or(0.0) * 1000.0,
        match &tokens_path {
            Some(path) => format!(" tokens={}", path.display()),
            None => String::new(),
        },
    ))
}

/// Deliver one reply line to the client's FIFO, or stdout when the
/// request named none.
fn respond(reply_to: Option<&Path>, reply: &str) -> Result<(), BoxErr> {
    match reply_to {
        Some(path) => respond_to_fifo(path, reply),
        None => {
            let mut out = std::io::stdout();
            writeln!(out, "{reply}")?;
            out.flush()?;
            Ok(())
        }
    }
}

/// Write one reply line to a client's reply FIFO.
///
/// The write side is opened `O_NONBLOCK` and retried to a deadline rather
/// than opened blocking. A blocking open on a FIFO with no reader waits
/// forever, so a client that died between sending its request and reading
/// its reply would wedge the server permanently — no CPU, no error, every
/// later request stuck behind a pipe nobody drains. That happened, and it
/// took the whole speech path down until a restart. Now a dead client
/// costs one dropped reply and a log line.
#[cfg(unix)]
fn respond_to_fifo(path: &Path, reply: &str) -> Result<(), BoxErr> {
    use std::os::unix::fs::OpenOptionsExt;

    let deadline = Instant::now() + REPLY_WAIT;
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
        {
            Ok(mut fifo) => {
                writeln!(fifo, "{reply}")?;
                fifo.flush()?;
                return Ok(());
            }
            // ENXIO is precisely "FIFO opened for writing with no
            // reader" — the client may simply not have reached its
            // read yet, so retry before giving up on it.
            Err(error) if error.raw_os_error() == Some(libc::ENXIO) => {
                if Instant::now() >= deadline {
                    return Err(format!(
                        "no reader on {} after {:.0}s — client gone, reply dropped",
                        path.display(),
                        REPLY_WAIT.as_secs_f64()
                    )
                    .into());
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

/// Reply FIFOs need the non-blocking write-open above to be safe against a
/// dead client, which is a Unix guarantee. Rather than emulate it, say so:
/// the reply is still available on stdout.
#[cfg(not(unix))]
fn respond_to_fifo(path: &Path, _reply: &str) -> Result<(), BoxErr> {
    Err(format!(
        "reply: {} — reply FIFOs are unix-only; omit reply: to take the \
         one-line reply on stdout",
        path.display()
    )
    .into())
}

/// Create the request FIFO if it is not already there, and refuse to run
/// against a path that exists as something else.
#[cfg(unix)]
fn ensure_fifo(path: &Path) -> Result<(), BoxErr> {
    use std::os::unix::fs::FileTypeExt;

    if path.exists() {
        if !std::fs::metadata(path)?.file_type().is_fifo() {
            return Err(format!("{} exists and is not a FIFO", path.display()).into());
        }
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    make_fifo(path)
}

/// `mkfifo(3)` directly rather than shelling out to `mkfifo(1)` — `libc`
/// is already a dependency, and this drops a silent dependency on the
/// binary being on `PATH`.
#[cfg(unix)]
fn make_fifo(path: &Path) -> Result<(), BoxErr> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| format!("{} contains an interior NUL", path.display()))?;
    // 0o666 & ~umask, matching mkfifo(1)'s default.
    if unsafe { libc::mkfifo(c_path.as_ptr(), 0o666) } != 0 {
        return Err(format!(
            "mkfifo {} failed: {}",
            path.display(),
            std::io::Error::last_os_error()
        )
        .into());
    }
    Ok(())
}

/// `--listen` is a FIFO server; the stdin path is the portable one.
#[cfg(not(unix))]
fn ensure_fifo(path: &Path) -> Result<(), BoxErr> {
    Err(format!(
        "--listen {} needs a unix FIFO; on this platform pipe request \
         blocks into stdin instead",
        path.display()
    )
    .into())
}

/// One frame as a token-row line, flushed on the spot — a frame sitting in
/// a buffer is silence the player could have been decoding.
fn write_frame_line(out: &mut impl Write, codes: &[u32]) -> std::io::Result<()> {
    let row: Vec<String> = codes.iter().map(u32::to_string).collect();
    writeln!(out, "{}", row.join(" "))?;
    out.flush()
}

/// ` | backbone p50 N ms | depth p50 N ms` — where a frame's time goes.
/// Empty when no stages were recorded.
fn stage_split(stages: &[larql_inference::speech::moss_realtime::FrameStages]) -> String {
    if stages.is_empty() {
        return String::new();
    }
    let median = |mut values: Vec<f64>| -> Option<f64> {
        values.retain(|&seconds| seconds > 0.0);
        if values.is_empty() {
            return None;
        }
        values.sort_by(|a, b| a.partial_cmp(b).expect("finite frame times"));
        Some(values[values.len() / 2] * 1000.0)
    };
    let backbone = median(stages.iter().map(|s| s.backbone_seconds).collect());
    let depth = median(stages.iter().map(|s| s.depth_seconds).collect());
    match (backbone, depth) {
        (Some(backbone), Some(depth)) => {
            format!(" | backbone p50 {backbone:.0} ms | depth p50 {depth:.0} ms")
        }
        _ => String::new(),
    }
}

fn truncate(text: &str, limit: usize) -> String {
    let trimmed: String = text.chars().take(limit).collect();
    if trimmed.chars().count() < text.chars().count() {
        format!("{trimmed}…")
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::speech::moss_realtime::FrameStages;
    use std::cell::RefCell;

    /// Run `drain` over a canned request stream, returning the `text:` of
    /// every request that actually reached the server. Replies go to
    /// stdout because no block names a `reply:`.
    fn drained(input: &str) -> Vec<String> {
        let seen = RefCell::new(Vec::new());
        let serve = |request: Request| -> Result<String, BoxErr> {
            seen.borrow_mut()
                .push(request.text.clone().unwrap_or_default());
            Ok("ok frames=1".to_string())
        };
        drain(BufReader::new(input.as_bytes()), &serve);
        seen.into_inner()
    }

    #[test]
    fn absorb_fills_every_known_key() {
        let mut request = Request::default();
        for line in [
            "text: Good evening Sam.",
            "voice: /voices/jarvis.tokens",
            "tokens: /out/frames.txt",
            "stream: /out/frames.fifo",
            "seed: 5",
            "max-frames: 200",
            "greedy: true",
            "reply: /out/reply.fifo",
        ] {
            request.absorb(line).expect("known key");
        }
        assert_eq!(request.text.as_deref(), Some("Good evening Sam."));
        assert_eq!(request.voice.as_deref(), Some("/voices/jarvis.tokens"));
        assert_eq!(request.tokens, Some(PathBuf::from("/out/frames.txt")));
        assert_eq!(request.stream, Some(PathBuf::from("/out/frames.fifo")));
        assert_eq!(request.seed, Some(5));
        assert_eq!(request.max_frames, Some(200));
        assert_eq!(request.greedy, Some(true));
        assert_eq!(request.reply, Some(PathBuf::from("/out/reply.fifo")));
    }

    /// The whole point of rejecting unknown keys: a typo'd `voise:` must
    /// not be silently dropped, or it synthesises the default voice and
    /// looks like a model bug.
    #[test]
    fn absorb_rejects_an_unknown_key() {
        let error = Request::default()
            .absorb("voise: /voices/jarvis.tokens")
            .expect_err("typo must not be a silent no-op");
        assert!(error.to_string().contains("voise"), "{error}");
    }

    #[test]
    fn absorb_rejects_a_line_without_a_colon() {
        let error = Request::default()
            .absorb("text is missing its colon")
            .expect_err("header lines are `key: value`");
        assert!(error.to_string().contains("malformed"), "{error}");
    }

    #[test]
    fn absorb_rejects_an_unparseable_number() {
        assert!(Request::default().absorb("seed: soon").is_err());
        assert!(Request::default().absorb("max-frames: lots").is_err());
    }

    #[test]
    fn absorb_trims_around_the_separator() {
        let mut request = Request::default();
        request
            .absorb("  text  :   Good evening.  ")
            .expect("trimmed");
        assert_eq!(request.text.as_deref(), Some("Good evening."));
    }

    /// `voice: -` must survive parsing as the sentinel so `serve_one` can
    /// tell "explicitly unconditioned" from "inherit the server default".
    #[test]
    fn absorb_keeps_the_unconditioned_sentinel() {
        let mut request = Request::default();
        request.absorb("voice: -").expect("sentinel");
        assert_eq!(request.voice.as_deref(), Some(UNCONDITIONED));
        assert!(Request::default().voice.is_none(), "absent != sentinel");
    }

    #[test]
    fn absorb_reads_greedy_as_a_flag() {
        for truthy in ["1", "true", "yes"] {
            let mut request = Request::default();
            request.absorb(&format!("greedy: {truthy}")).expect("flag");
            assert_eq!(request.greedy, Some(true), "{truthy}");
        }
        let mut request = Request::default();
        request.absorb("greedy: no").expect("flag");
        assert_eq!(request.greedy, Some(false));
    }

    #[test]
    fn drain_settles_one_block_per_blank_line() {
        assert_eq!(
            drained("text: first\n\ntext: second\n\n"),
            ["first", "second"]
        );
    }

    #[test]
    fn drain_settles_a_trailing_block_at_eof() {
        // No terminating blank line — EOF ends the block instead.
        assert_eq!(drained("text: only\n"), ["only"]);
    }

    #[test]
    fn drain_ignores_runs_of_blank_lines() {
        assert_eq!(drained("\n\ntext: first\n\n\n\n"), ["first"]);
    }

    #[test]
    fn drain_does_not_serve_a_malformed_block() {
        assert!(drained("voise: typo\ntext: unheard\n\n").is_empty());
    }

    /// A bad block must not take the server down with it — the next one
    /// still gets served.
    #[test]
    fn drain_keeps_serving_after_a_malformed_block() {
        assert_eq!(
            drained("voise: typo\n\ntext: still here\n\n"),
            ["still here"]
        );
    }

    /// …and neither must a request that fails inside the model.
    #[test]
    fn drain_keeps_serving_after_a_failing_request() {
        let served = RefCell::new(0usize);
        let serve = |_request: Request| -> Result<String, BoxErr> {
            *served.borrow_mut() += 1;
            Err("generation produced no frames before EOS".into())
        };
        drain(
            BufReader::new("text: one\n\ntext: two\n\n".as_bytes()),
            &serve,
        );
        assert_eq!(served.into_inner(), 2);
    }

    #[test]
    fn stage_split_is_blank_without_stages() {
        assert_eq!(stage_split(&[]), "");
    }

    #[test]
    fn stage_split_reports_medians_in_milliseconds() {
        let stages = [
            // Frame 0 rides the prefill, so its backbone time is absent
            // (0.0) and must not drag the median down.
            FrameStages {
                backbone_seconds: 0.0,
                depth_seconds: 0.030,
            },
            FrameStages {
                backbone_seconds: 0.012,
                depth_seconds: 0.032,
            },
            FrameStages {
                backbone_seconds: 0.014,
                depth_seconds: 0.034,
            },
        ];
        assert_eq!(
            stage_split(&stages),
            " | backbone p50 14 ms | depth p50 32 ms"
        );
    }

    #[test]
    fn stage_split_is_blank_when_a_stage_never_recorded() {
        let stages = [FrameStages {
            backbone_seconds: 0.0,
            depth_seconds: 0.030,
        }];
        assert_eq!(stage_split(&stages), "");
    }

    #[test]
    fn truncate_leaves_short_text_alone() {
        assert_eq!(truncate("Good evening.", 60), "Good evening.");
    }

    #[test]
    fn truncate_marks_what_it_cut() {
        assert_eq!(truncate("abcdef", 3), "abc…");
    }

    /// Counts characters, not bytes — a byte slice would panic here.
    #[test]
    fn truncate_is_multibyte_safe() {
        assert_eq!(truncate("héllo wörld", 4), "héll…");
    }
}
