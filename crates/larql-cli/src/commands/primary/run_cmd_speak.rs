//! `larql run --speak` — the model speaks the prompt.
//!
//! In speak mode the run target IS the speech model (MOSS-TTS-Realtime)
//! and the prompt is the text to synthesise: pure TTS, no chat LLM in
//! front. (The LLM-driven pipeline arrives when a chat model's token
//! stream feeds the same generation session.)
//!
//! Until funnel step 6 moves the model into the vindex, `model` is the
//! checkpoint's safetensors directory. The codec stays external
//! (`docs/tts-funnel.md` non-goals): audio tokens go out to a token
//! file, and a codec command — `--codec-cmd` or `$LARQL_MOSS_CODEC_CMD`,
//! with `{tokens}` / `{wav}` placeholders — turns them into a WAV.
//! Every stage is timed; the table at the end is the step-5 benchmark
//! surface (the budget that matters: 80 ms per frame).

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute::ffn::q4k_weight::Q4kFfn;
use larql_compute::{KvIndex, PackedAttnIndex};
use larql_inference::ffn::{FfnBackend, WeightFfn};
use larql_inference::speech::moss_prompt::build_prompt;
use larql_inference::speech::moss_realtime::generate_frames_streaming;
use larql_inference::speech::moss_sampling::{DecodeMode, MossSampling};
use larql_inference::speech::stream_timing::{analyze_stream, SECONDS_PER_FRAME};
use larql_inference::tokenizer::load_tokenizer;
use larql_models::loading::safetensors::load_model_dir;
use larql_models::speech::moss_tts_realtime::{
    depth_transformer_model, load_moss_tts_aux_from_safetensors, MossTtsRealtimeConfig,
};
use ndarray::Array2;

use super::run_cmd::RunArgs;

type BoxErr = Box<dyn std::error::Error>;

/// Environment fallback for `--codec-cmd`.
const CODEC_CMD_ENV: &str = "LARQL_MOSS_CODEC_CMD";
const TOKENS_PLACEHOLDER: &str = "{tokens}";
const WAV_PLACEHOLDER: &str = "{wav}";

pub fn run_speak(args: &RunArgs) -> Result<(), BoxErr> {
    let model_dir = PathBuf::from(&args.model);
    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        return Err(format!(
            "--speak expects the speech checkpoint's safetensors directory \
             (no config.json under {model_dir:?}); vindex residency is funnel step 6"
        )
        .into());
    }
    let text = args
        .prompt
        .as_deref()
        .ok_or("--speak needs a prompt: the text to say")?;

    // ── Load ──
    let started = Instant::now();
    let weights = load_model_dir(&model_dir)?;
    if weights.arch.family() != "moss_tts_realtime" {
        return Err(format!(
            "--speak supports moss_tts_realtime models; this is {:?}",
            weights.arch.family()
        )
        .into());
    }
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let config = MossTtsRealtimeConfig::from_config_json(&config_json)?;
    let aux =
        load_moss_tts_aux_from_safetensors(&model_dir, weights.arch.as_ref(), config.clone())?;
    let audio_tables = aux.audio_embed_tables;
    let depth = depth_transformer_model(aux.local, &config)?;
    let tokenizer = load_tokenizer(&model_dir)?;
    let load_seconds = started.elapsed().as_secs_f64();
    eprintln!("loaded in {load_seconds:.1}s");

    // ── Prompt (voice-clone splice when --voice is given) ──
    let reference = args
        .voice
        .as_deref()
        .map(|path| read_token_rows(path, config.rvq))
        .transpose()?;
    let prompt = build_prompt(&tokenizer, &config, reference.as_ref(), text)?;
    eprintln!(
        "prompt: {} rows ({}), {} text ids queued after the lead",
        prompt.prefill_matrix.nrows(),
        match &reference {
            Some(codes) => format!("voice reference {} frames", codes.nrows()),
            None => "no voice reference".to_string(),
        },
        prompt.text_queue.len(),
    );

    // ── Generate, timing every frame ──
    // `--metal` routes the backbone through the default backend
    // composition (Metal + CPU fallback when built with the gpu
    // feature); the depth transformer stays on CPU either way for now.
    let backend = if args.metal {
        larql_inference::default_engine_backend()
    } else {
        larql_inference::cpu_engine_backend()
    };
    let backbone_dense = WeightFfn { weights: &weights };
    let depth_dense = WeightFfn {
        weights: &depth.weights,
    };
    let (ffn, depth_ffn): (&dyn FfnBackend, &dyn FfnBackend);
    let (backbone_q4, depth_q4);
    let (backbone_attn_q4, depth_attn_q4);
    let (backbone_attn, depth_attn): (Option<&dyn KvIndex>, Option<&dyn KvIndex>);
    if args.q4 {
        let quant_start = Instant::now();
        backbone_q4 = Q4kFfn::quantize_from(&weights)?;
        depth_q4 = Q4kFfn::quantize_from(&depth.weights)?;
        backbone_attn_q4 = PackedAttnIndex::quantize_from(&weights)?;
        depth_attn_q4 = PackedAttnIndex::quantize_from(&depth.weights)?;
        eprintln!(
            "FFNs + attention projections quantised to Q4_K in {:.1}s",
            quant_start.elapsed().as_secs_f64()
        );
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
    let generation_start = Instant::now();
    let mut frame_instants: Vec<f64> = Vec::new();
    // With LARQL_PHASE_TIMING set, snapshot the accumulated phase split
    // at first frame — bounding it to prefill + one depth frame, the
    // TTFA-relevant region.
    let mut phase_split: Vec<(&'static str, f64)> = Vec::new();
    let mode = if args.greedy {
        DecodeMode::Greedy
    } else {
        DecodeMode::Sampled(MossSampling::default())
    };
    let generation = generate_frames_streaming(
        backend.as_ref(),
        &weights,
        ffn,
        &audio_tables,
        &depth,
        depth_ffn,
        &config,
        &prompt.prefill_matrix,
        &prompt.text_queue,
        prompt.text_pad_id,
        args.max_frames,
        mode,
        args.seed,
        backbone_attn,
        depth_attn,
        |index, _codes| {
            frame_instants.push(generation_start.elapsed().as_secs_f64());
            if index == 0 {
                phase_split = larql_compute::phase_timing::snapshot_and_reset();
            }
            if (index + 1) % 25 == 0 {
                eprintln!(
                    "  {} frames ({:.1}s audio)…",
                    index + 1,
                    (index + 1) as f64 * SECONDS_PER_FRAME
                );
            }
        },
    )?;
    let generate_seconds = generation_start.elapsed().as_secs_f64();
    let emitted = generation.emitted();
    if emitted.is_empty() {
        return Err("generation produced no frames before EOS".into());
    }

    // ── Emit tokens; codec; optional playback ──
    let wav_path = args
        .speech_out
        .clone()
        .unwrap_or_else(|| PathBuf::from("speech.wav"));
    let tokens_path = wav_path.with_extension("tokens.txt");
    write_token_rows(&tokens_path, emitted)?;
    eprintln!("audio tokens -> {}", tokens_path.display());

    let codec_cmd = args
        .codec_cmd
        .clone()
        .or_else(|| std::env::var(CODEC_CMD_ENV).ok());
    let mut codec_seconds = None;
    match codec_cmd {
        Some(template) => {
            let codec_start = Instant::now();
            run_codec(&template, &tokens_path, &wav_path)?;
            codec_seconds = Some(codec_start.elapsed().as_secs_f64());
            eprintln!("wav -> {}", wav_path.display());
            if args.play {
                std::process::Command::new("afplay")
                    .arg(&wav_path)
                    .status()?;
            }
        }
        None => eprintln!(
            "no codec command (--codec-cmd or ${CODEC_CMD_ENV}); tokens written, wav skipped"
        ),
    }

    // ── The step-5 benchmark table ──
    let frames = generation.frames.len();
    let audio_seconds = emitted.len() as f64 * SECONDS_PER_FRAME;
    let mut frame_durations: Vec<f64> = frame_instants
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    frame_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = |p: f64| -> f64 {
        let rank = ((frame_durations.len() as f64 - 1.0) * p / 100.0).round() as usize;
        frame_durations[rank]
    };
    println!(
        "\nframes                {frames} ({audio_seconds:.2}s audio, eos={})",
        generation.eos_at.is_some()
    );
    println!("model load            {load_seconds:.2}s");
    println!(
        "first frame (TTFA)    {:.0} ms after generation start",
        frame_instants[0] * 1000.0
    );
    println!(
        "generation            {generate_seconds:.2}s ({:.2} frames/s)",
        frames as f64 / generate_seconds
    );
    println!(
        "realtime factor       {:.2}x (1.0 = keeps up with playback)",
        audio_seconds / generate_seconds
    );
    if !frame_durations.is_empty() {
        println!(
            "frame compute         p50 {:.0} ms | p95 {:.0} ms | max {:.0} ms (budget {:.0} ms)",
            percentile(50.0) * 1000.0,
            percentile(95.0) * 1000.0,
            frame_durations.last().unwrap() * 1000.0,
            SECONDS_PER_FRAME * 1000.0
        );
    }
    // The step-5 streaming measurements: what a playback clock at the
    // frame rate would have experienced against these arrival times.
    if let Some(report) = analyze_stream(&frame_instants, SECONDS_PER_FRAME) {
        println!(
            "stream                min stable pre-buffer {:.0} ms | cold-start underruns {}",
            report.min_stable_prebuffer_seconds * 1000.0,
            report.cold_start_underruns
        );
        println!(
            "buffer                {:+.0} ms audio per second of playback | occupancy p50 {:.0} ms | p95 {:.0} ms",
            report.buffer_growth_per_second * 1000.0,
            report.occupancy_p50_seconds * 1000.0,
            report.occupancy_p95_seconds * 1000.0
        );
    }
    // Stage split: where a frame's milliseconds actually go.
    let stages = &generation.stage_timings;
    if !stages.is_empty() {
        let mut backbone: Vec<f64> = stages
            .iter()
            .map(|s| s.backbone_seconds)
            .filter(|&s| s > 0.0)
            .collect();
        let mut depth: Vec<f64> = stages.iter().map(|s| s.depth_seconds).collect();
        backbone.sort_by(|a, b| a.partial_cmp(b).unwrap());
        depth.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = |v: &[f64]| v[v.len() / 2] * 1000.0;
        println!(
            "prefill               {:.2}s ({} rows)",
            generation.prefill_seconds,
            prompt.prefill_matrix.nrows()
        );
        if !backbone.is_empty() {
            println!(
                "  backbone step       p50 {:.0} ms | max {:.0} ms per frame",
                mid(&backbone),
                backbone.last().unwrap() * 1000.0
            );
        }
        println!(
            "  depth (16 micro)    p50 {:.0} ms | max {:.0} ms per frame",
            mid(&depth),
            depth.last().unwrap() * 1000.0
        );
    }
    if let Some(seconds) = codec_seconds {
        println!("codec decode          {seconds:.2}s (external, includes model load)");
    }
    if !phase_split.is_empty() {
        let accounted: f64 = phase_split.iter().map(|(_, s)| s).sum();
        println!(
            "\nprefill phase split   ({} set; prefill + depth frame 0; {:.0} ms accounted)",
            larql_compute::phase_timing::ENV_VAR,
            accounted * 1000.0
        );
        phase_split.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite phase times"));
        for (name, seconds) in &phase_split {
            println!("  {name:<22}{:6.0} ms", seconds * 1000.0);
        }
    }
    Ok(())
}

/// Token-row files: one frame per line, `rvq` whitespace-separated ids,
/// `#` comments ignored — the same format `moss_codec_cli.py` speaks.
/// Shared with `speak_serve`, which reads a voice reference per request.
pub(super) fn read_token_rows(path: &Path, rvq: usize) -> Result<Array2<u32>, BoxErr> {
    let mut rows: Vec<u32> = Vec::new();
    let mut count = 0usize;
    for line in std::fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values: Vec<u32> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if values.len() != rvq {
            return Err(format!(
                "{}: expected {rvq} ids per row, found {}",
                path.display(),
                values.len()
            )
            .into());
        }
        rows.extend(values);
        count += 1;
    }
    if count == 0 {
        return Err(format!("{}: no token rows", path.display()).into());
    }
    Ok(Array2::from_shape_vec((count, rvq), rows)?)
}

pub(super) fn write_token_rows(path: &Path, frames: &[Vec<u32>]) -> Result<(), BoxErr> {
    let mut out = String::with_capacity(frames.len() * frames[0].len() * 5);
    out.push_str("# larql run --speak: generated MOSS audio tokens\n");
    for frame in frames {
        let row: Vec<String> = frame.iter().map(u32::to_string).collect();
        out.push_str(&row.join(" "));
        out.push('\n');
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn run_codec(template: &str, tokens: &Path, wav: &Path) -> Result<(), BoxErr> {
    if !template.contains(TOKENS_PLACEHOLDER) || !template.contains(WAV_PLACEHOLDER) {
        return Err(format!(
            "codec command must contain {TOKENS_PLACEHOLDER} and {WAV_PLACEHOLDER}"
        )
        .into());
    }
    let command = template
        .replace(TOKENS_PLACEHOLDER, &tokens.display().to_string())
        .replace(WAV_PLACEHOLDER, &wav.display().to_string());
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        .status()?;
    if !status.success() {
        return Err(format!("codec command failed ({status}): {command}").into());
    }
    Ok(())
}
