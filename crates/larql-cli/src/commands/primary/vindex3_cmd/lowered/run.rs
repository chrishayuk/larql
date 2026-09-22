//! `vindex3 exec --backend metal-lowered*`: the driver — load, prefill,
//! decode, and the report (timing, device/host split, optional stage
//! profile).

use larql_compute_metal::MetalBackend;
use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormats;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

use super::super::ExecArgs;
use super::dump::dump_lowered;
use super::step::host_argmax;
use super::teacher_force::run_teacher_force_lowered;
use super::LoweredSession;

/// `--bank` drives one resident model over many manifest entries on the
/// interpreter path (`bank.rs`); nothing on the lowered path reads it —
/// refuse rather than silently running `--tokens` as the only prompt and
/// writing none of the manifest's dumps (the failure this replaces:
/// exit 0, zero files, no indication the manifest was ever ignored).
///
/// Deliberately checked before any GPU/session state is touched — a
/// caller who passes `--bank` on a machine with no Metal device gets the
/// same clear refusal as one who has a device, not a confusing "no
/// Metal device available" instead of the real problem.
pub(super) fn refuse_bank_on_lowered(
    bank: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if bank.is_some() {
        return Err(
            "--bank is not supported on lowered backends (metal-lowered*); \
             it drives one resident model over a manifest and nothing in \
             the lowered path reads it. Use an interpreter backend \
             (e.g. --backend metal) for --bank runs."
                .into(),
        );
    }
    Ok(())
}

/// Run the plan through the lowering and report the final position's
/// logits, in the same shape `run_exec`'s other arms do.
pub(in super::super) fn run_lowered(
    args: &ExecArgs,
    tokens: &[u32],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    formats: WeightFormats,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    refuse_bank_on_lowered(args.bank.as_deref())?;
    let gpu = MetalBackend::new().ok_or("no Metal device available for --backend metal-lowered")?;
    let total = tokens.len() + args.generate.unwrap_or(0);
    let loading = std::time::Instant::now();
    let mut keep = Vec::new();
    let mut session = LoweredSession::new(&gpu, plan, store, formats, total.max(1), &mut keep)?;
    let load_seconds = loading.elapsed().as_secs_f64();
    eprintln!("weights resident in {load_seconds:.1} s");
    if let Some((rows, cols)) = session.head_geometry() {
        eprintln!("head geometry: [{rows}, {cols}]");
    }
    eprintln!(
        "plan: {} rope base(s), final norm {}",
        session.rope_bases(),
        if session.has_final_norm() {
            "present"
        } else {
            "absent"
        }
    );

    // ── per-layer dump: teacher-force the given tokens, capturing every
    //    layer's output per position into [seq, hidden] planes a
    //    `shannon layer-diff` reads (the lowered arm of the A-9.5 chain).
    if let Some(dir) = &args.dump_layers {
        return dump_lowered(&mut session, tokens, plan, &args.container, label, dir);
    }

    // ── teacher-forced logit dump: every position, not just the last ──
    // `clap` already forbids combining this with --generate (mod.rs's
    // `conflicts_with_all`), so there is no decode loop to reconcile
    // with — this is prefill-only, exactly like --dump-layers above.
    if let Some(path) = &args.logit_dump {
        let engine = format!("vindex3-metal-lowered-{label}");
        return run_teacher_force_lowered(&mut session, &engine, tokens, path);
    }

    let prompt_started = std::time::Instant::now();
    let mut next_id: Option<u32> = None;
    for &token in tokens {
        next_id = session.step(token)?;
    }
    // ── decode, kept strictly separate from prefill ─────────────────
    let mut decode_ms: Vec<f64> = Vec::new();
    let mut decode_gpu_ms: Vec<f64> = Vec::new();
    let mut decode_encode_ms: Vec<f64> = Vec::new();
    let mut generated: Vec<u32> = Vec::new();
    // The id the device produced for the last executed position.
    let mut next: u32 = 0;
    if let Some(n) = args.generate {
        next = next_id.ok_or("plan carries no output head — cannot generate")?;
        if args.profile {
            session.start_profile();
        }
        // From here every step continues from the device argmax: the
        // session gathers each next embedding on the device and commits
        // the look-ahead before its predecessor completes (1c).
        session.begin_decode();
        for _ in 0..n {
            generated.push(next);
            let started = std::time::Instant::now();
            let id = session.step(next)?.ok_or("plan carries no output head")?;
            decode_ms.push(started.elapsed().as_secs_f64() * 1e3);
            decode_gpu_ms.push(session.last_gpu_ms());
            decode_encode_ms.push(session.last_encode_ms());
            next = id;
        }
    }
    // Wait out the committed look-ahead step (its logits now occupy the
    // head slot; its argmax id is the one they belong to), then read the
    // final logits once for the summary line.
    let quiesced_id = session.quiesce();
    let logits = session.last_logits();

    let prompt_seconds = prompt_started.elapsed().as_secs_f64();
    if session.ablation_active() {
        println!("ABLATED RUN — numbers are wrong by construction; timing only");
    }
    println!("engine: vindex3-metal-lowered-{label}");
    println!("weights loaded: {load_seconds:.1} s");
    println!(
        "prompt: {} tokens in {prompt_seconds:.1} s ({:.0} ms/token)",
        tokens.len(),
        prompt_seconds * 1e3 / tokens.len().max(1) as f64,
    );
    if !decode_ms.is_empty() {
        let mut sorted = decode_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let pct = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
        // Steady state = the second half, so warmup and first-touch
        // residency do not flatter or penalise the median.
        let steady = &decode_ms[decode_ms.len() / 2..];
        let steady_mean = steady.iter().sum::<f64>() / steady.len() as f64;
        let steady_gpu = &decode_gpu_ms[decode_gpu_ms.len() / 2..];
        let steady_gpu_mean = steady_gpu.iter().sum::<f64>() / steady_gpu.len() as f64;
        let steady_enc = &decode_encode_ms[decode_encode_ms.len() / 2..];
        let steady_enc_mean = steady_enc.iter().sum::<f64>() / steady_enc.len() as f64;
        println!("decode tokens: {}", decode_ms.len());
        println!("first token: {:.2} ms", decode_ms[0]);
        println!("decode p50: {:.2} ms  p95: {:.2} ms", pct(0.50), pct(0.95));
        println!(
            "steady (last half): {:.2} ms/token ({:.3} tok/s)",
            steady_mean,
            1000.0 / steady_mean
        );
        // Device vs host split: the command buffer's own GPU span against
        // the wall step. Their difference is host work on the token's
        // critical path — embedding, commit latency, readback, argmax.
        // Encode is reported separately because `step` overlaps it with
        // the previous token's GPU execution (see `step.rs`).
        println!(
            "steady GPU span: {:.2} ms/token  host on critical path: {:.2} ms/token  (encode {:.2} ms/token, overlapped)",
            steady_gpu_mean,
            steady_mean - steady_gpu_mean,
            steady_enc_mean,
        );
        println!("generated ids: {generated:?}");
        // Which attention kernel actually ran — the seqpar port is judged
        // by this witness, not inferred from a throughput number.
        {
            use std::sync::atomic::Ordering;
            let serial =
                larql_compute_metal::route_witness::LOWERED_ATTEND_SERIAL.load(Ordering::Relaxed);
            let seqpar =
                larql_compute_metal::route_witness::LOWERED_ATTEND_SEQPAR.load(Ordering::Relaxed);
            println!("attention dispatches: serial {serial}  seqpar {seqpar}");
        }
        if let Some(lines) = session.profile_report() {
            for line in lines {
                println!("{line}");
            }
        }
    }
    match &logits {
        Some(l) => {
            // Host scan over the final logits, cross-checked against the
            // id the device argmax produced for the same position — a
            // standing gate on the kernel, free on every run.
            let best = host_argmax(l) as usize;
            let value = l[best];
            let device = quiesced_id.or(if generated.is_empty() {
                next_id
            } else {
                Some(next)
            });
            let check = match device {
                Some(d) if d as usize == best => "device argmax agrees",
                Some(d) => {
                    eprintln!("DEVICE ARGMAX MISMATCH: device {d}, host {best}");
                    "DEVICE ARGMAX MISMATCH"
                }
                None => "no device argmax",
            };
            println!("logits: {}, argmax {best} ({value:+.4}) — {check}", l.len());
            // --logit-dump returns above, via `run_teacher_force_lowered`,
            // before this summary block ever runs — there is no
            // last-position-only write here to duplicate or fall behind.
        }
        None => println!("logits: none (plan carries no output head)"),
    }
    Ok(())
}
