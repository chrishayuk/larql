//! Pure helpers for the VINDEX3 bench arm. The I/O body lives in
//! `vindex3_runtime.rs`; this file owns what can be decided without a
//! model:
//!   * `resolve_backends` — `--backends` names → V3 execution backends
//!   * `refuse_inapplicable_flags` — the V2-only flags, refused by name
//!   * `row_label` / `is_device_backend` — row identity and warm-up policy
//!   * `summarise` — one timed run → a `BenchRow`
//!
//! The row is built from the same statistic the V2 rows use — mean, p50
//! and p99 over every decode step after `--warmup` — so a V2 and a V3 row
//! in one table are the same measurement. `larql vindex3 exec --generate`
//! reports a last-half mean instead; the two are not interchangeable.

use clap::ValueEnum;

use super::args::BenchArgs;
use super::local::{append_repeat_note, format_early_stop_note};
use super::row::{compute_percentiles, BenchRow};
use crate::commands::primary::vindex3_cmd::ExecBackend;

/// Row label prefix, so a V3 row can never be mistaken for a V2 one.
const ROW_PREFIX: &str = "vindex3";

/// Backend names that realise the plan on a device rather than the CPU.
const DEVICE_PREFIX: &str = "metal";

/// The V2 bench's CPU name.
const CPU_ALIAS: &str = "cpu";

/// The V2 bench's GPU name.
const METAL_ALIAS: &str = "metal";

/// Map `--backends` entries onto V3 execution backends.
///
/// The V2 names keep their meaning: `cpu` is the `larql-compute` kernels
/// (`production`, what `larql run` and `larql serve` execute) and `metal`
/// is the Metal realisation (`larql run --metal`). Any other entry must
/// name a V3 backend exactly as `larql vindex3 exec --backend` spells it,
/// so a representation arm (`production-q4k`, `metal-lowered`, …) is
/// benchable without a second vocabulary.
pub(super) fn resolve_backends(names: &[&str]) -> Result<Vec<ExecBackend>, String> {
    names.iter().map(|name| resolve_backend(name)).collect()
}

fn resolve_backend(name: &str) -> Result<ExecBackend, String> {
    if name == CPU_ALIAS {
        return Ok(ExecBackend::Production);
    }
    if name == METAL_ALIAS {
        return metal_backend();
    }
    ExecBackend::from_str(name, false).map_err(|_| {
        format!(
            "unknown VINDEX3 bench backend {name:?} — use `{CPU_ALIAS}`, `{METAL_ALIAS}`, or a \
             `larql vindex3 exec --backend` name: {}",
            backend_names().join(", ")
        )
    })
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn metal_backend() -> Result<ExecBackend, String> {
    Ok(ExecBackend::Metal)
}

#[cfg(not(all(feature = "gpu", target_os = "macos")))]
fn metal_backend() -> Result<ExecBackend, String> {
    Err(format!(
        "`{METAL_ALIAS}` needs the `gpu` feature on macOS; this build has neither — \
         use `--backends {CPU_ALIAS}`"
    ))
}

/// Every name `larql vindex3 exec --backend` accepts in this build.
fn backend_names() -> Vec<String> {
    ExecBackend::value_variants()
        .iter()
        .filter_map(|b| b.to_possible_value())
        .map(|v| v.get_name().to_string())
        .collect()
}

/// The backend's CLI name — the same spelling `--backend` takes.
pub(super) fn backend_name(backend: ExecBackend) -> String {
    backend
        .to_possible_value()
        .map(|v| v.get_name().to_string())
        .unwrap_or_else(|| format!("{backend:?}"))
}

/// Table label for a V3 row.
pub(super) fn row_label(backend: ExecBackend) -> String {
    format!("{ROW_PREFIX}-{}", backend_name(backend))
}

/// Whether the backend executes on a device. The V2 bench pre-warms only
/// its Metal path (buffer caches, pipeline state); the V3 arm follows the
/// same policy so the prefill columns stay comparable.
pub(super) fn is_device_backend(backend: ExecBackend) -> bool {
    backend_name(backend).starts_with(DEVICE_PREFIX)
}

/// Refuse every flag the V3 arm cannot honour, together and by name.
///
/// They configure the V2 engine, its composition or its remote paths. A
/// V3 container executes its own program, so accepting one would silently
/// time something other than what the flag asked for. `--ollama` is kept:
/// it is an external baseline, not a V2 setting.
pub(super) fn refuse_inapplicable_flags(args: &BenchArgs) -> Result<(), String> {
    let set: Vec<&str> = [
        ("--engine", args.engine.is_some()),
        ("--ffn", args.ffn.is_some()),
        ("--wire", args.wire.is_some()),
        ("--ffn-policy", args.ffn_policy.is_some()),
        ("--moe-shards", args.moe_shards.is_some()),
        ("--routed-from", args.routed_from.is_some()),
        ("--bench-grid", args.bench_grid),
        ("--via-executor", args.via_executor),
        ("--profile", args.profile),
        ("--metal", args.metal),
        ("--concurrent", args.concurrent > 1),
    ]
    .into_iter()
    .filter_map(|(flag, given)| given.then_some(flag))
    .collect();
    if set.is_empty() {
        return Ok(());
    }
    Err(format!(
        "a VINDEX3 container is benched through its own program; these flags configure the \
         VINDEX2 engine and are not honoured here: {}",
        set.join(", ")
    ))
}

/// What a representation backend actually bound, as a row-note fragment.
///
/// A backend that asks for a representation (`production-q4k`, …) is
/// only benchmarking it when the store bound objects from a compiled
/// pack. With none, it executes the canonical weights under a label that
/// names a different encoding — so that is refused, naming the fix,
/// rather than written as a row. Partial coverage is legitimate (a role
/// policy keeps embeddings and norms at source precision) and is stated.
pub(super) fn representation_note(
    backend: ExecBackend,
    want: Option<&str>,
    from_pack: usize,
    objects: usize,
) -> Result<Option<String>, String> {
    let Some(want) = want else {
        return Ok(None);
    };
    if from_pack == 0 {
        return Err(format!(
            "`{}` asks for a {want} representation, but this container has no compiled {want} \
             pack (0/{objects} objects) — the row would time the canonical weights under a {want} \
             label. Compile a pack into the container first, or bench `cpu`.",
            backend_name(backend)
        ));
    }
    Ok(Some(format!("{want} pack {from_pack}/{objects} objects")))
}

/// Device time over the measured window, as a difference of the
/// backend's cumulative counters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DeviceWindow {
    pub device_ms: f64,
    pub submissions: u64,
}

/// One timed V3 generation, before it becomes a row.
pub(super) struct TimedRun<'a> {
    pub backend: ExecBackend,
    pub prefill_ms: f64,
    /// Wall ms of every decode step, warmup included.
    pub step_ms: &'a [f64],
    pub warmup: usize,
    pub target_tokens: usize,
    pub wall_ms: f64,
    pub prompt_tokens: usize,
    pub device: Option<DeviceWindow>,
    pub representation: Option<&'a str>,
    pub fingerprint: &'a str,
}

/// Build the row: statistics over the steps after `warmup`, and a note
/// that says what the numbers are about.
pub(super) fn summarise(run: &TimedRun<'_>) -> BenchRow {
    let n_warm = run.warmup.min(run.step_ms.len());
    let measured = &run.step_ms[n_warm..];
    let (avg, p50, p99) = compute_percentiles(measured);
    let tok_per_s = if measured.is_empty() {
        0.0
    } else {
        1000.0 / avg
    };

    let mut parts = vec![format!("prompt {} tok", run.prompt_tokens)];
    parts.extend(run.representation.map(str::to_string));
    let early = format_early_stop_note(measured.len(), run.target_tokens, run.wall_ms);
    if !early.is_empty() {
        parts.push(early);
    }
    if let Some(device) = run.device.filter(|_| !measured.is_empty()) {
        parts.push(format_device_note(device, measured.len(), avg));
    }
    let note = append_repeat_note(parts.join("; "), 0, 1, run.fingerprint);

    BenchRow {
        backend: row_label(run.backend),
        prefill_ms: run.prefill_ms,
        avg_decode_ms: avg,
        p50_ms: p50,
        p99_ms: p99,
        tok_per_s,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: measured.len(),
        note,
    }
}

/// Ids the reset witness compares: the prefill argmax, and the id the
/// first decode step produces.
#[cfg(any(test, all(feature = "gpu", target_os = "macos")))]
const RESET_WITNESS_IDS: usize = 2;

/// Refuse a lowered row whose timed run did not start where a fresh
/// session starts.
///
/// The warm-up runs from a fresh session and the timed run from a reset
/// one, over the same prompt, so their leading ids must agree. A reset
/// that leaked state (a position, a stale look-ahead, a decode chain)
/// would time a different generation and still print a plausible row.
/// Compared over the ids both runs produced, so an early EOS shortens
/// the check rather than failing it.
#[cfg(any(test, all(feature = "gpu", target_os = "macos")))]
pub(super) fn check_reset_witness(
    backend: ExecBackend,
    warm: &[u32],
    timed: &[u32],
) -> Result<(), String> {
    let n = RESET_WITNESS_IDS.min(warm.len()).min(timed.len());
    if warm[..n] == timed[..n] {
        return Ok(());
    }
    Err(format!(
        "{}: the run after reset began {:?}, the warm-up from a fresh session {:?} — \
         the reset leaked state, so the row would time a different generation",
        row_label(backend),
        &timed[..n],
        &warm[..n]
    ))
}

/// Device share of a decode token: time inside device calls, the rest
/// (the interpreter's glue), and submissions per token.
fn format_device_note(device: DeviceWindow, steps: usize, mean_ms: f64) -> String {
    let per_tok = device.device_ms / steps as f64;
    format!(
        "device {:.2} ms/tok + glue {:.2} ms/tok, {:.1} submissions/tok",
        per_tok,
        (mean_ms - per_tok).max(0.0),
        device.submissions as f64 / steps as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Harness {
        #[command(flatten)]
        args: BenchArgs,
    }

    fn args(extra: &[&str]) -> BenchArgs {
        let argv = ["bench", "model.vindex3"].iter().chain(extra);
        Harness::parse_from(argv).args
    }

    #[test]
    fn the_reset_witness_passes_matching_leading_ids() {
        let b = ExecBackend::Production;
        assert!(check_reset_witness(b, &[5, 9], &[5, 9, 11, 13]).is_ok());
        // Only the leading ids are compared; what follows may differ.
        assert!(check_reset_witness(b, &[5, 9, 1], &[5, 9, 2]).is_ok());
        // An early EOS shortens the comparison instead of failing it.
        assert!(check_reset_witness(b, &[5], &[5, 9]).is_ok());
        assert!(check_reset_witness(b, &[], &[5, 9]).is_ok());
    }

    #[test]
    fn the_reset_witness_refuses_a_run_that_began_elsewhere() {
        let b = ExecBackend::Production;
        let err = check_reset_witness(b, &[5, 9], &[7, 9]).unwrap_err();
        assert!(err.contains("reset leaked state"), "{err}");
        assert!(err.contains("[7, 9]") && err.contains("[5, 9]"), "{err}");
        // The second id is inside the window too.
        assert!(check_reset_witness(b, &[5, 9], &[5, 8]).is_err());
    }

    #[test]
    fn cpu_alias_is_the_production_backend() {
        let resolved = resolve_backends(&["cpu"]).unwrap();
        assert_eq!(backend_name(resolved[0]), "production");
        assert_eq!(row_label(resolved[0]), "vindex3-production");
        assert!(!is_device_backend(resolved[0]));
    }

    #[cfg(all(feature = "gpu", target_os = "macos"))]
    #[test]
    fn metal_alias_is_the_metal_backend() {
        let resolved = resolve_backends(&["metal"]).unwrap();
        assert_eq!(row_label(resolved[0]), "vindex3-metal");
        assert!(is_device_backend(resolved[0]));
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos")))]
    #[test]
    fn metal_alias_refuses_without_the_gpu_build() {
        let err = resolve_backends(&["metal"]).unwrap_err();
        assert!(err.contains("gpu"), "{err}");
    }

    #[test]
    fn exec_backend_names_pass_through() {
        let resolved = resolve_backends(&["production-q4k", "reference"]).unwrap();
        assert_eq!(row_label(resolved[0]), "vindex3-production-q4k");
        assert_eq!(row_label(resolved[1]), "vindex3-reference");
    }

    #[test]
    fn unknown_backend_names_the_accepted_ones() {
        let err = resolve_backends(&["cpu", "tpu"]).unwrap_err();
        assert!(err.contains("\"tpu\""), "{err}");
        assert!(err.contains("production-q4k"), "{err}");
    }

    #[test]
    fn plain_bench_flags_are_accepted() {
        assert!(refuse_inapplicable_flags(&args(&["--cpu", "--ollama", "gemma3:4b"])).is_ok());
    }

    #[test]
    fn every_v2_flag_is_refused_in_one_message() {
        let given = args(&[
            "--engine",
            "standard",
            "--ffn",
            "http://x",
            "--wire",
            "f32",
            "--ffn-policy",
            "dense",
            "--moe-shards",
            "0-1=http://x",
            "--routed-from",
            "dir",
            "--bench-grid",
            "--via-executor",
            "--profile",
            "--metal",
            "--concurrent",
            "2",
        ]);
        let err = refuse_inapplicable_flags(&given).unwrap_err();
        for flag in [
            "--engine",
            "--ffn,",
            "--wire",
            "--ffn-policy",
            "--moe-shards",
            "--routed-from",
            "--bench-grid",
            "--via-executor",
            "--profile",
            "--metal",
            "--concurrent",
        ] {
            assert!(err.contains(flag), "{flag} missing from: {err}");
        }
    }

    fn run<'a>(step_ms: &'a [f64], device: Option<DeviceWindow>) -> TimedRun<'a> {
        TimedRun {
            backend: ExecBackend::Production,
            prefill_ms: 40.0,
            step_ms,
            warmup: 2,
            target_tokens: 4,
            wall_ms: 100.0,
            prompt_tokens: 7,
            device,
            representation: None,
            fingerprint: "abc",
        }
    }

    #[test]
    fn canonical_backends_carry_no_representation_note() {
        assert_eq!(
            representation_note(ExecBackend::Production, None, 0, 5),
            Ok(None)
        );
    }

    #[test]
    fn a_representation_with_no_pack_is_refused() {
        let err = representation_note(ExecBackend::ProductionQ4k, Some("Q4_K"), 0, 5).unwrap_err();
        assert!(err.contains("`production-q4k`"), "{err}");
        assert!(err.contains("0/5"), "{err}");
    }

    #[test]
    fn pack_coverage_reaches_the_row_note() {
        let note = representation_note(ExecBackend::ProductionQ4k, Some("Q4_K"), 3, 5)
            .unwrap()
            .unwrap();
        assert_eq!(note, "Q4_K pack 3/5 objects");
        let steps = [1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        let row = summarise(&TimedRun {
            representation: Some(&note),
            ..run(&steps, None)
        });
        assert_eq!(row.note, "prompt 7 tok; Q4_K pack 3/5 objects; fp=abc");
    }

    #[test]
    fn summary_discards_warmup_steps() {
        let steps = [100.0, 90.0, 10.0, 10.0, 20.0, 20.0];
        let row = summarise(&run(&steps, None));
        assert_eq!(row.n_steps, 4);
        assert!((row.avg_decode_ms - 15.0).abs() < 1e-9);
        assert!((row.tok_per_s - 1000.0 / 15.0).abs() < 1e-9);
        assert_eq!(row.p99_ms, 20.0);
        assert_eq!(row.prefill_ms, 40.0);
        assert_eq!(row.backend, "vindex3-production");
        assert_eq!(row.note, "prompt 7 tok; fp=abc");
    }

    #[test]
    fn summary_reports_an_early_stop() {
        let steps = [100.0, 90.0, 10.0];
        let row = summarise(&run(&steps, None));
        assert_eq!(row.n_steps, 1);
        assert!(row.note.contains("early stop @1/4"), "{}", row.note);
    }

    #[test]
    fn summary_with_no_measured_steps_has_no_rate() {
        let device = DeviceWindow {
            device_ms: 5.0,
            submissions: 3,
        };
        let row = summarise(&run(&[100.0], Some(device)));
        assert_eq!(row.n_steps, 0);
        assert_eq!(row.tok_per_s, 0.0);
        assert!(row.note.contains("no decode steps"), "{}", row.note);
        assert!(!row.note.contains("device"), "{}", row.note);
    }

    #[test]
    fn summary_splits_device_from_glue() {
        let steps = [100.0, 90.0, 10.0, 10.0];
        let device = DeviceWindow {
            device_ms: 16.0,
            submissions: 70,
        };
        let row = summarise(&run(&steps, Some(device)));
        assert!(
            row.note
                .contains("device 8.00 ms/tok + glue 2.00 ms/tok, 35.0 submissions/tok"),
            "{}",
            row.note
        );
    }

    #[test]
    fn glue_never_reads_negative() {
        let note = format_device_note(
            DeviceWindow {
                device_ms: 30.0,
                submissions: 1,
            },
            1,
            20.0,
        );
        assert!(note.contains("glue 0.00"), "{note}");
    }
}
