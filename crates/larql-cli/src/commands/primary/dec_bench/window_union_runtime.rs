//! I/O driver for `larql dec-bench window-union` (BW11-1). Reads the trace
//! files named on the command line, calls into the pure logic in
//! `window_union.rs` for every (file, K) pair, pools the per-layer results
//! across files, and prints one spread row per K plus an optional pulse
//! line. All decision logic — parsing, windowing, the union arithmetic,
//! percentiles — lives in `window_union.rs` and is unit-tested there; this
//! module is thin enough to be covered directly with `tempfile` fixtures
//! (unlike `capture_runtime.rs` / `drift_runtime.rs`, it needs no live
//! model or GPU — a trace file is already a plain text artifact).

use std::fs::File;
use std::io::{BufReader, Write};

use super::args::WindowUnionArgs;
use super::window_union::{
    format_spread_row, parse_trace, shuffled_control_by_layer, summarize_spread,
    window_union_by_layer, LayerWindowUnion, TraceRecord,
};

pub fn run_window_union(args: &WindowUnionArgs) -> Result<(), Box<dyn std::error::Error>> {
    let ks = parse_k_list(&args.k)?;
    let traces = load_traces(&args.trace)?;

    let activation_fracs: Vec<f64> = traces
        .iter()
        .flat_map(|recs| recs.iter())
        .map(|r| r.experts.first().map(|e| e.len()).unwrap_or(0) as f64 / args.num_experts as f64)
        .collect();
    let activation_frac = mean(&activation_fracs).unwrap_or(f64::NAN);

    println!(
        "BW11-1 window-union — {} trace file(s), {} routed experts, activation {:.2}% \
         (observed top_k mean over every captured position — R2: this ratio is only valid \
         at this activation fraction, do not reuse DEC-0's or K3's numbers here)",
        traces.len(),
        args.num_experts,
        activation_frac * 100.0
    );
    println!(
        "R6: this is a LOGICAL union ceiling. No kernel in this codebase groups multiple \
         tokens through one expert's weight read yet — a favourable ratio licenses building \
         one, it is not a bandwidth claim on its own."
    );

    let trace_refs: Vec<&[TraceRecord]> = traces.iter().map(|v| v.as_slice()).collect();

    let mut pulse_lines = Vec::new();
    for k in &ks {
        let mut pooled: Vec<LayerWindowUnion> = Vec::new();
        for recs in &traces {
            pooled.extend(window_union_by_layer(recs, *k, args.per_expert_bytes)?);
        }
        let spread = summarize_spread(&pooled)?;
        println!("{}", format_spread_row(&spread, activation_frac));

        let mut pulse = serde_json::json!({
            "step": *k,
            "dec/window_k": *k,
            "dec/activation_frac": activation_frac,
            "dec/num_experts": args.num_experts,
            "dec/window_union_frac_mean": spread.union_frac_mean,
            "dec/window_union_frac_median": spread.union_frac_median,
            "dec/window_union_frac_p10": spread.union_frac_p10,
            "dec/window_union_frac_p90": spread.union_frac_p90,
            "dec/window_amortisation_mean": spread.amortisation_mean,
            "dec/window_n_layers": spread.n_layers,
        });

        if args.shuffle_trials > 0 {
            let control = shuffled_control_by_layer(
                &trace_refs,
                *k,
                args.per_expert_bytes,
                args.shuffle_trials,
                args.shuffle_seed,
            )?;
            let control_spread = summarize_spread(&control)?;
            println!(
                "      R9 control (same marginals, sequence order destroyed, {} trials): \
                 union_frac mean={:.4} median={:.4}  -- {}",
                args.shuffle_trials,
                control_spread.union_frac_mean,
                control_spread.union_frac_median,
                if control_spread.union_frac_mean <= spread.union_frac_mean + 1e-9 {
                    "windowed ratio is NOT below the marginal-only control -- \
                     no evidence of same-sequence structure beyond popularity skew"
                } else {
                    "windowed ratio sits BELOW the marginal-only control -- \
                     consistent with genuine same-sequence structure, not just skew"
                }
            );
            pulse["dec/window_control_union_frac_mean"] =
                serde_json::json!(control_spread.union_frac_mean);
            pulse["dec/window_control_trials"] = serde_json::json!(args.shuffle_trials);
        }

        pulse_lines.push(pulse);
    }

    if let Some(path) = &args.pulse_file {
        let mut f = File::create(path)?;
        for line in &pulse_lines {
            writeln!(f, "{line}")?;
        }
        if args.verbose {
            eprintln!(
                "wrote {} pulse line(s) to {}",
                pulse_lines.len(),
                path.display()
            );
        }
    }

    Ok(())
}

fn parse_k_list(s: &str) -> Result<Vec<usize>, String> {
    let mut ks = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let k: usize = part
            .parse()
            .map_err(|_| format!("invalid --k entry {part:?}: expected a positive integer"))?;
        if k == 0 {
            return Err("--k entries must be >= 1".to_string());
        }
        ks.push(k);
    }
    if ks.is_empty() {
        return Err("--k must name at least one window width".to_string());
    }
    Ok(ks)
}

fn load_traces(paths: &[std::path::PathBuf]) -> Result<Vec<Vec<TraceRecord>>, String> {
    if paths.is_empty() {
        return Err("--trace must name at least one file".to_string());
    }
    paths
        .iter()
        .map(|p| {
            let f = File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            parse_trace(BufReader::new(f)).map_err(|e| format!("{}: {e}", p.display()))
        })
        .collect()
}

fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else {
        Some(xs.iter().sum::<f64>() / xs.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_trace(dir: &std::path::Path, name: &str, lines: &[&str]) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = File::create(&path).unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        path
    }

    #[test]
    fn parse_k_list_accepts_a_comma_list_and_trims_whitespace() {
        assert_eq!(parse_k_list("1, 2,4 ,8").unwrap(), vec![1, 2, 4, 8]);
    }

    #[test]
    fn parse_k_list_rejects_zero_and_empty() {
        assert!(parse_k_list("0").is_err());
        assert!(parse_k_list("").is_err());
        assert!(parse_k_list("1,x").is_err());
    }

    #[test]
    fn load_traces_reads_multiple_files_independently() {
        let dir = tempfile::tempdir().unwrap();
        let a = write_trace(
            dir.path(),
            "a.jsonl",
            &[
                "{\"layer\":0,\"seq\":0,\"experts\":[[1,2]]}",
                "{\"layer\":0,\"seq\":1,\"experts\":[[2,3]]}",
            ],
        );
        let b = write_trace(
            dir.path(),
            "b.jsonl",
            &["{\"layer\":0,\"seq\":0,\"experts\":[[5,6]]}"],
        );
        let traces = load_traces(&[a, b]).unwrap();
        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].len(), 2);
        assert_eq!(traces[1].len(), 1);
    }

    #[test]
    fn load_traces_names_the_file_on_a_parse_error() {
        let dir = tempfile::tempdir().unwrap();
        let bad = write_trace(dir.path(), "bad.jsonl", &["not json"]);
        let err = load_traces(std::slice::from_ref(&bad)).unwrap_err();
        assert!(err.contains("bad.jsonl"), "must name the file: {err}");
    }

    #[test]
    fn run_window_union_end_to_end_over_two_prompt_traces() {
        let dir = tempfile::tempdir().unwrap();
        // 4 positions per file, K in {1,2}; per_expert_bytes chosen so the
        // arithmetic is easy to eyeball if this ever needs debugging.
        let a = write_trace(
            dir.path(),
            "prompt-a.jsonl",
            &[
                "{\"layer\":0,\"seq\":0,\"experts\":[[0,1]]}",
                "{\"layer\":0,\"seq\":1,\"experts\":[[0,1]]}",
                "{\"layer\":0,\"seq\":2,\"experts\":[[2,3]]}",
                "{\"layer\":0,\"seq\":3,\"experts\":[[2,3]]}",
            ],
        );
        let b = write_trace(
            dir.path(),
            "prompt-b.jsonl",
            &[
                "{\"layer\":0,\"seq\":0,\"experts\":[[0,1]]}",
                "{\"layer\":0,\"seq\":1,\"experts\":[[2,3]]}",
                "{\"layer\":0,\"seq\":2,\"experts\":[[4,5]]}",
                "{\"layer\":0,\"seq\":3,\"experts\":[[6,7]]}",
            ],
        );
        let pulse = dir.path().join("pulse.jsonl");
        let args = WindowUnionArgs {
            trace: vec![a, b],
            k: "1,2".to_string(),
            num_experts: 8,
            per_expert_bytes: 100.0,
            shuffle_trials: 10,
            shuffle_seed: 1,
            pulse_file: Some(pulse.clone()),
            verbose: true,
        };
        run_window_union(&args).unwrap();
        let pulse_text = std::fs::read_to_string(&pulse).unwrap();
        let lines: Vec<&str> = pulse_text.lines().collect();
        assert_eq!(lines.len(), 2, "one pulse line per K");
        let k1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(k1["dec/window_k"], 1);
        assert_eq!(k1["dec/window_union_frac_mean"], 1.0, "K=1 is always 1.0");
        assert_eq!(k1["dec/window_control_trials"], 10);
        let k2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(k2["dec/window_k"], 2);
        // File a: windows (0,1)=identical union_frac 0.5, (1,2)=partial 0.75,
        // (2,3)=identical 0.5 -> pooled naive 12, union 3+3+2=... computed by
        // routed_weight_bytes_per_token directly; just assert it moved off 1.0
        // and is a valid fraction, the exact value is covered by
        // window_union.rs's own unit tests.
        let mean = k2["dec/window_union_frac_mean"].as_f64().unwrap();
        assert!(
            (0.0..1.0).contains(&mean),
            "some sharing must show up: {mean}"
        );
        let control_mean = k2["dec/window_control_union_frac_mean"].as_f64().unwrap();
        assert!(
            (0.0..=1.0).contains(&control_mean),
            "control must also be a valid fraction: {control_mean}"
        );
    }

    #[test]
    fn run_window_union_with_shuffle_trials_zero_skips_the_control() {
        let dir = tempfile::tempdir().unwrap();
        let a = write_trace(
            dir.path(),
            "prompt-a.jsonl",
            &[
                "{\"layer\":0,\"seq\":0,\"experts\":[[0,1]]}",
                "{\"layer\":0,\"seq\":1,\"experts\":[[2,3]]}",
            ],
        );
        let pulse = dir.path().join("pulse.jsonl");
        let args = WindowUnionArgs {
            trace: vec![a],
            k: "1".to_string(),
            num_experts: 8,
            per_expert_bytes: 100.0,
            shuffle_trials: 0,
            shuffle_seed: 1,
            pulse_file: Some(pulse.clone()),
            verbose: false,
        };
        run_window_union(&args).unwrap();
        let pulse_text = std::fs::read_to_string(&pulse).unwrap();
        let line: serde_json::Value =
            serde_json::from_str(pulse_text.lines().next().unwrap()).unwrap();
        assert!(
            line.get("dec/window_control_union_frac_mean").is_none(),
            "shuffle_trials=0 must not emit a control field"
        );
    }

    #[test]
    fn run_window_union_errs_when_a_layer_is_too_short_for_k() {
        let dir = tempfile::tempdir().unwrap();
        let a = write_trace(
            dir.path(),
            "short.jsonl",
            &["{\"layer\":0,\"seq\":0,\"experts\":[[0,1]]}"],
        );
        let args = WindowUnionArgs {
            trace: vec![a],
            k: "4".to_string(),
            num_experts: 8,
            per_expert_bytes: 100.0,
            shuffle_trials: 200,
            shuffle_seed: 1,
            pulse_file: None,
            verbose: false,
        };
        assert!(run_window_union(&args).is_err());
    }
}
