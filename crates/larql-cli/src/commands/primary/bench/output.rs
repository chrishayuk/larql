//! Table rendering for `larql bench`. The user-facing `print_table` is a
//! thin loop over `println!`; the actual line composition is split into
//! pure `Vec<String>`-returning helpers so each one is unit-testable.
//! JSON output lives next to the orchestration in `run.rs` because it uses
//! the `BenchJsonResult` envelope.

use super::row::BenchRow;

const WIRE_COL_WIDTH: usize = 100;
const PLAIN_COL_WIDTH: usize = 85;

pub(super) fn print_table(rows: &[BenchRow]) {
    for line in render_table(rows) {
        println!("{line}");
    }
}

/// Pure renderer: returns the full table as a sequence of lines.
pub(super) fn render_table(rows: &[BenchRow]) -> Vec<String> {
    let has_wire = rows.iter().any(|r| r.wire_bytes_per_tok.is_some());
    let mut out = Vec::new();
    out.push(format_header_line(has_wire));
    out.push(format_separator(has_wire));
    for r in rows {
        out.push(format_data_row(r, has_wire));
    }
    out.extend(format_stage_breakdown(rows));
    out.extend(format_remote_ffn_breakdown(rows));
    out.extend(format_metal_vs_ollama_summary(rows));
    out
}

pub(super) fn format_header_line(has_wire: bool) -> String {
    let wire_col = if has_wire { "  wire_KB/tok" } else { "" };
    format!(
        "  {:<24} {:>10} {:>10} {:>10} {:>10} {:>6}{wire_col}  notes",
        "Backend", "prefill", "mean", "p50", "tok/s", "steps",
    )
}

pub(super) fn format_separator(has_wire: bool) -> String {
    let n = if has_wire {
        WIRE_COL_WIDTH
    } else {
        PLAIN_COL_WIDTH
    };
    format!("  {}", "─".repeat(n))
}

pub(super) fn format_data_row(r: &BenchRow, has_wire: bool) -> String {
    let wire_part = if has_wire {
        match r.wire_bytes_per_tok {
            Some(b) => format!("  {:>10.1}", b as f64 / 1024.0),
            None => "             ".to_string(),
        }
    } else {
        String::new()
    };
    format!(
        "  {:<24} {:>9.1}ms {:>9.2}ms {:>9.2}ms {:>9.1}  {:>6}{wire_part}  {}",
        r.backend, r.prefill_ms, r.avg_decode_ms, r.p50_ms, r.tok_per_s, r.n_steps, r.note,
    )
}

pub(super) fn format_stage_breakdown(rows: &[BenchRow]) -> Vec<String> {
    let Some(r) = rows.iter().find(|r| r.stages.is_some()) else {
        return Vec::new();
    };
    let s = r.stages.unwrap();
    let total = s.embed_ms_total
        + s.gpu_ms_total
        + s.cpu_fwd_ms_total
        + s.norm_ms_total
        + s.lm_head_ms_total
        + s.detok_ms_total;
    if total <= 0.0 {
        return Vec::new();
    }
    let pct = |v: f64| (v / total) * 100.0;
    let mut out = vec![
        String::new(),
        format!("  Per-stage average ({}):", r.backend),
        format!(
            "    embed     {:>6.3}ms  ({:>4.1}%)",
            s.embed_ms_total,
            pct(s.embed_ms_total)
        ),
    ];
    if s.cpu_fwd_ms_total > 0.0 {
        out.push(format!(
            "    CPU fwd   {:>6.3}ms  ({:>4.1}%)",
            s.cpu_fwd_ms_total,
            pct(s.cpu_fwd_ms_total)
        ));
        if s.dequant_ms_total > 0.0 {
            // Dequant is a slice of cpu_fwd time, not additive — show it
            // indented so the table still sums to 100%.
            out.push(format!(
                "      dequant {:>6.3}ms  ({:>4.1}% of cpu_fwd)",
                s.dequant_ms_total,
                if s.cpu_fwd_ms_total > 0.0 {
                    s.dequant_ms_total / s.cpu_fwd_ms_total * 100.0
                } else {
                    0.0
                }
            ));
        }
    }
    if s.gpu_ms_total > 0.0 {
        out.push(format!(
            "    GPU fwd   {:>6.3}ms  ({:>4.1}%)",
            s.gpu_ms_total,
            pct(s.gpu_ms_total)
        ));
    }
    if s.gate_up_ms_total > 0.0 {
        out.push(format!(
            "      gate+up {:>6.3}ms  ({:>4.1}%)",
            s.gate_up_ms_total,
            pct(s.gate_up_ms_total)
        ));
        out.push(format!(
            "      act+down{:>6.3}ms  ({:>4.1}%)",
            s.down_ms_total,
            pct(s.down_ms_total)
        ));
    }
    out.push(format!(
        "    final_norm{:>6.3}ms  ({:>4.1}%)",
        s.norm_ms_total,
        pct(s.norm_ms_total)
    ));
    out.push(format!(
        "    lm_head   {:>6.3}ms  ({:>4.1}%)",
        s.lm_head_ms_total,
        pct(s.lm_head_ms_total)
    ));
    out.push(format!(
        "    detok     {:>6.3}ms  ({:>4.1}%)",
        s.detok_ms_total,
        pct(s.detok_ms_total)
    ));
    out
}

pub(super) fn format_remote_ffn_breakdown(rows: &[BenchRow]) -> Vec<String> {
    let Some(r) = rows.iter().find(|r| r.ffn_rtt_ms.is_some()) else {
        return Vec::new();
    };
    let ffn = r.ffn_rtt_ms.unwrap();
    let attn = r.attn_ms.unwrap_or(r.avg_decode_ms);
    let total = r.avg_decode_ms;
    let pct = |v: f64| {
        if total > 0.0 {
            (v / total) * 100.0
        } else {
            0.0
        }
    };
    vec![
        String::new(),
        format!(
            "  Per-stage average (remote-ffn, {} layers × RTT):",
            r.backend.split('(').next().unwrap_or("").trim()
        ),
        format!(
            "    attn+norm+lmhead {:>7.2}ms  ({:>4.1}%)",
            attn,
            pct(attn)
        ),
        format!(
            "    ffn round-trips  {:>7.2}ms  ({:>4.1}%)  ← remote",
            ffn,
            pct(ffn)
        ),
        format!(
            "    total/tok        {:>7.2}ms  →  {:.1} tok/s",
            total, r.tok_per_s
        ),
    ]
}

pub(super) fn format_metal_vs_ollama_summary(rows: &[BenchRow]) -> Vec<String> {
    let metal = rows
        .iter()
        .find(|r| r.backend == "larql-metal" && r.tok_per_s > 0.0);
    let ollama = rows
        .iter()
        .find(|r| r.backend.starts_with("ollama") && r.tok_per_s > 0.0);
    let (Some(m), Some(o)) = (metal, ollama) else {
        return Vec::new();
    };
    let ratio = m.tok_per_s / o.tok_per_s;
    let (verb, sign) = if ratio >= 1.0 {
        ("faster", '>')
    } else {
        ("slower", '<')
    };
    vec![
        String::new(),
        format!(
            "  → larql-metal is {:.2}× {} {} ollama ({:.1} {} {:.1} tok/s)",
            if ratio >= 1.0 { ratio } else { 1.0 / ratio },
            verb,
            sign,
            m.tok_per_s,
            sign,
            o.tok_per_s,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_row(backend: &str, tok: f64) -> BenchRow {
        BenchRow {
            backend: backend.to_string(),
            prefill_ms: 0.0,
            avg_decode_ms: 0.0,
            p50_ms: 0.0,
            p99_ms: 0.0,
            tok_per_s: tok,
            stages: None,
            ffn_rtt_ms: None,
            attn_ms: None,
            wire_bytes_per_tok: None,
            shard_efficiency: None,
            n_steps: 0,
            note: String::new(),
        }
    }

    #[test]
    fn header_has_wire_column_only_when_requested() {
        let with = format_header_line(true);
        let without = format_header_line(false);
        assert!(with.contains("wire_KB/tok"));
        assert!(!without.contains("wire_KB/tok"));
        // Standard columns always present.
        for s in &[with.as_str(), without.as_str()] {
            assert!(s.contains("Backend"));
            assert!(s.contains("prefill"));
            assert!(s.contains("tok/s"));
            assert!(s.contains("steps"));
        }
    }

    #[test]
    fn separator_widens_with_wire_column() {
        let plain = format_separator(false);
        let wide = format_separator(true);
        // The separator string uses the box-drawing char `─`. Each `─` is
        // three bytes in UTF-8; checking the char-count via chars() is what
        // we mean.
        let plain_chars = plain.chars().filter(|&c| c == '─').count();
        let wide_chars = wide.chars().filter(|&c| c == '─').count();
        assert_eq!(plain_chars, PLAIN_COL_WIDTH);
        assert_eq!(wide_chars, WIRE_COL_WIDTH);
    }

    #[test]
    fn data_row_shows_wire_kb_only_when_present() {
        let mut r = empty_row("test", 10.0);
        r.wire_bytes_per_tok = Some(2048);
        let with = format_data_row(&r, true);
        assert!(with.contains("2.0"), "expected wire KB: {with}");

        r.wire_bytes_per_tok = None;
        let blank = format_data_row(&r, true);
        // Missing wire bytes leave the column blank, not "0.0".
        assert!(!blank.contains("0.0  test"));
    }

    #[test]
    fn data_row_without_wire_column_omits_kb_field() {
        let r = empty_row("test", 10.0);
        let line = format_data_row(&r, false);
        // No "wire_KB" formatting at the end.
        assert!(line.trim_end().ends_with("test") || line.contains("0  "));
        // Just confirm it includes the backend label.
        assert!(line.contains("test"));
    }

    #[test]
    fn stage_breakdown_empty_when_no_stages() {
        let rows = vec![empty_row("x", 1.0)];
        assert!(format_stage_breakdown(&rows).is_empty());
    }

    #[test]
    fn stage_breakdown_skips_when_total_is_zero() {
        let mut r = empty_row("x", 1.0);
        // Zero-everything StageTimings should be skipped.
        r.stages = Some(larql_inference::layer_graph::generate::StageTimings::default());
        let rows = vec![r];
        assert!(format_stage_breakdown(&rows).is_empty());
    }

    #[test]
    fn stage_breakdown_renders_when_stages_present() {
        let mut r = empty_row("metal", 1.0);
        let stages = larql_inference::layer_graph::generate::StageTimings {
            embed_ms_total: 1.0,
            gpu_ms_total: 10.0,
            gate_up_ms_total: 4.0,
            down_ms_total: 3.0,
            norm_ms_total: 0.5,
            lm_head_ms_total: 1.5,
            detok_ms_total: 0.2,
            ..Default::default()
        };
        r.stages = Some(stages);
        let lines = format_stage_breakdown(&[r]);
        // Section header + blank line + 8 rows (embed, gpu, gate+up, act+down, final_norm, lm_head, detok)
        assert!(!lines.is_empty());
        assert!(lines
            .iter()
            .any(|l| l.contains("Per-stage average (metal)")));
        assert!(lines.iter().any(|l| l.contains("embed")));
        assert!(lines.iter().any(|l| l.contains("gate+up")));
        assert!(lines.iter().any(|l| l.contains("final_norm")));
    }

    #[test]
    fn stage_breakdown_labels_cpu_fallback_forward() {
        let mut r = empty_row("larql-cpu", 1.0);
        let stages = larql_inference::layer_graph::generate::StageTimings {
            cpu_fwd_ms_total: 20.0,
            lm_head_ms_total: 2.0,
            ..Default::default()
        };
        r.stages = Some(stages);
        let lines = format_stage_breakdown(&[r]);
        assert!(lines.iter().any(|l| l.contains("CPU fwd")));
        assert!(!lines.iter().any(|l| l.contains("GPU fwd")));
        assert!(lines.iter().any(|l| l.contains("lm_head")));
    }

    #[test]
    fn stage_breakdown_omits_gate_up_when_zero() {
        let mut r = empty_row("metal", 1.0);
        let stages = larql_inference::layer_graph::generate::StageTimings {
            gpu_ms_total: 10.0,
            gate_up_ms_total: 0.0,
            ..Default::default()
        };
        r.stages = Some(stages);
        let lines = format_stage_breakdown(&[r]);
        assert!(!lines.iter().any(|l| l.contains("gate+up")));
    }

    #[test]
    fn remote_ffn_breakdown_empty_when_no_ffn_data() {
        let rows = vec![empty_row("x", 1.0)];
        assert!(format_remote_ffn_breakdown(&rows).is_empty());
    }

    #[test]
    fn remote_ffn_breakdown_renders_when_present() {
        let mut r = empty_row("remote-ffn (http://a)", 10.0);
        r.avg_decode_ms = 100.0;
        r.ffn_rtt_ms = Some(80.0);
        r.attn_ms = Some(20.0);
        let lines = format_remote_ffn_breakdown(&[r]);
        assert!(!lines.is_empty());
        assert!(lines
            .iter()
            .any(|l| l.contains("Per-stage average (remote-ffn")));
        assert!(lines.iter().any(|l| l.contains("ffn round-trips")));
    }

    #[test]
    fn remote_ffn_breakdown_uses_avg_decode_when_attn_missing() {
        let mut r = empty_row("remote-ffn (x)", 5.0);
        r.avg_decode_ms = 200.0;
        r.ffn_rtt_ms = Some(150.0);
        r.attn_ms = None;
        let lines = format_remote_ffn_breakdown(&[r]);
        // attn fallback should be the total decode value.
        assert!(lines.iter().any(|l| l.contains("200.00")));
    }

    #[test]
    fn metal_vs_ollama_summary_silent_when_either_missing() {
        let only_metal = vec![{
            let mut r = empty_row("larql-metal", 100.0);
            r.tok_per_s = 100.0;
            r
        }];
        assert!(format_metal_vs_ollama_summary(&only_metal).is_empty());

        let only_ollama = vec![{
            let mut r = empty_row("ollama gemma3:4b", 50.0);
            r.tok_per_s = 50.0;
            r
        }];
        assert!(format_metal_vs_ollama_summary(&only_ollama).is_empty());
    }

    #[test]
    fn metal_vs_ollama_summary_renders_when_both_present() {
        let rows = vec![
            {
                let mut r = empty_row("larql-metal", 80.0);
                r.tok_per_s = 80.0;
                r
            },
            {
                let mut r = empty_row("ollama gemma3:4b", 40.0);
                r.tok_per_s = 40.0;
                r
            },
        ];
        let lines = format_metal_vs_ollama_summary(&rows);
        assert!(!lines.is_empty());
        let summary = lines.last().unwrap();
        // 80 / 40 = 2.0x faster
        assert!(summary.contains("2.00×"), "got: {summary}");
        assert!(summary.contains("faster"));
    }

    #[test]
    fn metal_vs_ollama_summary_inverts_when_metal_slower() {
        let rows = vec![
            {
                let mut r = empty_row("larql-metal", 40.0);
                r.tok_per_s = 40.0;
                r
            },
            {
                let mut r = empty_row("ollama gemma3:4b", 80.0);
                r.tok_per_s = 80.0;
                r
            },
        ];
        let summary = format_metal_vs_ollama_summary(&rows)
            .last()
            .cloned()
            .unwrap();
        // Should still be displayed as 2.00× slower.
        assert!(summary.contains("2.00×"), "got: {summary}");
        assert!(summary.contains("slower"));
    }

    #[test]
    fn render_table_includes_header_separator_and_all_rows() {
        let rows = vec![empty_row("a", 1.0), empty_row("b", 2.0)];
        let lines = render_table(&rows);
        // header + separator + 2 rows = 4 lines minimum
        assert!(lines.len() >= 4);
        assert!(lines[0].contains("Backend"));
        assert!(lines[1].chars().any(|c| c == '─'));
        assert!(lines[2].contains("a"));
        assert!(lines[3].contains("b"));
    }
}
