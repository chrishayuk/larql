//! **KDA-1B rung 1b — one real KDA layer's projections at Q8_0, judged
//! by the same teacher-forced consequence metrics as the expert cells.**
//!
//! The question is scientific before it is economic: does quantising a
//! projection that FEEDS A RECURRENCE behave like quantising an expert
//! matrix (error absorbed or cascaded through later routers), or does
//! recurrent carry amplify representation error into a different
//! failure class? The expert cell at the same depth is the yardstick —
//! L25 routed experts at Q8_0 measured kl p99 2.3e-4 on this bank.
//!
//! The candidate arm is a TRANSIENT requant: the container's own bf16
//! bytes for the target layer's `q|k|v` bank and `o_proj` are widened
//! and re-encoded as Q8_0 at load, then dispatched through the real
//! quantised grouped kernel. No compile machinery, no overlay — that
//! machinery is earned only if this lever proves behaviourally cheap.
//! Everything else in both arms is pointer-identical by construction:
//! the ONLY difference is the two projections' physical representation.
//!
//! ```text
//! LARQL_KIMI_VINDEX3=~/chris-models/Kimi-Linear-48B-A3B-Instruct.aligned.vindex3 \
//! LARQL_KIMI_QUALITY_BANK=~/chris-models/qbanks/kimi-quality-bank-256x32 \
//! LARQL_KDA_Q8_LAYER=25 LARQL_Q2A_SEQUENCES=8 \
//!   cargo test -p larql-vindex --features gpu --release --lib kda_q8_real -- --nocapture
//! ```

use std::time::Instant;

use larql_compute::backend::ComputeBackend;
use larql_compute::cpu::ops::q4_common::quantize_q8_0;
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kimi_layer::ExpertEncoding as MetalEncoding;
use larql_compute_metal::MetalBackend;
use serde_json::Value;

use super::q2a_teacher_forced::{
    env_dir, observation, run_sequence, sequence_embeddings, BANK_ENV, SOURCE_ENV,
};
use crate::format::vindex3::opplan::exec::kimi_source::KimiSourceModel;
use crate::format::vindex3::opplan::exec::stack_metal::{DeviceAttn, DeviceLayer, HybridStack};
use crate::format::vindex3::represent::bank::BankBuilder;
use crate::format::vindex3::represent::quality::{kimi_logit_v3, Criterion, QualityEvidence};

/// Which KDA layer's projections the candidate arm re-encodes.
/// Default 25: the depth with the richest expert-side evidence, so the
/// recurrence-vs-router comparison is at matched depth.
const LAYER_ENV: &str = "LARQL_KDA_Q8_LAYER";
const LAYER_DEFAULT: usize = 25;
/// Diagnostic slice, same default as the expert probes.
const SEQUENCES_ENV: &str = "LARQL_Q2A_SEQUENCES";
const SEQUENCES_DEFAULT: usize = 8;

fn env_count(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Exact KL(baseline ‖ candidate) over the FULL vocabulary, one
/// position. The bank's own KL is authority; this exists for the
/// TOKEN-DISTANCE curve — each sequence starts from clean recurrent
/// state, so position index IS distance from the perturbation's onset,
/// and the curve answers the question this probe was built for: does
/// recurrent carry make projection error decay, hold, or accumulate?
fn full_kl(baseline: &[f32], candidate: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|x| (*x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (zb, zc) = (lse(baseline), lse(candidate));
    baseline
        .iter()
        .zip(candidate)
        .map(|(b, c)| {
            let pb = (*b as f64 - zb).exp();
            pb * ((*b as f64 - zb) - (*c as f64 - zc))
        })
        .sum()
}

fn widen_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Re-encode the layer's two wide projections in place. Returns the
/// (bf16, q8) byte counts so the caller can assert the swap really
/// happened — an arm that silently kept bf16 would measure nothing.
fn requant_kda_projections(layer: &mut DeviceLayer) -> (usize, usize) {
    let shape = layer.kda_shape;
    let (width, hidden) = (shape.num_heads * shape.head_dim, shape.hidden);
    let DeviceAttn::Kda {
        qkv_bank,
        qkv_offsets,
        o_proj,
        encoding,
        ..
    } = &mut layer.attn
    else {
        panic!("the target layer must be KDA — pick one from the interleave");
    };
    let per = width * hidden * 2;
    let before = qkv_bank.len() + o_proj.len();
    let mut q8 = Vec::new();
    let mut offsets = [ExpertOffset(0); 3];
    for (slot, off) in qkv_offsets.iter().enumerate() {
        let start = off.0 as usize;
        offsets[slot] = ExpertOffset(q8.len() as u32);
        q8.extend(quantize_q8_0(&widen_bf16(&qkv_bank[start..start + per])));
    }
    *qkv_bank = q8;
    *qkv_offsets = offsets;
    *o_proj = quantize_q8_0(&widen_bf16(o_proj));
    *encoding = MetalEncoding::Q80;
    (before, layer_bytes(layer))
}

fn layer_bytes(layer: &DeviceLayer) -> usize {
    match &layer.attn {
        DeviceAttn::Kda {
            qkv_bank, o_proj, ..
        } => qkv_bank.len() + o_proj.len(),
        _ => unreachable!("asserted KDA above"),
    }
}

/// `build_stack`, with one hook: the target layer's projections are
/// re-encoded BEFORE its attention banks are registered — a mutation
/// after registration would leave the residency declaration pointing at
/// freed bf16 bytes.
fn build_stack_with<'a>(
    metal: &MetalBackend,
    model: &KimiSourceModel,
    mutate: Option<usize>,
) -> (HybridStack<'a>, Option<(usize, usize)>) {
    let n = model.geometry.num_layers;
    let mut swapped = None;
    let mut device: Vec<Option<DeviceLayer>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut d = model
            .device_layer(metal, i, None)
            .unwrap_or_else(|e| panic!("layer {i} must load: {e}"));
        if mutate == Some(i) {
            swapped = Some(requant_kda_projections(&mut d));
        }
        device.push(Some(d));
    }
    for d in device.iter().flatten() {
        for bank in d.attention_banks() {
            metal.register_weight_region(bank);
        }
    }
    let host = (0..n).map(|_| None).collect();
    let mut stack = HybridStack::new(device, host);
    assert!(stack.attach_head(model.head().expect("the head must load")));
    (stack, swapped)
}

#[test]
fn one_kda_layers_projections_at_q8_through_the_consequence_metrics() {
    let (Some(source_dir), Some(bank_dir)) = (env_dir(SOURCE_ENV), env_dir(BANK_ENV)) else {
        eprintln!("skipped: set {SOURCE_ENV} and {BANK_ENV}");
        return;
    };
    if std::env::var("LARQL_RESIDENCY_SET").is_ok() {
        panic!("unset LARQL_RESIDENCY_SET: this run must use implicit residency");
    }
    let Some(metal) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let layer = env_count(LAYER_ENV, LAYER_DEFAULT);
    let sequences = env_count(SEQUENCES_ENV, SEQUENCES_DEFAULT);

    let manifest: Value =
        serde_json::from_slice(&std::fs::read(bank_dir.join("manifest.json")).expect("manifest"))
            .expect("bank manifest parses");
    let positions = manifest["positions"].as_u64().unwrap() as usize;

    let t0 = Instant::now();
    let model = KimiSourceModel::open(&source_dir).expect("source container opens");
    let g = model.geometry.clone();
    let moe_layers: Vec<u32> = (g.dense_prefix_layers..g.num_layers)
        .map(|l| l as u32)
        .collect();
    model
        .register_stores(&metal, &moe_layers)
        .expect("stores register");
    let (mut baseline, none) = build_stack_with(&metal, &model, None);
    let (mut candidate, swapped) = build_stack_with(&metal, &model, Some(layer));
    metal.seal_weight_regions();
    assert!(none.is_none());
    let (bf16_bytes, q8_bytes) = swapped.expect("the target layer was re-encoded");
    assert!(
        q8_bytes < bf16_bytes,
        "Q8_0 must be smaller than bf16 — the swap did not happen"
    );
    eprintln!(
        "[kda-q8] arms loaded in {:.1}s; layer {layer} projections {bf16_bytes} -> \
         {q8_bytes} bytes ({:.1}% of bf16); everything else identical by construction",
        t0.elapsed().as_secs_f64(),
        100.0 * q8_bytes as f64 / bf16_bytes as f64,
    );

    let t1 = Instant::now();
    let mut builder = BankBuilder::new();
    // KL by position index, across sequences — the token-distance curve.
    let mut kl_by_pos: Vec<Vec<f64>> = vec![Vec::new(); positions];
    for seq in 0..sequences {
        let rows = sequence_embeddings(&bank_dir, seq, positions, g.hidden);
        let base = run_sequence(&metal, &mut baseline, &rows, g.hidden);
        let cand = run_sequence(&metal, &mut candidate, &rows, g.hidden);
        for (pos, ((lb, tb), (lc, tc))) in base.into_iter().zip(cand).enumerate() {
            kl_by_pos[pos].push(full_kl(&lb, &lc));
            builder.observe(&observation(seq, pos, &lb, &tb, &lc, &tc));
        }
    }
    let min_covered = builder.min_covered_mass();
    let bank = builder.finish();
    let curve: Vec<serde_json::Value> = kl_by_pos
        .iter()
        .map(|v| {
            let mut s = v.clone();
            s.sort_by(|a, b| a.partial_cmp(b).expect("KL is finite"));
            let mean = s.iter().sum::<f64>() / s.len() as f64;
            serde_json::json!({"mean": mean, "max": s[s.len() - 1]})
        })
        .collect();
    // Quartile means of the curve, so decay/hold/accumulate reads off
    // one line without a plot.
    let q = positions / 4;
    let qmean = |r: std::ops::Range<usize>| {
        let vals: Vec<f64> = r.flat_map(|p| kl_by_pos[p].iter().copied()).collect();
        vals.iter().sum::<f64>() / vals.len() as f64
    };
    eprintln!(
        "[kda-q8] token-distance KL means by quartile of position 0..{positions}: \
         {:.3e} | {:.3e} | {:.3e} | {:.3e}",
        qmean(0..q),
        qmean(q..2 * q),
        qmean(2 * q..3 * q),
        qmean(3 * q..positions),
    );

    let evidence = QualityEvidence {
        gate: kimi_logit_v3(),
        bank: bank.clone(),
    };
    let verdict = evidence.verdict();
    let bank_manifest_sha256 = {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(bank_dir.join("manifest.json")).expect("bank manifest reads");
        format!("{:x}", Sha256::digest(&bytes))
    };
    let report = serde_json::json!({
        "run": format!("kda-q8-l{layer}"),
        "scope": "KDA projections (qkv bank + o_proj) of ONE layer, transient requant — no compiled candidate",
        "gate": evidence.gate,
        "authority_report": evidence.report(),
        "verdict_passed": verdict.passed(),
        "verdict_failures": verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "bank_manifest_sha256": bank_manifest_sha256,
        "bank": bank,
        "positions": bank.positions,
        "min_covered_mass": min_covered,
        "bytes": {"bf16": bf16_bytes, "q8_0": q8_bytes},
        "kl_by_position": curve,
        "wall_seconds": t1.elapsed().as_secs_f64(),
    });
    let path = format!("/tmp/kimi_kda-q8-l{layer}_report.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report).expect("serialises"),
    )
    .expect("report writes");
    eprintln!("{}", evidence.report());
    eprintln!("[kda-q8] verdict: {verdict:?} — report at {path}");

    if bank.positions < 4096 {
        assert!(!verdict.passed(), "sub-4096 positions can never pass v3");
        assert!(verdict
            .failures
            .iter()
            .any(|(c, d)| *c == Criterion::Positions && d.contains("< 4096")));
    }
}
