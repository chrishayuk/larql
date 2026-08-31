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
    env_dir, observation, run_sequence, sequence_embeddings, BANK_ENV, CANDIDATE_ENV, SOURCE_ENV,
};
use crate::format::vindex3::opplan::exec::kimi_source::{CandidateOverlay, KimiSourceModel};
use crate::format::vindex3::opplan::exec::stack_metal::{DeviceAttn, DeviceLayer, HybridStack};
use crate::format::vindex3::represent::bank::BankBuilder;
use crate::format::vindex3::represent::quality::{kimi_logit_v3, Criterion, QualityEvidence};

/// Which KDA layers' projections the candidate arm re-encodes — a
/// comma list (`"20,21,22,24,25"` is the late band; MLA positions are
/// refused by the requant itself). Default 25: the depth with the
/// richest expert-side evidence, so the recurrence-vs-router
/// comparison is at matched depth.
const LAYER_ENV: &str = "LARQL_KDA_Q8_LAYER";
const LAYER_DEFAULT: &str = "25";

fn target_layers() -> Vec<usize> {
    let spec = std::env::var(LAYER_ENV).unwrap_or_else(|_| LAYER_DEFAULT.into());
    if spec.trim().is_empty() {
        // A HEAD-ONLY probe: no layer scope at all.
        return Vec::new();
    }
    spec.split(',')
        .map(|v| v.trim().parse().expect("layer index"))
        .collect()
}

/// `LARQL_LMHEAD_Q8=1` re-encodes the candidate arm's OUTPUT HEAD to
/// Q8_0 — the head's perturbation lands directly at the logits, with
/// no downstream router or recurrence to mediate it, which makes this
/// probe a control against the interaction mechanisms measured
/// everywhere else as much as a lever measurement.
const LMHEAD_ENV: &str = "LARQL_LMHEAD_Q8";

fn head_q8() -> bool {
    std::env::var(LMHEAD_ENV).is_ok_and(|v| v == "1")
}
/// Diagnostic slice, same default as the expert probes.
const SEQUENCES_ENV: &str = "LARQL_Q2A_SEQUENCES";
const SEQUENCES_DEFAULT: usize = 8;
/// Null-arm sequences — the quality runner's own number.
const NULL_SEQUENCES: usize = 4;

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

/// Build one arm's layers; the target layer (if any) is re-encoded
/// BEFORE its attention banks are registered — a mutation after
/// registration would leave the residency declaration pointing at
/// freed bf16 bytes.
pub(super) fn build_layers(
    metal: &MetalBackend,
    model: &KimiSourceModel,
    mutate: &[usize],
    overlay: Option<&CandidateOverlay>,
) -> (Vec<Option<DeviceLayer>>, Vec<(usize, usize)>) {
    let n = model.geometry.num_layers;
    let mut swapped = Vec::new();
    let mut device: Vec<Option<DeviceLayer>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut d = model
            .device_layer(metal, i, overlay)
            .unwrap_or_else(|e| panic!("layer {i} must load: {e}"));
        if mutate.contains(&i) {
            swapped.push(requant_kda_projections(&mut d));
        }
        device.push(Some(d));
    }
    (device, swapped)
}

pub(super) fn assemble<'a>(
    metal: &MetalBackend,
    model: &KimiSourceModel,
    device: Vec<Option<DeviceLayer>>,
) -> HybridStack<'a> {
    assemble_with_head(metal, model, device, false)
}

/// `assemble`, optionally re-encoding the head to Q8_0. Returns with
/// the swap PROVEN: a candidate arm that silently kept the bf16 head
/// would measure nothing.
pub(super) fn assemble_with_head<'a>(
    metal: &MetalBackend,
    model: &KimiSourceModel,
    device: Vec<Option<DeviceLayer>>,
    q8_head: bool,
) -> HybridStack<'a> {
    for d in device.iter().flatten() {
        for bank in d.attention_banks() {
            metal.register_weight_region(bank);
        }
    }
    let host = (0..device.len()).map(|_| None).collect();
    let mut stack = HybridStack::new(device, host);
    let mut head = model.head().expect("the head must load");
    if q8_head {
        let before = head.weight.len();
        head.weight = quantize_q8_0(&widen_bf16(&head.weight));
        head.encoding = MetalEncoding::Q80;
        assert!(head.weight.len() < before, "the head swap did not happen");
        eprintln!(
            "[kda-q8] output head re-encoded Q8_0: {before} -> {} bytes ({:.1}%)",
            head.weight.len(),
            100.0 * head.weight.len() as f64 / before as f64
        );
    }
    assert!(stack.attach_head(head));
    stack
}

/// The attribution this probe's claim rests on: the arms differ at the
/// TARGET layer's two projections and NOWHERE else. Expert banks are
/// mmap-backed, so identity is the pointer; attention banks are owned
/// copies, so identity is the bytes.
fn assert_arms_differ_only_at(
    baseline: &[Option<DeviceLayer>],
    candidate: &[Option<DeviceLayer>],
    targets: &[usize],
    expert_layers: &[u32],
) {
    for (i, (b, c)) in baseline.iter().zip(candidate).enumerate() {
        let (b, c) = (
            b.as_ref().expect("all-device"),
            c.as_ref().expect("all-device"),
        );
        let compiled_here = expert_layers.contains(&(i as u32));
        for (name, pb, pc) in [
            ("gate", &b.bank.gate, &c.bank.gate),
            ("up", &b.bank.up, &c.bank.up),
            ("down", &b.bank.down, &c.bank.down),
        ] {
            if compiled_here {
                // The expert overlay substitutes this layer's routed
                // bank — the same asymmetry q2a proves, restated here
                // so the cross-family candidate's scope is explicit.
                assert_eq!(pb.store_id(), "kimi-source-expert-bank", "baseline {name}");
                assert_eq!(pc.store_id(), "kimi-candidate-bank", "candidate {name}");
            } else {
                assert_eq!(
                    pb.region.region.bytes().as_ptr(),
                    pc.region.region.bytes().as_ptr(),
                    "layer {i} {name}: both arms must bind ONE mmap expert region"
                );
            }
        }
        match (&b.attn, &c.attn) {
            (
                DeviceAttn::Kda {
                    qkv_bank: qb,
                    o_proj: ob,
                    encoding: eb,
                    ..
                },
                DeviceAttn::Kda {
                    qkv_bank: qc,
                    o_proj: oc,
                    encoding: ec,
                    ..
                },
            ) if targets.contains(&i) => {
                assert_eq!(*eb, MetalEncoding::Bf16, "baseline stays bf16");
                assert_eq!(*ec, MetalEncoding::Q80, "target layer is Q8_0");
                assert!(qc.len() < qb.len() && oc.len() < ob.len());
            }
            (
                DeviceAttn::Kda {
                    qkv_bank: qb,
                    o_proj: ob,
                    encoding: eb,
                    ..
                },
                DeviceAttn::Kda {
                    qkv_bank: qc,
                    o_proj: oc,
                    encoding: ec,
                    ..
                },
            ) => {
                assert_eq!((*eb, *ec), (MetalEncoding::Bf16, MetalEncoding::Bf16));
                assert!(
                    qb == qc && ob == oc,
                    "layer {i}: KDA banks must be byte-equal"
                );
            }
            (
                DeviceAttn::Mla {
                    q: qb,
                    kv_a: ab,
                    kv_b: bb,
                    o: ob,
                    ..
                },
                DeviceAttn::Mla {
                    q: qc,
                    kv_a: ac,
                    kv_b: bc,
                    o: oc,
                    ..
                },
            ) => {
                assert!(!targets.contains(&i), "every target must be a KDA layer");
                assert!(
                    qb == qc && ab == ac && bb == bc && ob == oc,
                    "layer {i}: MLA banks must be byte-equal"
                );
            }
            _ => panic!("layer {i}: the arms disagree on the attention operator"),
        }
    }
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
    let layers = target_layers();
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
    // Optional CROSS-FAMILY composition: an expert candidate (compiled
    // banks, q2a's own overlay machinery) beside the transient KDA
    // requant. The baseline arm binds neither.
    let overlay = env_dir(CANDIDATE_ENV).map(|dir| {
        let o = CandidateOverlay::open(&dir, &source_dir, &g).expect("candidate overlay opens");
        o.register_store(&metal);
        o
    });
    let expert_layers: Vec<u32> = overlay
        .as_ref()
        .map(|o| o.compiled_layers().to_vec())
        .unwrap_or_default();
    let q8h = head_q8();
    assert!(
        !layers.is_empty() || overlay.is_some() || q8h,
        "an empty scope with no overlay and no head flag measures nothing"
    );
    let (base_layers, none) = build_layers(&metal, &model, &[], None);
    let (cand_layers, swapped) = build_layers(&metal, &model, &layers, overlay.as_ref());
    assert!(none.is_empty());
    assert_eq!(swapped.len(), layers.len(), "every target was re-encoded");
    let bf16_bytes: usize = swapped.iter().map(|(b, _)| b).sum();
    let q8_bytes: usize = swapped.iter().map(|(_, q)| q).sum();
    assert!(
        layers.is_empty() || q8_bytes < bf16_bytes,
        "Q8_0 must be smaller than bf16 — the swap did not happen"
    );
    // Attribution BEFORE assembly: proven on the layers themselves, so
    // "identical except the targets' projections" is a checked fact,
    // not a construction argument.
    assert_arms_differ_only_at(&base_layers, &cand_layers, &layers, &expert_layers);
    let (null_layers, _) = build_layers(&metal, &model, &[], None);
    let mut baseline = assemble(&metal, &model, base_layers);
    let mut candidate = assemble_with_head(&metal, &model, cand_layers, q8h);
    let mut null_partner = assemble(&metal, &model, null_layers);
    metal.seal_weight_regions();
    eprintln!(
        "[kda-q8] arms loaded in {:.1}s; layers {layers:?} projections {bf16_bytes} -> \
         {q8_bytes} bytes ({:.1}% of bf16); every other bank byte-equal or \
         pointer-identical, PROVEN above",
        t0.elapsed().as_secs_f64(),
        100.0 * q8_bytes as f64 / bf16_bytes as f64,
    );

    // ── Null arm: BF16 against itself must be EXACTLY zero — the same
    // instrument-integrity gate the quality runner carries. ──
    {
        let mut builder = BankBuilder::new();
        for seq in 0..NULL_SEQUENCES {
            let rows = sequence_embeddings(&bank_dir, seq, positions, g.hidden);
            let a = run_sequence(&metal, &mut baseline, &rows, g.hidden);
            let b = run_sequence(&metal, &mut null_partner, &rows, g.hidden);
            for (pos, ((la, ta), (lb, tb))) in a.into_iter().zip(b).enumerate() {
                builder.observe(&observation(seq, pos, &la, &ta, &lb, &tb));
            }
        }
        let null_bank = builder.finish();
        assert_eq!(
            null_bank.logits.kl_p99, 0.0,
            "null arm KL must be exactly zero"
        );
        assert_eq!(null_bank.logits.max_logit_delta, 0.0);
        assert_eq!(null_bank.logits.top1_flips, 0);
        assert_eq!(null_bank.routing.route_flips, 0);
        eprintln!(
            "[kda-q8] null arm: {} positions, everything exactly zero",
            null_bank.positions
        );
    }

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
    let mut label = layers
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join("-");
    if let Some(o) = &overlay {
        label = format!("{label}-x-{}", o.index.map.name);
    }
    if q8h {
        label = if label.is_empty() || label == "" {
            "headq8".into()
        } else {
            format!("{label}-headq8")
        };
    }
    let report = serde_json::json!({
        "run": format!("kda-q8-l{label}"),
        "scope": "KDA projections (qkv bank + o_proj), transient requant; optional compiled expert candidate beside it",
        "expert_candidate": overlay.as_ref().map(|o| o.index.map.name.clone()),
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
    let path = format!("/tmp/kimi_kda-q8-l{label}_report.json");
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
