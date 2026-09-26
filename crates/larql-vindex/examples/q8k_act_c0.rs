//! q8k_act_c0 — Q8K-ACT-1 characterisation C0-a, run BEFORE the arm exists.
//!
//! What error does quantising the activation to Q8_K introduce into a
//! K-quant projection, on real weights and real activations? Only existing
//! kernels run here, so the answer is blind to the new arm:
//!
//! ```text
//! reference   codec.gemv(blocks, x)                    f32 activation (production-q4k today)
//! candidate   quantize_x_to_q8k(x) → q4k_q8k_matvec_parallel   the kernel V2's CPU decode uses
//! ```
//!
//! Both read the same stored blocks with the same captured `x`, so the
//! difference is the activation form alone. Per call it reports the
//! relative L2 error `‖y_c − y_r‖ / ‖y_r‖` and the worst element error
//! normalised by the output's RMS (an elementwise relative error would
//! blow up on near-zero outputs and read noise as signal).
//!
//! Frozen protocol and the rule that turns this into a tolerance:
//! `docs/q8k-act-1.md`.
//!
//! ```text
//! cargo run --release -p larql-vindex --example q8k_act_c0 -- \
//!   <container> --prompt 2,105,... --generate 24 --capture 8
//! ```

use std::path::Path;

use larql_compute::cpu::ops::q4k_q8k_dot::{q4k_q8k_matvec_parallel, quantize_x_to_q8k};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::cpu::replay::{start_capture, take_capture};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::plan_component_ops;
use larql_vindex::format::vindex3::represent::kquant;

/// The K-quant members the Q8_K route covers — the formats with a
/// `q8k_matvec` kernel in `larql-compute`.
const Q8K_MEMBERS: [&str; 2] = ["Q4_K", "Q6_K"];

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn ids(s: &str) -> Vec<u32> {
    s.split(',')
        .map(|t| t.trim().parse().expect("token id"))
        .collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// One call's error, and which member it was.
struct Error {
    member: &'static str,
    rel_l2: f64,
    max_over_rms: f64,
    /// Instrument control: the Q8_K kernel against the f32 kernel run on
    /// the Q8_K-DEQUANTISED activation. Near rounding means `rel_l2` is
    /// the activation's quantisation error and nothing else.
    control_rel_l2: f64,
}

/// Q8_K super-block width: one f32 scale per 256 activations.
const Q8K_BLOCK: usize = 256;

fn dequantise(q: &larql_compute::cpu::ops::q4k_q8k_dot::Q8KActivation) -> Vec<f32> {
    q.qs.iter()
        .enumerate()
        .map(|(i, &v)| f32::from(v) * q.d[i / Q8K_BLOCK])
        .collect()
}

fn compare(reference: &[f32], candidate: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for (&r, &c) in reference.iter().zip(candidate) {
        let d = f64::from(c) - f64::from(r);
        num += d * d;
        den += f64::from(r) * f64::from(r);
        worst = worst.max(d.abs());
    }
    let rms = (den / reference.len() as f64).sqrt();
    (num.sqrt() / den.sqrt(), worst / rms)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let container = args
        .first()
        .cloned()
        .ok_or("usage: q8k_act_c0 <container> --prompt ids --generate N --capture K")?;
    let prompt = ids(&flag(&args, "--prompt").ok_or("--prompt is required")?);
    let generate: usize = flag(&args, "--generate").map_or(24, |v| v.parse().expect("integer"));
    let capture: usize = flag(&args, "--capture").map_or(8, |v| v.parse().expect("integer"));
    assert!(capture <= generate, "--capture must not exceed --generate");

    let root = Path::new(&container);
    let inspection = inspect_container(root, false)?;
    let plan = plan_component_ops(&inspection, root, "target")?
        .plan
        .ok_or("component `target` produced no plan")?;
    // Stored: the characterisation must read the compiled pack, never a
    // representation manufactured at load.
    let store = OperandStore::open_for(
        root,
        &inspection,
        Some(kquant::Q4_K.name),
        RepresentationSource::Stored,
    )?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;

    let mut logits = Vec::new();
    for &t in &prompt {
        logits = session.step(t)?.logits.ok_or("no output head")?;
    }
    let mut generated = Vec::with_capacity(generate);
    let mut errors = Vec::new();
    for step in 0..generate {
        let next = argmax(&logits);
        generated.push(next);
        let capturing = step >= generate - capture;
        if capturing {
            start_capture();
        }
        logits = session.step(next)?.logits.ok_or("no output head")?;
        if !capturing {
            continue;
        }
        for call in take_capture() {
            // SAFETY: `session` holds the operands for this whole loop.
            let Some((blocks, codec)) = (unsafe { call.kquant() }) else {
                continue;
            };
            let Some(member) = Q8K_MEMBERS.iter().copied().find(|m| *m == codec.name) else {
                continue;
            };
            let x = call.x();
            let rows = call.out_dim();
            let reference = codec
                .gemv(blocks, x, rows, x.len())
                .ok_or("reference kernel refused the captured geometry")?;
            let q = quantize_x_to_q8k(x);
            let mut candidate = vec![0.0f32; rows];
            q4k_q8k_matvec_parallel(&mut candidate, &q, blocks, rows, x.len(), member)?;
            let (rel_l2, max_over_rms) = compare(&reference, &candidate);
            let control = codec
                .gemv(blocks, &dequantise(&q), rows, x.len())
                .ok_or("reference kernel refused the dequantised activation")?;
            let (control_rel_l2, _) = compare(&control, &candidate);
            errors.push(Error {
                member,
                rel_l2,
                max_over_rms,
                control_rel_l2,
            });
        }
    }

    println!("q8k_act_c0 — Q8K-ACT-1 C0-a (docs/q8k-act-1.md)");
    println!("  container {container}");
    println!(
        "  prompt {} ids, generated {generate}, captured last {capture} steps",
        prompt.len()
    );
    println!(
        "  generated ids: {}",
        generated
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "  LARQL_Q4K_ASM={}",
        std::env::var("LARQL_Q4K_ASM").unwrap_or_else(|_| "unset".into())
    );
    if errors.is_empty() {
        return Err("no Q4_K/Q6_K projection was captured — nothing to characterise".into());
    }
    for member in Q8K_MEMBERS {
        let mut rel: Vec<f64> = errors
            .iter()
            .filter(|e| e.member == member)
            .map(|e| e.rel_l2)
            .collect();
        if rel.is_empty() {
            println!("  {member}: no calls");
            continue;
        }
        let mut worst: Vec<f64> = errors
            .iter()
            .filter(|e| e.member == member)
            .map(|e| e.max_over_rms)
            .collect();
        let mut control: Vec<f64> = errors
            .iter()
            .filter(|e| e.member == member)
            .map(|e| e.control_rel_l2)
            .collect();
        rel.sort_by(f64::total_cmp);
        worst.sort_by(f64::total_cmp);
        control.sort_by(f64::total_cmp);
        println!(
            "  {member}: {} calls  rel_L2 median {:.3e}  p99 {:.3e}  max {:.3e}  |  max|Δ|/rms median {:.3e}  max {:.3e}",
            rel.len(),
            quantile(&rel, 0.5),
            quantile(&rel, 0.99),
            rel[rel.len() - 1],
            quantile(&worst, 0.5),
            worst[worst.len() - 1],
        );
        println!(
            "  {member} control (Q8_K kernel vs f32 kernel on dequantised x): rel_L2 median {:.3e}  max {:.3e}",
            quantile(&control, 0.5),
            control[control.len() - 1],
        );
    }
    Ok(())
}
