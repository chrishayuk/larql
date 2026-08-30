//! Q1's machine-readable artifact.

use super::*;
use crate::format::vindex3::represent::experiment::{RepresentationStatus, RoleScope};
use crate::format::vindex3::represent::map::Precision;
use crate::format::vindex3::represent::policy::Role;
use crate::format::vindex3::represent::selection::promote;

/// **Q1's artifact: one evidence record per candidate representation.**
///
/// The two reports above are the human view. This is the machine one —
/// what REPRESENT would consult to choose an encoding for this
/// component on this hardware, with the fields it did NOT measure left
/// explicitly absent so no consumer can mistake silence for a good
/// result.
///
/// Set `LARQL_Q1_RECORDS` to a directory to persist them; otherwise
/// they are printed.
#[test]
fn emit_representation_experiment_records() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    ramp_up(&metal);
    let oracle: Vec<f32> = fx.experts.iter().flat_map(|e| e.oracle.clone()).collect();
    let (n, k) = Stage::Gate.shape(&fx);

    // Quality and throughput from the SAME banks, all held at once.
    let arms: Vec<Arm> = Format::ALL.iter().map(|&f| Arm::build(&fx, f)).collect();
    let replicated: Vec<Reencoded> = arms
        .iter()
        .map(|a| a.gate.replicated(THROUGHPUT_REPLICAS))
        .collect();
    let outputs: Vec<Vec<f32>> = arms.iter().map(|a| a.ffn(&metal, &fx)).collect();
    let timings: Vec<Timing> = arms
        .iter()
        .zip(&replicated)
        .map(|(a, bank)| {
            measure(BENCH_WARMUP, BENCH_ITERS, || {
                a.format
                    .dispatch_profiled(
                        &metal,
                        &bank.bytes,
                        &bank.offsets,
                        &fx.x,
                        n,
                        k,
                        InputLayout::Shared,
                    )
                    .1
            })
        })
        .collect();

    let baseline_gpu = timings[0].gpu_median_ms;
    let source_bytes = arms[0].bank_bytes() as u64;
    let mut records = Vec::new();
    for ((a, out), t) in arms.iter().zip(&outputs).zip(&timings) {
        if a.format == Format::Bf16 {
            continue; // the baseline is a field, not a row
        }
        let native = a.format.native();
        let byte_ratio = 16.0 / a.format.bpw();
        let mut caveats = Vec::new();
        if !native {
            caveats.push(
                "no grouped kernel on the byte-offset-table convention: quality measured \
                 through decoded values carried in bf16, which costs 3.4e-3 relative — 23x \
                 below this format's own error. Throughput NOT measured."
                    .to_string(),
            );
        }
        caveats.push(format!(
            "component-level error only; no logit-level bank has been run, so kl/flip \
             fields are absent. GPU timer spread {:.2}.",
            t.gpu_spread()
        ));
        records.push(RepresentationExperiment {
            model: "Kimi-Linear-48B-A3B-Instruct".into(),
            // The routed bank IS the `ExpertWeight` role, scoped to the
            // layer measured — the precision map's own selector, so this
            // evidence can be promoted without a translation step.
            scope: RoleScope::role(Role::ExpertWeight).layers(1, 1),
            component: format!(
                "RoutedExpertBank(layer=1, slots={}, hidden={}, inter={})",
                fx.experts.len(),
                fx.hidden,
                fx.inter
            ),
            source: "BF16".into(),
            target: a.format.label().into(),
            hardware: "Apple M3 Max".into(),
            bits_per_weight: a.format.bpw(),
            source_bytes,
            target_bytes: a.bank_bytes() as u64,
            baseline_tokens_per_second: Some(BF16_REFERENCE_TOKENS_PER_SECOND),
            // Absent until the representation is wired into the token
            // loop. A projection is not a measurement.
            result_tokens_per_second: None,
            baseline_gpu_ms: Some(baseline_gpu),
            target_gpu_ms: native.then_some(t.gpu_median_ms),
            target_achieved_gb_per_s: native.then(|| {
                replicated[Format::ALL.iter().position(|f| *f == a.format).unwrap()]
                    .bytes
                    .len() as f64
                    / (t.gpu_median_ms / 1000.0)
                    / 1e9
            }),
            bandwidth_bound_fraction: native
                .then_some((baseline_gpu / t.gpu_median_ms) / byte_ratio),
            component_rel_rms: Some(rel_rms(out, &oracle) as f64),
            component_max_over_scale: Some(rel_err(out, &oracle) as f64),
            // No bank has been run, so there is no gate to name. This
            // being `None` is what makes `promote` refuse below.
            quality: None,
            // The ladder, as independently-known facts. Quality is not
            // among them on purpose: it cannot be a boolean, it has to
            // name a gate it passed. Q1 measured bytes, speed and
            // component error — none of which is evidence about the
            // model's output distribution.
            status: RepresentationStatus {
                represented: true,
                available: true,
                backend_supported: native,
                runnable: true,
                measured: native,
                selected: false,
            },
            provenance: Provenance {
                gate: "emit_representation_experiment_records".into(),
                fixture: "LARQL_KIMI_MOE_FIXTURE".into(),
                native_kernel: native,
                caveats,
            },
        });
    }

    for r in &records {
        assert!(
            !r.supports_quality_claim(),
            "{}: no logit bank has been run, so this record must NOT back a quality claim",
            r.target
        );
        assert_eq!(
            r.supports_throughput_claim(),
            r.provenance.native_kernel,
            "{}: throughput support must follow from having a real kernel",
            r.target
        );
    }
    // **What the evidence currently justifies: nothing.** Running the
    // promotion gate here is not decoration — it is the assertion that
    // Q1's numbers, however good, cannot become a deployment decision
    // until a quality bank exists. When Q2 lands, this same call starts
    // returning a map, and the diff says so.
    let promotion = promote("kimi-q1-candidates", "BF16", &records);
    eprintln!("[q1-promote] {}", promotion.describe());
    assert_eq!(
        promotion.promoted(),
        0,
        "no logit bank has been run, so no candidate may be promoted"
    );
    assert!(
        promotion.map.roles.is_empty(),
        "an unpromoted role must leave the map at source precision"
    );
    assert!(matches!(
        promotion
            .map
            .resolve(Role::ExpertWeight, "1.mlp.experts.7.down_proj.weight"),
        Precision::Source
    ));

    let json = serde_json::to_string_pretty(&records).expect("records serialise");
    match std::env::var_os("LARQL_Q1_RECORDS") {
        Some(dir) => {
            let path = std::path::Path::new(&dir).join("q1_expert_bank.json");
            std::fs::write(&path, &json).expect("write records");
            eprintln!(
                "[q1-record] {} records -> {}",
                records.len(),
                path.display()
            );
        }
        None => eprintln!("[q1-record]\n{json}"),
    }
}
