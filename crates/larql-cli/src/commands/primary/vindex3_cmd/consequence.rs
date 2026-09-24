//! `vindex3 consequence` — SENSITIVITY-1B', per-tensor activation-weighted
//! consequence.
//!
//! One number per tensor, and nothing else:
//!
//! ```text
//! consequence(W) = sum_j  d_j * || dW[:, j] ||^2       dW = W - dequant(quant(W))
//! ```
//!
//! `d_j = E[x_j^2]` comes from the frozen capture. There is no
//! normalisation here — not by `||XW||^2`, not by a model total, not by
//! anything. 1A divided by `||W||^2` and 1B-a divided by `||XW||^2`, and
//! both failed the same way: they rewarded operands for being small. The
//! pre-registration (`bench/prompts/quality-bank-1/SENSITIVITY-1B-PRIME.md`)
//! fixes absolute consequence as the one form, so this emits exactly that.
//!
//! **This command knows nothing about candidates.** It does not know what
//! `late5-ffn` means, which regions are negatives, or what the bar is.
//! Aggregation and judgment live in `candidates.py` and `score_1b_prime.py`,
//! so the measurement mechanism stays independent of the hypothesis being
//! judged. Per-MiB division happens there, not here.
//!
//! Three tensors classes, three provenances for `d_j`:
//!
//! ```text
//! q_proj, k_proj, v_proj    attention input site      captured directly
//! gate_proj, up_proj        FFN input site            captured directly
//! down_proj                 silu(gate(x)) * up(x)     RECONSTRUCTED, gated
//! o_proj                    no site exists            NOT EMITTED
//! ```
//!
//! `o_proj` is absent rather than zero, null or estimated. The capture has
//! no attention-output site, so there is no honest number for it; emitting
//! a cheap one would make any region containing it look cheap.

use std::path::PathBuf;

use clap::Args;
use larql_inference::vindex3::{open_component, OpenPolicy, OpenedComponent};
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::OperandRef;
use larql_vindex::format::vindex3::represent::policy::{classify_in, Role};

use super::input_moments::{
    self, ffn_activation, layer_of, projection_of, Matrix, TensorMoments, RECONSTRUCTION_TOLERANCE,
};

/// Codec and encoder identity, recorded per tensor so a later rung can tell
/// which encoding produced a number without re-deriving it.
const CODEC: &str = "nvfp4/rev1";
const ENCODER: &str = "nvfp4-nearest-v1";

#[derive(Args)]
pub struct ConsequenceArgs {
    /// Canonical container. Must be the one the moments were captured from.
    pub container: PathBuf,

    /// The frozen capture from `sensitivity --calibration`.
    #[arg(long, value_name = "JSON")]
    pub moments: PathBuf,

    /// The pre-registered calibration identity. Scoring refuses unless the
    /// moments were captured from exactly this token bank.
    #[arg(long, value_name = "SHA256")]
    pub expect_token_digest: String,

    /// Per-tensor consequence records.
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(serde::Serialize)]
struct Consequence {
    tensor: String,
    component: String,
    layer: usize,
    projection: String,
    /// sum_j d_j * ||dW[:, j]||^2 — the pre-registered score, unnormalised.
    num: f64,
    /// Where `d_j` came from: a captured site, or the gated reconstruction.
    moment_source: String,
    /// Bytes, so the aggregator can form the per-MiB return without
    /// re-reading the container.
    compiled_bytes: u64,
    source_bytes: u64,
    source_weight_digest: String,
    moment_artifact_digest: String,
    calibration_token_digest: String,
    codec: String,
    encoder: String,
}

pub fn run(args: ConsequenceArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (moments, moment_artifact_digest) = input_moments::read(&args.moments)?;

    // ---- provenance: refuse, never warn -------------------------------
    if moments.calibration.token_digest != args.expect_token_digest {
        return Err(format!(
            "REFUSED: moments were captured from a different calibration bank.\n  \
             expected {}\n  captured {}\n\
             SENSITIVITY-1B' is judged on a disjoint set. The 1B-a artefact carries a\n\
             numerically identical numerator computed from bank-derived activations, so\n\
             this is checked by digest rather than by filename.",
            args.expect_token_digest, moments.calibration.token_digest,
        )
        .into());
    }

    // The opener is the one authority on inspect → plan → open; the
    // provenance checks below read the inspection it judged.
    let OpenedComponent {
        inspection,
        store,
        plan,
        ..
    } = open_component(
        &args.container,
        "target",
        OpenPolicy {
            want: None,
            source: RepresentationSource::Transient,
        },
    )?;
    input_moments::check_provenance(&moments, &inspection)?;
    println!(
        "provenance OK  calibration {}  {} entries, {} positions",
        &args.expect_token_digest[..16],
        moments.calibration.entries,
        moments.positions,
    );

    // ---- the operands --------------------------------------------------
    let activation = ffn_activation(&plan)?;
    println!("ffn activation {activation:?}  (from the plan the executor runs)");

    let weights = collect_weights(&inspection, &args.container, &store)?;

    // ---- the down_proj reconstruction, gated ---------------------------
    let mut borrow =
        |layer: usize, proj: &str| -> Result<Option<Matrix<'_>>, Box<dyn std::error::Error>> {
            Ok(find(&weights, layer, proj).map(|w| Matrix {
                rows: w.rows,
                k: w.k,
                values: std::borrow::Cow::Borrowed(&w.values),
            }))
        };
    let (control_layer, rel) =
        input_moments::check_reconstruction(&moments, activation, &mut borrow)?;
    let reconstruction_ok = rel <= RECONSTRUCTION_TOLERANCE;
    println!(
        "reconstruction control  layer {control_layer}  rel {rel:.3e}  tolerance {RECONSTRUCTION_TOLERANCE:.0e}  {}",
        if reconstruction_ok { "PASS" } else { "FAIL" },
    );
    if !reconstruction_ok {
        return Err(format!(
            "REFUSED: the offline FFN reconstruction does not reproduce the executor \
             (rel {rel:.3e} > {RECONSTRUCTION_TOLERANCE:.0e}).\n\
             down_proj's moments come from that reconstruction, and `down-protected` is \
             one of the three frozen negatives — scoring it on a wrong intermediate could \
             manufacture a pass."
        )
        .into());
    }

    let down_moments = input_moments::reconstruct_down_moments(&moments, activation, &mut borrow)?;
    let tensor_moments = TensorMoments::new(&moments, down_moments);

    // ---- emit ----------------------------------------------------------
    let mut out = Vec::new();
    let mut skipped_no_site = 0usize;
    for w in &weights {
        let Some(proj) = projection_of(&w.tensor) else {
            continue;
        };
        let Some(layer) = layer_of(&w.tensor) else {
            continue;
        };
        // o_proj has no activation site: absent, not zero.
        let Some((d, source)) = tensor_moments.get(layer, proj) else {
            skipped_no_site += 1;
            continue;
        };
        if d.len() != w.k {
            return Err(format!(
                "REFUSED: {} expects {} input features, moments carry {}",
                w.tensor,
                w.k,
                d.len()
            )
            .into());
        }
        out.push(Consequence {
            tensor: w.tensor.clone(),
            component: w.object.clone(),
            layer,
            projection: proj.to_string(),
            num: weighted_column_energy(&w.delta, w.rows, w.k, d),
            moment_source: source.to_string(),
            compiled_bytes: w.compiled_bytes,
            source_bytes: w.source_bytes,
            source_weight_digest: w.digest.clone(),
            moment_artifact_digest: moment_artifact_digest.clone(),
            calibration_token_digest: args.expect_token_digest.clone(),
            codec: CODEC.to_string(),
            encoder: ENCODER.to_string(),
        });
    }

    std::fs::write(&args.output, serde_json::to_string(&out)?)?;
    println!(
        "emitted {} tensor consequences ({skipped_no_site} skipped for having no activation site)\n-> {}",
        out.len(),
        args.output.display()
    );
    Ok(())
}

/// `sum_j d_j * ||dW[:, j]||^2` over a row-major `[rows, k]` delta.
fn weighted_column_energy(delta: &[f32], rows: usize, k: usize, d: &[f64]) -> f64 {
    let mut per_column = vec![0f64; k];
    for r in 0..rows {
        let base = r * k;
        for (j, acc) in per_column.iter_mut().enumerate() {
            let v = delta[base + j] as f64;
            *acc += v * v;
        }
    }
    per_column.iter().zip(d).map(|(e, dj)| dj * e).sum()
}

struct Weight {
    object: String,
    tensor: String,
    rows: usize,
    k: usize,
    values: Vec<f32>,
    delta: Vec<f32>,
    compiled_bytes: u64,
    source_bytes: u64,
    digest: String,
}

fn collect_weights(
    inspection: &larql_vindex::format::vindex3::inspect::SystemInspection,
    container: &std::path::Path,
    store: &OperandStore,
) -> Result<Vec<Weight>, Box<dyn std::error::Error>> {
    use larql_models::quant::nvfp4::round_trip;
    use larql_vindex::format::vindex3::represent::nvfp4_pack::PackLayout;

    let text: std::collections::BTreeSet<&str> = inspection
        .graph
        .components
        .iter()
        .filter(|c| {
            c.role == larql_vindex::format::vindex3::graph::component::ComponentRole::PrimaryText
        })
        .map(|c| c.id.as_str())
        .collect();
    let primary: std::collections::BTreeSet<String> = inspection
        .graph
        .objects
        .iter()
        .filter(|o| text.contains(o.component.as_str()))
        .map(|o| o.id.clone())
        .collect();

    let mut out = Vec::new();
    for entry in inspection.index.representations.values() {
        let (header, _) = larql_vindex::format::vindex3::encode::segment::read_segment_header(
            &container.join(&entry.segment),
        )?;
        for t in &header.tensors {
            let role = classify_in(
                primary.contains(&entry.object),
                &entry.object,
                &t.name,
                &t.shape,
            );
            if !matches!(role, Role::DecoderLinear | Role::ExpertWeight) {
                continue;
            }
            let Ok(layout) = PackLayout::derive(&t.shape, &t.name) else {
                continue;
            };
            let values = store.load(&OperandRef {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
            })?;
            let back = round_trip(&values, layout.rows, layout.k)
                .map_err(|e| format!("{}: {e}", t.name))?;
            let delta: Vec<f32> = values.iter().zip(&back).map(|(a, b)| a - b).collect();
            out.push(Weight {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                rows: layout.rows,
                k: layout.k,
                values,
                delta,
                compiled_bytes: layout.total_len as u64,
                source_bytes: t.len,
                digest: entry.payload_sha256.clone(),
            });
        }
    }
    Ok(out)
}

fn find<'a>(weights: &'a [Weight], layer: usize, proj: &str) -> Option<&'a Weight> {
    weights
        .iter()
        .find(|w| layer_of(&w.tensor) == Some(layer) && w.tensor.contains(proj))
}
