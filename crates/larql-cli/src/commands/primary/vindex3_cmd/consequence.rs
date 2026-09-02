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

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Args;
use larql_inference::vindex3::{open_component, OpenPolicy, OpenedComponent};
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::OperandRef;
use larql_vindex::format::vindex3::represent::policy::{classify_in, Role};

/// Relative error the offline FFN reconstruction may differ from the
/// executor's own output before `down_proj` is refused.
///
/// The 1B-a capture measured 1.07e-07 relative, so this is ~100x the
/// observed agreement — loose enough not to trip on f32 reassociation,
/// tight enough that a wrong activation, a wrong operand order or a missed
/// scaling cannot pass. `down-protected` is one of the three frozen
/// negatives, so a silent reconstruction defect could manufacture a pass.
const RECONSTRUCTION_TOLERANCE: f64 = 1e-5;

/// Codec and encoder identity, recorded per tensor so a later rung can tell
/// which encoding produced a number without re-deriving it.
const CODEC: &str = "nvfp4/rev1";
const ENCODER: &str = "nvfp4-nearest-v1";

const ATTENTION_SITE: u8 = 0;
const FFN_SITE: u8 = 1;

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

#[derive(serde::Deserialize)]
struct Moments {
    positions: usize,
    calibration: CalibrationStamp,
    container: ContainerStamp,
    sites: Vec<Site>,
    ffn_samples: Vec<FfnSamples>,
    control: Option<Control>,
}

#[derive(serde::Deserialize)]
struct CalibrationStamp {
    token_digest: String,
    entries: usize,
}

#[derive(serde::Deserialize)]
struct ContainerStamp {
    model: String,
    representation_digests: BTreeMap<String, String>,
}

#[derive(serde::Deserialize)]
struct Site {
    layer: usize,
    site: String,
    second_moment: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct FfnSamples {
    layer: usize,
    rows: Vec<Vec<f32>>,
}

#[derive(serde::Deserialize)]
struct Control {
    layer: usize,
    ffn_input: Vec<f32>,
    ffn_output: Vec<f32>,
}

/// Ordered so `gate_proj`/`up_proj`/`down_proj` cannot be shadowed by a
/// shorter attention name appearing as a substring.
const PROJECTIONS: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

fn projection_of(tensor: &str) -> Option<&'static str> {
    PROJECTIONS.into_iter().find(|p| tensor.contains(p))
}

fn layer_of(tensor: &str) -> Option<usize> {
    tensor.split('.').next()?.parse().ok()
}

pub fn run(args: ConsequenceArgs) -> Result<(), Box<dyn std::error::Error>> {
    let moments: Moments = serde_json::from_str(&std::fs::read_to_string(&args.moments)?)?;
    let moment_artifact_digest = digest_file(&args.moments)?;

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
    if inspection.index.model != moments.container.model {
        return Err(format!(
            "REFUSED: container is not the one the moments were captured from.\n  \
             moments  {}\n  given    {}",
            moments.container.model, inspection.index.model,
        )
        .into());
    }
    for entry in inspection.index.representations.values() {
        let key = format!("{}@{}", entry.object, entry.encoding);
        match moments.container.representation_digests.get(&key) {
            Some(d) if *d == entry.payload_sha256 => {}
            Some(d) => {
                return Err(format!(
                    "REFUSED: {key} changed since capture.\n  moments {d}\n  \
                     container {}",
                    entry.payload_sha256,
                )
                .into())
            }
            None => {
                return Err(format!(
                    "REFUSED: {key} was not present when the moments were captured"
                )
                .into())
            }
        }
    }
    println!(
        "provenance OK  calibration {}  {} entries, {} positions",
        &args.expect_token_digest[..16],
        moments.calibration.entries,
        moments.positions,
    );

    // ---- the operands --------------------------------------------------
    let activation = ffn_activation(&plan)?;
    println!("ffn activation {activation:?}  (from the plan the executor runs)");

    let captured: BTreeMap<(usize, u8), &Vec<f64>> = moments
        .sites
        .iter()
        .filter_map(|s| {
            let code = match s.site.as_str() {
                "attention" => ATTENTION_SITE,
                "ffn" => FFN_SITE,
                _ => return None,
            };
            Some(((s.layer, code), &s.second_moment))
        })
        .collect();

    let weights = collect_weights(&inspection, &args.container, &store)?;

    // ---- the down_proj reconstruction, gated ---------------------------
    let control = moments
        .control
        .as_ref()
        .ok_or("REFUSED: capture carries no reconstruction control pair")?;
    let rel = check_reconstruction(control, &weights, activation)?;
    let reconstruction_ok = rel <= RECONSTRUCTION_TOLERANCE;
    println!(
        "reconstruction control  layer {}  rel {rel:.3e}  tolerance {RECONSTRUCTION_TOLERANCE:.0e}  {}",
        control.layer,
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

    let down_moments = reconstruct_down_moments(&moments.ffn_samples, &weights, activation)?;

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
        let (d, source) = match proj {
            "q_proj" | "k_proj" | "v_proj" => (
                captured.get(&(layer, ATTENTION_SITE)).copied(),
                "captured:attention",
            ),
            "gate_proj" | "up_proj" => (captured.get(&(layer, FFN_SITE)).copied(), "captured:ffn"),
            "down_proj" => (
                down_moments.get(&layer),
                "reconstructed:silu(gate(x))*up(x)",
            ),
            // No attention-output site exists. Absent, not zero.
            "o_proj" => {
                skipped_no_site += 1;
                continue;
            }
            _ => continue,
        };
        let Some(d) = d else {
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

/// `y = W x` for a row-major `[rows, k]` operand.
fn matvec(w: &Weight, x: &[f32]) -> Vec<f32> {
    (0..w.rows)
        .map(|r| {
            let base = r * w.k;
            (0..w.k).map(|j| w.values[base + j] * x[j]).sum()
        })
        .collect()
}

/// The FFN intermediate, named rather than inlined: `act(gate(x)) * up(x)`.
///
/// This is the one quantity in 1B' that is *reconstructed* rather than
/// observed, so it exists as its own function with its own control.
fn down_input(
    weights: &[Weight],
    layer: usize,
    x: &[f32],
    activation: larql_models::config::activation::Activation,
) -> Option<Vec<f32>> {
    use larql_vindex::format::vindex3::opplan::exec::kernels::activate;
    let gate = find(weights, layer, "gate_proj")?;
    let up = find(weights, layer, "up_proj")?;
    let g = matvec(gate, x);
    let u = matvec(up, x);
    Some(
        g.iter()
            .zip(&u)
            .map(|(a, b)| activate(activation, *a) * b)
            .collect(),
    )
}

/// Recompute the executor's own FFN output from its own input and compare.
///
/// A mathematically equivalent reconstruction can still be numerically
/// different — wrong activation, wrong operand order, a scaling the
/// executor applies and this does not — and it would corrupt exactly one
/// of the three frozen negatives.
fn check_reconstruction(
    control: &Control,
    weights: &[Weight],
    activation: larql_models::config::activation::Activation,
) -> Result<f64, Box<dyn std::error::Error>> {
    let inner = down_input(weights, control.layer, &control.ffn_input, activation)
        .ok_or("control layer has no gate/up operands")?;
    let down = find(weights, control.layer, "down_proj").ok_or("control layer has no down_proj")?;
    let recomputed = matvec(down, &inner);
    if recomputed.len() != control.ffn_output.len() {
        return Err(format!(
            "reconstruction produced {} outputs, executor recorded {}",
            recomputed.len(),
            control.ffn_output.len()
        )
        .into());
    }
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in recomputed.iter().zip(&control.ffn_output) {
        let d = (*a - *b) as f64;
        num += d * d;
        den += (*b as f64) * (*b as f64);
    }
    Ok(if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    })
}

/// `E[z_j^2]` for the FFN intermediate, from the sampled inputs.
///
/// The nonlinearity does not commute with the expectation, so these cannot
/// be derived from the FFN input's moments — the actual intermediate has to
/// be formed per sample.
fn reconstruct_down_moments(
    samples: &[FfnSamples],
    weights: &[Weight],
    activation: larql_models::config::activation::Activation,
) -> Result<BTreeMap<usize, Vec<f64>>, Box<dyn std::error::Error>> {
    let mut out = BTreeMap::new();
    for s in samples {
        if s.rows.is_empty() {
            continue;
        }
        let mut acc: Option<Vec<f64>> = None;
        for x in &s.rows {
            let z = down_input(weights, s.layer, x, activation)
                .ok_or_else(|| format!("layer {} has no gate/up operands", s.layer))?;
            let a = acc.get_or_insert_with(|| vec![0.0; z.len()]);
            for (slot, v) in a.iter_mut().zip(&z) {
                *slot += (*v as f64) * (*v as f64);
            }
        }
        if let Some(mut a) = acc {
            let n = s.rows.len() as f64;
            for v in a.iter_mut() {
                *v /= n;
            }
            out.insert(s.layer, a);
        }
    }
    Ok(out)
}

/// The activation the plan carries, so the reconstruction mirrors the
/// executor rather than assuming a family default.
fn ffn_activation(
    plan: &larql_vindex::format::vindex3::opplan::ComponentOpPlan,
) -> Result<larql_models::config::activation::Activation, Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::opplan::LayerFfn;
    for layer in &plan.layers {
        match &layer.ffn {
            Some(LayerFfn::Dense(f)) => return Ok(f.activation),
            Some(LayerFfn::Routed(r)) => return Ok(r.activation),
            Some(LayerFfn::Hybrid(_)) | None => continue,
        }
    }
    Err("plan carries no FFN op to read an activation from".into())
}

fn digest_file(path: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    use sha2::{Digest, Sha256};
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}
