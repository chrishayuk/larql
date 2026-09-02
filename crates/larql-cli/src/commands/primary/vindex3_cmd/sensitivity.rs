//! `vindex3 sensitivity` — SENSITIVITY-1A, the cheap local screen.
//!
//! Q-BANK is the promotion gate and costs 1,622 teacher-forced positions
//! per candidate. That does not scale to role x depth combinatorics, and it
//! certainly does not scale to K3, where no one will evaluate every
//! expert x layer x representation decision globally.
//!
//! So this asks a much cheaper question, from the weights alone and with no
//! forward pass anywhere:
//!
//! ```text
//! e(t) = || W - dequant(quant(W)) ||^2 / || W ||^2
//! ```
//!
//! the relative error quantising tensor `t` introduces. One pass over the
//! weights scores every tensor, and any candidate precision map is then a
//! sum over the tensors it protects — so hundreds of candidates cost one
//! screen rather than one screen each.
//!
//! **This may not work, and the response is fixed in advance** (see
//! `bench/prompts/quality-bank-1/SENSITIVITY-1.md`). Weight error measures
//! how far the weights move, not how strongly the model uses the directions
//! they move in. A clean failure says weight geometry alone does not
//! predict semantic sensitivity, which is the argument for the
//! activation-weighted rung, not a reason to tune this one until it agrees
//! with fifteen known answers.
//!
//! The quantiser here is the *same* `quantize_nvfp4` the compiler and the
//! loader use, so the screen cannot drift from the thing it is screening.

use std::path::PathBuf;

use clap::Args;
use larql_inference::vindex3::{open_component, OpenPolicy, OpenedComponent};
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
use larql_vindex::format::vindex3::opplan::OperandRef;
use larql_vindex::format::vindex3::represent::policy::{classify_in, Role};

#[derive(Args)]
pub struct SensitivityArgs {
    /// Canonical container to screen.
    pub container: PathBuf,

    /// Write per-tensor scores here as JSON.
    #[arg(long)]
    pub output: PathBuf,

    /// SENSITIVITY-1B: capture per-feature activation second moments over
    /// a calibration set instead of scoring weight error alone.
    ///
    /// The file is JSON lines of `{"id": ..., "ids": [...]}`. 1A needs no
    /// forward pass; 1B needs one per calibration prompt, which is still
    /// far cheaper than a Q-BANK run per candidate.
    #[arg(long, value_name = "JSONL")]
    pub calibration: Option<PathBuf>,

    /// Where `--calibration` writes the captured moments and the
    /// reconstruction control.
    #[arg(long, value_name = "JSON")]
    pub moments: Option<PathBuf>,
}

/// Progress cadence for the 1A weight scan — one line per layer's worth of
/// projections on a 7-projection decoder, so the output tracks depth.
const SCORE_PROGRESS_EVERY: usize = 40;

#[derive(serde::Serialize)]
struct TensorScore {
    object: String,
    tensor: String,
    role: String,
    shape: Vec<usize>,
    /// Bytes this tensor occupies compiled.
    compiled_bytes: u64,
    /// Bytes it occupies at source precision.
    source_bytes: u64,
    /// Relative quantisation error — the screen's whole signal.
    rel_error: f64,
    /// Weight energy, so a caller can re-weight by magnitude rather than
    /// by the normalised score if it wants to.
    energy: f64,
}

pub fn run(args: SensitivityArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(cal) = args.calibration.clone() {
        return capture_moments(&args, &cal);
    }
    use larql_models::quant::nvfp4::{round_trip, NVFP4_GROUP_ELEMS};
    use larql_vindex::format::vindex3::represent::nvfp4_pack::PackLayout;

    // Canonical bytes: the screen scores what quantisation would do to the
    // source, so it must read the source. The opener is the one authority
    // on inspect → plan → open, and refuses an unclosed program.
    let OpenedComponent {
        inspection, store, ..
    } = open_component(
        &args.container,
        "target",
        OpenPolicy {
            want: None,
            source: RepresentationSource::Transient,
        },
    )?;

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

    let mut scores = Vec::new();
    let started = std::time::Instant::now();

    for entry in inspection.index.representations.values() {
        let (header, _) = larql_vindex::format::vindex3::encode::segment::read_segment_header(
            &args.container.join(&entry.segment),
        )?;
        for t in &header.tensors {
            let role = classify_in(
                primary.contains(&entry.object),
                &entry.object,
                &t.name,
                &t.shape,
            );
            // Only tensors an encoding could apply to are worth scoring;
            // a norm has no candidate map to appear in.
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

            let mut num = 0f64;
            let mut den = 0f64;
            for (a, b) in values.iter().zip(&back) {
                let d = (*a - *b) as f64;
                num += d * d;
                den += (*a as f64) * (*a as f64);
            }
            scores.push(TensorScore {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                role: role.name().to_string(),
                shape: t.shape.clone(),
                compiled_bytes: layout.total_len as u64,
                source_bytes: t.len,
                rel_error: if den > 0.0 { num / den } else { 0.0 },
                energy: den,
            });
            if scores.len().is_multiple_of(SCORE_PROGRESS_EVERY) {
                println!(
                    "  scored {} tensors ({:.0}s)",
                    scores.len(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }

    let _ = NVFP4_GROUP_ELEMS;
    let n = scores.len();
    let mean = scores.iter().map(|s| s.rel_error).sum::<f64>() / n.max(1) as f64;
    std::fs::write(&args.output, serde_json::to_string(&scores)?)?;
    println!(
        "scored {n} tensors in {:.0}s  (mean relative error {mean:.6})\n-> {}",
        started.elapsed().as_secs_f64(),
        args.output.display()
    );
    Ok(())
}

/// Accumulates per-feature second moments per (layer, site), plus one
/// sample of the FFN output for the reconstruction control.
#[cfg(all(feature = "gpu", target_os = "macos"))]
#[derive(Default)]
pub struct MomentCollector {
    /// (layer, site) -> running sum of x_j^2, and the count.
    pub sums: std::collections::BTreeMap<(usize, u8), (Vec<f64>, u64)>,
    /// One captured (ffn input, ffn output) pair, for the control.
    pub control: Option<(usize, Vec<f32>, Vec<f32>)>,
    /// Sampled FFN inputs per layer.
    ///
    /// `down_proj`'s input is `act(gate(x)) * up(x)`, and its second
    /// moments cannot be derived from `x`'s: the nonlinearity does not
    /// commute with the expectation. So a stride of actual inputs is kept
    /// and the intermediate's moments are computed offline from them —
    /// the same reconstruction the control just verified, rather than a
    /// second approximation stacked on the first.
    pub ffn_samples: std::collections::BTreeMap<usize, Vec<Vec<f32>>>,
    pending_ffn_input: Option<(usize, Vec<f32>)>,
    seen: std::collections::BTreeMap<usize, u64>,
}

/// Keep one FFN input in this many, per layer. 395 calibration positions
/// give ~50 samples per layer, which is ample for a per-feature second
/// moment and small enough to serialise.
#[cfg(all(feature = "gpu", target_os = "macos"))]
const FFN_SAMPLE_STRIDE: u64 = 8;

#[cfg(all(feature = "gpu", target_os = "macos"))]
impl larql_vindex::format::vindex3::opplan::exec::observe::StepObserver for MomentCollector {
    fn event(&mut self, _e: larql_vindex::format::vindex3::opplan::exec::observe::StepEvent) {}

    fn operand_input(
        &mut self,
        layer: usize,
        site: larql_vindex::format::vindex3::opplan::exec::observe::InputSite,
        values: &[f32],
    ) {
        use larql_vindex::format::vindex3::opplan::exec::observe::InputSite;
        let code = match site {
            InputSite::Attention => 0u8,
            InputSite::Ffn => 1,
            InputSite::FfnOutput => 2,
        };
        if site == InputSite::Ffn {
            self.pending_ffn_input = Some((layer, values.to_vec()));
            let n = self.seen.entry(layer).or_insert(0);
            if n.is_multiple_of(FFN_SAMPLE_STRIDE) {
                self.ffn_samples
                    .entry(layer)
                    .or_default()
                    .push(values.to_vec());
            }
            *n += 1;
        }
        if site == InputSite::FfnOutput {
            // Keep exactly one pair: the control needs a single position
            // where both sides are known, not a corpus.
            if self.control.is_none() {
                if let Some((l, input)) = self.pending_ffn_input.take() {
                    if l == layer {
                        self.control = Some((layer, input, values.to_vec()));
                    }
                }
            }
            // The FFN output is not an input site; it carries no moments.
            return;
        }
        let entry = self
            .sums
            .entry((layer, code))
            .or_insert_with(|| (vec![0.0; values.len()], 0));
        for (acc, v) in entry.0.iter_mut().zip(values) {
            *acc += (*v as f64) * (*v as f64);
        }
        entry.1 += 1;
    }
}

/// SENSITIVITY-1B capture: run the calibration set through the reference
/// backend, accumulating per-feature second moments at each input site.
///
/// The forward pass is the *canonical* one with an observer attached —
/// `an_observed_step_is_bit_identical_to_an_unobserved_one` is what makes
/// the captured activations the ones execution actually sees.
/// One calibration prompt, pre-tokenised. The ids are fed to the executor
/// verbatim, so this file *is* what ran.
#[derive(serde::Deserialize)]
pub(super) struct Entry {
    id: String,
    ids: Vec<u32>,
}

/// SHA-256 over the entries actually consumed, in the canonical form
/// `bench/prompts/quality-bank-1/freeze_calibration.py` freezes:
///
/// ```text
/// json([{id, ids}], sort_keys=True, separators=(",", ":"))
/// ```
///
/// Built by hand rather than through `serde_json::to_string` because the
/// digest is only useful if it is byte-identical to the Python side. Two
/// keys, and `"id"` sorts before `"ids"`, so the ordering is fixed here as
/// it is there. List order is the file's order in both.
pub(super) fn calibration_digest(entries: &[Entry]) -> String {
    use sha2::{Digest, Sha256};
    let mut canonical = String::from("[");
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            canonical.push(',');
        }
        canonical.push_str("{\"id\":");
        // serde_json for the string so escaping matches json.dumps.
        canonical.push_str(&serde_json::to_string(&e.id).unwrap_or_default());
        canonical.push_str(",\"ids\":[");
        for (j, id) in e.ids.iter().enumerate() {
            if j > 0 {
                canonical.push(',');
            }
            canonical.push_str(&id.to_string());
        }
        canonical.push_str("]}");
    }
    canonical.push(']');
    format!("{:x}", Sha256::digest(canonical.as_bytes()))
}

/// Reads and digests the bank, then reports that this build cannot run it.
///
/// The capture goes through the Metal executor, which does not exist off
/// macOS or without the `gpu` feature. It still parses and digests the
/// calibration file first, so a mis-pointed `--calibration` fails as a bad
/// path or a malformed bank rather than as "no GPU" — and the digest
/// reported is the same function the real capture stamps, so the bank can
/// be frozen and checked on a machine that cannot capture on it.
#[cfg(not(all(feature = "gpu", target_os = "macos")))]
fn capture_moments(
    _args: &SensitivityArgs,
    calibration: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(calibration)?;
    let entries: Vec<Entry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    Err(format!(
        "--calibration parsed {} entries (token digest {}), but capturing \
         moments needs the Metal executor: build with the `gpu` feature on macOS",
        entries.len(),
        calibration_digest(&entries),
    )
    .into())
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn capture_moments(
    args: &SensitivityArgs,
    calibration: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
    use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
    let out = args
        .moments
        .clone()
        .ok_or("--calibration requires --moments")?;
    let OpenedComponent {
        inspection,
        plan,
        store,
        ..
    } = open_component(
        &args.container,
        "target",
        OpenPolicy {
            want: None,
            source: RepresentationSource::Transient,
        },
    )?;

    let text = std::fs::read_to_string(calibration)?;
    let entries: Vec<Entry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let token_digest = calibration_digest(&entries);

    // The f16 Metal realisation, the same one `exec --backend metal` uses
    // and the one Granite's external oracle was verified against. The
    // naive f32 reference is correct and takes hours on a 40-layer model
    // with no KV cache, which would make the screen more expensive than
    // the Q-BANK run it exists to avoid.
    let gpu = larql_compute_metal::MetalBackend::new()
        .ok_or("no Metal device available for the sensitivity capture")?;
    let backend = larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::new(
        gpu,
        "metal-r3-f16",
        larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat::F16,
    );
    let mut collector = MomentCollector::default();
    let started = std::time::Instant::now();
    let mut positions = 0usize;

    for (n, e) in entries.iter().enumerate() {
        // A fresh state per prompt, as Q-BANK-2 established: every
        // position must see the context its own prompt gives it.
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv)?;
        for &t in &e.ids {
            session.step_observed(t, &mut collector)?;
            positions += 1;
        }
        println!(
            "  {}/{} {} ({} ids, {:.0}s)",
            n + 1,
            entries.len(),
            e.id,
            e.ids.len(),
            started.elapsed().as_secs_f64()
        );
    }

    let moments: Vec<serde_json::Value> = collector
        .sums
        .iter()
        .map(|((layer, site), (sums, count))| {
            let n = (*count).max(1) as f64;
            serde_json::json!({
                "layer": layer,
                "site": match site { 0 => "attention", 1 => "ffn", _ => "other" },
                "positions": count,
                // E[x_j^2] per input feature — a vector, not a matrix.
                "second_moment": sums.iter().map(|v| v / n).collect::<Vec<f64>>(),
            })
        })
        .collect();

    let samples: Vec<serde_json::Value> = collector
        .ffn_samples
        .iter()
        .map(|(layer, rows)| serde_json::json!({ "layer": layer, "rows": rows }))
        .collect();
    let control = collector.control.map(|(layer, input, output)| {
        serde_json::json!({ "layer": layer, "ffn_input": input, "ffn_output": output })
    });

    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({
            "positions": positions,
            // Which calibration set these moments came from, stamped from
            // what was actually consumed rather than from the filename.
            // A screen judged on a disjoint set must be able to *prove* the
            // moments are disjoint: an artefact captured from bank-derived
            // activations is numerically tempting and would invalidate the
            // rung it exists to serve.
            "calibration": {
                "source": calibration.display().to_string(),
                "entries": entries.len(),
                "token_digest": token_digest,
                "digest_over":
                    "json([{id, ids}], sort_keys, compact) of the entries consumed",
            },
            // The weights these moments were observed through. Scoring must
            // be able to prove all three authorities agree — weights,
            // moments, calibration — so the container names itself here
            // rather than being asserted by whoever runs the scorer.
            "container": {
                "path": args.container.display().to_string(),
                "model": inspection.index.model,
                "representation_digests": inspection
                    .index
                    .representations
                    .values()
                    .map(|e| (
                        format!("{}@{}", e.object, e.encoding),
                        e.payload_sha256.clone(),
                    ))
                    .collect::<std::collections::BTreeMap<_, _>>(),
            },
            "sites": moments,
            "ffn_samples": samples,
            "control": control,
        }))?,
    )?;
    println!(
        "captured {} sites over {positions} positions in {:.0}s\n-> {}",
        collector.sums.len(),
        started.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}
