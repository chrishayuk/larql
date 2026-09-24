//! Per-tensor input-feature second moments `E[x_j^2]` from a
//! `sensitivity --calibration` capture — the one mapping `consequence`
//! scores with and `represent --moments` encodes with.
//!
//! Which tensor feeds from which site is read from the container's own
//! operation plan — the operands each layer's attention and FFN ops bind —
//! never from tensor names:
//!
//! ```text
//! attention q, k, v    attention input site      captured directly
//! FFN gate, up         FFN input site            captured directly
//! FFN down             act(gate(x)) * up(x)      RECONSTRUCTED, gated
//! attention o          no site exists            ABSENT
//! ```
//!
//! `o` is absent rather than zero, null or estimated: the capture has no
//! attention-output site, so there is no honest number for it. So is every
//! operand of an operator this mapping has no sites for (non-softmax
//! attention, routed experts): absent, and counted as such.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;

use larql_vindex::format::vindex3::inspect::SystemInspection;
use larql_vindex::format::vindex3::opplan::{ComponentOpPlan, OperandRef};
use larql_vindex::format::vindex3::represent::InputWeights;

type BoxErr = Box<dyn std::error::Error>;

/// Relative error the offline FFN reconstruction may differ from the
/// executor's own output before `down_proj` is refused.
///
/// The 1B-a capture measured 1.07e-07 relative, so this is ~100x the
/// observed agreement — loose enough not to trip on f32 reassociation,
/// tight enough that a wrong activation, a wrong operand order or a missed
/// scaling cannot pass.
pub(super) const RECONSTRUCTION_TOLERANCE: f64 = 1e-5;

const ATTENTION_SITE: u8 = 0;
const FFN_SITE: u8 = 1;

#[derive(serde::Deserialize)]
pub(super) struct Moments {
    pub(super) positions: usize,
    pub(super) calibration: CalibrationStamp,
    pub(super) container: ContainerStamp,
    sites: Vec<Site>,
    ffn_samples: Vec<FfnSamples>,
    control: Option<Control>,
}

#[derive(serde::Deserialize)]
pub(super) struct CalibrationStamp {
    pub(super) token_digest: String,
    pub(super) entries: usize,
}

#[derive(serde::Deserialize)]
pub(super) struct ContainerStamp {
    pub(super) model: String,
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

/// Read a capture, and the sha256 of its bytes.
pub(super) fn read(path: &Path) -> Result<(Moments, String), BoxErr> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    Ok((serde_json::from_slice(&bytes)?, digest))
}

/// Refuse a container other than the one the moments were captured from:
/// the model must match, and every representation it holds must carry the
/// digest it had at capture.
pub(super) fn check_provenance(
    moments: &Moments,
    inspection: &SystemInspection,
) -> Result<(), BoxErr> {
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
    Ok(())
}

/// The part an operand plays in its layer, as the plan binds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Part {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl Part {
    /// The label reports carry for this part.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Q => "q_proj",
            Self::K => "k_proj",
            Self::V => "v_proj",
            Self::O => "o_proj",
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

/// Every operand a moment site can describe, as the plan binds it:
/// softmax attention's q/k/v/o and a dense FFN's gate/up/down, by layer.
pub(super) struct PlanOperands {
    by_part: BTreeMap<(usize, Part), OperandRef>,
    by_tensor: BTreeMap<(String, String), (usize, Part)>,
}

impl PlanOperands {
    pub(super) fn from_plan(plan: &ComponentOpPlan) -> Self {
        use larql_vindex::format::vindex3::opplan::{LayerAttention, LayerFfn};
        let mut by_part = BTreeMap::new();
        for layer in &plan.layers {
            if let LayerAttention::Softmax(a) = &layer.attention {
                for (part, op) in [
                    (Part::Q, &a.q),
                    (Part::K, &a.k),
                    (Part::V, &a.v),
                    (Part::O, &a.o),
                ] {
                    by_part.insert((layer.layer, part), op.clone());
                }
            }
            if let Some(LayerFfn::Dense(f)) = &layer.ffn {
                if let Some(gate) = &f.gate {
                    by_part.insert((layer.layer, Part::Gate), gate.clone());
                }
                by_part.insert((layer.layer, Part::Up), f.up.clone());
                by_part.insert((layer.layer, Part::Down), f.down.clone());
            }
        }
        let by_tensor = by_part
            .iter()
            .map(|(&key, op)| ((op.object.clone(), op.tensor.clone()), key))
            .collect();
        Self { by_part, by_tensor }
    }

    pub(super) fn get(&self, layer: usize, part: Part) -> Option<&OperandRef> {
        self.by_part.get(&(layer, part))
    }

    /// The layer and part the plan binds `(object, tensor)` to.
    pub(super) fn part_of(&self, object: &str, tensor: &str) -> Option<(usize, Part)> {
        self.by_tensor
            .get(&(object.to_string(), tensor.to_string()))
            .copied()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, Part, &OperandRef)> {
        self.by_part.iter().map(|(&(l, p), op)| (l, p, op))
    }
}

/// A row-major `[rows, k]` operand, borrowed from a caller that already
/// holds it or loaded for the occasion.
pub(super) struct Matrix<'a> {
    pub(super) rows: usize,
    pub(super) k: usize,
    pub(super) values: Cow<'a, [f32]>,
}

/// `y = W x`.
fn matvec(w: &Matrix<'_>, x: &[f32]) -> Vec<f32> {
    (0..w.rows)
        .map(|r| {
            let base = r * w.k;
            (0..w.k).map(|j| w.values[base + j] * x[j]).sum()
        })
        .collect()
}

/// The FFN intermediate, named rather than inlined: `act(gate(x)) * up(x)`.
///
/// This is the one quantity that is *reconstructed* rather than observed,
/// so it exists as its own function with its own control.
fn down_input(
    gate: &Matrix<'_>,
    up: &Matrix<'_>,
    x: &[f32],
    activation: larql_models::config::activation::Activation,
) -> Vec<f32> {
    use larql_vindex::format::vindex3::opplan::exec::kernels::activate;
    let g = matvec(gate, x);
    let u = matvec(up, x);
    g.iter()
        .zip(&u)
        .map(|(a, b)| activate(activation, *a) * b)
        .collect()
}

/// The operands the reconstruction reads: `(layer, part)` → matrix, or
/// `None` if the layer has no such operand.
pub(super) type Loader<'a, 'b> = dyn FnMut(usize, Part) -> Result<Option<Matrix<'a>>, BoxErr> + 'b;

fn load_required<'a>(
    load: &mut Loader<'a, '_>,
    layer: usize,
    part: Part,
) -> Result<Matrix<'a>, BoxErr> {
    load(layer, part)?.ok_or_else(|| format!("layer {layer} binds no {part:?} operand").into())
}

/// Recompute the executor's own FFN output from its own input and compare,
/// returning the relative error.
///
/// A mathematically equivalent reconstruction can still be numerically
/// different — wrong activation, wrong operand order, a scaling the
/// executor applies and this does not.
pub(super) fn check_reconstruction(
    moments: &Moments,
    activation: larql_models::config::activation::Activation,
    load: &mut Loader<'_, '_>,
) -> Result<(usize, f64), BoxErr> {
    let control = moments
        .control
        .as_ref()
        .ok_or("REFUSED: capture carries no reconstruction control pair")?;
    let gate = load_required(load, control.layer, Part::Gate)?;
    let up = load_required(load, control.layer, Part::Up)?;
    let down = load_required(load, control.layer, Part::Down)?;
    let recomputed = matvec(
        &down,
        &down_input(&gate, &up, &control.ffn_input, activation),
    );
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
    let rel = if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    };
    Ok((control.layer, rel))
}

/// `E[z_j^2]` for the FFN intermediate, from the sampled inputs.
///
/// The nonlinearity does not commute with the expectation, so these cannot
/// be derived from the FFN input's moments — the actual intermediate has to
/// be formed per sample. Operands are asked for one layer at a time.
pub(super) fn reconstruct_down_moments(
    moments: &Moments,
    activation: larql_models::config::activation::Activation,
    load: &mut Loader<'_, '_>,
) -> Result<BTreeMap<usize, Vec<f64>>, BoxErr> {
    let mut out = BTreeMap::new();
    for s in &moments.ffn_samples {
        if s.rows.is_empty() {
            continue;
        }
        let gate = load_required(load, s.layer, Part::Gate)?;
        let up = load_required(load, s.layer, Part::Up)?;
        let mut acc: Option<Vec<f64>> = None;
        for x in &s.rows {
            let z = down_input(&gate, &up, x, activation);
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

/// Every tensor's input moments: `(layer, projection)` → the moment vector
/// and where it came from. `o_proj` never appears.
pub(super) struct TensorMoments {
    captured: BTreeMap<(usize, u8), Vec<f64>>,
    down: BTreeMap<usize, Vec<f64>>,
}

impl TensorMoments {
    pub(super) fn new(moments: &Moments, down: BTreeMap<usize, Vec<f64>>) -> Self {
        let captured = moments
            .sites
            .iter()
            .filter_map(|s| {
                let code = match s.site.as_str() {
                    "attention" => ATTENTION_SITE,
                    "ffn" => FFN_SITE,
                    _ => return None,
                };
                Some(((s.layer, code), s.second_moment.clone()))
            })
            .collect();
        Self { captured, down }
    }

    /// The moments for `part` at `layer`, and their provenance.
    pub(super) fn get(&self, layer: usize, part: Part) -> Option<(&[f64], &'static str)> {
        let (d, source) = match part {
            Part::Q | Part::K | Part::V => (
                self.captured.get(&(layer, ATTENTION_SITE)),
                "captured:attention",
            ),
            Part::Gate | Part::Up => (self.captured.get(&(layer, FFN_SITE)), "captured:ffn"),
            Part::Down => (self.down.get(&layer), "reconstructed:silu(gate(x))*up(x)"),
            Part::O => return None,
        };
        d.map(|d| (d.as_slice(), source))
    }
}

/// The activation the plan carries, so the reconstruction mirrors the
/// executor rather than assuming a family default.
pub(super) fn ffn_activation(
    plan: &larql_vindex::format::vindex3::opplan::ComponentOpPlan,
) -> Result<larql_models::config::activation::Activation, BoxErr> {
    use larql_vindex::format::vindex3::opplan::LayerFfn;
    for layer in &plan.layers {
        let (activation, gate_policy) = match &layer.ffn {
            Some(LayerFfn::Dense(f)) => (f.activation, f.gate_policy),
            Some(LayerFfn::Routed(r)) => (r.activation, r.gate_policy),
            Some(LayerFfn::Hybrid(_)) | None => continue,
        };
        // The reconstruction is `activate(gate) * up`, written out at
        // `down_input`. A gate policy that is not plain gating computes
        // something else entirely, and reconstructing it as plain gating
        // would put wrong moments under every `down_proj` — quietly, and
        // with every shape still closing. Refuse by name instead.
        if !matches!(gate_policy, larql_models::ExpertGatePolicy::Gated) {
            return Err(format!(
                "layer {} carries {gate_policy:?}; the reconstruction forms the FFN as \
                 `activation(gate) * up` and has no form for that combine, so it refuses \
                 rather than report moments computed from the wrong one",
                layer.layer,
            )
            .into());
        }
        return Ok(activation);
    }
    Err("plan carries no FFN op to read an activation from".into())
}

/// Input-feature weights for `represent --moments`: every decoder
/// projection the capture covers, keyed by `(object, tensor)`, with the
/// capture's digest; and how many tensors each provenance supplied.
pub(super) fn input_weights(
    container: &Path,
    moments_path: &Path,
) -> Result<(InputWeights, BTreeMap<&'static str, usize>), BoxErr> {
    use larql_inference::vindex3::{open_component, OpenPolicy, OpenedComponent};
    use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;

    let (moments, digest) = read(moments_path)?;
    let OpenedComponent {
        inspection,
        store,
        plan,
        ..
    } = open_component(
        container,
        "target",
        OpenPolicy {
            want: None,
            source: RepresentationSource::Transient,
        },
    )?;
    check_provenance(&moments, &inspection)?;
    let activation = ffn_activation(&plan)?;

    let operands = PlanOperands::from_plan(&plan);
    let mut load = |layer: usize, part: Part| -> Result<Option<Matrix<'static>>, BoxErr> {
        let Some(op) = operands.get(layer, part) else {
            return Ok(None);
        };
        let (rows, k) = matrix_shape(op)?;
        Ok(Some(Matrix {
            rows,
            k,
            values: Cow::Owned(store.load(op)?),
        }))
    };
    let (control_layer, rel) = check_reconstruction(&moments, activation, &mut load)?;
    if rel > RECONSTRUCTION_TOLERANCE {
        return Err(format!(
            "REFUSED: the offline FFN reconstruction does not reproduce the executor at \
             layer {control_layer} (rel {rel:.3e} > {RECONSTRUCTION_TOLERANCE:.0e}); \
             down_proj's weights would come from a wrong intermediate"
        )
        .into());
    }
    let down = reconstruct_down_moments(&moments, activation, &mut load)?;
    let tensor_moments = TensorMoments::new(&moments, down);

    let mut by_tensor = BTreeMap::new();
    let mut sources: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (layer, part, op) in operands.iter() {
        let Some((d, source)) = tensor_moments.get(layer, part) else {
            *sources.entry("absent (no activation site)").or_insert(0) += 1;
            continue;
        };
        let (_, k) = matrix_shape(op)?;
        if d.len() != k {
            return Err(format!(
                "REFUSED: {} expects {k} input features, moments carry {}",
                op.tensor,
                d.len()
            )
            .into());
        }
        by_tensor.insert((op.object.clone(), op.tensor.clone()), d.to_vec());
        *sources.entry(source).or_insert(0) += 1;
    }
    Ok((InputWeights { by_tensor, digest }, sources))
}

/// `(rows, k)` of a 2-D operand.
fn matrix_shape(op: &OperandRef) -> Result<(usize, usize), BoxErr> {
    match op.shape.as_slice() {
        [rows, k] => Ok((*rows, *k)),
        other => Err(format!("{}: expected a 2-D operand, shape {other:?}", op.tensor).into()),
    }
}
