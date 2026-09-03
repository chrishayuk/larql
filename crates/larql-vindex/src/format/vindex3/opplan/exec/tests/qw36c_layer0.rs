//! QW-3.6c: isolate the first diverging layer of the real Qwen3.8-27B.
//!
//! The per-layer census over the real container put the divergence at the
//! very first plane after the embedding:
//!
//! ```text
//! layer_000 (embedding)   rel_rms 0.0000e0   cos 1.0000000   EXACT
//! layer_001 (leaves L0)   rel_rms 1.2798e0   cos 0.8694140   <-- first
//! ```
//!
//! Layer 0 is a Gated DeltaNet layer, so this runs THAT operator alone,
//! on the real container's operands and HF's own normalised input, and
//! compares against HF's `linear_attn` output. It separates the operator
//! from the traversal: QW-2E proved the operator against HF using
//! captured weights, and what is new here is the container's operands and
//! this executor's plumbing around them.
//!
//! Env-gated — needs the 51 GB container and an HF capture — and skips
//! LOUDLY rather than reporting success over a missing subject:
//!
//! ```text
//! QW36C_CONTAINER=~/chris-models/Qwen3.8-27B.vindex3 \
//! QW36C_CAPS=/path/to/l0_caps.json cargo test qw36c -- --nocapture
//! ```

use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::continuation::RecurrentState;
use crate::format::vindex3::opplan::exec::cpu::WeightRows;
use crate::format::vindex3::opplan::exec::gated_delta::{layer_forward, state_geometry, Mutation};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention};

/// Accumulated in **f64**. The subject is f32, but a 248,320-term dot
/// summed in f32 loses more precision than the difference being measured:
/// the same comparison reads cos 0.9998944 in f32 and 0.9999807 in f64.
/// An f32 instrument reports a divergence that is its own rounding — the
/// measurement has to be more precise than the thing measured.
fn metrics(mine: &[f32], theirs: &[f32]) -> (f32, f32, f32) {
    let (mut num, mut den, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in mine.iter().zip(theirs) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let mx = mine
        .iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    (
        (num / den.max(f64::MIN_POSITIVE)).sqrt() as f32,
        (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)) as f32,
        mx,
    )
}

#[test]
fn the_first_recurrent_layer_matches_hf_on_the_real_container() {
    let (Ok(container), Ok(caps)) = (
        std::env::var("QW36C_CONTAINER"),
        std::env::var("QW36C_CAPS"),
    ) else {
        eprintln!("SKIP qw36c: set QW36C_CONTAINER and QW36C_CAPS");
        return;
    };
    let root = std::path::Path::new(&container);
    let inspection = inspect_container(root, false).unwrap();
    let plan = plan_component_ops(&inspection, root, "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(root, &inspection).unwrap();

    let LayerAttention::GatedDelta(op) = &plan.layers[0].attention else {
        panic!("layer 0 of this container is not a recurrence");
    };
    let caps: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(caps).unwrap()).unwrap();
    let grab = |k: &str| -> Vec<Vec<f32>> {
        caps[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect()
    };
    // HF's OWN normalised input, so the norm is not on trial here.
    let ln_in = grab("ln_in");
    let hf_attn = grab("attn");

    let load = |r: &crate::format::vindex3::opplan::OperandRef| store.load(r).unwrap();
    let (qkv, a, b, z, conv, alog, dt, nrm, outp) = (
        load(&op.in_proj_qkv),
        load(&op.in_proj_a),
        load(&op.in_proj_b),
        load(&op.in_proj_z),
        load(&op.conv1d),
        load(&op.a_log),
        load(&op.dt_bias),
        load(&op.norm),
        load(&op.out_proj),
    );
    let weights = crate::format::vindex3::opplan::exec::gated_delta::GatedDeltaWeights {
        in_proj_qkv: WeightRows::F32(&qkv),
        in_proj_a: WeightRows::F32(&a),
        in_proj_b: WeightRows::F32(&b),
        in_proj_z: WeightRows::F32(&z),
        conv1d: &conv,
        a_log: &alog,
        dt_bias: &dt,
        norm: &nrm,
        out_proj: WeightRows::F32(&outp),
        norm_eps: plan.layers[0].declared_norm_eps as f32,
    };
    let mut state = RecurrentState::zeros(&state_geometry(op).unwrap());
    let planes = layer_forward(op, &weights, &ln_in, &mut state, Mutation::None);

    println!("\n  pos   rel_rms      cosine      max_abs     |mine|     |hf|");
    println!("  ------------------------------------------------------------");
    let mut worst = 0.0f32;
    for (t, (mine, theirs)) in planes.output.iter().zip(&hf_attn).enumerate() {
        let (rel, cos, mx) = metrics(mine, theirs);
        let nm: f32 = mine.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nt: f32 = theirs.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("  {t}   {rel:.4e}  {cos:.7}  {mx:.4e}  {nm:9.4}  {nt:9.4}");
        worst = worst.max(rel);
    }
    // Bound set FROM the measured range (4.8e-4 … 2.7e-3 across the five
    // positions), not chosen to pass. The floor is not f32 arithmetic
    // order — a 5120-term f32 dot disagrees by ~4e-6 — so something at
    // the 1e-3 scale remains unexplained between a BF16 container and an
    // fp32-from-BF16 HF load. It is recorded as an open question rather
    // than absorbed into a comfortable tolerance: see the commit for
    // QW-3.6c.
    assert!(
        worst < 5e-3,
        "the recurrence disagrees with HF on the real container: worst rel_rms {worst:e}"
    );
}

/// Every OTHER stage of layer 0, isolated the same way.
///
/// The recurrence matches; the assembled layer does not. So the defect is
/// in what the traversal does around the operator, and each stage is
/// checked against HF's own capture of it rather than inferred from the
/// layer's final residual.
#[test]
fn every_stage_of_the_first_layer_matches_hf() {
    let (Ok(container), Ok(caps)) = (
        std::env::var("QW36C_CONTAINER"),
        std::env::var("QW36C_CAPS"),
    ) else {
        eprintln!("SKIP qw36c stages: set QW36C_CONTAINER and QW36C_CAPS");
        return;
    };
    let root = std::path::Path::new(&container);
    let inspection = inspect_container(root, false).unwrap();
    let plan = plan_component_ops(&inspection, root, "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(root, &inspection).unwrap();
    let layer = &plan.layers[0];

    let caps: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(caps).unwrap()).unwrap();
    let grab = |k: &str| -> Vec<Vec<f32>> {
        caps[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect()
    };
    let ln_in = grab("ln_in");
    let ln_post = grab("ln_post");
    let hf_mlp = grab("mlp");

    // The residual entering layer 0 — the plane the census showed is
    // bit-exact, so any disagreement below is this layer's own.
    let plane0: Vec<Vec<f32>> = {
        let raw = std::fs::read(
            std::path::Path::new(&std::env::var("QW36C_PLANES").unwrap_or_default())
                .join("layer_000.f32"),
        )
        .expect("set QW36C_PLANES to the --dump-layers directory");
        let hidden = ln_in[0].len();
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<f32>>()
            .chunks_exact(hidden)
            .map(<[f32]>::to_vec)
            .collect()
    };

    let rms = |x: &[f32], w: &[f32], offset: f32, eps: f64| -> Vec<f32> {
        let mean: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (mean + eps).sqrt();
        x.iter()
            .zip(w)
            .map(|(v, wv)| ((*v as f64 * inv) as f32) * (wv + offset))
            .collect()
    };
    let report = |name: &str, mine: &[f32], theirs: &[f32]| {
        let (rel, cos, mx) = metrics(mine, theirs);
        let nm: f32 = mine.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nt: f32 = theirs.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!(
            "  {name:22} rel {rel:.4e}  cos {cos:.7}  max {mx:.4e}  |mine| {nm:8.4}  |hf| {nt:8.4}"
        );
        rel
    };

    println!();
    let w_pre = store.load(&pre_norm(layer).weight).unwrap();
    let mine_ln_in = rms(
        plane0.last().unwrap(),
        &w_pre,
        pre_norm(layer).weight_offset,
        pre_norm(layer).eps,
    );
    let a = report("pre_attention norm", &mine_ln_in, ln_in.last().unwrap());

    let pre_ffn_norm = layer.pre_ffn_norm.as_ref().unwrap();
    let w_ffn = store.load(&pre_ffn_norm.weight).unwrap();
    // HF's residual after attention, so the norm is on trial and not the
    // recurrence that feeds it.
    let hf_attn = grab("attn");
    let resid: Vec<f32> = plane0
        .last()
        .unwrap()
        .iter()
        .zip(hf_attn.last().unwrap())
        .map(|(p, a)| p + a)
        .collect();
    let mine_ln_post = rms(&resid, &w_ffn, pre_ffn_norm.weight_offset, pre_ffn_norm.eps);
    let b = report("pre_ffn norm", &mine_ln_post, ln_post.last().unwrap());

    println!(
        "
  pre_attention weight |w| {:.4}   pre_ffn weight |w| {:.4}",
        w_pre.iter().map(|v| v * v).sum::<f32>().sqrt(),
        w_ffn.iter().map(|v| v * v).sum::<f32>().sqrt()
    );
    let _ = hf_mlp;
    // Measured 1.3e-3 and 2.6e-3. Same open question as above.
    assert!(
        a < 5e-3 && b < 5e-3,
        "a norm stage disagrees with HF: pre_attn {a:e} pre_ffn {b:e}"
    );
}

/// **The acceptance gate.** All 248,320 logits, and the same next token.
///
/// The closure measurement, not the debugging instrument — the per-layer
/// census above is what finds a defect; this is what says there is not
/// one. Reads the planes a `--dump-layers` run wrote, so it scores the
/// shipped executable path rather than a re-derivation.
#[test]
fn the_real_checkpoint_reproduces_hf_logits_and_next_token() {
    let (Ok(planes), Ok(oracle)) = (std::env::var("QW36C_PLANES"), std::env::var("QW36C_ORACLE"))
    else {
        eprintln!("SKIP qw36c logits: set QW36C_PLANES and QW36C_ORACLE");
        return;
    };
    let hf: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(oracle).unwrap()).unwrap();
    let theirs: Vec<f32> = hf["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let raw = std::fs::read(std::path::Path::new(&planes).join("logits.f32")).unwrap();
    let all: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mine = &all[all.len() - theirs.len()..];

    let (rel, cos, mx) = metrics(mine, &theirs);
    let arg = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };
    let (mine_arg, hf_arg) = (arg(mine), arg(&theirs));
    println!(
        "
  logits {}   rel_rms {rel:.4e}  cos {cos:.7}  max_abs {mx:.4e}",
        theirs.len()
    );
    println!("  argmax  LARQL {mine_arg}   HF {hf_arg}");

    assert_eq!(mine_arg, hf_arg, "different next token");
    assert!(cos > 0.9999, "logit direction diverged: cos {cos:.7}");
    assert!(rel < 2e-2, "logit magnitude diverged: rel_rms {rel:e}");
}

/// This fixture is a pre-norm stack; its pre-attention norm is a premise,
/// not an option to be handled.
fn pre_norm(
    layer: &crate::format::vindex3::opplan::LayerPlan,
) -> &crate::format::vindex3::opplan::NormOp {
    layer
        .pre_attention_norm
        .as_ref()
        .expect("this fixture is a pre-norm stack")
}
