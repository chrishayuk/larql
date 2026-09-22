//! PHYSICAL-1 QUALITY, real-weight arm: does Q4_K on KDA's two wide
//! projections preserve the recurrent state trajectory on TRAINED
//! weights?
//!
//! The synthetic gate (`trait_impl::kda::q4_trajectory`) qualifies the
//! mechanism and the controls. It cannot say whether its ~1.2 % state
//! error and ~15 % output error are artifacts of random weights, which
//! are the worst case for a block-scaled format. This binds one real KDA
//! layer and runs the same witness.
//!
//! ```text
//! cargo run --release -p larql-compute-metal --example kda_q4_trajectory_real -- \
//!     ~/chris-models/Kimi-Linear-48B-A3B-Instruct 1 512
//! ```
//!
//! **Kimi only, for now.** GLM's decay form is `ClampedSigmoid`, and
//! `shaders::kda`'s gate is hard-coded to Kimi's `Softplus` with no
//! family selection, so a GLM layer run here would be scored against the
//! wrong recurrence. See the funnel doc §8.2.2a.
//!
//! **What the input is, and is not.** Both arms see unit-RMS Gaussian-ish
//! vectors, which match the SCALE of the normalised hidden states KDA
//! consumes but not their direction distribution. That is adequate for a
//! representation question — the weights are real and both arms see the
//! same input — and it is NOT adequate for a claim about this model's
//! behaviour on text. Real hidden states need the layers below it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;

/// q, k and v — the three convolved streams the bank holds.
const CONV_STREAMS_N: usize = 3;
use larql_compute_metal::trait_impl::kda::trajectory::{
    drift_ratio, kda_trajectory, report, worst,
};
use larql_compute_metal::trait_impl::kda::{KdaDeviceWeights, KdaShape, SmallMatrix};
use larql_compute_metal::trait_impl::kimi_layer::ExpertEncoding;
use larql_compute_metal::MetalBackend;
use larql_models::config::KdaGateForm;

/// The decay-gate form this checkpoint's FAMILY computes, read from
/// `config.json`'s `architectures` and its declared `gate_lower_bound`.
///
/// **Derived from the family, never from the value.** Kimi and GLM both
/// declare `gate_lower_bound: -5.0` and only GLM applies it, so a
/// checkpoint whose family this build does not judge is REFUSED rather
/// than defaulted — serving one form for the other is a 2.8x per-step
/// decay error that compounds with context.
fn gate_form_of(dir: &Path) -> KdaGateForm {
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("parse config");
    let arch = cfg["architectures"][0].as_str().unwrap_or("").to_string();
    let declared = cfg["linear_attn_config"]["gate_lower_bound"].as_f64();
    match arch.as_str() {
        a if a.starts_with("KimiLinear") => {
            println!(
                "family {a}: KdaGateForm::Softplus (declares {declared:?}, applies it nowhere)"
            );
            KdaGateForm::Softplus
        }
        a if a.starts_with("Glm5Next") => {
            let lower_bound = declared.unwrap_or(-5.0) as f32;
            println!("family {a}: KdaGateForm::ClampedSigmoid {{ {lower_bound} }}");
            KdaGateForm::ClampedSigmoid { lower_bound }
        }
        a => panic!(
            "family {a:?} has no judged KDA decay-gate form. Declare it rather than \
             defaulting: gate_lower_bound is present on families that ignore it."
        ),
    }
}

/// One per-projection sensitivity arm: which projection it degraded,
/// its q|k|v bank, its `o_proj`, and its own offset table.
type SensArm = (String, Vec<u8>, Vec<u8>, [ExpertOffset; CONV_STREAMS_N]);

/// One small matrix as loaded:
type SmallLoaded = (&'static str, (Vec<u8>, Vec<f32>), String);

/// One precision arm: its encoding, its `q|k|v` bank, its `o_proj`, and
/// its own offset table. The table's ADDRESS matters — see where `arms`
/// is built.
type Arm = (ExpertEncoding, Vec<u8>, Vec<u8>, [ExpertOffset; 3]);

/// One tensor located inside one shard.
struct Located {
    shard: PathBuf,
    start: usize,
    end: usize,
    shape: Vec<usize>,
    dtype: String,
}

fn locate(dir: &Path, names: &[String]) -> HashMap<String, Located> {
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("model.safetensors.index.json")).expect("index json"),
    )
    .expect("parse index");
    let map = index["weight_map"].as_object().expect("weight_map");
    let mut headers: HashMap<PathBuf, serde_json::Value> = HashMap::new();
    let mut out = HashMap::new();
    for name in names {
        let shard = dir.join(
            map[name]
                .as_str()
                .unwrap_or_else(|| panic!("{name} not in index")),
        );
        let header = headers.entry(shard.clone()).or_insert_with(|| {
            let f = std::fs::File::open(&shard).expect("open shard");
            // SAFETY: read-only mapping of a file we do not mutate.
            let m = unsafe { memmap2::Mmap::map(&f) }.expect("mmap shard");
            let n = u64::from_le_bytes(m[..8].try_into().unwrap()) as usize;
            serde_json::from_slice(&m[8..8 + n]).expect("parse header")
        });
        let n = u64::from_le_bytes(
            std::fs::read(&shard).expect("read")[..8]
                .try_into()
                .unwrap(),
        ) as usize;
        let t = &header[name];
        let off = t["data_offsets"].as_array().expect("offsets");
        let base = 8 + n;
        out.insert(
            name.clone(),
            Located {
                shard,
                start: base + off[0].as_u64().unwrap() as usize,
                end: base + off[1].as_u64().unwrap() as usize,
                shape: t["shape"]
                    .as_array()
                    .expect("shape")
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
                dtype: t["dtype"].as_str().expect("dtype").to_string(),
            },
        );
    }
    out
}

/// Raw bytes, and the f32 values those bytes denote.
///
/// **The KDA block is not one dtype.** Kimi ships its wide projections
/// BF16 and some per-channel vectors F32 in the same layer, so the
/// reader dispatches on the declared dtype and refuses anything else
/// rather than reinterpreting bytes it does not recognise. This also
/// makes "small tensors at source precision" a per-tensor fact, not a
/// per-block one.
fn read_tensor(l: &Located) -> (Vec<u8>, Vec<f32>) {
    if l.dtype == "F32" {
        let f = std::fs::File::open(&l.shard).expect("open");
        // SAFETY: read-only mapping of a file we do not mutate.
        let m = unsafe { memmap2::Mmap::map(&f) }.expect("mmap");
        let bytes = m[l.start..l.end].to_vec();
        let exact = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        return (bytes, exact);
    }
    assert_eq!(l.dtype, "BF16", "unhandled dtype");
    let f = std::fs::File::open(&l.shard).expect("open");
    // SAFETY: read-only mapping of a file we do not mutate.
    let m = unsafe { memmap2::Mmap::map(&f) }.expect("mmap");
    let bytes = m[l.start..l.end].to_vec();
    let exact = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect();
    (bytes, exact)
}

/// The precision ladder, so a state error can be attributed to the
/// REPRESENTATION rather than to the binding: if Q8_0 and Q4_K land on
/// the same number, something other than the codec is moving the state.
fn quantise(enc: ExpertEncoding, v: &[f32]) -> Vec<u8> {
    use larql_compute::cpu::ops::q4_common as q;
    match enc {
        ExpertEncoding::Q80 => q::quantize_q8_0(v),
        ExpertEncoding::Q6K => q::quantize_q6_k(v),
        ExpertEncoding::Q4K => q::quantize_q4_k(v),
        ExpertEncoding::Bf16 => panic!("bf16 is the reference, not a candidate"),
    }
}

/// Read a flat little-endian f32 file written by the oracle's
/// `--raw-dir`.
fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Unit-RMS input, deterministic, shared by both arms.
fn inputs(hidden: usize, steps: usize) -> Vec<Vec<f32>> {
    (0..steps)
        .map(|s| {
            let v: Vec<f32> = (0..hidden)
                .map(|i| ((i as f32) * 0.7351 + s as f32 * 1.9).sin())
                .collect();
            let rms = (v.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
            v.into_iter()
                .map(|x| x / rms.max(f32::MIN_POSITIVE))
                .collect()
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| panic!("usage: <model-dir> [layer] [steps]")),
    );
    let layer: usize = args
        .next()
        .unwrap_or_else(|| "1".into())
        .parse()
        .expect("layer");
    let steps: usize = args
        .next()
        .unwrap_or_else(|| "512".into())
        .parse()
        .expect("steps");
    // Optional 4th argument: a directory of upstream reference tensors
    // from `tools/oracles/glm5/smoke.py --raw-dir`. Its presence turns
    // on the BASELINE gate below.
    let reference_dir = args.next().map(PathBuf::from);
    // 5th argument "sensitivity" runs the per-projection ladder instead
    // of the uniform one — KDA-REPRESENT-1's first question is not "how
    // few bits" but "which projection's bits actually move the state".
    let sensitivity = args.next().is_some_and(|a| a == "sensitivity");
    // A directory holding `kda_input.f32` from
    // `tools/oracles/glm5/kda_stream_ratio.py` — the REAL post-norm states
    // KDA sees. Unit-RMS synthetic input is 11.3x too large and this
    // operator is not scale-invariant, so no magnitude measured without
    // this is on-manifold.
    let state_bank = std::env::var("KDA_REAL_STATES").ok().map(PathBuf::from);

    // Kimi names its stack `model.layers.N`; GLM nests it under
    // `model.language_model.layers.N`. Same fifteen leaf names in both,
    // so the prefix is the only difference and it is DETECTED, not
    // guessed from the family.
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("model.safetensors.index.json")).expect("index json"),
    )
    .expect("parse index");
    let wm = index["weight_map"].as_object().expect("weight_map");
    let stem = ["model", "model.language_model"]
        .into_iter()
        .find(|s| wm.contains_key(&format!("{s}.layers.{layer}.self_attn.A_log")))
        .unwrap_or_else(|| panic!("no KDA layer {layer} under either known stack prefix"));
    println!("tensor stack prefix: {stem}.layers.{layer}");
    let p = |suffix: &str| format!("{stem}.layers.{layer}.self_attn.{suffix}");
    let names: Vec<String> = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_a_proj.weight",
        "g_b_proj.weight",
        "b_proj.weight",
        "A_log",
        "dt_bias",
        "o_norm.weight",
    ]
    .iter()
    .map(|s| p(s))
    .collect();
    let found = locate(&dir, &names);
    let get = |s: &str| read_tensor(&found[&p(s)]);
    println!(
        "dtypes: {}",
        names
            .iter()
            .map(|n| format!("{}={}", n.rsplit('.').nth(1).unwrap_or(n), found[n].dtype))
            .collect::<Vec<_>>()
            .join(" ")
    );

    let (q_b, q_e) = get("q_proj.weight");
    let (k_b, k_e) = get("k_proj.weight");
    let (v_b, v_e) = get("v_proj.weight");
    let (o_b, o_e) = get("o_proj.weight");

    // Geometry from the TENSOR SHAPES, not the config — the same rule the
    // MLA parity gate uses.
    let width = found[&p("q_proj.weight")].shape[0];
    let hidden = found[&p("q_proj.weight")].shape[1];
    let head_dim = found[&p("o_norm.weight")].shape[0];
    let num_heads = width / head_dim;
    let conv_kernel = *found[&p("q_conv1d.weight")]
        .shape
        .last()
        .expect("conv shape");
    let shape = KdaShape {
        hidden,
        num_heads,
        head_dim,
        conv_kernel,
    };
    println!(
        "layer {layer}: hidden={hidden} heads={num_heads} head_dim={head_dim} \
         width={width} conv_kernel={conv_kernel}"
    );
    assert!(
        ExpertEncoding::Q4K.matrix_bytes(width, hidden).is_some()
            && ExpertEncoding::Q4K.matrix_bytes(hidden, width).is_some(),
        "this layer's projections are not Q4_K-legal (k % 256)"
    );

    let mut bf16_qkv = Vec::new();
    for b in [&q_b, &k_b, &v_b] {
        bf16_qkv.extend_from_slice(b);
    }
    let per_bf16 = q_b.len();
    let bf16_offsets = [
        ExpertOffset(0),
        ExpertOffset(per_bf16 as u32),
        ExpertOffset((2 * per_bf16) as u32),
    ];

    // Per-channel vectors: the KDA kernels read these as f32, and they
    // are ~3 % of the block's non-projection parameters, so widening
    // them costs nothing worth binding differently.
    let small = |s: &str| get(s).1;
    let (qc, kc, vc) = (
        small("q_conv1d.weight"),
        small("k_conv1d.weight"),
        small("v_conv1d.weight"),
    );
    let (a_log, dt, o_norm) = (small("A_log"), small("dt_bias"), small("o_norm.weight"));

    // The five small MATRICES carry ~97 % of the block's non-projection
    // parameters and are bound at the precision the CHECKPOINT stores —
    // per tensor, because this block is not one dtype.
    let mats: Vec<SmallLoaded> = [
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_a_proj.weight",
        "g_b_proj.weight",
        "b_proj.weight",
    ]
    .iter()
    .map(|n| (*n, get(n), found[&p(n)].dtype.clone()))
    .collect();
    let bind = |i: usize| -> SmallMatrix<'_> {
        let (_, (bytes, exact), dtype) = &mats[i];
        match dtype.as_str() {
            "BF16" => SmallMatrix::Bf16(bytes),
            "F32" => SmallMatrix::F32(exact),
            d => panic!("unhandled small-matrix dtype {d}"),
        }
    };
    let stored: usize = mats.iter().map(|(_, (b, _), _)| b.len()).sum();
    let widened: usize = mats.iter().map(|(_, (_, e), _)| e.len() * 4).sum();
    println!(
        "small matrices: {} at source dtype = {:.2} MiB (widened to f32 would be {:.2} MiB, {:+.2} MiB/layer)",
        mats.iter().map(|(n, _, d)| format!("{}={d}", n.split('.').next().unwrap())).collect::<Vec<_>>().join(" "),
        stored as f64 / (1 << 20) as f64,
        widened as f64 / (1 << 20) as f64,
        (stored as f64 - widened as f64) / (1 << 20) as f64,
    );

    let reference = KdaDeviceWeights {
        qkv_bank: &bf16_qkv,
        qkv_offsets: &bf16_offsets,
        o_proj: &o_b,
        projection_encoding: ExpertEncoding::Bf16,
        q_conv1d: &qc,
        k_conv1d: &kc,
        v_conv1d: &vc,
        f_a_proj: bind(0),
        f_b_proj: bind(1),
        g_a_proj: bind(2),
        g_b_proj: bind(3),
        b_proj: bind(4),
        a_log: &a_log,
        dt_bias: &dt,
        o_norm: &o_norm,
        norm_eps: 1e-5,
        gate_form: gate_form_of(&dir),
    };

    let metal = MetalBackend::new().expect("Metal device");

    // BASELINE FIRST. If an upstream reference trajectory is supplied,
    // LARQL's BF16 arm is scored against it BEFORE any quantised arm
    // runs — otherwise a Qx miss has two possible causes and the rung
    // cannot attribute it. Written as an early `return` on failure
    // rather than a warning: scoring a representation on an unverified
    // executor is the mistake this whole rung exists to avoid.
    let xs = match reference_dir {
        Some(ref d) => {
            let flat = read_f32(&d.join("input.f32"));
            let want = read_f32(&d.join("output.f32"));
            let n = flat.len() / hidden;
            assert_eq!(
                flat.len(),
                n * hidden,
                "input.f32 is not a whole number of rows"
            );
            assert_eq!(
                want.len(),
                n * hidden,
                "output.f32 disagrees with input.f32"
            );
            let xs: Vec<Vec<f32>> = (0..n)
                .map(|i| flat[i * hidden..(i + 1) * hidden].to_vec())
                .collect();

            // The oracle ran all n positions in ONE parallel forward; this
            // steps them one at a time carrying recurrent state. Agreement
            // is the whole claim of the recurrent form, so a mismatch here
            // is an executor defect, not a numerical detail.
            let state = larql_compute_metal::trait_impl::kda::KdaDeviceState::zeros(&metal, shape);
            let mut worst_rel = 0.0f32;
            println!("\nBASELINE upstream GLM BF16 -> LARQL Metal BF16, {n} positions");
            for (i, x) in xs.iter().enumerate() {
                let (got, _) = metal
                    .kda_attention_step(reference, shape, &state, x)
                    .expect("bf16 baseline runs");
                let w = &want[i * hidden..(i + 1) * hidden];
                let num: f32 = w.iter().zip(&got).map(|(a, b)| (a - b) * (a - b)).sum();
                let den: f32 = w.iter().map(|a| a * a).sum();
                let rel = (num / den.max(f32::MIN_POSITIVE)).sqrt();
                worst_rel = worst_rel.max(rel);
                println!(
                    "    position {i:>3}  rel-L2 {rel:.3e}  |ref| {:.4}",
                    den.sqrt()
                );
            }
            // Numerical-ordering floor for an f32 reduction of this width,
            // not a tolerance chosen to pass.
            const FLOOR: f32 = 2e-3;
            println!("  worst baseline rel-L2 {worst_rel:.3e} (floor {FLOOR:.0e})");
            if worst_rel > FLOOR {
                println!(
                    "\nBASELINE RED — refusing to score any quantised arm. LARQL's BF16 \n\
                     KDA does not reproduce upstream on this layer, so a Qx result would \n\
                     have two causes. Fix the executor first."
                );
                return;
            }
            println!("  BASELINE GREEN\n");
            xs
        }
        None => inputs(hidden, steps),
    };
    let xs = match &state_bank {
        Some(d) => {
            let flat = read_f32(&d.join("kda_input.f32"));
            let n = flat.len() / hidden;
            let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
            let v: Vec<Vec<f32>> = (0..n)
                .map(|i| flat[i * hidden..(i + 1) * hidden].to_vec())
                .collect();
            println!(
                "ON-MANIFOLD: {n} real post-norm KDA states, mean RMS {:.6} \
                 (a unit-RMS synthetic vector is {:.1}x too large)",
                v.iter().map(|x| rms(x)).sum::<f32>() / n as f32,
                1.0 / (v.iter().map(|x| rms(x)).sum::<f32>() / n as f32)
            );
            v
        }
        None => xs,
    };
    let bf16_bytes = (bf16_qkv.len() + o_b.len()) as f64;

    println!(
        "\n{:<6} {:>8} {:>7} {:>11} {:>11} {:>8} {:>9} {:>9}",
        "enc", "MiB", "bpw", "state_rel", "state_cos", "drift", "out_raw", "out_traj"
    );
    // BUILT UP FRONT, NOT IN THE LOOP, AND THAT IS LOAD-BEARING.
    // `stable_offset_table` caches by (ptr, len). A loop-local
    // `[ExpertOffset; 3]` lands on the same stack address with the same
    // length every iteration, so arms 2 and 3 were handed arm 1's cached
    // table — Q8_0 slot strides applied to a Q4_K bank, reading past its
    // end. It surfaced as NaN on Q6_K and Q4_K while Q8_0, the first
    // arm, read a healthy 0.0084. Holding every arm's offsets alive at a
    // distinct address for the whole run is the fix.
    let arms: Vec<Arm> = [
        ExpertEncoding::Q80,
        ExpertEncoding::Q6K,
        ExpertEncoding::Q4K,
    ]
    .into_iter()
    .map(|enc| {
        let q: Vec<Vec<u8>> = [&q_e, &k_e, &v_e]
            .iter()
            .map(|e| quantise(enc, e))
            .collect();
        let per = q[0].len();
        let mut bank = Vec::new();
        for b in &q {
            bank.extend_from_slice(b);
        }
        let offsets = [
            ExpertOffset(0),
            ExpertOffset(per as u32),
            ExpertOffset((2 * per) as u32),
        ];
        (enc, bank, quantise(enc, &o_e), offsets)
    })
    .collect();

    for (enc, bank, o, offsets) in &arms {
        let candidate = KdaDeviceWeights {
            qkv_bank: bank,
            qkv_offsets: offsets,
            o_proj: o,
            projection_encoding: *enc,
            ..reference
        };
        let t = kda_trajectory(&metal, shape, reference, candidate, &xs, false);
        let (rel, cos, raw, traj) = worst(&t);
        let bytes = (bank.len() + o.len()) as f64;
        println!(
            "{:<6} {:>8.2} {:>7.3} {:>11.6} {:>11.6} {:>8.3} {:>9.4} {:>9.4}",
            enc.name(),
            bytes / (1 << 20) as f64,
            bytes * 8.0 / (4 * width * hidden) as f64,
            rel,
            cos,
            drift_ratio(&t, 8),
            raw,
            traj
        );
        if *enc == ExpertEncoding::Q4K {
            println!();
            report(
                &format!("REAL Q4_K vs BF16, layer {layer}, {steps} steps"),
                &t,
                8,
            );
            println!(
                "    bf16 {:.2} MiB -> Q4_K {:.2} MiB ({:.2}x)",
                bf16_bytes / (1 << 20) as f64,
                bytes / (1 << 20) as f64,
                bf16_bytes / bytes
            );
        }
    }

    // ---- KDA-REPRESENT-1 rung 0: WHERE DO THE BITS MATTER? ----
    //
    // Uniform Q4_K spends the same bits on every projection. The state
    // witness already showed that cannot be right: coarsening `q` moved
    // the recurrent state by EXACTLY zero while moving the output, so at
    // least one of the four is state-inert. This measures each one's
    // contribution separately.
    //
    // **Method: round-trip, not mixed binding.** `projection_encoding`
    // covers the whole q|k|v bank and `o_proj` together, so a mixed-
    // precision bank is not bindable today. Instead each arm quantises
    // ONE projection's values, dequantises them, and binds every arm as
    // BF16 — so the kernel is identical across arms and the only
    // difference is that projection's quantisation error. That answers
    // the sensitivity question without waiting for mixed-precision
    // execution to exist.
    if sensitivity {
        use larql_compute::cpu::ops::q4_common::{dequantize_q4_k, quantize_q4_k};
        let narrow = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        let roundtrip = |v: &[f32]| -> Vec<f32> { dequantize_q4_k(&quantize_q4_k(v), v.len()) };
        // A REFERENCE quantiser, not a shippable codec: symmetric
        // round-to-nearest with one absmax scale per 256-element block.
        // It exists to sweep bit width CONTINUOUSLY, which no fixed ggml
        // format can do, and so to expose the marginal curve — state
        // error reduced per added physical bit. It is strictly weaker
        // than Q4_K at the same width (Q4_K carries a per-block min and
        // 6-bit sub-scales), so every number it produces is a LOWER
        // BOUND on what a real codec can achieve at that width. The
        // Q4_K row below calibrates the offset.
        let rtn = |v: &[f32], bits: u32| -> Vec<f32> {
            let levels = ((1u32 << (bits - 1)) - 1) as f32;
            v.chunks(256)
                .flat_map(|blk| {
                    let amax = blk.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                    let s = if amax > 0.0 { amax / levels } else { 1.0 };
                    blk.iter()
                        .map(|x| (x / s).round().clamp(-levels, levels) * s)
                        .collect::<Vec<f32>>()
                })
                .collect()
        };
        let names = ["q_proj", "k_proj", "v_proj", "o_proj"];
        let exact = [&q_e, &k_e, &v_e, &o_e];

        println!("\nKDA-REPRESENT-1 rung 0: per-projection Q4_K sensitivity, layer {layer}");
        println!(
            "  (one projection round-tripped at a time; every arm binds BF16, so the\n                kernel is identical and only that projection's error differs)\n"
        );
        println!(
            "{:<10} {:>12} {:>12} {:>11} {:>11} {:>10}",
            "quantised", "weight rel", "state_rel", "state_cos", "out_traj", "drift"
        );

        // Every arm's bytes held live at a distinct address for the whole
        // run — the (ptr, len) offset-table aliasing that produced NaN in
        // the uniform ladder applies here too.
        let mut arms_s: Vec<SensArm> = Vec::new();
        // `sel` is compared against both `i` (a nested loop index) and the
        // sentinel `4`, not just used to index `names` — an enumerate()
        // rewrite would still need the raw index for those comparisons.
        #[allow(clippy::needless_range_loop)]
        for sel in 0..5usize {
            let rt: Vec<Vec<f32>> = (0..4)
                .map(|i| {
                    if sel == 4 || sel == i {
                        roundtrip(exact[i])
                    } else {
                        exact[i].clone()
                    }
                })
                .collect();
            let mut bank = Vec::new();
            for v in rt.iter().take(3) {
                bank.extend_from_slice(&narrow(v));
            }
            let per = narrow(&rt[0]).len();
            arms_s.push((
                if sel == 4 {
                    "ALL FOUR".into()
                } else {
                    names[sel].into()
                },
                bank,
                narrow(&rt[3]),
                [
                    ExpertOffset(0),
                    ExpertOffset(per as u32),
                    ExpertOffset((2 * per) as u32),
                ],
            ));
        }
        for (i, (name, bank, o, offsets)) in arms_s.iter().enumerate() {
            let candidate = KdaDeviceWeights {
                qkv_bank: bank,
                qkv_offsets: offsets,
                o_proj: o,
                projection_encoding: ExpertEncoding::Bf16,
                ..reference
            };
            let t = kda_trajectory(&metal, shape, reference, candidate, &xs, false);
            let (rel, cos, _, traj) = worst(&t);
            // Weight-space error, so a projection that is HARD to quantise
            // can be told apart from one whose error simply does not reach
            // the state.
            let src = if i == 4 { exact[0] } else { exact[i.min(3)] };
            let rt = roundtrip(src);
            let num: f32 = src.iter().zip(&rt).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = src.iter().map(|a| a * a).sum();
            println!(
                "{:<10} {:>12.5} {:>12.6} {:>11.6} {:>11.4} {:>10.3}",
                name,
                (num / den).sqrt(),
                rel,
                cos,
                traj,
                drift_ratio(&t, 8)
            );
        }
        println!(
            "\n  CONTROL: the ALL FOUR arm must reproduce the uniform Q4_K bank row printed\n  \
             above. If it does, the round-trip emulation is sound."
        );
        // ---- the allocation curve: state AND output vs bit width ----
        //
        // Two admission boundaries, kept INDEPENDENT rather than summed:
        // enough improvement on one must not buy a failure on the other.
        // Both are set at Q6_K parity, the representation 8.2.2d
        // qualified (state 0.0397, out_traj 0.0962).
        const STATE_BOUND: f32 = 0.10;
        const OUT_BOUND: f32 = 0.10;
        println!("\n\nKDA-REPRESENT-1 rung 1: allocation curve, per projection vs bit width");
        println!("  reference RTN quantiser (per-256 absmax, symmetric) — a LOWER BOUND on a");
        println!(
            "  real codec at the same width. Gates: state_rel <= {STATE_BOUND}, out_traj <= {OUT_BOUND}\n"
        );
        println!(
            "{:<10}{:>7}{:>12}{:>11}{:>12}{:>11}",
            "projection", "bits", "state_rel", "state", "out_traj", "output"
        );
        for (i, name) in names.iter().enumerate() {
            for bits in [2u32, 3, 4, 5, 6, 8] {
                let rt: Vec<Vec<f32>> = (0..4)
                    .map(|j| {
                        if j == i {
                            rtn(exact[j], bits)
                        } else {
                            exact[j].clone()
                        }
                    })
                    .collect();
                let mut bank = Vec::new();
                for v in rt.iter().take(3) {
                    bank.extend_from_slice(&narrow(v));
                }
                let per = narrow(&rt[0]).len();
                let o = narrow(&rt[3]);
                let offs = [
                    ExpertOffset(0),
                    ExpertOffset(per as u32),
                    ExpertOffset((2 * per) as u32),
                ];
                let cand = KdaDeviceWeights {
                    qkv_bank: &bank,
                    qkv_offsets: &offs,
                    o_proj: &o,
                    projection_encoding: ExpertEncoding::Bf16,
                    ..reference
                };
                let t = kda_trajectory(&metal, shape, reference, cand, &xs, false);
                let (rel, _, _, traj) = worst(&t);
                println!(
                    "{:<10}{:>7}{:>12.6}{:>11}{:>12.4}{:>11}",
                    name,
                    bits,
                    rel,
                    if rel <= STATE_BOUND { "pass" } else { "FAIL" },
                    traj,
                    if traj <= OUT_BOUND { "pass" } else { "FAIL" }
                );
            }
        }

        // ---- KDA-BEHAV-1: the FROZEN candidate roster ----
        //
        // Frozen BEFORE any downstream behavioural measurement, so that
        // experiment qualifies a fixed set rather than becoming a search
        // over allocations. Bit widths come from rung 1's on-manifold
        // binding widths and from deliberately more aggressive points
        // below them.
        //
        // The two gates stay INDEPENDENT: a candidate may not buy
        // behavioural quality by violating the recurrent-state gate.
        const ROSTER: [(&str, [u32; 4]); 4] = [
            ("strict", [3, 4, 5, 5]),
            ("slightly-aggr", [3, 4, 4, 4]),
            ("aggressive", [3, 4, 4, 3]),
            ("D-probe", [2, 3, 4, 3]),
        ];
        println!("\n\nKDA-BEHAV-1 candidate roster, FROZEN before behavioural measurement");
        println!("  (q, k, v, o widths; RTN reference quantiser; on-manifold states)\n");
        println!(
            "{:<15}{:>14}{:>10}{:>12}{:>11}{:>12}{:>11}",
            "candidate", "q/k/v/o", "RTN bpw", "state_rel", "state", "out_traj", "output"
        );
        let mut built: Vec<SensArm> = Vec::new();
        for (name, w) in ROSTER {
            let rt: Vec<Vec<f32>> = (0..4).map(|j| rtn(exact[j], w[j])).collect();
            let mut bank = Vec::new();
            for v in rt.iter().take(3) {
                bank.extend_from_slice(&narrow(v));
            }
            let per = narrow(&rt[0]).len();
            built.push((
                name.into(),
                bank,
                narrow(&rt[3]),
                [
                    ExpertOffset(0),
                    ExpertOffset(per as u32),
                    ExpertOffset((2 * per) as u32),
                ],
            ));
        }
        for ((name, bank, o, offs), (_, w)) in built.iter().zip(ROSTER) {
            let cand = KdaDeviceWeights {
                qkv_bank: bank,
                qkv_offsets: offs,
                o_proj: o,
                projection_encoding: ExpertEncoding::Bf16,
                ..reference
            };
            let t = kda_trajectory(&metal, shape, reference, cand, &xs, false);
            let (rel, _, _, traj) = worst(&t);
            println!(
                "{:<15}{:>14}{:>10.2}{:>12.6}{:>11}{:>12.4}{:>11}",
                name,
                format!("{}/{}/{}/{}", w[0], w[1], w[2], w[3]),
                w.iter().sum::<u32>() as f32 / 4.0,
                rel,
                if rel <= STATE_BOUND { "pass" } else { "FAIL" },
                traj,
                if traj <= OUT_BOUND { "pass" } else { "FAIL" }
            );
        }
        println!(
            "\n  The output column is scored against out_traj <= {OUT_BOUND}, which is Q6\n               PARITY BY ANALOGY and not yet an evidence-backed bound. Establishing it is\n               exactly what KDA-BEHAV-1 is for; these rows say which candidates are worth\n               taking to that experiment."
        );
        return;
    }

    // ---- PHYSICAL: steady-state GPU time for ONE real KDA layer ----
    //
    // `kda_attention_step` is one command buffer per step, so `gpu_ms`
    // is that buffer's own GPU span, not wall clock — host glue and
    // submission latency are excluded on purpose, and the dispatch and
    // command-buffer counts are reported beside it because §8.2.1
    // measured a submission at ~130x a dispatch.
    //
    // The median of REPS is reported, not the mean: a scheduler blip
    // moves the mean and not the median, and the whole point is the
    // steady state.
    const WARM: usize = 30;
    const REPS: usize = 200;
    let layers = 34usize; // GLM's declared KDA layer count.
    println!(
        "\nPHYSICAL: one real KDA layer, median of {REPS} steady-state steps \
         (GPU span of its single command buffer)\n"
    );
    println!(
        "{:<6} {:>10} {:>11} {:>12} {:>13} {:>12}",
        "enc", "ms/layer", "x34 layers", "wide MiB", "small MiB", "GB/s"
    );
    let small_bytes = stored as f64
        + (qc.len() + kc.len() + vc.len() + a_log.len() + dt.len() + o_norm.len()) as f64 * 4.0;
    let mut timed: Vec<(String, f64)> = Vec::new();
    for (enc, bank, o, offsets) in
        std::iter::once((&ExpertEncoding::Bf16, &bf16_qkv, &o_b, &bf16_offsets))
            .chain(arms.iter().map(|(e, b, o, f)| (e, b, o, f)))
    {
        let candidate = KdaDeviceWeights {
            qkv_bank: bank,
            qkv_offsets: offsets,
            o_proj: o,
            projection_encoding: *enc,
            ..reference
        };
        let state = larql_compute_metal::trait_impl::kda::KdaDeviceState::zeros(&metal, shape);
        for i in 0..WARM {
            let _ = metal.kda_attention_step(candidate, shape, &state, &xs[i % xs.len()]);
        }
        let mut ms: Vec<f64> = Vec::with_capacity(REPS);
        for i in 0..REPS {
            // Reset periodically: the state advances every step and an
            // unbounded run would drift out of the numeric range the
            // real decode occupies. The WORK is shape-invariant, so this
            // changes no cost.
            if i % 64 == 0 {
                state.reset();
            }
            let (_, t) = metal
                .kda_attention_step(candidate, shape, &state, &xs[i % xs.len()])
                .expect("timed step runs");
            ms.push(t);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let med = ms[REPS / 2];
        let wide = (bank.len() + o.len()) as f64;
        println!(
            "{:<6} {:>10.4} {:>11.2} {:>12.2} {:>13.2} {:>12.1}",
            enc.name(),
            med,
            med * layers as f64,
            wide / (1 << 20) as f64,
            small_bytes / (1 << 20) as f64,
            (wide + small_bytes) / 1e9 / (med / 1e3),
        );
        timed.push((enc.name().to_string(), med));
    }
    if let (Some(bf), Some(q6)) = (
        timed.iter().find(|(n, _)| n == "BF16"),
        timed.iter().find(|(n, _)| n == "Q6_K"),
    ) {
        println!(
            "\n  BF16 -> Q6_K: {:.4} -> {:.4} ms/layer ({:.2}x), {:.2} -> {:.2} ms across 34 layers",
            bf.1, q6.1, bf.1 / q6.1, bf.1 * layers as f64, q6.1 * layers as f64
        );
    }
}
