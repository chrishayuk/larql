//! Rung 5b: KDA's four wide projections on Metal, at real weights.
//!
//! Narrow by construction — only q/k/v/o move. The convolution, q/k
//! norms, low-rank gates, decay and the recurrence stay on the CPU path
//! the Kimi arc already proved byte-identical to `modeling_kimi.py`.
//!
//! **Why this rung is worth more than its own compute.** Rung 5a priced
//! one CPU↔GPU command-buffer boundary at ~0.23 ms. KDA's dependency
//! chain forces at least two crossings a layer once its projections are
//! on-device — q/k/v share an input and can go together, but `o_proj`'s
//! input does not exist until the recurrence has run. Two is the floor
//! for this shape; four (one per matrix) would spend ~0.9 ms a layer on
//! orchestration alone. So the arms differ in submission count as much
//! as in kernel, and the report says which.
//!
//! **Every boundary compared independently.** `KdaPlanes` exposes
//! fifteen — the three raw projections, the three convolutions, the two
//! normalised streams, the low-rank gate, the decay, beta, the
//! recurrence output, the output gate, the gated norm, and the layer
//! output. A Metal error in `q_proj` that a downstream L2 normalisation
//! would flatten shows up in the plane it happened in, not three stages
//! later.
//!
//! The attention output is additionally scored against
//! `out_attention_output.f32` — `modeling_kimi.py`'s own answer for this
//! layer on this input — so the claim is not two copies of larql
//! agreeing with each other.
//!
//! ```text
//! LARQL_KIMI_KDA_LAYER_FIXTURE=/tmp/kimi_kda_layer_fixture \
//!   cargo test -p larql-vindex --features gpu --release --lib kda_metal -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use larql_compute_metal::trait_impl::kimi_layer::ExpertEncoding;
use larql_compute_metal::MetalBackend;
use larql_models::config::{KdaGeometry, NormType};
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kda;
use crate::format::vindex3::opplan::exec::kda::{
    layer_forward_with, zero_state, CpuKdaProjections, KdaPlanes, KdaProjections, KdaWeights,
    Mutation,
};
use crate::format::vindex3::opplan::exec::kda_metal::{MetalKdaProjections, QkvSubmission};
use crate::format::vindex3::opplan::exec::kernels::norm;
use crate::format::vindex3::opplan::exec::timing::{ledger, OpClass};

const FIXTURE_ENV: &str = "LARQL_KIMI_KDA_LAYER_FIXTURE";

/// Same ceiling the layer's own oracle gate uses: full-width real
/// weights sum thousands of terms, so two orderings of the same
/// arithmetic separate further than a small fixture would.
const TOLERANCE: f32 = 3e-4;

/// Warmup and timed repeats. Sized against the ramp factor this test
/// reports, per the block rung's lesson that warmup must be
/// workload-shaped: one pass here moves ~72 MiB.
const WARMUP: usize = 12;
const ITERS: usize = 15;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(PathBuf::from)
}

fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("{name}.f32: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_bf16(dir: &Path, name: &str) -> Vec<u16> {
    let bytes = std::fs::read(dir.join(format!("{name}.bf16")))
        .unwrap_or_else(|e| panic!("{name}.bf16: {e}"));
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Everything one KDA layer needs, owned so `KdaWeights` can borrow it.
struct Fixture {
    hidden: usize,
    geometry: KdaGeometry,
    eps: f32,
    /// `input_layernorm(x)` — what the attention actually reads.
    normed: Vec<f32>,
    /// `modeling_kimi.py`'s own attention output for this input.
    oracle_attention: Vec<f32>,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    o: Vec<u16>,
    qc: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
    fa: Vec<f32>,
    fb: Vec<f32>,
    ga: Vec<f32>,
    gb: Vec<f32>,
    bp: Vec<f32>,
    al: Vec<f32>,
    dt: Vec<f32>,
    on: Vec<f32>,
}

impl Fixture {
    fn weights(&self) -> KdaWeights<'_> {
        KdaWeights {
            q_proj: WeightRows::Bf16(&self.q),
            k_proj: WeightRows::Bf16(&self.k),
            v_proj: WeightRows::Bf16(&self.v),
            q_conv1d: &self.qc,
            k_conv1d: &self.kc,
            v_conv1d: &self.vc,
            f_a_proj: &self.fa,
            f_b_proj: &self.fb,
            g_a_proj: &self.ga,
            g_b_proj: &self.gb,
            b_proj: &self.bp,
            a_log: &self.al,
            dt_bias: &self.dt,
            o_norm: &self.on,
            o_proj: WeightRows::Bf16(&self.o),
            norm_eps: self.eps,
            // The rank the two gate factorisations meet at, read from this
            // fixture's own `f_a_proj` rather than assumed equal to the head
            // dim: on this checkpoint the two coincide, and the executor no
            // longer takes that coincidence as its definition.
            gate_rank: self.fa.len() / self.hidden,
        }
    }

    /// Bytes the four projections read — what a GB/s figure is over.
    fn projection_bytes(&self) -> f64 {
        ((self.q.len() + self.k.len() + self.v.len() + self.o.len()) * 2) as f64
    }
}

fn load(dir: &Path) -> Fixture {
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    let hidden = manifest["hidden"].as_u64().unwrap() as usize;
    let eps = manifest["rms_eps"].as_f64().unwrap();
    let geometry = KdaGeometry {
        num_heads: manifest["num_heads"].as_u64().unwrap() as usize,
        head_dim: manifest["head_dim"].as_u64().unwrap() as usize,
        conv_kernel: 4,
    };
    assert_eq!(
        (geometry.num_heads, geometry.head_dim),
        (32, 128),
        "this gate exists to run REAL geometry"
    );

    let x = read_f32(dir, "input");
    let input_norm_weight = read_f32(dir, "input_norm_weight");
    let normed = norm(NormType::RmsNorm, &x, &input_norm_weight, 0.0, eps);
    let kda = |n: &str| read_f32(dir, &format!("kda_{n}"));
    let kda_bf16 = |n: &str| read_bf16(dir, &format!("kda_{n}"));

    Fixture {
        hidden,
        geometry,
        eps: eps as f32,
        normed,
        oracle_attention: read_f32(dir, "out_attention_output"),
        q: kda_bf16("q_proj"),
        k: kda_bf16("k_proj"),
        v: kda_bf16("v_proj"),
        o: kda_bf16("o_proj"),
        qc: kda("q_conv1d"),
        kc: kda("k_conv1d"),
        vc: kda("v_conv1d"),
        fa: kda("f_a_proj"),
        fb: kda("f_b_proj"),
        ga: kda("g_a_proj"),
        gb: kda("g_b_proj"),
        bp: kda("b_proj"),
        al: kda("a_log"),
        dt: kda("dt_bias"),
        on: kda("o_norm"),
    }
}

fn setup() -> Option<(MetalBackend, Fixture)> {
    let dir = match fixture_dir() {
        Some(d) => d,
        None => {
            eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
            return None;
        }
    };
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => {
            // On macOS this is NOT an absent device — it is a shader
            // library that failed to compile, and `MetalBackend::new`
            // reports both the same way. Skipping there turns a broken
            // build into a green run: a `log1p` that MSL does not have
            // once made every Metal gate in this tree "pass" by
            // skipping. Fail loudly where a device is supposed to exist.
            #[cfg(target_os = "macos")]
            panic!(
                "MetalBackend::new() returned None on macOS — the shader library \
                 almost certainly failed to compile. Run `cargo test -p \
                 larql-compute-metal --lib` to see the compiler's own message."
            );
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("skipped: no Metal device on this host");
                return None;
            }
        }
    };
    Some((metal, load(&dir)))
}

/// Every boundary `KdaPlanes` exposes, named, so a disagreement reports
/// the stage it happened in rather than "the layer".
fn planes(p: &KdaPlanes) -> [(&'static str, &Vec<f32>); 15] {
    [
        ("q_proj", &p.q_proj),
        ("k_proj", &p.k_proj),
        ("v_proj", &p.v_proj),
        ("q_conv", &p.q_conv),
        ("k_conv", &p.k_conv),
        ("v_conv", &p.v_conv),
        ("q_norm", &p.q_norm),
        ("k_norm", &p.k_norm),
        ("f_lowrank", &p.f_lowrank),
        ("g_decay", &p.g_decay),
        ("beta", &p.beta),
        ("recurrent_out", &p.recurrent_out),
        ("o_gate", &p.o_gate),
        ("o_norm", &p.o_norm),
        ("output", &p.output),
    ]
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run(fx: &Fixture, projections: &dyn KdaProjections) -> KdaPlanes {
    let mut state = zero_state(fx.geometry);
    layer_forward_with(
        projections,
        &fx.normed,
        fx.hidden,
        fx.weights(),
        fx.geometry,
        &mut state,
        Mutation::None,
    )
}

/// The gate: both Metal submission shapes, every plane, against the CPU
/// path AND against the checkpoint's own attention output.
#[test]
fn metal_projections_match_the_cpu_path_and_the_oracle_plane_by_plane() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let cpu = run(&fx, &CpuKdaProjections);
    assert!(
        max_abs(&cpu.output, &fx.oracle_attention) < TOLERANCE,
        "control: the CPU arm must match the checkpoint before Metal is judged against it"
    );

    for submission in [QkvSubmission::Batched, QkvSubmission::Grouped] {
        let projector = MetalKdaProjections::new(&metal, fx.weights(), submission);
        let got = run(&fx, &projector);
        for ((name, a), (_, b)) in planes(&got).into_iter().zip(planes(&cpu)) {
            let d = max_abs(a, b);
            eprintln!("[kda] {submission:?} {name:>13}: max|Δ| vs cpu {d:e}");
            assert!(
                d < TOLERANCE,
                "{submission:?} {name}: max|Δ| {d:e} vs the CPU path"
            );
        }
        let d = max_abs(&got.output, &fx.oracle_attention);
        eprintln!("[kda] {submission:?} {:>13}: max|Δ| vs HF  {d:e}", "output");
        assert!(
            d < TOLERANCE,
            "{submission:?}: max|Δ| {d:e} vs the checkpoint's own attention output"
        );
    }
}

/// **Control.** The gate must fail when the projections read the wrong
/// weights — otherwise a Metal path that quietly returned the CPU's
/// answer, or an all-zero plane a downstream norm rescued, would pass.
#[test]
fn a_transposed_projection_is_rejected() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let cpu = run(&fx, &CpuKdaProjections);

    // Swap q and k. Both are `[width, hidden]` real weights, so nothing
    // faults — the layer computes a plausible attention output from the
    // wrong streams, which is exactly the failure a tolerance-only gate
    // on the final vector would miss.
    let mut swapped = fx.weights();
    swapped.q_proj = WeightRows::Bf16(&fx.k);
    swapped.k_proj = WeightRows::Bf16(&fx.q);
    let projector = MetalKdaProjections::new(&metal, swapped, QkvSubmission::Grouped);
    let mut state = zero_state(fx.geometry);
    let got = layer_forward_with(
        &projector,
        &fx.normed,
        fx.hidden,
        swapped,
        fx.geometry,
        &mut state,
        Mutation::None,
    );
    let d = max_abs(&got.output, &cpu.output);
    assert!(
        d > TOLERANCE,
        "swapping q and k moved the output by only {d:e} — this gate cannot \
         tell the projections apart"
    );
    eprintln!("[kda] control: q/k swapped moves the output by {d:e} (tolerance {TOLERANCE:e})");
}

/// What the projections cost on each side, and how many CPU↔GPU
/// crossings each arm makes.
///
/// **Priced from the op-class ledger, not from layer wall.** The four
/// projections are a minority of a KDA layer's time — the convolution,
/// gates, gated norm and the `[32][128][128]` recurrence are all still
/// on the CPU and unchanged by this rung — so a whole-layer wall would
/// dilute the thing under test with work no arm varies. The ledger times
/// exactly the four classes that moved, and layer wall is reported
/// beside it so the dilution is visible rather than hidden.
///
/// On the Metal arms `KdaQProj` carries q, k and v together: they are one
/// indivisible submission there, and splitting one measured interval
/// three ways would invent a number.
///
/// Arms interleaved with a ramp factor, per this ladder's bench protocol.
/// The CPU arm is the proven row-parallel path, not a straw man.
#[test]
fn report_kda_projection_cost() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let bytes = fx.projection_bytes();
    let batched = MetalKdaProjections::new(&metal, fx.weights(), QkvSubmission::Batched);
    let grouped = MetalKdaProjections::new(&metal, fx.weights(), QkvSubmission::Grouped);
    // `None` for the CPU arm — it has no GPU window to report, which is
    // itself the point: its cost is all arithmetic and no boundary.
    let arms: [(
        &str,
        usize,
        &dyn KdaProjections,
        Option<&MetalKdaProjections<'_>>,
    ); 3] = [
        ("cpu (row-parallel)", 0, &CpuKdaProjections, None),
        ("metal qkv batched ", 2, &batched, Some(&batched)),
        ("metal qkv grouped ", 2, &grouped, Some(&grouped)),
    ];
    const PROJECTION_CLASSES: [OpClass; 4] = [
        OpClass::KdaQProj,
        OpClass::KdaKProj,
        OpClass::KdaVProj,
        OpClass::KdaOProj,
    ];

    // One pass, returning (layer wall ms, projection-only ms from the
    // ledger). The ledger is process-global, so it is zeroed immediately
    // before the pass and read immediately after.
    let once = |p: &dyn KdaProjections, m: Option<&MetalKdaProjections<'_>>| -> (f64, f64, f64) {
        ledger().reset();
        if let Some(m) = m {
            m.take_gpu_ms();
        }
        let t = Instant::now();
        std::hint::black_box(run(&fx, p));
        let wall = t.elapsed().as_secs_f64() * 1000.0;
        let proj: u64 = PROJECTION_CLASSES
            .iter()
            .map(|c| ledger().get(*c).nanos)
            .sum();
        assert_eq!(
            ledger().nested(),
            0,
            "overlapping timers invalidate the split"
        );
        let gpu = m.map(|m| m.take_gpu_ms()).unwrap_or(0.0);
        (wall, proj as f64 / 1e6, gpu)
    };

    for _ in 0..WARMUP {
        for (_, _, p, m) in &arms {
            once(*p, *m);
        }
    }
    let mut wall: Vec<Vec<f64>> = arms.iter().map(|_| Vec::new()).collect();
    let mut proj: Vec<Vec<f64>> = arms.iter().map(|_| Vec::new()).collect();
    let mut gpu: Vec<Vec<f64>> = arms.iter().map(|_| Vec::new()).collect();
    for _ in 0..ITERS {
        for (i, (_, _, p, m)) in arms.iter().enumerate() {
            let (w, pr, g) = once(*p, *m);
            wall[i].push(w);
            proj[i].push(pr);
            gpu[i].push(g);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (wall[0].len() / 3).max(1);
    let ramp = mean(&wall[0][..third]) / mean(&wall[0][wall[0].len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };

    eprintln!(
        "[kda] one layer, {} heads x {} dim, {:.1} MiB across q/k/v/o",
        fx.geometry.num_heads,
        fx.geometry.head_dim,
        bytes / (1024.0 * 1024.0),
    );
    let mut baseline: Option<f64> = None;
    for (i, (name, crossings, _, _)) in arms.iter().enumerate() {
        let (w, pr, g) = (
            median(&mut wall[i]),
            median(&mut proj[i]),
            median(&mut gpu[i]),
        );
        let device = if g > 0.0 {
            format!(
                "gpu {g:.3} ms {:>6.1} GB/s, host {:.3} ms over {crossings} crossings",
                bytes / 1e6 / g,
                pr - g,
            )
        } else {
            format!("{crossings} CPU<->GPU crossings")
        };
        eprintln!(
            "[kda]   {name}  projections {pr:.3} ms {:>6.1} GB/s  |  {device}  |  \
             layer wall {w:.3} ms{}",
            bytes / 1e6 / pr,
            match baseline {
                None => String::new(),
                Some(b) => format!("  [{:.2}x on projections]", b / pr),
            }
        );
        if baseline.is_none() {
            baseline = Some(pr);
        }
    }
    eprintln!("[kda]   ramp {ramp:.2}x — 1.00 means the machine held still");
    assert!(ramp.is_finite());
}

// ── R5c: one complete KDA attention operation, one GPU ownership interval ──

/// The device path's view of the layer's weights.
///
/// `q|k|v` concatenated because the grouped kernel binds one buffer, and
/// held by the caller for the projector's whole life: the device buffer
/// cache keys on `(ptr, len)`, so a bank dropped and reallocated at the
/// same size would silently alias the previous one.
struct DeviceWeights {
    qkv_bank: Vec<u8>,
    qkv_offsets: [ExpertOffset; 3],
    o_proj: Vec<u8>,
}

fn codes_to_bytes(w: &[u16]) -> Vec<u8> {
    w.iter().flat_map(|c| c.to_le_bytes()).collect()
}

impl DeviceWeights {
    fn build(fx: &Fixture) -> Self {
        let per = fx.q.len() * 2;
        let mut bank = Vec::with_capacity(3 * per);
        for m in [&fx.q, &fx.k, &fx.v] {
            bank.extend_from_slice(&codes_to_bytes(m));
        }
        Self {
            qkv_bank: bank,
            qkv_offsets: [
                ExpertOffset(0),
                ExpertOffset(per as u32),
                ExpertOffset((2 * per) as u32),
            ],
            o_proj: codes_to_bytes(&fx.o),
        }
    }

    fn refs<'a>(&'a self, fx: &'a Fixture) -> KdaDeviceWeights<'a> {
        KdaDeviceWeights {
            qkv_bank: &self.qkv_bank,
            qkv_offsets: &self.qkv_offsets,
            o_proj: &self.o_proj,
            projection_encoding: ExpertEncoding::Bf16,
            q_conv1d: &fx.qc,
            k_conv1d: &fx.kc,
            v_conv1d: &fx.vc,
            f_a_proj: &fx.fa,
            f_b_proj: &fx.fb,
            g_a_proj: &fx.ga,
            g_b_proj: &fx.gb,
            b_proj: &fx.bp,
            a_log: &fx.al,
            dt_bias: &fx.dt,
            o_norm: &fx.on,
            norm_eps: fx.eps,
        }
    }
}

fn device_shape(fx: &Fixture) -> KdaShape {
    KdaShape {
        hidden: fx.hidden,
        num_heads: fx.geometry.num_heads,
        head_dim: fx.geometry.head_dim,
        conv_kernel: fx.geometry.conv_kernel,
    }
}

/// **R5c's gate.** The whole attention operation on device — projections,
/// convolution, q/k norms, gates, the delta rule, the gated norm and
/// `o_proj` — in ONE command buffer, against the CPU path plane by
/// plane and against the checkpoint's own output.
///
/// Two of the fifteen planes are head-wide reductions the device sums in
/// a threadgroup tree where the CPU sums them in order, so those differ
/// by reassociation. The recurrence does not: one thread owns one value
/// column for a whole head, so its accumulation over `kk` is the same
/// ascending sequence with no cross-thread reduction to reorder it.
/// Which is which is reported rather than argued.
#[test]
fn the_whole_attention_on_device_matches_plane_by_plane() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let cpu = run(&fx, &CpuKdaProjections);
    let dw = DeviceWeights::build(&fx);
    let shape = device_shape(&fx);
    let state = KdaDeviceState::zeros(&metal, shape);
    let got = metal
        .kda_attention_step_traced(dw.refs(&fx), shape, &state, &fx.normed)
        .expect("device attention step");

    let pairs: [(&str, &Vec<f32>, &Vec<f32>); 15] = [
        ("q_proj", &got.q_proj, &cpu.q_proj),
        ("k_proj", &got.k_proj, &cpu.k_proj),
        ("v_proj", &got.v_proj, &cpu.v_proj),
        ("q_conv", &got.q_conv, &cpu.q_conv),
        ("k_conv", &got.k_conv, &cpu.k_conv),
        ("v_conv", &got.v_conv, &cpu.v_conv),
        ("q_norm", &got.q_norm, &cpu.q_norm),
        ("k_norm", &got.k_norm, &cpu.k_norm),
        ("f_lowrank", &got.f_lowrank, &cpu.f_lowrank),
        ("g_decay", &got.g_decay, &cpu.g_decay),
        ("beta", &got.beta, &cpu.beta),
        ("recurrent_out", &got.recurrent_out, &cpu.recurrent_out),
        ("o_gate", &got.o_gate, &cpu.o_gate),
        ("o_norm", &got.o_norm, &cpu.o_norm),
        ("output", &got.output, &cpu.output),
    ];
    for (name, a, b) in pairs {
        let d = max_abs(a, b);
        eprintln!("[r5c] {name:>13}: max|Δ| vs cpu {d:e}");
        assert!(d < TOLERANCE, "{name}: max|Δ| {d:e} vs the CPU path");
    }
    let d = max_abs(&got.output, &fx.oracle_attention);
    eprintln!(
        "[r5c] {:>13}: max|Δ| vs HF  {d:e}  (gpu {:.3} ms, 1 crossing)",
        "output", got.gpu_ms
    );
    assert!(d < TOLERANCE, "output vs the checkpoint: max|Δ| {d:e}");
}

/// **State parity across consecutive tokens.** The recurrent matrix and
/// the three convolution windows stay on device; nothing reads them
/// back. So the question is whether they still hold what the CPU's would
/// after the same sequence — a drift of one part in `1e7` per token
/// would be invisible at token one and fatal at token fifty.
///
/// Run the same positions through both, and compare the STATE as well as
/// the output at every step.
#[test]
fn device_state_tracks_the_cpu_across_consecutive_tokens() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let dw = DeviceWeights::build(&fx);
    let shape = device_shape(&fx);
    let device_state = KdaDeviceState::zeros(&metal, shape);
    let mut cpu_state = zero_state(fx.geometry);

    // The fixture is one position, so successive tokens are made by
    // perturbing it — the point is that the state ADVANCES and stays in
    // step, which needs the inputs to differ, not to be real.
    const TOKENS: usize = 8;
    for t in 0..TOKENS {
        let x: Vec<f32> = fx
            .normed
            .iter()
            .enumerate()
            .map(|(i, v)| v + ((i + t * 31) as f32 * 0.001).sin() * 0.05)
            .collect();

        let mut planes = KdaPlanes::default();
        let cpu_out = crate::format::vindex3::opplan::exec::kda::step_with(
            &CpuKdaProjections,
            &x,
            fx.weights(),
            fx.geometry,
            &mut cpu_state,
            &mut planes,
            Mutation::None,
        );
        let (got, _gpu) = metal
            .kda_attention_step(dw.refs(&fx), shape, &device_state, &x)
            .expect("device step");

        let (rec, conv) = device_state.read_back();
        let out_d = max_abs(&got, &cpu_out);
        let rec_d = max_abs(&rec, cpu_state.buffer(kda::RECURRENT).cells());
        let conv_d = (0..3)
            .map(|i| max_abs(&conv[i], cpu_state.buffer(kda::CONV_Q + i).cells()))
            .fold(0.0f32, f32::max);
        eprintln!("[r5c] token {t}: output max|Δ| {out_d:e}  recurrent {rec_d:e}  conv {conv_d:e}");
        assert!(out_d < TOLERANCE, "token {t} output: max|Δ| {out_d:e}");
        assert!(
            rec_d < TOLERANCE,
            "token {t} recurrent state: max|Δ| {rec_d:e}"
        );
        assert!(
            conv_d < TOLERANCE,
            "token {t} conv windows: max|Δ| {conv_d:e}"
        );
    }
}

/// **Control.** A fresh device state must NOT reproduce a state that has
/// already advanced — otherwise the test above would pass for a path
/// that quietly ignored its state buffer and recomputed from zero every
/// token, which is precisely the bug an on-device recurrence invites.
#[test]
fn a_reset_device_state_gives_a_different_answer() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let dw = DeviceWeights::build(&fx);
    let shape = device_shape(&fx);
    let carried = KdaDeviceState::zeros(&metal, shape);

    let (first, _) = metal
        .kda_attention_step(dw.refs(&fx), shape, &carried, &fx.normed)
        .expect("token 0");
    let (second, _) = metal
        .kda_attention_step(dw.refs(&fx), shape, &carried, &fx.normed)
        .expect("token 1, carrying state");
    let fresh = KdaDeviceState::zeros(&metal, shape);
    let (reset, _) = metal
        .kda_attention_step(dw.refs(&fx), shape, &fresh, &fx.normed)
        .expect("token 0 again, fresh state");

    assert_eq!(first, reset, "a fresh state must reproduce the first token");
    let d = max_abs(&second, &first);
    assert!(
        d > TOLERANCE,
        "the second token moved by only {d:e} — the device state is not being carried"
    );
    eprintln!("[r5c] control: carrying state moves the second token by {d:e}");
}

/// **R5c's measurement.** One complete KDA attention operation: the
/// whole host-side path against the whole device-side one.
///
/// The comparison rung 5b could not make. There, only four matvecs moved
/// and the layer still crossed twice; here everything between them moved
/// too, so the attention costs ONE crossing and the CPU does no
/// arithmetic at all. What that is worth is `compute saving + one
/// crossing removed`, and both halves are reported.
///
/// Arms interleaved with a ramp factor. The CPU arm is `step_with` on
/// the proven path, which is what production runs today.
#[test]
fn report_whole_attention_on_device_against_the_host() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let dw = DeviceWeights::build(&fx);
    let shape = device_shape(&fx);
    let bytes = fx.projection_bytes();

    let cpu_arm = || {
        let mut state = zero_state(fx.geometry);
        let mut planes = KdaPlanes::default();
        let t = Instant::now();
        std::hint::black_box(crate::format::vindex3::opplan::exec::kda::step_with(
            &CpuKdaProjections,
            &fx.normed,
            fx.weights(),
            fx.geometry,
            &mut state,
            &mut planes,
            Mutation::None,
        ));
        (t.elapsed().as_secs_f64() * 1000.0, 0.0)
    };
    let device_state = KdaDeviceState::zeros(&metal, shape);
    let gpu_arm = || {
        let t = Instant::now();
        let (out, gpu) = metal
            .kda_attention_step(dw.refs(&fx), shape, &device_state, &fx.normed)
            .expect("device attention");
        std::hint::black_box(out);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    for _ in 0..WARMUP {
        cpu_arm();
        gpu_arm();
    }
    let (mut cw, mut gw, mut gg) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERS {
        cw.push(cpu_arm().0);
        let (w, g) = gpu_arm();
        gw.push(w);
        gg.push(g);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (cw.len() / 3).max(1);
    let ramp = mean(&cw[..third]) / mean(&cw[cw.len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (cpu, gpu_wall, gpu_busy) = (median(&mut cw), median(&mut gw), median(&mut gg));

    eprintln!(
        "[r5c] whole KDA attention, {} heads x {} dim, {:.1} MiB of projections",
        fx.geometry.num_heads,
        fx.geometry.head_dim,
        bytes / (1024.0 * 1024.0),
    );
    eprintln!("[r5c]   host   {cpu:.3} ms  (all stages on CPU, 0 crossings)");
    eprintln!(
        "[r5c]   device {gpu_wall:.3} ms  gpu-busy {gpu_busy:.3} ms, \
         host {:.3} ms over 1 crossing  [{:.2}x]",
        gpu_wall - gpu_busy,
        cpu / gpu_wall,
    );
    eprintln!(
        "[r5c]   at rung 5a's 0.23 ms/crossing, {} more layers in one epoch would \
         remove ~{:.2} ms/token of the remaining host time",
        26,
        26.0 * (gpu_wall - gpu_busy) * 25.0 / 26.0,
    );
    eprintln!("[r5c]   ramp {ramp:.2}x — 1.00 means the machine held still");
    assert!(ramp.is_finite());
}
