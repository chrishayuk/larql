//! Rung 5d: one complete Kimi decoder layer on device, including the
//! router→expert-binding seam.
//!
//! The question this answers is not whether residuals and RMS norms can
//! run on Metal — they obviously can, and they cost nothing. It is
//! whether **Metal can compute the routing decision and consume it in
//! the grouped MoE without the host ever seeing a selected expert id.**
//! If it cannot, every layer pays the ~0.23 ms crossing rung 5a priced,
//! for the sake of eight integers, and the layer is not closed.
//!
//! The parity list below is deliberately long. Each of these semantics
//! was proved independently on the CPU already; what is under test now
//! is the DEVICE DEPENDENCY CHAIN, so every link gets its own
//! comparison rather than being inferred from the final vector.
//!
//! Two controls carried forward from the CPU router's own gates, because
//! they catch the failures that still produce plausible output:
//!   * weights gathered from the BIASED selection scores instead of the
//!     unbiased ones — preserves the selection, changes every routed
//!     contribution;
//!   * the correction bias omitted from selection — changes which
//!     experts run.
//!
//! ```text
//! LARQL_KIMI_KDA_LAYER_FIXTURE=/tmp/kimi_kda_layer_fixture \
//!   cargo test -p larql-vindex --features gpu --release --lib kimi_layer_metal -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute::cpu::ops::q4_common::{quantize_q6_k, quantize_q8_0};
use larql_compute_metal::shaders::kimi_layer::NOT_RESIDENT;
use larql_compute_metal::trait_impl::bf16_moe_block::{ExpertBankRef, MoeBlockCall, MoeFfnBanks};
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::grouped_experts::GroupedError;
use larql_compute_metal::trait_impl::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use larql_compute_metal::trait_impl::kimi_layer::{
    AttentionSpec, EncodedRegion, ExpertAddressing, ExpertEncoding, FfnSpec, KimiLayerWeights,
    KimiMoeWeights, ProjectionBank,
};
use larql_compute_metal::MetalBackend;
use larql_models::config::KdaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::kda::{KdaState, KdaWeights};
use crate::format::vindex3::opplan::exec::kimi_kda_layer::kda_decoder_layer_forward;
use crate::format::vindex3::opplan::exec::kimi_moe_block::ExpertWeights;

const FIXTURE_ENV: &str = "LARQL_KIMI_KDA_LAYER_FIXTURE";
/// The same ceiling the CPU layer's own oracle gate uses.
const TOLERANCE: f32 = 3e-4;
/// Warmup and repeats for the timing report. Workload-shaped, per the
/// block rung's lesson: one layer moves ~200 MiB.
const WARMUP: usize = 15;
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

fn read_bf16_bytes(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{name}.bf16"))).unwrap_or_else(|e| panic!("{name}.bf16: {e}"))
}

fn codes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// One projection's routed bank, its table, and the shared branch's own
/// region — everything bf16, as the checkpoint stores it.
fn projection<'a>(routed: &'a [u8], table: &'a [u32], shared: &'a [u8]) -> ProjectionBank<'a> {
    ProjectionBank {
        routed: EncodedRegion {
            bytes: routed,
            encoding: ExpertEncoding::Bf16,
        },
        addressing: ExpertAddressing::Table(table),
        shared: Some(EncodedRegion {
            bytes: shared,
            encoding: ExpertEncoding::Bf16,
        }),
    }
}

/// One layer's weights, owned, plus the resident expert bank the device
/// path binds.
///
/// **Only the selected experts plus the shared branch are resident.**
/// The router still scores all 256 from the real `[256, 2304]` matrix —
/// selection is genuine — but the bank holds nine, and every other
/// expert maps to `NOT_RESIDENT`. That is honest about what a fixture
/// can hold and is exactly the condition the device-side refusal counter
/// exists for: if the router ever picked an absent expert, the step
/// would be refused rather than served another expert's weights.
struct Fixture {
    hidden: usize,
    inter: usize,
    experts: usize,
    top_k: usize,
    renormalize: bool,
    branch_scale: f32,
    eps: f32,
    geometry: KdaGeometry,
    ids_order: Vec<usize>,

    x: Vec<f32>,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    router_weight: Vec<f32>,
    router_bias: Vec<f32>,
    oracle_layer_output: Vec<f32>,

    // KDA
    qkv_bank: Vec<u8>,
    qkv_offsets: [ExpertOffset; 3],
    o_proj: Vec<u8>,
    kda_f32: KdaF32,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    o: Vec<u16>,

    // MoE: gate/up/down banks over the resident experts. The shared
    // branch lives in its OWN allocations — semantic identity, never
    // co-location.
    bank_gate: Vec<u8>,
    bank_up: Vec<u8>,
    bank_down: Vec<u8>,
    residency: Vec<u32>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
    /// Widened codes, for the CPU arm.
    cpu_experts: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)>,
    cpu_shared: (Vec<u16>, Vec<u16>, Vec<u16>),
}

/// KDA's f32 vectors, grouped so `Fixture` stays readable.
struct KdaF32 {
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
    fn kda_cpu(&self) -> KdaWeights<'_> {
        let f = &self.kda_f32;
        KdaWeights {
            q_proj: &self.q,
            k_proj: &self.k,
            v_proj: &self.v,
            q_conv1d: &f.qc,
            k_conv1d: &f.kc,
            v_conv1d: &f.vc,
            f_a_proj: &f.fa,
            f_b_proj: &f.fb,
            g_a_proj: &f.ga,
            g_b_proj: &f.gb,
            b_proj: &f.bp,
            a_log: &f.al,
            dt_bias: &f.dt,
            o_norm: &f.on,
            o_proj: &self.o,
            norm_eps: self.eps,
        }
    }

    fn kda_device(&self) -> KdaDeviceWeights<'_> {
        let f = &self.kda_f32;
        KdaDeviceWeights {
            qkv_bank: &self.qkv_bank,
            qkv_offsets: &self.qkv_offsets,
            o_proj: &self.o_proj,
            q_conv1d: &f.qc,
            k_conv1d: &f.kc,
            v_conv1d: &f.vc,
            f_a_proj: &f.fa,
            f_b_proj: &f.fb,
            g_a_proj: &f.ga,
            g_b_proj: &f.gb,
            b_proj: &f.bp,
            a_log: &f.al,
            dt_bias: &f.dt,
            o_norm: &f.on,
            norm_eps: self.eps,
        }
    }

    fn layer<'a>(&'a self, state: &'a KdaDeviceState) -> KimiLayerWeights<'a> {
        KimiLayerWeights {
            input_norm: &self.input_norm,
            post_attention_norm: &self.post_norm,
            attention: AttentionSpec::Kda {
                weights: self.kda_device(),
                shape: self.shape(),
                state,
            },
            ffn: FfnSpec::Moe(KimiMoeWeights {
                router_weight: &self.router_weight,
                router_bias: &self.router_bias,
                gate: projection(&self.bank_gate, &self.residency, &self.shared_gate),
                up: projection(&self.bank_up, &self.residency, &self.shared_up),
                down: projection(&self.bank_down, &self.residency, &self.shared_down),
                inter: self.inter,
                top_k: self.top_k,
                renormalize: self.renormalize,
                branch_scale: self.branch_scale,
            }),
            norm_eps: self.eps,
        }
    }

    /// `input_layernorm(x)` on the host — only for the attention-alone
    /// decomposition, which needs the same input the layer's own first
    /// dispatch computes.
    fn input_norm_applied(&self) -> Vec<f32> {
        crate::format::vindex3::opplan::exec::kernels::norm(
            larql_models::config::NormType::RmsNorm,
            &self.x,
            &self.input_norm,
            0.0,
            self.eps as f64,
        )
    }

    fn shape(&self) -> KdaShape {
        KdaShape {
            hidden: self.hidden,
            num_heads: self.geometry.num_heads,
            head_dim: self.geometry.head_dim,
            conv_kernel: self.geometry.conv_kernel,
        }
    }
}

fn load(dir: &Path) -> Fixture {
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    let g = |k: &str| manifest[k].as_u64().unwrap() as usize;
    let (hidden, inter, experts, top_k) = (
        g("hidden"),
        g("moe_intermediate_size"),
        g("experts"),
        g("top_k"),
    );
    let geometry = KdaGeometry {
        num_heads: g("num_heads"),
        head_dim: g("head_dim"),
        conv_kernel: 4,
    };
    let ids_order: Vec<usize> = manifest["selected_ids_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!((experts, top_k), (256, 8), "this gate runs REAL geometry");

    let kda = |n: &str| read_f32(dir, &format!("kda_{n}"));
    let (qb, kb, vb) = (
        read_bf16_bytes(dir, "kda_q_proj"),
        read_bf16_bytes(dir, "kda_k_proj"),
        read_bf16_bytes(dir, "kda_v_proj"),
    );
    let per_qkv = qb.len();
    let mut qkv_bank = Vec::with_capacity(3 * per_qkv);
    for b in [&qb, &kb, &vb] {
        qkv_bank.extend_from_slice(b);
    }
    let o_proj = read_bf16_bytes(dir, "kda_o_proj");

    // The resident bank: the selected experts in `selected_ids_order`,
    // then the shared branch. Identity lives in the residency table, not
    // in this order — the router will pick whatever it picks.
    let mut bank_gate = Vec::new();
    let mut bank_up = Vec::new();
    let mut bank_down = Vec::new();
    let mut residency = vec![NOT_RESIDENT; experts];
    let mut cpu_experts = Vec::with_capacity(ids_order.len());
    for &id in &ids_order {
        residency[id] = bank_gate.len() as u32;
        let (g1, g3, g2) = (
            read_bf16_bytes(dir, &format!("expert{id}_w1")),
            read_bf16_bytes(dir, &format!("expert{id}_w3")),
            read_bf16_bytes(dir, &format!("expert{id}_w2")),
        );
        cpu_experts.push((codes(&g1), codes(&g3), codes(&g2)));
        bank_gate.extend_from_slice(&g1);
        bank_up.extend_from_slice(&g3);
        bank_down.extend_from_slice(&g2);
    }
    let (s1, s3, s2) = (
        read_bf16_bytes(dir, "shared_w1"),
        read_bf16_bytes(dir, "shared_w3"),
        read_bf16_bytes(dir, "shared_w2"),
    );
    let cpu_shared = (codes(&s1), codes(&s3), codes(&s2));

    Fixture {
        hidden,
        inter,
        experts,
        top_k,
        renormalize: manifest["moe_renormalize"].as_bool().unwrap(),
        branch_scale: manifest["routed_scaling_factor"].as_f64().unwrap() as f32,
        eps: manifest["rms_eps"].as_f64().unwrap() as f32,
        geometry,
        ids_order,
        x: read_f32(dir, "input"),
        input_norm: read_f32(dir, "input_norm_weight"),
        post_norm: read_f32(dir, "post_attention_norm_weight"),
        router_weight: read_f32(dir, "router_weight"),
        router_bias: read_f32(dir, "router_bias"),
        oracle_layer_output: read_f32(dir, "out_layer_output"),
        q: codes(&qb),
        k: codes(&kb),
        v: codes(&vb),
        o: codes(&o_proj),
        qkv_offsets: [
            ExpertOffset(0),
            ExpertOffset(per_qkv as u32),
            ExpertOffset((2 * per_qkv) as u32),
        ],
        qkv_bank,
        o_proj,
        kda_f32: KdaF32 {
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
        },
        bank_gate,
        bank_up,
        bank_down,
        residency,
        shared_gate: s1,
        shared_up: s3,
        shared_down: s2,
        cpu_experts,
        cpu_shared,
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
            // On macOS this means the shader library failed to compile,
            // not that a device is missing — skipping there turns a
            // broken build into a green run.
            #[cfg(target_os = "macos")]
            panic!(
                "MetalBackend::new() returned None on macOS — the shader library \
                 almost certainly failed to compile. Run `cargo test -p \
                 larql-compute-metal --lib`."
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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The CPU arm: the proven whole-layer path.
fn cpu_layer(
    fx: &Fixture,
) -> crate::format::vindex3::opplan::exec::kimi_kda_layer::KdaDecoderLayerTrace {
    let by_id = |id: usize| -> ExpertWeights<'_> {
        let slot = fx
            .ids_order
            .iter()
            .position(|&i| i == id)
            .unwrap_or_else(|| panic!("layer asked for un-resident expert {id}"));
        let (gate, up, down) = &fx.cpu_experts[slot];
        ExpertWeights { gate, up, down }
    };
    let shared = ExpertWeights {
        gate: &fx.cpu_shared.0,
        up: &fx.cpu_shared.1,
        down: &fx.cpu_shared.2,
    };
    let mut state = KdaState::zeros(fx.geometry);
    kda_decoder_layer_forward(
        &fx.x,
        fx.hidden,
        &fx.input_norm,
        &fx.post_norm,
        fx.eps as f64,
        fx.kda_cpu(),
        fx.geometry,
        &mut state,
        fx.inter,
        &fx.router_weight,
        &fx.router_bias,
        fx.experts,
        fx.top_k,
        fx.renormalize,
        fx.branch_scale as f64,
        by_id,
        Some((shared, fx.inter)),
    )
}

/// **R5d's gate.** Every link in the device dependency chain, in order.
#[test]
fn the_whole_layer_on_device_matches_link_by_link() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let cpu = cpu_layer(&fx);
    let state = KdaDeviceState::zeros(&metal, fx.shape());
    let got = metal
        .kimi_decoder_layer_traced(fx.layer(&state), &fx.x)
        .expect("device layer");

    let check = |name: &str, a: &[f32], b: &[f32]| {
        let d = max_abs(a, b);
        eprintln!("[r5d] {name:>24}: max|Δ| {d:e}");
        assert!(d < TOLERANCE, "{name}: max|Δ| {d:e}");
    };
    check("input_normed", &got.input_normed, &cpu.input_normed);
    check("attention", &got.attention, &cpu.attention.output);
    check(
        "after_attention",
        &got.after_attention,
        &cpu.after_attention,
    );
    check(
        "post_attention_normed",
        &got.post_attention_normed,
        &cpu.post_attention_normed,
    );
    check("router_logits", &got.router_logits, &cpu.moe.router.logits);
    check("router_scores", &got.router_scores, &cpu.moe.router.scores);
    check(
        "router_selection_scores",
        &got.router_selection_scores,
        &cpu.moe.router.selection_scores,
    );

    // Exact ids, not a tolerance — a route is a decision, not a number.
    let want_ids: Vec<u32> = cpu
        .moe
        .router
        .selected_ids
        .iter()
        .map(|&i| i as u32)
        .collect();
    eprintln!("[r5d] {:>24}: {:?}", "selected_ids", got.selected_ids);
    assert_eq!(got.selected_ids, want_ids, "the device chose other experts");

    // The routed weights the MoE multiplied by, then the shared branch's
    // unscaled 1.0 — the ordering the combine relies on.
    check(
        "combine_weights(routed)",
        &got.combine_weights[..fx.top_k],
        &cpu.moe.router.weights,
    );
    assert_eq!(
        got.combine_weights[fx.top_k], 1.0,
        "the shared branch is summed, never scaled"
    );

    // The offset table the router wrote is the one the residency map
    // names for the experts it chose — the seam itself.
    let want_offsets: Vec<u32> = want_ids
        .iter()
        .map(|&id| fx.residency[id as usize])
        .collect();
    assert_eq!(
        got.expert_offsets, want_offsets,
        "the GPU-written offset table does not match the residency map"
    );

    for (slot, want) in cpu.moe.expert_outputs.iter().enumerate() {
        let a = &got.expert_outputs[slot * fx.hidden..(slot + 1) * fx.hidden];
        let d = max_abs(a, want);
        assert!(d < TOLERANCE, "expert slot {slot}: max|Δ| {d:e}");
    }
    let shared = &got.expert_outputs[fx.top_k * fx.hidden..(fx.top_k + 1) * fx.hidden];
    check("shared_output", shared, &cpu.moe.shared_output);
    check("layer_output", &got.output, &cpu.output);
    check("layer_output vs HF", &got.output, &fx.oracle_layer_output);
    eprintln!(
        "[r5d] one command buffer, gpu {:.3} ms, the host never saw an expert id",
        got.gpu_ms
    );
}

/// **Control.** An expert the router picks that is not resident must be
/// REFUSED, not served another expert's weights.
///
/// Without this the seam would be trusting a table it cannot check: the
/// router writes offsets, and nothing downstream can tell a wrong offset
/// from a right one.
#[test]
fn selecting_a_non_resident_expert_is_refused() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let state = KdaDeviceState::zeros(&metal, fx.shape());
    let mut broken = fx.residency.clone();
    // Evict the expert the router ranks first.
    let cpu = cpu_layer(&fx);
    broken[cpu.moe.router.selected_ids[0]] = NOT_RESIDENT;
    let mut w = fx.layer(&state);
    {
        let m = moe_mut(&mut w);
        let a = ExpertAddressing::Table(&broken);
        m.gate.addressing = a;
        m.up.addressing = a;
        m.down.addressing = a;
    }

    let refused = metal.kimi_decoder_layer(w, &fx.x);
    assert!(
        matches!(
            refused,
            Err(larql_compute_metal::trait_impl::grouped_experts::GroupedError::
                LayerRouteNotResident { layer: 0, .. })
        ),
        "a non-resident selection must be refused as such, not served: {refused:?}"
    );
    eprintln!("[r5d] control: evicting a selected expert is refused — {refused:?}");
}

/// **Control.** The two router transcriptions that still produce
/// plausible output must not pass.
///
/// Perturbing the fixture rather than the kernel: a bias that cannot
/// change selection, and weights that would come from the biased scores.
#[test]
fn the_router_controls_still_bite_on_device() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let cpu = cpu_layer(&fx);
    let state = KdaDeviceState::zeros(&metal, fx.shape());

    // Omitting the correction bias must change the selection. On this
    // fixture the consequence is a REFUSAL rather than a different
    // answer, and that is the strongest possible outcome: only the
    // biased route's experts are resident, so a changed route
    // immediately names an expert that has no address, and the device
    // guard catches it. Two things are proved at once — the bias decides
    // selection, and a route that leaves the resident set is refused
    // rather than served someone else's weights.
    let zeros = vec![0.0f32; fx.experts];
    let mut unbiased = fx.layer(&state);
    moe_mut(&mut unbiased).router_bias = &zeros;
    let refused = metal.kimi_decoder_layer(unbiased, &fx.x);
    assert!(
        refused.is_err(),
        "dropping the correction bias changed nothing — the bias must decide \
         selection on this fixture, or this control proves nothing"
    );

    // And say WHY, from the CPU router, so the refusal is not mistaken
    // for an unrelated fault.
    let unbiased_route = crate::format::vindex3::opplan::exec::kimi_router::route(
        &cpu.post_attention_normed,
        &fx.router_weight,
        &zeros,
        fx.experts,
        fx.top_k,
        fx.renormalize,
        fx.branch_scale as f64,
        crate::format::vindex3::opplan::exec::kimi_router::Mutation::None,
    );
    assert_ne!(
        unbiased_route.selected_ids, cpu.moe.router.selected_ids,
        "the correction bias must change the route on this fixture"
    );
    eprintln!(
        "[r5d] control: dropping the correction bias moves the route {:?} -> {:?}, \
         which the device refuses as non-resident",
        cpu.moe.router.selected_ids, unbiased_route.selected_ids,
    );
}

/// What a whole layer costs on each side, and how many crossings it
/// makes.
#[test]
fn report_whole_layer_cost() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let state = KdaDeviceState::zeros(&metal, fx.shape());
    let host = || {
        let t = Instant::now();
        std::hint::black_box(cpu_layer(&fx));
        t.elapsed().as_secs_f64() * 1000.0
    };
    let device = || {
        // Reset outside the timer: the recurrent state advances every
        // step, and on this fixture a drifted state eventually routes to
        // an expert that is not resident. Holding the token constant is
        // also what makes the two arms comparable — `cpu_layer` starts
        // from a zero state every call.
        state.reset();
        let t = Instant::now();
        let (out, gpu) = metal
            .kimi_decoder_layer(fx.layer(&state), &fx.x)
            .expect("device layer");
        std::hint::black_box(out);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    for _ in 0..WARMUP {
        host();
        device();
    }
    let (mut h, mut dw, mut dg) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERS {
        h.push(host());
        let (w, g) = device();
        dw.push(w);
        dg.push(g);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (h.len() / 3).max(1);
    let ramp = mean(&h[..third]) / mean(&h[h.len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (host_ms, dev_ms, gpu_ms) = (median(&mut h), median(&mut dw), median(&mut dg));

    // Decompose the device GPU time: the attention alone is already
    // measured at rung 5c, so what the layer adds beyond it is the
    // router, the MoE and the norms. Without this the layer's total is
    // just a number with no stage attached.
    let kda_state = KdaDeviceState::zeros(&metal, fx.shape());
    let mut kda_gpu = Vec::new();
    for i in 0..WARMUP + ITERS {
        kda_state.reset();
        let (_, g) = metal
            .kda_attention_step(
                fx.kda_device(),
                fx.shape(),
                &kda_state,
                &fx.input_norm_applied(),
            )
            .expect("attention alone");
        if i >= WARMUP {
            kda_gpu.push(g);
        }
    }
    let attn_ms = median(&mut kda_gpu);

    // And the MoE alone, through the already-measured block path with
    // host-built offsets — so "everything else" can be split into the
    // experts and the rest rather than left as a residual.
    let cpu = cpu_layer(&fx);
    // The block path binds ONE bank per projection, so this probe
    // chooses the co-located layout: routed bytes with the shared
    // payload appended. A layout an arm MAY choose — the layer under
    // test binds the shared regions independently.
    let cat = |routed: &[u8], shared: &[u8]| {
        let mut v = routed.to_vec();
        v.extend_from_slice(shared);
        v
    };
    let (cat_gate, cat_up, cat_down) = (
        cat(&fx.bank_gate, &fx.shared_gate),
        cat(&fx.bank_up, &fx.shared_up),
        cat(&fx.bank_down, &fx.shared_down),
    );
    let offsets: Vec<ExpertOffset> = cpu
        .moe
        .router
        .selected_ids
        .iter()
        .map(|&id| ExpertOffset(fx.residency[id]))
        .chain(std::iter::once(ExpertOffset(fx.bank_gate.len() as u32)))
        .collect();
    let banks = MoeFfnBanks {
        gate: ExpertBankRef {
            weights: &cat_gate,
            offsets: &offsets,
        },
        up: ExpertBankRef {
            weights: &cat_up,
            offsets: &offsets,
        },
        down: ExpertBankRef {
            weights: &cat_down,
            offsets: &offsets,
        },
        hidden: fx.hidden,
        inter: fx.inter,
    };
    let call = [MoeBlockCall {
        banks,
        x: &cpu.post_attention_normed,
    }];
    let mut moe_gpu = Vec::new();
    for i in 0..WARMUP + ITERS {
        let (_, g) = metal
            .bf16_moe_ffn_blocks(
                &call,
                larql_compute_metal::trait_impl::bf16_moe_block::BlockLowering::Separate,
            )
            .expect("moe alone");
        if i >= WARMUP {
            moe_gpu.push(g);
        }
    }
    let moe_ms = median(&mut moe_gpu);

    eprintln!("[r5d] one complete decoder layer (KDA + router + 8 routed + shared MoE)");
    eprintln!("[r5d]   host   {host_ms:.3} ms   all stages on CPU");
    eprintln!(
        "[r5d]   device {dev_ms:.3} ms   gpu-busy {gpu_ms:.3} ms, host {:.3} ms \
         over 1 crossing  [{:.2}x]",
        dev_ms - gpu_ms,
        host_ms / dev_ms,
    );
    eprintln!(
        "[r5d]   of which attention {attn_ms:.3} ms, MoE {moe_ms:.3} ms, \
         remainder {:.3} ms (router + norms + residuals)",
        gpu_ms - attn_ms - moe_ms,
    );
    eprintln!("[r5d]   ramp {ramp:.2}x — 1.00 means the machine held still");
    assert!(ramp.is_finite());
}

/// Reach the routed weights of a layer, for the controls that corrupt
/// one field. Panics on a dense layer, which no control here builds.
fn moe_mut<'a, 'b>(
    w: &'b mut larql_compute_metal::trait_impl::kimi_layer::KimiLayerWeights<'a>,
) -> &'b mut larql_compute_metal::trait_impl::kimi_layer::KimiMoeWeights<'a> {
    match &mut w.ffn {
        FfnSpec::Moe(m) => m,
        FfnSpec::Dense(_) => panic!("this control corrupts a routed layer"),
    }
}

// **C rung 4 — projection identity, location, backing and encoding are
// independent.**
//
// Runs on the REAL layer fixture because the claim needs superblock-
// aligned shapes: at hidden 2304 / inter 1024 both projections are
// whole multiples of 256, while a toy fixture's 48-element projections
// cannot encode Q6_K at all — a test there would prove nothing about
// mixed representation.
//
// The four properties are varied AT ONCE, which is the point. Mixed
// encoding over identity addresses would leave open that some shared
// physical coordinate still exists; independent permutations under one
// encoding would leave open that representation is still a property of
// "the bank". Together they close both.
/// Three permutations of the bank's blocks that differ at every index.
fn perms(blocks: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    // Three PAIRWISE DISCORDANT permutations: no index maps to the same
    // place in any two of them.
    //
    // Affine maps `a*i + b` will not do. Two of them differ everywhere
    // only when their multipliers are equal, which makes them rotations
    // of one another — and a rotation is exactly what one hidden
    // coordinate plus a constant could still reproduce. So: seeded
    // shuffles, searched deterministically until a discordant triple
    // falls out.
    let shuffled = |seed: u64| {
        let mut v: Vec<usize> = (0..blocks).collect();
        let mut st = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        for i in (1..blocks).rev() {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            v.swap(i, (st >> 33) as usize % (i + 1));
        }
        v
    };
    let discordant = |a: &[usize], b: &[usize]| a.iter().zip(b).all(|(x, y)| x != y);
    for s in 0..4096u64 {
        let g = shuffled(s);
        for t in s + 1..s + 64 {
            let u = shuffled(t);
            if !discordant(&g, &u) {
                continue;
            }
            for w in t + 1..t + 64 {
                let d = shuffled(w);
                if discordant(&g, &d) && discordant(&u, &d) {
                    return (g, u, d);
                }
            }
        }
    }
    panic!("no discordant triple found for {blocks} blocks");
}

fn permute(src: &[u8], per: usize, perm: &[usize]) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for (i, &p) in perm.iter().enumerate() {
        out[p * per..(p + 1) * per].copy_from_slice(&src[i * per..(i + 1) * per]);
    }
    out
}

fn widen(bf16: &[u8]) -> Vec<f32> {
    bf16.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

/// Re-encode each block of a bf16 bank to Q6_K, block by block, so the
/// result is a bank of the same block count at a smaller stride.
fn to_q6k_blocks(src: &[u8], per: usize, n: usize, k: usize) -> (Vec<u8>, usize) {
    assert!(k.is_multiple_of(256), "k={k} cannot be Q6_K");
    let q_per = n * k / 256 * 210;
    let mut out = Vec::with_capacity(src.len() / per * q_per);
    for block in src.chunks_exact(per) {
        let q = quantize_q6_k(&widen(block));
        assert_eq!(q.len(), q_per);
        out.extend_from_slice(&q);
    }
    (out, q_per)
}

fn table(residency: &[u32], src_per: usize, dst_per: usize, perm: &[usize]) -> Vec<u32> {
    residency
        .iter()
        .map(|off| {
            if *off == larql_compute_metal::shaders::kimi_layer::NOT_RESIDENT {
                *off
            } else {
                (perm[*off as usize / src_per] * dst_per) as u32
            }
        })
        .collect()
}

/// **The signature: one logical expert, three physical slots, two
/// encodings, correct output.**
#[test]
fn location_and_representation_vary_independently_for_one_semantic_expert() {
    let Some((b, f)) = setup() else {
        return;
    };
    let per = f.inter * f.hidden * 2;
    let blocks = f.bank_gate.len() / per;
    let (pg, pu, pd) = perms(blocks);
    for i in 0..blocks {
        assert!(
            pg[i] != pu[i] && pu[i] != pd[i] && pg[i] != pd[i],
            "block {i} maps to {}/{}/{} — a shared coordinate would survive",
            pg[i],
            pu[i],
            pd[i]
        );
    }

    // The all-BF16 reference, unpermuted.
    let state_ref = KdaDeviceState::zeros(&b, f.shape());
    let (want, _) = b
        .kimi_decoder_layer(f.layer(&state_ref), &f.x)
        .expect("reference layer runs");

    // gate/up become Q6_K; down stays BF16. Each is permuted
    // differently. The SHARED branch follows each projection's encoding
    // from its own regions — it has no location in the routed banks to
    // permute, which is the whole point of it being a separate region.
    let (q_gate_src, q_per) = to_q6k_blocks(&f.bank_gate, per, f.inter, f.hidden);
    let (q_up_src, _) = to_q6k_blocks(&f.bank_up, per, f.inter, f.hidden);
    let (q_shared_gate, _) = to_q6k_blocks(&f.shared_gate, per, f.inter, f.hidden);
    let (q_shared_up, _) = to_q6k_blocks(&f.shared_up, per, f.inter, f.hidden);
    let gate = permute(&q_gate_src, q_per, &pg);
    let up = permute(&q_up_src, q_per, &pu);
    let down = permute(&f.bank_down, per, &pd);
    let (tg, tu, td) = (
        table(&f.residency, per, q_per, &pg),
        table(&f.residency, per, q_per, &pu),
        table(&f.residency, per, per, &pd),
    );
    let state = KdaDeviceState::zeros(&b, f.shape());
    let mut w = f.layer(&state);
    if let FfnSpec::Moe(m) = &mut w.ffn {
        m.gate = ProjectionBank {
            routed: EncodedRegion {
                bytes: &gate,
                encoding: ExpertEncoding::Q6K,
            },
            addressing: ExpertAddressing::Table(&tg),
            shared: Some(EncodedRegion {
                bytes: &q_shared_gate,
                encoding: ExpertEncoding::Q6K,
            }),
        };
        m.up = ProjectionBank {
            routed: EncodedRegion {
                bytes: &up,
                encoding: ExpertEncoding::Q6K,
            },
            addressing: ExpertAddressing::Table(&tu),
            shared: Some(EncodedRegion {
                bytes: &q_shared_up,
                encoding: ExpertEncoding::Q6K,
            }),
        };
        m.down = ProjectionBank {
            routed: EncodedRegion {
                bytes: &down,
                encoding: ExpertEncoding::Bf16,
            },
            addressing: ExpertAddressing::Table(&td),
            shared: Some(EncodedRegion {
                bytes: &f.shared_down,
                encoding: ExpertEncoding::Bf16,
            }),
        };
    }
    let (got, _) = b
        .kimi_decoder_layer(w, &f.x)
        .expect("the mixed layer must execute");

    // Every selected expert really does sit at three distinct slots.
    let ids: Vec<usize> = f.ids_order.clone();
    for &e in ids.iter().take(f.top_k) {
        let (a, c, d) = (
            tg[e] as usize / q_per,
            tu[e] as usize / q_per,
            td[e] as usize / per,
        );
        assert!(
            a != c && c != d && a != d,
            "expert {e} resolves to slots {a}/{c}/{d} — not three distinct places"
        );
    }

    // The tolerance is Q6's own, taken from this very layer: quantise
    // gate/up WITHOUT permuting and measure. Anything the permutation
    // adds beyond that is an addressing fault, not representation.
    let unpermuted = {
        let identity: Vec<usize> = (0..blocks).collect();
        let g = permute(&q_gate_src, q_per, &identity);
        let u = permute(&q_up_src, q_per, &identity);
        let (tgi, tui) = (
            table(&f.residency, per, q_per, &identity),
            table(&f.residency, per, q_per, &identity),
        );
        let st = KdaDeviceState::zeros(&b, f.shape());
        let mut w2 = f.layer(&st);
        if let FfnSpec::Moe(m) = &mut w2.ffn {
            m.gate = ProjectionBank {
                routed: EncodedRegion {
                    bytes: &g,
                    encoding: ExpertEncoding::Q6K,
                },
                addressing: ExpertAddressing::Table(&tgi),
                shared: Some(EncodedRegion {
                    bytes: &q_shared_gate,
                    encoding: ExpertEncoding::Q6K,
                }),
            };
            m.up = ProjectionBank {
                routed: EncodedRegion {
                    bytes: &u,
                    encoding: ExpertEncoding::Q6K,
                },
                addressing: ExpertAddressing::Table(&tui),
                shared: Some(EncodedRegion {
                    bytes: &q_shared_up,
                    encoding: ExpertEncoding::Q6K,
                }),
            };
            m.down.shared = Some(EncodedRegion {
                bytes: &f.shared_down,
                encoding: ExpertEncoding::Bf16,
            });
        }
        b.kimi_decoder_layer(w2, &f.x)
            .expect("unpermuted mixed runs")
            .0
    };
    let rel = |a: &[f32], c: &[f32]| {
        let se: f64 = a.iter().zip(c).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let ss: f64 = c.iter().map(|y| (*y as f64).powi(2)).sum();
        (se / ss).sqrt()
    };
    let envelope = rel(&unpermuted, &want);
    let mixed = rel(&got, &want);
    eprintln!(
        "[c4] Q6_K gate/up + BF16 down, three independent permutations: \
         rel_rms vs BF16 {mixed:.3e}; the same encodings UNPERMUTED give {envelope:.3e}"
    );
    assert!(
        envelope > 0.0,
        "quantising gate/up must move the answer, or this proves nothing"
    );
    assert!(
        mixed <= envelope * 1.05,
        "permuting the three projections independently must cost nothing beyond Q6's own \
         error: {mixed:.3e} against an envelope of {envelope:.3e}"
    );

    // The control: mismatch ONE projection's table and the answer must move.
    let state_bad = KdaDeviceState::zeros(&b, f.shape());
    let mut bad = f.layer(&state_bad);
    if let FfnSpec::Moe(m) = &mut bad.ffn {
        m.gate = ProjectionBank {
            routed: EncodedRegion {
                bytes: &gate,
                encoding: ExpertEncoding::Q6K,
            },
            addressing: ExpertAddressing::Table(&tu),
            shared: Some(EncodedRegion {
                bytes: &q_shared_gate,
                encoding: ExpertEncoding::Q6K,
            }),
        };
        m.up = ProjectionBank {
            routed: EncodedRegion {
                bytes: &up,
                encoding: ExpertEncoding::Q6K,
            },
            addressing: ExpertAddressing::Table(&tu),
            shared: Some(EncodedRegion {
                bytes: &q_shared_up,
                encoding: ExpertEncoding::Q6K,
            }),
        };
        m.down = ProjectionBank {
            routed: EncodedRegion {
                bytes: &down,
                encoding: ExpertEncoding::Bf16,
            },
            addressing: ExpertAddressing::Table(&td),
            shared: Some(EncodedRegion {
                bytes: &f.shared_down,
                encoding: ExpertEncoding::Bf16,
            }),
        };
    }
    let (wrong, _) = b.kimi_decoder_layer(bad, &f.x).expect("runs");
    assert!(
        rel(&wrong, &want) > envelope * 2.0,
        "giving gate another projection's table must break the answer, or the three \
         mappings are not really independent"
    );
}

/// **A projection whose bytes are not its declared encoding is refused
/// BEFORE any kernel runs.**
///
/// Q6_K bytes dispatched as BF16 would be read as roughly 2.4x more
/// data than exists — the layer must refuse rather than launch and let
/// a kernel interpret whatever follows.
///
/// This is the too-small direction, which a room check catches. The
/// opposite (BF16 bytes declared Q6_K, which are LARGER than the claim
/// and pass every room check) needs the bank's exact extent and is
/// covered by `represent::physical`'s
/// `a_bank_whose_bytes_are_not_its_declared_encoding_is_refused`.
#[test]
fn a_projection_declaring_the_wrong_encoding_is_refused_before_execution() {
    let Some((b, f)) = setup() else {
        return;
    };
    let per = f.inter * f.hidden * 2;
    let (q_gate, q_per) = to_q6k_blocks(&f.bank_gate, per, f.inter, f.hidden);
    assert!(
        q_per < per,
        "Q6_K must be smaller than bf16 for this to bite"
    );

    let ids: Vec<usize> = (0..f.bank_gate.len() / per).collect();
    let t = table(&f.residency, per, q_per, &ids);
    let state = KdaDeviceState::zeros(&b, f.shape());
    let mut w = f.layer(&state);
    if let FfnSpec::Moe(m) = &mut w.ffn {
        // Real Q6_K bytes, with the offsets they need — but claiming to
        // be BF16, which needs 2.4x the room.
        m.gate = ProjectionBank {
            routed: EncodedRegion {
                bytes: &q_gate,
                encoding: ExpertEncoding::Bf16,
            },
            addressing: ExpertAddressing::Table(&t),
            shared: Some(EncodedRegion {
                bytes: &f.shared_gate,
                encoding: ExpertEncoding::Bf16,
            }),
        };
    }
    let err = b
        .kimi_decoder_layer(w, &f.x)
        .expect_err("bytes that are not the declared encoding must be refused");
    assert!(
        matches!(err, GroupedError::OffsetOutOfRange { slot: 0, .. }),
        "expected the gate projection to be named, got {err:?}"
    );
}

/// Re-encode each block of a bf16 bank to Q8_0, block by block — the
/// Q8_0 sibling of `to_q6k_blocks`, at the 34-bytes-per-32 stride.
fn to_q8_0_blocks(src: &[u8], per: usize, n: usize, k: usize) -> (Vec<u8>, usize) {
    assert!(k.is_multiple_of(32), "k={k} cannot be Q8_0");
    let q_per = n * k / 32 * 34;
    let mut out = Vec::with_capacity(src.len() / per * q_per);
    for block in src.chunks_exact(per) {
        let q = quantize_q8_0(&widen(block));
        assert_eq!(q.len(), q_per);
        out.extend_from_slice(&q);
    }
    (out, q_per)
}

/// **Q8_0 sits strictly ABOVE Q6_K on the fidelity axis at a real
/// layer.** Both quantisations of all three projections move the layer
/// output — they are real representations, not pass-throughs — and
/// Q8_0's displacement is strictly smaller than Q6_K's on the same
/// weights and input.
///
/// This is the smoke gate the precision ladder stands on: if an ~8-bit
/// bank did NOT beat Q6_K here, running teacher-forced quality banks
/// against it would measure a defect, not a representation.
#[test]
fn q8_0_displaces_the_layer_output_less_than_q6_k() {
    let Some((b, f)) = setup() else {
        return;
    };
    let per = f.inter * f.hidden * 2;
    let per_down = f.hidden * f.inter * 2;

    let state_ref = KdaDeviceState::zeros(&b, f.shape());
    let (want, _) = b
        .kimi_decoder_layer(f.layer(&state_ref), &f.x)
        .expect("reference layer runs");

    // One closure per encoding, differing ONLY in the re-encoder, so
    // the two arms cannot drift in table construction or bank layout.
    type Requant<'a> = &'a dyn Fn(&[u8], usize, usize, usize) -> (Vec<u8>, usize);
    let run = |requant_gate_up: Requant, requant_down: Requant, enc: ExpertEncoding| -> Vec<f32> {
        let (g, q_per) = requant_gate_up(&f.bank_gate, per, f.inter, f.hidden);
        let (u, _) = requant_gate_up(&f.bank_up, per, f.inter, f.hidden);
        let (d, q_per_down) = requant_down(&f.bank_down, per_down, f.hidden, f.inter);
        let (sg, _) = requant_gate_up(&f.shared_gate, per, f.inter, f.hidden);
        let (su, _) = requant_gate_up(&f.shared_up, per, f.inter, f.hidden);
        let (sd, _) = requant_down(&f.shared_down, per_down, f.hidden, f.inter);
        let identity: Vec<usize> = (0..f.bank_gate.len() / per).collect();
        let t = table(&f.residency, per, q_per, &identity);
        let td = table(&f.residency, per_down, q_per_down, &identity);
        let state = KdaDeviceState::zeros(&b, f.shape());
        let mut w = f.layer(&state);
        if let FfnSpec::Moe(m) = &mut w.ffn {
            fn bank<'a>(
                bytes: &'a [u8],
                tbl: &'a [u32],
                shared: &'a [u8],
                enc: ExpertEncoding,
            ) -> ProjectionBank<'a> {
                ProjectionBank {
                    routed: EncodedRegion {
                        bytes,
                        encoding: enc,
                    },
                    addressing: ExpertAddressing::Table(tbl),
                    shared: Some(EncodedRegion {
                        bytes: shared,
                        encoding: enc,
                    }),
                }
            }
            m.gate = bank(&g, &t, &sg, enc);
            m.up = bank(&u, &t, &su, enc);
            m.down = bank(&d, &td, &sd, enc);
            let (out, _) = b.kimi_decoder_layer(w, &f.x).expect("quantised layer runs");
            return out;
        }
        unreachable!("the fixture layer is routed");
    };

    let q8 = run(&to_q8_0_blocks, &to_q8_0_blocks, ExpertEncoding::Q80);
    let q6 = run(&to_q6k_blocks, &to_q6k_blocks, ExpertEncoding::Q6K);

    let rel = |a: &[f32], c: &[f32]| {
        let se: f64 = a.iter().zip(c).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let ss: f64 = c.iter().map(|y| (*y as f64).powi(2)).sum();
        (se / ss).sqrt()
    };
    let (e8, e6) = (rel(&q8, &want), rel(&q6, &want));
    eprintln!("[q8] all-projection displacement vs BF16: Q8_0 {e8:.3e}, Q6_K {e6:.3e}");
    assert!(
        e8 > 0.0,
        "Q8_0 must move the answer, or this proves nothing about its decode"
    );
    assert!(
        e8 < e6,
        "Q8_0 ({e8:.3e}) must displace the output LESS than Q6_K ({e6:.3e}) on the same \
         weights, or the ladder's middle rung is below its bottom one"
    );
}
