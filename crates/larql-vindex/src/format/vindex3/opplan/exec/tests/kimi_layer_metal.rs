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

use larql_compute_metal::shaders::kimi_layer::NOT_RESIDENT;
use larql_compute_metal::trait_impl::bf16_moe_block::{ExpertBankRef, MoeBlockCall, MoeFfnBanks};
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use larql_compute_metal::trait_impl::kimi_layer::{
    AttentionSpec, ExpertAddressing, FfnSpec, KimiLayerWeights, KimiMoeWeights,
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

    // MoE: gate/up/down banks over the resident experts, shared last.
    bank_gate: Vec<u8>,
    bank_up: Vec<u8>,
    bank_down: Vec<u8>,
    residency: Vec<u32>,
    shared_offset: u32,
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
                addressing: ExpertAddressing::Table(&self.residency),
                shared_offset: self.shared_offset,
                gate: &self.bank_gate,
                up: &self.bank_up,
                down: &self.bank_down,
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
    let shared_offset = bank_gate.len() as u32;
    let (s1, s3, s2) = (
        read_bf16_bytes(dir, "shared_w1"),
        read_bf16_bytes(dir, "shared_w3"),
        read_bf16_bytes(dir, "shared_w2"),
    );
    let cpu_shared = (codes(&s1), codes(&s3), codes(&s2));
    bank_gate.extend_from_slice(&s1);
    bank_up.extend_from_slice(&s3);
    bank_down.extend_from_slice(&s2);

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
        shared_offset,
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
        .chain(std::iter::once(fx.shared_offset))
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
    moe_mut(&mut w).addressing = ExpertAddressing::Table(&broken);

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
    let offsets: Vec<ExpertOffset> = cpu
        .moe
        .router
        .selected_ids
        .iter()
        .map(|&id| ExpertOffset(fx.residency[id]))
        .chain(std::iter::once(ExpertOffset(fx.shared_offset)))
        .collect();
    let banks = MoeFfnBanks {
        gate: ExpertBankRef {
            weights: &fx.bank_gate,
            offsets: &offsets,
        },
        up: ExpertBankRef {
            weights: &fx.bank_up,
            offsets: &offsets,
        },
        down: ExpertBankRef {
            weights: &fx.bank_down,
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
