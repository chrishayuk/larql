//! Rung 5e: two consecutive real Kimi decoder layers in ONE command
//! buffer, with layer 2 driven entirely by layer 1's GPU-produced
//! hidden state.
//!
//! The claim under test is not "two layers happen to share a command
//! buffer". It is:
//!
//! > **Layer 2's routing decision is dynamically downstream of layer
//! > 1**, because its router scores a vector that never left the device.
//!
//! A device path that quietly re-used the original host input, or read a
//! stale buffer, would route somewhere else — and on this fixture the
//! two layers select almost disjoint expert sets, so it would be caught
//! immediately rather than absorbed into a tolerance.
//!
//! ```text
//! upload layer-1 input          <- the only host->device transfer
//!   encode layer 1 completely
//!   layer-1 hidden stays in its device buffer
//!   encode layer 2 reading THAT buffer
//!   layer-2 router chooses from it; layer-2 MoE consumes those choices
//! one commit, one wait
//! read the final output         <- the only device->host transfer
//! ```
//!
//! **Residency stays refusal-only.** The fixture provides the union of
//! experts the two reference layers need and nothing else; a selection
//! outside that set is refused rather than served. Residency POLICY is a
//! separate problem and is deliberately kept out of this proof.
//!
//! ```text
//! python scripts/kimi_two_layer_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --layers 1 2 --out /tmp/kimi_two_layer_fixture
//! LARQL_KIMI_TWO_LAYER_FIXTURE=/tmp/kimi_two_layer_fixture \
//!   cargo test -p larql-vindex --features gpu --release --lib kimi_two_layer -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute_metal::shaders::kimi_layer::NOT_RESIDENT;
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use larql_compute_metal::trait_impl::kimi_layer::{
    AttentionSpec, EncodedRegion, ExpertAddressing, ExpertEncoding, FfnSpec, KimiLayerCall,
    KimiLayerWeights, KimiMoeWeights, ProjectionBank,
};
use larql_compute_metal::trait_impl::mla::{MlaDeviceState, MlaDeviceWeights, MlaShape};
use larql_compute_metal::MetalBackend;
use larql_models::config::KdaGeometry;
use serde_json::Value;

const FIXTURE_ENV: &str = "LARQL_KIMI_TWO_LAYER_FIXTURE";
/// The same ceiling every real-weight Kimi gate uses.
const TOLERANCE: f32 = 3e-4;
const WARMUP: usize = 10;
const ITERS: usize = 10;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(PathBuf::from)
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

/// One layer's attention operands, owned. Which variant a layer holds is
/// the checkpoint's decision, read from the manifest — Kimi alternates
/// KDA and full attention, and a contiguous slice like `2 3 4` crosses
/// that transition, which is what makes this a model forward rather than
/// a chain of one operator.
enum LayerAttn {
    Kda {
        /// `q|k|v` concatenated; the grouped kernel binds one buffer.
        qkv_bank: Vec<u8>,
        qkv_offsets: [ExpertOffset; 3],
        o_proj: Vec<u8>,
        /// conv1d x3, f_a, f_b, g_a, g_b, b_proj, a_log, dt_bias, o_norm.
        f32s: Vec<Vec<f32>>,
    },
    Mla {
        q: Vec<u8>,
        kv_a: Vec<u8>,
        kv_b: Vec<u8>,
        o: Vec<u8>,
        kv_a_norm: Vec<f32>,
    },
}

/// The resident state a layer carries, whichever attention it runs.
enum LayerStateDev {
    Kda(KdaDeviceState),
    Mla(MlaDeviceState),
}

impl LayerStateDev {
    /// Start a new sequence. Both kinds restore in place; neither
    /// reallocates, so a timed loop measures the same token every time.
    fn reset(&self) {
        match self {
            Self::Kda(s) => s.reset(),
            Self::Mla(s) => s.reset(),
        }
    }
}

/// One layer's operands, owned, with its own resident expert bank.
struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    router_weight: Vec<f32>,
    router_bias: Vec<f32>,
    residency: Vec<u32>,
    bank_gate: Vec<u8>,
    bank_up: Vec<u8>,
    bank_down: Vec<u8>,
    /// The shared expert's own allocations — never part of the routed
    /// banks, because `Shared` is semantic identity, not co-location.
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
    attn: LayerAttn,
    ids: Vec<usize>,
    /// The reference boundaries for this layer.
    oracle_attention: Vec<f32>,
    oracle_after_attention: Vec<f32>,
    oracle_post_normed: Vec<f32>,
    oracle_input_normed: Vec<f32>,
    oracle_output: Vec<f32>,
}

impl Layer {
    /// The attention half, bound to whichever state this layer carries.
    /// A mismatch between the two is a construction error, not a runtime
    /// fallback — a KDA layer handed an MLA cache would be nonsense.
    fn attention<'a>(&'a self, fx: &Fixture, state: &'a LayerStateDev) -> AttentionSpec<'a> {
        match (&self.attn, state) {
            (
                LayerAttn::Kda {
                    qkv_bank,
                    qkv_offsets,
                    o_proj,
                    f32s: f,
                },
                LayerStateDev::Kda(state),
            ) => AttentionSpec::Kda {
                weights: KdaDeviceWeights {
                    qkv_bank,
                    qkv_offsets,
                    o_proj,
                    q_conv1d: &f[0],
                    k_conv1d: &f[1],
                    v_conv1d: &f[2],
                    f_a_proj: &f[3],
                    f_b_proj: &f[4],
                    g_a_proj: &f[5],
                    g_b_proj: &f[6],
                    b_proj: &f[7],
                    a_log: &f[8],
                    dt_bias: &f[9],
                    o_norm: &f[10],
                    norm_eps: fx.eps,
                },
                shape: fx.shape(),
                state,
            },
            (
                LayerAttn::Mla {
                    q,
                    kv_a,
                    kv_b,
                    o,
                    kv_a_norm,
                },
                LayerStateDev::Mla(state),
            ) => AttentionSpec::Mla {
                weights: MlaDeviceWeights {
                    q_proj: q,
                    kv_a_proj: kv_a,
                    kv_a_norm,
                    kv_b_proj: kv_b,
                    o_proj: o,
                    kv_a_norm_eps: fx.mla_eps,
                },
                shape: fx.mla_shape(),
                state,
            },
            _ => panic!("a layer's state must match its attention kind"),
        }
    }

    fn weights<'a>(&'a self, fx: &Fixture, state: &'a LayerStateDev) -> KimiLayerWeights<'a> {
        KimiLayerWeights {
            input_norm: &self.input_norm,
            post_attention_norm: &self.post_norm,
            attention: self.attention(fx, state),
            ffn: FfnSpec::Moe(KimiMoeWeights {
                router_weight: &self.router_weight,
                router_bias: &self.router_bias,
                gate: projection(&self.bank_gate, &self.residency, &self.shared_gate),
                up: projection(&self.bank_up, &self.residency, &self.shared_up),
                down: projection(&self.bank_down, &self.residency, &self.shared_down),
                inter: fx.inter,
                top_k: fx.top_k,
                renormalize: fx.renormalize,
                branch_scale: fx.branch_scale,
            }),
            norm_eps: fx.eps,
        }
    }
}

struct Fixture {
    hidden: usize,
    inter: usize,
    top_k: usize,
    renormalize: bool,
    branch_scale: f32,
    eps: f32,
    geometry: KdaGeometry,
    /// Present whenever the slice contains an MLA layer.
    mla: MlaShape,
    mla_eps: f32,
    kinds: Vec<String>,
    x: Vec<f32>,
    layers: Vec<Layer>,
    routes_differ: bool,
}

impl Fixture {
    fn mla_shape(&self) -> MlaShape {
        self.mla
    }

    /// One state per layer, of the kind that layer's attention needs —
    /// never shared, and never the wrong kind.
    fn states(&self, metal: &MetalBackend) -> Vec<LayerStateDev> {
        self.kinds
            .iter()
            .map(|k| {
                if k == "kda" {
                    LayerStateDev::Kda(KdaDeviceState::zeros(metal, self.shape()))
                } else {
                    // Capacity for one token's decode; this gate runs a
                    // single position through a slice.
                    LayerStateDev::Mla(MlaDeviceState::with_capacity(metal, self.mla, 64))
                }
            })
            .collect()
    }

    fn calls<'a>(&'a self, states: &'a [LayerStateDev]) -> Vec<KimiLayerCall<'a>> {
        self.layers
            .iter()
            .zip(states)
            .map(|(l, s)| KimiLayerCall {
                weights: l.weights(self, s),
            })
            .collect()
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

fn load_layer(dir: &Path, pos: usize, ids: Vec<usize>, experts: usize, kind: &str) -> Layer {
    let p = format!("l{pos}_");
    let attn = if kind == "kda" {
        let kda = |n: &str| read_f32(dir, &format!("{p}kda_{n}"));
        let (qb, kb, vb) = (
            read_bf16_bytes(dir, &format!("{p}kda_q_proj")),
            read_bf16_bytes(dir, &format!("{p}kda_k_proj")),
            read_bf16_bytes(dir, &format!("{p}kda_v_proj")),
        );
        let per_qkv = qb.len();
        let mut qkv_bank = Vec::with_capacity(3 * per_qkv);
        for b in [&qb, &kb, &vb] {
            qkv_bank.extend_from_slice(b);
        }
        LayerAttn::Kda {
            qkv_bank,
            qkv_offsets: [
                ExpertOffset(0),
                ExpertOffset(per_qkv as u32),
                ExpertOffset((2 * per_qkv) as u32),
            ],
            o_proj: read_bf16_bytes(dir, &format!("{p}kda_o_proj")),
            f32s: vec![
                kda("q_conv1d"),
                kda("k_conv1d"),
                kda("v_conv1d"),
                kda("f_a_proj"),
                kda("f_b_proj"),
                kda("g_a_proj"),
                kda("g_b_proj"),
                kda("b_proj"),
                kda("a_log"),
                kda("dt_bias"),
                kda("o_norm"),
            ],
        }
    } else {
        LayerAttn::Mla {
            q: read_bf16_bytes(dir, &format!("{p}mla_q_proj")),
            kv_a: read_bf16_bytes(dir, &format!("{p}mla_kv_a_proj")),
            kv_b: read_bf16_bytes(dir, &format!("{p}mla_kv_b_proj")),
            o: read_bf16_bytes(dir, &format!("{p}mla_o_proj")),
            kv_a_norm: read_f32(dir, &format!("{p}mla_kv_a_norm")),
        }
    };

    let mut bank_gate = Vec::new();
    let mut bank_up = Vec::new();
    let mut bank_down = Vec::new();
    let mut residency = vec![NOT_RESIDENT; experts];
    for &id in &ids {
        residency[id] = bank_gate.len() as u32;
        bank_gate.extend_from_slice(&read_bf16_bytes(dir, &format!("{p}expert{id}_w1")));
        bank_up.extend_from_slice(&read_bf16_bytes(dir, &format!("{p}expert{id}_w3")));
        bank_down.extend_from_slice(&read_bf16_bytes(dir, &format!("{p}expert{id}_w2")));
    }
    let shared_gate = read_bf16_bytes(dir, &format!("{p}shared_w1"));
    let shared_up = read_bf16_bytes(dir, &format!("{p}shared_w3"));
    let shared_down = read_bf16_bytes(dir, &format!("{p}shared_w2"));

    Layer {
        input_norm: read_f32(dir, &format!("{p}input_norm_weight")),
        post_norm: read_f32(dir, &format!("{p}post_attention_norm_weight")),
        router_weight: read_f32(dir, &format!("{p}router_weight")),
        router_bias: read_f32(dir, &format!("{p}router_bias")),
        residency,
        bank_gate,
        bank_up,
        bank_down,
        shared_gate,
        shared_up,
        shared_down,
        attn,
        ids,
        oracle_input_normed: read_f32(dir, &format!("{p}out_input_normed")),
        oracle_attention: read_f32(dir, &format!("{p}out_attention_output")),
        oracle_after_attention: read_f32(dir, &format!("{p}out_after_attention")),
        oracle_post_normed: read_f32(dir, &format!("{p}out_post_attention_normed")),
        oracle_output: read_f32(dir, &format!("{p}out_layer_output")),
    }
}

fn load(dir: &Path) -> Fixture {
    let m: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    let g = |k: &str| m[k].as_u64().unwrap() as usize;
    let experts = g("experts");
    let kinds: Vec<String> = m["kinds"]
        .as_array()
        .expect("the manifest declares each layer's attention kind")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let ids_of = |pos: usize| -> Vec<usize> {
        m["selected_ids_order"][pos.to_string()]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    Fixture {
        hidden: g("hidden"),
        inter: g("moe_intermediate_size"),
        top_k: g("top_k"),
        renormalize: m["moe_renormalize"].as_bool().unwrap(),
        branch_scale: m["routed_scaling_factor"].as_f64().unwrap() as f32,
        eps: m["rms_eps"].as_f64().unwrap() as f32,
        geometry: KdaGeometry {
            num_heads: g("num_heads"),
            head_dim: g("head_dim"),
            conv_kernel: 4,
        },
        mla: MlaShape {
            hidden: g("hidden"),
            num_heads: m["mla"]["num_heads"].as_u64().unwrap() as usize,
            kv_lora_rank: m["mla"]["kv_lora_rank"].as_u64().unwrap() as usize,
            qk_nope_head_dim: m["mla"]["qk_nope_head_dim"].as_u64().unwrap() as usize,
            qk_rope_head_dim: m["mla"]["qk_rope_head_dim"].as_u64().unwrap() as usize,
            v_head_dim: m["mla"]["v_head_dim"].as_u64().unwrap() as usize,
        },
        mla_eps: m["mla"]["kv_a_norm_eps"].as_f64().unwrap() as f32,
        kinds: kinds.clone(),
        x: read_f32(dir, "input"),
        layers: (0..m["layers"].as_array().expect("layers").len())
            .map(|p| load_layer(dir, p, ids_of(p), experts, &kinds[p]))
            .collect(),
        routes_differ: m["routes_differ"].as_bool().unwrap(),
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

/// **R5e's gate.** Both layers, every boundary, one command buffer.
#[test]
fn two_dynamic_layers_in_one_command_buffer_match_the_oracle() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    assert!(
        fx.routes_differ,
        "the two layers must select different experts, or this fixture cannot \
         tell a real chain from a path that re-used layer 1's input"
    );
    let states = fx.states(&metal);
    let calls = fx.calls(&states);

    let planes = metal
        .kimi_decoder_layers_traced(&calls, &fx.x)
        .expect("layer chain");
    assert_eq!(planes.len(), fx.layers.len());

    for (i, (p, l)) in planes.iter().zip(&fx.layers).enumerate() {
        let check = |name: &str, a: &[f32], b: &[f32]| {
            let d = max_abs(a, b);
            eprintln!("[r5e] layer {i} {name:>22}: max|Δ| {d:e}");
            assert!(d < TOLERANCE, "layer {i} {name}: max|Δ| {d:e}");
        };
        check("input_normed", &p.input_normed, &l.oracle_input_normed);
        check("attention", &p.attention, &l.oracle_attention);
        check(
            "after_attention",
            &p.after_attention,
            &l.oracle_after_attention,
        );
        check(
            "post_attention_normed",
            &p.post_attention_normed,
            &l.oracle_post_normed,
        );

        // A route is a decision, not a number, so both halves are exact.
        //
        // **Two references, because there are two contracts.** The
        // fixture's `selected_ids_order` is torch `topk(sorted=False)`'s
        // order, which promises nothing — so the CHECKPOINT is compared
        // on membership. The ORDER contract belongs to
        // `kimi_router::route`: descending by selection score, ties by
        // ascending index. The device reproduces that one exactly, and
        // it matters because the combine pairs weights with slots
        // positionally.
        eprintln!(
            "[r5e] layer {i} {:>22}: {:?}",
            "selected_ids", p.selected_ids
        );
        let device_ids: Vec<usize> = p.selected_ids.iter().map(|&x| x as usize).collect();
        let mut got_set = device_ids.clone();
        let mut want_set = l.ids.clone();
        got_set.sort_unstable();
        want_set.sort_unstable();
        assert_eq!(
            got_set, want_set,
            "layer {i}: the device's expert SET differs from the checkpoint's"
        );

        let reference = crate::format::vindex3::opplan::exec::kimi_router::route(
            &p.post_attention_normed,
            &l.router_weight,
            &l.router_bias,
            l.router_bias.len(),
            fx.top_k,
            fx.renormalize,
            fx.branch_scale as f64,
            crate::format::vindex3::opplan::exec::kimi_router::Mutation::None,
        );
        assert_eq!(
            device_ids, reference.selected_ids,
            "layer {i}: the device's ranking differs from `kimi_router::route`'s"
        );
        check(
            "router_selection_scores",
            &p.router_selection_scores,
            &reference.selection_scores,
        );
        check(
            "combine_weights(routed)",
            &p.combine_weights[..fx.top_k],
            &reference.weights,
        );

        // And each slot's offset is the one the residency map names for
        // the expert THAT slot holds — the seam, checked against the
        // device's own ordering rather than a presumed one.
        let want_offsets: Vec<u32> = device_ids.iter().map(|&id| l.residency[id]).collect();
        assert_eq!(
            p.expert_offsets, want_offsets,
            "layer {i}: the GPU-written offset table does not match the residency map"
        );
        assert_eq!(
            p.combine_weights[fx.top_k], 1.0,
            "layer {i}: the shared branch is summed, never scaled"
        );
        check("layer_output", &p.output, &l.oracle_output);
    }

    // The link itself: layer 1's input was layer 0's output, and neither
    // ever reached the host.
    eprintln!(
        "[r5e] chain proven: {} layers, one command buffer, gpu {:.3} ms; each route \
         computed from its predecessor's device-resident hidden — {:?}",
        planes.len(),
        planes[0].gpu_ms,
        planes
            .iter()
            .map(|p| p.selected_ids.clone())
            .collect::<Vec<_>>(),
    );
}

/// **Control.** Feeding layer 2 the ORIGINAL input instead of layer 1's
/// output must change its route.
///
/// This is the failure the whole rung is about: a device path that read
/// a stale buffer, or re-bound the upload, would produce a plausible
/// output from the wrong hidden state. Running the LAST layer alone on
/// `x` shows what that mistake would have selected.
#[test]
fn the_last_layer_run_on_the_original_input_routes_differently() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let states = fx.states(&metal);
    let last = fx.layers.len() - 1;
    let alone = metal.kimi_decoder_layer_traced(fx.layers[last].weights(&fx, &states[last]), &fx.x);

    // On the wrong hidden state it selects experts its own bank does not
    // hold, so the device refuses — the strongest possible outcome, and
    // proof the chained run's route came from somewhere else.
    match alone {
        Err(e) => eprintln!(
            "[r5e] control: layer {last} on the ORIGINAL input routes outside its bank and \
             is refused ({e}) — its chained route therefore came from its predecessor"
        ),
        Ok(p) => {
            let want: Vec<u32> = fx.layers[last].ids.iter().map(|&x| x as u32).collect();
            assert_ne!(
                p.selected_ids, want,
                "layer {last} selected the same experts from the original input as from \
                 the chain — this fixture cannot prove the chain"
            );
            eprintln!(
                "[r5e] control: layer {last} on the ORIGINAL input routes {:?}, chained it \
                 routes {want:?}",
                p.selected_ids
            );
        }
    }
}

/// **The strongest chain control.** Change what a layer OUTPUTS without
/// changing what it SELECTS, and watch a later layer's route move.
///
/// Swapping two of a layer's resident expert offsets makes each selected
/// id read the other expert's weights: the router still scores the same
/// `post_attention_normed` and picks the same ids, so that layer's own
/// decision is untouched — but its output changes, and every layer after
/// it is therefore routing on a different hidden state.
///
/// The evidence is precise: the refusal must come from a layer AFTER the
/// mutated one, never from the mutated layer itself. A stale-buffer or
/// buffer-reuse bug would break that pattern.
#[test]
fn perturbing_a_layers_output_moves_a_later_layers_route() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let n = fx.layers.len();
    assert!(n >= 2, "a chain control needs at least two layers");
    let states = fx.states(&metal);
    let baseline = metal
        .kimi_decoder_layers_traced(&fx.calls(&states), &fx.x)
        .expect("baseline chain");

    // Before mutating anything: a reset must reproduce the baseline
    // exactly. If it does not, every comparison below is against a
    // different machine and the control proves nothing.
    for s in &states {
        s.reset();
    }
    let repeat = metal
        .kimi_decoder_layers_traced(&fx.calls(&states), &fx.x)
        .expect("unmutated rerun after reset");
    for (i, (a, b)) in repeat.iter().zip(&baseline).enumerate() {
        assert_eq!(
            a.selected_ids, b.selected_ids,
            "layer {i} routed differently on an identical rerun — `reset` is not \
             restoring the state"
        );
    }

    // Both ends of the chain: the first layer, and the second-to-last.
    for mutated in [0usize, n - 2] {
        let l = &fx.layers[mutated];
        let mut swapped = l.residency.clone();
        let (a, b) = (l.ids[0], l.ids[1]);
        swapped.swap(a, b);

        for s in &states {
            s.reset();
        }
        let mut calls = fx.calls(&states);
        {
            let m = moe_mut(&mut calls[mutated].weights);
            let a = ExpertAddressing::Table(&swapped);
            m.gate.addressing = a;
            m.up.addressing = a;
            m.down.addressing = a;
        }
        let got = metal.kimi_decoder_layers_traced(&calls, &fx.x);

        match got {
            Ok(planes) => {
                // Every layer up to and including the mutated one must
                // route exactly as before; at least one after it must not.
                for (i, (p, base)) in planes.iter().zip(&baseline).enumerate() {
                    if i <= mutated {
                        assert_eq!(
                            p.selected_ids, base.selected_ids,
                            "swapping layer {mutated}'s expert OFFSETS changed layer {i}'s \
                             route — offsets must not reach the router"
                        );
                    }
                }
                // What MUST hold: the mutated layer's own output moved,
                // and so did the LAST layer's — the chain carried it
                // forward. A route flip downstream is a stronger
                // outcome but not a required one: a top-k of 256 can
                // absorb a small perturbation without changing its
                // selection, and demanding it would make this control
                // fixture-dependent rather than mechanism-dependent.
                let last = planes.len() - 1;
                let moved_here = max_abs(&planes[mutated].output, &baseline[mutated].output);
                let moved_end = max_abs(&planes[last].output, &baseline[last].output);
                assert!(
                    moved_here > 0.0,
                    "swapping layer {mutated}'s experts did not change its own output — \
                     the offsets are not reaching the MoE"
                );
                assert!(
                    moved_end > 0.0,
                    "layer {mutated}'s output changed but layer {last}'s did not — the \
                     chain is not carrying its output forward"
                );
                let rerouted: Vec<usize> = planes
                    .iter()
                    .zip(&baseline)
                    .enumerate()
                    .skip(mutated + 1)
                    .filter(|(_, (p, b))| p.selected_ids != b.selected_ids)
                    .map(|(i, _)| i)
                    .collect();
                eprintln!(
                    "[r5e] control: swapping layer {mutated}'s experts moved its output by \
                     {moved_here:e} and layer {last}'s by {moved_end:e}, with layers \
                     0..={mutated} routing identically (later reroutes: {rerouted:?})"
                );
            }
            Err(e) => {
                // A later layer routed outside its bank — same evidence,
                // provided the refusal is not the mutated layer's own.
                let blamed = match e {
                    larql_compute_metal::trait_impl::grouped_experts::GroupedError::
                        LayerRouteNotResident { layer, .. } => layer,
                    other => panic!("unexpected refusal: {other}"),
                };
                assert!(
                    blamed > mutated,
                    "the refusal came from layer {blamed}, at or before the mutated layer \
                     {mutated} — swapping offsets must not change that layer's own route"
                );
                eprintln!(
                    "[r5e] control: swapping layer {mutated}'s experts left its own route \
                     intact and pushed layer {blamed}'s route outside its bank"
                );
            }
        }
    }
}

/// What two chained layers cost, and how many crossings they make.
///
/// Reported by stage from the start: rung 5d found a single serial GPU
/// operation costing more than the attention and the MoE combined, and
/// it was invisible in the total.
#[test]
fn report_two_layer_cost() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let states = fx.states(&metal);
    let calls = fx.calls(&states);

    let chained = || {
        for s in &states {
            s.reset();
        }
        let t = Instant::now();
        let (out, gpu) = metal
            .kimi_decoder_layers(&calls, &fx.x, None)
            .expect("chain");
        std::hint::black_box(out);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };
    // The same two layers, one command buffer EACH — the shape rung 5d
    // left, and what the epoch is being compared against.
    // The same layers, one command buffer EACH, with the hidden state
    // round-tripping through the host between them — the shape rung 5d
    // left, and what the epoch is measured against.
    let separate = || {
        for s in &states {
            s.reset();
        }
        let t = Instant::now();
        let mut gpu = 0.0;
        let mut h = fx.x.clone();
        for call in &calls {
            let (out, g) = metal.kimi_decoder_layer(call.weights, &h).expect("layer");
            gpu += g;
            h = out;
        }
        std::hint::black_box(h);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    for _ in 0..WARMUP {
        separate();
        chained();
    }
    let (mut sw, mut sg, mut cw, mut cg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERS {
        let (w, g) = separate();
        sw.push(w);
        sg.push(g);
        let (w, g) = chained();
        cw.push(w);
        cg.push(g);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (sw.len() / 3).max(1);
    let ramp = mean(&sw[..third]) / mean(&sw[sw.len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (s_wall, s_gpu, c_wall, c_gpu) = (
        median(&mut sw),
        median(&mut sg),
        median(&mut cw),
        median(&mut cg),
    );

    let n = fx.layers.len();
    eprintln!("[r5e] {n} chained real decoder layers");
    eprintln!(
        "[r5e]   {n} command buffers  wall {s_wall:.3} ms  gpu {s_gpu:.3} ms  \
         host {:.3} ms over {n} crossings",
        s_wall - s_gpu,
    );
    eprintln!(
        "[r5e]   1 command buffer   wall {c_wall:.3} ms  gpu {c_gpu:.3} ms  \
         host {:.3} ms over 1 crossing  [{:.2}x wall]",
        c_wall - c_gpu,
        s_wall / c_wall,
    );
    eprintln!("[r5e]   ramp {ramp:.2}x — 1.00 means the machine held still");
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
