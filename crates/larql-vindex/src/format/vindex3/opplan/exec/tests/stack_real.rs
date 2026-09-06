//! The mixed 27-layer stack at REAL Kimi Linear weights — the exit
//! condition the user specified: same initial hidden states → same
//! layer-by-layer outputs across all 27 layers and all three positions,
//! against the Python reference.
//!
//! Every layer's own attention weights are real (all 20 KDA, all 7
//! MLA), every norm is real, every router decision is real, and every
//! FFN branch — `KimiMLP` at layer 0's real 9216-wide dense weights,
//! `KimiSparseMoeBlock` at every other layer's real routed+shared
//! weights — is real. `scripts/kimi_stack_export.py` builds this
//! SEQUENTIALLY, layer by layer: a layer's routing decision depends on
//! the TRUE hidden state its predecessor produced, so there is no
//! cross-layer version of the single-layer "probe first" trick — each
//! layer is computed for real before the next layer's routing is even
//! decided, and only that layer's own union-of-3-positions experts are
//! loaded.
//!
//! Env-gated, same reason every other real-weight gate in this file is
//! — at real width this is the largest fixture in the ladder (order
//! 18 GiB: 26 routed layers × real per-expert weights):
//!
//! ```text
//! python scripts/kimi_stack_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --out /tmp/kimi_stack_fixture
//! LARQL_KIMI_STACK_FIXTURE=/tmp/kimi_stack_fixture \
//!     cargo test -p larql-vindex --lib stack_real --release
//! ```

use std::path::{Path, PathBuf};

use larql_models::config::{KdaGeometry, MlaGeometry};
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kda::{zero_state, KdaOutputGateWeights, KdaWeights};
use crate::format::vindex3::opplan::exec::kimi_moe_block::ExpertWeights;
use crate::format::vindex3::opplan::exec::mla::{MlaState, MlaWeights};
use crate::format::vindex3::opplan::exec::stack::{
    stack_forward, AttentionKind, LayerAttention, LayerFfn, LayerSpec, LayerState, LoadedExpert,
};

const FIXTURE_ENV: &str = "LARQL_KIMI_STACK_FIXTURE";
/// Same as every other full-width real-weight gate: hundreds of terms
/// summed per element, now compounded over 27 layers — wider than a
/// single layer's `3e-4` for exactly that reason.
const TOLERANCE: f32 = 2e-3;

pub(super) const KDA_FIELDS: [&str; 15] = [
    "q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d", "f_a_proj", "f_b_proj",
    "g_a_proj", "g_b_proj", "b_proj", "a_log", "dt_bias", "o_norm", "o_proj",
];
/// `KdaWeights`'s four BF16 operands (P4c-4) — a SUBSET of `KDA_FIELDS`,
/// named separately because they load via `read_bf16` while the other
/// eleven stay `read_f32`. `RealLayer.kda_fields` still carries an (empty,
/// unused) placeholder at these four positions so `KDA_FIELDS.iter().
/// position(...)` keeps working unchanged for the eleven it does load.
pub(super) const KDA_BF16_FIELDS: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];
pub(super) const MLA_FIELDS: [&str; 5] =
    ["q_proj", "kv_a_proj", "kv_a_norm", "kv_b_proj", "o_proj"];

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(PathBuf::from)
}

pub(super) fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("{name}.f32: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Expert weights only (P4a — `ExpertWeights` is BF16 code units, never
/// F32; see `kimi_moe_block.rs`'s own doc comment). Each `u16` is the
/// top 16 bits of the f32 the checkpoint's own BF16 tensor denotes —
/// `scripts/kimi_moe_export.py`'s `write_bf16` truncates rather than
/// re-derives, so this recovers the checkpoint's OWN bits exactly, not
/// an independent rounding of an already-lossy f32 upcast.
pub(super) fn read_bf16(dir: &Path, name: &str) -> Vec<u16> {
    let bytes = std::fs::read(dir.join(format!("{name}.bf16")))
        .unwrap_or_else(|e| panic!("{name}.bf16: {e}"));
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    let d = max_abs_diff(got, want);
    assert!(d < TOLERANCE, "{name}: max|Δ| {d:e}");
}

pub(super) struct LoadedExpertOwned {
    id: usize,
    gate: Vec<u16>,
    up: Vec<u16>,
    down: Vec<u16>,
}

/// Everything one layer needs, read from disk. A struct rather than the
/// synthetic test's `BTreeMap` fields — real names are already known
/// (`KDA_FIELDS`/`MLA_FIELDS`), so a map buys nothing here.
pub(super) struct RealLayer {
    pub(super) kind: &'static str,
    dense: bool,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    kda_fields: Vec<Vec<f32>>, // indexed by KDA_FIELDS order; BF16 slots empty, see kda_bf16_fields
    kda_bf16_fields: Vec<Vec<u16>>, // indexed by KDA_BF16_FIELDS order
    mla_fields: Vec<Vec<f32>>, // indexed by MLA_FIELDS order
    dense_ffn: Option<(Vec<u16>, Vec<u16>, Vec<u16>)>,
    router_weight: Option<Vec<f32>>,
    router_bias: Option<Vec<f32>>,
    experts: Vec<LoadedExpertOwned>,
    shared: Option<(Vec<u16>, Vec<u16>, Vec<u16>)>,
}

pub(super) fn load_real_layer(dir: &Path, i: usize, manifest: &Value) -> RealLayer {
    let l = &manifest["layers"][i];
    let kind: &'static str = if l["kind"] == "mla" { "mla" } else { "kda" };
    let dense = l["dense"].as_bool().unwrap();

    let kda_fields = if kind == "kda" {
        KDA_FIELDS
            .iter()
            .map(|f| {
                if KDA_BF16_FIELDS.contains(f) {
                    Vec::new() // loaded as bf16 below, see kda_bf16_fields
                } else {
                    read_f32(dir, &format!("layer{i}_kda_{f}"))
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let kda_bf16_fields = if kind == "kda" {
        KDA_BF16_FIELDS
            .iter()
            .map(|f| read_bf16(dir, &format!("layer{i}_kda_{f}")))
            .collect()
    } else {
        Vec::new()
    };
    let mla_fields = if kind == "mla" {
        MLA_FIELDS
            .iter()
            .map(|f| read_f32(dir, &format!("layer{i}_mla_{f}")))
            .collect()
    } else {
        Vec::new()
    };
    let dense_ffn = dense.then(|| {
        (
            read_bf16(dir, &format!("layer{i}_dense_w1")),
            read_bf16(dir, &format!("layer{i}_dense_w3")),
            read_bf16(dir, &format!("layer{i}_dense_w2")),
        )
    });
    let (router_weight, router_bias, experts, shared) = if dense {
        (None, None, Vec::new(), None)
    } else {
        let union: Vec<usize> = l["selected_ids_union_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let experts = union
            .into_iter()
            .map(|id| LoadedExpertOwned {
                id,
                gate: read_bf16(dir, &format!("layer{i}_expert{id}_w1")),
                up: read_bf16(dir, &format!("layer{i}_expert{id}_w3")),
                down: read_bf16(dir, &format!("layer{i}_expert{id}_w2")),
            })
            .collect();
        (
            Some(read_f32(dir, &format!("layer{i}_router_weight"))),
            Some(read_f32(dir, &format!("layer{i}_router_bias"))),
            experts,
            Some((
                read_bf16(dir, &format!("layer{i}_shared_w1")),
                read_bf16(dir, &format!("layer{i}_shared_w3")),
                read_bf16(dir, &format!("layer{i}_shared_w2")),
            )),
        )
    };

    RealLayer {
        kind,
        dense,
        input_norm: read_f32(dir, &format!("layer{i}_input_norm_weight")),
        post_norm: read_f32(dir, &format!("layer{i}_post_norm_weight")),
        kda_fields,
        kda_bf16_fields,
        mla_fields,
        dense_ffn,
        router_weight,
        router_bias,
        experts,
        shared,
    }
}

/// One layer's [`LoadedExpert`] list, built from its own loaded
/// `experts` — a separate step from [`spec`] for the same reason
/// `stack_parity.rs`'s own `Fixture::loaded_experts` is: the list must
/// outlive the `LayerSpec` that borrows it. Shared with `token_real.rs`
/// so the real-weight loading code exists in exactly one place.
pub(super) fn expert_list_for(layer: &RealLayer) -> Vec<LoadedExpert<'_>> {
    layer
        .experts
        .iter()
        .map(|e| LoadedExpert {
            id: e.id,
            weights: ExpertWeights {
                gate: &e.gate,
                up: &e.up,
                down: &e.down,
            },
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn spec<'a>(
    layer: &'a RealLayer,
    kda_geometry: KdaGeometry,
    mla_geometry: MlaGeometry,
    mla_kv_a_norm_eps: f64,
    experts: usize,
    top_k: usize,
    moe_intermediate: usize,
    dense_intermediate: usize,
    renormalize: bool,
    branch_scale: f64,
    eps: f64,
    loaded: &'a [LoadedExpert<'a>],
) -> LayerSpec<'a> {
    let attention = if layer.kind == "kda" {
        let f = |name: &str| {
            layer.kda_fields[KDA_FIELDS.iter().position(|&n| n == name).unwrap()].as_slice()
        };
        let fb = |name: &str| {
            layer.kda_bf16_fields[KDA_BF16_FIELDS.iter().position(|&n| n == name).unwrap()]
                .as_slice()
        };
        LayerAttention::Kda(
            KdaWeights {
                q_proj: WeightRows::Bf16(fb("q_proj")),
                k_proj: WeightRows::Bf16(fb("k_proj")),
                v_proj: WeightRows::Bf16(fb("v_proj")),
                q_conv1d: f("q_conv1d"),
                k_conv1d: f("k_conv1d"),
                v_conv1d: f("v_conv1d"),
                f_a_proj: f("f_a_proj"),
                f_b_proj: f("f_b_proj"),
                output_gate: KdaOutputGateWeights::LowRank {
                    g_a_proj: f("g_a_proj"),
                    g_b_proj: f("g_b_proj"),
                },
                b_proj: f("b_proj"),
                a_log: f("a_log"),
                dt_bias: f("dt_bias"),
                o_norm: f("o_norm"),
                o_proj: WeightRows::Bf16(fb("o_proj")),
                norm_eps: eps as f32,
                // The rank the gate factorisations meet at — read from
                // this fixture's own `f_b_proj` (`[width, rank]`), not
                // the head dim the executor used to assume.
                gate_rank: f("f_b_proj").len() / kda_geometry.value_width(),
            },
            kda_geometry,
        )
    } else {
        let f = |name: &str| {
            layer.mla_fields[MLA_FIELDS.iter().position(|&n| n == name).unwrap()].as_slice()
        };
        LayerAttention::Mla(
            MlaWeights {
                output_gate: None,
                q_proj: WeightRows::F32(f("q_proj")),
                kv_a_proj: WeightRows::F32(f("kv_a_proj")),
                kv_a_norm: f("kv_a_norm"),
                kv_b_proj: WeightRows::F32(f("kv_b_proj")),
                o_proj: WeightRows::F32(f("o_proj")),
                kv_a_norm_eps: mla_kv_a_norm_eps,
            },
            mla_geometry,
        )
    };
    let ffn = if let Some((gate, up, down)) = &layer.dense_ffn {
        LayerFfn::Dense {
            weights: ExpertWeights { gate, up, down },
            inter: dense_intermediate,
        }
    } else {
        let shared = layer
            .shared
            .as_ref()
            .map(|(gate, up, down)| (ExpertWeights { gate, up, down }, moe_intermediate));
        LayerFfn::Moe {
            router_weight: layer.router_weight.as_ref().unwrap(),
            router_bias: layer.router_bias.as_ref().unwrap(),
            experts,
            top_k,
            renormalize,
            branch_scale,
            loaded,
            shared,
            inter: moe_intermediate,
        }
    };
    LayerSpec {
        attention,
        ffn,
        input_norm_weight: &layer.input_norm,
        post_attention_norm_weight: &layer.post_norm,
        norm_eps: eps,
    }
}

#[test]
fn three_positions_through_all_27_real_layers_match_the_oracle() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");

    let hidden = manifest["hidden"].as_u64().unwrap() as usize;
    let eps = manifest["rms_eps"].as_f64().unwrap();
    let kda_geometry = KdaGeometry {
        num_heads: manifest["kda_num_heads"].as_u64().unwrap() as usize,
        head_dim: manifest["kda_head_dim"].as_u64().unwrap() as usize,
        conv_kernel: manifest["kda_conv_kernel"].as_u64().unwrap() as usize,
    };
    let mla_geometry = MlaGeometry {
        num_heads: manifest["mla_num_heads"].as_u64().unwrap() as usize,
        kv_lora_rank: manifest["mla_kv_lora_rank"].as_u64().unwrap() as usize,
        qk_nope_head_dim: manifest["mla_qk_nope_head_dim"].as_u64().unwrap() as usize,
        qk_rope_head_dim: manifest["mla_qk_rope_head_dim"].as_u64().unwrap() as usize,
        v_head_dim: manifest["mla_v_head_dim"].as_u64().unwrap() as usize,
    };
    let mla_kv_a_norm_eps = manifest["mla_kv_a_norm_eps"].as_f64().unwrap();
    let experts = manifest["experts"].as_u64().unwrap() as usize;
    let top_k = manifest["top_k"].as_u64().unwrap() as usize;
    let moe_intermediate = manifest["moe_intermediate_size"].as_u64().unwrap() as usize;
    let dense_intermediate = manifest["dense_intermediate_size"].as_u64().unwrap() as usize;
    let renormalize = manifest["moe_renormalize"].as_bool().unwrap();
    let branch_scale = manifest["routed_scaling_factor"].as_f64().unwrap();
    let num_layers = manifest["num_layers"].as_u64().unwrap() as usize;
    let positions = manifest["positions"].as_u64().unwrap() as usize;
    assert_eq!(
        num_layers, 27,
        "this gate exists to run the REAL 27-layer topology"
    );
    assert_eq!(
        positions, 3,
        "MLA's attention math needs ≥2 real positions to be non-degenerate"
    );
    assert_eq!(
        (experts, top_k),
        (256, 8),
        "this gate exists to run REAL geometry"
    );
    assert_eq!(
        dense_intermediate, 9216,
        "layer 0's dense KimiMLP is 9216-wide, NOT moe_intermediate_size"
    );

    let inputs: Vec<Vec<f32>> = (0..positions)
        .map(|p| read_f32(&dir, &format!("input_{p}")))
        .collect();

    eprintln!(
        "  loading {num_layers} real layers from {}...",
        dir.display()
    );
    let layers: Vec<RealLayer> = (0..num_layers)
        .map(|i| load_real_layer(&dir, i, &manifest))
        .collect();
    for (i, l) in layers.iter().enumerate() {
        assert_eq!(
            l.dense,
            l.dense_ffn.is_some(),
            "layer {i}: the manifest's declared `dense` flag must agree with which weights it dumped"
        );
    }
    let expert_lists: Vec<Vec<LoadedExpert<'_>>> = layers.iter().map(expert_list_for).collect();
    let specs: Vec<LayerSpec<'_>> = layers
        .iter()
        .zip(&expert_lists)
        .map(|(l, loaded)| {
            spec(
                l,
                kda_geometry,
                mla_geometry,
                mla_kv_a_norm_eps,
                experts,
                top_k,
                moe_intermediate,
                dense_intermediate,
                renormalize,
                branch_scale,
                eps,
                loaded,
            )
        })
        .collect();

    let mut states: Vec<LayerState> = layers
        .iter()
        .map(|l| {
            if l.kind == "kda" {
                LayerState::Kda(zero_state(kda_geometry))
            } else {
                LayerState::Mla(MlaState::default())
            }
        })
        .collect();

    for (p, input) in inputs.iter().enumerate().take(positions) {
        eprintln!("  position {p}: running all {num_layers} layers...");
        let traces = stack_forward(input, hidden, &specs, &mut states);
        assert_eq!(traces.len(), num_layers);

        for (i, t) in traces.iter().enumerate() {
            let expected_kind = if layers[i].kind == "mla" {
                AttentionKind::Mla
            } else {
                AttentionKind::Kda
            };
            assert_eq!(
                t.kind, expected_kind,
                "layer {i} position {p}: operator dispatch"
            );

            assert_close(
                &format!("layer {i} position {p} input_residual"),
                &t.input_residual,
                &read_f32(&dir, &format!("layer{i}_out_input_residual_{p}")),
            );
            assert_close(
                &format!("layer {i} position {p} attention_output"),
                &t.attention_output,
                &read_f32(&dir, &format!("layer{i}_out_attention_output_{p}")),
            );
            assert_close(
                &format!("layer {i} position {p} post_attention_residual"),
                &t.post_attention_residual,
                &read_f32(&dir, &format!("layer{i}_out_post_attention_residual_{p}")),
            );

            // The router's own selected-ids/weights split is already
            // proven in isolation (`kimi_router.rs`/`kimi_moe_block.rs`)
            // and re-proven per real layer in `kimi_kda_layer_real.rs`/
            // `kimi_mla_layer_real.rs`. `StackLayerTrace` intentionally
            // exposes only `ffn_output` — the COMBINED result — not the
            // router's internals, so this gate's evidence that routing
            // stayed correct is numeric: a wrong selected SET would move
            // `ffn_output` far outside `TOLERANCE`, not agree within it.
            assert_close(
                &format!("layer {i} position {p} ffn_output"),
                &t.ffn_output,
                &read_f32(&dir, &format!("layer{i}_out_ffn_output_{p}")),
            );
            assert_close(
                &format!("layer {i} position {p} layer_output"),
                &t.layer_output,
                &read_f32(&dir, &format!("layer{i}_out_layer_output_{p}")),
            );
        }

        let final_out = &traces.last().unwrap().layer_output;
        assert_close(
            &format!("position {p} final stack output"),
            final_out,
            &read_f32(&dir, &format!("final_output_{p}")),
        );
    }

    eprintln!(
        "  all {num_layers} layers, {positions} positions, real Kimi Linear weights, every boundary within {TOLERANCE:e}"
    );
}
