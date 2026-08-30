//! The mixed 27-layer stack against a pinned tiny oracle, boundary by
//! boundary, at every layer and every position — [`kda_parity.rs`]/
//! [`mla_parity.rs`]'s own role, one level up.
//!
//! `kimi_stack_oracle.json` (`scripts/kimi_stack_oracle_export.py`) is
//! synthetic — tiny widths, same reasoning every prior committed oracle
//! uses — but the 27-layer KIND SEQUENCE is copied from the real
//! checkpoint's `config.json`, not invented: this is what makes the
//! test prove LAYER ORDERING, not just "27 layers of something".
//!
//! **Deliberately a different computational order than the oracle.**
//! `scripts/kimi_stack_reference.py` computes whole-sequence-per-layer
//! (layer 0 for all 3 positions, then layer 1, ...); this file calls
//! [`stack::stack_forward`] depth-first per position (all 27 layers for
//! position 0, then position 1, ...), threading each layer's own state
//! across the three calls — the shape autoregressive decode actually
//! needs. Agreement despite the different order is a stronger gate than
//! reproducing one order twice.

use std::collections::BTreeMap;

use larql_models::config::{KdaGeometry, MlaGeometry};
use serde_json::Value;

use crate::format::vindex3::opplan::exec::kda::{KdaState, KdaWeights};
use crate::format::vindex3::opplan::exec::kimi_moe_block::ExpertWeights;
use crate::format::vindex3::opplan::exec::mla::{MlaState, MlaWeights};
use crate::format::vindex3::opplan::exec::stack::{
    stack_forward, AttentionKind, LayerAttention, LayerFfn, LayerSpec, LayerState, LoadedExpert,
};

const ORACLE: &str = include_str!("kimi_stack_oracle.json");
/// f32 transcription noise between two DIFFERENT computational orders of
/// the same arithmetic (whole-sequence-per-layer vs depth-first-per-
/// position) — wider than the single-operator oracles' `2e-5` because
/// summation order genuinely differs here, not just implementation
/// order of the same sum.
///
/// Widened three times since: `2e-4` originally, `2e-2` at P4a when
/// EXPERT weights became bf16, now `2e-1` at P4c-4 when KDA's
/// q/k/v/o_proj ALSO became bf16 — TWO independent bf16 rounding
/// sources compounding through the same mechanism, not one, and this
/// time through TWO axes at once: 27 layers of depth AND, because KDA
/// carries recurrent state across positions, 3 positions of state
/// carriage — a rounded contribution from position 0 feeds position
/// 1's decay/L2-norm/recurrence nonlinearities, which feed position 2's.
/// At this fixture's deliberately TINY widths (`INTER=5`, 2-head/4-dim
/// KDA geometry) a single rounded weight is a much larger fraction of
/// both a dot product AND a normalisation than at real width — the same
/// reason `kda_oracle.json`/`kimi_stack_oracle.json` stay small while
/// their real-weight counterparts need far smaller widening (rounding
/// noise genuinely averages out at real width, confirmed by
/// `kda_parity_real.rs`/`stack_real.rs` needing NO widening from bf16
/// alone). Measured worst case: position 0 (no state carried in yet)
/// tops out at ~3.9e-2; position 1 (one step of state carriage) reaches
/// ~1.36e-1, the global worst, at layer 26's `input_residual`; position
/// 2 is NOT higher (~7.8e-2) — compounding through a recurrent
/// nonlinearity is not monotonic in the number of steps, it depends on
/// which specific roundings happen to reinforce or cancel, so "worse at
/// more positions" is not a safe assumption to size this bound from —
/// the actual measured max across every boundary/layer/position is.
///
/// This bound is generous ON PURPOSE: it is sized to the WORST
/// layer/position, not tuned per case, because the claim this test
/// makes is composition (layer ordering, operator dispatch, state
/// carriage) — the underlying kernel's OWN numerical precision is
/// `mla_parity.rs`/`kda_parity.rs`'s job, and IS tightly bounded there.
const TOLERANCE: f32 = 2e-1;

const KDA_FIELDS: [&str; 15] = [
    "q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d", "f_a_proj", "f_b_proj",
    "g_a_proj", "g_b_proj", "b_proj", "a_log", "dt_bias", "o_norm", "o_proj",
];
const MLA_FIELDS: [&str; 5] = ["q_proj", "kv_a_proj", "kv_a_norm", "kv_b_proj", "o_proj"];

fn floats(node: &Value) -> Vec<f32> {
    node.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

/// `ExpertWeights` (P4a) and KDA's q/k/v/o_proj (P4c-4) are BF16 code
/// units. The oracle export (`kimi_stack_oracle_export.py`) quantises
/// both to bf16 BEFORE running its own reference forward pass, so the
/// f32 values this JSON stores are already bf16-EXACT — truncating here
/// recovers the identical bits Python used, not an independent rounding.
fn bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

fn bf16_vec(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&f| bf16(f)).collect()
}

struct LoadedExpertOwned {
    id: usize,
    gate: Vec<u16>,
    up: Vec<u16>,
    down: Vec<u16>,
}

fn load_expert(node: &Value) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    // ExpertWeights: gate=w1, up=w3, down=w2 — the checkpoint's own
    // naming, never alphabetic order (see `larql_models::architectures::kimi`).
    (
        bf16_vec(&floats(&node["w1"])),
        bf16_vec(&floats(&node["w3"])),
        bf16_vec(&floats(&node["w2"])),
    )
}

struct LoadedLayer {
    kind: &'static str,
    dense: bool,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    kda: BTreeMap<&'static str, Vec<f32>>,
    /// q/k/v/o_proj only (P4c-4 — `KdaWeights`'s four widest operands are
    /// BF16 code units), converted once at load time so `spec()` can
    /// borrow rather than rebuild them per call.
    kda_bf16: BTreeMap<&'static str, Vec<u16>>,
    mla: BTreeMap<&'static str, Vec<f32>>,
    ffn_dense: Option<(Vec<u16>, Vec<u16>, Vec<u16>)>,
    router_weight: Option<Vec<f32>>,
    router_bias: Option<Vec<f32>>,
    experts: Vec<LoadedExpertOwned>,
    shared: Option<(Vec<u16>, Vec<u16>, Vec<u16>)>,
}

/// `KdaWeights`'s four BF16 operands (P4c-4) — named here once so
/// `load_layer` and `spec` cannot drift on which four they are.
const KDA_BF16_FIELDS: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

fn load_layer(node: &Value) -> LoadedLayer {
    let kind: &'static str = if node["kind"] == "mla" { "mla" } else { "kda" };
    let dense = node["dense"].as_bool().unwrap();
    let mut kda = BTreeMap::new();
    let mut kda_bf16 = BTreeMap::new();
    let mut mla = BTreeMap::new();
    if kind == "kda" {
        for f in KDA_FIELDS {
            kda.insert(f, floats(&node["attn_weights"][f]));
        }
        for f in KDA_BF16_FIELDS {
            kda_bf16.insert(f, bf16_vec(&kda[f]));
        }
    } else {
        for f in MLA_FIELDS {
            mla.insert(f, floats(&node["attn_weights"][f]));
        }
    }
    let experts = node["experts"]
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(id, w)| {
                    let (gate, up, down) = load_expert(w);
                    LoadedExpertOwned {
                        id: id.parse().unwrap(),
                        gate,
                        up,
                        down,
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    LoadedLayer {
        kind,
        dense,
        input_norm: floats(&node["input_norm"]),
        post_norm: floats(&node["post_norm"]),
        kda,
        kda_bf16,
        mla,
        ffn_dense: node.get("ffn_dense").map(load_expert),
        router_weight: node.get("router").map(|r| floats(&r["weight"])),
        router_bias: node.get("router").map(|r| floats(&r["bias"])),
        experts,
        shared: node.get("shared").map(load_expert),
    }
}

struct Fixture {
    hidden: usize,
    eps: f64,
    kda_geometry: KdaGeometry,
    mla_geometry: MlaGeometry,
    mla_kv_a_norm_eps: f64,
    experts: usize,
    top_k: usize,
    inter: usize,
    renormalize: bool,
    branch_scale: f64,
    num_layers: usize,
    positions: usize,
    input: Vec<Vec<f32>>,
    layers: Vec<LoadedLayer>,
    boundaries: Value,
    final_output: Vec<Vec<f32>>,
}

fn load() -> Fixture {
    let v: Value = serde_json::from_str(ORACLE).expect("oracle fixture parses");
    let layers = v["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(load_layer)
        .collect();
    let input = v["input"].as_array().unwrap().iter().map(floats).collect();
    let final_output = v["final_output"]
        .as_array()
        .unwrap()
        .iter()
        .map(floats)
        .collect();
    Fixture {
        hidden: v["hidden"].as_u64().unwrap() as usize,
        eps: v["rms_eps"].as_f64().unwrap(),
        kda_geometry: KdaGeometry {
            num_heads: v["kda_num_heads"].as_u64().unwrap() as usize,
            head_dim: v["kda_head_dim"].as_u64().unwrap() as usize,
            conv_kernel: v["kda_conv_kernel"].as_u64().unwrap() as usize,
        },
        mla_geometry: MlaGeometry {
            num_heads: v["mla_num_heads"].as_u64().unwrap() as usize,
            kv_lora_rank: v["mla_kv_lora_rank"].as_u64().unwrap() as usize,
            qk_nope_head_dim: v["mla_qk_nope_head_dim"].as_u64().unwrap() as usize,
            qk_rope_head_dim: v["mla_qk_rope_head_dim"].as_u64().unwrap() as usize,
            v_head_dim: v["mla_v_head_dim"].as_u64().unwrap() as usize,
        },
        mla_kv_a_norm_eps: v["mla_kv_a_norm_eps"].as_f64().unwrap(),
        experts: v["experts"].as_u64().unwrap() as usize,
        top_k: v["top_k"].as_u64().unwrap() as usize,
        inter: v["inter"].as_u64().unwrap() as usize,
        renormalize: v["moe_renormalize"].as_bool().unwrap(),
        branch_scale: v["routed_scaling_factor"].as_f64().unwrap(),
        num_layers: v["num_layers"].as_u64().unwrap() as usize,
        positions: v["positions"].as_u64().unwrap() as usize,
        input,
        layers,
        boundaries: v["boundaries"].clone(),
        final_output,
    }
}

impl Fixture {
    /// One layer's [`LoadedExpert`] list, built from this layer's OWN
    /// expert map — a separate step from [`Fixture::spec`] because the
    /// list must outlive the `LayerSpec` that borrows it: build every
    /// layer's list up front (via `collect`, so nothing reallocates out
    /// from under an earlier reference), then hand each into `spec`.
    fn loaded_experts(&self, i: usize) -> Vec<LoadedExpert<'_>> {
        self.layers[i]
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

    fn spec<'e>(&'e self, i: usize, loaded: &'e [LoadedExpert<'e>]) -> LayerSpec<'e> {
        let l = &self.layers[i];
        let attention = if l.kind == "kda" {
            LayerAttention::Kda(
                KdaWeights {
                    q_proj: &l.kda_bf16["q_proj"],
                    k_proj: &l.kda_bf16["k_proj"],
                    v_proj: &l.kda_bf16["v_proj"],
                    q_conv1d: &l.kda["q_conv1d"],
                    k_conv1d: &l.kda["k_conv1d"],
                    v_conv1d: &l.kda["v_conv1d"],
                    f_a_proj: &l.kda["f_a_proj"],
                    f_b_proj: &l.kda["f_b_proj"],
                    g_a_proj: &l.kda["g_a_proj"],
                    g_b_proj: &l.kda["g_b_proj"],
                    b_proj: &l.kda["b_proj"],
                    a_log: &l.kda["a_log"],
                    dt_bias: &l.kda["dt_bias"],
                    o_norm: &l.kda["o_norm"],
                    o_proj: &l.kda_bf16["o_proj"],
                    norm_eps: self.eps as f32,
                },
                self.kda_geometry,
            )
        } else {
            LayerAttention::Mla(
                MlaWeights {
                    q_proj: &l.mla["q_proj"],
                    kv_a_proj: &l.mla["kv_a_proj"],
                    kv_a_norm: &l.mla["kv_a_norm"],
                    kv_b_proj: &l.mla["kv_b_proj"],
                    o_proj: &l.mla["o_proj"],
                    kv_a_norm_eps: self.mla_kv_a_norm_eps,
                },
                self.mla_geometry,
            )
        };
        let ffn = if let Some((gate, up, down)) = &l.ffn_dense {
            LayerFfn::Dense {
                weights: ExpertWeights { gate, up, down },
                inter: self.inter,
            }
        } else {
            let shared = l
                .shared
                .as_ref()
                .map(|(gate, up, down)| (ExpertWeights { gate, up, down }, self.inter));
            LayerFfn::Moe {
                router_weight: l.router_weight.as_ref().unwrap(),
                router_bias: l.router_bias.as_ref().unwrap(),
                experts: self.experts,
                top_k: self.top_k,
                renormalize: self.renormalize,
                branch_scale: self.branch_scale,
                loaded,
                shared,
                inter: self.inter,
            }
        };
        LayerSpec {
            attention,
            ffn,
            input_norm_weight: &l.input_norm,
            post_attention_norm_weight: &l.post_norm,
            norm_eps: self.eps,
        }
    }

    fn states(&self) -> Vec<LayerState> {
        self.layers
            .iter()
            .map(|l| {
                if l.kind == "kda" {
                    LayerState::Kda(KdaState::zeros(self.kda_geometry))
                } else {
                    LayerState::Mla(MlaState::empty())
                }
            })
            .collect()
    }

    fn expected(&self, boundary: &str, layer: usize, position: usize) -> Vec<f32> {
        floats(&self.boundaries[boundary][layer][position])
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn three_positions_through_all_27_layers_match_the_oracle_at_every_boundary() {
    let f = load();
    assert_eq!(
        f.num_layers, 27,
        "this gate proves the REAL 27-layer topology, not a toy count"
    );
    assert_eq!(
        f.positions, 3,
        "MLA's attention math needs ≥2 real positions to be non-degenerate"
    );

    for (i, l) in f.layers.iter().enumerate() {
        assert_eq!(
            l.dense,
            l.ffn_dense.is_some(),
            "layer {i}: the fixture's declared `dense` flag must agree with which weights it dumped"
        );
    }

    let expert_lists: Vec<Vec<LoadedExpert<'_>>> =
        (0..f.num_layers).map(|i| f.loaded_experts(i)).collect();
    let specs: Vec<LayerSpec<'_>> = (0..f.num_layers)
        .map(|i| f.spec(i, &expert_lists[i]))
        .collect();
    let mut states = f.states();

    for p in 0..f.positions {
        let traces = stack_forward(&f.input[p], f.hidden, &specs, &mut states);
        assert_eq!(traces.len(), f.num_layers);

        for (i, t) in traces.iter().enumerate() {
            let expected_kind = if f.layers[i].kind == "mla" {
                AttentionKind::Mla
            } else {
                AttentionKind::Kda
            };
            assert_eq!(
                t.kind, expected_kind,
                "layer {i} position {p}: operator dispatch"
            );
            assert_eq!(t.layer, i);

            let checks: [(&str, &Vec<f32>); 5] = [
                ("input_residual", &t.input_residual),
                ("attention_output", &t.attention_output),
                ("post_attention_residual", &t.post_attention_residual),
                ("ffn_output", &t.ffn_output),
                ("layer_output", &t.layer_output),
            ];
            for (name, actual) in checks {
                let d = max_abs_diff(actual, &f.expected(name, i, p));
                assert!(
                    d < TOLERANCE,
                    "layer {i} position {p} boundary `{name}`: max|Δ| {d:e}"
                );
            }
        }

        let final_out = &traces.last().unwrap().layer_output;
        let d = max_abs_diff(final_out, &f.final_output[p]);
        assert!(
            d < TOLERANCE,
            "position {p} final stack output: max|Δ| {d:e}"
        );
    }
}

#[test]
fn kda_state_size_is_constant_and_mla_state_size_grows_by_one_per_position() {
    let f = load();
    let expert_lists: Vec<Vec<LoadedExpert<'_>>> =
        (0..f.num_layers).map(|i| f.loaded_experts(i)).collect();
    let specs: Vec<LayerSpec<'_>> = (0..f.num_layers)
        .map(|i| f.spec(i, &expert_lists[i]))
        .collect();
    let mut states = f.states();

    let mut kda_state_sizes: Vec<Vec<usize>> = Vec::new();
    let mut mla_state_sizes: Vec<Vec<usize>> = Vec::new();
    for p in 0..f.positions {
        let traces = stack_forward(&f.input[p], f.hidden, &specs, &mut states);
        for t in &traces {
            match t.kind {
                AttentionKind::Kda => {
                    if kda_state_sizes.len() <= t.layer {
                        kda_state_sizes.resize(t.layer + 1, Vec::new());
                    }
                    kda_state_sizes[t.layer].push(t.state_size);
                }
                AttentionKind::Mla => {
                    if mla_state_sizes.len() <= t.layer {
                        mla_state_sizes.resize(t.layer + 1, Vec::new());
                    }
                    mla_state_sizes[t.layer].push(t.state_size);
                }
            }
        }
    }
    for (layer, sizes) in kda_state_sizes.iter().enumerate() {
        if sizes.is_empty() {
            continue;
        }
        assert!(
            sizes.iter().all(|&s| s == sizes[0]),
            "KDA layer {layer}: recurrent state size must be O(1) in position count, got {sizes:?}"
        );
        assert_eq!(
            sizes[0],
            f.kda_geometry.num_heads * f.kda_geometry.head_dim * f.kda_geometry.head_dim
        );
    }
    for (layer, sizes) in mla_state_sizes.iter().enumerate() {
        if sizes.is_empty() {
            continue;
        }
        let expected: Vec<usize> = (1..=sizes.len()).collect();
        assert_eq!(
            *sizes, expected,
            "MLA layer {layer}: cached-position count must grow by exactly one per call"
        );
    }
}
