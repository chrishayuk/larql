//! **The real trajectory.** Sixteen deterministic greedy tokens through
//! the genuine mixed stack: Metal KDA+MoE segments, CPU MLA layers.
//!
//! Every ladder rung up to 5f ran on a fixture. This runs the model:
//! the same prompt, the same 19 positions, the same 16/16 token gate and
//! top-10 ranking checks as `generate_real.rs`, with persistent state
//! everywhere — KDA recurrent and convolution state resident on device
//! for the whole decode, MLA KV cache on the host — and every routing
//! decision computed on device from a hidden state the host never sees.
//!
//! **Kimi Linear's topology decides the epoch length, not us.** Layers
//! 3, 7, 11, 15, 19, 23 and 26 are MLA and are not ported, so the decode
//! is roughly three KDA layers per GPU epoch:
//!
//! ```text
//! [Metal: 1..2] MLA 3 [Metal: 4..6] MLA 7 [Metal: 8..10] MLA 11 ...
//! ```
//!
//! Seven deliberate crossings a token instead of twenty-six. Rung 5a's
//! curve says four blocks per command buffer capture ~88% of the epoch
//! prize, so three collects most of it — the topology lands near the
//! measured sweet spot by accident.
//!
//! Layer 0 is dense (`KimiMLP`, no router) and stays on the host: it is
//! one layer of twenty-seven and porting a second FFN shape to save it
//! would be work aimed at 4% of the stack.
//!
//! ```text
//! python scripts/kimi_generate_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --tokens 1008 10484 318 --new 16 --out /tmp/kimi_generate_fixture
//! LARQL_KIMI_GENERATE_FIXTURE=/tmp/kimi_generate_fixture \
//!   cargo test -p larql-vindex --features gpu --release --lib generate_metal -- --nocapture
//! ```

use std::path::Path;
use std::time::Instant;

use crate::format::vindex3::represent::physical::{
    EncodedRegion, ExpertBankBinding, ExpertEncoding, ExpertLayout, ExtentPolicy, PhysicalStore,
};
use larql_compute::backend::ComputeBackend;
use larql_compute_metal::shaders::kimi_layer::NOT_RESIDENT;
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kda::{KdaDeviceState, KdaShape};
use larql_compute_metal::trait_impl::kimi_layer::KimiHead;
use larql_compute_metal::MetalBackend;
use larql_models::config::{KdaGeometry, MlaGeometry, NormType};
use serde_json::Value;

use super::stack_real::{expert_list_for, load_real_layer, read_f32, spec, KDA_FIELDS};
use crate::format::vindex3::opplan::exec::cpu::kernels::BlasF32;
use crate::format::vindex3::opplan::exec::cpu::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::opplan::exec::kernels::norm;
use crate::format::vindex3::opplan::exec::stack::{LayerSpec, LayerState};
use crate::format::vindex3::opplan::exec::stack_metal::{
    DeviceAttn, DeviceLayer, DeviceState, HybridHead, HybridStack, HybridTiming,
};
use larql_compute_metal::trait_impl::mla::{MlaDeviceState, MlaShape};

fn larql_compute_metal_timing_default() -> HybridTiming {
    HybridTiming::default()
}

const FIXTURE_ENV: &str = "LARQL_KIMI_GENERATE_FIXTURE";
const TOP_K_RANK: usize = 10;

fn fixture_dir() -> Option<std::path::PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(std::path::PathBuf::from)
}

/// The checkpoint's own bytes for a bf16 tensor — no widening pass.
fn read_bf16_bytes(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{name}.bf16"))).unwrap_or_else(|e| panic!("{name}.bf16: {e}"))
}

/// An f32 tensor read back as the bf16 codes the checkpoint stores.
///
/// The fixture writes MLA's matrices f32 because the host operator
/// consumes f32, but they are a LOSSLESS upcast of the checkpoint's own
/// bf16 — so truncating recovers exactly those bits. The same argument
/// P4c-4 made for KDA's q/k/v/o.
fn read_f32_as_bf16(dir: &Path, name: &str) -> Vec<u8> {
    read_f32(dir, name)
        .iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .expect("logits are never NaN")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// Build one KDA+MoE layer's device form, reading each tensor's bytes
/// straight into the banks.
///
/// Straight in, because the alternative — load every expert as `Vec<u16>`
/// and concatenate afterwards — would hold this layer's ~65 experts
/// twice, and nineteen such layers is ~17 GiB of duplicate. The device
/// arm therefore never materialises the host-side expert form at all for
/// the layers it runs.
#[allow(clippy::too_many_arguments)]
fn device_layer(
    metal: &MetalBackend,
    dir: &Path,
    i: usize,
    manifest: &Value,
    shape: KdaShape,
    mla_shape: MlaShape,
    experts: usize,
    inter: usize,
    top_k: usize,
) -> DeviceLayer {
    let l = &manifest["layers"][i];
    // Layer 0 is `KimiMLP`: three banks holding ONE gated MLP at its own
    // (four times wider) intermediate size, and no router. The banks are
    // the same storage the routed layers use, so everything downstream —
    // registration, residency, the grouped kernel — is unchanged.
    if l["dense"].as_bool().unwrap() {
        let dense_inter = manifest["dense_intermediate_size"].as_u64().unwrap() as usize;
        let (attn, state) = attention_for(metal, dir, i, manifest, shape, mla_shape);
        return DeviceLayer {
            attn,
            state,
            bank: owned_bank(
                read_bf16_bytes(dir, &format!("layer{i}_dense_w1")),
                read_bf16_bytes(dir, &format!("layer{i}_dense_w3")),
                read_bf16_bytes(dir, &format!("layer{i}_dense_w2")),
                ExpertLayout::Mapped { ids: vec![0] },
                false,
            ),
            offsets: Vec::new(),
            expert_stride: 0,
            shared_offset: 0,
            input_norm: read_f32(dir, &format!("layer{i}_input_norm_weight")),
            post_norm: read_f32(dir, &format!("layer{i}_post_norm_weight")),
            router_weight: Vec::new(),
            router_bias: Vec::new(),
            inter: dense_inter,
            top_k: 0,
            dense: true,
            renormalize: manifest["moe_renormalize"].as_bool().unwrap(),
            branch_scale: manifest["routed_scaling_factor"].as_f64().unwrap() as f32,
            norm_eps: manifest["rms_eps"].as_f64().unwrap() as f32,
            kda_shape: shape,
            mla_shape,
            mla_norm_eps: manifest["mla_kv_a_norm_eps"].as_f64().unwrap() as f32,
        };
    }
    let union: Vec<usize> = l["selected_ids_union_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let mut bank_gate = Vec::new();
    let mut bank_up = Vec::new();
    let mut bank_down = Vec::new();
    let mut residency = vec![NOT_RESIDENT; experts];
    for &id in &union {
        residency[id] = bank_gate.len() as u32;
        bank_gate.extend_from_slice(&read_bf16_bytes(dir, &format!("layer{i}_expert{id}_w1")));
        bank_up.extend_from_slice(&read_bf16_bytes(dir, &format!("layer{i}_expert{id}_w3")));
        bank_down.extend_from_slice(&read_bf16_bytes(dir, &format!("layer{i}_expert{id}_w2")));
    }
    let shared_offset = bank_gate.len() as u32;
    bank_gate.extend_from_slice(&read_bf16_bytes(dir, &format!("layer{i}_shared_w1")));
    bank_up.extend_from_slice(&read_bf16_bytes(dir, &format!("layer{i}_shared_w3")));
    bank_down.extend_from_slice(&read_bf16_bytes(dir, &format!("layer{i}_shared_w2")));

    let (attn, state) = attention_for(metal, dir, i, manifest, shape, mla_shape);

    DeviceLayer {
        attn,
        state,
        expert_stride: (bank_gate.len() / (union.len() + 1)) as u32,
        bank: owned_bank(
            bank_gate,
            bank_up,
            bank_down,
            // The fixture's packed union: physical slot != expert id, so
            // this stays a `Mapped` view and the existing offset table
            // is what addresses it. Deliberately NOT migrated to the
            // compiled full-bank layout — one changed variable at a time.
            ExpertLayout::Mapped {
                ids: union.iter().map(|e| *e as u32).collect(),
            },
            true,
        ),
        offsets: residency,
        shared_offset,
        input_norm: read_f32(dir, &format!("layer{i}_input_norm_weight")),
        post_norm: read_f32(dir, &format!("layer{i}_post_norm_weight")),
        router_weight: read_f32(dir, &format!("layer{i}_router_weight")),
        router_bias: read_f32(dir, &format!("layer{i}_router_bias")),
        inter,
        top_k,
        dense: false,
        renormalize: manifest["moe_renormalize"].as_bool().unwrap(),
        branch_scale: manifest["routed_scaling_factor"].as_f64().unwrap() as f32,
        norm_eps: manifest["rms_eps"].as_f64().unwrap() as f32,
        kda_shape: shape,
        mla_shape,
        mla_norm_eps: manifest["mla_kv_a_norm_eps"].as_f64().unwrap() as f32,
    }
}

/// The attention half of one layer, whichever operator it runs.
///
/// Shared by the routed and dense paths: which FFN a layer has says
/// nothing about its attention, and Kimi's dense layer 0 is a KDA layer
/// like any other.
fn attention_for(
    metal: &MetalBackend,
    dir: &Path,
    i: usize,
    manifest: &Value,
    shape: KdaShape,
    mla_shape: MlaShape,
) -> (DeviceAttn, DeviceState) {
    let l = &manifest["layers"][i];
    let is_kda = l["kind"] != "mla";
    let (attn, state) = if is_kda {
        let (qb, kb, vb) = (
            read_bf16_bytes(dir, &format!("layer{i}_kda_q_proj")),
            read_bf16_bytes(dir, &format!("layer{i}_kda_k_proj")),
            read_bf16_bytes(dir, &format!("layer{i}_kda_v_proj")),
        );
        let per = qb.len();
        let mut qkv_bank = Vec::with_capacity(3 * per);
        for b in [&qb, &kb, &vb] {
            qkv_bank.extend_from_slice(b);
        }
        drop((qb, kb, vb));

        // `KdaDeviceWeights`'s own field order, taken from `KDA_FIELDS`
        // by name so a reordering there cannot silently mis-bind an
        // operand.
        let f32_order = [
            "q_conv1d", "k_conv1d", "v_conv1d", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj",
            "b_proj", "a_log", "dt_bias", "o_norm",
        ];
        for name in f32_order {
            assert!(KDA_FIELDS.contains(&name), "{name} is not a KDA operand");
        }
        (
            DeviceAttn::Kda {
                qkv_bank,
                qkv_offsets: [
                    ExpertOffset(0),
                    ExpertOffset(per as u32),
                    ExpertOffset((2 * per) as u32),
                ],
                o_proj: read_bf16_bytes(dir, &format!("layer{i}_kda_o_proj")),
                f32s: f32_order
                    .iter()
                    .map(|f| read_f32(dir, &format!("layer{i}_kda_{f}")))
                    .collect(),
            },
            DeviceState::Kda(KdaDeviceState::zeros(metal, shape)),
        )
    } else {
        let mla = |n: &str| read_f32_as_bf16(dir, &format!("layer{i}_mla_{n}"));
        (
            DeviceAttn::Mla {
                q: mla("q_proj"),
                kv_a: mla("kv_a_proj"),
                kv_b: mla("kv_b_proj"),
                o: mla("o_proj"),
                kv_a_norm: read_f32(dir, &format!("layer{i}_mla_kv_a_norm")),
            },
            DeviceState::Mla(MlaDeviceState::with_capacity(
                metal,
                mla_shape,
                MLA_CACHE_POSITIONS,
            )),
        )
    };
    (attn, state)
}

/// Wrap owned bytes as three physical regions.
///
/// The fixture path keeps its packed banks exactly as they were; only
/// the OWNERSHIP vocabulary changes, so `DeviceLayer` can hold the same
/// bytes as regions without the loader being rewritten.
fn owned_bank(
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    layout: ExpertLayout,
    shared_branch: bool,
) -> ExpertBankBinding {
    let region = |id: &str, bytes: Vec<u8>| {
        let len = bytes.len() as u64;
        let store = std::sync::Arc::new(PhysicalStore::owned(
            id,
            bytes,
            std::collections::BTreeMap::from([("bank".to_string(), (0, len))]),
        ));
        EncodedRegion {
            region: PhysicalStore::whole(&store, "bank").expect("the bank is its own only tensor"),
            // The fixture stores the checkpoint's own bf16.
            encoding: ExpertEncoding::Bf16,
        }
    };
    ExpertBankBinding {
        gate: region("fixture", gate),
        up: region("fixture", up),
        down: region("fixture", down),
        layout,
        // The fixture's packed banks ARE the bank.
        extent: ExtentPolicy::Exact,
        shared_branch,
    }
}

/// Cache positions each MLA layer is built for. The trajectory runs 19;
/// a sequence longer than this is refused, not truncated.
const MLA_CACHE_POSITIONS: usize = 64;

#[test]
fn sixteen_greedy_tokens_through_the_mixed_metal_stack() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => {
            #[cfg(target_os = "macos")]
            panic!(
                "MetalBackend::new() returned None on macOS — the shader library almost \
                 certainly failed to compile. Run `cargo test -p larql-compute-metal --lib`."
            );
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("skipped: no Metal device on this host");
                return;
            }
        }
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    let g = |k: &str| manifest[k].as_u64().unwrap() as usize;
    let (hidden, experts, top_k) = (g("hidden"), g("experts"), g("top_k"));
    let (inter, num_layers, positions, vocab) = (
        g("moe_intermediate_size"),
        g("num_layers"),
        g("positions"),
        g("vocab_size"),
    );
    let eps = manifest["rms_eps"].as_f64().unwrap();
    let kda_geometry = KdaGeometry {
        num_heads: g("kda_num_heads"),
        head_dim: g("kda_head_dim"),
        conv_kernel: g("kda_conv_kernel"),
    };
    let mla_geometry = MlaGeometry {
        num_heads: g("mla_num_heads"),
        kv_lora_rank: g("mla_kv_lora_rank"),
        qk_nope_head_dim: g("mla_qk_nope_head_dim"),
        qk_rope_head_dim: g("mla_qk_rope_head_dim"),
        v_head_dim: g("mla_v_head_dim"),
    };
    let shape = KdaShape {
        hidden,
        num_heads: kda_geometry.num_heads,
        head_dim: kda_geometry.head_dim,
        conv_kernel: kda_geometry.conv_kernel,
    };
    let mla_device_shape = MlaShape {
        hidden,
        num_heads: mla_geometry.num_heads,
        kv_lora_rank: mla_geometry.kv_lora_rank,
        qk_nope_head_dim: mla_geometry.qk_nope_head_dim,
        qk_rope_head_dim: mla_geometry.qk_rope_head_dim,
        v_head_dim: mla_geometry.v_head_dim,
    };
    let ids = |k: &str| -> Vec<usize> {
        manifest[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    let (token_ids, want_argmax) = (ids("token_ids"), ids("argmax_ids"));
    let prompt_len = ids("prompt_tokens").len();
    let want_top10: Vec<Vec<usize>> = manifest["top10_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();

    // Which layers go where — read from the checkpoint's own topology,
    // never assumed.
    // Every layer, including the dense one. The FFN half is a parameter
    // of the decoder layer the same way attention became one, so there
    // is no host break left in a token at all — one command buffer from
    // the embedding to the logits.
    let on_device: Vec<bool> = vec![true; num_layers];
    let device_count = on_device.iter().filter(|b| **b).count();

    let t0 = Instant::now();
    eprintln!("[traj] loading {num_layers} layers ({device_count} on device)...");
    let mut device: Vec<Option<DeviceLayer>> = Vec::with_capacity(num_layers);
    let mut host_raw = Vec::with_capacity(num_layers);
    for (i, &device_side) in on_device.iter().enumerate() {
        if device_side {
            device.push(Some(device_layer(
                &metal,
                &dir,
                i,
                &manifest,
                shape,
                mla_device_shape,
                experts,
                inter,
                top_k,
            )));
            host_raw.push(None);
        } else {
            device.push(None);
            host_raw.push(Some(load_real_layer(&dir, i, &manifest)));
        }
    }
    let host_experts: Vec<Option<Vec<_>>> = host_raw
        .iter()
        .map(|l| l.as_ref().map(expert_list_for))
        .collect();
    let host: Vec<Option<(LayerSpec<'_>, LayerState)>> = host_raw
        .iter()
        .zip(&host_experts)
        .map(|(l, e)| {
            l.as_ref().map(|l| {
                let s = spec(
                    l,
                    kda_geometry,
                    mla_geometry,
                    manifest["mla_kv_a_norm_eps"].as_f64().unwrap(),
                    experts,
                    top_k,
                    inter,
                    g("dense_intermediate_size"),
                    manifest["moe_renormalize"].as_bool().unwrap(),
                    manifest["routed_scaling_factor"].as_f64().unwrap(),
                    eps,
                    e.as_ref().expect("host layer has experts"),
                );
                let state = if l.kind == "kda" {
                    LayerState::Kda(crate::format::vindex3::opplan::exec::kda::KdaState::zeros(
                        kda_geometry,
                    ))
                } else {
                    LayerState::Mla(crate::format::vindex3::opplan::exec::mla::MlaState::empty())
                };
                (s, state)
            })
        })
        .collect();

    // Declare the expert banks resident before the first timed token.
    //
    // ~17 GiB of bank across nineteen layers is bound into every command
    // buffer, and with implicit residency the driver re-establishes that
    // working set per submission. `LARQL_RESIDENCY_SET=2` builds one
    // `MTLResidencySet` over the registered regions and attaches it to
    // the queue instead. Registering is harmless when the arm is off, so
    // the A/B is one environment variable.
    // The head's 755 MB is read here, ahead of sealing, so it joins the
    // residency set with every other weight. bf16, not the f32 the
    // fixture stores: the checkpoint was bf16, the f32 is a lossless
    // upcast of it (verified: no element carries a non-zero low 16
    // bits), so truncating recovers the original values exactly while
    // halving the traffic.
    let final_norm_weight = read_f32(&dir, "final_norm_weight");
    let lm_head_bf16 = read_f32_as_bf16(&dir, "lm_head_weight");
    let vocab_from_head = lm_head_bf16.len() / (hidden * 2);

    let mut registered = 0usize;
    for d in device.iter().flatten() {
        for bank in [
            d.bank.gate.region.bytes(),
            d.bank.up.region.bytes(),
            d.bank.down.region.bytes(),
        ] {
            metal.register_weight_region(bank);
            registered += 1;
        }
        for bank in d.attention_banks() {
            metal.register_weight_region(bank);
            registered += 1;
        }
    }
    metal.register_weight_region(&lm_head_bf16);
    registered += 1;
    metal.seal_weight_regions();
    eprintln!("[traj] {registered} weight regions registered and sealed");

    // Every device layer's banks checked before a single dispatch: the
    // declared encoding must match the bytes actually bound.
    for (i, d) in device.iter().enumerate() {
        if let Some(d) = d {
            d.validate_banks(hidden)
                .unwrap_or_else(|e| panic!("layer {i}'s banks are not what they claim: {e}"));
        }
    }
    let mut stack = HybridStack::new(device, host);
    assert!(
        stack.attach_head(HybridHead {
            norm_weight: final_norm_weight.clone(),
            weight: lm_head_bf16,
            vocab: vocab_from_head,
            norm_eps: eps as f32,
        }),
        "the stack must end on a device layer for the head to ride in its epoch"
    );
    let epochs = stack.epochs();
    eprintln!(
        "[traj] loaded in {:.1}s; {} GPU epochs a token, lengths {:?}",
        t0.elapsed().as_secs_f64(),
        epochs.len(),
        epochs.iter().map(|r| r.len()).collect::<Vec<_>>()
    );

    let embeddings: Vec<Vec<f32>> = (0..positions)
        .map(|p| read_f32(&dir, &format!("embedding_{p}")))
        .collect();
    let mut got_argmax = Vec::with_capacity(positions);
    // Logits kept, top-10 ranked AFTER the timed loop: sorting 163840
    // logits a position is gate instrumentation, not decode work, and
    // costs ~10 ms a token — more than the lm_head it is checking.
    let mut got_logits: Vec<Vec<f32>> = Vec::with_capacity(positions);
    let mut acc = larql_compute_metal_timing_default();
    let head_ms = 0.0f64;
    let decode = Instant::now();
    for embedding in embeddings.iter().take(positions) {
        // With the head attached this IS the logits vector — the final
        // norm and the 163840x2304 projection ran inside the same
        // command buffer as the last 26 layers, so the hidden state
        // never crossed to the host at all.
        let (logits, _traces, t) = stack
            .forward(&metal, embedding, hidden)
            .expect("the mixed stack must not refuse — every selected expert is resident");
        assert_eq!(logits.len(), vocab, "the device head returns logits");
        acc.device_wall_ms += t.device_wall_ms;
        acc.device_gpu_ms += t.device_gpu_ms;
        acc.host_wall_ms += t.host_wall_ms;
        acc.epochs = t.epochs;
        got_argmax.push(
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are never NaN"))
                .expect("vocab is never empty")
                .0,
        );
        got_logits.push(logits);
    }
    let decode_s = decode.elapsed().as_secs_f64();
    let got_top10: Vec<Vec<usize>> = got_logits
        .iter()
        .map(|l| top_k_ids(l, TOP_K_RANK))
        .collect();

    // ── The gate: every one of the 16 generated tokens ──
    let mut first_divergence = None;
    for step in 0..16 {
        let pos = prompt_len - 1 + step;
        if got_argmax[pos] != token_ids[pos + 1] && first_divergence.is_none() {
            first_divergence = Some((step, got_argmax[pos], token_ids[pos + 1]));
        }
    }
    assert_eq!(
        first_divergence, None,
        "the mixed Metal stack diverged from the oracle's trajectory"
    );
    assert_eq!(got_argmax, want_argmax, "argmax at every position");
    for (pos, (got, want)) in got_top10.iter().zip(&want_top10).enumerate() {
        assert_eq!(got, want, "top-{TOP_K_RANK} ranking at position {pos}");
    }

    let generated: Vec<usize> = (0..16).map(|s| got_argmax[prompt_len - 1 + s]).collect();
    eprintln!(
        "[traj] prompt {:?} -> generated {generated:?} — 16/16 tokens match, \
         top-{TOP_K_RANK} stable at all {positions} positions",
        &token_ids[..prompt_len]
    );
    let n = positions as f64;
    eprintln!(
        "[traj] {positions} positions in {decode_s:.3} s = {:.2} tok/s \
         ({:.1} ms/token) — all {device_count} of {num_layers} layers plus the \
         lm_head on device, {} on the host",
        n / decode_s,
        1000.0 * decode_s / n,
        num_layers - device_count,
    );
    let (encode_ms, wait_ms) = larql_compute_metal::trait_impl::kimi_layer::take_chain_timing_ms();
    eprintln!(
        "[traj]   of the device host time: encode {:.1} ms/token, commit+wait {:.1} ms/token \
         (wait includes the {:.1} ms of GPU execution)",
        encode_ms / n,
        wait_ms / n,
        acc.device_gpu_ms / n,
    );
    eprintln!(
        "[traj]   per token: device {:.1} ms (gpu-busy {:.1}, host {:.1} over {} epochs)  \
         host layers {:.1} ms  host head {:.1} ms  unattributed {:.1} ms",
        acc.device_wall_ms / n,
        acc.device_gpu_ms / n,
        (acc.device_wall_ms - acc.device_gpu_ms) / n,
        acc.epochs,
        acc.host_wall_ms / n,
        head_ms / n,
        (1000.0 * decode_s - acc.device_wall_ms - acc.host_wall_ms - head_ms) / n,
    );
}

// The submission-cost probe that used to live here has been removed. It
// asked whether the trajectory's ~3.4 ms an epoch was per-submission
// cost scaling with a 17 GiB working set, or the GPU going idle across
// the CPU MLA layers between epochs, and answered it: 0.23 ms a
// submission, flat in working set, so it was the gaps. That answer is
// what motivated putting MLA on device, and with the gaps gone its
// premise — many epochs a token — no longer holds.

/// **R7 — the device head against the oracle, on its own.**
///
/// The trajectory gate already covers the head: it compares argmax and
/// the top-10 ranking at all 19 positions. This exists so a
/// disagreement names the head instead of the twenty-six layers in
/// front of it, and so the bf16-vs-f32 question gets an actual number
/// rather than an argument.
///
/// The input is the oracle's own last-layer output, so nothing here
/// depends on the stack being correct.
#[test]
fn the_device_head_matches_the_oracle_from_the_last_layers_output() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let Some(metal) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — see the trajectory test");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    let g = |k: &str| manifest[k].as_u64().unwrap() as usize;
    let (hidden, vocab, num_layers) = (g("hidden"), g("vocab_size"), g("num_layers"));
    let eps = manifest["rms_eps"].as_f64().unwrap();

    let h = read_f32(&dir, &format!("layer{}_out_layer_output_0", num_layers - 1));
    let final_norm_weight = read_f32(&dir, "final_norm_weight");
    let want = read_f32(&dir, "logits_0");
    let w_f32 = read_f32(&dir, "lm_head_weight");
    let w_bf16: Vec<u8> = w_f32
        .iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    assert!(
        w_f32.iter().all(|f| f.to_bits() & 0xFFFF == 0),
        "the fixture's f32 lm_head must be a lossless bf16 upcast, or the \
         device arm is reading different VALUES and not merely accumulating \
         them differently"
    );

    let (device, gpu_ms) = metal
        .kimi_head(
            &KimiHead {
                norm_weight: &final_norm_weight,
                norm_eps: eps as f32,
                weight: &w_bf16,
                vocab,
            },
            &h,
        )
        .expect("the head must not refuse at the fixture's real shapes");

    // The CPU control on the same input: identical values, different
    // accumulation order and f32 loads.
    let normed = norm(NormType::RmsNorm, &h, &final_norm_weight, 0.0, eps);
    let mut host = vec![0.0f32; vocab];
    BlasF32.project_rows(WeightRows::F32(&w_f32), &normed, &mut host);

    let d_vs_o = max_abs_delta(&device, &want);
    let h_vs_o = max_abs_delta(&host, &want);
    let d_vs_h = max_abs_delta(&device, &host);
    eprintln!(
        "[r7] {vocab}x{hidden} bf16 head on device in {gpu_ms:.3} ms gpu; \
         max|Δ| device-vs-oracle {d_vs_o:.3e}, host-vs-oracle {h_vs_o:.3e}, \
         device-vs-host {d_vs_h:.3e}"
    );
    assert_eq!(
        argmax(&device),
        argmax(&want),
        "the device head must pick the oracle's token"
    );
    assert_eq!(
        top_k_ids(&device, TOP_K_RANK),
        top_k_ids(&want, TOP_K_RANK),
        "top-{TOP_K_RANK} ranking against the oracle"
    );
    // A control that must FAIL on a different input, so the assertions
    // above are not passing on some property every vector has.
    let mut other = h.clone();
    other[0] += 1.0;
    let (moved, _) = metal
        .kimi_head(
            &KimiHead {
                norm_weight: &final_norm_weight,
                norm_eps: eps as f32,
                weight: &w_bf16,
                vocab,
            },
            &other,
        )
        .expect("head runs");
    assert!(
        max_abs_delta(&moved, &device) > 0.0,
        "perturbing the input must move the logits, or the head is not \
         reading its input"
    );
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are never NaN"))
        .expect("non-empty")
        .0
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
