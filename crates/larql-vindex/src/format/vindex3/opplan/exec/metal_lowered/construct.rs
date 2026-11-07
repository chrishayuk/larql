//! `LoweredSession::new`: bind the container, resolve residents, and build every layer's lowering.

use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::backend::WeightFormats;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::weights::LoadedWeight;
use crate::format::vindex3::opplan::ComponentOpPlan;
use larql_compute::backend::MatMul;
use larql_compute_metal::lowering::DeviceBuffer;
use larql_compute_metal::MetalBackend;
use larql_models::config::PositionPolicy;
use resident::{
    resident_attn, resident_matrix, resident_norm, resident_vector, rope_inv_freq_table,
    rope_table_key, Ablation,
};
use routed::build_ffn;
use std::collections::HashMap;

#[allow(unused_imports)]
use super::*;

impl<'a> LoweredSession<'a> {
    /// Load every operand the plan consumes, once, resident on the
    /// device.
    /// `formats` is the plan's per-class policy, applied here rather
    /// than assumed: attention, FFN and head may each be resident in a
    /// different representation and still execute under one schedule.
    pub fn new(
        gpu: &'a MetalBackend,
        plan: &'a ComponentOpPlan,
        store: &OperandStore,
        formats: WeightFormats,
        max_positions: usize,
        keep: &mut Vec<LoadedWeight>,
    ) -> Result<Self, VindexError> {
        // YaRN, sinks and Q/K/V/O biases are lowered (A-9.4): the
        // amplitude rides slot 6 of the rope kernel, the sinks slot 10/11
        // of the attention kernel, the biases the `bias_add` kernel after
        // each projection, and a routed FFN through the served descriptor
        // MoE path (build_routed).
        //
        // The dense gate policy is checked by `ffn_activation`, layer by
        // layer, in the loop below — not by a blanket "is it plain
        // gating" scan here. That scan predated any dense policy having a
        // kernel; since K3-ACT-1 one does (`SituGlu` → the `situ_glu`
        // kernel) and one still does not (`ClampedGlu`, A-9.4), so the
        // question is no longer whether the policy is plain but whether
        // its combine has a kernel — which `ffn_activation` answers, in
        // one place, for both this pre-check and the encode. A policy
        // with no kernel still refuses BY NAME, naming its parameters.
        // Gemma 4's semantics are lowered (G4.3): K≡V binds the K matrix
        // as V, the V norm and the weighted Q/K norms ride the served
        // norm kernels, the proportional partial rotary rides a per-layer
        // table, the hybrid FFN is composed in `encode_stack`. What the
        // lowering still has no kernel for is refused, typed, here: a
        // rotary-width partial rotary (a prefix rotated as its own block)
        // and a gate activation other than SiLU / tanh-GELU.
        // No DeltaNet kernel exists on this path. Refuse the whole plan
        // rather than lower the 16 softmax layers of a 64-layer hybrid and
        // silently drop the other 48.
        if let Some(l) = plan.layers.iter().find(|l| l.attention.softmax().is_none()) {
            return Err(VindexError::Parse(format!(
                "layer {} carries `{}`, which this lowering has no kernel for; refusing",
                l.layer,
                l.attention.declared_name(),
            )));
        }
        if let Some(l) = plan.layers.iter().find(|l| {
            matches!(
                l.attention.softmax().map(|op| &op.position),
                // M-RoPE joins the refusal on the same ground: its
                // rotary block is prefix-shaped, and its per-slot axis
                // assignment is a second thing the rope kernel does not
                // express.
                Some(
                    PositionPolicy::PartialRope {
                        basis: larql_models::config::RotaryFrequencyBasis::RotaryWidth,
                        ..
                    } | PositionPolicy::MRope { .. }
                )
            )
        }) {
            return Err(VindexError::Parse(format!(
                "layer {} carries {:?}, whose prefix-block rotation the rope kernel does not \
                 express; refusing rather than rotating the whole head",
                l.layer,
                l.attention.softmax().map(|op| &op.position)
            )));
        }
        if let Some(l) = plan
            .layers
            .iter()
            .find(|l| l.layer_scale.is_some() && l.ffn.as_ref().and_then(|f| f.hybrid()).is_none())
        {
            return Err(VindexError::Parse(format!(
                "layer {} carries a layer scalar on a non-hybrid FFN, which the stack encoder \
                 applies only on the hybrid arm today; refusing",
                l.layer
            )));
        }
        // A residual-scale op (Granite `residual_multiplier`) rides the
        // attention and dense/hybrid FFN residual adds; the routed FFN's
        // served MoE path owns its own residual combine and has no slot
        // for it, so refuse rather than silently add the branch unscaled.
        if let Some(l) = plan.layers.iter().find(|l| {
            l.residual_scale.is_some()
                && matches!(
                    l.ffn,
                    Some(crate::format::vindex3::opplan::LayerFfn::Routed(_))
                )
        }) {
            return Err(VindexError::Parse(format!(
                "layer {} carries a residual scale on a routed FFN, whose served combine has \
                 no residual-scale slot; refusing",
                l.layer
            )));
        }
        for l in &plan.layers {
            let activation = match &l.ffn {
                Some(crate::format::vindex3::opplan::LayerFfn::Dense(op)) => {
                    Some((op.activation, op.gate_policy))
                }
                Some(crate::format::vindex3::opplan::LayerFfn::Hybrid(op)) => {
                    Some((op.dense.activation, op.dense.gate_policy))
                }
                Some(crate::format::vindex3::opplan::LayerFfn::Routed(_)) | None => None,
            };
            if let Some((activation, gate_policy)) = activation {
                ffn_activation(activation, gate_policy)
                    .map_err(|e| VindexError::Parse(format!("layer {}: {e}", l.layer)))?;
            }
        }
        let embedding = plan
            .embedding
            .as_ref()
            .ok_or_else(|| VindexError::Parse("plan carries no embedding op".into()))?;
        let embed_table = store.load(&embedding.table)?;
        let hidden = embed_table.len() / embedding.vocab_size;

        let mut layers = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            let a = layer.attention.softmax().unwrap_or_else(|| {
                panic!(
                    "layer {} is not softmax; the lowering refused this plan in `new`",
                    layer.layer
                )
            });
            let kv_rows = a.num_kv_heads * a.head_dim;
            let zeros = vec![0.0f32; max_positions * kv_rows];
            // On a K≡V layer the plan's `v` IS the K operand: the same
            // bytes load through the same cache, so the V projection
            // binds the K matrix — the raw K projection lands in the V
            // slot before the key's own norm and rotation.
            // Q, K, V and O as slices of one allocation, in touch order,
            // where format and alignment admit it; otherwise four
            // separate residents, unchanged.
            let [q_m, k_m, v_m, o_m] = resident_attn(
                gpu,
                store,
                [&a.q, &a.k, &a.v, &a.o],
                formats.attention,
                keep,
            )?;
            layers.push(LayerResident {
                q: q_m,
                k: k_m,
                v: v_m,
                o: o_m,
                qk_norm: match &a.qk_norm {
                    Some(qk) => {
                        let q = resident_vector(gpu, store, Some(&qk.q))?.expect("q norm weight");
                        let k = resident_vector(gpu, store, Some(&qk.k))?.expect("k norm weight");
                        Some((q, k, qk.weight_offset))
                    }
                    None => None,
                },
                rope_key: rope_table_key(&a.position, a.head_dim),
                layer_scale: match &layer.layer_scale {
                    Some(op) => {
                        Some(store.load(op).and_then(|v| {
                            crate::format::vindex3::opplan::exec::layer_scalar_of(&v)
                        })?)
                    }
                    None => None,
                },
                q_bias: resident_vector(gpu, store, a.q_bias.as_ref())?,
                k_bias: resident_vector(gpu, store, a.k_bias.as_ref())?,
                v_bias: resident_vector(gpu, store, a.v_bias.as_ref())?,
                o_bias: resident_vector(gpu, store, a.o_bias.as_ref())?,
                sinks: resident_vector(gpu, store, a.sinks.as_ref().map(|s| &s.logits))?,
                gate: match &a.output_gate {
                    Some(g) => Some(resident_matrix(
                        gpu,
                        store,
                        &g.projection,
                        formats.attention,
                        keep,
                    )?),
                    None => None,
                },
                ffn: build_ffn(gpu, store, layer, formats, keep)?,
                // The Metal trunk applies a norm before attention. A
                // post-norm stack has none, and this path refuses rather
                // than lowering an identity norm that would read as one.
                pre_attn_norm: resident_norm(
                    gpu,
                    store,
                    layer.pre_attention_norm.as_ref().ok_or_else(|| {
                        VindexError::Parse(format!(
                            "layer {} carries no pre-attention norm (post-norm placement); the \
                             Metal trunk has no lowering for it",
                            layer.layer
                        ))
                    })?,
                )?
                .0,
                post_attn_norm: match &layer.post_attention_norm {
                    Some(op) => Some(resident_norm(gpu, store, op)?),
                    None => None,
                },
                pre_ffn_norm: resident_norm(
                    gpu,
                    store,
                    layer.pre_ffn_norm.as_ref().ok_or_else(|| {
                        VindexError::Parse(format!(
                            "layer {} carries no pre-FFN norm (mixer-only); the Metal \
                             lowering has no arm for it",
                            layer.layer
                        ))
                    })?,
                )?
                .0,
                post_ffn_norm: match &layer.post_ffn_norm {
                    Some(op) => Some(resident_norm(gpu, store, op)?),
                    None => None,
                },
                k_cache: gpu
                    .lowering_upload(&zeros)
                    .ok_or_else(|| VindexError::Parse("KV cache allocation failed".into()))?,
                v_cache: gpu
                    .lowering_upload(&zeros)
                    .ok_or_else(|| VindexError::Parse("KV cache allocation failed".into()))?,
            });
        }

        let final_norm = match &plan.final_norm {
            Some(op) => Some(resident_norm(gpu, store, op)?),
            None => None,
        };
        let (head, vocab, head_multiplier, head_softcap) = match &plan.output {
            Some(out) => {
                let m = resident_matrix(gpu, store, &out.projection, formats.head, keep)?;
                let v = m.rows;
                (
                    Some(m),
                    v,
                    out.multiplier.map(|m| m as f32),
                    out.softcapping,
                )
            }
            None => (None, 0, None, None),
        };

        // Scratch sized from the widest layer, allocated once.
        let max_q = plan
            .layers
            .iter()
            .filter_map(|l| l.attention.softmax())
            .map(|op| op.num_q_heads * op.head_dim)
            .max()
            .unwrap_or(hidden);
        let max_q_heads = plan
            .layers
            .iter()
            .filter_map(|l| l.attention.softmax())
            .map(|op| op.num_q_heads)
            .max()
            .unwrap_or(1);
        let splitk_widths = (max_q_heads, max_q);
        let (splitk_o, splitk_ml) =
            larql_compute_metal::ops::kv_splitk::SplitKScratch::lens(1, max_q_heads, max_q);
        let splitk = [
            gpu.lowering_scratch(splitk_o),
            gpu.lowering_scratch(splitk_ml),
        ];
        let max_inter = plan
            .layers
            .iter()
            .filter_map(|l| {
                let ffn = l.ffn.as_ref()?;
                ffn.dense()
                    .map(|f| f.intermediate_size)
                    .or_else(|| ffn.hybrid().map(|h| h.dense.intermediate_size))
            })
            .max()
            .unwrap_or(hidden);
        // Slots 16 and 17 are both vocabulary-sized: the head writes raw
        // logits into one and the scaled/softcapped result into the
        // other. Sizing 16 as `hidden` made the readback fail closed —
        // `try_read_buffer_f32` refuses a buffer shorter than the
        // requested length, which is why this surfaced as "no output
        // head" rather than as garbage logits.
        let sizes = [
            hidden,
            hidden,
            hidden,
            max_q,
            max_q,
            max_q,
            hidden,
            hidden,
            hidden,
            max_inter,
            max_inter,
            max_inter,
            max_q,
            hidden,
            hidden,
            hidden,
            vocab.max(1),
            vocab.max(1),
        ];
        let mut scratch: Vec<DeviceBuffer> =
            sizes.iter().map(|n| gpu.lowering_scratch(*n)).collect();
        let scratch_widths = sizes.to_vec();
        // A hybrid layer's own intermediates (slots 18..24), and a zero
        // buffer for the expert combine's residual input.
        let has_hybrid = plan
            .layers
            .iter()
            .any(|l| l.ffn.as_ref().and_then(|f| f.hybrid()).is_some());
        if has_hybrid {
            for _ in 0..larql_compute_metal::lowering::stack::StackScratch::HYBRID_BUFFERS {
                scratch.push(gpu.lowering_scratch(hidden));
            }
            let zero = vec![0.0f32; hidden];
            scratch.push(
                gpu.lowering_upload(&zero)
                    .ok_or_else(|| VindexError::Parse("zero buffer upload failed".into()))?,
            );
        }

        // One inverse-frequency table per distinct rotary policy in the
        // plan — keyed on (theta, yarn-or-plain), so a YaRN layer's ramped
        // frequencies and a plain layer's `theta^(-2i/d)` never collide on
        // theta alone. The table matches the interpreter's exactly: plain
        // rope from `rope_rotate`, YaRN from `kernels::yarn_frequencies`.
        let mut inv_freq: HashMap<u64, DeviceBuffer> = HashMap::new();
        for layer in &plan.layers {
            let a = layer.attention.softmax().unwrap_or_else(|| {
                panic!(
                    "layer {} is not softmax; the lowering refused this plan in `new`",
                    layer.layer
                )
            });
            let key = rope_table_key(&a.position, a.head_dim);
            if let Some(key) = key {
                inv_freq.entry(key).or_insert_with(|| {
                    let table = rope_inv_freq_table(&a.position, a.head_dim);
                    gpu.lowering_upload(&table).expect("inv_freq upload")
                });
            }
        }

        // Residency bootstrap. Without it the driver's wired-page
        // collector un-wires weights that sit idle between submissions,
        // and a decode walking ~15 GB per token pays a re-wire on every
        // touch — measured at 10x on a large f16 working set. One command
        // buffer referencing everything re-wires it at memcpy speed, and
        // steps fast enough thereafter keep themselves wired.
        //
        // The slices are the same allocations `lowering_weight` cached on,
        // so this wires the buffers the stack will actually bind.
        let mut streams: Vec<&[u8]> = Vec::with_capacity(keep.len() * 2);
        for w in keep.iter() {
            match w {
                LoadedWeight::Nvfp4 { packed, scales, .. } => {
                    streams.push(packed.as_slice());
                    streams.push(scales.as_slice());
                }
                LoadedWeight::Mxfp4 { packed, scales } => {
                    streams.push(packed.as_slice());
                    streams.push(scales.as_slice());
                }
                LoadedWeight::F16(b) => streams.push(b.as_slice()),
                _ => {}
            }
        }
        let wiring = std::time::Instant::now();
        gpu.wire_resident(&streams);
        eprintln!(
            "wired {} weight streams in {:.1} s",
            streams.len(),
            wiring.elapsed().as_secs_f64()
        );

        // The embedding table as a device buffer over the same host
        // floats — a row lookup on the GPU reads only the sampled row's
        // 4·hidden bytes, so residency does not change. Refused when the
        // plan judges a weightless embedding norm (host f64 semantics).
        let device_embed = (plan.embedding.as_ref().is_some_and(|e| e.norm.is_none())).then(|| {
            // SAFETY: an f32 slice viewed as bytes, same length; the Vec
            // lives in `Self` for the session, outliving the buffer's use.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    embed_table.as_ptr() as *const u8,
                    std::mem::size_of_val(embed_table.as_slice()),
                )
            };
            gpu.lowering_weight(bytes)
        });

        // Argmax scratch sized for the head's vocabulary. The control arm
        // (`LARQL_LOWERED_HOST_ARGMAX=1`) keeps the argmax on the host so
        // the device kernel can be A/B'd under one power state.
        let host_argmax = std::env::var_os(HOST_ARGMAX_ENV).is_some();
        let argmax = (vocab > 0 && !host_argmax).then(|| {
            use larql_compute_metal::lowering::head::argmax_partials;
            let parts = argmax_partials(vocab);
            [
                gpu.lowering_scratch(parts),
                gpu.lowering_scratch(parts),
                gpu.lowering_scratch(1),
                gpu.lowering_scratch(1),
            ]
        });
        Ok(Self {
            gpu,
            plan,
            hidden,
            embed_table,
            layers,
            final_norm,
            head,
            head_multiplier,
            head_softcap,
            vocab,
            scratch,
            inv_freq,
            position: 0,
            ablate: Ablation::from_env(),
            ledger: None,
            verify_ledger: None,
            last_gpu_ms: 0.0,
            last_encode_ms: 0.0,
            prepared: None,
            max_positions,
            argmax,
            device_embed,
            last_device_id: None,
            decode_chain: false,
            submissions: 0,
            scratch_widths,
            verify: None,
            splitk,
            splitk_widths,
        })
    }
}
