//! Lowering a routed FFN: the expert bank into page-aligned,
//! region-registered buffers and the `MoeLayerWeights` the served
//! descriptor MoE path consumes — routing, layout and format all from
//! the plan's `RoutedFfnOp`, never a model name.

use larql_compute_metal::MetalBackend;
use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormats;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::weights::{AlignedBytes, LoadedWeight};
use larql_vindex::format::vindex3::opplan::{FfnOp, LayerPlan};

use super::resident::resident_matrix;
use super::DeviceMatrix;

/// A layer's resident FFN: dense gate/up/down matrices, or a routed
/// expert bank resolved to registered regions plus the served MoE
/// scratch and descriptor table.
pub(super) enum FfnResident {
    Dense {
        gate: DeviceMatrix,
        up: DeviceMatrix,
        down: DeviceMatrix,
    },
    Routed(Box<RoutedLayer>),
    /// Gemma 4: both branches plus the router conditioning, the branch
    /// norms and the layer scalar (see `HybridFfnLowering`).
    Hybrid(Box<HybridResident>),
}

impl FfnResident {
    /// `(dense matrix bytes, active expert bytes)` one token reads
    /// through this layer's FFN.
    pub(super) fn bytes_per_token(&self) -> (usize, usize) {
        match self {
            FfnResident::Dense { gate, up, down } => (gate.bytes() + up.bytes() + down.bytes(), 0),
            FfnResident::Routed(r) => (0, r.active_expert_bytes()),
            FfnResident::Hybrid(h) => (
                h.gate.bytes() + h.up.bytes() + h.down.bytes(),
                h.routed.active_expert_bytes(),
            ),
        }
    }
}

/// A hybrid layer's resident operands beyond the two branches.
pub(super) struct HybridResident {
    pub(super) gate: DeviceMatrix,
    pub(super) up: DeviceMatrix,
    pub(super) down: DeviceMatrix,
    pub(super) routed: RoutedLayer,
    /// `router.scale · hidden^-0.5` as an RMS-norm weight.
    pub(super) router_conditioning: super::DeviceBuffer,
    pub(super) per_expert_scale: super::DeviceBuffer,
    pub(super) pre_experts_norm: super::DeviceBuffer,
    pub(super) post_dense_norm: super::DeviceBuffer,
    pub(super) post_experts_norm: super::DeviceBuffer,
    pub(super) branch_norm_eps: f32,
    pub(super) branch_norm_weight_offset: f32,
}

/// One routed layer's expert bank and MoE machinery, held for the
/// session: the packed expert bytes in page-aligned, region-registered
/// buffers (bound zero-copy, never copied per token), the f32 router and
/// bias operands, the per-layer `MoeScratch`, and the descriptor table.
pub(super) struct RoutedLayer {
    gate_up_blocks: AlignedBytes,
    gate_up_scales: AlignedBytes,
    down_blocks: AlignedBytes,
    down_scales: AlignedBytes,
    router_proj: Vec<f32>,
    router_bias: Vec<f32>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
    pre_ffn_norm: Vec<f32>,
    gu_expert_bytes: usize,
    gu_scale_bytes: usize,
    dn_expert_bytes: usize,
    dn_scale_bytes: usize,
    experts: usize,
    top_k: usize,
    inter: usize,
    gate_rule: larql_compute::MoeGateRule,
    routing_policy: larql_compute::MoeRoutingPolicy,
    fused_row_layout: larql_compute::MoeFusedRowLayout,
    expert_qformat: larql_compute::QuantFormat,
    pub(super) table: std::sync::Arc<larql_compute_metal::moe_descriptor::MoeExpertDescriptorTable>,
    pub(super) scratch: larql_compute_metal::MoeScratch,
    pub(super) eps: f32,
}

/// The served routing policy for the plan's judged router kind — a
/// mapping, not a model-name lookup: the routed op carries the kind and
/// this turns it into the compute-layer policy.
fn routing_policy(kind: larql_models::MoeRouterKind) -> larql_compute::MoeRoutingPolicy {
    use larql_compute::MoeRoutingPolicy;
    match kind {
        larql_models::MoeRouterKind::TopKSoftmax => MoeRoutingPolicy::top_k_softmax(),
        larql_models::MoeRouterKind::TopKThenSoftmax => MoeRoutingPolicy::top_k_then_softmax(),
        larql_models::MoeRouterKind::Gemma4Hybrid => MoeRoutingPolicy::gemma4_hybrid(),
        // Represented, not executable — see
        // `larql_compute::pipeline_layer::moe_build::moe_routing_policy`.
        // Every policy here normalises across experts in a way sigmoid
        // does not, so substituting one produces plausible, wrong
        // expert weights.
        larql_models::MoeRouterKind::Sigmoid => {
            unimplemented!("sigmoid expert routing is represented but not executable")
        }
    }
}

/// The served fused-row layout for the plan's declared gate/up layout.
fn fused_row_layout(layout: larql_models::GateUpLayout) -> larql_compute::MoeFusedRowLayout {
    use larql_compute::MoeFusedRowLayout;
    match layout {
        larql_models::GateUpLayout::Interleaved => MoeFusedRowLayout::Interleaved,
        larql_models::GateUpLayout::ContiguousHalves => MoeFusedRowLayout::ContiguousHalves,
    }
}

/// The served quant format the descriptor MoE path serves this bank in:
/// native MXFP4 as stored, or a packed BF16 bank QUANTISED TO MXFP4 at
/// load (Gemma 4 — the descriptor kernels read Q6_K and MXFP4 only, so a
/// bf16 bank is a representation choice made here, priced against the
/// f32 interpreter like every other lowered representation). `None` for
/// a format with no path.
fn expert_quant_format(format: larql_models::ExpertFormat) -> Option<larql_compute::QuantFormat> {
    match format {
        larql_models::ExpertFormat::PackedMxfp4 | larql_models::ExpertFormat::PackedBF16 => {
            Some(larql_compute::QuantFormat::MXFP4)
        }
        larql_models::ExpertFormat::PerExpert => None,
    }
}

/// A packed expert projection as the descriptor path binds it: every
/// expert's MXFP4 blocks back to back, and every expert's e8m0 scales
/// back to back. Native MXFP4 is read verbatim; BF16 is widened and
/// quantised per expert with the interpreter's own quantiser, so the
/// lowered bank is byte-identical to what the interpreter's MXFP4 device
/// arm binds.
fn packed_bank(
    store: &OperandStore,
    projection: &larql_vindex::format::vindex3::opplan::PackedProjection,
    format: larql_models::ExpertFormat,
    experts: usize,
    rows: usize,
    k: usize,
    layer: usize,
    what: &str,
) -> Result<(AlignedBytes, AlignedBytes), VindexError> {
    use larql_vindex::format::vindex3::opplan::exec::weights::quantize_mxfp4;
    match format {
        larql_models::ExpertFormat::PackedMxfp4 => {
            let blocks = store.load_raw(&projection.weights)?;
            let scales = store.load_raw(projection.scales.as_ref().ok_or_else(|| {
                VindexError::Parse(format!("layer {layer}: routed {what} carries no scales"))
            })?)?;
            Ok((
                AlignedBytes::from_bytes(&blocks.bytes),
                AlignedBytes::from_bytes(&scales.bytes),
            ))
        }
        larql_models::ExpertFormat::PackedBF16 => {
            let raw = store.load_raw(&projection.weights)?;
            const BF16: &str = "BF16";
            if raw.dtype != BF16 {
                return Err(VindexError::Parse(format!(
                    "layer {layer}: {what} bank declared packed BF16 but stored as {}",
                    raw.dtype
                )));
            }
            let per_expert = rows * k;
            if raw.bytes.len() != experts * per_expert * 2 {
                return Err(VindexError::Parse(format!(
                    "layer {layer}: {what} bank holds {} bytes, not [{experts}, {rows}, {k}] bf16",
                    raw.bytes.len()
                )));
            }
            let mut packed = Vec::new();
            let mut scales = Vec::new();
            for e in 0..experts {
                let bytes = &raw.bytes[e * per_expert * 2..(e + 1) * per_expert * 2];
                let values: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                    .collect();
                match quantize_mxfp4(&values, rows, k, what)? {
                    LoadedWeight::Mxfp4 {
                        packed: p,
                        scales: sc,
                    } => {
                        packed.extend_from_slice(&p.as_slice()[..p.logical_len()]);
                        scales.extend_from_slice(&sc.as_slice()[..sc.logical_len()]);
                    }
                    _ => unreachable!("quantize_mxfp4 yields an MXFP4 weight"),
                }
            }
            Ok((
                AlignedBytes::from_bytes(&packed),
                AlignedBytes::from_bytes(&scales),
            ))
        }
        larql_models::ExpertFormat::PerExpert => Err(VindexError::Parse(format!(
            "layer {layer}: per-expert tensors have no descriptor path"
        ))),
    }
}

/// Per-expert byte slices into a packed bank: expert `e` occupies
/// `[e*per .. (e+1)*per]` of the bank's logical bytes.
fn expert_slices(bank: &AlignedBytes, per: usize, experts: usize) -> Vec<&[u8]> {
    let all = &bank.as_slice()[..per * experts];
    (0..experts).map(|e| &all[e * per..(e + 1) * per]).collect()
}

impl RoutedLayer {
    /// Expert bytes one token reads: `top_k` experts' gate/up and down
    /// blocks plus their scales. The router itself is f32 and small.
    pub(super) fn active_expert_bytes(&self) -> usize {
        self.top_k
            * (self.gu_expert_bytes
                + self.gu_scale_bytes
                + self.dn_expert_bytes
                + self.dn_scale_bytes)
    }

    /// The `MoeLayerWeights` view the served descriptor path consumes,
    /// assembled from a `RoutedFfnOp` — per-expert slices into the
    /// registered banks, router/bias from f32 storage, GPT-OSS routing
    /// and gate semantics from the plan. Rebuilt per step (borrows are
    /// cheap; no bytes move).
    pub(super) fn moe(&self) -> larql_compute::MoeLayerWeights<'_> {
        use larql_compute::{MoeExpertScales, MoeWeightLayout};
        larql_compute::MoeLayerWeights {
            experts_gate_up: expert_slices(
                &self.gate_up_blocks,
                self.gu_expert_bytes,
                self.experts,
            ),
            experts_down: expert_slices(&self.down_blocks, self.dn_expert_bytes, self.experts),
            routing_policy: self.routing_policy,
            weight_layout: MoeWeightLayout::unpadded(),
            expert_scales: MoeExpertScales::Paired {
                gate_up: expert_slices(&self.gate_up_scales, self.gu_scale_bytes, self.experts),
                down: expert_slices(&self.down_scales, self.dn_scale_bytes, self.experts),
            },
            fused_row_layout: self.fused_row_layout,
            expert_data_format: self.expert_qformat,
            router_proj: &self.router_proj,
            router_bias: &self.router_bias,
            experts_gate_up_bias: &self.gate_up_bias,
            experts_down_bias: &self.down_bias,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &self.pre_ffn_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: self.experts,
            top_k: self.top_k,
            intermediate_size: self.inter,
            gate_rule: self.gate_rule,
        }
    }
}

/// Build a layer's resident FFN: dense matrices, or the routed expert
/// bank loaded into registered regions with its MoE machinery.
pub(super) fn build_ffn(
    gpu: &MetalBackend,
    store: &OperandStore,
    layer: &LayerPlan,
    formats: WeightFormats,
    keep: &mut Vec<LoadedWeight>,
) -> Result<FfnResident, VindexError> {
    if let Some(op) = layer.ffn.as_ref().and_then(|f| f.routed()) {
        if op.router_kind == larql_models::MoeRouterKind::Gemma4Hybrid {
            return Err(VindexError::Parse(format!(
                "layer {}: a pure routed FFN with the Gemma 4 router kind has no lowering arm \
                 that runs its router conditioning (only the hybrid arm does); refusing",
                layer.layer
            )));
        }
        return Ok(FfnResident::Routed(Box::new(build_routed(
            gpu, store, layer, op,
        )?)));
    }
    if let Some(op) = layer.ffn.as_ref().and_then(|f| f.hybrid()) {
        return Ok(FfnResident::Hybrid(Box::new(build_hybrid(
            gpu, store, layer, op, formats, keep,
        )?)));
    }
    let dense = dense_ffn(layer)?;
    Ok(FfnResident::Dense {
        gate: resident_matrix(
            gpu,
            store,
            dense
                .gate
                .as_ref()
                .ok_or_else(|| VindexError::Parse("lowering requires a gated FFN".into()))?,
            formats.ffn,
            keep,
        )?,
        up: resident_matrix(gpu, store, &dense.up, formats.ffn, keep)?,
        down: resident_matrix(gpu, store, &dense.down, formats.ffn, keep)?,
    })
}

/// Load a routed layer's expert bank into page-aligned, region-registered
/// buffers and build its MoE scratch + descriptor table. The expert bytes
/// are bound zero-copy through the same registered-region path the served
/// `--routed-from` run uses — never copied per token.
fn build_routed(
    gpu: &MetalBackend,
    store: &OperandStore,
    layer: &LayerPlan,
    op: &larql_vindex::format::vindex3::opplan::RoutedFfnOp,
) -> Result<RoutedLayer, VindexError> {
    // Every routing/layout/format fact comes from the plan's RoutedFfnOp,
    // never a model name. A storage format the descriptor path cannot
    // serve, or a fused operand with no declared row layout, refuses here
    // rather than guessing.
    let expert_qformat = expert_quant_format(op.expert_format).ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {}: the descriptor MoE path does not serve expert format {:?}",
            layer.layer, op.expert_format
        ))
    })?;
    // The `expert_qformat` refusal above already rejects
    // `ExpertFormat::PerExpert`, so this is a packed bank whenever it is
    // reached — stated again here rather than trusted, the same posture
    // `packed_bank`'s own exhaustive match takes.
    let larql_vindex::format::vindex3::opplan::ExpertBank::Packed {
        gate_up: gate_up_projection,
        down: down_projection,
    } = &op.bank
    else {
        return Err(VindexError::Parse(format!(
            "layer {}: routed FFN op carries a per-expert (unfused) bank; the descriptor MoE \
             path has no lowering arm for it",
            layer.layer
        )));
    };
    let fused_row_layout = fused_row_layout(op.gate_up_layout.ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {}: routed FFN carries no gate_up layout",
            layer.layer
        ))
    })?);
    // The policy the DESCRIPTOR path is asked to run. Gemma 4's kind
    // carries router conditioning, renormalisation, a per-expert scale
    // and a post-expert norm — none of which the descriptor encode does:
    // the hybrid lowering performs them itself around the experts, so the
    // view it hands the experts states the plain select-and-combine
    // contract (no post-expert norm — the combine asserts that) rather
    // than the served `gemma4_hybrid` policy. A pure routed layer with
    // that kind has no arm that runs its conditioning: refused in
    // `build_ffn`.
    let routing_policy = match op.router_kind {
        larql_models::MoeRouterKind::Gemma4Hybrid => {
            larql_compute::MoeRoutingPolicy::top_k_then_softmax()
        }
        kind => routing_policy(kind),
    };
    let hidden = op.router.shape.get(1).copied().unwrap_or(0);
    let experts = op.experts;
    let inter = op.expert_intermediate_size;

    // Packed blocks and scales into aligned, registered banks — native
    // MXFP4 verbatim, packed BF16 quantised per expert.
    const FUSED: usize = larql_models::quant::mxfp4::FUSED_HALVES;
    let (gate_up_blocks, gate_up_scales) = packed_bank(
        store,
        gate_up_projection,
        op.expert_format,
        experts,
        FUSED * inter,
        hidden,
        layer.layer,
        "gate_up",
    )?;
    let (down_blocks, down_scales) = packed_bank(
        store,
        down_projection,
        op.expert_format,
        experts,
        hidden,
        inter,
        layer.layer,
        "down",
    )?;
    for (bank, what) in [
        (&gate_up_blocks, "gate_up blocks"),
        (&gate_up_scales, "gate_up scales"),
        (&down_blocks, "down blocks"),
        (&down_scales, "down scales"),
    ] {
        if !gpu.lowering_register_region(bank.as_slice()) {
            return Err(VindexError::Parse(format!(
                "layer {}: could not register the routed {what} region (not page-aligned)",
                layer.layer
            )));
        }
    }
    let gu_expert_bytes = gate_up_blocks.logical_len() / experts;
    let gu_scale_bytes = gate_up_scales.logical_len() / experts;
    let dn_expert_bytes = down_blocks.logical_len() / experts;
    let dn_scale_bytes = down_scales.logical_len() / experts;

    let f32_or_empty =
        |o: Option<&larql_vindex::format::vindex3::opplan::OperandRef>| -> Result<Vec<f32>, VindexError> {
            match o {
                Some(op) => store.load(op),
                None => Ok(Vec::new()),
            }
        };
    let router_proj = store.load(&op.router)?;
    let router_bias = f32_or_empty(op.router_bias.as_ref())?;
    let gate_up_bias = f32_or_empty(gate_up_projection.bias.as_ref())?;
    let down_bias = f32_or_empty(down_projection.bias.as_ref())?;
    let pre_ffn_op = layer.pre_ffn_norm.as_ref().ok_or_else(|| {
        VindexError::Parse("layer carries no pre-FFN norm (mixer-only)".to_string())
    })?;
    let pre_ffn_norm = store.load(&pre_ffn_op.weight)?;
    let gate_rule = larql_compute::MoeGateRule::from_arch(op.gate_policy, op.activation);

    let scratch = larql_compute_metal::MoeScratch::new_public_with_format(
        gpu,
        op.top_k,
        hidden,
        inter,
        expert_qformat,
        hidden,
    );
    // Build the descriptor table from a temporary `MoeLayerWeights`
    // borrowing the freshly-loaded storage, before that storage moves
    // into `RoutedLayer` — the table keeps only region buffers, no borrow.
    let table = {
        use larql_compute::{MoeExpertScales, MoeRoutingPolicy, MoeWeightLayout, QuantFormat};
        let moe = larql_compute::MoeLayerWeights {
            experts_gate_up: expert_slices(&gate_up_blocks, gu_expert_bytes, experts),
            experts_down: expert_slices(&down_blocks, dn_expert_bytes, experts),
            routing_policy: MoeRoutingPolicy::top_k_then_softmax(),
            weight_layout: MoeWeightLayout::unpadded(),
            expert_scales: MoeExpertScales::Paired {
                gate_up: expert_slices(&gate_up_scales, gu_scale_bytes, experts),
                down: expert_slices(&down_scales, dn_scale_bytes, experts),
            },
            fused_row_layout,
            expert_data_format: QuantFormat::MXFP4,
            router_proj: &router_proj,
            router_bias: &router_bias,
            experts_gate_up_bias: &gate_up_bias,
            experts_down_bias: &down_bias,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_ffn_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: experts,
            top_k: op.top_k,
            intermediate_size: inter,
            gate_rule,
        };
        if !gpu.lowering_moe_supported(&moe, &scratch) {
            return Err(VindexError::Parse(format!(
                "layer {}: the descriptor MoE path does not support this routed layer \
                 (format/policy/geometry) — refusing before encode",
                layer.layer
            )));
        }
        gpu.lowering_moe_descriptor(layer.layer, &moe, inter, hidden)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "layer {}: expert operands did not resolve inside their registered regions",
                    layer.layer
                ))
            })?
    };
    Ok(RoutedLayer {
        gate_up_blocks,
        gate_up_scales,
        down_blocks,
        down_scales,
        router_proj,
        router_bias,
        gate_up_bias,
        down_bias,
        pre_ffn_norm,
        gu_expert_bytes,
        gu_scale_bytes,
        dn_expert_bytes,
        dn_scale_bytes,
        experts,
        top_k: op.top_k,
        inter,
        gate_rule,
        routing_policy,
        fused_row_layout,
        expert_qformat,
        table,
        scratch,
        eps: pre_ffn_op.eps as f32,
    })
}

/// Build a hybrid layer: the dense matrices resident in the FFN format,
/// the routed bank through `build_routed` (whose descriptor table and
/// scratch the lowering drives directly), and the conditioning /
/// branch-norm vectors resident as f32.
fn build_hybrid(
    gpu: &MetalBackend,
    store: &OperandStore,
    layer: &LayerPlan,
    op: &larql_vindex::format::vindex3::opplan::HybridFfnOp,
    formats: WeightFormats,
    keep: &mut Vec<LoadedWeight>,
) -> Result<HybridResident, VindexError> {
    let routed = build_routed(gpu, store, layer, &op.routed)?;
    let hidden = op.routed.router.shape.get(1).copied().unwrap_or(0);
    let missing = |what: &str| {
        VindexError::Parse(format!(
            "layer {}: hybrid FFN's router carries no {what}; the plan must",
            layer.layer
        ))
    };
    let router_scale = store.load(
        op.routed
            .router_scale
            .as_ref()
            .ok_or_else(|| missing("router scale"))?,
    )?;
    let per_expert_scale = store.load(
        op.routed
            .router_per_expert_scale
            .as_ref()
            .ok_or_else(|| missing("per-expert scale"))?,
    )?;
    // HF: rms_no_weight(r) · scale · hidden^-0.5 — folded into one
    // weighted norm.
    let root_hidden_inv = (hidden as f32).powf(-0.5);
    let conditioning: Vec<f32> = router_scale.iter().map(|s| s * root_hidden_inv).collect();
    let upload = |v: &[f32], what: &str| {
        gpu.lowering_upload(v).ok_or_else(|| {
            VindexError::Parse(format!("layer {}: {what} upload failed", layer.layer))
        })
    };
    let gate = op
        .dense
        .gate
        .as_ref()
        .ok_or_else(|| VindexError::Parse("lowering requires a gated dense FFN".into()))?;
    Ok(HybridResident {
        gate: resident_matrix(gpu, store, gate, formats.ffn, keep)?,
        up: resident_matrix(gpu, store, &op.dense.up, formats.ffn, keep)?,
        down: resident_matrix(gpu, store, &op.dense.down, formats.ffn, keep)?,
        routed,
        router_conditioning: upload(&conditioning, "router conditioning")?,
        per_expert_scale: upload(&per_expert_scale, "per-expert scale")?,
        pre_experts_norm: upload(
            &store.load(&op.pre_experts_norm.weight)?,
            "pre-experts norm",
        )?,
        post_dense_norm: upload(&store.load(&op.post_dense_norm.weight)?, "post-dense norm")?,
        post_experts_norm: upload(
            &store.load(&op.post_experts_norm.weight)?,
            "post-experts norm",
        )?,
        branch_norm_eps: op.pre_experts_norm.eps as f32,
        branch_norm_weight_offset: op.pre_experts_norm.weight_offset,
    })
}

/// The dense FFN op of a layer the lowering has already admitted (routed
/// layers are refused in `new`, so this only fails on a plan that changed
/// under us).
fn dense_ffn(layer: &LayerPlan) -> Result<&FfnOp, VindexError> {
    layer.ffn.as_ref().and_then(|f| f.dense()).ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {} carries a routed FFN the lowering does not execute (A-9.4)",
            layer.layer
        ))
    })
}
