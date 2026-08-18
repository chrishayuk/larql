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

/// The served quant format for the plan's expert storage format, or
/// `None` for a format the descriptor MoE path does not serve.
fn expert_quant_format(format: larql_models::ExpertFormat) -> Option<larql_compute::QuantFormat> {
    match format {
        larql_models::ExpertFormat::PackedMxfp4 => Some(larql_compute::QuantFormat::MXFP4),
        larql_models::ExpertFormat::PackedBF16 | larql_models::ExpertFormat::PerExpert => None,
    }
}

/// Per-expert byte slices into a packed bank: expert `e` occupies
/// `[e*per .. (e+1)*per]` of the bank's logical bytes.
fn expert_slices(bank: &AlignedBytes, per: usize, experts: usize) -> Vec<&[u8]> {
    let all = &bank.as_slice()[..per * experts];
    (0..experts).map(|e| &all[e * per..(e + 1) * per]).collect()
}

impl RoutedLayer {
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
    if let Some(op) = layer.ffn.routed() {
        return Ok(FfnResident::Routed(Box::new(build_routed(
            gpu, store, layer, op,
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
    use larql_vindex::format::vindex3::opplan::exec::weights::AlignedBytes;
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
    let fused_row_layout = fused_row_layout(op.gate_up_layout.ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {}: routed FFN carries no gate_up layout",
            layer.layer
        ))
    })?);
    let routing_policy = routing_policy(op.router_kind);
    let hidden = op.router.shape.get(1).copied().unwrap_or(0);
    let experts = op.experts;
    let inter = op.expert_intermediate_size;

    // Packed blocks and scales into aligned, registered banks.
    let aligned = |operand: &larql_vindex::format::vindex3::opplan::OperandRef| -> Result<AlignedBytes, VindexError> {
        let raw = store.load_raw(operand)?;
        Ok(AlignedBytes::from_bytes(&raw.bytes))
    };
    let gate_up_blocks = aligned(&op.gate_up.weights)?;
    let gate_up_scales = aligned(op.gate_up.scales.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {}: routed gate_up carries no scales",
            layer.layer
        ))
    })?)?;
    let down_blocks = aligned(&op.down.weights)?;
    let down_scales = aligned(op.down.scales.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {}: routed down carries no scales",
            layer.layer
        ))
    })?)?;
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
    let gate_up_bias = f32_or_empty(op.gate_up.bias.as_ref())?;
    let down_bias = f32_or_empty(op.down.bias.as_ref())?;
    let pre_ffn_norm = store.load(&layer.pre_ffn_norm.weight)?;
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
        use larql_compute::{
            MoeExpertScales, MoeFusedRowLayout, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
        };
        let moe = larql_compute::MoeLayerWeights {
            experts_gate_up: expert_slices(&gate_up_blocks, gu_expert_bytes, experts),
            experts_down: expert_slices(&down_blocks, dn_expert_bytes, experts),
            routing_policy: MoeRoutingPolicy::top_k_then_softmax(),
            weight_layout: MoeWeightLayout::unpadded(),
            expert_scales: MoeExpertScales::Paired {
                gate_up: expert_slices(&gate_up_scales, gu_scale_bytes, experts),
                down: expert_slices(&down_scales, dn_scale_bytes, experts),
            },
            fused_row_layout: MoeFusedRowLayout::Interleaved,
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
        eps: layer.pre_ffn_norm.eps as f32,
    })
}

/// The dense FFN op of a layer the lowering has already admitted (routed
/// layers are refused in `new`, so this only fails on a plan that changed
/// under us).
fn dense_ffn(layer: &LayerPlan) -> Result<&FfnOp, VindexError> {
    layer.ffn.dense().ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {} carries a routed FFN the lowering does not execute (A-9.4)",
            layer.layer
        ))
    })
}
