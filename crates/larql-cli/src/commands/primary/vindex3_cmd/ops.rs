//! `larql vindex3 ops` — print the generic operation plan (G5b-1).

use larql_vindex::format::vindex3::opplan::{
    plan_component_ops, AttentionOp, GatedDeltaOp, LayerAttention, LayerPlan, NormOp,
};

use super::optional_op::scalar;
use super::OpsArgs;

pub(super) fn run_ops(args: OpsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let inspection =
        larql_vindex::format::vindex3::inspect::inspect_container(&args.container, false)?;
    let outcome = plan_component_ops(&inspection, &args.container, &args.component)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&outcome)?);
    } else if let Some(plan) = &outcome.plan {
        if let Some(embedding) = &plan.embedding {
            println!(
                "embedding: {} vocab {} scale {}",
                embedding.table.object,
                embedding.vocab_size,
                scalar(embedding.scale)
            );
        }
        match args.layer {
            Some(layer) => match plan.layers.iter().find(|l| l.layer == layer) {
                Some(layer_plan) => print_layer(&plan.component, layer_plan),
                None => return Err(format!("no layer {layer} in the plan").into()),
            },
            None => {
                for layer_plan in &plan.layers {
                    match &layer_plan.attention {
                        LayerAttention::Softmax(attention) => println!(
                            "layer {:3}: {:?}{} position {:?}  {}/{} operands accounted",
                            layer_plan.layer,
                            attention.span,
                            attention
                                .window
                                .map(|w| format!("({w})"))
                                .unwrap_or_default(),
                            attention.position,
                            layer_plan.operands_accounted,
                            layer_plan.operands_present,
                        ),
                        LayerAttention::GatedDelta(op) => println!(
                            "layer {:3}: GatedDelta({}k/{}v heads) state {} elems  \
                             {}/{} operands accounted",
                            layer_plan.layer,
                            op.num_key_heads,
                            op.num_value_heads,
                            op.state_elements(),
                            layer_plan.operands_accounted,
                            layer_plan.operands_present,
                        ),
                        LayerAttention::Kda(op) => println!(
                            "layer {:3}: KDA({} heads x {}) state {} elems  \
                             {}/{} operands accounted",
                            layer_plan.layer,
                            op.num_heads,
                            op.head_dim,
                            op.state_elements(),
                            layer_plan.operands_accounted,
                            layer_plan.operands_present,
                        ),
                        LayerAttention::Mla(op) => println!(
                            "layer {:3}: MLA({} heads, q {} / kv {} compressed) \
                             {}/{} operands accounted",
                            layer_plan.layer,
                            op.num_heads,
                            op.q_head_dim(),
                            op.compressed_kv_width(),
                            layer_plan.operands_accounted,
                            layer_plan.operands_present,
                        ),
                    }
                }
            }
        }
        if let Some(output) = &plan.output {
            println!(
                "output: {} multiplier {}{}",
                output.projection.object,
                scalar(output.multiplier),
                output
                    .softcapping
                    .map(|c| format!(" softcap {c}"))
                    .unwrap_or_default(),
            );
        }
        eprintln!(
            "plan closed: {} layer(s), every executable operand accounted",
            plan.layers.len()
        );
    }
    if outcome.closed() {
        Ok(())
    } else {
        for defect in &outcome.defects {
            println!("defect: {defect}");
        }
        Err(format!(
            "operand closure failed: {} defect(s)",
            outcome.defects.len()
        )
        .into())
    }
}

/// The object one line of `bank=…` names: the fused bank's own object for
/// `ExpertBank::Packed`, or the first of `experts` independent objects for
/// `ExpertBank::PerExpert` — the summary line names ONE object either way,
/// with the per-expert count so a reader does not mistake it for a fused
/// bank of one.
fn bank_object(bank: &larql_vindex::format::vindex3::opplan::ExpertBank) -> String {
    use larql_vindex::format::vindex3::opplan::ExpertBank;
    match bank {
        ExpertBank::Packed { gate_up, .. } => gate_up.weights.object.clone(),
        ExpertBank::PerExpert { gate, .. } => match gate.first() {
            Some(first) => format!("{} (per-expert × {})", first.object, gate.len()),
            None => "per-expert (0 experts)".to_string(),
        },
    }
}

fn print_layer(component: &str, layer: &LayerPlan) {
    println!("{component}.layer[{}]", layer.layer);
    let norm = |op: &NormOp, site: &str| {
        println!("  {:?}({site}, eps {:e})", op.kind, op.eps);
    };
    norm(&layer.pre_attention_norm, "pre_attention");
    match &layer.attention {
        LayerAttention::Softmax(op) => print_softmax(op),
        LayerAttention::GatedDelta(op) => print_gated_delta(op),
        LayerAttention::Kda(op) => print_kda(op),
        LayerAttention::Mla(op) => print_mla(op),
    }
    println!("  residual");
    if let Some(op) = &layer.post_attention_norm {
        norm(op, "post_attention");
    }
    norm(&layer.pre_ffn_norm, "pre_ffn");
    match &layer.ffn {
        larql_vindex::format::vindex3::opplan::LayerFfn::Dense(ffn) => println!(
            "  {}FFN({:?}, {})",
            if ffn.gate.is_some() { "Gated" } else { "" },
            ffn.activation,
            ffn.intermediate_size
        ),
        larql_vindex::format::vindex3::opplan::LayerFfn::Routed(ffn) => println!(
            "  RoutedFFN({} experts, top-{}, {:?}, {:?}, {}, {:?}{}) router={}/{}, bank={}",
            ffn.experts,
            ffn.top_k,
            ffn.routing_policy,
            ffn.gate_policy,
            ffn.expert_intermediate_size,
            ffn.expert_format,
            if ffn.router_bias.is_some() {
                ", router bias"
            } else {
                ""
            },
            ffn.router.object,
            ffn.router.tensor,
            bank_object(&ffn.bank),
        ),
        larql_vindex::format::vindex3::opplan::LayerFfn::Hybrid(ffn) => {
            println!(
                "  HybridFFN: dense {}FFN({:?}, {}) → post_dense_norm  +  routed({} experts, \
                 top-{}, {:?}, {:?}, {}, {:?}{}) over pre_experts_norm(residual) → \
                 post_experts_norm; router={}/{}, bank={}",
                if ffn.dense.gate.is_some() {
                    "Gated"
                } else {
                    ""
                },
                ffn.dense.activation,
                ffn.dense.intermediate_size,
                ffn.routed.experts,
                ffn.routed.top_k,
                ffn.routed.router_kind,
                ffn.routed.gate_policy,
                ffn.routed.expert_intermediate_size,
                ffn.routed.expert_format,
                if ffn.routed.router_scale.is_some() {
                    ", router scale + per-expert scale"
                } else {
                    ""
                },
                ffn.routed.router.object,
                ffn.routed.router.tensor,
                bank_object(&ffn.routed.bank),
            );
        }
    }
    if let Some(op) = &layer.post_ffn_norm {
        norm(op, "post_ffn");
    }
    println!("  residual");
    if let Some(scale) = &layer.layer_scale {
        println!("  × layer_scale {}/{}", scale.object, scale.tensor);
    }
}

/// The softmax attention section of one layer.
fn print_softmax(attention: &AttentionOp) {
    println!("  Attention");
    println!(
        "    geometry: {}q / {}kv, head_dim {}",
        attention.num_q_heads, attention.num_kv_heads, attention.head_dim
    );
    println!(
        "    query_scale {} score_scale {}",
        scalar(attention.query_scale),
        attention.score_scale
    );
    if attention.parameter_free_qk_norm.q || attention.parameter_free_qk_norm.k {
        println!(
            "    parameter_free_qk_norm q={} k={}",
            attention.parameter_free_qk_norm.q, attention.parameter_free_qk_norm.k
        );
    }
    println!(
        "    span {:?}{}",
        attention.span,
        attention
            .window
            .map(|w| format!("({w})"))
            .unwrap_or_default()
    );
    println!("    position {:?}", attention.position);
    if let Some(qk) = &attention.qk_norm {
        println!("    qk_norm {:?}", qk.scope);
    }
    for (name, operand) in [
        ("q", &attention.q),
        ("k", &attention.k),
        ("v", &attention.v),
        ("o", &attention.o),
    ] {
        println!(
            "    {name} = {}/{} {:?}",
            operand.object, operand.tensor, operand.shape
        );
    }
    if let Some(gate) = &attention.output_gate {
        println!(
            "    output_gate {:?} = {}/{}",
            gate.spec.activation, gate.projection.object, gate.projection.tensor
        );
    }
}

/// The Gated DeltaNet section of one layer.
///
/// Deliberately does NOT reuse the softmax vocabulary: there is no span,
/// no window and no KV head count to print, and the one number a reader
/// most needs — the recurrent state's size — has no softmax counterpart.
/// Kimi Delta Attention. Prints the geometry that separates it from Gated
/// DeltaNet — one head count, a gate rank, and a per-channel `dt_bias` —
/// rather than a shape a reader would have to compare by hand.
fn print_kda(op: &larql_vindex::format::vindex3::opplan::KdaOp) {
    println!("  KDA (Kimi Delta Attention)");
    println!(
        "    geometry: {} heads x {} (value width {}), conv kernel {}, gate rank {}",
        op.num_heads,
        op.head_dim,
        op.value_width(),
        op.conv_kernel,
        op.gate_rank
    );
    println!(
        "    decay clamp: {}",
        op.gate_lower_bound
            .map_or_else(|| "undeclared".to_string(), |b| format!("{b}"))
    );
    println!(
        "    state: {} elements/layer — constant in sequence length",
        op.state_elements()
    );
    for (name, operand) in [
        ("q_proj", &op.q_proj),
        ("k_proj", &op.k_proj),
        ("v_proj", &op.v_proj),
        ("q_conv1d", &op.q_conv1d),
        ("k_conv1d", &op.k_conv1d),
        ("v_conv1d", &op.v_conv1d),
        ("f_a_proj", &op.f_a_proj),
        ("f_b_proj", &op.f_b_proj),
        ("g_a_proj", &op.g_a_proj),
        ("g_b_proj", &op.g_b_proj),
        ("b_proj", &op.b_proj),
        ("a_log", &op.a_log),
        ("dt_bias", &op.dt_bias),
        ("o_norm", &op.o_norm),
        ("out_proj", &op.out_proj),
    ] {
        println!("    {name}: {}/{}", operand.object, operand.tensor);
    }
}

fn print_mla(op: &larql_vindex::format::vindex3::opplan::MlaOp) {
    println!("  MLA (Multi-Latent Attention)");
    println!(
        "    geometry: {} heads, q/k {} (nope {} + rope {}), v {}, kv_lora_rank {}",
        op.num_heads,
        op.q_head_dim(),
        op.qk_nope_head_dim,
        op.qk_rope_head_dim,
        op.v_head_dim,
        op.kv_lora_rank
    );
    println!(
        "    compressed KV cache: {} elements/position (vs {} decompressed)",
        op.compressed_kv_width(),
        op.num_heads * (op.qk_nope_head_dim + op.v_head_dim)
    );
    for (name, operand) in [
        ("q_proj", &op.q_proj),
        ("kv_a_proj", &op.kv_a_proj),
        ("kv_a_norm", &op.kv_a_norm),
        ("kv_b_proj", &op.kv_b_proj),
        ("out_proj", &op.out_proj),
    ] {
        println!("    {name}: {}/{}", operand.object, operand.tensor);
    }
}

fn print_gated_delta(op: &GatedDeltaOp) {
    println!("  GatedDeltaNet");
    println!(
        "    geometry: {}k/{}v heads, key_head_dim {}, value_head_dim {}",
        op.num_key_heads, op.num_value_heads, op.key_head_dim, op.value_head_dim
    );
    println!(
        "    conv kernel {}  qkv channels {}",
        op.conv_kernel,
        op.qkv_channels()
    );
    println!(
        "    state: {} elements/layer at {} — constant in sequence length",
        op.state_elements(),
        op.state_dtype
            .map(|d| d.declared_name())
            .unwrap_or("undeclared")
    );
    for (name, operand) in [
        ("in_proj_qkv", &op.in_proj_qkv),
        ("in_proj_a", &op.in_proj_a),
        ("in_proj_b", &op.in_proj_b),
        ("in_proj_z", &op.in_proj_z),
        ("conv1d", &op.conv1d),
        ("a_log", &op.a_log),
        ("dt_bias", &op.dt_bias),
        ("norm", &op.norm),
        ("out_proj", &op.out_proj),
    ] {
        println!(
            "    {name} = {}/{} {:?}",
            operand.object, operand.tensor, operand.shape
        );
    }
}
