//! Walk the hero container's primary-text surface and print the ledger.
//!
//! Not a CI test — it needs the 51 GB artifact. This is the production
//! canary the fixtures cannot stand in for.

use larql_vindex::format::vindex3::gguf::geometry::{semantic_digest, TargetGeometry};
use larql_vindex::format::vindex3::gguf::plan::Qwen35Lowering;
use larql_vindex::format::vindex3::gguf::walk::WalkError;
use larql_vindex::format::vindex3::gguf::walk::{inventory_from_container, walk_primary_text};
use larql_vindex::format::vindex3::inspect::inspect_container;

fn role_for(name: &str) -> Option<(String, Option<usize>)> {
    let layer = name.split('.').next().and_then(|s| s.parse::<usize>().ok());
    let r = if name.contains("linear_attn.in_proj_qkv") {
        "fused recurrent q|k|v"
    } else if name.contains("linear_attn.in_proj_z") {
        "output-gate projection"
    } else if name.contains("linear_attn.in_proj_a") {
        "decay projection"
    } else if name.contains("linear_attn.in_proj_b") {
        "write-strength projection"
    } else if name.contains("linear_attn.conv1d") {
        "causal conv over q|k|v"
    } else if name.contains("linear_attn.A_log") {
        "log decay"
    } else if name.contains("linear_attn.dt_bias") {
        "timestep bias"
    } else if name.contains("linear_attn.norm") {
        "gated norm"
    } else if name.contains("linear_attn.out_proj") {
        "output projection"
    } else if name.contains("self_attn.q_norm") {
        "attention q norm"
    } else if name.contains("self_attn.k_norm") {
        "attention k norm"
    } else if name.contains("self_attn.q_proj") {
        "query"
    } else if name.contains("self_attn.k_proj") {
        "key"
    } else if name.contains("self_attn.v_proj") {
        "value"
    } else if name.contains("self_attn.o_proj") {
        "output"
    } else if name.contains("input_layernorm") {
        "input layer norm"
    } else if name.contains("post_attention_layernorm") {
        "post-attention layer norm"
    } else if name.contains("mlp.gate_proj") {
        "ffn gate"
    } else if name.contains("mlp.up_proj") {
        "ffn up"
    } else if name.contains("mlp.down_proj") {
        "ffn down"
    } else {
        return None;
    };
    Some((r.to_string(), layer))
}

fn main() {
    let root = std::path::PathBuf::from(std::env::args().nth(1).expect("container path"));
    let inspection = inspect_container(&root, false).expect("inspect");

    let (sources, excluded) = inventory_from_container(
        &root,
        &inspection.index,
        &|object, name| {
            // Model-scope surfaces carry no layer index.
            match object {
                "target.embedding" => Some(("embedding".into(), None)),
                "target.final_norm" => Some(("final norm".into(), None)),
                "target.output_head" => Some(("output head".into(), None)),
                _ => role_for(name),
            }
        },
        &|object| object.starts_with("target."),
        // The programme's choice, spelled out: prefer the compiled pack
        // where one exists, else the canonical bytes. The real exporter
        // reads this from the precision map rather than a preference.
        &|_object, ids| {
            ids.iter()
                .find(|id| id.ends_with("@NVFP4"))
                .or_else(|| ids.first())
                .map(|s| s.to_string())
        },
    )
    .expect("read");

    let required = vec![
        ("token_embd.weight", "every model needs an embedding"),
        ("output_norm.weight", "final norm before the head"),
        ("output.weight", "graph says head_reuses_embedding = false"),
    ];
    // The expectation's authority: the graph, read off the primary-text
    // component. Not the tensors — those are the other side.
    let component = inspection
        .graph
        .primary_text_component()
        .expect("one primary-text component");
    let lowering = Qwen35Lowering::from_surface(
        component
            .execution
            .as_ref()
            .expect("the component carries its execution surface"),
        component.hidden_size,
    )
    .expect("the graph carries every fact the expectation needs");

    let (plans, ledger) = walk_primary_text(&sources, excluded, &required, &lowering);

    println!("SOURCE");
    for (obj, n) in &ledger.source_by_object {
        println!("  {obj:32} {n:>5}");
    }
    println!("  {:32} {:>5}", "primary_text", ledger.source_total);
    println!("  {:32} {:>5}", "accounted", ledger.accounted);
    println!("\nEXCLUDED");
    for e in &ledger.excluded {
        println!("  {:32} {:>5}   {}", e.object, e.count, e.reason);
    }
    println!("\nTARGET");
    println!("  {:32} {:>5}", "plans", plans.len());
    println!(
        "  {:32} {:>5}",
        "scale siblings", ledger.generated_scale_tensors
    );
    println!("  {:32} {:>5}", "errors", ledger.errors.len());
    for e in ledger.errors.iter().take(10) {
        println!("    {e:?}");
    }
    // Planner vs metadata, per tensor, on the real container. The
    // digest below proves the two selections agree with each other;
    // this proves both agree with the graph.
    println!("\nGEOMETRY");
    println!(
        "  {:32} {:>5}",
        "reconciled with graph", ledger.geometry_reconciled
    );
    let disagreements = ledger
        .errors
        .iter()
        .filter(|e| matches!(e, WalkError::Geometry(_)))
        .count();
    println!("  {:32} {:>5}", "disagreements", disagreements);
    for e in ledger
        .errors
        .iter()
        .filter_map(|e| match e {
            WalkError::Geometry(g) => Some(g),
            _ => None,
        })
        .take(5)
    {
        println!("    {e}");
    }
    // Semantic geometry only: names and dims, no encoding, no scales.
    // Both selections of one model must agree.
    let geometry: Vec<TargetGeometry> = plans
        .iter()
        .map(|p| TargetGeometry {
            name: p.target_name.clone(),
            dims: p.target_shape.clone(),
        })
        .collect();
    // The transform programme, tallied — a dropped surface shows here
    // before it shows as wrong output tokens.
    let reordered = plans.iter().filter(|p| !p.layout.is_empty()).count();
    let arithmetic = plans.iter().filter(|p| !p.value.is_empty()).count();
    println!("\nPROGRAMME");
    println!("  {:32} {:>5}", "layout-transformed", reordered);
    println!("  {:32} {:>5}", "value-transformed", arithmetic);

    println!("\nSEMANTIC");
    println!("  {:32} {:>5}", "targets", geometry.len());
    // Names and dims, no encoding and no scale siblings — so the two
    // selections of one model must produce the same value.
    println!(
        "  {:32} {}",
        "semantic-shape digest",
        semantic_digest(geometry)
    );
    println!("\nready  {}", ledger.ready());
}
