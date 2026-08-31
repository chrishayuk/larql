//! Emit a hero container's primary-text surface as a GGUF, then verify
//! the file through the independent reader before any runtime sees it.
//!
//! Everything below the walk is execution: the plans and the resolved
//! metadata carry every decision, and the emitter refuses to make any.
//! The one judgment this example itself holds is the representation
//! selection (prefer the compiled NVFP4 pack) — the same one the walk
//! canary uses, spelled once and passed to both.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use larql_models::config::PositionPolicy;
use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::gguf::emit::{emit_gguf, metadata_to_gguf, verify_emitted};
use larql_vindex::format::vindex3::gguf::metadata::qwen35_metadata;
use larql_vindex::format::vindex3::gguf::plan::Qwen35Lowering;
use larql_vindex::format::vindex3::gguf::vocab::qwen35_vocab;
use larql_vindex::format::vindex3::gguf::walk::{inventory_from_container, walk_primary_text};
use larql_vindex::format::vindex3::graph::LayerOperator;
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

/// The precision programme's one judgment, spelled once.
fn select(_object: &str, ids: &[&str]) -> Option<String> {
    ids.iter()
        .find(|id| id.ends_with("@NVFP4"))
        .or_else(|| ids.first())
        .map(|s| s.to_string())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("container path"));
    let out = PathBuf::from(args.next().expect("output .gguf path"));

    let inspection = inspect_container(&root, false).expect("inspect");
    let (sources, excluded) = inventory_from_container(
        &root,
        &inspection.index,
        &|object, name| match object {
            "target.embedding" => Some(("embedding".into(), None)),
            "target.final_norm" => Some(("final norm".into(), None)),
            "target.output_head" => Some(("output head".into(), None)),
            _ => role_for(name),
        },
        &|object| object.starts_with("target."),
        &|object, ids| select(object, ids),
    )
    .expect("read inventory");

    let component = inspection
        .graph
        .primary_text_component()
        .expect("one primary-text component");
    let surface = component
        .execution
        .as_ref()
        .expect("the component carries its execution surface");
    let lowering = Qwen35Lowering::from_surface(surface, component.hidden_size)
        .expect("the graph carries every fact the lowering needs");

    let required = vec![
        ("token_embd.weight", "every model needs an embedding"),
        ("output_norm.weight", "final norm before the head"),
        ("output.weight", "graph says head_reuses_embedding = false"),
    ];
    let (plans, ledger) = walk_primary_text(&sources, excluded, &required, &lowering);
    if !ledger.ready() {
        eprintln!(
            "walk not ready: {:?}",
            ledger.errors.iter().take(10).collect::<Vec<_>>()
        );
        std::process::exit(1);
    }
    println!(
        "walk      {} plans, {} scale siblings, geometry {}/{}",
        plans.len(),
        ledger.generated_scale_tensors,
        ledger.geometry_reconciled,
        ledger.accounted
    );

    // Metadata inputs, each from the graph's own declarations.
    let policies = component.attention.as_ref().expect("per-layer policies");
    let attending: Vec<usize> = policies
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.operator, LayerOperator::Softmax))
        .map(|(i, _)| i)
        .collect();
    let position = component.position_policy(0).expect("layer 0 position");
    let PositionPolicy::MRope {
        theta,
        rotary_fraction,
        section,
        ..
    } = position
    else {
        panic!("qwen35 metadata expects MRoPE; the graph declares {position:?}");
    };
    let sections: Vec<u32> = section.iter().map(|s| *s as u32).collect();
    let table = qwen35_metadata(
        surface,
        component.num_layers,
        component.hidden_size,
        theta,
        &sections,
        rotary_fraction,
        &attending,
        ledger.generated_scale_tensors > 0,
    )
    .expect("metadata table");

    let vocab_size = surface.head.as_ref().expect("head surface").vocab_size;
    let vocab = qwen35_vocab(&root, vocab_size).expect("tokenizer table");
    println!(
        "vocab     {} tokens + {} pad, {} merges, {} control, {} user-defined",
        vocab.tokens, vocab.padded, vocab.merges, vocab.control, vocab.user_defined
    );
    let mut metadata = metadata_to_gguf(&table);
    metadata.extend(vocab.entries);

    // The payload source: object-qualified address → (segment, span).
    let mut spans: BTreeMap<String, (PathBuf, u64, u64)> = BTreeMap::new();
    let mut by_object: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for id in inspection.index.representations.keys() {
        let object = id.split('@').next().unwrap_or(id).to_string();
        by_object.entry(object).or_default().push(id.clone());
    }
    for (object, ids) in &by_object {
        if !object.starts_with("target.") {
            continue;
        }
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let chosen = select(object, &refs).expect("selection");
        let entry = &inspection.index.representations[&chosen];
        let path = root.join(&entry.segment);
        let (header, data_start) = read_segment_header(&path).expect("segment header");
        for t in &header.tensors {
            spans.insert(
                format!("{object}/{}", t.name),
                (path.clone(), data_start + t.offset, t.len),
            );
        }
    }
    let mut open = |source: &str| -> std::io::Result<Box<dyn Read>> {
        let (path, at, len) = spans
            .get(source)
            .ok_or_else(|| std::io::Error::other(format!("no source span for `{source}`")))?;
        let mut f = std::fs::File::open(path)?;
        f.seek(SeekFrom::Start(*at))?;
        Ok(Box::new(f.take(*len)))
    };

    let t0 = std::time::Instant::now();
    let report = emit_gguf(&metadata, &plans, &mut open, &out).expect("emit");
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "emitted   {} tensors + {} scale siblings, {} metadata keys, {:.2} GB in {:.1?}",
        report.tensors,
        report.scale_siblings,
        report.metadata_keys,
        bytes as f64 / 1e9,
        t0.elapsed()
    );

    let required_names: Vec<&str> = required.iter().map(|(n, _)| *n).collect();
    match verify_emitted(&out, &metadata, &plans, &required_names) {
        Ok(r) => println!(
            "VERIFIED  {} tensors ({} NVFP4, {} scale siblings), {} metadata keys — \
             independent reader agrees with the plan exactly",
            r.tensors, r.nvfp4_tensors, r.scale_siblings, r.metadata_keys
        ),
        Err(wrong) => {
            eprintln!("VERIFY FAILED: {} mismatches", wrong.len());
            for w in wrong.iter().take(20) {
                eprintln!("  {w}");
            }
            std::process::exit(1);
        }
    }
}
