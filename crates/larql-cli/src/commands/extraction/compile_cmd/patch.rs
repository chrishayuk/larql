//! Vindex-patch compilation: read .vlp patches, install one edge per Insert op.
//!
//! Trigger comes from each patch's stored gate vector; write comes from the
//! down_meta target token's embedding when present.

use std::collections::HashMap;
use std::path::PathBuf;

use ndarray::ArcArray2;

use super::detect::{decode_f32_b64, detect_ffn_pattern, ensure_cloned};
use super::edge::install_edge;
use super::save::{copy_model_config, merge_for_save, write_safetensors};
use super::CompileArgs;

/// What this mode calls each kind of op it does not install, for the refusal.
fn uncompilable_kind(op: &larql_vindex::PatchOp) -> Option<&'static str> {
    use larql_vindex::PatchOp;
    match op {
        PatchOp::Insert { .. } => None,
        PatchOp::Update { .. } => Some("update"),
        PatchOp::Delete { .. } => Some("delete"),
        PatchOp::InsertKnn { .. } => Some("insert_knn"),
        PatchOp::DeleteKnn { .. } => Some("delete_knn"),
        PatchOp::WeightDelta(_) => Some("weight_delta"),
        PatchOp::StructuredWeight(_) => Some("structured_weight"),
        PatchOp::LowRankColumnsWeight(_) => Some("low_rank_columns_weight"),
    }
}

/// Refuse a patch set holding any op this mode does not install. Patch mode writes one FFN edge per `Insert`;
/// it has no form for a tombstone, a relabel, a retrieval entry or a stored-weight edit. Compiling the inserts
/// and reporting success would silently drop the rest, so the whole set is refused and nothing is written.
fn refuse_uncompilable(ops: &[larql_vindex::PatchOp]) -> Result<(), String> {
    let mut kinds: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for kind in ops.iter().filter_map(uncompilable_kind) {
        *kinds.entry(kind).or_default() += 1;
    }
    if kinds.is_empty() {
        return Ok(());
    }
    let held: Vec<String> = kinds.iter().map(|(k, n)| format!("{n} {k}")).collect();
    Err(format!(
        "patch mode compiles `insert` ops into FFN weights and cannot compile the other ops these patches \
         hold ({}); nothing was written. Stored-weight ops (weight_delta, structured_weight, \
         low_rank_columns_weight) compile into a VINDEX3 container with APPLY PATCH and \
         COMPILE CURRENT INTO VINDEX",
        held.join(", ")
    ))
}

pub fn run(args: CompileArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex_path = args.vindex.as_ref().unwrap();
    eprintln!("LARQL AOT Compiler — patch mode");
    eprintln!("  base model: {}", args.base.display());
    eprintln!("  vindex:     {}", vindex_path.display());
    eprintln!("  output:     {}", args.output.display());

    eprintln!("\nLoading patches...");
    let patch_files: Vec<PathBuf> = if vindex_path.is_file() {
        vec![vindex_path.clone()]
    } else {
        std::fs::read_dir(vindex_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "vlp"))
            .collect()
    };

    let mut all_ops = Vec::new();
    for pf in &patch_files {
        let patch = larql_vindex::VindexPatch::load(pf)?;
        eprintln!("  patch: {} ({} ops)", pf.display(), patch.operations.len());
        all_ops.extend(patch.operations);
    }

    eprintln!("  total patch operations: {}", all_ops.len());
    if all_ops.is_empty() {
        eprintln!("  no patches found — nothing to compile");
        return Ok(());
    }
    // Before the model is loaded and long before anything is written: every op is installed, or none is.
    refuse_uncompilable(&all_ops)?;

    eprintln!("\nLoading base model...");
    let weights = larql_models::loading::load_model_dir(&args.base)?;
    let config = weights.arch.config();
    eprintln!(
        "  {} layers, hidden={}, ffn={}",
        config.num_layers, config.hidden_size, config.intermediate_size
    );

    let gate_pattern = detect_ffn_pattern(&weights.tensors, "gate");
    let up_pattern = detect_ffn_pattern(&weights.tensors, "up");
    let down_pattern = detect_ffn_pattern(&weights.tensors, "down");
    eprintln!("  gate pattern: {}", gate_pattern.replace("{}", "N"));
    eprintln!("  up pattern:   {}", up_pattern.replace("{}", "N"));
    eprintln!("  down pattern:  {}", down_pattern.replace("{}", "N"));

    eprintln!("\nCompiling patches into weights...");
    let mut modified: HashMap<String, ArcArray2<f32>> = HashMap::new();
    let mut n_compiled = 0;

    for op in &all_ops {
        let larql_vindex::PatchOp::Insert {
            layer,
            feature,
            gate_vector_b64,
            entity,
            target,
            down_meta,
            ..
        } = op
        else {
            continue;
        };

        let Some(b64) = gate_vector_b64 else {
            eprintln!(
                "  skip: insert at L{}[{}] has no gate vector",
                layer, feature
            );
            continue;
        };
        let gate_vec = decode_f32_b64(b64)?;

        let gate_key = gate_pattern.replace("{}", &layer.to_string());
        let up_key = up_pattern.replace("{}", &layer.to_string());
        let down_key = down_pattern.replace("{}", &layer.to_string());

        ensure_cloned(&mut modified, &weights.tensors, &gate_key)?;
        ensure_cloned(&mut modified, &weights.tensors, &up_key)?;
        ensure_cloned(&mut modified, &weights.tensors, &down_key)?;

        let write: Vec<f32> = match down_meta {
            Some(dm) => {
                let tid = dm.top_token_id as usize;
                if tid >= weights.embed.shape()[0] {
                    eprintln!(
                        "  skip: insert at L{}[{}] target token {} out of vocab",
                        layer, feature, tid
                    );
                    continue;
                }
                weights.embed.row(tid).to_vec()
            }
            None => {
                eprintln!(
                    "  skip: insert at L{}[{}] has no down_meta target",
                    layer, feature
                );
                continue;
            }
        };

        let stats = install_edge(
            &mut modified,
            &gate_key,
            &up_key,
            &down_key,
            *feature,
            &gate_vec,
            &write,
            args.gate_scale,
            args.alpha,
        )?;

        n_compiled += 1;
        eprintln!(
            "  compiled: L{}[{}] {} → {} (gate ‖{:.3}‖, down ‖{:.3}‖)",
            layer, feature, entity, target, stats.g_norm, stats.d_norm
        );
    }

    eprintln!("\n  {} edges compiled into weights", n_compiled);

    eprintln!("\nSaving compiled model...");
    std::fs::create_dir_all(&args.output)?;
    let merged = merge_for_save(&weights, modified);
    let output_file = args.output.join("model.safetensors");
    write_safetensors(&merged.tensors, &merged.vectors, &output_file)?;

    let file_size = std::fs::metadata(&output_file)?.len();
    eprintln!(
        "  saved: {} ({:.1} GB, {} tensors, {} vectors)",
        output_file.display(),
        file_size as f64 / 1e9,
        merged.tensors.len(),
        merged.vectors.len(),
    );

    copy_model_config(&args.base, &args.output)?;

    eprintln!("\nDone. The compiled model runs in any inference engine:");
    eprintln!(
        "  transformers: AutoModelForCausalLM.from_pretrained(\"{}\")",
        args.output.display()
    );
    eprintln!("  ollama:       convert to GGUF, then `ollama create`");
    Ok(())
}
