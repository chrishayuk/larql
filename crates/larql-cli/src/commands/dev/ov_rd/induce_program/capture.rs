use larql_inference::encode_prompt;
use larql_vindex::VectorIndex;

use super::super::basis::{build_roundtrip_bases, fit_z_pca_bases};
use super::super::input::{limit_prompts_per_stratum, load_prompts, split_prompt_records};
use super::super::metrics::{argmax, log_softmax};
use super::super::oracle_pq_fit::fit_pq_codebooks;
use super::super::oracle_pq_forward::{
    capture_attention_relation_rows, final_logits, forward_q4k_oracle_pq_head,
};
use super::super::oracle_pq_mode_d::materialize_mode_d_tables;
use super::super::program::fingerprint::codebook_fingerprint;
use super::super::static_replace::fit_static_means;
use super::super::types::{HeadId, PqConfig};
use super::args::InduceProgramArgs;
use super::context::{FitContext, MetalBackendOpt, PromptCapture};

pub fn build_fit_context(
    args: &InduceProgramArgs,
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    head: HeadId,
    config: PqConfig,
) -> Result<FitContext, Box<dyn std::error::Error>> {
    let mut all_records = load_prompts(&args.prompts, None)?;
    if args.max_per_stratum > 0 {
        all_records = limit_prompts_per_stratum(all_records, args.max_per_stratum);
    }
    let (fit_prompts, eval_prompts) =
        split_prompt_records(&all_records, args.eval_mod, args.eval_offset)?;

    eprintln!(
        "Prompts: {} fit, {} eval",
        fit_prompts.len(),
        eval_prompts.len()
    );

    let selected_heads = vec![head];
    let configs = vec![config];

    eprintln!("Fitting static bases");
    let means = fit_static_means(weights, index, tokenizer, &fit_prompts, &selected_heads)?;

    eprintln!("Building W_O-visible bases");
    let bases = build_roundtrip_bases(weights, index, &selected_heads, args.sigma_rel_cutoff)?;

    eprintln!("Fitting PCA bases");
    let pca_bases = fit_z_pca_bases(
        weights,
        index,
        tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
    )?;

    eprintln!("Fitting PQ codebooks");
    let codebooks = fit_pq_codebooks(
        weights,
        index,
        tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &configs,
        args.pq_iters,
        &[],
    )?;

    let fp = codebooks
        .get(&(head, config))
        .map(|cb| codebook_fingerprint(&cb.centroids));
    eprintln!("Codebook fingerprint: {}", fp.as_deref().unwrap_or("?"));

    eprintln!("Materializing Mode D table");
    let mut mode_d_tables = materialize_mode_d_tables(
        weights,
        index,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &codebooks,
        &[],
    )?;

    let mode_d_table = mode_d_tables
        .remove(&(head, config))
        .ok_or_else(|| format!("Mode D table missing for L{}H{}", head.layer, head.head))?;
    let basis = bases.get(&head).ok_or("W_O basis missing")?;
    let pca_basis = pca_bases.get(&head).ok_or("PCA basis missing")?;
    let head_means = means.get(&head).ok_or("Position means missing")?;
    let codebook = codebooks.get(&(head, config)).ok_or("Codebook missing")?;

    // Pre-capture oracle codes, attention rows, and baseline logits for every eval prompt.
    eprintln!(
        "Capturing oracle codes + attention for {} eval prompts",
        eval_prompts.len()
    );
    let mut captures = Vec::with_capacity(eval_prompts.len());
    for (idx, record) in eval_prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", idx + 1, eval_prompts.len(), label);

        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");

        let baseline_h =
            larql_inference::vindex::predict_kquant_hidden(weights, &token_ids, index, None);
        let baseline_logits = final_logits(weights, &baseline_h);
        let baseline_logp = log_softmax(&baseline_logits);
        let baseline_top1 = argmax(&baseline_logits);

        let (_, _, oracle_codes) = forward_q4k_oracle_pq_head(
            weights, &token_ids, index, head, basis, pca_basis, head_means, codebook, stratum,
        )?;

        let attention_rows = capture_attention_relation_rows(weights, &token_ids, index, head)?;

        captures.push(PromptCapture {
            id: label.to_string(),
            stratum: stratum.to_string(),
            token_ids,
            oracle_codes,
            attention_rows,
            baseline_logp,
            baseline_top1,
        });
    }

    let metal: MetalBackendOpt = if args.metal {
        init_metal_backend()
    } else {
        None
    };
    if metal.is_some() {
        eprintln!("Metal backend: active");
    }

    Ok(FitContext {
        head,
        group: args.group,
        config,
        mode_d_table,
        captures,
        codebook_fingerprint: fp,
        metal,
    })
}

fn init_metal_backend() -> MetalBackendOpt {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match larql_compute_metal::MetalBackend::new() {
            Some(b) => {
                eprintln!("Metal backend: initialized");
                Some(Box::new(b))
            }
            None => {
                eprintln!("Metal backend: unavailable (MetalBackend::new returned None)");
                None
            }
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        eprintln!("Metal backend: not compiled in (rebuild with --features metal on macOS)");
        None
    }
}
