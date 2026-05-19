use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::encode_prompt;
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use std::collections::HashMap;

use super::address::{
    attention_argmax, attention_relation_key, ffn_first_feature_key, prev_ffn_feature_key,
};
use super::basis::*;
use super::gamma_address::fit_gamma_projected_address_models;
use super::input::*;
use super::metrics::*;
use super::oracle_pq_address::{
    collect_code_occurrences, fit_address_attention_cluster_group_models,
    fit_address_attention_relation_group_models, fit_address_ffn_first_feature_group_models,
    fit_address_lsh_group_models, fit_address_prev_ffn_feature_group_models,
    fit_address_probe_models, fit_address_reduced_qk_cluster_group_models,
    fit_address_supervised_group_models, fit_majority_codes_for_codebooks,
};
use super::oracle_pq_eval::evaluate_predicted_address;
use super::oracle_pq_fit::fit_pq_codebooks;
use super::oracle_pq_forward::{
    capture_attention_relation_rows, capture_ffn_first_feature_keys, capture_layer_input_hidden,
    capture_prev_ffn_feature_keys, capture_reduced_qk_attention_rows, final_logits,
    forward_q4k_oracle_pq_head, forward_q4k_oracle_pq_mode_d_head,
};
use super::oracle_pq_mode_d::{corruption_keep_values, materialize_mode_d_tables};
use super::oracle_pq_reports::OraclePqPointAccumulator;
use super::oracle_pq_stability::measure_code_stability;
use super::reports::*;
use super::static_replace::fit_static_means;
use super::types::*;

#[derive(Args)]
pub(super) struct OraclePqArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 0:6.
    #[arg(long)]
    heads: String,

    /// Comma-separated PQ configs as K:groups:bits, e.g. 128:16:4,192:24:4.
    #[arg(long)]
    configs: String,

    /// Relative singular value cutoff for retained W_O-visible directions.
    #[arg(long, default_value_t = 1e-6)]
    sigma_rel_cutoff: f64,

    /// Lloyd iterations per product-codebook group.
    #[arg(long, default_value_t = 25)]
    pq_iters: usize,

    /// Also materialize residual-space additive tables and compare Mode D injection.
    #[arg(long)]
    mode_d_check: bool,

    /// Fit and evaluate graph-native discrete address probes.
    ///
    /// The probes use only prompt metadata and token ids, not residual vectors.
    /// Requires --mode-d-check because predicted addresses are evaluated through
    /// the materialized residual-space tables.
    #[arg(long)]
    address_probes: bool,

    /// Add a mixed simple-key address probe that picks the best discrete key
    /// independently for each PQ group on the training split.
    #[arg(long)]
    address_mixed_key_probe: bool,

    /// Evaluate simple discrete keys on selected PQ groups only. Selected
    /// groups are predicted from each key; unselected groups are evaluated as
    /// either oracle-correct or majority/default.
    #[arg(long)]
    address_key_group_probe: bool,

    /// Comma-separated PQ groups for --address-key-group-probe.
    #[arg(long, default_value = "0")]
    address_key_groups: String,

    /// Optional comma-separated simple-key probe names for
    /// --address-key-group-probe. Empty evaluates all simple-key probes.
    #[arg(long, default_value = "")]
    address_key_group_probe_names: String,

    /// Evaluate selected PQ groups by replacing them with train-set majority
    /// codes while all unselected groups remain oracle-correct.
    #[arg(long)]
    address_majority_group_probe: bool,

    /// Comma-separated PQ groups for --address-majority-group-probe.
    #[arg(long, default_value = "0")]
    address_majority_groups: String,

    /// Evaluate code-level behavioral substitution for selected PQ groups.
    ///
    /// Positions whose oracle group code equals a selected from-code are
    /// substituted to each selected to-code while all other groups and
    /// positions remain oracle-correct.
    #[arg(long)]
    address_code_substitution_group_probe: bool,

    /// Comma-separated PQ groups for --address-code-substitution-group-probe.
    #[arg(long, default_value = "0")]
    address_code_substitution_groups: String,

    /// Optional comma-separated source codes. Empty means all codes.
    #[arg(long, default_value = "")]
    address_code_substitution_from_codes: String,

    /// Target codes. Use "majority" or a comma-separated list of codes.
    #[arg(long, default_value = "majority")]
    address_code_substitution_to_codes: String,

    /// Evaluate simultaneous behavioral class-collapse substitutions.
    ///
    /// Spec format:
    ///   name=6+10+13:13
    ///   name=6+10+13:13|7:10
    /// Multiple specs are separated by semicolons.
    #[arg(long)]
    address_code_class_collapse_group_probe: bool,

    /// Comma-separated PQ groups for --address-code-class-collapse-group-probe.
    #[arg(long, default_value = "0")]
    address_code_class_collapse_groups: String,

    /// Semicolon-separated class-collapse specs.
    #[arg(long, default_value = "")]
    address_code_class_collapse_specs: String,

    /// Probe position-local interactions for one prompt and one PQ group.
    ///
    /// This is a targeted diagnostic for quotient failures: selected primary
    /// and secondary source codes are changed to one target code only within
    /// the requested prompt, while all other positions/groups remain oracle.
    #[arg(long)]
    address_code_position_interaction_probe: bool,

    /// Prompt id for --address-code-position-interaction-probe.
    #[arg(long, default_value = "")]
    address_code_position_prompt_id: String,

    /// PQ group for --address-code-position-interaction-probe.
    #[arg(long, default_value_t = 0)]
    address_code_position_group: usize,

    /// Primary source codes for --address-code-position-interaction-probe.
    #[arg(long, default_value = "10")]
    address_code_position_primary_codes: String,

    /// Secondary source codes for --address-code-position-interaction-probe.
    #[arg(long, default_value = "6")]
    address_code_position_secondary_codes: String,

    /// Target code for --address-code-position-interaction-probe.
    #[arg(long, default_value_t = 13)]
    address_code_position_target_code: usize,

    /// Evaluate split-wide conditional quotient rules for one PQ group.
    ///
    /// Primary codes are mapped to the target unconditionally. Secondary codes
    /// are mapped to the target except where a built-in guard preserves the
    /// oracle code. This tests whether a quotient plus local exception guard
    /// clears the held-out gate.
    #[arg(long)]
    address_code_conditional_quotient_group_probe: bool,

    /// PQ group for --address-code-conditional-quotient-group-probe.
    #[arg(long, default_value_t = 0)]
    address_code_conditional_quotient_group: usize,

    /// Primary source codes for the conditional quotient probe.
    #[arg(long, default_value = "10")]
    address_code_conditional_quotient_primary_codes: String,

    /// Secondary source codes for the conditional quotient probe.
    #[arg(long, default_value = "6")]
    address_code_conditional_quotient_secondary_codes: String,

    /// Target code for the conditional quotient probe.
    #[arg(long, default_value_t = 13)]
    address_code_conditional_quotient_target_code: usize,

    /// Max early position guarded by early-prose conditional quotient variants.
    #[arg(long, default_value_t = 1)]
    address_code_conditional_quotient_early_position_max: usize,

    /// Conditional quotient guards to evaluate.
    ///
    /// Supported: early_prose_position, early_prose_bos_prev, prose_bos_prev.
    #[arg(
        long,
        default_value = "early_prose_position,early_prose_bos_prev,prose_bos_prev"
    )]
    address_code_conditional_quotient_guards: String,

    /// Extra source:target mappings layered on top of the conditional quotient.
    ///
    /// Spec format matches class-collapse specs. Empty adds only the base
    /// conditional quotient. Example:
    ///   code4_to13=4:13;code7_to10=7:10
    #[arg(long, default_value = "")]
    address_code_conditional_quotient_extra_specs: String,

    /// Export per-position occurrences for selected PQ group codes.
    #[arg(long)]
    address_code_occurrences: bool,

    /// Comma-separated PQ groups for --address-code-occurrences.
    #[arg(long, default_value = "0")]
    address_code_occurrence_groups: String,

    /// Optional comma-separated codes for --address-code-occurrences.
    /// Empty means all codes.
    #[arg(long, default_value = "")]
    address_code_occurrence_codes: String,

    /// Occurrence split to export: train, eval, or all.
    #[arg(long, default_value = "eval")]
    address_code_occurrence_split: String,

    /// Evaluate a hard-coded code7 fallback rule for L0H6-style probes.
    ///
    /// For selected groups, predict special code when attention argmax is BOS
    /// and stratum is not arithmetic; otherwise predict the train majority
    /// code. Unselected groups remain oracle-correct.
    #[arg(long)]
    address_code7_bos_rule_group_probe: bool,

    /// Comma-separated PQ groups for --address-code7-bos-rule-group-probe.
    #[arg(long, default_value = "0")]
    address_code7_bos_rule_groups: String,

    /// Special code used by --address-code7-bos-rule-group-probe.
    #[arg(long, default_value_t = 7)]
    address_code7_bos_rule_code: usize,

    /// Evaluate oracle upper bounds for a binary code7-vs-default address.
    ///
    /// Selected groups use the special code only where the oracle address has
    /// that code and the requested structural filter matches; all other
    /// positions use the train majority code. Unselected groups remain
    /// oracle-correct.
    #[arg(long)]
    address_code7_oracle_binary_group_probe: bool,

    /// Comma-separated PQ groups for --address-code7-oracle-binary-group-probe.
    #[arg(long, default_value = "0")]
    address_code7_oracle_binary_groups: String,

    /// Special code used by --address-code7-oracle-binary-group-probe.
    #[arg(long, default_value_t = 7)]
    address_code7_oracle_binary_code: usize,

    /// Comma-separated filters for oracle binary code7 upper bounds.
    ///
    /// Supported: all, natural_prose_bos, natural_prose_bos_or_prev.
    #[arg(
        long,
        default_value = "all,natural_prose_bos,natural_prose_bos_or_prev"
    )]
    address_code7_oracle_binary_filters: String,

    /// Evaluate how sensitive Mode D is to address corruption.
    ///
    /// This keeps a prefix of oracle PQ groups and replaces the rest with
    /// per-group majority codes learned from the training split. It estimates
    /// how many groups must be addressed correctly before predicted addressing
    /// can pass the KL gate.
    #[arg(long)]
    address_corruption_sweep: bool,

    /// Evaluate one-group-at-a-time address importance by replacing each group
    /// with its train-set majority code while all other groups remain oracle.
    #[arg(long)]
    address_group_importance: bool,

    /// Fit and evaluate fixed random-hyperplane LSH probes for selected PQ
    /// groups. The selected groups are predicted from the residual entering the
    /// target layer; other groups are evaluated both oracle-correct and
    /// majority/default.
    #[arg(long)]
    address_lsh_group_probe: bool,

    /// Comma-separated PQ groups for --address-lsh-group-probe.
    #[arg(long, default_value = "0")]
    address_lsh_groups: String,

    /// Number of LSH bits per selected group. For a 4-bit PQ group, 4 LSH bits
    /// creates 16 buckets.
    #[arg(long, default_value_t = 4)]
    address_lsh_bits: usize,

    /// Number of deterministic random-hyperplane seeds to try per selected
    /// group. The best seed is selected by train code accuracy.
    #[arg(long, default_value_t = 32)]
    address_lsh_seeds: usize,

    /// Fit and evaluate supervised binary-hyperplane address probes for
    /// selected PQ groups. The selected groups are predicted from the residual
    /// entering the target layer; other groups are evaluated both
    /// oracle-correct and majority/default.
    #[arg(long)]
    address_supervised_group_probe: bool,

    /// Comma-separated PQ groups for --address-supervised-group-probe.
    #[arg(long, default_value = "0")]
    address_supervised_groups: String,

    /// SGD epochs for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 16)]
    address_supervised_epochs: usize,

    /// SGD learning rate for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 0.05)]
    address_supervised_lr: f32,

    /// L2 weight decay for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 1e-4)]
    address_supervised_l2: f32,

    /// Fit and evaluate supervised group address probes after a diagonal
    /// affine gamma-alignment projection from the layer input toward later
    /// post-layer residual snapshots.
    #[arg(long)]
    address_gamma_projected_group_probe: bool,

    /// Comma-separated PQ groups for --address-gamma-projected-group-probe.
    #[arg(long, default_value = "0")]
    address_gamma_projected_groups: String,

    /// Comma-separated post-layer residual snapshots used as gamma-alignment
    /// targets, e.g. 20,26,29,33. The raw layer-input supervised probe is
    /// always included as gamma_raw for comparison.
    #[arg(long, default_value = "20,26,29,33")]
    address_gamma_projected_layers: String,

    /// Comma-separated random projection ranks for the gamma bridge control,
    /// e.g. 64,128. These are fixed Rademacher low-rank projections of the
    /// layer input followed by the same supervised bit probes.
    #[arg(long, default_value = "")]
    address_gamma_random_ranks: String,

    /// Comma-separated deterministic seeds for random projection ranks.
    #[arg(long, default_value = "0")]
    address_gamma_random_seeds: String,

    /// Comma-separated learned bridge ranks for the gamma bridge test. These
    /// fit a low-rank target-PCA proxy from layer input to later residual
    /// snapshots before training the same supervised group-bit probes.
    #[arg(long, default_value = "")]
    address_gamma_learned_ranks: String,

    /// SGD epochs for learned low-rank gamma bridge fitting.
    #[arg(long, default_value_t = 8)]
    address_gamma_learned_epochs: usize,

    /// Normalized LMS learning rate for learned low-rank gamma bridge fitting.
    #[arg(long, default_value_t = 0.5)]
    address_gamma_learned_lr: f32,

    /// L2 weight decay for learned low-rank gamma bridge fitting.
    #[arg(long, default_value_t = 1e-5)]
    address_gamma_learned_l2: f32,

    /// Power-iteration steps for the learned bridge target PCA basis.
    #[arg(long, default_value_t = 8)]
    address_gamma_learned_pca_iters: usize,

    /// Report train/eval PQ code distribution stability for selected groups.
    #[arg(long)]
    address_code_stability: bool,

    /// Comma-separated PQ groups for --address-code-stability.
    #[arg(long, default_value = "0")]
    address_code_stability_groups: String,

    /// Fit and evaluate selected PQ groups from previous-layer FFN top-feature
    /// keys. This is the first model-native discrete-state address probe for
    /// non-layer-0 dynamic heads.
    #[arg(long)]
    address_prev_ffn_feature_group_probe: bool,

    /// Comma-separated PQ groups for --address-prev-ffn-feature-group-probe.
    #[arg(long, default_value = "0")]
    address_prev_ffn_feature_groups: String,

    /// Number of previous-layer FFN activation features retained for feature
    /// hash keys.
    #[arg(long, default_value_t = 4)]
    address_prev_ffn_feature_top_k: usize,

    /// Fit and evaluate selected PQ groups from an FFN-first diagnostic state:
    /// run the target layer's FFN on the pre-attention residual, use top
    /// activation features as keys, but leave the real forward ordering
    /// unchanged. This tests whether computed L0 FFN features would bootstrap
    /// attention addressability under an FFN-first reorder.
    #[arg(long)]
    address_ffn_first_feature_group_probe: bool,

    /// Comma-separated PQ groups for --address-ffn-first-feature-group-probe.
    #[arg(long, default_value = "0")]
    address_ffn_first_feature_groups: String,

    /// Number of FFN-first activation features retained for feature hash keys.
    #[arg(long, default_value_t = 4)]
    address_ffn_first_feature_top_k: usize,

    /// Fit and evaluate selected PQ groups from discrete attention/relation
    /// state keys. This tests whether the dominant address is carried by QK
    /// routing structure rather than token or FFN-feature state.
    #[arg(long)]
    address_attention_relation_group_probe: bool,

    /// Comma-separated PQ groups for --address-attention-relation-group-probe.
    #[arg(long, default_value = "0")]
    address_attention_relation_groups: String,

    /// Fit and evaluate selected PQ groups from learned attention-pattern
    /// cluster IDs. This is a discrete relation-catalogue probe over fixed
    /// features derived from the full attention distribution.
    #[arg(long)]
    address_attention_cluster_group_probe: bool,

    /// Comma-separated PQ groups for --address-attention-cluster-group-probe.
    #[arg(long, default_value = "0")]
    address_attention_cluster_groups: String,

    /// Comma-separated k values for attention-pattern clustering.
    #[arg(long, default_value = "16,32")]
    address_attention_cluster_ks: String,

    /// Optional comma-separated attention-cluster probe names. Empty evaluates
    /// all cluster probe names for the selected k values.
    #[arg(long, default_value = "")]
    address_attention_cluster_probe_names: String,

    /// Fit/evaluate selected PQ groups from attention-pattern clusters where
    /// the attention distribution is recomputed from only the first r Q/K
    /// dimensions. Use rank 0 for the full-QK control.
    #[arg(long)]
    address_reduced_qk_cluster_group_probe: bool,

    /// Comma-separated PQ groups for --address-reduced-qk-cluster-group-probe.
    #[arg(long, default_value = "0")]
    address_reduced_qk_cluster_groups: String,

    /// Comma-separated QK ranks. Rank 0 means full QK; positive ranks are
    /// clamped to the layer head dimension.
    #[arg(long, default_value = "0,128,64,32,16")]
    address_reduced_qk_ranks: String,

    /// Comma-separated k values for reduced-QK attention-pattern clustering.
    #[arg(long, default_value = "16,32")]
    address_reduced_qk_cluster_ks: String,

    /// Optional comma-separated reduced-QK cluster probe names. Empty evaluates
    /// all generated names.
    #[arg(long, default_value = "")]
    address_reduced_qk_cluster_probe_names: String,

    /// Comma-separated PQ groups whose centroids are fit separately per
    /// prompt stratum. This is a codebook-layout diagnostic for cases where a
    /// single global PQ group carries a hard prose/structured tail.
    #[arg(long, default_value = "")]
    stratum_conditioned_pq_groups: String,

    /// Limit prompts for bounded oracle runs.
    #[arg(long)]
    max_prompts: Option<usize>,

    /// Keep at most N prompts per stratum after loading. Useful for balanced
    /// held-out smoke runs from a larger ordered corpus.
    #[arg(long)]
    max_per_stratum: Option<usize>,

    /// Evaluate only prompts where prompt_index % eval_mod == eval_offset.
    /// The remaining prompts are used to fit static means, PCA, and PQ.
    #[arg(long)]
    eval_mod: Option<usize>,

    /// Held-out modulo offset used with --eval-mod.
    #[arg(long, default_value_t = 0)]
    eval_offset: usize,
}

pub(super) fn run_oracle_pq(args: OraclePqArgs) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    eprintln!("Loading vindex: {}", args.index.display());
    let start = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_kquant(&args.index)?;
    index.load_interleaved_kquant(&args.index)?;
    let mut weights = load_model_weights_kquant(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;
    if weights.arch.is_hybrid_moe() {
        return Err("ov-rd oracle-pq currently supports dense FFN vindexes only".into());
    }
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );

    let selected_heads = parse_head_spec(&args.heads)?;
    if selected_heads.is_empty() {
        return Err("no heads selected for oracle PQ".into());
    }
    let configs = parse_pq_configs(&args.configs)?;
    if configs.is_empty() {
        return Err("no PQ configs selected".into());
    }
    let mut key_groups = parse_usize_list(&args.address_key_groups)?;
    key_groups.sort_unstable();
    key_groups.dedup();
    let key_group_probe_names = parse_string_list(&args.address_key_group_probe_names);
    if args.address_key_group_probe {
        if key_groups.is_empty() {
            return Err(
                "--address-key-group-probe requires at least one --address-key-groups value".into(),
            );
        }
        for config in &configs {
            for &group in &key_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-key-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut majority_groups = parse_usize_list(&args.address_majority_groups)?;
    majority_groups.sort_unstable();
    majority_groups.dedup();
    if args.address_majority_group_probe {
        if majority_groups.is_empty() {
            return Err("--address-majority-group-probe requires at least one --address-majority-groups value".into());
        }
        for config in &configs {
            for &group in &majority_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-majority-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut code_substitution_groups = parse_usize_list(&args.address_code_substitution_groups)?;
    code_substitution_groups.sort_unstable();
    code_substitution_groups.dedup();
    let mut code_substitution_from_codes =
        parse_usize_list(&args.address_code_substitution_from_codes)?;
    code_substitution_from_codes.sort_unstable();
    code_substitution_from_codes.dedup();
    let code_substitution_to_specs =
        parse_code_substitution_to_specs(&args.address_code_substitution_to_codes)?;
    if args.address_code_substitution_group_probe {
        if code_substitution_groups.is_empty() {
            return Err("--address-code-substitution-group-probe requires at least one --address-code-substitution-groups value".into());
        }
        if code_substitution_to_specs.is_empty() {
            return Err("--address-code-substitution-group-probe requires at least one --address-code-substitution-to-codes value".into());
        }
        for config in &configs {
            let levels = 1usize << config.bits_per_group;
            for &group in &code_substitution_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-code-substitution-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
            for &code in &code_substitution_from_codes {
                if code >= levels {
                    return Err(format!(
                        "--address-code-substitution-from-codes includes code {code}, but config {:?} has only {levels} levels",
                        config
                    )
                    .into());
                }
            }
            for spec in &code_substitution_to_specs {
                if let CodeSubstitutionToSpec::Code(code) = spec {
                    if *code >= levels {
                        return Err(format!(
                            "--address-code-substitution-to-codes includes code {code}, but config {:?} has only {levels} levels",
                            config
                        )
                        .into());
                    }
                }
            }
        }
    }
    let mut code_class_collapse_groups =
        parse_usize_list(&args.address_code_class_collapse_groups)?;
    code_class_collapse_groups.sort_unstable();
    code_class_collapse_groups.dedup();
    let code_class_collapse_specs =
        parse_code_class_collapse_specs(&args.address_code_class_collapse_specs)?;
    if args.address_code_class_collapse_group_probe {
        if code_class_collapse_groups.is_empty() {
            return Err("--address-code-class-collapse-group-probe requires at least one --address-code-class-collapse-groups value".into());
        }
        if code_class_collapse_specs.is_empty() {
            return Err(
                "--address-code-class-collapse-specs must include at least one spec".into(),
            );
        }
        for config in &configs {
            let levels = 1usize << config.bits_per_group;
            for &group in &code_class_collapse_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-code-class-collapse-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
            for spec in &code_class_collapse_specs {
                for mapping in &spec.mappings {
                    if mapping.target >= levels {
                        return Err(format!(
                            "class-collapse spec {:?} targets code {}, but config {:?} has only {levels} levels",
                            spec.name, mapping.target, config
                        )
                        .into());
                    }
                    for &source in &mapping.sources {
                        if source >= levels {
                            return Err(format!(
                                "class-collapse spec {:?} includes source code {source}, but config {:?} has only {levels} levels",
                                spec.name, config
                            )
                            .into());
                        }
                    }
                }
            }
        }
    }
    let mut code_position_primary_codes =
        parse_usize_list(&args.address_code_position_primary_codes)?;
    code_position_primary_codes.sort_unstable();
    code_position_primary_codes.dedup();
    let mut code_position_secondary_codes =
        parse_usize_list(&args.address_code_position_secondary_codes)?;
    code_position_secondary_codes.sort_unstable();
    code_position_secondary_codes.dedup();
    let code_position_prompt_id = args.address_code_position_prompt_id.trim().to_string();
    if args.address_code_position_interaction_probe {
        if code_position_prompt_id.is_empty() {
            return Err("--address-code-position-interaction-probe requires --address-code-position-prompt-id".into());
        }
        if code_position_primary_codes.is_empty() {
            return Err(
                "--address-code-position-primary-codes must include at least one code".into(),
            );
        }
        if code_position_secondary_codes.is_empty() {
            return Err(
                "--address-code-position-secondary-codes must include at least one code".into(),
            );
        }
        for config in &configs {
            let levels = 1usize << config.bits_per_group;
            if args.address_code_position_group >= config.groups {
                return Err(format!(
                    "--address-code-position-group is {}, but config {:?} has only {} groups",
                    args.address_code_position_group, config, config.groups
                )
                .into());
            }
            if args.address_code_position_target_code >= levels {
                return Err(format!(
                    "--address-code-position-target-code is {}, but config {:?} has only {levels} levels",
                    args.address_code_position_target_code, config
                )
                .into());
            }
            for &code in code_position_primary_codes
                .iter()
                .chain(code_position_secondary_codes.iter())
            {
                if code >= levels {
                    return Err(format!(
                        "--address-code-position primary/secondary code {code} exceeds config {:?} with {levels} levels",
                        config
                    )
                    .into());
                }
            }
        }
    }
    let mut code_conditional_quotient_primary_codes =
        parse_usize_list(&args.address_code_conditional_quotient_primary_codes)?;
    code_conditional_quotient_primary_codes.sort_unstable();
    code_conditional_quotient_primary_codes.dedup();
    let mut code_conditional_quotient_secondary_codes =
        parse_usize_list(&args.address_code_conditional_quotient_secondary_codes)?;
    code_conditional_quotient_secondary_codes.sort_unstable();
    code_conditional_quotient_secondary_codes.dedup();
    let code_conditional_quotient_guards =
        parse_conditional_quotient_guards(&args.address_code_conditional_quotient_guards)?;
    let mut code_conditional_quotient_extra_specs =
        parse_code_class_collapse_specs(&args.address_code_conditional_quotient_extra_specs)?;
    code_conditional_quotient_extra_specs.insert(
        0,
        CodeClassCollapseSpec {
            name: "base".to_string(),
            mappings: Vec::new(),
        },
    );
    if args.address_code_conditional_quotient_group_probe {
        if code_conditional_quotient_primary_codes.is_empty() {
            return Err(
                "--address-code-conditional-quotient-primary-codes must include at least one code"
                    .into(),
            );
        }
        if code_conditional_quotient_secondary_codes.is_empty() {
            return Err("--address-code-conditional-quotient-secondary-codes must include at least one code".into());
        }
        if code_conditional_quotient_guards.is_empty() {
            return Err(
                "--address-code-conditional-quotient-guards must include at least one guard".into(),
            );
        }
        for config in &configs {
            let levels = 1usize << config.bits_per_group;
            if args.address_code_conditional_quotient_group >= config.groups {
                return Err(format!(
                    "--address-code-conditional-quotient-group is {}, but config {:?} has only {} groups",
                    args.address_code_conditional_quotient_group, config, config.groups
                )
                .into());
            }
            if args.address_code_conditional_quotient_target_code >= levels {
                return Err(format!(
                    "--address-code-conditional-quotient-target-code is {}, but config {:?} has only {levels} levels",
                    args.address_code_conditional_quotient_target_code, config
                )
                .into());
            }
            for &code in code_conditional_quotient_primary_codes
                .iter()
                .chain(code_conditional_quotient_secondary_codes.iter())
            {
                if code >= levels {
                    return Err(format!(
                        "--address-code-conditional-quotient primary/secondary code {code} exceeds config {:?} with {levels} levels",
                        config
                    )
                    .into());
                }
            }
            for spec in &code_conditional_quotient_extra_specs {
                for mapping in &spec.mappings {
                    if mapping.target >= levels {
                        return Err(format!(
                            "conditional quotient extra spec {:?} targets code {}, but config {:?} has only {levels} levels",
                            spec.name, mapping.target, config
                        )
                        .into());
                    }
                    for &source in &mapping.sources {
                        if source >= levels {
                            return Err(format!(
                                "conditional quotient extra spec {:?} includes source code {source}, but config {:?} has only {levels} levels",
                                spec.name, config
                            )
                            .into());
                        }
                    }
                }
            }
        }
    }
    let mut code_occurrence_groups = parse_usize_list(&args.address_code_occurrence_groups)?;
    code_occurrence_groups.sort_unstable();
    code_occurrence_groups.dedup();
    let mut code_occurrence_codes = parse_usize_list(&args.address_code_occurrence_codes)?;
    code_occurrence_codes.sort_unstable();
    code_occurrence_codes.dedup();
    let code_occurrence_split = args
        .address_code_occurrence_split
        .trim()
        .to_ascii_lowercase();
    if args.address_code_occurrences {
        if code_occurrence_groups.is_empty() {
            return Err(
                "--address-code-occurrences requires at least one --address-code-occurrence-groups value"
                    .into(),
            );
        }
        if !matches!(code_occurrence_split.as_str(), "train" | "eval" | "all") {
            return Err("--address-code-occurrence-split must be train, eval, or all".into());
        }
        for config in &configs {
            let levels = 1usize << config.bits_per_group;
            for &group in &code_occurrence_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-code-occurrence-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
            for &code in &code_occurrence_codes {
                if code >= levels {
                    return Err(format!(
                        "--address-code-occurrence-codes includes code {code}, but config {:?} has only {levels} levels",
                        config
                    )
                    .into());
                }
            }
        }
    }
    let mut code7_bos_rule_groups = parse_usize_list(&args.address_code7_bos_rule_groups)?;
    code7_bos_rule_groups.sort_unstable();
    code7_bos_rule_groups.dedup();
    if args.address_code7_bos_rule_group_probe {
        if code7_bos_rule_groups.is_empty() {
            return Err("--address-code7-bos-rule-group-probe requires at least one --address-code7-bos-rule-groups value".into());
        }
        for config in &configs {
            let levels = 1usize << config.bits_per_group;
            for &group in &code7_bos_rule_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-code7-bos-rule-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
            if args.address_code7_bos_rule_code >= levels {
                return Err(format!(
                    "--address-code7-bos-rule-code is {}, but config {:?} has only {levels} levels",
                    args.address_code7_bos_rule_code, config
                )
                .into());
            }
        }
    }
    let mut code7_oracle_binary_groups =
        parse_usize_list(&args.address_code7_oracle_binary_groups)?;
    code7_oracle_binary_groups.sort_unstable();
    code7_oracle_binary_groups.dedup();
    let code7_oracle_binary_filters = parse_string_list(&args.address_code7_oracle_binary_filters);
    if args.address_code7_oracle_binary_group_probe {
        if code7_oracle_binary_groups.is_empty() {
            return Err("--address-code7-oracle-binary-group-probe requires at least one --address-code7-oracle-binary-groups value".into());
        }
        if code7_oracle_binary_filters.is_empty() {
            return Err(
                "--address-code7-oracle-binary-filters must include at least one filter".into(),
            );
        }
        for filter in &code7_oracle_binary_filters {
            if !matches!(
                filter.as_str(),
                "all" | "natural_prose_bos" | "natural_prose_bos_or_prev"
            ) {
                return Err(format!(
                    "unsupported --address-code7-oracle-binary-filters value {filter:?}; expected all, natural_prose_bos, or natural_prose_bos_or_prev"
                )
                .into());
            }
        }
        for config in &configs {
            let levels = 1usize << config.bits_per_group;
            for &group in &code7_oracle_binary_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-code7-oracle-binary-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
            if args.address_code7_oracle_binary_code >= levels {
                return Err(format!(
                    "--address-code7-oracle-binary-code is {}, but config {:?} has only {levels} levels",
                    args.address_code7_oracle_binary_code, config
                )
                .into());
            }
        }
    }
    let mut lsh_groups = parse_usize_list(&args.address_lsh_groups)?;
    lsh_groups.sort_unstable();
    lsh_groups.dedup();
    if args.address_lsh_group_probe {
        if lsh_groups.is_empty() {
            return Err(
                "--address-lsh-group-probe requires at least one --address-lsh-groups value".into(),
            );
        }
        if args.address_lsh_bits == 0 {
            return Err("--address-lsh-bits must be greater than zero".into());
        }
        if args.address_lsh_bits > 16 {
            return Err("--address-lsh-bits is capped at 16 for bounded diagnostics".into());
        }
        if args.address_lsh_seeds == 0 {
            return Err("--address-lsh-seeds must be greater than zero".into());
        }
        for config in &configs {
            for &group in &lsh_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-lsh-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut supervised_groups = parse_usize_list(&args.address_supervised_groups)?;
    supervised_groups.sort_unstable();
    supervised_groups.dedup();
    if args.address_supervised_group_probe {
        if supervised_groups.is_empty() {
            return Err(
                "--address-supervised-group-probe requires at least one --address-supervised-groups value".into(),
            );
        }
        if args.address_supervised_epochs == 0 {
            return Err("--address-supervised-epochs must be greater than zero".into());
        }
        if args.address_supervised_lr <= 0.0 {
            return Err("--address-supervised-lr must be greater than zero".into());
        }
        if args.address_supervised_l2 < 0.0 {
            return Err("--address-supervised-l2 must be non-negative".into());
        }
        for config in &configs {
            for &group in &supervised_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-supervised-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut gamma_projected_groups = parse_usize_list(&args.address_gamma_projected_groups)?;
    gamma_projected_groups.sort_unstable();
    gamma_projected_groups.dedup();
    let mut gamma_projected_layers = parse_usize_list(&args.address_gamma_projected_layers)?;
    gamma_projected_layers.sort_unstable();
    gamma_projected_layers.dedup();
    let mut gamma_random_ranks = parse_usize_list(&args.address_gamma_random_ranks)?;
    gamma_random_ranks.sort_unstable();
    gamma_random_ranks.dedup();
    let mut gamma_random_seeds = parse_usize_list(&args.address_gamma_random_seeds)?
        .into_iter()
        .map(|seed| seed as u64)
        .collect::<Vec<_>>();
    gamma_random_seeds.sort_unstable();
    gamma_random_seeds.dedup();
    let mut gamma_learned_ranks = parse_usize_list(&args.address_gamma_learned_ranks)?;
    gamma_learned_ranks.sort_unstable();
    gamma_learned_ranks.dedup();
    if args.address_gamma_projected_group_probe {
        if gamma_projected_groups.is_empty() {
            return Err("--address-gamma-projected-group-probe requires at least one --address-gamma-projected-groups value".into());
        }
        if gamma_projected_layers.is_empty()
            && gamma_random_ranks.is_empty()
            && gamma_learned_ranks.is_empty()
        {
            return Err("--address-gamma-projected-layers, --address-gamma-random-ranks, or --address-gamma-learned-ranks must include at least one value".into());
        }
        if !gamma_learned_ranks.is_empty() && gamma_projected_layers.is_empty() {
            return Err(
                "--address-gamma-learned-ranks requires at least one --address-gamma-projected-layers value"
                    .into(),
            );
        }
        for &layer in &gamma_projected_layers {
            if layer >= weights.num_layers {
                return Err(format!(
                    "--address-gamma-projected-layers includes layer {layer}, but the model has only {} layers",
                    weights.num_layers
                )
                .into());
            }
        }
        for head in &selected_heads {
            for &layer in &gamma_projected_layers {
                if layer < head.layer {
                    return Err(format!(
                        "--address-gamma-projected-layers includes post-L{layer}, before target L{}H{}",
                        head.layer, head.head
                    )
                    .into());
                }
            }
        }
        for &rank in &gamma_random_ranks {
            if !(1..=weights.hidden_size).contains(&rank) {
                return Err(format!(
                    "--address-gamma-random-ranks includes rank {rank}, expected 1..={}",
                    weights.hidden_size
                )
                .into());
            }
        }
        if !gamma_random_ranks.is_empty() && gamma_random_seeds.is_empty() {
            return Err(
                "--address-gamma-random-seeds must include at least one seed when random ranks are enabled"
                    .into(),
            );
        }
        for &rank in &gamma_learned_ranks {
            if !(1..=weights.hidden_size).contains(&rank) {
                return Err(format!(
                    "--address-gamma-learned-ranks includes rank {rank}, expected 1..={}",
                    weights.hidden_size
                )
                .into());
            }
        }
        if args.address_gamma_learned_epochs == 0 {
            return Err("--address-gamma-learned-epochs must be greater than zero".into());
        }
        if args.address_gamma_learned_lr <= 0.0 {
            return Err("--address-gamma-learned-lr must be greater than zero".into());
        }
        if args.address_gamma_learned_l2 < 0.0 {
            return Err("--address-gamma-learned-l2 must be non-negative".into());
        }
        if args.address_gamma_learned_pca_iters == 0 {
            return Err("--address-gamma-learned-pca-iters must be greater than zero".into());
        }
        for config in &configs {
            for &group in &gamma_projected_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-gamma-projected-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut code_stability_groups = parse_usize_list(&args.address_code_stability_groups)?;
    code_stability_groups.sort_unstable();
    code_stability_groups.dedup();
    if args.address_code_stability {
        if code_stability_groups.is_empty() {
            return Err(
                "--address-code-stability requires at least one --address-code-stability-groups value"
                    .into(),
            );
        }
        for config in &configs {
            for &group in &code_stability_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-code-stability-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut prev_ffn_feature_groups = parse_usize_list(&args.address_prev_ffn_feature_groups)?;
    prev_ffn_feature_groups.sort_unstable();
    prev_ffn_feature_groups.dedup();
    if args.address_prev_ffn_feature_group_probe {
        if prev_ffn_feature_groups.is_empty() {
            return Err("--address-prev-ffn-feature-group-probe requires at least one --address-prev-ffn-feature-groups value".into());
        }
        if args.address_prev_ffn_feature_top_k == 0 {
            return Err("--address-prev-ffn-feature-top-k must be greater than zero".into());
        }
        for head in &selected_heads {
            if head.layer == 0 {
                eprintln!(
                    "warning: L{}H{} has no previous layer; previous-FFN feature keys will be 'none'",
                    head.layer, head.head
                );
            }
        }
        for config in &configs {
            for &group in &prev_ffn_feature_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-prev-ffn-feature-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut ffn_first_feature_groups = parse_usize_list(&args.address_ffn_first_feature_groups)?;
    ffn_first_feature_groups.sort_unstable();
    ffn_first_feature_groups.dedup();
    if args.address_ffn_first_feature_group_probe {
        if ffn_first_feature_groups.is_empty() {
            return Err("--address-ffn-first-feature-group-probe requires at least one --address-ffn-first-feature-groups value".into());
        }
        if args.address_ffn_first_feature_top_k == 0 {
            return Err("--address-ffn-first-feature-top-k must be greater than zero".into());
        }
        for config in &configs {
            for &group in &ffn_first_feature_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-ffn-first-feature-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut attention_relation_groups = parse_usize_list(&args.address_attention_relation_groups)?;
    attention_relation_groups.sort_unstable();
    attention_relation_groups.dedup();
    if args.address_attention_relation_group_probe {
        if attention_relation_groups.is_empty() {
            return Err("--address-attention-relation-group-probe requires at least one --address-attention-relation-groups value".into());
        }
        for config in &configs {
            for &group in &attention_relation_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-attention-relation-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut attention_cluster_groups = parse_usize_list(&args.address_attention_cluster_groups)?;
    attention_cluster_groups.sort_unstable();
    attention_cluster_groups.dedup();
    let mut attention_cluster_ks = parse_usize_list(&args.address_attention_cluster_ks)?;
    attention_cluster_ks.sort_unstable();
    attention_cluster_ks.dedup();
    let attention_cluster_probe_names =
        parse_string_list(&args.address_attention_cluster_probe_names);
    if args.address_attention_cluster_group_probe {
        if attention_cluster_groups.is_empty() {
            return Err("--address-attention-cluster-group-probe requires at least one --address-attention-cluster-groups value".into());
        }
        if attention_cluster_ks.is_empty() {
            return Err("--address-attention-cluster-ks must include at least one k".into());
        }
        for &cluster_count in &attention_cluster_ks {
            if !(2..=128).contains(&cluster_count) {
                return Err(
                    "--address-attention-cluster-ks values must be between 2 and 128".into(),
                );
            }
        }
        for config in &configs {
            for &group in &attention_cluster_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-attention-cluster-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut reduced_qk_cluster_groups = parse_usize_list(&args.address_reduced_qk_cluster_groups)?;
    reduced_qk_cluster_groups.sort_unstable();
    reduced_qk_cluster_groups.dedup();
    let mut reduced_qk_ranks = parse_usize_list(&args.address_reduced_qk_ranks)?;
    reduced_qk_ranks.sort_unstable();
    reduced_qk_ranks.dedup();
    let mut reduced_qk_cluster_ks = parse_usize_list(&args.address_reduced_qk_cluster_ks)?;
    reduced_qk_cluster_ks.sort_unstable();
    reduced_qk_cluster_ks.dedup();
    let reduced_qk_cluster_probe_names =
        parse_string_list(&args.address_reduced_qk_cluster_probe_names);
    if args.address_reduced_qk_cluster_group_probe {
        if reduced_qk_cluster_groups.is_empty() {
            return Err("--address-reduced-qk-cluster-group-probe requires at least one --address-reduced-qk-cluster-groups value".into());
        }
        if reduced_qk_ranks.is_empty() {
            return Err("--address-reduced-qk-ranks must include at least one rank".into());
        }
        if reduced_qk_cluster_ks.is_empty() {
            return Err("--address-reduced-qk-cluster-ks must include at least one k".into());
        }
        for &cluster_count in &reduced_qk_cluster_ks {
            if !(2..=128).contains(&cluster_count) {
                return Err(
                    "--address-reduced-qk-cluster-ks values must be between 2 and 128".into(),
                );
            }
        }
        for config in &configs {
            for &group in &reduced_qk_cluster_groups {
                if group >= config.groups {
                    return Err(format!(
                        "--address-reduced-qk-cluster-groups includes group {group}, but config {:?} has only {} groups",
                        config, config.groups
                    )
                    .into());
                }
            }
        }
    }
    let mut stratum_conditioned_pq_groups = parse_usize_list(&args.stratum_conditioned_pq_groups)?;
    stratum_conditioned_pq_groups.sort_unstable();
    stratum_conditioned_pq_groups.dedup();
    for config in &configs {
        for &group in &stratum_conditioned_pq_groups {
            if group >= config.groups {
                return Err(format!(
                    "--stratum-conditioned-pq-groups includes group {group}, but config {:?} has only {} groups",
                    config, config.groups
                )
                .into());
            }
        }
    }
    let mut prompts = load_prompts(&args.prompts, args.max_prompts)?;
    if let Some(max_per_stratum) = args.max_per_stratum {
        prompts = limit_prompts_per_stratum(prompts, max_per_stratum);
    }
    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("PQ configs: {:?}", configs);
    eprintln!("Prompts: {}", prompts.len());
    let (fit_prompts, eval_prompts): (Vec<PromptRecord>, Vec<PromptRecord>) =
        if let Some(eval_mod) = args.eval_mod {
            split_prompt_records(&prompts, eval_mod, args.eval_offset)?
        } else {
            (prompts.clone(), prompts.clone())
        };
    eprintln!(
        "Oracle PQ split: fit_prompts={}, eval_prompts={}",
        fit_prompts.len(),
        eval_prompts.len()
    );

    eprintln!("Fitting position-mean static bases");
    let means = fit_static_means(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
    )?;

    eprintln!("Building W_O-visible bases");
    let bases =
        build_roundtrip_bases(&mut weights, &index, &selected_heads, args.sigma_rel_cutoff)?;

    eprintln!("Fitting empirical z-space PCA bases");
    let pca_bases = fit_z_pca_bases(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
    )?;

    eprintln!("Fitting product quantizers");
    let codebooks = fit_pq_codebooks(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &configs,
        args.pq_iters,
        &stratum_conditioned_pq_groups,
    )?;
    let mode_d_tables = if args.mode_d_check {
        eprintln!("Materializing Mode D residual-space tables");
        materialize_mode_d_tables(
            &mut weights,
            &index,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &stratum_conditioned_pq_groups,
        )?
    } else {
        HashMap::new()
    };
    let run_address_probes =
        args.address_probes || args.address_mixed_key_probe || args.address_key_group_probe;
    let address_probe_models = if run_address_probes {
        if !args.mode_d_check {
            return Err(
                "--address-probes/--address-mixed-key-probe requires --mode-d-check".into(),
            );
        }
        eprintln!("Fitting graph-native address probes");
        fit_address_probe_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            args.address_mixed_key_probe,
        )?
    } else {
        HashMap::new()
    };
    let address_lsh_models = if args.address_lsh_group_probe {
        if !args.mode_d_check {
            return Err("--address-lsh-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting LSH group address probes for groups {:?} (bits={}, seeds={})",
            lsh_groups, args.address_lsh_bits, args.address_lsh_seeds
        );
        fit_address_lsh_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &lsh_groups,
            args.address_lsh_bits,
            args.address_lsh_seeds,
        )?
    } else {
        HashMap::new()
    };
    let address_supervised_models = if args.address_supervised_group_probe {
        if !args.mode_d_check {
            return Err("--address-supervised-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting supervised group address probes for groups {:?} (epochs={}, lr={}, l2={})",
            supervised_groups,
            args.address_supervised_epochs,
            args.address_supervised_lr,
            args.address_supervised_l2
        );
        fit_address_supervised_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &supervised_groups,
            args.address_supervised_epochs,
            args.address_supervised_lr,
            args.address_supervised_l2,
        )?
    } else {
        HashMap::new()
    };
    let address_gamma_projected_models = if args.address_gamma_projected_group_probe {
        if !args.mode_d_check {
            return Err("--address-gamma-projected-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting gamma-projected supervised group address probes for groups {:?} (post_layers={:?}, random_ranks={:?}, random_seeds={:?}, learned_ranks={:?}, learned_epochs={}, learned_lr={}, learned_l2={}, learned_pca_iters={}, epochs={}, lr={}, l2={})",
            gamma_projected_groups,
            gamma_projected_layers,
            gamma_random_ranks,
            gamma_random_seeds,
            gamma_learned_ranks,
            args.address_gamma_learned_epochs,
            args.address_gamma_learned_lr,
            args.address_gamma_learned_l2,
            args.address_gamma_learned_pca_iters,
            args.address_supervised_epochs,
            args.address_supervised_lr,
            args.address_supervised_l2
        );
        fit_gamma_projected_address_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &gamma_projected_groups,
            &gamma_projected_layers,
            &gamma_random_ranks,
            &gamma_random_seeds,
            &gamma_learned_ranks,
            args.address_gamma_learned_epochs,
            args.address_gamma_learned_lr,
            args.address_gamma_learned_l2,
            args.address_gamma_learned_pca_iters,
            args.address_supervised_epochs,
            args.address_supervised_lr,
            args.address_supervised_l2,
        )?
    } else {
        HashMap::new()
    };
    let address_prev_ffn_feature_models = if args.address_prev_ffn_feature_group_probe {
        if !args.mode_d_check {
            return Err("--address-prev-ffn-feature-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting previous-FFN feature group address probes for groups {:?} (top_k={})",
            prev_ffn_feature_groups, args.address_prev_ffn_feature_top_k
        );
        fit_address_prev_ffn_feature_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &prev_ffn_feature_groups,
            args.address_prev_ffn_feature_top_k,
        )?
    } else {
        HashMap::new()
    };
    let address_ffn_first_feature_models = if args.address_ffn_first_feature_group_probe {
        if !args.mode_d_check {
            return Err("--address-ffn-first-feature-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting FFN-first feature group address probes for groups {:?} (top_k={})",
            ffn_first_feature_groups, args.address_ffn_first_feature_top_k
        );
        fit_address_ffn_first_feature_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &ffn_first_feature_groups,
            args.address_ffn_first_feature_top_k,
        )?
    } else {
        HashMap::new()
    };
    let address_attention_relation_models = if args.address_attention_relation_group_probe {
        if !args.mode_d_check {
            return Err("--address-attention-relation-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting attention-relation group address probes for groups {:?}",
            attention_relation_groups
        );
        fit_address_attention_relation_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &attention_relation_groups,
        )?
    } else {
        HashMap::new()
    };
    let address_attention_cluster_models = if args.address_attention_cluster_group_probe {
        if !args.mode_d_check {
            return Err("--address-attention-cluster-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting attention-pattern cluster group address probes for groups {:?} (k={:?})",
            attention_cluster_groups, attention_cluster_ks
        );
        fit_address_attention_cluster_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &attention_cluster_groups,
            &attention_cluster_ks,
        )?
    } else {
        HashMap::new()
    };
    let address_reduced_qk_cluster_models = if args.address_reduced_qk_cluster_group_probe {
        if !args.mode_d_check {
            return Err("--address-reduced-qk-cluster-group-probe requires --mode-d-check".into());
        }
        eprintln!(
            "Fitting reduced-QK cluster group address probes for groups {:?} (ranks={:?}, k={:?})",
            reduced_qk_cluster_groups, reduced_qk_ranks, reduced_qk_cluster_ks
        );
        fit_address_reduced_qk_cluster_group_models(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &reduced_qk_cluster_groups,
            &reduced_qk_ranks,
            &reduced_qk_cluster_ks,
        )?
    } else {
        HashMap::new()
    };
    if args.address_corruption_sweep && !args.mode_d_check {
        return Err("--address-corruption-sweep requires --mode-d-check".into());
    }
    if args.address_group_importance && !args.mode_d_check {
        return Err("--address-group-importance requires --mode-d-check".into());
    }
    if args.address_majority_group_probe && !args.mode_d_check {
        return Err("--address-majority-group-probe requires --mode-d-check".into());
    }
    if args.address_code_substitution_group_probe && !args.mode_d_check {
        return Err("--address-code-substitution-group-probe requires --mode-d-check".into());
    }
    if args.address_code_class_collapse_group_probe && !args.mode_d_check {
        return Err("--address-code-class-collapse-group-probe requires --mode-d-check".into());
    }
    if args.address_code_position_interaction_probe && !args.mode_d_check {
        return Err("--address-code-position-interaction-probe requires --mode-d-check".into());
    }
    if args.address_code_conditional_quotient_group_probe && !args.mode_d_check {
        return Err(
            "--address-code-conditional-quotient-group-probe requires --mode-d-check".into(),
        );
    }
    if args.address_code7_bos_rule_group_probe && !args.mode_d_check {
        return Err("--address-code7-bos-rule-group-probe requires --mode-d-check".into());
    }
    if args.address_code7_oracle_binary_group_probe && !args.mode_d_check {
        return Err("--address-code7-oracle-binary-group-probe requires --mode-d-check".into());
    }
    let majority_codes = if args.address_corruption_sweep
        || args.address_group_importance
        || args.address_lsh_group_probe
        || args.address_supervised_group_probe
        || args.address_gamma_projected_group_probe
        || args.address_key_group_probe
        || args.address_majority_group_probe
        || args.address_code_substitution_group_probe
        || args.address_code_class_collapse_group_probe
        || args.address_code_position_interaction_probe
        || args.address_code_conditional_quotient_group_probe
        || args.address_code7_bos_rule_group_probe
        || args.address_code7_oracle_binary_group_probe
        || args.address_prev_ffn_feature_group_probe
        || args.address_ffn_first_feature_group_probe
        || args.address_attention_relation_group_probe
        || args.address_attention_cluster_group_probe
        || args.address_reduced_qk_cluster_group_probe
    {
        eprintln!("Fitting per-group majority codes for address diagnostics");
        fit_majority_codes_for_codebooks(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
        )?
    } else {
        HashMap::new()
    };
    let code_stability = if args.address_code_stability {
        eprintln!(
            "Measuring PQ code stability for groups {:?}",
            code_stability_groups
        );
        measure_code_stability(
            &mut weights,
            &index,
            &tokenizer,
            &fit_prompts,
            &eval_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &code_stability_groups,
        )?
    } else {
        HashMap::new()
    };

    if args.address_code_occurrences {
        let occurrence_prompts = match code_occurrence_split.as_str() {
            "train" => fit_prompts.clone(),
            "eval" => eval_prompts.clone(),
            "all" => prompts.clone(),
            _ => unreachable!("validated code occurrence split"),
        };
        eprintln!(
            "Exporting code occurrences for groups {:?}, codes {:?}, split {}",
            code_occurrence_groups, code_occurrence_codes, code_occurrence_split
        );
        let occurrences = collect_code_occurrences(
            &mut weights,
            &index,
            &tokenizer,
            &occurrence_prompts,
            &selected_heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            &code_occurrence_groups,
            &code_occurrence_codes,
        )?;
        let occurrence_path = args.out.join("code_occurrences.json");
        let file = std::fs::File::create(&occurrence_path)?;
        serde_json::to_writer_pretty(file, &occurrences)?;
        eprintln!("Wrote {}", occurrence_path.display());
    }

    let mut accumulators: HashMap<(HeadId, PqConfig), OraclePqPointAccumulator> = HashMap::new();
    for head in &selected_heads {
        for &config in &configs {
            accumulators.insert((*head, config), OraclePqPointAccumulator::new());
        }
    }

    for (prompt_idx, record) in eval_prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", prompt_idx + 1, eval_prompts.len(), label);

        let token_ids = encode_prompt(&tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");

        let baseline_hidden =
            larql_inference::vindex::predict_kquant_hidden(&mut weights, &token_ids, &index, None);
        let baseline_logits = final_logits(&weights, &baseline_hidden);
        let baseline_logp = log_softmax(&baseline_logits);
        let baseline_top1 = argmax(&baseline_logits);
        let baseline_top2 = top_k_indices(&baseline_logits, 2);
        let baseline_top2_token = baseline_top2.get(1).copied().unwrap_or(baseline_top1);
        let baseline_top1_prob = token_prob(&baseline_logp, baseline_top1);
        let baseline_top2_prob = token_prob(&baseline_logp, baseline_top2_token);
        let baseline_top1_margin = baseline_top1_prob - baseline_top2_prob;

        for head in &selected_heads {
            let basis = bases.get(head).ok_or_else(|| {
                format!("missing basis for oracle PQ L{} H{}", head.layer, head.head)
            })?;
            let head_means = means.get(head).ok_or_else(|| {
                format!(
                    "missing position means for oracle PQ L{} H{}",
                    head.layer, head.head
                )
            })?;
            let pca_basis = pca_bases.get(head).ok_or_else(|| {
                format!(
                    "missing empirical PCA basis for oracle PQ L{} H{}",
                    head.layer, head.head
                )
            })?;
            for &config in &configs {
                let codebook = codebooks.get(&(*head, config)).ok_or_else(|| {
                    format!("missing PQ codebook for L{} H{}", head.layer, head.head)
                })?;
                let (pq_hidden, metrics, oracle_codes_by_position) = forward_q4k_oracle_pq_head(
                    &mut weights,
                    &token_ids,
                    &index,
                    *head,
                    basis,
                    pca_basis,
                    head_means,
                    codebook,
                    stratum,
                )?;
                let pq_logits = final_logits(&weights, &pq_hidden);
                let pq_logp = log_softmax(&pq_logits);
                let kl = kl_logp(&baseline_logp, &pq_logp);
                let pq_top1 = argmax(&pq_logits);
                let pq_top5 = top_k_indices(&pq_logits, 5);
                let pq_top2 = top_k_indices(&pq_logits, 2);
                let pq_top2_token = pq_top2.get(1).copied().unwrap_or(pq_top1);
                let pq_top1_prob = token_prob(&pq_logp, pq_top1);
                let pq_top2_prob = token_prob(&pq_logp, pq_top2_token);
                let pq_top1_margin = pq_top1_prob - pq_top2_prob;
                let pq_prob_of_baseline_top1 = token_prob(&pq_logp, baseline_top1);

                let (
                    mode_d_kl,
                    mode_d_top1,
                    mode_d_top1_agree,
                    baseline_top1_in_mode_d_top5,
                    coeff_mode_d_max_abs_logit_diff,
                ) = if args.mode_d_check {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let mode_d_hidden = forward_q4k_oracle_pq_mode_d_head(
                        &mut weights,
                        &token_ids,
                        &index,
                        *head,
                        basis,
                        pca_basis,
                        head_means,
                        codebook,
                        mode_d_table,
                        stratum,
                    )?;
                    let mode_d_logits = final_logits(&weights, &mode_d_hidden);
                    let mode_d_logp = log_softmax(&mode_d_logits);
                    let mode_d_top1 = argmax(&mode_d_logits);
                    let mode_d_top5 = top_k_indices(&mode_d_logits, 5);
                    (
                        Some(kl_logp(&baseline_logp, &mode_d_logp)),
                        Some(mode_d_top1),
                        Some(baseline_top1 == mode_d_top1),
                        Some(mode_d_top5.contains(&baseline_top1)),
                        Some(max_abs_diff(&pq_logits, &mode_d_logits)),
                    )
                } else {
                    (None, None, None, None, None)
                };

                if run_address_probes {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for address probes L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let probe_models =
                        address_probe_models.get(&(*head, config)).ok_or_else(|| {
                            format!(
                                "missing address probe models for L{} H{} {:?}",
                                head.layer, head.head, config
                            )
                        })?;
                    for probe_model in probe_models {
                        let full_probe_enabled =
                            args.address_probes || probe_model.name == "mixed_best_simple_key";
                        if full_probe_enabled {
                            let predicted_codes_by_position = (0..token_ids.len())
                                .map(|pos| probe_model.predict_codes(&token_ids, stratum, pos))
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_model.name,
                                    &probe_model.selected_group_keys,
                                    prompt_report,
                                );
                        }
                        if args.address_key_group_probe {
                            if !key_group_probe_names.is_empty()
                                && !key_group_probe_names.contains(&probe_model.name)
                            {
                                continue;
                            }
                            let group_majority =
                                majority_codes.get(&(*head, config)).ok_or_else(|| {
                                    format!(
                                        "missing majority codes for key group probe L{} H{} {:?}",
                                        head.layer, head.head, config
                                    )
                                })?;
                            for (probe_name, use_oracle_rest) in [
                                (
                                    format!(
                                        "{}_groups_{:?}_oracle_rest",
                                        probe_model.name, key_groups
                                    ),
                                    true,
                                ),
                                (
                                    format!(
                                        "{}_groups_{:?}_majority_rest",
                                        probe_model.name, key_groups
                                    ),
                                    false,
                                ),
                            ] {
                                let predicted_codes_by_position = oracle_codes_by_position
                                    .iter()
                                    .enumerate()
                                    .map(|(pos, oracle_codes)| {
                                        let mut codes = if use_oracle_rest {
                                            oracle_codes.clone()
                                        } else {
                                            group_majority.clone()
                                        };
                                        let probe_codes =
                                            probe_model.predict_codes(&token_ids, stratum, pos);
                                        for &group in &key_groups {
                                            codes[group] = probe_codes[group];
                                        }
                                        codes
                                    })
                                    .collect::<Vec<_>>();
                                let prompt_report = evaluate_predicted_address(
                                    &mut weights,
                                    &token_ids,
                                    &index,
                                    *head,
                                    mode_d_table,
                                    &predicted_codes_by_position,
                                    stratum,
                                    label,
                                    &baseline_logp,
                                    baseline_top1,
                                    &oracle_codes_by_position,
                                )?;
                                accumulators
                                    .get_mut(&(*head, config))
                                    .expect("oracle PQ accumulator missing")
                                    .add_address_probe(
                                        &probe_name,
                                        &probe_model.selected_group_keys,
                                        prompt_report,
                                    );
                            }
                        }
                    }
                }

                if args.address_majority_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for majority group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for majority group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let predicted_codes_by_position = oracle_codes_by_position
                        .iter()
                        .map(|oracle_codes| {
                            let mut codes = oracle_codes.clone();
                            for &group in &majority_groups {
                                codes[group] = group_majority[group];
                            }
                            codes
                        })
                        .collect::<Vec<_>>();
                    let prompt_report = evaluate_predicted_address(
                        &mut weights,
                        &token_ids,
                        &index,
                        *head,
                        mode_d_table,
                        &predicted_codes_by_position,
                        stratum,
                        label,
                        &baseline_logp,
                        baseline_top1,
                        &oracle_codes_by_position,
                    )?;
                    let selected_group_keys = (0..config.groups)
                        .map(|group| {
                            if majority_groups.contains(&group) {
                                "majority".to_string()
                            } else {
                                "oracle".to_string()
                            }
                        })
                        .collect::<Vec<_>>();
                    accumulators
                        .get_mut(&(*head, config))
                        .expect("oracle PQ accumulator missing")
                        .add_address_probe(
                            &format!("majority_groups_{:?}_oracle_rest", majority_groups),
                            &selected_group_keys,
                            prompt_report,
                        );
                }

                if args.address_code_substitution_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for code substitution probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for code substitution probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let levels = 1usize << config.bits_per_group;
                    let from_codes = if code_substitution_from_codes.is_empty() {
                        (0..levels).collect::<Vec<_>>()
                    } else {
                        code_substitution_from_codes.clone()
                    };
                    for &group in &code_substitution_groups {
                        for &from_code in &from_codes {
                            let source_code_present = oracle_codes_by_position
                                .iter()
                                .any(|codes| codes[group] == from_code);
                            for to_spec in &code_substitution_to_specs {
                                let to_code = match *to_spec {
                                    CodeSubstitutionToSpec::Majority => group_majority[group],
                                    CodeSubstitutionToSpec::Code(code) => code,
                                };
                                if to_code == from_code {
                                    continue;
                                }
                                let predicted_codes_by_position = oracle_codes_by_position
                                    .iter()
                                    .map(|oracle_codes| {
                                        let mut codes = oracle_codes.clone();
                                        if codes[group] == from_code {
                                            codes[group] = to_code;
                                        }
                                        codes
                                    })
                                    .collect::<Vec<_>>();
                                let prompt_report = if source_code_present {
                                    evaluate_predicted_address(
                                        &mut weights,
                                        &token_ids,
                                        &index,
                                        *head,
                                        mode_d_table,
                                        &predicted_codes_by_position,
                                        stratum,
                                        label,
                                        &baseline_logp,
                                        baseline_top1,
                                        &oracle_codes_by_position,
                                    )?
                                } else {
                                    oracle_mode_d_address_report(
                                        label,
                                        stratum,
                                        token_ids.len(),
                                        config.groups,
                                        mode_d_kl.unwrap_or(kl),
                                        mode_d_top1_agree.unwrap_or(false),
                                        baseline_top1_in_mode_d_top5.unwrap_or(false),
                                    )
                                };
                                let to_label = match *to_spec {
                                    CodeSubstitutionToSpec::Majority => {
                                        format!("majority{}", group_majority[group])
                                    }
                                    CodeSubstitutionToSpec::Code(code) => code.to_string(),
                                };
                                let selected_group_keys = (0..config.groups)
                                    .map(|candidate_group| {
                                        if candidate_group == group {
                                            format!("from{from_code}_to{to_label}")
                                        } else {
                                            "oracle".to_string()
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                accumulators
                                    .get_mut(&(*head, config))
                                    .expect("oracle PQ accumulator missing")
                                    .add_address_probe(
                                        &format!(
                                            "code_subst_g{group}_from{from_code}_to{to_label}_oracle_rest"
                                        ),
                                        &selected_group_keys,
                                        prompt_report,
                                    );
                            }
                        }
                    }
                }

                if args.address_code_class_collapse_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for code class-collapse probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    for collapse_spec in &code_class_collapse_specs {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .map(|oracle_codes| {
                                let mut codes = oracle_codes.clone();
                                for &group in &code_class_collapse_groups {
                                    for mapping in &collapse_spec.mappings {
                                        if mapping.sources.contains(&oracle_codes[group]) {
                                            codes[group] = mapping.target;
                                            break;
                                        }
                                    }
                                }
                                codes
                            })
                            .collect::<Vec<_>>();
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        let selected_group_keys = (0..config.groups)
                            .map(|group| {
                                if code_class_collapse_groups.contains(&group) {
                                    collapse_spec.mapping_label()
                                } else {
                                    "oracle".to_string()
                                }
                            })
                            .collect::<Vec<_>>();
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(
                                &format!(
                                    "code_class_collapse_{}_groups_{:?}_oracle_rest",
                                    collapse_spec.name, code_class_collapse_groups
                                ),
                                &selected_group_keys,
                                prompt_report,
                            );
                    }
                }

                if args.address_code_position_interaction_probe
                    && label == code_position_prompt_id.as_str()
                {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for code position-interaction probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group = args.address_code_position_group;
                    let target_code = args.address_code_position_target_code;
                    let primary_positions = oracle_codes_by_position
                        .iter()
                        .enumerate()
                        .filter_map(|(pos, codes)| {
                            code_position_primary_codes
                                .contains(&codes[group])
                                .then_some(pos)
                        })
                        .collect::<Vec<_>>();
                    let secondary_positions = oracle_codes_by_position
                        .iter()
                        .enumerate()
                        .filter_map(|(pos, codes)| {
                            code_position_secondary_codes
                                .contains(&codes[group])
                                .then_some(pos)
                        })
                        .collect::<Vec<_>>();

                    let mut emit_position_variant =
                        |variant_name: String,
                         mut changed_positions: Vec<usize>|
                         -> Result<(), Box<dyn std::error::Error>> {
                            changed_positions.sort_unstable();
                            changed_positions.dedup();
                            if changed_positions.is_empty() {
                                return Ok(());
                            }
                            let predicted_codes_by_position = oracle_codes_by_position
                                .iter()
                                .enumerate()
                                .map(|(pos, oracle_codes)| {
                                    let mut codes = oracle_codes.clone();
                                    if changed_positions.binary_search(&pos).is_ok() {
                                        codes[group] = target_code;
                                    }
                                    codes
                                })
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            let selected_group_keys = (0..config.groups)
                                .map(|candidate_group| {
                                    if candidate_group == group {
                                        format!(
                                            "{variant_name}_positions_{}",
                                            changed_positions
                                                .iter()
                                                .map(ToString::to_string)
                                                .collect::<Vec<_>>()
                                                .join("+")
                                        )
                                    } else {
                                        "oracle".to_string()
                                    }
                                })
                                .collect::<Vec<_>>();
                            accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(
                                &format!(
                                    "pos_interaction_g{group}_{variant_name}_to{target_code}_oracle_rest"
                                ),
                                &selected_group_keys,
                                prompt_report,
                            );
                            Ok(())
                        };

                    emit_position_variant("A0_all_primary".to_string(), primary_positions.clone())?;
                    emit_position_variant(
                        "A1_all_secondary".to_string(),
                        secondary_positions.clone(),
                    )?;
                    let mut all_primary_secondary = primary_positions.clone();
                    all_primary_secondary.extend(secondary_positions.iter().copied());
                    emit_position_variant(
                        "A2_all_primary_all_secondary".to_string(),
                        all_primary_secondary,
                    )?;
                    for (idx, &secondary_pos) in secondary_positions.iter().enumerate() {
                        let mut changed = primary_positions.clone();
                        changed.push(secondary_pos);
                        emit_position_variant(
                            format!("A{}_all_primary_secondary_pos{secondary_pos}", idx + 3),
                            changed,
                        )?;
                    }
                    let leave_one_offset = 3 + secondary_positions.len();
                    for (idx, &secondary_pos) in secondary_positions.iter().enumerate() {
                        let mut changed = primary_positions.clone();
                        changed.extend(
                            secondary_positions
                                .iter()
                                .copied()
                                .filter(|pos| *pos != secondary_pos),
                        );
                        emit_position_variant(
                            format!(
                                "A{}_all_primary_all_secondary_except_pos{secondary_pos}",
                                leave_one_offset + idx
                            ),
                            changed,
                        )?;
                    }
                    for &primary_pos in &primary_positions {
                        let mut changed = secondary_positions.clone();
                        changed.push(primary_pos);
                        emit_position_variant(
                            format!("all_secondary_primary_pos{primary_pos}"),
                            changed,
                        )?;
                    }
                    for &primary_pos in &primary_positions {
                        let mut changed = secondary_positions.clone();
                        changed.extend(
                            primary_positions
                                .iter()
                                .copied()
                                .filter(|pos| *pos != primary_pos),
                        );
                        emit_position_variant(
                            format!("all_primary_except_pos{primary_pos}_all_secondary"),
                            changed,
                        )?;
                    }
                }

                if args.address_code_conditional_quotient_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for code conditional-quotient probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group = args.address_code_conditional_quotient_group;
                    let target_code = args.address_code_conditional_quotient_target_code;
                    let early_position_max =
                        args.address_code_conditional_quotient_early_position_max;
                    let attention_rows =
                        capture_attention_relation_rows(&mut weights, &token_ids, &index, *head)?;
                    for &guard in &code_conditional_quotient_guards {
                        for extra_spec in &code_conditional_quotient_extra_specs {
                            let predicted_codes_by_position = oracle_codes_by_position
                                .iter()
                                .enumerate()
                                .map(|(pos, oracle_codes)| {
                                    let mut codes = oracle_codes.clone();
                                    let group_code = oracle_codes[group];
                                    if code_conditional_quotient_primary_codes.contains(&group_code)
                                    {
                                        codes[group] = target_code;
                                    } else if code_conditional_quotient_secondary_codes
                                        .contains(&group_code)
                                        && !guard.keeps_secondary_oracle(
                                            stratum,
                                            pos,
                                            early_position_max,
                                            attention_rows
                                                .get(pos)
                                                .map(Vec::as_slice)
                                                .unwrap_or(&[]),
                                        )
                                    {
                                        codes[group] = target_code;
                                    }
                                    for mapping in &extra_spec.mappings {
                                        if mapping.sources.contains(&group_code) {
                                            codes[group] = mapping.target;
                                            break;
                                        }
                                    }
                                    codes
                                })
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            let selected_group_keys = (0..config.groups)
                                .map(|candidate_group| {
                                    if candidate_group == group {
                                        format!(
                                            "{}_primary{}_secondary{}_to{}_extra{}",
                                            guard.label(),
                                            code_conditional_quotient_primary_codes
                                                .iter()
                                                .map(ToString::to_string)
                                                .collect::<Vec<_>>()
                                                .join("+"),
                                            code_conditional_quotient_secondary_codes
                                                .iter()
                                                .map(ToString::to_string)
                                                .collect::<Vec<_>>()
                                                .join("+"),
                                            target_code,
                                            extra_spec.mapping_label_or_base()
                                        )
                                    } else {
                                        "oracle".to_string()
                                    }
                                })
                                .collect::<Vec<_>>();
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &format!(
                                        "code_conditional_quotient_g{group}_{}_extra{}_to{target_code}_oracle_rest",
                                        guard.label(),
                                        extra_spec.name
                                    ),
                                    &selected_group_keys,
                                    prompt_report,
                                );
                        }
                    }
                }

                if args.address_code7_bos_rule_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for code7 BOS rule probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for code7 BOS rule probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let attention_rows =
                        capture_attention_relation_rows(&mut weights, &token_ids, &index, *head)?;
                    let use_special_code = stratum != "arithmetic";
                    let predicted_codes_by_position = oracle_codes_by_position
                        .iter()
                        .enumerate()
                        .map(|(pos, oracle_codes)| {
                            let mut codes = oracle_codes.clone();
                            let attention_weights =
                                attention_rows.get(pos).map(Vec::as_slice).unwrap_or(&[]);
                            let predicts_special = use_special_code
                                && !attention_weights.is_empty()
                                && attention_argmax(attention_weights, pos) == 0;
                            for &group in &code7_bos_rule_groups {
                                codes[group] = if predicts_special {
                                    args.address_code7_bos_rule_code
                                } else {
                                    group_majority[group]
                                };
                            }
                            codes
                        })
                        .collect::<Vec<_>>();
                    let prompt_report = evaluate_predicted_address(
                        &mut weights,
                        &token_ids,
                        &index,
                        *head,
                        mode_d_table,
                        &predicted_codes_by_position,
                        stratum,
                        label,
                        &baseline_logp,
                        baseline_top1,
                        &oracle_codes_by_position,
                    )?;
                    let selected_group_keys = (0..config.groups)
                        .map(|group| {
                            if code7_bos_rule_groups.contains(&group) {
                                format!(
                                    "bos_non_arithmetic_to_code{}_else_majority{}",
                                    args.address_code7_bos_rule_code, group_majority[group]
                                )
                            } else {
                                "oracle".to_string()
                            }
                        })
                        .collect::<Vec<_>>();
                    accumulators
                        .get_mut(&(*head, config))
                        .expect("oracle PQ accumulator missing")
                        .add_address_probe(
                            &format!(
                                "code{}_bos_non_arithmetic_groups_{:?}_oracle_rest",
                                args.address_code7_bos_rule_code, code7_bos_rule_groups
                            ),
                            &selected_group_keys,
                            prompt_report,
                        );
                }

                if args.address_code7_oracle_binary_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for code7 oracle binary probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for code7 oracle binary probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let attention_rows =
                        capture_attention_relation_rows(&mut weights, &token_ids, &index, *head)?;
                    for filter in &code7_oracle_binary_filters {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .enumerate()
                            .map(|(pos, oracle_codes)| {
                                let mut codes = oracle_codes.clone();
                                let attention_weights =
                                    attention_rows.get(pos).map(Vec::as_slice).unwrap_or(&[]);
                                let relation_matches = match filter.as_str() {
                                    "all" => true,
                                    "natural_prose_bos" => {
                                        stratum == "natural_prose"
                                            && !attention_weights.is_empty()
                                            && attention_argmax(attention_weights, pos) == 0
                                    }
                                    "natural_prose_bos_or_prev" => {
                                        stratum == "natural_prose"
                                            && (!attention_weights.is_empty()
                                                && (attention_argmax(attention_weights, pos) == 0
                                                    || attention_argmax(attention_weights, pos)
                                                        == pos.saturating_sub(1)))
                                    }
                                    _ => unreachable!("validated oracle binary filter"),
                                };
                                for &group in &code7_oracle_binary_groups {
                                    codes[group] = if relation_matches
                                        && oracle_codes[group]
                                            == args.address_code7_oracle_binary_code
                                    {
                                        args.address_code7_oracle_binary_code
                                    } else {
                                        group_majority[group]
                                    };
                                }
                                codes
                            })
                            .collect::<Vec<_>>();
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        let selected_group_keys = (0..config.groups)
                            .map(|group| {
                                if code7_oracle_binary_groups.contains(&group) {
                                    format!(
                                        "oracle_{}_code{}_else_majority{}",
                                        filter,
                                        args.address_code7_oracle_binary_code,
                                        group_majority[group]
                                    )
                                } else {
                                    "oracle".to_string()
                                }
                            })
                            .collect::<Vec<_>>();
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(
                                &format!(
                                    "oracle_binary_{}_code{}_groups_{:?}_oracle_rest",
                                    filter,
                                    args.address_code7_oracle_binary_code,
                                    code7_oracle_binary_groups
                                ),
                                &selected_group_keys,
                                prompt_report,
                            );
                    }
                }

                if args.address_group_importance {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for address group importance L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for address group importance L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    for replaced_group in 0..config.groups {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .map(|codes| {
                                codes
                                    .iter()
                                    .enumerate()
                                    .map(|(group, &code)| {
                                        if group == replaced_group {
                                            group_majority[group]
                                        } else {
                                            code
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_group_importance(replaced_group, prompt_report);
                    }
                }

                if args.address_lsh_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for LSH group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let lsh_model = address_lsh_models.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing LSH group probe model for L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for LSH group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let layer_input =
                        capture_layer_input_hidden(&mut weights, &token_ids, &index, head.layer)?;
                    let selected_group_keys = lsh_model.selected_group_keys();
                    for (probe_name, use_oracle_rest) in [
                        (
                            format!("lsh_groups_{:?}_oracle_rest", lsh_model.groups),
                            true,
                        ),
                        (
                            format!("lsh_groups_{:?}_majority_rest", lsh_model.groups),
                            false,
                        ),
                    ] {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .enumerate()
                            .map(|(pos, oracle_codes)| {
                                let base_codes = if use_oracle_rest {
                                    oracle_codes.as_slice()
                                } else {
                                    group_majority.as_slice()
                                };
                                lsh_model.predict_selected_groups(&layer_input, pos, base_codes)
                            })
                            .collect::<Vec<_>>();
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(&probe_name, &selected_group_keys, prompt_report);
                    }
                }

                if args.address_supervised_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for supervised group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let supervised_model = address_supervised_models
                        .get(&(*head, config))
                        .ok_or_else(|| {
                            format!(
                                "missing supervised group probe model for L{} H{} {:?}",
                                head.layer, head.head, config
                            )
                        })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for supervised group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let layer_input =
                        capture_layer_input_hidden(&mut weights, &token_ids, &index, head.layer)?;
                    let selected_group_keys = supervised_model.selected_group_keys();
                    for (probe_name, use_oracle_rest) in [
                        (
                            format!(
                                "supervised_hyperplane_groups_{:?}_oracle_rest",
                                supervised_model.groups
                            ),
                            true,
                        ),
                        (
                            format!(
                                "supervised_hyperplane_groups_{:?}_majority_rest",
                                supervised_model.groups
                            ),
                            false,
                        ),
                    ] {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .enumerate()
                            .map(|(pos, oracle_codes)| {
                                let base_codes = if use_oracle_rest {
                                    oracle_codes.as_slice()
                                } else {
                                    group_majority.as_slice()
                                };
                                supervised_model.predict_selected_groups(
                                    &layer_input,
                                    pos,
                                    base_codes,
                                )
                            })
                            .collect::<Vec<_>>();
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_probe(&probe_name, &selected_group_keys, prompt_report);
                    }
                }

                if args.address_gamma_projected_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for gamma-projected group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let gamma_models = address_gamma_projected_models
                        .get(&(*head, config))
                        .ok_or_else(|| {
                            format!(
                                "missing gamma-projected group probe models for L{} H{} {:?}",
                                head.layer, head.head, config
                            )
                        })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for gamma-projected group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let layer_input =
                        capture_layer_input_hidden(&mut weights, &token_ids, &index, head.layer)?;
                    for gamma_model in gamma_models {
                        let projected_input = gamma_model.project_layer_input(&layer_input)?;
                        let selected_group_keys = gamma_model.selected_group_keys();
                        for (probe_name, use_oracle_rest) in [
                            (
                                format!(
                                    "{}_groups_{:?}_oracle_rest",
                                    gamma_model.name, gamma_projected_groups
                                ),
                                true,
                            ),
                            (
                                format!(
                                    "{}_groups_{:?}_majority_rest",
                                    gamma_model.name, gamma_projected_groups
                                ),
                                false,
                            ),
                        ] {
                            let predicted_codes_by_position = oracle_codes_by_position
                                .iter()
                                .enumerate()
                                .map(|(pos, oracle_codes)| {
                                    let base_codes = if use_oracle_rest {
                                        oracle_codes.as_slice()
                                    } else {
                                        group_majority.as_slice()
                                    };
                                    gamma_model.supervised.predict_selected_groups(
                                        &projected_input,
                                        pos,
                                        base_codes,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_name,
                                    &selected_group_keys,
                                    prompt_report,
                                );
                        }
                    }
                }

                if args.address_prev_ffn_feature_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for previous-FFN feature group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let prev_feature_models = address_prev_ffn_feature_models
                        .get(&(*head, config))
                        .ok_or_else(|| {
                            format!(
                                "missing previous-FFN feature group probe model for L{} H{} {:?}",
                                head.layer, head.head, config
                            )
                        })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for previous-FFN feature group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let prev_features_by_position = capture_prev_ffn_feature_keys(
                        &mut weights,
                        &token_ids,
                        &index,
                        head.layer,
                        args.address_prev_ffn_feature_top_k,
                    )?;
                    for probe_model in prev_feature_models {
                        let selected_group_keys = probe_model.selected_group_keys.clone();
                        for (probe_name, use_oracle_rest) in [
                            (
                                format!(
                                    "{}_groups_{:?}_oracle_rest",
                                    probe_model.name, prev_ffn_feature_groups
                                ),
                                true,
                            ),
                            (
                                format!(
                                    "{}_groups_{:?}_majority_rest",
                                    probe_model.name, prev_ffn_feature_groups
                                ),
                                false,
                            ),
                        ] {
                            let predicted_codes_by_position = oracle_codes_by_position
                                .iter()
                                .enumerate()
                                .map(|(pos, oracle_codes)| {
                                    let mut codes = if use_oracle_rest {
                                        oracle_codes.clone()
                                    } else {
                                        group_majority.clone()
                                    };
                                    let prev_features = prev_features_by_position
                                        .get(pos)
                                        .map(Vec::as_slice)
                                        .unwrap_or(&[]);
                                    let key = prev_ffn_feature_key(
                                        &probe_model.name,
                                        &token_ids,
                                        stratum,
                                        pos,
                                        prev_features,
                                    );
                                    let probe_codes = probe_model.predict_codes_from_key(&key);
                                    for &group in &prev_ffn_feature_groups {
                                        codes[group] = probe_codes[group];
                                    }
                                    codes
                                })
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_name,
                                    &selected_group_keys,
                                    prompt_report,
                                );
                        }
                    }
                }

                if args.address_ffn_first_feature_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for FFN-first feature group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let ffn_first_models = address_ffn_first_feature_models
                        .get(&(*head, config))
                        .ok_or_else(|| {
                        format!(
                            "missing FFN-first feature group probe model for L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for FFN-first feature group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let ffn_first_features_by_position = capture_ffn_first_feature_keys(
                        &mut weights,
                        &token_ids,
                        &index,
                        head.layer,
                        args.address_ffn_first_feature_top_k,
                    )?;
                    for probe_model in ffn_first_models {
                        let selected_group_keys = probe_model.selected_group_keys.clone();
                        for (probe_name, use_oracle_rest) in [
                            (
                                format!(
                                    "{}_groups_{:?}_oracle_rest",
                                    probe_model.name, ffn_first_feature_groups
                                ),
                                true,
                            ),
                            (
                                format!(
                                    "{}_groups_{:?}_majority_rest",
                                    probe_model.name, ffn_first_feature_groups
                                ),
                                false,
                            ),
                        ] {
                            let predicted_codes_by_position = oracle_codes_by_position
                                .iter()
                                .enumerate()
                                .map(|(pos, oracle_codes)| {
                                    let mut codes = if use_oracle_rest {
                                        oracle_codes.clone()
                                    } else {
                                        group_majority.clone()
                                    };
                                    let ffn_first_features = ffn_first_features_by_position
                                        .get(pos)
                                        .map(Vec::as_slice)
                                        .unwrap_or(&[]);
                                    let key = ffn_first_feature_key(
                                        &probe_model.name,
                                        &token_ids,
                                        stratum,
                                        pos,
                                        ffn_first_features,
                                    );
                                    let probe_codes = probe_model.predict_codes_from_key(&key);
                                    for &group in &ffn_first_feature_groups {
                                        codes[group] = probe_codes[group];
                                    }
                                    codes
                                })
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_name,
                                    &selected_group_keys,
                                    prompt_report,
                                );
                        }
                    }
                }

                if args.address_attention_relation_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for attention-relation group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let relation_models = address_attention_relation_models
                        .get(&(*head, config))
                        .ok_or_else(|| {
                        format!(
                            "missing attention-relation group probe model for L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for attention-relation group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let attention_rows =
                        capture_attention_relation_rows(&mut weights, &token_ids, &index, *head)?;
                    for probe_model in relation_models {
                        let selected_group_keys = probe_model.selected_group_keys.clone();
                        for (probe_name, use_oracle_rest) in [
                            (
                                format!(
                                    "{}_groups_{:?}_oracle_rest",
                                    probe_model.name, attention_relation_groups
                                ),
                                true,
                            ),
                            (
                                format!(
                                    "{}_groups_{:?}_majority_rest",
                                    probe_model.name, attention_relation_groups
                                ),
                                false,
                            ),
                        ] {
                            let predicted_codes_by_position = oracle_codes_by_position
                                .iter()
                                .enumerate()
                                .map(|(pos, oracle_codes)| {
                                    let mut codes = if use_oracle_rest {
                                        oracle_codes.clone()
                                    } else {
                                        group_majority.clone()
                                    };
                                    let attention_weights =
                                        attention_rows.get(pos).map(Vec::as_slice).unwrap_or(&[]);
                                    let key = attention_relation_key(
                                        &probe_model.name,
                                        &token_ids,
                                        stratum,
                                        pos,
                                        attention_weights,
                                    );
                                    let probe_codes = probe_model.predict_codes_from_key(&key);
                                    for &group in &attention_relation_groups {
                                        codes[group] = probe_codes[group];
                                    }
                                    codes
                                })
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_name,
                                    &selected_group_keys,
                                    prompt_report,
                                );
                        }
                    }
                }

                if args.address_attention_cluster_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for attention-cluster group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let cluster_models = address_attention_cluster_models
                        .get(&(*head, config))
                        .ok_or_else(|| {
                            format!(
                                "missing attention-cluster group probe model for L{} H{} {:?}",
                                head.layer, head.head, config
                            )
                        })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for attention-cluster group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let attention_rows =
                        capture_attention_relation_rows(&mut weights, &token_ids, &index, *head)?;
                    for cluster_model in cluster_models {
                        if !attention_cluster_probe_names.is_empty()
                            && !attention_cluster_probe_names.contains(&cluster_model.name)
                        {
                            continue;
                        }
                        let selected_group_keys = cluster_model.selected_group_keys.clone();
                        for (probe_name, use_oracle_rest) in [
                            (
                                format!(
                                    "{}_groups_{:?}_oracle_rest",
                                    cluster_model.name, attention_cluster_groups
                                ),
                                true,
                            ),
                            (
                                format!(
                                    "{}_groups_{:?}_majority_rest",
                                    cluster_model.name, attention_cluster_groups
                                ),
                                false,
                            ),
                        ] {
                            let predicted_codes_by_position = oracle_codes_by_position
                                .iter()
                                .enumerate()
                                .map(|(pos, oracle_codes)| {
                                    let base_codes = if use_oracle_rest {
                                        oracle_codes.as_slice()
                                    } else {
                                        group_majority.as_slice()
                                    };
                                    let attention_weights =
                                        attention_rows.get(pos).map(Vec::as_slice).unwrap_or(&[]);
                                    cluster_model.predict_selected_groups(
                                        &token_ids,
                                        stratum,
                                        pos,
                                        attention_weights,
                                        base_codes,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_name,
                                    &selected_group_keys,
                                    prompt_report,
                                );
                        }
                    }
                }

                if args.address_reduced_qk_cluster_group_probe {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for reduced-QK cluster group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let cluster_models = address_reduced_qk_cluster_models
                        .get(&(*head, config))
                        .ok_or_else(|| {
                            format!(
                                "missing reduced-QK cluster group probe model for L{} H{} {:?}",
                                head.layer, head.head, config
                            )
                        })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for reduced-QK cluster group probe L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let mut rows_by_rank: HashMap<Option<usize>, Vec<Vec<f32>>> = HashMap::new();
                    for cluster_model in cluster_models {
                        if !reduced_qk_cluster_probe_names.is_empty()
                            && !reduced_qk_cluster_probe_names.contains(&cluster_model.name)
                        {
                            continue;
                        }
                        if !rows_by_rank.contains_key(&cluster_model.qk_rank) {
                            let rows = if let Some(qk_rank) = cluster_model.qk_rank {
                                capture_reduced_qk_attention_rows(
                                    &mut weights,
                                    &token_ids,
                                    &index,
                                    *head,
                                    qk_rank,
                                )?
                            } else {
                                capture_attention_relation_rows(
                                    &mut weights,
                                    &token_ids,
                                    &index,
                                    *head,
                                )?
                            };
                            rows_by_rank.insert(cluster_model.qk_rank, rows);
                        }
                        let attention_rows = rows_by_rank
                            .get(&cluster_model.qk_rank)
                            .expect("reduced-QK rows were just inserted");
                        let selected_group_keys = cluster_model.selected_group_keys.clone();
                        for (probe_name, use_oracle_rest) in [
                            (
                                format!(
                                    "{}_groups_{:?}_oracle_rest",
                                    cluster_model.name, reduced_qk_cluster_groups
                                ),
                                true,
                            ),
                            (
                                format!(
                                    "{}_groups_{:?}_majority_rest",
                                    cluster_model.name, reduced_qk_cluster_groups
                                ),
                                false,
                            ),
                        ] {
                            let predicted_codes_by_position = oracle_codes_by_position
                                .iter()
                                .enumerate()
                                .map(|(pos, oracle_codes)| {
                                    let base_codes = if use_oracle_rest {
                                        oracle_codes.as_slice()
                                    } else {
                                        group_majority.as_slice()
                                    };
                                    let attention_weights =
                                        attention_rows.get(pos).map(Vec::as_slice).unwrap_or(&[]);
                                    cluster_model.predict_selected_groups(
                                        &token_ids,
                                        stratum,
                                        pos,
                                        attention_weights,
                                        base_codes,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let prompt_report = evaluate_predicted_address(
                                &mut weights,
                                &token_ids,
                                &index,
                                *head,
                                mode_d_table,
                                &predicted_codes_by_position,
                                stratum,
                                label,
                                &baseline_logp,
                                baseline_top1,
                                &oracle_codes_by_position,
                            )?;
                            accumulators
                                .get_mut(&(*head, config))
                                .expect("oracle PQ accumulator missing")
                                .add_address_probe(
                                    &probe_name,
                                    &selected_group_keys,
                                    prompt_report,
                                );
                        }
                    }
                }

                if args.address_corruption_sweep {
                    let mode_d_table = mode_d_tables.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing Mode D table for address corruption L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let group_majority = majority_codes.get(&(*head, config)).ok_or_else(|| {
                        format!(
                            "missing majority codes for address corruption L{} H{} {:?}",
                            head.layer, head.head, config
                        )
                    })?;
                    let keep_values = corruption_keep_values(config.groups);
                    for oracle_groups_kept in keep_values {
                        let predicted_codes_by_position = oracle_codes_by_position
                            .iter()
                            .map(|codes| {
                                codes
                                    .iter()
                                    .enumerate()
                                    .map(|(group, &code)| {
                                        if group < oracle_groups_kept {
                                            code
                                        } else {
                                            group_majority[group]
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        let prompt_report = evaluate_predicted_address(
                            &mut weights,
                            &token_ids,
                            &index,
                            *head,
                            mode_d_table,
                            &predicted_codes_by_position,
                            stratum,
                            label,
                            &baseline_logp,
                            baseline_top1,
                            &oracle_codes_by_position,
                        )?;
                        accumulators
                            .get_mut(&(*head, config))
                            .expect("oracle PQ accumulator missing")
                            .add_address_corruption(oracle_groups_kept, prompt_report);
                    }
                }

                accumulators
                    .get_mut(&(*head, config))
                    .expect("oracle PQ accumulator missing")
                    .add(OraclePqPromptReport {
                        id: label.to_string(),
                        stratum: stratum.to_string(),
                        kl,
                        delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
                        baseline_top1,
                        pq_top1,
                        top1_agree: baseline_top1 == pq_top1,
                        baseline_top1_in_pq_top5: pq_top5.contains(&baseline_top1),
                        baseline_top1_prob,
                        baseline_top2: baseline_top2_token,
                        baseline_top2_prob,
                        baseline_top1_margin,
                        pq_top1_prob,
                        pq_prob_of_baseline_top1,
                        pq_top1_margin,
                        mode_d_kl,
                        mode_d_top1,
                        mode_d_top1_agree,
                        baseline_top1_in_mode_d_top5,
                        coeff_mode_d_max_abs_logit_diff,
                        pre_wo_l2: metrics.pre_wo_l2,
                        wo_visible_l2: metrics.wo_visible_l2,
                    });
            }
        }
    }

    let mut head_reports = Vec::new();
    for head in &selected_heads {
        let basis = bases
            .get(head)
            .ok_or_else(|| format!("missing basis for L{} H{}", head.layer, head.head))?;
        let pca_basis = pca_bases
            .get(head)
            .ok_or_else(|| format!("missing PCA basis for L{} H{}", head.layer, head.head))?;
        let static_train_samples = means.get(head).map(|m| m.count).unwrap_or(0);
        let mut points = Vec::new();
        for &config in &configs {
            let acc = accumulators
                .remove(&(*head, config))
                .expect("oracle PQ accumulator missing at finish");
            let stability = code_stability
                .get(&(*head, config))
                .cloned()
                .unwrap_or_default();
            points.push(acc.finish(config, weights.hidden_size, stability));
        }
        head_reports.push(OraclePqHeadReport {
            layer: head.layer,
            head: head.head,
            head_dim: basis.head_dim,
            rank_retained: basis.rank_retained(),
            empirical_rank: pca_basis.rank(),
            sigma_max: basis.sigma_max,
            sigma_min_retained: basis.sigma_min_retained,
            static_train_samples,
            points,
        });
    }

    let report = OraclePqReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        train_prompts_seen: fit_prompts.len(),
        eval_prompts_seen: eval_prompts.len(),
        max_per_stratum: args.max_per_stratum,
        eval_mod: args.eval_mod,
        eval_offset: args.eval_offset,
        static_base: "position_mean".to_string(),
        configs,
        sigma_rel_cutoff: args.sigma_rel_cutoff,
        pq_iters: args.pq_iters,
        mode_d_check: args.mode_d_check,
        address_probes: args.address_probes,
        address_mixed_key_probe: args.address_mixed_key_probe,
        address_key_group_probe: args.address_key_group_probe,
        address_key_groups: if args.address_key_group_probe {
            key_groups
        } else {
            Vec::new()
        },
        address_key_group_probe_names: if args.address_key_group_probe {
            key_group_probe_names
        } else {
            Vec::new()
        },
        address_majority_group_probe: args.address_majority_group_probe,
        address_majority_groups: if args.address_majority_group_probe {
            majority_groups
        } else {
            Vec::new()
        },
        address_code_substitution_group_probe: args.address_code_substitution_group_probe,
        address_code_substitution_groups: if args.address_code_substitution_group_probe {
            code_substitution_groups
        } else {
            Vec::new()
        },
        address_code_substitution_from_codes: if args.address_code_substitution_group_probe {
            code_substitution_from_codes
        } else {
            Vec::new()
        },
        address_code_substitution_to_codes: if args.address_code_substitution_group_probe {
            code_substitution_to_specs
                .into_iter()
                .map(|spec| match spec {
                    CodeSubstitutionToSpec::Majority => "majority".to_string(),
                    CodeSubstitutionToSpec::Code(code) => code.to_string(),
                })
                .collect()
        } else {
            Vec::new()
        },
        address_code_class_collapse_group_probe: args.address_code_class_collapse_group_probe,
        address_code_class_collapse_groups: if args.address_code_class_collapse_group_probe {
            code_class_collapse_groups
        } else {
            Vec::new()
        },
        address_code_class_collapse_specs: if args.address_code_class_collapse_group_probe {
            code_class_collapse_specs
                .iter()
                .map(CodeClassCollapseSpec::label)
                .collect()
        } else {
            Vec::new()
        },
        address_code_position_interaction_probe: args.address_code_position_interaction_probe,
        address_code_position_prompt_id: if args.address_code_position_interaction_probe {
            code_position_prompt_id
        } else {
            String::new()
        },
        address_code_position_group: if args.address_code_position_interaction_probe {
            args.address_code_position_group
        } else {
            0
        },
        address_code_position_primary_codes: if args.address_code_position_interaction_probe {
            code_position_primary_codes
        } else {
            Vec::new()
        },
        address_code_position_secondary_codes: if args.address_code_position_interaction_probe {
            code_position_secondary_codes
        } else {
            Vec::new()
        },
        address_code_position_target_code: if args.address_code_position_interaction_probe {
            args.address_code_position_target_code
        } else {
            0
        },
        address_code_conditional_quotient_group_probe: args
            .address_code_conditional_quotient_group_probe,
        address_code_conditional_quotient_group: if args
            .address_code_conditional_quotient_group_probe
        {
            args.address_code_conditional_quotient_group
        } else {
            0
        },
        address_code_conditional_quotient_primary_codes: if args
            .address_code_conditional_quotient_group_probe
        {
            code_conditional_quotient_primary_codes
        } else {
            Vec::new()
        },
        address_code_conditional_quotient_secondary_codes: if args
            .address_code_conditional_quotient_group_probe
        {
            code_conditional_quotient_secondary_codes
        } else {
            Vec::new()
        },
        address_code_conditional_quotient_target_code: if args
            .address_code_conditional_quotient_group_probe
        {
            args.address_code_conditional_quotient_target_code
        } else {
            0
        },
        address_code_conditional_quotient_early_position_max: if args
            .address_code_conditional_quotient_group_probe
        {
            args.address_code_conditional_quotient_early_position_max
        } else {
            0
        },
        address_code_conditional_quotient_guards: if args
            .address_code_conditional_quotient_group_probe
        {
            code_conditional_quotient_guards
                .iter()
                .map(|guard| guard.label().to_string())
                .collect()
        } else {
            Vec::new()
        },
        address_code_conditional_quotient_extra_specs: if args
            .address_code_conditional_quotient_group_probe
        {
            code_conditional_quotient_extra_specs
                .iter()
                .map(CodeClassCollapseSpec::label)
                .collect()
        } else {
            Vec::new()
        },
        address_code7_bos_rule_group_probe: args.address_code7_bos_rule_group_probe,
        address_code7_bos_rule_groups: if args.address_code7_bos_rule_group_probe {
            code7_bos_rule_groups
        } else {
            Vec::new()
        },
        address_code7_bos_rule_code: if args.address_code7_bos_rule_group_probe {
            args.address_code7_bos_rule_code
        } else {
            0
        },
        address_code7_oracle_binary_group_probe: args.address_code7_oracle_binary_group_probe,
        address_code7_oracle_binary_groups: if args.address_code7_oracle_binary_group_probe {
            code7_oracle_binary_groups
        } else {
            Vec::new()
        },
        address_code7_oracle_binary_code: if args.address_code7_oracle_binary_group_probe {
            args.address_code7_oracle_binary_code
        } else {
            0
        },
        address_code7_oracle_binary_filters: if args.address_code7_oracle_binary_group_probe {
            code7_oracle_binary_filters
        } else {
            Vec::new()
        },
        address_corruption_sweep: args.address_corruption_sweep,
        address_group_importance: args.address_group_importance,
        address_lsh_group_probe: args.address_lsh_group_probe,
        address_lsh_groups: if args.address_lsh_group_probe {
            lsh_groups
        } else {
            Vec::new()
        },
        address_lsh_bits: args.address_lsh_bits,
        address_lsh_seeds: args.address_lsh_seeds,
        address_supervised_group_probe: args.address_supervised_group_probe,
        address_supervised_groups: if args.address_supervised_group_probe {
            supervised_groups
        } else {
            Vec::new()
        },
        address_supervised_epochs: args.address_supervised_epochs,
        address_supervised_lr: args.address_supervised_lr,
        address_supervised_l2: args.address_supervised_l2,
        address_gamma_projected_group_probe: args.address_gamma_projected_group_probe,
        address_gamma_projected_groups: if args.address_gamma_projected_group_probe {
            gamma_projected_groups
        } else {
            Vec::new()
        },
        address_gamma_projected_layers: if args.address_gamma_projected_group_probe {
            gamma_projected_layers
        } else {
            Vec::new()
        },
        address_gamma_random_ranks: if args.address_gamma_projected_group_probe {
            gamma_random_ranks
        } else {
            Vec::new()
        },
        address_gamma_random_seeds: if args.address_gamma_projected_group_probe {
            gamma_random_seeds
        } else {
            Vec::new()
        },
        address_gamma_learned_ranks: if args.address_gamma_projected_group_probe {
            gamma_learned_ranks
        } else {
            Vec::new()
        },
        address_gamma_learned_epochs: if args.address_gamma_projected_group_probe {
            args.address_gamma_learned_epochs
        } else {
            0
        },
        address_gamma_learned_lr: if args.address_gamma_projected_group_probe {
            args.address_gamma_learned_lr
        } else {
            0.0
        },
        address_gamma_learned_l2: if args.address_gamma_projected_group_probe {
            args.address_gamma_learned_l2
        } else {
            0.0
        },
        address_gamma_learned_pca_iters: if args.address_gamma_projected_group_probe {
            args.address_gamma_learned_pca_iters
        } else {
            0
        },
        address_code_stability: args.address_code_stability,
        address_code_stability_groups: if args.address_code_stability {
            code_stability_groups
        } else {
            Vec::new()
        },
        address_prev_ffn_feature_group_probe: args.address_prev_ffn_feature_group_probe,
        address_prev_ffn_feature_groups: if args.address_prev_ffn_feature_group_probe {
            prev_ffn_feature_groups
        } else {
            Vec::new()
        },
        address_prev_ffn_feature_top_k: args.address_prev_ffn_feature_top_k,
        address_ffn_first_feature_group_probe: args.address_ffn_first_feature_group_probe,
        address_ffn_first_feature_groups: if args.address_ffn_first_feature_group_probe {
            ffn_first_feature_groups
        } else {
            Vec::new()
        },
        address_ffn_first_feature_top_k: args.address_ffn_first_feature_top_k,
        address_attention_relation_group_probe: args.address_attention_relation_group_probe,
        address_attention_relation_groups: if args.address_attention_relation_group_probe {
            attention_relation_groups
        } else {
            Vec::new()
        },
        address_attention_cluster_group_probe: args.address_attention_cluster_group_probe,
        address_attention_cluster_groups: if args.address_attention_cluster_group_probe {
            attention_cluster_groups
        } else {
            Vec::new()
        },
        address_attention_cluster_ks: if args.address_attention_cluster_group_probe {
            attention_cluster_ks
        } else {
            Vec::new()
        },
        address_attention_cluster_probe_names: if args.address_attention_cluster_group_probe {
            attention_cluster_probe_names
        } else {
            Vec::new()
        },
        address_reduced_qk_cluster_group_probe: args.address_reduced_qk_cluster_group_probe,
        address_reduced_qk_cluster_groups: if args.address_reduced_qk_cluster_group_probe {
            reduced_qk_cluster_groups
        } else {
            Vec::new()
        },
        address_reduced_qk_ranks: if args.address_reduced_qk_cluster_group_probe {
            reduced_qk_ranks
        } else {
            Vec::new()
        },
        address_reduced_qk_cluster_ks: if args.address_reduced_qk_cluster_group_probe {
            reduced_qk_cluster_ks
        } else {
            Vec::new()
        },
        address_reduced_qk_cluster_probe_names: if args.address_reduced_qk_cluster_group_probe {
            reduced_qk_cluster_probe_names
        } else {
            Vec::new()
        },
        stratum_conditioned_pq_groups,
        selected_heads,
        heads: head_reports,
    };

    let out_path = args.out.join("oracle_pq.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn parse_string_list(spec: &str) -> Vec<String> {
    spec.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn oracle_mode_d_address_report(
    label: &str,
    stratum: &str,
    positions: usize,
    groups: usize,
    kl: f64,
    top1_agree: bool,
    baseline_top1_in_predicted_top5: bool,
) -> AddressProbePromptReport {
    AddressProbePromptReport {
        id: label.to_string(),
        stratum: stratum.to_string(),
        kl,
        positions,
        groups_correct: positions * groups,
        groups_total: positions * groups,
        exact_address_match: true,
        top1_agree,
        baseline_top1_in_predicted_top5,
    }
}

#[derive(Debug, Clone)]
struct CodeClassCollapseSpec {
    name: String,
    mappings: Vec<CodeClassCollapseMapping>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConditionalQuotientGuard {
    EarlyProsePosition,
    EarlyProseBosPrev,
    ProseBosPrev,
}

impl ConditionalQuotientGuard {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "early_prose_position" | "E_early_prose_position_guard" => {
                Some(ConditionalQuotientGuard::EarlyProsePosition)
            }
            "early_prose_bos_prev" | "F_early_prose_bos_prev_guard" => {
                Some(ConditionalQuotientGuard::EarlyProseBosPrev)
            }
            "prose_bos_prev" | "G_prose_bos_prev_guard" => {
                Some(ConditionalQuotientGuard::ProseBosPrev)
            }
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ConditionalQuotientGuard::EarlyProsePosition => "E_early_prose_position_guard",
            ConditionalQuotientGuard::EarlyProseBosPrev => "F_early_prose_bos_prev_guard",
            ConditionalQuotientGuard::ProseBosPrev => "G_prose_bos_prev_guard",
        }
    }

    fn keeps_secondary_oracle(
        self,
        stratum: &str,
        pos: usize,
        early_position_max: usize,
        attention_weights: &[f32],
    ) -> bool {
        if stratum != "natural_prose" {
            return false;
        }
        let is_early = pos <= early_position_max;
        match self {
            ConditionalQuotientGuard::EarlyProsePosition => is_early,
            ConditionalQuotientGuard::EarlyProseBosPrev => {
                is_early && is_bos_or_previous_attention(pos, attention_weights)
            }
            ConditionalQuotientGuard::ProseBosPrev => {
                is_bos_or_previous_attention(pos, attention_weights)
            }
        }
    }
}

fn is_bos_or_previous_attention(pos: usize, attention_weights: &[f32]) -> bool {
    if attention_weights.is_empty() {
        return false;
    }
    let source = attention_argmax(attention_weights, pos);
    source == 0 || (pos > 0 && source + 1 == pos)
}

impl CodeClassCollapseSpec {
    fn label(&self) -> String {
        format!("{}={}", self.name, self.mapping_label())
    }

    fn mapping_label(&self) -> String {
        self.mappings
            .iter()
            .map(|mapping| {
                let sources = mapping
                    .sources
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("+");
                format!("{sources}:{}", mapping.target)
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    fn mapping_label_or_base(&self) -> String {
        if self.mappings.is_empty() {
            "base".to_string()
        } else {
            self.mapping_label()
        }
    }
}

#[derive(Debug, Clone)]
struct CodeClassCollapseMapping {
    sources: Vec<usize>,
    target: usize,
}

fn parse_code_class_collapse_specs(
    spec: &str,
) -> Result<Vec<CodeClassCollapseSpec>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for (idx, raw_spec) in spec
        .split(';')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .enumerate()
    {
        let (raw_name, raw_mappings) = raw_spec
            .split_once('=')
            .map(|(name, mappings)| (name.trim(), mappings.trim()))
            .unwrap_or(("", raw_spec));
        let mappings = parse_code_class_collapse_mappings(raw_mappings)?;
        let fallback_name = sanitize_probe_name(
            &mappings
                .iter()
                .map(|mapping| {
                    let sources = mapping
                        .sources
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("+");
                    format!("{sources}_to_{}", mapping.target)
                })
                .collect::<Vec<_>>()
                .join("_and_"),
        );
        let name = if raw_name.is_empty() {
            format!("collapse{idx}_{fallback_name}")
        } else {
            sanitize_probe_name(raw_name)
        };
        if name.is_empty() {
            return Err(format!("invalid empty class-collapse name in spec {raw_spec:?}").into());
        }
        out.push(CodeClassCollapseSpec { name, mappings });
    }
    Ok(out)
}

fn parse_conditional_quotient_guards(
    spec: &str,
) -> Result<Vec<ConditionalQuotientGuard>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for raw in spec
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let guard = ConditionalQuotientGuard::parse(raw).ok_or_else(|| {
            format!(
                "unsupported conditional quotient guard {raw:?}; expected early_prose_position, early_prose_bos_prev, or prose_bos_prev"
            )
        })?;
        if !out.contains(&guard) {
            out.push(guard);
        }
    }
    Ok(out)
}

fn parse_code_class_collapse_mappings(
    spec: &str,
) -> Result<Vec<CodeClassCollapseMapping>, Box<dyn std::error::Error>> {
    let mut mappings = Vec::new();
    let mut seen_sources = Vec::new();
    for raw_mapping in spec
        .split('|')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let (raw_sources, raw_target) = raw_mapping.split_once(':').ok_or_else(|| {
            format!("invalid class-collapse mapping {raw_mapping:?}; expected sources:target")
        })?;
        let mut sources = Vec::new();
        for part in raw_sources
            .split('+')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            sources
                .push(part.parse::<usize>().map_err(|err| {
                    format!("invalid class-collapse source code {part:?}: {err}")
                })?);
        }
        sources.sort_unstable();
        sources.dedup();
        if sources.is_empty() {
            return Err(format!("class-collapse mapping {raw_mapping:?} has no sources").into());
        }
        for &source in &sources {
            if seen_sources.contains(&source) {
                return Err(format!(
                    "class-collapse source code {source} appears in more than one mapping"
                )
                .into());
            }
            seen_sources.push(source);
        }
        let target = raw_target.trim().parse::<usize>().map_err(|err| {
            format!(
                "invalid class-collapse target code {:?}: {err}",
                raw_target.trim()
            )
        })?;
        mappings.push(CodeClassCollapseMapping { sources, target });
    }
    if mappings.is_empty() {
        return Err(format!("class-collapse spec {spec:?} has no mappings").into());
    }
    Ok(mappings)
}

fn sanitize_probe_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
enum CodeSubstitutionToSpec {
    Majority,
    Code(usize),
}

fn parse_code_substitution_to_specs(
    spec: &str,
) -> Result<Vec<CodeSubstitutionToSpec>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for part in spec
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if part.eq_ignore_ascii_case("majority") {
            out.push(CodeSubstitutionToSpec::Majority);
        } else {
            out.push(CodeSubstitutionToSpec::Code(
                part.parse::<usize>()
                    .map_err(|err| format!("invalid code substitution target {part:?}: {err}"))?,
            ));
        }
    }
    Ok(out)
}
