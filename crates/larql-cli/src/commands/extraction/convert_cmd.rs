use larql_vindex::format::filenames::*;
use std::path::PathBuf;

use clap::{Args, Subcommand};

mod gguf_tokenizer;
#[cfg(test)]
mod tests;

#[derive(Args)]
pub struct ConvertArgs {
    #[command(subcommand)]
    command: ConvertCommand,
}

#[derive(Subcommand)]
enum ConvertCommand {
    /// Convert a GGUF model to a vindex.
    GgufToVindex {
        /// Path to the .gguf file.
        input: PathBuf,

        /// Output vindex directory.
        #[arg(short, long)]
        output: PathBuf,

        /// Extract level: browse (default), inference, all.
        #[arg(long, default_value = "browse")]
        level: String,

        /// Store in f16 (half precision).
        #[arg(long)]
        f16: bool,

        /// Retain native ternary (I2_S, GGML type 36) BitLinear
        /// weights verbatim instead of dequantizing them at
        /// convert time.  Writes a `bitnet/` subdirectory + a
        /// `bitnet_layout.json` describing tensor shapes and
        /// per-channel scale offsets.  Loaders dispatch on
        /// `index.json`'s `bitnet_layout` field at runtime.
        ///
        /// Only meaningful when the source GGUF is BitNet 1.58
        /// shaped (architecture = `bitnet-b1.58`).  No-op for
        /// non-BitNet GGUFs.  Closes BUG-infer-deadlock §5.4.
        #[arg(long)]
        keep_quant: bool,

        /// Dense-only build: skip the gate-vector + clustering
        /// stages (walk / browse).  Only valid with `--keep-quant`
        /// on a BitNet GGUF.  Produces a vindex that supports
        /// native-ternary `/v1/infer` (dense mode) at the ~1.4 GB
        /// resident footprint, without the ~2 GB f32 gate matrix or
        /// the 20-30 min HNSW clustering build.  Walk / browse /
        /// describe endpoints will return nothing useful on the
        /// resulting vindex.
        #[arg(long)]
        dense_only: bool,
    },

    /// Convert a safetensors model to a vindex (alias for extract-index).
    SafetensorsToVindex {
        /// Path to the model directory.
        input: PathBuf,

        /// Output vindex directory.
        #[arg(short, long)]
        output: PathBuf,

        /// Extract level: browse (default), inference, all.
        #[arg(long, default_value = "browse")]
        level: String,

        /// Store in f16.
        #[arg(long)]
        f16: bool,
    },

    /// Show GGUF file metadata and tensor info.
    GgufInfo {
        /// Path to the .gguf file.
        input: PathBuf,
    },

    /// Quantize an existing vindex into a different storage format.
    /// Each sub-format has its own flag surface — see
    /// `docs/specs/quantize-cli-spec.md` for the shape and how new
    /// formats slot in. FP4 is the only format wired as of exp 26;
    /// Q4K and future formats land as additional subcommands.
    #[command(subcommand)]
    Quantize(QuantizeCommand),

    /// Retrofit `down_features_q4k.bin` (W2 feature-major down) into
    /// an existing Q4K vindex without re-quantising. Reads the down
    /// portion of `interleaved_kquant.bin` per layer, transposes to
    /// `[intermediate, hidden]`, re-quantises at the same precision
    /// the source used, and writes the W2 file + manifest in place.
    /// Idempotent — silent no-op when the file is already present.
    /// See ADR-009 for the architectural rationale.
    AddFeatureMajorDown {
        /// Vindex directory to retrofit. Must already have
        /// `interleaved_kquant.bin` + manifest (i.e. `quant: q4k` in
        /// `index.json`).
        #[arg(long)]
        input: PathBuf,

        /// Suppress the per-layer progress line printed during write.
        #[arg(long)]
        quiet: bool,
    },
}

#[derive(Subcommand)]
enum QuantizeCommand {
    /// Convert an f32/f16 vindex into a Q4_K/Q6_K vindex (the Ollama-
    /// compatible "Q4_K_M" mix: attention Q/K/O + FFN gate/up at
    /// Q4_K, attention V + FFN down at Q6_K). `--down-q4k` switches
    /// FFN down to Q4_K uniformly — saves ~30 MB/layer on 31B at
    /// modest precision cost.
    ///
    /// Source must be extracted with `--level inference` or `--level all`
    /// (needs the full f32/f16 weights to quantise).
    Q4K {
        /// Existing vindex directory (the source).
        #[arg(long)]
        input: PathBuf,

        /// Output vindex directory. Written atomically (to `<out>.tmp/`
        /// then renamed on success).
        #[arg(long)]
        output: PathBuf,

        /// Quantise FFN down-proj as Q4_K instead of Q6_K. Default off
        /// preserves the Ollama Q4_K_M mix (Q4_K gate/up + Q6_K down).
        #[arg(long)]
        down_q4k: bool,

        /// Emit `down_features_q4k.bin` (W2 feature-major down) so per-feature
        /// row decode can skip the `kquant_ffn_layer` cache. Adds ~14 MB / layer
        /// at Gemma 4B dims; eliminates the ~840 MB heap cache ceiling.
        /// Recommended for CPU sparse walk and grid/MoE workloads.
        #[arg(long)]
        feature_major_down: bool,

        /// Overwrite the output directory if it already exists.
        #[arg(long)]
        force: bool,

        /// Suppress the backend-describe summary printed after write.
        #[arg(long)]
        quiet: bool,
    },

    /// Convert an f32/f16 vindex into an FP4/FP8 vindex per the
    /// chosen policy. Exp 26. Policy spec: `docs/specs/fp4-precision-policy.md`.
    Fp4 {
        /// Existing vindex directory (the source).
        #[arg(long)]
        input: PathBuf,

        /// Output vindex directory. Written atomically (to `<out>.tmp/`
        /// then renamed on success).
        #[arg(long)]
        output: PathBuf,

        /// Precision policy for up / down (gate stays at source dtype
        /// in all three policies — FP4 gate is blocked on an FP4-aware
        /// gate KNN path, see policy spec §2).
        #[arg(long, default_value = "option-b", value_parser = ["option-a", "option-b", "option-c"])]
        policy: String,

        /// Min compliance fraction for an FP4-targeted projection at
        /// the given threshold. Projections below this are downgraded
        /// to the manifest's fallback precision (FP8). Doesn't apply
        /// to FP8 / F16 projections — those don't use the
        /// distributional assumption.
        #[arg(long, default_value_t = 0.99)]
        compliance_floor: f32,

        /// max(sub-block scale)/min(sub-block scale) threshold for
        /// the FP4 compliance gate. 16.0 is the E4M3/E2M1 exponent
        /// budget (the format's derived default); lower = stricter,
        /// higher = more permissive.
        #[arg(long, default_value_t = 16.0)]
        threshold: f32,

        /// Overwrite the output directory if it already exists.
        #[arg(long)]
        force: bool,

        /// Fail (non-zero exit) if any FP4-targeted projection misses
        /// the compliance floor, instead of downgrading it.
        #[arg(long)]
        strict: bool,

        /// Skip emitting `fp4_compliance.json` in the output directory.
        #[arg(long)]
        no_sidecar: bool,

        /// Suppress the backend-describe summary printed after write.
        #[arg(long)]
        quiet: bool,
    },
}

pub fn run(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        ConvertCommand::GgufToVindex {
            input,
            output,
            level,
            f16,
            keep_quant,
            dense_only,
        } => run_gguf_to_vindex(&input, &output, &level, f16, keep_quant, dense_only),
        ConvertCommand::SafetensorsToVindex {
            input,
            output,
            level,
            f16,
        } => run_safetensors_to_vindex(&input, &output, &level, f16),
        ConvertCommand::GgufInfo { input } => run_gguf_info(&input),
        ConvertCommand::Quantize(cmd) => run_quantize(cmd),
        ConvertCommand::AddFeatureMajorDown { input, quiet } => {
            run_add_feature_major_down(&input, quiet)
        }
    }
}

fn run_add_feature_major_down(
    input: &std::path::Path,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::quant::add_feature_major_down;

    if !quiet {
        eprintln!("Retrofitting feature-major down → {}", input.display());
    }
    let report = add_feature_major_down(input)?;
    if report.skipped {
        if !quiet {
            eprintln!(
                "  down_features_q4k.bin already present — no-op (skipped {} layers)",
                report.num_layers,
            );
        }
        return Ok(());
    }
    if !quiet {
        let mb = report.bytes_written as f64 / (1024.0 * 1024.0);
        eprintln!(
            "  wrote down_features_q4k.bin: {} layers, {:.1} MB, {:.2?}",
            report.num_layers, mb, report.wall_time,
        );
        eprintln!(
            "  per-feature down decode now skips kquant_ffn_layer cache \
             (verify via GET /v1/stats → q4k_ffn.feature_major_down: true)"
        );
    }
    Ok(())
}

fn run_quantize(cmd: QuantizeCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        QuantizeCommand::Fp4 {
            input,
            output,
            policy,
            compliance_floor,
            threshold,
            force,
            strict,
            no_sidecar,
            quiet,
        } => run_quantize_fp4(QuantizeFp4Opts {
            input,
            output,
            policy,
            compliance_floor,
            threshold,
            force,
            strict,
            no_sidecar,
            quiet,
        }),
        QuantizeCommand::Q4K {
            input,
            output,
            down_q4k,
            feature_major_down,
            force,
            quiet,
        } => run_quantize_q4k(QuantizeQ4kOpts {
            input,
            output,
            down_q4k,
            feature_major_down,
            force,
            quiet,
        }),
    }
}

struct QuantizeQ4kOpts {
    input: PathBuf,
    output: PathBuf,
    down_q4k: bool,
    feature_major_down: bool,
    force: bool,
    quiet: bool,
}

fn run_quantize_q4k(opts: QuantizeQ4kOpts) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::quant::{vindex_to_q4k, Q4kConvertConfig};

    let config = Q4kConvertConfig {
        down_q4k: opts.down_q4k,
        feature_major_down: opts.feature_major_down,
        force: opts.force,
    };

    if !opts.quiet {
        eprintln!("== quantize q4k ==");
        eprintln!("  in       : {}", opts.input.display());
        eprintln!("  out      : {}", opts.output.display());
        eprintln!(
            "  down_q4k : {} ({})",
            opts.down_q4k,
            if opts.down_q4k {
                "Q4_K down (uniform)"
            } else {
                "Q6_K down (Q4_K_M mix)"
            }
        );
        eprintln!();
    }

    let report = vindex_to_q4k(&opts.input, &opts.output, &config)?;

    if !opts.quiet {
        eprintln!("── summary ──");
        eprintln!(
            "  FFN storage : {:.2} GB → {:.2} GB  ({:.2}× compression)",
            report.src_ffn_bytes as f64 / 1_073_741_824.0,
            report.dst_ffn_bytes as f64 / 1_073_741_824.0,
            report.compression,
        );
        eprintln!(
            "  Linked aux  : {} files ({:.2} GB)",
            report.aux_linked_count,
            report.aux_linked_bytes as f64 / 1_073_741_824.0
        );
        eprintln!("  Wall time   : {:.1}s", report.wall_time.as_secs_f64());
        eprintln!("  Walk backend: {}", report.walk_backend);
        eprintln!();
        eprintln!("→ {}", opts.output.display());
    }

    Ok(())
}

struct QuantizeFp4Opts {
    input: PathBuf,
    output: PathBuf,
    policy: String,
    compliance_floor: f32,
    threshold: f32,
    force: bool,
    strict: bool,
    no_sidecar: bool,
    quiet: bool,
}

fn run_quantize_fp4(opts: QuantizeFp4Opts) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::quant::{vindex_to_fp4, Fp4ConvertConfig, Policy, ProjectionOutcome};

    let policy = Policy::parse(&opts.policy)?;
    let config = Fp4ConvertConfig {
        policy,
        compliance_floor: opts.compliance_floor,
        threshold: opts.threshold,
        strict: opts.strict,
        force: opts.force,
        emit_sidecar: !opts.no_sidecar,
    };

    if !opts.quiet {
        eprintln!("== quantize fp4 ==");
        eprintln!("  in     : {}", opts.input.display());
        eprintln!("  out    : {}", opts.output.display());
        eprintln!("  policy : {}", policy.label());
        eprintln!(
            "  floor  : {:.1}% @ R<{}",
            opts.compliance_floor * 100.0,
            opts.threshold
        );
        eprintln!();
    }

    let (report, _scan) = vindex_to_fp4(&opts.input, &opts.output, &config)?;

    if !opts.quiet {
        eprintln!("── per-projection ──");
        for p in &report.per_projection {
            let compliance = p
                .compliance_at_threshold
                .map(|c| format!("{:.4}%", c * 100.0))
                .unwrap_or_else(|| "N/A".into());
            let downgrade_flag = matches!(
                p.outcome,
                ProjectionOutcome::DowngradedFp4ToFp8 | ProjectionOutcome::DowngradedFp4ToF16,
            );
            let marker = if downgrade_flag { "⚠" } else { " " };
            eprintln!(
                "  {marker} {:<5}  compliance={:<12}  → {:?}  ({})",
                p.name,
                compliance,
                p.chosen_precision,
                p.outcome.action_str(),
            );
        }
        eprintln!();
        eprintln!("── summary ──");
        eprintln!(
            "  FFN storage : {:.2} GB → {:.2} GB  ({:.2}× compression)",
            report.src_ffn_bytes as f64 / 1_073_741_824.0,
            report.dst_ffn_bytes as f64 / 1_073_741_824.0,
            report.compression,
        );
        eprintln!(
            "  Linked aux  : {} files ({:.2} GB)",
            report.aux_linked_count,
            report.aux_linked_bytes as f64 / 1_073_741_824.0
        );
        eprintln!("  Wall time   : {:.1}s", report.wall_time.as_secs_f64());
        eprintln!("  Walk backend: {}", report.walk_backend);
        eprintln!();
        if report.per_projection.iter().any(|p| {
            matches!(
                p.outcome,
                ProjectionOutcome::DowngradedFp4ToFp8 | ProjectionOutcome::DowngradedFp4ToF16
            )
        }) {
            eprintln!("⚠ compliance floor missed on ≥ 1 projection; see fp4_compliance.json.");
            if !opts.strict {
                eprintln!("(Use --strict to treat this as a fatal error.)");
            }
        }
        eprintln!("→ {}", opts.output.display());
    }

    Ok(())
}

fn run_gguf_to_vindex(
    input: &std::path::Path,
    output: &std::path::Path,
    level: &str,
    use_f16: bool,
    keep_quant: bool,
    dense_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading GGUF: {}", input.display());

    let gguf = larql_models::loading::gguf::GgufFile::open(input)?;

    if let Some(name) = gguf.metadata.get("general.name") {
        eprintln!("  Model: {:?}", name);
    }
    if let Some(arch) = gguf.metadata.get("general.architecture") {
        eprintln!("  Architecture: {:?}", arch);
    }

    // Detect BitNet so --keep-quant can be a no-op rather than an
    // error on non-BitNet inputs.  Match by architecture name to
    // stay forward-compatible with future BitNet variants that ship
    // additional ggml types.
    let is_bitnet = gguf
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .map(|s| s.starts_with("bitnet"))
        .unwrap_or(false);
    let do_keep_quant = keep_quant && is_bitnet;
    if keep_quant && !is_bitnet {
        eprintln!(
            "  --keep-quant: ignored (architecture is not BitNet; no I2_S \
             tensors to retain)"
        );
    }
    if dense_only && !do_keep_quant {
        return Err("--dense-only is only valid with --keep-quant on a BitNet \
             GGUF (it skips the gate-vector + clustering stages that only \
             walk / browse use; native-ternary BitNet /v1/infer does not \
             need them)."
            .into());
    }

    // Before the tensor load: a missing or unparseable tokenizer should
    // fail in seconds, not after dequantising every tensor.
    let tokenizer = gguf_tokenizer::load_sibling_tokenizer(input)?;

    eprintln!("  Loading and dequantizing tensors...");
    let weights = if do_keep_quant {
        // Retain raw bytes for I2_S BitLinear tensors so the
        // bitnet_writer can copy them verbatim into bitnet/.
        const TYPE_I2_S: u32 = 36;
        larql_models::loading::gguf::load_gguf_keep_quant(input, &[TYPE_I2_S])?
    } else {
        larql_models::load_gguf(input)?
    };

    eprintln!(
        "  {} layers, hidden_size={}, intermediate_size={}, vocab_size={}",
        weights.num_layers, weights.hidden_size, weights.intermediate_size, weights.vocab_size
    );

    let extract_level = match level {
        "inference" => larql_vindex::ExtractLevel::Inference,
        "all" => larql_vindex::ExtractLevel::All,
        _ => larql_vindex::ExtractLevel::Browse,
    };

    // --keep-quant needs the dense norms / embeddings / lm_head that
    // only inference/all levels extract; load_bitnet_model fails at
    // serve time on a browse-level vindex with "vindex does not
    // contain model weights".  Reject up front rather than producing
    // an unusable vindex after a multi-minute extract.  (Skipped for
    // --dense-only, which runs its own inference-equivalent build
    // regardless of --level.)
    if do_keep_quant && !dense_only && extract_level == larql_vindex::ExtractLevel::Browse {
        return Err("--keep-quant requires --level inference (or all): BitNet \
             inference needs the dense norm/embed/lm_head tensors that browse \
             level does not extract. Re-run with --level inference."
            .into());
    }

    let dtype = if use_f16 {
        larql_vindex::StorageDtype::F16
    } else {
        larql_vindex::StorageDtype::F32
    };

    let model_name = gguf
        .metadata
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("gguf-model")
        .to_string();

    let tokenizer_ref = &tokenizer;

    eprintln!("\nExtracting to {}", output.display());

    let mut callbacks = SilentCallbacks;
    if dense_only {
        eprintln!(
            "  Dense-only build: skipping gate-vector + clustering stages \
             (walk / browse disabled; native-ternary /v1/infer only)"
        );
        larql_vindex::build_vindex_dense_only(
            &weights,
            tokenizer_ref,
            &model_name,
            output,
            dtype,
            &mut callbacks,
        )?;
    } else {
        larql_vindex::build_vindex(
            &weights,
            tokenizer_ref,
            &model_name,
            output,
            10,
            extract_level,
            dtype,
            &mut callbacks,
        )?;
    }

    // BitNet --keep-quant: write the I2_S bytes + per-channel scales
    // and stamp `bitnet_layout` into index.json.
    if do_keep_quant {
        // The architecture the loader built from this file's own header
        // carries every dim the runtime loader needs; nothing is assumed.
        let arch = larql_vindex::extract::bitnet_writer::BitnetArchMeta::from_architecture(
            &*weights.arch,
        )?;
        let layout =
            larql_vindex::extract::bitnet_writer::write_bitnet_artifacts(output, &weights, arch)?;
        eprintln!(
            "  BitNet keep-quant: wrote {} I2_S tensors + {} scale entries \
             (heads={}q/{}kv, head_dim={}, eps={:.0e}, rope_base={})",
            layout.tensors.len(),
            layout.total_scale_count,
            arch.n_q_heads,
            arch.n_kv_heads,
            arch.head_dim,
            arch.rms_eps,
            arch.rope_base,
        );
        // Patch index.json with bitnet_layout = layout.
        patch_index_json_with_bitnet_layout(output, &layout)?;
    }
    // GGUF conversion: HF metadata (tokenizer_config.json etc.) is not
    // packed in the GGUF itself, but if the user kept the HF files next
    // to the `.gguf`, snapshot them. Missing-file case is a no-op.
    if let Some(src_dir) = input.parent() {
        if let Err(e) = larql_vindex::snapshot_hf_metadata(src_dir, output) {
            eprintln!("  warning: failed to snapshot HF metadata: {e}");
        }
    }

    eprintln!("Done: {}", output.display());
    Ok(())
}

fn run_safetensors_to_vindex(
    input: &std::path::Path,
    output: &std::path::Path,
    level: &str,
    use_f16: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // This is essentially extract-index
    eprintln!("Loading safetensors: {}", input.display());
    let weights = larql_models::load_model_dir(input)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(input).or_else(|_| {
        // Try to load from the model directory
        let tok_path = input.join(TOKENIZER_JSON);
        larql_vindex::tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| larql_vindex::VindexError::Parse(e.to_string()))
    })?;

    let extract_level = match level {
        "inference" => larql_vindex::ExtractLevel::Inference,
        "all" => larql_vindex::ExtractLevel::All,
        _ => larql_vindex::ExtractLevel::Browse,
    };

    let dtype = if use_f16 {
        larql_vindex::StorageDtype::F16
    } else {
        larql_vindex::StorageDtype::F32
    };

    let model_name = input
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());

    eprintln!("Extracting to {}", output.display());

    let mut callbacks = SilentCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        &model_name,
        output,
        10,
        extract_level,
        dtype,
        &mut callbacks,
    )?;
    // Snapshot HF-side metadata (chat template, special tokens, generation
    // config) from the source directory. `input` here is the safetensors
    // model dir, which is where these files live in the HF cache.
    if let Err(e) = larql_vindex::snapshot_hf_metadata(input, output) {
        eprintln!("  warning: failed to snapshot HF metadata: {e}");
    }

    eprintln!("Done: {}", output.display());
    Ok(())
}

fn run_gguf_info(input: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let gguf = larql_models::loading::gguf::GgufFile::open(input)?;

    println!("GGUF: {}", input.display());
    println!();

    // Print metadata
    println!("Metadata ({} keys):", gguf.metadata.len());
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in &keys {
        let val = &gguf.metadata[*key];
        match val {
            larql_models::loading::gguf::GgufValue::String(s) => {
                if s.len() > 80 {
                    println!("  {}: \"{}...\"", key, &s[..80]);
                } else {
                    println!("  {}: \"{}\"", key, s);
                }
            }
            larql_models::loading::gguf::GgufValue::Array(arr) => {
                println!("  {}: [{} elements]", key, arr.len());
            }
            other => println!("  {}: {:?}", key, other),
        }
    }

    println!();

    // Print tensor info table (name, dims, ggml type id) — the layout spec a
    // consumer (e.g. a vindex→GGUF exporter) must match. Sorted by name.
    println!();
    println!("Tensors ({}):", gguf.tensor_infos.len());
    let mut infos: Vec<&larql_models::loading::gguf::GgufTensorInfo> =
        gguf.tensor_infos.iter().collect();
    infos.sort_by(|a, b| a.name().cmp(b.name()));
    for t in &infos {
        println!(
            "  {:<40} dims={:?} type={}",
            t.name(),
            t.dims(),
            t.tensor_type(),
        );
    }

    println!();

    // Print synthesised config
    let config = gguf.to_config_json();
    println!("Detected config:");
    println!("  {}", serde_json::to_string_pretty(&config)?);

    Ok(())
}

struct SilentCallbacks;
impl larql_vindex::IndexBuildCallbacks for SilentCallbacks {}

/// Re-write `index.json` to include the `bitnet_layout` block.
///
/// `build_vindex` writes index.json before we know whether the
/// `--keep-quant` artifacts will be produced, so we round-trip
/// the file through serde to add the layout in place.  Stable
/// across reruns: the load -> patch -> write cycle is idempotent.
fn patch_index_json_with_bitnet_layout(
    out_dir: &std::path::Path,
    layout: &larql_vindex::config::BitnetLayout,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = out_dir.join(INDEX_JSON);
    let bytes = std::fs::read(&path)?;
    let mut config: larql_vindex::VindexConfig = serde_json::from_slice(&bytes)?;
    config.bitnet_layout = Some(layout.clone());
    let json = serde_json::to_string_pretty(&config)?;
    std::fs::write(&path, json)?;
    Ok(())
}
