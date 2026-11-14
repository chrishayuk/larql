//! `--moe-shards`: attention and dense FFN local, MoE experts on remote shards.

#[allow(unused_imports)]
use super::*;

/// `--moe-shards` dispatch path.
///
/// Metal runs attention + dense FFN on GPU (same as normal `larql run --metal`).
/// MoE expert blocks are dispatched to remote mini-processes via binary
/// `POST /v1/expert/batch` instead of running locally.
pub(super) fn run_with_moe_shards(
    vindex_path: &std::path::Path,
    prompt: &str,
    shards_str: Option<&str>,
    units_manifest: Option<&std::path::Path>,
    max_tokens: usize,
    dispatch: &str,
    predispatch_iters: usize,
    metal: bool,
    engine_spec: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Remote MoE needs an engine that dispatches FFN per-layer through the
    // `ffn` trait (where `RemoteMoeFfn` hooks the experts): `standard` and
    // `boundary_kv` (which wraps a StandardEngine and adds compressed-residual
    // boundary frames — same dispatch, wire-efficient cold-context). The
    // compression engines (markov_residual / turbo_quant / windowed_checkpoint /
    // boundary_per_layer) route FFN through the backend's fused coarse path, and
    // apollo/no_cache re-forward — none have a remote-expert hook, so they'd
    // silently drop experts. Reject them clearly.
    if let Some(spec) = engine_spec {
        let kind = larql_kv::EngineKind::from_name(spec)
            .ok_or_else(|| format!("unknown --engine `{spec}`"))?;
        if !matches!(
            kind,
            larql_kv::EngineKind::Standard { .. }
                | larql_kv::EngineKind::BoundaryKv { .. }
                | larql_kv::EngineKind::WindowedCheckpoint { .. }
                | larql_kv::EngineKind::MarkovResidual { .. }
                | larql_kv::EngineKind::MarkovResidualCodec { .. }
                | larql_kv::EngineKind::TurboQuant { .. }
                | larql_kv::EngineKind::BoundaryPerLayer { .. }
        ) {
            return Err(format!(
                "`--engine {}` is not supported with remote MoE (--moe-shards). Supported: \
                 standard, boundary, windowed-checkpoint, markov-rs, markov-residual-codec, \
                 turbo-quant, boundary-per-layer (they dispatch FFN per-layer through the ffn \
                 trait where experts hook in). `no-cache` / `apollo` re-forward and would \
                 multiply expert round-trips. See larql-kv ROADMAP §\"MoE-aware KV engines (C1)\".",
                kind.display_name()
            )
            .into());
        }
    }
    use larql_inference::ffn::moe_remote::{parse_unit_manifest, RemoteMoeBackend, ShardConfig};
    use larql_inference::{
        generate_kquant_cpu_remote, generate_with_remote_moe, generate_with_remote_moe_batch,
    };

    // Pick ownership mode: legacy `--moe-shards` (layer-uniform ranges) or
    // `--moe-units-manifest` (fine-grained per-(layer, expert) sets).  The
    // mutually-exclusive guard at the caller means at most one is set here.
    let configs: Vec<ShardConfig> = if let Some(path) = units_manifest {
        let cfgs = parse_unit_manifest(path).map_err(|e| format!("--moe-units-manifest: {e}"))?;
        if cfgs.is_empty() {
            return Err("--moe-units-manifest: manifest contains no shards".into());
        }
        eprintln!(
            "Loaded {} shard(s) from unit manifest at {}",
            cfgs.len(),
            path.display()
        );
        cfgs
    } else if let Some(s) = shards_str {
        // Parse "START-END=URL,START-END=URL,..." into Vec<ShardConfig>.
        let mut cfgs: Vec<ShardConfig> = Vec::new();
        for segment in s.split(',') {
            let segment = segment.trim();
            if segment.is_empty() {
                continue;
            }
            let mut parts = segment.splitn(2, '=');
            let range_str = parts
                .next()
                .ok_or_else(|| format!("malformed shard segment: {segment:?}"))?;
            let url = parts
                .next()
                .ok_or_else(|| format!("missing URL in shard segment: {segment:?}"))?;
            let (start, end_incl) = ShardConfig::parse_range(range_str)
                .ok_or_else(|| format!("bad expert range {range_str:?} in --moe-shards"))?;
            cfgs.push(ShardConfig::new(start, end_incl, url));
        }
        if cfgs.is_empty() {
            return Err("--moe-shards: no valid shard segments found".into());
        }
        cfgs
    } else {
        return Err("internal error: run_with_moe_shards called with neither flag".into());
    };

    let num_shards = configs.len();
    // Initialise compute backend early so we can report it in the topology banner.
    // An explicit `--metal` with no usable Metal device is a loud error, not a
    // CPU fallback — see `backend_select`.
    let backend: Box<dyn larql_compute::ComputeBackend> =
        crate::backend_select::backend_for_metal_flag(metal)?;
    eprintln!("Connecting to {} MoE shard(s)…", num_shards);
    let remote = RemoteMoeBackend::connect(configs)
        .map_err(|e| format!("failed to connect to MoE shards: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  Router:     local");
    eprintln!(
        "  Experts:    remote  (sharded across {} endpoint{})",
        num_shards,
        if num_shards == 1 { "" } else { "s" }
    );

    // Client loads attn + dense FFN + norms + router weights — no expert bytes.
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index
        .load_attn_kquant(vindex_path)
        .map_err(|e| format!("failed to load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(vindex_path)
        .map_err(|e| format!("failed to load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(vindex_path);

    // Prompt-shape options (centralised in `larql_inference::chat::render_user_prompt`):
    //   default              → chat_template.jinja with auto-injected default system prompt for Gemma 4
    //   LARQL_RAW_PROMPT=1   → raw user string with <bos> prepended (no template)
    //   LARQL_THINKING=1     → enable_thinking=true (skips empty thought block)
    //   LARQL_SYSTEM=<text>  → explicit system message
    //   LARQL_NO_DEFAULT_SYSTEM=1 → suppress the auto-injected Gemma 4 default
    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), prompt)?;
    if std::env::var("LARQL_DUMP_PROMPT").is_ok() {
        let mode = if std::env::var("LARQL_RAW_PROMPT").is_ok() {
            "raw"
        } else if std::env::var("LARQL_THINKING").is_ok() {
            "thinking"
        } else {
            "default"
        };
        eprintln!(
            "[chat] mode={mode} ---PROMPT START---\n{wrapped_prompt}\n[chat] ---PROMPT END---"
        );
    }
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("failed to tokenise prompt: {e}"))?;
    eprintln!("[chat] tokenised to {} ids", prompt_ids.len());

    // Backend-aware dispatch, probed on the constructed backend instance
    // rather than the `--metal` flag: backends that implement the fused
    // `DecodeBackend::decode_token_with_moe` trait method (Metal today,
    // CUDA post-G-ladder) take the fused path; backends that don't (CPU —
    // the method returns `None`, which previously surfaced as
    // "decode_token_with_moe returned None during prefill" whenever
    // `--metal` was omitted, #146) route through the CPU remote-MoE
    // forward instead: per-token `predict_kquant_hidden(Some(remote))` →
    // `run_moe_layer_cpu` → `forward_moe_seq`, which dispatches each MoE
    // layer's experts to the shards over HTTP.
    let (tokens, decode_ms): (Vec<String>, Vec<f64>) = if backend
        .supports(larql_compute::Capability::DecodeMoe)
    {
        let eos =
            larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
        let result = if dispatch == "batch" {
            generate_with_remote_moe_batch(
                &weights,
                &tokenizer,
                prompt_ids,
                max_tokens,
                &index,
                &remote,
                &*backend,
                &eos,
                predispatch_iters,
            )
        } else {
            generate_with_remote_moe(
                &weights, &tokenizer, prompt_ids, max_tokens, &index, &remote, &*backend, &eos,
            )
        }
        .map_err(|e| format!("grid generate failed ({dispatch}): {e}"))?;
        (result.tokens, result.decode_ms)
    } else {
        if dispatch == "batch" {
            eprintln!(
                "  note: --moe-dispatch batch is GPU-only; using sequential CPU expert dispatch"
            );
        }
        // CPU remote-MoE. Default: the KV-cached engine path — dequantize the
        // client weights (attn + dense FFN; experts stay remote) to f32 once,
        // then drive a StandardEngine with `RemoteMoeFfn` so attention is
        // KV-cached and only MoE experts round-trip to the shards. ~10× faster
        // than full-recompute and byte-identical output (verified on
        // Gemma-4-26B-A4B, 2 shards).
        //
        // Falls back to the standalone full-recompute path (closes #146) for
        // Per-Layer-Embedding archs (the engine path doesn't apply PLE) or when
        // `LARQL_MOE_FULL_RECOMPUTE=1` is set as an escape hatch.
        let uses_ple = weights.arch.per_layer_input_gate_key(0).is_some();
        let force_recompute = std::env::var("LARQL_MOE_FULL_RECOMPUTE").is_ok();
        if uses_ple || force_recompute {
            if uses_ple {
                eprintln!(
                    "  note: model uses Per-Layer Embeddings — full-recompute CPU path (no KV cache)"
                );
            }
            let started = std::time::Instant::now();
            // Fatal by policy: a shard failure aborts the run rather than
            // finishing the sentence from a model missing an expert layer.
            let toks = generate_kquant_cpu_remote(
                &mut weights,
                &tokenizer,
                &prompt_ids,
                max_tokens,
                &index,
                &remote,
            )
            .map_err(|e| format!("remote MoE dispatch failed, generation aborted: {e}"))?;
            let total_ms = started.elapsed().as_secs_f64() * 1000.0;
            let strings: Vec<String> = toks.into_iter().map(|(s, _)| s).collect();
            let n = strings.len();
            let per = if n == 0 { 0.0 } else { total_ms / n as f64 };
            (strings, vec![per; n])
        } else {
            // Dequantize attn + dense FFN to f32 for every layer, kept resident
            // for the whole generation (experts are not loaded on the client).
            for layer in 0..weights.num_layers {
                larql_inference::vindex::insert_q4k_layer_tensors_resident(
                    &mut weights,
                    &index,
                    layer,
                )
                .map_err(|e| format!("failed to dequantize layer {layer} to f32: {e}"))?;
            }
            let moe_ffn = larql_inference::ffn::RemoteMoeFfn {
                weights: &weights,
                remote: &remote,
            };
            // Build the chosen MoE-capable engine (validated above): `standard`
            // or `boundary_kv`. Drive through `generate_with_engine` (not
            // `generate_cached`): it routes prefill/decode through
            // `engine.prefill/decode_step` → the MoE-aware `kv_*_via_dispatch`
            // path. `generate_cached` uses the legacy `kv_prefill_run` path,
            // which has no MoE hook (experts never dispatched).
            let kind = engine_spec
                .and_then(larql_kv::EngineKind::from_name)
                .unwrap_or(larql_kv::EngineKind::Standard { window_size: None });
            eprintln!("  Engine:     {} (CPU, KV-cached)", kind.display_name());
            let mut engine = kind.build(larql_inference::cpu_engine_backend());
            let mut strings: Vec<String> = Vec::new();
            // Capture a timestamp per emitted token so the banner reports TRUE
            // steady-state decode (inter-token intervals), not total/n which
            // conflates model-load + prefill into the per-token number.
            let mut tok_times: Vec<std::time::Instant> = Vec::new();
            let started = std::time::Instant::now();
            // Resident-weights quant path: weights were dequantised f32-resident
            // above, so this threads the `index` to the backend (no `&mut`, so
            // `moe_ffn` can borrow `&weights` concurrently) — letting the
            // Q4K-direct attention kernel fire under `LARQL_Q4K_DIRECT_ATTN`.
            // With the flag unset the backend ignores the index → identical f32.
            //
            // Checked: `RemoteMoeFfn` is strict, so a shard that fails (5xx,
            // unreachable, timeout) ends the stream with an error naming the
            // shard, layer and experts. Return it rather than print the tokens
            // that came before as if they were the answer — the same policy as
            // the full-recompute branch above.
            let outcome = larql_kv::generation::generate_with_engine_resident_checked(
                &mut engine,
                &weights,
                &tokenizer,
                &moe_ffn,
                &index,
                &prompt_ids,
                max_tokens,
                |_id, tok| {
                    strings.push(tok.to_string());
                    tok_times.push(std::time::Instant::now());
                },
            );
            if let Some(err) = outcome.error {
                return Err(format!(
                    "remote MoE dispatch failed, generation aborted after {} token(s): {err}",
                    outcome.tokens.len()
                )
                .into());
            }
            // Time-to-first-token (prefill) is the gap start→first emit; decode
            // is the gap between consecutive emits.
            if let Some(first) = tok_times.first() {
                eprintln!(
                    "  prefill (TTFT):  {:.0} ms",
                    first.duration_since(started).as_secs_f64() * 1000.0
                );
            }
            let decode_ms: Vec<f64> = tok_times
                .windows(2)
                .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1000.0)
                .collect();
            if larql_inference::decode_stages::is_enabled() {
                let (attn_ms, dense_ms, expert_ms, lmhead_ms) =
                    larql_inference::decode_stages::snapshot_ms();
                // Accumulated over prefill + decode. attn/dense/lm_head are
                // client-side; experts are server-side. "Everything else"
                // (router, combine, embed) is the remainder of wall-time.
                eprintln!(
                    "  [stages] attn: {attn_ms:.0} ms | dense FFN: {dense_ms:.0} ms | \
                     lm_head: {lmhead_ms:.0} ms | remote experts: {expert_ms:.0} ms"
                );
            }
            (strings, decode_ms)
        }
    };

    for tok in &tokens {
        print!("{tok}");
    }
    if !tokens.is_empty() {
        println!();
    }
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let n = decode_ms.len();
    if n > 0 {
        let avg = decode_ms.iter().sum::<f64>() / n as f64;
        let tok_s = 1000.0 / avg;
        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let top_k = weights.arch.num_experts_per_token();
        let experts_invoked = num_layers * top_k * n;
        // One f32 residual vector per layer per shard in each direction.
        let bytes_per_token = num_layers * num_shards * hidden * std::mem::size_of::<f32>();
        let kb = |b: usize| b as f64 / 1024.0;
        eprintln!();
        eprintln!("  decode:          {tok_s:.1} tok/s");
        eprintln!(
            "  experts invoked: {experts_invoked}  ({num_layers} layers × top-{top_k} × {n} token{})",
            if n == 1 { "" } else { "s" }
        );
        eprintln!(
            "  bytes sent:      ~{:.0} KB  ({num_layers} layers × {num_shards} shard{} × hidden × f32)",
            kb(bytes_per_token * n),
            if num_shards == 1 { "" } else { "s" }
        );
        eprintln!(
            "  bytes recv:      ~{:.0} KB  ({num_layers} layers × {num_shards} shard{} × hidden × f32)",
            kb(bytes_per_token * n),
            if num_shards == 1 { "" } else { "s" }
        );
    }
    Ok(())
}
