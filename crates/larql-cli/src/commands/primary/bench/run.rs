//! Bench orchestration: pick backends, drive each, render the table, emit
//! JSON. Heavy lifting lives in the per-backend modules; this file is just
//! the dispatch + JSON envelope.

use larql_kv::EngineKind;

use crate::commands::primary::cache;

use super::args::BenchArgs;
use super::engine_runtime::run_engine;
use super::grid_lan_runtime::{self, GridLanOptions};
use super::helpers;
use super::local_moe_runtime;
use super::local_runtime::run_larql;
use super::ollama::run_ollama;
use super::output::print_table;
use super::remote_ffn_runtime::run_concurrent_ffn;
use super::remote_moe_runtime::run_concurrent_moe;
use super::row::{BenchJsonLatency, BenchJsonResult, BenchJsonRow, BenchJsonStages, BenchRow};

pub fn run(mut args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    // The steady-state A/B vehicle: arm the execution seam here and hold
    // the guard for the whole bench, so the tok/s table and the BW10
    // summary below describe the SAME policy. Inert unless
    // LARQL_EXEC_POLICY is set; a malformed value stops the run.
    let _exec_policy = larql_compute::exec_policy::spec::from_env()?;
    // Configure rayon's global thread pool up front. Auto-select picks
    // 8 on Apple silicon — empirically the sweet spot for Q4_K × Q8_K
    // matvec on M3 Max's LPDDR5 controllers (12-thread default
    // saturates DRAM channels + adds rayon work-steal overhead, see
    // `bench/baselines/cpu/DIAGNOSIS-2026-05-16-thread-scaling.md`).
    // `RAYON_NUM_THREADS` in the environment overrides everything.
    configure_rayon_threads(args.threads);

    // `--cpu` is shorthand for a CPU-only run. Two normalisations:
    //  1. Force `backends = "cpu"` so the engine path (which decides
    //     CPU-vs-Metal via `args.backends.contains("metal")`) builds a
    //     CpuBackend rather than silently running the engine on Metal.
    //  2. Unless the user picked engines explicitly, also surface the
    //     production `standard` StandardEngine CPU row. That is the path
    //     `larql run`/`larql walk` actually use and it is ~12% faster
    //     than the legacy fused `larql-cpu` bench row (26.4 vs 23.5 tok/s
    //     on Gemma 3 4B Q4K, M3 Max). Without this, `larql bench --cpu`
    //     reported only the slower legacy path — understating the real
    //     CPU number (see `bench/baselines/c10_gemma3-4b_cpu_reconciled.json`).
    if args.cpu {
        args.backends = "cpu".to_string();
        if args.engine.is_none() {
            args.engine = Some("standard".to_string());
        }
    }

    // --bench-grid-lan short-circuits the normal flow: it orchestrates
    // a matrix of independent `larql bench` invocations from a JSON
    // config, mirroring `experiments/41_residual_transport_grid/run.py`.
    if let Some(config_path) = args.bench_grid_lan.clone() {
        let out_dir = args.grid_lan_out.clone().unwrap_or_else(|| {
            let parent = config_path
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| std::path::PathBuf::from("."));
            parent.join("results")
        });
        let only = if args.grid_lan_only.is_empty() {
            None
        } else {
            Some(args.grid_lan_only.clone())
        };
        return grid_lan_runtime::run(GridLanOptions {
            config_path,
            out_dir,
            only,
            include_disabled: args.grid_lan_include_disabled,
            dry_run: args.grid_lan_dry_run,
            timeout_secs: None,
            cov_threshold: args.grid_lan_cov_threshold,
            cov_extra_repeats: args.grid_lan_extra_repeats,
        });
    }

    let vindex_path = cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display(),
        )
        .into());
    }

    let requested_backends: Vec<&str> = if args.cpu {
        vec!["cpu"]
    } else {
        args.backends
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let want_metal = requested_backends.contains(&"metal");
    let want_cpu = requested_backends.contains(&"cpu");
    let want_engine = args.engine.is_some();
    let want_ffn = args.ffn.is_some();
    let want_moe = args.moe_shards.is_some();
    if !want_metal && !want_cpu && args.ollama.is_none() && !want_engine && !want_ffn && !want_moe {
        return Err(
            "no backends selected: pass --backends metal,cpu, --ollama, --engine, --ffn, or --moe-shards".into(),
        );
    }

    println!("larql bench: {}", vindex_path.display());
    println!("Prompt: {:?}", args.prompt);
    if want_cpu {
        let active = rayon::current_num_threads();
        println!("CPU threads: {} (rayon)", active);
    }
    println!(
        "Decode: {} tokens after {} warmup; backends={}{}",
        args.tokens,
        args.warmup,
        if args.cpu {
            "cpu"
        } else {
            args.backends.as_str()
        },
        args.ollama
            .as_deref()
            .map(|m| format!(", ollama={m}"))
            .unwrap_or_default(),
    );
    println!();

    let mut rows: Vec<BenchRow> = Vec::new();

    // GPU/CPU bench requires Q4K vindex. Skip silently when running engine-only
    // (engines need f32 weights from a non-Q4K vindex).
    let cfg = larql_vindex::load_vindex_config(&vindex_path)?;
    let is_q4k = cfg.quant == larql_vindex::QuantFormat::Q4K;

    // Hybrid-MoE vindexes (Gemma 4 26B-A4B etc.) need the in-process KV-cached
    // MoE decode path, not the dense legacy CPU decode (`run_larql`) or the
    // NullFfn engine path — both ignore the experts. Detect up front (mmap
    // load, cheap) so the CPU rows route correctly. `--moe-shards` keeps the
    // remote path; Metal MoE keeps its existing GPU dispatch.
    let arch_is_moe = is_q4k && vindex_is_hybrid_moe(&vindex_path);
    // CPU MoE without shards → drive the in-process LocalMoeFfn engine path.
    let want_local_moe = arch_is_moe && !want_metal && args.moe_shards.is_none();

    if want_metal {
        if is_q4k {
            rows.push(run_larql(&vindex_path, &args, /* metal */ true)?);
        } else if !want_engine {
            return Err(format!(
                "GPU bench requires a Q4K vindex (got quant={:?}). \
                 Use a q4k vindex for GPU bench, or omit --backends and use --engine only.",
                cfg.quant,
            )
            .into());
        }
    }
    if want_cpu && !arch_is_moe {
        if is_q4k {
            rows.push(run_larql(&vindex_path, &args, /* metal */ false)?);
        } else if !want_engine {
            return Err(format!(
                "CPU bench requires a Q4K vindex (got quant={:?}).",
                cfg.quant,
            )
            .into());
        }
    }
    if let Some(ref ollama_model) = args.ollama {
        // `--ollama-cpu` forces num_gpu=0 + num_thread=<bench threads> so
        // Ollama is a true CPU baseline; otherwise it runs on its default
        // backend (Metal GPU on Apple silicon) and is a GPU comparison.
        let ollama_cpu_threads = if args.ollama_cpu {
            Some(rayon::current_num_threads())
        } else {
            None
        };
        rows.push(run_ollama(
            ollama_model,
            &args.prompt,
            args.tokens,
            ollama_cpu_threads,
        ));
    }

    // KV engine rows.
    //
    // Q4K vindex → prefill_q4k / decode_step_q4k (Metal pipeline, fast path).
    // f16/f32 vindex → prefill / decode_step (f32 CPU path, slow but correct).
    if let Some(ref engine_list) = args.engine {
        let mut cb = larql_vindex::SilentLoadCallbacks;

        if is_q4k {
            let mut weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
            let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
            index.load_attn_kquant(&vindex_path)?;
            index.load_interleaved_kquant(&vindex_path)?;
            let token_ids =
                larql_inference::encode_prompt(&tokenizer, &*weights.arch, args.prompt.as_str())
                    .map_err(|e| format!("tokenize: {e}"))?;
            if want_local_moe {
                // In-process KV-cached MoE: attention is KV-cached on the
                // engine, experts are computed locally from the resident
                // vindex (no shards, no loopback round-trip). This is the
                // fair single-box CPU MoE decode number — the one that pins
                // the medium-term tier against llama.cpp-on-26B-CPU.
                rows.extend(local_moe_runtime::run_local_moe(
                    &mut weights,
                    &index,
                    &tokenizer,
                    &token_ids,
                    engine_list,
                    &args,
                )?);
            } else if arch_is_moe && !want_metal {
                // CPU MoE with --moe-shards: the remote-MoE block below
                // dispatches experts to the shards. The dense NullFfn engine
                // loop would silently ignore the experts, so skip it here.
                eprintln!(
                    "[bench] MoE vindex on CPU with --moe-shards — experts via \
                     the remote block; dense engine row skipped"
                );
            } else {
                let kv_ref_bytes = larql_kv::markov_residual::kv_memory_bytes_for_seq(
                    larql_inference::WeightsView::dense(&weights),
                    token_ids.len(),
                );

                // Parse + validate --ffn-policy once before the engine loop
                // (multi-engine sweep reuses the same validated policy).
                // Q4K path accepts but doesn't yet honor — engine_runtime
                // logs a warning if non-None.
                let validated_policy =
                    parse_ffn_policy(args.ffn_policy.as_deref(), weights.num_layers)?;

                for engine_name in EngineKind::split_specs(engine_list) {
                    match EngineKind::from_name(&engine_name) {
                        Some(kind) => {
                            let backend = if want_metal {
                                larql_inference::default_engine_backend()
                            } else {
                                larql_inference::cpu_engine_backend()
                            };
                            rows.push(run_engine(
                                &mut weights,
                                Some(&index),
                                &token_ids,
                                kv_ref_bytes,
                                kind,
                                backend,
                                validated_policy.as_ref(),
                                &args,
                            )?);
                        }
                        None => eprintln!(
                            "unknown engine {:?} — supported: {}",
                            engine_name,
                            EngineKind::supported_names().join(", "),
                        ),
                    }
                }
            }
        } else {
            // `&mut` so the unified `run_engine` signature works for the
            // dense path too (`run_engine` requires &mut for the quant
            // path's lazy dequant; the dense path reborrows immutably
            // inside).
            let mut weights = larql_vindex::load_model_weights(&vindex_path, &mut cb)?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
            let token_ids =
                larql_inference::encode_prompt(&tokenizer, &*weights.arch, args.prompt.as_str())
                    .map_err(|e| format!("tokenize: {e}"))?;
            let kv_ref_bytes = larql_kv::markov_residual::kv_memory_bytes_for_seq(
                larql_inference::WeightsView::dense(&weights),
                token_ids.len(),
            );

            let validated_policy =
                parse_ffn_policy(args.ffn_policy.as_deref(), weights.num_layers)?;

            for engine_name in EngineKind::split_specs(engine_list) {
                match EngineKind::from_name(&engine_name) {
                    Some(kind) => {
                        let backend = if want_metal {
                            larql_inference::default_engine_backend()
                        } else {
                            larql_inference::cpu_engine_backend()
                        };
                        rows.push(run_engine(
                            &mut weights,
                            None,
                            &token_ids,
                            kv_ref_bytes,
                            kind,
                            backend,
                            validated_policy.as_ref(),
                            &args,
                        )?);
                    }
                    None => eprintln!(
                        "unknown engine {:?} — supported: {}",
                        engine_name,
                        EngineKind::supported_names().join(", "),
                    ),
                }
            }
        }
    }

    if let Some(ref ffn_url) = args.ffn {
        let wire_prefs = match args.wire.as_deref() {
            Some(spec) => {
                let parsed = helpers::parse_wire_list(spec);
                if parsed.is_empty() {
                    vec![larql_inference::WirePreference::BestAvailable]
                } else {
                    parsed
                }
            }
            None => vec![larql_inference::WirePreference::BestAvailable],
        };
        for pref in wire_prefs {
            rows.push(run_concurrent_ffn(&vindex_path, &args, ffn_url, pref)?);
        }
    }

    if let Some(ref shards_str) = args.moe_shards {
        if args.bench_grid {
            // Grid scaling sweep: run with 1..N shards from the shard map.
            let shard_entries: Vec<&str> = shards_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            // Track single-shard tok/s so we can compute shard_efficiency
            // for the larger configurations (ADR-0012 §--bench-grid Mode).
            let mut single_shard_tok_per_s: Option<f64> = None;
            for n_shards in 1..=shard_entries.len() {
                let partial = shard_entries[..n_shards].join(",");
                let mut row = run_concurrent_moe(&vindex_path, &args, &partial)?;
                if n_shards == 1 {
                    single_shard_tok_per_s = Some(row.tok_per_s);
                }
                row.shard_efficiency = single_shard_tok_per_s
                    .and_then(|base| helpers::shard_efficiency(row.tok_per_s, n_shards, base));
                row.note = format!(
                    "{} shard{} | {}",
                    n_shards,
                    if n_shards == 1 { "" } else { "s" },
                    row.note
                );
                rows.push(row);
            }
        } else {
            rows.push(run_concurrent_moe(&vindex_path, &args, shards_str)?);
        }
    }

    print_table(&rows);

    // BW10 steady-state movement/causality summary — inert unless
    // LARQL_MOVEMENT_LEDGER is set. Printed after the table so the tok/s
    // figure and the movement attribution for the SAME run sit together;
    // reading a throughput delta without the attribution is what put a
    // scheduling win into the bandwidth column once already.
    larql_compute::movement_ledger::session::flush();

    // JSON output (ADR-0012).
    let want_json = args
        .output
        .as_deref()
        .map(|o| o.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
        || args.output_file.is_some();
    if want_json {
        let json_rows: Vec<BenchJsonRow> = rows
            .iter()
            .map(|r| BenchJsonRow {
                backend: r.backend.clone(),
                prefill_ms: r.prefill_ms,
                ms_per_tok: BenchJsonLatency {
                    mean: r.avg_decode_ms,
                    p50: r.p50_ms,
                    p99: r.p99_ms,
                },
                tok_per_s: r.tok_per_s,
                wire_bytes_per_tok: r.wire_bytes_per_tok,
                shard_efficiency: r.shard_efficiency,
                stages: r.stages.map(BenchJsonStages::from),
                n_steps: r.n_steps,
                note: r.note.clone(),
            })
            .collect();
        let result = BenchJsonResult {
            timestamp: {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                format!("{secs}")
            },
            model: vindex_path.display().to_string(),
            prompt: args.prompt.clone(),
            tokens: args.tokens,
            wire: args.wire.clone(),
            concurrent: args.concurrent,
            results: json_rows,
        };
        let json_str = serde_json::to_string_pretty(&result)?;
        if let Some(ref path) = args.output_file {
            std::fs::write(path, &json_str)?;
            eprintln!("[bench] JSON written to {path}");
        } else {
            println!("{json_str}");
        }
    }
    Ok(())
}

/// Whether the vindex at `dir` is a hybrid-MoE architecture (Gemma 4
/// 26B-A4B, 31B-MoE, …). Loads the Q4K model weights (mmap — cheap, no full
/// read) and probes the architecture. Returns `false` on any load error so a
/// non-MoE / unreadable vindex falls through to the standard CPU/engine paths
/// and surfaces its own error there.
fn vindex_is_hybrid_moe(dir: &std::path::Path) -> bool {
    let mut cb = larql_vindex::SilentLoadCallbacks;
    larql_vindex::load_model_weights_kquant(dir, &mut cb)
        .map(|w| w.arch.is_moe() || w.arch.is_hybrid_moe())
        .unwrap_or(false)
}

/// Configure rayon's global thread pool for the bench. Precedence:
/// 1. `RAYON_NUM_THREADS` env var if set (rayon's standard override).
/// 2. `--threads N` arg if non-zero.
/// 3. Auto: 8 on Apple silicon (M-series), rayon default elsewhere.
///
/// Apple silicon (M1/M2/M3/M4 Max) shows a clear sweet spot at 8
/// threads for memory-bandwidth-bound Q4_K matvec kernels. Past 8
/// threads, LPDDR5 channel contention plus rayon's lack of P-core
/// pinning causes a ~15% throughput regression vs the 8-thread
/// configuration. Documented in
/// `bench/baselines/cpu/DIAGNOSIS-2026-05-16-thread-scaling.md`.
fn configure_rayon_threads(threads_arg: usize) {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        // Rayon will read the env var on first pool access. Don't
        // build_global() here — we'd race with that lazy init.
        return;
    }
    let n_threads = if threads_arg > 0 {
        threads_arg
    } else {
        auto_default_threads()
    };
    if n_threads == 0 {
        return;
    }
    // build_global only succeeds once per process. Ignore errors —
    // they only happen when rayon's already been initialised (e.g.
    // from a parallel test harness or a previous `run` in the same
    // process), which is fine.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global();
}

/// Pick a sensible default thread count when the user hasn't set one.
/// Returns 0 to fall through to rayon's own default on unknown CPUs.
fn auto_default_threads() -> usize {
    // Apple silicon: 8 threads is the empirical optimum for the
    // Q4_K × Q8_K matvec on M3 Max LPDDR5. Confirmed across the
    // 12-P-core M3 Max (best 24.6 tok/s) — see thread-scaling
    // diagnosis doc. M1/M2/M4 are likely similar but unverified;
    // user can override with `--threads`.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        8
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        0
    }
}

/// Parse + validate the `--ffn-policy <spec>` flag value. Returns
/// `None` when the flag was omitted; `Some(validated)` when a spec
/// was provided. Surfaces parse / validation errors with a `--ffn-policy:`
/// prefix so the user sees which flag failed.
fn parse_ffn_policy(
    spec: Option<&str>,
    num_layers: usize,
) -> Result<Option<larql_inference::ffn_policy::ValidatedFfnLayerPolicy>, Box<dyn std::error::Error>>
{
    let Some(spec) = spec else {
        return Ok(None);
    };
    let validated = larql_inference::ffn_policy::FfnLayerPolicy::from_spec(spec)
        .map_err(|e| format!("--ffn-policy parse: {e}"))?
        .validate_for(num_layers)
        .map_err(|e| format!("--ffn-policy validation: {e}"))?;
    Ok(Some(validated))
}
