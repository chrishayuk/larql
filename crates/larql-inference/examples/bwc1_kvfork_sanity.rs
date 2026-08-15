//! BW-C1 prerequisite: validate KV-fork mechanics on the resident CPU
//! MoE decode path before trusting anything built on top of it.
//!
//! Mirrors the R0-R5 gate discipline from `larql-kv`'s
//! `semantic_promotion` checkpoint/replay work
//! (`crates/larql-kv/src/model_walk/tests/replay_gates.rs`), reduced to
//! the two gates that matter here:
//!
//! - **R1 (null case)**: capture a checkpoint, decode N tokens from it
//!   TWICE with nothing armed. The two replays must be BIT-IDENTICAL —
//!   if a bare restore already perturbs the trajectory, nothing
//!   downstream means anything.
//! - **R4 (control)**: capture a checkpoint, decode N tokens once
//!   clean and once with a real oracle ablation armed. The two runs
//!   MUST diverge — a harness where the ablation never does anything
//!   would pass R1 vacuously and prove nothing.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwc1_kvfork_sanity -- \
//!     --vindex /path/to/gpt-oss-20b-q4k.vindex

use std::path::PathBuf;

use larql_compute::cpu::ops::moe::expert_override;
use larql_inference::ffn::LocalMoeFfn;
use larql_inference::kv_engine::PerLayerKvAccess;
use larql_kv::engines::semantic_promotion::checkpoint::BoundaryCheckpoint;
use larql_kv::engines::semantic_promotion::ids::CheckpointId;
use larql_kv::AnyEngine;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn per_layer_kv(engine: &mut AnyEngine) -> Option<&mut dyn PerLayerKvAccess> {
    match engine {
        AnyEngine::Kv(e) => e.per_layer_kv_mut(),
        AnyEngine::Retrieval(_) => None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = PathBuf::from(
        std::env::var("HOME").unwrap_or_default() + "/chris-models/gpt-oss-20b-q4k.vindex",
    );
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" {
            i += 1;
            vindex_path = PathBuf::from(&args[i]);
        }
        i += 1;
    }
    if !vindex_path.is_dir() {
        eprintln!("vindex directory not found: {}", vindex_path.display());
        std::process::exit(1);
    }

    println!("=== BW-C1 prerequisite: KV-fork sanity (R1 null case, R4 control) ===\n");

    let mut cb = SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
    for layer in 0..weights.num_layers {
        larql_inference::vindex::insert_q4k_layer_tensors_resident(&mut weights, &index, layer)?;
    }
    let weights_ref = &weights;
    let moe_ffn = LocalMoeFfn {
        weights: weights_ref,
        index: Some(&index),
    };

    let prompt = "The history of the Roman Empire began when";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

    let mut engine = larql_kv::EngineKind::from_name("standard")
        .expect("standard engine exists")
        .build(larql_inference::cpu_engine_backend());

    // Prefill puts the engine into PrefillDispatchMode::PerLayer
    // unconditionally (`StandardEngine::do_prefill`), which is what
    // makes `per_layer_kv_mut` return `Some` afterward.
    let _h = engine
        .prefill_resident(weights_ref, &moe_ffn, &index, &prompt_ids)
        .map_err(|e| format!("prefill failed: {e}"))?;
    println!("prefilled {} prompt tokens", prompt_ids.len());

    // Capture right after prefill — position 0 of generation. Greedy
    // decode can fall into a repetition attractor a few steps in (seen
    // empirically: at +3 steps here, ALL 96 single-expert ablations
    // across all 24 layers left a 6-token repeat of the same token
    // unchanged — not a restore bug, `replay_a` itself already showed
    // the repeat before any restore ran, but a bad position for a
    // "prove the harness is sensitive" gate: repetition attractors are
    // robust to small perturbations by construction). Position 0 is
    // less likely to have degenerated yet.
    let current = *prompt_ids.last().unwrap();

    let kv = per_layer_kv(&mut engine).ok_or("per_layer_kv_mut returned None — R1 cannot run")?;
    let ckpt = BoundaryCheckpoint::capture(CheckpointId::from_counter(1), kv)
        .map_err(|e| format!("capture failed: {e:?}"))?;
    println!(
        "captured checkpoint: {} layers, logical_next_position={}\n",
        ckpt.per_layer.len(),
        ckpt.logical_next_position
    );

    const N: usize = 6;

    // ── R1: two clean replays from the SAME checkpoint must be
    // bit-identical. ──
    let replay_a = decode_n(&mut engine, weights_ref, &moe_ffn, &index, current, N);
    restore(&mut engine, &ckpt)?;
    let replay_b = decode_n(&mut engine, weights_ref, &moe_ffn, &index, current, N);
    restore(&mut engine, &ckpt)?;

    let r1_pass = replay_a == replay_b;
    println!("R1 (null case): replay_a={replay_a:?}");
    println!("R1 (null case): replay_b={replay_b:?}");
    println!("R1 PASS: {r1_pass}\n");
    if !r1_pass {
        return Err(
            "R1 FAILED — a bare restore already perturbs the trajectory; \
                     KV-fork is not sound on this path, stop here"
                .into(),
        );
    }

    // ── R4: capture the real routing at this position, then try several
    // ablation targets until one DIVERGES. Requiring the FIRST candidate
    // to diverge would be the wrong gate — BW-C already established that
    // roughly half of real ablations leave the trajectory unchanged, so
    // a "safe" first pick is real evidence the harness works, not a
    // failure. R4's actual job is proving the harness CAN detect a real
    // divergence when one exists, which needs trying candidates, not
    // trusting the first. ──
    expert_override::start_observing();
    let _ = decode_n(&mut engine, weights_ref, &moe_ffn, &index, current, 1);
    let observed = expert_override::stop_observing();
    restore(&mut engine, &ckpt)?;
    if observed.is_empty() {
        return Err("no expert calls observed at the checkpoint position".into());
    }

    let mut r4_pass = false;
    for obs in &observed {
        expert_override::arm_once(obs.layer, obs.expert);
        let replay_ablated = decode_n(&mut engine, weights_ref, &moe_ffn, &index, current, N);
        let fired = expert_override::fired();
        expert_override::disarm();
        restore(&mut engine, &ckpt)?;

        let diverged = replay_a != replay_ablated;
        println!(
            "R4 candidate: layer={} expert={} weight={:.4} \
             fired={fired} diverged={diverged} ablated={replay_ablated:?}",
            obs.layer, obs.expert, obs.router_weight
        );
        if fired && diverged {
            r4_pass = true;
            break;
        }
    }
    println!("R4 (control): clean ={replay_a:?}");
    println!("R4 PASS: {r4_pass}\n");

    if r1_pass && r4_pass {
        println!("Both gates pass — KV-fork is sound on this path, safe to build BW-C1 on.");
        Ok(())
    } else {
        Err(
            "R4 FAILED — either the override never fired, or firing it made no observable \
             difference; the harness cannot distinguish a real null result from a broken one"
                .into(),
        )
    }
}

fn restore(
    engine: &mut AnyEngine,
    ckpt: &BoundaryCheckpoint,
) -> Result<(), Box<dyn std::error::Error>> {
    let kv = per_layer_kv(engine).ok_or("per_layer_kv_mut returned None on restore")?;
    ckpt.restore(kv)
        .map_err(|e| format!("restore failed: {e:?}"))?;
    Ok(())
}

fn decode_n(
    engine: &mut AnyEngine,
    weights: &larql_inference::ModelWeights,
    ffn: &LocalMoeFfn,
    index: &VectorIndex,
    mut current: u32,
    n: usize,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let h = engine
            .decode_step_resident(weights, ffn, index, current)
            .expect("decode_step_resident failed");
        current = argmax_token(weights, &h);
        out.push(current);
    }
    out
}

fn argmax_token(weights: &larql_inference::ModelWeights, h: &ndarray::Array2<f32>) -> u32 {
    let logits = larql_inference::research::hidden_to_raw_logits(weights, h);
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
