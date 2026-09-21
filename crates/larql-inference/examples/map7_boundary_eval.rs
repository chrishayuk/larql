//! MAP-7 boundary-map arm evaluator.
//!
//! Scores a fitted global boundary map (from bench/map7-kshape/
//! fit_global_map.py) under the frozen injection mode of
//! docs/preregistration/map7-kshape-prereg.md §4.2: **final position per
//! prompt, single-row splice** — the canonical forward supplies the full
//! target-boundary matrix, only the last row is replaced by the arm's
//! prediction, and the canonical suffix runs from there. Canonical and arm
//! logits come from the same process (paired by construction).
//!
//! Blinding (prereg §3.6/§4.1): evaluating the T3 pair (20,29) refuses
//! unless `t2-gate-pass.json` exists in the output directory — the T2
//! instrument gate must be recorded first.
//!
//! Env: LARQL_M7B_VINDEX, LARQL_M7B_CORPUS, LARQL_M7B_MAP (fitted map
//! .safetensors; its .json sidecar carries pair/mode/identity),
//! LARQL_M7B_OUT, LARQL_M7B_SPLIT (e.g. dev), LARQL_M7B_STRATUM
//! (e.g. general), optional LARQL_M7B_LIMIT.

use std::path::{Path, PathBuf};

use ndarray::{s, Array2};
use sha2::{Digest, Sha256};

use larql_compute::ffn::WeightFfn;
use larql_compute::forward::layer::apply_layer_scalar;
use larql_compute::forward::predict::raw::hidden_to_raw_logits;
use larql_compute::forward::{
    apply_per_layer_embedding, embed_tokens_pub, precompute_per_layer_inputs, run_attention,
    run_ffn,
};
use larql_models::ModelWeights;

/// Conjunctive faithfulness contract (prereg §3.3): top-1 match AND KL ≤ ε.
const KL_EPSILON_NATS: f64 = 0.10;
/// Containment metric depths (prereg §3.4).
const TOPK_CONTAIN: [usize; 2] = [5, 20];

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn refuse(msg: &str) -> ! {
    eprintln!("REFUSE: {msg}");
    std::process::exit(1);
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn vindex_digest(dir: &Path) -> String {
    let mut entries: Vec<(String, u64, bool)> = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| refuse(&format!("vindex dir: {e}"))) {
        let entry = entry.expect("dir entry");
        let meta = entry.metadata().expect("metadata");
        if meta.is_file() {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_json = name.ends_with(".json");
            entries.push((name, meta.len(), is_json));
        }
    }
    entries.sort();
    let mut h = Sha256::new();
    for (name, size, is_json) in &entries {
        h.update(format!("{name}:{size}\n").as_bytes());
        if *is_json {
            h.update(std::fs::read(dir.join(name)).expect("read json"));
        }
    }
    format!(
        "vdir-{}",
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    )
}

fn shellexpand_home(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = env("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}

// ---------------------------------------------------------------------------
// Fitted map
// ---------------------------------------------------------------------------

struct FittedMap {
    u: Array2<f32>,   // (d, r)
    vt: Array2<f32>,  // (r, d)
    x_mean: Vec<f32>, // (d)
    y_mean: Vec<f32>, // (d)
    delta_mode: bool,
    src: usize,
    tgt: usize,
    manifest: serde_json::Value,
}

fn load_map(path: &Path) -> FittedMap {
    let bytes = std::fs::read(path).unwrap_or_else(|e| refuse(&format!("read map: {e}")));
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| refuse(&format!("parse map: {e}")));
    let f32_tensor = |name: &str| -> (Vec<usize>, Vec<f32>) {
        let t = st
            .tensor(name)
            .unwrap_or_else(|_| refuse(&format!("map missing tensor {name}")));
        let vals: Vec<f32> = t
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (t.shape().to_vec(), vals)
    };
    let (us, uv) = f32_tensor("u");
    let (vs, vv) = f32_tensor("vt");
    let (_, xm) = f32_tensor("x_mean");
    let (_, ym) = f32_tensor("y_mean");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}.json", path.display()))
            .unwrap_or_else(|e| refuse(&format!("map sidecar json: {e}"))),
    )
    .unwrap_or_else(|e| refuse(&format!("map sidecar parse: {e}")));
    let pair = manifest["pair"]
        .as_array()
        .unwrap_or_else(|| refuse("map sidecar missing pair"));
    FittedMap {
        u: Array2::from_shape_vec((us[0], us[1]), uv).expect("u shape"),
        vt: Array2::from_shape_vec((vs[0], vs[1]), vv).expect("vt shape"),
        x_mean: xm,
        y_mean: ym,
        delta_mode: manifest["mode"].as_str() == Some("delta"),
        src: pair[0].as_u64().expect("src") as usize,
        tgt: pair[1].as_u64().expect("tgt") as usize,
        manifest,
    }
}

impl FittedMap {
    fn predict(&self, x: &[f32]) -> Vec<f32> {
        let d = x.len();
        let xc: Vec<f32> = (0..d).map(|i| x[i] - self.x_mean[i]).collect();
        let xc = Array2::from_shape_vec((1, d), xc).expect("xc");
        let low = xc.dot(&self.u); // (1, r)
        let y = low.dot(&self.vt); // (1, d)
        (0..d)
            .map(|i| self.y_mean[i] + y[[0, i]] + if self.delta_mode { x[i] } else { 0.0 })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Forward machinery (canonical pass + suffix resume; splice-floor-validated)
// ---------------------------------------------------------------------------

struct CanonicalPass {
    i_src_last: Vec<f32>,
    i_tgt: Array2<f32>,
    logits_last: Vec<f32>,
}

fn canonical_pass(weights: &ModelWeights, ids: &[u32], src: usize, tgt: usize) -> CanonicalPass {
    let ffn = WeightFfn { weights };
    let h0 = embed_tokens_pub(weights, ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h0, ids);
    let last = ids.len() - 1;
    let mut i_src_last = Vec::new();
    let mut i_tgt = None;
    let mut h = h0;
    for layer in 0..weights.num_layers {
        if layer == src {
            i_src_last = h.row(last).to_vec();
        }
        if layer == tgt {
            i_tgt = Some(h.clone());
        }
        h = step_layer(weights, &ffn, &ple_inputs, h, layer);
    }
    let logits_last = hidden_to_raw_logits(weights, &h.slice(s![last..last + 1, ..]).to_owned());
    CanonicalPass {
        i_src_last,
        i_tgt: i_tgt.expect("tgt boundary reached"),
        logits_last,
    }
}

fn step_layer(
    weights: &ModelWeights,
    ffn: &WeightFfn,
    ple_inputs: &[Array2<f32>],
    h: Array2<f32>,
    layer: usize,
) -> Array2<f32> {
    let h_post_attn = match run_attention(larql_models::WeightsView::dense(weights), &h, layer) {
        Some(pa) => pa,
        None => h.clone(),
    };
    let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
    let mut h_new = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
    apply_layer_scalar(weights, &mut h_new, layer);
    h_new
}

fn suffix_logits(
    weights: &ModelWeights,
    ids: &[u32],
    mut h: Array2<f32>,
    from_layer: usize,
) -> Vec<f32> {
    let ffn = WeightFfn { weights };
    let h0 = embed_tokens_pub(weights, ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h0, ids);
    for layer in from_layer..weights.num_layers {
        h = step_layer(weights, &ffn, &ple_inputs, h, layer);
    }
    let last = ids.len() - 1;
    hidden_to_raw_logits(weights, &h.slice(s![last..last + 1, ..]).to_owned())
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max
        + logits
            .iter()
            .map(|&v| ((v as f64) - max).exp())
            .sum::<f64>()
            .ln();
    logits.iter().map(|&v| v as f64 - lse).collect()
}

fn topk_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    order
}

struct ArmMetrics {
    top1_match: bool,
    kl_nats: f64,
    rank_of_native_top1: usize,
    containment: Vec<f64>,
    faithful: bool,
    margin_logit: f32,
}

fn score(canon: &[f32], arm: &[f32]) -> ArmMetrics {
    let lp = log_softmax(canon);
    let lq = log_softmax(arm);
    let kl: f64 = lp.iter().zip(&lq).map(|(&p, &q)| p.exp() * (p - q)).sum();
    let canon_top = topk_ids(canon, TOPK_CONTAIN[1].max(2));
    let native = canon_top[0];
    let margin_logit = canon[canon_top[0] as usize] - canon[canon_top[1] as usize];
    let mut arm_order: Vec<u32> = (0..arm.len() as u32).collect();
    arm_order.sort_unstable_by(|&a, &b| {
        arm[b as usize]
            .partial_cmp(&arm[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let rank_of_native_top1 = arm_order
        .iter()
        .position(|&i| i == native)
        .unwrap_or(usize::MAX);
    let top1_match = arm_order[0] == native;
    let containment = TOPK_CONTAIN
        .iter()
        .map(|&k| {
            let canon_k = &canon_top[..k];
            let arm_k = &arm_order[..k];
            canon_k.iter().filter(|i| arm_k.contains(i)).count() as f64 / k as f64
        })
        .collect();
    ArmMetrics {
        top1_match,
        kl_nats: kl,
        rank_of_native_top1,
        containment,
        faithful: top1_match && kl <= KL_EPSILON_NATS,
        margin_logit,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let vindex = PathBuf::from(shellexpand_home(
        &env("LARQL_M7B_VINDEX").unwrap_or_else(|| refuse("LARQL_M7B_VINDEX unset")),
    ));
    let corpus_dir = PathBuf::from(shellexpand_home(
        &env("LARQL_M7B_CORPUS").unwrap_or_else(|| refuse("LARQL_M7B_CORPUS unset")),
    ));
    let map_path = PathBuf::from(shellexpand_home(
        &env("LARQL_M7B_MAP").unwrap_or_else(|| refuse("LARQL_M7B_MAP unset")),
    ));
    let out_dir = PathBuf::from(shellexpand_home(
        &env("LARQL_M7B_OUT").unwrap_or_else(|| refuse("LARQL_M7B_OUT unset")),
    ));
    let split = env("LARQL_M7B_SPLIT").unwrap_or_else(|| refuse("LARQL_M7B_SPLIT unset"));
    let stratum = env("LARQL_M7B_STRATUM").unwrap_or_else(|| refuse("LARQL_M7B_STRATUM unset"));
    let limit: usize = env("LARQL_M7B_LIMIT")
        .map(|v| v.parse().expect("LIMIT"))
        .unwrap_or(usize::MAX);

    let map = load_map(&map_path);

    // Blinding: the T3 pair is locked until the T2 gate result is recorded.
    if (map.src, map.tgt) == (20, 29) && !out_dir.join("t2-gate-pass.json").exists() {
        refuse("T3 pair (20,29) is blinded until t2-gate-pass.json exists (prereg §3.6/§4.1)");
    }

    // Identity checks against the map's recorded fit identity.
    let prompts_bytes = std::fs::read(corpus_dir.join("prompts.jsonl"))
        .unwrap_or_else(|e| refuse(&format!("read corpus: {e}")));
    let corpus_sha = sha256_hex(&prompts_bytes);
    let fit_identity = &map.manifest["identity"];
    if fit_identity["prompts_sha256"].as_str() != Some(corpus_sha.as_str()) {
        refuse("corpus digest does not match the map's fit identity");
    }
    let vdigest = vindex_digest(&vindex);
    if fit_identity["vindex_digest"].as_str() != Some(vdigest.as_str()) {
        refuse("vindex digest does not match the map's fit identity");
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_with_opts(
        &vindex,
        &mut cb,
        larql_vindex::LoadWeightsOptions::default(),
    )
    .expect("load weights");
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex).expect("load tokenizer");

    let mut rows = Vec::new();
    let mut per_prompt = Vec::new();
    for line in String::from_utf8(prompts_bytes).expect("utf8").lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("corpus line");
        if v["split"].as_str() != Some(split.as_str())
            || v["stratum"].as_str() != Some(stratum.as_str())
        {
            continue;
        }
        rows.push(v);
        if rows.len() >= limit {
            break;
        }
    }
    if rows.is_empty() {
        refuse(&format!("no prompts for {split}/{stratum}"));
    }
    println!(
        "evaluating arm '{}' pair ({},{}) on {}/{}: {} prompts",
        map.manifest["arm"].as_str().unwrap_or("?"),
        map.src,
        map.tgt,
        split,
        stratum,
        rows.len()
    );

    let started = std::time::Instant::now();
    for (n, v) in rows.iter().enumerate() {
        let prompt_id = v["prompt_id"].as_str().expect("prompt_id");
        let text = v["text"].as_str().expect("text");
        let ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, text)
            .unwrap_or_else(|e| refuse(&format!("tokenize {prompt_id}: {e:?}")));
        let canon = canonical_pass(&weights, &ids, map.src, map.tgt);
        let y_hat = map.predict(&canon.i_src_last);
        let mut spliced = canon.i_tgt.clone();
        let last = ids.len() - 1;
        spliced
            .row_mut(last)
            .iter_mut()
            .zip(&y_hat)
            .for_each(|(dst, &v)| *dst = v);
        let arm_logits = suffix_logits(&weights, &ids, spliced, map.tgt);
        let m = score(&canon.logits_last, &arm_logits);
        per_prompt.push(serde_json::json!({
            "prompt_id": prompt_id,
            "entity": v["entity"], "family": v["family"], "answer": v["answer"],
            "top1_match": m.top1_match, "kl_nats": m.kl_nats,
            "rank_of_native_top1": m.rank_of_native_top1,
            "containment_top5": m.containment[0], "containment_top20": m.containment[1],
            "faithful": m.faithful, "margin_logit": m.margin_logit,
        }));
        if (n + 1) % 50 == 0 {
            println!(
                "  {}/{} ({:.1}s elapsed)",
                n + 1,
                rows.len(),
                started.elapsed().as_secs_f32()
            );
        }
    }

    // Aggregate + mandatory margin-tertile table (prereg §3.5).
    let n = per_prompt.len();
    let rate = |f: &dyn Fn(&serde_json::Value) -> bool| {
        per_prompt.iter().filter(|r| f(r)).count() as f64 / n as f64
    };
    let top1_rate = rate(&|r| r["top1_match"].as_bool() == Some(true));
    let faithful_rate = rate(&|r| r["faithful"].as_bool() == Some(true));
    let mut kls: Vec<f64> = per_prompt
        .iter()
        .map(|r| r["kl_nats"].as_f64().unwrap())
        .collect();
    kls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_kl = kls[n / 2];

    let mut margins: Vec<f64> = per_prompt
        .iter()
        .map(|r| r["margin_logit"].as_f64().unwrap())
        .collect();
    margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t1 = margins[n / 3];
    let t2 = margins[2 * n / 3];
    let tertile_rates: Vec<serde_json::Value> = [(f64::MIN, t1), (t1, t2), (t2, f64::MAX)]
        .iter()
        .map(|&(lo, hi)| {
            let sel: Vec<_> = per_prompt
                .iter()
                .filter(|r| {
                    let m = r["margin_logit"].as_f64().unwrap();
                    m >= lo && m < hi
                })
                .collect();
            let tn = sel.len().max(1);
            serde_json::json!({
                "n": sel.len(),
                "top1_rate": sel.iter().filter(|r| r["top1_match"].as_bool() == Some(true)).count() as f64 / tn as f64,
                "faithful_rate": sel.iter().filter(|r| r["faithful"].as_bool() == Some(true)).count() as f64 / tn as f64,
            })
        })
        .collect();

    let summary = serde_json::json!({
        "arm": map.manifest["arm"],
        "pair": [map.src, map.tgt],
        "split": split, "stratum": stratum, "n": n,
        "top1_rate": top1_rate,
        "faithful_rate": faithful_rate,
        "median_kl_nats": median_kl,
        "kl_epsilon": KL_EPSILON_NATS,
        "margin_tertile_boundaries": [t1, t2],
        "tertiles_low_mid_high": tertile_rates,
        "injection_mode": "final-position single-row splice",
        "map_manifest": map.manifest,
        "vindex_digest": vdigest,
        "corpus_sha256": corpus_sha,
    });
    std::fs::create_dir_all(&out_dir).expect("mkdir out");
    let stem = format!(
        "eval-{}-{}-{}-{}",
        map.manifest["arm"].as_str().unwrap_or("arm"),
        map.src,
        map.tgt,
        format_args!("{split}-{stratum}")
    );
    std::fs::write(
        out_dir.join(format!("{stem}.jsonl")),
        per_prompt
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .expect("write jsonl");
    std::fs::write(
        out_dir.join(format!("{stem}.summary.json")),
        serde_json::to_string_pretty(&summary).unwrap() + "\n",
    )
    .expect("write summary");
    println!("n {n}  top1 {top1_rate:.4}  faithful {faithful_rate:.4}  medianKL {median_kl:.4}");
    println!(
        "summary: {}",
        out_dir.join(format!("{stem}.summary.json")).display()
    );
}
