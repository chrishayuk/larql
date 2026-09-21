//! MAP-7 / K-SHAPE shared corpus capture harness.
//!
//! Implements docs/preregistration/map7-kshape-capture-spec.md: per sampled
//! position it persists all 35 boundary residuals `i_L` (residual entering
//! layer L; i_0 = embedding, i_34 = final pre-norm), FFN-input residuals at
//! the six frozen probe layers, top-256 canonical logits, and per-position
//! metrics — everything in f32, safetensors shards with JSONL sidecars.
//! Gate/up/tail intermediates are never persisted; a 0.1% subsample of
//! post-GEGLU activation rows is checksummed for recompute parity.
//!
//! Modes (env `LARQL_M7K_MODE`):
//!   `selftest` — smoke determinism, recompute parity, splice floor; writes
//!               `selftest-report.json`. Must pass before capture.
//!   `capture`  — bulk capture, append-only by shard, resumable.
//!
//! Env: LARQL_M7K_VINDEX, LARQL_M7K_CORPUS, LARQL_M7K_OUT,
//!      LARQL_M7K_MODE, optional LARQL_M7K_MODEL_DIGEST, LARQL_M7K_GIT_SHA,
//!      optional LARQL_M7K_SHARD_FILTER (substring; capture only matching shards).
//!
//! Usage:
//!   LARQL_M7K_VINDEX=~/chris-models/gemma3-4b-f16.vindex \
//!   LARQL_M7K_CORPUS=bench/map7-kshape/corpus-v1 \
//!   LARQL_M7K_OUT=~/chris-models/map7-kshape-capture-v1 \
//!   LARQL_M7K_MODE=selftest \
//!   cargo run --release -p larql-inference --example map7_kshape_capture

use std::collections::BTreeMap;
use std::io::Write as _;
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

/// Frozen probe layers (capture spec §6.3).
const PROBE_LAYERS: [usize; 6] = [3, 9, 17, 21, 26, 31];
/// Top-K logits persisted per position (capture spec §5).
const TOP_LOGITS: usize = 256;
/// Position rule (capture spec §4.2): final min(MAX_POS, len - MIN_POS_INDEX).
const MAX_POS: usize = 32;
const MIN_POS_INDEX: usize = 8;
/// Full-vocab logit audit subsample: 5% of positions (per-mille units).
const AUDIT_PERMILLE: u64 = 50;
/// Recompute-parity checksum subsample: 0.1% of (position, probe-layer) cells.
const CHECKSUM_PERMILLE: u64 = 1;
/// Prompts per shard (capture spec §5: keeps the write buffer ~1.5 GB).
const SHARD_SIZE: usize = 128;
/// Refuse bulk capture below this free-disk floor (capture spec §5.1).
const MIN_FREE_GB: u64 = 120;
/// Smoke-set size for self-tests.
const SMOKE_N: usize = 8;
const CORPUS_VERSION: &str = "map7-kshape-corpus-v1";

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
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// FNV-1a 64 — stable cell-selection hash (audit + checksum subsampling).
fn stable_hash(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for p in parts {
        for b in p.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0x1f;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct PromptRec {
    prompt_id: String,
    stratum: String,
    split: String,
    domain: Option<String>,
    family: Option<String>,
    entity: Option<String>,
    answer: Option<String>,
    text: String,
}

struct Corpus {
    prompts: Vec<PromptRec>,
    prompts_sha256: String,
}

fn load_corpus(dir: &Path) -> Corpus {
    let prompts_path = dir.join("prompts.jsonl");
    let manifest_path = dir.join("manifest.json");
    let bytes = std::fs::read(&prompts_path)
        .unwrap_or_else(|e| refuse(&format!("cannot read {}: {e}", prompts_path.display())));
    let actual_sha = sha256_hex(&bytes);
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| refuse(&format!("cannot read {}: {e}", manifest_path.display()))),
    )
    .unwrap_or_else(|e| refuse(&format!("bad corpus manifest JSON: {e}")));
    let expected_sha = manifest["prompts_sha256"].as_str().unwrap_or_default();
    if expected_sha != actual_sha {
        refuse(&format!(
            "corpus digest mismatch: manifest {expected_sha} vs file {actual_sha}"
        ));
    }
    if manifest["corpus_version"].as_str() != Some(CORPUS_VERSION) {
        refuse("corpus_version mismatch");
    }
    let mut prompts = Vec::new();
    for line in String::from_utf8(bytes).expect("utf8 corpus").lines() {
        let v: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| refuse(&format!("bad corpus line: {e}")));
        let sget = |k: &str| v[k].as_str().map(|s| s.to_string());
        prompts.push(PromptRec {
            prompt_id: sget("prompt_id").unwrap_or_else(|| refuse("missing prompt_id")),
            stratum: sget("stratum").unwrap_or_else(|| refuse("missing stratum")),
            split: sget("split").unwrap_or_else(|| refuse("missing split")),
            domain: sget("domain"),
            family: sget("family"),
            entity: sget("entity"),
            answer: sget("answer"),
            text: sget("text").unwrap_or_else(|| refuse("missing text")),
        });
    }
    Corpus {
        prompts,
        prompts_sha256: actual_sha,
    }
}

// ---------------------------------------------------------------------------
// Identity: vindex digest, model digest, free disk
// ---------------------------------------------------------------------------

/// Fast container identity: SHA-256 over sorted "name:size" lines of the
/// vindex directory plus the full content of every *.json file inside it.
fn vindex_digest(dir: &Path) -> String {
    let mut entries: Vec<(String, u64, bool)> = Vec::new();
    let rd = std::fs::read_dir(dir)
        .unwrap_or_else(|e| refuse(&format!("cannot read vindex dir {}: {e}", dir.display())));
    for entry in rd {
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
    format!("vdir-{}", hex(&h.finalize()))
}

fn model_digest() -> String {
    if let Some(d) = env("LARQL_M7K_MODEL_DIGEST") {
        return d;
    }
    let home = env("HOME").unwrap_or_else(|| refuse("HOME unset"));
    let refs =
        PathBuf::from(home).join(".cache/huggingface/hub/models--google--gemma-3-4b-it/refs/main");
    match std::fs::read_to_string(&refs) {
        Ok(s) => s.trim().to_string(),
        Err(_) => refuse(
            "cannot resolve model digest: set LARQL_M7K_MODEL_DIGEST or ensure the HF cache ref exists",
        ),
    }
}

fn free_disk_gb(dir: &Path) -> u64 {
    let out = std::process::Command::new("df")
        .arg("-k")
        .arg(dir)
        .output()
        .unwrap_or_else(|e| refuse(&format!("df failed: {e}")));
    let text = String::from_utf8_lossy(&out.stdout);
    let last = text.lines().last().unwrap_or_default();
    let avail_kb: u64 = last
        .split_whitespace()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| refuse(&format!("cannot parse df output: {last}")));
    avail_kb / (1024 * 1024)
}

// ---------------------------------------------------------------------------
// Forward capture
// ---------------------------------------------------------------------------

struct PosMetrics {
    pos: usize,
    top_ids: Vec<u32>,
    top_vals: Vec<f32>,
    margin_logit: f32,
    top1_prob: f32,
    entropy_nats: f32,
    realized_next_id: Option<u32>,
    nll_next: Option<f32>,
    audit_full: bool,
    full_logits: Option<Vec<f32>>,
}

struct PromptCapture {
    /// (n_pos, 35, hidden) row-major.
    boundaries: Vec<f32>,
    /// (n_pos, 6, hidden) row-major.
    ffn_in: Vec<f32>,
    positions: Vec<PosMetrics>,
    /// (prompt_id, pos, layer, sha256-of-activation-row) checksum cells.
    checksums: Vec<(String, usize, usize, String)>,
    /// Full-seq boundary i_29, kept only when `keep_i29` (splice-floor test).
    i29_full: Option<Array2<f32>>,
    seq_len: usize,
}

fn selected_positions(seq_len: usize) -> Vec<usize> {
    if seq_len <= MIN_POS_INDEX {
        return Vec::new();
    }
    let n = MAX_POS.min(seq_len - MIN_POS_INDEX);
    (seq_len - n..seq_len).collect()
}

fn capture_prompt(
    weights: &ModelWeights,
    token_ids: &[u32],
    prompt_id: &str,
    keep_i29: bool,
) -> PromptCapture {
    let hidden = weights.hidden_size;
    let n_layers = weights.num_layers;
    assert_eq!(
        n_layers, 34,
        "zone map and boundary layout assume Gemma 3 4B"
    );
    let seq_len = token_ids.len();
    let sel = selected_positions(seq_len);
    if sel.is_empty() {
        refuse(&format!(
            "{prompt_id}: no sampleable positions (len {seq_len})"
        ));
    }

    let ffn = WeightFfn { weights };
    let h0 = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h0, token_ids);

    let n_pos = sel.len();
    let mut boundaries = vec![0.0f32; n_pos * (n_layers + 1) * hidden];
    let mut ffn_in = vec![0.0f32; n_pos * PROBE_LAYERS.len() * hidden];
    let mut checksums = Vec::new();
    let mut i29_full = None;

    let copy_rows = |dst: &mut [f32], slot: usize, slots: usize, h: &Array2<f32>| {
        for (pi, &p) in sel.iter().enumerate() {
            let row = h.row(p);
            let off = (pi * slots + slot) * hidden;
            dst[off..off + hidden].copy_from_slice(row.as_slice().expect("contiguous row"));
        }
    };

    let mut h = h0;
    for layer in 0..n_layers {
        copy_rows(&mut boundaries, layer, n_layers + 1, &h);
        if keep_i29 && layer == 29 {
            i29_full = Some(h.clone());
        }
        let h_post_attn = match run_attention(larql_models::WeightsView::dense(weights), &h, layer)
        {
            Some(pa) => pa,
            None => h.clone(),
        };
        let probe_slot = PROBE_LAYERS.iter().position(|&l| l == layer);
        if let Some(slot) = probe_slot {
            copy_rows(&mut ffn_in, slot, PROBE_LAYERS.len(), &h_post_attn);
        }
        let want_checksum = probe_slot.is_some()
            && sel.iter().any(|&p| {
                stable_hash(&[prompt_id, &p.to_string(), &layer.to_string()]) % 1000
                    < CHECKSUM_PERMILLE
            });
        let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, &ffn, want_checksum);
        if want_checksum {
            let act = activation.as_ref().expect("activation requested");
            for &p in &sel {
                if stable_hash(&[prompt_id, &p.to_string(), &layer.to_string()]) % 1000
                    < CHECKSUM_PERMILLE
                {
                    let row = act.row(p);
                    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                    checksums.push((prompt_id.to_string(), p, layer, sha256_hex(&bytes)));
                }
            }
        }
        let mut h_new =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_new, layer);
        h = h_new;
    }
    copy_rows(&mut boundaries, n_layers, n_layers + 1, &h);

    let mut positions = Vec::with_capacity(n_pos);
    for &p in &sel {
        let row = h.slice(s![p..p + 1, ..]).to_owned();
        let logits = hidden_to_raw_logits(weights, &row);
        positions.push(position_metrics(prompt_id, p, token_ids, logits));
    }

    PromptCapture {
        boundaries,
        ffn_in,
        positions,
        checksums,
        i29_full,
        seq_len,
    }
}

fn position_metrics(
    prompt_id: &str,
    pos: usize,
    token_ids: &[u32],
    logits: Vec<f32>,
) -> PosMetrics {
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_ids: Vec<u32> = order[..TOP_LOGITS].to_vec();
    let top_vals: Vec<f32> = top_ids.iter().map(|&i| logits[i as usize]).collect();
    let margin_logit = top_vals[0] - top_vals[1];

    // Log-sum-exp once; entropy, top-1 prob, and teacher-forced NLL from it.
    let max = top_vals[0];
    let mut sum_exp = 0.0f64;
    let mut sum_p_logit = 0.0f64;
    for &v in &logits {
        let e = ((v - max) as f64).exp();
        sum_exp += e;
        sum_p_logit += e * v as f64;
    }
    let lse = max as f64 + sum_exp.ln();
    let entropy_nats = (lse - sum_p_logit / sum_exp) as f32;
    let top1_prob = ((top_vals[0] as f64 - lse).exp()) as f32;
    let realized_next_id = token_ids.get(pos + 1).copied();
    let nll_next = realized_next_id.map(|t| (lse - logits[t as usize] as f64) as f32);
    let audit_full = stable_hash(&["audit", prompt_id, &pos.to_string()]) % 1000 < AUDIT_PERMILLE;
    PosMetrics {
        pos,
        top_ids,
        top_vals,
        margin_logit,
        top1_prob,
        entropy_nats,
        realized_next_id,
        nll_next,
        audit_full,
        full_logits: if audit_full { Some(logits) } else { None },
    }
}

// ---------------------------------------------------------------------------
// Shard serialization
// ---------------------------------------------------------------------------

struct ShardBuf {
    boundaries: Vec<f32>,
    ffn_in: Vec<f32>,
    top_ids: Vec<u32>,
    top_vals: Vec<f32>,
    meta_lines: Vec<String>,
    audit_logits: Vec<f32>,
    audit_meta: Vec<String>,
    checksum_lines: Vec<String>,
    rows: usize,
    vocab: usize,
    hidden: usize,
}

impl ShardBuf {
    fn new(hidden: usize, vocab: usize) -> Self {
        ShardBuf {
            boundaries: Vec::new(),
            ffn_in: Vec::new(),
            top_ids: Vec::new(),
            top_vals: Vec::new(),
            meta_lines: Vec::new(),
            audit_logits: Vec::new(),
            audit_meta: Vec::new(),
            checksum_lines: Vec::new(),
            rows: 0,
            vocab,
            hidden,
        }
    }

    fn push(&mut self, rec: &PromptRec, cap: &PromptCapture) {
        self.boundaries.extend_from_slice(&cap.boundaries);
        self.ffn_in.extend_from_slice(&cap.ffn_in);
        for pm in &cap.positions {
            self.top_ids.extend_from_slice(&pm.top_ids);
            self.top_vals.extend_from_slice(&pm.top_vals);
            let meta = serde_json::json!({
                "prompt_id": rec.prompt_id, "stratum": rec.stratum,
                "split": rec.split, "domain": rec.domain, "family": rec.family,
                "entity": rec.entity, "answer": rec.answer,
                "seq_len": cap.seq_len, "pos": pm.pos,
                "top1_id": pm.top_ids[0], "margin_logit": pm.margin_logit,
                "top1_prob": pm.top1_prob, "entropy_nats": pm.entropy_nats,
                "realized_next_id": pm.realized_next_id, "nll_next": pm.nll_next,
                "audit_full": pm.audit_full,
            });
            self.meta_lines.push(meta.to_string());
            if let Some(full) = &pm.full_logits {
                self.audit_logits.extend_from_slice(full);
                self.audit_meta.push(
                    serde_json::json!({"prompt_id": rec.prompt_id, "pos": pm.pos}).to_string(),
                );
            }
            self.rows += 1;
        }
        for (pid, pos, layer, sha) in &cap.checksums {
            self.checksum_lines.push(
                serde_json::json!({"prompt_id": pid, "pos": pos, "layer": layer,
                                   "sha256_activation_f32le": sha})
                .to_string(),
            );
        }
    }

    fn serialize_main(&self) -> Vec<u8> {
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let u32b = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let b_bytes = f32b(&self.boundaries);
        let f_bytes = f32b(&self.ffn_in);
        let tv_bytes = f32b(&self.top_vals);
        let ti_bytes = u32b(&self.top_ids);
        let tensors: Vec<(&str, safetensors::tensor::TensorView)> = vec![
            (
                "boundaries",
                safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    vec![self.rows, 35, self.hidden],
                    &b_bytes,
                )
                .expect("boundaries view"),
            ),
            (
                "ffn_in",
                safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    vec![self.rows, PROBE_LAYERS.len(), self.hidden],
                    &f_bytes,
                )
                .expect("ffn_in view"),
            ),
            (
                "logits_top_v",
                safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    vec![self.rows, TOP_LOGITS],
                    &tv_bytes,
                )
                .expect("top_v view"),
            ),
            (
                "logits_top_i",
                safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::U32,
                    vec![self.rows, TOP_LOGITS],
                    &ti_bytes,
                )
                .expect("top_i view"),
            ),
        ];
        safetensors::serialize(tensors, None).expect("serialize shard")
    }

    fn serialize_audit(&self) -> Option<Vec<u8>> {
        if self.audit_meta.is_empty() {
            return None;
        }
        let bytes: Vec<u8> = self
            .audit_logits
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::tensor::Dtype::F32,
            vec![self.audit_meta.len(), self.vocab],
            &bytes,
        )
        .expect("audit view");
        Some(safetensors::serialize(vec![("logits_full", view)], None).expect("serialize audit"))
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)
        .unwrap_or_else(|e| refuse(&format!("write {}: {e}", tmp.display())));
    std::fs::rename(&tmp, path)
        .unwrap_or_else(|e| refuse(&format!("rename {}: {e}", path.display())));
}

// ---------------------------------------------------------------------------
// Batch capture — shared by self-tests and bulk capture. Prompt order in
// the output is the input order regardless of `parallel` (rayon's ordered
// collect), and each prompt's computation is independent, so parallel and
// sequential batches serialize byte-identically (asserted by self-test 1).
// ---------------------------------------------------------------------------

fn capture_batch(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    recs: &[&PromptRec],
    parallel: bool,
) -> ShardBuf {
    use rayon::prelude::*;
    let one = |rec: &&PromptRec| -> PromptCapture {
        let ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, &rec.text)
            .unwrap_or_else(|e| refuse(&format!("tokenize {}: {e:?}", rec.prompt_id)));
        capture_prompt(weights, &ids, &rec.prompt_id, false)
    };
    let caps: Vec<PromptCapture> = if parallel {
        recs.par_iter().map(one).collect()
    } else {
        recs.iter().map(one).collect()
    };
    let mut buf = ShardBuf::new(weights.hidden_size, weights.vocab_size);
    for (rec, cap) in recs.iter().zip(&caps) {
        buf.push(rec, cap);
    }
    buf
}

// ---------------------------------------------------------------------------
// Self-tests
// ---------------------------------------------------------------------------

fn run_selftests(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    corpus: &Corpus,
    out_dir: &Path,
    identity: &serde_json::Value,
) {
    let smoke: Vec<&PromptRec> = corpus.prompts.iter().take(SMOKE_N).collect();

    // 1. Determinism: sequential and parallel captures of the smoke set
    //    must serialize byte-identically (licenses parallel bulk capture).
    let mut serialized = Vec::new();
    for parallel in [false, true] {
        let buf = capture_batch(weights, tokenizer, &smoke, parallel);
        println!(
            "selftest determinism {} round: {} rows",
            if parallel { "parallel" } else { "sequential" },
            buf.rows
        );
        serialized.push(buf.serialize_main());
    }
    let determinism = serialized[0] == serialized[1];
    if !determinism {
        refuse("determinism self-test FAILED: sequential and parallel captures differ byte-wise");
    }
    println!("selftest determinism: PASS (sequential == parallel)");

    // 2. Recompute parity: activation rows recomputed via the independent
    //    capture_ffn_activation_matrix path must match the run_ffn capture.
    let mut parity_cells = 0usize;
    for rec in &smoke {
        let ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, &rec.text)
            .unwrap_or_else(|e| refuse(&format!("tokenize {}: {e:?}", rec.prompt_id)));
        let sel = selected_positions(ids.len());
        for &layer in PROBE_LAYERS.iter().take(2) {
            let full =
                larql_inference::forward::capture_ffn_activation_matrix(weights, &ids, layer)
                    .unwrap_or_else(|| refuse("capture_ffn_activation_matrix returned None"));
            // Reference from the harness's own loop, via a fresh capture with
            // every cell of this (prompt, layer) checksummed.
            let ffn = WeightFfn { weights };
            let h0 = embed_tokens_pub(weights, &ids);
            let ple_inputs = precompute_per_layer_inputs(weights, &h0, &ids);
            let mut h = h0;
            for l in 0..=layer {
                let h_post_attn =
                    match run_attention(larql_models::WeightsView::dense(weights), &h, l) {
                        Some(pa) => pa,
                        None => h.clone(),
                    };
                let (h_post_ffn, act) = run_ffn(weights, &h_post_attn, l, &ffn, l == layer);
                if l == layer {
                    let act = act.expect("activation");
                    for &p in &sel {
                        let a = act.row(p);
                        let b = full.row(p);
                        if a != b {
                            refuse(&format!(
                                "recompute parity FAILED at {} pos {p} layer {layer}",
                                rec.prompt_id
                            ));
                        }
                        parity_cells += 1;
                    }
                }
                let mut h_new =
                    apply_per_layer_embedding(weights, &h_post_ffn, l, ple_inputs.get(l));
                apply_layer_scalar(weights, &mut h_new, l);
                h = h_new;
            }
        }
    }
    println!("selftest recompute parity: PASS ({parity_cells} cells bit-identical)");

    // 3. Splice floor: resume from captured i_29, logits must be bit-identical.
    let mut floor_prompts = 0usize;
    for rec in &smoke {
        let ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, &rec.text)
            .unwrap_or_else(|e| refuse(&format!("tokenize {}: {e:?}", rec.prompt_id)));
        let cap = capture_prompt(weights, &ids, &rec.prompt_id, true);
        let i29 = cap.i29_full.expect("i29 kept");
        let ffn = WeightFfn { weights };
        let h0 = embed_tokens_pub(weights, &ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h0, &ids);
        let mut h = i29;
        for layer in 29..weights.num_layers {
            let h_post_attn =
                match run_attention(larql_models::WeightsView::dense(weights), &h, layer) {
                    Some(pa) => pa,
                    None => h.clone(),
                };
            let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
            let mut h_new =
                apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
            apply_layer_scalar(weights, &mut h_new, layer);
            h = h_new;
        }
        let last = ids.len() - 1;
        let resumed = hidden_to_raw_logits(weights, &h.slice(s![last..last + 1, ..]).to_owned());
        let canon_row = {
            // Canonical logits recomputed from the capture's own stored final
            // boundary (i_34) row for the last selected position.
            let sel = selected_positions(ids.len());
            let pi = sel.iter().position(|&p| p == last).expect("last selected");
            let hlen = weights.hidden_size;
            let off = (pi * 35 + 34) * hlen;
            let row = Array2::from_shape_vec((1, hlen), cap.boundaries[off..off + hlen].to_vec())
                .expect("row");
            hidden_to_raw_logits(weights, &row)
        };
        if resumed != canon_row {
            refuse(&format!(
                "splice floor FAILED on {}: resumed logits differ",
                rec.prompt_id
            ));
        }
        floor_prompts += 1;
    }
    println!("selftest splice floor: PASS ({floor_prompts} prompts bit-identical)");

    let report = serde_json::json!({
        "determinism": "pass",
        "recompute_parity": "pass",
        "splice_floor": "pass",
        "smoke_n": SMOKE_N,
        "identity": identity,
    });
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    write_atomic(
        &out_dir.join("selftest-report.json"),
        serde_json::to_string_pretty(&report).unwrap().as_bytes(),
    );
    println!(
        "selftest report written: {}",
        out_dir.join("selftest-report.json").display()
    );
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// Shard order is a pipeline priority, not part of the data contract:
/// shard content is keyed by (split, stratum, chunk) alone. The T2
/// instrument gate needs train/dev general first; T3 fitting needs
/// train/dev entity-swap next; test shards run last (evaluated once,
/// at the end).
fn shard_plan(prompts: &[PromptRec]) -> Vec<(String, Vec<usize>)> {
    let mut groups: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, p) in prompts.iter().enumerate() {
        groups
            .entry((p.split.clone(), p.stratum.clone()))
            .or_default()
            .push(i);
    }
    let split_rank = |s: &str| match s {
        "train" => 0,
        "dev" => 1,
        _ => 2,
    };
    let stratum_rank = |s: &str| match s {
        "general" => 0,
        "entity-swap" => 1,
        "templated" => 2,
        "code" => 3,
        _ => 4,
    };
    let mut ordered: Vec<_> = groups.into_iter().collect();
    ordered.sort_by_key(|((split, stratum), _)| {
        let test_last = if split == "test" { 1 } else { 0 };
        (test_last, stratum_rank(stratum), split_rank(split))
    });
    let mut shards = Vec::new();
    for ((split, stratum), idxs) in ordered {
        for (ci, chunk) in idxs.chunks(SHARD_SIZE).enumerate() {
            shards.push((format!("{split}-{stratum}-c{ci:02}"), chunk.to_vec()));
        }
    }
    shards
}

fn run_capture(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    corpus: &Corpus,
    out_dir: &Path,
    identity: &serde_json::Value,
) {
    let report_path = out_dir.join("selftest-report.json");
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap_or_else(|_| {
            refuse("no selftest-report.json — run LARQL_M7K_MODE=selftest first")
        }))
        .unwrap_or_else(|e| refuse(&format!("bad selftest report: {e}")));
    for key in ["determinism", "recompute_parity", "splice_floor"] {
        if report[key].as_str() != Some("pass") {
            refuse(&format!("selftest '{key}' did not pass"));
        }
    }
    if &report["identity"] != identity {
        refuse("selftest report identity does not match current corpus/container digests");
    }
    let free = free_disk_gb(out_dir);
    if free < MIN_FREE_GB {
        refuse(&format!("only {free} GB free (< {MIN_FREE_GB} GB floor)"));
    }

    let filter = env("LARQL_M7K_SHARD_FILTER");
    let plan = shard_plan(&corpus.prompts);
    let total = plan.len();
    for (si, (shard_id, idxs)) in plan.into_iter().enumerate() {
        if let Some(f) = &filter {
            if !shard_id.contains(f.as_str()) {
                continue;
            }
        }
        let shard_path = out_dir.join(format!("{shard_id}.safetensors"));
        let meta_path = out_dir.join(format!("{shard_id}.meta.jsonl"));
        if shard_path.exists() && meta_path.exists() {
            println!(
                "[{}/{}] {shard_id}: exists, skipping (append-only)",
                si + 1,
                total
            );
            continue;
        }
        let started = std::time::Instant::now();
        let recs: Vec<&PromptRec> = idxs.iter().map(|&i| &corpus.prompts[i]).collect();
        let buf = capture_batch(weights, tokenizer, &recs, true);
        write_atomic(&shard_path, &buf.serialize_main());
        if let Some(audit) = buf.serialize_audit() {
            write_atomic(
                &out_dir.join(format!("{shard_id}.audit.safetensors")),
                &audit,
            );
            write_atomic(
                &out_dir.join(format!("{shard_id}.audit.meta.jsonl")),
                (buf.audit_meta.join("\n") + "\n").as_bytes(),
            );
        }
        if !buf.checksum_lines.is_empty() {
            write_atomic(
                &out_dir.join(format!("{shard_id}.checksums.jsonl")),
                (buf.checksum_lines.join("\n") + "\n").as_bytes(),
            );
        }
        // meta last: a shard counts as complete only when meta exists.
        write_atomic(&meta_path, (buf.meta_lines.join("\n") + "\n").as_bytes());
        println!(
            "[{}/{}] {shard_id}: {} prompts, {} rows, {:.1}s",
            si + 1,
            total,
            idxs.len(),
            buf.rows,
            started.elapsed().as_secs_f32()
        );
        std::io::stdout().flush().ok();
    }
    println!("capture complete (or filtered subset done)");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let vindex = env("LARQL_M7K_VINDEX").unwrap_or_else(|| refuse("LARQL_M7K_VINDEX unset"));
    let corpus_dir = env("LARQL_M7K_CORPUS").unwrap_or_else(|| refuse("LARQL_M7K_CORPUS unset"));
    let out = env("LARQL_M7K_OUT").unwrap_or_else(|| refuse("LARQL_M7K_OUT unset"));
    let mode = env("LARQL_M7K_MODE").unwrap_or_else(|| refuse("LARQL_M7K_MODE unset"));
    let vindex_dir = PathBuf::from(shellexpand_home(&vindex));
    let corpus_dir = PathBuf::from(shellexpand_home(&corpus_dir));
    let out_dir = PathBuf::from(shellexpand_home(&out));

    let corpus = load_corpus(&corpus_dir);
    println!(
        "corpus: {} prompts, sha256 {}",
        corpus.prompts.len(),
        corpus.prompts_sha256
    );

    let vdigest = vindex_digest(&vindex_dir);
    let mdigest = model_digest();
    let identity = serde_json::json!({
        "corpus_version": CORPUS_VERSION,
        "prompts_sha256": corpus.prompts_sha256,
        "vindex_digest": vdigest,
        "model_digest": mdigest,
        "probe_layers": PROBE_LAYERS,
        "top_logits": TOP_LOGITS,
        "max_pos": MAX_POS,
        "min_pos_index": MIN_POS_INDEX,
        "shard_size": SHARD_SIZE,
    });

    // Capture manifest: created once, then must match on every run.
    std::fs::create_dir_all(&out_dir).expect("mkdir out");
    let cm_path = out_dir.join("capture-manifest.json");
    let git_sha = env("LARQL_M7K_GIT_SHA").unwrap_or_else(|| "unrecorded".to_string());
    if cm_path.exists() {
        let existing: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&cm_path).expect("read capture manifest"),
        )
        .expect("parse capture manifest");
        if existing["identity"] != identity {
            refuse("capture-manifest identity mismatch: output dir belongs to a different corpus/container");
        }
    } else {
        let cm = serde_json::json!({
            "identity": identity,
            "harness": "crates/larql-inference/examples/map7_kshape_capture.rs",
            "harness_git_sha": git_sha,
            "vindex_path": vindex_dir.to_string_lossy(),
        });
        write_atomic(
            &cm_path,
            serde_json::to_string_pretty(&cm).unwrap().as_bytes(),
        );
    }

    println!("loading vindex: {}", vindex_dir.display());
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_with_opts(
        &vindex_dir,
        &mut cb,
        larql_vindex::LoadWeightsOptions::default(),
    )
    .expect("load vindex weights");
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_dir).expect("load tokenizer");
    println!(
        "model: {} layers, hidden {}, vocab {}",
        weights.num_layers, weights.hidden_size, weights.vocab_size
    );

    match mode.as_str() {
        "selftest" => run_selftests(&weights, &tokenizer, &corpus, &out_dir, &identity),
        "capture" => run_capture(&weights, &tokenizer, &corpus, &out_dir, &identity),
        other => refuse(&format!("unknown LARQL_M7K_MODE '{other}'")),
    }
}

fn shellexpand_home(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = env("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}
