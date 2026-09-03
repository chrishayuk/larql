//! Checkpoint geometry over HTTP range requests — I/O, excluded from coverage.
//!
//! Reads `config.json` plus two safetensors headers (one KDA layer, one MLA
//! layer) without downloading any weights. On a 1.5 TB repository that is under
//! 2 MB of traffic.
//!
//! Index-base trap, recorded because it cost a wrong figure once: the config's
//! `kda_layers` / `full_attn_layers` lists are **1-indexed** against 0-indexed
//! tensor names, so config layer 4 is tensor layer 3. Fetching the wrong one
//! silently returns another KDA layer and over-counts the dense half.

use std::collections::BTreeMap;

use serde_json::Value;

use super::access::{AccessMode, DenseCensus, Family, LayerTopology};
use super::geometry::K3Geometry;

const SAFETENSORS_LEN_PREFIX: u64 = 8;

pub struct Repo {
    pub id: String,
    client: reqwest::blocking::Client,
}

impl Repo {
    pub fn new(id: impl Into<String>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            id: id.into(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(180))
                .build()?,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{path}", self.id)
    }

    fn range(&self, path: &str, from: u64, to: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let resp = self
            .client
            .get(self.url(path))
            .header(reqwest::header::RANGE, format!("bytes={from}-{to}"))
            .send()?
            .error_for_status()?;
        Ok(resp.bytes()?.to_vec())
    }

    pub fn config(&self) -> Result<Value, Box<dyn std::error::Error>> {
        let body = self
            .client
            .get(self.url("config.json"))
            .send()?
            .error_for_status()?
            .text()?;
        let v: Value = serde_json::from_str(&body)?;
        Ok(v.get("text_config").cloned().unwrap_or(v))
    }

    /// Parse one shard's safetensors header without reading its tensor data.
    pub fn shard_header(&self, shard: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let prefix = self.range(shard, 0, SAFETENSORS_LEN_PREFIX - 1)?;
        let n = u64::from_le_bytes(prefix[..8].try_into()?);
        let body = self.range(
            shard,
            SAFETENSORS_LEN_PREFIX,
            SAFETENSORS_LEN_PREFIX + n - 1,
        )?;
        Ok(serde_json::from_slice(&body)?)
    }
}

/// Byte range of one named tensor within a shard, from its header entry.
pub fn tensor_range(header: &Value, name: &str) -> Option<(u64, u64)> {
    let o = header.get(name)?.get("data_offsets")?.as_array()?;
    Some((o.first()?.as_u64()?, o.get(1)?.as_u64()?))
}

/// Every `weight_scale` tensor in a shard, with its shape and byte range.
pub fn scale_tensors(header: &Value) -> Vec<(String, Vec<u64>, u64, u64)> {
    let Some(obj) = header.as_object() else {
        return Vec::new();
    };
    let mut out: Vec<_> = obj
        .iter()
        .filter(|(k, _)| k.ends_with("weight_scale"))
        .filter_map(|(k, v)| {
            let shape: Vec<u64> = v
                .get("shape")?
                .as_array()?
                .iter()
                .filter_map(Value::as_u64)
                .collect();
            let (lo, hi) = tensor_range(header, k)?;
            Some((k.clone(), shape, lo, hi))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Every routed-expert `weight_packed` tensor in a shard, with shape and range.
pub fn packed_expert_tensors(header: &Value) -> Vec<(String, Vec<u64>, u64, u64)> {
    let Some(obj) = header.as_object() else {
        return Vec::new();
    };
    let mut out: Vec<_> = obj
        .iter()
        .filter(|(k, _)| k.ends_with("weight_packed") && k.contains(".experts."))
        .filter_map(|(k, v)| {
            let shape: Vec<u64> = v
                .get("shape")?
                .as_array()?
                .iter()
                .filter_map(Value::as_u64)
                .collect();
            let (lo, hi) = tensor_range(header, k)?;
            Some((k.clone(), shape, lo, hi))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

impl Repo {
    /// Read a byte range out of a shard, offset by the header length.
    pub fn tensor_bytes_at(
        &self,
        shard: &str,
        header_len: u64,
        lo: u64,
        hi: u64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let base = SAFETENSORS_LEN_PREFIX + header_len;
        self.range(shard, base + lo, base + hi - 1)
    }

    /// The header plus its byte length, so tensor offsets can be resolved.
    pub fn shard_header_with_len(
        &self,
        shard: &str,
    ) -> Result<(Value, u64), Box<dyn std::error::Error>> {
        let prefix = self.range(shard, 0, SAFETENSORS_LEN_PREFIX - 1)?;
        let n = u64::from_le_bytes(prefix[..8].try_into()?);
        let body = self.range(
            shard,
            SAFETENSORS_LEN_PREFIX,
            SAFETENSORS_LEN_PREFIX + n - 1,
        )?;
        Ok((serde_json::from_slice(&body)?, n))
    }
}

fn tensor_bytes(entry: &Value) -> u64 {
    entry
        .get("data_offsets")
        .and_then(|o| o.as_array())
        .and_then(|a| Some(a.get(1)?.as_u64()? - a.first()?.as_u64()?))
        .unwrap_or(0)
}

fn tensor_params(entry: &Value) -> u64 {
    entry
        .get("shape")
        .and_then(|s| s.as_array())
        .map(|dims| dims.iter().filter_map(Value::as_u64).product())
        .unwrap_or(0)
}

/// Non-expert parameters, and one expert's total bytes plus per-branch bytes.
fn split_layer(header: &Value) -> (u64, u64, [u64; 3]) {
    let obj = match header.as_object() {
        Some(o) => o,
        None => return (0, 0, [0; 3]),
    };
    let mut dense = 0u64;
    let mut expert0 = 0u64;
    let mut branches: BTreeMap<String, u64> = BTreeMap::new();

    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        if let Some(rest) = name.split(".experts.").nth(1) {
            let idx = rest.split('.').next().unwrap_or("");
            if idx == "0" {
                expert0 += tensor_bytes(entry);
                for w in ["w1", "w2", "w3"] {
                    if name.contains(&format!(".{w}.")) {
                        *branches.entry(w.to_string()).or_default() += tensor_bytes(entry);
                    }
                }
            }
        } else {
            dense += tensor_params(entry);
        }
    }
    let b = |k: &str| branches.get(k).copied().unwrap_or(0);
    (dense, expert0, [b("w1"), b("w2"), b("w3")])
}

fn usize_at(cfg: &Value, key: &str) -> usize {
    cfg.get(key).and_then(Value::as_u64).unwrap_or(0) as usize
}

fn layer_list(cfg: &Value, key: &str) -> Vec<usize> {
    cfg.get("linear_attn_config")
        .and_then(|c| c.get(key))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default()
}

/// All-in bits per weight for the families `quantization_config.ignore`
/// excludes, from the checkpoint's own `dtype`.
///
/// K3-L1-F1: this is the number the ledger used to hardcode at 4.25. The
/// routed side has always been derived; now the dense side is too.
fn dense_stored(cfg: &Value) -> (f64, String) {
    let dtype = cfg
        .get("dtype")
        .or_else(|| cfg.get("torch_dtype"))
        .and_then(Value::as_str)
        .unwrap_or("bfloat16");
    let bits = match dtype {
        "float32" | "float" => 32.0,
        "float64" | "double" => 64.0,
        _ => 16.0,
    };
    let label = match dtype {
        "bfloat16" => "BF16".to_string(),
        "float16" | "half" => "FP16".to_string(),
        other => other.to_uppercase(),
    };
    (bits, label)
}

/// Split one layer's header into dense families by tensor name.
///
/// The patterns are the checkpoint's own: `quantization_config.ignore`
/// names `self_attn`, `shared_experts` and `mlp.*_proj`, and the LatentMoE
/// wrapper and router sit beside them under `block_sparse_moe`.
fn family_split(header: &Value) -> BTreeMap<&'static str, u64> {
    let mut out: BTreeMap<&'static str, u64> = BTreeMap::new();
    let Some(obj) = header.as_object() else {
        return out;
    };
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        // Routed experts are not dense. `shared_experts` must not be caught
        // by the `.experts.` test — it is BF16 and always-on.
        if name.contains(".experts.") && !name.contains("shared_experts") {
            continue;
        }
        let family = if name.contains("shared_experts") {
            "shared_experts"
        } else if name.contains("self_attn") {
            "self_attn"
        } else if name.contains("routed_expert_") {
            "LatentMoE wrapper"
        } else if name.contains("block_sparse_moe.gate") {
            "router"
        } else if name.contains("mlp.") {
            "dense MLP"
        } else {
            "norms / res_proj"
        };
        *out.entry(family).or_default() += tensor_bytes(entry);
    }
    out
}

/// The measured dense census: every BF16 family and how it is TOUCHED.
///
/// K3-LEDGER-2. `embed_tokens` and `lm_head` are byte-identical and untied;
/// only the head is a full read. Layer 0 is dense, so shared experts, the
/// wrapper and the router multiply by `n_moe()`, not by `n_layers`.
fn dense_census(
    cfg: &Value,
    kda: &BTreeMap<&'static str, u64>,
    mla: &BTreeMap<&'static str, u64>,
    topo: LayerTopology,
    n_kda: usize,
    n_mla: usize,
) -> DenseCensus {
    let hidden = usize_at(cfg, "hidden_size") as u64;
    let vocab = usize_at(cfg, "vocab_size") as u64;
    let ffn = usize_at(cfg, "intermediate_size") as u64;
    let dtype_bytes = 2u64;
    let get = |m: &BTreeMap<&'static str, u64>, k: &str| m.get(k).copied().unwrap_or(0);

    // A dense layer has no shared experts, wrapper or router; it has an MLP.
    let dense0 = 3 * ffn * hidden * dtype_bytes;
    let head = vocab * hidden * dtype_bytes;

    DenseCensus::new(vec![
        Family::full_read("KDA self_attn", get(kda, "self_attn"), n_kda),
        Family::full_read("shared_experts", get(kda, "shared_experts"), topo.n_moe()),
        Family::full_read("MLA self_attn", get(mla, "self_attn"), n_mla),
        Family::full_read(
            "LatentMoE wrapper",
            get(kda, "LatentMoE wrapper"),
            topo.n_moe(),
        ),
        Family::full_read("lm_head", head, 1),
        Family::full_read("dense layer-0 MLP", dense0, topo.n_dense()),
        Family::full_read("router", get(kda, "router"), topo.n_moe()),
        Family::full_read(
            "norms / res_proj",
            get(kda, "norms / res_proj"),
            topo.n_layers,
        ),
        Family {
            name: "embed_tokens".into(),
            bytes_per_unit: head,
            units: 1,
            // A gather, not a matmul. This is the -2.35 GB correction.
            access: AccessMode::Gather {
                rows_per_token: 1,
                total_rows: vocab,
            },
        },
    ])
}

/// One layer's family byte profile, from its shard header alone.
///
/// Layer `n` lives in shard `n + 1` — verified against layers 0, 3 and 49.
pub fn layer_profile(
    repo: &Repo,
    template: &str,
    layer: usize,
) -> Result<super::homogeneity::LayerProfile, Box<dyn std::error::Error>> {
    let shard = template.replace("{:05}", &format!("{:05}", layer + 1));
    let header = repo.shard_header(&shard)?;
    Ok(super::homogeneity::LayerProfile {
        index: layer,
        families: family_split(&header)
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    })
}

pub fn load_geometry(
    repo: &Repo,
    kda_shard: &str,
    mla_shard: &str,
) -> Result<K3Geometry, Box<dyn std::error::Error>> {
    let cfg = repo.config()?;
    let kda_hdr = repo.shard_header(kda_shard)?;
    let mla_hdr = repo.shard_header(mla_shard)?;

    let (kda_dense, expert_bytes, branch_bytes) = split_layer(&kda_hdr);
    let (mla_dense, _, _) = split_layer(&mla_hdr);

    if kda_dense <= mla_dense {
        return Err(format!(
            "MLA layer ({mla_dense} params) is not lighter than the KDA layer \
             ({kda_dense} params) — the 1-indexed config vs 0-indexed tensor \
             trap: config layer N is tensor layer N-1, so '{mla_shard}' is \
             probably another KDA layer"
        )
        .into());
    }

    let n_layers = usize_at(&cfg, "num_hidden_layers");
    let full_attn_list = layer_list(&cfg, "full_attn_layers");
    let kda_list = layer_list(&cfg, "kda_layers");
    let full_attn = full_attn_list.len();
    let kda = kda_list.len();
    // 1-indexed in the config, 0-indexed in tensor names.
    let kda_layer_indices: Vec<usize> = kda_list.iter().filter_map(|l| l.checked_sub(1)).collect();
    let vocab = usize_at(&cfg, "vocab_size") as u64;
    let (dense_stored_bits, dense_stored_label) = dense_stored(&cfg);
    let topology = LayerTopology::new(n_layers, usize_at(&cfg, "first_k_dense_replace"));
    let census = dense_census(
        &cfg,
        &family_split(&kda_hdr),
        &family_split(&mla_hdr),
        topology,
        if kda > 0 { kda } else { n_layers - full_attn },
        full_attn,
    );
    let hidden = usize_at(&cfg, "hidden_size");

    Ok(K3Geometry {
        hidden_size: hidden,
        latent_dim: usize_at(&cfg, "routed_expert_hidden_size"),
        moe_intermediate: usize_at(&cfg, "moe_intermediate_size"),
        n_layers,
        n_dense_layers: usize_at(&cfg, "first_k_dense_replace"),
        n_kda_layers: if kda > 0 { kda } else { n_layers - full_attn },
        n_mla_layers: full_attn,
        n_experts: usize_at(&cfg, "num_experts"),
        top_k: usize_at(&cfg, "num_experts_per_token"),
        expert_bytes,
        branch_bytes,
        kda_layer_dense_params: kda_dense,
        mla_layer_dense_params: mla_dense,
        // Untied embed + lm_head. Vision is deliberately excluded.
        embedding_params: 2 * vocab * hidden as u64,
        dense_stored_bits,
        dense_stored_label,
        kda_layer_indices,
        topology,
        dense_census: census,
        vision_included: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_params_multiplies_the_shape() {
        let v: Value = serde_json::json!({"shape": [3072, 1792], "data_offsets": [0, 100]});
        assert_eq!(tensor_params(&v), 3072 * 1792);
        assert_eq!(tensor_bytes(&v), 100);
    }

    #[test]
    fn split_layer_separates_experts_from_dense_and_sums_branch_zero_only() {
        let hdr: Value = serde_json::json!({
            "layers.9.self_attn.q_proj.weight": {"shape": [8, 4], "data_offsets": [0, 64]},
            "layers.9.mlp.experts.0.w1.weight_packed": {"shape": [2, 2], "data_offsets": [64, 74]},
            "layers.9.mlp.experts.0.w2.weight_packed": {"shape": [2, 2], "data_offsets": [74, 84]},
            "layers.9.mlp.experts.0.w3.weight_packed": {"shape": [2, 2], "data_offsets": [84, 94]},
            "layers.9.mlp.experts.1.w1.weight_packed": {"shape": [2, 2], "data_offsets": [94, 104]},
        });
        let (dense, expert0, branches) = split_layer(&hdr);
        assert_eq!(dense, 32, "only non-expert tensors count toward dense");
        assert_eq!(expert0, 30, "expert 1 must not be summed into expert 0");
        assert_eq!(branches, [10, 10, 10]);
    }

    #[test]
    fn metadata_key_is_ignored() {
        let hdr: Value = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "layers.0.norm.weight": {"shape": [4], "data_offsets": [0, 8]},
        });
        assert_eq!(split_layer(&hdr).0, 4);
    }
}
