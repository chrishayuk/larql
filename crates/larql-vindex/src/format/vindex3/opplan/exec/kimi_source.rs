//! **The Kimi decoder stack, loaded from the source VINDEX3 container.**
//!
//! Q2a's loader, and the shape serving wants afterwards: no exported
//! fixture, no duplicate bytes — the container's own mmap'd segments are
//! the physical stores, and a [`DeviceLayer`] binds regions into them.
//!
//! ```text
//! target.decoder_stack   norms, router, KDA/MLA, SHARED experts, dense MLP
//! target.expert_bank     routed experts (arbitrary order, Table-addressed)
//! target.final_norm      final RMSNorm
//! target.output_head     lm_head
//! ```
//!
//! An optional [`CandidateOverlay`] substitutes a compiled bank for one
//! or more layers' ROUTED experts — and only those. Everything else,
//! including the substituted layers' shared experts, still resolves from
//! the source stores, so two arms differing only in the overlay differ
//! in exactly one physical fact.
//!
//! Geometry comes from the container's own `system_graph.json`, never
//! from a hardcoded family table — the graph carried it through
//! admission, so the loader consumes what the container says.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use larql_compute_metal::trait_impl::kda::KdaShape;
use larql_compute_metal::trait_impl::mla::MlaShape;

use crate::error::VindexError;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::represent::physical::{PhysicalStore, WeightRegion};

mod model;
mod overlay;
pub use model::*;
pub use overlay::*;

/// Positions each MLA layer's device cache is sized for. A sequence
/// longer than this is refused by the operator, not truncated.
const MLA_CACHE_POSITIONS: usize = 64;

/// Everything the loader needs to build a layer, read from the
/// container's own system graph.
#[derive(Debug, Clone)]
pub struct KimiGeometry {
    pub hidden: usize,
    pub num_layers: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub dense_prefix_layers: usize,
    pub dense_intermediate: usize,
    pub experts: u32,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub branch_scale: f32,
    pub renormalize: bool,
    pub kda: KdaShape,
    pub kda_gate_form: Option<larql_models::config::KdaGateForm>,
    pub mla: MlaShape,
    /// The epsilon MLA's latent norm runs at, READ FROM THE GRAPH.
    ///
    /// This was `MLA_KV_A_NORM_EPS`, a constant in this file, and the
    /// ontology drill's F6: the one judged semantic the container could
    /// not carry, so deleting the checkpoint could not restore it. The
    /// surface carries it since lift 2, and this loader consumes what
    /// the container says — refusing a container that predates it
    /// rather than re-supplying the number from memory, which is the
    /// same posture every other field here takes.
    pub mla_norm_eps: f32,
    /// Per layer, in order: `true` = MLA full attention, `false` = KDA.
    pub mla_layer: Vec<bool>,
    /// The container declares KDA's output gate full-rank
    /// (`use_full_rank_gate`, Kimi-K3). This loader binds the low-rank
    /// pair by NAME into `f32s[5]`/`[6]`, so a full-rank layer is refused
    /// before any tensor is read rather than failing on a missing name.
    pub kda_full_rank_gate: bool,
    /// The container declares an MLA output gate (`mla_use_output_gate`,
    /// Kimi-K3). The device MLA path carries no gate, so a gated layer is
    /// refused rather than run ungated with every shape still closing.
    pub mla_output_gate: bool,
    /// The container declares a factorised MLA query (`q_lora_rank`,
    /// Kimi-K3). The device MLA path binds one dense `q_proj` and
    /// `q_b_proj` has the same row count, so a factorised layer is
    /// refused rather than bound into a slot it does not belong in.
    pub mla_q_lora_rank: Option<usize>,
}

impl KimiGeometry {
    /// Bytes one BF16 routed-expert projection occupies in the source.
    pub fn source_projection_bytes(&self) -> u64 {
        self.moe_intermediate as u64 * self.hidden as u64 * 2
    }
}

fn graph_value(dir: &Path) -> Result<serde_json::Value, VindexError> {
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)
        .map_err(|e| VindexError::Parse(format!("index.json: {e}")))?;
    let graph_name = index["system_graph"]
        .as_str()
        .unwrap_or("system_graph.json")
        .to_string();
    let graph_path = crate::format::vindex3::container_path::resolve(dir, &graph_name)?;
    serde_json::from_slice(&std::fs::read(graph_path)?)
        .map_err(|e| VindexError::Parse(format!("system_graph: {e}")))
}

/// Read the geometry from the graph, refusing anything absent — a
/// defaulted width would build a layer that binds plausibly and computes
/// the wrong function.
fn geometry_from_graph(graph: &serde_json::Value) -> Result<KimiGeometry, VindexError> {
    let comp = graph["components"]
        .get(0)
        .ok_or_else(|| VindexError::Parse("graph has no components".into()))?;
    let need = |v: &serde_json::Value, what: &str| -> Result<u64, VindexError> {
        v.as_u64()
            .ok_or_else(|| VindexError::Parse(format!("graph is missing `{what}`")))
    };
    let need_f = |v: &serde_json::Value, what: &str| -> Result<f64, VindexError> {
        v.as_f64()
            .ok_or_else(|| VindexError::Parse(format!("graph is missing `{what}`")))
    };
    let exec = &comp["execution"];
    let (kda, mla, moe, norm, head, ffn) = (
        &exec["kda"],
        &exec["mla"],
        &exec["ffn"]["moe"],
        &exec["norm"],
        &exec["head"],
        &exec["ffn"],
    );
    let hidden = need(&comp["hidden_size"], "hidden_size")? as usize;
    let num_layers = need(&comp["num_layers"], "num_layers")? as usize;
    let routing = moe["routing_policy"].as_str().unwrap_or("");
    let renormalize = match routing {
        "normalised_over_selected" => true,
        other => {
            return Err(VindexError::Parse(format!(
                "routing policy `{other}` is not one this loader has judged"
            )))
        }
    };
    let attention = comp["attention"]
        .as_array()
        .ok_or_else(|| VindexError::Parse("graph has no per-layer attention".into()))?;
    if attention.len() != num_layers {
        return Err(VindexError::Parse(format!(
            "graph declares {num_layers} layers but {} attention entries",
            attention.len()
        )));
    }
    let mla_layer = attention
        .iter()
        .enumerate()
        .map(|(i, a)| match a["operator"].as_str() {
            Some("mla") => Ok(true),
            Some("kda") => Ok(false),
            other => Err(VindexError::Parse(format!(
                "layer {i} declares operator {other:?}, which this loader cannot build"
            ))),
        })
        .collect::<Result<Vec<bool>, _>>()?;
    Ok(KimiGeometry {
        hidden,
        num_layers,
        vocab: need(&head["vocab_size"], "head.vocab_size")? as usize,
        rms_eps: need_f(&norm["pre"]["eps"], "norm.pre.eps")? as f32,
        dense_prefix_layers: need(&moe["dense_prefix_layers"], "moe.dense_prefix_layers")? as usize,
        dense_intermediate: need(&ffn["intermediate_size"], "ffn.intermediate_size")? as usize,
        experts: need(&moe["experts"], "moe.experts")? as u32,
        top_k: need(&moe["top_k"], "moe.top_k")? as usize,
        moe_intermediate: need(
            &moe["expert_intermediate_size"],
            "moe.expert_intermediate_size",
        )? as usize,
        branch_scale: need_f(&moe["branch_scale"], "moe.branch_scale")? as f32,
        renormalize,
        kda_gate_form: serde_json::from_value(exec["kda_gate_form"].clone())
            .map_err(|e| VindexError::Parse(format!("kda_gate_form: {e}")))?,
        kda: KdaShape {
            hidden,
            num_heads: need(&kda["num_heads"], "kda.num_heads")? as usize,
            head_dim: need(&kda["head_dim"], "kda.head_dim")? as usize,
            conv_kernel: need(&kda["conv_kernel"], "kda.conv_kernel")? as usize,
        },
        mla_norm_eps: need_f(&mla["kv_a_norm_eps"], "mla.kv_a_norm_eps")? as f32,
        mla: MlaShape {
            hidden,
            num_heads: need(&mla["num_heads"], "mla.num_heads")? as usize,
            kv_lora_rank: need(&mla["kv_lora_rank"], "mla.kv_lora_rank")? as usize,
            qk_nope_head_dim: need(&mla["qk_nope_head_dim"], "mla.qk_nope_head_dim")? as usize,
            qk_rope_head_dim: need(&mla["qk_rope_head_dim"], "mla.qk_rope_head_dim")? as usize,
            v_head_dim: need(&mla["v_head_dim"], "mla.v_head_dim")? as usize,
        },
        mla_layer,
        // Both read from the surface as DECLARED, never from whether a
        // `g_proj` happens to exist in the segment (K3-REP-GATE-1).
        kda_full_rank_gate: exec["kda_use_full_rank_gate"].as_bool() == Some(true),
        mla_output_gate: mla["output_gate"].is_object(),
        // The surface's declared query form, read as the graph writes it.
        mla_q_lora_rank: mla["query"]["rank"].as_u64().map(|r| r as usize),
    })
}

/// One named tensor of a mapped segment, with its stated dtype.
struct SegmentTensors {
    store: Arc<PhysicalStore>,
    dtypes: BTreeMap<String, String>,
    offsets: BTreeMap<String, u64>,
}

impl SegmentTensors {
    fn open(id: &str, path: &Path) -> Result<Self, VindexError> {
        let (header, _) = read_segment_header(path)?;
        let mut dtypes = BTreeMap::new();
        let mut offsets = BTreeMap::new();
        for t in &header.tensors {
            dtypes.insert(t.name.clone(), t.dtype.clone());
            offsets.insert(t.name.clone(), t.offset);
        }
        Ok(Self {
            store: Arc::new(PhysicalStore::map_segment(id, path)?),
            dtypes,
            offsets,
        })
    }

    /// The region for `tensor`, or a refusal naming it — the "zero
    /// missing operands" criterion is enforced here, at every lookup,
    /// rather than tallied afterwards.
    fn region(&self, tensor: &str) -> Result<WeightRegion, VindexError> {
        self.store.whole(tensor).ok_or_else(|| {
            VindexError::Parse(format!(
                "`{}` has no `{tensor}` — a source operand is missing",
                self.store.id()
            ))
        })
    }

    /// Raw bytes, copied out — for operands the device holds owned.
    fn bytes(&self, tensor: &str) -> Result<Vec<u8>, VindexError> {
        Ok(self.region(tensor)?.bytes().to_vec())
    }

    /// The tensor widened to f32, whatever the segment stored.
    ///
    /// BF16 widens losslessly; F32 reinterprets. Anything else is
    /// refused by name rather than mis-read.
    fn f32s(&self, tensor: &str) -> Result<Vec<f32>, VindexError> {
        let bytes = self.region(tensor)?;
        let bytes = bytes.bytes();
        match self.dtypes.get(tensor).map(String::as_str) {
            Some("BF16") => Ok(bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect()),
            Some("F32") => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            other => Err(VindexError::Parse(format!(
                "`{tensor}` is stored as {other:?}, which this loader does not widen"
            ))),
        }
    }
}

/// The source container, opened once and shared by every arm built from
/// it — which is what makes "layer 2+ is the same store in both arms" a
/// structural fact rather than a hope.
pub struct KimiSourceModel {
    pub geometry: KimiGeometry,
    dir: PathBuf,
    decoder: SegmentTensors,
    experts: SegmentTensors,
}
