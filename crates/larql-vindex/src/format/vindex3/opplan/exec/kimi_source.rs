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

use larql_compute::backend::ComputeBackend;
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kda::{KdaDeviceState, KdaShape};
use larql_compute_metal::trait_impl::mla::{MlaDeviceState, MlaShape};
use larql_compute_metal::MetalBackend;

use super::stack_metal::{DeviceAttn, DeviceLayer, DeviceState, HybridHead};
use crate::error::VindexError;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::represent::compile::CandidatePlacement;
use crate::format::vindex3::represent::compiler::{
    bank_base, read_source_identity, CandidateIndex,
};
use crate::format::vindex3::represent::physical::{
    EncodedRegion, ExpertBankBinding, ExpertEncoding, ExtentPolicy, PhysicalStore,
    ProjectionAddressing, RoutedProjection, SharedExpertBinding, WeightRegion,
};
use crate::format::vindex3::represent::policy::{layer_of, projection_of, Role};
use crate::format::vindex3::represent::source_bank::source_expert_bank;

/// `kv_a_layernorm`'s epsilon is the reference class's DEFAULT, not the
/// config's `rms_norm_eps`: `KimiMLAAttention.__init__` constructs it
/// with no override (P3d-i, measured against the real checkpoint). The
/// graph carries the config value; this one fact it cannot carry.
const MLA_KV_A_NORM_EPS: f32 = 1e-6;

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
    pub mla: MlaShape,
    /// Per layer, in order: `true` = MLA full attention, `false` = KDA.
    pub mla_layer: Vec<bool>,
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
    serde_json::from_slice(&std::fs::read(dir.join(graph_name))?)
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
        kda: KdaShape {
            hidden,
            num_heads: need(&kda["num_heads"], "kda.num_heads")? as usize,
            head_dim: need(&kda["head_dim"], "kda.head_dim")? as usize,
            conv_kernel: need(&kda["conv_kernel"], "kda.conv_kernel")? as usize,
        },
        mla: MlaShape {
            hidden,
            num_heads: need(&mla["num_heads"], "mla.num_heads")? as usize,
            kv_lora_rank: need(&mla["kv_lora_rank"], "mla.kv_lora_rank")? as usize,
            qk_nope_head_dim: need(&mla["qk_nope_head_dim"], "mla.qk_nope_head_dim")? as usize,
            qk_rope_head_dim: need(&mla["qk_rope_head_dim"], "mla.qk_rope_head_dim")? as usize,
            v_head_dim: need(&mla["v_head_dim"], "mla.v_head_dim")? as usize,
        },
        mla_layer,
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

impl KimiSourceModel {
    pub fn open(dir: &Path) -> Result<Self, VindexError> {
        let geometry = geometry_from_graph(&graph_value(dir)?)?;
        let seg = |name: &str| dir.join("segments").join(format!("{name}.bin"));
        Ok(Self {
            geometry,
            dir: dir.to_path_buf(),
            decoder: SegmentTensors::open(
                "kimi-source-decoder-stack",
                &seg("target.decoder_stack"),
            )?,
            experts: SegmentTensors::open("kimi-source-expert-bank", &seg("target.expert_bank"))?,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The shared expert's binding for one layer — always from the
    /// decoder stack, never from any expert bank, whichever arm asked.
    fn shared_binding(&self, layer: usize) -> Result<SharedExpertBinding, VindexError> {
        let region = |proj: &str| -> Result<EncodedRegion, VindexError> {
            Ok(EncodedRegion {
                region: self.decoder.region(&format!(
                    "{layer}.block_sparse_moe.shared_experts.{proj}.weight"
                ))?,
                encoding: ExpertEncoding::Bf16,
            })
        };
        Ok(SharedExpertBinding {
            gate: region("gate_proj")?,
            up: region("up_proj")?,
            down: region("down_proj")?,
        })
    }

    /// One layer's attention operands, whichever operator the graph
    /// declares for it.
    fn attention(
        &self,
        metal: &MetalBackend,
        layer: usize,
    ) -> Result<(DeviceAttn, DeviceState), VindexError> {
        let g = &self.geometry;
        let t = |suffix: &str| format!("{layer}.self_attn.{suffix}");
        if g.mla_layer[layer] {
            return Ok((
                DeviceAttn::Mla {
                    q: self.decoder.bytes(&t("q_proj.weight"))?,
                    kv_a: self.decoder.bytes(&t("kv_a_proj_with_mqa.weight"))?,
                    kv_b: self.decoder.bytes(&t("kv_b_proj.weight"))?,
                    o: self.decoder.bytes(&t("o_proj.weight"))?,
                    kv_a_norm: self.decoder.f32s(&t("kv_a_layernorm.weight"))?,
                },
                DeviceState::Mla(MlaDeviceState::with_capacity(
                    metal,
                    g.mla,
                    MLA_CACHE_POSITIONS,
                )),
            ));
        }
        let (q, k, v) = (
            self.decoder.bytes(&t("q_proj.weight"))?,
            self.decoder.bytes(&t("k_proj.weight"))?,
            self.decoder.bytes(&t("v_proj.weight"))?,
        );
        let per = q.len();
        if k.len() != per || v.len() != per {
            return Err(VindexError::Parse(format!(
                "layer {layer}: KDA q/k/v projections differ in size ({per}/{}/{})",
                k.len(),
                v.len()
            )));
        }
        let mut qkv_bank = Vec::with_capacity(3 * per);
        for b in [&q, &k, &v] {
            qkv_bank.extend_from_slice(b);
        }
        // `KdaDeviceWeights`'s own field order. `A_log` is stored
        // `[1,1,H,1]` and flattens to `[H]`; the conv weights' middle
        // `1` dimension is inert under row-major flattening — the only
        // two transforms the proven exporter ever applied.
        let f32s: Vec<Vec<f32>> = [
            "q_conv1d.weight",
            "k_conv1d.weight",
            "v_conv1d.weight",
            "f_a_proj.weight",
            "f_b_proj.weight",
            "g_a_proj.weight",
            "g_b_proj.weight",
            "b_proj.weight",
            "A_log",
            "dt_bias",
            "o_norm.weight",
        ]
        .iter()
        .map(|suffix| self.decoder.f32s(&t(suffix)))
        .collect::<Result<_, _>>()?;
        Ok((
            DeviceAttn::Kda {
                qkv_bank,
                qkv_offsets: [
                    ExpertOffset(0),
                    ExpertOffset(per as u32),
                    ExpertOffset((2 * per) as u32),
                ],
                o_proj: self.decoder.bytes(&t("o_proj.weight"))?,
                f32s,
            },
            DeviceState::Kda(KdaDeviceState::zeros(metal, g.kda)),
        ))
    }

    /// Build one layer, routed experts from the source bank or — when
    /// the overlay compiled this layer — from the candidate's own store.
    pub fn device_layer(
        &self,
        metal: &MetalBackend,
        layer: usize,
        overlay: Option<&CandidateOverlay>,
    ) -> Result<DeviceLayer, VindexError> {
        let g = &self.geometry;
        let (attn, state) = self.attention(metal, layer)?;
        let common = |bank, router_weight, router_bias, inter, top_k, dense| {
            DeviceLayer {
                attn,
                state,
                bank,
                input_norm: Vec::new(), // filled below
                post_norm: Vec::new(),
                router_weight,
                router_bias,
                inter,
                top_k,
                dense,
                renormalize: g.renormalize,
                branch_scale: g.branch_scale,
                norm_eps: g.rms_eps,
                kda_shape: g.kda,
                mla_shape: g.mla,
                mla_norm_eps: MLA_KV_A_NORM_EPS,
            }
        };
        let mut d = if layer < g.dense_prefix_layers {
            // The dense MLP: three whole tensors of the decoder stack,
            // bound as one-expert regions.
            let region = |proj: &str| -> Result<EncodedRegion, VindexError> {
                Ok(EncodedRegion {
                    region: self.decoder.region(&format!("{layer}.mlp.{proj}.weight"))?,
                    encoding: ExpertEncoding::Bf16,
                })
            };
            // One "expert" at offset zero, per projection. The dense
            // MLP is three whole tensors of the decoder stack, and each
            // is exactly its own region — so `Exact` is a real claim
            // here and the surplus-byte check does bite.
            let dense = |proj: &str| -> Result<RoutedProjection, VindexError> {
                Ok(RoutedProjection {
                    region: region(proj)?,
                    addressing: ProjectionAddressing::Table(vec![0]),
                    extent: ExtentPolicy::Exact,
                })
            };
            common(
                ExpertBankBinding {
                    gate: dense("gate_proj")?,
                    up: dense("up_proj")?,
                    down: dense("down_proj")?,
                    shared: None,
                },
                // A dense layer carries no router at all.
                Vec::new(),
                Vec::new(),
                g.dense_intermediate,
                0,
                true,
            )
        } else {
            let shared = self.shared_binding(layer)?;
            let router_weight = self
                .decoder
                .f32s(&format!("{layer}.block_sparse_moe.gate.weight"))?;
            let router_bias = self.decoder.f32s(&format!(
                "{layer}.block_sparse_moe.gate.e_score_correction_bias"
            ))?;
            // **The source binding is always built**, and the overlay
            // then replaces only the projections it actually compiled.
            //
            // Per projection, not per bank: a projection-scoped
            // candidate holds `w1` alone, and `w3`/`w2` must still
            // resolve from the source segment — a different store, a
            // different encoding and a different addressing mode, in
            // the same layer. Composing in this direction also makes
            // the fallback the source rather than a hole, so an overlay
            // that compiled nothing yields the baseline arm exactly.
            let source = source_expert_bank(
                &self.experts.store,
                &self.experts.offsets,
                layer as u32,
                g.experts,
                g.source_projection_bytes(),
            )?;
            let mut bank = source.binding;
            bank.shared = Some(shared);
            if let Some(o) = overlay {
                for (name, slot) in [
                    ("w1", &mut bank.gate),
                    ("w3", &mut bank.up),
                    ("w2", &mut bank.down),
                ] {
                    if let Some(compiled) = o.projection_binding(layer as u32, name) {
                        *slot = compiled?;
                    }
                }
            }
            common(
                bank,
                router_weight,
                router_bias,
                g.moe_intermediate,
                g.top_k,
                false,
            )
        };
        d.input_norm = self
            .decoder
            .f32s(&format!("{layer}.input_layernorm.weight"))?;
        d.post_norm = self
            .decoder
            .f32s(&format!("{layer}.post_attention_layernorm.weight"))?;
        d.validate_banks(g.hidden)?;
        Ok(d)
    }

    /// The final norm and vocabulary projection, for the head to ride in
    /// the last device epoch.
    pub fn head(&self) -> Result<HybridHead, VindexError> {
        let seg = |name: &str| self.dir.join("segments").join(format!("{name}.bin"));
        let final_norm = SegmentTensors::open("kimi-source-final-norm", &seg("target.final_norm"))?;
        let head = SegmentTensors::open("kimi-source-output-head", &seg("target.output_head"))?;
        Ok(HybridHead {
            norm_weight: final_norm.f32s("weight")?,
            weight: head.bytes("weight")?,
            vocab: self.geometry.vocab,
            norm_eps: self.geometry.rms_eps,
        })
    }

    /// Register the mmap-backed stores with the backend, page-aligned.
    ///
    /// Region bases must be page-aligned for zero-copy registration, and
    /// a payload span almost never is — so registration cuts from each
    /// store's BACKING allocation (whose mmap base is aligned by
    /// construction) at page boundaries. Without this, the first
    /// `weights()` resolution misses and stages a multi-gigabyte copy,
    /// silently.
    ///
    /// The expert bank is registered as one span per MoE layer (~3.6 GB
    /// each) rather than one 94 GB buffer; `moe_layers` names which.
    pub fn register_stores(
        &self,
        metal: &MetalBackend,
        moe_layers: &[u32],
    ) -> Result<usize, VindexError> {
        const PAGE: usize = larql_compute_metal::buffers::PAGE_SIZE;
        let mut registered = 0usize;
        metal.register_weight_region(self.decoder.store.backing_bytes());
        registered += 1;
        let backing = self.experts.store.backing_bytes();
        let payload_start = self.experts.store.payload_start();
        for &layer in moe_layers {
            let bank = source_expert_bank(
                &self.experts.store,
                &self.experts.offsets,
                layer,
                self.geometry.experts,
                self.geometry.source_projection_bytes(),
            )?;
            let start = (payload_start + bank.layer_base) as usize / PAGE * PAGE;
            let end = (payload_start + bank.layer_base + bank.layer_len) as usize;
            metal.register_weight_region(&backing[start..end]);
            registered += 1;
        }
        Ok(registered)
    }
}

/// A compiled candidate bank, opened against the source it depends on.
pub struct CandidateOverlay {
    pub index: CandidateIndex,
    store: Arc<PhysicalStore>,
    /// Layers whose routed bank the candidate holds, from the ledger's
    /// own seals — never from the map alone, which states intent rather
    /// than bytes.
    layers: Vec<u32>,
    /// Projections the candidate holds, in the checkpoint's own
    /// spelling (`w1` gate, `w3` up, `w2` down). Fewer than three is a
    /// projection-scoped candidate, and the rest stay source-backed.
    /// Derived from the SEALS, for the same reason as `layers`.
    projections: Vec<String>,
    /// Physical representation per compiled LAYER — a composed map's
    /// layers need not share one (Q8_0 band beside a Q6_K layer).
    encodings: std::collections::BTreeMap<u32, ExpertEncoding>,
    /// Where each layer's bank sits in the segment — the same
    /// definition the compiler wrote and `verify_complete` proved.
    /// Geometry lives in the placement's layouts; the overlay keeps
    /// only the expert count, which the layouts do not carry.
    placement: CandidatePlacement,
    experts: u32,
}

impl CandidateOverlay {
    /// Open the overlay, verify its source dependency against the
    /// container actually present, and prove the ledger COMPLETE for
    /// every layer it claims: all `experts x 3` operands sealed at
    /// exactly the layout's offsets, no two seals overlapping.
    ///
    /// Completeness at load is what makes "remove one compiled operand"
    /// a refusal instead of a silent fallback — an Identity-addressed
    /// bank has no table entry to mark absent, so the ledger is the only
    /// witness that every slot's bytes were actually compiled.
    pub fn open(
        dir: &Path,
        source_dir: &Path,
        geometry: &KimiGeometry,
    ) -> Result<Self, VindexError> {
        let index: CandidateIndex = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)
            .map_err(|e| VindexError::Parse(format!("candidate index: {e}")))?;
        index.source.verify(&read_source_identity(source_dir)?)?;
        let (layers, projections, placement) = verify_complete(&index, geometry)?;
        // Per LAYER, from the placement the completeness proof ran on —
        // a composed map's layers need not share one encoding.
        let mut encodings = std::collections::BTreeMap::new();
        for &layer in &layers {
            let name = &placement.layout(layer)?.encoding;
            let enc = ExpertEncoding::parse(name).ok_or_else(|| {
                VindexError::Parse(format!(
                    "candidate encodes layer {layer} as `{name}`, which no grouped \
                     kernel reads"
                ))
            })?;
            encodings.insert(layer, enc);
        }
        let segment = dir.join("segments").join(format!("{}.bin", index.object));
        let store = Arc::new(PhysicalStore::map_compiled(
            // Encoding-agnostic on purpose: the id lands in evidence
            // attributions, and a Q8_0 candidate labelled "q6" would
            // misstate the representation under test.
            "kimi-candidate-bank",
            &segment,
            &index.ledger,
        )?);
        Ok(Self {
            index,
            store,
            layers,
            projections,
            encodings,
            placement,
            experts: geometry.experts,
        })
    }

    pub fn compiled_layers(&self) -> &[u32] {
        &self.layers
    }

    /// Which projections this candidate compiled. Fewer than three
    /// means the rest execute from the source.
    /// The physical representation this LAYER's sealed operands carry —
    /// from the map the compiler executed, verified parseable at open.
    pub fn encoding_of(&self, layer: u32) -> Result<ExpertEncoding, VindexError> {
        self.encodings.get(&layer).copied().ok_or_else(|| {
            VindexError::Parse(format!("layer {layer} is not compiled in this candidate"))
        })
    }

    pub fn compiled_projections(&self) -> &[String] {
        &self.projections
    }

    /// How the arm reads in a report: the scope, not the file name.
    pub fn scope(&self) -> String {
        format!(
            "layers {:?} / {} / {}",
            self.layers,
            if self.projections.len() == 3 {
                "all projections".to_string()
            } else {
                self.projections.join("+")
            },
            self.layers
                .iter()
                .map(|l| {
                    let enc = self.encodings.get(l).map(|e| e.name()).unwrap_or("?");
                    format!("L{l}:{enc}")
                })
                .collect::<Vec<_>>()
                .join(" ")
        )
    }

    /// **The bytes this overlay READS are the bytes the compiler
    /// WROTE**, proven per operand against the ledger's own
    /// `target_hash`.
    ///
    /// Region extent and seal offsets agreeing (`verify_complete`) shows
    /// the layout is self-consistent; it cannot show that the view is
    /// positioned where the writer put the payload. A whole-bank shift —
    /// a projection base off by one stride, a header the reader skips
    /// and the writer did not — satisfies every offset check and serves
    /// a neighbouring expert's weights, which decode to plausible
    /// numbers and would read as "quantisation is catastrophic".
    ///
    /// `sample` operands per projection, evenly spread across the
    /// expert range so a shift anywhere in the bank is caught.
    pub fn verify_reads_match_seals(
        &self,
        layer: u32,
        sample: usize,
    ) -> Result<usize, VindexError> {
        // Only the projections this candidate actually compiled: a
        // scoped one has nothing to say about the other two, and
        // demanding seals for them would refuse a complete overlay.
        let compiled: Vec<(String, RoutedProjection)> = self
            .projections
            .iter()
            .map(|proj| {
                let binding = self.projection_binding(layer, proj).ok_or_else(|| {
                    VindexError::Parse(format!("layer {layer} / {proj} is not compiled here"))
                })??;
                Ok((proj.clone(), binding))
            })
            .collect::<Result<_, VindexError>>()?;
        let mut checked = 0usize;
        let step = (self.experts as usize / sample.max(1)).max(1);
        for expert in (0..self.experts as usize).step_by(step) {
            for (proj, projection) in &compiled {
                let stride = match projection.addressing {
                    ProjectionAddressing::Identity { stride, .. } => stride,
                    ProjectionAddressing::Table(_) => {
                        return Err(VindexError::Parse(
                            "a compiled projection is identity-addressed by construction".into(),
                        ))
                    }
                };
                let tensor = format!("{layer}.block_sparse_moe.experts.{expert}.{proj}.weight");
                let seal = self
                    .index
                    .ledger
                    .get(&self.index.object, &tensor)
                    .ok_or_else(|| VindexError::Parse(format!("no seal for `{tensor}`")))?;
                let base = expert * stride as usize;
                let bytes = &projection.region.region.bytes()[base..base + stride as usize];
                let got = super::super::super::represent::compile::hash_bytes(bytes);
                if got != seal.target_hash {
                    return Err(VindexError::Parse(format!(
                        "`{tensor}`: the loader reads {} at slot {expert} but the compiler                          sealed {} — the view is not positioned where the payload was                          written, so this bank serves the wrong expert's weights",
                        &got[..12],
                        &seal.target_hash[..12]
                    )));
                }
                checked += 1;
            }
        }
        Ok(checked)
    }

    /// **One projection's compiled binding**, when this overlay holds
    /// that projection of that layer — `None` when it does not, which
    /// tells the caller to keep the source's.
    ///
    /// Per projection because a precision map may scope one: `w1` alone
    /// is a candidate, and the sweep asking WHICH projection initiates
    /// the routing cascade needs exactly that. `None` is a real answer
    /// rather than a failure, which is what makes composition additive
    /// over the source binding.
    ///
    /// No shared branch: the overlay compiles ROUTED experts only, and
    /// the caller attaches the shared binding from the source — the
    /// composition D0 exists to make expressible.
    pub fn projection_binding(
        &self,
        layer: u32,
        projection: &str,
    ) -> Option<Result<RoutedProjection, VindexError>> {
        if !self.layers.contains(&layer) || !self.projections.iter().any(|p| p == projection) {
            return None;
        }
        Some(self.binding_inner(layer, projection))
    }

    /// The whole routed bank, for an overlay that compiled all three
    /// projections. `None` if it compiled fewer.
    pub fn routed_binding(&self, layer: u32) -> Option<Result<ExpertBankBinding, VindexError>> {
        if !self.layers.contains(&layer) || self.projections.len() != 3 {
            return None;
        }
        Some((|| {
            Ok(ExpertBankBinding {
                gate: self.binding_inner(layer, "w1")?,
                up: self.binding_inner(layer, "w3")?,
                down: self.binding_inner(layer, "w2")?,
                shared: None,
            })
        })())
    }

    fn binding_inner(&self, layer: u32, projection: &str) -> Result<RoutedProjection, VindexError> {
        // The SAME placement `verify_complete` proved every seal
        // against — one definition of where an expert's bytes are.
        let layout = self.placement.layout(layer)?.clone();
        let layer_base = self.placement.layer_base(layer)?;
        if layout.gate_up_stride != layout.down_stride {
            return Err(VindexError::Parse(format!(
                "layer {layer}: gate/up stride {} != down stride {} — identity addressing \
                 carries ONE stride for all three projections",
                layout.gate_up_stride, layout.down_stride
            )));
        }
        let stride = u32::try_from(layout.gate_up_stride).map_err(|_| {
            VindexError::Parse("a compiled expert stride does not fit 32 bits".to_string())
        })?;
        let bank_bytes = layout.bank_bytes("w1")?;
        let encoding = self.encoding_of(layer)?;
        let region = |proj: &str| -> Result<EncodedRegion, VindexError> {
            let base = layer_base + bank_base(&layout, proj)?;
            Ok(EncodedRegion {
                region: self.store.span(base, bank_bytes).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "candidate segment is too short for the {proj} bank at {base}"
                    ))
                })?,
                encoding,
            })
        };
        // Every compiled projection is a full execution-shaped bank:
        // addressed by identity at its own stride, and exactly its own
        // region. The stride now travels WITH the projection rather
        // than beside the binding, so a caller cannot pair one
        // projection's bytes with another's stride.
        Ok(RoutedProjection {
            region: region(projection)?,
            addressing: ProjectionAddressing::Identity {
                experts: self.experts,
                stride,
            },
            // A compiled bank IS its region, exactly.
            extent: ExtentPolicy::Exact,
        })
    }

    pub fn store_id(&self) -> &str {
        self.store.id()
    }

    /// Register the compiled segment for zero-copy binding.
    pub fn register_store(&self, metal: &MetalBackend) {
        metal.register_weight_region(self.store.backing_bytes());
    }
}

/// Every operand the map compiled is sealed, at the layout's exact
/// offset and length. Returns the layers the ledger covers.
pub fn verify_complete(
    index: &CandidateIndex,
    geometry: &KimiGeometry,
) -> Result<(Vec<u32>, Vec<String>, CandidatePlacement), VindexError> {
    let overlaps = index.ledger.overlaps();
    if !overlaps.is_empty() {
        return Err(VindexError::Parse(format!(
            "candidate ledger has overlapping seals: {overlaps:?}"
        )));
    }
    let mut layers: Vec<u32> = index
        .ledger
        .sealed
        .values()
        .filter_map(|s| layer_of(&s.tensor))
        .collect();
    layers.sort_unstable();
    layers.dedup();
    if layers.is_empty() {
        return Err(VindexError::Parse(
            "candidate ledger seals nothing — an overlay with no overlay".to_string(),
        ));
    }
    // Which projections the candidate actually SEALED. A
    // projection-scoped map compiles one, and completeness is then a
    // claim about that one — demanding all three would refuse a
    // candidate that is complete for what it set out to hold.
    let mut projections: Vec<String> = index
        .ledger
        .sealed
        .values()
        .filter_map(|s| projection_of(&s.tensor).map(str::to_string))
        .collect();
    projections.sort();
    projections.dedup();
    // Whatever the map INTENDED, the seals are what exist. A map naming
    // a projection the ledger does not hold is an incomplete compile,
    // caught by the per-operand check below.
    if let Some(scoped) = index
        .map
        .exceptions
        .iter()
        .find(|e| e.encoding.is_some())
        .and_then(|e| e.projection.clone())
    {
        if !projections.contains(&scoped) {
            return Err(VindexError::Parse(format!(
                "the map scopes projection `{scoped}` but the ledger seals none of it"
            )));
        }
    }
    // ONE placement, the same definition the compiler wrote under: each
    // layer's base is the sum of the preceding compiled layers' extents,
    // each at its OWN encoding — which is what lets a composed map hold
    // a Q8_0 band beside a Q6_K layer in one candidate.
    let placement = CandidatePlacement::resolve(
        &index.map,
        Role::ExpertWeight,
        &layers,
        geometry.experts,
        geometry.hidden,
        geometry.moe_intermediate,
    )?;
    for &layer in &layers {
        let layout = placement.layout(layer)?;
        let layer_base = placement.layer_base(layer)?;
        for expert in 0..geometry.experts {
            for proj in projections.iter().map(String::as_str) {
                let tensor = format!("{layer}.block_sparse_moe.experts.{expert}.{proj}.weight");
                let seal = index.ledger.get(&index.object, &tensor).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "candidate ledger has no seal for `{tensor}` — the compiled bank is \
                         INCOMPLETE and an identity-addressed route to expert {expert} would \
                         read unsealed bytes; refusing rather than falling back to source"
                    ))
                })?;
                let slot = layout.slot(proj, expert)?;
                let want_offset = layer_base + bank_base(layout, proj)? + slot.offset;
                if seal.target_offset != want_offset || seal.target_len != slot.len {
                    return Err(VindexError::Parse(format!(
                        "`{tensor}` is sealed at {}+{} but the layout places it at \
                         {want_offset}+{} — the ledger and the layout disagree",
                        seal.target_offset, seal.target_len, slot.len
                    )));
                }
                if seal.encoding != layout.encoding {
                    return Err(VindexError::Parse(format!(
                        "`{tensor}` is sealed as {} but the map resolves layer {layer} \
                         to {}",
                        seal.encoding, layout.encoding
                    )));
                }
            }
        }
    }
    Ok((layers, projections, placement))
}
