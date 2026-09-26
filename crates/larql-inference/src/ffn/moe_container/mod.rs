//! The routed-bank override: expert bytes sourced from a VINDEX3 container.
//!
//! # What this closes
//!
//! Every VINDEX3 execution result before this one bound its operands out of a
//! **VINDEX2** file. `moe_bound` says so plainly, and the container ladder
//! (c8/c9) proved only that the same bytes could be *written* to a VINDEX3
//! container — not that anything would ever *read* one to compute with. That
//! left a gap nothing on the execution path could cross: `Vindex3Container`
//! appeared nowhere in this crate.
//!
//! ```text
//! before   VINDEX2 index ──> MoeLayerWeights ──> bound executor
//! now      VINDEX3 container ──> expert regions ─┘
//!                                (everything else still VINDEX2)
//! ```
//!
//! # Exactly one operand source is replaced
//!
//! The spine — tokenizer, config, embeddings, attention, norms, routers,
//! shared/dense FFN, LM head — is read from the VINDEX2 model, unchanged and
//! byte-for-byte. Only classes 4 and 5 (routed gate/up and routed down, spec
//! §4) come from the container. That is what makes a comparison against a
//! plain VINDEX2 run a statement about *the routed bytes* and nothing else: if
//! this route substituted anything further, a divergence would have somewhere
//! else to hide.
//!
//! This is also the K3 deployment topology in miniature — a small resident
//! spine plus huge paged routed banks — but it is **not** artifact identity. A
//! composed run is two directories, and a VINDEX3 model that still needs a
//! VINDEX2 directory to run is not yet a model. Container completeness (the
//! `control/`, `dense/`, `shared/` and sidecar classes of spec §5) is a
//! separate job and this type does not pretend to stand in for it.
//!
//! # Public API only, deliberately
//!
//! Regions are reached through `Vindex3Container -> segment -> region_bytes`
//! and never by parsing LYRW descriptors here. The binary layout is mid-change
//! (24 B → 28 B bank descriptors with an explicit `group_width`); a backend
//! that read the descriptor itself would have to be rewritten alongside it,
//! and would have been a second implementation of the reader in the meantime.
//!
//! # No fallback, ever
//!
//! If a layer, an expert or a role is missing, this refuses. It does not quietly
//! serve the VINDEX2 bytes that are sitting right there in the same process —
//! doing so would make "the model ran from VINDEX3" unfalsifiable, which is the
//! only claim the route exists to support.

use std::path::Path;

use larql_models::ModelWeights;
use ndarray::Array2;

use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;
use larql_vindex::format::lyrw2::read::Lyrw2Reader;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::format::vindex3::import::routed_storage_key;
use larql_vindex::format::vindex3::Vindex3Container;

use super::moe_backend::{MoeBackendError, MoeExpertBackend};
use super::moe_bound::BoundMoeBackend;

/// Route name for diagnostics. Never branched on.
const ROUTE_NAME: &str = "moe-vindex3-container";

/// Bank ordinal of the routed bank within a segment. c8/c9 write one bank per
/// segment file, and spec §6 (draft-3) makes that a MUST.
const ROUTED_BANK: u16 = 0;

/// Expert bytes served from a VINDEX3 container, everything else from VINDEX2.
pub struct ContainerRoutedBackend {
    container: Vindex3Container,
    inner: BoundMoeBackend,
}

/// Why a container could not be composed with a loaded model.
///
/// Every variant names both sides. A composition refusal that said only
/// "incompatible" would send the reader to the wrong one of two artifacts.
#[derive(Debug)]
pub enum CompositionError {
    Open(String),
    /// The container describes a different model than the spine does.
    Identity {
        spine: String,
        container: String,
    },
    /// A routed layer the model needs is not in the container.
    MissingLayer {
        layer: usize,
    },
    /// The container's segment for a layer cannot be resolved or read.
    UnreadableLayer {
        layer: usize,
        why: String,
    },
    /// A shape the container declares disagrees with the model's.
    Shape {
        layer: usize,
        what: &'static str,
        spine: u64,
        container: u64,
    },
}

impl std::fmt::Display for CompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(why) => write!(f, "cannot open the routed container: {why}"),
            Self::Identity { spine, container } => write!(
                f,
                "the container describes `{container}` but the model is `{spine}` — \
                 composing them would serve one model's experts inside another"
            ),
            Self::MissingLayer { layer } => write!(
                f,
                "layer {layer} routes to experts but the container has no bank for \
                 it; `--routed-from` never falls back to the VINDEX2 bytes, because \
                 a run that silently did could not support the claim it exists for"
            ),
            Self::UnreadableLayer { layer, why } => {
                write!(f, "layer {layer}'s bank is unreadable: {why}")
            }
            Self::Shape {
                layer,
                what,
                spine,
                container,
            } => write!(
                f,
                "layer {layer}: the model expects {what} {spine}, the container \
                 declares {container}"
            ),
        }
    }
}

impl std::error::Error for CompositionError {}

impl ContainerRoutedBackend {
    /// Open `root` and check it can serve every routed layer `weights` needs.
    ///
    /// All validation happens here — before a prompt is encoded, before a
    /// token is generated. A composition that failed on layer 17 of the first
    /// forward pass would have already printed part of an answer produced by a
    /// model half of which was not the one asked for.
    pub fn open(
        root: &Path,
        weights: &ModelWeights,
        production: bool,
    ) -> Result<Self, CompositionError> {
        let container =
            Vindex3Container::open(root).map_err(|e| CompositionError::Open(e.to_string()))?;

        let backend = Self {
            container,
            inner: if production {
                BoundMoeBackend::production()
            } else {
                BoundMoeBackend::reference()
            },
        };
        backend.check_composable(weights)?;
        Ok(backend)
    }

    /// Refuse any container that cannot serve this model's routed layers.
    ///
    /// Coverage is checked against the **model**, not the container: a
    /// container with thirty banks still cannot serve a model with thirty-one
    /// routed layers, and asking the container what it has would let the
    /// missing one pass unnoticed.
    fn check_composable(&self, weights: &ModelWeights) -> Result<(), CompositionError> {
        let arch = &*weights.arch;
        let spine_model = weights.arch.family().to_string();
        let container_model = self.container.index().family.clone();
        if spine_model != container_model {
            return Err(CompositionError::Identity {
                spine: spine_model,
                container: container_model,
            });
        }

        for layer in 0..weights.num_layers {
            let Some(moe) = build_moe_weights(weights, arch, layer) else {
                continue; // A dense layer routes nowhere; the container owes it nothing.
            };
            let declared = self
                .container
                .layer(layer as u32)
                .ok_or(CompositionError::MissingLayer { layer })?;

            let experts = declared.routed_bank.experts as usize;
            if experts != moe.num_experts {
                return Err(CompositionError::Shape {
                    layer,
                    what: "experts",
                    spine: moe.num_experts as u64,
                    container: experts as u64,
                });
            }
            if let Some(dims) = declared.routed_bank.expert_dims.as_ref() {
                if dims.input as usize != weights.hidden_size {
                    return Err(CompositionError::Shape {
                        layer,
                        what: "hidden size",
                        spine: weights.hidden_size as u64,
                        container: dims.input as u64,
                    });
                }
                if dims.intermediate as usize != moe.intermediate_size {
                    return Err(CompositionError::Shape {
                        layer,
                        what: "semantic intermediate width",
                        spine: moe.intermediate_size as u64,
                        container: dims.intermediate as u64,
                    });
                }
            }

            // Resolve and read the bank now rather than at first token: a key
            // that resolves to a missing file is a composition failure, and
            // discovering it mid-generation would be one too late.
            self.check_layer_regions(layer, experts, &moe)?;
        }
        Ok(())
    }

    /// Every expert's two regions must be present and the size the model expects.
    fn check_layer_regions(
        &self,
        layer: usize,
        experts: usize,
        moe: &MoeLayerWeights<'_>,
    ) -> Result<(), CompositionError> {
        let reader = self
            .container
            .segment(&routed_storage_key(layer as u32))
            .map_err(|e| CompositionError::UnreadableLayer {
                layer,
                why: e.to_string(),
            })?;
        // The byte authority is the CONTAINER's declared representation,
        // not the model's in-memory one: a native bank legitimately
        // differs from the spine's transcode, and asking the model what
        // the container should weigh reintroduces the coupling VINDEX3
        // exists to remove. Semantic dims stay the model's; only the
        // bytes-per-dim formula comes from the declaration. Formats
        // without a declared formula keep the legacy same-representation
        // contract (the model's own slice lengths) — strictness is
        // preserved either way, only the authority moves.
        let declared = reader
            .resolve(ROUTED_BANK, SCHEMA_ENTRY, RegionRole::GateUpFused)
            .map_err(|e| CompositionError::UnreadableLayer {
                layer,
                why: format!("gate/up schema: {e}"),
            })?
            .ok_or_else(|| CompositionError::UnreadableLayer {
                layer,
                why: "no gate/up region".into(),
            })?
            .schema
            .format;
        let (want_gate_up, want_down) = match declared_region_sizes(
            declared,
            moe.intermediate_size,
            moe.num_experts,
            self.container.index().hidden_size,
        ) {
            Some(native) => native,
            None => expert_region_sizes(moe),
        };

        for expert in 0..experts as u32 {
            for (role, want) in [
                (RegionRole::GateUpFused, want_gate_up),
                (RegionRole::Down, want_down),
            ] {
                let got = reader
                    .region_bytes(ROUTED_BANK, expert, role)
                    .map_err(|e| CompositionError::UnreadableLayer {
                        layer,
                        why: format!("expert {expert} {role:?}: {e}"),
                    })?
                    .ok_or_else(|| CompositionError::UnreadableLayer {
                        layer,
                        why: format!("expert {expert} has no {role:?} region"),
                    })?;
                if got.len() != want {
                    return Err(CompositionError::Shape {
                        layer,
                        what: "routed region bytes",
                        spine: want as u64,
                        container: got.len() as u64,
                    });
                }
            }
        }

        // A format whose scales live in partner streams is unservable
        // without them, and a stream of the wrong length resolves into the
        // wrong expert's exponents — arithmetically valid, silently wrong.
        // Both are composition failures, caught here for the same reason
        // the payload sizes are.
        if let Some((want_gu_scales, want_dn_scales)) = declared_scale_sizes(
            declared,
            moe.intermediate_size,
            self.container.index().hidden_size,
        ) {
            for expert in 0..experts as u32 {
                for (role, want) in [
                    (RegionRole::GateUpFused, want_gu_scales),
                    (RegionRole::Down, want_dn_scales),
                ] {
                    let got = reader
                        .paired_region_bytes(ROUTED_BANK, expert, role, RegionRole::Scales)
                        .map_err(|e| CompositionError::UnreadableLayer {
                            layer,
                            why: format!("expert {expert} {role:?} scales: {e}"),
                        })?
                        .ok_or_else(|| CompositionError::UnreadableLayer {
                            layer,
                            why: format!(
                                "expert {expert} {role:?} declares a split-scale \
                                 format but carries no scale partner"
                            ),
                        })?;
                    if got.len() != want {
                        return Err(CompositionError::Shape {
                            layer,
                            what: "scale partner bytes",
                            spine: want as u64,
                            container: got.len() as u64,
                        });
                    }
                }
            }
        }

        // Resolve the two storage facts here as well, and discard them. They
        // are re-read per forward, but a bank whose scales are half-present or
        // whose row arrangement this build cannot act on must fail at
        // composition — the same reason the payload regions are read now
        // rather than at first token. Deferring it to the forward turns a
        // startup refusal into a mid-generation one.
        let describe = |e: MoeBackendError| CompositionError::UnreadableLayer {
            layer,
            why: e.to_string(),
        };
        read_bank_storage(&reader, layer, experts).map_err(describe)?;
        Ok(())
    }

    /// This layer's expert bank as an override for a spine-built layer —
    /// the routed container's representation authority, packaged for
    /// `MoeLayerWeights::apply_expert_bank_override`. `Ok(None)` when the
    /// spine layer routes nowhere (dense layers owe the container nothing).
    ///
    /// Everything returned borrows from `self`; the container must
    /// outlive the layer views it feeds (the composing caller owns both,
    /// which is what makes this a reference move and never a bank copy).
    pub fn expert_bank_override(
        &self,
        num_experts: usize,
        spine_format: larql_compute::QuantFormat,
        layer: usize,
    ) -> Result<larql_compute::ExpertBankOverride<'_>, MoeBackendError> {
        let reader = self
            .container
            .segment(&routed_storage_key(layer as u32))
            .map_err(|e| MoeBackendError::Container(format!("layer {layer} bank: {e}")))?;

        let mut gate_up = Vec::with_capacity(num_experts);
        let mut down = Vec::with_capacity(num_experts);
        for expert in 0..num_experts as u32 {
            for (role, sink) in [
                (RegionRole::GateUpFused, &mut gate_up),
                (RegionRole::Down, &mut down),
            ] {
                let bytes = reader
                    .region_bytes(ROUTED_BANK, expert, role)
                    .map_err(|e| {
                        MoeBackendError::Container(format!(
                            "layer {layer} expert {expert} {role:?}: {e}"
                        ))
                    })?
                    .ok_or_else(|| {
                        MoeBackendError::Container(format!(
                            "layer {layer} expert {expert} has no {role:?} region"
                        ))
                    })?;
                sink.push(bytes);
            }
        }
        let storage = read_bank_storage(&reader, layer, num_experts)?;
        Ok(larql_compute::ExpertBankOverride {
            experts_gate_up: gate_up,
            experts_down: down,
            expert_scales: storage.scales,
            fused_row_layout: storage.fused_row_layout,
            // The container's declared representation travels with its
            // bytes; a declaration without a compute binding keeps the
            // spine's format (the legacy same-representation contract —
            // the caller states it, nothing is hardcoded here).
            expert_data_format: declared_quant_format(storage.format).unwrap_or(spine_format),
        })
    }

    /// The container's mapped segment regions, for the compute backend to
    /// register alongside the spine's packed mmaps — the routed bank's
    /// bytes must alias into GPU buffers the same way the spine's do, or
    /// zero-copy resolve refuses every expert the override supplies.
    pub fn weight_regions(&self) -> impl Iterator<Item = &[u8]> {
        self.container.segment_regions()
    }

    /// One line describing what this run actually composed.
    pub fn describe(&self, spine: &Path) -> String {
        format!(
            "composed run: VINDEX2 spine {} + legacy LYRW bank-shape routed banks {}",
            spine.display(),
            self.container.root().display()
        )
    }
}

/// The byte length each of a layer's two region kinds must have.
///
/// Taken from the incumbent's own slices rather than recomputed from shapes:
/// the model already holds the authoritative lengths, and deriving them here
/// would be a second implementation of the layout to disagree with.
fn expert_region_sizes(moe: &MoeLayerWeights<'_>) -> (usize, usize) {
    (
        moe.experts_gate_up.first().map_or(0, |s| s.len()),
        moe.experts_down.first().map_or(0, |s| s.len()),
    )
}

/// Payload byte lengths a bank of `format` must have, from the
/// container's declared representation + the model's semantic dims —
/// or `None` for formats whose containers carry the model's own
/// representation (the legacy contract: model slice lengths apply).
fn declared_region_sizes(
    format: RegionFormat,
    inter: usize,
    _experts: usize,
    hidden: usize,
) -> Option<(usize, usize)> {
    use larql_models::quant::mxfp4::{FUSED_HALVES, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};
    match format {
        RegionFormat::Mxfp4 => {
            let row = |cols: usize| (cols / MXFP4_GROUP_ELEMS) * MXFP4_GROUP_BYTES;
            Some((FUSED_HALVES * inter * row(hidden), hidden * row(inter)))
        }
        _ => None,
    }
}

/// Scale-partner byte lengths a bank of `format` must carry per expert —
/// `None` for formats whose scales ride inside the payload blocks.
///
/// For a split-scale format this is one e8m0 exponent per group of
/// [`MXFP4_GROUP_ELEMS`](larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS)
/// input elements, per output row: the gate/up region is
/// `[FUSED_HALVES x inter, hidden]` and the down region `[hidden, inter]`.
/// A `Some` here also states the format cannot be served without its
/// partner streams — nibbles with no exponents decode to nothing.
fn declared_scale_sizes(
    format: RegionFormat,
    inter: usize,
    hidden: usize,
) -> Option<(usize, usize)> {
    use larql_models::quant::mxfp4::{FUSED_HALVES, MXFP4_GROUP_ELEMS};
    match format {
        RegionFormat::Mxfp4 => Some((
            FUSED_HALVES * inter * (hidden / MXFP4_GROUP_ELEMS),
            hidden * (inter / MXFP4_GROUP_ELEMS),
        )),
        _ => None,
    }
}

/// The compute-tier format a container declaration binds to, when it
/// differs from carrying the model's own representation.
fn declared_quant_format(format: RegionFormat) -> Option<larql_compute::QuantFormat> {
    match format {
        RegionFormat::Mxfp4 => Some(larql_compute::QuantFormat::MXFP4),
        _ => None,
    }
}

/// The two payload roles a routed bank always carries, in operand order.
const PAYLOAD_ROLES: [RegionRole; 2] = [RegionRole::GateUpFused, RegionRole::Down];

/// The entry whose schema speaks for the bank. Schemas are per-bank, not
/// per-entry, so any entry answers — expert 0 is simply the one that always
/// exists.
const SCHEMA_ENTRY: u32 = 0;

/// What a routed bank declares about how it is stored, as the compute tier
/// spells it.
///
/// Read together because they come from one place — the gate/up region's
/// schema — and because a consumer needs both or neither: knowing the rows
/// are interleaved is no use without the exponents that decode them.
struct BankStorage<'a> {
    scales: larql_compute::MoeExpertScales<'a>,
    fused_row_layout: larql_compute::MoeFusedRowLayout,
    format: RegionFormat,
}

/// Read both storage facts off the container's own schema.
///
/// Neither is inferred from the format. A codec does not determine where its
/// scales sit, and a checkpoint's row arrangement does not survive a
/// canonicalising extraction — the schema is the only thing that knows, which
/// is the whole reason it records them.
fn read_bank_storage<'a>(
    reader: &Lyrw2Reader<'a>,
    layer: usize,
    num_experts: usize,
) -> Result<BankStorage<'a>, MoeBackendError> {
    let refuse = |why: String| MoeBackendError::Container(format!("layer {layer}: {why}"));

    let gate_up = reader
        .resolve(ROUTED_BANK, SCHEMA_ENTRY, RegionRole::GateUpFused)
        .map_err(|e| refuse(format!("gate/up schema: {e}")))?
        .ok_or_else(|| refuse("no gate/up region".into()))?
        .schema;

    let fused_row_layout = gate_up.layout.fused_row_layout().ok_or_else(|| {
        refuse(format!(
            "gate/up rows are declared `{}`, which this binary cannot act on. \
             Reading it as contiguous halves would silently mix the gate and \
             up branches; re-extract with a layout this build understands.",
            gate_up.layout.name()
        ))
    })?;

    // Ask the schema whether a partner exists, rather than attempting the
    // lookup and reading its error as "no". `resolve_paired` treats a partner
    // request on an unpaired region as a caller mistake by design, and that
    // same error path also carries `MissingPartner` — a corrupt bank, which
    // must not be silently served as an inline one.
    if !gate_up.is_paired() {
        return Ok(BankStorage {
            scales: larql_compute::MoeExpertScales::Inline,
            fused_row_layout,
            format: gate_up.format,
        });
    }
    let format = gate_up.format;

    let mut streams: [Vec<&'a [u8]>; PAYLOAD_ROLES.len()] = Default::default();
    for (slot, role) in PAYLOAD_ROLES.iter().enumerate() {
        for expert in 0..num_experts as u32 {
            let bytes = reader
                .paired_region_bytes(ROUTED_BANK, expert, *role, RegionRole::Scales)
                .map_err(|e| refuse(format!("expert {expert} {role:?} scales: {e}")))?
                .ok_or_else(|| {
                    refuse(format!(
                        "expert {expert} has no {role:?} scale partner, but the \
                         bank declares itself paired — a split-scale bank must \
                         carry exponents for every expert"
                    ))
                })?;
            streams[slot].push(bytes);
        }
    }
    let [gate_up_scales, down_scales] = streams;
    Ok(BankStorage {
        scales: larql_compute::MoeExpertScales::Paired {
            gate_up: gate_up_scales,
            down: down_scales,
        },
        fused_row_layout,
        format,
    })
}

impl MoeExpertBackend for ContainerRoutedBackend {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        let arch = &*weights.arch;
        let Some(mut moe) = build_moe_weights(weights, arch, layer) else {
            // No experts to route into. Zeros, matching the in-process path —
            // erroring here would change the model rather than the route.
            return Ok(Array2::zeros((h.nrows(), h.ncols())));
        };

        // Replace *only* the routed operands. Every other field of `moe` —
        // router projection and scales, the norms, the policy flags — stays as
        // the VINDEX2 model produced it.
        let reader = self
            .container
            .segment(&routed_storage_key(layer as u32))
            .map_err(|e| MoeBackendError::Container(format!("layer {layer} bank: {e}")))?;

        let mut gate_up = Vec::with_capacity(moe.num_experts);
        let mut down = Vec::with_capacity(moe.num_experts);
        for expert in 0..moe.num_experts as u32 {
            for (role, sink) in [
                (RegionRole::GateUpFused, &mut gate_up),
                (RegionRole::Down, &mut down),
            ] {
                let bytes = reader
                    .region_bytes(ROUTED_BANK, expert, role)
                    .map_err(|e| {
                        MoeBackendError::Container(format!(
                            "layer {layer} expert {expert} {role:?}: {e}"
                        ))
                    })?
                    .ok_or_else(|| {
                        MoeBackendError::Container(format!(
                            "layer {layer} expert {expert} has no {role:?} region"
                        ))
                    })?;
                sink.push(bytes);
            }
        }
        moe.experts_gate_up = gate_up;
        moe.experts_down = down;
        // Both remaining facts are read off the container, not inferred from
        // the format: a codec does not determine where its scales sit, and a
        // checkpoint's row arrangement does not survive a canonicalising
        // extraction. The schema is the only thing that knows.
        let storage = read_bank_storage(&reader, layer, moe.num_experts)?;
        moe.expert_scales = storage.scales;
        moe.fused_row_layout = storage.fused_row_layout;
        // The container's declared representation travels with its bytes:
        // a native bank must not execute under the spine's format.
        if let Some(qf) = declared_quant_format(storage.format) {
            moe.expert_data_format = qf;
        }

        Ok(self.inner.run_layer(layer, &moe, h, norm_offset, eps)?)
    }

    fn name(&self) -> &'static str {
        ROUTE_NAME
    }
}

#[cfg(test)]
mod tests;
