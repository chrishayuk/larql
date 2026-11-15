//! Pending stored-weight ops as the session executes them.
//!
//! A stored-weight op (`weight_delta`, `structured_weight`, `low_rank_columns_weight`) names elements of a stored
//! BF16 matrix. Compiling writes a container that holds the result. Until then the op lives in the overlay, and
//! this is how execution sees it meanwhile:
//!
//! ```text
//! weight ops in effect
//!     → declare_patches + materialise_declared_patches   (COMPILE's own resolution and byte path; nothing written)
//!     → each patched tensor's whole BF16 bytes
//!     → OperandOverrides::replace_bf16                   (a whole-operand stored-byte replacement)
//! ```
//!
//! so a session with ops pending and a fresh binding of the container compiled from them select the same
//! realization for every patched tensor and load the same bytes into it.
//!
//! The view is built once per change of the op set ([`KnowledgeOverlay::refresh_live_weights`]), not per
//! execution, and it is the overlay's own mutators that mark it stale. Everything here fails closed: with ops in
//! effect, [`KnowledgeOverlay::live_weight_overrides`] hands back replacements for every one of them or a refusal
//! that says why. It never hands back "nothing", which would execute the stored weights.

use std::path::Path;
use std::sync::Arc;

use super::KnowledgeOverlay;
use crate::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use crate::format::vindex3::opplan::exec::operands::{
    OperandOverrides, OperandStore, RepresentationSource,
};
use crate::format::vindex3::opplan::exec::residency_record::source_name;
use crate::format::vindex3::opplan::ComponentOpPlan;
use crate::format::vindex3::weight_patch::{
    declare_patches, materialise_declared_patches, WeightOpRef,
};

pub mod read_path;

/// What a session executes on: where its pending ops are checked against.
pub struct ExecutionBinding<'a> {
    /// The bound container: the source the ops were written against, and compile's source.
    pub container: &'a Path,
    pub plan: &'a ComponentOpPlan,
    pub store: &'a OperandStore,
    /// The lowering provider the session executes through.
    pub lowering: LoweringIdentity,
}

/// Why a session cannot execute with its pending stored-weight ops.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LiveWeightRefusal {
    #[error(
        "the session's representation source is `{requested}`, under which the loader may quantise a patched \
         matrix at load and execute that instead; stored-weight ops execute only under `stored`, where the \
         patched bytes are the bytes computed with"
    )]
    SourceNotStored { requested: &'static str },
    #[error(
        "the session executes through `{lowering}`; stored-weight ops execute only through `{qualified}`"
    )]
    LoweringNotQualified { lowering: String, qualified: String },
    #[error(
        "knowledge edits (INSERT / UPDATE / DELETE, or a patch carrying them) and stored-weight ops are both \
         pending. COMPILE refuses that mix, so no container would hold what this session would compute; \
         remove one of them"
    )]
    KnowledgePending,
    #[error(
        "tensor `{tensor}` is not a matrix operand the execution plan lists, so nothing shows execution would \
         read its edit"
    )]
    NotExecuted { tensor: String },
    #[error("tensor `{tensor}` is executed as `{operation}`, and {why}")]
    OutsideSeam {
        tensor: String,
        operation: &'static str,
        why: &'static str,
    },
    #[error(
        "object `{object}` is bound to its compiled `{encoding}` pack in this session, not to the canonical \
         weights `{tensor}` is edited in"
    )]
    BoundToPack {
        tensor: String,
        object: String,
        encoding: String,
    },
    #[error(
        "the plan executes `{tensor}` as `{planned}`, which is not the `{patched}` the edit compiles to"
    )]
    OperandMismatch {
        tensor: String,
        planned: String,
        patched: String,
    },
    /// What `COMPILE CURRENT INTO VINDEX` would refuse before it wrote anything.
    #[error("COMPILE would refuse these ops: {0}")]
    Compile(String),
    #[error(
        "the stored-weight ops changed and their live view was not rebuilt; refusing to execute the stored \
         weights in their place"
    )]
    Stale,
}

/// One tensor a live view replaces.
#[derive(Clone)]
pub struct LiveTensor {
    /// The graph object whose canonical representation holds the tensor.
    pub object: String,
    pub tensor: String,
    pub shape: [usize; 2],
    /// SHA-256 of [`Self::bytes`]: equal to hashing this tensor's bytes in a container compiled from the ops.
    pub sha256: String,
    /// The whole tensor as COMPILE would store it: BF16, little-endian, row-major.
    pub bytes: Arc<[u8]>,
}

impl std::fmt::Debug for LiveTensor {
    /// The tensor's identity and digest, never its bytes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveTensor")
            .field("object", &self.object)
            .field("tensor", &self.tensor)
            .field("shape", &self.shape)
            .field("sha256", &self.sha256)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

/// The stored-weight ops in effect, as bytes execution loads.
#[derive(Debug)]
pub struct LiveWeightView {
    tensors: Vec<LiveTensor>,
    overrides: OperandOverrides,
}

/// Which ops something was built for: their digests, in the order they are in effect. An op's digest covers
/// everything it computes with, so two op sets with the same identity materialise the same bytes.
pub(super) type OpSetIdentity = Vec<String>;

impl LiveWeightView {
    /// Every patched tensor, in the order each was first edited.
    pub fn tensors(&self) -> &[LiveTensor] {
        &self.tensors
    }
}

/// What the overlay holds for its weight ops.
#[derive(Debug, Default)]
pub(super) enum LiveWeightState {
    /// The op set changed since anything was built for it (or nothing was ever built).
    #[default]
    Stale,
    /// The view of exactly the ops `built_for` names.
    Ready {
        built_for: OpSetIdentity,
        view: LiveWeightView,
    },
    /// Why exactly the ops `built_for` names cannot execute.
    Refused {
        built_for: OpSetIdentity,
        refusal: LiveWeightRefusal,
    },
}

impl KnowledgeOverlay {
    /// Rebuild the live view of the stored-weight ops in effect, and report it as execution will see it. Call
    /// after anything that may have changed them: a patch applied or removed, an op staged.
    pub fn refresh_live_weights(
        &mut self,
        binding: &ExecutionBinding<'_>,
    ) -> Result<Option<&LiveWeightView>, LiveWeightRefusal> {
        self.live = if self.has_weight_ops() {
            let built_for = self.weight_op_identity();
            let ops: Vec<WeightOpRef<'_>> =
                self.weight_ops().into_iter().map(|(_, op)| op).collect();
            match build_view(&ops, binding) {
                Ok(view) => LiveWeightState::Ready { built_for, view },
                Err(refusal) => LiveWeightState::Refused { built_for, refusal },
            }
        } else {
            LiveWeightState::Stale
        };
        self.live_weights()
    }

    /// The live view execution would run over: `None` with no stored-weight op in effect, the view when every
    /// op is materialised, and otherwise the reason nothing may execute.
    pub fn live_weights(&self) -> Result<Option<&LiveWeightView>, LiveWeightRefusal> {
        if !self.has_weight_ops() {
            return Ok(None);
        }
        if self.has_knowledge_state() {
            return Err(LiveWeightRefusal::KnowledgePending);
        }
        // What was built is served only for the ops it was built for. A mutator that changed the op set and
        // did not mark the view stale is caught here, by comparing identities, not trusted to have remembered.
        let current = self.weight_op_identity();
        match &self.live {
            LiveWeightState::Ready { built_for, view } if *built_for == current => Ok(Some(view)),
            LiveWeightState::Refused { built_for, refusal } if *built_for == current => {
                Err(refusal.clone())
            }
            LiveWeightState::Ready { .. }
            | LiveWeightState::Refused { .. }
            | LiveWeightState::Stale => Err(LiveWeightRefusal::Stale),
        }
    }

    /// The identity of the stored-weight ops in effect now.
    fn weight_op_identity(&self) -> OpSetIdentity {
        self.weight_ops()
            .into_iter()
            .map(|(_, op)| op.digest().to_string())
            .collect()
    }

    /// The replacements an execution runs over for the stored-weight ops in effect: `None` with no such op,
    /// otherwise a fresh copy of the view's (so [`OperandOverrides::unread_patched`] witnesses one execution)
    /// or the refusal. With ops in effect this is never `None`, which would execute the stored weights.
    ///
    /// A caller with no ops in effect derives its compose edits as before ([`Self::operand_overrides`]); the
    /// two never combine, since [`Self::live_weights`] refuses while knowledge edits are pending.
    pub fn live_weight_overrides(&self) -> Result<Option<OperandOverrides>, LiveWeightRefusal> {
        Ok(self.live_weights()?.map(|view| view.overrides.clone()))
    }

    /// Mark the live view stale. Every mutator of the op set calls this; [`Self::live_weights`] does not
    /// depend on it, since it also holds what was built to the identity of the ops now in effect.
    pub(super) fn invalidate_live_weights(&mut self) {
        self.live = LiveWeightState::Stale;
    }
}

/// Materialise `ops` for `binding`, refusing whatever execution could not honour exactly.
fn build_view(
    ops: &[WeightOpRef<'_>],
    binding: &ExecutionBinding<'_>,
) -> Result<LiveWeightView, LiveWeightRefusal> {
    let requested = binding.store.representation_source();
    if requested != RepresentationSource::Stored {
        return Err(LiveWeightRefusal::SourceNotStored {
            requested: source_name(requested),
        });
    }
    let qualified = LoweringIdentity::cpu_production();
    if binding.lowering != qualified {
        return Err(LiveWeightRefusal::LoweringNotQualified {
            lowering: binding.lowering.to_string(),
            qualified: qualified.to_string(),
        });
    }
    // Structure before bytes: an op execution cannot honour is refused without reading a tensor.
    for op in ops {
        read_path::seam_operands(binding.plan, op.tensor())?;
    }
    let compile = |e: crate::format::vindex3::weight_patch::WeightPatchError| {
        LiveWeightRefusal::Compile(e.to_string())
    };
    let (patches, _origins) = declare_patches(binding.container, ops).map_err(compile)?;
    let materialised =
        materialise_declared_patches(binding.container, &patches).map_err(compile)?;

    let mut tensors = Vec::with_capacity(materialised.len());
    let mut overrides = OperandOverrides::new();
    for tensor in materialised {
        if let Some(selected) = binding
            .store
            .selection()
            .get(&tensor.object)
            .filter(|selected| selected.stored)
        {
            return Err(LiveWeightRefusal::BoundToPack {
                tensor: tensor.tensor,
                object: tensor.object,
                encoding: selected.encoding.clone(),
            });
        }
        let sha256 = tensor.sha256();
        let bytes: Arc<[u8]> = Arc::from(tensor.bytes);
        for operand in read_path::seam_operands(binding.plan, &tensor.tensor)? {
            // The replacement is installed only for an operand that is this tensor, shape and bytes.
            let installed = operand.object == tensor.object
                && operand.shape == tensor.shape
                && overrides.replace_bf16(&operand, Arc::clone(&bytes)).is_ok();
            if !installed {
                return Err(LiveWeightRefusal::OperandMismatch {
                    tensor: tensor.tensor,
                    planned: format!("{} {:?}", operand.object, operand.shape),
                    patched: format!("{} {:?}", tensor.object, tensor.shape),
                });
            }
        }
        tensors.push(LiveTensor {
            object: tensor.object,
            tensor: tensor.tensor,
            shape: tensor.shape,
            sha256,
            bytes,
        });
    }
    Ok(LiveWeightView { tensors, overrides })
}
