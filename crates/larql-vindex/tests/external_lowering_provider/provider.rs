//! **The provider this build does not ship.**
//!
//! An integration test is a separate crate in cargo's model, so
//! everything below is written against what `larql-vindex` exports and
//! nothing else. What the provider OWNS is the part that has to be its
//! own for the claim to mean anything: the candidates it derives from
//! declared representation facts, the realization it pins, the refusal it
//! raises when nothing is admissible, and the kernel that multiplies. The
//! glue between projections — norms, RoPE, softmax, activations, the
//! residual write — it borrows from the reference oracle, exactly as the
//! in-tree device provider borrows the production CPU glue, and for the
//! same reason: what is being demonstrated is authority, not arithmetic.
//!
//! Every method counts its dispatches, so the test can say which parts of
//! the seam the interpreter actually asked this provider for rather than
//! assuming.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::opplan::exec::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, FfnCall, NormCall,
    PlanBackend, ProjectCall, RoutedFfnCall, WeightFormat, WeightSlice,
};
use larql_vindex::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;
use larql_vindex::format::vindex3::opplan::exec::kernels::activate;
use larql_vindex::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use larql_vindex::format::vindex3::opplan::exec::realization::{
    common_selection, realization_residency, RealizationForm, RealizationId, RefusalKind,
    RepresentationFacts, Selection, SelectionReason, SelectionRefusal,
};
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::planned::PlannedOperand;

/// The family this crate registers and `larql-vindex` does not. The
/// genericity scan is over this string.
pub const FAMILY: &str = "outside-lowering";
pub const REVISION: u32 = 1;
/// The diagnostic name — also scanned for, since a provider leaking into
/// core by its NAME would be no better than by its family.
pub const NAME: &str = "outside-lowering-r1";

/// A SECOND external provider, equally capable: the same candidate
/// derivation, the same kernel, a different identity. It exists so that
/// "the caller named this provider and got it" can be told apart from
/// "it was the only implementation that could have run this plan".
pub const SIBLING_FAMILY: &str = "outside-lowering-sibling";
pub const SIBLING_NAME: &str = "outside-lowering-sibling-r1";

pub fn identity() -> LoweringIdentity {
    LoweringIdentity::new(FAMILY, REVISION)
}

pub fn sibling_identity() -> LoweringIdentity {
    LoweringIdentity::new(SIBLING_FAMILY, REVISION)
}

/// What the interpreter asked this provider for, by seam method.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Dispatches {
    pub embed: u64,
    pub norm: u64,
    pub project: u64,
    pub attention: u64,
    pub attention_step: u64,
    pub ffn: u64,
    pub routed_ffn: u64,
    pub output_head: u64,
    pub residual_add: u64,
    /// Matrices this provider multiplied with its OWN kernel — the work,
    /// as distinct from the calls.
    pub matrices: u64,
}

impl Dispatches {
    pub fn total(&self) -> u64 {
        self.embed
            + self.norm
            + self.project
            + self.attention
            + self.attention_step
            + self.ffn
            + self.routed_ffn
            + self.output_head
            + self.residual_add
    }
}

#[derive(Default)]
pub struct Counts {
    embed: AtomicU64,
    norm: AtomicU64,
    project: AtomicU64,
    attention: AtomicU64,
    attention_step: AtomicU64,
    ffn: AtomicU64,
    routed_ffn: AtomicU64,
    output_head: AtomicU64,
    residual_add: AtomicU64,
    matrices: AtomicU64,
}

/// A handle on one provider's dispatch counters, kept by the test after
/// the registry has taken ownership of the provider itself — registration
/// is by `Box<dyn PlanBackend>`, so the concrete type is gone the moment
/// it is registered, which is itself part of the claim.
#[derive(Clone, Default)]
pub struct Tally(Arc<Counts>);

impl Tally {
    pub fn dispatches(&self) -> Dispatches {
        let c = &*self.0;
        Dispatches {
            embed: c.embed.load(Ordering::Relaxed),
            norm: c.norm.load(Ordering::Relaxed),
            project: c.project.load(Ordering::Relaxed),
            attention: c.attention.load(Ordering::Relaxed),
            attention_step: c.attention_step.load(Ordering::Relaxed),
            ffn: c.ffn.load(Ordering::Relaxed),
            routed_ffn: c.routed_ffn.load(Ordering::Relaxed),
            output_head: c.output_head.load(Ordering::Relaxed),
            residual_add: c.residual_add.load(Ordering::Relaxed),
            matrices: c.matrices.load(Ordering::Relaxed),
        }
    }
}

/// A lowering provider from outside the tree.
pub struct OutsideProvider {
    /// Which of the two external identities this instance states.
    family: &'static str,
    name: &'static str,
    /// The glue, borrowed through exported API.
    glue: ReferenceBackend,
    /// A deliberate defect in this provider's OWN kernel, for the control
    /// that the numbers are its: `0.0` is the honest provider.
    defect: f32,
    counts: Arc<Counts>,
}

impl Default for OutsideProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OutsideProvider {
    pub fn new() -> Self {
        Self::with_defect(0.0)
    }

    /// The equally capable second provider — same policy, same kernel,
    /// its own identity.
    pub fn sibling() -> Self {
        Self {
            family: SIBLING_FAMILY,
            name: SIBLING_NAME,
            ..Self::new()
        }
    }

    /// The same provider with a deliberate defect in its projection
    /// kernel — the instrument control: if the logits do not move, the
    /// numbers were never this provider's.
    pub fn with_defect(defect: f32) -> Self {
        Self {
            family: FAMILY,
            name: NAME,
            glue: ReferenceBackend::new(),
            defect,
            counts: Arc::new(Counts::default()),
        }
    }

    /// A handle on this provider's counters that outlives handing the
    /// provider to a registry.
    pub fn tally(&self) -> Tally {
        Tally(Arc::clone(&self.counts))
    }

    /// This provider's own kernel: one row-major `[out, in]` matrix
    /// against one vector, accumulated in order. Not borrowed — the
    /// arithmetic that makes the logits this provider's.
    fn project_rows(
        &self,
        weight: WeightSlice<'_>,
        out_dim: usize,
        in_dim: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        // One kernel, over f32 values. Anything else is refused by name
        // rather than silently widened — this provider does not pretend
        // to formats it has no code for.
        let w = weight.as_f32()?;
        self.counts.matrices.fetch_add(1, Ordering::Relaxed);
        let mut out = vec![0.0f32; out_dim];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (w, v) in w[row * in_dim..(row + 1) * in_dim].iter().zip(x) {
                acc += w * v;
            }
            *slot = acc + self.defect;
        }
        Ok(out)
    }
}

impl PlanBackend for OutsideProvider {
    fn name(&self) -> &str {
        self.name
    }

    fn identity(&self) -> LoweringIdentity {
        LoweringIdentity::new(self.family, REVISION)
    }

    /// **This provider's own selection.** Candidates are derived from
    /// what the registry DECLARES for the stored representation — never
    /// from its label — and the one it takes is the one it has a kernel
    /// for: the universal decode to f32, which is what its single scalar
    /// kernel reads. A direct realization the codec declares is
    /// considered and recorded as a candidate, and declined, because this
    /// provider has no kernel over stored bytes; nothing about that
    /// choice is core's to make.
    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        // The operations no backend chooses between — embedding gather,
        // bank slicing, the mapped bank — answered the one way the
        // contract defines them.
        if let Some(common) = common_selection(operand, facts, WeightFormat::F32) {
            return common;
        }
        let refuse = |kind, considered: Vec<(RealizationId, String)>| {
            Box::new(SelectionRefusal {
                operand: operand.operand.clone(),
                operation: operand.operation,
                representation: facts.label.clone(),
                requested: operand.access,
                kind,
                considered,
            })
        };
        if facts.registered.is_none() {
            return Err(refuse(RefusalKind::UnregisteredRepresentation, Vec::new()));
        }
        let decode = RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32));
        let mut candidates: Vec<RealizationId> = facts
            .direct_cpu_plans()
            .into_iter()
            .map(|plan| RealizationId::cpu(RealizationForm::Direct(plan)))
            .collect();
        let declined = candidates.len();
        candidates.push(decode);
        Ok(Selection {
            realization: decode,
            residency: realization_residency(facts, decode),
            reason: if declined > 0 {
                SelectionReason::ArmPrefersDecode
            } else {
                SelectionReason::NoDirectRealization
            },
            candidates,
        })
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.counts.embed.fetch_add(1, Ordering::Relaxed);
        self.glue.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.counts.norm.fetch_add(1, Ordering::Relaxed);
        self.glue.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.counts.project.fetch_add(1, Ordering::Relaxed);
        self.project_rows(call.weight, call.out_dim, call.in_dim, call.x)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        self.counts.attention.fetch_add(1, Ordering::Relaxed);
        self.glue.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        self.counts.attention_step.fetch_add(1, Ordering::Relaxed);
        self.glue.attention_step(call)
    }

    /// The dense feed-forward — three matrices, this provider's own
    /// kernel for each, and the plan's own semantics between them.
    ///
    /// The gate POLICY is the plan's meaning, not a lowering choice: this
    /// provider honours the plain gated policy and refuses anything else
    /// by name, exactly as the contract requires of every backend. It
    /// states that refusal itself because the shipped helper that states
    /// it (`require_executable_gate`) is crate-internal — an outside
    /// provider can read the policy and must re-derive what honouring it
    /// means, which is a real limit of the seam and is recorded as one.
    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.counts.ffn.fetch_add(1, Ordering::Relaxed);
        if !matches!(call.gate_policy, larql_models::ExpertGatePolicy::Gated) {
            return Err(VindexError::Parse(format!(
                "`{}` implements only the plain gated feed-forward policy; this plan declares \
                 {:?} and this provider will not approximate it",
                self.name, call.gate_policy
            )));
        }
        let up = self.project_rows(call.up, call.intermediate, call.hidden, call.x)?;
        let inner: Vec<f32> = match call.gate {
            Some(gate) => {
                let gate = self.project_rows(gate, call.intermediate, call.hidden, call.x)?;
                gate.iter()
                    .zip(&up)
                    .map(|(g, u)| activate(call.activation, *g) * u)
                    .collect()
            }
            None => up.iter().map(|u| activate(call.activation, *u)).collect(),
        };
        self.project_rows(call.down, call.hidden, call.intermediate, &inner)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.counts.routed_ffn.fetch_add(1, Ordering::Relaxed);
        self.glue.routed_ffn(call)
    }

    /// The vocabulary projection — this provider's own kernel, so the
    /// logits a caller reads are literally what it computed.
    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        self.counts.output_head.fetch_add(1, Ordering::Relaxed);
        let mut logits = self.project_rows(projection, vocab, hidden, x)?;
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = (*logit / cap).tanh() * cap;
            }
        }
        Ok(logits)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.counts.residual_add.fetch_add(1, Ordering::Relaxed);
        self.glue.residual_add(acc, delta);
    }
}
