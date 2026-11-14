//! `RemoteMoeFfn` — an [`FfnBackend`] adapter that lets the KvEngine layer
//! drive CPU remote-MoE decode with a real KV cache.
//!
//! The engine owns attention (and its KV cache) and calls
//! [`FfnBackend::forward_moe_full_layer`] per MoE layer; this adapter
//! computes that layer's MoE FFN block via
//! [`moe_ffn_block_cpu`](crate::vindex::kquant_forward::moe_ffn_block_cpu)
//! — dense `h1` locally + experts `h2` dispatched to the remote shards
//! through [`RemoteMoeBackend`]. This is the engine-routed counterpart to
//! the standalone full-recompute `generate_kquant_cpu_remote` path that
//! closed #146; see the larql-kv "MoE-aware KV engines (C1)" roadmap item.

use larql_compute::ffn::FfnBackend;
use larql_models::ModelWeights;
use ndarray::Array2;

use super::RemoteMoeBackend;
use crate::ffn::WeightFfn;
use crate::vindex::moe_ffn_block_cpu;

/// `FfnBackend` for CPU remote-MoE decode through a `KvEngine`.
///
/// The dense `h1` contribution runs through [`WeightFfn`] (f32 dense FFN over
/// `weights.tensors` — the caller pre-dequantizes the client's Q4K FFN), and
/// the expert `h2` contribution dispatches to the remote shards via
/// `forward_moe_seq`.
///
/// (A `WalkFfn`-based Q4K-direct `h1` was tried 2026-05-29 and reverted: its
/// dense mode runs the per-position sparse-walk machinery → ~8.5× slower than
/// f32 BLAS. The genuine Q4K-direct dense kernel is
/// `kquant_ffn_forward_layer_q8k`; see the bottleneck-diagnosis follow-up.)
///
/// PLE is **not** applied on this path (`moe_ffn_block_cpu` is called with
/// `ple_input = None`), so callers must route Per-Layer-Embedding
/// architectures (Gemma 4 E-series) through the full-recompute path
/// instead. Non-PLE MoE models (Gemma 4 26B-A4B, 31B-MoE) are unaffected.
pub struct RemoteMoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub remote: &'a RemoteMoeBackend,
}

impl<'a> FfnBackend for RemoteMoeFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.general().forward(layer, x)
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        self.general().forward_observed(layer, x)
    }

    fn name(&self) -> &str {
        REMOTE_MOE_FFN_NAME
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, larql_execution::BoxRefusal> {
        self.general().forward_moe_full_layer(layer, h_post_attn)
    }
}

impl<'a> RemoteMoeFfn<'a> {
    /// This adapter as the general one. Kept as a delegation rather than a type
    /// alias so the `{ weights, remote }` literal its callers use still
    /// type-checks — the name is load-bearing at one CLI call site and in two
    /// roadmaps.
    fn general(&self) -> MoeFfn<'a> {
        // Strict: a shard that fails (5xx, unreachable, timeout, malformed
        // reply) fails the layer, and the engine surfaces it to the caller of
        // generation. This adapter used to be best-effort — the failed
        // experts contributed zeros and decode carried on from a model that
        // had silently skipped them. A caller that wants that degradation must
        // now ask for it by name with `MoeFfn::best_effort`.
        MoeFfn::strict(self.weights, self.remote)
    }
}

/// The name `RemoteMoeFfn` reports. Unchanged from before the generalisation:
/// it appears in engine diagnostics and is matched on in at least one test.
const REMOTE_MOE_FFN_NAME: &str = "remote-moe";

/// `FfnBackend` for CPU MoE decode through a `KvEngine`, over **any** expert
/// route.
///
/// The engine owns attention and its KV cache and calls
/// [`FfnBackend::forward_moe_full_layer`] per MoE layer; this computes that
/// layer's block — dense `h1` locally, experts `h2` through whichever
/// [`MoeExpertBackend`] is installed.
///
/// The remote-specific adapter above predates this and now delegates to it. The
/// generalisation is what lets a VINDEX3 bound route reach the decode path at
/// all: the block loop's seam became a trait in the composition rung, but the
/// *engine's* adapter still named one concrete backend, so the bound route was
/// reachable only from the full-recompute path.
///
/// PLE is **not** applied here (`moe_ffn_block_cpu` is called with
/// `ple_input = None`), so Per-Layer-Embedding architectures must go through
/// the full-recompute path instead.
pub struct MoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub moe: &'a dyn crate::ffn::MoeExpertBackend,
    /// Whether a refusing route is a diagnosis or a fallback.
    policy: RefusalPolicy,
    /// The first refusal seen, with the layer that produced it.
    ///
    /// A cell rather than a return value because `FfnBackend` has no error
    /// channel — `forward_moe_full_layer` returns `Option<Array2<f32>>`, and
    /// widening it would reach every engine. The refusal is therefore caught on
    /// the way *in*, before `moe_ffn_block_cpu` handles it.
    refusal: std::cell::RefCell<Option<RecordedRefusal>>,
}

/// What a refusing expert route means to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefusalPolicy {
    /// Log it, contribute zeros, return the dense half. An explicit opt-in
    /// for analysis that wants a degraded continuation over none; no
    /// production generation path uses it.
    #[default]
    BestEffort,
    /// Record it, so the caller can refuse the output.
    ///
    /// A selected-but-absent expert must not silently become an FFN-skipping
    /// approximation. Under this policy the block still returns its dense half —
    /// the trait cannot say otherwise — but [`MoeFfn::refusal`] is set, and a
    /// caller that accepts logits without checking it is accepting a
    /// numerically wrong continuation that looks plausible.
    Strict,
}

/// A refusal, with where it happened.
///
/// An error in its own right, so it can cross `FfnBackend`'s boundary as a
/// `BoxRefusal` while keeping both levels: the response category an engine
/// switches on, and the concrete message the route produced.
#[derive(Debug, Clone)]
pub struct RecordedRefusal {
    pub layer: usize,
    /// The refusal's own classification, preserved rather than flattened —
    /// `Residency` means fetch the operand, and is not a defect.
    pub kind: larql_execution::RefusalKind,
    pub message: String,
}

impl std::fmt::Display for RecordedRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "layer {}: {} ({})", self.layer, self.message, self.kind)
    }
}

impl std::error::Error for RecordedRefusal {}

impl larql_execution::ExecutionRefusal for RecordedRefusal {
    fn kind(&self) -> larql_execution::RefusalKind {
        self.kind
    }
}

/// Wraps the route so a refusal is recorded before the block loop swallows it.
struct RefusalRecorder<'a> {
    inner: &'a dyn crate::ffn::MoeExpertBackend,
    sink: &'a std::cell::RefCell<Option<RecordedRefusal>>,
}

impl crate::ffn::MoeExpertBackend for RefusalRecorder<'_> {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, crate::ffn::MoeBackendError> {
        let out = self
            .inner
            .forward_moe_seq(weights, layer, h, norm_offset, eps);
        if let Err(err) = &out {
            let kind = match err {
                crate::ffn::MoeBackendError::Bound(inner) => inner.refusal(),
                // A remote dispatch failure is the operand not being here.
                crate::ffn::MoeBackendError::Remote(_) => larql_execution::RefusalKind::Residency,
                // So is a routed region missing from its container — same
                // diagnosis, different distance: the bytes this layer needs
                // are not reachable, rather than reachable and rejected.
                crate::ffn::MoeBackendError::Container(_) => {
                    larql_execution::RefusalKind::Residency
                }
                // A declared routing rule no policy computes: a valid
                // operation this route cannot execute.
                crate::ffn::MoeBackendError::UnsupportedRouting(_) => {
                    larql_execution::RefusalKind::Unsupported
                }
            };
            // First only: the earliest refusal is the diagnosis, and later ones
            // are usually the same cause repeating per layer.
            let mut sink = self.sink.borrow_mut();
            if sink.is_none() {
                *sink = Some(RecordedRefusal {
                    layer,
                    kind,
                    message: err.to_string(),
                });
            }
        }
        out
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

impl<'a> MoeFfn<'a> {
    /// Best-effort: a refusing route contributes zeros and the dense half is
    /// returned. Historical behaviour, unchanged.
    pub fn best_effort(
        weights: &'a ModelWeights,
        moe: &'a dyn crate::ffn::MoeExpertBackend,
    ) -> Self {
        Self {
            weights,
            moe,
            policy: RefusalPolicy::BestEffort,
            refusal: std::cell::RefCell::new(None),
        }
    }

    /// Strict: a refusal is recorded and the caller must check [`Self::refusal`]
    /// before accepting the output.
    ///
    /// Strictness is in the constructor rather than a field a caller can forget
    /// to set, and the check is a method rather than a log line someone has to
    /// read.
    pub fn strict(weights: &'a ModelWeights, moe: &'a dyn crate::ffn::MoeExpertBackend) -> Self {
        Self {
            weights,
            moe,
            policy: RefusalPolicy::Strict,
            refusal: std::cell::RefCell::new(None),
        }
    }

    pub fn policy(&self) -> RefusalPolicy {
        self.policy
    }

    /// The first refusal seen, if any. Always recorded; only *meaningful* as a
    /// gate under [`RefusalPolicy::Strict`], where the caller is contracted to
    /// consult it.
    pub fn refusal(&self) -> Option<RecordedRefusal> {
        self.refusal.borrow().clone()
    }

    /// Whether every layer so far executed its expert route.
    pub fn all_experts_executed(&self) -> bool {
        self.refusal.borrow().is_none()
    }
}

impl FfnBackend for MoeFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        WeightFfn {
            weights: self.weights,
        }
        .forward(layer, x)
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        WeightFfn {
            weights: self.weights,
        }
        .forward_observed(layer, x)
    }

    fn name(&self) -> &str {
        self.moe.name()
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, larql_execution::BoxRefusal> {
        let recorder = RefusalRecorder {
            inner: self.moe,
            sink: &self.refusal,
        };
        let out = moe_ffn_block_cpu(
            self.weights,
            h_post_attn,
            layer,
            &WeightFfn {
                weights: self.weights,
            },
            None,
            // Recording, because this adapter has its own `RefusalPolicy` one
            // level up: it needs the refusal *captured* so `Strict` can turn it
            // into an error and `BestEffort` can degrade having said so.
            // Making the block itself fatal here would bypass that decision.
            Some(crate::ffn::MoeRoute::recording(&recorder)),
        )
        // Unreachable: `recording` never propagates out of the block.
        .expect("RecordRefusal never returns Err");
        // `moe_ffn_block_cpu` has already logged the refusal and left the
        // expert contribution at zero, so `out` is the dense half wearing the
        // shape of an answer. Under `Strict` it must not escape — which is the
        // difference between this and the audited contract it replaces, where
        // the caller had to remember to ask.
        //
        // The dense half is computed and discarded on that path. It is an error
        // path, and paying for it buys the guarantee that no caller can consume
        // a partial layer.
        match self.refusal.borrow().as_ref() {
            Some(recorded) if self.policy == RefusalPolicy::Strict => {
                Err(Box::new(recorded.clone()))
            }
            // Best-effort keeps its historical behaviour: degrade, having said
            // so. Only an explicit `MoeFfn::best_effort` reaches this arm with
            // a refusal recorded.
            _ => Ok(Some(out)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_gemma4_moe_weights;
    use ndarray::Array2;

    /// A remote adapter with no shard to dispatch to fails the layer — the
    /// experts it cannot reach are an error, not a zero contribution.
    #[test]
    fn forward_moe_full_layer_refuses_when_no_shard_serves_the_experts() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let ffn = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        let h_post_attn = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let err = ffn
            .forward_moe_full_layer(0, &h_post_attn)
            .expect_err("an unserved expert must fail the layer");
        assert_eq!(err.kind(), larql_execution::RefusalKind::Residency);
        assert!(err.to_string().contains("no shard owns expert"), "{err}");
    }

    /// `forward` / `forward_with_activation` run the dense FFN fallback;
    /// `name` is stable.
    #[test]
    fn dense_fallbacks_and_name() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let ffn = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        assert_eq!(ffn.name(), "remote-moe");
        let x = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let dense = ffn.forward(0, &x);
        assert_eq!(dense.shape()[0], 2);
        assert!(dense.iter().all(|v| v.is_finite()));
        let (out, obs) = ffn.forward_observed(0, &x);
        assert_eq!(out.shape()[0], 2);
        let act = obs.into_dense().expect("dense fallback observes densely");
        assert_eq!(act.shape()[0], 2);
    }

    /// The remote adapter is the strict general one. Same weights, same
    /// input, same outcome — the delegation relocates the call and nothing
    /// else.
    #[test]
    fn the_remote_adapter_still_equals_the_general_one() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let specific = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        let general = MoeFfn::strict(&weights, &remote);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        assert_eq!(
            specific
                .forward_moe_full_layer(0, &h)
                .expect_err("no shard")
                .to_string(),
            general
                .forward_moe_full_layer(0, &h)
                .expect_err("no shard")
                .to_string()
        );
        assert_eq!(specific.forward(0, &h), general.forward(0, &h));
        // The name is the one thing that deliberately differs: the remote
        // adapter keeps its historical name for diagnostics, the general one
        // reports whichever route it carries.
        assert_eq!(specific.name(), REMOTE_MOE_FFN_NAME);
        assert_eq!(general.name(), crate::ffn::MoeExpertBackend::name(&remote));
    }

    /// The point of the generalisation: a VINDEX3 bound route can now drive the
    /// engine's FFN seam, which the remote-typed adapter made impossible.
    #[test]
    fn a_bound_route_can_drive_the_engine_adapter() {
        let weights = make_test_gemma4_moe_weights();
        // The fixture stores BF16 experts, so the reference pairing is the one
        // that executes; the production Q4_K pairing would refuse them.
        let bound = crate::ffn::BoundMoeBackend::reference();
        let ffn = MoeFfn::best_effort(&weights, &bound);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out = ffn
            .forward_moe_full_layer(0, &h)
            .expect("executes")
            .expect("the adapter always returns Some");
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|v| v.abs() > f32::EPSILON));
    }

    /// A refused route must not silently contribute zeros through the adapter.
    /// `moe_ffn_block_cpu` logs and leaves `h2` zero on error, so this pins that
    /// the dense half still flows — the failure is visible as a *different*
    /// output rather than as a plausible one.
    #[test]
    fn a_refusing_route_still_returns_the_dense_half() {
        let weights = make_test_gemma4_moe_weights();
        // Q4_K kernels against a BF16 store: refused at bind.
        let refusing = crate::ffn::BoundMoeBackend::production();
        let executing = crate::ffn::BoundMoeBackend::reference();
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let refused = MoeFfn::best_effort(&weights, &refusing)
            .forward_moe_full_layer(0, &h)
            .expect("executes")
            .expect("returns Some even when the route refused");
        let executed = MoeFfn::best_effort(&weights, &executing)
            .forward_moe_full_layer(0, &h)
            .expect("executes")
            .expect("produces a layer");
        assert_ne!(
            refused, executed,
            "a refused expert route must not produce the same output as one that ran"
        );
    }

    // ── Strictness ─────────────────────────────────────────────────────────

    #[test]
    fn a_strict_adapter_records_a_refusal_that_best_effort_would_swallow() {
        // The behaviour that must not become the MAP-3A contract. Best-effort
        // logs, contributes zeros and returns the dense half — a plausible but
        // numerically wrong continuation. Strict records it so the caller can
        // refuse the output.
        let weights = make_test_gemma4_moe_weights();
        // Q4_K kernels against the fixture's BF16 store: refused at bind.
        let refusing = crate::ffn::BoundMoeBackend::production();
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);

        let lenient = MoeFfn::best_effort(&weights, &refusing);
        let _ = lenient.forward_moe_full_layer(0, &h);

        let strict = MoeFfn::strict(&weights, &refusing);
        let _ = strict.forward_moe_full_layer(0, &h);

        assert_eq!(strict.policy(), RefusalPolicy::Strict);
        assert_eq!(lenient.policy(), RefusalPolicy::BestEffort);
        let recorded = strict.refusal().expect("a refusal must be recorded");
        assert_eq!(recorded.layer, 0);
        assert_eq!(
            recorded.kind,
            larql_vindex::runtime::RefusalKind::Unsupported
        );
        assert!(!strict.all_experts_executed());
    }

    #[test]
    fn an_executing_route_records_nothing() {
        // The counter-case. If a refusal were recorded for a route that ran,
        // the gate would reject every output and be quietly useless.
        let weights = make_test_gemma4_moe_weights();
        let executing = crate::ffn::BoundMoeBackend::reference();
        let strict = MoeFfn::strict(&weights, &executing);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let _ = strict.forward_moe_full_layer(0, &h);
        assert!(strict.refusal().is_none());
        assert!(strict.all_experts_executed());
    }

    #[test]
    fn the_refusal_keeps_its_classification_rather_than_a_flattened_message() {
        // `Residency` means fetch the operand and is not a defect; `Unsupported`
        // means write a kernel. A gate that only had a string could not tell a
        // sharded deployment from a broken one.
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let strict = MoeFfn::strict(&weights, &remote);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let _ = strict.forward_moe_full_layer(0, &h);
        let recorded = strict.refusal().expect("an unserved expert is refused");
        assert_eq!(
            recorded.kind,
            larql_execution::RefusalKind::Residency,
            "an unreachable shard is the operand not being here"
        );
        assert!(!recorded.message.is_empty());
    }

    #[test]
    fn only_the_first_refusal_is_kept() {
        // Later layers usually repeat the same cause, so the earliest one is
        // the diagnosis. A sink that overwrote would report the last layer
        // rather than the first — the same first-versus-causal confusion the
        // divergence report exists to avoid.
        let weights = make_test_gemma4_moe_weights();
        let refusing = crate::ffn::BoundMoeBackend::production();
        let strict = MoeFfn::strict(&weights, &refusing);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let _ = strict.forward_moe_full_layer(0, &h);
        let _ = strict.forward_moe_full_layer(1, &h);
        assert_eq!(strict.refusal().expect("recorded").layer, 0);
    }

    #[test]
    fn the_remote_adapter_is_strict() {
        // A shard failure must stop generation rather than finish it from a
        // model missing that shard's experts.
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let specific = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        assert_eq!(specific.general().policy(), RefusalPolicy::Strict);
    }

    #[test]
    fn best_effort_remains_an_explicit_opt_in() {
        // The degradation is still reachable, but only by name.
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let lenient = MoeFfn::best_effort(&weights, &remote);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out = lenient
            .forward_moe_full_layer(0, &h)
            .expect("best-effort degrades")
            .expect("returns the dense half");
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(lenient.refusal().is_some(), "but records that it did");
    }
}
