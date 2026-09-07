//! The VINDEX3-bound expert route.
//!
//! Binds each layer's expert population from the mapped Q4_K bytes and
//! executes it through `larql-vindex`'s bound runtime, using the production
//! router and expert kernels. Substituting this for the in-process route is
//! what makes a whole-model comparison a statement about *composition*: every
//! other step — attention, KV, the dense slab, the norms, PLE — is the same
//! code, so a divergence is the MoE path or nothing.
//!
//! # Bound per call, deliberately
//!
//! Binding is not cached across tokens. The regions come from the mapped
//! index, so binding is pointer arithmetic and a shape check rather than a
//! read, and a cache here would be the resolution creep the bound object
//! exists to prevent — with the added hazard that a stale bound plan would
//! outlive the weights it borrows.
//!
//! # What this route does *not* do
//!
//! Fall back. If an operand is missing, a kernel unavailable or a shape wrong,
//! it returns the refusal. The in-process path guards several of these by
//! quietly substituting another kernel; doing that here would make a parity
//! result a statement about whichever route happened to run.

use larql_models::ModelWeights;
use ndarray::Array2;

use larql_vindex::format::capability::binding::{ComponentView, RepresentationIdentity};
use larql_vindex::format::capability::component::ComponentContract;
use larql_vindex::format::capability::coordinate::BankCoordinate;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::runtime::consts::{COL_DIM, FUSED_PROJECTION_HALVES};
use larql_vindex::runtime::{
    execute, BoundBankOperation, BoundExpert, BoundExpertScaling, BoundMoeOperation,
    BoundProjection, BoundReduction, BoundRouter, BoundTensor, ExecutionError, ExpertKernel,
    MoeInputs, RouterKernel,
};

use larql_compute::cpu::ops::moe::{moe_expert_input, moe_post_expert_output, moe_router_input};
use larql_compute::MoeLayerWeights;

use super::moe_backend::{MoeBackendError, MoeExpertBackend};

/// Which representation these operands claim to come from.
const VARIANT: &str = "vindex-mapped";
const ROUTER_REGION_SET: &str = "router";
const PER_EXPERT_SCALE_REGION_SET: &str = "router_per_expert_scale";
/// Gemma-shaped layers carry one routed bank. Multi-bank layers arrive with
/// the Mini-K3 rung and will index this rather than assume it.
const BANK_ID: u16 = 0;
const ROUTE_NAME: &str = "vindex3-bound";

/// Executes hybrid-MoE layers through a VINDEX3 bound route.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundMoeBackend {
    /// Which kernel runs the experts. Defaults to the production
    /// Q4_K × Q8_K kernel; the reference is available for differential runs.
    pub expert_kernel: ExpertKernel,
    /// Which kernel scores the router.
    pub router_kernel: RouterKernel,
}

impl BoundMoeBackend {
    /// The production pairing: both kernels bound to `larql-compute`'s own.
    pub fn production() -> Self {
        Self {
            expert_kernel: ExpertKernel::IncumbentQ4kQ8k,
            router_kernel: RouterKernel::Incumbent,
        }
    }

    /// Both kernels on the reference implementations — the oracle pairing.
    /// Requires directly-decodable operands, so it refuses Q4_K stores.
    pub fn reference() -> Self {
        Self {
            expert_kernel: ExpertKernel::Reference,
            router_kernel: RouterKernel::Reference,
        }
    }
}

impl MoeExpertBackend for BoundMoeBackend {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        let seq_len = h.nrows();
        let hidden = h.ncols();
        let arch = &*weights.arch;
        let Some(moe) = larql_compute::pipeline_layer::build_moe_weights(weights, arch, layer)
        else {
            // No expert weights to route into. Zeros, matching the in-process
            // path — a backend that errored here would change the model.
            return Ok(Array2::zeros((seq_len, hidden)));
        };
        Ok(self.run_layer(layer, &moe, h, norm_offset, eps)?)
    }

    fn name(&self) -> &'static str {
        ROUTE_NAME
    }
}

impl BoundMoeBackend {
    /// Bind one layer and run every position through it.
    ///
    /// Split out of [`MoeExpertBackend::forward_moe_seq`] so the execution is
    /// reachable from a `MoeLayerWeights` alone. The wrapper's only remaining
    /// job is to get one out of a `ModelWeights`, which needs a whole loaded
    /// model and is therefore the part a unit test cannot reach — so it is kept
    /// to the smallest possible body.
    pub fn run_layer(
        &self,
        layer: usize,
        moe: &MoeLayerWeights<'_>,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, ExecutionError> {
        let seq_len = h.nrows();
        let hidden = h.ncols();
        // Owned so the bound operands have something with a stable address to
        // borrow; the router is f32 in the index and the expert regions are
        // borrowed from the map directly.
        let router_bytes = f32_bytes(moe.router_proj);
        let scale_bytes = f32_bytes(moe.router_per_expert_scale);
        let operation = self.bind(layer, hidden, moe, &router_bytes, &scale_bytes)?;

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        for pos in 0..seq_len {
            let residual: Vec<f32> = h.row(pos).to_vec();
            // The two inputs are produced by the surrounding block's norms and
            // scales, exactly as the in-process path produces them. Deriving
            // them here would re-model the incumbent's routing policy with
            // somewhere for the two to disagree.
            let expert_input = moe_expert_input(&residual, moe, norm_offset, eps);
            let router_in = moe_router_input(&residual, &expert_input, moe, norm_offset, eps);

            let delta = execute(&operation, MoeInputs::split(&expert_input, &router_in))?;
            // The post-expert norm belongs to the block, not to the operation —
            // which is why `execute` returns a delta.
            let contribution = moe_post_expert_output(&delta, moe, norm_offset, eps);
            for (dst, src) in out.row_mut(pos).iter_mut().zip(contribution.iter()) {
                *dst = *src;
            }
        }
        Ok(out)
    }

    /// Bind one layer's whole expert population from the mapped bytes.
    fn bind<'a>(
        &self,
        layer: usize,
        hidden: usize,
        moe: &MoeLayerWeights<'a>,
        router_bytes: &'a [u8],
        scale_bytes: &'a [u8],
    ) -> Result<BoundMoeOperation<'a>, ExecutionError> {
        let inter = moe.intermediate_size;
        let inter_padded = moe.inter_padded();
        let format = expert_format(moe);

        let mut experts = Vec::with_capacity(moe.num_experts);
        for (expert_id, (&gate_up, &down)) in moe
            .experts_gate_up
            .iter()
            .zip(moe.experts_down.iter())
            .enumerate()
        {
            experts.push(BoundExpert {
                expert_id: expert_id as u32,
                projection: BoundProjection::Fused {
                    gate_up: BoundTensor::direct(
                        identity(&RegionRole::GateUpFused.name()),
                        gate_up,
                        format,
                        ComponentContract::matrix(
                            (FUSED_PROJECTION_HALVES * inter) as u32,
                            hidden as u32,
                        ),
                    )?,
                },
                // Stored at the padded intermediate width; the role sees the
                // live columns. One binding, and each kernel takes from it what
                // it needs.
                down: BoundTensor::new(
                    identity(&RegionRole::Down.name()),
                    down,
                    format,
                    ComponentContract::matrix(hidden as u32, inter_padded as u32),
                    ComponentView::Slice {
                        dim: COL_DIM,
                        start: 0,
                        len: inter as u32,
                    },
                )?,
            });
        }

        let operation = BoundMoeOperation {
            router: BoundRouter {
                weight: BoundTensor::direct(
                    identity(ROUTER_REGION_SET),
                    router_bytes,
                    RegionFormat::F32,
                    ComponentContract::matrix(moe.num_experts as u32, hidden as u32),
                )?,
                top_k: moe.top_k,
                selected_weight: moe.routing_policy.selected_weight,
                scaling: if scale_bytes.is_empty() {
                    BoundExpertScaling::None
                } else {
                    BoundExpertScaling::PerExpert {
                        scales: BoundTensor::direct(
                            identity(PER_EXPERT_SCALE_REGION_SET),
                            scale_bytes,
                            RegionFormat::F32,
                            ComponentContract::vector(moe.num_experts as u32),
                        )?,
                    }
                },
                kernel: self.router_kernel,
            },
            transforms: Vec::new(),
            banks: vec![BoundBankOperation {
                bank: BankCoordinate::new(layer as u32, BANK_ID),
                experts,
                intermediate_dim: inter,
                hidden_dim: hidden,
                // The bound path's expert kernels implement the gated
                // combine only. A ClampedGlu layer (GPT-OSS) or a SiTU-GLU
                // layer (Kimi-K3) must refuse at bind — with the rule
                // named — rather than run as SiLU and produce a
                // plausible-looking wrong forward. Both arms are spelled
                // out rather than collapsed into a wildcard, so a fourth
                // combine rule has to be considered here rather than
                // inheriting a refusal by default.
                activation: match moe.gate_rule {
                    larql_compute::MoeGateRule::Gated(a) => a,
                    larql_compute::MoeGateRule::ClampedGlu { .. }
                    | larql_compute::MoeGateRule::SituGlu { .. }
                    // GLM-5.3-Flash's clamped SwiGLU refuses here too: the
                    // clamp is real arithmetic, and running the unclamped
                    // gated kernel would silently drop it.
                    | larql_compute::MoeGateRule::ClampedGated { .. } => {
                        return Err(ExecutionError::UnsupportedFormat {
                            format: format!("{:?}", moe.gate_rule),
                            operand: "expert gate rule (bound MoE kernels are gated-only)"
                                .to_string(),
                        })
                    }
                },
                kernel: self.expert_kernel,
            }],
            reduction: BoundReduction::WeightedSum,
            residual_dim: hidden,
        };
        operation.validate()?;
        Ok(operation)
    }
}

fn identity(region_set: &str) -> RepresentationIdentity {
    RepresentationIdentity::new(region_set, VARIANT)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The region encoding this layer's experts are stored in.
///
/// Mapped from the loader's own `QuantFormat` rather than assumed, so a store
/// that is not Q4_K refuses at bind — with a message naming the encoding —
/// instead of being read as Q4_K super-blocks and producing noise.
fn expert_format(moe: &MoeLayerWeights<'_>) -> RegionFormat {
    match moe.expert_data_format {
        larql_compute::QuantFormat::Q4_K => RegionFormat::Q4K,
        larql_compute::QuantFormat::Q6_K => RegionFormat::Q6K,
        larql_compute::QuantFormat::Q4_0 => RegionFormat::Q4_0,
        larql_compute::QuantFormat::BF16 => RegionFormat::BF16,
        larql_compute::QuantFormat::F16 => RegionFormat::F16,
        larql_compute::QuantFormat::F32 => RegionFormat::F32,
        // Q4_KF, Q8_0 and I2S have no region-format equivalent this runtime
        // binds. Naming the tag keeps the refusal legible rather than
        // defaulting to Q4_K and reading the bytes at the wrong stride.
        other => RegionFormat::Unknown(unmapped_tag(other)),
    }
}

/// A distinct tag for an encoding with no `RegionFormat`, so the refusal says
/// which one rather than `format_0`.
fn unmapped_tag(format: larql_compute::QuantFormat) -> u16 {
    // Offset past every registered `RegionFormat` tag, so an unmapped codec can
    // never collide with a real one.
    const UNMAPPED_BASE: u16 = 1_000;
    UNMAPPED_BASE
        + match format {
            larql_compute::QuantFormat::Q4_KF => 0,
            larql_compute::QuantFormat::Q8_0 => 1,
            larql_compute::QuantFormat::I2S => 2,
            _ => 3,
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::cpu::ops::moe::cpu_moe_forward;
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    use larql_compute::{Activation, QuantFormat};

    /// One super-block wide, so Q8_K is satisfied and the fixture stays small.
    const HIDDEN: usize = 256;
    /// Half a block, so `down` is stored padded — the shape a binding gets wrong.
    const INTER: usize = 128;
    const INTER_PADDED: usize = 256;
    const POPULATION: usize = 2;
    const TOP_K: usize = 1;
    const LAYER: usize = 0;
    const EPS: f32 = 1e-6;
    const NORM_OFFSET: f32 = 0.0;

    fn ramp(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i + seed) % 19) as f32 * 0.01 - 0.09)
            .collect()
    }

    struct Fixture {
        gate_up: Vec<u8>,
        down: Vec<u8>,
        router: Vec<f32>,
        input: Vec<f32>,
    }

    impl Fixture {
        fn new() -> Self {
            let input: Vec<f32> = (0..HIDDEN).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
            // Expert 0's router row is the input, so its logit is ‖x‖² and the
            // selection is decided by the scores rather than by a tie.
            let mut router = vec![0.0f32; POPULATION * HIDDEN];
            router[..HIDDEN].copy_from_slice(&input);
            Self {
                gate_up: quantize_q4_k(&ramp(2 * INTER * HIDDEN, 1)),
                down: quantize_q4_k(&ramp(HIDDEN * INTER_PADDED, 7)),
                router,
                input,
            }
        }

        fn moe(&self) -> MoeLayerWeights<'_> {
            MoeLayerWeights {
                expert_scales: larql_compute::MoeExpertScales::Inline,
                fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
                experts_gate_up: vec![&self.gate_up, &self.gate_up],
                experts_down: vec![&self.down, &self.down],
                routing_policy: larql_compute::MoeRoutingPolicy::gemma4_hybrid(),
                weight_layout: larql_compute::MoeWeightLayout::default(),
                expert_data_format: QuantFormat::Q4_K,
                router_proj: &self.router,
                router_scale: &[],
                router_per_expert_scale: &[],
                router_norm: &[],
                router_norm_parameter_free: false,
                router_input_scalar: 1.0,
                pre_experts_norm: &[],
                post_ffn1_norm: &[],
                post_experts_norm: &[],
                num_experts: POPULATION,
                top_k: TOP_K,
                intermediate_size: INTER,
                router_bias: &[],
                experts_gate_up_bias: &[],
                experts_down_bias: &[],
                gate_rule: larql_compute::MoeGateRule::Gated(Activation::GeluTanh),
            }
        }

        fn h(&self) -> Array2<f32> {
            Array2::from_shape_vec((1, HIDDEN), self.input.clone()).expect("one row")
        }
    }

    #[test]
    fn the_bound_route_reproduces_the_in_process_route_bit_for_bit() {
        // The claim the whole seam rests on, at unit scale: substituting the
        // bound route must not change the layer's contribution.
        let fixture = Fixture::new();
        let moe = fixture.moe();
        let h = fixture.h();

        let bound = BoundMoeBackend::production()
            .run_layer(LAYER, &moe, &h, NORM_OFFSET, EPS)
            .expect("the fixture binds and executes");
        let incumbent = cpu_moe_forward(&fixture.input, &moe, NORM_OFFSET, EPS);

        assert_eq!(bound.nrows(), 1);
        assert_eq!(bound.row(0).to_vec(), incumbent);
    }

    #[test]
    fn the_bound_route_is_not_producing_zeros() {
        // Guards the guard: an all-zero agreement would be two failures
        // agreeing rather than a parity result.
        let fixture = Fixture::new();
        let out = BoundMoeBackend::production()
            .run_layer(LAYER, &fixture.moe(), &fixture.h(), NORM_OFFSET, EPS)
            .expect("executes");
        assert!(out.iter().any(|v| v.abs() > f32::EPSILON));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn every_position_is_routed_independently() {
        // A backend that routed once and broadcast would pass a single-row
        // test. Two different rows must produce two different contributions.
        let fixture = Fixture::new();
        let mut rows = fixture.input.clone();
        rows.extend(fixture.input.iter().map(|v| -v));
        let h = Array2::from_shape_vec((2, HIDDEN), rows).expect("two rows");
        let out = BoundMoeBackend::production()
            .run_layer(LAYER, &fixture.moe(), &h, NORM_OFFSET, EPS)
            .expect("executes");
        assert_ne!(out.row(0).to_vec(), out.row(1).to_vec());
    }

    #[test]
    fn the_reference_pairing_refuses_quantised_operands() {
        // The reference decoder implements the directly-readable encodings
        // only, so pairing it with a Q4_K store must refuse rather than
        // silently fall back to the kernel that can read them.
        let fixture = Fixture::new();
        let err = BoundMoeBackend::reference()
            .run_layer(LAYER, &fixture.moe(), &fixture.h(), NORM_OFFSET, EPS)
            .unwrap_err();
        assert!(
            matches!(err, ExecutionError::KernelOperandUnsuitable { .. })
                || matches!(err, ExecutionError::UnsupportedFormat { .. }),
            "{err}"
        );
    }

    #[test]
    fn a_layer_stored_in_an_unmappable_encoding_refuses_by_name() {
        // Q8_0 has no `RegionFormat`, so binding must refuse with a tag that
        // says which codec — not default to Q4_K and read at the wrong stride.
        let fixture = Fixture::new();
        let moe = MoeLayerWeights {
            expert_data_format: QuantFormat::Q8_0,
            ..fixture.moe()
        };
        let err = BoundMoeBackend::production()
            .run_layer(LAYER, &moe, &fixture.h(), NORM_OFFSET, EPS)
            .unwrap_err();
        assert!(
            matches!(err, ExecutionError::UnsupportedFormat { .. }),
            "{err}"
        );
    }

    #[test]
    fn each_registered_encoding_maps_to_its_region_format() {
        let fixture = Fixture::new();
        for (quant, expected) in [
            (QuantFormat::Q4_K, RegionFormat::Q4K),
            (QuantFormat::Q6_K, RegionFormat::Q6K),
            (QuantFormat::Q4_0, RegionFormat::Q4_0),
            (QuantFormat::BF16, RegionFormat::BF16),
            (QuantFormat::F16, RegionFormat::F16),
            (QuantFormat::F32, RegionFormat::F32),
        ] {
            let moe = MoeLayerWeights {
                expert_data_format: quant,
                ..fixture.moe()
            };
            assert_eq!(expert_format(&moe), expected, "{quant:?}");
        }
    }

    #[test]
    fn unmappable_encodings_get_distinct_tags_that_cannot_collide() {
        // A shared tag would make two different refusals read identically, and
        // a tag inside the registered range would name someone else's codec.
        let tags: Vec<u16> = [QuantFormat::Q4_KF, QuantFormat::Q8_0, QuantFormat::I2S]
            .into_iter()
            .map(unmapped_tag)
            .collect();
        let mut sorted = tags.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), tags.len(), "two codecs share a tag: {tags:?}");
        for tag in tags {
            assert!(
                matches!(RegionFormat::from_u16(tag), RegionFormat::Unknown(_)),
                "tag {tag} collides with a registered codec"
            );
        }
    }

    #[test]
    fn the_route_names_itself_and_defaults_to_the_reference_kernels() {
        assert_eq!(BoundMoeBackend::production().name(), ROUTE_NAME);
        let default = BoundMoeBackend::default();
        assert_eq!(default.expert_kernel, ExpertKernel::Reference);
        assert_eq!(default.router_kernel, RouterKernel::Reference);
        assert_eq!(
            BoundMoeBackend::production().expert_kernel,
            ExpertKernel::IncumbentQ4kQ8k
        );
    }

    // ── The wrapper, over a loaded model ───────────────────────────────────
    //
    // `run_layer` above is reachable from a `MoeLayerWeights` alone. These
    // cover the part that is not: resolving one out of a `ModelWeights`, which
    // needs the synthetic Gemma fixture. Its experts are BF16, which is the
    // useful case here — the reference kernel decodes them and the Q4_K kernel
    // must refuse them.

    use crate::test_utils::{make_test_gemma4_moe_weights, GEMMA4_MOE_HIDDEN};
    use larql_vindex::runtime::RefusalKind;

    const MOE_LAYER: usize = 0;
    /// Past the fixture's two layers, so `build_moe_weights` finds nothing.
    const ABSENT_LAYER: usize = 99;

    fn model_residual(rows: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, GEMMA4_MOE_HIDDEN), |(r, c)| {
            ((r * 7 + c) % 11) as f32 * 0.1 - 0.5
        })
    }

    #[test]
    fn the_bound_route_over_a_loaded_model_matches_the_in_process_route() {
        // End to end through the wrapper: resolve the layer, bind its whole
        // population, execute every position. The reference pairing, because
        // the fixture stores BF16 experts.
        let weights = make_test_gemma4_moe_weights();
        let h = model_residual(3);
        let bound = BoundMoeBackend::reference()
            .forward_moe_seq(&weights, MOE_LAYER, &h, NORM_OFFSET, EPS)
            .expect("the fixture binds and executes");
        let incumbent = super::super::moe_backend::InProcessMoeBackend
            .forward_moe_seq(&weights, MOE_LAYER, &h, NORM_OFFSET, EPS)
            .expect("executes");

        assert_eq!(bound.shape(), incumbent.shape());
        let worst = bound
            .iter()
            .zip(incumbent.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = incumbent.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // The reference sums in index order where the incumbent uses BLAS, so
        // this is a summation-order band rather than a claim of bit-equality.
        assert!(
            scale > 0.0 && worst / scale <= 1e-3,
            "diverged by {worst} of {scale}"
        );
    }

    #[test]
    fn the_production_pairing_refuses_a_bf16_store_by_format() {
        // It reads Q4_K super-blocks. Reading BF16 bytes as super-blocks is
        // exactly the silent substitution the format check exists to refuse.
        let weights = make_test_gemma4_moe_weights();
        let err = BoundMoeBackend::production()
            .forward_moe_seq(&weights, MOE_LAYER, &model_residual(1), NORM_OFFSET, EPS)
            .unwrap_err();
        let MoeBackendError::Bound(inner) = err else {
            panic!("a bound route must refuse as a bound route");
        };
        assert_eq!(inner.refusal(), RefusalKind::Unsupported, "{inner}");
    }

    #[test]
    fn a_layer_with_no_expert_weights_contributes_zeros() {
        // Matches the in-process path. A backend that errored here would
        // change the model rather than the route.
        let weights = make_test_gemma4_moe_weights();
        let h = model_residual(2);
        let out = BoundMoeBackend::production()
            .forward_moe_seq(&weights, ABSENT_LAYER, &h, NORM_OFFSET, EPS)
            .expect("an absent MoE layer is not an error");
        assert_eq!(out.shape(), h.shape());
        assert!(out.iter().all(|v| *v == 0.0));
    }
}
