//! Opening a container for execution — the one authority on how a
//! container gets prepared.
//!
//! `larql vindex3 exec` and `larql run <container>` execute the same
//! program through the same interpreter; they differ only in what wraps
//! it (token ids and a research report, or text and a stream). Both
//! therefore come through here: inspect the container, close the
//! component's plan, bind the operands the chosen backend wants, and
//! hand the backend — as a concrete type, chosen exactly once — to a
//! [`BackendVisitor`]. A second copy of this sequence would be a second
//! authority on what a container *is* when it runs, and the two would
//! drift.

use std::path::Path;

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::plan_component_ops;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

use super::ExecBackend;

type BoxErr = Box<dyn std::error::Error>;

/// The component a container executes when the caller names none — the
/// same default `larql vindex3 exec --component` declares.
pub(crate) const DEFAULT_COMPONENT: &str = "target";

/// Engine tag prefix; the backend name completes it so a dump or a
/// transcript can never be mistaken for one produced by another
/// realisation.
pub(crate) const ENGINE_PREFIX: &str = "vindex3";

/// A container opened for execution: the plan closed, the operands bound.
pub(crate) struct PreparedContainer {
    pub(crate) plan: ComponentOpPlan,
    pub(crate) store: OperandStore,
    /// The encoding a compiled pack would have to carry for the backend
    /// this was prepared for. `None` on arms that execute the canonical
    /// bytes directly, which then never look for a pack.
    pub(crate) want: Option<&'static str>,
}

/// Inspect, plan and bind — everything that has to happen before a
/// backend can be handed the program.
///
/// A component that does not close is refused here, with every defect
/// printed, so no caller can execute a partial plan.
pub(crate) fn prepare(
    container: &Path,
    component: &str,
    backend: ExecBackend,
    source: RepresentationSource,
) -> Result<PreparedContainer, BoxErr> {
    let inspection = inspect_container(container, false)?;
    let outcome = plan_component_ops(&inspection, container, component)?;
    if !outcome.defects.is_empty() {
        for defect in &outcome.defects {
            eprintln!("defect: {defect}");
        }
        return Err(format!(
            "component `{component}` does not close: {} defect(s)",
            outcome.defects.len()
        )
        .into());
    }
    let plan = outcome
        .plan
        .ok_or_else(|| format!("component `{component}` produced no plan"))?;
    let want = wanted_representation(backend);
    let store = OperandStore::open_for(container, &inspection, want, source)?;
    Ok(PreparedContainer { plan, store, want })
}

/// Parse `--representation-source`.
///
/// What execution wants, and whether it may be manufactured now, are
/// separate questions; this is the second one.
pub(crate) fn parse_representation_source(spec: &str) -> Result<RepresentationSource, BoxErr> {
    match spec {
        "auto" => Ok(RepresentationSource::Auto),
        "stored" => Ok(RepresentationSource::Stored),
        "transient" => Ok(RepresentationSource::Transient),
        other => Err(format!(
            "unknown --representation-source `{other}`; expected auto, stored or transient"
        )
        .into()),
    }
}

/// The stored encoding a backend could be served from, if one is compiled.
///
/// Only the NVFP4 arms have a compiled counterpart today. Arms that run
/// the canonical bytes return `None` and never look for a pack, so adding
/// packs to a container cannot change what they execute.
pub(crate) fn wanted_representation(backend: ExecBackend) -> Option<&'static str> {
    use larql_vindex::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
    // Exhaustive, with no wildcard arm, and that is the point: `_ =>
    // None` stood here and a newly added NVFP4 backend silently
    // inherited "wants nothing", bound the canonical bytes and produced
    // logits bit-identical to BF16. A run that looks like perfect
    // fidelity is the exact failure this file must not be able to
    // express, so a new backend is now a compile error until someone
    // states which representation it executes.
    match backend {
        ExecBackend::Reference | ExecBackend::Production => None,
        ExecBackend::ProductionNvfp4 => Some(DTYPE_NVFP4),
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::Metal
        | ExecBackend::MetalMxfp4
        | ExecBackend::MetalMxfp4All
        | ExecBackend::MetalLoweredMxfp4
        | ExecBackend::MetalLoweredMxfp4Ffn
        | ExecBackend::MetalLoweredF16 => None,
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalNvfp4
        | ExecBackend::MetalNvfp4Ffn
        | ExecBackend::MetalNvfp4NoHead
        | ExecBackend::MetalLowered
        | ExecBackend::MetalLoweredFfn
        | ExecBackend::MetalLoweredNoHead => Some(DTYPE_NVFP4),
    }
}

/// The lowered path's per-class policy, for the arms that take it.
///
/// Same scheduling for every arm — one command buffer per token — so a
/// comparison between them prices the *representation*, which the
/// pre-lowering numbers could not (they mixed kernel families and
/// starvation). `None` for every interpreted arm.
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub(super) fn lowered_formats(
    backend: ExecBackend,
) -> Option<(
    larql_vindex::format::vindex3::opplan::exec::backend::WeightFormats,
    &'static str,
)> {
    use larql_vindex::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats};
    match backend {
        ExecBackend::MetalLowered => {
            Some((WeightFormats::uniform(WeightFormat::Nvfp4), "nvfp4-all"))
        }
        ExecBackend::MetalLoweredFfn => Some((
            WeightFormats {
                attention: WeightFormat::F16,
                ffn: WeightFormat::Nvfp4,
                head: WeightFormat::F16,
            },
            "nvfp4-ffn",
        )),
        ExecBackend::MetalLoweredNoHead => Some((
            WeightFormats {
                attention: WeightFormat::Nvfp4,
                ffn: WeightFormat::Nvfp4,
                head: WeightFormat::F16,
            },
            "nvfp4-no-head",
        )),
        ExecBackend::MetalLoweredMxfp4 => {
            Some((WeightFormats::uniform(WeightFormat::Mxfp4), "mxfp4-all"))
        }
        ExecBackend::MetalLoweredF16 => {
            Some((WeightFormats::uniform(WeightFormat::F16), "f16-all"))
        }
        ExecBackend::MetalLoweredMxfp4Ffn => Some((
            WeightFormats {
                attention: WeightFormat::F16,
                ffn: WeightFormat::Mxfp4,
                head: WeightFormat::F16,
            },
            "mxfp4-ffn",
        )),
        ExecBackend::Reference
        | ExecBackend::Production
        | ExecBackend::ProductionNvfp4
        | ExecBackend::Metal
        | ExecBackend::MetalMxfp4
        | ExecBackend::MetalMxfp4All
        | ExecBackend::MetalNvfp4
        | ExecBackend::MetalNvfp4NoHead
        | ExecBackend::MetalNvfp4Ffn => None,
    }
}

/// Work that runs against a backend chosen at runtime.
///
/// The backends are distinct concrete types and the interpreter is
/// generic over them, so the choice cannot be a trait object; it is made
/// once, in [`with_plan_backend`], and the visitor sees the concrete
/// type.
pub(crate) trait BackendVisitor {
    type Out;
    fn visit<B: PlanBackend>(self, backend: &B) -> Result<Self::Out, BoxErr>;
}

/// Construct the interpreted backend `backend` names and hand it to
/// `visitor` — the single place a `--backend` value becomes a type.
///
/// The lowered arms are not interpreted backends and are refused here:
/// they run through `lowered::run_lowered`, which the exec verb reaches
/// via [`lowered_formats`] before it comes this way.
pub(crate) fn with_plan_backend<V: BackendVisitor>(
    backend: ExecBackend,
    visitor: V,
) -> Result<V::Out, BoxErr> {
    match backend {
        ExecBackend::Reference => visitor.visit(&ReferenceBackend::new()),
        ExecBackend::Production => visitor.visit(&ProductionBackend::new()),
        // Same kernels as `production`; the difference is upstream, in
        // `wanted_representation`, which makes the store bind the
        // compiled NVFP4 pack instead of the canonical bytes. The
        // projector then dispatches `FusedNvfp4` off the resident
        // representation, exactly as it dispatches every other arm.
        ExecBackend::ProductionNvfp4 => visitor.visit(&ProductionBackend::new()),
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalMxfp4 => {
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for --backend metal-mxfp4")?;
            // FFN-only MXFP4 — the gpt-oss precedent. The gates
            // falsified the wider presets on the 6-token fixture:
            // all-MXFP4 flipped the argmax (top-2 gap 0.08 vs
            // upstream's 1.13) and an f16 head alone did not recover
            // it (gap 0.01) — 4-bit attention projections accumulate
            // ~14% rel_rms across 52 layers. Attention and head stay
            // f16; the FFN bulk (~3/4 of the bytes) is quantised.
            use larql_vindex::format::vindex3::opplan::exec::backend::{
                WeightFormat, WeightFormats,
            };
            let backend =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu,
                    "metal-q1-mxfp4-ffn",
                    WeightFormats {
                        attention: WeightFormat::F16,
                        ffn: WeightFormat::Mxfp4,
                        head: WeightFormat::F16,
                    },
                );
            visitor.visit(&backend)
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalLowered
        | ExecBackend::MetalLoweredFfn
        | ExecBackend::MetalLoweredNoHead
        | ExecBackend::MetalLoweredMxfp4
        | ExecBackend::MetalLoweredMxfp4Ffn
        | ExecBackend::MetalLoweredF16 => Err(format!(
            "`{backend:?}` is a lowered arm: it does not execute through the interpreter"
        )
        .into()),
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalMxfp4All => {
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for --backend metal-mxfp4-all")?;
            // The control arm: the preset Q1 falsified. Its job is to
            // fail, so that a Q2 arm holding the prediction is evidence
            // about the format rather than about the harness.
            use larql_vindex::format::vindex3::opplan::exec::backend::{
                WeightFormat, WeightFormats,
            };
            let backend =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu,
                    "metal-q1-mxfp4-all",
                    WeightFormats::uniform(WeightFormat::Mxfp4),
                );
            visitor.visit(&backend)
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalNvfp4 | ExecBackend::MetalNvfp4Ffn | ExecBackend::MetalNvfp4NoHead => {
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for the nvfp4 backends")?;
            // The VINDEX3-Q2 ladder. Q1 established that *this model's*
            // attention does not survive MXFP4; NVFP4 keeps the same
            // e2m1 elements and changes only the scale geometry, which a
            // weight-reconstruction sweep with an equal-bit-budget
            // control (E8M0 at group 16) isolated as the whole source of
            // the difference. Arm A is the one that matters — it is the
            // ~17 GB regime — and B and C exist so a failure says which
            // class it came from rather than only that it failed.
            use larql_vindex::format::vindex3::opplan::exec::backend::{
                WeightFormat, WeightFormats,
            };
            let (name, formats) = match backend {
                // A — everything 4-bit.
                ExecBackend::MetalNvfp4 => (
                    "metal-q2-nvfp4-all",
                    WeightFormats::uniform(WeightFormat::Nvfp4),
                ),
                // B — attention and FFN 4-bit, head wide. Isolates the
                // head, which Q1's second rung showed was not the whole
                // story under MXFP4.
                ExecBackend::MetalNvfp4NoHead => (
                    "metal-q2-nvfp4-no-head",
                    WeightFormats {
                        attention: WeightFormat::Nvfp4,
                        ffn: WeightFormat::Nvfp4,
                        head: WeightFormat::F16,
                    },
                ),
                // C — the Q1-passing partition, re-run under NVFP4, so
                // the two formats are compared at the same class split.
                _ => (
                    "metal-q2-nvfp4-ffn",
                    WeightFormats {
                        attention: WeightFormat::F16,
                        ffn: WeightFormat::Nvfp4,
                        head: WeightFormat::F16,
                    },
                ),
            };
            let backend =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu, name, formats,
                );
            visitor.visit(&backend)
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::Metal => {
            // vindex never links Metal: the CLI injects the concrete
            // device through larql-compute's MatMul seam. f16 weights so
            // the Metal buffer cache keeps the model resident (r2); the
            // engine tag names the realisation so a dump can never be
            // mistaken for the f32 r1 lowering.
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for --backend metal")?;
            let backend =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::new(
                    gpu,
                    "metal-r3-f16",
                    larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat::F16,
                );
            visitor.visit(&backend)
        }
    }
}
