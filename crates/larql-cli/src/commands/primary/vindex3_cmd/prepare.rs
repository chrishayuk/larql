//! The CLI's policy over the one VINDEX3 opener.
//!
//! `larql vindex3 exec` and `larql run <container>` execute the same
//! program through the same interpreter; they differ only in what wraps
//! it (token ids and a research report, or text and a stream). Neither
//! opens a container itself: `larql_inference`'s `open_component` is the
//! single authority on what a container *is* when it runs — inspect,
//! close the plan, bind the operands — and `larql serve` binds through
//! the same call. What this module owns is the CLI's side of that line:
//! which encoding a `--backend` asks for, whether the runtime may
//! manufacture it, and composing the lowering registry a `--backend`
//! names — in [`lowerings_for`], the one place this crate constructs a
//! provider — then resolving the provider by identity and handing it to
//! a [`BackendVisitor`]. Realisation is not interpretation, so it stays
//! here.

use std::path::Path;

use larql_inference::vindex3::{open_component, OpenPolicy, OpenedComponent};
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::lowering::{LoweringIdentity, LoweringRegistry};
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;

use super::ExecBackend;

type BoxErr = Box<dyn std::error::Error>;

/// The component a container executes when the caller names none — the
/// same default `larql vindex3 exec --component` declares.
pub(crate) const DEFAULT_COMPONENT: &str = "target";

/// Engine tag prefix; the backend name completes it so a dump or a
/// transcript can never be mistaken for one produced by another
/// realisation.
pub(crate) const ENGINE_PREFIX: &str = "vindex3";

/// Open `container`'s `component` for `backend`.
///
/// The CLI decides the policy — the encoding this backend executes and
/// whether it may be manufactured at load — and the runtime opener does
/// everything else, refusing a component that does not close with every
/// defect in the error.
pub(crate) fn prepare(
    container: &Path,
    component: &str,
    backend: ExecBackend,
    source: RepresentationSource,
) -> Result<OpenedComponent, BoxErr> {
    let policy = OpenPolicy {
        want: wanted_representation(backend).map(str::to_string),
        source,
    };
    Ok(open_component(container, component, policy)?)
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
    use larql_vindex::format::vindex3::represent::kquant;
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
        ExecBackend::ProductionQ8 => Some(kquant::Q8_0.name),
        ExecBackend::ProductionQ6k => Some(kquant::Q6_K.name),
        ExecBackend::ProductionQ4k => Some(kquant::Q4_K.name),
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
        // The K-quant arms are interpreted CPU arms; there is no lowered
        // GPU path for them, and this model has no Metal kernel at all.
        | ExecBackend::ProductionQ8
        | ExecBackend::ProductionQ6k
        | ExecBackend::ProductionQ4k
        | ExecBackend::Metal
        | ExecBackend::MetalMxfp4
        | ExecBackend::MetalMxfp4All
        | ExecBackend::MetalNvfp4
        | ExecBackend::MetalNvfp4NoHead
        | ExecBackend::MetalNvfp4Ffn => None,
    }
}

/// Work that runs against a provider resolved at runtime.
///
/// The interpreter is generic over the provider type; the visitor sees
/// the shared handle the registry hands out, which implements the trait
/// by delegation, so the choice is made once — in [`lowerings_for`] —
/// and the work is written once.
pub(crate) trait BackendVisitor {
    type Out;
    fn visit<B: PlanBackend>(self, backend: &B) -> Result<Self::Out, BoxErr>;
}

/// The lowering registry a `--backend` value composes, and the identity
/// it asks that registry for — **the single place a provider is
/// constructed in this crate** (LOWERING-PLUGIN-1, L3).
///
/// The shipped providers come from the registry's own constructor. A
/// device arm registers the device provider it configures — the injected
/// Metal device and the per-class format table that make that arm what it
/// is — under the device provider's identity, and asks for it by that
/// identity. Nothing downstream constructs a provider: every verb
/// resolves through the registry this returns.
pub(crate) fn lowerings_for(
    backend: ExecBackend,
) -> Result<(LoweringRegistry, LoweringIdentity), BoxErr> {
    let shipped = LoweringRegistry::shipped();
    match backend {
        ExecBackend::Reference => Ok((shipped, LoweringIdentity::reference())),
        // Same kernels for every production arm; the arms differ
        // upstream, in `wanted_representation`, which makes the store
        // bind a compiled NVFP4 or K-quant pack instead of the canonical
        // bytes. The projector then dispatches `FusedNvfp4` /
        // `FusedKQuant` off the resident representation — or, under
        // `LARQL_KQUANT_EXEC=widen`, a stored K-quant is decoded to f32
        // in `OperandStore::load` and run through BLAS as rung A's
        // authority path did. The executor reports which one ran in its
        // `projection plans:` line rather than leaving it to the
        // provider's name.
        ExecBackend::Production
        | ExecBackend::ProductionNvfp4
        | ExecBackend::ProductionQ8
        | ExecBackend::ProductionQ6k
        | ExecBackend::ProductionQ4k => Ok((shipped, LoweringIdentity::cpu_production())),
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
            let device =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu,
                    "metal-q1-mxfp4-ffn",
                    WeightFormats {
                        attention: WeightFormat::F16,
                        ffn: WeightFormat::Mxfp4,
                        head: WeightFormat::F16,
                    },
                );
            Ok((
                shipped.register(Box::new(device))?,
                LoweringIdentity::device_matmul(),
            ))
        }
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
            let device =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu,
                    "metal-q1-mxfp4-all",
                    WeightFormats::uniform(WeightFormat::Mxfp4),
                );
            Ok((
                shipped.register(Box::new(device))?,
                LoweringIdentity::device_matmul(),
            ))
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
            let device =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu, name, formats,
                );
            Ok((
                shipped.register(Box::new(device))?,
                LoweringIdentity::device_matmul(),
            ))
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
            let device =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::new(
                    gpu,
                    "metal-r3-f16",
                    larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat::F16,
                );
            Ok((
                shipped.register(Box::new(device))?,
                LoweringIdentity::device_matmul(),
            ))
        }
    }
}

/// Resolve the provider `backend` names and hand it to `visitor`:
/// compose the registry, then [`with_lowerings`]. The visitor sees a
/// shared handle to the registered provider, never a type this function
/// constructed for it.
pub(crate) fn with_plan_backend<V: BackendVisitor>(
    backend: ExecBackend,
    visitor: V,
) -> Result<V::Out, BoxErr> {
    let (lowerings, identity) = lowerings_for(backend)?;
    with_lowerings(&lowerings, &identity, visitor)
}

/// Hand `visitor` the provider `identity` names in `lowerings`, or
/// refuse by identity naming every provider the registry holds — before
/// the visitor runs, so before any operand is read. The seam a test hands
/// a registry of its own to: a registry without the provider must refuse
/// here, and nothing underneath may construct one to answer anyway.
pub(crate) fn with_lowerings<V: BackendVisitor>(
    lowerings: &LoweringRegistry,
    identity: &LoweringIdentity,
    visitor: V,
) -> Result<V::Out, BoxErr> {
    let provider = lowerings.provider_shared(identity)?;
    visitor.visit(&provider)
}
