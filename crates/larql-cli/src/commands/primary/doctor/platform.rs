//! The machine: OS, CPU features the compute kernels dispatch on, memory,
//! and whether the Metal backend finds a device.

use super::report::{Check, Group};
use super::requirements::QUICKSTART_RAM_BYTES;
use super::units::format_bytes;

/// One CPU feature a CPU kernel dispatches on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CpuFeature {
    pub name: &'static str,
    pub present: bool,
}

/// What the Metal probe found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MetalProbe {
    /// Not macOS: the CPU backend is the backend.
    NotApplicable,
    /// macOS, but this binary was built without Metal.
    NotCompiled,
    /// A Metal device, by backend name.
    Available(String),
    /// Metal compiled in, but the backend would not start.
    Unavailable(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlatformFacts {
    pub os: String,
    pub arch: String,
    pub cpu_threads: Option<usize>,
    pub cpu_features: Vec<CpuFeature>,
    /// The Q4_K/Q6_K kernel classes this CPU selects
    /// (`larql_compute`'s own summary).
    pub kernels: String,
    pub ram_bytes: Option<u64>,
    pub metal: MetalProbe,
}

/// The CPU features LARQL's kernels check at run time, as this CPU
/// reports them: dotprod on aarch64 (the NEON Q4_K/Q6_K kernels), AVX2 on
/// x86_64 (the Q4_K matvec). NEON itself is a compile-time baseline.
pub(crate) fn detect_cpu_features() -> Vec<CpuFeature> {
    #[cfg(target_arch = "aarch64")]
    {
        vec![
            CpuFeature {
                name: "neon",
                present: cfg!(target_feature = "neon"),
            },
            CpuFeature {
                name: "dotprod",
                present: std::arch::is_aarch64_feature_detected!("dotprod"),
            },
        ]
    }
    #[cfg(target_arch = "x86_64")]
    {
        vec![CpuFeature {
            name: "avx2",
            present: is_x86_feature_detected!("avx2"),
        }]
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        Vec::new()
    }
}

pub(crate) fn checks(p: &PlatformFacts) -> Vec<Check> {
    vec![
        system(p),
        cpu(p),
        memory(p.ram_bytes),
        backend(&p.metal, &p.os),
    ]
}

fn system(p: &PlatformFacts) -> Check {
    let threads = p
        .cpu_threads
        .map(|n| format!(", {n} CPU threads"))
        .unwrap_or_default();
    Check::ok(
        Group::Platform,
        "System",
        format!("{} {}{threads}", p.os, p.arch),
    )
}

fn cpu(p: &PlatformFacts) -> Check {
    const NAME: &str = "CPU kernels";
    if p.cpu_features.is_empty() {
        return Check::warn(
            Group::Platform,
            NAME,
            format!("no SIMD kernels for {} ({})", p.arch, p.kernels),
            "CPU decode uses the scalar kernels here; expect it to be slow",
        );
    }
    let listed: Vec<String> = p
        .cpu_features
        .iter()
        .map(|f| {
            if f.present {
                f.name.to_string()
            } else {
                format!("no {}", f.name)
            }
        })
        .collect();
    let summary = format!("{} ({})", listed.join(", "), p.kernels);
    let missing: Vec<&str> = p
        .cpu_features
        .iter()
        .filter(|f| !f.present)
        .map(|f| f.name)
        .collect();
    if missing.is_empty() {
        Check::ok(Group::Platform, NAME, summary)
    } else {
        Check::warn(
            Group::Platform,
            NAME,
            summary,
            format!(
                "this CPU lacks {}; LARQL falls back to slower kernels",
                missing.join(", ")
            ),
        )
    }
}

fn memory(ram: Option<u64>) -> Check {
    const NAME: &str = "Memory";
    let need = format_bytes(QUICKSTART_RAM_BYTES);
    match ram {
        None => Check::warn(
            Group::Platform,
            NAME,
            "could not read physical memory",
            format!("the quick start model needs about {need} of RAM"),
        ),
        Some(bytes) if bytes < QUICKSTART_RAM_BYTES => Check::fail(
            Group::Platform,
            NAME,
            format!("{} (quick start needs {need})", format_bytes(bytes)),
            "the quick start model will not fit; use a machine with more RAM",
        ),
        Some(bytes) => Check::ok(
            Group::Platform,
            NAME,
            format!("{} (quick start needs {need})", format_bytes(bytes)),
        ),
    }
}

fn backend(metal: &MetalProbe, os: &str) -> Check {
    const NAME: &str = "Backend";
    match metal {
        MetalProbe::NotApplicable => Check::ok(
            Group::Platform,
            NAME,
            format!("CPU (Metal is macOS-only; this is {os})"),
        ),
        MetalProbe::NotCompiled => Check::warn(
            Group::Platform,
            NAME,
            "CPU only: this build has no Metal backend",
            "install the macOS release archive, or build with default features for Metal",
        ),
        MetalProbe::Available(name) => Check::ok(Group::Platform, NAME, name.clone()),
        MetalProbe::Unavailable(why) => Check::warn(
            Group::Platform,
            NAME,
            format!("Metal is compiled in but unavailable: {why}"),
            "models run on the CPU backend instead, more slowly",
        ),
    }
}
