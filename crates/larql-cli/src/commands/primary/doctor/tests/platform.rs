use super::super::platform::{checks, detect_cpu_features, CpuFeature, MetalProbe, PlatformFacts};
use super::super::report::Status;
use super::{healthy, statuses, GB};

fn platform() -> PlatformFacts {
    healthy().platform
}

fn check(p: &PlatformFacts, name: &str) -> super::Check {
    checks(p).into_iter().find(|c| c.name == name).unwrap()
}

#[test]
fn a_healthy_machine_passes_every_platform_check() {
    let p = platform();
    assert_eq!(statuses(&checks(&p)), [Status::Ok; 4]);
    assert_eq!(check(&p, "System").summary, "macos aarch64, 8 CPU threads");
    assert_eq!(
        check(&p, "CPU kernels").summary,
        "neon, dotprod (q4k_matvec=neon)"
    );
    assert_eq!(
        check(&p, "Memory").summary,
        "16.0 GB (quick start needs 4.0 GB)"
    );
    assert_eq!(check(&p, "Backend").summary, "metal (GPU)");
}

#[test]
fn an_unknown_thread_count_is_left_out() {
    let mut p = platform();
    p.cpu_threads = None;
    assert_eq!(check(&p, "System").summary, "macos aarch64");
}

#[test]
fn a_missing_cpu_feature_warns_and_names_it() {
    let mut p = platform();
    p.cpu_features[1].present = false;
    let c = check(&p, "CPU kernels");
    assert_eq!(c.status, Status::Warn);
    assert!(c.summary.starts_with("neon, no dotprod"), "{}", c.summary);
    assert!(c.hint.unwrap().contains("lacks dotprod"));
}

#[test]
fn an_architecture_without_simd_kernels_warns() {
    let mut p = platform();
    p.arch = "riscv64".into();
    p.cpu_features.clear();
    let c = check(&p, "CPU kernels");
    assert_eq!(c.status, Status::Warn);
    assert!(c.summary.contains("riscv64"));
}

#[test]
fn too_little_memory_fails_and_unknown_memory_warns() {
    let mut p = platform();
    p.ram_bytes = Some(2 * GB);
    assert_eq!(check(&p, "Memory").status, Status::Fail);
    p.ram_bytes = None;
    assert_eq!(check(&p, "Memory").status, Status::Warn);
}

#[test]
fn every_metal_probe_outcome_has_a_line() {
    let mut p = platform();
    p.metal = MetalProbe::NotApplicable;
    p.os = "linux".into();
    let c = check(&p, "Backend");
    assert_eq!(c.status, Status::Ok);
    assert!(c.summary.contains("linux"));

    p.metal = MetalProbe::NotCompiled;
    assert_eq!(check(&p, "Backend").status, Status::Warn);

    p.metal = MetalProbe::Unavailable("no device".into());
    let c = check(&p, "Backend");
    assert_eq!(c.status, Status::Warn);
    assert!(c.summary.contains("no device"));
}

#[test]
fn detected_features_match_this_architecture() {
    let names: Vec<&str> = detect_cpu_features().iter().map(|f| f.name).collect();
    if cfg!(target_arch = "aarch64") {
        assert_eq!(names, ["neon", "dotprod"]);
    } else if cfg!(target_arch = "x86_64") {
        assert_eq!(names, ["avx2"]);
    } else {
        assert!(names.is_empty());
    }
    let _: Vec<CpuFeature> = detect_cpu_features();
}
