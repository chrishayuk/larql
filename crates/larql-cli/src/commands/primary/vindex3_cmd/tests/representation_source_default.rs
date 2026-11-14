//! The `--representation-source` default: `stored` for the arms that name no
//! representation, `auto` for every arm that names a format.
//!
//! Under `auto`, `production` requantises every BF16 matrix whose BF16 bytes
//! exceed the host's L2 to Q8 at load — a lossy, machine-dependent change of
//! arithmetic nobody asked for (PREC-DIAG-1, Gemma 3 4B: 3.0 max |Δ logprob|
//! against transformers f32, where the stored bytes agree within 6e-4).

use clap::{Parser, ValueEnum};
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;

use super::super::prepare::{
    default_representation_source, representation_source_flag, resolve_representation_source,
    wanted_representation,
};
use super::super::{ExecArgs, ExecBackend};

#[derive(Parser)]
struct ExecCli {
    #[command(flatten)]
    exec: ExecArgs,
}

fn parse(extra: &[&str]) -> ExecArgs {
    let base = ["exec", "some.vindex3", "--tokens", "1,2,3"];
    ExecCli::try_parse_from(base.iter().chain(extra))
        .expect("valid arguments")
        .exec
}

#[test]
fn the_canonical_bytes_arms_default_to_stored() {
    for backend in [ExecBackend::Reference, ExecBackend::Production] {
        assert_eq!(
            default_representation_source(backend),
            RepresentationSource::Stored,
            "{backend:?}"
        );
    }
}

/// The opt-in is naming the format: an arm that asks for a representation
/// keeps `auto`, so it still quantises at load when no pack is compiled.
#[test]
fn every_arm_that_names_a_format_keeps_auto() {
    for &backend in ExecBackend::value_variants() {
        if matches!(backend, ExecBackend::Reference | ExecBackend::Production) {
            continue;
        }
        assert_eq!(
            default_representation_source(backend),
            RepresentationSource::Auto,
            "{backend:?}"
        );
    }
}

/// `stored` is the default exactly where the backend names no
/// representation — the only arms whose manufactured image is the size
/// policy's choice rather than the user's — among the CPU arms, where
/// that policy lives.
#[test]
fn stored_is_the_default_exactly_for_the_cpu_arms_that_want_no_representation() {
    for backend in [
        ExecBackend::Reference,
        ExecBackend::Production,
        ExecBackend::ProductionNvfp4,
        ExecBackend::ProductionQ8,
        ExecBackend::ProductionQ6k,
        ExecBackend::ProductionQ4k,
        ExecBackend::ProductionQ4kQ8k,
        ExecBackend::ProductionNvfp4Q8,
    ] {
        assert_eq!(
            default_representation_source(backend) == RepresentationSource::Stored,
            wanted_representation(backend).is_none(),
            "{backend:?}"
        );
    }
}

#[test]
fn an_explicit_source_wins_over_the_default() {
    assert_eq!(
        resolve_representation_source(Some("auto"), ExecBackend::Production).unwrap(),
        RepresentationSource::Auto
    );
    assert_eq!(
        resolve_representation_source(Some("transient"), ExecBackend::Reference).unwrap(),
        RepresentationSource::Transient
    );
    assert_eq!(
        resolve_representation_source(Some("stored"), ExecBackend::ProductionQ8).unwrap(),
        RepresentationSource::Stored
    );
    let err = resolve_representation_source(Some("cached"), ExecBackend::Production)
        .err()
        .unwrap();
    assert!(err.to_string().contains("cached"), "{err}");
}

#[test]
fn no_source_given_resolves_to_the_backend_default() {
    assert_eq!(
        resolve_representation_source(None, ExecBackend::Production).unwrap(),
        RepresentationSource::Stored
    );
    assert_eq!(
        resolve_representation_source(None, ExecBackend::ProductionNvfp4).unwrap(),
        RepresentationSource::Auto
    );
}

/// What a run record states is the resolved value, never an absent flag.
#[test]
fn exec_without_the_flag_runs_production_stored_and_says_so() {
    let args = parse(&["--backend", "production"]);
    assert_eq!(args.representation_source, None);
    assert_eq!(
        args.representation_source().unwrap(),
        RepresentationSource::Stored
    );
    assert_eq!(args.representation_source_flag().unwrap(), "stored");
}

#[test]
fn exec_keeps_auto_when_asked_for() {
    let args = parse(&["--backend", "production", "--representation-source", "auto"]);
    assert_eq!(
        args.representation_source().unwrap(),
        RepresentationSource::Auto
    );
    assert_eq!(args.representation_source_flag().unwrap(), "auto");
}

#[test]
fn exec_defaults_to_the_reference_arm_and_it_is_stored() {
    let args = parse(&[]);
    assert!(matches!(args.backend, ExecBackend::Reference));
    assert_eq!(args.representation_source_flag().unwrap(), "stored");
}

#[test]
fn the_flag_spelling_round_trips() {
    for source in [
        RepresentationSource::Auto,
        RepresentationSource::Stored,
        RepresentationSource::Transient,
    ] {
        assert_eq!(
            resolve_representation_source(
                Some(representation_source_flag(source)),
                ExecBackend::Production
            )
            .unwrap(),
            source
        );
    }
}

/// The logit dump's sidecar states the source the run resolved, how many
/// tensors were quantised at load, and which projection kernels ran — so a
/// plane says whether its matrices ran stored bytes or a manufactured image.
#[test]
fn the_logit_dump_records_the_resolved_source_and_what_ran() {
    use super::super::{run, Vindex3Command};
    use super::rope_angles::{encoded_fixture, exec_args, read_json};

    let (dir, container) = encoded_fixture();
    let mut planes = Vec::new();
    for (label, source) in [("default", None), ("auto", Some("auto"))] {
        let dump = dir.path().join(format!("production-{label}.f32"));
        let mut args = exec_args(container.clone(), ExecBackend::Production);
        args.representation_source = source.map(str::to_string);
        args.logit_dump = Some(dump.clone());
        run(Vindex3Command::Exec(args)).unwrap();
        let meta = read_json(&dump.with_extension("meta.json"));
        assert_eq!(
            meta["representation_source"],
            if source.is_none() { "stored" } else { "auto" }
        );
        assert_eq!(meta["runtime_quantised"], 0, "{label}: {meta}");
        let plans = meta["projection_plans"].as_object().expect("a map");
        assert!(!plans.is_empty(), "{label}: {meta}");
        planes.push(std::fs::read(&dump).unwrap());
    }
    // The fixture's matrices sit far below any L2, so the size policy keeps
    // them exact under either source: the two planes are the same bytes.
    assert_eq!(planes[0], planes[1]);
}
