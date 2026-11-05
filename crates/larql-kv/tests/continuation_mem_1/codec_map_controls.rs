//! CONTINUATION-CODEC-MAP-1: controls for the simulated partial codec.
//! SIM-0 on a fixture (the real-model SIM-0 is the run's first stage):
//! every row compressed ⇒ decode logits bit-identical to codec/v1, the same
//! retained range after every append, the same layer/range reads. Then the
//! map grammar and its selections.

use larql_kv::CodecKvState;

use super::codec_map_sim::{Age, Clause, CompressMap, Layers, MappedCodec, Recorder, DEPTH_GROUPS};
use super::*;

const BITS: u8 = 4;

/// A journey that crosses the fixture's sliding window, as CODEC-1's
/// residency control uses.
fn journey() -> Journey {
    Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9, 13],
        decode: vec![1, 2, 3, 4, 6, 7, 8],
    }
}

/// Decode logits, the recorder's trace, and `read` of the provider after
/// the journey, for one provider on the fixture.
fn traced<P, R>(inner: P, read: impl Fn(&P) -> R) -> (Vec<Vec<f32>>, (u64, u64), R)
where
    P: super::measured::Inspect + super::codec_map_sim::Retained,
{
    let subject = subjects::fixture(miniature_glimmer, "codec-map-sim0");
    let backend = ReferenceBackend::new();
    let ops = subject.prepare(&backend);
    let mut kv = Measured::new(Recorder::new(inner));
    let out = subjects::run(&subject, &ops, &backend, &mut kv, &journey());
    let rows = out
        .logits
        .into_iter()
        .filter(|(phase, _)| *phase == subjects::DECODE)
        .map(|(_, r)| r)
        .collect();
    (rows, kv.inner.trace(), read(&kv.inner.inner))
}

fn bits(rows: &[Vec<f32>]) -> Vec<u32> {
    rows.iter().flatten().map(|x| x.to_bits()).collect()
}

#[test]
fn sim0_all_compressed_is_behaviourally_codec_v1() {
    let _serial = serial();
    let (codec, codec_trace, _) = traced(CodecKvState::new(BITS), |_| ());
    let (sim, sim_trace, exact) = traced(
        MappedCodec::new(BITS, CompressMap::all()),
        MappedCodec::exact_fraction,
    );
    assert!(!codec.is_empty());
    assert_eq!(bits(&sim), bits(&codec), "logits bit-identical to codec/v1");
    assert_eq!(
        sim_trace, codec_trace,
        "same retention after every append, same reads"
    );
    assert!(sim_trace.1 > 0);
    assert_eq!(exact, 0.0);
}

#[test]
fn a_map_that_compresses_nothing_is_the_exact_path_and_differs_from_codec() {
    let _serial = serial();
    let none = CompressMap::single(
        "m",
        Clause {
            layers: Layers::Only(vec![]),
            ..Clause::all()
        },
    );
    let (exact, trace, fraction) =
        traced(MappedCodec::new(BITS, none), MappedCodec::exact_fraction);
    let (codec, codec_trace, _) = traced(CodecKvState::new(BITS), |_| ());
    let window = {
        let subject = subjects::fixture(miniature_glimmer, "codec-map-window");
        let backend = ReferenceBackend::new();
        let ops = subject.prepare(&backend);
        let mut kv = Measured::new(WindowKvState::new());
        subjects::run(&subject, &ops, &backend, &mut kv, &journey())
            .logits
            .into_iter()
            .filter(|(p, _)| *p == subjects::DECODE)
            .map(|(_, r)| r)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(&exact),
        bits(&window),
        "nothing compressed = window/v1, bit for bit"
    );
    assert_ne!(
        bits(&exact),
        bits(&codec),
        "the codec must be visible: SIM-0 is not vacuous"
    );
    assert_eq!(trace, codec_trace, "same geometry either way");
    assert_eq!(fraction, 1.0);
}

#[test]
fn the_recorder_trace_sees_a_changed_retention() {
    let _serial = serial();
    let (_, a, _) = traced(MappedCodec::new(BITS, CompressMap::all()), |_| ());
    let longer = Journey {
        decode: vec![1, 2, 3, 4, 6, 7, 8, 10],
        ..journey()
    };
    let subject = subjects::fixture(miniature_glimmer, "codec-map-longer");
    let backend = ReferenceBackend::new();
    let ops = subject.prepare(&backend);
    let mut kv = Measured::new(Recorder::new(MappedCodec::new(BITS, CompressMap::all())));
    subjects::run(&subject, &ops, &backend, &mut kv, &longer);
    assert_ne!(kv.inner.trace(), a, "one more step is a different trace");
}

#[test]
fn the_map_grammar_parses_and_refuses() {
    let m = CompressMap::parse("v_full=full:v:all").unwrap();
    assert_eq!(
        (
            m.name.as_str(),
            m.clauses.len(),
            &m.clauses[0].layers,
            m.clauses[0].k,
            m.clauses[0].v,
            m.clauses[0].age
        ),
        ("v_full", 1, &Layers::Full, false, true, Age::All)
    );
    assert_eq!(
        CompressMap::parse("x=q2:kv:older64").unwrap().clauses[0].layers,
        Layers::Depth(1)
    );
    assert_eq!(
        CompressMap::parse("x=all:k:older64").unwrap().clauses[0].age,
        Age::OlderThan(64)
    );
    assert_eq!(
        CompressMap::parse("x=all:k:newer256").unwrap().clauses[0].age,
        Age::NewerThan(256)
    );
    assert_eq!(
        CompressMap::parse("x=l3+l7:v:all").unwrap().clauses[0].layers,
        Layers::Only(vec![3, 7])
    );
    for bad in [
        "nameless",
        "x=all:v",
        "x=q0:v:all",
        "x=q5:v:all",
        "x=all:q:all",
        "x=all:v:recent9",
        "x=l3+z:v:all",
    ] {
        assert!(CompressMap::parse(bad).is_err(), "{bad} must be refused");
    }
}

#[test]
fn selections_partition_the_layers() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "codec-map-parts");
    let geometry = &subject.geometry;
    let kv: Vec<_> = geometry.iter().filter_map(|g| g.kv().cloned()).collect();
    let n = kv.len();
    let chosen = |layers: Layers| -> Vec<bool> {
        let mut m = MappedCodec::new(
            BITS,
            CompressMap::single(
                "m",
                Clause {
                    layers,
                    ..Clause::all()
                },
            ),
        );
        use larql_vindex::format::vindex3::opplan::exec::kv::ContinuationProvider;
        m.prepare(&kv);
        (0..n).map(|l| m.selected(l)).collect()
    };
    let (sliding, full) = (chosen(Layers::Sliding), chosen(Layers::Full));
    assert!(
        sliding.iter().zip(&full).all(|(a, b)| a != b),
        "sliding and full partition the layers"
    );
    let mut counts = vec![0; n];
    for q in 0..DEPTH_GROUPS {
        for (l, on) in chosen(Layers::Depth(q)).into_iter().enumerate() {
            counts[l] += on as usize;
        }
    }
    assert!(
        counts.iter().all(|&c| c == 1),
        "depth groups partition the layers"
    );
}

#[test]
fn age_rules_split_at_their_boundary_and_invert() {
    let _serial = serial();
    // Rows held at the end of the fixture journey, newest age 0; a map that
    // keeps the newest `w` exact and its inverse must together cover every
    // compressed row exactly once, so their exact fractions sum to 1.
    let w = 3;
    let older = CompressMap::single(
        "m",
        Clause {
            age: Age::OlderThan(w),
            ..Clause::all()
        },
    );
    let newer = CompressMap::single(
        "m",
        Clause {
            age: Age::NewerThan(w),
            ..Clause::all()
        },
    );
    let (_, _, keep_recent) = traced(MappedCodec::new(BITS, older), MappedCodec::exact_fraction);
    let (_, _, keep_old) = traced(MappedCodec::new(BITS, newer), MappedCodec::exact_fraction);
    assert!(
        keep_recent > 0.0 && keep_old > 0.0,
        "{keep_recent} {keep_old}"
    );
    assert!(
        (keep_recent + keep_old - 1.0).abs() < 1e-12,
        "{keep_recent} + {keep_old}"
    );
    assert!(
        keep_recent < keep_old,
        "the journey holds more than 2w rows per layer, so recent-w exact protects less"
    );
}

#[test]
fn a_union_parses_and_protection_is_the_complement() {
    let _serial = serial();
    let u = CompressMap::parse("p=all:v:all|l0:k:all").unwrap();
    assert_eq!(u.clauses.len(), 2);
    assert!(
        CompressMap::parse("p=all:v:all|").is_err(),
        "an empty clause is refused"
    );
    // Protecting nothing — every V, every K — is the full codec, bit for
    // bit and trace for trace (SIM-0 through the union path).
    let (codec, codec_trace, _) = traced(CodecKvState::new(BITS), |_| ());
    let union_all = CompressMap::parse("u=all:v:all|all:k:all").unwrap();
    let (sim, trace, exact) = traced(
        MappedCodec::new(BITS, union_all),
        MappedCodec::exact_fraction,
    );
    assert_eq!(bits(&sim), bits(&codec));
    assert_eq!(trace, codec_trace);
    assert_eq!(exact, 0.0);
    // Disjoint clauses add: V everywhere (half the bytes) plus K on layer 0
    // compresses exactly the sum of the two single maps.
    let exact_of = |spec: &str| {
        traced(
            MappedCodec::new(BITS, CompressMap::parse(spec).unwrap()),
            MappedCodec::exact_fraction,
        )
        .2
    };
    let (v, k0, both) = (
        exact_of("a=all:v:all"),
        exact_of("b=l0:k:all"),
        exact_of("c=all:v:all|l0:k:all"),
    );
    assert!(
        ((1.0 - both) - ((1.0 - v) + (1.0 - k0))).abs() < 1e-12,
        "{v} {k0} {both}"
    );
}
