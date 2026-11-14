//! What an LQL session actually executes a REAL Gemma 3 4B container from, under each representation source:
//! the tensors the loader quantised, by name. Instrument evidence about the execution path, on one non-cohort
//! prompt; no behavioural claim. Ignored by default:
//!
//! ```sh
//! LARQL_V3_CONTAINER=~/chris-models/gemma3-4b-it.vindex3 \
//!   cargo test --release -p larql-lql --test vindex3_representation_source_real -- --ignored --nocapture
//! ```

use larql_lql::{parse, RepresentationSource, Session};

const CONTAINER_VAR: &str = "LARQL_V3_CONTAINER";
/// The matrix CONSUMPTION-BF16-1 qualified: 52 MB as BF16, larger than any performance-core L2.
const LARGE_MATRIX: &str = "12.mlp.down_proj.weight";
const PROMPT: &str = "The capital of France is";

fn infer_under(container: &str, source: RepresentationSource) -> (Vec<String>, Vec<String>) {
    let mut session = Session::new();
    session.set_representation_source(source);
    let mut run = |text: &str| session.execute(&parse(text).unwrap()).unwrap();
    run(&format!("USE \"{container}\";"));
    let reply = run(&format!("INFER \"{PROMPT}\" TOP 3;"));
    (reply, session.runtime_quantised_tensors().unwrap())
}

#[test]
#[ignore = "needs a real Gemma 3 4B container in LARQL_V3_CONTAINER"]
fn real_infer_executes_stored_weights_by_default_and_a_manufactured_residency_only_under_auto() {
    let container = std::env::var(CONTAINER_VAR)
        .unwrap_or_else(|_| panic!("set {CONTAINER_VAR} to a real container"));

    let (stored_reply, stored_quantised) = infer_under(&container, RepresentationSource::Stored);
    println!(
        "stored: {} tensor(s) quantised at load",
        stored_quantised.len()
    );
    println!(
        "stored: {}",
        stored_reply[1..stored_reply.len() - 1].join(" | ")
    );
    assert!(
        stored_quantised.is_empty(),
        "under stored nothing may be manufactured: {stored_quantised:?}"
    );

    let (auto_reply, auto_quantised) = infer_under(&container, RepresentationSource::Auto);
    println!(
        "auto:   {} tensor(s) quantised at load",
        auto_quantised.len()
    );
    println!(
        "auto:   {}",
        auto_reply[1..auto_reply.len() - 1].join(" | ")
    );
    assert!(
        auto_quantised.iter().any(|t| t == LARGE_MATRIX),
        "under auto the loader quantises {LARGE_MATRIX} on a host whose L2 it does not fit: \
         {} quantised, first {:?}",
        auto_quantised.len(),
        auto_quantised.iter().take(5).collect::<Vec<_>>()
    );

    // The default is the first of the two: a session that sets nothing binds under stored.
    assert_eq!(
        Session::new().representation_source(),
        RepresentationSource::Stored
    );
}
