//! DESCRIBE on MoE-router fixture (try_moe_describe path)

use super::*;

#[test]
fn describe_on_moe_fixture_loads_router() {
    // USE on the MoE fixture should populate Backend::Vindex.router via
    // `RouterIndex::load`. DESCRIBE then routes through `try_moe_describe`
    // and reports per-expert hit counts.
    let (mut session, dir) = moe_vindex_session("describe_moe");

    // Sanity: router file is on disk and the live backend's router
    // got constructed (otherwise try_moe_describe short-circuits to None).
    assert!(
        dir.join("router_weights.bin").exists(),
        "router_weights.bin should exist at {}",
        dir.display()
    );
    if let Backend::Vindex { router, .. } = &session.backend {
        assert!(
            router.is_some(),
            "RouterIndex::load should populate Backend::Vindex.router for the MoE fixture",
        );
    } else {
        panic!("expected Backend::Vindex");
    }

    let stmt = parser::parse(r#"DESCRIBE "[1]";"#).unwrap();
    let out = session.execute(&stmt).expect("DESCRIBE on MoE fixture");
    let joined = out.join("\n");
    // Expected output format from describe/moe.rs:
    //   [1]
    //     Experts (L0-1):
    //       E<id>  <count>/<layers> layers  (<pct>% avg)
    //   ...
    assert!(
        joined.contains("Experts"),
        "expected 'Experts' header in MoE DESCRIBE, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_verbose_on_moe_fixture_shows_routing() {
    // VERBOSE branch prints the per-layer routing table.
    let (mut session, dir) = moe_vindex_session("describe_moe_verbose");

    let stmt = parser::parse(r#"DESCRIBE "[1]" VERBOSE;"#).unwrap();
    let out = session
        .execute(&stmt)
        .expect("DESCRIBE VERBOSE on MoE fixture");
    let joined = out.join("\n");
    assert!(
        joined.contains("Routing (L"),
        "expected verbose 'Routing' header, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_on_moe_fixture_unknown_entity_reports_not_found() {
    // The "(not found)" branch fires when the entity doesn't tokenise
    // to anything in vocab. Use a literal that the WordLevel tokenizer
    // can't resolve to avoid the embedding lookup short-circuit.
    let (mut session, dir) = moe_vindex_session("describe_moe_unknown");

    let stmt = parser::parse(r#"DESCRIBE "totally_unknown_entity_that_wont_tokenize";"#).unwrap();
    let out = session.execute(&stmt).expect("DESCRIBE on unknown entity");
    let joined = out.join("\n");
    // Either "(not found)" or the entity name with empty experts is
    // acceptable — the test is that we don't panic on unknown input.
    assert!(
        joined.contains("not found") || joined.contains("Experts"),
        "expected sensible MoE DESCRIBE output, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn browse_on_a_weight_backend_points_at_extraction() {
    let session = weight_session();
    let err = match session.browse() {
        Ok(_) => panic!("weight backend must not browse"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("requires a vindex"), "{err}");
    assert!(err.contains("EXTRACT"), "{err}");
}

#[test]
fn browse_without_any_backend_is_no_backend() {
    let session = Session::new();
    let err = match session.browse() {
        Ok(_) => panic!("empty session must not browse"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("No backend"), "{err}");
}
