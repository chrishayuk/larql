use larql_lql::install_fault::InstallFaultPoint as P;

use crate::support::*;

#[test]
fn the_fingerprint_is_stable_and_sees_a_successful_install() {
    let fixture = Fixture::new();
    let mut session = fixture.session();
    let before = session.state_fingerprint();
    assert_eq!(
        before,
        session.state_fingerprint(),
        "nothing happened between two calls"
    );
    assert!(before.changed_since(&before).is_empty());

    run(&mut session, INSERT);
    let changed = session.state_fingerprint().changed_since(&before);
    for expected in [
        "overlay.counts",
        "overlay.vectors",
        "session.decoy_residual_cache",
        "session.installed_edges",
        "session.patch_recording",
        "session.raw_install_residuals",
    ] {
        assert!(
            changed.contains(&expected),
            "a successful install must change {expected}: {changed:?}"
        );
    }
}

#[test]
fn an_unarmed_or_unreached_fault_leaves_an_install_bit_identical() {
    let fixture = Fixture::new();
    let plain = install_and_save(
        &fixture,
        &mut fixture.session(),
        "plain.vlp",
        &[INSERT, INSERT_SECOND],
    );

    // Armed and then cleared.
    let mut cleared = fixture.session();
    cleared.set_install_fault(Some(P::AfterRefine));
    cleared.set_install_fault(None);
    assert_eq!(
        plain,
        install_and_save(
            &fixture,
            &mut cleared,
            "cleared.vlp",
            &[INSERT, INSERT_SECOND]
        )
    );

    // Armed at an iteration the balance pass never reaches (it runs at most 16).
    let mut unreached = fixture.session();
    unreached.set_install_fault(Some(P::DuringBalance { iteration: 99 }));
    assert_eq!(
        plain,
        install_and_save(
            &fixture,
            &mut unreached,
            "unreached.vlp",
            &[INSERT, INSERT_SECOND]
        )
    );
}

#[test]
fn a_fault_fires_once_and_the_session_installs_normally_afterwards() {
    let fixture = Fixture::new();
    let mut session = fixture.session();
    session.set_install_fault(Some(P::BeforeInstall));
    let first = try_run(&mut session, INSERT).unwrap_err();
    assert!(
        first.contains("injected fault at before-install"),
        "{first}"
    );
    // Disarmed by firing: the same statement now succeeds, and so does another.
    run(&mut session, INSERT);
    run(&mut session, INSERT_SECOND);
}

#[test]
fn every_point_reports_its_own_name() {
    let fixture = Fixture::new();
    for (point, name) in [
        (P::BeforeInstall, "before-install"),
        (P::AfterCachesCommitted, "after-caches-committed"),
        (P::AfterSlotStaged, "after-slot-staged"),
        (P::AfterRefine, "after-refine"),
        (P::DuringBalance { iteration: 0 }, "during-balance-0"),
        (P::DuringBalance { iteration: 1 }, "during-balance-1"),
        (P::AfterBalance, "after-balance"),
        (P::BeforeRecord, "before-record"),
    ] {
        assert_eq!(point.to_string(), name);
        let mut session = fixture.session();
        session.set_install_fault(Some(point));
        let err = try_run(&mut session, INSERT).unwrap_err();
        assert!(err.contains(&format!("injected fault at {name}")), "{err}");
    }
}

#[test]
fn the_fingerprint_names_every_part_of_the_state_it_covers() {
    let fixture = Fixture::new();
    let names: Vec<&str> = fixture
        .session()
        .state_fingerprint()
        .components()
        .keys()
        .copied()
        .collect();
    assert_eq!(
        names,
        [
            "overlay.counts",
            "overlay.patches",
            "overlay.vectors",
            "overlay.weight_ops",
            "session.counters",
            "session.decoy_residual_cache",
            "session.installed_edges",
            "session.patch_recording",
            "session.raw_install_residuals",
        ]
    );
}

#[test]
fn the_fingerprint_sees_an_applied_patch() {
    let fixture = Fixture::new();
    let mut author = fixture.session();
    install_and_save(&fixture, &mut author, "applied.vlp", &[INSERT]);

    let mut session = fixture.session();
    let before = session.state_fingerprint();
    run(
        &mut session,
        &format!("APPLY PATCH \"{}\";", q(&fixture.path("applied.vlp"))),
    );
    let changed = session.state_fingerprint().changed_since(&before);
    assert!(changed.contains(&"overlay.patches"), "{changed:?}");
    assert!(changed.contains(&"overlay.vectors"), "{changed:?}");
}

#[test]
fn the_fingerprint_sees_a_staged_weight_edit() {
    let dir = tempfile::tempdir().unwrap();
    let container =
        larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack(dir.path());
    let mut session = larql_lql::Session::new();
    run(&mut session, &format!("USE \"{}\";", q(&container)));
    let before = session.state_fingerprint();
    run(
        &mut session,
        r#"UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = "1.mlp.down_proj.weight" AND ROW = 5 AND COL IN (32, 33);"#,
    );
    let changed = session.state_fingerprint().changed_since(&before);
    assert!(changed.contains(&"overlay.weight_ops"), "{changed:?}");
    assert!(changed.contains(&"overlay.counts"), "{changed:?}");
}

#[test]
fn a_session_with_no_container_still_fingerprints() {
    let session = larql_lql::Session::new();
    let fingerprint = session.state_fingerprint();
    assert_eq!(
        fingerprint.components()["overlay.vectors"],
        "no-vindex3-overlay"
    );
    assert_eq!(fingerprint, session.state_fingerprint());
}
