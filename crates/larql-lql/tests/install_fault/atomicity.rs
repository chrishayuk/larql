//! INSERT atomicity (Gate B, B5): an INSERT that fails, at any point, leaves
//! the session exactly as it found it, on both backends.
//!
//! The install is forced to fail at every fault point, with and without an
//! earlier install at the layer. After the failure, everything an install can
//! write is compared with what it was before: every state fingerprint
//! component, the VINDEX2 override vectors as bits, the asserted facts, and
//! INFER on several prompts. Then the effective state is checked a second
//! way: the INSERT that failed, run again, writes exactly what it writes in a
//! session where it never failed.
//!
//! Before the fix, these tables were the defect: the decoy cache, the raw
//! residual, the slot and every refined slot at the layer survived the
//! failure, and after a fault just before recording, the overlay held an edit
//! the saved patch lacked. A refused INSERT also opened an automatic
//! recording.

use larql_lql::install_fault::InstallFaultPoint as P;
use larql_lql::Session;

use crate::support::*;

const POINTS: [P; 8] = [
    P::BeforeInstall,
    P::AfterCachesCommitted,
    P::AfterSlotStaged,
    P::AfterRefine,
    P::DuringBalance { iteration: 0 },
    P::DuringBalance { iteration: 1 },
    P::AfterBalance,
    P::BeforeRecord,
];

const V3_PROMPTS: [&str; 3] = ["The a of is", "The c of is", "a b c"];
const V2_PROMPTS: [&str; 3] = ["The [7] of [3] is", "The [7] of [4] is", "[1] [2] [3]"];

/// One backend's fixture, statements and prompts.
struct Case {
    name: &'static str,
    session: Box<dyn Fn() -> Session>,
    recording: std::path::PathBuf,
    insert: &'static str,
    insert_second: &'static str,
    prompts: [&'static str; 3],
    _keep: Box<dyn std::any::Any>,
}

fn v3() -> Case {
    let fx = std::rc::Rc::new(Fixture::new());
    let f = fx.clone();
    Case {
        name: "VINDEX3",
        session: Box::new(move || f.session()),
        recording: fx.path("recording.vlp"),
        insert: INSERT,
        insert_second: INSERT_SECOND,
        prompts: V3_PROMPTS,
        _keep: Box::new(fx),
    }
}

fn v2() -> Case {
    let fx = std::rc::Rc::new(V2Fixture::new());
    let f = fx.clone();
    Case {
        name: "VINDEX2",
        session: Box::new(move || f.session()),
        recording: fx.path("recording.vlp"),
        insert: V2_INSERT,
        insert_second: V2_INSERT_SECOND,
        prompts: V2_PROMPTS,
        _keep: Box::new(fx),
    }
}

/// The state and the predictions on every prompt.
fn observe(case: &Case, session: &mut Session) -> String {
    let predictions: Vec<Vec<String>> = case.prompts.iter().map(|p| infer(session, p)).collect();
    format!("{}\n--\n{predictions:?}", state(session))
}

/// A session in a named recording, with an earlier install if asked.
fn prepared(case: &Case, earlier_install: bool) -> Session {
    let mut session = (case.session)();
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", q(&case.recording)),
    );
    if earlier_install {
        run(&mut session, case.insert);
    }
    session
}

fn the_install(case: &Case, earlier_install: bool) -> &'static str {
    if earlier_install {
        case.insert_second
    } else {
        case.insert
    }
}

/// Every fault point, with and without an earlier install: the INSERT fails
/// with the injected fault, and the session is exactly as before.
fn assert_a_failed_install_changes_nothing(case: &Case) {
    for earlier in [false, true] {
        for point in POINTS {
            let mut session = prepared(case, earlier);
            let before = observe(case, &mut session);
            session.set_install_fault(Some(point));
            let error = try_run(&mut session, the_install(case, earlier)).unwrap_err();
            assert!(
                error.contains(&format!("injected fault at {point}")),
                "{}, fault at {point}: {error}",
                case.name
            );
            assert_eq!(
                observe(case, &mut session),
                before,
                "{}, earlier install: {earlier}, fault at {point}",
                case.name
            );
        }
    }
}

#[test]
fn vindex3_a_failed_install_changes_nothing() {
    assert_a_failed_install_changes_nothing(&v3());
}

#[test]
fn vindex2_a_failed_install_changes_nothing() {
    assert_a_failed_install_changes_nothing(&v2());
}

/// The effective state, checked by what the session does next: after a
/// failure at any point, the same INSERT run again writes exactly what it
/// writes in a session where it never failed, and the session ends equal to
/// that one.
fn assert_the_retried_install_is_what_it_would_have_been(case: &Case) {
    for earlier in [false, true] {
        let mut clean = prepared(case, earlier);
        run(&mut clean, the_install(case, earlier));
        let want = observe(case, &mut clean);
        for point in POINTS {
            let mut session = prepared(case, earlier);
            session.set_install_fault(Some(point));
            try_run(&mut session, the_install(case, earlier)).unwrap_err();
            run(&mut session, the_install(case, earlier));
            assert_eq!(
                observe(case, &mut session),
                want,
                "{}, earlier install: {earlier}, fault at {point}",
                case.name
            );
        }
    }
}

#[test]
fn vindex3_the_retried_install_is_what_it_would_have_been() {
    assert_the_retried_install_is_what_it_would_have_been(&v3());
}

#[test]
fn vindex2_the_retried_install_is_what_it_would_have_been() {
    assert_the_retried_install_is_what_it_would_have_been(&v2());
}

/// After a fault just before the patch operation is recorded, the overlay
/// holds no edit the recording lacks: SAVE PATCH writes a patch with no
/// operation, and the session it leaves is a session that never inserted.
#[test]
fn a_failed_install_leaves_no_edit_the_saved_patch_lacks() {
    for case in [v3(), v2()] {
        let mut session = prepared(&case, false);
        let before = observe(&case, &mut session);
        session.set_install_fault(Some(P::BeforeRecord));
        try_run(&mut session, case.insert).unwrap_err();
        assert_eq!(observe(&case, &mut session), before, "{}", case.name);
        run(&mut session, "SAVE PATCH;");
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&case.recording).unwrap()).unwrap();
        assert_eq!(
            saved["operations"].as_array().unwrap().len(),
            0,
            "{}: {saved}",
            case.name
        );
    }
}

/// A failed INSERT with no recording open does not leave one: SAVE PATCH
/// still says there is none. With an anonymous recording already open, the
/// failure keeps it and its operations as they were.
#[test]
fn a_failed_insert_opens_no_recording_and_keeps_an_open_one() {
    for case in [v3(), v2()] {
        let mut session = (case.session)();
        let before = observe(&case, &mut session);
        session.set_install_fault(Some(P::AfterRefine));
        try_run(&mut session, case.insert).unwrap_err();
        assert_eq!(observe(&case, &mut session), before, "{}", case.name);
        let save = try_run(&mut session, "SAVE PATCH;").unwrap_err();
        assert!(
            save.contains("no active patch session"),
            "{}: {save}",
            case.name
        );

        run(&mut session, case.insert);
        let before = observe(&case, &mut session);
        session.set_install_fault(Some(P::AfterRefine));
        try_run(&mut session, case.insert_second).unwrap_err();
        assert_eq!(observe(&case, &mut session), before, "{}", case.name);
        let save = try_run(&mut session, "SAVE PATCH;").unwrap_err();
        assert!(
            save.contains("anonymous patch session with 1 pending operation(s)"),
            "{}: {save}",
            case.name
        );
    }
}

/// The audit probe (`engine_lql::b5a_refused_insert_leaves_no_auto_patch`):
/// an INSERT refused for a real reason, with no fault armed, on a container
/// with no tokenizer. The session is unchanged and no recording is open.
#[test]
fn a_refused_insert_leaves_no_automatic_recording() {
    let checkpoint = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    larql_vindex::format::vindex3::fixtures::encode_fixture_container(
        larql_vindex::format::vindex3::fixtures::miniature_glimmer,
        checkpoint.path(),
        dir.path(),
        "tokless",
    );
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", q(dir.path())));
    let before = state(&mut session);
    try_run(&mut session, INSERT).expect_err("no tokenizer: the INSERT is refused");
    assert_eq!(state(&mut session), before);
    let save = try_run(&mut session, "SAVE PATCH;").unwrap_err();
    assert!(
        save.contains("no active patch session"),
        "a refused INSERT opened a recording: {save}"
    );
}
