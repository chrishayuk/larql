use super::super::build_info::{checks, BuildFacts};
use super::only;

fn facts(metal: bool, research: bool) -> BuildFacts {
    BuildFacts {
        version: "0.2.1".into(),
        target: "x86_64-unknown-linux-gnu".into(),
        metal,
        research,
    }
}

#[test]
fn a_portable_release_build_says_so() {
    let check = only(checks(&facts(false, false)));
    assert_eq!(
        check.summary,
        "0.2.1 for x86_64-unknown-linux-gnu (portable CPU build)"
    );
}

#[test]
fn a_metal_research_build_says_so() {
    let check = only(checks(&facts(true, true)));
    assert!(check
        .summary
        .ends_with("(Metal + CPU build, research tools)"));
}

#[test]
fn the_current_build_reports_this_crate() {
    let current = BuildFacts::current();
    assert_eq!(current.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(current.target, env!("LARQL_TEST_TARGET"));
    assert_eq!(current.research, cfg!(feature = "research"));
    assert_eq!(
        current.metal,
        cfg!(all(feature = "gpu", target_os = "macos"))
    );
}
