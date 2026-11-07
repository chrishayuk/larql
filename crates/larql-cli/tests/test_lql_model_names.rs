//! `larql lql 'USE "<name>"'` through the real binary: a cached name binds,
//! an uncached one is refused with the fetch guidance and nothing is
//! downloaded. The caches are temp dirs named by `HF_HOME`/`LARQL_HOME`
//! on the child process only.

#[path = "support/continuation_fixture.rs"]
#[allow(dead_code)]
mod fixture;

use std::path::Path;
use std::process::{Command, Output};

/// The shorthand the fixture is linked under.
const NAME: &str = "glimmer-lql";
/// The `larql link` registry suffix and subdir under `LARQL_HOME`.
const LINK_SUFFIX: &str = ".vindex";
const LOCAL_SUBDIR: &str = "local";

fn lql(statement: &str, home: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_larql"))
        .env("HF_HOME", home.join("hf"))
        .env("LARQL_HOME", home.join("larql"))
        .env("LARQL_VERIFIER", "")
        .args(["lql", statement])
        .output()
        .expect("runs larql")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn a_cached_name_binds_by_shorthand() {
    let home = tempfile::tempdir().unwrap();
    let build = tempfile::tempdir().unwrap();
    let container = fixture::glimmer_container(build.path());
    let local = home.path().join("larql").join(LOCAL_SUBDIR);
    std::fs::create_dir_all(&local).unwrap();
    std::fs::rename(&container, local.join(format!("{NAME}{LINK_SUFFIX}"))).unwrap();

    let o = lql(&format!("USE \"{NAME}\"; STATS;"), home.path());
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    assert!(stdout(&o).contains("VINDEX3"), "{}", stdout(&o));
}

#[test]
fn an_uncached_name_is_refused_with_the_fetch_guidance() {
    let home = tempfile::tempdir().unwrap();
    let o = lql("USE \"larql-test-no-such-model\";", home.path());
    assert_eq!(o.status.code(), Some(1), "{}", stdout(&o));
    let out = stdout(&o);
    assert!(out.contains("never downloads"), "{out}");
    assert!(
        out.contains("larql run hf://<owner>/larql-test-no-such-model"),
        "{out}"
    );
    assert!(!home.path().join("hf").exists(), "nothing was fetched");
}
