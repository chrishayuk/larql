use std::path::Path;
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

const A: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#;
const C: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#;

fn tok() -> String {
    let vocab = r#""[UNK]":0,"The":1,"of":2,"is":3,"a":4,"b":5,"c":6,"[5]":7,"[6]":8"#;
    format!("{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},\"post_processor\":null,\"decoder\":null,\"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}")
}
struct Fx { _c: tempfile::TempDir, dir: tempfile::TempDir }
fn fx() -> Fx {
    let c = tempfile::tempdir().unwrap(); let dir = tempfile::tempdir().unwrap();
    encode_fixture_container(miniature_glimmer, c.path(), dir.path(), "trace-fixture");
    std::fs::write(dir.path().join("tokenizer.json"), tok()).unwrap();
    Fx { _c: c, dir }
}
fn q(p: &Path) -> String { p.display().to_string() }
fn run(s: &mut Session, stmt: &str) -> Vec<String> {
    s.execute(&parse(stmt).unwrap()).unwrap_or_else(|e| panic!("{stmt}: {e}"))
}
fn open(f: &Fx) -> Session { let mut s = Session::new(); run(&mut s, &format!("USE \"{}\";", q(f.dir.path()))); s }
fn vectors(s: &Session) -> String { s.state_fingerprint().components()["overlay.vectors"].clone() }
fn infer(s: &mut Session, p: &str) -> Vec<String> {
    run(s, &format!("INFER \"{p}\" TOP 5;")).into_iter().filter(|l| !l.trim_end().ends_with("ms")).collect()
}

#[test]
fn b3_two_compose_inserts_same_layer_replay_exactly() {
    let f = fx();
    let patch = f.dir.path().join("two.vlp");
    let mut s = open(&f);
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&patch)));
    run(&mut s, A); run(&mut s, C);
    run(&mut s, "SAVE PATCH;");
    let live = vectors(&s);
    let live_inf = infer(&mut s, "The b of a is");
    let mut fresh = open(&f);
    run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&patch)));
    let replay = vectors(&fresh);
    let replay_inf = infer(&mut fresh, "The b of a is");
    eprintln!("live infer {live_inf:?}\nreplay infer {replay_inf:?}");
    assert_eq!(live, replay, "overlay vectors after fresh APPLY differ from the session that saved the patch");
}

#[test]
fn b3_split_across_two_patches_replay_exactly() {
    let f = fx();
    let p1 = f.dir.path().join("p1.vlp"); let p2 = f.dir.path().join("p2.vlp");
    let mut s = open(&f);
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&p1))); run(&mut s, A); run(&mut s, "SAVE PATCH;");
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&p2))); run(&mut s, C); run(&mut s, "SAVE PATCH;");
    let live = vectors(&s);
    let mut fresh = open(&f);
    run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&p1)));
    run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&p2)));
    assert_eq!(live, vectors(&fresh), "two-patch replay differs from live session");
}

#[test]
fn b1_reuse_same_path_then_insert_equals_fresh_session() {
    let f = fx();
    let mut s = open(&f);
    run(&mut s, A); run(&mut s, C);
    run(&mut s, &format!("USE \"{}\";", q(f.dir.path())));
    let after_use = s.state_fingerprint();
    let pristine = open(&f).state_fingerprint();
    eprintln!("changed vs pristine after re-USE: {:?}", after_use.changed_since(&pristine));
    run(&mut s, A);
    let mut fresh = open(&f);
    run(&mut fresh, A);
    assert_eq!(vectors(&s), vectors(&fresh), "INSERT after re-USE differs from INSERT in a fresh session");
}

#[test]
fn b4_knn_reinsert_same_key_takes_effect() {
    let f = fx();
    let mut s = open(&f);
    let o1 = run(&mut s, r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#);
    let o2 = run(&mut s, r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[6]");"#);
    eprintln!("{o1:?}\n{o2:?}");
    let out = infer(&mut s, "The b of a is").join("\n");
    eprintln!("{out}");
    let d = run(&mut s, r#"DESCRIBE "a";"#).join("\n");
    eprintln!("{d}");
    assert!(out.contains("[6]") && !out.contains("[5]"), "second assertion not in effect:\n{out}");
}

#[test]
fn control_single_insert_replays_exactly_and_is_deterministic() {
    let f = fx();
    let patch = f.dir.path().join("one.vlp");
    let mut s = open(&f);
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&patch)));
    run(&mut s, A);
    run(&mut s, "SAVE PATCH;");
    let mut fresh = open(&f);
    run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&patch)));
    assert_eq!(vectors(&s), vectors(&fresh));
    let mut s2 = open(&f); run(&mut s2, A); run(&mut s2, C);
    let mut s3 = open(&f); run(&mut s3, A); run(&mut s3, C);
    assert_eq!(vectors(&s2), vectors(&s3));
}

#[test]
fn b5a_refused_insert_leaves_no_auto_patch() {
    let c = tempfile::tempdir().unwrap(); let dir = tempfile::tempdir().unwrap();
    encode_fixture_container(miniature_glimmer, c.path(), dir.path(), "tokless");
    let mut s = Session::new();
    run(&mut s, &format!("USE \"{}\";", q(dir.path())));
    let before = s.state_fingerprint();
    let err = s.execute(&parse(A).unwrap()).expect_err("no tokenizer");
    eprintln!("insert refused: {err}");
    eprintln!("changed: {:?}", s.state_fingerprint().changed_since(&before));
    let save = s.execute(&parse("SAVE PATCH;").unwrap()).expect_err("save");
    assert!(save.to_string().contains("no active patch session"), "refused INSERT opened a recording: {save}");
}
