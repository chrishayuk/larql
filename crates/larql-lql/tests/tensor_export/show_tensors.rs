//! `SHOW TENSORS`: every tensor by its qualified name, with what its bare name finds.

use crate::support::*;

fn line_of<'a>(lines: &'a [String], qualified: &str) -> &'a str {
    lines
        .iter()
        .find(|l| l.split_whitespace().next() == Some(qualified))
        .unwrap_or_else(|| panic!("{qualified} not listed: {lines:#?}"))
}

#[test]
fn every_tensor_is_listed_by_its_qualified_name() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let lines = run(&mut s, "SHOW TENSORS;");
    let matrix = line_of(&lines, &format!("{CANONICAL}::{MATRIX}"));
    assert!(matrix.contains("BF16"), "{matrix}");
    assert!(matrix.contains("[64, 256]"), "{matrix}");
    assert!(matrix.trim_end().ends_with("yes"), "{matrix}");
    let derived = line_of(&lines, &format!("{DERIVED}::{MATRIX}"));
    assert!(derived.contains("Q8_0"), "{derived}");
    assert!(derived.contains("no (derived)"), "{derived}");
}

#[test]
fn the_shared_bare_name_is_marked_ambiguous() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let lines = run(&mut s, "SHOW TENSORS;");
    for representation in [
        EMBEDDING,
        "target.final_norm@BF16",
        "target.output_head@BF16",
    ] {
        let line = line_of(&lines, &format!("{representation}::{AMBIGUOUS}"));
        assert!(line.contains("ambiguous"), "{line}");
    }
    let summary = lines.last().unwrap();
    assert!(summary.contains("3 bare name(s) ambiguous"), "{summary}");
}

#[test]
fn every_listed_float_name_exports() {
    // What SHOW TENSORS lists is what EXPORT WEIGHT accepts: every qualified name of a float tensor exports.
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let lines = run(&mut s, "SHOW TENSORS \"embedding\";");
    let names: Vec<String> = lines
        .iter()
        .filter(|l| l.contains("::"))
        .filter_map(|l| l.split_whitespace().next().map(str::to_string))
        .collect();
    assert_eq!(names, vec![format!("{EMBEDDING}::{AMBIGUOUS}")]);
    for (i, name) in names.iter().enumerate() {
        run(&mut s, &export(name, "", &fx.path(&format!("{i}.npy")), ""));
    }
}

#[test]
fn a_filter_keeps_matching_representations_only() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let lines = run(&mut s, "SHOW TENSORS \"Q8_0\";");
    let listed: Vec<&String> = lines.iter().filter(|l| l.contains("::")).collect();
    assert!(!listed.is_empty());
    assert!(listed.iter().all(|l| l.starts_with(DERIVED)), "{listed:#?}");
    let none = run(&mut s, "SHOW TENSORS \"no-such-object\";");
    assert_eq!(none.len(), 1);
    assert!(none[0].contains("no representations match"), "{none:?}");
}

#[test]
fn an_f32_container_lists_f32() {
    let fx = Fixture::f32();
    let mut s = fx.session();
    let lines = run(&mut s, "SHOW TENSORS;");
    let line = line_of(&lines, &format!("{F32_CANONICAL}::{F32_MATRIX}"));
    assert!(line.contains("F32") && line.contains("[12, 20]"), "{line}");
}
