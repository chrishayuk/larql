//! Golden, on the real Qwen3-0.6B container (ignored by default; needs the 1.3 GB
//! container). WSP-001 Stage 1's seven-weight edit, typed as LQL:
//!
//! ```text
//! USE "<qwen3-0.6b-q8_0.vindex3>";
//! BEGIN PATCH "wsp001.vlp";
//! UPDATE WEIGHTS SET DELTA = 0.0078125 WHERE TENSOR = "26.mlp.down_proj.weight"
//!   AND ROW = 100 AND COL IN (200, 201, 202, 203, 204, 206, 207);
//! SAVE PATCH;
//! -- a completely fresh session:
//! USE "<qwen3-0.6b-q8_0.vindex3>";  APPLY PATCH "wsp001.vlp";
//! COMPILE CURRENT INTO VINDEX "<out>";
//! ```
//!
//!   WSP_GOLDEN_SOURCE=<shipped container> WSP_GOLDEN_EXPECT=<the Python prototype's edited container> \
//!   WSP_GOLDEN_OUT=<new container path> \
//!   cargo test --release -p larql-lql --test weight_patch_lql golden -- --ignored --nocapture
//!
//! The expected container was produced by an independent implementation and
//! checked by two independent checkers; byte agreement is the gate.

use std::path::PathBuf;

use crate::support::*;

#[test]
#[ignore]
fn the_stage_1_edit_typed_as_lql_reproduces_the_verified_bytes() {
    let var = |n: &str| PathBuf::from(std::env::var(n).unwrap_or_else(|_| panic!("set {n}")));
    let (source, expect, out) = (
        var("WSP_GOLDEN_SOURCE"),
        var("WSP_GOLDEN_EXPECT"),
        var("WSP_GOLDEN_OUT"),
    );
    let scratch = tempfile::tempdir().unwrap();
    let vlp = scratch.path().join("wsp001.vlp");
    let edit = "UPDATE WEIGHTS SET DELTA = 0.0078125 WHERE TENSOR = \"26.mlp.down_proj.weight\" \
                AND ROW = 100 AND COL IN (200, 201, 202, 203, 204, 206, 207);";

    let mut author = session_on(&source);
    run(&mut author, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    for line in run(&mut author, edit) {
        println!("  {line}");
    }
    println!("  {:?}", run(&mut author, "SAVE PATCH;"));

    let mut fresh = session_on(&source);
    println!(
        "  {:?}",
        run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&vlp)))
    );
    let started = std::time::Instant::now();
    for line in run(
        &mut fresh,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
    ) {
        println!("  {line}");
    }
    println!("  compiled in {:.1?}", started.elapsed());

    assert!(is_coherent(&out));
    let (mine, theirs) = (digests(&out), digests(&expect));
    for id in [CANONICAL, DERIVED] {
        println!(
            "{id}: payload {} segment {}",
            &mine[id].0[..16],
            &mine[id].1[..16]
        );
        assert_eq!(mine[id], theirs[id], "{id}");
    }
    std::fs::copy(&vlp, out.with_extension("vlp")).unwrap();
}
