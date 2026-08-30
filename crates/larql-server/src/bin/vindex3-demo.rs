//! vindex3-demo — write the public explorer's demo container.
//!
//! The public deployment (`deploy/fly-explorer/`) serves one immutable
//! VINDEX3 container. This binary produces it at boot: the miniature
//! Glimmer system — two layers with a mixed attention policy
//! (Sliding(3)+RoPE, Full+NoPE), gated attention, synthetic weights —
//! encoded through the real inventory → encode pipeline, plus the
//! synthetic tokenizer, under the name `vindex3-demo`.
//!
//! Why generate instead of download: the container is kilobytes, the
//! generator is the same fixture every LQL/serve gate runs against, and
//! a boot-time regeneration means the public box holds no state worth
//! keeping — wipe it and it rebuilds identically. What this is NOT: a
//! production model. The weights are synthetic; the format, graph,
//! provenance, and execution are real, which is exactly what the public
//! explorer demonstrates.

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

fn main() {
    let Some(dest) = std::env::args().nth(1) else {
        eprintln!("usage: vindex3-demo <output-dir>");
        std::process::exit(2);
    };
    let dest = std::path::PathBuf::from(dest);
    if dest.join("index.json").exists() {
        println!("demo container already present at {}", dest.display());
        return;
    }
    std::fs::create_dir_all(&dest).expect("create output dir");
    let checkpoint =
        std::env::temp_dir().join(format!("vindex3-demo-checkpoint-{}", std::process::id()));
    std::fs::create_dir_all(&checkpoint).expect("create checkpoint dir");
    encode_fixture_container(miniature_glimmer, &checkpoint, &dest, "vindex3-demo");
    std::fs::write(
        dest.join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .expect("write tokenizer");
    let _ = std::fs::remove_dir_all(&checkpoint);
    println!("demo container written to {}", dest.display());
}
