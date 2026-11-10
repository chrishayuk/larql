//! Criterion benchmarks for the LQL parser.
//!
//! Measures parser throughput across the four major statement families
//! (lifecycle, query, mutation, introspection) and the cost of parsing
//! a 100-statement batch (the typical REPL hot-reload workload).
//!
//! Run: `cargo bench -p larql-lql --bench parser`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_lql::parse;

const LIFECYCLE: &str = r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3.vindex" COMPONENTS FFN_GATE, FFN_DOWN LAYERS 0-33 WITH ALL;"#;
const COMPILE: &str = r#"COMPILE CURRENT INTO MODEL "gemma3-edited/" FORMAT safetensors;"#;
const COMPILE_INTO_VINDEX: &str = r#"COMPILE CURRENT INTO VINDEX "gemma3-baked.vindex";"#;
const WALK: &str = r#"WALK "The capital of France is" TOP 5 LAYERS 25-33 MODE hybrid COMPARE;"#;
const SELECT: &str = r#"SELECT entity, target FROM EDGES WHERE relation = "capital" ORDER BY confidence DESC LIMIT 10;"#;
const DESCRIBE: &str = r#"DESCRIBE "France" KNOWLEDGE RELATIONS ONLY;"#;
const INFER: &str = r#"INFER "The capital of France is" TOP 5 COMPARE;"#;
const EXPLAIN_INFER: &str = r#"EXPLAIN INFER "The capital of France is" TOP 5;"#;

const INSERT_MIN: &str =
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John", "lives-in", "London");"#;
const INSERT_FULL: &str = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital-of", "Poseidon") AT LAYER 24 CONFIDENCE 0.95 ALPHA 0.30 MODE COMPOSE;"#;
const UPDATE: &str =
    r#"UPDATE EDGES SET target = "London", confidence = 0.9 WHERE layer = 26 AND feature = 8821;"#;
const DELETE: &str = r#"DELETE FROM EDGES WHERE layer = 26 AND feature = 8821;"#;
const MERGE: &str = r#"MERGE "src.vindex" INTO "dst.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#;

const SHOW_RELATIONS: &str = "SHOW RELATIONS AT LAYER 26 WITH EXAMPLES;";
const SHOW_LAYERS: &str = "SHOW LAYERS RANGE 0-10;";
const SHOW_FEATURES: &str = r#"SHOW FEATURES 26 WHERE token = "Paris" LIMIT 5;"#;
const STATS: &str = "STATS;";

const PIPE: &str = r#"WALK "test" TOP 5 |> EXPLAIN WALK "test";"#;

/// Single-statement parse throughput across the major families.
fn bench_parse_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_single");

    let cases: &[(&str, &str)] = &[
        ("extract", LIFECYCLE),
        ("compile_into_model", COMPILE),
        ("compile_into_vindex", COMPILE_INTO_VINDEX),
        ("walk", WALK),
        ("select", SELECT),
        ("describe", DESCRIBE),
        ("infer", INFER),
        ("explain_infer", EXPLAIN_INFER),
        ("insert_min", INSERT_MIN),
        ("insert_full_compose_with_alpha", INSERT_FULL),
        ("update", UPDATE),
        ("delete", DELETE),
        ("merge", MERGE),
        ("show_relations", SHOW_RELATIONS),
        ("show_layers", SHOW_LAYERS),
        ("show_features", SHOW_FEATURES),
        ("stats", STATS),
        ("pipe", PIPE),
    ];

    group.throughput(Throughput::Elements(1));
    for (name, src) in cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), src, |b, src| {
            b.iter(|| {
                let _ = parse(src).unwrap();
            });
        });
    }
    group.finish();
}

/// Batch parse — simulates a vindexfile build script (~100 statements
/// of mixed shapes parsed in one go).
fn bench_parse_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_batch");

    // Build a 100-statement script by repeating a representative mix.
    let mix = [
        LIFECYCLE,
        WALK,
        DESCRIBE,
        INSERT_FULL,
        UPDATE,
        DELETE,
        SELECT,
        SHOW_FEATURES,
        STATS,
        INFER,
    ];
    let mut script = String::new();
    for i in 0..100 {
        script.push_str(mix[i % mix.len()]);
        script.push('\n');
    }
    let stmt_count = 100u64;

    group.throughput(Throughput::Elements(stmt_count));
    group.bench_function("100_mixed_statements", |b| {
        b.iter(|| {
            // Parse the lines individually — same shape the REPL uses.
            for line in script.lines() {
                let _ = parse(line).unwrap();
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_parse_single, bench_parse_batch);
criterion_main!(benches);
