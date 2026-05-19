//! End-to-end demo of all four KV engines on synthetic weights.
//!
//! Loads the `larql_inference::test_utils` 2-layer fixture, then runs each
//! engine through prefill + a few decode steps, printing the per-engine
//! diagnostics so you can see the trait surface in action.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p larql-kv --example engine_ladder
//! ```

use larql_inference::cpu_engine_backend;
use larql_inference::ffn::WeightFfn;
use larql_inference::test_utils::make_test_weights;
use larql_kv::{EngineKind, KvEngine};

fn run_engine(label: &str, mut engine: Box<dyn KvEngine>) {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..8).collect();

    print!("{label:<32} ");
    let prefill = engine.prefill(&weights, &ffn, &prompt);
    if prefill.is_none() {
        println!("prefill returned None (engine not configured)");
        return;
    }

    for tok in 0..3 {
        let _ = engine.decode_step(&weights, &ffn, tok as u32);
    }

    let info = engine.info();
    println!(
        "memory={:>8} bytes  window={:<5}  cold={:>8} bytes  [{}]",
        engine.memory_bytes(),
        engine.window_tokens(),
        engine.cold_bytes(),
        info.summary(),
    );
}

fn main() {
    let specs = [
        "markov-rs",
        "markov-rs:window=4",
        "unlimited-context:window=4",
        "turbo-quant:bits=4",
        "tq3",
        "apollo:layer=1,coef=8.0,top_k=4",
        "boundary-kv:chunk_tokens=4,sequence_id=demo",
        "boundary-kv:window=4,chunk_tokens=4,sequence_id=demo-bounded",
        "markov-rs-codec",
        "markov-rs-codec:window=4",
    ];

    println!("larql-kv engine ladder (synthetic 2-layer model)\n");
    println!("{:<32} diagnostics", "engine");
    println!("{}", "-".repeat(96));

    for spec in specs {
        let kind = match EngineKind::from_name(spec) {
            Some(k) => k,
            None => {
                println!("{spec:<32} <unparseable>");
                continue;
            }
        };
        run_engine(spec, kind.build(cpu_engine_backend()));
    }
}
