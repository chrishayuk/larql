//! Emit a hero container as a qwen35 GGUF through the one assembled
//! pipeline (`gguf::export::export_qwen35`) and print the observed
//! ledger. The example holds no logic of its own any more — roles come
//! from the operation plan, selection from the precision map, and the
//! file is verified through the independent reader before this prints.

use larql_vindex::format::vindex3::gguf::export::export_qwen35;

fn main() {
    let mut args = std::env::args().skip(1);
    let root = std::path::PathBuf::from(args.next().expect("container path"));
    let out = std::path::PathBuf::from(args.next().expect("output .gguf path"));

    let t0 = std::time::Instant::now();
    match export_qwen35(&root, &out) {
        Ok(r) => {
            println!("selected  {}", r.selected_encoding);
            println!(
                "walk      {} tensors, geometry {}/{}, {} scale siblings",
                r.ledger.source_total,
                r.ledger.geometry_reconciled,
                r.ledger.accounted,
                r.ledger.generated_scale_tensors
            );
            println!(
                "vocab     {} tokens + {} pad, {} merges",
                r.vocab_tokens, r.vocab_padded, r.vocab_merges
            );
            println!(
                "VERIFIED  {} tensors ({} NVFP4, {} scale siblings), {} metadata keys",
                r.verify.tensors,
                r.verify.nvfp4_tensors,
                r.verify.scale_siblings,
                r.verify.metadata_keys
            );
            println!(
                "written   {} — {:.2} GB in {:.1?}",
                r.out.display(),
                r.bytes as f64 / 1e9,
                t0.elapsed()
            );
        }
        Err(e) => {
            eprintln!("export failed: {e}");
            std::process::exit(1);
        }
    }
}
