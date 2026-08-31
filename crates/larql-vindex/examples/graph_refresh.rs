//! Rebuild a container's system graph from its source checkpoint, at
//! the current schema.
//!
//! The hero containers were encoded before `context_length` had a home
//! on the execution surface (ce693017), so their graphs lack a fact the
//! metadata gate requires. The fix is NOT to edit the artifact by hand
//! — it is to re-derive the graph from the same checkpoint with the
//! same builder and let the diff show that nothing changed except what
//! the schema learned to record.
//!
//! Prints the rebuilt graph to the given path; writes nothing into the
//! container. The caller inspects the diff and copies deliberately.

use larql_models::inventory::build_inventory;
use larql_vindex::format::vindex3::graph::build_from_inventories;

fn main() {
    let mut args = std::env::args().skip(1);
    let checkpoint = std::path::PathBuf::from(args.next().expect("checkpoint dir"));
    let out = std::path::PathBuf::from(args.next().expect("output json path"));

    let name = checkpoint
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .expect("checkpoint path has a stem");
    let inventory = build_inventory(&checkpoint).expect("read checkpoint");
    let built = build_from_inventories(&[(name, inventory)]);

    println!("unplaced groups        {}", built.unplaced.len());
    println!("unresolved interfaces  {}", built.unresolved_interfaces.len());
    println!("incomplete surfaces    {}", built.incomplete_surfaces.len());
    for s in &built.incomplete_surfaces {
        println!("  {}: {:?}", s.component, s.missing);
    }
    let defects = built.graph.validate();
    println!("validation defects     {}", defects.len());
    for d in defects.iter().take(5) {
        println!("  {d:?}");
    }

    let json = serde_json::to_string_pretty(&built.graph).expect("serialise");
    std::fs::write(&out, json).expect("write");
    println!("wrote {}", out.display());
}
