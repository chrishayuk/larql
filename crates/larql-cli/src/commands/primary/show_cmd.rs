//! `larql show <model>` — print vindex metadata.
//!
//! Resolves the model the same way `run` does, then dumps `index.json` plus
//! file inventory (size per component) so you can see what's actually in
//! this vindex before you load it.

use clap::Args;

use crate::commands::primary::cache;

#[derive(Args)]
pub struct ShowArgs {
    /// Vindex directory, `hf://owner/name`, `owner/name`, or cache shorthand.
    pub model: String,
}

pub fn run(args: ShowArgs) -> Result<(), Box<dyn std::error::Error>> {
    let path = cache::resolve_model(&args.model)?;

    println!("Model:      {}", args.model);
    println!("Path:       {}", path.display());

    // Dispatch on the container's own discriminator, then let each generation
    // describe itself in its own vocabulary. Normalising VINDEX3 back into a
    // VINDEX2-shaped summary would drop exactly what a VINDEX3 user needs to
    // see — programme id, storage key, manifest validity.
    // `Quant: Q4K` names a format but not its cost. `source_dtype` carries the
    // float precision that format replaced, so the precision map below can
    // state the compression outright instead of leaving it to be inferred.
    let mut source_dtype: Option<larql_vindex::StorageDtype> = None;

    match larql_vindex::format::generation::detect_generation(&path)? {
        larql_vindex::format::generation::ContainerGeneration::V3 => show_v3(&path)?,
        larql_vindex::format::generation::ContainerGeneration::V2 => {
            let cfg = larql_vindex::load_vindex_config(&path)?;
            println!("Generation: VINDEX2");
            println!("Layers:     {}", cfg.num_layers);
            println!("Hidden:     {}", cfg.hidden_size);
            println!("Dtype:      {:?}", cfg.dtype);
            println!("Quant:      {:?}", cfg.quant);
            source_dtype = Some(cfg.dtype);
        }
    }

    show_precision(&path, source_dtype);

    println!("\nFiles:");
    let mut entries: Vec<_> = std::fs::read_dir(&path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.metadata().map(|m| m.is_file()).unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        println!("  {:<32} {:>12}", name, human_size(size));
    }
    Ok(())
}

/// Describe a VINDEX3 container in its own terms.
///
/// The per-layer lines are the ones with no VINDEX2 equivalent: which
/// programme interprets the bank, and which storage key it resolves to. Those
/// are what a binding failure is diagnosed from, so `show` is where they
/// belong.
fn show_v3(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    // Read the index directly rather than through `Vindex3Container::open`.
    // That reader is the *routed-MoE* open path and refuses a system container
    // (dense, no routed programme) outright — but `show` describes containers,
    // it doesn't serve them, and a dense container is exactly one a caller may
    // be running `show` to understand. The MoE reader is opened below, only
    // once there is a programme for it to describe.
    let raw = std::fs::read_to_string(path.join(larql_vindex::format::filenames::INDEX_JSON))?;
    let index: larql_vindex::format::vindex3::Vindex3Index =
        serde_json::from_str(&raw).map_err(|e| format!("parse VINDEX3 index.json: {e}"))?;

    println!("Generation: VINDEX3 (index.json schema {})", index.version);
    println!("Layers:     {}", index.num_layers);
    println!("Hidden:     {}", index.hidden_size);
    println!("Family:     {}", index.family);
    println!("Profiles:   {}", index.profile_names().join(", "));
    let shape = index.shape()?;
    println!("Shape:      {}", shape.describe());
    if !shape.is_normative() {
        println!(
            "Migrate:    {}",
            larql_vindex::format::vindex3::LEGACY_BANK_MIGRATION
        );
    }

    // Report the authorities this container carries; print no negative for the
    // ones it doesn't.
    //
    // `moe_manifest` is the *routed-MoE programme* manifest, an artifact of the
    // import path. A graph-encoded container carries none whether or not it
    // routes — gpt-oss-20b is a routed MoE, expert bank and all, with
    // `moe_manifest: None`. An absence line here would say nothing true about
    // the model's shape while strongly implying "dense", so there isn't one.
    // What the model contains is the representations table's job.
    if let Some(graph) = index.system_graph.as_deref() {
        println!("Graph:      {graph}");
    }
    if let Some(manifest) = index.moe_manifest.as_deref() {
        println!("Manifest:   {manifest}  (routed-MoE programme)");
    }

    match index.authority {
        larql_vindex::format::vindex3::index::ContainerAuthority::Canonical => {}
        larql_vindex::format::vindex3::index::ContainerAuthority::Derived => {
            println!("Authority:  derived  (executable; not re-compilable)");
            if let Some(model) = index.derived_from_model.as_deref() {
                println!("Source:     {model}");
            }
            let digests: Vec<&str> = index
                .representations
                .values()
                .filter_map(|e| e.source_representation_digest.as_deref())
                .collect();
            if let Some(d) = digests.first() {
                println!("Derives:    sha256:{}…", &d[..d.len().min(16)]);
            }
        }
    }

    show_v3_representations(&index);

    // §9.1: show what each profile actually selects, not just that it exists.
    // A container that carries alternative packs is exactly the one where
    // "which bytes does this profile run" stops being obvious.
    if !index.variants.is_empty() {
        println!("\nVariants:");
        for region_set in index.variants.region_sets() {
            let set = index
                .variants
                .get(&region_set)
                .expect("region_sets() lists catalogued keys");
            println!(
                "  {region_set}: {} (baseline {})",
                set.present().join(", "),
                set.baseline
            );
        }
        for name in index.profile_names() {
            let resolved = index.select_profile(name)?;
            println!("\nProfile '{name}' selects:");
            for (region_set, stored) in resolved.entries() {
                println!(
                    "  {region_set} -> {} [{}]",
                    stored.storage,
                    stored.fidelity.name()
                );
            }
        }
    }

    // Routed-programme detail needs the MoE reader, so it is reached for only
    // when the index declares a programme. A system container simply has none
    // of this to report — that is a shape, not a failure.
    if index.moe_manifest.is_some() {
        let container = larql_vindex::format::vindex3::Vindex3Container::open(path)?;
        println!("\nMoE layers:");
        for layer in &container.manifest().layers {
            let bank = &layer.routed_bank;
            let known = if bank.resolve_programme().is_some() {
                ""
            } else {
                "  [programme not implemented by this binary]"
            };
            println!(
                "  layer {:<3} {:<24} experts {:<4} storage {}{}",
                layer.layer, bank.programme, bank.experts, bank.storage, known
            );
        }

        let defects = container.verify();
        if defects.is_empty() {
            println!("\nStructure:  bindable (no defects)");
        } else {
            println!("\nStructure:  {} defect(s)", defects.len());
            for d in &defects {
                println!("  - {d}");
            }
        }
    }
    Ok(())
}

/// What each stored object is encoded as — VINDEX3's answer to the VINDEX2
/// precision map.
///
/// The granularity differs on purpose: VINDEX2 records a block format per
/// tensor, so its map is per-projection. VINDEX3 records an encoding per
/// stored object, so this is per-object, and an object carrying more than one
/// encoding says so in its tag (`BF16+MXFP4`). Both answer the same question —
/// what precision is actually on disk — at the granularity their format keeps.
fn show_v3_representations(index: &larql_vindex::format::vindex3::Vindex3Index) {
    if index.representations.is_empty() {
        return;
    }
    println!("\nRepresentations (what is stored, and how):");
    println!(
        "  {:<34} {:<12} {:>8} {:>14}",
        "object", "encoding", "tensors", "bytes"
    );
    println!("  {}", "-".repeat(72));

    let mut total = 0u64;
    // Declaration order is the encoder's, which groups an object with its
    // variants; sorting by size would split them.
    for entry in index.representations.values() {
        total += entry.payload_bytes;
        println!(
            "  {:<34} {:<12} {:>8} {:>14}",
            entry.object,
            entry.encoding,
            thousands(entry.tensor_count as u64),
            human_size(entry.payload_bytes),
        );
    }
    println!("  {}", "-".repeat(72));
    println!(
        "  {:<34} {:<12} {:>8} {:>14}",
        "TOTAL",
        "",
        "",
        human_size(total)
    );
}

/// Print the per-projection precision map, if this vindex has one.
///
/// Answers the question `Quant: Q4K` raises but doesn't settle: how many bits
/// per weight is that *actually*, and did every tensor get the same deal? For
/// the Ollama-compatible Q4_K_M mix the answer is neither 4 bits nor uniform,
/// and both facts are load-bearing when sizing or debugging a build.
///
/// Silent when nothing is quantised — a float vindex has no map to draw — and
/// on a manifest read failure. `show` is an inventory command; a precision
/// table it couldn't build is not a reason to fail the listing the caller
/// asked for.
fn show_precision(path: &std::path::Path, source_dtype: Option<larql_vindex::StorageDtype>) {
    let Ok(tensors) = larql_vindex::quant::read_quant_inventory(path) else {
        return;
    };
    let map = larql_vindex::quant::precision_map(&tensors);
    if map.is_empty() {
        return;
    }

    println!("\nPrecision (from this vindex's own manifests):");
    println!(
        "  {:<11} {:<7} {:>15} {:>15} {:>12}",
        "projection", "format", "weights", "bytes", "bits/weight"
    );
    println!("  {}", "-".repeat(65));
    for row in &map.rows {
        println!(
            "  {:<11} {:<7} {:>15} {:>15} {:>12.4}",
            row.projection,
            row.format,
            thousands(row.weights),
            thousands(row.bytes),
            row.bits_per_weight()
        );
    }
    println!("  {}", "-".repeat(65));
    println!(
        "  {:<11} {:<7} {:>15} {:>15} {:>12.4}",
        "TOTAL",
        if map.is_mixed() { "mixed" } else { "uniform" },
        thousands(map.total_weights),
        thousands(map.total_bytes),
        map.bits_per_weight()
    );

    // Compression is only meaningful against a stated source precision.
    if let Some(dtype) = source_dtype {
        println!(
            "\n  {:<22}{:>10}",
            format!("same weights at {dtype}"),
            human_size(map.source_bytes(dtype))
        );
        println!(
            "  {:<22}{:>10}   ({:.2}x)",
            "quantised",
            human_size(map.total_bytes),
            map.compression_vs(dtype)
        );
    }
}

/// `3208642560` → `3,208,642,560`. Big weight counts are the whole point of
/// the table; unseparated they're unreadable at a glance.
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn human_size(bytes: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if bytes >= G {
        format!("{:.2} GB", bytes as f64 / G as f64)
    } else if bytes >= M {
        format!("{:.1} MB", bytes as f64 / M as f64)
    } else if bytes >= K {
        format!("{:.1} KB", bytes as f64 / K as f64)
    } else {
        format!("{bytes} B")
    }
}
