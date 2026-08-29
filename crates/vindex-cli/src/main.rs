//! vindex — open a VINDEX3 artifact and interrogate it.
//!
//! Deliberately small: the format-native verbs only, each answering
//! from the container's own declarations, each speaking `--json`. The
//! text renderings below are projections of the same facts object the
//! JSON emits — one result, two views, and the web Explorer's designed
//! panels are the third.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "vindex",
    about = "The format-native VINDEX3 tool: inspect, describe, representations, precision, verify.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Emit the structured result instead of text.
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Command {
    /// The container, reconstructed from itself: identity, census, coherence.
    Inspect { container: PathBuf },
    /// One logical object, in full — identity, bindings, representations, tensor-table head.
    Describe {
        container: PathBuf,
        /// A logical object id, or an unambiguous suffix of one.
        address: String,
        /// How many tensor-table rows to show per representation.
        #[arg(long, default_value_t = 8)]
        values: usize,
    },
    /// The physical directory: what exists as bytes, with recorded fidelity.
    Representations { container: PathBuf },
    /// Bits per weight — derived from stored bytes over tensor elements, never asserted.
    Precision { container: PathBuf },
    /// The container against its own recorded hashes, re-derived from the artifact alone.
    Verify { container: PathBuf },
}

fn kv(label: &str, value: impl std::fmt::Display) {
    println!("{label:<14} {value}");
}

fn render_inspect(v: &Value) {
    kv("model", v["model"].as_str().unwrap_or("?"));
    kv("family", v["family"].as_str().unwrap_or("?"));
    kv("generation", &v["generation"]);
    kv(
        "geometry",
        format!("{} layers · hidden {}", v["num_layers"], v["hidden_size"]),
    );
    kv("authority", v["authority"].as_str().unwrap_or("?"));
    if let Some(d) = v["derived_from_model"].as_str() {
        kv("derives from", d);
    }
    println!();
    println!(
        "{:<22} {:<16} {:>7} {:>8}",
        "COMPONENT", "ROLE", "LAYERS", "HIDDEN"
    );
    for c in v["components"].as_array().into_iter().flatten() {
        println!(
            "{:<22} {:<16} {:>7} {:>8}",
            c["id"].as_str().unwrap_or("?"),
            c["role"].as_str().unwrap_or("?"),
            c["num_layers"],
            c["hidden_size"]
        );
    }
    println!();
    kv(
        "graph",
        format!(
            "{} object(s) · {} edge(s) · {}",
            v["objects"],
            v["edges"],
            if v["coherent"].as_bool().unwrap_or(false) {
                "coherent"
            } else {
                "DEFECTS PRESENT"
            }
        ),
    );
}

fn render_representations(v: &Value) {
    println!(
        "{:<34} {:<12} {:<22} {:>8} {:>14}",
        "REPRESENTATION", "ENCODING", "FIDELITY", "TENSORS", "BYTES"
    );
    for e in v["entries"].as_array().into_iter().flatten() {
        println!(
            "{:<34} {:<12} {:<22} {:>8} {:>14}",
            e["id"].as_str().unwrap_or("?"),
            e["encoding"].as_str().unwrap_or("?"),
            e["fidelity"].as_str().unwrap_or("—"),
            e["tensor_count"],
            e["payload_bytes"]
        );
    }
}

fn render_describe(v: &Value) {
    let o = &v["object"];
    kv("object", o["id"].as_str().unwrap_or("?"));
    kv("kind", o["kind"].as_str().unwrap_or("?"));
    kv("component", o["component"].as_str().unwrap_or("?"));
    for r in o["representations"].as_array().into_iter().flatten() {
        kv(
            "representation",
            format!(
                "{} · {}",
                r["encoding"].as_str().unwrap_or("?"),
                r["fidelity"].as_str().unwrap_or("?")
            ),
        );
    }
    for d in v["directory"].as_array().into_iter().flatten() {
        println!();
        kv(
            "stored as",
            format!(
                "{} — {} bytes, {} tensors",
                d["segment"].as_str().unwrap_or("?"),
                d["payload_bytes"],
                d["tensor_count"]
            ),
        );
        for t in d["tensor_table_head"].as_array().into_iter().flatten() {
            println!(
                "  {:<40} {:<8} {}",
                t["name"].as_str().unwrap_or("?"),
                t["dtype"].as_str().unwrap_or("?"),
                serde_json::to_string(&t["shape"]).unwrap_or_default()
            );
        }
    }
}

fn render_precision(v: &Value) {
    println!(
        "{:<34} {:<10} {:>16} {:>14} {:>10}",
        "REPRESENTATION", "ENCODING", "WEIGHTS", "BYTES", "BITS/W"
    );
    for e in v["entries"].as_array().into_iter().flatten() {
        println!(
            "{:<34} {:<10} {:>16} {:>14} {:>10.4}",
            e["id"].as_str().unwrap_or("?"),
            e["encoding"].as_str().unwrap_or("?"),
            e["weights"],
            e["payload_bytes"],
            e["bits_per_weight"].as_f64().unwrap_or(0.0)
        );
    }
    println!();
    kv("weights", &v["total_weights"]);
    kv(
        "effective",
        format!(
            "{:.4} bits / weight",
            v["effective_bits_per_weight"].as_f64().unwrap_or(0.0)
        ),
    );
    if !v["precision_map"].is_null() {
        kv(
            "precision map",
            "present — compiled policy carried by the index (see --json)",
        );
    }
}

fn render_verify(v: &Value) {
    for e in v["entries"].as_array().into_iter().flatten() {
        let ok = e["segment_ok"].as_bool().unwrap_or(false)
            && e["payload_ok"].as_bool().unwrap_or(false);
        println!(
            "{:<34} {}",
            e["id"].as_str().unwrap_or("?"),
            if ok { "ok" } else { "MISMATCH" }
        );
    }
    println!();
    kv(
        "verified",
        if v["verified"].as_bool().unwrap_or(false) {
            "yes — the artifact agrees with its own record"
        } else {
            "NO"
        },
    );
    kv("scope", v["scope"].as_str().unwrap_or(""));
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Inspect { container } => vindex_cli::inspect_facts(container),
        Command::Describe {
            container,
            address,
            values,
        } => vindex_cli::describe_facts(container, address, *values),
        Command::Representations { container } => vindex_cli::representations_facts(container),
        Command::Precision { container } => vindex_cli::precision_facts(container),
        Command::Verify { container } => vindex_cli::verify_facts(container),
    };
    match result {
        Ok(v) => {
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            } else {
                match &cli.command {
                    Command::Inspect { .. } => render_inspect(&v),
                    Command::Describe { .. } => render_describe(&v),
                    Command::Representations { .. } => render_representations(&v),
                    Command::Precision { .. } => render_precision(&v),
                    Command::Verify { .. } => render_verify(&v),
                }
            }
            if let Some(false) = v["verified"].as_bool() {
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("vindex: {e}");
            ExitCode::from(2)
        }
    }
}
