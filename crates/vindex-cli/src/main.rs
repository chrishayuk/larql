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

mod update;

#[derive(Parser)]
#[command(
    name = "vindex",
    about = "The format-native VINDEX3 tool: inspect, describe, representations, diff, represent, precision, verify.",
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
        /// Decode and print the first values of this tensor (name or
        /// suffix) — the numbers themselves, from the canonical bytes.
        #[arg(long)]
        peek: Option<String>,
    },
    /// The physical directory: what exists as bytes, with recorded fidelity.
    Representations { container: PathBuf },
    /// Every layer's token-mixer programme, from the operation plan.
    Layers { container: PathBuf },
    /// One object under two of the container's representations, decoded and
    /// compared value by value — the error derived, never asserted.
    Diff {
        container: PathBuf,
        /// First encoding (e.g. F32, BF16).
        a: String,
        /// Second encoding (e.g. NVFP4).
        b: String,
        /// A logical object id, or an unambiguous suffix of one.
        address: String,
        /// How many per-value rows to show.
        #[arg(long, default_value_t = 8)]
        values: usize,
        /// Show values from this tensor (name or suffix) instead of the
        /// tensor with the largest error.
        #[arg(long)]
        tensor: Option<String>,
    },
    /// Compile a representation through the reference compiler into a new
    /// container beside the original. Nothing is destroyed.
    Represent {
        container: PathBuf,
        /// Where to write the new container.
        out: PathBuf,
        /// Target encoding.
        #[arg(long, default_value = "NVFP4")]
        encoding: String,
    },
    /// Compile the selected representation to a GGUF for an independent
    /// runtime, verified against the plan before the command returns.
    Export {
        container: PathBuf,
        /// The .gguf file to write.
        out: PathBuf,
    },
    /// Bits per weight — derived from stored bytes over tensor elements, never asserted.
    Precision {
        container: PathBuf,
        /// The precision map, seen: bits per layer × semantic role, from the
        /// representation each object would execute.
        #[arg(long)]
        matrix: bool,
    },
    /// The container against its own recorded hashes, re-derived from the artifact alone.
    Verify { container: PathBuf },
    /// Install the latest release of this tool. Only ever runs when you ask:
    /// no verb checks for updates on its own, and nothing phones home.
    Update {
        /// Report whether a newer release exists, without installing.
        #[arg(long)]
        check: bool,
    },
}

fn render_export(v: &Value) {
    kv("selected", v["selected"].as_str().unwrap_or("?"));
    let w = &v["walk"];
    kv(
        "walk",
        format!(
            "{} tensors · geometry {}/{} · {} scale siblings",
            w["source_tensors"], w["geometry_reconciled"], w["accounted"], w["scale_siblings"]
        ),
    );
    let vc = &v["vocab"];
    kv(
        "vocab",
        format!(
            "{} tokens + {} pad · {} merges",
            vc["tokens"], vc["padded"], vc["merges"]
        ),
    );
    let ve = &v["verified"];
    kv(
        "verified",
        format!(
            "{} tensors ({} NVFP4 · {} scale siblings) · {} metadata keys",
            ve["tensors"], ve["nvfp4_tensors"], ve["scale_siblings"], ve["metadata_keys"]
        ),
    );
    kv(
        "written",
        format!(
            "{} — {:.2} GB, independent reader agrees with the plan",
            v["out"].as_str().unwrap_or("?"),
            v["bytes"].as_f64().unwrap_or(0.0) / 1e9
        ),
    );
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
            c["num_layers"].as_u64().unwrap_or(0),
            c["hidden_size"].as_u64().unwrap_or(0)
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
            e["tensor_count"].as_u64().unwrap_or(0),
            e["payload_bytes"].as_u64().unwrap_or(0)
        );
    }
}

fn render_peek(v: &Value) {
    let Some(p) = v.get("peek").filter(|p| !p.is_null()) else {
        return;
    };
    println!();
    println!(
        "{} {} {} — first values",
        p["tensor"].as_str().unwrap_or("?"),
        p["dtype"].as_str().unwrap_or("?"),
        serde_json::to_string(&p["shape"]).unwrap_or_default()
    );
    for x in p["values"].as_array().into_iter().flatten() {
        let x = x.as_f64().unwrap_or(0.0);
        println!("  {}{:.6}", if x >= 0.0 { "+" } else { "" }, x);
    }
}

fn render_layers(v: &Value) {
    let rows: Vec<&Value> = v["layers"].as_array().into_iter().flatten().collect();
    let line = |r: &Value| {
        format!(
            "{:<18} ffn {}",
            r["mixer"].as_str().unwrap_or("?"),
            r["ffn"].as_str().unwrap_or("?")
        )
    };
    let mut i = 0;
    while i < rows.len() {
        let l = line(rows[i]);
        let start = rows[i]["layer"].as_u64().unwrap_or(0);
        let mut end = start;
        while i + 1 < rows.len() && line(rows[i + 1]) == l {
            i += 1;
            end = rows[i]["layer"].as_u64().unwrap_or(end);
        }
        let label = if start == end {
            format!("{start}")
        } else {
            format!("{start}–{end}")
        };
        println!("{label:<10}{l}");
        i += 1;
    }
}

fn render_describe(v: &Value) {
    if let Some(mixer) = v["mixer"].as_str() {
        kv("layer", v["layer"].as_u64().unwrap_or(0));
        kv("token mixer", mixer);
        // An operator the container names but this release cannot
        // describe operand-by-operand says so, in place of the empty
        // table that once stood in for the answer.
        if let Some(why) = v["undescribed"].as_str() {
            println!();
            println!("operands      — {why}");
            return;
        }
        println!();
        println!("{:<32} {:<38} SHAPE", "SEMANTICS", "TENSOR");
        for op in v["operands"].as_array().into_iter().flatten() {
            println!(
                "{:<32} {:<38} {}",
                op["role"].as_str().unwrap_or("?"),
                op["tensor"].as_str().unwrap_or("?"),
                serde_json::to_string(&op["shape"]).unwrap_or_default()
            );
        }
        return;
    }
    if let Some(role) = v["role"].as_str() {
        kv("role", role);
        kv("object", v["object"].as_str().unwrap_or("?"));
        kv("tensor", v["tensor"].as_str().unwrap_or("?"));
        kv(
            "shape",
            serde_json::to_string(&v["shape"]).unwrap_or_default(),
        );
        for r in v["representations"].as_array().into_iter().flatten() {
            kv(
                "representation",
                format!(
                    "{} · {} · {:.4} bits/weight",
                    r["encoding"].as_str().unwrap_or("?"),
                    r["dtype"].as_str().unwrap_or("?"),
                    r["bits_per_weight"].as_f64().unwrap_or(0.0)
                ),
            );
        }
        println!();
        println!("VALUES");
        for x in v["values"].as_array().into_iter().flatten() {
            let x = x.as_f64().unwrap_or(0.0);
            println!("  {}{:.6}", if x >= 0.0 { "+" } else { "" }, x);
        }
        return;
    }
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
    render_peek(v);
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

fn render_precision_matrix(v: &Value) {
    for prog in v["programmes"].as_array().into_iter().flatten() {
        let roles: Vec<&str> = prog["roles"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| r.as_str())
            .collect();
        println!(
            "{} · {} layer(s)",
            prog["label"].as_str().unwrap_or("?"),
            prog["layers"].as_u64().unwrap_or(0)
        );
        let fmt_row = |bits: &Value| -> String {
            roles
                .iter()
                .map(|r| match bits[r].as_f64() {
                    Some(b) => format!("{b:>7.2}"),
                    None => format!("{:>7}", "—"),
                })
                .collect::<String>()
        };
        println!(
            "{:<10}{}",
            "LAYER",
            roles.iter().map(|r| format!("{r:>7}")).collect::<String>()
        );
        // Collapse runs of identical rows into ranges — a 64-layer
        // model with a five-layer map should read as its regions.
        let rows: Vec<&Value> = prog["rows"].as_array().into_iter().flatten().collect();
        let mut i = 0;
        while i < rows.len() {
            let line = fmt_row(&rows[i]["bits"]);
            let start = rows[i]["layer"].as_u64().unwrap_or(0);
            let mut end = start;
            while i + 1 < rows.len() && fmt_row(&rows[i + 1]["bits"]) == line {
                i += 1;
                end = rows[i]["layer"].as_u64().unwrap_or(end);
            }
            let label = if start == end {
                format!("{start}")
            } else {
                format!("{start}–{end}")
            };
            println!("{label:<10}{line}");
            i += 1;
        }
        println!();
    }
    println!("MODEL SURFACES");
    for s in v["surfaces"].as_array().into_iter().flatten() {
        println!(
            "{:<34} {:<8} {:>8.4} bits/weight",
            s["object"].as_str().unwrap_or("?"),
            s["representation"].as_str().unwrap_or("?"),
            s["bits_per_weight"].as_f64().unwrap_or(0.0)
        );
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
            e["weights"].as_u64().unwrap_or(0),
            e["payload_bytes"].as_u64().unwrap_or(0),
            e["bits_per_weight"].as_f64().unwrap_or(0.0)
        );
    }
    println!();
    kv(
        "weight slots",
        v["total_weight_slots"].as_u64().unwrap_or(0),
    );
    kv(
        "stored",
        format!(
            "{:.4} bits / weight slot, across every representation carried — an archival container counts each representation of an object; the execution-relevant figure is the BITS/W of the representation a profile selects",
            v["stored_bits_per_weight_slot"].as_f64().unwrap_or(0.0)
        ),
    );
    if !v["precision_map"].is_null() {
        kv(
            "precision map",
            "present — compiled policy carried by the index (see --json)",
        );
    }
}

fn render_diff(v: &Value) {
    kv("object", v["object"].as_str().unwrap_or("?"));
    kv(
        "comparing",
        format!(
            "{} against {}",
            v["a"].as_str().unwrap_or("?"),
            v["b"].as_str().unwrap_or("?")
        ),
    );
    println!();
    println!(
        "{:<38} {:>10} {:>10} {:>13} {:>13}",
        "TENSOR", "WEIGHTS", "CHANGED", "RMS", "MAX"
    );
    for t in v["tensors"].as_array().into_iter().flatten() {
        if let Some(note) = t["note"].as_str() {
            println!("{:<38} {note}", t["tensor"].as_str().unwrap_or("?"));
            continue;
        }
        println!(
            "{:<38} {:>10} {:>10} {:>13.6} {:>13.6}",
            t["tensor"].as_str().unwrap_or("?"),
            t["weights"].as_u64().unwrap_or(0),
            t["changed"].as_u64().unwrap_or(0),
            t["rms_error"].as_f64().unwrap_or(0.0),
            t["max_error"].as_f64().unwrap_or(0.0)
        );
    }
    if let Some(head) = v["values"].as_object() {
        println!();
        println!("{} — first values", head["tensor"].as_str().unwrap_or("?"));
        println!("{:>13} {:>13} {:>13}", "A", "B", "ERROR");
        for r in head["rows"].as_array().into_iter().flatten() {
            println!(
                "{:>13.6} {:>13.6} {:>13.6}",
                r["a"].as_f64().unwrap_or(0.0),
                r["b"].as_f64().unwrap_or(0.0),
                r["error"].as_f64().unwrap_or(0.0)
            );
        }
    }
    println!();
    if v["identical"].as_bool().unwrap_or(false) {
        kv("result", "identical — every decoded value agrees");
    } else {
        kv(
            "result",
            format!(
                "{} of {} values differ · rms {:.6} · max {:.6}",
                v["changed_values"],
                v["total_weights"],
                v["rms_error"].as_f64().unwrap_or(0.0),
                v["max_error"].as_f64().unwrap_or(0.0)
            ),
        );
    }
}

fn render_represent(v: &Value) {
    kv("encoding", v["encoding"].as_str().unwrap_or("?"));
    kv("map", v["map"].as_str().unwrap_or("?"));
    kv("out", v["out"].as_str().unwrap_or("?"));
    for c in v["compiled"].as_array().into_iter().flatten() {
        println!();
        kv("compiled", c["object"].as_str().unwrap_or("?"));
        kv(
            "tensors",
            format!(
                "{} re-encoded · {} carried verbatim",
                c["compiled_tensors"], c["carried_tensors"]
            ),
        );
        kv(
            "bytes",
            format!(
                "{} → {} ({:.2}× smaller)",
                c["source_bytes"],
                c["compiled_bytes"],
                c["compression"].as_f64().unwrap_or(0.0)
            ),
        );
    }
    for p in v["preserved"].as_array().into_iter().flatten() {
        println!();
        kv(
            "preserved",
            format!(
                "{} — wholly at {} ({} bytes)",
                p["object"].as_str().unwrap_or("?"),
                p["encoding"].as_str().unwrap_or("?"),
                p["bytes"]
            ),
        );
    }
    println!();
    kv(
        "linked",
        format!("{} segment(s) untouched", v["linked_segments"]),
    );
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
    if let Command::Update { check } = &cli.command {
        return match update::run(*check) {
            Ok(line) => {
                println!("{line}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("vindex: {e}");
                ExitCode::from(2)
            }
        };
    }
    let result = match &cli.command {
        Command::Inspect { container } => vindex_cli::inspect_facts(container),
        Command::Describe {
            container,
            address,
            values,
            peek,
        } => vindex_cli::describe_facts(container, address, *values, peek.as_deref()),
        Command::Representations { container } => vindex_cli::representations_facts(container),
        Command::Layers { container } => vindex_cli::layers_facts(container),
        Command::Diff {
            container,
            a,
            b,
            address,
            values,
            tensor,
        } => vindex_cli::diff_facts(container, a, b, address, *values, tensor.as_deref()),
        Command::Represent {
            container,
            out,
            encoding,
        } => vindex_cli::represent_facts(container, out, encoding),
        Command::Precision { container, matrix } => {
            if *matrix {
                vindex_cli::precision_matrix_facts(container)
            } else {
                vindex_cli::precision_facts(container)
            }
        }
        Command::Verify { container } => vindex_cli::verify_facts(container),
        Command::Export { container, out } => vindex_cli::export_facts(container, out),
        Command::Update { .. } => unreachable!("handled above"),
    };
    match result {
        Ok(v) => {
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            } else {
                match &cli.command {
                    Command::Inspect { .. } => render_inspect(&v),
                    Command::Export { .. } => render_export(&v),
                    Command::Describe { .. } => render_describe(&v),
                    Command::Representations { .. } => render_representations(&v),
                    Command::Layers { .. } => render_layers(&v),
                    Command::Diff { .. } => render_diff(&v),
                    Command::Represent { .. } => render_represent(&v),
                    Command::Precision { matrix, .. } => {
                        if *matrix {
                            render_precision_matrix(&v)
                        } else {
                            render_precision(&v)
                        }
                    }
                    Command::Verify { .. } => render_verify(&v),
                    Command::Update { .. } => unreachable!("handled above"),
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
