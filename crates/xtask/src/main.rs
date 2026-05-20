mod audit;
mod certify;
mod github;
mod mir_facts;
mod new_crate_detector;
mod rules;
mod status;
mod wasm_facts;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", about = "larql-to-sparql build and certification tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the wasm32 certification cascade for workspace members.
    ///
    /// Levels run in order: 1 (compile), call-graph closure, 2 (runtime
    /// confirmation), 4 (boundary map), 5/6 (mutation).  All results are
    /// reported regardless of pass/fail.  Exit code is non-zero only when a
    /// crate regresses below its claimed-level.
    WasmCertify {
        /// Certify only this crate (default: all workspace members).
        #[arg(long)]
        crate_name: Option<String>,
    },

    /// Print per-crate certification status table (reads manifests + last run).
    WasmStatus {
        /// Emit JSON instead of Markdown.
        #[arg(long)]
        json: bool,
    },

    /// Surface audit: wasm32-accessible modules, runtime-trap candidates, and
    /// the Level-4 native-only boundary map.  Always exits 0.
    WasmAudit {
        /// Audit only this crate (default: all workspace members).
        #[arg(long)]
        crate_name: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::WasmCertify { crate_name } => certify::run(crate_name.as_deref()),
        Command::WasmStatus { json } => status::run(json),
        Command::WasmAudit { crate_name } => audit::run(crate_name.as_deref()),
    }
}
