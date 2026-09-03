//! **`larql optimizer-mcp` — the read-only MCP facade.**
//!
//! Stage 4 of the physical optimiser. The whole command is: read one
//! stored search record, wrap it in the view layer, and answer
//! questions about it on stdin and stdout until the client goes away.
//!
//! ```text
//! snapshot.json  →  SearchSnapshot  →  OptimizerView  →  seven tools
//!    facts            deserialise        derive            render
//! ```
//!
//! The record is opened read-only and never written back. Nothing in
//! this module computes: [`server`] dispatches and serialises, and every
//! conclusion in every answer was drawn by the optimiser just now, from
//! stored facts, under the semantics the record itself declares.

pub mod protocol;
pub mod server;
pub mod tools;

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Args;
use larql_vindex::format::vindex3::represent::state::snapshot::SearchSnapshot;
use larql_vindex::format::vindex3::represent::view::OptimizerView;

use server::Server;

#[derive(Args, Debug)]
pub struct OptimizerMcpArgs {
    /// The stored search record to serve. Opened read-only.
    #[arg(long, value_name = "PATH")]
    pub snapshot: PathBuf,

    /// Print the tool declarations and exit, without serving. Useful
    /// for checking what a client would see.
    #[arg(long)]
    pub list_tools: bool,
}

pub fn run(args: OptimizerMcpArgs) -> Result<(), Box<dyn std::error::Error>> {
    dispatch(
        &args,
        BufReader::new(std::io::stdin().lock()),
        std::io::stdout().lock(),
    )
}

/// The verb, with its input and output supplied.
///
/// Separate from [`run`] so a test can drive the whole command — record
/// on disk, frames in, frames out — rather than only the half of it
/// below the transport. What is left in `run` is the stdio wiring, and
/// nothing else.
fn dispatch(
    args: &OptimizerMcpArgs,
    input: impl BufRead,
    output: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    match args.list_tools {
        true => declare(output),
        false => serve(&args.snapshot, input, output),
    }
}

/// Print the tool declarations and serve nothing.
///
/// Deliberately does not open the record: an operator should be able to
/// see what a client would see without having a search to point at.
fn declare(mut output: impl Write) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(output, "{}", serde_json::to_string_pretty(&tools::all())?)?;
    Ok(())
}

/// Open one record and answer questions about it until the input ends.
fn serve(
    path: &std::path::Path,
    input: impl BufRead,
    output: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = load(path)?;
    Server::new(OptimizerView::new(&snapshot)).serve(input, output)?;
    Ok(())
}

/// Read a record and check its schema before serving a word of it.
///
/// A snapshot whose schema this build does not know is refused rather
/// than served partially: a reader that does not know the schema string
/// should not trust its reading of anything under it.
fn load(path: &std::path::Path) -> Result<SearchSnapshot, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("could not open the record at {}: {e}", path.display()))?;
    let snapshot: SearchSnapshot = serde_json::from_reader(BufReader::new(file))
        .map_err(|e| format!("could not read the record at {}: {e}", path.display()))?;
    snapshot.check_schema()?;
    Ok(snapshot)
}

#[cfg(test)]
mod tests;
