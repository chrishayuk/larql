//! `vindex3 token-bank` — the sealed corpus of MEASURE-PLAN-1
//! (`docs/measure-plan-1.md`).
//!
//! `export` tokenises a prompt file (Q-BANK-1's `prompts.json` shape) with a
//! container's own tokenizer, so the bank is bound to that model from the
//! start. `check` opens a bank and reads every sample, confirming each seal
//! and that the bank belongs to a container's tokenizer. The format and every
//! refusal live in `larql_vindex::format::vindex3::represent::token_bank`;
//! this file only names the arguments.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use larql_vindex::format::vindex3::represent::token_bank::{
    container_tokenizer_sha256, export, TokenBank, TOKENIZER_FILE,
};

/// `run_bank.py`'s default truncation cap, so an export with no `--max-tokens`
/// yields the streams that path already scores.
pub(super) const DEFAULT_MAX_TOKENS: usize = 128;

#[derive(Args)]
pub struct TokenBankArgs {
    #[command(subcommand)]
    pub command: TokenBankCommand,
}

#[derive(Subcommand)]
pub enum TokenBankCommand {
    /// Tokenise a prompt file with a container's tokenizer into a new bank.
    Export(ExportArgs),
    /// Read every sample of a bank against its seals, and check that it
    /// belongs to a container's tokenizer.
    Check(CheckArgs),
}

#[derive(Args)]
pub struct ExportArgs {
    /// Container whose `tokenizer.json` tokenises the prompts.
    pub container: PathBuf,
    /// Prompt file, e.g. `bench/prompts/quality-bank-1/prompts.json`.
    #[arg(long)]
    pub prompts: PathBuf,
    /// Truncation cap, in ids.
    #[arg(long, default_value_t = DEFAULT_MAX_TOKENS)]
    pub max_tokens: usize,
    /// New bank directory. Refused if it exists.
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args)]
pub struct CheckArgs {
    /// Bank directory.
    pub bank: PathBuf,
    /// Container the bank must belong to.
    #[arg(long)]
    pub container: PathBuf,
}

pub fn run(args: TokenBankArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        TokenBankCommand::Export(a) => {
            run_export(&a.container, &a.prompts, a.max_tokens, &a.output)
        }
        TokenBankCommand::Check(a) => run_check(&a.bank, &a.container),
    }
}

pub(super) fn run_export(
    container: &Path,
    prompts: &Path,
    max_tokens: usize,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = export(prompts, &container.join(TOKENIZER_FILE), max_tokens, output)?;
    let positions: usize = manifest.samples.iter().map(|s| s.tokens).sum();
    println!(
        "bank {} ({}): {} samples, {positions} ids, cap {max_tokens}",
        manifest.bank_id,
        manifest.prompts.bank,
        manifest.samples.len()
    );
    println!("tokenizer {}", manifest.tokenizer_sha256);
    println!("-> {}", output.display());
    Ok(())
}

pub(super) fn run_check(bank: &Path, container: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let opened = TokenBank::open(bank)?;
    opened.check_tokenizer(&container_tokenizer_sha256(container)?)?;
    let mut positions = 0;
    for i in 0..opened.sample_count() {
        positions += opened.read(i)?.len();
    }
    println!(
        "bank {}: {} samples, {positions} ids, every seal read; belongs to {}",
        opened.manifest().bank_id,
        opened.sample_count(),
        container.display()
    );
    Ok(())
}
