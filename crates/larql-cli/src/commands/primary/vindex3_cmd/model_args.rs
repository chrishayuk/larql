//! Which `larql vindex3` arguments name an existing model, resolved once
//! before any verb runs.
//!
//! Every such argument accepts what `larql run` accepts — a directory,
//! `hf://owner/name`, `owner/name` or a cache shorthand — through
//! [`cache::resolve_installed_model`], which never downloads (see its
//! module docs for the rule). An existing path is returned unchanged, so a
//! verb sees exactly the path it saw before.
//!
//! The match below is exhaustive on purpose: a new verb must say which of
//! its arguments are models, and a verb with none says so explicitly.
//! Checkpoint artifacts (`plan`, `encode`, `verify`'s sources), token
//! banks, records and every output path are not models and stay paths.

use std::path::{Path, PathBuf};

use super::auto_rep::AutoRepCommand;
use super::token_bank::TokenBankCommand;
use super::Vindex3Command;
use crate::commands::primary::cache;

type BoxErr = Box<dyn std::error::Error>;

/// Maps one model argument to the directory the verb reads.
pub type ModelResolver<'a> = dyn Fn(&Path) -> Result<PathBuf, BoxErr> + 'a;

/// Resolve every model argument of `cmd` in place with the installed-only
/// resolver.
pub fn resolve_installed(cmd: &mut Vindex3Command) -> Result<(), BoxErr> {
    resolve_with(cmd, &cache::resolve_installed_model)
}

/// Resolve every model argument of `cmd` in place with `resolve`.
pub fn resolve_with(cmd: &mut Vindex3Command, resolve: &ModelResolver<'_>) -> Result<(), BoxErr> {
    for slot in model_slots(cmd) {
        *slot = resolve(slot)?;
    }
    Ok(())
}

/// The arguments of `cmd` that name an existing model.
pub fn model_slots(cmd: &mut Vindex3Command) -> Vec<&mut PathBuf> {
    match cmd {
        Vindex3Command::Plan(_) | Vindex3Command::Encode(_) => Vec::new(),
        Vindex3Command::Inspect(a) => vec![&mut a.container],
        Vindex3Command::References(a) => vec![&mut a.container],
        Vindex3Command::Verify(a) => vec![&mut a.container],
        Vindex3Command::Ops(a) => vec![&mut a.container],
        Vindex3Command::Exec(a) => vec![&mut a.container],
        Vindex3Command::Represent(a) => vec![&mut a.container],
        Vindex3Command::Observe(a) => vec![&mut a.container],
        Vindex3Command::Sensitivity(a) => vec![&mut a.container],
        Vindex3Command::Consequence(a) => vec![&mut a.container],
        Vindex3Command::TokenBank(a) => match &mut a.command {
            TokenBankCommand::Export(e) => vec![&mut e.container],
            TokenBankCommand::Import(i) => vec![&mut i.container],
            TokenBankCommand::Check(c) => vec![&mut c.container],
        },
        Vindex3Command::Measure(a) => vec![&mut a.reference, &mut a.candidate],
        Vindex3Command::AutoRep(a) => match &mut a.command {
            AutoRepCommand::Init(i) => vec![&mut i.container],
            AutoRepCommand::Run(r) => vec![&mut r.source],
        },
    }
}
