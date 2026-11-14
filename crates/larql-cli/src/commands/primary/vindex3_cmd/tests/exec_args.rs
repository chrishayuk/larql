//! `vindex3 exec` argument relationships that clap enforces.

use super::*;
use clap::Parser;

#[derive(Parser)]
struct ExecCli {
    #[command(flatten)]
    exec: ExecArgs,
}

fn parse(extra: &[&str]) -> Result<ExecCli, clap::Error> {
    let base = ["exec", "some.vindex3", "--tokens", "1,2,3"];
    ExecCli::try_parse_from(base.iter().chain(extra))
}

/// `--dump-dir` is only read by the `--bank` branch. Alone it used to
/// be accepted and silently ignored, so a run looked like it had
/// written dumps when it had written none.
#[test]
fn dump_dir_without_bank_is_refused() {
    let err = parse(&["--dump-dir", "out"])
        .err()
        .expect("--dump-dir without --bank must be refused");
    assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    assert!(err.to_string().contains("--bank"), "{err}");
}

#[test]
fn dump_dir_with_bank_parses() {
    let cli = parse(&["--bank", "bank.jsonl", "--dump-dir", "out"]).expect("valid pair");
    assert_eq!(
        cli.exec.dump_dir.as_deref(),
        Some(std::path::Path::new("out"))
    );
    assert_eq!(
        cli.exec.bank.as_deref(),
        Some(std::path::Path::new("bank.jsonl"))
    );
}
