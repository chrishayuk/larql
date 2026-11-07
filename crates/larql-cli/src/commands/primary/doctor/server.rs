//! The `larql-server` executable `larql serve` runs, found the way
//! `serve` finds it, and whether it is this release's.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use super::super::serve_cmd::{server_file_name, sibling_server, SERVER_BINARY};
use super::report::{Check, Group};
use super::units::display_path;

/// Where the server executable was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ServerLocation {
    /// Beside the running `larql` — what `serve` prefers.
    Sibling(PathBuf),
    /// On `PATH` — what `serve` falls back to.
    OnPath(PathBuf),
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServerFacts {
    pub location: ServerLocation,
    /// `--version`'s answer, when the server was found.
    pub version: Option<Result<String, String>>,
    /// The version the server should report (this `larql`'s).
    pub expected_version: String,
    pub home: Option<PathBuf>,
}

/// The first `file_name` in the directories of `path_var` (a `PATH`
/// value), the lookup the OS makes for `serve`'s bare-name fallback.
pub(crate) fn find_on_path(file_name: &str, path_var: Option<&OsStr>) -> Option<PathBuf> {
    std::env::split_paths(path_var?)
        .map(|dir| dir.join(file_name))
        .find(|candidate| candidate.is_file())
}

/// Find the server as `larql serve` would: beside `exe`, else on `PATH`.
pub(crate) fn locate(exe: Option<&Path>, path_var: Option<&OsStr>) -> ServerLocation {
    if let Some(path) = sibling_server(exe) {
        return ServerLocation::Sibling(path);
    }
    find_on_path(&server_file_name(), path_var)
        .map(ServerLocation::OnPath)
        .unwrap_or(ServerLocation::Missing)
}

/// The version in a clap `--version` line (`larql-server 0.2.1`).
pub(crate) fn parse_version(stdout: &str) -> Option<String> {
    stdout
        .lines()
        .next()?
        .trim()
        .strip_prefix(SERVER_BINARY)?
        .split_whitespace()
        .next()
        .map(str::to_string)
}

pub(crate) fn checks(s: &ServerFacts) -> Vec<Check> {
    let (path, how) = match &s.location {
        ServerLocation::Sibling(p) => (p, "beside larql"),
        ServerLocation::OnPath(p) => (p, "on PATH"),
        ServerLocation::Missing => {
            return vec![Check::fail(
                Group::Server,
                SERVER_BINARY,
                "not found beside larql or on PATH",
                "`larql serve` needs it: install both executables from the release archive, \
                 or `cargo install --path crates/larql-server`",
            )]
        }
    };
    let shown = display_path(path, s.home.as_deref());
    let check = match &s.version {
        Some(Ok(v)) if *v == s.expected_version => Check::ok(
            Group::Server,
            SERVER_BINARY,
            format!("{v} ({how}: {shown})"),
        ),
        Some(Ok(v)) => Check::warn(
            Group::Server,
            SERVER_BINARY,
            format!("{v} ({how}: {shown}), but larql is {}", s.expected_version),
            "reinstall so larql and larql-server come from the same release",
        ),
        Some(Err(why)) => Check::warn(
            Group::Server,
            SERVER_BINARY,
            format!("found ({how}: {shown}) but `--version` failed: {why}"),
            "reinstall larql-server from the same release as larql",
        ),
        None => Check::ok(Group::Server, SERVER_BINARY, format!("{how}: {shown}")),
    };
    vec![check]
}
