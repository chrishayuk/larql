use super::*;
use crate::{Cli, Commands};
use clap::Parser;

fn forwarded(argv: &[&str]) -> Vec<String> {
    let cli = Cli::try_parse_from(["larql", "serve"].iter().chain(argv)).unwrap();
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve")
    };
    serve_command_args(&args).unwrap()
}

/// Server flags reach the server exactly as typed, including ones the
/// old mirror never declared and pairs it could not express.
#[test]
fn flags_are_forwarded_verbatim() {
    let argv = [
        "--dir",
        "/models",
        "--hnsw",
        "--warmup-hnsw",
        "--v3-backend",
        "metal",
        "--session-ttl-secs=60",
        "--port",
        "9000",
    ];
    assert_eq!(forwarded(&argv), argv);
}

/// Nothing is added: the server's defaults apply, not copies of them.
#[test]
fn no_arguments_forward_nothing() {
    assert!(forwarded(&[]).is_empty());
}

/// `--help` and `-h` go to the server, which owns the flag list.
#[test]
fn help_is_forwarded_to_the_server() {
    assert_eq!(forwarded(&["--help"]), ["--help"]);
    assert_eq!(forwarded(&["-h"]), ["-h"]);
}

/// A leading non-flag argument is the model, and it is resolved.
#[test]
fn a_leading_model_is_resolved_and_the_rest_forwarded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap();
    let out = forwarded(&[path, "--port", "9000"]);
    assert_eq!(out[1..], ["--port", "9000"]);
    assert_eq!(
        out[0],
        super::super::serve_resolve::resolve_serve_target(path).unwrap()
    );
}

/// A resolution failure is an error, not a silently unresolved name.
#[test]
fn an_unresolvable_model_is_an_error() {
    let cli = Cli::try_parse_from([
        "larql",
        "serve",
        "definitely-not-a-cached-vindex-serve-test",
    ])
    .unwrap();
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve")
    };
    assert!(serve_command_args(&args).is_err());
}

/// Only a leading argument is treated as the model; later non-flag
/// arguments are flag values and are never resolved.
#[test]
fn only_the_first_argument_can_be_the_model() {
    let argv = ["--layers", "0-14", "--join", "grpc://router:50052"];
    assert_eq!(forwarded(&argv), argv);
}

fn strings(argv: &[&str]) -> Vec<String> {
    argv.iter().map(|s| s.to_string()).collect()
}

fn fake_resolve(_: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok("/cache/local/qwen3-0.6b.vindex".to_string())
}

/// A model served by name is also forwarded as `--model-alias <name>`, so
/// a request can use the name it was served under.
#[test]
fn a_named_model_is_forwarded_as_its_own_alias() {
    let out = forward_args(&strings(&["qwen3-0.6b", "--port", "9000"]), fake_resolve).unwrap();
    assert_eq!(
        out,
        [
            "/cache/local/qwen3-0.6b.vindex",
            "--model-alias",
            "qwen3-0.6b",
            "--port",
            "9000"
        ]
    );
}

/// An explicit `--model-alias`, in either spelling, is the caller's choice
/// and is not doubled.
#[test]
fn an_explicit_alias_is_not_overridden() {
    for argv in [
        &["qwen3-0.6b", "--model-alias", "mine"][..],
        &["qwen3-0.6b", "--model-alias=mine"][..],
    ] {
        let out = forward_args(&strings(argv), fake_resolve).unwrap();
        assert_eq!(out[0], "/cache/local/qwen3-0.6b.vindex");
        assert_eq!(out[1..], argv[1..]);
    }
}

/// A directory path is not a name: nothing is aliased, and a flag that
/// merely starts with the alias flag's spelling is not mistaken for it.
#[test]
fn a_path_gets_no_alias() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap();
    let out = forward_args(&strings(&[path]), |p| Ok(p.to_string())).unwrap();
    assert_eq!(out, [path]);
    assert!(!has_explicit_alias(&strings(&["--model-aliasx"])));
}

// ── Server executable lookup ────────────────────────────────────────

#[test]
fn the_server_beside_larql_is_preferred() {
    let dir = tempfile::tempdir().unwrap();
    let exe = dir.path().join("larql");
    let server = dir.path().join(server_file_name());
    std::fs::write(&server, b"").unwrap();
    assert_eq!(sibling_server(Some(&exe)), Some(server.clone()));
    assert_eq!(server_program(Some(&exe)), server.to_string_lossy());
}

#[test]
fn without_a_sibling_the_os_finds_the_server_on_path() {
    let dir = tempfile::tempdir().unwrap();
    let exe = dir.path().join("larql");
    assert_eq!(sibling_server(Some(&exe)), None);
    assert_eq!(sibling_server(None), None);
    assert_eq!(server_program(Some(&exe)), SERVER_BINARY);
}

/// The Windows rule, proven on any host: the sibling is `larql-server.exe`,
/// and a suffix-less file beside `larql.exe` is not it.
#[test]
fn a_windows_sibling_is_found_by_its_exe_name() {
    let dir = tempfile::tempdir().unwrap();
    let exe = dir.path().join("larql.exe");
    std::fs::write(dir.path().join("larql-server"), b"").unwrap();
    assert_eq!(
        sibling_server_named(Some(&exe), &server_file_name_with(".exe")),
        None
    );
    let server = dir.path().join("larql-server.exe");
    std::fs::write(&server, b"").unwrap();
    assert_eq!(
        sibling_server_named(Some(&exe), &server_file_name_with(".exe")),
        Some(server)
    );
    assert_eq!(server_file_name_with(".exe"), "larql-server.exe");
    assert_eq!(server_file_name_with(""), "larql-server");
}

#[test]
fn the_server_file_name_carries_the_platform_suffix() {
    assert_eq!(
        server_file_name(),
        format!("larql-server{}", std::env::consts::EXE_SUFFIX)
    );
}
