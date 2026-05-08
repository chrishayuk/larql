#
# nix/shell.nix - LARQL development shell
#
# Provides build dependencies plus Rust tooling, Python bindings support,
# and debugging tools.
#
{ pkgs, lib, larql }:
let
  banner = import ./banner.nix;

  # Python with packages needed for larql-python development and testing
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    pytest
    numpy
  ]);
in
pkgs.mkShell {
  inputsFrom = [ larql ];

  packages = with pkgs; [
    # Rust development tools
    rust-analyzer
    clippy
    rustfmt
    cargo-watch

    # Python bindings (larql-python via maturin)
    pythonEnv
    maturin
  ] ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux [
    gdb
    valgrind
  ] ++ lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
    lldb
  ];

  shellHook = banner;
}
