#
# flake.nix - LARQL Nix packaging
#
# Quick Start:
#   nix build                         # Build larql-cli binary
#   nix develop                       # Development shell
#   nix flake show                    # List all outputs
#
# Demo (models fetched via git-lfs, vindexes built at nix-build time):
#   nix run .#demo-list               # List available models
#   nix run .#demo                    # Walk default model (gemma4-4b)
#   nix run .#demo-<key>              # Walk a specific model
#   nix run .#demo-info               # Show GGUF metadata for local models
#   nix build .#vindex-<key>          # Build a vindex from a model
#
# OCI Containers (Linux only):
#   nix build .#container             # larql-server image
#   nix build .#container-cli         # larql CLI image
#   docker load < result
#
# See also: ./nix/README.md
#
{
  description = "LARQL - query engine for transformer model weights";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        nixDir = ./nix;
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;

        # Import modular package definition
        larql = import (nixDir + "/package.nix") { inherit pkgs lib; src = self; };

        # Import OCI container images (Linux only)
        containers = lib.optionalAttrs pkgs.stdenv.isLinux (
          import (nixDir + "/container.nix") { inherit pkgs lib larql; }
        );

        # Import demo (model fetch + vindex extraction + apps)
        demo = import (nixDir + "/demo.nix") { inherit pkgs lib larql; };
      in
      {
        packages = {
          default = larql;
          inherit larql;
        } // lib.optionalAttrs pkgs.stdenv.isLinux {
          container = containers.server;
          container-cli = containers.cli;
        }
        # Demo packages (model + vindex)
        // demo.packages;

        # Demo and utility apps
        apps = demo.apps;

        # Import modular development shell
        devShells.default = import (nixDir + "/shell.nix") { inherit pkgs lib larql; };
      }
    );
}
