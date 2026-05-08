#
# nix/demo.nix - LARQL demo apps
#
# Fetches models from HuggingFace as Nix derivations (git-lfs),
# extracts vindexes at build time, and provides `nix run` apps to query them.
#
# Everything is cached in /nix/store — no runtime downloads needed.
#
# Apps:
#   nix run .#demo-list                  List available models
#   nix run .#demo                       Walk default model (gemma4-4b)
#   nix run .#demo-<model>               Walk a specific model
#   nix run .#repl-<model>               Interactive REPL with model pre-loaded
#   nix run .#demo-info                  Show GGUF metadata for local models
#
# Packages:
#   nix build .#model-<name>             Fetch a model from HuggingFace
#   nix build .#vindex-<name>            Extract a vindex from a model
#
{ pkgs, lib, larql }:
let
  models = import ./models.nix { inherit pkgs; };

  # Default model for `nix run .#demo`
  defaultModel = "gemma4-4b";
  defaultPrompt = "The capital of France is";

  # Build a vindex derivation from a model
  mkVindex = name: model: pkgs.runCommand "larql-vindex-${name}" {
    nativeBuildInputs = [ larql ];
  } ''
    mkdir -p $out
    larql extract-index ${model.src} -o $out
  '';

  mkApp = name: script: {
    type = "app";
    program = "${pkgs.writeShellScript name script}";
  };

  # Generate per-model packages and apps
  modelNames = builtins.attrNames models;

  vindexes = lib.mapAttrs mkVindex models;

  modelPackages = lib.foldl' (acc: name: acc // {
    "model-${name}" = models.${name}.src;
    "vindex-${name}" = vindexes.${name};
  }) {} modelNames;

  modelApps = lib.foldl' (acc: name:
    let
      model = models.${name};
      vindex = vindexes.${name};
    in acc // {
      "demo-${name}" = mkApp "larql-demo-${name}" ''
        set -euo pipefail
        PROMPT="''${1:-${defaultPrompt}}"

        echo "=== LARQL Demo: ${model.name} ==="
        echo "Vindex: ${vindex}"
        echo "Prompt: $PROMPT"
        echo ""

        ${larql}/bin/larql walk --index ${vindex} -p "$PROMPT" -k 10
      '';

      "repl-${name}" = mkApp "larql-repl-${name}" ''
        echo "=== LARQL REPL: ${model.name} ==="
        echo ""
        exec ${pkgs.expect}/bin/expect -c '
          spawn ${larql}/bin/larql repl
          expect "larql>"
          send "USE \"${vindex}\";\r"
          interact
        '
      '';
    }
  ) {} modelNames;

  # Model listing (sorted by size for display)
  modelList = lib.concatStringsSep "\n" (map (name:
    let m = models.${name};
    in "  ${name}|${m.name}|${m.size}"
  ) modelNames);

in
{
  packages = modelPackages;

  apps = modelApps // {
    # Default demo (uses default model)
    demo = mkApp "larql-demo" ''
      set -euo pipefail
      PROMPT="''${1:-${defaultPrompt}}"

      echo "=== LARQL Demo: ${models.${defaultModel}.name} ==="
      echo "Vindex: ${vindexes.${defaultModel}}"
      echo "Prompt: $PROMPT"
      echo ""

      ${larql}/bin/larql walk --index ${vindexes.${defaultModel}} -p "$PROMPT" -k 10
    '';

    # Default REPL (uses default model)
    repl = mkApp "larql-repl" ''
      echo "=== LARQL REPL: ${models.${defaultModel}.name} ==="
      echo ""
      exec ${pkgs.expect}/bin/expect -c '
        spawn ${larql}/bin/larql repl
        expect "larql>"
        send "USE \"${vindexes.${defaultModel}}\";\r"
        interact
      '
    '';

    # List available models
    demo-list = mkApp "larql-demo-list" ''
      echo "Available LARQL demo models:"
      echo ""
      printf "  %-20s %-35s %s\n" "KEY" "MODEL" "SIZE"
      printf "  %-20s %-35s %s\n" "---" "-----" "----"
      ${lib.concatStringsSep "\n" (map (name:
        let m = models.${name};
        in ''printf "  %-20s %-35s %s\n" "${name}" "${m.name}" "${m.size}"''
      ) modelNames)}
      echo ""
      echo "Run a model:"
      echo "  nix run .#demo                          # default (${defaultModel})"
      echo "  nix run .#demo-<key>                    # specific model"
      echo "  nix run .#demo-<key> -- \"your prompt\"   # custom prompt"
      echo ""
      echo "Interactive REPL:"
      echo "  nix run .#repl                          # default (${defaultModel})"
      echo "  nix run .#repl-<key>                    # specific model"
      echo ""
      echo "Build a vindex:"
      echo "  nix build .#vindex-<key>"
    '';

    # Show GGUF file metadata (works with local llama.cpp cache)
    demo-info = mkApp "larql-demo-info" ''
      set -euo pipefail
      GGUF_DIR="$HOME/.cache/llama.cpp"

      if [ -n "''${1:-}" ]; then
        ${larql}/bin/larql convert gguf-info "$1"
        exit 0
      fi

      if [ ! -d "$GGUF_DIR" ]; then
        echo "No GGUF files found in $GGUF_DIR"
        echo "Usage: nix run .#demo-info -- /path/to/model.gguf"
        exit 1
      fi

      echo "Available GGUF models in $GGUF_DIR:"
      echo ""
      for f in "$GGUF_DIR"/*.gguf; do
        size=$(du -h "$f" | cut -f1)
        name=$(basename "$f")
        echo "  $name ($size)"
      done
      echo ""
      echo "Run with: nix run .#demo-info -- <path>"
    '';
  };
}
