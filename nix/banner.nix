#
# nix/banner.nix - LARQL development shell banner
#
# ASCII art and welcome message displayed on `nix develop`.
#
''
  echo ""
  echo "   ╦   ╔═╗ ╦═╗ ╔═╗ ╦"
  echo "   ║   ╠═╣ ╠╦╝ ║═╬╗║"
  echo "   ╩═╝ ╩ ╩ ╩╚═ ╚═╝╚╩═╝"
  echo "   Lazarus Query Language v0.1"
  echo ""
  echo "  Query engine for transformer model weights"
  echo "  The model is the database."
  echo ""
  echo "  cargo build --release     Build optimised binary"
  echo "  cargo test                Run all workspace tests"
  echo "  make ci                   fmt-check + clippy + test"
  echo "  nix build                 Build via Nix"
  echo ""
  echo "  Demo (models fetched + vindexes built by Nix):"
  echo "  nix run .#demo-list       List available models"
  echo "  nix run .#demo            Walk default model (gemma4-4b)"
  echo '  nix run .#demo-<key>      Walk a specific model'
  echo "  nix run .#repl            Interactive REPL (gemma4-4b)"
  echo '  nix run .#repl-<key>      REPL with a specific model'
  echo ""
''
