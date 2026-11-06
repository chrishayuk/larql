#
# nix/package.nix - LARQL Rust package derivation
#
# Builds the larql workspace (excluding larql-python which requires maturin).
# Provides the larql-cli binary and all library crates.
#
# Patches:
#   use-system-protoc.patch - Removes protobuf-src build dependency from
#     larql-server so it uses nixpkgs protoc instead of compiling from source.
#
{ pkgs, lib, src }:
let
  # Filter out files not needed for the build
  srcFiltered = lib.cleanSourceWith {
    inherit src;
    filter = path: type:
      let
        baseName = builtins.baseNameOf path;
        relPath = lib.removePrefix (toString src + "/") (toString path);
      in
        # Exclude nix packaging files, build artifacts, and editor config
        !(lib.hasPrefix "nix/" relPath)
        && !(lib.hasPrefix "flake" baseName)
        && !(lib.hasPrefix "result" baseName)
        && !(lib.hasPrefix "target/" relPath)
        && !(lib.hasPrefix ".git/" relPath);
  };
in
pkgs.rustPlatform.buildRustPackage {
  pname = "larql";
  version = "0.1.0";
  src = srcFiltered;

  cargoHash = "sha256-UEFD9S0jgpD4+x1LaXgIzp0OZvAcVQifqcBa8hFVp6o=";

  # No cargoPatches: `crates/*/build.rs` already honours an explicit
  # PROTOC (set below), so the historical use-system-protoc.patch has
  # nothing left to do. See the note under `cmake` in nativeBuildInputs.

  nativeBuildInputs = with pkgs; [
    pkg-config
    protobuf   # provides protoc for tonic-build
    cmake      # needed by protobuf-src build script (still in Cargo.lock as transitive dep)
  ];

  buildInputs = with pkgs; [
    openssl
  ] ++ lib.optionals stdenv.hostPlatform.isLinux [
    openblas
  ] ++ lib.optionals stdenv.hostPlatform.isDarwin (with darwin.apple_sdk.frameworks; [
    Accelerate
    Security
    SystemConfiguration
  ]);

  # Point tonic-build to nixpkgs protoc
  PROTOC = "${pkgs.protobuf}/bin/protoc";

  # utoipa-swagger-ui's build script otherwise tries to `curl` the Swagger
  # UI dist tarball from the network, which cannot work in the Nix sandbox
  # (no network, and no curl in the build inputs). Hand it a pre-fetched
  # zip via the env var the build script documents, so it copies dist/
  # out of a fixed-output derivation instead of fetching.
  #
  # The version is pinned to what the crate wants; if a utoipa-swagger-ui
  # bump changes it, the build fails loudly with the new URL in the log
  # rather than silently serving a mismatched UI.
  SWAGGER_UI_DOWNLOAD_URL =
    let
      swaggerUi = pkgs.fetchurl {
        url = "https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.17.14.zip";
        hash = "sha256-SBJE0IEgl7Efuu73n3HZQrFxYX+cn5UU5jrL4T5xzNw=";
      };
    in
    "file://${swaggerUi}";

  # Point openblas-src to system library
  OPENBLAS_LIB_DIR = lib.optionalString pkgs.stdenv.hostPlatform.isLinux
    "${pkgs.openblas}/lib";

  # Exclude larql-python (requires maturin, handled separately in dev shell)
  # Patch sets larql-cli default = [] (removes metal); re-enable on Darwin
  cargoBuildFlags = [
    "--workspace"
    "--exclude" "larql-python"
  ] ++ lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
    "--features" "larql-cli/metal"
  ];

  cargoTestFlags = [
    "--workspace"
    "--exclude" "larql-python"
  ] ++ lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
    "--features" "larql-cli/metal"
  ];

  # Skip tests — upstream has a pre-existing compile error in test_architectures
  # (missing fields packed_byte_ranges and packed_mmaps in ModelWeights)
  doCheck = false;

  meta = with lib; {
    description = "Query engine for transformer model weights — the model is the database";
    homepage = "https://github.com/chrishayuk/chuk-larql-rs";
    license = licenses.asl20;
    mainProgram = "larql";
  };
}
