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

  cargoHash = "sha256-6mvESL1m5sZZCx8YdArgTkwNnGHRQz3RPub/hVYelqg=";

  # Use system protoc instead of bundled protobuf-src
  cargoPatches = [
    ./patches/use-system-protoc.patch
  ];

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
