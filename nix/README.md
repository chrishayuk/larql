# LARQL — Nix Flake Development Environment

This document explains how to use **Nix** to set up a reproducible development environment for the **LARQL** project.

## Goals

The goals of using Nix in this repository are to:
- **Simplify onboarding** — make it easy to get started with LARQL on any Linux or macOS system
- **Improve reproducibility** — ensure consistent build environments across developers (no more "it worked on my machine")
- **Reduce setup friction** — eliminate dependency conflicts and version mismatches (Rust toolchain, OpenBLAS, protobuf, etc.)

Feedback and pull requests are welcome. If we're missing a tool, please open an issue or PR. See `nix/package.nix` for package definitions and `nix/shell.nix` for development tools.

---

## Table of Contents

- [LARQL — Nix Flake Development Environment](#larql--nix-flake-development-environment)
  - [Goals](#goals)
  - [Background](#background)
  - [Quick Start](#quick-start)
    - [1. Install Nix](#1-install-nix)
    - [2. Enter Development Environment](#2-enter-development-environment)
    - [3. First Run Considerations](#3-first-run-considerations)
    - [4. Build and Test](#4-build-and-test)
  - [File Structure](#file-structure)
  - [Common Commands](#common-commands)
  - [Python Bindings](#python-bindings)
  - [Notes](#notes)
  - [Troubleshooting](#troubleshooting)

---

## Background

[Nix](https://nixos.org) is a package manager for Linux and other Unix-like systems that provides **reproducible, isolated** environments. By tracking all dependencies and hashing their content, it ensures every developer uses the same versions of every package.

### What This Repository Provides

This repository includes `flake.nix`, `flake.lock`, and modular Nix files in `nix/`:

- **`flake.nix`** — Main entry point; imports modules from `nix/` and wires up the package and development shell
- **`flake.lock`** — Pins exact versions so all developers use **identical** inputs
- **`nix/package.nix`** — Rust package derivation (`rustPlatform.buildRustPackage`)
- **`nix/shell.nix`** — Development shell configuration (Rust tools, Python, debuggers)
- **`nix/README.md`** — This file

Running `nix develop` spawns a shell with the correct Rust toolchain, system libraries (OpenBLAS, protobuf, OpenSSL), and development tools configured for you.

### Current Toolchain Versions

The Nix development environment provides the following (from nixpkgs unstable):

| Package | Purpose |
|---------|---------|
| Rust (~1.92) | Compiler and cargo (well above `rust-version = "1.75"` minimum) |
| OpenBLAS | BLAS backend for `ndarray` (Linux) |
| Accelerate | BLAS backend (macOS, via Apple framework) |
| protobuf | gRPC code generation for `larql-server` |
| OpenSSL | TLS for HTTP clients (`reqwest`) |
| rust-analyzer | LSP server for IDE integration |
| clippy / rustfmt | Linting and formatting |
| cargo-watch | Auto-rebuild on file changes |
| Python 3 + pytest + maturin | PyO3 bindings development |

---

## Quick Start

Install Nix if you don't already have it, then enter the development environment.

### 1. Install Nix

Choose **multi-user** (daemon) or **single-user**:

- **Multi-user install** (recommended on most distros)
  [Install Nix (multi-user)](https://nix.dev/manual/nix/2.24/installation/#multi-user)
  ```bash
  bash <(curl -L https://nixos.org/nix/install) --daemon
  ```

- **Single-user install**
  [Install Nix (single-user)](https://nix.dev/manual/nix/2.24/installation/#single-user)
  ```bash
  bash <(curl -L https://nixos.org/nix/install) --no-daemon
  ```

#### Video Tutorials

| Platform | Video |
|----------|-------|
| Ubuntu | [Installing Nix on Ubuntu](https://youtu.be/cb7BBZLhuUY) |
| Fedora | [Installing Nix on Fedora](https://youtu.be/RvaTxMa4IiY) |

### 2. Enter Development Environment

#### Enable Flakes (if needed)

If you don't have the "flakes" feature enabled, run this command:
```bash
nix --extra-experimental-features 'nix-command flakes' develop .
```

To permanently enable the Nix "flakes" feature, update `/etc/nix/nix.conf`:
```bash
test -d /etc/nix || sudo mkdir /etc/nix
echo 'experimental-features = nix-command flakes' | sudo tee -a /etc/nix/nix.conf
```

With flakes enabled, simply run:
```bash
nix develop
```

See also: [Nix Flakes Wiki](https://nixos.wiki/wiki/flakes)

### 3. First Run Considerations

On first execution, Nix will download and build all dependencies, which might take several minutes. On subsequent executions, Nix will reuse the cache in `/nix/store/` and will be essentially instantaneous.

> **Note:** Nix will not interact with any "system" packages you may already have installed. The Nix versions are isolated and will effectively "disappear" when you exit the development shell.

### 4. Build and Test

Once inside the Nix development shell:

```bash
# Build the larql CLI binary
cargo build --release

# Run all workspace tests
cargo test

# Run the full CI check (fmt + clippy + test)
make ci

# Or build via Nix directly (outside the shell)
nix build
./result/bin/larql --help
```

---

## Remote Usage (no clone needed)

Nix flakes can be run directly from GitHub — no need to clone the repository. This is useful for trying out LARQL or running demos on any machine with Nix installed.

### From the main repository

Once the Nix flake is merged into `main`:

```bash
# Build the larql binary
nix build github:chrishayuk/larql

# Enter the development shell
nix develop github:chrishayuk/larql

# List available demo models
nix run github:chrishayuk/larql#demo-list

# Run the default demo (gemma4-4b)
nix run github:chrishayuk/larql#demo

# Run a specific model demo
nix run github:chrishayuk/larql#demo-tinyllama-1b

# Launch the interactive REPL with a model
nix run github:chrishayuk/larql#repl

# Run with a custom prompt
nix run github:chrishayuk/larql#demo-qwen-15b -- "The meaning of life is"
```

### From a fork or branch

Before the PR is merged, you can run from a fork or a specific branch:

```bash
# From a fork (e.g. randomizedcoder's fork, nix branch)
nix run github:randomizedcoder/larql/nix#demo-list
nix run github:randomizedcoder/larql/nix#repl
nix develop github:randomizedcoder/larql/nix

# General syntax: github:<owner>/<repo>/<branch>#<target>
nix run github:<owner>/<repo>/<branch>#demo
```

### Pinning a specific revision

For reproducibility, you can pin to an exact commit:

```bash
nix run github:chrishayuk/larql/<commit-sha>#demo
```

> **Note:** The first run downloads and builds everything from scratch, which may take several minutes. Subsequent runs are cached in `/nix/store/` and are essentially instant.

---

## File Structure

| File | Purpose |
|------|---------|
| `flake.nix` | Main entry point — imports from `nix/`, defines packages and devShell |
| `flake.lock` | Pins nixpkgs and flake-utils to exact revisions |
| `nix/package.nix` | Rust package derivation: source filtering, native deps (OpenBLAS, protobuf), build flags |
| `nix/shell.nix` | Development shell: inherits build deps + adds Rust tools, Python/maturin, debuggers |
| `nix/banner.nix` | ASCII art banner displayed on `nix develop` |
| `nix/container.nix` | OCI container images for larql-server and larql CLI (Linux only) |
| `nix/models.nix` | Model catalog: HuggingFace models fetched as fixed-output derivations via git-lfs |
| `nix/demo.nix` | Demo apps: per-model vindex extraction and `nix run` targets |
| `nix/patches/use-system-protoc.patch` | Build patch: uses nixpkgs protoc instead of bundled protobuf-src |
| `nix/README.md` | This documentation |

### Adding New Modules

To add a new Nix module (e.g., NixOS module, test runner):

1. Create `nix/your-module.nix` following the `{ pkgs, lib, ... }: ...` pattern
2. Import it in `flake.nix` with `import (nixDir + "/your-module.nix") { inherit pkgs lib; }`
3. Wire it into the appropriate output (`packages`, `apps`, `checks`, etc.)

---

## Common Commands

| Command | Description |
|---------|-------------|
| `nix develop` | Enter development shell with all tools |
| `nix build` | Build the larql package (output in `./result/`) |
| `nix flake show` | List all flake outputs |
| `nix flake check` | Validate flake structure |
| `cargo build --release` | Build optimised binary (inside dev shell) |
| `cargo test` | Run all workspace tests |
| `make ci` | fmt-check + clippy + test |
| `make fmt` | Format all code |
| `make lint` | Run clippy with `-D warnings` |
| `nix build .#container` | Build larql-server OCI image (Linux only) |
| `nix build .#container-cli` | Build larql CLI OCI image (Linux only) |
| `nix run .#demo-list` | List available demo models |
| `nix run .#demo` | Walk default model (gemma4-4b) |
| `nix run .#demo-<key>` | Walk a specific model |
| `nix run .#demo-info` | Show GGUF metadata for local models |
| `nix build .#model-<key>` | Fetch a model from HuggingFace |
| `nix build .#vindex-<key>` | Build a vindex from a model |

---

## Demo Models

The flake includes a catalog of HuggingFace models that are fetched as Nix derivations (via `git-lfs`), with vindexes extracted at build time. Everything is cached in `/nix/store` — no runtime downloads needed.

### Available Models

| Key | Model | Size |
|-----|-------|------|
| `smollm2-360m` | SmolLM2 360M Instruct | ~725MB |
| `qwen-05b` | Qwen2.5 0.5B Instruct | ~970MB |
| `gemma4-2b` | Gemma 4 E2B Instruct | ~2.0GB |
| `tinyllama-1b` | TinyLlama 1.1B Chat v1.0 | ~2.2GB |
| `qwen-15b` | Qwen2.5 1.5B Instruct | ~3.1GB |
| `stablelm-2b` | StableLM 2 Zephyr 1.6B | ~3.2GB |
| `gemma4-4b` | Gemma 4 E4B Instruct (default) | ~4.0GB |
| `phi-35-mini` | Phi-3.5 Mini Instruct (3.8B) | ~7.6GB |

### Running a Demo

```bash
# List all available models
nix run .#demo-list

# Run default model (gemma4-4b) with default prompt
nix run .#demo

# Run a specific model
nix run .#demo-tinyllama-1b

# Run with a custom prompt
nix run .#demo-qwen-15b -- "The meaning of life is"

# Build just the vindex (without running)
nix build .#vindex-smollm2-360m
```

On first run, Nix fetches the model from HuggingFace and builds the vindex. Subsequent runs are instant (cached in `/nix/store`).

### Adding a Model

1. Prefetch the model hash:
   ```bash
   nix run nixpkgs#nix-prefetch-git -- --fetch-lfs https://huggingface.co/<org>/<model>
   ```
2. Copy the `rev` and `hash` into a new entry in `nix/models.nix`
3. The model automatically gets `model-<key>`, `vindex-<key>`, and `demo-<key>` targets

> **Note:** Gated models (Llama, Gemma) require HuggingFace authentication and won't work without `HF_TOKEN`. Prefer ungated models for the catalog.

---

## OCI Containers

Two container images are available (Linux only), built with `dockerTools.buildLayeredImage` for optimal layer caching.

### larql-server

```bash
# Build and load
nix build .#container
docker load < result

# Run with a vindex directory
docker run -d -p 8080:8080 \
  -v /path/to/vindexes:/data \
  larql-server:latest /data/my.vindex

# With gRPC enabled
docker run -d -p 8080:8080 -p 50051:50051 \
  -v /path/to/vindexes:/data \
  larql-server:latest /data/my.vindex --grpc-port 50051

# With CORS and API key
docker run -d -p 8080:8080 \
  -v /path/to/vindexes:/data \
  larql-server:latest /data/my.vindex --cors --api-key mysecret
```

### larql CLI

```bash
# Build and load
nix build .#container-cli
docker load < result

# Run a command
docker run --rm -v /path/to/vindexes:/data larql:latest repl /data/my.vindex
```

Both containers:
- Run as non-root user `larql` (UID 1000)
- Include TLS certificates for HuggingFace downloads (`hf://` paths)
- Mount vindex data at `/data`

---

## Python Bindings

The `larql-python` crate (PyO3 bindings) is excluded from the Nix package build because it requires `maturin`. Instead, build it inside the dev shell:

```bash
nix develop
cd crates/larql-python
maturin develop --release
pytest tests/ -v
```

---

## Notes

- **`Cargo.lock` is committed** — `rustPlatform.buildRustPackage` requires a lock file for reproducible builds. This is best practice for application repositories (per Cargo documentation).
- **Cross-platform** — The flake supports both Linux (OpenBLAS) and macOS (Accelerate framework). Metal GPU support is available on Apple Silicon via `--features metal` in the dev shell.
- **`result` symlinks** — `nix build` creates a `result` symlink in the repo root. These are gitignored.

---

## Troubleshooting

**Issue: `nix develop` fails with "experimental features" error**
```bash
# Solution: Enable flakes permanently
echo 'experimental-features = nix-command flakes' | sudo tee -a /etc/nix/nix.conf
```

**Issue: Build fails with missing `Cargo.lock`**
```bash
# Solution: Generate the lock file
cargo generate-lockfile
```

**Issue: OpenBLAS not found during build**
```bash
# This should be handled automatically by the Nix derivation.
# If building outside Nix, install openblas-dev (or equivalent) for your distro.
```

**Issue: protobuf version mismatch**
```bash
# The Nix derivation sets PROTOC to use the Nix-provided protobuf.
# Inside nix develop, protoc is available on PATH automatically.
```
