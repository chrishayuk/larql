#!/usr/bin/env python3
"""Keep system TLS out of the shipped target graph (host build tools are allowed)."""

import argparse
from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parent.parent
TLS_DEPS = {"reqwest", "tungstenite"}
SYSTEM_TLS = {"native-tls", "tokio-native-tls", "openssl", "openssl-sys", "hyper-tls"}


def manifest_errors(path: Path) -> list[str]:
    """Check every dependency table, including dev/build and target-specific ones."""
    errors = []

    def visit(table):
        for name, value in table.items():
            if name in {"dependencies", "dev-dependencies", "build-dependencies"}:
                for alias, dep in value.items():
                    package = dep.get("package", alias) if isinstance(dep, dict) else alias
                    if package not in TLS_DEPS:
                        continue
                    if not isinstance(dep, dict) or dep.get("workspace") is not True:
                        errors.append(f"{path}: {alias} must inherit workspace TLS settings")
                    elif dep.get("default-features") is True or any(
                        feature in {"default", "default-tls"} or "native-tls" in feature
                        for feature in dep.get("features", [])
                    ):
                        errors.append(f"{path}: {alias} re-enables system TLS")
            elif isinstance(value, dict):
                visit(value)

    visit(tomllib.loads(path.read_text()))
    return errors


def system_tls_packages(tree: str) -> set[str]:
    return {line.split()[0] for line in tree.splitlines() if line.split()} & SYSTEM_TLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="aarch64-unknown-linux-gnu")
    args = parser.parse_args()
    errors = [error for path in (ROOT / "crates").glob("*/Cargo.toml") for error in manifest_errors(path)]
    if errors:
        raise SystemExit("\n".join(errors))
    # Metadata only: no cross compiler, target stdlib or OpenBLAS install needed.
    # Normal edges exclude OpenBLAS/Swagger's host-side download build tools.
    tree = subprocess.run(
        ["cargo", "tree", "--locked", "--target", args.target, "--no-default-features",
         "-p", "larql-cli", "-p", "larql-router", "-p", "larql-server",
         "-e", "normal", "--prefix", "none", "--format", "{p}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if tree.returncode:
        raise SystemExit(tree.stderr.strip() or f"cargo tree failed with status {tree.returncode}")
    forbidden = system_tls_packages(tree.stdout)
    if forbidden:
        raise SystemExit(f"{args.target}: system TLS in target dependencies: {', '.join(sorted(forbidden))}")
    print(f"{args.target}: workspace TLS inheritance and shipped target graph are clean")


if __name__ == "__main__":
    main()
