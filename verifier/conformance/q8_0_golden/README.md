# Q8_0 conformance fixtures

**Class: CURRENT.** Generates `crates/larql-models/src/quant/ggml/tests/q8_0_ggml_golden.rs`, the expected bytes that LARQL's Q8_0 encoder is tested
against. Every expected block comes from ggml's reference encoder at a pinned commit. None is written by hand.

```sh
./build.sh
../../.venv/bin/python -I gen_q8_golden.py > ../../../crates/larql-models/src/quant/ggml/tests/q8_0_ggml_golden.rs
rustfmt --edition 2021 ../../../crates/larql-models/src/quant/ggml/tests/q8_0_ggml_golden.rs
```

`build.sh` fetches ggml from GitHub and checks each source file it uses against `ggml-SHA256SUMS`. It needs git, a C compiler and network access, so
it isn't part of the verifier's tests, and CI doesn't regenerate the fixture. It builds a macOS `.dylib` and has only been run on macOS on Apple silicon.

This is a test fixture for the engine, not part of the contract measurement. No code in `larql_contract_verifier/` or `refcheck/` uses it.
