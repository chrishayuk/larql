# GW-CAR-1A pre-seal review

**Status:** Executable design draft. No CAR-1A candidate replay or adjudication
exists. The registered CAR-1A grid and gates are in the
[specification](gw-car-1.md) and
[machine draft](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1a-protocol.draft.json).

The machine draft's file SHA-256 is
`9ada55da0ddae37195a04051904f39e632242ecf3660d2dd7c04b6ef6ecc746d`.
The already-built `target/release/examples/gwcar1a_record` executable's
SHA-256 is `f25b603c04573472a0d7cf3110850e0d467458b4b724f014dd5712fa1937a38f`.
The draft leaves `protocol_sha256` and `runner.executable_sha256` null; the
separate seal operation fills those fields after review and refuses to
overwrite an existing seal.

The 19 candidate IDs consist of exact and donor controls, f16, signed
2/4/8-bit quantization, six sparse-coordinate sizes, and seven PCA ranks.
PCA uses only the 396 train carriers. Every candidate changes only the
entering L24 carrier; both H1 arms retain the same seven donor heads. The
runner requires bit-exact reproduction of STATE-1 contexts 128 and 0 by the
exact and donor controls. The denominator remains STATE-1 context 255, and
one held-out max-t family covers all 19 candidates and four raw/z readouts in
each split. Each candidate reports encoded row bytes, shared model bytes,
formula-derived decode work, and natural-prefix dependence. CAR-1B has no
candidate or outcome surface in this draft.

Checks completed before seal:

- The draft validates against the sealed STATE-1 protocol, selection,
  adjudication, replay tensors, and erratum witness.
- The inherited context-255 denominator recomputes exactly from the
  historical replay tensors on both held-out splits.
- Seven synthetic Python codec/analysis tests and the Rust H1-arm test pass.
- The separate Rust example builds in release mode, passes Clippy with
  warnings denied, and passes its own `rustfmt --check`.
- The candidate fitter rejects the unsealed draft before creating a fit
  directory. The release replay binary rejects it before creating an output
  directory.

The source and executable are ready for a joint protocol/implementation
review. Once reviewed, the seal command is:

```bash
python3 scripts/gwcar1a_preregister.py seal \
  bench/gw-car-1/gemma3-4b-it-phase1/gwcar1a-protocol.json \
  bench/gw-car-1/gemma3-4b-it-phase1/gwcar1a-protocol.draft.json \
  target/release/examples/gwcar1a_record
```

The subsequent fit, train replay, held-out replay, and adjudication must use
that frozen protocol and the bound executable. Any pre-outcome correction to
the grid, gate, code, or authorities requires refreshing the draft and
rebuilding before seal. Any post-seal correction requires an explicit
amendment; the original protocol remains unchanged.
