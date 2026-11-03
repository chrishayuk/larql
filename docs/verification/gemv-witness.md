# GEMV witness `larql-gemv-witness/1`

**Class: CURRENT.** What `vindex3 exec --gemv-witness` records about the CPU Q8_0 kernel, for a program outside the engine to judge.

```sh
larql vindex3 exec <container> --backend production-q8 --representation-source stored \
  --tokens <ids> --logit-dump <file> --gemv-witness <new dir> [--gemv-witness-max-operands N]
```

It records the first call on each of up to `N` distinct Q8_0 operands (default 1). A record stores that matrix's bytes in full, so a large `N` is
large on disk. `--gemv-witness` requires `--logit-dump`, and it fails rather than writing an empty witness if the run never reaches this kernel: for
example under `LARQL_KQUANT_EXEC=widen`, or on a device backend.

## What it holds

The directory is new: an existing one is refused. `witness.json` is written last, so a directory without it is incomplete.

```json
{
  "schema": "larql-gemv-witness/1",
  "kernel": "cpu.FusedKQuant.q8_0_gemv",
  "activation": "f32",
  "execution_arm": "Direct",
  "build_commit": "a23b37b8…",
  "binary_sha256": "4d80dd42…",
  "container": "/models/m.vindex3",
  "representation": "Q8_0",
  "calls_observed": 1120,
  "operands_noted": 196,
  "records": [
    {
      "call": 0,
      "tensor": "layers.0.self_attn.q_proj.weight",
      "encoding": "Q8_0",
      "rows": 1024,
      "columns": 1024,
      "blocks": {"file": "000000.blocks", "bytes": 1114112, "sha256": "…"},
      "x": {"file": "000000.x", "bytes": 4096, "sha256": "…"},
      "y": {"file": "000000.y", "bytes": 4096, "sha256": "…"}
    }
  ]
}
```

`blocks` are the stored bytes as the kernel read them. `x` and `y` are little-endian IEEE 754 binary32 bit patterns. `build_commit` is `null` unless the
build set `LARQL_BUILD_COMMIT`. It's never read from a working tree.

## What it shows, and doesn't

It only observes. It runs after the kernel returns, and the engine computes no expected value. Judging the result belongs to the verifier.

**A witness is the engine's report of what the engine did.** A verifier can check that `y` is right for that `x` and those bytes, and that the bytes
are the container's. It can't check that the production kernel really produced `y`. Every field in the binding, including the tensor name and the commit,
is reported rather than established. This doesn't defend against a dishonest engine.

**Capture on versus off.** The hook comes after the kernel and reads only, and a test holds the output bits equal with it on and off, on a small matrix.
Comparing a real run's logit dumps with it on and off, as an acceptance check, is stronger. Both are tests of that run, not a proof that the instrumented
and uninstrumented programs are equivalent.
