# GEMV witness `larql-gemv-witness/1`

**Class: CURRENT.** What `vindex3 exec --gemv-witness` records about the CPU Q8_0 kernel, for a program outside the engine to judge.

```sh
larql vindex3 exec <container> --backend production-q8 --representation-source stored \
  --tokens <ids> --logit-dump <file> --gemv-witness <new dir> [--gemv-witness-max-operands N]
```

It records the first call reached on each of up to `N` distinct Q8_0 operands (default 1). A record stores that matrix's bytes in full, so a large `N` is large on disk. `--gemv-witness`
requires `--logit-dump`. It refuses `--bank`, `--plugin`, `--lowering`, `--representation`, a limit of 0, and an existing directory before the run starts. It fails rather than writing an empty witness if the run never reaches this kernel:
for example under `LARQL_KQUANT_EXEC=widen`, or on a device backend. Don't compare timings from a capture run with others, because capturing digests and copies the matrices.

## What it holds

The directory is new. `witness.json` is renamed into place after every stream is written and synced, so a witness.json that exists is whole, and a directory without it is incomplete.

```json
{
  "schema": "larql-gemv-witness/1",
  "kernel": "cpu.FusedKQuant.q8_0_gemv",
  "activation": "f32",
  "execution_arm": "Direct",
  "build_commit": "a23b37b8…",
  "binary_sha256": "4d80dd42…",
  "container": "/models/m.vindex3",
  "container_index_sha256": "…",
  "component": "target",
  "backend": "ProductionQ8",
  "representation": "Q8_0",
  "representation_source": "stored",
  "tokens": [9707, 1879],
  "draft_depth": null,
  "logit_dump": {"path": "/runs/logits.bin", "sha256": "…", "bytes": 16000},
  "environment": {"LARQL_CPU_ARITHMETIC": "…"},
  "platform": "macos-aarch64",
  "projection_plans": {"FusedKQuant": 3136, "FusedQ8": 16},
  "calls_observed": 3136,
  "operands_noted": 196,
  "max_operands": 196,
  "records": [
    {
      "call": 0,
      "object": "target.decoder_stack",
      "tensor": "0.self_attn.q_proj.weight",
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

| Field | Meaning |
|---|---|
| `call` | The order calls reached the capture, counting every Q8_0 call from 0. With positions running concurrently, this, the first `x` recorded for an operand, and which operands fill the limit aren't reproducible |
| `calls_observed` | Q8_0 kernel calls during the run. Each batched position is one |
| `operands_noted` | Q8_0 operands loaded |
| `max_operands` | The limit asked for, so that "limited to N" can be told from "fewer were reached" |
| `representation` | The encoding the run asked for. `encoding` on a record is that operand's own |
| `container_index_sha256` | The container's `index.json` when the run started, which holds its segments' digests |
| `logit_dump` | Computed as this run wrote the bytes, not by reading the path later |
| `environment` | `LARQL_*` and `RAYON_NUM_THREADS`: settings that can change the numbers |
| `projection_plans` | Calls per kernel in the whole run. This witness covers only `FusedKQuant`, so it's part of a run, not all of it |
| `object`, `tensor` | The operand's name. A tensor name is only unique within its object |

`blocks` are the stored bytes as the kernel read them. `x` and `y` are little-endian IEEE 754 binary32 bit patterns. `build_commit` is `null` unless the
build set `LARQL_BUILD_COMMIT`. It's never read from a working tree.

## What it shows, and doesn't

It only observes. It runs after the kernel returns, and the engine computes no expected value. Judging the result belongs to the verifier.

**A witness is the engine's report of what the engine did.** A verifier can check that `y` is right for that `x` and those bytes, and that the bytes
are what the container holds under that object and tensor name. It can't check that the production kernel really produced `y`. Every field in the binding, including the tensor name and the commit,
is reported rather than established. This doesn't defend against a dishonest engine.

**`binary_sha256` doesn't bind the running image.** It's the file at the executable's path just after start-up, so replacing that file at the wrong moment gives another file's digest. It never covers linked libraries, or anything injected through the environment.

**Capture on versus off.** The hook comes after the kernel and reads only, and a test holds the output bits equal with it on and off, on a small matrix.
Comparing a real run's logit dumps with it on and off, as an acceptance check, is stronger. Both are tests of that run, not a proof that the instrumented
and uninstrumented programs are equivalent.
