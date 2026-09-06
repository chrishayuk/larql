# Native record experiments

Campaign synthesis: [reversible value control, measured addressing failure](CAMPAIGN.md).
The bounded twelve-slot fitting branch is **closed**; selective writable records
have not been established. Composition, capacity scaling and HNSW remain deferred.

Latest result: [cached rejection analysis and one gate-loss correction](GATE_AUX_RESULTS.md).
All suppressed training positives recover and training delivery reaches 116/116
on both rosters, but validation delivery/rejection still fail the frozen gate.
No full-model forwards were run. [Cached-only protocol](GATE_AUX_PROTOCOL.md).

Previous result: [restricted native gate/up fit](CONTROLLER_RESULTS.md). Fresh
sentence answers improve to 33/48 on the observed roster and 32/48 on a fresh
installation, but both fail activation delivery and selective-record controls.
The C+A reference reaches 48/48 on these new sentence forms. Values remain
frozen during fitting; no inference-time labels or masks are used by the fitted
arm. [Controller protocol](CONTROLLER_PROTOCOL.md).

Previous result: [eight-arm oracle decomposition](DECOMPOSITION_RESULTS.md).
Canonical activation replacement alone recovers 24/30 failed sentence reads;
adding final-position competitor suppression reaches 26/30 and matches every
full-oracle top-1 answer without suppressing earlier writes. Earlier suppression
adds no correct reads at any background. This remains an artificial diagnostic,
not a working controller. [Decomposition protocol](DECOMPOSITION_PROTOCOL.md).

Previous [oracle routing and value strength](ORACLE_RESULTS.md): oracle routing
recovers 26/30 failed sentence reads; a fixed larger down-column scale makes
both preselected records read 7/7 under oracle. Composition remains closed.
[Oracle protocol](ORACLE_PROTOCOL.md).

Previous results: [activation/discrimination diagnostic](DISCRIMINATION_RESULTS.md)
and [4×3 factorial write](FACTORIAL_RESULTS.md), under the
[frozen follow-up protocol](DISCRIMINATION_PROTOCOL.md). Both factorial editors
fail the selective-record gate; native composition remains closed.

The [original entry-gate verdict](RESULTS.md) and [original protocol](PROTOCOL.md)
are preserved. That three-write recipe reproduces and generalizes, but fails
entity–relation selectivity.

Worktree: `.claude/worktrees/native-record-composition`, branch
`exp/native-record-composition`, forked from `a4dfe541`.

## Reproduce

Run from this worktree's root using the existing Python 3.12 environment with
MLX, NumPy, Transformers and the editable `chuk_lazarus` installation. The run
artifact records dependency versions, loader commit, original source digest,
and snapshot file locations. Metal access is required. Loading is offline and
the supplied model path must be the original local, unquantized 4B checkpoint.

```sh
python3 bench/native-record/run_address.py \
  --model /Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767 \
  --out bench/native-record/results/address-v2
```

An existing output directory is refused. `--fixture-only` writes the frozen
fixture without loading MLX or model weights. Each scored row is flushed to
`rows.jsonl`; only a completed run writes `summary.json`. The ignored
`edit_slots.npz` stores the few original/edited rows and columns, not a model
copy. No checkpoint or vindex is mutated.

Audit a completed artifact and run instrument checks without GPU access:

```sh
python3 bench/native-record/audit_run.py bench/native-record/results/address-v1
cd bench/native-record
python3 -m unittest -v test_address.py
```

`vendor/native.py` is the exact original source, retained for provenance and
its `unit`/`unique_part` operations. It is imported, not invoked as a standalone
experiment. The extended runner retains its mathematical write recipe and
adds the frozen evaluation and state transitions. It reads only the required
embedding rows instead of materializing the full embedding table.

## Diagnostic and factorial reproduction

The diagnostic captures the original bank and disjoint development/evaluation
rosters. It evaluates L26 first and runs the fixed layer screen only when both
reader gates fail. The factorial runner uses a third entity roster and the
unchanged, predeclared method comparison.

```sh
native_snapshot=/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
python3 bench/native-record/run_discrimination.py \
  --model "$native_snapshot" --out bench/native-record/results/discrimination-v2
python3 bench/native-record/run_factorial.py \
  --model "$native_snapshot" --out bench/native-record/results/factorial-v2
python3 bench/native-record/audit_binding.py \
  --diagnostic bench/native-record/results/discrimination-v1 \
  --factorial bench/native-record/results/factorial-v1
cd bench/native-record
python3 -m unittest -v test_address.py test_binding.py
```

Every run requires a new output directory. Capture NPZ files are intentionally
ignored by Git but retained locally for vector-level analysis and hash audits;
keep them with the JSON artifacts when archiving the experiment. The scripts
need no changes to LARQL runtime crates or the external chuk-lazarus checkout.

## Oracle-routing reproduction

Uses the same local snapshot and existing factorial-v1 baseline. The run is
bounded at 1,500 scored forwards. A new output directory is required.

```sh
python3 bench/native-record/run_oracle.py \
  --model "$native_snapshot" --out bench/native-record/results/oracle-v3
python3 bench/native-record/audit_oracle.py bench/native-record/results/oracle-v2
cd bench/native-record
python3 -m unittest -v test_address.py test_binding.py test_oracle.py
```

`oracle-v1` is a preserved instrument failure with zero scored rows; `oracle-v2`
is the completed run. Its ignored `normalization_vectors.npz` is required for
the vector-level audit and should accompany its JSON artifacts when archived.

## Oracle-component decomposition reproduction

Uses the same snapshot and the completed `oracle-v2` fixture and endpoint
artifacts. Six new combinations plus opening/closing endpoint verification
require 840 scored forwards. All down-column scales stay at 1×.

```sh
python3 bench/native-record/run_decomposition.py \
  --model "$native_snapshot" --out bench/native-record/results/decomposition-v2
python3 bench/native-record/audit_decomposition.py \
  bench/native-record/results/decomposition-v1
cd bench/native-record
python3 -m unittest -v test_address.py test_binding.py test_oracle.py test_decomposition.py
```

The completed run is `decomposition-v1`. Preserve its ignored `vectors.npz`
alongside the JSON artifacts; the audit uses it to recheck all intervention
scopes and the 336 earlier-position pairs with identical final local vectors.

## Restricted gate/up controller reproduction

The runner uses the observed factorial development bank to choose among four
fixed optimizer settings, freezes the choice, then evaluates new forms on that
roster and a fresh installation. The fresh roster receives the same fitting
setting and step count without retuning. Seven evaluation states per roster
give 2,352 scored forwards, plus installation and development captures.

```sh
python3 bench/native-record/run_controller.py \
  --model "$native_snapshot" --out bench/native-record/results/controller-v3
python3 bench/native-record/audit_controller.py \
  bench/native-record/results/controller-v2
cd bench/native-record
python3 -m unittest -v test_address.py test_binding.py test_oracle.py test_decomposition.py test_controller.py
```

`controller-v1` preserves a zero-scored-row harness failure. `controller-v2`
is the completed model experiment. Keep its ignored `captures.npz` with its fixture,
optimizer traces, frozen selection, scores and summary when archiving.

## Cached rejection and auxiliary gate-loss reproduction

No model path is accepted and no checkpoint is loaded. The runner reads only
the allowed development inputs and row arrays from `controller-v2`, performs
two exact zero-auxiliary controls, three fixed-strength candidates and one
fixed-strength second-roster repeat. The result gates any future full-model run;
the completed artifact does **not** permit one.

```sh
python3 bench/native-record/run_gate_aux.py \
  --out bench/native-record/results/gate-aux-v2
python3 bench/native-record/audit_gate_aux.py \
  bench/native-record/results/gate-aux-v1
cd bench/native-record
python3 -m unittest -v test_address.py test_binding.py test_oracle.py test_decomposition.py test_controller.py test_rejection.py
```

MLX/Metal is used only for cached small-matrix fitting and bf16 activation
replay. The audit and unit tests need no GPU. Preserve `arrays.npz` alongside
the analyses, complete threshold curves, traces, frozen choice and hashes.
