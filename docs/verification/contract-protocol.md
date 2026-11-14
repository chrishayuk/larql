# Verifier protocol `larql-verify-contract/1`

**Class: CURRENT.** What a program must do to answer LQL's `VERIFY REPRESENTATION`.

LQL does not evaluate contracts. A check written in LARQL would be LARQL marking its own work, so the measurement belongs to a
separate program, with no LARQL code in it. LQL runs that program, checks that its answer is well formed and about the question
asked, and renders it.

**LQL trusts the verifier it finds.** It validates the structure of the answer and what the answer is bound to, not whether the
measurement is true. The identity it prints, including any code digest, is what the verifier reports about itself, so it doesn't
authenticate the verifier. Replacing the executable can make a contract appear satisfied without any change to the model. See
[independence.md](independence.md#the-verifier-is-not-authenticated).

## Configuration

The first of these that applies names the verifier:

1. `Session::set_verifier`, for an embedding program;
2. the `LARQL_VERIFIER` environment variable;
3. `larql-verifier` on `PATH`, absolute entries only;
4. `verifier/.venv/bin/larql-verifier` in a directory above the running `larql` binary, which is where a checkout's verifier is
   installed. The current directory isn't searched, so that a statement can't run a program from wherever it happens to be run.

If none applies, the statement refuses. Setting the first or second to an empty value turns discovery off, and the statement
refuses: an explicitly disabled verifier never falls through to a later choice.

## Invocation

```text
<verifier> contract --contract <id> --container <dir> --source <dir>
```

`<dir>` values are canonical absolute paths, and LQL checks that they exist before running anything. The verifier runs directly,
with no shell. Stdin is closed. It has one hour, after which it is killed and the statement fails.

The verifier must not modify the container or the source: the statement is read-only.

## Exit status

| Exit | Meaning |
|---|---|
| 0 | The verifier reached an answer, including `violated` and `could_not_evaluate` |
| anything else | The verifier itself failed. The statement fails, showing the end of its stderr |

## Answer

Exactly one JSON object on stdout, and nothing else there. Diagnostics go to stderr.

```json
{
  "schema": "larql-verify-contract/1",
  "contract": "Q8_0_T2",
  "contract_statement": "every element of every Q8_0 tensor is within the T2 reconstruction bound of its source value",
  "evidence_kind": "numerical_measurement",
  "status": "violated",
  "violations": 7,
  "container": {"path": "/models/m-q8_0.vindex3", "sha256": "7309e605..."},
  "source": {"path": "/models/m-source", "identity": "9 files match pinned SHA-256 hashes"},
  "measurements": [{"name": "elements over T2", "value": "7"}],
  "offending": [{"name": "model.layers.0.self_attn.q_proj.weight", "violations": 7, "detail": "1 block"}],
  "reasons": [],
  "verifier": {"name": "example", "version": "1"},
  "report_sha256": "06fb2586..."
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | Exactly `larql-verify-contract/1` |
| `contract` | yes | The `--contract` value, unchanged |
| `contract_statement` | no | The property, in words |
| `evidence_kind` | yes | Exactly `numerical_measurement`. This version accepts no other kind of evidence |
| `status` | yes | `satisfied`, `violated` or `could_not_evaluate` |
| `violations` | yes | How many elements, or other units the contract counts, break it |
| `container.path`, `source.path` | yes | The `--container` and `--source` values, unchanged |
| `container.sha256` | yes, may be null | A digest of what was measured. The definition is the verifier's |
| `source.identity` | yes, may be null | How the source was identified |
| `measurements` | no | Name and value pairs, both strings, shown in order |
| `offending` | no | What broke the contract, worst first. Only the first 10 are shown |
| `reasons` | no | Why it could not be evaluated |
| `verifier` | yes | An object naming the verifier. Its members are shown as given |
| `report_sha256` | no | A digest of the full report, if the verifier has one |

Unknown fields are ignored.

## What makes an answer acceptable

LQL refuses the answer, and the statement fails, if any of these hold:

- it isn't valid JSON or lacks a required field;
- `schema` or `evidence_kind` isn't the exact value above;
- `contract`, `container.path` or `source.path` differs from what was asked;
- `status` is `satisfied` with `violations` other than 0, or with anything in `offending`;
- `status` is `violated` with `violations` of 0, or with `offending` not summing to `violations`;
- `status` is `could_not_evaluate` with no `reasons`;
- `status` is anything else.

A contract the verifier doesn't know is `could_not_evaluate`, as is any case where the result would be vacuous, such as a contract
about Q8_0 tensors on a container with none. **A verifier must never report `satisfied` for something it did not measure.**

## What the statement does

| Answer | Statement |
|---|---|
| `satisfied` | Returns the report |
| `violated` | Returns the report. A violation is an answer, not a failure |
| `could_not_evaluate` | Fails, with the reasons |
| The verifier fails, or the answer is refused | Fails |

## Operator contracts: `larql-verify-operator/1`

The same arrangement, for one recorded call of a computation rather than one container. The verifier reads a witness (`larql-gemv-witness/1`) and doesn't run anything.

```text
<verifier> operator --contract <id> --operator <tensor> --witness <witness.json> --container <dir>
```

The exit status, stdout and the rules for an acceptable answer are as above, with these differences:

| Field | Meaning |
|---|---|
| `schema` | Exactly `larql-verify-operator/1` |
| `operator` | The `--operator` value, unchanged |
| `witness.path` | The `--witness` value, unchanged |
| `witness.sha256` | A digest of the witness's manifest |
| `witness.basis` | Exactly `engine_reported` |
| `witness.reported` | What the witness says about its own run: reported, not established. This includes the path it says it ran against, which isn't checked: only the bytes are compared |
| `report_sha256` | Covers the contract, operator, object, the witness and container digests, the streams' digests and the result, not the verifier's code digest |

There's no `source`: the reference for these contracts is exact arithmetic on the container's own bytes.

**Before judging anything, the verifier checks** the witness's schema, kernel, activation, execution arm and representation, read as strict JSON; that it holds exactly one record for this operator; each stream's name, length and digest, read only from a regular file in the witness's directory, never through a symlink; and that the container, read by the validating reader with no warnings and no symlinks, holds exactly one Q8_0 tensor of that name and shape with the same bytes. A record that names its `object` is held to that object, and the answer reports the object judged. A witness that records `container_index_sha256` is held to it. If any of those fails, the answer is `could_not_evaluate`, because there's nothing to judge. The container is hashed again afterwards, and a change during the evaluation is also `could_not_evaluate`.

**What tying a witness to a container means.** The bytes are that tensor's in a container that agrees with itself when read. Nothing here ties those bytes to a source checkpoint, which is `Q8_0_T2`'s job, and a container changed and changed back between the two digests wouldn't be noticed.

| Contract | What it measures | What `violated` means |
|---|---|---|
| `Q8_0_GEMV_SEQ` | The recorded output equals a declared sequential binary32 program, bit for bit | This recorded output isn't that program's. That could be the kernel or a fault in the recording: for example, swapped `x` and `y` on a square tensor can't be told apart. **It doesn't mean less accurate:** a more accurate kernel violates this too |
| `Q8_0_GEMV_B1` | The recorded output is within `\|y − y*\| ≤ γ_k·S + k·2⁻¹⁵⁰·(1 + γ_k)` of exact arithmetic, with `u = 2⁻²⁴`, `γ_k = k·u/(1 − k·u)` and `S = Σ\|w·x\|` | The output is further from the exact result than the bound allows |

Both are numerical measurements of one recorded call.

- **`Q8_0_GEMV_B1` is experimental.** Its inequality has a derivation on paper that no proof system has checked. A satisfied result is a measured bound check on that call, not a machine-checked proof, and doesn't show the bound holds on other inputs.
- **The bound is loose.** Real faults that change the arithmetic can stay well inside it, so a satisfied `Q8_0_GEMV_B1` doesn't rule out small defects. That's why there are two contracts.
- **`Q8_0_GEMV_SEQ`'s order was chosen to describe LARQL's own CPU kernel.** Passing shows this recorded output matches the declared program, not that the declared order is the right one.
- **A witness is the engine's report.** These contracts judge the recorded output. They can't show the production kernel produced it, and a faulty or malicious engine could forge a witness.

## Patch contracts: `larql-verify-patch/1`

The same arrangement, for whether a compiled container implements what a `.vlp` patch declares of a base container. It answers `VERIFY PATCH`.

```text
<verifier> patch --contract <id> --patch <patch.vlp> --base <dir> --container <dir>
```

The exit status, stdout and the rules for an acceptable answer are as above, with these differences:

| Field | Meaning |
|---|---|
| `schema` | Exactly `larql-verify-patch/1` |
| `patch.path`, `base.path`, `container.path` | The values asked about, unchanged. `sha256` is the verifier's digest of each |
| `recomputed_from` | What the expected result was derived from. For `STRUCTURED_REMOVE_DIRECTION`: the operands and the base weights, **not the compiler's ledger** |

There's no `source`.

**What a verdict covers.** `status` is the verdict on the weights and is unchanged. Three further fields, additive, say what it does not cover, so a satisfied weight check can't be read as more:

| Field | Values | Meaning |
|---|---|---|
| `numerical_satisfaction` | `satisfied`, `violated`, `could_not_evaluate` | The same as `status`: do the actual BF16 weights implement the declared transformation |
| `ledger_integrity` | `not_checked` | The ledger is not read, so nothing is claimed of it. A corrupted ledger leaves the weight verdict as it was |
| `provenance` | `not_checked` | Never `satisfied` by this contract. `provenance_detail` breaks it down: `base_binding` is `satisfied` when the patch was judged (a stale binding is refused) and `could_not_evaluate` otherwise; `operand_origin`, `operand_derivation` and `op_digest` are `not_checked` |

| Contract | What it measures | What `violated` means |
|---|---|---|
| `STRUCTURED_REMOVE_DIRECTION` | Every element of each edited tensor equals the BF16 result of the declared `remove_direction` recomputed on exact rationals (declared columns), or the base (all other elements), and every other BF16 tensor equals the base | Some element differs. `offending` names the tensor and the first element, with the expected and found bits |

**Before judging anything, the verifier checks** that the patch holds only `structured_weight` operations of an implemented profile (`f64-seq-rne1/v1`, `f64-seq-rne2/v1`), that each op's binding matches the base (the SHA-256 of its `index.json` and the representation's payload digest), that operands and columns fit the tensor, and that no two ops declare the same column. Otherwise the answer is `could_not_evaluate`. Both containers are read by the validating reader, and all three inputs are digested again afterwards: a change during the evaluation is `could_not_evaluate`.

**What it does not establish:** the derived (quantised) representations a compile re-derives, the digest the op carries (which detects accidental damage, not tampering), and the operand's `origin`. It recomputes with exact rational arithmetic, not by running LARQL's interpreter, so agreement between the two is evidence that neither shares the other's mistake; the interpreter's profile definitions are the common specification.

## The contract `Q8_0_T2`

LARQL's own verifier, [`verifier/`](../../verifier/README.md), implements this protocol for one contract, `Q8_0_T2`. It accepts only
source checkpoints whose hashes it pins, which today means Qwen3-0.6B at `c1899de2`. Anything else is `could_not_evaluate`. A different
verifier can be configured instead.

The contract's bound, T2, is derived from ggml's reference encoder:
`|x − q·d₁₆| ≤ d₁₆·(½ + 2⁻¹² + 2⁻¹⁶ + 127·2⁻¹¹) + 127·2⁻²⁵`, for an element `x` in a block with stored scale `d₁₆` and code `q`.

A result is a numerical measurement of one container. It is not a formal proof, and not a claim about the encoder that produced
the container.
