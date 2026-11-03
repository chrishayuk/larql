# Verifier protocol `larql-verify-contract/1`

**Class: CURRENT.** What a program must do to answer LQL's `VERIFY REPRESENTATION`.

LQL does not evaluate contracts. A check written in LARQL would be LARQL marking its own work, so the measurement belongs to a
separate program, with no LARQL code in it. LQL runs that program, checks that its answer is well formed and about the question
asked, and renders it.

**LQL trusts the verifier you configure.** It validates the structure of the answer and what the answer is bound to, not whether
the measurement is true. The identity it prints is what the verifier reports about itself.

## Configuration

The first of these that applies names the verifier:

1. `Session::set_verifier`, for an embedding program;
2. the `LARQL_VERIFIER` environment variable;
3. `larql-verifier` on `PATH`;
4. `verifier/.venv/bin/larql-verifier` in a directory above the running `larql` binary, which is where a checkout's verifier is
   installed. The current directory isn't searched, so that a statement can't run a program from wherever it happens to be run.

If none applies, the statement refuses. Setting the first or second to an empty value turns discovery off, and the statement refuses.

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

## The contract `Q8_0_T2`

LARQL's own verifier, [`verifier/`](../../verifier/README.md), implements this protocol for one contract, `Q8_0_T2`. It accepts only
source checkpoints whose hashes it pins, which today means Qwen3-0.6B at `c1899de2`. Anything else is `could_not_evaluate`. A different
verifier can be configured instead.

The contract's bound, T2, is derived from ggml's reference encoder:
`|x − q·d₁₆| ≤ d₁₆·(½ + 2⁻¹² + 2⁻¹⁶ + 127·2⁻¹¹) + 127·2⁻²⁵`, for an element `x` in a block with stored scale `d₁₆` and code `q`.

A result is a numerical measurement of one container. It is not a formal proof, and not a claim about the encoder that produced
the container.
