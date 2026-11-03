# LARQL contract verifier

**Class: CURRENT.** Measures a VINDEX3 container against a numerical contract. It's what LQL's `VERIFY REPRESENTATION` runs.

It ships with LARQL, but it isn't built from LARQL: it's a separate Python package with no LARQL code in it, so that LARQL can't mark
its own work. [docs/verification/independence.md](../docs/verification/independence.md) says what enforces that, and what doesn't.

**A result is a numerical measurement of one container.** It isn't a formal proof, and it isn't a claim about the encoder that produced
the container. There is one contract and one pinned model so far.

**A result doesn't authenticate the verifier.** The code digest in an answer is the program's report about itself, which helps reproduce a result.
Replacing the program could make a contract appear satisfied without changing the model, so a result must not gate a release on this basis.
[What a release gate would need](../docs/verification/independence.md#the-verifier-is-not-authenticated).

## Results

`Q8_0_T2` requires every element of every Q8_0 tensor to be within the T2 reconstruction bound of its source value. On two containers of
Qwen3-0.6B at `c1899de2`:

| Container | Encoded with LARQL | Result | Elements over T2 |
|---|---|---|---:|
| Original | `2536bb79` | `violated` | 57,634 |
| Corrected | `a23b37b8` | `satisfied` | 0 |

## Install

From `verifier/`, with Python 3.12 and [uv](https://docs.astral.sh/uv/). It doesn't need `torch`, or a built LARQL.

```sh
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip sync --require-hashes requirements.lock
VIRTUAL_ENV=$PWD/.venv uv pip install --no-deps .
```

## Use

```sh
.venv/bin/larql-verifier contract --contract Q8_0_T2 --container <dir> --source <checkpoint dir>
```

The [protocol](../docs/verification/contract-protocol.md) specifies the invocation and the answer, how LQL finds a verifier, and how to
write a different one.

## Reproduce

```sh
./reproduce.sh
```

It downloads the checkpoint from Hugging Face, builds LARQL at each commit in a separate worktree, encodes a container with it, measures it,
and compares the status, the violation count, the report digest and the Q8_0 segment's digest with `expected/qwen3-0.6b.json`. It needs git,
curl, a Rust toolchain, whatever LARQL needs to build and about 10 GB of disk.

**Where it has been run:** macOS on Apple silicon only. Encoding on other platforms hasn't been compared.

A container's overall digest isn't in the expected results, because `candidate.json` records the path of the encoder's input. So it differs
between machines even when the weights are identical.

## Tests

```sh
.venv/bin/python -I -m unittest discover -s tests
```

The tests are synthetic, apart from one that runs if you set `LCV_SOURCE` to a real checkpoint directory.

## What it does and doesn't establish

- **T2** is derived from ggml's reference encoder: `|x − q·d₁₆| ≤ d₁₆·(½ + 2⁻¹² + 2⁻¹⁶ + 127·2⁻¹¹) + 127·2⁻²⁵`, for an element `x` in a block
  with stored scale `d₁₆` and code `q`. It has been checked empirically, not formally proved.
- **`satisfied`** means no element of those tensors exceeded the bound in that container. It says nothing about any other container, or about
  how a model behaves when run.
- **`could_not_evaluate`** is never a pass. It's what you get for an unknown contract, an unpinned source, a container with no Q8_0 tensors, or
  anything T2 makes no statement about.

## Licence

Apache-2.0, the same as LARQL. See [`LICENSE`](LICENSE).

