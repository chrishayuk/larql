# GLM-5.3-Flash reference oracle

The checkpoint ships **no modeling code**, so the architecture named by its
`architectures` field resolves to upstream `transformers`. **Upstream *is* the
contract**, which makes this environment one of the programme's authorities
rather than a debugging convenience — a `transformers` upgrade can change every
GLM parity number in the funnel.

Reconstructing it must therefore not depend on which scratch directory happened
to survive.

```sh
tools/oracles/glm5/bootstrap.sh                 # venv + pinned deps + source check
export LARQL_GLM_ORACLE=$PWD/.glm-oracle-venv
$LARQL_GLM_ORACLE/bin/python tools/oracles/glm5/smoke.py
```

| file | role |
|---|---|
| `requirements.txt` | exact pins (torch 2.14.0, transformers 5.16.1, numpy 2.5.2, safetensors 0.8.0; resolved on Python 3.12) |
| `bootstrap.sh` | build or reuse the venv, then verify sources. Idempotent |
| `verify_sources.sh` | the installed `glm5_next` must hash to `scripts/glm_reference_sources.sha256`. **Fails loudly on drift** |
| `smoke.py` | four checks in order: environment → source hashes → one real KDA layer strict-loads → it runs and emits the boundaries PHYSICAL-1 scores |

`smoke.py --raw-dir DIR` writes `input.f32`, `output.f32`, `decay.f32` and
`meta.txt` as flat little-endian f32 — the reference trajectory the Metal arm is
scored against:

```sh
$LARQL_GLM_ORACLE/bin/python tools/oracles/glm5/smoke.py --positions 512 --raw-dir /tmp/glm_ref
cargo run --release -p larql-compute-metal --example kda_q4_trajectory_real -- \
    /Volumes/model-drive/models/GLM-5.3-Flash 0 512 /tmp/glm_ref
```

Passing a reference directory turns on the **baseline gate**: LARQL's BF16 arm is
scored against upstream first, and the run **returns without scoring any
quantised arm** if it misses. That ordering is the point — a Qx miss on an
unverified executor has two causes.

**The venv is not in the repo; its construction is.** Steps 1 and 2 of the smoke
test pass without the weights mounted, and say so, so the *environment* can be
verified separately from the *checkpoint*.
