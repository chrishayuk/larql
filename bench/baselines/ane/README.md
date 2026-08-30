# ANE-0b — frozen GPU-alone baseline

The control ANE-3 scores against. **Immutable once ANE-3 begins.**

## What it is

GPU-alone `f16_gemv` on the shape ANE-1 puts on the Neural Engine:
Qwen3.8-27B's FFN gate/up projection, `5120 -> 17408`, f16 weights and
f32 activation. 178.26 MB of weights per call, the most repeated large
op in the model, and FFN is ~63% of the bytes read per token.

Three lines are measured; **only `ffn_gate_up` is the control**:

| line | shape | why it is here |
|---|---|---|
| `ffn_gate_up` | 5120 → 17408 | the control |
| `ffn_down` | 17408 → 5120 | same bytes, 3.4× fewer threadgroups — guards against a `ROWS_PER_TG` geometry surprise |
| `dispatch_floor` | 5120 → 8 | per-call submission cost with almost no weight traffic, so the control's ms can be read as work rather than overhead |

## Why not just use the ~367 GB/s roofline number

It came from a different access pattern. Scoring a concurrency result
against a differently-shaped control is how a 5–10% "gain" gets
manufactured. This exists so ANE-3 has a same-shape, same-dtype,
same-kernel-family control measured on this machine.

## Protocol

Enforced by `run_ane0b.sh`, not by memory:

- **AC power required** — the script refuses on battery.
- **Two independent sessions**, run separately, min-of-N within each.
  The cross-session e2e floor on this machine is ~±6%.
- **Nothing else touching the GPU.** Recorded in the provenance file.
- **Immutable**: the script refuses to overwrite an existing session.
- Thermal state recorded before and after; git SHA and dirty count
  recorded.

## The probe's own controls

Both fire inside the binary, and a failure exits non-zero:

1. **It computed the projection.** One row is recomputed on the CPU from
   the same f16 bytes and compared (tolerance 2e-3; observed ~1e-6). A
   harness that is fast because it computed nothing fails here.
2. **The implied rate is physical.** Above 600 GB/s — beyond this
   machine's ~400 GB/s fabric — the clock is broken, not the result.

## This is a control, not a performance rung

Do not optimise the GPU path here. If something looks wrong, diagnose
it and say so; otherwise freeze it.

## Running

```
cargo build --release -p larql-compute-metal --example ane0b_gpu_baseline
./bench/baselines/ane/run_ane0b.sh s1     # session 1
./bench/baselines/ane/run_ane0b.sh s2     # session 2, separately
```

## Status

**BANKED 2026-08-25** — sessions `s1-battery`, `s2-battery`, on battery
by explicit override (`ANE0B_ALLOW_BATTERY=1`, which requires the session
label to say so). Denominator for ANE-3: **288.7 / 289.1 GB/s raw
effective**, agreeing to 0.15%. See `ADJUDICATION.md`.
