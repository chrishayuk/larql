# Kimi Linear 48B — the PRECISION-1 topology

**The first complete REPRESENT chain: a per-layer precision frontier,
a composed map earned against a frozen behavioural contract at
authority scale, and an in-session decode benchmark.** Closed
2026-08-30 on Kimi-Linear-48B-A3B-Instruct
(`~/chris-models/Kimi-Linear-48B-A3B-Instruct.aligned.vindex3`).

Contract: **kimi-logit-v3, frozen at `ce6e87a3`** — six authority
criteria (positions ≥ 4096, KL p99 ≤ 1e-3, covered mass ≥ 0.6, top-1
mass displaced ≤ 5e-2, top-10 mass p99 ≤ 1e-1, route mixture mass
p99 ≤ 0.15 / max ≤ 0.25). Evidence bank: 256 × 32-position
teacher-forced sequences from real prose, identity
`a73f437aeb1d…` (manifest + per-file SHA256 in
`~/chris-models/kimi-quality-bank-provenance/`).

## The earned topology

```text
layers 0–23   BF16      (Q8_0 refused at L19 and below; see frontier)
layers 24–26  Q8_0      composed map, PASSED v3 at 8192 positions
```

Composed authority (8192 positions): **kl p99 4.153e-4** (2.4×
headroom), covered 0.631, top-1 mass max 1.56e-2, top-10 mass p99
6.15e-2, route mass p99 0.113. Report
`kimi_map-l24-26q80-8192_report.json`. Counts at that scale — 23
top-1 flips, 1041 top-10 changes, 163 route flips — would have
REFUSED under the count-based v1/v2 gates; the consequence contract
is what makes 8192-position authority reachable at all.

Bytes: expert banks for L24–26 drop 3 × 3.62 GB (BF16) → 3 × 1.93 GB
(Q8_0), **5.1 GB saved**, all other tensors untouched.

Decode (in-session, interleaved blocks, min-of-N floors, two
sessions, AC power): baseline BF16 **35.46 / 35.52 tok/s** vs
candidate **36.44 / 36.27 tok/s** → **1.021–1.028×**; GPU time
27.0 → 26.4–26.5 ms/token (−2.2%, consistent with the roofline for
the byte reduction). Harness:
`opplan/exec/tests/q2a_decode_bench.rs`.

## The per-layer Q8_0 frontier (256-position diagnostics)

```text
layer  kl p99      verdict
 25    2.305e-4    PASS   (confirmed at 8192: 1.446e-4)
 24    8.804e-5    PASS
 23    7.049e-4    PASS
 22    1.569e-4    PASS
 21    8.988e-4    PASS   (non-monotone inside the band)
 20    4.258e-4    PASS   ← frontier
 19    1.144e-3    FAIL   kl alone (14% over)
 18    8.906e-4    FAIL   top10_mass alone (0.1042) — kl passes
 16    4.962e-3    FAIL   kl + route_mass
 13    8.001e-3    FAIL   kl + route_mass
  6    1.665e-2    FAIL   kl + top10_mass + route_mass
  1    4.665e-2    FAIL   kl + top10_mass + route_mass
 26    1.161e-4    PASS   (Q8_0; Q6_K also passed individually at
                           8.53e-4 @ 8192 — see below for why the
                           map still prefers Q8_0 here)
```

The failure mode rotates at the edge (L18/L19 fail single criteria)
and enriches with depth. L18/L19 are the balanced-profile seeds.

## Composition is the finding

Individually admissible layers do not compose, and no scalar
correction fixes that:

```text
map                       measured    raw Σ      α       verdict
V1 {20-25:Q8, 26:Q6}      2.489e-3    3.2e-3     0.77    FAIL kl
V2 {22-25:Q8, 26:Q6}      1.110e-3    ~2.0e-3    0.55    FAIL kl
V3 {22,24,25,26:Q8}       1.124e-3    5.92e-4    1.90    FAIL kl  (super-additive)
V4 {24,25,26:Q8}          2.262e-4    4.35e-4    0.52    PASS → 8192 PASS
```

* V2 and V3 land nearly identical composed KL with raw sums 3.4×
  apart: the composed tail is dominated by interaction, not member
  costs.
* V3 − V4 attributes it: **adding L22 to the late band costs
  ~9.0e-4 composed against 1.57e-4 solo (~6×)**. The mechanism is
  the routing cascade — L22's hidden-state displacement flips
  routes from L23 onward in every map containing it.
* **L26 composes almost free** (last routed layer — no downstream
  router to amplify it), which is why the map takes Q8_0 there
  (1.16e-4) over the individually-cheapest Q6_K (8.53e-4): *the
  best representation for a scope in isolation need not belong to
  the best admissible map.* A per-layer recipe cannot discover
  this; only a composed gate can.
* Diagnostic→8192 drift on the winner was +84% (2.26e-4 →
  4.15e-4): the ~7e-4 promotion margin for a 1e-3 gate is not
  conservatism theatre.

## Reproduction

```text
bank      scripts/kimi_quality_bank_export.py  (deterministic; verify
          per-file SHA256 against provenance SHA256SUMS.json)
compile   LARQL_Q6_MAP="24-26:Q8_0" → represent/compile_real_tests.rs
quality   q2a_teacher_forced (LARQL_Q2A_SEQUENCES=8 diagnostic,
          256 authority)
speed     q2a_decode_bench (both arms in-process, interleaved)
evidence  ~/chris-models/kimi-quality-bank-provenance/
```
