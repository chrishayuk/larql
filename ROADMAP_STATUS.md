# Roadmap Status

Canonical rollup for the next execution slice. Keep the detailed design in
`ROADMAP.md` and crate-local roadmaps; use this file to answer "what is active
now?" without rereading every crate document.

Last updated: 2026-05-16

## Recently shipped (delta since last update)

- **Cross-engine forward-pass correctness gate** (2026-05-16): `larql shannon verify` CLI + multi-arch sweep `scripts/diagnose_models.py` + CI workflow `.github/workflows/shannon-verify.yml`. Four config-loading bugs surfaced and fixed in `larql-models` (rms_norm_eps not parsed; Gemma 3 per-layer-type rope_scaling missing; llama3 rope_scaling missing; StarCoder2 norm_epsilon alias). 7/9 archs in the sweep PASS at <0.5% bits/char vs HF F32 with no env-var overrides. See [`docs/diagnoses/shannon-cross-engine-divergence.md`](docs/diagnoses/shannon-cross-engine-divergence.md).

## Active Sequence

| Order | Item | Status | Owner | Exit criterion |
|---:|---|---|---|---|
| 1 | V0 aim-validation harness | started | `bench/aim-validation`, `scripts/aim_validation.py` | V1-V4 runs share one model/prompt/metric matrix and emit comparable JSON records. |
| 2 | V1 hash routing across all layers | queued | experiments + `larql-inference` | Per-layer top-k table and end-to-end divergence/tok/s report across the cross-arch matrix. |
| 3 | V2 FP4 generality | queued | experiments + `larql-vindex`/`larql-compute` | FP4-friendliness report by architecture/layer with QAT-required thresholds flagged. |
| 4 | C10 CPU baseline bench | started | `larql-cli`, `bench/` | `larql bench --cpu --output json` works and quant-matched llama.cpp CPU baseline is recorded; next exit is KV-cached CPU Q4K decode plus cross-arch repeats. |
| 5 | MI4/T7 trace truthfulness gate | queued | `larql-inference` | TRACE final residual/logit parity pinned for WalkFfn and patched-vindex paths, then Q4K/MoE. |
| 6 | R6 depth-fraction probe API | queued | `larql-inference`, `larql-models` | Stable probe API available before MTP3 layer-choice validation. |
| 7 | MTP1-MTP2 | queued | `larql-models`, `larql-vindex`, `larql-inference` | Gemma 4 assistant drafter loads; verify-loop decode exists before activation-feedback work. |

## Current P0/P1 Boundaries

| Area | Decision |
|---|---|
| Highest leverage | Run V1-V4 aim-validation before expanding long-term CPU/MoE engineering. |
| GPU credibility | Keep D-ATTN-MTG, D-PREFILL-MM2, D-METAL-PLE, and MTP on the baseline-credibility path. |
| CPU credibility | C10 comes first because the CPU track cannot enforce the 10% threshold without measurement. |
| Multi-machine MoE | Stays P2 unless a specific experiment or frontier-scale release re-promotes it. |
| Production-engine features | Continuous batching, PagedAttention, broad OpenAI surface, MCP, and thinking toggles stay deferred unless an experiment needs them. |

## Drift Checks

- If a crate roadmap says an item shipped, but this rollup still says queued,
  update this file in the same change.
- If a benchmark number changes in `README.md`, record whether it updates a
  baseline JSON, a roadmap claim, or both.
- If V1, V2, V3, or V4 fails, update the achievability table in `ROADMAP.md`
  before starting dependent engineering.
