# larql-compute-metal

The Metal GPU backend: 199 source files, 61 shader modules carrying 111
`kernel void` entry points, 31 examples and 63 integration-test binaries.

**There is no `metal` feature.** The crate is gated on
`target_os = "macos"` and is pulled in by the `gpu` feature of its
consumers (`larql-cli`, `larql-inference`, `larql-vindex`, `larql-kv`) —
and `gpu` is **on by default** in `larql-cli`. Any command line reading
`--features metal` predates ADR-019 and fails; there is nothing to
enable.

Metal is a first-class peer, not a special case (ADR-0022): this crate is
the same shape a future `larql-compute-vulkan` would be — its own crate,
implementing the same trait surface, owning its kernels.

## Why this file exists

This crate had no documentation for its first three months. The
consequence was not absence but *misdirection*: `larql-compute`'s docs
kept describing kernels, shaders and dispatch policy that had moved here,
so roughly half of that crate's documented paths pointed at code that was
no longer in it, and readers had no way to tell which half. A 2026-08-22
audit found the whole `crates/larql-compute/src/metal/…` tree cited
across ~20 documents including two `Status: Accepted` ADRs.

If you are adding to this crate, document it **here**. The rule the
audit produced: a doc lives in the crate that owns the code it describes.

## Layout

```
src/
  shaders/     61 modules of embedded MSL, one per kernel family
  kernels/     KernelHandle / pipeline construction and the registry
  ops/         encode-side primitives (one function per dispatch)
  stages/      composed stages (qkv_proj, ffn, attention, …)
  decode/      the served decode path: token loop, KV setup, head
  lowering/    VINDEX3 plan → Metal execution (attention, ffn, stack,
               head, nvfp4, profile)
  moe_gpu_route/   GPU-resident MoE routing: router → top-K → descriptor
                   gather → grouped experts → combine, all on device
  moe_zero_copy/   expert dispatch over registered mmap regions
  moe_descriptor/  the expert descriptor table (identity as GPU data)
  buffers/     the buffer cache and residency
  trait_impl/  ComputeBackend / MatMul / KvDispatch impls
  diag/        profilers and benches that are NOT on the decode path
  cb_status/   command-buffer completion checking (see below)
```

Two module boundaries are load-bearing rather than tidy:

- **`diag/` is never on the decode path.** Anything that samples
  timestamps drains the pipeline; keeping it separate is what makes the
  production encoder one uninstrumented encoder.
- **Every command-buffer wait refuses on failure.** A raw
  `wait_until_completed()` returns for a failed or ignored buffer exactly
  as for a finished one, with stale bytes in the outputs, so a step that
  only waited would report garbage at full speed (#229). The rule is that
  any number this crate contributes to a benchmark comes from an execution
  that actually completed: `cb_status::wait_checked` hands the failure to
  callers with an error channel (`GroupedError::CommandBufferFailed` on
  the Kimi/MLA/KDA and grouped-expert paths, which refuse before any cache
  advances), `cb_status::wait_or_abort` ends the step for callers without
  one (a `None` there would be taken as "fall back to the CPU" and the
  fallback's number reported as the GPU's), and tests `.expect` it.
  `cb_status/tests.rs` scans the source tree for both a naked wait and a
  discarded result. A real fault cannot be produced deterministically, so
  the refusal paths are witnessed through `cb_status::inject_fault_at_for_test`.

## Operator controls

Every one of these is read once and defaults to the shipped behaviour, so
an unset environment is the production configuration. They exist as A/B
arms — the control for a measurement, not tuning knobs.

| variable | effect |
|---|---|
| `LARQL_GPU_ROUTE=1` | GPU-resident MoE routing (router, top-K, descriptor gather, combine all on device) |
| `LARQL_NVFP4_KERNEL=<arm>` | pick an NVFP4 GEMV arm; default `x2` (two rows/lane sharing X loads) |
| `LARQL_NVFP4_FUSE=0\|seg` | disable projection fusion, or keep only the Q/K/V + gate/up segment fusion |
| `LARQL_MXFP4_EXPERT_X2=0` | fall back from the default `mxfp4g_split_lut16_vec_x2` expert kernel |
| `LARQL_MXFP4_EXPERT_GU=1` | opt into the fused gate+up expert arm (measured null; retained as a control) |
| `LARQL_MXFP4_EXPERT_DC=1` | opt into fused down+combine — **measured a 3.6% regression on AC**, kept as a control only |
| `LARQL_SPIN_WAIT=1` | spin on command-buffer status instead of blocking (obsoleted by one-CB-per-token; burns a core) |
| `LARQL_RESIDENCY_SET=1\|2` | explicit `MTLResidencySet` arms — **refuted**, no effect; retained so the refutation stays reproducible |
| `LARQL_EXTRA_BARRIERS=N` | dose-response control: insert N empty commit+wait pairs per layer |
| `LARQL_ABLATE_MOE=bias,act,combine` | in-situ tail ablation. Pair with a semantics-preserving fusion arm — an ablation that removes a dependency edge **overprices** the component |
| `LARQL_MOE_INLINE_DIAG=1` | name the first unmet merged-CB precondition per layer |
| `LARQL_SKIP_OUTER_NORM=1` | skip the outer norm (diagnostic) |

Retained-but-refuted arms are deliberate. A refuted arm that has been
deleted cannot stop the same idea being re-proposed next quarter.

## Building and testing

```bash
cargo build --release -p larql-compute-metal
make larql-compute-metal-test              # lib tests, --test-threads=1
make larql-compute-metal-ci                # fmt + lint + test + coverage
make larql-compute-metal-coverage-summary  # coverage gate (96% total floor)
```

`--test-threads=1` is not optional: many tests set process-global env
vars, and the default parallel runner races on them.

The toolchain is pinned by `rust-toolchain.toml` (1.98.0). CI installs
newest-stable, so before the pin a local clippy could pass against lints
CI would fail on.

## Measuring anything in this crate

Read `bench/prompts/README.md` first — it is the protocol of record. The
short version, all learned the expensive way:

1. **AC power, full charge, idle GPU.** On battery the same probe reads
   roughly half speed. Bulk-charging is not the same as charged.
2. **Warm before every arm.** This GPU accelerates hugely under sustained
   load: an unwarmed single-dispatch arm read 150.5 µs where the warmed
   arm read 39.5 — a 3.8× fake that fabricates clean-looking curves out
   of nothing but the frequency ramp.
3. **Pair every arm with an adjacent baseline.** A global open/close
   bracket can read −36% drift while paired measurements sit inside 1.3%.
4. **The end-to-end decode instrument reproduces to ~±6% across
   sessions.** No sub-6% claim is banked from one block, however clean
   its internal brackets — an interleaved A/B/A/B with a 0.48% control
   bracket and self-consistent arms still produced a false +2.4% that
   replication erased.
5. **If a kernel probe predicts a delta below that floor, e2e cannot
   adjudicate it.** A measured effect several times larger than its own
   model is evidence of an instrument problem, not of a missing cost
   term.
6. **Score in per-stage GB/s ratios, not tok/s.** Per-stage bandwidth
   reproduces to 0.4–1.9% where wall-clock wobbles ~6%.

`larql vindex3 exec --backend metal-lowered --generate N --profile` is
the stage profiler. Its sampling drains the pipeline at every stage
boundary, so judge *throughput* from an unprofiled run and *attribution*
from a profiled one — the two are not the same number.

## Known-open

- Coverage sits below the 96% total floor; `backend/mod.rs`,
  `lowering/{ffn,stack}.rs`, `moe_gpu_route/encode.rs` and
  `trait_impl/matmul.rs` are under the 90% per-file default.
- `crates/larql-compute/ROADMAP.md` §F1–F22 is the capability checklist:
  F17 and F21 remain open, the rest are fixed.
- `docs/metal-kernel-capabilities.md` is a useful kernel *reference*
  (§1–§6) but its *findings* half is stale — 20 of 22 are fixed.
