# Hardening backlog

Open remediation items from whole-codebase and subsystem reviews, extracted
from `ROADMAP.md` on 2026-08-10. **These are tracked work, not history** —
each review's ordered action list still contains unclosed P0/P1 items, and
per-crate items are mirrored into the crate-local roadmaps. Closed items
belong in [`CHANGELOG.md`](../CHANGELOG.md).

Reviews collected here: 2026-05-28 (whole codebase), 2026-06-12 (follow-up),
2026-07-22 (DEC readiness), 2026-07-30 (vindex + walk-FFN), 2026-07-31
(extraction tensor coverage), 2026-08-05 (compute-layer hygiene), plus the
standing cleanup / consolidation track.

---

## Codebase hardening (review 2026-05-28)

Whole-codebase multi-agent review (17 crates, ~415K LOC; one reader per crate +
adversarial verification of every high/critical finding). Full record:
[`docs/audits/codebase-review-2026-05-28.md`](../docs/audits/codebase-review-2026-05-28.md).
Verdict: mature, defensively-engineered; exposure is concentrated and thematic,
not pervasive. `cargo clippy --workspace --all-targets` is clean (2 trivial nits).
Per-crate items below are mirrored into each crate-local roadmap.

Ordered actions (✅ = also confirmed by hand):

1. **Make `FfnBackend::forward` fallible** (P0) — the trait returns an infallible
   `Array2<f32>`, forcing process-abort on served paths. Convert
   `larql-inference` `cached.rs:123,200`, `hidden.rs:38`, ✅`http.rs:519` and
   `larql-compute` `moe/forward.rs:191,211` to `?`-propagation into the existing
   `GenerateError` channel. Highest leverage — removes the top serving-abort
   class. [larql-inference, larql-compute]
2. ✅ **Bound the Metal KV cache** (P0) — `kv_attention.rs:186-187` (+ `attn_fused`,
   `kv_append_attend_fused`) write `K_cache[pos*total+tid]` with no `pos<max_seq`
   clamp; sessions exceeding the 4096-row cache write OOB on the GPU during
   normal decode. Add the position guard and extend `ensure_prompt_fits` to
   `prompt_len + max_tokens`; expose cache sizing to the caller. The only
   verified memory-corruption bug. [larql-compute-metal — no crate roadmap]
3. **Fix `larql-python` soundness gaps** (P0) — `trace_py.rs:14-28` raw
   `*const ModelWeights`/`*const Tokenizer` is use-after-free across `del model`
   (give `PyResidualTrace` a `Py<PyWalkModel>`); `walk.rs:207-223` zero-copy
   embed `Vec::from_raw_parts` lacks the length check its sibling paths use.
   [larql-python — no crate roadmap]
4. **Validate router layer ranges + wire server eviction** (P1) — `larql-router`
   `routing.rs:237` builds an unbounded route table from gRPC-announced ranges
   (clamp to model depth before `rebuild_route_table`); `larql-server`
   `session.rs:184` + `ratelimit.rs:83` never evict (dead eviction logic).
   Memory/DoS class. [larql-router, larql-server]
5. **Shared NaN-safe top-K/sort helper** (P1) — route the ~10
   `partial_cmp().unwrap()` sites (vindex router:107/lm_head:322/gate_store:330,
   core graph:278/walk:35/pagerank:19, cli parity:1119, python vindex:847,1432)
   and `larql-lql`'s four `embed.row()` callers through bounds-checked helpers.
   [larql-vindex, larql-core, larql-cli, larql-lql]
6. **SQL expert UTF-8 offset bug + typed cross-crate contracts** (P2) —
   `larql-experts/sql/src/lib.rs:161` slices the original string with offsets
   from an uppercased copy (panic on non-ASCII SQL); use `char_indices`. Then
   consider typing the `*const f32` reinterpret, positional-QKVO
   (`attn_data[1]/[2]`), and `per_layer_ffn_key` conventions to stop silent
   drift. `larql-router-protocol`: `None` fingerprint disables TLS verification.
   [larql-experts — no crate roadmap, larql-router-protocol — no crate roadmap]

Hygiene (separate from the sweep): 2 clippy nits in `larql-cli` (unused
`ProjectorWeights`, dead `total_tiles`); coverage below the ≥90% floor on
`larql-inference` (70.7%) and `larql-cli` (12.0%).

### Follow-up review (2026-06-12)

Diff review of the in-flight C10/FR3 changes + fresh whole-workspace sweep
(10 subsystem readers + adversarial verification; several headline claims
refuted — GGUF overflow, kernel release-mode bounds, `attn_fused` overflow
all died under verification). Full record:
[`docs/audits/codebase-review-2026-06-12.md`](../docs/audits/codebase-review-2026-06-12.md).
Items 1 and 5 of the 2026-05-28 list were re-confirmed still open
(`cached.rs:123,200`/`hidden.rs:38` panics; python `vindex.rs:847` NaN sort)
— they stay tracked there, not duplicated here.

Ordered actions:

1. **Sanitize `model_id` in shard loader** (P0, security) —
   `larql-server/shard_loader.rs:30` joins router-supplied `model_id`
   (`announce.rs:544`) into the store path unvalidated; `../` escapes the
   shard dir (tar unpack itself is safe, tar 0.4.45). Reject path
   separators / `..`. Follow-on (P2): grid non-join RPCs (`drain_server`,
   `assign_range`, `grid/service.rs:114`) don't require the grid key.
   [larql-server, larql-router]
2. **Check Metal command-buffer status** (P0) — all 77
   `wait_until_completed()` sites read buffers with no `status()`/`error()`
   inspection (e.g. `ops/full_pipeline/dispatch.rs:456,783`); a failed GPU
   command yields stale data straight into logits. Add a `wait_and_check()`
   helper and migrate. Cheap insurance against the next phantom-drift hunt.
   [larql-compute-metal — no crate roadmap]
3. **Route the 2 hardcoded dispatches through `KernelHandle`** (P1, latent
   but a 3×-historical bug class) — `decode_hybrid.rs:388-391` hardcodes
   256 threads/TG while `q8_matvec_pipeline` is already a `KernelHandle`
   carrying the geometry; `stages/qkv_proj.rs:241` takes a raw
   `ComputePipelineState` so it can't consult one. Correct today, silently
   fast-but-wrong on any shader geometry change. [larql-compute-metal]
4. **Corrupt-vindex load robustness** (P1) — `larql-vindex
   format/load.rs:81,293` index `gate_slices[info.layer]` with
   `info.layer` straight from `index.json`, no bounds check (panic on
   corrupt manifest; validate `< num_layers` → `VindexError::Parse`);
   `load.rs:317` defaults missing manifest `offset`/`length` to 0,
   masking the real error. [larql-vindex]
5. **Validate Q4K lm_head buffer size** (P1, from the diff review) —
   `larql-kv/generation.rs:657` + `larql-inference
   forward/predict/dense.rs:189` never check buffer len vs
   `vocab_size × bytes_per_row`; truncated weights panic mid-decode,
   padded ones decode garbage logits. One length check → clean f32
   fallback. [larql-kv, larql-inference]
6. **Release the GIL in larql-python** (P1) — zero `allow_threads` in the
   crate; `predict`/`trace`/`generate_with_hooks`/`infer`/`infer_trace`
   block all Python threads for whole forward passes. Wrap compute in
   `py.allow_threads`. (NaN sort at `vindex.rs:847` already tracked as
   2026-05-28 item 5.) [larql-python — no crate roadmap]
7. **Env-flag registry** (P1) — 145 distinct `LARQL_*` flags, ~18
   documented; accepted values already diverge (`LARQL_Q4K_ASM=true` works,
   the three new C10 flags accept only `"1"` — a bench run with `=true`
   silently measures the wrong config). Route flags through the
   `larql-compute/src/options.rs` taxonomy + generate `docs/env-flags.md`.
   [workspace]
8. **Diff-review cleanups before/with the C10 commit** (P2) — fold
   `hidden == 0` into the padded-down guard (`larql-compute
   kquant_forward/cached.rs:861` + twin); extract the duplicated ~35-line
   padded-down block into one `larql-compute` helper with a reusable
   scratch buffer (kills the lockstep-comment hazard + ~69 KB/token alloc
   on 26B); drop the unnecessary `relations.clone()`
   (`larql-lql edges.rs:186`); length-check `labels`/`counts` at load
   (`relations.rs:35`); OnceLock the `LARQL_FR3_EXPLICIT` read
   (`edges.rs:279`). [larql-compute, larql-inference, larql-lql]
9. **Forward-pass loop unification** (P2, ADR first) — five parallel
   layer-step loops in `larql-inference/vindex/kquant_forward/`
   (`hidden`/`prefill`/`decode_step`/`decode_step_direct`/remote-FFN) each
   repeat the same sentinel logic; every stepping change lands 5× or
   numerics silently diverge. Big-ticket; cuts across the C10-hot files,
   so sequence behind the current residency arc. [larql-inference]
10. **Dead weight** (P2) — 4 unreferenced Metal shader modules
    (`graph_walk_knn`, `q4_sparse_matvec`, `turboquant_{encode,decode}`)
    need an ADR-017 retention rationale or deletion; `model-compute` crate
    has no second consumer (no-speculative-extraction policy); `larql-inference`
    `test_utils.rs` (1,228 lines) ships as public API. [larql-compute-metal,
    model-compute, larql-inference]
11. **Serving posture** (P2, plausible-not-verified) — document or fix:
    streaming completions serialize on the weights guard
    (`completions.rs:302`) with no per-request timeout (`:366`); no
    graceful drain on shutdown (`bootstrap.rs:1255`); grid join stream has
    no malformed-message rate limit (`grid/service.rs:121`). [larql-server,
    larql-router]

### DEC-readiness review (2026-07-22)

Targeted review of the **DEC data-plane** ahead of the DEC funnel programme
([`docs/dec-funnel.md`](../docs/dec-funnel.md)) — the code the programme runs on
rented x86 marketplace hosts, against non-Gemma models, over adversarial
links. Four parallel readers (security, hardcoding/config, modularity,
performance), verified findings only. Full record:
[`docs/audits/dec-readiness-review-2026-07-22.md`](../docs/audits/dec-readiness-review-2026-07-22.md).
Verdict: structurally sound and the wire decoders are mostly hardened, but a
**silent-corruption cluster** (produces a number, the number is a lie) is the
dominant risk because the whole programme is a measurement exercise. The B-row
f32/f16/i8 serving path is well-built; only the Q8K path — the wire DEC prefers
— does not batch. Work through in the order below (roughly DEC-stage sequencing).

**Batch A — silent corruption + the security HIGH (before any claim-bearing run): ✅ DONE (2026-07-22)**

1. ✅ **Q8K batched compute** (P0, corrupts C1/C6, gates DEC-0) — `walk_ffn/q8k.rs:199`
   ran B same-layer rows as B independent matvecs, each re-streaming the full
   layer's weights, so the Q8K batch curve was ~linear by construction. Fixed:
   the handler now groups request entries by layer; groups of >1 dequantise to
   f32 and run ONE batched GEMM through `kquant_ffn_forward_layer` (preserving
   the Q8K upload win, amortising weights across rows); singleton groups keep
   the existing single-row Q4K×Q8K kernel unchanged (no batching problem there,
   and it avoids dequantising gate/up on the latency-critical single-token
   decode path). Numerical equivalence between the two paths is pinned by
   `walk_ffn_kquant_layer_q8k_batched_gemm_matches_per_row_single_kernel`.
   [larql-server, larql-inference]
2. ✅ **Multi-layer decoder allocation bomb** (P0, security) — `Vec::with_capacity(n)`
   from an attacker u32 → one 16-byte packet aborts the server
   (`moe_remote/multi_layer_wire.rs:105,146,211,254,292`). Regression from the
   repo's own `max_possible_entries` guard (PR 104). Fixed: mirrored the guard
   at every task/result/expert-count allocation site, plus inside the shared
   `read_f32_slice`/`read_i16_slice` helpers so `hidden`/`nb`-derived lengths
   are bounded before any allocation, covering both `decode_multi_layer_request`
   and the client-side `decode_multi_layer_response`. 8 new
   `rejects_impossible_*_before_allocating` regression tests. [larql-inference]
3. ✅ **Shard failure zero-fills FFN output** (P0, silent generation corruption) —
   `sharded.rs:117` returned zeros on a panicked/unowned shard and decode
   continued on a corrupt hidden state. Fixed: `forward_predispatch_all` now
   panics loudly on an unowned layer or a shard's transport failure (propagating
   the worker thread's panic via `resume_unwind` instead of swallowing it),
   matching `RemoteWalkBackend::forward`'s existing panic-on-error convention.
   [larql-inference]
4. ✅ **Down-proj ignores its format tag on the Q8K fast path** (P0, gates
   DEC-4/6) — `kquant_forward/walk_ffn.rs:135` fed `ffn[2].0` into a Q4_K-only
   kernel without checking `ffn[2].1`; a non-Q4_K down slab (Inkling/K3) would
   have decoded garbage. Fixed: the fast path now additionally gates on
   `ffn[2].1 == "Q4_K"`, falling back to the format-aware `dequantize_matrix`
   path otherwise. Fixed in both the `larql-inference` copy (the live serving
   path) and the `larql-compute` twin (same bug, not yet wired to a serving
   path — see item 4g). Regression:
   `walk_ffn_kquant_layer_q8k_rejects_down_slab_with_non_q4k_format_tag`.
   [larql-inference, larql-compute]
5. ✅ **x86 scalar-fallback is silent** (P0 *observability*, gates DEC-0.5) —
   `q4k_q8k_gate_up_into` (:1377) and `q6k_q8k_matvec_into` (:2119) have no AVX2
   branch; the serving-path doc-comments falsely claimed "NEON/AVX2". Building
   the AVX2 kernels remains **C-ladder** work (not done here). Fixed: added
   `larql_compute::cpu::ops::q4k_q8k_dot::kernel_class_summary()`, logged once
   at server startup (`larql-server/bootstrap.rs`), and corrected the false doc
   comments on `q4k_q8k_gate_up_into` and the `q8k.rs` module doc — so no DEC
   number is ever recorded on an unlogged scalar path. [larql-compute, larql-server]

**Batch B — fleet/config landmines (before the x86 + Linux arms):**

6. **`127.0.0.1` announce on `--join`** (P1, breaks multi-host grid) — refuse a
   wildcard host without `--public-url`, or detect the outbound IP
   (`bootstrap.rs:1222`). [larql-server]
7. ✅ **Backend factory + capability dispatch** (P1, unblocks x86 + pre-work for
   G-ladder) — DONE 2026-07-22 (except the capture-portability doc note, folded
   into #8's script work). `larql_compute::backend::factory` adds
   `BackendKind` (+`FromStr` for a future `--backend`/`DEC0_BACKEND` string) and
   `backend_from_spec(kind, registry)` with injected constructors (ADR-019: the
   trait crate names no backend crate); `larql-cli/src/backend_select.rs` builds
   the registry once, and all 7 `if metal` cfg copy-paste sites collapse onto it
   (`run_cmd.rs` ×2, `bench/remote_ffn_runtime.rs`, `bench/local_runtime.rs`,
   `dec_bench/capture_runtime.rs`, `shannon_cmd.rs`, `walk_cmd.rs`). Semantics
   tightened: an explicit `--metal` with no usable device now errors loudly
   instead of silently benching on CPU. Dispatch de-`bool`ed: the remote-MoE
   fork probes `supports(Capability::DecodeMoe)` on the constructed instance,
   and run_cmd's experts module fixes TWO latent bugs — `metal_ready_for_q4`
   probed `default_backend()` (always CPU post-ADR-019, so the check was
   vacuous) and `Strategy::MetalQ4K` then ran `layer_graph::generate` on a
   fresh `default_backend()` (CPU) — the constructed backend is now stored in
   `Runtime` and probed via the canonical `PrefillQ4 && DecodeToken` pair.
   [larql-cli, larql-compute]
8. **`--metal` / `--backends metal` hardcoded for x86** (P1) — `DEC0_BACKEND`
   env in `scripts/dec0-loopback.sh:80,97`, platform-conditional `--backends`
   default (`bench/args.rs:26`). Couples to #7. [larql-cli]
9. ✅ **`SKIP_MOE` vs `LARQL_SKIP_MOE` name split** (P1, corrupts the anchor's
   ceiling arm) — DONE 2026-07-22. One canonical prefixed name for all three
   unprefixed vars (`LARQL_SKIP_MOE`, `LARQL_SKIP_OUTER_NORM`,
   `LARQL_DECODE_DEBUG`), read through shared accessors in
   `larql_compute::options` (`skip_moe_enabled` / `skip_outer_norm_enabled` /
   `decode_debug_enabled`) that honour the historical unprefixed names as
   deprecated aliases with a one-time stderr warning. The grid path's
   `GridRuntimeConfig` now reads the same accessor as the local path, so the
   DEC-0 ceiling arm measures one thing regardless of which name the operator
   types; dec-funnel.md DEC-0 anchor note updated to the canonical name
   (README already used it). Alias behaviour pinned by
   `unprefixed_legacy_aliases_still_enable_their_flags`.
   [larql-inference, larql-compute, larql-compute-metal, docs]
10. **DEC deployment auth posture** (P1, security) — the data plane is open
    unless `--api-key` is set (`/v1/shard` streams the whole vindex as a tar);
    router admin RPCs (`drain_server`/`assign_range`) and the grid port are
    unauthenticated (`grid/service.rs:386,397,455`; overlaps 2026-06-12 item 1
    follow-on). Decide: mandatory data-plane auth off-loopback, or private-network
    binding as a documented DEC deployment rule. Constant-time the gRPC grid-key
    compare (`service.rs:105`) while here. [larql-server, larql-router]
11. **Timeout defaults + no-op grid-LAN timeout** (P2) — the 30/60/120s defaults
    assume 26B+LAN and will 504 on Inkling cold-start (note at DEC-4/5
    provisioning); `grid_lan_runtime.rs:179` timeout is a `let _ =` no-op (wire
    it). Arch-tag the grid-regress baselines (`bench-grid-regress.sh:35`).
    [larql-cli, larql-inference]

**Batch C — structural pre-work (schedule per-ladder, not a DEC-0 blocker):**

12. **Compute admission control** (P1, protects C3/DEC-2) — ~192 concurrent
    multithreaded-BLAS `spawn_blocking` tasks at 4 clients × 48-layer fan-out
    look like tier saturation but are oversubscription. Semaphore sized to
    physical cores + `OPENBLAS_NUM_THREADS=1` for the serving build.
    [larql-server]
13. ✅ **q8k endpoint drain/heartbeat/latency blindness** (P1, breaks C7 router
    demo) — DONE 2026-07-23, extended to the whole expert surface per the
    expert-serving review (§1d): shared `track_model_request` helper
    (`RifGuard` + `requests_total`) on the q8k walk-ffn handler AND all
    expert endpoints (single/legacy-batch/layer-batch×2/multi-layer×2), with
    `layer_latency_tracker.record` on q8k walk-ffn and the expert batch
    handlers. See `docs/audits/expert-serving-review-2026-07-23.md`.
    [larql-server]
14. ✅ **dec_bench `Endpoint` seam + routing capture** (P1, gates the
    routed-experts arm that gates the C1-on-MoE verdict) — DONE 2026-07-23,
    preceded by a three-reader expert-serving review
    (`docs/audits/expert-serving-review-2026-07-23.md`) whose Phase-A server
    hardening + pre-measurement perf batch landed first (batch handlers 400
    on unresolvable experts; q8k shape validation; owned-entry
    `per_expert_bytes` probe; bulk LE codecs off the reactor thread; stale
    parallelism docs corrected). Built: `Endpoint` enum (walk-ffn ×2 +
    experts-multi-layer ×2 — path/frame/decoder/`server_ms`/denominator per
    variant); capture `--routing` flag with additive pool sidecars
    (`raw.bin`/`normed.bin`/`routing.bin`, manifest stays v1, the shipped
    330M pool still replays the dense arms); routing computed at the capture
    sink via the now-`pub` `build_moe_router_weights` + client router,
    gated by a router twin-parity test (inference `route()` ≡ compute
    policy pipeline, 4 shapes); per-point batch-aware denominators
    (`weight_bytes_tok_naive` primary — server streams per-row, no
    cross-row sharing — + `_union` as the DEC-3 bound) and
    `dec/endpoint(_code)`/`dec/experts_union_frac`/`client_rayon_threads`
    in the pulse/run record; warmup non-zero-response guard (§1a class).
    [larql-cli, larql-inference, larql-server]
15. **Server expert dispatcher** (P2, before G4 cuda-experts) — extract one
    `run_experts(state, backend, …)` from the per-handler Metal/CPU branches
    (`q8k.rs:107`, `grpc_expert.rs:178`, `expert/{layer,multi_layer}_batch.rs`).
    [larql-server]
16. ✅ **Wire consolidation** (P2) — DONE 2026-07-24, the trigger having arrived
    early (DEC-1A's asymmetric codecs + timing field, not DEC-6a). The dense
    binary frame is single-sourced in `larql-inference` `ffn/remote/codec.rs`
    (encoder+decoder+constants; server `binary.rs` is a shim; router imports);
    every CT string and `BATCH_MARKER` declared once; byte-identical wire
    pinned by encode-decode-reencode tests; all allocation-bomb guards moved
    verbatim; `call_q8k_layers` byte-counter gap fixed. Three extensions then
    landed on the consolidated seam same-night (ADR-0025): the header-gated
    `serve_us` timing trailer, independent inbound/return wire formats
    (f16/i8 REQUEST encodings — previously f32-only — with `Content-Type`=in
    / `Accept`=out decoupled), and the `dec-bench drift` C6 fidelity
    instrument. [larql-inference, larql-server, larql-router, larql-cli]
17. **MoE parity seams + hot-path cleanups** (P2) — make `build_moe_router_weights`
    `pub` and share the combine math before the DEC-6b KDA/LatentMoE port
    (`hidden.rs:93` vs `core.rs:111`); `model.patched` arc-swap so compute
    doesn't hold the read lock across FFN (C3 shared-tier landmine); drop the
    4–6 full-buffer request-lifecycle passes (`core.rs:48,235,284`,
    `binary.rs:88`); gate `--release-mmap-after-request` on `requests_in_flight`;
    persistent client fan-out pool. [larql-inference, larql-server]

### Vindex + WalkFFN review (2026-07-30)

Subsystem review of `larql-vindex` (~51K LOC) and the walk-FFN engine
(`larql-inference/src/vindex/walk_ffn/`), merged with an external strategic
review of the architecture and a kernel deep-dive. Full record:
[`docs/audits/vindex-walkffn-review-2026-07-30.md`](../docs/audits/vindex-walkffn-review-2026-07-30.md).
Verdict: both subsystems structurally healthy (the storage layer and spec
crate are defensive engineering done right; the trait-dispatch refactor
paid off — FP4 cost zero kernel code), but four high-severity runtime bugs,
a silent-wrong-numerics cluster in the quantized walk paths (same
"produces a number, the number is a lie" theme as the DEC review), and
**no walk-vs-dense numerical parity test anywhere in the tree**.

**Status 2026-08-01: PROGRAMME CLOSED — 24 of 24.** Tiers 0–1 in full (2026-07-30,
incl. all four HIGHs); item 13 resolved with the finding inverted (the
exact-first gate chain is now actually wired — `enable_hnsw()` had been
leaking approximate selection into walk numerics); Tier 2 complete:
base+delta (16), forward/forward_observed split (15), runtime trace
emission (17), execution planner (18), two-stage selection (19) all
shipped; parity suite (20) landed with the per-file ≥90% coverage
pass; KnnStore unified at the retrieval-kernel level (21 — full arch-B
retirement explicitly gated in the spec, see the item); v1 conformance
contract (22) shipped 2026-08-01 (corruption suite + LE golden
vectors + `docs/conformance-v1.md`; perf benchmark protocol is a
documented follow-up in that doc); doc drift (23) closed 2026-08-01
(every number re-verified against its bench/experiment source — the
0.008 ms headline was the pre-2026-04-05 reduced-shape `vindex_bench`
example; extract-default contradiction resolved in favour of the code,
per surface; walk.md K=8092 kept — it is the literal harness constant,
now documented as such — and WalkFfn reframed as the
instrumentable/editable layer + CPU sparse path); hygiene (24) closed
2026-08-01, triaged per its own licence — done: generic-engine
vocabularies → data files behind a loud-fallback search chain, the two
deferred 16384→10240 fixes, the activation dispatch (27 sites) onto one
exhaustive helper, FFN component constants unified, 41 colocated tests
for `hnsw.rs`/`mutate`/`write_f32.rs` (97/96/93% line coverage);
documented remainder: the >250-line file splits (see the item).
Standing follow-ups carried out of the programme: server/lql
`try_apply_patch` migration, remote transport coverage harness,
logit-contribution trace field, walk-FFN thresholds surfaced into
`WalkFfnConfig`, HNSW level-0 graph fragmentation at n≳64 (new finding
from item 24's test pass — naive `add_connection` eviction orphans
nodes; recall@10 collapses to 0.16 at n=200 uniform), and the remaining
file splits (`huggingface/download/mod.rs` 1329, `patch/overlay.rs`
1071, `quant/convert.rs` 653).

Sequencing is interaction-driven: Tier 0's padded-stride fix **gates**
Tier 2's base+delta (the delta path leans on the same row-dot/sidecar
machinery, and GPT-OSS-20B hidden=2880 is K3 rung 1); the
`forward`/`forward_observed` split *is* the fix for the zero-activation
bugs (don't patch them twice); the planner enum subsumes the
wrong-capability-gate class but the live panic gets its two-line fix now.

**Tier 0 — correctness (small independent diffs, before any Q4K walk
claim on a non-256-aligned model):**

1. ✅ **Q4K cache padded-stride fix + non-aligned fixture** (DONE 2026-07-30) (P0, silent
   garbage) — `kquant_cache.rs:138-161` decodes assuming unpadded
   `[rows, cols]`; the writer pads each row's cols to 256
   (`write_kquant/ffn.rs:70`). Wrong FFN outputs, no diagnostic, on
   hidden%256≠0 models (GPT-OSS-20B 2880, Gemma3-1B 1152). The fix already
   exists in one of three copies (`kquant_forward/walk_ffn.rs:63-70`).
   Add a hidden=320 fixture — every current Q4K fixture is 256-aligned so
   the suite structurally cannot catch this class. Victims: parallel-down
   path, per-feature down accumulate, selector row norms. [larql-vindex,
   larql-inference]
2. ✅ **Q4_0 ladder gates on the wrong format → CPU panic** (DONE 2026-07-30) (P0) —
   `walk_ffn/mod.rs:405` admits Q4_0 data on `supports_quant(Q4_K)`;
   `CpuBackend` says yes but leaves `q4_matvec_pair_batch` defaulted to
   `None`, and `interleaved_q4.rs:58-62` unwraps it. Gate on Q4_0 / actual
   batch-kernel availability; unwraps → fallthrough. `interleaved_q4.rs`
   has zero tests. [larql-inference]
3. ✅ **Overlay gate cache poisoned by zero-width gate vectors** (DONE 2026-07-30) (P0,
   nondeterministic panic/wrong-scores) — `patch/overlay.rs:176-191`
   mixed-width guard misses `len==0`; `vindexfile/mod.rs:125` inserts
   `vec![]` gates on every INSERT, so the trigger is in-tree. Guard the
   zero-width case AND stop inserting empty gate vectors. [larql-vindex]
4. ✅ **Loader panics on malformed `index.json`** (DONE 2026-07-30) (P0) — `format/load.rs:81`
   and `:293` index `gate_slices[info.layer]` unchecked from parsed JSON;
   return `VindexError::Parse` per the crate's own stated standard.
   [larql-vindex]
5. ✅ **Override fallthrough** (DONE 2026-07-30 — routes to the extracted override-aware `weights_fallback` instead of erroring; step 10 honours overrides, so availability is preserved) (P1, stopgap until base+delta) —
   `mod.rs:333-339`: sparse returning `None` on an overridden layer falls
   through to override-blind whole-layer paths — the exact failure the
   module doc warns about. [larql-inference]
6. ✅ **Unaligned f32 transmutes (UB) + patch decode swallowing** (DONE 2026-07-30 — new `format/le_floats.rs`; `try_apply_patch` is the error-surfacing entry, `apply_patch` kept as an infallible wrapper that drops corrupt patches wholesale; migrating larql-server/larql-lql callers to `try_apply_patch` is a follow-up) (P1) —
   `patch/format.rs:202`, `quant/convert.rs:565`, `config/dtype.rs:60` →
   `from_le_bytes`/bytemuck (also fixes the native-endian `.vlp`
   portability gap); `overlay_apply.rs:86,122` must surface
   `decode_gate_vector` failures instead of applying meta-only half-state;
   the hand-rolled base64 decoder silently truncates trailing chars.
   [larql-vindex]

**Tier 1 — kernel-semantics campaign (one PR neighbourhood: make explicit
what's exact, approximate, observed, reconstructed):**

7. ✅ **Wire `activation_floor`** (DONE 2026-07-30 — `effective_activation_floor()` = max(user floor, named `ACTIVATION_NOISE_FLOOR`), applied on all three sparse accumulate loops, behavioral test) — documented, settable from
   `predict_cmd.rs:241`, read by nothing; the real threshold is a
   hardcoded `1e-10` ×3 (`sparse.rs:338,411,549`). [larql-inference]
8. ✅ **Name the 80% full-K threshold, align doc/code** (DONE 2026-07-30 — `walk_ffn/thresholds.rs` FULL_K_DENSITY 4/5 + PARALLEL_DOWN_MIN_HITS + GATHER_MIN_FEATURES; helper doc now states the [80%,100%) band is dense) —
   `helpers.rs:24` fires the dense gemm at `k >= intermediate*8/10` while
   docs say "K ≥ feature count"; fidelity-vs-K points above 0.8 density
   are secretly dense unless `force_walk`. Named const in config; consider
   true `k >= intermediate`. [larql-inference]
9. ✅ **`selector:fallback` trace suffix** (DONE 2026-07-30 — dispatch-trace entry + `selector_fallback_count()`) — `joint_gate_knn` silently
   degrades to GateOnly when norms/batched scores are missing; A/B sweeps
   can't currently be trusted. [larql-inference]
10. ✅ **Resolve the gather caveat** (DONE 2026-07-30 — STALE: the phrase dates from task #24's transposed-down striding; task #25's hard sidecar requirement (`down_features_q4k_layer_data(layer)?` + decline-without-sidecar pin) resolved it, validated vs dense at |err|/‖ref‖≈6e-3. Caveat deleted, history documented in `sparse_gather.rs`. Remaining issue on this path is the documented 0.15× full-forward perf collapse, not correctness) — `sparse.rs:450` says "experimental —
    not yet correct for production down" on a kernel production routing
    reaches (route-pool + sidecar). Stale comment (predates the
    feature-major sidecar?) → delete; live → opt-in flag. [larql-inference]
11. ✅ **Unify the NaN contract** (DONE 2026-07-30 — shared `selection_weight_cmp_desc` panics on NaN matching `top_k_by_abs`; 4 sites unified, `#[should_panic]` pins incl. a NaN-gate-scores mock through `joint_gate_knn`) — `top_k_by_abs` panics;
    `selector.rs:267,320` `unwrap_or(Equal)` scrambles silently. Pick one
    (also see the 2026-05-28 item 5 shared helper). [larql-inference]
12. ✅ **Delete the orphaned `larql-vindex/src/walk/` module** (DONE 2026-07-30) — no
    `mod walk;` anywhere, never compiles, stale `WalkFfnConfig` duplicate
    (left by `3944359b`). [larql-vindex]
13. ✅ **Decide the HNSW hot-path question** (DONE 2026-07-30 — the exact-first ordering is DELIBERATE (`735f570e` 2026-04-04 call-site comment; brute gemv break-even-or-better at walk N per `docs/ffn-graph-layer.md`/`benches/hnsw_decode.rs`; HNSW's 80–95% recall would break the exact-top-K selection-quality gates) **but it had never actually executed**: `impl GateLookup for VectorIndex` was missing the `gate_walk` override — the trait default's "Override in VectorIndex" comment dates to the same 2026-04-04 commit — so every `&dyn GateIndex` walk selection silently took the `None` default into `gate_knn`, and `enable_hnsw()` DID leak approximate HNSW into walk numerics (pin test caught it: exact `[1,19,30,0]` became signed-biased `[1,29,9,26]` on the f32 fixture). Fixed by wiring the intended chain, not HNSW: delegation shim in `index/core/gate_lookup.rs` + guarded `PatchedVindex::gate_walk` (declines on gate-overridden/tombstoned layers so the overlay-aware `gate_knn` merge stays authoritative); Q4K-only gates and patched layers still reach `gate_knn_q4`/`gate_knn` as before, so the MoE-expert HNSW win is preserved. `enable_hnsw()` doc now maps exactly which paths consult HNSW incl. the 2026-04→07 leak window; pinned by `gate_walk_ignores_hnsw_toggle`, `gate_walk_delegates_to_inherent_on_a_populated_index`, the 3 `PatchedVindex` gate_walk pins, and `walk_ffn_sparse_hot_path_ignores_enable_hnsw`) — verified: `gate_walk` is tried
    first (`sparse.rs:231,268`) and HNSW lives only inside the `gate_knn`
    fallback, so `enable_hnsw()` changes nothing whenever `gate_walk`
    succeeds. Intentional (brute gemv wins at these N) → document at
    `enable_hnsw()`; otherwise wire it. [larql-vindex, larql-inference]
14. ✅ **Tombstone semantics for Delete→Update + pinning test** (DONE 2026-07-30 — Update resurrects, matching Insert; pinned-None meta cleared when Update carries no replacement; oversampling named `BASE_KNN_OVERSAMPLE_FACTOR`=2 with 2×→4×→all-features escalation only on layers with tombstones; 7 regression tests) —
    Update never clears `deleted` (`overlay_apply.rs:102-138`);
    `feature_meta()` and `gate_knn()` disagree about the same feature.
    Also the 2× deletion-oversampling under-fill (`overlay.rs:426`).
    [larql-vindex]

**Tier 2 — capability (the strategic-review core, in this order):**

15. ✅ **`forward` / `forward_observed` split** (DONE 2026-07-31 —
    `FfnBackend::forward_with_activation` is GONE; the trait is
    `forward` (hot, never touches an activation buffer) +
    `forward_observed` returning `FfnActivations` (new module
    `larql-compute/src/ffn/observe.rs`): `Dense` for dense paths (the
    matrix is an intrinsic intermediate), `Sparse` per-position
    `(feature, activation)` pairs for exactly the K computed features,
    `Absent {reason}` for paths that observe nothing — the trait default,
    so unobserving backends (remote walk's fabricated `[seq,1]` zeros,
    MoE's output-as-activation, seven larql-kv/server stubs) now say so
    instead of inventing tensors. `WalkFfn` routes both entry points
    through one `forward_routed(.., Observe)` body — identical routing by
    construction; `Skip` mode threads through every walk path
    (sparse/gather/parallel/base_delta/weights_fallback +
    `sparse_compute`'s split plain/`_observed` API) so the old
    `seq_len × intermediate` zero-fill no longer exists on generation.
    The parallel Q4K down branch reports its REAL per-feature activations
    (the pinned all-zeros parity test flipped to assert bit-equality with
    the serial halves); an L1 hit serves `forward` but an observed call
    BYPASSES the cache read and recomputes (pinned); base_delta reports
    post-patch slot activations (new `base_delta_tests.rs`, incl. decline
    branches — 20%→95% file coverage). `run_ffn`'s capture arm densifies
    via `FfnActivations::into_dense()` (Absent → `None`, never zeros), so
    hooks/trace/server consumers kept their `Option<Array2>` shape;
    changed files ≥90% line coverage except the pre-existing
    network-debt pair `remote/http.rs` / `remote/sharded.rs` (12%→35%
    with new no-shard observation pins; rest needs a mock-server
    harness)) — activations
    become opt-in; sparse paths emit `(FeatureId, f32)` pairs instead of a
    dense `seq_len × intermediate` zero-fill. Subsumes (by construction)
    the parallel-path zero activations (`sparse.rs:283-371`) and the L1
    cache's fabricated-zeros hit (`mod.rs:367`), and removes the dense
    allocation from ordinary generation. [larql-inference, larql-server]
16. **Base-plus-delta patched FFN execution** (after item 1) —
    `y_patched = y_base + Σ_{i∈P}(contribᵢ_new − contribᵢ_old)` is exact
    and O(|P|) on top of the fast dense path; retires the
    override-forces-sparse cliff and makes editing production-viable.
    Exactness conditions: old-term subtraction through the SAME quantised
    row_dot bytes as the dense base (not f32-recomputed), old-down rows
    from the feature-major sidecar. Lands as a routing-ladder branch, not
    a rewrite. [larql-inference, larql-vindex]
17. ✅ **Runtime trace emission** (DONE 2026-07-31 — the post-hoc
    `gate_knn` re-run is GONE: `with_trace` upgrades every call to
    `Observe::Record` and folds the executed path's observation into
    per-(position, layer) records at the routing-ladder exit (new
    `walk_ffn/trace.rs`), riding the item-15 seam rather than a parallel
    channel. `SparseActivations` entries carry the kernels' own gate/up
    scores (`record_scored`) plus per-position kernel labels, so
    serial/gather/parallel/weights-fallback report the values they
    actually computed (gather now returns its fused gate/up dots);
    records carry gate_score/up_score/activation/rank/path +
    residual_delta_norm (`‖out_row‖`); `‖down_row‖` is served only from
    the selector's prebuilt lazy norm cache, never computed for tracing;
    dense whole-layer paths emit the layer summary and decline
    per-feature records rather than fabricating. `take_trace` rebuilds
    the public `WalkTrace` from the runtime records — hits are the
    EXECUTED features, `WalkHit` extended additively
    (up_score/activation/down_row_norm/rank; post-hoc KNN views build
    via the new `WalkHit::from_gate` and stay honestly `None`) — and
    `take_runtime_trace` exposes full fidelity. Field names follow the
    chuk-introspect snake_case vocabulary; no dependency added. Pinned
    by `take_trace_reports_executed_route_not_gate_knn`: a pool route
    vs a decoy `gate_knn` — the trace must equal the executed route,
    which the old re-run structurally cannot return. Target-logit
    contribution needs lm_head access → documented out of scope in
    `trace.rs`) — replace `take_trace`'s post-hoc
    `gate_knn` re-run (`mod.rs:281-306`, which ignores selector/pools/
    cell-router and records scores, not contributions) with emission from
    the executed path: gate, up, activation, ‖down‖, residual-delta,
    logit contribution, rank, path. Align with the chuk-introspect schema
    — no second trace format. [larql-inference]
18. ✅ **Execution planner** (DONE 2026-07-31 — path selection is an
    explicit decision value: `FfnPlan` (new `walk_ffn/plan.rs`), one
    variant per ladder destination incl. `OverrideBaseDelta` as a plan
    variant per the freeze condition, names aligned to the trace_path
    vocabulary. Every variant carries a structured `PlanReason` —
    layer/seq_len/num_features/has_overrides, the `selected`
    condition, a `skipped` list stating why EACH higher-priority rung
    did not fire (base+delta declines name the exact failed
    precondition — `base_delta_preconditions` now returns
    `Result<slots, &'static str>`), and pre-execution `ThresholdCheck`s
    (requested K vs FULL_K_DENSITY, single-sourced from
    `hits_len_ge_intermediate` so `satisfied` honours `force_walk`).
    The planner (`planner.rs` `plan_layer`) is the ladder's ONLY
    condition source: `forward_ladder` plans, then `execute_plan`
    matches condition-free, and `forward_unpatched_whole_layer`
    (base+delta's base) iterates the same `WHOLE_LAYER_RUNGS` table.
    Try-then-fallthrough handled honestly: a path returning `None`
    mid-execution re-plans with that rung in a `PlanExclusions` set,
    and the executed plan's reason records "declined at execution" —
    pinned by a test where six lying capability flags each decline and
    the ladder lands exactly where the pre-planner code did.
    Inspection: public `WalkFfn::plan_for` (pure — L1 probed via new
    stats-free `FfnL1Cache::peek`, no dispatch entries, no execution);
    the runtime trace's `LayerTraceRecord` gains `plan_reason`
    (additive — `DispatchEntry`'s literal construction is pinned by
    routing tests). Routing is decision-identical: every dispatch/
    routing/trace test passes unchanged, same trace_path strings; the
    executed forward keeps exactly one L1 `get` per eligible call so
    hit/miss accounting is preserved. 20 planner tests (one per rung +
    decline-re-plan + purity); changed/new files ≥96% line coverage.
    Thresholds stay in `thresholds.rs`, REFERENCED by reasons —
    surfacing them into `WalkFfnConfig` is a tracked follow-up) —
    `VindexFfnPlan` enum + structured reason
    (plan/reason/layer/features/overrides), config-surfaced thresholds
    replacing the magic ratios; only freeze once base+delta exists as a
    plan variant. The ladder's trace_path names + routing tests are the
    seed; add the reason field. [larql-inference]
19. ✅ **Two-stage selection: shortlist top-M by gate, exact rerank**
    (DONE 2026-07-31 — opt-in `WalkFfnConfig::shortlist_m:
    Option<usize>` (+ `with_shortlist_m`; `None` = single-stage,
    default everywhere), consumed on the selector-dispatch route (new
    `walk_ffn/shortlist.rs`): stage 1 takes the top-M through the
    production `gate_walk` → `gate_knn_q4` → `gate_knn` chain (now
    factored as `production_gate_chain`, shared with the `GateOnly`
    route and the joint fallback — no new projection code); stage 2
    evaluates the configured criterion for ONLY those M candidates —
    per-candidate up dots via the per-row `ffn_row_dot`, norms from
    the existing lazy caches, O(M·d), never a full projection — and
    fully sorts to the final top-K (`rerank_cmp`: weight desc, feature
    asc on ties; the runtime trace's `rank` field is therefore the
    FINAL rerank order, and `joint_gate_knn` sorts its top-K by the
    same comparator so the two paths report identical order). The
    weight formulas are single-sourced in `criterion_weight` /
    `criterion_inputs` — `joint_gate_knn`'s inline per-variant
    closures were extracted onto them, so the full-projection and
    two-stage paths cannot drift. Hits keep the
    `(feat_idx, raw_gate_score)` contract; `shortlist_m` forces the
    per-position walk (like pools — the full-K gemv rewrite would
    bypass the structure); the Sparse plan reason records a
    `SHORTLIST_M` `ThresholdCheck` (actual=M, cutoff=K, satisfied =
    two-stage actually runs). M < K, `Random` (no criterion), or
    missing stage-2 inputs decline to single-stage OBSERVABLY — a
    `shortlist:declined` dispatch-trace entry +
    `shortlist_decline_count`, the M10 `selector:fallback` precedent.
    Pinned by 13 tests (`shortlist_tests.rs`): M=N two-stage ==
    `joint_gate_knn` (same features, same order, raw scores) for every
    scored selector; a huge-‖down‖/tiny-gate decoy the full-projection
    rerank picks but the top-M gate shortlist structurally excludes;
    observable declines; default-off bit-identical to single-stage;
    and the cost pin — a delegating index that PANICS on
    `gate_scores_batch`/`gate_scores_batch_backend`/
    `kquant_matmul_transb` runs a full two-stage forward clean, while
    its counting twin shows single-stage joint pays ≥2 full
    projections. Changed/new files ≥94% line coverage) — the
    rerank criterion already exists as
    `FeatureSelector::ActXUpScoreXDownNorm`; add the shortlist structure
    so it stops paying full projections. Production-cost shape of the
    existing experiment harness. [larql-inference]

**Tier 3 — productization:**

20. ✅ **Walk-vs-dense parity suite** (DONE 2026-07-30 — landed with the per-file 90% coverage pass: serial-vs-parallel, gather-vs-serial on a real sidecar, walk-vs-dense WeightFfn parity for gemv + exact/full_mmap/interleaved, dispatch-trace assertions against the REAL ladder in the moved dispatch_tests.rs; every walk_ffn file >= 90% line coverage) — the four tests that would have caught
    the four worst bugs: non-aligned Q4K fixture through cache + serial
    walk vs dequant baseline; CpuBackend + Q4_0 forward; serial-vs-parallel
    parity at hits ≥ 512 asserting output AND activation; dispatch-trace
    assertions against the REAL ladder (routing_tests.rs currently tests a
    hand-copied replica that can drift without failing). No test anywhere
    compares walk output against dense ground truth on a served vindex.
    [larql-inference, larql-server]
21. ✅ **KnnStore unification** (DONE 2026-07-31 — unified at the
    RETRIEVAL-KERNEL level; honestly short of full arch-B retirement,
    which is now explicitly gated in the spec rather than silently
    pending. The parallel scoring implementation is GONE: `KnnStore`'s
    private `key_matrices` GEMM + `dirty`-flag rebuild machinery is
    deleted, and its L2-normalized keys now live as rows in the new
    shared `patch/gate_overlay.rs::GateOverlay` — the same structure
    that holds `PatchedVindex`'s gate overrides — so `gate_knn` and
    every KNN query score through ONE kernel carrying the campaign's
    hardening (H3 zero-width guard, mixed-width slow-path fallback,
    per-layer snapshot cache). Mutators invalidate their own layer's
    snapshot, retiring the manual `invalidate_gate_cache*` calls (a
    forgotten-invalidation hazard class). What stays KnnStore-specific
    is POLICY, not machinery: entity/relation/target entry metadata,
    normalize-on-insert, rank-by-raw-cosine (vs `gate_knn`'s `|score|`
    merged with base hits — match the statistic to the operation).
    Public API, `.vlp` `InsertKnn`/`DeleteKnn` ops and the
    `knn_store.bin` format are unchanged; all five consumer crates
    (inference/lql/server/python/engine) compile untouched. Full
    "FFN = KNN index = vindex" (spec §3: appended-slot
    `AppendFeature`, delete the post-logits override) is NOT done —
    the FR1/FR2/early-exit routers (2026-06/07) shipped ON the
    post-logits override after the spec was written, and the α
    calibration (spec Q2) plus the 189-fact parity benchmark are
    unvalidated empirical work; `FFN_VINDEX_UNIFICATION_SPEC.md`
    rewritten to describe the post-unification reality and the
    remaining gate. Regression pins: query correctness after entity
    removal renumbers indices, clone-preserves-retrieval; changed
    files ≥90% line coverage in-crate) — still exported and live in
    `patch/knn_store_io.rs`/`overlay.rs`/`overlay_apply.rs`; the
    unification spec still describes it. Until removed, "FFN = KNN index =
    vindex" is partly aspiration. [larql-vindex]
22. ✅ **Vindex v1 conformance contract** (DONE 2026-08-01 —
    `crates/larql-vindex/docs/conformance-v1.md` + the pinning suite
    `tests/conformance_v1_{index,kquant,patches,down_meta,golden_le}.rs`
    (38 tests over the shared `tests/common/` fixture): every v1
    artifact × corruption class asserts error-not-panic-not-garbage —
    index.json (malformed/missing/wrong-typed fields, unknown
    dtype/quant tags, the H4 out-of-range-layer fix pinned as
    contract), interleaved_kquant slab+manifest (unknown format tag →
    Err; truncated slab / offset-length overflow / short manifest →
    checked_view decline; H1 padded-stride pinned at the writer),
    down_features sidecar (bin-without-manifest and missing shape[1]
    → Err, OOB → decline), .vlp (corrupt/truncated base64 → wholesale
    rejection, zero half-applied ops — M4/M5 pinned), .lknn
    (magic/version/truncation/absurd-count), down_meta.bin (truncation,
    checked-arithmetic overflow, allocation-bomb regression on both
    readers). Cross-platform: byte-level LE golden vectors for
    le_floats, .vlp base64, down_meta.bin, .lknn — exact bytes, not
    round-trip equality; no BE runner exists, the goldens are the
    guard. The two §3 LOW conformance violations fixed: legacy
    `down_meta::read_binary` now bounds every allocation by the real
    file size with checked arithmetic (mirrors `mmap_binary`; module
    split into `down_meta/{mod,read}.rs`), and the Vindexfile parser
    got quote-aware tuple splitting (`INSERT ("Acme, Inc", …)`),
    hard errors on missing/unknown/duplicate DELETE condition keys,
    and `find_free_feature().unwrap_or(0)` → error instead of
    silently overwriting feature 0; `.lknn` capacity hints bounded by
    remaining bytes as part of the same pass. Perf benchmark protocol
    is a documented follow-up in conformance-v1.md §4 (walk-vs-dense
    parity exists as item 20; no numbers faked). [larql-vindex,
    larql-vindex-spec]
23. ✅ **Doc drift** (DONE 2026-08-01 — every number traced to its
    source before editing. The `0.008 ms/layer` + `0.3 ms` 34-layer
    walk headline (repo README, vindex `operations-spec.md`) was the
    pre-2026-04-05 `vindex_bench` example at its reduced 1024×256
    synthetic shape ("reduced from 10240/2560/34 for bench speed"),
    scaled to 34 layers — replaced with the current criterion
    `vindex_ops` numbers at BOTH shapes (22.7 µs at 1024×256, 2.64 ms
    at the Gemma 10240×2560 production shape) plus an explicit
    exact-brute-gemv note: the walk hot path never consults HNSW,
    `enable_hnsw` is gate-KNN-consumers-only (item 13 inversion), now
    also stated in the crate README's interpretability recipe.
    Extract-level default contradiction resolved IN FAVOUR OF THE
    CODE: `larql extract` defaults to `--level inference`
    (`extract_index_cmd.rs:46`) while bare LQL `EXTRACT MODEL`
    defaults to browse (`lql parser/lifecycle.rs:17`) — the README
    table now says which default belongs to which surface, and the
    stale "add `--f16`" footer became "f16 is the default, `--f32`
    opts out". walk.md "Lossless at K=8092": NOT fixed by swapping
    8092→8192 — the 2026-04-03 boundary sweep, sparse.md and the
    remote-codec tests all literally ran K=8092 (the typo is baked
    into the harness), so the doc now says exactly that, notes
    8092 = 79% of 10240 stays genuinely sparse while K≥8192 hits the
    80% full-K dense rewrite (`thresholds.rs`), and date-qualifies
    the 97.91% figure (LQL-spec INFER example run, not the sweep).
    walk.md/ffn-README "production" framing reframed: WalkFfn =
    instrumentable/editable execution layer + CPU sparse path, Q4K
    GPU decode (~88 tok/s vs ~1.9 tok/s CPU INFER walk) is the perf
    centre; historical results kept, date-qualified. Campaign-sweep
    fixes: runtime trace emission + `new_with_trace` in walk.md
    (item 17), base+delta-first for patched layers in walk.md + the
    crate README W2 note (item 16), `gate_overlay.rs`/KnnStore
    GateOverlay-backed scoring in the crate README tree (item 21),
    `walk_ffn.rs` → `walk_ffn/` paths. No code changes.) [docs]
24. ✅ **Hygiene** (DONE 2026-08-01 — triaged per the item's own
    licence: worked in priority order, each piece fully or not at all,
    remainder documented. **(1) Generic-engine violations:** the
    English word lists (countries/languages/months/numbers + the
    148-word stop list) and the Wikidata category vocabulary are OUT
    of `clustering/` engine code and into `data/entity_patterns.json`
    + `data/stop_words.json` (+ the existing
    `data/wikidata_categories.json`), loaded through the new
    `clustering/data_files.rs` search chain — `LARQL_DATA_DIR` env dir
    → compile-time workspace `data/`, explicit config path via the
    `*_from(path)` loaders, NEVER cwd; `load_reference_databases`'s
    identical cwd-probe (`data`/`../data`/`../../data`) fixed with the
    same resolver; fallbacks are minimal built-in core sets and LOUD
    (stderr `warning:`). Bare `0.25` floor → `MIN_CATEGORY_SIMILARITY`;
    the "60%+" doc-vs-`0.5`-code pattern threshold resolved in favour
    of the code as `PATTERN_MATCH_FRACTION` (+ `MORPHOLOGICAL_MAX_LEN`);
    class order is data (language before country), pinned. Tests cover
    data-file loading, env-dir precedence, missing/invalid/empty-file
    loud fallbacks, threshold boundaries, and a behavioural
    similarity-floor pair through `auto_label_clusters_from_embeddings`;
    clustering files 93–100% line coverage. **(2) The two deferred
    item-23 16384 fixes:** Gemma 3 4B intermediate is 10240 (verified
    against `larql-models` `gemma3.rs:195`) — `docs/ffn-cache.md:46`
    now states the real sparse gate (below the 4/5 `FULL_K_DENSITY`
    rewrite ⇒ `top_k < 8192`, 8092 qualifies) and lql
    `insert/capture.rs:99` says 10240. **(3) Activation dispatch:**
    27 copies of the GeluTanh|Gelu → gelu-tanh-else-SiLU match (10
    `walk_ffn/` files, `sparse_compute.rs` ×3, `layer_graph/template.rs`,
    `kquant_forward/walk_ffn.rs` ×4 across larql-inference AND
    larql-compute, `cached.rs`, `ffn/weight.rs` ×4,
    `expert_weight/gate.rs`, 3 examples) now route through ONE helper,
    `larql_models::Activation::uses_gelu_tanh_gate_up()` — a
    wildcard-free exhaustive match (a hypothetical new variant is a
    compile error, not a silent SiLU landing; pinned by tests incl. a
    `#[should_panic]` for `Relu`, which has no kernel and no in-tree
    arch). Two silently-drifted copies found en route (`weight.rs`
    gated arms and `cached.rs` matched `GeluTanh` only, dropping exact
    `Gelu` to SiLU) are now consistent. **(4) Component constants:**
    `FFN_GATE`/`FFN_UP` added beside `FFN_DOWN` +
    `FFN_COMPONENTS_PER_LAYER`, pub in larql-vindex (crate-root
    export) and mirrored in `larql_compute::kv_index` with
    compile-time equality pins in `kv_index_impl.rs`; every bare
    `0/1/2` walk/kquant call site replaced (selector norms, sparse
    row-dot/scaled-add, sparse_parallel, `interleaved_q4`'s `* 3` →
    `FFN_COMPONENTS_PER_LAYER` + component-slice helper, both
    `kquant_forward/walk_ffn.rs`, and `base_delta.rs`'s local consts
    unified on the vindex ones). **(5) Colocated tests, 41 new:**
    `index/compute/hnsw.rs` 0 → 12 tests at 97.3% line coverage
    (insert/search, recall@10 = 0.97 vs brute force on clustered
    synthetic, level-RNG determinism with the LCG constants pinned);
    `index/mutate/mod.rs` 14 tests at 96.0% (meta/gate/override
    mutation, INSERT/DELETE-then-query, save→load round trips incl.
    mmap→heap promotion); `format/weights/write_f32.rs` 15 tests at
    92.6% (round trip through the f32 loader, MoE/MLA/BitNet writer
    branches, error paths). New finding pinned honestly rather than
    papered over: HNSW's level-0 graph FRAGMENTS as n grows — naive
    `add_connection` eviction orphans nodes (~33/200 BFS-reachable,
    recall@10 0.16 at n=200 uniform even with ef=n; fully connected
    ≤~64) — production gate-KNN at 10K+ features may be silently
    degraded; carried as a standing follow-up. **REMAINDER (documented,
    not done): (6) file splits** — `huggingface/download/mod.rs` 1329,
    `patch/overlay.rs` 1071, `quant/convert.rs` 653 still exceed the
    250-line rule (94/196 vindex src files over; `walk_ffn/mod.rs` is
    already down to 553 and `sparse.rs` to 861 via the Tier-2 sibling
    decompositions). Verification: larql-vindex 1296 lib tests +
    integration suites, larql-inference 1423 lib tests, larql-models /
    larql-compute / larql-lql all green; clippy + fmt clean on changed
    files; changed/new files ≥90% line coverage except
    `larql-models/src/config.rs` (63% file-wide pre-existing
    trait-default debt; the added helper's lines are 100% covered))
    — file splits (`walk_ffn/mod.rs` 926 → timings/ladder/
    builders; `sparse.rs` 842 → gemv/route/parallel/gather;
    `overlay.rs` 959; `huggingface/download/mod.rs` 1329;
    `quant/convert.rs` 655 — 88/186 vindex files exceed the 250-line
    rule); dedupe the 8-site GeluTanh/SiLU activation dispatch (new
    activations silently land in the SiLU arm); English word lists +
    Wikidata categories out of `clustering/` into data files (+ fix
    cwd-relative probing); colocated tests for `hnsw.rs` (455L, zero
    tests), `index/mutate/mod.rs`, `write_f32.rs` (777L); bare `0/1/2`
    component indices → `FFN_DOWN` et al. [larql-vindex, larql-inference]

### Extraction tensor-coverage audit + silent-drop follow-ups (2026-07-31)

Built the audit §4.6 work-item 2 asked for: every source tensor is classified
as **recognised** (an architecture accessor names it), **dropped by a named
rule**, or **unrecognised** — and the third bucket is loud.
`extract::coverage` + the `tensor_audit` stage, which runs *first* in
`build_vindex_streaming` so an unaddressable checkpoint fails in seconds
rather than after a multi-minute extraction. Reports always; fatal under
`LARQL_EXTRACT_STRICT=1`, which is now set in the `larql-vindex` CI workflow.

The case for it was five silent drops in one week, none caught automatically:
5 of 11 attention tensors (§4.6.1), 3 of 8 MLP tensors (§4.7), the
`gate_walk` trait default silently `None` (review item 13), a
`moe_intermediate_size()` defaulting to 0, and LayerNorm `β` — see item 3.

Validated on ten checkpoints: Qwen3-30B-A3B (18,867 tensors), OLMoE (3,219),
gpt-oss-20b, Gemma 3 4B (439 SigLIP tensors correctly classified
`non-text-tower`), all clean. **GPT-2 from HF safetensors: 1 of 160
recognised** — see item 2.

1. **Migrate `residual_diff` off process-global env vars onto the
   thread-local override.** `larql_compute::options::set_env_override`
   exists precisely to replace `std::env::set_var`, "which races concurrent
   `getenv` on the decode path and SIGSEGVs libc" — and all three dump sites
   already read through `options::env_value`, which consults it first.
   `run_with_dump_dir` / `run_with_two_env_vars` never adopted it; they were
   fixed on 2026-07-31 with a shared mutex, which is correct but serialises
   four ~110 s captures. Thread-local removes the shared state instead of
   guarding it. **Prerequisite:** the dump hook must read the var on the same
   thread that set it — true for the CPU path (`hidden.rs`'s own test relies
   on it), plausible for Metal encoding, but a read inside a rayon worker
   would silently stop dumping. Verify against the 4-model parity suite
   (~7 min) before switching. Needs an additive `clear_env_override(name)`;
   today only `clear_fast_path_overrides()` (clears all) exists.
   [larql-inference, larql-compute]
2. **GPT-2: rename or add an HF-safetensors variant.** `gpt2.rs` matches the
   trait defaults only *after* the GGUF→HF normalisation, so a raw HF
   checkpoint (`h.N.attn.c_attn.weight`, `wte.weight`, `ln_1.*`) is
   unaddressable. It fails late at the embeddings stage with a one-tensor
   message rather than silently, but 159 of 160 tensors are unreachable.
   Needs the `h.N.` prefix + `c_attn`/`c_fc`/`c_proj` spellings, and a drop
   rule for `h.N.attn.bias` (the causal mask — a derived constant, not a
   weight, so it belongs in `coverage::rules`). Unblocked by item 3.
   [larql-models]
3. **Verify the restored LayerNorm `β` numerically.** No accessor named a
   norm bias until 2026-07-31, so extraction never wrote one and
   `build_pipeline_layers` hardcoded `input_norm_bias: None` — while the
   Metal `layer_norm` shader implemented `+ bias` and always took its
   no-bias variant. The CPU dense path got away with it by mangling the
   weight key, which is why raw-safetensors inference was right and every
   vindex-backed path dropped the shift, for **GPT-2 and StarCoder2**. Now
   declared, extracted and resolved; the honest status is "the tensor flows
   end to end", not "the output is correct". Wants a GB-shaped measurement.
   [larql-models, larql-vindex, larql-compute]
4. **Consumption-level coverage audit.** The current audit measures
   *naming*, not consumption — a recognised tensor is one extraction *can*
   reach, not one it wrote. Naming is where all five drops actually lived,
   so this closes the bug class that has bitten; recording `WeightSource`
   reads would subsume it and also catch "named but never asked for".
   [larql-vindex]
5. **`capture.rs` cannot reach the 90 % floor in Linux CI.** ~250 of its 408
   lines are `metal_decode` / `metal_decode_steps` / `metal_prefill`, which
   need a Metal device by construction — which is why the file sits outside
   the crate's `include_globs`. Raised 34 % → ~50 % by testing `cpu_prefill`
   for the first time, plus a macOS-gated `metal_prefill` test so the
   constructor is exercised somewhere. Either accept the exclusion
   permanently and say so in the policy note, or split the GPU dispatch from
   the dump-readback logic so the latter is testable everywhere.
   [larql-inference]
6. **`named_keys` is hand-maintained against 62 trait accessors.** A new
   `*_key` accessor not wired into `collect()` makes its tensors report as
   *unrecognised* — noisy, never quiet, and the pin test catches it at the
   source. It has already fired twice for real (`moe_post_ffn1_norm_key`;
   then the three norm-bias accessors). If the accessor count keeps growing,
   consider deriving the list rather than pinning a count.
   [larql-vindex]

---

### Compute-layer hygiene review — `larql-compute` / `larql-compute-metal` / `larql-models` (2026-08-05)

Scanned for the four standing rules: architecture-driven rather than
model-hardcoded, no magic strings/numbers, modular and decoupled, no large
files. **The architecture-independence story is much better than the file-size
one**, and the one real hardcoding leak is a stringly-typed protocol.

| # | Finding | Where | Priority |
|---|---|---|---|
| H1 | **`moe_router_type()` was a `&str` protocol between models and compute.** `pipeline_layer::moe_routing_policy` matched the literal `"gemma4_top_k_softmax"` and fell through to a default for everything else — so `gpt_oss`'s `"gpt_oss_topk_then_softmax"`, a genuinely different rule, **silently took the ordinary policy**. That is the mechanism behind §4.7.10's open quantised-MoE defect, not merely a style issue. **FIXED**: typed `MoeRouterKind` with the string kept as the vindex wire form (`as_str`/`from_wire`); compute now matches exhaustively, so a new variant fails to compile rather than defaulting. The predecessor test called the function twice and asserted nothing — replaced with one that pins each kind to a distinct policy. | `larql-compute`, `larql-models` | **done** |
| H2 | **`diag/shader_bench.rs` — 1 759 non-test lines**, and it hardcodes `"gemma3"` profiles and `"gemma3-4b"` labels. Diagnostics may name models, but not at this size in one file. **Attempted and reverted:** a line-based carve into config / shapes / measure / benches kept cutting across item boundaries (a trailing `#[derive]`, a truncated function body, the `mod tests {` wrapper). It wants an AST-aware split or a careful manual one, not a `sed` pass — and it is the lowest-value item here, so it was not worth finishing badly. | `larql-compute-metal` | low (was medium) |
| H3 | **`kquant_forward/cached.rs` — tests. DONE (2026-08-06).** "Zero tests" was half right: the sibling `kquant_forward/mod.rs` suite already drove most of the public surface, but **every one of those tests asserts a shape, not a value** (`h.shape() == [1, hidden]`, "must complete without panic"). That is exactly the hole §4.10 fell through — a RoPE defect keeps the shapes correct. 16 new tests in `cached/tests.rs` built on *agreement*: prefill-vs-decode on the rope-scaled Q4_K fixture (CPU analogue of M5), the padded-intermediate refusal in `layer_supports_direct_matvec`, and the guard clauses of `matvec_q4k_or_q6k_q8k`. Also replaced mod.rs's `let _: bool = supports_direct_matvec_decode(...)` — a test that asserted nothing — with a real assertion. | `larql-compute` | **done** |
| H4 | **`decode/mod.rs` — tests. DONE (2026-08-06), and it found a bug.** "Zero tests" again described coverage, not correctness: `tests/test_metal_decode_synthetic.rs` already drives `decode_token` end to end and says so in its own header ("smoke tests, not numerical-parity tests"). What nothing touched was the **KV cache geometry** layer — `kv_shapes_for_layers` / `ensure_kv_cache_for_{layers,shapes}` — where every failure mode is silent: decode still runs, still returns finite numbers of the right shape, and is simply wrong past some position. Six tests in `decode/tests.rs` on the Gemma-4 sliding(16×256)/global(4×512) pair, GPU-guarded. **`grow_to_shapes` ignored its `max_seq` argument** — see below. | `larql-compute-metal` | **done** |
| H5a | **`lm_head` silently tied to the embedding matrix.** `unwrap_or_else(\|\| embed.clone())` fired whenever `lm_head.weight` was absent, and **`tie_word_embeddings` was never parsed at all** despite appearing in every checkpoint config and several fixtures. A model declaring `false` (GPT-OSS, OLMoE) that lost the tensor to a key mismatch or skip filter would have served a wrong output projection and still produced fluent text. **FIXED**: field parsed (outer *and* `text_config`), and untied-but-missing is now a `MissingTensor` error naming the conflict. Absent stays `None` — not a claim either way — so tie-on-absence is unchanged for models that really are tied. | `larql-models` | **done** |
| H5b | **Split `loading/safetensors.rs`. DONE (2026-08-06).** 1 205 lines → `safetensors/{mod,mxfp4,dtype,paths}.rs`, largest 491. Moved whole functions with the compiler as the check (the H2 lesson), then moved each concern's tests and helpers to sit with it. Seams: MXFP4 packed-expert expansion (+ the seven `MXFP4_*` name constants, which now live with the layout they describe), raw-dtype/FP8 decode, model-path resolution; the shard walk and key normalisation stay in `mod.rs`. 649 `larql-models` tests green, clippy clean. | `larql-models` | **done** |
| H6 | `attention/gqa.rs` was 1 308 lines but **377 non-test** — the bulk was its (good) test suite. **FIXED**: split to `gqa/mod.rs` (379) + `gqa/tests.rs` (934), the same pattern `rope/` uses. | `larql-compute` | **done** |

**What is already right, and worth not regressing.** Architecture behaviour is
genuinely trait-driven: model-type strings appear almost exclusively in
`detect/mod.rs` (the dispatcher, where they belong) and in test fixtures. The
`stages::sinks` / `stages::rope_freq` modules are the pattern to copy — each
owns one binding convention in one place, with the reason it exists documented
against the defect that motivated it. Numeric constants are named
(`ROPE_BASE_DEFAULT`, `DEFAULT_NORM_EPS`, `YARN_BETA_FAST`,
`UNIT_AMPLITUDE`, `LAYER_TYPE_*`, `ROPE_TYPE_*`) rather than inline.

**Status after the second pass (2026-08-06), updated 2026-08-07:** H1, H3, H4,
H5a, H5b and H6 are done; M3, M4 and M5 are done, and M6/M7/M8 landed on
2026-08-07. **H2 is the only listed item still open**, and it is low by choice.

The second pass also turned up **three** defects that were not on the list —
two from the standing scan, one from writing H4's tests. All three are the same
family as H1/H5a: a value that answers a question nobody asked it. Two are
`_ =>`/omission defaults; the third is an argument accepted and ignored, which
is a pattern the scan did not previously look for and now should.

#### Next actions for the open items

**H2 is the only one left**, and it is deliberately last.

**H2 — split `diag/shader_bench.rs`.** Unchanged and still lowest value. Do
**not** repeat the line-range carve: it cut across a trailing `#[derive]`,
truncated a function body, and orphaned the `mod tests {` wrapper. Move items
one at a time with the compiler as the check — that is how H5b was done this
pass and it worked without incident — or leave it; it is diagnostics code with
no known defect behind it.

#### Standing follow-ups from the same pass

- **M4** — **done 2026-08-07**, and the defect was the mirror of what this line
  described: Metal *decode* windowed correctly while Metal *prefill* took no
  window at all. See the M4 row above. **Still open from it:** a model-level
  long-prompt parity fixture. The existing suites prompt with ~16 tokens
  against a 1024 window, so they remain blind to this class; the kernel test is
  what guards it today.

##### Ninth instance — an argument accepted and ignored (H4, 2026-08-06)

Not found by grepping for `_ =>` or `unwrap_or`: this one is a **parameter
that is taken and then not used**, which the scan's three patterns do not
catch. Worth adding as a fourth thing to look for at the boundary.

`KVCache::grow_to_shapes(bufs, shapes, max_seq)` only ever grew the *layer
count*; it never looked at `max_seq` for layers that already existed. Its
caller `ensure_kv_cache_for_shapes` rebuilds only on a **shape** mismatch — so
a second, longer prompt with the same attention geometry kept buffers sized
for the first one, while the caller had just asked for more room and had no
way to learn it did not get it. `encode_kv_append` then writes at
`current_len` and bumps it with **no bound check** against `max_seq`, so the
appends run off the end of a buffer allocated as
`max_seq * num_kv_heads * head_dim * 4`.

Reachable on a real path, not just in theory:
`vindex::kquant_forward::metal` sizes the cache as
`token_ids.len().max(MIN_KV_CACHE_SEQ)`, so it varies with prompt length
across calls on one backend. The uniform call sites (`kv_cache_mut*`) pass the
constant `DEFAULT_KV_CACHE_MAX_SEQ` and never take the branch, which is why
nothing had hit it.

Fixed by reallocating undersized layers in `grow_to_shapes`; regrowing drops
that layer's cached K/V, which matches what a shape mismatch already does, and
the one caller that varies `max_seq` calls `reset_kv_cache()` immediately
after. `ensure_kv_cache_grows_max_seq_for_a_longer_prompt` pins it — verified
to **fail** with the fix reverted while the other five geometry tests still
pass, so it discriminates the defect rather than merely covering the line.

##### The standing scan found two more (2026-08-06)

The scan works. Run it: grep the model→compute boundary for `_ =>`,
`unwrap_or`, and `&str` parameters that carry a behavioural choice — and now
also for **parameters that are accepted and never read** (the H4 instance
above, which none of the first three patterns would have caught). Two hits
from this run, both fixed, both worth reading as a pair because they fail in
opposite directions.

**Seventh instance — `Activation` collapsed to SiLU at the pipeline boundary.**
`pipeline_layer.rs` translated `arch.activation()` into the compute enum with
`match { GeluTanh => GeluTanh, _ => Silu }`, re-spelled at three construction
sites. `larql_models::Activation` has four variants, so **`Relu` and `Gelu`
both became `Silu`** — and the compute enum already had `ReLU` and `GeluExact`
waiting to receive them, so nothing was lost for lack of a destination. The
damning part: `larql-compute-metal`'s `assert_metal_activation_supported`
exists precisely to "fail loud rather than silently routing GeluExact / ReLU
layers to SiLU (the prior behaviour, which produced wrong logits with no
signal)". That guard is correct, tested, and **was unreachable** — the
wildcard upstream guaranteed it could never be handed either variant. The fix
was applied at the consumer and missed at the producer. Now three `From` impls
in `pipeline/enums.rs` are the one definition, exhaustive so a new variant
fails to compile; the five CPU MoE expert loops that carried the same
`_ => silu` wildcard route through one `gate_up_is_gelu_tanh()`. No in-tree
architecture returns `Gelu` or `Relu`, so this is behaviour-preserving today —
it converts a latent silent-wrong into the loud refusal that already existed.

**Eighth instance — the vindex format could not carry `rope_scaling`.** This
one is an *omission*, not a wildcard: `VindexModelConfig` simply had no such
field, so `from_arch` dropped it and every vindex-served model read back
`rope_scaling: None`. `google/gemma-3-4b-it` declares
`{"factor": 8.0, "rope_type": "linear"}`; served from a vindex it ran with a
position divisor of **1.0 on its five global layers**, rotating them eight
times faster than the checkpoint asks. This is a served-model correctness
defect, not a test gap — and it is why M5 above needed an env override to mean
anything.

Note *why it was invisible*: CPU and Metal both read the same `index.json`, so
both were wrong identically and `test_decode_consistency` stayed green.
**A parity gate cannot see a defect in a config that both of its arms share.**
That belongs next to R14 (gate–claim congruence) as a standing rule.

Fixed by `RopeScaling::to_config_json` (an inverse of the detector's parser,
with per-family `parse(emit(x)) == x` round-trip tests, because an inverse
that drifts from its forward is worse than none) plus five previously-dropped
fields — `rope_scaling`, `attn_logit_softcapping`, `swiglu_limit`,
`norm_topk_prob`, `tie_word_embeddings`. That last one is H5a's field: the
H5a fix could not reach a vindex-served model. All are `#[serde(default)]`, so
**existing vindexes still load and still answer `None` — they must be
re-extracted to pick the values up.** The two production writers that had
open-coded their own copy of `from_arch` now call it.

Two guards so the class cannot recur: `model_config_persists_every_forward_
affecting_field` scrapes both structs and fails on any `ModelConfig` field
with no home (a deliberate not-persisted list carries the reasons — MLA
geometry and `has_vision_config` are named as real gaps, `embedding_multiplier`
as already carried via `embed_scale`), and
`gemma3_global_rope_divisor_survives_the_vindex_round_trip` asserts the
divisor end to end with a precondition that the source arch had one.

---

## Cleanup / consolidation track (added 2026-06-12)

Standing recommendations from the 2026-06-12 review, distinct from the
hardening bug-fixes above: this is the maintenance-debt layer. The repeated
observation across both reviews is that bugs in this codebase come back from
the dead through **duplication** — parallel paths created to avoid
destabilising a parity-verified one, then maintained in lockstep by comment
("keep in lockstep" twins, `KernelHandle` bypassed at 2 new sites, 6 copies
of the env-flag helper with diverging semantics). The corrective habit, made
policy:

> **Prefer a parameter on the existing path over a parallel path.** A new
> code path needs the same justification as a new crate: a reason the
> existing one cannot be parameterised. Opt-in experiment paths are fine,
> but they get a removal-or-promotion condition when added, not after.

Themes, in leverage order (concrete first steps live in hardening items
7–10 above; this section tracks the policy-level work):

1. **One forward-pass spine** — the five parallel layer-step loops in
   `larql-inference/vindex/kquant_forward/` are the canonical instance.
   ADR first (what is the shared layer-step contract: sentinels, MoE
   detection, KV dispatch, capture hooks), then fold
   `hidden`/`prefill`/`decode_step`/`decode_step_direct`/remote-FFN onto
   it. Sequenced behind the C10 residency arc (same hot files). The
   padded-down twin extraction (hardening item 8) is the cheap pilot for
   the same move one level down. [larql-inference, larql-compute]
2. **Flags → config** — beyond the registry (hardening item 7): any
   `LARQL_*` flag that changes numerics and has survived its experiment
   (e.g. the Q4K residency trio once C10 lands) gets promoted to real
   config/CLI surface or deleted; env vars stay for diagnostics and
   short-lived experiments only. Uniform parsing through the
   `options.rs` taxonomy so `=true` vs `=1` can never again silently
   change what a bench measured. [workspace]
3. **Experiment-path lifecycle** — opt-in paths that lost their A/B keep
   accumulating (ADR-017 covers shaders; nothing covers CPU/env paths).
   Extend the ADR-017 rule workspace-wide: every opt-in path carries a
   retention rationale + revival story, and reviews may delete any that
   lack one. Current deletions/decisions owed: 4 unreferenced Metal shader
   modules, `model-compute` (no second consumer), `larql-experts`
   integration status, `test_utils.rs` out of larql-inference's public
   API. [workspace]
4. **API surface honesty** — `larql-inference/vindex` re-exports ~28
   implementation-named functions (`predict_kquant_*` variants); external
   callers choose forward paths by fuzzy naming. After (1), expose one
   facade that dispatches internally; deprecate the variants. Pairs with
   the Engine/StatePolicy framing already proposed. [larql-inference]
5. **Coverage debt** — per-file ≥90% floor policy vs reality:
   `larql-inference` 70.7%, `larql-cli` 12.0% (snapshot 2026-05-16).
   Raise toward the floor opportunistically as files are touched by (1)
   and (4) rather than as a standalone sweep; new/split files land at
   ≥90% (existing policy). [larql-inference, larql-cli]
6. **Scratch-artifact hygiene** — underscore-prefixed bench baselines
   (`bench/baselines/_*.json`) are scratch by convention but accumulate
   untracked/half-tracked; adopt the rule that `_`-prefixed artifacts are
   gitignored, and reconciled baselines get real names + a RUNBOOK line.
   [bench]

---
