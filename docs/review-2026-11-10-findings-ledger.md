# Codebase review 2026-11-10 — findings ledger

Reviewed: `origin/main` at `6890239d`, by static reading only; the review itself compiled and ran nothing. Entries updated through 2026-11-11 (main `df20de1e`) as reproducers and fixes landed.

Evidence states used below:

- **code-read** — the defect is visible at the cited line and was read twice (reviewer, then a second read). The runtime consequence is still inferred until a reproducer exists.
- **reported** — one reviewer's reading; not re-read.
- **checked-live** — established against the live system (Fly, GitHub).
- **reproduced** — a failing test or run exists.

A finding moves to **reproduced** only with a failing test committed ahead of the fix. Owner is blank until assigned.

Proposed invariant these findings are instances of: *no successful response from LARQL conceals an operation that was skipped, partially applied, or substituted with different semantics.*

## P0 — exposure and enforcement

| ID | Finding | Where | Evidence | Next step | Owner |
|---|---|---|---|---|---|
| SEC-1 | Fly expert shards run keyless with `--insecure-public` while `fly.toml` publishes HTTP and gRPC | `deploy/fly/start.sh:53`, `deploy/fly/fly.toml` | code-read; **checked-live 2026-11-10: not deployed** — no `larql-expert-server`, `-b`, `-c`, `-d` app exists in the Fly account | Fix the config before any deploy: require an `API_KEY` secret or drop the public services | |
| SEC-2 | QUIC fingerprint verifier accepts any handshake signature | `larql-router-protocol/src/transport/quic.rs:262-264` | code-read | Delegate to the rustls provider's signature verification; test with a replayed certificate and a wrong key | |
| SEC-3 | Grid `DrainServer`, `AssignRange`, `Status` are unauthenticated; only `Join` checks the grid key | `larql-router/src/grid/service.rs:99,392,403,461` | code-read (router default bind `0.0.0.0` is reported, not re-read) | Interceptor applying the grid key to every RPC; loopback default | |
| SEC-4 | `--rate-limit` is inert without `--trust-forwarded-for`: no `ConnectInfo` on the served app | `larql-server/src/bootstrap/mod.rs:573`, `ratelimit/middleware.rs:29-55` | code-read | One test through the real listener; serve with connect-info | |
| SEC-5 | With `--trust-forwarded-for`, the leftmost `X-Forwarded-For` entry is trusted | `larql-server/src/ratelimit/middleware.rs:38` | code-read; **checked-live: `vindex3-explorer` exists on Fly, stopped with auto-start, so it wakes on request** and runs in this mode. Whether Fly appends or replaces the header is unverified | Use `Fly-Client-IP` or the rightmost entry | |
| CI-1 | Only `boundary` is a required check on main; nothing that compiles or tests gates a merge | ruleset 16196186 | checked-live; **fixed 2026-11-11**: `ci-gate` (#772, `df20de1e`) is required alongside `boundary`. It judges only workflows that ran, so CI-2 and CI-3 still stand | Done; keep CI-2 and CI-3 open | |
| CI-2 | Eight workspace crates plus `larql-experts` have no workflow (about 418 tests) | `.github/workflows/` | reported | One leaf-crates workflow: fmt, clippy, test | |
| CI-3 | Coverage jobs for `larql-cli` and `larql-inference` have no threshold or policy check | `larql-cli.yml`, `larql-inference.yml` | code-read (no `--fail-under`, no policy call) | Call the existing `make <crate>-coverage-summary` targets | |

Note on the five red `larql-cli` runs on main (2026-11-10, `65461938`..`c86aca04`): the failing test was `site_capture_dump::the_logit_dump_is_byte_identical_with_and_without_capture_on_both_backends`, a real race on the process-wide projection ledger. It was fixed at root by #767 (`fa0a7243`), with a mutation-checked regression test, not quarantined. No open flake is known from that incident.

## P1 — V2 editing and compile (Gate A)

| ID | Finding | Where | Evidence | Gate | Owner |
|---|---|---|---|---|---|
| EDIT-1 | `SAVE PATCH` on an anonymous patch takes the recording before the empty-path check, losing the ops | `larql-lql/src/executor/mod.rs:484-492` | **reproduced** — 6 of 7 tests in `tests/save_patch_transactional.rs` fail on `6890239d`; **fixed, merged in #770 (`4a872dfa`)** | A2 | |
| EDIT-1b | **Still owed from A2's original acceptance:** a `.vlp` is not published atomically; a write that fails part-way can leave a truncated file. Dropped from the A2 PR for lack of a failing reproducer, by decision after review | `larql-vindex/src/patch/format.rs` `VindexPatch::save` | code-read; no reproducer (needs fault injection) | open, unassigned | |
| EDIT-2 | `apply_patch` discards the result of `try_apply_patch`; callers report a dropped patch as applied | `larql-vindex/src/patch/overlay_apply.rs:84-86`; callers in lql and server | **reproduced and fixed, merged in #773 (`d2c9224c`)**: 15 of 18 LQL and 5 of 5 server reproducers failed on `6890239d` | A5 | |
| EDIT-3 | V2 `COMPILE INTO VINDEX` bakes from `patched.base()`; tombstones and meta overrides are never read | `larql-lql/src/executor/lifecycle/compile/into_vindex.rs:210` | code-read | A4 | |
| EDIT-4 | The V2 compile carry list omits `lm_head.bin` and every k-quant file; no quant guard | `into_vindex.rs:246-256` | code-read; the wrong-logits consequence is reported | A4 | |
| EDIT-5 | Remote DELETE/UPDATE are posted without the session header and apply server-wide | `larql-lql/src/executor/remote/mutation.rs:64-68,126-130` | code-read | Gate B (moved out of A5) | |
| EDIT-6 | Vindexfile `DELETE` matches on target only, ignoring entity and relation | `larql-vindex/src/vindexfile/mod.rs:150-156` | code-read; the substring-match breadth is reported | A4 or own PR | |
| EDIT-7 | `VindexPatch::save` is a bare write; a patch path can name a base file | `larql-vindex/src/patch/format.rs:152` | reported | A2 | |
| EDIT-8 | `UPDATE SET` fields and value types are unvalidated | `larql-lql/src/executor/mutation/update.rs:16-33` | reported | A1 follow-up | |
| EDIT-9 | `REMOVE PATCH` leaves up/down overrides on base metadata | `larql-vindex/src/patch/overlay_apply.rs:248-258` | reported | Gate B (moved out of A5) | |
| EDIT-10 | MERGE ignores `INTO`, records nothing, checks no compatibility | `larql-lql/src/executor/mutation/merge.rs` | reported | A6 | |

Follow-ups opened by A5 (2026-11-10), none of them fixed by it. Also not validated by A5: `target_id`, `top_token_id`, non-finite vector values, and the model named by a Vindexfile `PATCH`.

| ID | Finding | Where | Evidence | Gate | Owner |
|---|---|---|---|---|---|
| ID-1 | Model identity is a name, and V2 and VINDEX3 take it from different places (`index.json` `model` versus the encode-time name). No stable identifier or source-weight fingerprint exists in either format, so same-name different-weights is accepted and same-model different-name is refused | `larql-vindex` patch `base_model`; V3 `index.model` | code-read; pinned by the parity fixtures' shared `PARITY_MODEL` | separate design, not Gate A | |
| ID-2 | The server apply route checks slots, bounds and widths but not `base_model`; remote DELETE/UPDATE send an empty one. Model identity is enforced for local LQL `APPLY PATCH` only | `larql-server/src/routes/patches.rs`; `larql-lql/src/executor/remote/mutation.rs` | code-read | Gate B | |
| ID-3 | `DIFF … INTO PATCH` writes `base_model: "unknown"` with no bound model or on V3/remote, producing a patch `APPLY PATCH` now refuses | `larql-lql/src/executor/lifecycle/diff.rs:222` | code-read | follow-up: capture the real identity or refuse to produce the patch | |
| ID-4 | Server slot enrichment chooses a slot from a fixed 1..=10240 range, whatever the layer's width. A5 left enrichment unchanged and refuses a synthesised slot the layer does not have (a draft that folded the slot into the layer's width was rejected in review, because it reinterprets the patch) | `larql-server/src/routes/patches.rs` `enrich_patch_ops` | code-read; refusal pinned by `http_patches_apply_refuses_a_synthesised_slot_the_layer_does_not_have` | follow-up: allocate from the layer's real feature space | |

A4 acceptance cases to add: INSERT then DELETE then COMPILE; UPDATE then COMPILE; a model with a separate `lm_head.bin`; a k-quant base; a required file missing from the source. Each must either produce an artifact that contains the edit and every required component, or fail by name.

## P1 — inference correctness

| ID | Finding | Where | Evidence | Reproducer | Owner |
|---|---|---|---|---|---|
| INF-1 | `populate_kv_layer` and `truncate_kv_cache` set `current_len` but not `abs_position` | `larql-compute-metal/src/trait_impl/decode.rs:325,359` | code-read; callers reported | Populate-then-decode versus a fresh full-context forward at a nonzero position | |
| INF-2 | `populate_kv_layer` copy has no destination bound | `decode.rs:314-324` | code-read | Refusal test at `seq_len > max_seq` and on shape mismatch | |
| INF-3 | Windowed-checkpoint clips on `handle.cached_len()`, which is always 0 on Metal | `larql-kv/src/engines/windowed_checkpoint/dispatch.rs:197`; `larql-compute-metal/src/kv_dispatch_impl.rs:533` | code-read; the wrong-checkpoint-row consequence is reported | Metal test across a window close: bounded rows and output parity with CPU | |
| INF-4 | Remote MoE adapter is `best_effort`: a failed shard leaves the expert contribution at zero | `larql-inference/src/ffn/moe_remote/ffn_adapter.rs:73-78` | code-read | Shard returning 5xx must fail the generation | |
| INF-5 | A layer no shard owns returns a zero FFN; the shard map is not checked for coverage | `larql-inference/src/ffn/remote/sharded.rs:283` | code-read | `--ffn` map with a gap must be refused at connect | |
| INF-6 | CPU fallback in streaming generation drops sampling and the token callback | `larql-inference/src/layer_graph/generate/gpu/mod.rs:267`; `generate/cpu.rs:35-42` | code-read; the SSE consequence is reported | CPU-backend streaming test asserting callbacks and non-greedy sampling | |
| INF-7 | Q4 FFI wrappers pass unchecked lengths to C in release | `larql-compute/src/cpu/ops/q4_vecmat.rs:8`, `q4_matvec.rs:29-54` | code-read | Short-slice input must return a refusal | |
| INF-8 | Boundary gate fragility test is `<`, false for NaN | `larql-boundary/src/gate.rs:141-142` | code-read | NaN logit must not yield a continuation-safe decision | |
| INF-9 | Mid-generation backend failure breaks the loop with `error: None` | `generate/gpu/decode_loop.rs:190-202,239-250`; `generate/cpu.rs` | reported | Fault injection at step N must surface an error | |
| INF-10 | MLA positions advance before the route-refusal check | `larql-compute-metal/src/trait_impl/kimi_layer/encode.rs:208-231` | reported | "Advances nothing" test on refusal | |

## P2 — carried forward and smaller items

All **reported** unless noted.

- W10 parity gate toggles `LARQL_W10_HONLY`, which nothing reads (`larql-kv/examples/w10_parity_gate.rs:191-199`; code-read). Standing since 2026-09-01.
- Seven `let _ = index.load_*` on the V2 load path (`larql-vindex/src/format/load.rs:194-205`; code-read). Standing.
- `make test-models` runs a `larql-kv` test under `-p larql-inference` and aborts (`Makefile:31`; code-read).
- `larql pull --output` renames the cache directory (`larql-cli/src/commands/primary/pull_cmd.rs:246`; code-read; the dangling-symlink consequence is inferred).
- `larql hf download` copies top-level files only, so a V3 repo loses `segments/` (`extraction/hf_cmd.rs:220-232`; code-read).
- `larql recipe build` always fails at preflight while `docs/cli.md` documents it working (`larql-factory/src/build/stages/verify.rs:23-33`; code-read).
- Remote shard names are joined to the cache dir with no containment check (`format/huggingface/metadata_checkpoint/mod.rs:226-236,283`; code-read).
- Segment tensor `len` drives an unchecked allocation (`vindex3/opplan/exec/operands/load.rs:428`; code-read).
- V3 streaming and WebSocket generation cannot be cancelled.
- The router forwards client `Authorization` headers to any joined server.
- `larql verify` checksums cover nine float files only.
- A V2 Hub pull ignores per-file fetch failures.
- `_ => Silu` (`larql-inference/src/lib.rs:174`), missing norm weight becomes 1.0, invented per-family MoE defaults, `skipped_tensors` never read. Standing.
- `SHOW RELATIONS` hardcodes layers 14..27; parsed-and-dropped LQL clauses. Standing.
- Research KV engines ungated; `LARQL_MARKOV_WALK_KV_TOPK` alters the exact-contract engine with no trace in `info()`.
- Eleven Metal tests skip with no device; five `#[should_panic]` tests pass through the skip branch.
- `release.yml` triggers on `v*`, which also matches `vindex-v*`.
- AGENTS.md stale claims; `ROADMAP_STATUS.md` last updated 2026-08-22; 204 merged remote branches; actions on floating tags.
