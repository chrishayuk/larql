> Preserved verbatim. Independent targeted review of `11376af6:docs/mem-002-i1-i3-proposal.md` (draft PR #671), 2026-11-01. Verdict: FAIL (F1 blocking).

# PR #671 targeted review — MEM-002 I1 + I3 implementation proposal

Reviewed: `11376af6:docs/mem-002-i1-i3-proposal.md` (previous `a4fa09cc`). Code read at `11376af6`;
`git diff 11376af6 b1e6a524` touches only the proposal doc, so the code is identical to main.
Context: `5c14a712:docs/mem-002-cold-state-qualification.md` ([Q]). Read-only. Nothing was built or run.

## 1. Corrections

| # | Correction | Where | Verdict |
|---|---|---|---|
| 1 | One capture per observation; both views derived from it; start/end and scan order kept; a scan is not called atomic | §3.2 (`ResidencyObservation`, `observe_residency`, `summary()`, wall and monotonic `started`/`ended`, `scan_index`, "A scan over many regions is not atomic"); §4 "One observation" test compares both views from one snapshot | **IMPLEMENTED.** Gap (minor): the per-step delta needs the *previous full observation*, so the curve's `before_res: MappedResidency` closure state (`residency_curve.rs:229,234,274`) must become a `ResidencyObservation`. This is not stated. |
| 2 | PC1: cross-process visibility on controlled known ranges; no whole-file residency assumed; positive detection, attribution and read success | §5 general rules + PC1 | **IMPLEMENTED.** dd ranges, observation window, exit status and byte count, pass = zero failed + fraction + attribution, "does not require … whole file to remain resident". See F4 (range-to-region mapping) and F3 (process side effects). |
| 3 | PC2: no exact-zero requirement; prior residency, readahead, boundary pages and concurrent activity accounted for; positive and null controls; classes fixed before the outcome | §5 PC2 class table, sham, pass criteria, "Exact zero … not required" | **IMPLEMENTED, with a material weakness (F3).** The known-positive increase is not compared with the same regions under the sham, and process A runs a decode between reports A and B. |
| 4 | Versioned JSONL with phase, region identity, status and failure reason | §3.4 | **IMPLEMENTED.** `schema` `larql.residency-observation/1`, `record`, `observation_id`, `phase`, `scan_index`/`store_id`/`store_offset`/`len`, `status`, `error{kind,errno}`, and refusal of unknown versions. Minor gaps: F6. |
| 5 | A failed region never enters an admission-quality aggregate silently; admission stays I2 | §3.2 (`resident_bytes` sums read regions only, `failed_regions`, `complete_resident_bytes()`), §3.4 "Nothing here admits", §1 out-of-scope | **IMPLEMENTED.** The guard is by convention: `resident_bytes` stays a public, partial field (§6 Q2 leaves this open). `Default` gives `failed_regions = 0`, so `MappedResidency::default().complete_resident_bytes()` would be `Some(0)` (F7). |
| 6 | Code and API claims checked against the code | see §2–§3 below | **Mostly correct. One infeasible test row (F1, BLOCKING).** §2.3 is incomplete, but not on the witness path (F2). |
| 7 | Stays a proposal | whole document | **YES.** Only signatures and pseudo-structs, no code. I2, I4, I5 and I6 are out of scope. PC1 and PC2 are "not authorised to run" until the freeze. |

## 2. Code and API claims

| Claim | Checked at `11376af6` | Verdict |
|---|---|---|
| `WeightRegion::resident_bytes()` at `physical.rs:269-307`: `Some(0)` when empty, `None` when `sysconf<=0` or `mincore` rc≠0, `None` on non-unix, pages×page clipped to len | fn at 269, closing brace at 307; behaviour matches | ✓ |
| Boundary pages are counted by every overlapping region | `first = start & !(page-1)`; each region queries its own aligned span | ✓ |
| The failure cause is discarded | `if rc != 0 { return None; }` | ✓ |
| errno can be captured at the call site | ✓ feasible: `std::io::Error::last_os_error().raw_os_error()` right after `rc != 0`. The crate's examples already do this (`examples/glm_moe_residency/probe.rs:169`, `examples/vindex3_residency_probe.rs:224`) | ✓ |
| `WeightRegion` exposes what `RegionResidency` needs | `store_id() -> &str` (= the store's `id`, which is the operand **object** name, see `operands.rs:282-284`), `store_offset()` (offset in the whole backing, header included, i.e. the **file offset** for `map_segment`/`map_compiled`), `len()` | ✓ (see F4 on the file path) |
| `LoadedWeight::resident_bytes` maps `None`→0 at `loaded.rs:37` | `LoadedWeight::Mapped { region, .. } => region.resident_bytes().unwrap_or(0) as usize` | ✓ |
| `mapped_region() -> Option<&WeightRegion>` implementable | `LoadedWeight::Mapped { region: WeightRegion, form }` (`weights/mod.rs:186-189`) | ✓ |
| `mapped_residency()` at `residency.rs:417-…` sums weights with `mapped_bytes>0` | 417-456. Traversal covers dense_ffns, routed_experts, every attention variant (Softmax/GatedDelta/Mamba2/ConvQkv/Kda/Mla), layer.ffn and the output head. `LoadedWeight::Mapped` is constructed only at `experts/routed.rs:347`, and `embed_table` is a `Vec<f32>`. So the traversal covers every place a mapped weight can sit. Sources list "412-440" stops short of the output head (452-453); §2.2's "417-…" is fine | ✓ (cosmetic) |
| `MappedResidency { regions, mapped_bytes, resident_bytes }` at `census.rs:51-55` | ✓ `Debug, Clone, Copy, PartialEq, Eq, Default`. Adding `failed_regions: usize` keeps every derive, and `kimi_per_expert_prepared.rs:417` (`== Default::default()`) still compiles | ✓ |
| `SiteResidency::add` (`census.rs:70-80`) is not a `None→0` consumer for mapped weights | `mapped>0` takes the first branch, so `resident_bytes()` is called only for non-mapped weights | ✓ |
| `DecodeSession::mapped_residency` at `steps.rs:121-123` | ✓ | ✓ |
| `budgets.rs:410-419`, `:417` | `resident_bytes: self.weights.iter().map(\|w\| w.resident_bytes() as u64).sum()` at 417 | ✓ |
| reconcile never reconciles mapped resident bytes | `reconcile.rs:63-82`: for mapped, it adds to `mapped_resident` and `continue`s | ✓ |
| `residency_curve.rs:187`, print `:195-199`, `:229`, `:237` | ✓ all exact. `resident_delta` at `:255` = `now_res.resident_bytes.saturating_sub(before_res.resident_bytes)` | ✓ |
| Reads stay outside timers | binding read at 187 is before any pass; per-step `observe` runs after `started.elapsed()` (`:288`, `:311-313`) | ✓ (the new per-step `Vec` and `String` allocations are also untimed) |
| `vindex3_cmd/tests/generate.rs:79` is `the_residency_curve_runs_cold_and_warm_passes_over_one_bound_image` | ✓ name correct. **But** it runs `fixture_dir(true)`, a dense Glimmer (`tests/mod.rs:49-…`, no experts), so it has **zero mapped regions**. It also only calls `run(...)` and asserts nothing about stdout | see F1 |
| `represent/tests/physical_store.rs:56-67` | `an_empty_region_has_no_resident_bytes`, unix-gated `Some(0)` assertion | ✓ |
| `gwv2.rs:257` | ✓ the line exists, but it reads `ops.mapped_residency()` (the aggregate), **not** `resident_bytes()` directly. The classification is right; the mechanism shown in the table is slightly off | minor |
| "One observation" test on `kimi_per_expert_prepared.rs` | the fixture binds mapped expert regions (`:393-406`), so the test is feasible there | ✓ |

## 3. `None → 0` inventory (complete, from `git grep "resident_bytes()"` and `mapped_residency|MappedResidency|.resident_bytes` at `11376af6`)

`WeightRegion::resident_bytes()` has exactly **one** production caller: `loaded.rs:37` (`unwrap_or(0)`). There is no other `unwrap_or`/`map_or` on it. (The `map_or(0, …)` sites in `larql-kv` act on `Option<KvHandle>`, which is unrelated.) The other test call is `physical_store.rs:66`.

Downstream of `loaded.rs:37`, for mapped weights:

| Consumer | Kind | In proposal? | Witness path? |
|---|---|---|---|
| `prepared/residency.rs:423` `mapped_residency()` | aggregate | §2.2 (changed) | yes |
| `decode/steps.rs:122` delegate | aggregate | §2.2 (changed) | yes |
| `residency_curve.rs:187/198, 229, 237/255/266` | printed witness and delta | §2.2, §3.4 (changed) | yes |
| `accounting/budgets.rs:417` `Observed.resident_bytes` → `reconcile.rs:81` `mapped_resident` | accounting | §2.3 (unchanged) | no |
| `vindex3_cmd/realizations.rs:111` prints `reconciled.mapped_resident` | user-visible print | **not listed** (it is the "reported" end of the budgets row) | no |
| `larql-demos/examples/inference/gwv2.rs:257/259` `mapped_resident_bytes` in `execution-image.json` | demo artifact, via the aggregate | §2.3 (unchanged) | no |
| `opplan/exec/tests/carrier_write_real.rs:166-173` prints `mapped.resident_bytes` as run evidence | test print | **not listed** | no |
| `opplan/exec/tests/kimi_per_expert_prepared.rs:193-195, 393-417` | test asserts | not listed (tests) | no |
| `opplan/exec/tests/accounting/every_resident_form_the_cpu_loader_produ.rs:232` | test sum of `Observed.resident_bytes` | not listed (test) | no |
| `prepared/census.rs:76,78` `SiteResidency::add` | non-mapped weights only | §2.3 (correctly excluded) | n/a |

Conclusion: §2.3 is **correct but incomplete**. It misses `realizations.rs:111` and `carrier_write_real.rs:166`, plus test asserts. No consumer on the **witness path** is missed. Note that after the change, gwv2 and carrier_write_real read a sum over *read* regions and ignore `failed_regions`. They stay failure-blind, but they no longer add failures as 0.

## 4. Findings

**F1 — BLOCKING: the CLI and Schema test rows are infeasible or vacuous as located.**
- The "Schema" row says the CLI test pins "a failed region carries `status: "failed"` and an error kind". §4's own "Honest limits" says a real `mincore` failure cannot be induced in CI, so no CLI run can emit a failed region.
- `generate.rs:79` uses the dense Glimmer `fixture_dir(true)`. The only constructor of `LoadedWeight::Mapped` is routed experts (`experts/routed.rs:347`), so that fixture has **zero mapped regions**. "`--residency-regions` writes one JSON line per mapped region" is therefore vacuous (only header lines), and "the binding line names read and failed counts" is always `0 read, 0 failed`.
- The test only calls `run(...)`, which `println!`s to stdout. Without refactoring the formatting into a pure function or a writer, it cannot assert on the binding line or on the absence of `WITNESS INCOMPLETE`.
- Fix:
  - pin the JSONL serialiser (ok and failed region records, every error kind, `errno` present only for `mincore`) and the binding-line formatter directly, from constructed `ResidencyObservation`s;
  - keep the CLI test for the end-to-end smoke check (file exists, header record carries the version, `regions == failed + ok lines`), and either state that it has 0 mapped regions or add an MoE CLI fixture.

**F2 — minor: complete the §2.3 inventory** with `realizations.rs:111` (the user-visible print of `mapped_resident`) and `carrier_write_real.rs:166`. Show gwv2 as an aggregate (`mapped_residency()`) consumer.

**F3 — major (should fix before the freeze): the PC2 positive control is not defended against process side effects or concurrent activity.**
- (a) `--residency-regions` is a flag on the residency curve, which always runs at least one pass (prompt plus decode). So the process that writes report A then executes and touches routed experts before step 2's read, and report B sees those pages too. The proposal has no bind-only mode.
- (b) The pass criterion compares only *unobserved* regions with the sham. "Known-positive regions increase" is not compared with the same regions' change under the sham. A known-positive region that the run routed to, or that background activity touched, passes without the dd read having caused anything.
- Fix: either add a bind-observe-exit mode (the binding observation is already taken before any pass), or require known-positive change minus the same regions' sham change to exceed the frozen fraction.
- The same applies to PC1: the binding observation is untouched by the fresh process itself, but the observation window must not include an earlier process's run.

**F4 — minor: range→region attribution needs a file identity.** `store_id` is the operand object name (`operands.rs:282-284`), not the segment path. `store_offset` is the file offset including the segment header. PC1 and PC2 map dd byte ranges of `.s7` files to regions, so the JSONL should carry, or the procedure should state how to resolve, the segment path per `store_id`. The proposal should also state explicitly that `store_offset` is a file offset.

**F5 — minor: the delta's arithmetic and its empty case are undefined.**
- "Delta over regions read in both" does fix the mixing of successful and failed reads that the current aggregate `saturating_sub` (`residency_curve.rs:255`) would allow.
- Not specified:
  - (i) signed net vs per-region saturating;
  - (ii) the match key. `scan_index` is valid within one immutable `PreparedOperands`, but should be asserted against `(store_id, store_offset, len)`;
  - (iii) the value when no region was read in both. That must print as n/a, not `0.000`, or the fix reintroduces a `None→0`.
- The per-step "mapped res" column (`:266`) is a partial sum when failures exist, and should be marked so.
- The region set cannot change within one curve run (the image is immutable). Across processes (PC2) the match must be by identity.

**F6 — minor schema details.**
- `errno` should be present only for `kind:"mincore"`.
- `page_size` needs a defined value when it is unavailable.
- `observation_id` scope is unstated.
- The header has no process or container identity (pid, container path and hash, binary or commit), which PC2 needs to pair reports A, B and the sham.
- The observation header could also carry the read regions' mapped bytes, so a resident fraction is computable over read regions only. `summary().mapped_bytes` is presumably over all regions; state it.

**F7 — minor API notes.**
- An empty region on non-unix is `None` today. "Empty stays `Ok(0)`" changes that unless `Unsupported` is returned first; state which.
- `Default` makes `complete_resident_bytes()` `Some(0)`.
- The "Compatibility" test is tautological once `resident_bytes()` is literally `residency().ok()`.
- `RegionResidency` owning a `String` per region per step is untimed but allocates between tokens on banks with many experts; an index into a store table is cheaper.

**F8 — cosmetic.**
- `[D §1.1]` and `[R §1.1]` are cited but not defined in this document. They are defined only in [Q].
- The relaxed PC2 differs from [Q §4]'s text ("exactly the regions … none elsewhere beyond boundary pages"), so #670 needs the matching amendment.

## 5. Verdict

**FAIL (one blocking item: F1).** Corrections 1–5 and 7 are implemented, and the code and line claims are accurate. errno capture and `mapped_region()` are implementable, and the traversal covers every place a mapped weight can sit. Fixing F1 is a test-plan edit. F3 should be fixed before the freeze; it is a procedure change, not an implementation blocker. F2 and F4–F8 are small edits.
