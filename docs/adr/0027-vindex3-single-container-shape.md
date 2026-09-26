# ADR-0027 — VINDEX3 3.0 has one container shape; the LYRW bank shape is import-only

**Status:** Accepted 2026-09-26 as a decision; **execution pending** (closure
criteria below). This is gate 1 of the candidate specification's §21.
**Affects:** `crates/larql-vindex/docs/vindex3-format-spec.md` §5.4–§5.6,
§16, §21; `docs/vindex3-format.md` §9; `docs/vindex3-experiments.md` (E8
rule, bank-ABI rows); `format::vindex3::write` (the bank writer) and its
one production caller, `larql extract-index --expert-banks native
--expert-banks-out`.
**Related:** `docs/lyrw-v2.md` (the bank codec's own specification);
`docs/vindex-generation-policy.md` (M4).

---

## Context

The candidate specification records two V3 writers that share nothing but
the `index.json` envelope:

| Shape | Written by | Carries |
| ----- | ---------- | ------- |
| **graph** (§5.3) | `larql vindex3 encode`, `extract --generation v3`, LQL `FORMAT VINDEX3`, the factory | `system_graph`, representations, tensor-table segments, no `moe_manifest` |
| **bank** (§5.4) | `format::vindex3::write::write_container` — reached in production only by `extract-index --expert-banks-out`; otherwise the conformance fixtures and the `vindex3_import_gemma_*` examples | `moe_manifest.json`, `.lyrw` segment files, no graph |

§5.6 fixed the *direction* ("the graph/logical representation is the
format; a bank layout is an encoding of a representation") and left two
ways of executing it open: make LYRW a segment codec that graph
representations emit, or retire the bank writer to import-only status.
Gate 1 is choosing between them, and gates 2 (required/optional freeze) and 3
(`vindex-core`) cannot be scoped until it is chosen. Whatever top-level
shapes the freeze admits, an independent reader must implement and every
later release must keep supporting.

Three facts from the checkout (2026-09-26) settle the choice:

1. **The graph shape already represents packed routed banks without
   LYRW files.** `gpt-oss-20b.vindex3` is a routed MoE with a
   `target.expert_bank` representation and `moe_manifest: None`
   (`format/vindex3/index.rs`). `opplan/exec/experts.rs` binds a whole
   `[experts × rows, k]` bank as one codec region, with an optional paired
   scales stream, and decodes each expert as a row range of it.
2. **The graph shape has already absorbed the LYRW vocabulary that
   matters.** It uses LYRW's *region-binding* semantics — format tags → codec
   labels, `BlocksValues`/`BlocksScales` → paired streams —
   through `represent/codec/codecs/lyrw2.rs::bind_region`. It uses none
   of LYRW's *file* layout. No graph-path code opens a `.lyrw` file.
3. **Shared experts execute on the graph path.** `opplan::SharedExpertOp`
   carries DeepSeek-/Kimi-lineage shared experts beside the routed selection.
   Kimi-Linear-48B executes through it on the production CPU backend
   (`docs/v3-obs-1-carrier-observation.md`). The bank-shape runtime, by
   contrast, documents shared banks as unimplemented ("arriving with the
   Mini-K3 rung").

So the bank shape's container topology does nothing the graph shape cannot
already do. If it were made normative, a second container ontology would be
frozen into the public ABI purely to preserve the vehicle that proved the
sparse-bank machinery.

## Decision

1. **The graph container is the only normative top-level VINDEX3 shape
   at 3.0 Final.** A conforming 3.0 artifact carries `system_graph`.
2. **The bank shape becomes a legacy input, not a VINDEX3 3.0 shape.** A
   conforming 3.0 reader MUST recognise it (`moe_manifest` present,
   `system_graph` absent). It MUST then either open it or refuse it
   *by name*, as a legacy bank-shape container, and state the migration
   path. It MUST NOT report it as a conforming 3.0 container, and it MUST
   NOT fail on it with a generic parse error. The LARQL reader keeps
   opening it. An independent reader (`vindex-core`, gate 3) MAY refuse it.
3. **LYRW v2 stays an import/interchange codec, specified in
   `docs/lyrw-v2.md`, outside the VINDEX3 3.0 normative contract.** The
   part of it the graph shape uses, region format tags and packing →
   codec streams, is specified (gate 2) as part of the representation/codec
   contract and does not depend on the `.lyrw` file layout.
4. **`extract-index --expert-banks native|auto --expert-banks-out` is a
   legacy LYRW import/export facility and does not produce a VINDEX3 3.0
   container.** It is *not* taught to emit the graph shape. That would add
   a second graph producer, starting from a partially interpreted V2
   extraction context, and break the one path from source admission to
   SystemGraph to encoder. VINDEX3 comes from `extract-index --generation
   v3` or `larql vindex3 encode`. The V3 arm already refuses `--expert-banks`
   by name (`extract_index_cmd.rs`, `run_v3`). The gate-1 implementation PR
   rewrites the flag's help and output so they no longer suggest a
   normative VINDEX3 container. Where the wording cannot be made honest,
   the PR makes the flag refuse. `write_container` stays for the LYRW
   conformance fixtures, kernel bring-up and the import examples.
   (Executed: the route stays as the one pinned legacy facility, because
   `run`/`bench --routed-from` consume its output; see closure
   criterion 1.)
5. **The "bank-ABI pre-freeze" gate (§21 gate 6) is re-based on the graph
   shape.** The LYRW bank ABI is no longer frozen into 3.0, so rows that
   certify *that* ABI stop being 3.0 gates. Rows that certify a
   *shape-independent property* move to the graph shape and stay (see
   *Consequences*).

### Rejected: LYRW as a graph-representation segment codec

This would give the graph shape a second segment family (tensor-table
*and* `.lyrw`). Every conforming reader would then have to implement LYRW
descriptor parsing to read routed models, and it buys no representational
power. Fact 1 shows packed banks already live in tensor-table segments.
If a future representation needs LYRW's physical layout (for example,
page-aligned expert slabs for remote residency), it can be added as a
*registered segment codec* in a later minor version, without reopening
the container model.

## Consequences

### Gate 6 rows, re-based

| Row (experiments doc) | Shape-independent? | Fate under this ADR |
| --- | --- | --- |
| V2-0 profile authority cannot exceed derived authority | yes (`capability/authority.rs`) | **closed**, carries over |
| V2-0 variant-selection refusal | the property is | **open on the graph shape.** The pinned evidence is `read.rs::a_profile_selecting_an_absent_variant_is_refused_at_open` and `…::the_refusal_happens_before_any_segment_byte_is_read`. Both build fixture A with `write_container`, so they run on the **bank** reader. The graph encoder writes no variant catalogue. Re-homing means either graph containers carry variants and the same two assertions are re-pinned on one, or gate 2 moves variants out of the 3.0 MUST set. The bank-side ceiling still applies: steering has never run end to end |
| V2-1 native oracle vs container exact | yes | **graph execution *seam* parity: closed, scoped.** `larql-inference/src/vindex3/tests/mod.rs`: `reference_runtime_matches_the_decode_harness_bit_for_bit`, `production_…` and `dense_…` open a freshly encoded graph container and reproduce the direct `DecodeSession` harness bit for bit (logits) and id for id (generation), and `the_parity_instrument_detects_a_diverged_prompt` shows the instrument can fail. **Scope:** runtime vs harness over the *same* container. That is not the bank-era claim of container vs an independent oracle. The independent-oracle witnesses that exist are operator-level (below) |
| V2-1 fused vs decomposed identical | yes | **open, evidence needed.** No graph-shape test found that executes two representations (fused and decomposed FC1) of one model identically. The plan understands fused layouts, but that is not parity. The LYRW result is not inherited |
| V2-1 expert count / top-K not hard-coded | yes | **re-home, likely cheap**: real models with different (experts, top-k) through one plan path. Cite the pinned tests at gate-1 execution |
| V2-1 shared banks not hard-coded | yes | **open, re-scoped, no K3 dependency.** An operator-level oracle exists: `opplan/exec/tests/kimi_moe_real.rs::router_and_block_match_the_oracle_at_kimis_real_geometry`, which matches Kimi-Linear layer-1 (256 experts, top-8, one shared expert) against `modeling_kimi.py` through the shared branch. It is **env-gated** (`LARQL_KIMI_MOE_FIXTURE`; without the fixture it prints "skipped" and passes) and reads exported tensors, not a container. What is missing is container → plan → shared branch against a reference. Close it with a Kimi-Linear-48B token-parity run *or* a small committed graph fixture with a shared expert and an oracle, whichever is cheaper. A skip-on-absent test does not close it |
| V2-1 WALK/DESCRIBE parity | yes | **open** on the graph shape |
| Fixtures B–D | no (bank-shape conformance) | **leave 3.0**: they become LYRW-importer fixtures. Gate 3's conformance suite needs graph-shape fixtures instead |

### Specification text that must change (at gate 2, not in this ADR)

- §5.4 "Readers MUST accept it" → decision 2. "Writers SHOULD NOT extend
  it" → decision 4.
- §5.5 "carries at least one of `system_graph` / `moe_manifest`" →
  a conforming 3.0 container carries `system_graph`. The convergence
  configuration (both present) is no longer a target.
- §5.6 becomes a record of this decision.
- §16 criteria 2–4 are written in LYRW-bank terms ("the same LYRW2 bank
  machinery", "K3 … served"). Criteria 2 and 3 restate over the graph
  shape. **Criterion 4 (K3 served) is a K3 dependency inside the success
  criteria**, and gate 2 must say explicitly whether 3.0 Final requires all
  seven criteria or only the ones §21 names. This ADR does not decide that.
- The E8 rule's forbidden list (`LYRW2 byte-layout changes · new region
  roles · new packing modes`) must be restated in graph-shape terms
  *before* E8 runs. The draft wording is: no `index.json` or SystemGraph
  schema change, no new `ObjectKind` / `ComponentRole`, no new codec label,
  no kernel-interface change, and no loader case keyed on the model.

### E8 candidate eligibility (recorded for later)

Mixtral appears only in `larql-models` detection and has no footprint in
`format/vindex3`. Qwen-MoE is **not** clean: `SharedExpertOp` documents
Qwen-lineage shared-expert sizing, so Qwen shaped the operator. The
eligibility audit belongs to E8. This entry only keeps the observation.

## Closure criteria

Gate 1 is **executed** when all of these hold, each backed by a test or
recorded check:

1. **Exactly one production bank-shape producer, and it is labelled
   legacy.** The bank shape has a live consumer: `larql run` and `larql
   bench --routed-from DIR` compose a VINDEX2 spine with bank-shape
   routed banks, on CPU and Metal, through `ContainerRoutedBackend`.
   Its only producer is `extract-index --expert-banks native|auto
   --expert-banks-out`, which runs through
   `extract::orchestrate::write_native_container` → `ContainerBuilder`.
   Chosen 2026-09-26: keep that route as the single enumerated legacy
   LYRW facility, rather than retiring it or first migrating
   `--routed-from` to graph containers. Its CLI help and runtime output
   name it as legacy LYRW and not a VINDEX3 3.0 model.
   `crates/larql-vindex/tests/bank_shape_producer_closure.rs` scans every
   non-test source file in the workspace, and pins every production
   reference to `write_container` or `ContainerBuilder` to that one owner.
   A new producer fails the test. Retiring the facility later means
   removing the owner and emptying the pin together.
2. **Named legacy recognition.** `larql vindex3 inspect` and `verify` on
   fixture A report it as a legacy bank-shape container, non-normative
   since 3.0, and name the migration path. A test pins the wording's
   fields.
   *Executed.* `Vindex3Index::shape()` (`format/vindex3/shape.rs`) is the
   one discriminator, read from which authorities `index.json` names;
   an index naming neither is refused (§5.5). The graph-path surfaces
   (`inspect_container`, and through it `vindex3 inspect`, `vindex3
   verify`, exec, represent and compile) refuse a bank container with
   the typed `VindexError::LegacyBankContainer`, which names the ADR and
   `LEGACY_BANK_MIGRATION` (re-extract from source for now; see
   criterion 3). `larql show` and `larql verify` print the shape, plus the
   migration line for a bank container, and keep reading it. Recognising
   the shape exposed the reverse gap: `larql verify` sent *every* V3
   container to the bank reader, which refuses a graph container, so the
   normative shape could not be verified through the top-level command.
   It now dispatches on shape, and a graph container is verified through
   `inspect_container` with payload re-hashing. Pinned by
   `format/vindex3/shape_tests.rs` and
   `larql-cli/src/commands/extraction/verify_cmd_tests.rs`.
3. **Artifact inventory.** The bank-shape containers that exist outside
   tests are listed. For each one that must survive, a re-encode to the
   graph shape keeps every weight-region payload hash identical,
   carries the programme semantics into the graph's routed op, and does
   not raise authority or fidelity. Where the inventory finds none, this
   criterion is met by recording that, and re-extraction from source is
   the migration path.
4. **A graph-shape conformance fixture** exists: tiny, deterministic,
   routed + shared, with an oracle. It replaces fixture A as the fixture
   gate 3's reader is certified against.
5. **Specification updated.** §5.4–§5.6, §9 of the living doc, §16 and the
   E8 rule carry the text above, and §21 records gate 1 as executed.
