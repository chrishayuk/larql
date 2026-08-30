# Container-generation policy: the VINDEX2 → VINDEX3 transition

Status: **contract adopted; default not yet flipped.**
The semantic catch-up phase closed 2026-08-22: VINDEX3 reached full LQL
parity with VINDEX2 (execution, inference, browse, mutation, patching,
compose + parity, COMPILE, logical DIFF, COMPACT), gated cross-platform,
with a real-model compose smoke green on granite-4.1-3b. VINDEX3 is the
**candidate primary generation**; VINDEX2 is the **compatibility
generation**.

## The migration contract

```text
New extraction        → VINDEX3 by default            (after the flip)
Existing VINDEX2      → continues to open and run
Generation detection  → bind once at load
Higher layers         → never care which generation was bound
New semantic features → V3 only, unless compatibility requires V2
V2                    → compatibility generation, no architectural expansion
Escape hatch          → explicit V2 extraction/binding during the transition
```

## The default-flip gate

The flip is a **named decision, made in exactly one place** — never a
side effect of a CLI default, a recipe template, or a surface-local
fallback:

- `crates/larql-vindex/src/format/generation.rs` —
  `DEFAULT_EXTRACTION_GENERATION` is the generation an extraction with no
  expressed preference writes. It is `V2` today.
- `crates/larql-vindex/src/format/generation_tests.rs` —
  `auto_extraction_resolves_to_v2_until_the_default_flip_is_decided` pins
  it. Flipping means changing the constant **and** this test in the same
  commit; that pair is the decision.

Every extraction surface resolves caller intent to a `GenerationRequest`
(`Auto` | `Explicit(generation)`) and passes it through
`admit_extraction_generation`. `Auto` gains its meaning only there.
"V3 was requested and refused" and "V3 was never requested" are distinct
by construction, and an explicit request is **never downgraded** — a
surface that cannot produce the requested generation refuses by name.

## Surfaces

| Surface | Request spelling | Today (`Auto` → V2) |
|---|---|---|
| LQL | `EXTRACT MODEL "m" INTO "o" [FORMAT VINDEX2\|VINDEX3]` | `FORMAT VINDEX3` encodes + auto-binds; `FORMAT VINDEX2` explicit; absent = policy |
| CLI | `larql extract --generation {v2\|v3}` | `--generation v3` encodes; omitted = policy |
| Factory | `extractor.options.generation: "v2"\|"v3"` → forwarded as `--generation` | absent = tool policy; a pin participates in `build_id` |
| V3 producer | `larql vindex3 encode <hf-artifacts>` | the multi-artifact system encoder |
| Bank import | `larql extract-index --expert-banks native --expert-banks-out DIR` | writes a V3 container in the **bank shape** (candidate spec §5.4): `moe_manifest.json` + LYRW segments, no system graph — the transitional import layout, not the graph shape the four surfaces above produce |

### What a V3 request produces

Both extraction surfaces run one shared pipeline —
`larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint`:

```text
checkpoint dir (config.json + *.safetensors)
  → build_inventory     interpretation, once
  → plan_system         admissibility, blocking findings ITEMISED in the refusal
  → encode_system       segments + system_graph + index
  → capability snapshot tokenizer.json + tokenizer_config/special_tokens_map/
                        generation_config/chat_template
```

The capability snapshot is what keeps a V3-produced container servable:
the encoder proper writes only what execution needs, so without it a
fresh container binds with token-id capability only and refuses `INFER`.
`larql vindex3 encode` places the same files, so all three producers
agree.

Sources the encoder cannot consume refuse **by name**, never by
downgrade: GGUF (a container may not transcode on the way in — use
`FORMAT VINDEX2`), non-checkpoint directories, and V2-only transcoding
flags (`--quant`, `--expert-banks`, `--compact`, …) passed alongside
`--generation v3`.

After a V3 extraction the LQL session is **already bound**, through the
same `bind_v3_session` block `USE` uses — gated by an equality test, so
`EXTRACT` cannot grow a second binding path that drifts.

Binding needs no policy: `detect_generation` (`index.json` schema
version, the sole discriminator — no filename or shape sniffing; V2 spans
schemas 1–2, V3 spans 3–4) already dispatches `USE`, the server
bootstrap, and `DIFF` operands, and fails closed on unknown schemas.

## Cross-generation contract: prompt encoding

A user prompt must reach both generations as the **same token sequence**.

The two surfaces learn the BOS fact from different places, and that is
deliberate:

| | Source of the BOS fact |
|---|---|
| V2 | the reconstructed `ModelArchitecture` — `arch.bos_token_id()` |
| V3 | the container's own carried `generation_config.json` → `tokenizer_config.json` |

V3 must not reconstruct architecture, so it reads the checkpoint's own
declaration, carried into the container by the M2 capability snapshot.
Both then prepend through one shared contract
(`larql_inference::maybe_prepend_bos`: prepend only when the tokenizer's
post-processor did not already).

This matters for exactly the models where the tokenizer's
`TemplateProcessing.single` template omits BOS while the model requires
it — Gemma 4 is the live case: architecture and `generation_config.json`
both say id 2. Gated by
`v2_and_v3_prompt_encoders_agree_on_a_bos_requiring_model`, whose control
(passing no declared BOS) reproduces the pre-fix divergence `[2,3]` vs
`[3]`.

A container that declares nothing — including any predating the
capability snapshot — encodes exactly as before.

## Consumer matrix (M3)

**Acceptance criterion: no VINDEX3 artifact can enter the system and
then silently disappear from a consumer surface.** Every consumer is in
one of two states — *supports*, or *refuses explicitly, naming the
generation*. "Hidden", "not found", and an empty listing are failures of
this contract, not neutral outcomes.

| Consumer | V3 state | How |
|---|---|---|
| `USE` / binding | supports | `detect_generation`, bind-once |
| `INFER` / `TRACE` / browse / mutation / lifecycle | supports | the V3 statement arms |
| `EXTRACT ... FORMAT VINDEX3` | supports | M2's shared encode pipeline |
| `SHOW MODELS` | supports | `summarize_container`, generation column |
| `/v1/models` | supports | `generation` reported for **both** classes |
| `larql vindex3 *` | supports | V3-native verbs |
| `link` | supports | aliasing is generation-agnostic |
| `run` | explicit refuse | `load_vindex_config` names the generation |
| Vindexfile `FROM` | explicit refuse | `load_vindex_config` names the generation |
| `publish` | explicit refuse | `load_vindex_config` names the generation |
| `slice` | explicit refuse | `unsupported_generation("slice", …)` |
| quantisation conversion | explicit refuse | `unsupported_generation(…)` |

Two of these were genuine defects rather than gaps:

- **`SHOW MODELS` hid V3 containers.** It listed only directories whose
  `index.json` parsed as a V2 config, so a container the user had just
  extracted did not appear at all. It now reads the generation-neutral
  `summarize_container`, names the generation in its own column, and
  lists a container it cannot identify as `unreadable` rather than
  dropping the row.
- **`slice` and quantisation conversion would have produced wrong
  output**, not an error: both select or parse by V2 layout, so a V3
  container yields a copy missing its weights, or a serde error about a
  missing field rather than about the generation. Both now refuse up
  front through `unsupported_generation`, one wording to grep for when
  these refusals start becoming implementations.

Capability strings are part of this contract. The V3 `SUPPORTED` string
lists `EXTRACT` and `SHOW MODELS`, which execute on a V3 binding and
previously went unadvertised — a binding that understates what it serves
is the same class of problem as a listing that hides what exists.

## Rungs to the flip

Three ladders touch VINDEX3 and measure different things — the crosswalk,
so no reader mistakes one for another:

- **M1–M4** (this document) — the *migration* ladder: when V3 becomes the
  default generation.
- **G0–G8** (`docs/vindex3-format.md` §2) — the *capability* ladder: what
  the graph pipeline can prove, from inventory to alternate plans.
- **V2-0..V2-4 / E-rows** (`docs/vindex3-experiments.md`) — the *bank-ABI*
  ladder: the pre-registered gates that freeze the LYRW/manifest bytes.

M4 does not wait on G8 or on the bank-ABI freeze; its evidence rows are
named below.

- **M1 — the seam. DONE** (PR #290): policy type, pinned default,
  explicit request spellings on every surface, refusals instead of
  downgrades. No behaviour change.
- **M2 — V3 production reachable from the extraction surfaces. DONE**:
  one shared `encode_checkpoint` pipeline behind `FORMAT VINDEX3` /
  `--generation v3`, capability snapshot closing the tokenizer/HF
  metadata gap on all three producers, post-extract auto-bind through
  `USE`'s own binding block.
- **M3 — consumer readiness. DONE**: every consumer supports V3 or
  refuses naming the generation; see the consumer matrix above.
- **M4 — the flip.** `DEFAULT_EXTRACTION_GENERATION = V3` + the pinned
  test, in one commit. V2 becomes the explicitly-requested compatibility
  generation. Evidence rows required by then: the real-model compose
  smoke (done — granite-4.1-3b) and whatever Metal-path/scale rows the
  release checklist names.

After the flip, V2 receives no architectural expansion — compatibility
fixes only.
