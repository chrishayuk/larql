# The representation/execution contract

**Status:** rung 1 landed — the trait, extracted from the five encodings the
container already carries, with the registry in front of it and three
consumers rewired to it. Code: `crates/larql-vindex/src/format/vindex3/represent/codec/`.

## Why a contract, and why now

VINDEX3's claim is that it can describe what a model *means* independently of
how that meaning is physically represented, resident, or lowered for
execution. Until this rung, that claim was carried by discipline rather than
by a type: the facts a stored encoding has to declare were scattered across
seven surfaces — a K-quant geometry table, an expert-encoding match, the
NVFP4 pack layout, private group constants in the loader, the runtime's
residency-format enum, the compute crate's `QuantMatVec` dispatch and the
K3 ledger's maturity ladder — and two formats (ternary I2_S and MXFP4) had
already been routed *around* the trait that should have served them
because a single `&[u8]` had nowhere to put their scales.

Left alone, the next hard model's physical constraints would have defined
the abstraction. The contract is written before K3 and residency become
dominant, so that they plug into it rather than shape it.

## The contract

`RepresentationCodec` is not a quantisation trait. Six things it gets
right, each already paid for once in this tree:

| Concern | What the codec declares | Why it is there |
| --- | --- | --- |
| identity | `encoding_label()` — the label a container writes; `identity()` — ABI family + revision + geometry | the file names its contract independently of whichever implementation is registered |
| streams | `streams()` — a **named set**; `CodecOperands` = streams + `AuxiliaryOperands` | one slice was the wall MXFP4 and I2_S hit; auxiliaries are represented objects a decode depends on (a codebook), named now while empty |
| capabilities | access granularity (sequential / block / row / element), logical grouping, physical alignment | the planner refuses by name in preflight — Walk FFN needs rows and a stream-sequential codec cannot serve it — rather than a kernel discovering it |
| extents | `extents()` — one certificate per admissible depth, bits/weight and an optional error radius | every codec answers depth 0 today; a progressive codec exposes one per prefix, and residency can then ask for *enough representation* |
| decode | `decode_rows()` — **mandatory**, range-aware, to canonical f32 | the universal correctness surface; kernels are acceleration, never semantics |
| realizations | `decode_residency()` required; `accelerations()` each carry their own `ResidencyProfile` | residency belongs to the realization, so a fallback is a different realization with a different declared cost, never a quiet substitution |

Two rules follow. *Adding a codec requires proving representation
correctness; adding a kernel must not change it.* And a codec with no direct
realization is not a defect: it executes through the reference path, flagged
— spec §11's `representable-but-no-kernel`, made structural.

What a certificate deliberately does **not** say: fidelity to a source. That
is a property of an instance, set at extraction from provenance and carried
by the stored variant. A native MXFP4 checkpoint stored as MXFP4 is
source-exact; the same bytes compiled from bf16 are approximate; the codec is
the same in both.

## What rung 1 extracted

Five encodings, none privileged:

| Codec | Streams | Access | bits/weight | CPU direct realization |
| --- | --- | --- | --- | --- |
| BF16 / F16 / F32 tensor tables | 1 | element-random | 16 / 16 / 32 | bf16: `FusedBf16`, `Bf16xQ8`; f32: `BlasF32`, `ScalarF32`; f16: none |
| Q4_K / Q6_K / Q8_0 (scales inline) | 1 | row-random | 4.5 / 6.5625 / 8.5 | none on the V3 CPU executor |
| NVFP4 (one row, split derived from shape) | 3, via `bind_packed` | row-random | 4.5 | `FusedNvfp4` (rebound) |
| MXFP4 (codes and e8m0 scales **apart**) | 2 | row-random | 4.25 | none on the V3 CPU executor |
| LYRW v2 banks | — | — | — | a *storage arrangement*: `RegionFormat` → codec label, `Packing` → single / paired-values / paired-scales binding |

The multi-stream witness is MXFP4: it does not override `bind_packed`, so
handing it one payload is refused naming both streams — the answer
`QuantMatVec` could only give as `None`.

Consumers now derived from the codec rather than restating its facts:

- `OperandStore::load` dispatches every stored dtype through the registry
  (floats, K-quants, NVFP4; an unregistered dtype is refused naming the
  registered ones).
- `CodecIdentity::admit` is the registry's `admit` — unknown family,
  foreign revision and disagreeing geometry stay three distinct refusals.
- `ExpertEncoding::matrix_bytes` prices a bank through the codec.
- The loader's MXFP4 group constants and the graph builder's MXFP4 label
  each have one home.

Tests (`codec/tests/`) run every check over the whole registry, pin each
codec's decode to the path it replaced, and pin the older tables to the
codec — so a drift fails in a unit test rather than in a 20 GB segment.

## Boundaries of this rung — stated, not implied

- **The executor does not yet select a realization through the trait.**
  `PhysicalProjectionPlan` still chooses; the codec *declares* which plans
  serve it (positive evidence, keyed to the plan enum) and what each costs.
  The trace record `requested / selected / reason / residency` is the next
  rung.
- **Only CPU realizations are declared.** Device kernels (NVFP4 and MXFP4 on
  Metal, the grouped K-quant expert kernels) are declared by the peer crate
  that owns them, not claimed here.
- **K-quant packs have no direct realization on the V3 CPU path.** That is
  the truth of `PhysicalProjectionPlan` today; the K3 ledger's `Production`
  rung for Q4_K/Q6_K refers to the V2 `QuantMatVec` executor.
- **Loading is stricter than before, in two places.** A K-quant row must
  be a whole number of blocks — the compiler always enforced this at write
  time and the loader now enforces it at read time, so a `[2, 128]` Q6_K
  tensor, decodable and meaningless before, is refused. And a float operand
  whose bytes are shorter than its declared shape implies is refused rather
  than silently decoded short. The routed-FFN loader consequently checks
  the expert bank's *declared* geometry before it reads any operand, which
  its own test had always claimed ("refuses before touching the bytes") and
  which only held before because the widener under-decoded in silence.

## The programme this opens

Four proofs, each with a minimal witness, then adversarial confirmations:

| Proof | Witness | Forces |
| --- | --- | --- |
| Representation is not a dtype | progressive / residual codec (`R = R₀ + Δ₁ + … + Δₙ`) | extents with meaning; prefix identity; residency choosing depth; "enough representation" as a planner request |
| Representation is not self-contained bytes | VQ / codebook | `AuxiliaryOperands` in anger: an encoded operand depending on another represented object |
| Storage is not execution residency | entropy-coded bf16 (zstd / ANS) | sequential access refused by name for row plans; storage / decoded / executable / workspace residency told apart |
| Canonical semantics are source-independent | HF round-trip, then `.fs3`/`.fsc` lowering | meaning → foreign vocabulary without special-case reconstruction |

Adversarial confirmations: ternary / base-243 (element and byte boundaries
diverge), per-row mixed rate (shape does not determine offset), permutation
sidecar (physical order is not logical order), MLX lowering. If any of these
takes major surgery, the representation layer is still leaking a
physical-layout assumption.

Order: entropy-coded bf16 first — it is cheap and tells immediately whether
the extracted contract preserved an mmap / random-access assumption. Then
progressive. Then VQ. If those three arrive without changing the trait, it
is frozen, and LARQL no longer knows what quantisation formats exist — only
what properties an executable representation must declare.

## Rung 2 — entropy-coded bf16, the hostile sixth codec: HELD

Preregistered before any code in
[`represent/forecasts/rung2-entropy-coded-bf16.json`](represent/forecasts/rung2-entropy-coded-bf16.json)
(frozen, unedited); scored in
[`represent/forecasts/rung2-execution-notes.json`](represent/forecasts/rung2-execution-notes.json).
Code: `codec/codecs/bf16_zlib.rs`, the ninth registered codec.

`BF16_ZLIB` is one RFC 1950 stream per tensor inflating to the row-major
little-endian bf16 image — sequential by construction, with a stored
length that is instance-dependent (a repetitive tensor stores fewer bytes
than raw bf16, a noise tensor more) while the decoded length stays
shape-derived, and with no direct realization registered. The identity names
the wire format and the element grid, never the library; the lossless
claim is proved at the bit level against a stream written by a *different*
implementation (Python's zlib, `scripts/gen_bf16_zlib_fixture.py`).

| Property | Result |
| --- | --- |
| executable through registration alone | one `.register` line; prepared / production / physical / weights / operands untouched; candidate logits **bit-exact** to a raw-bf16 control under the production backend |
| the contract leak | exactly the one forecast — `stored_bytes(shape)` — costing one refusal variant, `InstanceSized`; every other contract file byte-identical to `f92fac65` |
| sequential, refused by class | the packed-bank preflight asks the registry `require(RowRandom)` **before** reading; refusal names `sequential` vs `row-random`, and `load_count` does not move |
| no direct realization | `accelerations()` empty; the executor observes `BlasF32` over an f32 image |
| residency | the census agrees with `decode_residency()` for every transcoded site, and a mutated declaration would break the agreement — a check with teeth, not two readings of the f32 default |
| source touch vs working set | the container's recorded length (≠ 2·elements, either direction) vs 4·elements resident |
| pre-registration control | the eight-codec registry refuses the label naming the eight; the same bytes under an unregistered label are refused at load, by name |

Seven whole-registry gates collided, not the six forecast: each was
classified in place — accidental universals (row access for every codec,
size from shape, validate-by-length) generalised **by declaration, not by
label**; the genuine requirement (a codec with an acceleration provides
rows) retained; rosters extended. The generalised short-stream gate then
caught a real gap in the first implementation: a reader adapter that
reports a missing Adler-32 trailer as end-of-file. Whole decodes now
require a positively witnessed stream end.

What this earns, in the user's wording: LARQL supports plug-in
representations with different storage and access semantics.
"Representation-open" waits on VQ and progressive; pluggable *lowering
targets* are enabled by the seam but not delivered — rung 3 (planner
admission, realization trace) is their prerequisite.

## Rung 3 — planned execution realizations: preregistered, not yet built

Frozen before any code in
[`represent/forecasts/rung3-planned-realizations.json`](represent/forecasts/rung3-planned-realizations.json),
with the baseline measured at the rung-2 merge: the seam between plan and
backend is three stored-dtype booleans, selection is a boolean ladder over
them, the kernel is observed from resident bytes rather than pinned, MXFP4
banks enter through a `U8` label nobody registered, and realization identity
is a closed enum. The forecast fixes the contract-level design — a
hardware-independent `PlannedOperand`, a derived candidate set, one pinned
`RealizationId` with its reason and resource profile — and predicts the
transition per wave (3a requirements, 3b admission and selection, 3c trace
and accounting, 3d privileged paths removed), including the blocker an
external provider is expected to hit. Execution notes go in a sibling
`rung3-execution-notes.json`; the forecast is immutable once committed.

### Rung 3 — built and held, with the falsifier that fired

All four waves ran on branch `represent-rung3`; the execution record, wave by
wave with every finding the forecast did not contain, is
[`represent/forecasts/rung3-execution-notes.json`](represent/forecasts/rung3-execution-notes.json).
What the rung leaves behind:

- **A plan is a set of hardware-independent planned operands** (`PlannedOperand`:
  operation, required access, logical extent, logical elements, and — since 3d —
  the codec the plan *declares* when a container's label is a carrier dialect
  rather than a codec). Nothing in it names a dtype or a kernel.
- **Admission and selection happen at preparation, before any byte.** The
  backend derives a candidate set from the codec's declarations, pins one
  `RealizationId` with its reason, or refuses with every reason. The three
  stored-dtype booleans, the `U8` bank dialect and every dtype-name judgement
  under `exec/` are gone; `ExpertFormat` keys exactly one thing, the declared
  codec of a packed bank.
- **The declaration is bound against the census, not into it.** An expectation
  is priced from the pin, the codec's declared residency and the container's
  recorded length; an observation is read off the bound objects; the two are
  reconciled, and a mutated declaration breaks the reconciliation of exactly
  the forms it prices.
- **The registry is a value, not a default.** The store carries the registry it
  decodes through, the prepared image records the one it was prepared through,
  and execution checks its providers against that one. This is the
  falsifier that fired (F8): the first run of the provider proof was refused at
  *execution* by three hidden built-in registries that registration could not
  reach, after selection had already succeeded.
- **The acceptance test is external.**
  [`crates/larql-vindex/tests/external_codec_provider.rs`](../crates/larql-vindex/tests/external_codec_provider.rs)
  is an integration test — a separate crate in cargo's model — that registers a
  representation this build does not ship and executes a container stored in
  it: the same selection as the shipped label, bit-exact logits, refusal by name
  without the provider, invalidation of an image that loses it. It touches only
  exported API.

**The claim rung 3 earns:** LARQL can accept a representation implementation
through exported API, carry its registry through preparation and execution,
select and account for it before I/O, execute it bit-exactly, and invalidate
prepared state when that provider disappears, without privileged dtype control
flow.

Two wording boundaries stay explicit:

- The integration test proves *external-crate compatibility*, because Rust
  integration tests compile as separate crates against exported API. It does
  not claim runtime discovery or distributed package loading; a plugin here
  is an external crate linked at compile time.
- "Stored label wins" makes the **stored representation authoritative**. The
  expert-format declaration is a legacy default consulted only for a carrier
  dialect whose label names no codec. They are one rule with a fallback, not
  two competing semantic truths.

The region exactness rule is correct for fixed-size codecs — each stream
satisfies its declared minimum, all stream lengths total the declared
representation size, so no stream can contain unclaimed bytes — and an
instance-sized codec keeps taking its exact length from the bound container
record, never from a shape-derived declaration.

What it does not claim: a CPU shared-expert realization (the hard refusal is
preserved; the missing realizations are inventoried), a provider that brings
its own *kernel* (the kernel-plane blocker was measured at five closed enums
across 33 files and is measured debt, not the next task), or any
performance. The architecture phase has turned K3 failures from
format-specific surprises into named missing realizations with physical
costs; the next work binds a real K3 plan through this machinery.

## Rung 4 — the registry closed over its consumers: FP8, Q5_K/Q3_K, and the admission seam

Built after the v1 freeze and inside it: no invariant of the nine moved, and
the change is what the freeze was for — a consumer that had been routed
*around* the contract brought through it, with the evidence that nothing
else still is.

**Fine-grained FP8 was the conspicuous exception.** GLM-5.3-Flash's own
bytes — 95.8 % of a 306 GiB checkpoint — had a loader, a `WeightSlice`, a
resident format and a fused kernel (`FusedFp8Block`), and no codec. The
loader spelled the checkpoint's sibling name, a `companion` accessor existed
for that one format, the planner skipped the scale grid by its spelling, and
the kernel was reachable only by a caller that already knew the answer:
selection could not offer it, the reference oracle could not decode it, and
an external provider could not have done what it did.

It is now `F8_E4M3` (family `fp8-block`), the twelfth registered codec, and
the first PRODUCTION codec whose bytes mean nothing on their own:

| | |
|---|---|
| streams | one, the E4M3 codes |
| dependency | the f32 scale grid, named `scales`, **another represented object** addressed by the container's reference table and decoded through its own codec |
| tile | derived per tensor from the two shapes — never a config field, never a stream: a stream is bytes and the tile needs the grid's SHAPE, which is why the grid is a dependency and not a second stream (the VQ codebook's rule, met in production) |
| certificate | codes at 8 bits/weight; the grid is the grid's own footprint. No radius: a fitted codec, like every quantiser here |
| decode | bit-exact to `fp8_finegrained::dequantize_into`, the transcription of upstream's `Fp8Dequantize` |
| direct realization | `FusedFp8Block`, declared, with the grid **retained** — the first pin in the tree whose dependency lifetime is `Retained`, priced by the ledger that had been waiting for one |

Three things had to move for that to be one registration rather than a
special case:

- **The encoder declares the dependency.** The checkpoint's
  `*.weight_scale_inv` convention is consumed exactly once, at encode, where
  it becomes a row in the auxiliary reference table beside the segment it
  describes. The planner's name-based skip is gone: a grid is skipped
  because something *references* it, and an orphan is the unclassified
  operand it always was. A container encoded before the encoder declared
  dependencies migrates with `larql vindex3 references --declare`, which
  applies the encoder's rule late to the headers on disk and refuses to
  touch a table that already exists.
- **The dependency lifetime is the realization's.** Every pin used to be
  `PreparationOnly` because every realization decoded. A direct kernel over
  codes keeps its dependency for every token, so the lifetime is set from
  the pinned realization — and re-set by every re-pin, budget re-selection
  included, through one method.
- **The CPU policy ranks the stored FP8 bytes with the compiled packs.**
  Executed in place, never widened: on GLM the alternative is 612 GB of a
  306 GB checkpoint, which stays a candidate for the oracle.

**Q5_K and Q3_K** fill the ~5.5 and ~3.4 bits-per-weight points the
K-quant ladder skipped. This workspace decodes them (a GGUF import, a pack
another tool wrote) and does not compile them, so the K-quant vocabulary now
keeps two tables: `COMPILABLE`, what `vindex represent` can write, and
`DECODABLE`, what the registry ships. Neither has a kernel, and neither
declares one: the direct realization is declared only where the dispatch
answers, and a test holds the declaration and the dispatch together over
every member.

**The admission seam.** `OperandStore::open` admitted a pack's declared
identity against the built-in registry before `with_registry` could point
the store anywhere else — the F8 falsifier one seam further in. The
registry is now the constructor's parameter (`open_in`), re-pointing an
opened store re-runs the admission, and `ExpertEncoding` prices a bank
through the codec it *is* rather than a registry lookup. The witness is
external: a pack under the provider's identity is refused by the built-in
registry naming every family it does know, admitted by the provider's, and
executes bit-exact to the control.

**The closure.** Every direct CPU kernel over stored bytes is declared by a
registered codec, and the executor's own re-quantised forms by none —
checked over the plan enum, exhaustively, so a kernel added without a
declaration is a red test rather than a privileged path
(`codec/tests/closure.rs`).

Fourteen codecs: BF16, F16, F32, Q4_K, Q6_K, Q8_0, NVFP4, MXFP4,
BF16_ZLIB, F32_PLANES, VQ8_SHARED, F8_E4M3, Q5_K, Q3_K. Eight are the
production estate, three forced the contract, and the last three arrived
through it.

Not claimed: a device realization of FP8 (a device backend still narrows
only floats, and an FP8 operand under it is refused at load rather than at
selection — the next thing to make honest), a Q5_K or Q3_K kernel, or any
change to the lowering plane.
