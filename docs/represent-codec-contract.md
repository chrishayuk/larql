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
