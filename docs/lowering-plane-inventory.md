# The lowering plane — inventory before it is opened

**Status: measured, not built.** Read against main `a03efc29` (PR #460)
with PR #461 (the codec plane closed over its consumers) beside it. This
document exists so that "open the lowering plane" starts from what the tree
actually contains rather than from the diagram, the way rung 3's forecast
started from a measured baseline. Nothing here is frozen; the forecast that
freezes the transition is the next document, not this one.

The question is the codec plane's, asked of the other half:

> Representation plugins answer "what is stored, and what guarantees does
> it carry?" Lowering plugins answer "how is a planned graph realized for
> CPU, Metal, MLX, GGUF export, or a remote FFN service?"

REPRESENT can now say "here is a representation you have never heard of"
through registration alone. The realization side cannot say "here is a
lowering provider you have never heard of": the seam exists, and the
identity, the registry and the closure do not.

## What is already right

- **The seam is a trait, and the interpreter drives it.** `PlanBackend`
  (`…/opplan/exec/backend.rs`) owns arithmetic only; `exec/mod.rs` owns
  meaning. One interpreter drives every backend, `execute_plan` takes
  `&dyn PlanBackend`, and `Vindex3Runtime<B: PlanBackend>` is generic —
  a second implementation cannot become a second reading of the model.
- **Selection is the provider's, from declarations.** `PlanBackend::select`
  is asked per operand at preparation, before any byte, with the
  `RepresentationFacts` the registry declares, and answers with one pinned
  `RealizationId` or a refusal naming every candidate. The CPU policy
  derives its candidates from the codec's accelerations; nothing there
  reads a label.
- **The pin carries the codec provider and the ledger prices the pin.**
  `RealizationRecord.provider` is the codec identity the label resolved to,
  `ensure_providers_in` invalidates an image whose codec provider is lost or
  substituted, and the census is reconciled against the declaration.

Three in-tree providers implement the trait: `ReferenceBackend`,
`ProductionBackend`, `DevicePlanBackend<M: MatMul>`.

## What is closed, measured

Non-test files under `opplan/exec/` and `represent/` naming a variant of
each closed enum (2026-09-08, same method as rung 3's 2026-09-05 count):

| enum | files | what it closes |
|---|---|---|
| `PhysicalProjectionPlan` | 12 | the CPU kernel vocabulary; a codec's `Acceleration.plan` is one of these |
| `WeightFormat` | 10 | the resident forms a backend may ask the loader for |
| `WeightSlice` | 7 | the borrowed operand a backend receives |
| `WeightRows` | 14 | the row view a kernel consumes |
| `LoadedWeight` | 5 | the owned operand the loader produces |
| union of the five | **33** | unchanged since 2026-09-05 — the kernel plane has not widened or narrowed |
| `RealizationForm` | 4 | the seven ways an operand is executed |
| `RealizationBackend` | 2 | `Cpu \| Device` — WHERE a realization runs, as a closed pair |
| `AccelerationBackend` | 2 | `Cpu` — the only backend a CODEC may declare a kernel for |

Five more non-test files outside `larql-vindex` (`larql-cli`, `larql-inference`,
`larql-server`) name a kernel-plane variant.

Two of these matter more than the count. `RealizationBackend { Cpu, Device }`
is the identity a pin records for its lowering provider, and it is a
closed pair with no revision: a prepared image knows which CODEC produced
its bytes by family and revision, and knows which LOWERING qualified its
pin only as "device". `AccelerationBackend { Cpu }` means a codec can
declare a direct kernel only for the CPU executor; the device backend's
kernels are "declared by the peer crate that owns them" — and there is no
seam through which that declaration reaches selection. `DevicePlanBackend::select`
consults its own per-class format table (`SelectionReason::DeviceClassTable`),
asks the facts only whether the label is registered, and offers exactly one
candidate. That is a backend choosing, not a backend deriving.

## What goes around the seam

| path | size | what it is |
|---|---|---|
| `larql-cli/…/vindex3_cmd/lowered/` | 2,890 lines | `--backend metal-lowered*` (six of `ExecBackend`'s eighteen arms): a second driver that loads the plan, encodes whole command buffers per position and steps them itself. No `PlanBackend::select`, no pinned `RealizationId`, no `PreparedOperands`, no ledger, no provider invalidation. The perf instrument of record runs outside the accounting the freeze built. |
| `…/opplan/exec/{stack_metal,kda_metal,kimi_source}.rs` | 1,929 lines | Metal-specific lowerings of the Kimi ladder INSIDE the device-agnostic crate, gated by `cfg(feature = "gpu")` in `exec/mod.rs`. The crate's own contract ("this crate never links a GPU API; the caller injects the device") holds for `device.rs` and not for these. |
| `ExecBackend` (CLI) | 18 arms | The only registry of lowering providers is a `clap::ValueEnum`; each arm hand-constructs a concrete backend (13 construction sites outside `larql-vindex`, non-test). A provider that is not an arm cannot be asked for. |
| `PlanBackend::name()` | — | "for diagnostics and parity reports, not dispatched on": the trait has a name and no identity. |

The first row is the lowering plane's fine-grained FP8: a real, load-bearing
realization that the seam cannot see. The second is the analogue of the
`companion` accessor — an exception that lives where the rule does.

## The programme this measures for: LOWERING-PLUGIN-1

Authority movement first, kernels never. The six steps, mapped to the tree:

1. **A lowering identity.** `LoweringIdentity { family, revision }` beside
   `CodecIdentity`, required of every `PlanBackend` (replacing `name()` as
   the thing a record carries; a name stays for reports). CPU, reference,
   device-over-`MatMul`, and later Metal-lowered, MLX, GGUF export and a
   remote FFN become identities, not arms.
2. **A registry in front of candidate derivation.** `LoweringRegistry::register(Box<dyn PlanBackend>)`,
   built and consulted the way `CodecRegistry` is: a value the store or the
   session carries, never a hidden default. `ExecBackend` becomes a lookup
   into it, and the 13 construction sites become one.
3. **CPU and device behind the registry, behaviour unchanged.** The first
   transition changes no selection, no number and no ledger line — the
   witness is byte-identical realization records on the existing fixtures.
4. **Provider identity in the pin.** `RealizationRecord` gains the lowering
   identity beside the codec `provider`; `PreparedOperands::ensure_providers_in`
   gains a sibling for lowerings, so an image prepared by a provider that is
   gone, or at another revision, is invalidated by name — the symmetry the
   codec plane already proved (v1 contract §8).
5. **One hostile external lowering provider.** An integration test, outside
   core, that registers a provider the build does not ship and proves
   `register → candidates → selection → accounting → binding → execute`,
   with the genericity scan (`external_attested_provider/genericity.rs`) run
   over its identity strings. The `lowered/` driver is the candidate to
   move behind it, because it is the one that will fail first.
6. **Realization-error certificates.** Named unpaid by the v1 freeze:
   representation fidelity ends at decoded values; accelerated-kernel error
   belongs to this plane. Not before 1–5.

## Falsifiers worth preregistering

- **F1 — the seam's granularity.** `PlanBackend` is per-operation
  (`project`, `attention`, `ffn`, …). The Metal-lowered driver encodes a
  whole position — stack and head — into one command buffer, and its prize
  is exactly that it does not return to the host between operations. If it
  cannot be moved behind `PlanBackend` without a whole-step method, the
  trait needs one (`step`, with the per-op methods as its default
  decomposition), and that is a contract change to record, not a workaround.
- **F2 — declarations for devices.** `AccelerationBackend` must open before
  a device provider can derive candidates from a codec's declarations rather
  than a class table. Whether it opens to an identity (the provider names
  itself in the declaration) or to a capability (the codec declares what a
  kernel would need and the provider matches) decides whether a codec ever
  has to know a provider exists. The codec plane's answer to the same
  question — a codec never names a backend — suggests the second.
- **F3 — the five-enum kernel plane.** A provider that brings its own kernel
  identity meets 33 files. Steps 1–5 do not open it (the CPU and device
  providers keep their forms); a provider with a new resident form is the
  transition after this one, and its cost is this number.
- **F4 — the hidden defaults.** The codec plane found three hidden built-in
  registries at execution (rung 3, F8) and one more at open (PR #461).
  Expect the same class here: every `ProductionBackend::new()` in a
  non-test path outside the CLI is a candidate.

## Not claimed

No provider is moved, no enum is opened, no forecast is frozen. This is the
baseline the forecast will be measured against, and the counts above are
what a wave must move.
