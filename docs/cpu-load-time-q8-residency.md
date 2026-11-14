# The production CPU loader's Q8 residency

**Class: CURRENT behaviour, written down for the first time; plus a PROPOSED verification contract.** Found by
CONSUMPTION-BF16-1 (`docs/consumption-bf16-1.md`).

A VINDEX3 container that stores BF16 weights is not necessarily executed from them. On the production CPU backend
the loader chooses, per matrix, what to hold in memory, and when it is allowed to (§2a) its choice for a large
matrix is a lossy Q8 derivation it manufactures at load. That derivation is in no container, has no digest, and
until now had no written rule.

**Correction (2026-11-14, same day).** The first version of this document called the manufacture "the default
policy". That was wrong as a general statement. PREC-DIAG-1 had already found this behaviour (2026-11-11) and PR
#778 made `vindex3 exec --backend production` and the server's CPU path default to a policy that forbids it. It is
the default of the library entry point the CONSUMPTION-BF16-1 harness used, and, as read from the code, of LQL's
`INFER`. §2a says which entry point runs which.

## 1. Three representations, not two

| Representation | Where it lives | Rule | Verifier |
|---|---|---|---|
| Canonical BF16 | the container | the edit ledger, for an edited container | `WEIGHT_PATCHES_V4` |
| Stored Q8_0 pack | the container, when one was compiled | `kquant-larql-native` rev 2 (32-element blocks, f16 scale) | `Q8_0_PACK_DERIVATION` |
| **Load-time Q8 residency** | process memory only | §3 (64-element blocks, f32 scale) | none |

The stored pack and the load-time residency are different quantisations of the same weights. Verifying one says
nothing about the other.

## 2. When the loader manufactures it (`opplan/exec/cpu/physical/projection.rs`, `choose_for_l2`)

For a matrix of `elements = rows × cols` stored as BF16, against the host's performance-core L2 size:

| Condition | Resident as | Kernel (default arm) |
|---|---|---|
| `elements × 4 < L2` | f32, widened exactly | BLAS f32 |
| else `elements × 2 < L2` | BF16, the stored bytes | fused BF16 × f32 |
| else, and Q8 is permitted | **Q8, manufactured** | fused Q8 × f32 activation |

- **Q8 is permitted unless `LARQL_CPU_MAX_FORMAT=bf16`.** Only that exact value forbids it; the cap does not affect
  a stored pack.
- **The boundary is read from the hardware** (`hw.perflevel0.l2cachesize` on macOS, a default elsewhere). Which
  matrices are quantised is therefore a property of the host, not of the container. On this machine (16 MiB L2)
  a Gemma 3 4B `down_proj` (`[2560, 10240]`, 52 MB as BF16) is quantised; a `[1024, 2560]` `k_proj` is not.
- **`LARQL_CPU_ARITHMETIC`** selects other arms for the same population (`bf16xq8`, `q8xq8`, `q4xq8`), which also
  quantise the activation. Unset, the activation stays f32.

## 2a. Which entry point is allowed to manufacture it

Whether the loader may manufacture anything is `RepresentationSource` (`opplan/exec/operands.rs`): `Stored` forbids
it (a matrix that would be quantised keeps its stored BF16 bytes), `Auto` permits it, `Transient` requires it.
`LARQL_CPU_MAX_FORMAT=bf16` is a second, independent cap that forbids Q8 under any source.

| Entry point | Source | Large BF16 matrices execute as | How established |
|---|---|---|---|
| `larql vindex3 exec --backend production` / `reference` | `stored` by default since #778 | stored BF16 | read from `vindex3_cmd/prepare.rs` |
| `larql-server`, CPU | `Stored` | stored BF16 | read from `larql-server/src/vindex3.rs` |
| `larql vindex3 exec --representation-source auto` | `Auto` | manufactured Q8 | read from the same |
| **LQL `USE` + `INFER`** | `OpenPolicy::default()` = `Auto` | **manufactured Q8** | read from `larql-lql/src/executor/vindex3/session.rs` and `larql-inference/src/vindex3/runtime.rs`; **not confirmed by a run** |
| `OperandStore::open` (library; the I5 and CONSUMPTION-BF16-1 harnesses) | `Auto` | manufactured Q8 | measured (CONSUMPTION-BF16-1 exploratory run) |
| `larql run`, Metal backends | not traced | — | — |

So the exact path already exists and is the CLI's and the server's default. What is new here is the written rule
(§3), the measurement that an `Auto` run computes with it (§4), and that LQL's `INFER` appears to be an `Auto`
caller: the statement a user edits and verifies through would execute the requantised arithmetic.

## 3. The rule (`opplan/exec/quantise.rs`, `quantise_q8`)

Over the matrix widened to f32, row by row, in blocks of 64 consecutive elements (a row's last block is short;
no block spans two rows):

1. `peak = max |w|` over the block.
2. `scale = peak / 127` in f32, or `1` if `peak = 0`. The scale is kept as an **f32**.
3. `code = round_half_away(w / scale)`, one f32 division per element, clamped to `[−127, 127]`.

A weight is read back as `code × scale`. It differs from the stored Q8_0 pack in block length (64, not 32), scale
width (f32, not f16), the division (by the scale, not a multiplication by its reciprocal) and the clamp (±127).

## 4. Evidence that the default run computes with it

CONSUMPTION-BF16-1 §11, exploratory: an independent re-implementation of §3 reproduces a default-policy run's
layer-12 FFN write to 2.9e-8 .. 6.7e-7 at 40 run-positions, where the stored BF16 weights reproduce it only to
2.3e-4 .. 1.0e-2. One tensor, one layer, one machine, on a record examined before the rule was read.

## 5. What this means for claims

- "The container's canonical weights are verified" gives "the run computed with the verified weights" only for a
  run whose representation source is `stored` (or that sets `LARQL_CPU_MAX_FORMAT=bf16`), and CONSUMPTION-BF16-1
  qualified that for one tensor under the second setting. Which source a run used has to be recorded, because it
  differs by entry point (§2a).
- An `Auto` run of an edited container computes with a quantisation of the edit. A difference in its
  output is partly the edit and partly that quantisation, and the two are not separated by anything recorded.
- A run's manifest must record the policy variables and the host's L2 size, or it does not say which model ran.

## 6. Proposed: an execution-representation contract

Not built. A design for review.

**The engine reports** (a witness, like `larql-gemv-witness/1`), per operand it loaded:

- object, tensor name, shape, and the SHA-256 of the canonical tensor bytes it read;
- the resident representation chosen (`f32`, `bf16`, `q8-load/1`, a stored pack's codec) and whether it was
  manufactured at load;
- the derivation rule's name and revision (`q8-load/1` = §3), and the inputs to the choice: the L2 size used, the
  policy variables;
- the SHA-256 of the resident bytes (codes and scales, in a declared layout);
- the kernel that will consume it and the arithmetic arm.

**The verifier checks, independently**, for each operand:

1. the canonical digest is the container's tensor (and the container passes its own contracts);
2. the choice follows from §2 given the reported L2 size and policy — so the selection is judged, which the stored
   pack contract cannot do;
3. the resident bytes are §3 applied to the canonical bytes, every block: recomputed and compared by digest and,
   on mismatch, by block;
4. no operand of the plan is missing from the witness.

**A consumption record then binds a kernel call to a resident digest**, so the chain is: declared edit → canonical
bytes (`WEIGHT_PATCHES_V4`) → resident bytes (this contract) → the call that consumed them (a witness of the kind
CONSUMPTION-BF16-1 and the Q8_0 GEMV contracts judge).

What it would still not establish: that the engine's report is honest. A witness is the engine's own account, as
the GEMV witness document already says.

## 7. Open decisions

1. Should LQL's `INFER` and the library default open under `Stored`, as the CLI and server already do, so that
   `Auto` is always an explicit choice?
2. Should the selection stop depending on the host's cache size, or be recorded in every run's identity?
3. Does the load-time rule get frozen and versioned (`q8-load/1`) with conformance vectors, like the VLP numerics?
