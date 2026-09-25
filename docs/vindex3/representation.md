# Represent a model

**Class: CURRENT.** [Status](status.md).

REPRESENT explores physical realizations under declared behavior and cost
constraints. A logical object can have canonical bytes and alternate encoded
representations. A profile selects among representations; encoding a smaller
pack does not by itself establish that its behavior is acceptable.

The standalone tool supports reference compilation and inspection:

```bash
vindex represent model.vindex3 model-nvfp4.vindex3 --encoding NVFP4
vindex representations model-nvfp4.vindex3
vindex precision model-nvfp4.vindex3 --matrix
```

Supported encodings, geometry and export targets are checked by the compiler.
Use command help and an admitted artifact; the existence of a codec is not
a promise that every backend supports its execution.

## Compile a representation

`larql vindex3 represent` is the same compiler with the precision policy and
plugins exposed:

```bash
# An archival container: canonical bytes kept, the pack added beside them.
larql vindex3 represent model.vindex3 --output model-nvfp4.vindex3 --encoding NVFP4

# A deployment image: the pack replaces the source bytes it was compiled from.
larql vindex3 represent model.vindex3 --output deploy-nvfp4.vindex3 \
  --encoding NVFP4 --deployment

# Hold some projections at source precision.
larql vindex3 represent model.vindex3 --output model-r1.vindex3 --encoding Q4_K \
  --protect v_proj --protect down_proj@30-39 --protect-layers 0-1
```

- **Encodings.** `NVFP4`, the K-quants the compiler names in its refusal, and
  any encoding a `--plugin`'s encoder registers ([Plugins](plugins.md)).
- **Policy.** The default compiles the parameter mass: the decoder's bulk
  projections, a recurrence's projections and routed experts. It preserves
  the embedding, the head and small vectors. `--include-role` opts a
  preserved role in. `--protect` and `--protect-layers` hold named
  projections or depths at source precision. The container records the
  program as its precision map (`r0-uniform`, or `r1-protect-…`).
- **What is written.** One pack per compiled object, as
  `segments/<object>@<ENCODING>.bin`, marked `approximate` in the system
  graph. Every other segment is hard-linked. An archival container keeps the
  canonical representation; a `--deployment` image drops the source bytes of
  every compiled object, is marked `Derived`, and names the digests it came
  from. The command prints each object's source and compiled bytes and what
  the policy preserved.
- **Input-feature weights.** `--moments <capture>` encodes under per-input-feature
  weights `E[x²]` from `larql vindex3 sensitivity --calibration … --moments`, captured
  from the same container on a calibration set disjoint from any bank you will
  measure on. The encoder then minimises `Σ E[x_i²]·(w_i − ŵ_i)²`, spending its
  error where the activation is small. q/k/v and gate/up use the captured
  attention and FFN input sites; `down_proj` uses a reconstruction of
  `act(gate(x))·up(x)` that is checked against the executor before use;
  `o_proj` has no captured site and is encoded unweighted. The command prints
  how many tensors each source covered. Only an encoder that implements
  weighting takes them (a plugin encoder may; NVFP4 and the K-quants refuse),
  and the pack's recipe records the capture's digest.
- **Provenance.** Each pack records its codec identity and encoder recipe.
  A pack LARQL compiled can be re-derived byte for byte by the same recipe; a
  plugin encoder's pack cannot, and says so.

Execute a pack with the backend that asks for it (`production-nvfp4` for
`NVFP4`, `production-q4k` for `Q4_K`, …), or name it with `--representation`
([Execution](execution.md)).

## Measure a representation

A smaller pack is not evidence of acceptable behavior. `larql vindex3
measure` teacher-forces a candidate realization against a reference over a
sealed token bank ([MEASURE-PLAN-1](../measure-plan-1.md)).

**1. Build a bank.** A bank is token ids sealed against one tokenizer:

```bash
# From prompts (text), tokenised by the container's own tokenizer:
larql vindex3 token-bank export model.vindex3 \
  --prompts bench/prompts/quality-bank-1/prompts.json --output bank/

# From ids another harness already tokenised (fixed evaluation windows):
larql vindex3 token-bank import model.vindex3 --ids windows.json --output bank/

larql vindex3 token-bank check bank/ --container deploy-nvfp4.vindex3
```

`export` truncates each prompt to `--max-tokens` (128 by default). `import`
takes `{"bank": NAME, "samples": [{"id", "category", "ids": [...]}]}` and
seals the ids verbatim: no special tokens, no cap. It refuses an id outside
the tokenizer's vocabulary. A bank belongs to every container with the same
tokenizer.

**2. Measure.**

```bash
larql vindex3 measure \
  --reference model.vindex3 --reference-backend production \
  --candidate deploy-nvfp4.vindex3 --candidate-backend production-nvfp4 \
  --bank bank/ --sequences 69 --label nvfp4 --output out/nvfp4
```

It prints KL, top-1 agreement, and their spread by prompt category and by the
reference's top-1 margin. `out/` holds `report.json`, `positions.jsonl` (per
position: KL, top-1, top-5 overlap, ΔNLL of the true next token, reference
margin and entropy) and `receipt.json`.

**3. Read the receipt first.** Before any number counts, the procedure proves
the null arm (the reference run twice, bit for bit), the changed variable,
each arm's physical attribution, byte identity of what both arms share, every
seal, and that the bank belongs to both models. `Inadmissible` means numbers
exist but are not evidence; `ExecutionFailure` means nothing was measured.
Either exits non-zero and still writes a receipt. The procedure characterises;
it applies no gate. Pre-register the decision a measurement serves separately.

Each arm is any `exec` backend, including lowered Metal arms. A plugin's
codec, encoder and lowering provider take part through `--plugin` and the
per-arm `--*-lowering` / `--*-representation` overrides ([Plugins](plugins.md)).

## Search and evidence contracts

The research machinery extends beyond compilation. Candidate authority binds
actual payloads to a representation state. Deterministic evidence ingestion
feeds the search state; measurement keys bind candidate, protocol and
instrument identity. Actuation reconstructs and validates the selected
measurement before preparation and execution. These boundaries make a search
loop auditable; none independently grants a candidate behavioral approval.

The implementation lives in
[`represent/`](../../crates/larql-vindex/src/format/vindex3/represent/), including
`candidate_authority`, `ingest`, `state`, `search_evidence` and `actuate`.
The [REPRESENT contract index](../represent-v1-contract-index.md) and
[optimizer contract index](../optimizer-contract-index.md) name the tests that
protect their respective frozen contracts. They are stronger authorities than
a blanket claim that the optimizer is finished.

The [codec contract](../represent-codec-contract.md) separates representation
decoding, compilation and execution support. The
[optimizer MCP design](../represent-optimizer-mcp.md) is a design document,
not an inventory of shipped commands.

[REPRESENT-CAL-1](../represent-cal-1.md) is the calibrated-recipe implementation
contract. CAL-1.1 supplies content-bound calibration artifacts and exact dense
projection input capture through the library API; GPTQ dispatch and its admission
run remain later stages.
