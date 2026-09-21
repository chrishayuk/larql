# Observation and intervention

**Class: CURRENT.** [Capability boundaries](status.md).

A model execution can leave an evidence record. Observation subscribes to the
canonical decode traversal: it records actual execution sites and carrier
writes, rather than reconstructing a second forward pass for display.

```bash
larql vindex3 observe model.vindex3 --backend production \
  --prompt "The capital of France is" --record run.jsonl
```

The recorder carries run identity, provenance, events and a receipt. Standard
statistics include carrier/write norms and fixed projections. A supplied basis
must retain its identity. A true logit lens is separately armed with
`--lens-tokens` and uses the execution image's normalization and vocabulary
head; a raw selected-token probe is not a probability or rank.

Open the JSONL in the [Observatory](../../observatory/README.md). The bridge
validates record structure, identities and receipt bindings and replays the
recorded values. Importing a record does not run a model or prove its output
agrees with an independent implementation.

| Evidence | What it can establish |
|---|---|
| Carrier observation | What was written at the declared execution boundary |
| Lens or projection | What a declared reader sees in that state or write |
| Per-head/source decomposition | Descriptive support under the capture and normalization contract |
| Intervention plus controlled comparison | A scoped counterfactual effect under the declared manipulation |

Head capture has a narrower backend/operator scope than carrier observation.
`observe --heads` records per-query-head contributions, top source positions,
sink mass and head-sum residuals, with coverage on the receipt. The
[per-head observation contract](../v3-head-obs-1-per-head-observation.md) defines
what the softmax tap observes and the reconstruction/parity laws.
[ATTR-1D](../v3-attr-1d-descriptive-support.md) normalizes descriptive support;
attention weights and additive contributions alone do not establish causality.
The earlier [CPU head-capture record](../v3-observatory-head-capture.md) remains
a historical account of the separate Granite capture tooling.

`observe --intervene declarations.json` accepts carrier and head interventions.
Carrier declarations name a layer, attention/FFN site and positions, with
`zero`, `add` or `replace`. Head declarations name a layer, query head and
positions, with `zero`, `scale` or `replace`. The declaration and vector
provenance join the record identity; receipts account for firings. See
[carrier intervention](../v3-intervene-1-carrier-intervention.md) and
[head intervention](../v3-intervene-2-head-intervention.md) for the contracts.

A true head counterfactual changes the mixed head value inside attention,
before the gate, output projection and post-attention normalization, then
resumes downstream execution. Subtracting an observed head contribution is a
different experiment: changing one head can change the normalization applied
to all heads. The implemented head-intervention scope is CPU softmax decode;
MLA, conv-QKV, batch prefill and Metal head intervention remain outside it.
There is no separate `intervene` verb: the interface is part of `observe`.

Use `--capture` with `--capture-out` for carrier donors and `--capture-heads`
with `--capture-heads-out` for head donors. Reproduction needs the declared
addresses, captured vectors, model/representation identity and comparison
protocol. A successful intervention is not by itself a causal attribution
result; the [instrument calibration](../instrument-1-calibration.md) and
[HEAD-OBS-1 evidence programme](../head-obs-1-per-head-observation.md) explain
why reader calibration and a discriminating decision matter.

The [carrier observation record](../v3-obs-1-carrier-observation.md),
[logit-lens record](../v3-lens-1-logit-lens.md) and
[observation contract draft](../vindex3-observation-contract.md) retain their
own status. Research results belong in records; the CURRENT overview links
them without rewriting their experimental claims.
