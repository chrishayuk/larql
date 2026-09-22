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
The [CPU softmax head-capture record](../v3-observatory-head-capture.md)
describes the separate local capture tooling and measured Granite evidence.
It explicitly distinguishes that tooling from the merged recording CLI.
[ATTR-1D](../v3-attr-1d-descriptive-support.md) normalizes descriptive support;
attention weights and additive contributions alone do not establish causality.

Carrier intervention changes a declared write or state. A true head
counterfactual changes attention computation and resumes downstream execution;
subtracting an observed head contribution is a different experiment. Head
replay, carrier intervention and graph-walk work must retain their own scope,
controls and provenance. This checkout contains local intervention work;
that is not a release claim or a portable `intervene` command. Verify the
specific revision and interface before reproducing a protocol.

The [carrier observation record](../v3-obs-1-carrier-observation.md),
[logit-lens record](../v3-lens-1-logit-lens.md) and
[observation contract draft](../vindex3-observation-contract.md) retain their
own status. Research results belong in records; the CURRENT overview links
them without rewriting their experimental claims.
