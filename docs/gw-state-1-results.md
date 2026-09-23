# GW-STATE-1 — closed result

**Status:** CLOSED — `carrier_plus_edge`, with [Erratum 1](gw-state-1-erratum-1.md) attached.

**Population:** The frozen 666-execution GW-V2 factual cohort.

**Historical protocol:** `sha256:b50965f4eec68a25b91d6707758d4c834468299a31e827dc4c8d7ed025c0ff6e`.
**Historical adjudication:** `sha256:ce0498a74200bb43c317f6df65d78bf9afbfaa5916ef1b943842db142080a2a5`.

Train selected no donor-carrier context and context `128` for the natural-carrier
branch. Context `128` retains the target-natural entering carrier and exact
target-natural H1 treatment while supplying all seven non-H1 heads from their
matched donors. Held out replay evaluated the frozen contexts `0, 127, 128,
255` on validation and test separately.

| Context | Validation | Test | Registered interpretation |
|---|---|---|---|
| `0`: donor carrier, donor non-H1 heads | fail | fail | `edge_only` does not clear |
| `127`: donor carrier, natural non-H1 heads | fail | fail | fixed donor-carrier sentinel |
| `128`: natural carrier, donor non-H1 heads | pass | pass | `carrier_plus_edge` clears |
| `255`: all-natural context | pass | pass | exact control clears |

Raw / independently z-scored proximal retention for context `128` was
`0.96567 / 1.02859` on validation and `1.02869 / 1.04391` on test. Its
simultaneous one-sided lower bounds were `0.87689 / 0.97909` and
`0.91899 / 0.98026`, respectively. Its terminal-effect simultaneous lower
bounds were positive on both metrics in both splits. Both tested donor-carrier
sentinels failed proximal point retention. The registered ordered adjudicator
therefore returned `carrier_plus_edge`.

Under the registered gate, the natural carrier plus H1 sufficed despite
donor-supplied values for all seven neighbouring heads. The failed
donor-carrier contexts do not by themselves prove a necessary-component
contrast. The result does not claim cheap construction or execution of the
natural carrier, task correctness, or efficient WALK.

The sealed [protocol](../bench/gw-state-1/gemma3-4b-it-phase1/gwstate1-protocol.json),
[specification](gw-state-1.md), implementation, selection, replay, and
adjudication retain their historical bytes. The selection and adjudication
manifests and replay tensors reside in the local ignored `output/` tree; the
[invariance witness](../bench/gw-state-1/gemma3-4b-it-phase1/gwstate1-erratum1-invariance-witness.json)
records their hashes and recomputes both held-out analyses. Erratum 1 records a
post-execution prose/code disagreement in a later `broad_context` fallback.
The witness shows that this disagreement cannot change this historical verdict:
the `carrier_plus_edge` branch resolves first.
