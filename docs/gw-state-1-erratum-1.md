# GW-STATE-1 erratum 1 — broad-context fallback

**Recorded:** 23 September 2026, in a post-execution audit.

**Status:** Historical discrepancy record; no amendment to the sealed experiment.
**Observed verdict:** `carrier_plus_edge`.

GW-STATE-1 was sealed, executed, selected, replayed on held-out rows, and
adjudicated before this discrepancy was discovered. Filesystem timestamps put
the frozen protocol at 00:02:27 BST and the adjudication manifest at 02:14:54
BST on 23 September 2026; the other stage manifests fall between them.
The audit that found the discrepancy did not inspect outcomes. The separate
invariance check below was performed after the discrepancy was identified.
The earlier statement that there were zero STATE-1 outcomes is superseded.

## The two frozen rules

The [sealed prose](gw-state-1.md) says `broad_context` applies when a frozen
selected subset of at least four non-H1 heads clears, **or only the all-natural
sentinel clears**. The [sealed executable adjudicator](../scripts/gwstate1_analysis.py)
first checks the exact/all-natural control, `edge_only`, `carrier_plus_edge`, and
selected small circuits; it then returns `broad_context` unconditionally.
Consequently, when the all-natural sentinel and donor-carrier/all-seven-head
sentinel clear, no simpler class clears, and no selected four-plus-head context
clears, the executable returns `broad_context` although the prose condition
is false. A synthetic branch check reproduced that disagreement.

This is a rule inconsistency, not evidence that model execution, artifact
identity, selection, or replay was corrupted. Literal prose/executable
agreement must not be claimed for this verdict fallback.

## Historical outcome invariance

The [separate witness](../bench/gw-state-1/gemma3-4b-it-phase1/gwstate1-erratum1-invariance-witness.json),
reproducible with `python3 scripts/gwstate1_erratum1_witness.py`, verifies the
protocol and selection, checks the adjudication seal and replay hashes, and
recomputes both held-out analyses from the existing replay tensors. Its
recomputed results equal the historical adjudication, including all subject
lists and reported analysis values.

The train selections are `[none, 128]`. On both validation and test, the exact
control and context `128` pass; context `0` fails. Therefore the ordered
adjudicator reaches `carrier_plus_edge` before the disputed `broad_context`
fallback. The historical `carrier_plus_edge` verdict is invariant to this
specific discrepancy. This witness does not make any broader claim about
unexamined branches or future experiments.

Historical identities: protocol `sha256:b50965f4eec68a25b91d6707758d4c834468299a31e827dc4c8d7ed025c0ff6e`;
adjudication `sha256:ce0498a74200bb43c317f6df65d78bf9afbfaa5916ef1b943842db142080a2a5`.
The witness records file and tensor hashes as well.

The sealed protocol, prose, adjudicator, selection, replay tensors, and
adjudication artifact retain their original bytes. Any future use of the
STATE-1 verdict scheme should choose and preregister one `broad_context`
definition before new outcomes are available. That prospective choice does
not change this historical adjudication.
