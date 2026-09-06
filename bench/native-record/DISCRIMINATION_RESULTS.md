# Discrimination diagnostic: detector overlap and reader transfer failure

Completed 2026-09-05 on the same Gemma 3 4B IT bf16 checkpoint and loader as
address-v1. [Protocol](DISCRIMINATION_PROTOCOL.md) was frozen before capture.
The original editor was unchanged. Graph causality, speed and composition were
not tested in this diagnostic.

## Actual edited-slot activity

The instrument hooks the actual gate/up projections and the actual down_proj
input (post-GELU product). Contribution magnitude is the Euclidean norm of the
slot's down column times its absolute activation, **before** Gemma's post-FFN
normalization. It is not an additive decomposition after that normalization.

| Zelandia-capital slot queried with | Gate | Up | Activation | Contribution norm |
|---|---:|---:|---:|---:|
| The capital of Zelandia is | 362 | 12.5625 | 4544 | 116.612 |
| The capital of Qtaria is | 286 | 9.9375 | 2848 | 73.088 |
| The capital of Vornholt is | 203 | 7.0625 | 1432 | 36.749 |

These unwanted reads activate the edited slot substantially. Their downstream
counter-swaps from address-v1 therefore have a directly observed slot-level
mechanism; they are not merely generic output drift.

The seven intended prompts for each original slot (installation + four forms +
two aliases) overlap the same-relation controls in activation magnitude:

| Slot | Intended activation range | Unwanted range | AUROC | Unwanted n |
|---|---:|---:|---:|---:|
| Zelandia capital | 752–4544 | 0–4096 | 0.892 | 29 |
| Qtaria currency | 1352–3040 | 0.002–2640 | 0.931 | 25 |
| Vornholt language | 490–4320 | 1.578–3088 | 0.834 | 25 |

The AUROCs include the broader same-relation control bank and must not hide the
hard-case overlap. No single activation cutoff can perfectly separate intended
and unwanted reads in these observed ranges. This concerns the original
detector, not whether another direction could recover binding identity.

## Reader evaluation on fresh entities and structures

Development used four entities × three relations, with 24 training queries,
24 validation queries, 12 canonical enrollment addresses, and 12 near-name
unknowns. Settings were selected there only. Evaluation used four different
entities, their 12 canonical enrollment addresses, 48 queries in fresh sentence
structures, 24 explicit alias queries, and 12 fresh near-name unknowns.

Nearest-address retrieval uses cosine similarity. The ridge probe learns a
pairwise binding-match score from normalized query/address pairs, with balanced
positive/negative training weights. This lets it score newly enrolled binding
labels. It tests transfer of a learned decision rule; it is not an exhaustive
test of fixed-roster linear decodability. Enrollment contains no answer values.

| Layer | Nearest binding accuracy | Ridge binding accuracy | Nearest accepted + correct | Ridge accepted + correct |
|---|---:|---:|---:|---:|
| 22 | 30/72 | 25/72 | 2/72 | 0/72 |
| 24 | 38/72 | 35/72 | 0/72 | 0/72 |
| **26 (primary)** | **36/72** | **34/72** | **1/72** | **3/72** |
| 28 | 35/72 | 33/72 | 2/72 | 6/72 |
| 30 | 43/72 | 48/72 | 5/72 | 7/72 |

Both readers fail at layer 26, which triggers the frozen additional-layer
screen. None of the five layers passes the reader gate. The threshold chosen
on development data mostly rejects fresh valid queries: low false acceptance
alone must not be mistaken for successful selectivity. At layer 26 each reader
accepts 1/12 unknowns; at layer 30 each accepts 3/12. The full per-layer rates
are in the summary.

The selected ridge regularization is 1.0 at every layer. Its development
validation accuracy is 24/24 at L24/L26/L28/L30 (20/24 at L22). This falls to
34/72 at the primary fresh L26 evaluation. The held-out shift changes entities
and sentence structures together, so it does not isolate which shift causes
the failure. The additional-layer results are a bounded screen, not a newly
validated editing-layer selection.

At L26, nearest retrieval gets the **relation** right in 66/72 queries but the
**entity** right in 38/72; ridge gets 64/72 and 38/72 respectively. Nearest has
24/48 sentence and 12/24 alias binding hits; ridge has 21/48 and 13/24. There
is usable signal, but neither tested reader reliably identifies bindings across
the frozen transfer. These failures do not establish that entity information
is absent from the residual.

## Same-example original-bank comparison

The original 114 prompts are diagnostic data. For a nontrivial retrieval check,
each of the 12 original/near-name entities is enrolled under all three
relations using `The RELATION listed for ENTITY is`, which is distinct from
every queried prompt. There are 36 candidate bindings and 36 known-binding
queries (the 21 intended reads plus 15 cross-relation/near-name queries).

Nearest retrieves 4/36 correctly and the development-trained ridge retrieves
7/36. The remaining 78 prompts are unknown to that enrollment bank. These
numbers use more candidates and a different enrollment format than the fresh
12-candidate test; they are not directly comparable. They do not rescue the
original slot failures or license a claim that a particular probe family cannot
decode the original bindings after being trained on them.

## Verification and next experiment

All 342 clean/written/removed instrumented forwards reproduce address-v1's
persisted top-five tokens and log probabilities exactly. Removal reproduces
the new clean full logits exactly on all 114 prompts. Actual MLP input vectors
are saved in `captures.npz`; scalar slot measurements are saved per prompt.
The audit verifies hashes, row coverage, and every reader's decisions and
summary from its saved scores. No model checkpoint was mutated.

The next run is the already-frozen 4×3 factorial comparison. The original
factorial editor already includes canonical cross-record binding negatives;
the single changed method adds those binding negatives in the two development
sentence forms. It remains at L26 and uses one slot per record. No probe result
was used to tune its layer, scales, positive addresses, or negative templates.

Artifacts: [summary](results/discrimination-v1/summary.json),
[slot measurements](results/discrimination-v1/written.jsonl),
[reader predictions](results/discrimination-v1/reader_rows.json),
[fixture](results/discrimination-v1/fixture.json),
[metadata](results/discrimination-v1/metadata.json).
