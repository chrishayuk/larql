# Model-as-Database V3: residual keys and future addressability

**Status:** P2 open. FAST-TAP is parity-gated. The pilot's perfect late-layer
operation purity was confounded, while the fixed adversarial replication finds
a gradual, moderate semantic signal peaking at 0.7125 rather than 1.0. KNN-3
remains unbanked; coarse KNN-1 remains small in absolute terms.

The first question is deliberately narrower than routing:

> Is the residual stream an index into the model database?

MAD-V3-KNN-1 measures whether neighbours of a residual predict which model
objects carry future FFN contribution. MAD-V3-KNN-3 measures whether lexical
variants of one graph query converge to a canonical residual key. kNN remains
offline in both experiments. The model always executes its complete plan.

## Pre-registered first run

- Model: the VINDEX3 Muse-Glimmer container used by the G6 lowering work.
- Corpus: 5,600 examples generated from 300 history graphs and 100 held-out
  graphs. The first feasibility pilot used one graph in each partition.
- Residual keys: final prompt-token residuals entering layers
  `1,8,16,24,32,40,48,51`.
- Object address: one layer's gated-FFN channel block, initially 128
  intermediate channels. Its byte cost is the canonical stored bytes for the
  corresponding gate rows, up rows and down columns. These are logical byte
  slices, not yet a claim that the container stores each slice contiguously.
- Contribution mass: squared L2 norm of the block's raw down-projection vector,
  before Glimmer's post-FFN norm. This is an exact, non-interventional boundary;
  attributing through RMS normalisation would require a separate nonlinear
  causal definition.
- Search: cosine. Exact brute force is the scientific default and headline.
  The scalable arm is projected HNSW plus exact full-dimensional reranking, always named
  approximate and accompanied by sampled Recall@K against brute force.
- History/query partitions are disjoint. `group_id` enables a second,
  held-out-graph arm in which no historical residual from the query's source
  graph is eligible.

Primary outputs:

1. semantic neighbour purity and wording neighbour purity by layer;
2. candidate object bytes versus future contribution mass covered, by source
   layer and horizon;
3. the same coverage under the layer-popularity control and the per-query
   oracle ceiling, plus a deterministic within-layer signature-shuffle null;
4. HNSW mean and minimum Recall@K against exact cosine on an audited query
   sample when the approximate arm is run;
5. contribution entropy and `exp(H)`, the effective number of active blocks,
   so a failed predictor can be distinguished from an intrinsically dense
   address distribution.

For KNN-1, each object's predicted score is the unweighted mean contribution
mass across the K neighbours. Objects are ranked by predicted mass per stored
byte; the report records the actual admitted byte fraction as well as the
requested budget, so unequal tail blocks cannot make the x-axis lie.

The first interesting result is a layer where semantic purity rises above
wording purity **and** kNN future coverage separates from layer popularity. A
semantic-purity result alone is KNN-3, not an address-generation result. A
future-coverage result without a popularity separation is only a globally hot
object table.

Kill or downgrade conditions:

- HNSW audit is poor: no model claim is evaluated; raise `ef_search` or run the
  exact control first.
- kNN does not beat popularity at matched actual bytes: future addressability
  is not supported at that layer/horizon.
- the effect disappears under `--exclude-same-group`: report graph-instance
  memorisation, not a transferable query key.
- semantic and wording purity rise together: the key is lexical, not
  canonical.
- only the oracle ceiling is high: the address space is useful but the
  residual-neighbour predictor is not.

## Fixture

Capture input is JSONL, one independent prompt per row. Token IDs are explicit
because the VINDEX3 execution fixture must not silently choose a tokenizer or
chat template:

```json
{"id":"g17-q4-w1","semantic_id":"g17:successor:northbridge","wording_id":"successor-template-1","split":"history","group_id":"graph-17","token_ids":[1,42,73]}
{"id":"g17-q4-w3","semantic_id":"g17:successor:northbridge","wording_id":"successor-template-3","split":"query","group_id":"graph-17","token_ids":[1,91,28]}
```

`semantic_id` names the underlying database query, while `wording_id` names the
lexical template shared across many semantic queries. The partition must put
different wordings of a semantic query on both sides for the canonical-key arm.
`operation_id` names the graph-independent operation/relation class, and
`regime` selects one of the three query arms without recapturing weights:

- `a_same_graph_unseen_wording` — exact semantic query appears in history,
  phrasing does not;
- `b_unseen_graph_seen_wording` — graph/entity is new, operation and wording
  family appeared in history;
- `c_unseen_graph_unseen_wording` — both graph/entity and wording family are
  held out; operation purity is the relevant label.

## Run

```bash
cargo run --release -p larql-cli --features gpu -- \
  dev mad-knn capture /path/to/glimmer.vindex3 \
  --queries bench/mad-v3-knn/queries.jsonl \
  --output bench/mad-v3-knn/capture-metal-f16 \
  --backend metal \
  --contribution-engine fast \
  --residual-layers 0-51 \
  --block-channels 128

cargo run --release -p larql-cli -- \
  dev mad-knn census bench/mad-v3-knn/capture-metal-f16 \
  --byte-budgets 0.05,0.10,0.20 \
  --output bench/mad-v3-knn/census.json

cargo run --release -p larql-cli -- \
  dev mad-knn evaluate bench/mad-v3-knn/capture-metal-f16 \
  --search exact \
  --neighbors 1,4,8,16 \
  --horizons 1,4,8 \
  --byte-budgets 0.01,0.05,0.10,0.20 \
  --audit-queries 64 \
  --query-regime c_unseen_graph_unseen_wording \
  --exclude-same-group \
  --output bench/mad-v3-knn/report.json

python3 scripts/audit_mad_knn_neighbors.py \
  bench/mad-v3-knn/capture-metal-f16 \
  --query-regime c_unseen_graph_unseen_wording \
  --neighbors 4 \
  --exclude-same-group \
  --output bench/mad-v3-knn/neighbour-audit-c.json
```

Repeat evaluation with `--exclude-same-group`. Run a smaller
HNSW arm only after the exact result exists.

Generate the fixed corpus with Glimmer's tokenizer authority:

```bash
python3 scripts/generate_mad_knn_corpus.py /path/to/Muse-Glimmer-30B \
  --profile adversarial \
  --out bench/mad-v3-knn/queries-adversarial.jsonl
```

`--profile pilot` reproduces the first P1 corpus. The adversarial profile is
the current scientific default for the next run: it permutes entity-name
families independently of relation, crosses generic syntax with every
operation, randomises record layout, and reserves an unseen relation synonym
for arm C.

The Metal capture uses the generic VINDEX3 device executor with f16 resident
operands because that path exposes the diagnostic seam. It is not the G6d
one-command-buffer `metal-lowered` schedule, so it supports a structural claim,
not a throughput claim. `--contribution-engine exact` is the scalar f64
accumulation authority. `fast` launches one Metal threadgroup per channel
block and returns only one f32 mass per logical object; it changes neither
logits nor the forward schedule.

## First Glimmer feasibility pilot

The 2026-08-16 pilot captured the final prompt-token residual at layer 32 and
the exact raw FFN-block contribution at layer 33 for 24 examples (12 history,
four queries per A/B/C regime). This is too small for a model claim. Its job was
to exercise every control on real Glimmer weights and measure whether the
generic observer path can support the planned corpus.

At `K=4` and a 20% future-byte budget:

| regime | semantic purity/base | operation purity/base | kNN | shuffle | popularity | oracle | effective blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.438 / 0.250 | 0.438 / 0.250 | 0.316 | 0.309 | 0.309 | 0.328 | 143.9 / 156 |
| B | 0.000 / 0.000 | 0.500 / 0.250 | 0.311 | 0.305 | 0.308 | 0.319 | 145.7 / 156 |
| C | 0.000 / 0.000 | 0.313 / 0.250 | 0.312 | 0.305 | 0.307 | 0.325 | 144.8 / 156 |

Regime A deliberately permits same-graph history so exact semantic identity
exists; B and C use `--exclude-same-group`. `K=1` purity was 1.0 for A exact
semantics, 0.75 for B operations, and 1.0 for C operations, but four queries
per arm make those values descriptive only.

The weak positive address separation was not yet evidence: the layer-33 oracle
ceiling is only 0.8–1.3 contribution points above kNN at the 20% budget, and
the contribution distribution is very dense (`exp(H)` is 92–93% of blocks).
The generic scalar observer took about 30 seconds per 90-token example for one
source/target layer pair after its one-off 60 GB load. That made unbiased layer
search impractical and motivated FAST-TAP rather than retrieval optimisation.

## FAST-TAP parity gate

FAST-TAP computes every FFN channel-block contribution on the device during a
normal observed decode. On a fixed real-Glimmer fixture it preserved the
captured residual plane byte-for-byte relative to the scalar observer. For the
contribution plane, maximum absolute error was `4.768e-7`, maximum relative
error was `1.195e-7`, mean relative error was `2.439e-8`, and the selected
block set agreed exactly at 5%, 10%, 20%, 40%, and 60% byte budgets. This is
the numerical contract: the observer is exact for the experiment's rankings
within f32 reduction tolerance, not bit-identical to f64 accumulation.

Independent prompts reset decode state. The capture path also reasserts the
device residency set after each reset; without this, later examples silently
fell off the sustained resident path even though their logits remained valid.

## First whole-model Glimmer census

The 2026-08-16 P1 scan used 120 balanced examples: 60 historical rows and 20
queries in each A/B/C regime. It captured the final prompt-token residual at
all 52 layers and all 52 FFN contribution signatures. Exact cosine is used
throughout; `K=4` is the headline and K=1/8 are robustness checks. This is
still a topology-finding pilot, not a significance claim.

The physical census found that low contribution entropy is not sufficient for
query-specific addressability. Layer 4 has only 30.3 effective blocks out of
156, yet at a 20% byte budget popularity covers 0.7211 and the per-query oracle
only 0.7239. Its sparse support is mostly the same globally hot support. The
largest all-query oracle advantages were instead modest and late:

| target layer | budget | popularity | oracle | oracle advantage |
| ---: | ---: | ---: | ---: | ---: |
| 49 | 5% | 0.2375 | 0.2751 | +0.0375 |
| 47 | 5% | — | — | +0.0347 |
| 35 | 10% | — | — | +0.0344 |
| 49 | 20% | 0.4750 | 0.5084 | +0.0333 |

No coarse 128-channel FFN address layer has a large oracle ceiling. The
database distinction is therefore `sparse support != query-specific physical
address`: a globally popular sparse page set does not create a useful index.

The operation-neighbour result is much stronger. In the hard C arm (unseen
graph and unseen wording, with same-group history excluded), operation purity
at `K=4` rises from a 0.25 base rate to 0.75 at layer 23, 0.975 at layer 31,
and 1.0 at layer 32; it remains 1.0 through layer 51. The transition is not
monotonic in earlier layers, which argues against summarising it as generic
depth. Regime A reaches roughly 0.20 exact-semantic purity against a 0.05 base
rate late in the network. Regime B also reaches 1.0 operation purity from
layer 32 onward.

### Layer-1 adversarial audit

The 2026-08-17 nearest-neighbour inspection found that the pilot corpus does
not support semantic canonicalisation independently of lexical cues:

- every query and history rendering repeats the canonical relation word;
- source names are deterministically coupled to relation
  (`facility -> Emberford`, `supervisor -> Vantfall`, `route -> Wrenrel`,
  `archive -> Sablevale`), even on held-out graphs;
- all 80 layer-1 top-four neighbours come from history template 1, regardless
  of the query's record layout or C-arm template;
- layer-1 purity is heterogeneous by operation: archive 1.0, facility 1.0,
  route 0.60, and supervisor 0.25.

Mean absolute query/neighbor token-length difference is 2.4 tokens versus
3.07 over all eligible pairs; no top-four pair has identical token length, and
record-layout agreement is 0.35 versus a 0.333 base. Length and record order
therefore do not explain the result, but the relation token and entity family
can. The layer-32 value of 1.0 remains a strong operation-decodability result;
it is not yet evidence for a canonical semantic key. MAD-KNN-3 is not banked.

### Adversarial C replication

The fixed 2026-08-17 replication used the same 120-row shape (60 history and
20 queries per arm) and exact cosine search. Entity families are permuted per
graph, syntax and record layout are crossed with operation, and hard arm C
uses held-out synonyms: `facility -> venue`, `supervisor -> director`,
`route -> way`, and `archive -> storehouse`.

At K=4, C changes from the pilot's early lexical jump and perfect late purity
to a broad mid/late semantic hump:

| source layer | operation purity | balanced base |
| ---: | ---: | ---: |
| 0 | 0.2500 | 0.2500 |
| 1 | 0.2750 | 0.2500 |
| 12 | 0.3500 | 0.2500 |
| 20 | 0.4625 | 0.2500 |
| 23 | 0.5000 | 0.2500 |
| 31 | 0.5875 | 0.2500 |
| 36 | 0.7125 | 0.2500 |
| 40 | 0.7125 | 0.2500 |
| 45 | 0.7125 | 0.2500 |
| 48 | 0.6125 | 0.2500 |
| 51 | 0.5125 | 0.2500 |

The effect is not tied to K: at layer 36 purity is 0.75, 0.7125, and 0.625
for K=1, 4, and 8. A graph-stratified label permutation that preserves all
four queries within each held-out graph gives `p=0.00002` after taking the
maximum over all 52 layers (100,000 deterministic draws). This is a descriptive
small-corpus randomisation test, not a population-level significance claim.

Token position does not account for the hump. Restricting candidates to
within one token of each query retains 16/20 queries and reaches 0.6406 at
layer 41 against a matched 0.2634 base. A two-token window retains 19/20 and
reaches 0.6579 against 0.2464. Token-count-only KNN is exactly 0.25.

The signal is uneven across held-out aliases. At layer 36, per-operation purity
is archive/storehouse 0.40, facility/venue 0.75, route/way 0.90, and
supervisor/director 0.80. The result therefore supports genuine mid/late
relation-semantic structure, but not an alias-invariant canonical key. Under
the predeclared interpretation it is a moderate positive, not the dramatic
`0.9+` result required to bank KNN-3. Alias rotation is the next replication.

The coarse physical result remains small. For adversarial C at source 48 to
target 49, 5% bytes cover 0.2184 under popularity, 0.2367 under KNN, and 0.2735
under the oracle: +1.82 contribution points and 33.1% of oracle headroom. At
20%, the corresponding values are 0.4677, 0.4838, and 0.5110 (+1.61 points,
37.2% of headroom). Source 47 to target 48 is the best 5% KNN gain at +2.11
points. This is still too little absolute leverage to optimise retrieval.

For reproducibility, the original confounded P1 coarse-address table is
retained below. Its strongest conjunction was source layer 48 predicting
target layer 49 at `K=4`:

| regime | budget | popularity | kNN | oracle | oracle headroom recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 5% | 0.2203 | 0.2422 | 0.2549 | 62.9% |
| B | 5% | 0.2311 | 0.2612 | 0.2693 | 79.1% |
| C | 5% | 0.2612 | 0.2921 | 0.3010 | 77.8% |
| A | 20% | 0.4453 | 0.4601 | 0.4746 | 50.6% |
| B | 20% | 0.4774 | 0.5042 | 0.5135 | 74.4% |
| C | 20% | 0.5024 | 0.5212 | 0.5370 | 54.3% |

The C-arm 5% coverage is stable across neighbour counts: 0.2927 at K=1,
0.2921 at K=4, and 0.2923 at K=8, versus 0.2612 popularity. Absolute gains
remain small because the coarse-block oracle ceiling is small, but kNN
recovers most of the available headroom at this layer. The present reading is:

> The adversarial corpus has a broad relation-semantic phase around layers
> 36–45; the best coarse FFN address opportunity remains around layers 47–49.

This keeps MAD-KNN-3 open rather than established, and keeps the narrower
MAD-KNN-1 alive at L47->49. The next logical gate is a preregistered rotation
of held-out aliases to distinguish a stable operation representation from
particular pretrained synonym geometry. The next physical gate varies only
the page schema (smaller channel groups, projection tiles, or attention
objects) while keeping the residual graph fixed. HNSW still does not belong
in the scientific path.

## MAD-V3-ROUTE-1: Successful Basin Multiplicity

ROUTE-1 replaces the canonical-key requirement with a stricter computational
question: do successful executions of one operation occupy multiple stable
residual basins that imply different downstream work? The candidate frontier
is fixed before inspection at layers 36–45, and the first physical target is
the existing coarse opportunity at FFN layer 49. A basin-conditioned predictor
cannot enlarge layer 49's oracle headroom; it can only recover more of it than
plain or operation-conditioned KNN.

The route corpus uses repeated executions for every operation/answer cell.
History and query graphs are disjoint. History rotates through three relation
aliases; the fourth is held out for every query. Syntax, record order, filler,
entity identity and token length vary within cells. `--alias-fold 0..3`
preregisters four independent lexical rotations rather than pooling synonyms
after seeing the result. The tokenizer-authoritative expected continuation is
captured, and the unchanged model is greedily decoded for exactly that many
tokens. By default ROUTE-1 substitutes unique, neutral, tokenizer-verified
single-token answer values; this preserves answer-identity diversity while
avoiding six extra decode steps for the old synthetic names. The slower
multi-token control remains available as `--route-answer-tokens synthetic`.
Basin discovery and banking use exact-answer executions only.

```bash
python3 scripts/generate_mad_knn_corpus.py /path/to/Muse-Glimmer-30B \
  --profile adversarial --alias-fold 3 --route-variants 8 \
  --train-graphs 20 --heldout-graphs 10 \
  --out bench/mad-v3-knn/route-fold-3.jsonl

target/release/larql dev mad-knn capture /path/to/glimmer.vindex3 \
  --queries bench/mad-v3-knn/route-fold-3.jsonl \
  --output bench/mad-v3-knn/route-fold-3-capture \
  --backend metal --contribution-engine fast \
  --residual-layers 36-45 --contribution-layers 49 \
  --block-channels 128

# The identical command may be restarted after interruption:
target/release/larql dev mad-knn capture /path/to/glimmer.vindex3 \
  --queries bench/mad-v3-knn/route-fold-3.jsonl \
  --output bench/mad-v3-knn/route-fold-3-capture \
  --backend metal --contribution-engine fast \
  --residual-layers 36-45 --contribution-layers 49 \
  --block-channels 128 --resume

python3 scripts/audit_mad_route_basins.py \
  bench/mad-v3-knn/route-fold-3-capture \
  --frontier-layers 36-45 --target-layer 49 --neighbors 16 \
  --output bench/mad-v3-knn/route-fold-3-audit.json
```

The audit constructs a mutual-cosine-KNN graph separately within each known
operation, then finds local Louvain communities without supplying wording,
alias, graph, layout, length, answer, correctness or contribution labels.
Held-out executions are assigned through their nearest same-operation history
residual. This operation stratification deliberately controls the already
established relation geometry; communities must explain variation within an
operation to count as candidate routes.

A ROUTE-1 basin counts only when all of the following survive on held-out
graphs and repeat across at least three of four alias folds in the same layer
band:

- the mutual-KNN community is stable under 80% node resampling (median
  within-operation adjusted Rand index at least 0.6);
- every operation has more successful history rows than `K+1` and at least
  twice the minimum credited-community size; a saturated mutual-KNN graph is
  reported as underpowered rather than as a one-basin result;
- every credited community has at least eight successful samples; answer,
  wording, graph and layout each have at least three categories, normalized
  entropy at least 0.65 and maximum share at most 0.50; alias has at least two
  categories, entropy at least 0.65 and maximum share at most 0.67; token count
  has at least two categories, entropy at least 0.50 and maximum share at most
  0.75;
- multiple answer cells themselves span more than one basin, ruling out a
  disguised answer partition;
- basin labels predict a distinct layer-49 contribution distribution after
  subtracting each operation/answer cell mean, with a within-cell permutation
  `p <= 0.05` corrected across the ten preregistered frontier layers;
- basin-conditioned KNN improves held-out future-contribution coverage over
  both plain and operation-conditioned KNN at the identical 5% and 20% byte
  budgets; and
- all credited basins contain exact-answer executions. A correct/incorrect
  split is failure geometry, not successful route multiplicity.

The script reports normalized nuisance entropy and maximum category share per
community, answer cells spanning multiple basins, bootstrap stability,
answer-conditioned downstream-signature R², and popularity/plain/operation/
basin/oracle byte coverage. It intentionally does not emit a banked verdict
from one capture: fold replication and the ten-layer multiplicity correction
remain study-level gates.

### Resumable capture contract

Long Glimmer runs commit one row at a time. `capture-state.json` pins the full
ordered fixture, model, backend, residual layers, physical objects,
contribution engine and block size. `progress.jsonl` is the commit authority:
a row is appended only after its residual and contribution bytes have flushed.
On `--resume`, the CLI requires the requested configuration to equal that
checkpoint exactly, discards an unterminated progress tail, truncates both
binary planes to the last committed row, reloads prior answer outcomes and
continues at the next fixture. `manifest.json` remains the sole completion
marker, so normal evaluators still refuse interrupted captures.

The real-Glimmer interruption drill stopped a 16-row capture after 12 committed
rows and resumed it to completion. Its residual plane, contribution plane and
manifest were exactly identical to an uninterrupted capture of the same
fixture. Partial captures made before this checkpoint protocol have no stored
answer outcomes and cannot be resumed.

### Fold 0 discovery result

The 2026-08-19 discovery fold used alias fold 0, ten history graphs, five
held-out graphs and four renderings per operation/answer: 160 history and 80
held-out executions. The resumable capture completed all 240 rows without an
interruption and recorded L36–45 residuals plus 156 coarse FFN objects at
layer 49.

Exact-answer filtering left only 46/160 history rows and 47/80 held-out rows.
Successful history counts were archive 7, facility 15, route 11 and supervisor
13; held-out counts were 10, 13, 12 and 12. The frozen `K=16` route gate is
therefore **underpowered rather than negative**: every within-operation KNN
graph is saturated, producing one complete community per operation at every
candidate layer. Bootstrap ARI is undefined, zero answer cells span multiple
basins, answer-conditioned downstream R² is zero and basin conditioning is
identical to operation conditioning. The audit now detects this condition and
credits no basin.

The power loss is structured. History template 6 is exact on 5/40 examples;
`archive/storehouse` and `route/way` are each 0/10. By contrast, fold 0 holds
the canonical aliases out for query, where archive, facility, route and
supervisor achieve 50%, 65%, 60% and 60%. This fold is an unusually hard
history-side lexical rotation, not a balanced sample of successful routes.

A separately labelled, non-bankable `K=4` diagnostic asks whether smaller
local graphs reveal enough structure to justify a larger successful corpus.
It finds small communities with median bootstrap ARI from approximately 0.57
to 0.77, and 5–10 of 30 answer cells span multiple communities depending on
layer. Almost none pass nuisance mixing: only one four-sample supervisor
community at L44–45 does, below the preregistered eight-sample floor. No
downstream-signature result survives the ten-layer correction (best corrected
`p ~= 0.086`).

The downstream diagnostic is adverse to the route interpretation. At a 5%
byte budget, popularity covers 0.2226, plain KNN peaks at 0.2273, operation-KNN
at 0.2263 and basin-KNN at 0.2246, against a 0.2471 oracle. At 20%, the maxima
are 0.4515, 0.4620, 0.4565 and 0.4537 against a 0.4843 oracle. Basin
conditioning is worse than operation conditioning at every L36–45 source
layer and both budgets. Fold 0 therefore supports, at most, representational
variation without demonstrated computational multiplicity. ROUTE-1 is not
banked, and causal triplets are not yet justified.

### Fold 1 powered confirmation

Fold 1 keeps the observational gate frozen: mutual `K=16`, minimum community
size eight, L36–45 residual discovery, layer 49 coarse FFN contributions, and
5%/20% byte budgets. Only corpus size and the preregistered alias rotation
change. Sixty history graphs provide 240 executions per operation before
correctness filtering; 25 disjoint held-out graphs provide another 100 per
operation. With four renderings per operation/answer cell, the capture has 960
history and 400 held-out executions.

The larger corpus requires 340 answer identities. The deterministic neutral
name pool was expanded rather than weakening the global-uniqueness control;
every selected value must still be a distinct tokenizer token whose decoded
continuation exactly round-trips the source spelling. The capture uses alias
fold 1 and records the same L36–45 residual and layer-49 contribution planes as
fold 0. The resumable capture completed all 1,360 rows on 2026-08-19; no
fold-1 outcome was inspected before freezing these settings.

Exact-answer filtering retains 436/960 history rows (45.4%) and 134/400
held-out rows (33.5%). History successes by operation are archive 98, facility
106, route 113 and supervisor 119; held-out successes are 29, 35, 35 and 35.
Every operation therefore clears the frozen `K=16` power guard at every layer.

The powered result is a **ROUTE-1 non-pass**. Communities are reproducible:
median bootstrap ARI ranges from 0.778 to 0.888 across L36–45, and 130–148 of
213 operation/answer cells span more than one community. Their layer-49
contribution signatures also differ after operation/answer residualisation
(`R² = 0.210–0.248`; within-cell permutation `p = 0.00010`, Bonferroni
`p = 0.0010`, at every frontier layer). This separation is not yet credible as
route multiplicity, however: only 0–2 of 19–22 size-eligible communities per
layer pass all nuisance-mixing thresholds. Most communities are pure or nearly
pure in wording and relation alias even while strongly mixing graph, answer,
layout and token length.

Most decisively, basin conditioning reduces held-out physical prediction at
both frozen budgets and every frontier layer. Relative to operation-conditioned
KNN, basin coverage is lower by 0.00242–0.00398 at 5% and by 0.00172–0.00392
at 20%. The best plain KNN coverage is 0.2603 at 5% and 0.4872 at 20%, versus
popularity 0.2456/0.4735 and oracle 0.2733/0.5078. Thus the continuous residual
neighbourhood recovers useful coarse address headroom, but discretising it into
the observed communities destroys predictive information. Fold 1 supports
stable lexical/representational multiplicity with downstream correlates, not
nuisance-invariant successful computational basins. Causal triplets remain
unjustified under the preregistered gate.

## MAD-V3-FIELD-1: Computational Frontier Geometry

FIELD-1 drops the discrete-route assumption. Its hypothesis is that a
successful residual encodes a continuous future-computation state: nearby
states should induce smoothly related mixtures of future physical work, even
when no nuisance-invariant community label exists. The completed powered
fold-1 capture is reused without touching execution. Because its uniform-KNN
result is already known, this run is a discovery fold rather than a bankable
replication.

The scientific split, exact-success filter, L36–45 frontier, layer-49 objects,
exact cosine retrieval, `K=16`, same-graph exclusion, and 5%/20% byte budgets
remain fixed. Each query is evaluated with five non-oracle profiles:

1. history popularity;
2. the uniform mean of its 16 nearest successful history contributions;
3. an adaptive Gaussian mean with residual cosine distance `d` and weights
   `exp(-0.5 * (d / d_K)^2)`, where `d_K` is the farthest selected neighbour;
4. local-linear/LLE reconstruction weights from the regularised neighbour
   Gram matrix, with ridge `1e-3 * trace(C) / K` and weights constrained to sum
   to one; and
5. local low-rank regression: centre the same residual/contribution neighbours,
   retain the smallest residual-PC rank explaining 95% variance capped at
   eight and `K-2`, then fit multivariate ridge with `1e-3` times mean retained
   variance.

Negative predicted contributions are clipped to zero; a degenerate predictor
falls back to the uniform neighbour mean. Rankings use contribution per object
byte, exactly as ROUTE-1. In addition to contribution coverage and recovered
popularity-to-oracle headroom, the audit reports cosine similarity and
Jensen–Shannon divergence between predicted and observed contribution
distributions. Candidate-vs-uniform coverage differences use 10,000 paired
sign-flip permutations. The discovery table reports raw and Bonferroni values
over three candidate methods, ten layers and two budgets (60 comparisons).

A continuous estimator counts as a FIELD-1 improvement only if it beats
uniform KNN at both budgets in the same layer band, survives the 60-way
correction, and improves contribution-vector similarity rather than exploiting
only a ranking tie. A banked smooth-field claim additionally requires the same
layer band and estimator ordering in at least three of four preregistered alias
folds. If continuous estimators fail while uniform KNN remains useful, the
supported claim is local neighbour similarity, not a smooth coordinate field.
Offline interpolation and approximation/rescue trajectories remain deferred
until an estimator clears this observational gate.

### FIELD-1 discovery result

The fold-1 evaluator reproduces ROUTE-1's uniform-KNN coverage exactly
(`max_abs_error = 0`) before comparing the new estimators. FIELD-1 is a
**discovery non-pass**. Adaptive distance weighting changes byte coverage by
only -0.000132 to +0.000312 at 5% and -0.000148 to +0.000285 at 20% across
L36–45. Its strongest same-layer result is L43 (`+0.000312/+0.000285`), with
raw one-sided paired `p = 0.031/0.048`, but both 60-way corrected values are
1.0. The improvement is concentrated in very few queries because most
rankings tie the uniform estimator.

Distance weighting does reveal a tiny consistent full-vector effect at every
layer: cosine rises by 0.000078–0.000112 and Jensen–Shannon divergence falls by
0.000031–0.000045 relative to uniform KNN. That perturbation is too small to
produce robust byte-ranking gains. Local-linear reconstruction is worse at
the 5% budget at every layer and degrades vector cosine by 0.00038–0.00361.
Local low-rank regression is also worse at 5% everywhere and degrades cosine
by 0.00031–0.00117. Neither method clears both budgets at any layer; all
corrected coverage tests are non-significant. No estimator used its fallback,
and the preregistered low-rank rule selected the cap of eight for every query.

The strongest supported statement is therefore narrower than the smooth-field
hypothesis: **local residual neighbourhood membership predicts future coarse
physical work, but distance and local affine coordinates add no material
held-out address information at this granularity.** Uniform local averaging
still recovers up to 53.2%/40.0% of popularity-to-oracle headroom at 5%/20%,
respectively. Discrete lexical communities discard that signal, but richer
continuous estimators do not improve it. This is compatible with a locally
flat or noisy field, an inadequate FFN page schema, or useful geometry that is
not affine in cosine coordinates; FIELD-1 does not distinguish those cases.
Interpolation and causal trajectory experiments remain gated off.

## MAD-V3-STATE-1: Local Computational Equivalence

STATE-1 asks whether the useful FIELD-1 object is a local cohort rather than a
metric coordinate system. It reuses the sealed powered fold-1 capture and is a
discovery study. No contribution target participates in neighbour selection.

The scale arm runs exact, uniform cosine KNN independently at every L36–45
source layer for `K = 1, 2, 4, 8, 16, 32, 64, 128`. The split, exact-success
filter, same-graph exclusion, L49 physical target and 5%/20% budgets remain
unchanged. It reports contribution-vector cosine/Jensen–Shannon divergence,
future-byte coverage, oracle-headroom recovery, operation purity, query-template
purity, exact-alias purity and +/-1-token-count proximity. Adjacent-layer
Jaccard, retention with a finite-history chance adjustment, and the size of
each query's ten-layer neighbour union measure cohort stability. Each
alternative K is compared with the already established
`K=16` using 10,000 paired sign flips; discovery multiplicity is 7 alternatives
times 10 layers times 2 budgets (140 comparisons).

The persistence arm fixes `K=16` at every frontier layer. For each held-out
query/history pair, edge strength is the number of L36–45 layers in which the
history execution belongs to the query's top-16 set. The persistent cohort is
the 16 histories with greatest edge strength, breaking ties by mean ordinal
rank across all ten layers and then capture order. Contributions and labels are
not used. Its uniform future-contribution vote is compared conservatively with
L38, the strongest known single-layer uniform baseline on fold 1, using two
paired sign-flip tests corrected across the two byte budgets. Comparisons with
all other source layers remain descriptive.

A characteristic scale requires a reproducible interior plateau or optimum at
both budgets, not a one-budget maximum selected from this discovery fold. A
persistence improvement requires greater coverage than L38 at both budgets,
two-way-corrected `p <= 0.05`, improved vector cosine and Jensen–Shannon
divergence, and replication in the same frontier across three of four alias
folds. Otherwise STATE-1 may establish neighbourhood persistence or semantic
composition, but not an execution-state equivalence graph with superior
physical predictivity. Radius neighbourhoods are deliberately deferred until
the fixed-K sweep establishes whether a characteristic boundary exists.

### STATE-1 discovery result

The evaluator reproduces FIELD-1's `K=16` coverage exactly
(`max_abs_error = 0`). The scale curve is real but broad rather than a sharp
neighbourhood boundary. Averaged over L36–45, 5%/20% coverage progresses as:

```text
K=1    0.25366 / 0.47690
K=2    0.25624 / 0.48374
K=4    0.25685 / 0.48436
K=8    0.25825 / 0.48564
K=16   0.25981 / 0.48700
K=32   0.26055 / 0.48601
K=64   0.26048 / 0.48647
K=128  0.25958 / 0.48477
```

Thus K=16–64 forms a budget-dependent plateau: K=32 is best on average at 5%,
while K=16 is best at 20%, and K=128 declines at both budgets. No alternative
K beats K=16 after the preregistered 140-way correction. The closest result is
K=32 at L45/5% (`delta = +0.00162`, raw `p = 0.00050`, corrected `p = 0.070`).
There is no single characteristic scale to bank from this fold.

Neighbourhood identity is nevertheless highly persistent. At K=16, adjacent
layers retain 90.1% of members versus 3.67% chance retention
(chance-adjusted retention 0.897). The ten-layer union contains 25.17 histories
on average, while 8.84 of 16 members survive in every L36–45 set. The explicit
persistence selector chooses histories present in 87.3% of layers on average;
55.2% of its selected histories occur in all ten layer neighbourhoods.

That topology does **not** establish a superior execution-state equivalence
graph. Persistent voting covers 0.25977/0.48709, below the conservative L38
baseline by 0.00055/0.00013 (`p = 0.935/0.655`; corrected 1.0). It improves the
full contribution vector relative to L38 (cosine `+0.00124`, Jensen–Shannon
`-0.00030`) but not the high-density object ranking. Its cohort also remains
wording-loaded: wording-family purity is 0.750 versus operation purity 0.631,
so persistence does not remove the lexical nuisance exposed by ROUTE-1.

STATE-1 is therefore **topology-positive and physical-predictivity-negative**.
Glimmer maintains remarkably stable local cohorts across the frontier, and
uniform averaging over a moderate cohort is useful, but neither persistence
nor a unique neighbourhood boundary adds robust future-byte information. The
current evidence is consistent with local-constant denoising over stable
lexical/semantic cohorts; it does not yet identify computational equivalence.
Radius tuning, trajectory graph communities and causal interventions remain
gated off rather than being used to optimise around this result.

## MAD-V3-PAGE-1: Physical Schema Search

PAGE-1 freezes the logical side at the powered fold-1 corpus, exact-success
rows, history/query split, L36–45 residuals, exact cosine, uniform `K=16`,
same-graph exclusion and 5%/20% byte budgets. Only the physical partition of
L49's 19,968 FFN intermediate channels changes. The preregistered contiguous
schemas are 16, 32, 64, 128 and 256 channels per page (1,248, 624, 312, 156 and
78 objects). L38 is the primary source layer because it was the strongest
known fixed-index control before PAGE-1; all L36–45 results are secondary
robustness checks, not independently optimised predictors.

All schemas are observed during the same unchanged forward. The gated FFN
activation is computed once, then a read-only exact contribution reduction is
run for each schema. A 19,968-channel one-page schema is captured as an
authority for the squared norm of the complete raw down-projection output. It
is a control, not a candidate at 5%/20% budgets. Coarse page energy cannot be
reconstructed by summing fine-page energies because squared norms contain
cross-channel terms; the multi-schema observer therefore measures every page
definition directly. Before the powered run, a real-Glimmer fixture must show
that every schema's multi-capture contribution vector matches a separate
single-schema capture and that residual/logit outcomes remain identical.

For each candidate schema PAGE-1 reports popularity, frozen uniform KNN and
oracle contribution coverage at identical byte fractions, plus:

```text
oracle headroom = oracle coverage - popularity coverage

fraction recovered = (KNN coverage - popularity coverage)
                     / oracle headroom
```

It also reports effective contribution entropy and the interference factor
`sum(page norm²) / full-output norm²`; cross-schema coverage is interpreted
only alongside that factor. At primary L38, each non-control schema is compared
with 128-channel pages using paired 10,000-sign-flip tests for oracle-headroom
change and KNN-gain change at both budgets, corrected over four alternative
schemas times two quantities times two budgets (16 comparisons).

PAGE-1 outcome A requires oracle headroom to grow under a finer schema while
the frozen KNN retains or increases a substantial recovered fraction. Outcome
B is growing oracle headroom with falling recovered fraction. Outcome C is no
material oracle-headroom growth, which closes further subdivision of
contiguous checkpoint channels and motivates history-only functional pages.
This discovery fold cannot bank a physical-design claim without an independent
alias rotation, but it can choose among A/B/C without changing the predictor.

The implementation gate passed on a four-row real-Glimmer fixture before the
powered capture. The six schema slices from one multi-schema forward are
byte-identical to independent 16/32/64/128/256/19,968-channel captures; all
residual planes and answer outcomes are also identical, with maximum
contribution error exactly zero. A copied capture truncated after three commits
resumed its fourth row to byte-identical residual, contribution, progress and
manifest artifacts. Multi-schema observation is therefore exact and resumable
under the same contract as the earlier single-schema observer.

### PAGE-1 discovery result

The powered capture completed all 1,360 examples and the frozen evaluator used
the same 436 successful history executions and 134 successful held-out queries
as ROUTE-1/FIELD-1/STATE-1. PAGE-1 is an **Outcome B discovery result**: finer
contiguous pages reveal more query-specific oracle selectivity, but the frozen
residual-neighbour index does not recover the additional opportunity.

At the preregistered primary source layer L38, the schema ladder is:

```text
channels  objects   5%: popularity / KNN / oracle   headroom / recovered
     256       78       .20040 / .20304 / .20984       .00944 / 28.0%
     128      156       .24563 / .26033 / .27325       .02762 / 53.2%
      64      312       .30855 / .31712 / .33911       .03056 / 28.1%
      32      624       .36963 / .38462 / .40903       .03941 / 38.0%
      16     1248       .43307 / .44582 / .47150       .03843 / 33.2%

channels  objects  20%: popularity / KNN / oracle   headroom / recovered
     256       78       .41402 / .42455 / .44592       .03190 / 33.0%
     128      156       .47349 / .48722 / .50782       .03433 / 40.0%
      64      312       .52552 / .53922 / .56270       .03718 / 36.8%
      32      624       .57420 / .58466 / .61088       .03668 / 28.5%
      16     1248       .61786 / .63060 / .65956       .04170 / 30.5%
```

Relative to the 128-channel control, 16-channel pages increase oracle headroom
by 0.01081 at 5% and 0.00737 at 20%; both paired sign-flip tests have
16-way-corrected `p = 0.00160`. Thirty-two-channel pages also increase 5%
headroom by 0.01179 (`corrected p = 0.00160`), and 64-channel pages increase it
by 0.00294 (`corrected p = 0.0240`). None of the eight finer-schema/budget tests
shows a corrected improvement in KNN gain. At 16 channels the KNN gain actually
changes by -0.00195/-0.00099 relative to control, while at 32 channels its 5%
change is only +0.00030 despite the +0.01179 oracle-headroom increase.

The effective contribution fraction falls from 0.670 at 128 channels to 0.442
at 32 and 0.334 at 16, confirming that finer pages expose conditional
concentration. The full-output interference factor remains tightly grouped
across schemas (mean 0.402--0.407), so the ordering is not explained by a gross
cross-schema change in summed-page energy. Coverage nevertheless remains a
schema-relative squared-output-energy quantity, not an additive decomposition
of the full FFN output.

The supported conclusion is therefore deliberately split: **contiguous L49
channels contain finer physical selectivity than the 128-channel layout
exposes, but the fixed L36--45 cosine neighbourhood is not sufficient to
address that extra selectivity.** PAGE-1 does not support dynamic residency or
a better page size, because finer pages lower rather than preserve the fraction
of oracle headroom recovered. It also does not justify predictor tuning on this
fold. An independent alias rotation is still required before banking the
physical-selectivity claim; any next gate must address the logical key/page
mapping rather than merely subdividing the tensor again.

## MAD-V3-KEY-1: Address Refinement

KEY-1 asks whether the PAGE-1 gap is caused by representing execution as a
single residual position rather than a position plus recent transition state.
It is an offline discovery analysis over the already completed capture; model
execution, successful rows, history/query split, same-graph exclusion, L49
targets, uniform `K=16`, exact cosine and 5%/20% byte budgets remain frozen.
The primary physical schemas are the PAGE-1 16- and 32-channel pages. The
128/64/32/16 ladder is also used for a descriptive within-neighbourhood
disagreement census, but only 16 and 32 enter the key-comparison family.

The preregistered keys use only state available at or before L38:

```text
h38                 current residual control
[h37, h38]          two-position state
delta38             h38 - h37
[h38, delta38]      position plus current transition
[h38, delta37,
      delta38]      position plus two-transition summary
```

Every component is independently L2-normalised before combination. Cosine of
a compound key is therefore the mean component cosine, exactly equivalent to
cosine on the concatenated unit component vectors. This prevents residual
magnitude or a high-norm delta from silently choosing the key weighting. The
neighbours vote uniformly; no labels, contributions, learned metric or fitted
predictor participate in retrieval.

For the original `h38` cohort, KEY-1 reports contribution-profile agreement
among the 16 histories at each page scale: neighbour-to-centroid cosine and
Jensen--Shannon divergence plus pairwise top-page Jaccard at both byte budgets.
This determines whether PAGE-1's finer oracle opportunity appears as increasing
unresolved variation inside an otherwise fixed residual neighbourhood.

Each of the four alternative keys is compared with `h38` at two schemas and
two budgets using paired 10,000-sign-flip tests, corrected over all 16
comparisons. A discovery address-refinement pass requires the same augmented
key to improve held-out coverage at both budgets on both 16- and 32-channel
pages with corrected `p <= 0.05`, while improving contribution-vector cosine
and Jensen--Shannon divergence. A delta-only result is evidence that transition
state is address-bearing but not yet a sufficient query key. No claim is banked
without an independent alias rotation reproducing the estimator and layer.
Attention/FFN internal keys and trained projections remain gated off until this
read-only residual-transition test answers whether another capture is warranted.

### KEY-1 discovery result

The evaluator reproduces PAGE-1's L38 control exactly
(`max_abs_error = 0`). KEY-1 is an **address-refinement non-pass**. None of the
four alternative keys improves fine-page coverage after the preregistered
16-way correction; every corrected value is 1.0. The two-position
`[h37,h38]` key is the least negative result: relative to `h38`, its
16-channel coverage changes by -0.00013/-0.00033 at 5%/20%, while its
32-channel coverage changes by only +0.00012/+0.00009. It produces tiny
full-vector improvements (cosine about +0.00017 and Jensen--Shannon about
-0.00004) that do not change the useful page ranking.

Transition-only retrieval is worse. `delta38` changes 16-channel coverage by
-0.00124/-0.00154 and 32-channel coverage by -0.00061/-0.00109. Combining
position and transition does not rescue it: `[h38,delta38]` and the
two-transition summary are below `h38` at both budgets on both schemas. Thus
the missing fine address information is not exposed by an equally weighted
first-order residual velocity over L36--38.

The fixed-`h38` disagreement census nevertheless validates the physical/logical
gap directly:

```text
channels  neighbour-centroid cosine / JS    top-page Jaccard 5% / 20%
     128                 .97142 / .00952              .5105 / .5866
      64                 .96711 / .01332              .5071 / .5606
      32                 .96457 / .01794              .5328 / .5239
      16                 .96351 / .02382              .5318 / .4831
```

As pages refine from 128 to 16 channels, neighbour-to-centroid JS divergence
increases 2.5x and cosine agreement declines monotonically. Top-page agreement
also declines monotonically at the 20% budget; the 5% Jaccard is not monotonic,
so no boundary claim is made there. PAGE-1's finer oracle opportunity therefore
does appear as unresolved downstream variation inside a stable residual cohort,
but recent residual position/delta keys do not identify it.

The supported narrowing is: **the L38 residual locates a broad future-work
cohort, while the fine L49 addressing distinctions are not recoverable from an
unweighted first-order residual trajectory.** Further cosine/trajectory tuning
is not licensed by this fold. If the arc continues, the next read-only capture
must test state created by the intervening computation itself--for example
attention output, FFN input or gate/preactivation summaries--with PAGE-1's
16/32-channel targets and KEY-1's split/predictor still frozen.

## MAD-V3-ADDR-1: Address Formation Census

ADDR-1 asks where the model manufactures the fine L49 address information that
PAGE-1 established and KEY-1 could not recover from a residual trajectory. The
first stage is a localisation census, not an attempt to make every internal
tensor a new predictor. It freezes the powered fold-1 exact-success IDs (436
history, 134 query), their order and metadata, same-graph exclusion, exact
cosine, uniform `K=16`, L49 16/32-channel targets and 5%/20% byte budgets. A
filtered 570-row fixture is derived solely from the sealed correctness manifest
before capture; answer outcomes must remain exact in the new run.

ADDR-1A captures these exact hidden-width seams at every layer L38--49:

```text
layer_input       h_l before pre-attention norm (the residual control)
attention_input   normalised vector consumed by Q/K/V projections
attention_output  attention branch after any branch norm, before residual add
post_attention    residual after the attention branch is added
ffn_input         normalised vector consumed by gate/up projections
ffn_output        FFN branch after any branch norm, before residual add
```

The decode observer only copies values already produced by the unchanged
forward. It cannot replace an operation. `post_attention` and the next
`layer_input` retain the cumulative-state boundaries; branch outputs are kept
separate so a predictive jump can be attributed to the computation that
created it. Gate/up preactivations and activated FFN state are deliberately
deferred to ADDR-1B: they are intermediate-width, would add several gigabytes,
and should be captured only at layers selected by the structural census.

Every `(layer,seam)` is independently used as a cosine key into successful
history, with no labels or contribution values in retrieval. The report gives
popularity/KNN/oracle coverage, vector cosine/Jensen--Shannon divergence and
oracle-headroom recovery. Formation tests compare cumulative state across the
actual residual joins: `layer_input -> post_attention` for attention and
`post_attention -> next_layer_input` for the FFN. Pre-attention and pre-FFN
normalisation transitions are also tested as coordinate transformations.
Branch-only `attention_output`/`ffn_output` remain descriptive keys; comparing
a branch alone with the recombined residual would confound new information with
the old residual being added back. The complete layer/transition/schema/budget
family uses 10,000 sign flips under one correction. A formation boundary
requires the same residual join to improve both budgets on both page schemas
after correction and to improve both vector metrics. Replication on an alias
rotation is required before banking it.

Lead time is reported separately as the number of complete layers remaining
before L49 consumes its down-projection pages. All L38--48 seams are admissible
advance predictors. At L49, `layer_input`, `attention_input`,
`attention_output`, `post_attention` and `ffn_input` are pre-down observations;
`ffn_output` is explicitly a post-consumption formation control and can never
support a prefetch claim. If ADDR-1A localises a jump to an FFN boundary,
ADDR-1B will capture actual gate/up/activated state only around that boundary
and will preserve the same distinction between formation and usable notice.

The ADDR-1A implementation gate passed before the powered run. On a four-row
real-Glimmer fixture, the seam-enabled forward matches PAGE-1 residuals,
16/32-channel contributions and answer outcomes exactly. Both execution
identities--`layer_input + attention_output = post_attention` and
`post_attention + ffn_output = next_layer_input`--have maximum absolute error
zero. A separate live run was interrupted after one committed row and resumed
to two; its residual, contribution, seam and progress artifacts are byte-for-
byte identical to an uninterrupted two-row capture. The evaluator then
completed over the fixture. The observer is therefore read-only, exact and
resumable at the new seam-plane boundary.

The first powered evaluator draft mechanically compared every adjacent listed
seam and therefore labelled `attention_output -> post_attention` as a formation
edge. That is not a valid information-formation contrast: `post_attention`
adds the pre-existing layer residual back to the branch output. Its apparent
L43/L47/L48 passes are retained in the v1 artifact but superseded before any
claim was banked. Audit schema v2 uses only the cumulative residual joins above;
all raw seam-key curves remain unchanged.

### ADDR-1A discovery result

The powered capture completed all 570 frozen exact-success rows (436 history,
134 query). Against the independent PAGE-1 capture, residuals, 16/32-channel
contributions, answer outcomes and both residual-join identities have maximum
absolute error zero. Audit v2 reproduces every overlapping PAGE-1 residual
baseline exactly (`max_abs_error = 0`).

ADDR-1A is a **localisation non-pass**. None of the 47 cumulative residual-join
or normalisation transitions clears the complete 188-test correction family;
there are no formation passes. The strongest residual-join near miss is the
L47 FFN join on 32-channel pages at 20%: coverage increases by 0.00066
(`raw p = 0.00050`, `corrected p = 0.0940`), but it does not clear correction
and does not reproduce at the other budget or 16-channel schema. There is no
defensible sharp attention/FFN boundary at which fine-address information
appears in the hidden-width state.

The descriptive lead-time curve does show gradual late enrichment rather than
a flat null:

```text
key/seam                    16ch recovered 5% / 20%   32ch recovered 5% / 20%
L38 layer_input                      .3317 / .3054              .3805 / .2853
L47 ffn_output (2 layers ahead)      .3594 / .3218              .4285 / .3200
L49 attention_input (pre-down)       .3669 / .3374              .4108 / .3199
L49 ffn_output (post-consumption)    .4171 / .3402              .5115 / .3521
```

The L47 branch-only result is an admissible advance association but not a
formation edge, and its selection from the full census is descriptive. The L49
FFN-output control is strongest at 5%, especially for 32-channel pages, but it
exists only after the target down projection has consumed those pages and
cannot support prefetch. Its comparatively small 20% improvement also argues
against treating it as a general address authority.

The supported conclusion is narrower than the motivating hypothesis:
**fine-page predictivity becomes modestly richer as execution approaches L49,
but ADDR-1A does not locate a discrete hidden-state seam where the model
manufactures the address.** Formation may be distributed, may live transiently
in intermediate-width gate/up/activated coordinates, or may simply remain
below this workload's observable ceiling. Under the preregistered ordering,
ADDR-1B gate/activation capture remains gated off because no FFN residual join
passed across both schemas and budgets; it must not be opened by selecting the
best branch output after inspection.

## MAD-V3-PAGE-1R: Alias-Rotation Replication

PAGE-1R is the independent lexical replication required before banking
PAGE-1's physical-selectivity result. It uses preregistered alias fold 2 after
fold 1 discovery, with the same 60 history graphs, 25 held-out graphs, four
renderings per operation/answer cell, neutral tokenizer-verified one-token
answers, row-generation seed and history/query graph split. The underlying
graphs, operations and answer identities are unchanged; only which relation
alias is held out for queries is rotated. No fold-2 outcome is inspected before
this gate is frozen.

The logical and physical analysis remains unchanged: exact-answer executions
only, same-graph exclusion, exact cosine, uniform `K=16`, primary residual L38,
secondary L36--45 census, L49 contiguous 16/32/64/128/256-channel schemas plus
the 19,968-channel norm authority, and 5%/20% byte budgets. Popularity, KNN,
oracle, effective contribution fraction and the full-output interference
factor use the PAGE-1 evaluator without parameter changes. The paired
10,000-sign-flip family remains correction over four alternative schemas,
oracle-headroom/KNN-gain and two budgets (16 comparisons).

The replication is powered only if every operation has at least 32 successful
history executions and 20 successful held-out executions. The physical claim
is banked only if 16-channel pages increase oracle headroom over the
128-channel control at both 5% and 20% with positive effect and corrected
`p < 0.05`, effective contribution fraction again decreases at finer
granularity, and the interference factor shows no gross schema-dependent shift.
The frozen KNN may continue to recover a smaller fraction of the finer oracle
opportunity: PAGE-1R tests whether conditional physical concentration
replicates, not whether the current logical key addresses it. Thirty-two- and
64-channel outcomes and all L36--45 source-layer variation are robustness
descriptions and cannot rescue a failed primary 16-channel gate.

If the gate passes, the banked pair of results is deliberately asymmetric:
**dense Glimmer has query-dependent fine-grained L49 FFN concentration, while
the current long-horizon residual key resolves coarse working-set structure
better than exact fine pages.** Together with ADDR-1A, this motivates
progressive multi-resolution working-set prediction rather than a hidden
single-step router. If the gate fails, PAGE-1 remains discovery-only and no
physical residency claim is licensed.

## MAD-V3-HORIZON-1: Progressive Resolution

HORIZON-1 is a read-only interaction test over the completed ADDR-1A and
PAGE-1 artifacts. It asks whether approaching L49 preferentially improves
prediction of fine pages relative to coarse pages, rather than merely making
all resolutions more predictable. It introduces no Glimmer execution and
does not inspect PAGE-1R. The frozen exact-success 436/134 history/query split,
same-graph exclusion, exact cosine, uniform `K=16`, L49 targets and 5%/20%
byte budgets remain unchanged.

Four states are selected before inspecting the interaction:

```text
stage 0   L38 layer_input
stage 1   L44 layer_input
stage 2   L47 ffn_output
stage 3   L49 ffn_input       (last captured pre-FFN hidden-width state)
```

Their fixed proximity coordinates are `[0, 6, 9, 11]`, measured as progress
from the L38 control toward L49. `L49 ffn_output` is excluded because it is
post-consumption. The physical resolutions are the independently observed
128-, 32- and 16-channel PAGE-1 schemas. Coarser squared-output page energies
must not be synthesized by summing 16-channel energies: cross-page terms make
that operation invalid, and PAGE-1 already captured every schema exactly in
the same unchanged execution.

For query `q`, state `s`, schema `c` and budget `b`, define a query-level
recovered-headroom contribution

```text
R(q,s,c,b) = (KNN_coverage(q,s,c,b) - popularity_coverage(q,c,b))
             / mean_q(oracle_coverage(q,c,b) - popularity_coverage(q,c,b))
```

The fixed schema-level denominator avoids unstable per-query ratios, while
`mean_q R` exactly equals the reported aggregate fraction of oracle headroom
recovered. For each query and budget, fit the OLS slope of
`R(16ch) - R(128ch)` over the four proximity coordinates. The primary statistic
is each query's mean slope across the 5% and 20% budgets. One 10,000-draw
one-sided paired sign-flip test evaluates whether its mean is positive.
HORIZON-1 passes only if `p < 0.05` and the aggregate 16-vs-128 slope is
positive separately at both budgets. The identically defined 32-vs-128 slopes
are prespecified corroboration but cannot rescue a failed primary gate.

A pass supports the narrow statement that **fine-address resolution improves
preferentially as L49 consumption approaches**. Equal improvement at all
resolutions, a flat fine penalty, or a failed primary interaction is a
progressive-resolution non-pass even if individual late states predict work
better. No individual seam significance tests, alternate state selection or
post-consumption result may rescue the interaction.

### HORIZON-1 result

The frozen offline audit reproduces PAGE-1's L38 coverage at all three schemas
and both budgets with maximum absolute error zero. Its deterministic rerun is
byte-identical (`sha256 20e73778f9cb0f2cc90344e776d0017b740a5f1cc8e7f02f1435c48e813f9fa4`).
HORIZON-1 is a **progressive-resolution non-pass**.

The recovered-headroom fractions are:

```text
state                 128ch 5/20%       32ch 5/20%       16ch 5/20%
L38 layer_input       .5320 / .3999     .3805 / .2853    .3317 / .3054
L44 layer_input       .4974 / .3853     .3848 / .2992    .3408 / .3159
L47 ffn_output        .5486 / .4164     .4285 / .3200    .3594 / .3218
L49 ffn_input         .5602 / .3842     .4043 / .3191    .3611 / .3265
```

At 5%, the primary 16-vs-128 fine penalty is -0.2003 at L38 and -0.1991
at L49: it does not close. At 20% it descriptively narrows from -0.0945 to
-0.0577. The budget-averaged per-query slope is only +0.00119 per proximity
unit (`55.2%` positive queries; one-sided sign-flip `p = 0.3178`), so the
primary gate fails despite positive aggregate slopes at both budgets.

The prespecified 32-vs-128 corroboration also fails. Its 5% deficit slightly
worsens from -0.1515 to -0.1558 while the 20% deficit narrows from -0.1146 to
-0.0651; the omnibus slope is +0.00213 (`p = 0.1685`). Thus late states can
predict somewhat more future work without preferentially resolving fine pages
in a stable cross-query manner.

The supported conclusion is deliberately negative: **these existing states do
not provide direct evidence that prediction horizon determines useful page
granularity.** Coarse 128-channel prediction retains a recovered-headroom
advantage at every tested state, and proximity does not reliably close it.
This does not invalidate a hierarchical residency design as an engineering
hypothesis, but HORIZON-1 does not establish it as Glimmer's observed internal
organisation. No alternate seam, distance coordinate or post-consumption state
is selected after this result.

### Causal follow-up (not part of the observational gate)

After a basin survives ROUTE-1, exact/approximation/rescue triplets will track
distance to the paired exact trajectory, distance to the nearest successful
manifold, and basin membership by layer. Returning near the paired trajectory
is repair; entering another successful basin while remaining far from the
paired exact is an alternate route; leaving the successful manifold is failure
rerouting. The strongest planned case fixes prompt, starting residual and
answer while two distinct rescue interventions produce stable, physically
different successful basins. The current read-only MAD observer does not yet
claim this causal result.

## Capture artifact

`manifest.json` is written last and is the completion marker. The two data
files are little-endian f32:

```text
residuals.f32      [sample][residual_layer][hidden]
contributions.f32 [sample][object]
```

The manifest pins sample labels and token count, object layer/address/byte
cost, binary file names, contribution-engine provenance, and the contribution
semantics `l2_norm_squared_raw_down_block`. Evaluation refuses wrong sizes,
duplicate IDs, missing history/query partitions, zero residuals, non-finite values,
negative contribution masses, path traversal, or another contribution
definition.

## Deferred

- attention head/head-group and O-projection slice objects;
- physical segment/tile addresses rather than logical channel slices;
- packed NVFP4 contribution accounting;
- heterogeneous residual/temporal/semantic/VINDEX graph walks (KNN-2);
- query-to-physical-plan stability (KNN-4);
- graph delta and mutation-frontier analysis (KNN-5);
- any use of kNN in the inference hot path.
