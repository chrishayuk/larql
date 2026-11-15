# VERIFY-REMOTE-1: prompt-lookup speculation over remote experts

Status: **DRAFT, revision 0. Unqualified, not approved for any model run,
not frozen.** Nothing below has been built or measured. This document holds
the interface audit, the proposed protocol, the blockers and the testing
plan, for review.

## Question

Does the existing prompt-lookup speculative decoding make **greedy
generation** of GPT-OSS 20B faster, in committed new tokens per second,
when the routed experts run on a remote worker, once drafting,
verification, rejection and rollback are all paid for?

BLOCK-TIME-1 and BLOCK-TIME-2 measured verification of *supplied* tokens:
3.37× more evaluated positions per second at 20 ms for width 8, and the
5-to-20 ms slope following `24 / W`. Neither generated anything. Whether
real acceptance turns that into faster generation is the question here.

**Scope of this first rung:** greedy decoding only; the prompt-lookup
proposer only; no model drafter, no sampled-output acceptance rule, no new
hardware topology, no physical network.

## Interface audit

What exists on main (`4a1279a9`), who owns each step, and what is missing.

| Step | Owner today | Path | Remote-ready? |
|---|---|---|---|
| Proposing | `prompt_lookup::propose` (pure: longest trailing n-gram, 2 ≤ n ≤ 16, earliest occurrence, prompt only) | `crates/larql-cli/src/commands/primary/vindex3_cmd/lowered/prompt_lookup.rs` | **No**: a private module of the local Metal command (`mod prompt_lookup;`), unreachable from the remote path |
| Acceptance | `prompt_lookup::accepted_prefix` (pure: leading agreement of the guess with the target's greedy ids) | same file | **No**, for the same reason |
| Speculative loop | `speculative_decode` over `LoweredSession` | `.../lowered/run.rs:336` | **No**: written for the local Metal session; interleaves its own decode-chain look-ahead (`begin_decode`, `quiesce`) and reads device argmax ids |
| Block verification | `RoutedBlockSession::verify(tokens) → logits per position` | `crates/larql-inference/src/vindex3/routed_block.rs:113` | **Yes**: one multi-row frame per contacted worker per routed layer (BLOCK-CHECK, BLOCK-TIME) |
| Accepted-prefix commit and rejected-suffix rollback | `RoutedBlockSession::commit(count)`, through the continuation's own `truncate` | `routed_block.rs:156` | **Yes**: tests show committing any prefix equals decoding only that prefix, and that chained verify plus partial commit leave no trace of rejected positions (`tests/routed_block.rs:199, 268`) |
| Session position | `RoutedBlockSession::position()` (committed only) | `routed_block.rs:103` | Yes |
| Single step (no proposal) | `DecodeSession::step(token)` over its own continuation | used by `v3_block_time`'s `sequential` arm | **No** on a block session: `RoutedBlockSession` owns its continuation and has no single-step method |
| Stop handling | `truncate_at_stop` | `.../lowered/run.rs` | Pure, `pub(super)`; not reachable |
| Remote dispatch accounting | `Timed` transport (route, rows, bytes, round trip per exchange) and `Recorder` (worker timing) | `crates/larql-cli/examples/v3_block_time.rs:61-155` | Yes, but inside one example |

The two sessions' contracts differ:

| | `LoweredSession` (local) | `RoutedBlockSession` (remote) |
|---|---|---|
| verify | returns argmax ids | returns logits; the caller takes the argmax |
| keep a prefix | `truncate(absolute position)` | `commit(count)`, relative, required before the next verify |
| single step | `step(id)` | none |

### Missing seams, and what this programme would add

1. **Move the proposer, not copy it.** `propose`, `accepted_prefix` and
   `truncate_at_stop` are pure. Move them to a shared module in
   `larql-inference` (proposed: `larql_inference::speculation`) and have
   the local Metal command import them from there, unchanged. The local
   path's behaviour is pinned by its existing tests.
2. **A single step on the block session.** Without one, every position
   with no proposal would run as a width-1 block over the multi-row route.
   BLOCK-TIME-1 measured that as about 7% slower per position than
   `sequential` (coordinator 44.8 against 35.3 ms per position at 0 ms). It
   would penalise the speculative arms for reasons unrelated to
   speculation. Proposed: `RoutedBlockSession::step(token) → logits`, which
   runs `DecodeSession::step` over the session's own continuation (the
   single-row route) and commits that one position. It needs tests that its
   logits and continuation equal the sequential session's, and that it
   refuses while a block is pending.
3. **A remote speculative loop.** This is a second loop, and the
   justification is specific: the local loop's structure is its Metal
   decode-chain pipelining (a committed look-ahead step collected by
   `quiesce`), which has no remote counterpart. The remote session reports
   logits, not device ids, and commits relatively. The **policy** is not
   duplicated: it calls the moved `propose`, `accepted_prefix` and
   `truncate_at_stop`, and keeps the same accounting (proposed, accepted,
   verify blocks, single steps, rewinds). Proposed home:
   `larql_inference::vindex3::speculation::greedy_speculate(session,
   prompt, first, n, width, stop)`. It is generic over the session only
   through `verify`, `commit`, `step` and `position`.
4. **A trial binary** (`examples/v3_verify_remote.rs`), sharing `Timed`
   and `Recorder` with `v3_block_time` through one examples support module
   rather than a copy. It records per-exchange routes and sizes, worker
   timing, every block's proposal, acceptance and rejection offset, and the
   generated ids.
5. **A driver** reusing BLOCK-TIME-2's runner, telemetry, stall watch,
   `PAUSE`/`PAUSED` hold, provenance, admission checks and quiet-machine
   handshake.

No second implementation of proposing, acceptance or rollback is
introduced. `RoutedBlockSession` remains the only owner of
verified-against-committed state on the remote path.

## Proposed protocol

### Fixed

- GPT-OSS 20B (`gpt-oss-20b.vindex3`), stored weights (production
  arithmetic); one CPU expert worker owning every routed layer; LINK-1's
  thread settings.
- The frozen relay `7b1097f8…` at unlimited rate, at **0, 5 and 20 ms**.
- Greedy decoding; ties broken to the lowest id, identically in every arm.

### Arms

| Arm | What it does |
|---|---|
| `sequential` | Greedy generation through `DecodeSession::step`: the exact reference and the baseline |
| `verify-2`, `verify-4`, `verify-8` | `greedy_speculate` with blocks of at most 2, 4 or 8 positions (the fed token plus up to 1, 3 or 7 proposed) |

### Workloads

Four prompts, committed as text and token ids before any run. Neither the
selection nor any prompt is changed after a run.

- **Copy-heavy**: two edit-style prompts whose answers repeat much of the
  prompt: a JSON edit and a code docstring transform. Proposed: the two
  prompts #729 measured locally, recovered from that run's archive. If the
  originals cannot be recovered, two new prompts of the same kinds are
  written and committed blind.
- **Low-copy**: two ordinary prose prompts (an explanation and an
  open-ended paragraph), where prompt lookup is expected to propose
  little.

**Copy-heavy and low-copy results are reported separately and never
aggregated into one figure.**

### Measure

`committed new tokens per second = new ids generated ÷ wall time of
generation`

This is timed from the first generated position to the last. It includes
proposing, every verify block, every single step, every commit or rollback,
argmax, and coordinator work. **Prompt prefill is excluded, identically in
every arm.** It is done untimed, by the same method in every arm (proposed:
block verification at width 8 and commit, which BLOCK-CHECK showed is exact).

Per trial, recorded:
- generated ids, and blocks with their proposal length, accepted count and
  rejection offset;
- single steps;
- exchanges by route with rows and bytes;
- worker execution per exchange;
- the decomposition used in BLOCK-TIME.

### Correctness gates

1. **Exact greedy identity.** Every arm's generated ids equal the
   `sequential` arm's for the same workload, at every delay, through any
   stop id. Any mismatch stops the run. A mismatch is a correctness
   finding, reported with the first differing position.
2. **Acceptance accounting.** The trial's counts must reconcile: generated
   ids = 1 (the prompt's own greedy id, from prefill) + accepted proposals +
   verify blocks (each adds the target's id after its accepted prefix) +
   single steps, less any ids after a stop. Proposed, accepted, rejected and committed positions are
   reported separately; verified positions are never counted as generated
   tokens.
3. **Remote-execution integrity.**
   - every verify block of `k` positions sends exactly one multi-row frame
     of `k` rows per routed layer (24), and no single-row frame;
   - every single step sends exactly 24 single-row frames;
   - no other exchange occurs, and the one configured worker is the one
     contacted.

   A trial that fails this is void by name. It shows a silent local or
   single-row fallback.
4. **Rollback.** This is covered by tests before any run (see the testing
   plan), and by gate 1 during it: a rollback that left rejected state
   would change later ids.

The relay-schedule gate (LINK-1's), BLOCK-TIME-2's stall watch, and the
quiet-machine and handshake rules apply unchanged.

### Schedule

- Warm-up: BLOCK-TIME's rule (`sequential` at 0 ms until two trials agree
  within 1%, at most four).
- **Units.** One unit is one workload at one delay: the four arms in a
  seeded order, opened and closed by a short `sequential` 0 ms bracket.
  That is 4 workloads × 3 delays = **12 units per repeat, two repeats**, in
  a seeded order balanced so every workload and delay appears once per
  block of units.
- **Generation length**: 64 new ids, or until a stop id.
- **Validity**: a unit is void if its brackets differ by more than 1%, or
  if any of its trials fails gate 3, the relay gate or the stall rule.
  Void units are kept and never rerun. Only a gate 1 failure stops the run.
- A within-unit cost-stability rule like BLOCK-TIME-2's rule 4 is **open
  for review**. The arms here do different work by design, so coordinator
  time is not expected to match between them. A candidate is worker
  execution per verified position.

### Statements, fixed in advance

For each workload, delay and width, with both repeats' units valid:

- **"Faster than sequential"**: committed new tokens per second higher than
  `sequential`'s by more than 1% in both repeats. **"Slower"**: lower by
  more than 1% in both. Otherwise **no difference shown**.
- The ratio is reported per repeat, along with the mean accepted length per
  block, the share of positions verified in blocks, and the rejection
  offsets.
- A width that proposes nothing on a workload is reported as such, with
  its tokens per second. It is not dropped.
- A missing cell (no valid unit in a repeat) is reported as missing.
- **Descriptive only:** the speed the BLOCK-TIME exchange model predicts
  from the observed acceptance, against the measured speed.
- **Never claimed:** sampled-output exactness, model-drafter behaviour,
  physical-network performance, or an aggregate across copy-heavy and
  low-copy workloads.

### Relay scope

The block-frame qualification covers frames of widths 1, 2, 4 and 8.
Speculative blocks also run at widths 3, 5, 6 and 7, whose frame sizes fall
between qualified sizes. Those widths are **interpolated, not qualified**,
and the results say so. The largest qualified error (−96 µs per exchange at
width 8 and 5 ms) is about 0.1 ms per exchange.

### Budget (estimate, to be confirmed by the diagnostic)

Per trial: prefill of a prompt of about 150 tokens at width 8, untimed
(about 20 s at 0 ms, about 30 s at 20 ms), plus 64 generated ids. At
20 ms, `sequential` takes about 40 s for those; speculative arms take
less in proportion to acceptance. About 100 trials, plus brackets, comes
to roughly **90 to 120 minutes**. The run can be held between units.

## Blockers before any model run

1. Seams 1 to 5 built, with tests, reviewed and merged through the normal
   ruleset.
2. Workload prompts recovered or written, and committed as text and ids.
3. The within-unit cost-stability rule decided.
4. One declared functional diagnostic (busy host allowed, never counted)
   showing gates 1 to 3 working end to end.
5. Independent review of this protocol and the implementation, then a
   freeze with hashes in `frozen-protocols.json`.
6. A quiet window agreed with the WSP-002 session, which has priority.

## Testing plan

**Rust, `larql-inference`, against the existing routed-block fixture (fake
workers, no model):**
- `greedy_speculate` generates exactly the ids of sequential greedy
  decoding for widths 2, 4 and 8, on prompts built so proposals are
  always, sometimes and never accepted;
- **corrupted proposals** (negative control): a proposer wrapper that
  changes one token of every proposal, at the first, a middle and the last
  proposed position. The verifier rejects at exactly that offset, the
  committed ids still equal the sequential reference, and acceptance falls
  as expected;
- **rejection at every offset** 0 to `W - 1`, after which the next block's
  logits equal a fresh sequential session that consumed only the committed
  prefix (KV state carries nothing rejected);
- a stop id inside an accepted block ends generation there, and ids after
  it are not emitted;
- the generation limit and the session's position limit truncate a
  proposal and never overrun;
- `RoutedBlockSession::step`: logits and continuation equal the sequential
  session's; it refuses while a block is pending; a failed step
  invalidates the session;
- **frame accounting** through the fixture's transport: each block of `k`
  positions produces one `k`-row frame per routed layer, each single step
  one single-row frame per layer, and nothing else.

**The local path, unchanged:** the existing `prompt_lookup` and lowered
speculation tests keep passing after the move. CI on the Metal path runs
them.

**Python, bench folder (CI's protocol workflow):** schedule shape, seed and
balance pinned; unit validity; reconciliation of the acceptance accounting;
the "faster/slower/no difference" rule at, below and above 1%; missing and
zero-proposal reporting; and an end-to-end `execute()` with fake trials,
as for BLOCK-TIME-2.

**Before the admissible run:** the functional diagnostic, including one
corrupted-proposer control trial per width on one copy-heavy workload at
0 ms. Its ids must equal the reference, and its acceptance must fall. The
control's timing is not a result.
