# Proposal: a batched, exact remote expert operation for VINDEX3

Status: **proposal only. Nothing here is implemented**, and no wire, provider
or scheduling change belongs to the DEC-V3-1 branch that carries this file.
It is separate from the link experiment
([LINK-1](../../bench/v3-routed-experts/LINK-1.md)), which measures the
existing single-row operation and should report first: whether latency,
bandwidth or per-call overhead limits that operation decides how much this
is worth building.

## Why

Every seam of the [routed expert provider](v3-routed-experts.md) carries one
row. `RoutedExpertProvider::apply` takes one input; a `VEX1` request carries
one hidden-width row; the coordinator fans out once per layer per position.
Two things the DEC programme wants therefore cannot happen on VINDEX3:

- **Speculative width.** Verifying a block of k drafted positions should cost
  one crossing per layer, not k. Today `--speculate` exists only on the Metal
  lowered path, remote placement requires CPU production lowering, and the
  frame has room for one row.
- **Expert-grouped execution.** When several rows select the same expert, the
  worker should read that expert's weights once. Today each row is its own
  request and its own read.

One primitive serves both: a remote operation over R rows, each with its own
selected experts, executed grouped by expert, answered per (row, expert).

## The operation

```text
request:   layer, R rows; per row: row id, selected expert ids, one input row
response:  one unweighted output row per (row id, expert id)
```

- **Row identity is the caller's.** The coordinator assigns each row an id
  and the response must answer every requested (row, expert) pair exactly
  once. Omitted, duplicated or unsolicited pairs refuse, as `VEY1` does for
  expert ids today.
- **The worker stays stateless and session-free.** It never learns which
  session or position a row came from. That is what lets the same frame carry
  k positions of one session or one position from each of several sessions.
- **No routing weights on the wire.** Routing, selection and weighted
  reduction stay in the coordinator.
- **Reduction is unchanged.** Each row's outputs are accumulated in that
  row's production selection order. Grouping changes which rows an expert
  kernel sees together, never the order of any row's sum.
- **Exact F32 carriers**, bind-once authority and worker-incarnation handles
  as today. A new frame version, not a change to `VEX1`/`VEY1`.

## Minimum changes, in dependency order

1. **Wire frame v2** in `larql-router-protocol`: the request and response
   above, with exact length, count, dimension and finiteness checks.
2. **Provider trait**: a rows form beside `RoutedExpertProvider::apply`, with
   the existing single-row call defined as its R = 1 case.
3. **Worker execution**: group the frame's pairs by expert and run each
   owned expert once over its rows.
4. **Invocation accounting**: worker counters of expert kernel invocations
   against (row, expert) pairs served. This counts grouping, not memory
   traffic: a kernel invoked once can still re-read its weights for every
   row. Physical reuse is a separate, later claim (below).
5. **Coordinator fan-out**: group a multi-row call's pairs by owning shard,
   dispatch in parallel as now, reassemble per row.

Items 1–5 are shared. Two further pieces are specific to one use each and
are **not** part of the first rung:

6. **Multi-position coordinator session** (speculative width only): verify
   and truncate on the CPU routed session, or a Metal lowered coordinator
   that accepts a remote expert provider. Neither exists. This is the largest
   piece of work and has its own correctness questions (causal attention
   within a block, acceptance, rollback).
7. **Cross-client scheduler** (shared tier only): something that merges
   frames from independent coordinators.

## Acceptance for the first rung (items 1–5, one client)

- R = 1 reproduces the existing single-row operation bit for bit.
- R > 1 reproduces the same rows executed one at a time, bit for bit.
- A response cannot omit, duplicate or substitute a (row, expert) identity.
- Worker counters show expert invocations below pairs served when rows share
  experts. This is reported as grouping only, never as bandwidth or memory
  saved.
- A frame succeeds or fails whole, and a failure invalidates the one
  coordinator session that sent it, as a failed step does today.

## Physical reuse is a separate rung

The first rung establishes numerical equivalence and grouping. It makes no
claim that any weight read was avoided. Three quantities are distinct and
must not be substituted for one another:

| Quantity | What it shows |
|---|---|
| Network requests per layer | Crossings, nothing about weights |
| Expert kernel invocations | Grouping happened |
| Weight bytes actually read from memory per pair served | Physical reuse |

Only the third supports a reuse claim, and it needs a kernel that
demonstrably makes one pass over an expert's weights for several rows
(a multi-row kernel), measured with suitable evidence: hardware memory
counters or a bandwidth-bound timing argument with controls. PROFILE-1
already notes that transform time alone cannot establish DRAM traffic. The
DEC funnel's rule that a union is only potential reuse applies here
unchanged.

## Open, deliberately deferred

**Failure isolation across clients.** Whole-frame atomicity is right for one
session. Applied to a frame that mixes sessions, one bad row would invalidate
unrelated sessions. Until that is designed, rows from different trust domains
are never placed in one frame. The design should wait for a proven
single-client implementation.

## What existing evidence does and does not say

- GPT-OSS 20B verify blocks of 8 positions touch 12.6–13.7 distinct experts
  of 32 slots (registry `gptoss-verify-n-locality`). That is a ceiling on
  logical reuse along one sequence, measured on the f32 reference tier.
- The cross-client batch-union figures in the DEC funnel were measured on a
  different model and the legacy stack. They motivate item 4; they are not a
  prediction for this operation.
- Nothing yet shows that grouped execution on the CPU worker turns shared
  experts into avoided reads, and item 4 will not show it either: it counts
  invocations. That question belongs to the physical-reuse rung.
