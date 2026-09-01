# Compiling a VINDEX3 container from a repo you never download

`larql vindex3 plan` and `larql vindex3 encode` accept an `hf://` artifact
alongside a checkpoint directory:

```bash
larql vindex3 plan   hf://zai-org/GLM-5.3-Flash
larql vindex3 encode hf://ibm-granite/granite-4.2-3b --output granite.vindex3
```

Neither downloads the checkpoint. `plan` reads safetensors *headers*;
`encode` additionally reads the byte range of each tensor it actually
binds, and nothing else. The canonical BF16 weights are never on the local
disk, in whole or in part.

Measured against the real hub:

```text
hf://zai-org/GLM-5.3-Flash
  staged 39.39 MB (10.68 MB of headers over 62 shard(s), 28.71 MB of metadata)
    standing in for 328.33 GB (305.78 GiB) of tensor payload
  plan: 78 representable, 1 mismatched, 33 unrepresented — 32 blocking
```

That verdict is the same one the weights would produce. Deciding whether a
328 GB checkpoint is worth converting now costs forty megabytes — a ratio
of about 8,300:1.

Headers and metadata are quoted separately on purpose. They scale
differently: header bytes track the tensor count, while a tokenizer is
whatever size it is — GLM's is 19 MB, nearly twice every shard header put
together. Quoting the header figure alone would understate the transfer
fourfold, and the point of the line is that the claim is checkable.

GB and GiB are both given at GB scale and only there. That is where a 7%
difference is large enough that two correct figures side by side read as a
bug in one of them.

## `metadata.total_size` is not the payload total

The payload census comes from the staged **headers**, never from the shard
index's `metadata.total_size`. The two disagree whenever the source model
tied weights: HF computes `total_size` from *deduplicated parameter
storage*, so it declares a tied embedding once while the file serialises it
twice.

granite-4.2-3b really does this. `lm_head.weight` and
`model.embed_tokens.weight` are two physically distinct 513,802,240-byte
regions — offsets `[0, 513802240)` and `[513802240, 1027604480)` of shard
one — and the index declares 6,805,672,960 bytes against headers summing to
7,319,475,200. Short by exactly one of them.

```text
staged 7.26 MB (0.04 MB of headers over 2 shard(s), 7.21 MB of metadata)
  standing in for 7.32 GB (6.82 GiB) of tensor payload
  note: the shard index declares 6.81 GB (6.34 GiB) — 513.80 MB less than
        its own headers sum to (tied weights counted once there, serialised
        twice here); the header sum is what transfers
```

The headers are what the file holds and what a range-read encode will
transfer, so they are the authority; the index's declaration is reported
alongside when it disagrees, because a silent 7% gap between "standing in
for" and "fetched" reads like a units bug. GLM-5.3-Flash ties nothing and
its two figures agree exactly, which is why the note does not appear there.

**This retires a self-check.** "The inventory's `total_bytes` equals the
index's declared `total_size` exactly" was used as evidence that a
header-only stub faithfully stands in for a checkpoint. It holds only when
nothing is tied; on granite it fails by construction, and applying it would
reject a perfectly good stub.

## The two steps, and why they are separate

```text
hf://org/name[@revision]
        │
        ├── resolve revision → commit sha          one request
        │
        ├── stage headers                          config.json, the shard
        │     ~39 MB for a 328 GB model            index, tokenizer files,
        │                                          and <u64 len><header>
        │                                          per shard
        │
        ├── build_inventory  ─┐
        ├── plan_system       ├── ADMISSION — headers only, no payload
        ├── capability gate  ─┘
        │
        │   refused? stop here. Nothing was transferred.
        │
        └── encode                                 GET Range: bytes=… per
              per tensor: range → segment          tensor the plan binds
              source bytes dropped immediately
```

Admission runs first and runs on headers, which is the whole reason this
works: an inadmissible model costs its headers and not one tensor byte.
`larql vindex3 encode hf://Qwen/Qwen2.5-0.5B --capability text-generation`
refuses with four blocking findings having transferred 0.03 MB.

### The stub is also the payload manifest

A safetensors header states every payload's offset and length, so the
staged header-only checkpoint *is* the offset table. `RemoteArtifactSource`
indexes it with the same `index_shard_header` the local source uses —
there is **one** implementation of the offset arithmetic, not two that
agree.

Staged headers and metadata are cached under
`~/.cache/larql/hf-headers/{owner}--{name}/{commit}/` (honouring
`LARQL_HOME`), keyed by the *resolved commit*, never by the branch name.
Because the key is a commit, nothing under it can have changed, so a
second run re-fetches neither the headers nor the tokenizer. Headers read at one commit and payloads read at another would
address a different checkpoint with the same offsets — the one failure
this path could have that still produces plausible bytes.

## What is gated

`crates/larql-vindex/src/format/vindex3/encode/source/tests/remote.rs`
encodes one fixture checkpoint twice — once from the directory, once from a
mock repo serving that same directory by byte range — and requires the
segments and the system graph to be **byte-identical**. Shifting the remote
source's offset by one byte makes it fail, which is what makes it a gate
rather than a decoration.

Its anti-vacuity check proves the remote arm could not have read the local
directory: the staged shard is strictly smaller than the real one, so the
payload bytes are not reachable from where it read.

### Two controls, because a range read fails quietly

| Failure | What the caller sees | What we do |
| --- | --- | --- |
| Host ignores `Range:`, answers `200` with the whole file | Bytes arrive; the first N are a plausible tensor from the wrong offset | Refuse — only `206 Partial Content` is accepted |
| Host rate-limits, answers `206` with a short body | HTTP succeeded | Retry with backoff, then refuse naming what was measured |

`scripts/hf_metadata_checkpoint.py` — the Python precursor this path
replaces for the LARQL flow — learned the second the hard way and cannot
detect the first at all, because `curl -sL` hides the status.

## What the separation buys, measured

Planning is now cheap enough to be used as an instrument. Six sizes of one
family, staged from headers only — a 44× span in payload:

| model | payload | representable | blocking | text-gen required |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1.50 GB | 37 | 3 | 40 |
| Qwen3-1.7B | 4.06 GB | 37 | 3 | 40 |
| Qwen3-4B | 8.04 GB | **36** | 3 | 39 |
| Qwen3-8B | 16.38 GB | 37 | 3 | 40 |
| Qwen3-14B | 29.54 GB | 37 | 3 | 40 |
| Qwen3-32B | 65.52 GB | 37 | 3 | 40 |

**The blocking set is identical at every size** — the same three findings,
verbatim: `max_window_layers` and `use_sliding_window` declared and read by
nothing, and one `sliding_window` execution-semantic rule. What VINDEX3
must learn in order to support Qwen3 does not depend on which Qwen3.
Semantic bring-up is size-invariant, and that is measured rather than
assumed.

The **physical inventory is not**. Qwen3-4B is missing `target.output_head`
— and not for the reason the config suggests:

| model | `tie_word_embeddings` | `lm_head.weight` in the shards |
| --- | --- | --- |
| 0.6B | `True` | yes |
| 1.7B | `True` | yes |
| 4B | `True` | **no** |
| 8B / 14B / 32B | `False` | yes |

Same family, same flag, different serialisation. 0.6B and 1.7B declare
tying and ship the tensor anyway; 4B declares tying and omits it. Nothing
in `config.json` distinguishes them.

### The same lesson three times in one day

```text
declared                              physical
─────────────────────────────────     ────────────────────────────────
granite  metadata.total_size      ≠   sum of header data_offsets
Qwen3    tie_word_embeddings      ≠   whether lm_head.weight exists
HF index weight_map dedup         ≠   what a range read must address
```

For range-backed work only the right-hand column is addressable, and only
the headers report it. This is the empirical case for keeping **model
authority**, **architecture interpretation** and **physical
representation** as separate stages: upstream collapses them into one
repository, and the three columns above are what that collapse hides.

```text
canonical model
      │  authority — what the model IS, pinned to a commit
      ▼
semantic model graph
      │  interpretation — how VINDEX3 understands it (size-invariant)
      ▼
execution-specific representation
      │  derivation — what this machine should execute
      ▼
remote / cached / mapped / resident
         residency — where those bytes currently are
```

Each arrow is a place where a declared fact and a physical fact can
disagree. Each stage has to be able to say which one it trusts.

## Two meanings of "streaming", and this is the first one

Worth keeping distinct in VINDEX vocabulary:

| | What streams | What it solves |
| --- | --- | --- |
| **A. From source** *(built)* | BF16 → tensor at a time → VINDEX3 | **Storage.** A 328 GB model converts on a laptop with nothing like 328 GB of scratch. |
| **B. During execution** *(next)* | operand fetched if and when required | **Transfer/residency.** You receive the representation you execute, not the authority it derives from. |

A solves storage, not transfer: if every tensor is represented, roughly
every tensor still crosses the network once.

### The decision for B: promote before `prepare`

A prepared model must **not** secretly fault remote operands into
existence. The invariant to keep is:

> Prepared execution means every operand the execution plan requires has a
> stable execution residency.

```text
PLAN → select representation → HYDRATE required operands → PREPARE → RUN
```

not

```text
PREPARE → RUN → operand missing → HTTP → continue
```

`OperandStore`'s load counter and `opplan/exec/tests/residency.rs` — "a
served model's operands are lowered once" — are not an obstacle here. They
are a statement about what `prepare` means, and it is worth preserving. The
user still experiences streaming; the execution model stays clean.

Three axes that must stay separate:

```text
SOURCE                  RESIDENCY               EXECUTION
where authority         where its bytes         what execution
originates              currently live          consumes

HF repo ──────────┐
                  │     Remote      ─┐
local artifact ───┼──→  LocalCached  ├──→  Prepared
                  │     LocalMapped  │
VINDEX repo ──────┘     Resident    ─┘
```

Note that `RepresentationSource {Auto, Stored, Transient}` is **none of
these**. It says whether the runtime may *manufacture* a representation at
load, and `Transient` is a permanently retained oracle that a network
fallback would destroy. Residency needs its own axis.

Demand paging is not forbidden — it is a *different, explicit* contract
(`FaultableRemote` or similar), the only class allowed to cross the network
after preparation, with its own accounting (`remote_fetches`,
`remote_bytes`, `cache_hits`, `promotions`, `evictions`) rather than
pretending those events are ordinary `OperandStore` loads.

## Ladder

```text
[x] HF metadata/header-only planning
[x] immutable revision pinning
[x] remote tensor Range source
[x] streamed HF → local VINDEX3 encode
[x] no source checkpoint materialisation
[x] byte-identical local/remote parity
[x] admission before payload transfer

[ ] 2a  remote VINDEX3 representation source; hydrate execution set;
        prepare exactly once; execute identically
[ ] 2b  explicit FaultableRemote contract with separate accounting
[ ] 3   persistent content-addressed operand cache
[ ] 4   derived representation cache — REPRESENT transforms the streamed
        tensor so canonical BF16 exists only transiently in memory, then
        derivation identity (revision + programme + ABI + layout) decides
        whether the derived object can be streamed instead of the BF16
[ ] 5   sparse/on-demand remote execution
```
