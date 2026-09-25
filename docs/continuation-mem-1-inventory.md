# Continuation state: what exists and what is read

**Status: reconnaissance (code reading only), not measurement.** Read
against main `0aae2cea`, where CONTINUATION-PLUGIN-1 C6 (#545) merged and
the CONTINUATION-PLUGIN-1 merge train ended. It is the evidence CONTINUATION-MEM-1 is
frozen against: every claim below is a statement about the code, with its
location, that the measurement is designed to confirm or refute in bytes.
The forecast is
[`continuation-mem-1.json`](represent/forecasts/continuation-mem-1.json).

CONTINUATION-PLUGIN-1 made the continuation plane open. This programme asks
what the state behind that seam costs, before VIEW-1 changes the read
contract. The question is: what continuation bytes exist, which are read,
when, and what does the current API force into existence that nothing
needs?

## The read contract

`ContinuationProvider::keys(layer)` and `values(layer)` return
`&[Vec<f32>]`: every row appended to the layer, from position 0, each its
own heap allocation (`opplan/exec/kv.rs`). A provider must hold every row
(`kv.rs:21-26`). All rows are `f32`, and a row's length is
`kv_dim = num_kv_heads * head_dim` (`continuation.rs:258-261`). No `bf16`
or `f16` appears anywhere on the continuation path.

## Storage, by provider

| Provider | K/V rows | Copies per appended row | Stored twice |
|---|---|---|---|
| `row/v1` (`RowKvState`, `kv.rs:285-368`) | `Vec<Vec<f32>>` per layer; the backend's row `Vec` is moved in | 0 | no |
| `canonical/v1` (`CanonicalKvState`, `larql-kv/src/vindex3/mod.rs:88-340`) | an `Array2<f32>` matrix per layer (the storage authority) **and** a `Vec<Vec<f32>>` view per layer | 2 per K and 2 per V: `push_row` into the matrix (`:313`), then `row(last).to_vec()` into the view (`:322`) | **yes** |

The view exists only because the read contract returns `&[Vec<f32>]`,
which a matrix cannot lend. `push_row` grows the matrix geometrically, so
it also carries up to about 2× capacity slack, and it periodically
reallocates and copies the whole matrix. Nothing checks that the two
copies agree: `append` is the only writer, and it fills the view from the
matrix.

**The server runs `canonical/v1`** (`larql-server/src/vindex3.rs:295-304`).
**The CLI's default is `row/v1`** (`larql-cli/.../continuation.rs:29-35`).
The two paths hold different amounts of memory for the same conversation.

Recurrent buffers (`RecurrentState`) are allocated once, by
`prepare_continuation`, and updated in place. Latent rows (`LatentKvRows`)
are a `Vec<Vec<f32>>` that moves each appended row in.

## Reads, by operator

| Operator | Phase | What it takes from the provider | Copied? |
|---|---|---|---|
| Softmax attention | decode (`decode.rs:855-856`), resumed prefill (`mod.rs:2334-2335`) | the full `&[Vec<f32>]`; the kernel reads `start..=position` | borrowed |
| Softmax, sliding window | same | the full slice, from which the kernel reads only the last `window` rows (`production.rs:694-705`, `source_start`) | borrowed. **Rows before the window stay resident and are never read again.** |
| ConvQkv (Mamba2Attention) | decode (`decode.rs:821-822`), prefill (`mod.rs:1441-1442`) | **every past row, copied with `to_vec()`**, only to release the borrow before `recurrent_state` | **deep copy, every step** |
| MLA | every call (`mla.rs:309-345`) | every latent row. Each call re-normalises it and re-runs `kv_b_proj` on it, and clones the new row on append (`mla.rs:310`). Batched prefill calls it once per position (`mod.rs:1412-1426`). | re-derived each call; prefill is quadratic |
| GatedDelta, Mamba2, KDA | every call | `&mut` recurrent buffers, updated in place; each copies its small conv history out first (`gated_delta.rs:602`, `mamba2.rs:188`, `conv_qkv.rs:142`) | small |

Batched prefill appends one moved row per position per layer
(`mod.rs:1466-1477`). Resumed prefill (the server's chained turns) runs
its suffix one position at a time (`mod.rs:1478-1490`).

## Handoff and resume

State is moved at every step, never copied or serialized:
`ContinuationHandoff` has no `Clone`, `resume` returns or drops it,
`V3KvHandoff` moves it, and `ResponseKvCache` moves it in and out
(take-once). Only `absorbed_ids` is rebuilt per turn.

## Instruments that exist, and what they cannot see

- `exec/accounting.rs` (`ResourceLedger`) prices weights and operands.
  It never sees continuation state.
- `exec/cpu/ledger.rs` counts projection weight bytes. The attention site
  is tagged, but K/V reads are not counted.
- `exec/timing.rs` `OpClass::AttentionCore` records time only.
- The size formulas `LayerContinuationGeometry::elements_at` and
  `RecurrentGeometry::bytes` are never called at runtime.
- The server counts hits, misses, resumptions, refusals and reused
  tokens, with no bytes.

**No instrument sees continuation bytes.** MEM-1 has to build one.

## Surprises, stated as hypotheses for MEM-1 to price

1. `canonical/v1` holds about 2 to 3 times the K/V bytes `row/v1` does,
   for identical output. The extra is the view the read contract needs,
   plus matrix capacity slack.
2. ConvQkv copies all past K/V rows on every step: quadratic bytes over a
   decode, purely to satisfy the borrow checker.
3. MLA re-derives every latent row on every call: O(n) matvecs per decode
   step, O(n²) for prefill.
4. Sliding layers hold every row forever. Attended bytes are bounded by
   the window; resident bytes are not.
5. Resumed prefill runs one position at a time, not batched.
