# Extend LARQL with plugins

**Class: CURRENT.** [Representation](representation.md) ·
[Execution](execution.md) · [Status](status.md).

A plugin is a shared library (`.dylib` / `.so`) that adds what this build
does not ship. It can register four kinds of thing:

| Kind | Trait | What it adds |
|---|---|---|
| Codec | `RepresentationCodec` | Reads a stored encoding: a new `dtype` label a container may hold |
| Encoder | `RepresentationEncoder` | Writes that encoding, so `represent --encoding <LABEL>` can compile a pack in it |
| Lowering provider | `PlanBackend` | Executes a planned graph: select a realization per operand, run projections, possibly over the codec's own bytes |
| Continuation provider | `ContinuationFactory` | Holds a conversation's state between steps: K/V rows, recurrent buffers, latent rows |

A plugin is named on the command line and loaded for that command only.
Nothing is discovered: a library not passed with `--plugin` is never opened.

## What loading guarantees, and what it does not

Rust has no stable ABI, so a plugin and the `larql` binary exchange Rust
trait objects only when both were compiled **by the same compiler from the
same `larql-vindex` commit**. Every plugin exports a C-ABI stamp,

```text
larql-plugin/2 larql-vindex/<version> (<rustc version>) commit <sha>
```

and the host compares it with its own before calling anything Rust-typed. A
mismatch, or either side built from an unknown commit, is refused with both
stamps in the message. Uncommitted edits to the same commit are not detected:
build the plugin and the host from one checkout.

- Libraries are opened `RTLD_LOCAL` and never unloaded; registrations hold
  their vtables and `&'static` labels for the life of the process.
- A plugin statically links its own copy of `larql-vindex`. Process-wide
  state inside it (a `OnceLock` registry, a ledger) is the plugin's copy, not
  the host's, so a plugin reports through the values it returns, not globals.
- A registered codec label or lowering identity that clashes with a shipped
  one, or with another plugin's, is refused as a duplicate, never a
  replacement.
- Loading is implemented for Unix `dlopen`; elsewhere `--plugin` refuses by
  name.

## Write a plugin

A plugin is an ordinary crate with a `cdylib` target that depends on
`larql-vindex` from the checkout the host is built from:

```toml
[lib]
crate-type = ["rlib", "cdylib"]

[dependencies]
larql-vindex = { path = "../larql/crates/larql-vindex" }
```

It exports one registration function through the `larql_plugin!` macro,
which also exports the ABI stamp:

```rust
use larql_vindex::format::vindex3::plugin::PluginRegistrar;

fn register(r: &mut PluginRegistrar) {
    r.codec(Box::new(MyCodec));    // read MY_CODEC packs
    r.encoder(Box::new(MyCodec));  // compile MY_CODEC packs (optional)
    r.lowering(|| Box::new(MyProvider::new())); // execute them (optional)
    r.continuation(Box::new(MyStateFactory));   // hold conversation state (optional)
}

larql_vindex::larql_plugin!(register);
```

Registering an encoder does not register its codec. A plugin whose packs must
also be read registers both, as above.

### Codecs and encoders

The codec contract is [`represent-codec-contract.md`](../represent-codec-contract.md):
a codec names its label and identity (`CodecIdentity`: family, revision,
geometry), its streams, capabilities and extents, and decodes rows to the
canonical f32 surface. An encoder adds one method,

```rust
fn encode_packed(&self, values: &[f32], shape: &[usize],
                 extent: RepresentationExtent, tensor: &str)
    -> Result<Vec<u8>, CodecError>;
```

whose bytes, handed back to `decode_packed`, must decode to what this codec's
own `decode_rows` produces. A lossy encoder need not invert its input; it must
be stable under its own round trip.

An encoder may also implement `encode_packed_weighted(…, input_weights: &[f64])`,
minimising `Σ_rows Σ_i input_weights[i]·(w − ŵ)²` with one weight per input
feature; `represent --moments` calls it. The default refuses
(`WeightingUnsupported`), so an encoder that cannot honour weights never
ignores them silently. Weighting is an encoder choice: the bytes decode like
any other pack of that codec, and the recipe (`codec-encoder-weighted`)
records the weights' digest.

When `represent` compiles with an encoder:

- **Eligibility** is the encoder's `stored_bytes(shape, …)`. An error means
  the shape cannot hold the encoding and the tensor is carried at source
  precision, as a K-quant row that does not divide is.
- **Length.** A codec that can state a tensor's length from its shape must
  produce exactly that many bytes. A codec whose length depends on the values
  answers `CodecError::InstanceSized` and is encoded while planning, because
  the segment table needs every length before the payload is written; that
  object's encoded bytes are held in memory until its segment is written.
- **Provenance.** The pack's directory entry records the codec's
  `CodecIdentity` and an encoder recipe of `codec-encoder`, with the codec's
  `family/rN` as its source. LARQL never claims it can reproduce those bytes
  itself.

### Lowering providers

A lowering provider is a `PlanBackend` registered under a
`LoweringIdentity` (`family/vN`). Its `select` chooses, per operand, a
realization: decode to f32 and run a shipped kernel, or a direct realization
the codec declares. A provider that runs its own kernel over the stored bytes
asks for `WeightFormat::CodecOwned` and receives
`WeightSlice::CodecOwned { bytes, label }`: the container's raw payload and
its label, uninterpreted. LARQL records such a plan's arithmetic as
`CODEC_OWNED`. It does not know whether the provider quantises the activation
or accumulates in integers, so the provider must state that in its own
documentation.

A provider may delegate everything it does not specialise to a shipped
backend. The worked example is `crates/larql-vindex/tests/external_lowering_provider/`,
which registers a provider this build does not ship and executes it through
registration alone.

### Continuation providers

A continuation provider is a `ContinuationFactory` registered under a
`ContinuationIdentity` (`family/vN`). It declares the continuation regions it
can hold (`Kv`, `LatentKv`, `Recurrent`, `KvAndRecurrent`), validates its
`key=value` options, and builds one conversation's state. Selection matches the
plan's geometry against the declared regions and refuses before prefill,
naming the first layer the provider cannot hold. State is sealed with the
provider's identity and a digest of its configuration, and resumes only under
both (CONTINUATION-PLUGIN-1).

Loaded continuation providers join the shipped ones (`row/v1`,
`canonical/v1`) in the one registry the run selects from. A clashing identity
is refused as a duplicate. The worked example is
`crates/larql-kv/tests/external_continuation_provider/`; the same provider,
built as a plugin, is `crates/larql-continuation-fixture/`, which exists only
for the CLI's plugin gate.

## Use a plugin

Three commands take `--plugin <PATH>` (repeatable):

| Command | What the plugin supplies |
|---|---|
| `larql vindex3 represent` | Encoders that `--encoding` may name, alongside `NVFP4` and the K-quants |
| `larql vindex3 exec` | Codecs to decode the container, lowering providers `--lowering` may select |
| `larql vindex3 measure` | The same, for both arms |
| `larql run` (VINDEX3) | Codecs and lowering providers as for `exec`, and continuation providers `--continuation` may select |

Each loaded library reports what it registered on stderr:

```text
plugin: libmy_codec.dylib — codecs [MY_CODEC], encoders [MY_CODEC], lowerings [my-provider/v1], continuations []
```

### Compile a pack

```bash
larql vindex3 represent model.vindex3 --output model-mycodec.vindex3 \
  --encoding MY_CODEC --deployment --plugin ./libmy_codec.dylib
```

The role policy, `--protect`, `--protect-layers`, `--include-role`,
`--object` and `--deployment` apply exactly as for a shipped encoding; see
[Representation](representation.md). The graph marks the new representation
`approximate`.

### Execute it

A backend names the stored representation it asks for (`production-nvfp4`
asks for `NVFP4`). A plugin's encoding has no backend of its own, so name it:

```bash
larql vindex3 exec model-mycodec.vindex3 --backend production \
  --plugin ./libmy_codec.dylib \
  --representation MY_CODEC \
  --lowering my-provider/v1 \
  --tokens 1,2,3 --generate 8
```

- `--representation <ENCODING>` asks for that stored representation instead
  of the backend's. The container must hold it; nothing is manufactured under
  this name.
- `--lowering <family/vN>` executes on that provider instead of the one
  `--backend` names. Without it, the pack is decoded through the plugin's
  codec and run on the backend's own kernels.
- A lowered Metal backend (`metal-lowered*`) executes its own formats, not
  through a lowering provider, and refuses both flags.

### Hold state with a plugin's continuation provider

```bash
larql run model.vindex3 "The capital of France is" \
  --plugin ./libmy_state.dylib \
  --continuation my-state/v1 --continuation-option bits=4 --verbose
```

- `--continuation <family/vN>` holds continuation state with that provider
  instead of the one `--engine` names. Giving both is refused.
- `--continuation-option key=value` (repeatable) configures it. The provider
  accepts or refuses each option before anything runs. An option without
  `--continuation` is refused.
- `--verbose` reports the identity the run resolved (`continuation=my-state/v1`).

### Measure it

`measure` loads `--plugin` once for both arms and takes each override per arm:
`--reference-lowering` / `--candidate-lowering` and
`--reference-representation` / `--candidate-representation`. An arm run on a
plugin provider is recorded as `<backend>@<family/vN>`.

```bash
larql vindex3 measure \
  --reference model.vindex3 --reference-backend production \
  --candidate model-mycodec.vindex3 --candidate-backend production \
  --candidate-representation MY_CODEC \
  --plugin ./libmy_codec.dylib --candidate-lowering my-provider/v1 \
  --bank bank/ --sequences 69 --label mycodec --output out/mycodec
```

The procedure's admissibility checks (null arm, changed variable, physical
attribution, seals) apply unchanged; a plugin arm is evidence on the same
terms as a shipped one. See [Representation](representation.md#measure-a-representation).
