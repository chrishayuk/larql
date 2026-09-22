# Execute a VINDEX3 artifact

**Class: CURRENT.** [Command inventory](../generated/current-facts.md).

Build the source checkout with the pinned Rust toolchain:

```bash
cargo build --release -p vindex-cli -p larql-cli
```

On Linux or Windows add `--no-default-features` to avoid the default Metal
backend. Examples below use a local Hugging Face checkpoint directory and
an admitted text component; replace the paths with your own.

```bash
vindex plan /path/to/checkpoint --json
vindex encode /path/to/checkpoint --output model.vindex3
vindex inspect model.vindex3 --json
vindex verify model.vindex3
larql vindex3 ops model.vindex3
larql vindex3 exec model.vindex3 --backend production --tokens 1,2,3 --generate 8
larql vindex3 observe model.vindex3 --backend production \
  --prompt "The capital of France is" --record run.jsonl
```

Cargo puts the binaries in its configured release target directory; add that
directory to PATH or invoke the binaries there. Token IDs are model-specific;
the numeric example demonstrates the interface, not a portable prompt.
`observe --prompt` uses the container's tokenizer. `plan` and `encode` also
accept `hf://org/repo@revision`; remote access and gated repositories require
the appropriate network access and credentials.

`vindex verify` checks the artifact against its recorded hashes. It does not
establish HF forward parity or behavioral fidelity of an approximate pack.
The source-versus-container verifier, layer comparisons, and representation
quality instruments answer those separate questions.

Opening resolves a component program and representation policy. Preparation
binds operands and a numerical realization; a `DecodeSession` advances
continuation state. `LogitsSession` lets generation and serving consume logits
without reconstructing a V2 `ModelWeights` object. Backend support and
representation compatibility are checked at their respective boundaries.

`larql bench model.vindex3 --backends cpu,metal` times the serving path's
prefill and decode with the same statistic as a VINDEX2 bench row; see the
[CLI reference](../cli.md#larql-bench). Its numbers describe the machine and
backend they ran on and do not establish behavioral fidelity.

Use `larql serve model.vindex3 --port 8080` for the server surface; build
`larql-server` as well. HTTP state and API behavior are detailed in the
[runtime guide](../vindex3-runtime.md). To inspect execution evidence, continue
with [observation](observation-and-intervention.md).
