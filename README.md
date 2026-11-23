# LARQL

**Run, serve, query and edit AI models — and see what they're doing inside.**

LARQL is a model runtime written in Rust. Point it at a Hugging Face model and
you can chat with it, serve it over an OpenAI-compatible API, ask it questions
in a query language (LQL), record what happens inside it while it runs, and
experiment with edits — all from one file format, **VINDEX3**, that describes
the model as something you can execute *and* inspect.

```bash
larql run   hf://Qwen/Qwen3-0.6B "What is the capital of France?"
larql chat  Qwen3-0.6B
larql serve Qwen3-0.6B --port 8080
```

The first `run` downloads the model from Hugging Face and converts it to
VINDEX3 once; after that it's cached under its short name.

## Install

Download a prebuilt binary — no Rust toolchain needed:

```bash
# macOS (Apple Silicon)
curl -L https://github.com/chrishayuk/larql/releases/latest/download/larql-aarch64-apple-darwin.tar.gz | tar xz
sudo mv larql-aarch64-apple-darwin/larql* /usr/local/bin/

# Linux (x86_64). Needs the OpenBLAS runtime: sudo apt-get install -y libopenblas0
curl -L https://github.com/chrishayuk/larql/releases/latest/download/larql-x86_64-unknown-linux-gnu.tar.gz | tar xz
sudo mv larql-x86_64-unknown-linux-gnu/larql* /usr/local/bin/
```

Windows: download `larql-x86_64-pc-windows-msvc.zip` from the
[latest release](https://github.com/chrishayuk/larql/releases/latest).
Each archive contains `larql` and `larql-server` (used by `larql serve`).
GPU execution uses Metal on Apple Silicon. Linux and Windows builds run on the
CPU, and there is no CUDA backend.
Then run `larql doctor` to check the install, hardware, backends, caches and
Hugging Face access before the quick start.

<details>
<summary>Build from source instead</summary>

```bash
git clone https://github.com/chrishayuk/larql && cd larql
cargo build --release -p larql-cli -p larql-server   # Linux/Windows: add --no-default-features
export PATH="$PWD/target/release:$PATH"
```

The repository pins its Rust toolchain, so `cargo` picks the right version.
macOS builds use Metal; `--no-default-features` builds the portable CPU backend.
</details>

## Quick start

Full walkthrough with expected output: **[Getting started](docs/getting-started.md)**.

**1. Run a model.** Give `larql run` any supported Hugging Face model. The first
run converts it to VINDEX3 and caches it (Qwen3-0.6B is about 1.5 GB); later
runs start immediately, offline.

```console
$ larql run hf://Qwen/Qwen3-0.6B "What is the capital of France? Answer in one word. /no_think"
Qwen/Qwen3-0.6B is a Hugging Face checkpoint, not a vindex; planning a VINDEX3 encode (first run only)…
  ...
<think>

</think>

Paris
```

Prompts go through the model's chat template, so instruct models answer them.
Add `--raw` for plain text completion. `larql chat Qwen3-0.6B` starts an
interactive conversation. Or pull a ready-made VINDEX3 model:

```bash
larql pull granite-4.1-3b
larql run  granite-4.1-3b "Write a haiku about Rust."
```

`qwen3-0.6b`, `gemma-3-270m-it`, `gemma-3-1b-it`, `gemma-3-4b-it`,
`gemma-3-12b-it` and `gemma-4-12b-it` are ready to pull the same way.

**2. Serve it** over an OpenAI-compatible API:

```bash
larql serve Qwen3-0.6B --port 8080

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Write a haiku about Rust."}]}'
```

**3. Look inside it.** This is what most runtimes can't do. These commands
take the same model name:

```console
$ larql lql 'USE "Qwen3-0.6B"; INFER "The capital of France is" TOP 3;'
   1.  Paris               (65.71%)  [id 12095]
   2.  located             (2.78%)  [id 7407]
   3.  the                 (2.35%)  [id 279]
```

**At which layer does the model settle on "Paris"?** `observe` runs the prompt
and records what every layer writes. `--lens-tokens` adds a logit lens: at each
layer it reads the state through the model's own final norm and output head, and
reports where a token ranks among the 151,936 in the vocabulary:

```console
$ larql vindex3 observe Qwen3-0.6B --prompt "The capital of France is" \
    --lens-tokens 12095 --record run.jsonl
  lens: 140 head passes; position 4 by depth (token: logp rank; then top-1):
    L  0 Ffn  12095 " Paris": -18.081 #11876  top 1172 " only"
  ...
    L 19 Ffn  12095 " Paris": -15.878 #1049  top 2130 "____"
    L 20 Ffn  12095 " Paris": -10.384 #43  top 2130 "____"
    L 21 Ffn  12095 " Paris": -4.359 #9  top 2130 "____"
    L 22 Ffn  12095 " Paris": -0.596 #1  top 12095 " Paris"
  ...
    L 27 Ffn  12095 " Paris": -0.420 #1  top 12095 " Paris"
```

Through the first twenty of its 28 layers, "Paris" ranks below 1,000. It reaches
the top over layers 20 to 22. `run.jsonl` holds every layer's write with its
provenance and a verified receipt, ready to analyse or to load into the
[Observatory](observatory/README.md). `larql repl` opens an interactive LQL shell. `larql lql 'HELP;'` prints every
LQL statement with its syntax, and `larql lql --help` says how several
statements share one session and what the exit status means.

**4. Change it.** Write an edit to the stored weights as a `.vlp` patch, apply it
to a running session, and take it away again. The model on disk never changes:

```console
$ larql lql 'USE "gemma-3-4b-it";
    BEGIN PATCH "edit.vlp";
    UPDATE WEIGHTS SET DELTA = 4.0 WHERE TENSOR = "20.mlp.down_proj.weight" AND ROW = 7 AND COL IN (0, 1, 2, 3);
    SAVE PATCH;'

$ larql lql 'USE "gemma-3-4b-it";
    INFER "The capital of France is" TOP 3;
    APPLY PATCH "edit.vlp";
    INFER "The capital of France is" TOP 3;
    REMOVE PATCH "edit.vlp";
    INFER "The capital of France is" TOP 3;'
   1.  Paris               (81.11%)  [id 9079]
   2.  a                   (2.31%)  [id 496]
   3. :                    (2.04%)  [id 236787]
Applied: edit.vlp (... 4 stored-weight edit(s), bound to this container ...)
   1.  Paris               (80.96%)  [id 9079]
   2.  a                   (2.29%)  [id 496]
   3.  the                 (1.92%)  [id 506]
Removed patch #1
   1.  Paris               (81.11%)  [id 9079]
   2.  a                   (2.31%)  [id 496]
   3. :                    (2.04%)  [id 236787]
```

Removing the patch restores the original predictions exactly. The four-weight edit
is only an illustration of the mechanism. It isn't meant to improve anything.

**A real edit, with its evidence.** Asked for a raw JSON object with "no code
fence", Gemma 3 4B IT wraps it in a ```` ```json ```` fence anyway. In
[examples/gemma-json-neuron](examples/gemma-json-neuron/README.md), an AI agent
working through LQL repurposed one feed-forward neuron (7,676 stored weights, no
training) so the model answers with raw JSON when asked to, and still fences it
when a code block is requested. It was then scored on 48 prompts that were sealed
before the run:

| Sealed prompts | Unedited | With the patch |
| --- | --- | --- |
| Asking for raw JSON (24) | 3 pass | 16 pass |
| Asking for fenced JSON or a code block (24) | | every reply byte-identical to the unedited model |

Compiled and exported to GGUF, llama.cpp produces the same tokens as LARQL on 53
of the 54 prompts. The example's README has the patch, the commands to reproduce
it, and its limits. It doesn't fix every prompt: 8 of the 24 still fail.

Then make it permanent: compile the patch into a new container and export that
as GGUF for llama.cpp or Ollama.

```bash
larql lql 'USE "gemma-3-4b-it"; APPLY PATCH "edit.vlp"; COMPILE CURRENT INTO VINDEX "gemma-edited.vindex3";'
larql convert vindex-to-gguf gemma-edited.vindex3 gemma-edited.gguf --text-only
```

`CAPTURE` and `EXPORT WEIGHT` write activations, logits and weights to NumPy
files, and the [Python binding](crates/larql-python/README.md) has the same
operations. `larql lql 'HELP CAPTURE;'` and the [LQL guide](docs/lql-guide.md)
have the details.

## What you can do

| I want to… | Start here |
|---|---|
| Run or chat with a model | [Getting started](docs/getting-started.md) |
| Serve a model to an app or agent | [Serving](docs/getting-started.md#3-serve-it) · [runtime/API reference](docs/vindex3-runtime.md) |
| Query a model with LQL | [LQL guide](docs/lql-guide.md) |
| See what happens inside during a run | [Observation and intervention](docs/vindex3/observation-and-intervention.md) |
| Shrink a model (4-bit etc.) and measure the cost | [Representation](docs/vindex3/representation.md) |
| Edit a model's weights, live, and export the result to GGUF | [Quick start, step 4](#quick-start) · [worked example](examples/gemma-json-neuron/README.md) · [LQL guide](docs/lql-guide.md) |
| Edit what a model knows | [LQL guide — knowledge edits](docs/lql-guide.md) (experimental) |
| Use it from Python | [Python bindings](crates/larql-python/README.md) |
| Understand the format | [What is VINDEX3?](docs/vindex3/what-is-vindex3.md) · [vindex3.org](https://vindex3.org) |
| Check what works today | [Status](docs/vindex3/status.md) |

New to the vocabulary? See the [glossary](docs/glossary.md).

## Which models work?

LARQL checks every part of a model against what it knows how to run before
it downloads the weights. If something is missing, `larql run` refuses and
lists exactly what, after reading only the file headers, rather than
producing something that runs wrong.

Models run end to end so far include Qwen3, Gemma 3, Gemma 4 and Granite 4.1;
[Getting started](docs/getting-started.md#which-models-work) has the current
list. Older VINDEX2 `.vindex` files still open and run; new work uses VINDEX3.

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md): setup, where things live, a
first-change walkthrough and the checks to run before a pull request.
[AGENTS.md](AGENTS.md) is the detailed workspace reference;
[stack architecture](docs/architecture-stack.md) maps every crate.
The [documentation index](docs/README.md) leads to specifications, design
records and research results.

## License

See [LICENSE](LICENSE).
