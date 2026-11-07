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

# Linux (x86_64)
curl -L https://github.com/chrishayuk/larql/releases/latest/download/larql-x86_64-unknown-linux-gnu.tar.gz | tar xz
sudo mv larql-x86_64-unknown-linux-gnu/larql* /usr/local/bin/
```

Windows: download `larql-x86_64-pc-windows-msvc.zip` from the
[latest release](https://github.com/chrishayuk/larql/releases/latest).
Each archive contains `larql` and `larql-server` (used by `larql serve`).
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

`gemma-3-4b-it`, `gemma-3-12b-it` and `gemma-4-12b-it` are ready to pull the
same way.

**2. Serve it** over an OpenAI-compatible API:

```bash
larql serve Qwen3-0.6B --port 8080

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Write a haiku about Rust."}]}'
```

**3. Look inside it.** This is what most runtimes can't do. `larql vindex3`
commands take the model's name; LQL's `USE` takes its directory, which
`larql show` prints:

```console
$ M=$(larql show Qwen3-0.6B | awk '/^Path:/ {print $2}')

$ larql lql "USE \"$M\"; INFER \"The capital of France is\" TOP 3;"
   1.  Paris               (74.36%)  [id 12095]
   2.  the                 (2.18%)  [id 279]
   3.  in                  (2.07%)  [id 304]

$ larql vindex3 observe Qwen3-0.6B --backend production \
    --prompt "The capital of France is" --record run.jsonl
```

`observe` writes a record of every layer's contribution while the model runs;
`larql repl` opens an interactive LQL shell.

## What you can do

| I want to… | Start here |
|---|---|
| Run or chat with a model | [Getting started](docs/getting-started.md) |
| Serve a model to an app or agent | [Serving](docs/getting-started.md#3-serve-it) · [runtime/API reference](docs/vindex3-runtime.md) |
| Query a model with LQL | [LQL guide](docs/lql-guide.md) |
| See what happens inside during a run | [Observation and intervention](docs/vindex3/observation-and-intervention.md) |
| Shrink a model (4-bit etc.) and measure the cost | [Representation](docs/vindex3/representation.md) |
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
