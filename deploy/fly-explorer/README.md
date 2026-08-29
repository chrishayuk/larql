# vindex3-explorer — the public explorer on fly.io

The hardened public endpoint behind vindex3.org's **ENTER A MODEL**
terminal: `larql-server --public-explorer` over one immutable demo
container.

## What it serves

- `POST /v1/query` — one LQL statement per request, executed under
  `CapabilityProfile::PublicExplorer` (read surface + `INFER` with
  `GENERATE ≤ 32`; everything else refuses with 403 after parsing,
  before execution — see `crates/larql-lql/docs/spec.md` §4.5).
- `GET /v1/health`, `/v1/models`, `/v1/describe`, `/v1/walk`,
  `/v1/relations`, `/v1/stats` — the read-only REST nouns.
- Nothing else. Mutating routes are not mounted.

## The container

`vindex3-demo`, regenerated at every boot by the `vindex3-demo` binary
(`crates/larql-server/src/bin/vindex3-demo.rs`): the miniature Glimmer
system — two layers, mixed attention policy (Sliding(3)+RoPE,
Full+NoPE), gated attention — encoded through the real inventory →
encode pipeline, ~40 KB on disk. Synthetic weights, real format: the
graph, directory, provenance, authority, and execution the terminal
walks are all genuine. No volume, no HF download, no state worth
keeping: wipe the machine and it rebuilds identically.

## Deploy

```bash
fly apps create vindex3-explorer
fly deploy --app vindex3-explorer --config deploy/fly-explorer/fly.toml --remote-only
```

Hardening is layered: fly's connection limits → per-IP rate limit
(`RATE_LIMIT`, default 120/min, trusting `X-Forwarded-For` from fly's
proxy) → server concurrency limit (`MAX_CONCURRENT`) → the capability
profile inside the LQL session. The machine auto-stops when idle and
cold-starts in a few seconds (the container regeneration is
milliseconds).

## Custom domain

```bash
fly certs add public.vindex3.org --app vindex3-explorer
```

then add the CNAME (`public` → `vindex3-explorer.fly.dev`) at the DNS
host.
