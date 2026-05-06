# Codex-as-verifier for Claude Code

This repo wires a **two-agent loop** in the spirit of
[disler/the-verifier-agent](https://github.com/disler/the-verifier-agent),
adapted for Claude Code (builder) + the Codex CLI (verifier).

The pattern: after every Claude turn, an independent Codex run reads
Claude's last assistant message, falsifies it against the actual
filesystem, grades it, and either passes through or feeds a correction
back to Claude.

## Why

Half of an engineer's day reviewing agent output is the bottleneck.
Putting a second model on the verification side moves that work
off-human while keeping the writes single-tenant: the verifier is
**read-only** by sandbox policy and persona, so it can never silently
"fix" something the builder broke.

## Architecture

```
┌──────────────┐  Stop event   ┌─────────────────┐  read-only sandbox  ┌──────────────┐
│ Claude Code  │──────────────▶│ verify.sh hook  │────────────────────▶│ codex exec   │
│  (builder)   │               │ (.claude/hooks) │                     │  (verifier)  │
└──────┬───────┘               └────────┬────────┘                     └──────┬───────┘
       ▲                                │ schema-validated JSON                 │
       │                                ▼                                       │
       │                        ┌─────────────────┐                             │
       │  decision:"block"      │   verdict.json  │◀────────────────────────────┘
       │ + correction text      │  (PERFECT |     │
       └────────────────────────│   PARTIAL |     │
                                │   FAILED)       │
                                └─────────────────┘
```

Files:

| Path | Purpose |
|---|---|
| `.claude/settings.json` | Wires `Stop` + `SubagentStop` hooks at the project level. |
| `.claude/hooks/verify.sh` | Orchestrator. Reads hook stdin, builds prompt, runs codex, parses verdict, writes block decision when needed. |
| `.claude/verifier/prompt.md` | Codex persona — read-only verifier with explicit grading rubric. |
| `.claude/verifier/schema.json` | JSON Schema for the verdict (`grade`, `feedback`, `details`, `commands_run`). |
| `.claude/verifier/state/<session>.{count,last,log}` | Per-session loop counter, last verdict, NDJSON log. Excluded from git. |

## Enable / disable

Opt-in by environment variable. Without it, the hook is a silent no-op
so the repo is friendly for cloners who don't want this loop.

```bash
export LARQL_VERIFIER=1            # turn on for this shell
unset LARQL_VERIFIER               # turn off
```

Tunables (also via env vars):

| Variable | Default | Purpose |
|---|---|---|
| `LARQL_VERIFIER` | unset | Master switch. Set to `1` to enable. |
| `LARQL_VERIFIER_MAX_LOOPS` | `3` | Maximum corrective cycles per builder session. After this, the verifier escalates to the human (no block). |
| `LARQL_VERIFIER_TIMEOUT` | `120` | Seconds before `codex exec` is killed. The Stop hook itself has a 180s outer timeout in `settings.json`. |
| `LARQL_VERIFIER_MODEL` | (codex default) | Override `-m <model>` for codex. |

## Grading rubric

Codex returns one of five grades (see `.claude/verifier/prompt.md` for
the full rubric):

- **PERFECT / VERIFIED** — claims match reality. Hook exits 0; Claude
  stops normally. Loop counter is reset.
- **PARTIAL / FEEDBACK** — substantive divergence. Hook prints
  `{"decision":"block","reason":"[verifier:codex] <feedback>"}` to
  stdout and exits 0. Claude resumes with the feedback as a new user
  message.
- **FAILED** — verifier cannot reach a conclusion (sandbox failure,
  divergence too large to correct via feedback, etc.). Hook exits 0
  without blocking; the human reads the log to decide.

Loop guard: after `LARQL_VERIFIER_MAX_LOOPS` consecutive blocks for the
same session, the hook stops blocking and escalates. This mirrors the
Pi verifier-agent's 3-attempt cap.

Re-entrancy guard: if Claude is already in a Stop-hook loop (the hook
payload's `stop_hook_active` is true), the hook exits 0 immediately to
avoid runaway cycles.

## What gets verified

The verifier prompt instructs Codex to check, in roughly this order:

1. File existence + content (claimed creates/edits actually landed).
2. Git state (claimed commits stuck; tree is clean).
3. `cargo check --workspace --all-targets` for any Rust touch.
4. `cargo fmt --check` and `cargo clippy --workspace --tests -- -D warnings` if Claude claimed clean output.
5. Specific named tests via `cargo test -p <crate> <name>` — never the full suite.
6. OpenSpec gates: `openspec validate <change> --strict` and `python3 scripts/spec-trace.py --check` if `openspec/` was touched.
7. Spec → test linkage: `<!-- test: -->` annotations resolve to real tests, real wildcards, or explicit `unbacked`.
8. Counts and claims (e.g., "added 12 scenarios") spot-checked via the trace tool.

Codex stops early on the first failure rather than running every check —
the goal is fast actionable feedback, not exhaustive coverage.

## Logs

Every verdict is appended NDJSON-style to
`.claude/verifier/state/<session>.log`:

```jsonl
{"ts":"2026-05-06T18:42:01Z","attempt":1,"grade":"PARTIAL","verdict":"{...}"}
{"ts":"2026-05-06T18:42:55Z","attempt":2,"grade":"PERFECT","verdict":"{...}"}
```

Inspect the most recent verdict directly:

```bash
cat .claude/verifier/state/<session>.last | python3 -m json.tool
```

## Safety properties

- **Read-only by sandbox**: `codex exec --sandbox read-only` blocks any
  mutation by the verifier model itself. Even a misbehaving prompt
  cannot mutate the workspace.
- **Read-only by persona**: the prompt explicitly forbids `git commit`,
  `cargo install`, `rm`, `mv`, etc. Belt-and-suspenders with the sandbox.
- **Fail open**: any verifier-side error (codex missing, schema parse
  failure, timeout, unknown grade) results in *exit 0 without
  blocking* — the user is never trapped behind a broken verifier.
- **Bounded blast radius**: the hook only runs when `LARQL_VERIFIER=1`;
  cloners aren't affected by default.

## Caveats and known gaps

- **Codex auth**: `codex` must be logged in (`codex login`) on the
  machine where Claude runs. Without auth, the hook fails open — the
  user gets a one-line stderr and the turn passes through.
- **Cost**: each builder turn spawns one codex run. Keep
  `LARQL_VERIFIER_TIMEOUT` reasonable to bound cost. Disable with
  `unset LARQL_VERIFIER` for spec-only / docs-only sessions where
  verification adds no value.
- **Latency**: 30–120s per turn is the realistic range for substantive
  verification on this workspace. Don't enable on pure-discussion
  sessions where the cost outweighs the catch rate.
- **Claude can't read the verdict directly**, only the `feedback` text
  that comes back in `decision.reason`. The longer `details` field is
  for humans reviewing the log.
- **Subagent verification**: the `SubagentStop` hook applies the same
  loop to subagents. This is aggressive — set the counter cap to 1 for
  subagents if you find it too chatty (`LARQL_VERIFIER_MAX_LOOPS=1`
  and re-enable for the parent session).

## Provenance

The architecture is a faithful adaptation of the Pi Builder/Verifier
loop documented at <https://github.com/disler/the-verifier-agent>. The
key changes for Claude Code:

- Coordination uses Claude's `Stop` hook payload (`session_id`,
  `transcript_path`, `stop_hook_active`) instead of a Unix socket.
- Correction is delivered via the hook's `decision:"block"` JSON, not
  via a side-channel tool call.
- Read-only enforcement uses `codex --sandbox read-only` plus persona
  rules instead of a custom tool allowlist.
