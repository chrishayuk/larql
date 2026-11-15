# The restricted session service: `larql serve-session` and `larql.connect`

**Class: CURRENT.** A LARQL process that serves one VINDEX3 container to one client over a Unix socket, through
typed operations only, under a policy it enforces itself. Its client is `larql.connect(socket)`, a pure-Python
object with the same methods as the in-process `larql.session(path)` (see the
[Python crate README](../crates/larql-python/README.md)).

It exists so that a client can be given every weight, activation, logit and generation it needs, and the means
to change the model, from LARQL, while its own process has no access to the model files. **The second half is
not this service's job.** The service is not an operating-system sandbox; see [What it does not
enforce](#what-it-does-not-enforce).

```bash
larql serve-session --config policy.json --socket /path/to/larql.sock
```

```python
import larql
s = larql.connect("/path/to/larql.sock")
logits = s.capture("logits", "The capital of France is").array     # [1, vocab], float32
s.apply_patch("edit.vlp")                                           # a file in the workdir
s.capture("logits", "The capital of France is")                     # counts against the budget
s.remove_patch("edit.vlp")
s.budget()                                                          # edited-model executions left
```

## The policy

A JSON file, read once at start. Unknown fields are refused, so a misspelt limit fails the start.

```json
{
  "policy": "larql-session-policy/v1",
  "container": "/models/gemma3-4b-it.vindex3",
  "workdir": "/runs/e1/agent",
  "audit_log": "/runs/e1/audit/session.jsonl",
  "resume": false,
  "operations": ["tensors", "tensor", "tokenize", "decode", "capture", "infer", "generate",
                 "begin_patch", "add_low_rank", "save_patch", "apply_patch", "remove_patch",
                 "patches", "discard_weight_edits"],
  "budget": {"edited_executions": 40},
  "limits": {"max_generate_tokens": 256, "max_capture_positions": 4096},
  "representation_source": "stored",
  "staging_dir": "/runs/e1/staging"
}
```

| Field | Meaning |
|---|---|
| `container` | The one VINDEX3 container. Bound at start; never another (there is no `USE`). |
| `workdir` | The only directory request paths may name. Must not contain, or be inside, the container. |
| `audit_log` | JSONL. Must be outside the workdir and the container. Refused if it exists, unless `resume`. |
| `resume` | Continue an existing log: its sequence numbers carry on, and the budget used is the number of its `admitted` records (never a figure a record states). Refused if the log is missing (a resume never starts from a fresh budget), does not parse whole, or its last `start` record names another container (by `index.json` digest). |
| `operations` | The allow-list. `update_weights` (and any other) is allowed only if listed. `hello` and `budget` are always answered. |
| `budget.edited_executions` | How many executions of an edited model are answered. |
| `limits` | Optional; each field defaults: `max_response_bytes` 256 MiB (array bytes per response), `max_request_bytes` 64 MiB (one request line), `max_generate_tokens` 256, `max_capture_positions` 4096, `max_capture_layers` 64, `max_sequence_tokens` 4096 (prompt plus continuation; also the most ids one `decode` takes), `max_patch_bytes` 64 MiB, `max_infer_top` 1000, `max_applied_patches` 16, `max_text_bytes` 64 KiB (a prompt, a continuation, text to tokenize, and the text `decode` returns), `max_staging_bytes` 256 MiB (the staged copies held at once), `max_audit_records` 10,000,000 (past it every request is refused). `max_request_bytes` defaults to 16 MiB: the largest request is an `add_low_rank` operand, and a float64 rank-1 column over Gemma 3 4B's largest tensor (262,208 rows) is 2.8 MB of base64, so 16 MiB carries rank 5 there. |
| `require_capture_bound` | Default `true`: a capture whose response size the service cannot bound from the plan before running it (a site the plan gives no width, such as a hyper-connection residual or a routed FFN's down projection input) is refused up front, uncharged, with the site named. Then no capture ever runs and is afterwards refused on size. `false` lets such a capture run, admitted and charged, under the post-run size check. On a dense single-stream program (Gemma 3 4B) every site is bounded. |
| `representation_source` | `stored` (default), `auto` or `transient`, as `larql lql --representation-source`. |
| `staging_dir` | Required. Where the service makes its private staging directory (mode 0700). Must be outside the workdir and the container, must not contain the workdir, and should be a directory the client cannot write. |

Paths may be anywhere the service's user can reach; nothing ties them to one tree.

## The protocol: `larql-session/v1`

JSON lines over the Unix socket, one request, one response, in order.

```json
{"protocol": "larql-session/v1", "id": 12, "op": "capture",
 "args": {"site": "logits", "prompt": "The capital of France is", "positions": "ALL"}}
```

```json
{"protocol": "larql-session/v1", "id": 12, "seq": 31, "ok": true,
 "result": {"array": {"npy_base64": "k05VTVBZ…", "sha256": "…", "dtype": "<f4", "shape": [5, 262208]},
            "provenance": {"site": "logits", "executes": "patched", "patches": […], …}},
 "budget": {"limit": 40, "used": 3, "remaining": 37}, "edited": true, "counted": true}
```

- `id` is echoed (at most 256 bytes as JSON; a longer one is refused and answered with `id` null). `seq` is the
  request's record number in the audit log. `request_uid` is the service's own identifier of the request,
  unique for the service's lifetime and across resumes (an instance identifier drawn at start, and a count).
- `budget` is after the request. `edited`: the execution ran (or was refused for) an edited model. `counted`:
  it was charged.
- An error is `"ok": false, "error": {"kind", "reason"}`. Kinds: `invalid_request` (not JSON, not a request,
  a bad argument), `unsupported_protocol`, `refused` (the policy: an operation not allowed, a path, a limit),
  `budget_exhausted`, `failed` (LARQL's own refusal or error, its reason verbatim), `busy`. After any error the
  service keeps answering on the same connection, except after an over-long line, which is answered and the
  connection closed.
- **Arrays** travel as a whole `.npy` file, base64 in the JSON: the bytes the matching LQL statement writes, so
  `sha256` equals the provenance's `npy_sha256`. The client checks the digest. Base64 (not a binary frame) keeps
  every message one JSON line, at a third more bytes.
- **Operand arrays** (`add_low_rank`'s `a` and `b`) go the other way the same way: `{"npy_base64": …}`, read
  strictly as 2-D, C-order, little-endian float64 or float32.

| `op` | `args` | Result (as the in-process method returns) |
|---|---|---|
| `hello` | — | protocol, version, container index digest, allowed operations, limits, budget |
| `budget` | — | `{"limit", "used", "remaining"}` |
| `tensors` | `filter` | list of tensor dicts |
| `tensor` | `name`, `rows`, `cols`, `bits` | array + provenance |
| `tokenize` / `decode` | `text`, `raw` / `ids` | ids / text |
| `capture` | `site`, `prompt`, `layer`/`layers`, `continue_text`/`continue_tokens`/`generate`, `positions` | array + provenance |
| `infer` / `generate` | `prompt`, `top` / `prompt`, `n` | dicts |
| `begin_patch` | `path` | null |
| `add_low_rank` | `tensor`, `a`, `b`, `columns`, `numerics` | stage report |
| `update_weights` | `tensor`, `row`, `cols`, `delta` | stage report |
| `save_patch` | — | counts; `path` is the workdir name |
| `apply_patch` / `remove_patch` | `path` | dicts; `path` is the workdir name |
| `patches` | — | the `SHOW PATCHES` dict |
| `discard_weight_edits` | — | `{"discarded", "unrecorded", "from_patches"}` |

Each operation calls the typed Rust operation in `larql_lql::operations` that the LQL statement and the
in-process Python method call; no LQL text is accepted or built.

## What it enforces

- **One container, read-only, bound once.** Nothing in the protocol rebinds, compiles or writes into it. Its
  path is never sent. Redaction is structural: a provenance record's `container` field becomes
  `<bound container>` (the index digest identifies the container), and a patch's staged path becomes its
  workdir name; arrays and the strings the client sent are never searched or rewritten. A LARQL reason that
  names the container's path, the staging directory, or repeats an absolute path the client sent is withheld
  whole (the audit log keeps it), so a guessed path is told the same thing whether it is right or wrong.
- **The allow-list.** Any other `op` is refused with the list.
- **Paths stay in the workdir.** A request names a file relative to the workdir (or absolutely under it). The
  name must pass the container path rule (no root, `.`, `..`, backslash or NUL). The file is then opened one
  component at a time with `openat` and `O_NOFOLLOW` from a descriptor of the workdir taken at start, so no
  symlink is followed anywhere on the way and no swap between a check and the open can redirect it. The file must
  be a regular file with one link (hard links to files elsewhere are refused; FIFOs are refused without blocking).
  LARQL itself never sees a workdir path: an applied patch is read here and copied to the private staging
  directory, and LARQL reads the copy; a saved patch is written by LARQL into staging and copied into a workdir
  file opened as above. Each staged file has a name unique for the service's lifetime (`<n>.vlp`, flat: no
  directories are mirrored), so re-applying a name never touches a live patch's copy. A copy is deleted when
  LARQL rejects the patch, and when its patch is removed however it is named (by path or by its own
  description); their total is capped by `max_staging_bytes`. Staged files get the same discipline: the place is cleared (a planted symlink is
  removed, never followed) and the copy created with `O_CREAT | O_EXCL | O_NOFOLLOW` before LARQL reads it; the
  place is cleared before LARQL saves there, and what it saved is read back confined (a link or hard link is
  refused). `begin_patch` checks the
  destination when the recording starts, and `save_patch` again before LARQL writes; a refused save keeps the
  recording.
- **The budget, server-side.** An `infer`, `generate` or `capture` executed while the session has pending
  edits (LARQL's own `has_pending_edits`: a weight op applied or staged, or a knowledge edit) or any patch is
  applied is an edited execution. It is **charged when admitted, before it runs**: an `admitted` audit record
  carrying the budget after it is appended and synced, and only then does the model run. Whatever happens next
  (LARQL refuses after running, the response is too large, the client hangs up, the service is killed) the
  charge stands, and a resumed log recounts it. A capture over many positions or layers is still one. A
  capture's provenance (`executes`) is a second witness: a capture it says ran edited weights is charged even
  if the pre-check did not. When the budget is spent, edited executions are refused (`budget_exhausted`) and
  logged; base-model executions and reads (`tensor`, `tokenize`, …) stay free. If a capture's provenance says
  edited weights ran when the gate said stored, the run is charged regardless of the budget (it may go past
  its limit, recorded as `overdrawn`), and the service treats it as a fault and stops answering. Removing every patch and
  discarding staged ops makes executions free again.
- **Checks before execution.** Everything a capture's arguments allow is checked before it is admitted, so a
  refusal there costs nothing and computes nothing: the site's need for a layer (or none), empty or repeated
  `layers`/`positions`, layers against the plan, `REPLY` without a continuation, an empty continuation, token
  ids outside the vocabulary, `generate=0`, positions against the sequence (exactly, or for `generate` against
  the prompt plus `n`), and the limits. A position the model's own generation decides (one past where
  generation actually stopped) can only be refused after running; that refusal is charged, and under
  `generate=` it is told only "position is beyond the generated sequence", with no length or token.
- **Limits.** Generation length, capture positions and layers, sequence length, `top` and the patches
  applied at once are refused before anything executes. A capture's array bytes are bounded before it runs
  (layers × positions × the site's width from the plan × 4, plus the header) and refused up front; where the
  plan gives no width (a hyper-connection residual, a routed FFN's down input) the capture is refused up front
  under `require_capture_bound` (the default), or left to the check after running when that is false. A tensor read is refused before it is read when the listing shows it would be too large. Every
  response's array bytes are checked again before it is sent.
- **The audit log.** Every request gets one record, appended (`O_APPEND`) and synced before its response is
  sent: `seq`, `time_unix_ms`, `request_uid`, `id`, `op`, `args_sha256` and the arguments (operand arrays, and
  strings over 1 KiB, replaced by their digest; the whole summary by digest and length when it passes 4 KiB; a
  line that is not a request by its digest and length), `ok`, the error (with the withheld reason as
  `internal`), `edited`, `counted`,
  `admitted_seq`, `outcome`, `budget_used`, `budget_remaining`, each returned array's digest, dtype and shape,
  and the SHA-256 of the response line. The `admitted` record is the evidence that budget was spent: the
  `request_uid` (also on the response and on the outcome record), the client's `id`, the op, `args_sha256` and
  the summarised arguments, the edits in effect (as a capture's provenance lists them) with their digest
  `edits_sha256`, each applied `.vlp`'s SHA-256 and each weight op's digest, and the budget after it. Three
  states are distinguishable: a request `rejected` before
  execution (no `admitted` record, not charged); an `admitted` record (charged, written before the model
  runs); and the request record of an admitted execution, `completed` or `failed` (outcome only, no further
  charge). A `start` record holds the policy digest, the container path and index
  digest, the workdir, the allow-list, budget, limits and every `LARQL_*` environment variable; `connect`,
  `disconnect`, `connection_rejected` and `stop` are recorded too; `connect` records the peer's uid, gid and (on
  macOS and Linux) pid. If the log cannot be written, the request is
  answered with an error and the service stops answering.
- **One client.** While a client is connected, a second connection is sent one `busy` error and closed (checked
  between requests). Requests are answered one at a time. A response must be written whole within 10 s (a
  client that stopped reading is hung up on, and its `disconnect` record says `write_deadline_expired`, apart
  from `client_gone` for a write the socket refused outright); it is written a chunk at a time, each write waiting at most
  50 ms, and a shutdown between chunks ends it at once. The audit log is locked (`flock`) for the service's
  lifetime: a second service on the same log refuses to start.
- **Shutdown.** SIGTERM, SIGINT or SIGHUP: the request in flight is finished, the connection closed, the `stop` record
  written, the socket file removed (only if it is still the socket this process bound). The service refuses to
  start on an existing socket path rather than unlink it.

## What it does not enforce

- **It is not an OS sandbox.** It cannot stop the client's process from reading the model files, the HF cache, a
  GGUF, the audit log or the staging directory directly, or from running another model runtime. The staging
  directory in particular must not be writable by the client: LARQL itself opens staged paths by name, so a
  party that can write there between the service's checks and LARQL's open (the service's own user) could
  redirect a read or a save; the service's checks narrow that window and catch it afterwards, they do not
  close it. That must be
  enforced by the operating system: the model directory unreadable (and untraversable) by the client's user or
  sandbox, no other copy of the checkpoint reachable, no model runtime installed. The pure-Python client means the
  client environment needs no native `larql` extension at all.
- **Socket access.** Anyone who can connect to the socket is the client. The socket is created with the
  service's umask; restrict it with the directory's permissions.
- **Counterfactuals in the client.** Arithmetic the client does on exported weights and activations is not
  metered.
- **Resource use inside an allowed request.** The limits bound sizes, not time: a long capture is not
  interrupted, and memory use of a single allowed execution is LARQL's.
- **The workdir's contents.** The client can write anything in its workdir; the service only guarantees it reads
  and writes inside it. A workdir on the same filesystem as the model and writable by a party that can also
  hard-link model files is refused file by file (link count), not prevented.
- **Integrity of the log against the operator.** The log is append-only from the service's side; protecting it
  from other writers is the operating system's job (keep it outside anything the client can write).

### Unix sockets from a Claude Code sandbox

A client run inside Claude Code's Bash sandbox on macOS cannot connect to a Unix socket by default: the
settings reference documents `sandbox.network.allowUnixSockets` (a list of socket paths, macOS only; default
unset, which blocks every Unix socket) and `sandbox.network.allowAllUnixSockets` (the only switch on Linux/WSL2).
For example, in the user's settings (repository settings can be ignored under an admin-required sandbox):

```json
{"sandbox": {"network": {"allowedDomains": [], "strictAllowlist": true,
                         "allowUnixSockets": ["/Users/me/e1-run/larql.sock"]}}}
```

`allowedDomains`/`strictAllowlist` govern the network proxy; the documentation does not say they affect Unix
sockets, so check the connect from inside the sandbox before relying on it.

## Tests

- Rust, in `crates/larql-cli/src/commands/primary/session_service/tests/`: every operation against the
  in-process typed operation (arrays bit for bit), refusals, paths (`..`, absolute, symlinks to files and
  directories, a link swapped in after `begin_patch`, links planted in staging, hard links, FIFOs), the budget
  (charged at admission; every early refusal uncharged; a failure after admission charged; the edited
  predicate checked against capture's `executes` after every state-changing operation on the allow-list), the
  limits (the capture size bound before running), the audit log and its resume (budget recounted from
  `admitted` records), redaction of the container path, the envelope, and the socket (ordering, `busy`,
  over-long lines, a client that stops reading, a client that hangs up mid-request, shutdown). `review.rs`
  holds one named regression test per finding of the security reviews (`m1_…` to `l6_…`, `early_stop_…`).
- The process tests in `crates/larql-cli/tests/test_serve_session.rs`: SIGTERM stops the service, records the
  stop, removes the socket, and a second start on the same log is refused; SIGKILL during an admitted
  generate, then a resume, leaves that generate spent.
- Python, `crates/larql-python/tests/test_session_service.py`: every client method against `larql.session` on
  the synthetic container, refusals as exceptions, the budget, an answer naming another request refused, and
  an import of the client with the native extension blocked.
