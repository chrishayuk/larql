# COMPOSE-CORRECTION-1 harness

Measures `INSERT ... MODE COMPOSE` installs for the COMPOSE-CORRECTION-1 study (pre-registration draft:
PR #702). It uses only the public `larql_lql` API
(the balance observer in `larql_lql::balance_trace`, `Session`, the AST) and touches no product code.

```sh
cargo build --release -p larql-demos --example compose_harness
target/release/examples/compose_harness \
  --container ~/chris-models/q8enc-ad82acba/qwen3-0.6b-q8_0.vindex3 --tag qwen --arm A1 \
  --set controls|development|faults \
  --bank bench/compose-correction-1/bank.json \
  --controls crates/larql-demos/examples/lql/compose_harness/controls.json \
  --out DIR [--alpha F] [--unrelated N] [--check-determinism]
```

Exit status: 0 success, 1 a control did not reproduce, 2 refused or failed to run.

## Sets

| `--set` | Reads | Purpose |
|---|---|---|
| `controls` | `controls.json` | Replay outcomes observed before the harness existed. Must reproduce under the arm the binary was built for. |
| `development` | the bank's two `development` facts | Build and debug the harness. Never scored. Every install records the session's state fingerprint before and after. |
| `faults` | the bank's two `development` facts | Failure atomicity (prereg §7.1): each of the seven fault points of `larql_lql::install_fault` is injected in a fresh session, alone and after an earlier successful install; the cell reports which fingerprint components changed and which of the bank's 50 atomicity-probe prompts moved. Measures the implementation; it is not used to tune anything. |
| `registered` | the bank's registered facts | **Unreachable until the pre-registration is frozen.** Requires a `FROZEN` file beside the bank holding exactly the bank's SHA-256. Every attempt, allowed or refused, is appended to `registered-attempts.jsonl` beside the bank. |

Registered facts are read on no other path (`main.rs`, behind `guard::admit_registered`).

## Arms

The harness is built once per arm and records which it was built for; it does not switch arms at run
time. A0 is the commit before the leading-space matcher fix (#701), A1 is A0 plus #701, both with the
balance observer (#704).

### Pinning at freeze

For a registered run each arm is built from a single commit that contains this harness, and the run's
provenance must report that commit with zero tracked changes. A0 is the pre-#701 library plus the
observer plus this harness; A1 is the same with #701. A binary built from a checkout with tracked
changes says so in its provenance block and is not admissible for a registered run.

## What a run records

Per fact: the prompt the install actually captured its residual from, the evaluation prompt, and
whether they are equal; the target's token ids and pieces; every balance iteration (the probability the
controller read, the down-vector norm, the decision, the canonical prompt's top-1 and the target's
first-token probability by id, and whether a watched neighbour's or unrelated prompt's top-1 moved
from its baseline); the termination; every prompt's baseline and post-install top-1; and the section 6
outcome, scored on token ids, never on the string matcher under study.

The provenance block names the container (index and tokenizer SHA-256), the bank SHA-256, the commit
and tracked changes of the checkout the binary was built from, the harness source hash, and the
command.

## Not yet measured

Predictions in the fault matrix are compared as the printed top 20 (ids and probabilities to two
decimals), not as raw logits. KL divergence is not computed; the harness sees the top 20 candidates,
not full distributions.
