# Verification independence

**Class: CURRENT.** Independence is an architectural property. It doesn't need a separate repository, but it does need boundaries that are
enforced, because one pull request could otherwise change both the engine and the verifier to make a wrong result look right.

| Requirement | What enforces it | Status |
|---|---|---|
| No LARQL code in the numerics | `verifier/tests/test_boundaries.py` refuses any import but the standard library, NumPy and `gguf`, and any way of running a program | Enforced by CI |
| Measurement code stays what was approved | `verifier/tests/test_verifier.py` checks `refcheck/` and its tests against the hash table in `verifier/provenance/PROVENANCE.md` | Enforced by CI |
| One change can't move both sides | The `verifier` workflow's `boundary` job fails a pull request that changes both the measurement and `crates/` | Enforced by CI, if required (below) |
| Results are bound to what they measured | Each answer carries the verifier's version and code digest, the container's digest, and the source's pinned identity | By construction |
| No circular answers | The expected results in `verifier/expected/` come from the verifier, and LARQL's tests don't generate them | By review |
| Separate review of verifier changes | `.github/CODEOWNERS` | **Needs a repository setting** |

**What this leaves to a repository setting.** `main` has no branch protection today, so neither the code owner review nor the `boundary` job can
block a merge yet. An administrator needs to protect `main`, require code owner review, and make `verifier / boundary` a required check.

**The `boundary` job's escape hatch.** Some changes do need both sides, such as a format change. The label `verifier-and-engine` lets one through.
A person has to apply it, so that the decision is visible on the pull request.

**The isolated checker stays separate.** The reference checker in its own repository remains the research and adversarial evaluation authority,
with its own test history. `verifier/refcheck/` is a pinned copy of its approved measurement code. See
[PROVENANCE.md](../../verifier/provenance/PROVENANCE.md).

**What this doesn't give you.**

- `verifier/larql_contract_verifier/` wasn't written under the checker's isolation: its author had read LARQL's source.
- LQL trusts whichever verifier it's configured with. It checks an answer's form and what it's bound to, not whether it's true.
- This is empirical checking of one model against one contract, not proof checking.
