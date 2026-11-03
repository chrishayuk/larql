# Verification independence

**Class: CURRENT.** Independence is an architectural property. It doesn't need a separate repository, but it does need boundaries that are
enforced, because one pull request could otherwise change both the engine and the verifier to make a wrong result look right.

| Requirement | What enforces it | Status |
|---|---|---|
| No LARQL code in the numerics | `verifier/tests/test_boundaries.py` refuses any import but the standard library, NumPy and `gguf`, and any way of running a program | Enforced by CI |
| Measurement code stays what was approved | `verifier/tests/test_verifier.py` checks `refcheck/` and its tests against the hash table in `verifier/provenance/PROVENANCE.md` | Enforced by CI |
| One change can't move both sides | The `boundary` check in `verifier-boundary.yml`, required on `main`, fails a pull request that changes both the measurement and `crates/`. It runs from the base branch, so a pull request can't rewrite it | Enforced by a required check |
| Results are bound to what they measured | Each answer carries the verifier's version and code digest, the container's digest, and the source's pinned identity. These say what was measured. They don't authenticate who measured it (below) | By construction |
| No circular answers | The expected results in `verifier/expected/` come from the verifier, and LARQL's tests don't generate them | By review |
| Separate review of verifier changes | `.github/CODEOWNERS`, required on `main` | **Not independent yet** (below) |

**Repository settings.** These live outside the repository, so check them rather than trusting this page:
`gh api repos/chrishayuk/larql/rules/branches/main`. Two rulesets apply to `main`:

- `main branch`, with no bypass: changes arrive by pull request, `boundary` must pass, and the branch can't be deleted or force-pushed.
- `verifier code owners`: a code owner must approve changes to the paths in `CODEOWNERS`. Repository admins can bypass this when merging a
  pull request, not by pushing directly, and GitHub records each bypass.

**A sole code owner is not an independent reviewer.** The repository has one code owner, who writes most changes and can't approve their own
pull request, so in practice their verifier changes merge through the recorded admin bypass. The rule stops changes from anyone else reaching
the verifier unreviewed, and leaves a record. It does not put a second person between an author and the measurement. A change to a verifier
contract or a numerical algorithm doesn't have independent human review until another trusted reviewer is added to `CODEOWNERS`.

**The `verifier-and-engine` override.** Some changes do need both sides, such as a format change. To let one through:

1. A maintainer or admin comments or reviews with a line starting `Verifier-and-engine justification:`, saying why the change can't be split.
2. A maintainer or admin then applies the label `verifier-and-engine`.

GitHub lets anyone with triage access apply a label, so the gate asks GitHub who applied it and who wrote the reason, and refuses anyone else.
Removing the label, or pushing again, runs the gate again. The check doesn't judge the reason. It makes sure a maintainer gave one on the record.

## The verifier is not authenticated

LQL runs whichever verifier it finds, and checks the form of its answer and what the answer is bound to. The version and code digest in the answer
are reported by the program being identified, so they support reproducing a result, and a replaced or modified program can report anything.

**Replacing the verifier can make a contract appear satisfied without changing the model.** That is acceptable for what this is today, a measurement
tool run by someone who controls their own machine. `VERIFY REPRESENTATION` and its exit status **must not gate a release** on this basis.

A verification-gated release would first need the gate to pin an approved verifier, through configuration the checked change can't modify, and to
check that identity itself rather than ask the verifier. For this Python package, the identity would have to cover the interpreter and locked
dependencies as well as the package's own files. None of that exists yet.

**The isolated checker stays separate.** The reference checker in its own repository remains the research and adversarial evaluation authority,
with its own test history. `verifier/refcheck/` is a pinned copy of its approved measurement code. See
[PROVENANCE.md](../../verifier/provenance/PROVENANCE.md).

**What this doesn't give you.**

- `verifier/larql_contract_verifier/` wasn't written under the checker's isolation: its author had read LARQL's source.
- Authentication of the verifier: see above.
- Independent human review: see above.
- This is empirical checking of one model against one contract, not proof checking.
