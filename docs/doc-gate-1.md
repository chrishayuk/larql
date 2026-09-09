# DOC-GATE-1 — the documentation instrument must have examined something

**Status:** frozen before implementation
**Date:** 2026-09-09
**Qualified against:** main `9c3dbc91` (OPT-CONTRACT-1, #467)

---

## The defect, measured before it is repaired

`scripts/check_doc_references.py` matches `SKIP_DIRS` against the ABSOLUTE
directory path:

```python
for dirpath, dirnames, filenames in os.walk(ROOT):
    if any(s in dirpath + "/" for s in SKIP_DIRS):
        dirnames[:] = []
        continue
```

`SKIP_DIRS` contains `/.claude/`, and every worktree in this repository lives
under `.claude/worktrees/`. So `ROOT` itself matches, every `dirpath` beneath
it matches, the walk is pruned at the first directory, and the corpus is
empty. The gate then prints `no broken references` and **exits 0**.

Measured at `9c3dbc91`, from `.claude/worktrees/doc-gate-1`:

| walk | files found |
|---|---|
| absolute match — what ships today | **0** |
| `/.claude/` removed from `SKIP_DIRS` | 3855 |
| skip matched relative to the repository root | 3855 |

The same run from a checkout outside `.claude/` examines **242 documents and
1441 references**. From a worktree it examines **0 of each and passes**.

This is not a CI defect. GitHub Actions checks out at a clean path, so the
gate is real where it runs. It is a LOCAL verification defect: every
pre-push run any session made from a worktree was vacuous, and it failed
open while saying so in the language of success. Two sessions hit it
independently on 2026-09-09.

## Why this is its own transition and not part of #467

OPT-CONTRACT-1 asks *are these frozen semantic contracts continuously
enforced?* This asks *did the documentation instrument examine anything at
all?* They share a fail-closed philosophy and nothing else. Merging them
would make #467 harder to describe and harder to revert.

## Acceptance conditions, frozen

1. **Skip semantics are repository-relative.** A checkout beneath
   `.claude/worktrees/...` inspects the same logical corpus as an ordinary
   checkout.
2. **Zero documents examined is a hard failure.** Not a warning, not a
   green — a non-zero exit, so no future path or filter bug can produce
   another vacuous pass.
3. **An ordinary checkout and a `.claude/` checkout produce the SAME
   document and reference counts** on the same commit. This is the strongest
   of the four: condition 2 alone would still admit a partial-tree bug that
   scanned half the corpus and reported a confident non-zero number.
4. **The check name describes what it enforces.** The job named `doc links`
   now also decides whether contract indexes name real tests (#467), so a
   contract-index failure currently reaches a reviewer as "doc links"
   failing.

Plus the control that keeps condition 1 honest:

5. **Legitimately skipped directories stay skipped.** `target/`, `.git/`,
   `.claude/`, `.venv/`, `node_modules/`, `coverage/` must contribute
   nothing to the corpus under the repaired semantics.

## How they are enforced

Conditions 1, 2, 3 and 5 become controls in
`scripts/check_doc_references.py selftest`, wired into CI beside the
`ci_throughput.py` and `conformance_evidence.py` selftests. That is the
lesson this repository already recorded on 2026-09-07, in the workflow
comment beside the first of those: **a checker nobody runs is not a check.**
Controls that live only in a transcript are the next thing nobody runs.

## Out of scope

- The content of any reference the gate reports. This transition changes
  which files are EXAMINED, not what counts as broken.
- `check_doc_links.py` and `check_contract_index.py`, which walk from
  explicit roots and are not vacuous from a worktree (857 links, 85 named
  tests, both measured from one).
- Any change to `SKIP_DIRS` membership. The six entries are correct; only
  the matching is wrong.
