# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Reference the `./AGENTS.md` file for the broader context of this project.

## Spec-first workflow

LARQL is spec-driven via OpenSpec. **Every code change must reference an
OpenSpec capability** under `openspec/specs/<capability>/spec.md` (or, for
in-flight work, `openspec/changes/<id>/specs/<capability>/spec.md`).

- 44 capabilities are catalogued in
  [openspec/changes/backfill-specs/proposal.md](openspec/changes/backfill-specs/proposal.md).
- Scenarios are linked to tests by `<!-- test: <fqn> -->` annotations.
- Run `make ci` before pushing — it chains fmt, clippy, tests, traceability check, and openspec validation.
- See [openspec/README.md](openspec/README.md) for authoring rules.