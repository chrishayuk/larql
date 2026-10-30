# GW-CAR-1B — raw evidence archive

**Status:** archived and verified on 2026-10-30. This document records where
the raw outputs that produced [the GW-CAR-1B result](gw-car-1b-results.md)
are kept. It adds no gate, re-runs nothing, and changes no conclusion. The
result, protocol and adjudication are as sealed on 2026-09-28.

## What is archived

The protocol's `output_root`, `output/gwcar1b-capital-v1/`: 57 files (114 MB)
under `candidates/`, `capture/`, `reference/` and `replay/`, written
2026-09-28 08:35–08:42 local time. `output/` is gitignored, so these bytes
were never committed, as with GW-CAR-1A.

The per-file SHA-256 manifest is committed beside the protocol:
[`gwcar1b-capital-raw-outputs.sha256`](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-raw-outputs.sha256)
(sha256 `b1d185bf563657462b235e2223914054dbc28f9073832b61fb947cfda9d0cd60`,
paths relative to `output/`).

## Where it is

| | |
|---|---|
| Registry experiment | `gw-car-1b-fresh-capital-carrier-sources` (`EXP-20261030-212128-00980`), programme `larql` |
| Registry run | `RUN-20261030-212135-01170`: `workspec.kind = historical-import`, never claimed or executed |
| Archive artifact | **2443**: `s3://chuk-experiments/runs/RUN-20261030-212135-01170/dataset/gwcar1b-capital-v1.tar.zst` |
| Archive sha256 | `d9c7a1ae98fd6fe7e8e7982f6bb145041f126bfeb09470697879e9611037e3b6` (54,926,418 bytes; tar + zstd -19) |
| Manifest artifact | **2444**: `s3://chuk-experiments/runs/RUN-20261030-212135-01170/dataset/gwcar1b-capital-v1.sha256` |

The registry entry is a **retrospective import**. Its design records the
protocol, plan, executable, adjudication and audit hashes, plus an
`import_note` saying it is not a prospective registration. The
pre-registration authority remains the committed protocol and its hash.

## Retrieval

```bash
B=https://chuk-experiments-server.fly.dev
mkdir gwcar1b-restore && cd gwcar1b-restore
curl -sSL -H "Authorization: Bearer $CHUK_EXPERIMENTS_API_KEY" \
  -o gwcar1b-capital-v1.tar.zst "$B/v1/artifacts/2443/download"
shasum -a 256 gwcar1b-capital-v1.tar.zst
# expect d9c7a1ae98fd6fe7e8e7982f6bb145041f126bfeb09470697879e9611037e3b6
zstd -dc gwcar1b-capital-v1.tar.zst | tar -xf -
shasum -a 256 -c <larql>/bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-raw-outputs.sha256
```

## Verification performed

1. The manifest was built from the original `output/gwcar1b-capital-v1/`
   (57 files, no symlinks).
2. The archive was extracted into a fresh directory, and all 57 files matched
   the manifest with no extra files.
3. After upload, both artifacts were downloaded back through the registry
   (`/v1/artifacts/{id}/download`) into an empty directory. The archive
   hash matched, the downloaded manifest was byte-identical to the local
   one, and all 57 extracted files matched it.
4. The original `output/gwcar1b-capital-v1/` still matches the manifest.
   It has not been moved or deleted.

A second verified copy is kept locally at `~/larql-evidence/gw-car-1b/`
(archive and manifest, same hashes) on the internal data volume.
