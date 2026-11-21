"""PREC-DIAG-1 E1–E3 runner (unit 4): execution and admission bound to FREEZE.json.

`FREEZE.json` (`freeze.py`) holds every identity. This runner:
- **derives** each run's expected identity from it (`expected_identity`);
- **runs** each run under the lifecycle, with that identity bound into the
  run's own record;
- **hands analysis** the same derived identities.

Nothing is caller-supplied. Every run's identity carries the sha256 of the
`FREEZE.json` it was made under. A run made under another freeze is
therefore never admitted, even when its bytes are valid and self-consistent.

  E1  transformers family: A32, A32 again (A32-rep2), A32s, A32r, A64, on
      each input; logits only
  E2  LARQL family: C, D, C32a, D32a cached; C, D batched prefill
      (all positions); on each input
  E3  captures at the frozen positions: C and D (LARQL); A32 and A64
  N1  control (hf-f64-accuracy/2): Dauto = production CPU from the `auto`
      representation source, cached, on every input; read only by
      `control_runs`
      (transformers); on I_rep and I_nr1

Usage:
  experiment.py run E1|E2|E3|N1 --freeze FREEZE.json --freeze-sha256 HEX --root DIR
  experiment.py readings --freeze FREEZE.json --freeze-sha256 HEX --root DIR

`readings` runs `analysis.analyse` over every run in the three experiments.
The V0 bridge is separate (`v0.py`).

The operation-level dtype audit (A64 audit part 5, `op_audit.py`, #897) is a
freeze decision, never a code default, so runs made under an older freeze
keep the identity they were made with and reports re-read them unchanged.
An experiment with A64 runs refuses to start unless the freeze decides
explicitly (`op_audit_policy`):
- `values.a64_op_audit` = the audit's schema (`op_audit.SCHEMA`): every A64
  run is made with `--op-audit SCHEMA` (the oracle refuses a schema it does
  not implement), binds `op_audit.py` and `op_probes.py`, and carries the
  schema in its identity, so a run made without it is never admitted;
- `values.a64_op_audit` = "waived", with a reason in
  `values.a64_op_audit_waiver`: allowed only for a family whose backfill is
  reported (`OP_AUDIT_BACKFILL`), and recorded as a preflight warning.
Leaving the value out refuses the experiment, so a new freeze cannot forget
the audit silently.

This module names no schema itself and imports nothing new: the reports
(`f64_accuracy_v2.py`) import it, and a reporting repair binds exactly the
modules they import.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prec_diag import analysis  # noqa: E402
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import families as fam  # noqa: E402
from prec_diag import freeze as fz  # noqa: E402
from prec_diag import v0  # noqa: E402

INPUTS = ("I_rep", "I_nr1", "I_nr2")
CAPTURE_INPUTS = ("I_rep", "I_nr1")
LARQL_ARMS = {"C": ["--backend", "reference"],
              "D": ["--backend", "production", "--representation-source", "stored"],
              # N1 control only: production CPU from the `auto` source, which may
              # build a lossy (Q8) residency at load. Never an admission path.
              "Dauto": ["--backend", "production", "--representation-source", "auto"]}
#: Control experiments: run on demand, read only by `control_runs`, never by
#: `runs_and_identities` (so PREC-DIAG readings and v1/v2 admission never see them).
CONTROL_EXPERIMENTS = ("N1",)
#: The experiments `runs_and_identities` reads unless told otherwise.
READING_EXPERIMENTS = ("E1", "E2", "E3")
#: The modeling file A64 patches, and the thread settings, are the oracle's own constants.
HF_THREADS = {"torch_num_threads": 8, "torch_num_interop_threads": 1, "OMP_NUM_THREADS": "8"}
HF_ENVIRONMENT = {"OMP_NUM_THREADS": "8"}
#: The freeze value that decides A64's operation-level audit, its waiver, and
#: the files a freeze that requires it must bind.
OP_AUDIT_VALUE = "a64_op_audit"
OP_AUDIT_WAIVED = "waived"
OP_AUDIT_WAIVER = "a64_op_audit_waiver"
OP_AUDIT_FILES = {"script.op_audit": "op_audit.py", "script.op_probes": "op_probes.py"}
#: Families whose A64 forward has had the audit backfilled (#897), and where it
#: is reported. Only these may waive it; a family added later never can.
OP_AUDIT_BACKFILL_REPORT = "docs/prec-diag-a64-op-audit-backfill.md"
OP_AUDIT_BACKFILL = {"gemma3_text": OP_AUDIT_BACKFILL_REPORT, "qwen3": OP_AUDIT_BACKFILL_REPORT}
#: FREEZE.json values this runner needs beyond freeze.REQUIRED_*.
REQUIRED_VALUES = ("larql_environment", "python", "capture_positions", "resource_limits", "packages", "threads")


def plan(experiment: str) -> list[dict]:
    """The runs of one experiment, in execution order."""
    runs = []
    if experiment == "E1":
        for i in INPUTS:
            for arm, base in (("A32", "A32"), ("A32-rep2", "A32"), ("A32s", "A32s"), ("A32r", "A32r"),
                              ("A64", "A64")):
                runs.append({"run_id": f"E1.{arm}.{i}", "kind": "hf", "arm": arm, "base": base, "input": i,
                             "capture": False})
    elif experiment == "E2":
        for i in INPUTS:
            for b in ("C", "D"):
                runs.append({"run_id": f"E2.{b}.{i}", "kind": "larql", "arm": b, "base": b, "input": i,
                             "mode": "cached", "rope_angles": "larql", "capture": False})
                runs.append({"run_id": f"E2.{b}32a.{i}", "kind": "larql", "arm": f"{b}32a", "base": b,
                             "input": i, "mode": "cached", "rope_angles": "hf-f32", "capture": False})
                runs.append({"run_id": f"E2.{b}-batched.{i}", "kind": "larql", "arm": f"{b}-batched", "base": b,
                             "input": i, "mode": "batched", "rope_angles": "larql", "capture": False})
    elif experiment == "N1":
        for i in INPUTS:
            runs.append({"run_id": f"N1.Dauto.{i}", "kind": "larql", "arm": "Dauto", "base": "Dauto", "input": i,
                         "mode": "cached", "rope_angles": "larql", "capture": False})
    elif experiment == "E3":
        for i in CAPTURE_INPUTS:
            for b in ("C", "D"):
                runs.append({"run_id": f"E3.{b}-capture.{i}", "kind": "larql", "arm": f"{b}-capture", "base": b,
                             "input": i, "mode": "cached", "rope_angles": "larql", "capture": True})
            for a in ("A32", "A64"):
                runs.append({"run_id": f"E3.{a}-capture.{i}", "kind": "hf", "arm": f"{a}-capture", "base": a,
                             "input": i, "capture": True})
    else:
        raise ValueError(f"unknown experiment {experiment!r}")
    return runs


# ------------------------------------------------------------ the freeze

def load_freeze(path: Path, expected_sha256: str) -> dict:
    """The manifest, refused unless its bytes are the reviewed ones and it
    names everything this runner derives identities from."""
    raw = Path(path).read_bytes()
    if ev.sha256_bytes(raw) != expected_sha256:
        raise ev.EvidenceError("FREEZE.json differs from the reviewed digest")
    manifest = json.loads(raw)
    if manifest.get("schema") != fz.SCHEMA:
        raise ev.EvidenceError(f"not a freeze manifest: {manifest.get('schema')!r}")
    missing = [k for k in REQUIRED_VALUES if k not in manifest["values"]]
    missing += [lt for lt in rope_table_labels(manifest) if lt not in manifest["files"]]
    missing += [label for label in ("script.families",) if label not in manifest["files"]]
    if op_audit_required(manifest):
        missing += [label for label in OP_AUDIT_FILES if label not in manifest["files"]]
    if missing:
        raise ev.EvidenceError(f"FREEZE.json lacks what the runner needs: {missing}")
    return manifest


def op_audit_schema(manifest: dict) -> str | None:
    """The schema of A64's operation-level audit (part 5) the freeze requires, if any."""
    value = manifest["values"].get(OP_AUDIT_VALUE)
    return value if isinstance(value, str) and value and value != OP_AUDIT_WAIVED else None


def op_audit_required(manifest: dict) -> bool:
    return op_audit_schema(manifest) is not None


def runs_a64(experiment: str) -> bool:
    return any(r["kind"] == "hf" and r["base"] == "A64" for r in plan(experiment))


def op_audit_policy(manifest: dict, experiment: str) -> tuple[list[str], list[str]]:
    """(problems, warnings) of the freeze's decision on A64's operation-level
    audit, for an experiment about to run. Only experiments with A64 runs are
    concerned; readings never call this, so runs made before the audit
    existed are read exactly as before."""
    if not runs_a64(experiment):
        return [], []
    value = manifest["values"].get(OP_AUDIT_VALUE)
    if op_audit_required(manifest):
        missing = [label for label in OP_AUDIT_FILES if label not in manifest["files"]]
        return [f"values.{OP_AUDIT_VALUE} requires the audit, and the freeze binds no {missing}"] if missing else [], []
    if value == OP_AUDIT_WAIVED:
        family = fam.family_of(checkpoint_config(manifest))
        backfill = OP_AUDIT_BACKFILL.get(family)
        reason = manifest["values"].get(OP_AUDIT_WAIVER)
        problems = []
        if not (isinstance(reason, str) and reason.strip()):
            problems.append(f"values.{OP_AUDIT_VALUE} is waived with no reason in values.{OP_AUDIT_WAIVER}")
        if not backfill:
            problems.append(f"{family} has no operation-level backfill: its A64 runs cannot waive the audit")
        warning = (f"{experiment}: A64 runs without the operation-level dtype audit (waived: {reason}); "
                   f"{family} backfill: {backfill}")
        return problems, [] if problems else [warning]
    if value is None:
        return [f"FREEZE.json sets no values.{OP_AUDIT_VALUE}: {experiment}'s A64 runs need the operation-level "
                f"dtype audit (its schema, op_audit.SCHEMA) or an explicit waiver ({OP_AUDIT_WAIVED!r} with "
                f"values.{OP_AUDIT_WAIVER})"], []
    return [f"values.{OP_AUDIT_VALUE} = {value!r}: expected the audit's schema or {OP_AUDIT_WAIVED!r}"], []


def checkpoint_config(manifest: dict) -> dict:
    return json.loads((Path(manifest["checkpoint"]["directory"]) / "config.json").read_text())


def layer_types(manifest: dict) -> list[str]:
    return sorted(set(config_layer_types(checkpoint_config(manifest))))


config_layer_types = fam.config_layer_types


def modeling_identity(manifest: dict) -> dict:
    """The transformers modeling file the oracle must find for this checkpoint's family."""
    family = fam.family_of(checkpoint_config(manifest))
    return {"family": family, "sha256": fam.FAMILIES[family]["modeling_sha256"]}


def rope_table_labels(manifest: dict) -> list[str]:
    return [f"rope_table.{lt}.{t}" for lt in layer_types(manifest) for t in ("cos", "sin")]


def ids_of(manifest: dict, input_name: str) -> list[int]:
    entry = manifest["files"][f"ids.{input_name}"]
    raw = Path(entry["path"]).read_bytes()
    if ev.sha256_bytes(raw) != entry["sha256"]:
        raise ev.EvidenceError(f"ids.{input_name} differs from the freeze")
    value = json.loads(raw)
    return value["ids"] if isinstance(value, dict) else value


def vocab_of(manifest: dict) -> int:
    graph = json.loads((Path(manifest["container"]["path"]) / "system_graph.json").read_text())
    primary = [c for c in graph["components"] if c.get("role") == "primary_text"]
    return int(primary[0]["execution"]["head"]["vocab_size"])


def file_sha(manifest: dict, label: str) -> str:
    return manifest["files"][label]["sha256"]


def capture_positions_of(manifest: dict, run: dict) -> list[int]:
    """The frozen capture positions of a capture run, refused (never guessed)
    when the freeze names none for its input."""
    positions = manifest["values"]["capture_positions"].get(run["input"])
    if positions is None:
        raise ev.EvidenceError(f"{run['run_id']}: FREEZE.json values.capture_positions names no positions for "
                               f"{run['input']}, so this capture run has no identity")
    return positions


def expected_identity(manifest: dict, freeze_sha256: str, run: dict, experiment: str) -> dict:
    """Every identity field a run must carry, derived from the freeze alone."""
    ids = ids_of(manifest, run["input"])
    bind = {"experiment": experiment, "run_id": run["run_id"], "freeze_sha256": freeze_sha256}
    if run["kind"] == "larql":
        return {**bind, "tool": "larql", "arm": run["arm"], "input": run["input"],
                "binary": {"sha256": file_sha(manifest, "larql_binary")},
                "container_index_sha256": manifest["container"]["index_json_sha256"],
                "input_ids_json_sha256": v0.ids_json_sha256(ids),
                "environment": manifest["values"]["larql_environment"], "rope_angles": run["rope_angles"],
                "mode": run["mode"], "capture": run["capture"]}
    positions = capture_positions_of(manifest, run) if run["capture"] else None
    identity = {"tool": "hf_oracle", "arm": run["base"], "bind": bind,
             "scripts": {"hf_oracle.py": file_sha(manifest, "script.hf_oracle"),
                         "evidence.py": file_sha(manifest, "script.evidence"),
                         "families.py": file_sha(manifest, "script.families"),
                         "oracle_loading.py": file_sha(manifest, "script.oracle_loading")},
             "checkpoint": {"config_sha256": manifest["checkpoint"]["config_json_sha256"],
                            "safetensors": manifest["checkpoint"]["safetensors"]},
             "input": {"ids_json_sha256": v0.ids_json_sha256(ids), "length": len(ids)},
             "positions": positions, "modeling": modeling_identity(manifest),
             "threads": manifest["values"]["threads"], "environment": HF_ENVIRONMENT,
             "packages": manifest["values"]["packages"]}
    if run["base"] == "A64" and op_audit_required(manifest):
        identity["op_audit"] = op_audit_schema(manifest)
        identity["scripts"].update({name: file_sha(manifest, label) for label, name in OP_AUDIT_FILES.items()})
    return identity


# -------------------------------------------------------------- preflight

def frozen_world_problems(manifest_path: Path, freeze_sha256: str) -> list[str]:
    """Everything that must still be the reviewed world, at execution AND at
    readings: every frozen file, container and checkpoint digest
    (`freeze.preflight`), every script actually running (this process's own
    modules, against their frozen labels), and the frozen thread settings."""
    problems = fz.preflight(manifest_path, freeze_sha256)
    manifest = json.loads(Path(manifest_path).read_bytes())
    here = Path(__file__).resolve().parent
    own = {"script.experiment": here / "experiment.py", "script.v0": here / "v0.py",
           "script.evidence": here / "evidence.py", "script.analysis": here / "analysis.py",
           "script.freeze": here / "freeze.py", "script.hf_oracle": here / "hf_oracle.py",
           "script.families": here / "families.py",
           "script.oracle_loading": here.parent / "oracle_loading.py",
           **{label: here / name for label, name in OP_AUDIT_FILES.items()},
           "script.equivalence_metrics": here.parent / "equivalence_metrics.py"}
    for label, path in own.items():
        if label in manifest["files"] and ev.sha256_file(path) != manifest["files"][label]["sha256"]:
            problems.append(f"the running {path.name} is not the frozen {label}")
    if manifest["values"].get("threads") != HF_THREADS:
        problems.append("frozen thread settings differ from the oracle's")
    return problems


class EvidenceLimitError(ValueError):
    pass


def total_limit(manifest: dict, experiment: str) -> float:
    """The experiment's own wall-clock total (C11): `wall_s_total` maps each
    experiment to its seconds. The clock starts after a passing preflight and
    covers every run of that invocation, failed ones included; it is checked
    before each run starts, and a run that has started runs to completion
    under `wall_s_per_run`."""
    totals = manifest["values"].get("resource_limits", {}).get("wall_s_total")
    if not isinstance(totals, dict):
        raise EvidenceLimitError("resource_limits.wall_s_total must map each experiment to its seconds")
    value = totals.get(experiment)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise EvidenceLimitError(f"resource_limits.wall_s_total has no positive limit for {experiment}")
    return value


def preflight(manifest_path: Path, freeze_sha256: str, root: Path, experiment: str) -> list[str]:
    """Before an experiment: the frozen world, plus run-once and disk."""
    problems = frozen_world_problems(manifest_path, freeze_sha256)
    manifest = json.loads(Path(manifest_path).read_bytes())
    problems += op_audit_policy(manifest, experiment)[0]
    family = fam.family_of(checkpoint_config(manifest))
    if any(r["capture"] for r in plan(experiment)) and not fam.FAMILIES[family]["captures"]:
        problems.append(f"{experiment} captures sites, and the oracle maps no sites for {family}")
    if (root / experiment).exists():
        problems.append(f"{root / experiment} exists: an experiment's runs are made once")
    limits = manifest["values"].get("resource_limits", {})
    try:
        total_limit(manifest, experiment)
    except EvidenceLimitError as e:
        problems.append(str(e))
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    if free < limits.get("min_free_bytes", 0):
        problems.append(f"free disk {free} < {limits.get('min_free_bytes')}")
    return problems


# ------------------------------------------------------------------ runs

def larql_argv(manifest: dict, run: dict, run_dir: Path, ids: list[int]) -> list[str]:
    argv = [manifest["files"]["larql_binary"]["path"], "vindex3", "exec", manifest["container"]["path"],
            *LARQL_ARMS[run["base"]], "--tokens", ",".join(map(str, ids))]
    if run["rope_angles"] != "larql":
        argv += ["--rope-angles", run["rope_angles"]]
    if run["mode"] == "batched":
        argv += ["--prefill-logit-dump", str(run_dir / v0.LOGITS),
                 "--prefill-lengths", ",".join(str(n) for n in range(1, len(ids) + 1)),
                 "--prefill-method", "batched"]
    else:
        argv += ["--logit-dump", str(run_dir / v0.LOGITS)]
    if run["capture"]:
        positions = capture_positions_of(manifest, run)
        argv += ["--capture-sites", str(run_dir / v0.SITES_DIR), "--capture-positions",
                 ",".join(map(str, positions)),
                 "--capture-budget-bytes", str(manifest["values"]["resource_limits"]["capture_bytes_per_run"])]
    return argv


def v0_view(manifest: dict, run: dict, ids: list[int]) -> dict:
    """The fields `v0.verify_logits`/`verify_capture` read, from the freeze."""
    return {"input": {"length": len(ids), "ids_u32le_sha256": v0.ids_u32le_sha256(ids)},
            "vocab": vocab_of(manifest),
            "capture": {"positions": manifest["values"]["capture_positions"].get(run["input"]),
                        "geometry": v0.frozen_geometry(Path(manifest["container"]["path"])),
                        "layers": len(v0.frozen_geometry(Path(manifest["container"]["path"]))["layers"])},
            "binaries": {"experimental": {"sha256": file_sha(manifest, "larql_binary")}},
            "container": {"index_json_sha256": manifest["container"]["index_json_sha256"]},
            "environment": manifest["values"]["larql_environment"]}


def run_larql(manifest: dict, freeze_sha256: str, run: dict, run_dir: Path, experiment: str) -> dict:
    ids = ids_of(manifest, run["input"])
    identity = {**expected_identity(manifest, freeze_sha256, run, experiment), "os_build": v0.os_build()}
    rr = ev.RunRecord.begin(run_dir, run["run_id"], "prec-diag-1-larql", identity)
    limits = manifest["values"]["resource_limits"]
    try:
        done = v0.spawn(larql_argv(manifest, run, run_dir, ids), dict(manifest["values"]["larql_environment"]),
                        run_dir, limits["wall_s_per_run"])
    except (OSError, ValueError) as e:
        return rr.fail(ev.FAULT, {"spawn": f"{type(e).__name__}: {e}"})
    measured = {"wall_s": done["wall_s"], "peak_rss_bytes": done["peak_rss_bytes"],
                "resource_measure": v0.RSS_METHOD}
    if done["timed_out"]:
        return rr.fail(ev.RESOURCE_LIMIT, {"limit": "wall_s_per_run"}, **measured)
    if done["exit"] != 0:
        return rr.fail(ev.FAULT, {"exit": done["exit"]}, **measured)
    if done["peak_rss_bytes"] > limits["rss_bytes"]:
        return rr.fail(ev.RESOURCE_LIMIT, {"limit": "rss_bytes", "value": done["peak_rss_bytes"]}, **measured)
    try:
        view = v0_view(manifest, run, ids)
        vrun = {"binary": "experimental", "arm": run["base"], "capture": run["capture"]}
        if run["mode"] == "batched":
            size = (run_dir / v0.LOGITS).stat().st_size if (run_dir / v0.LOGITS).is_file() else None
            want = len(ids) * view["vocab"] * v0.F32
            problems = [] if size == want else [f"{v0.LOGITS} is {size} bytes, expected {want}"]
            outputs = [v0.LOGITS, "stdout.txt", "stderr.txt"]
        else:
            problems = v0.verify_logits(view, vrun, run_dir, rope_angles=run["rope_angles"])
            outputs = [v0.LOGITS, v0.LOGITS_META, "stdout.txt", "stderr.txt"]
        if run["capture"] and not problems:
            problems += v0.verify_capture(view, vrun, run_dir)
            outputs += v0.capture_outputs(run_dir)
        if problems:
            return rr.fail(ev.FAULT, {"verification": problems}, **measured)
        return rr.complete(outputs, nonfinite_logits=v0.count_nonfinite(run_dir / v0.LOGITS, view["vocab"]),
                           **measured)
    except ev.EvidenceError:
        raise
    except Exception as e:
        return rr.fail(ev.FAULT, {"verification_error": f"{type(e).__name__}: {e}"}, **measured)


def run_hf(manifest: dict, freeze_sha256: str, run: dict, run_dir: Path, experiment: str) -> dict:
    """`hf_oracle.py` in its own process (fixed environment, fixed threads).
    It owns its run directory and record; the launcher's own output goes
    beside it in `<run id>.launcher/`."""
    launcher = run_dir.with_name(run_dir.name + ".launcher")
    launcher.mkdir(parents=True)
    expected = expected_identity(manifest, freeze_sha256, run, experiment)
    (launcher / "bind.json").write_text(json.dumps(expected["bind"]))
    (launcher / "ids.json").write_text(json.dumps(ids_of(manifest, run["input"])))
    argv = [manifest["values"]["python"], "-I", str(Path(__file__).resolve().parent / "hf_oracle.py"),
            "--arm", run["base"], "--checkpoint", manifest["checkpoint"]["directory"],
            "--ids", str(launcher / "ids.json"), "--boundary", manifest["files"]["loading_boundary"]["path"],
            "--out", str(run_dir), "--bind-json", str(launcher / "bind.json")]
    if run["capture"]:
        argv += ["--positions", ",".join(map(str, expected["positions"]))]
    if run["base"] == "A64" and op_audit_required(manifest):
        argv += ["--op-audit", op_audit_schema(manifest)]
    if run["base"] == "A32r":
        for lt in layer_types(manifest):
            argv += ["--tables", f"{lt}={Path(manifest['files'][f'rope_table.{lt}.cos']['path']).parent}"]
    env = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin", **HF_ENVIRONMENT, "HF_HUB_OFFLINE": "1"}
    done = v0.spawn(argv, env, launcher, manifest["values"]["resource_limits"]["wall_s_per_run"])
    record_path = run_dir / ev.RESULT_FILE
    if not record_path.is_file():
        rr = ev.RunRecord.begin(run_dir, run["run_id"], "hf-oracle", expected)
        status = ev.RESOURCE_LIMIT if done["timed_out"] else ev.FAULT
        return rr.fail(status, {"launcher": done, "reason": "the oracle wrote no record"})
    record = json.loads(record_path.read_text())
    status = ev.RESOURCE_LIMIT if done["timed_out"] else ev.FAULT
    if record.get("status") == ev.IN_PROGRESS:  # killed at the limit, or died: finalise it here
        return ev.RunRecord(run_dir, record).fail(status, {"launcher": done})
    if done["timed_out"] or done["exit"] != 0:
        # A final record is not enough: the process must also have ended
        # cleanly. The oracle's directory is kept beside the run, and the
        # run itself is recorded as failed, so admission can never take it.
        kept = run_dir.with_name(f"{run_dir.name}.launch-failed")
        run_dir.rename(kept)
        rr = ev.RunRecord.begin(run_dir, run["run_id"], "hf-oracle", expected)
        return rr.fail(status, {"launcher": done, "oracle_record_status": record.get("status"),
                                "oracle_record_kept_at": kept.name,
                                "reason": "the oracle process did not end cleanly"})
    return record


def run_experiment(manifest_path: Path, freeze_sha256: str, root: Path, experiment: str) -> list[dict]:
    manifest = load_freeze(manifest_path, freeze_sha256)
    root.mkdir(parents=True, exist_ok=True)
    attempt = len(sorted(root.glob(f"{experiment}.preflight-*"))) + 1
    pre = ev.RunRecord.begin(root / f"{experiment}.preflight-{attempt}", f"{experiment}-preflight",
                             "prec-diag-1-preflight", {"freeze_sha256": freeze_sha256, "experiment": experiment})
    problems = preflight(manifest_path, freeze_sha256, root, experiment)
    if problems:
        pre.fail(ev.ABORTED_PREFLIGHT, problems)
        return [pre.record]
    warnings = op_audit_policy(manifest, experiment)[1]
    (pre.run_dir / "checks.json").write_text(json.dumps({"problems": [], "warnings": warnings}))
    pre.complete(["checks.json"])
    records, started = [], time.monotonic()
    total = total_limit(manifest, experiment)
    for run in plan(experiment):
        run_dir = root / experiment / run["run_id"]
        if time.monotonic() - started > total:
            rr = ev.RunRecord.begin(run_dir, run["run_id"], "prec-diag-1",
                                    expected_identity(manifest, freeze_sha256, run, experiment))
            records.append(rr.fail(ev.RESOURCE_LIMIT, {"limit": "wall_s_total"}))
            continue
        go = run_larql if run["kind"] == "larql" else run_hf
        records.append(go(manifest, freeze_sha256, run, run_dir, experiment))
    return records


# --------------------------------------------------------------- readings

def runs_and_identities(manifest: dict, freeze_sha256: str, root: Path,
                        experiments: tuple[str, ...] = READING_EXPERIMENTS) -> tuple[dict, dict]:
    """Analysis's two inputs, keyed `(arm, input)`: the run directories
    where each run should be, and the identity each must carry.

    `experiments` names which experiments to read. PREC-DIAG readings read all
    three (the default). A report whose test reads only some of them names
    those, so an experiment outside its scope (E3 for `hf-f64-accuracy`) is
    never looked at. A named experiment that cannot be identified is refused."""
    unknown = [e for e in experiments if e not in READING_EXPERIMENTS]
    if unknown or not experiments:
        raise ValueError(f"experiments must be a non-empty subset of {READING_EXPERIMENTS}: {experiments!r}")
    runs, expected = {}, {}
    for experiment in experiments:
        for run in plan(experiment):
            if run["arm"] == "A32-rep2":
                continue  # consumed by the repeatability reading, not by analysis
            key = (run["arm"], run["input"])
            runs[key] = root / experiment / run["run_id"]
            expected[key] = expected_identity(manifest, freeze_sha256, run, experiment)
    return runs, expected


def control_runs(manifest: dict, freeze_sha256: str, root: Path, experiment: str) -> tuple[dict, dict]:
    """A control experiment's runs and identities, keyed `(arm, input)`."""
    if experiment not in CONTROL_EXPERIMENTS:
        raise ValueError(f"not a control experiment: {experiment!r}")
    runs, expected = {}, {}
    for run in plan(experiment):
        key = (run["arm"], run["input"])
        runs[key] = root / experiment / run["run_id"]
        expected[key] = expected_identity(manifest, freeze_sha256, run, experiment)
    return runs, expected


def repeatability(manifest: dict, freeze_sha256: str, root: Path, log: ev.ResultsLog) -> list[dict]:
    """Validity check: A32 run twice is bit-identical (rev 3 "Validity checks")."""
    out = []
    by_id = {r["run_id"]: r for r in plan("E1")}
    for i in INPUTS:
        a, b = by_id[f"E1.A32.{i}"], by_id[f"E1.A32-rep2.{i}"]
        adm = [ev.admit(root / "E1" / r["run_id"], expected_identity(manifest, freeze_sha256, r, "E1"))
               for r in (a, b)]
        if all(x.admitted for x in adm):
            values = v0.first_difference(root / "E1" / a["run_id"] / "logits.f32",
                                         root / "E1" / b["run_id"] / "logits.f32", vocab_of(manifest))
            entry = ev.reading(f"Validity A32 repeatability {i}", ev.READING_COMPLETE, adm, values,
                               evidence_class="E1")
        else:
            entry = ev.reading(f"Validity A32 repeatability {i}", ev.NOT_ADMITTED, adm)
        out.append(log.append(entry))
    return out


def readings(manifest_path: Path, freeze_sha256: str, root: Path) -> dict:
    """Readings over E1–E3. Refused, before anything is written, unless the
    frozen world still holds: the runs may be intact while the analysis that
    reads them has drifted. (Unlike `preflight`, the experiment directories
    must exist here.)"""
    manifest = load_freeze(manifest_path, freeze_sha256)
    problems = frozen_world_problems(manifest_path, freeze_sha256)
    if problems:
        raise ev.EvidenceError(f"readings refused, the frozen world has drifted: {problems}")
    log_path = root / "PREC-DIAG-1.results.jsonl"
    if any(not e.get("reading", "").startswith("V0-") for e in ev.ResultsLog(log_path).verify()):
        raise ev.EvidenceError(f"{log_path} already holds PREC-DIAG-1 readings: they are recorded once")
    runs, expected = runs_and_identities(manifest, freeze_sha256, root)
    repeatability(manifest, freeze_sha256, root, ev.ResultsLog(log_path))
    attempt = len(sorted(root.glob("analysis-*"))) + 1
    return analysis.analyse(runs=runs, expected=expected, vocab=vocab_of(manifest),
                            out_dir=root / f"analysis-{attempt}", log_path=log_path, freeze_sha256=freeze_sha256)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "readings"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("experiment", choices=("E1", "E2", "E3", *CONTROL_EXPERIMENTS))
        p.add_argument("--freeze", required=True, type=Path)
        p.add_argument("--freeze-sha256", required=True)
        p.add_argument("--root", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        for rec in run_experiment(a.freeze, a.freeze_sha256, a.root, a.experiment):
            print(rec["run_id"], rec["status"])
    else:
        print(readings(a.freeze, a.freeze_sha256, a.root)["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
