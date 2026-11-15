"""Host telemetry for timing runs (BLOCK-TIME-2.md, "Host controls").

Observational only: nothing here voids, excludes or reorders a trial, and no
threshold is applied to what it records. It exists so that a disturbed trial
can be attributed to something observed rather than left as "cause not
identified".

Parsing is pure and tested; `Sampler` and `spotlight_starts` run commands.
"""
import calendar
from collections import Counter
import json
import re
import subprocess
import threading
import time

PROCESS_INTERVAL_S = 5.0
STATE_INTERVAL_S = 30.0
TOP_PROCESSES = 10
COMMAND_TIMEOUT_S = 10
LOG_TIMEOUT_S = 600
PS = ["ps", "-Ao", "pid=,pcpu=,rss=,comm="]
THERMAL = ["pmset", "-g", "therm"]
SYSCTL_KEYS = ("vm.memory_pressure", "kern.memorystatus_level")
UNIFIED_LOG = "/usr/bin/log"  # zsh's `log` is a builtin; always the binary
SPOTLIGHT_PREDICATE = 'process == "launchd" AND eventMessage CONTAINS "spawned mdworker"'
LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} (\d{2}):(\d{2}):\d{2}")


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_ps(text, top=TOP_PROCESSES):
    """Top processes by CPU share, and the total over every process."""
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        pid, cpu, rss, command = parts
        try:
            rows.append({"pid": int(pid), "cpu_percent": float(cpu), "rss_kb": int(rss), "command": command})
        except ValueError:
            continue
    rows.sort(key=lambda row: row["cpu_percent"], reverse=True)
    return {"total_cpu_percent": sum(row["cpu_percent"] for row in rows), "processes": len(rows),
            "top": rows[:top]}


def parse_thermal(text):
    """`pmset -g therm`: the recorded warning levels, or a note that none is recorded."""
    levels = {}
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^(.*?level)\s*=\s*(\S+)", line, re.IGNORECASE)
        if match:
            levels[match.group(1).strip()] = match.group(2)
        elif line.lower().startswith("note:"):
            levels.setdefault("notes", []).append(line[5:].strip())
    return levels


def parse_sysctl(text):
    values = {}
    for line, key in zip(text.splitlines(), SYSCTL_KEYS):
        try:
            values[key] = int(line.strip())
        except ValueError:
            values[key] = line.strip()
    return values


def count_per_minute(log_text):
    """Unified-log lines per minute of the log's own timestamps (local time), `HH:MM` to count."""
    counts = Counter()
    for line in log_text.splitlines():
        match = LOG_LINE.match(line)
        if match:
            counts[f"{match.group(1)}:{match.group(2)}"] += 1
    return dict(sorted(counts.items()))


def run(command, timeout=COMMAND_TIMEOUT_S):
    started = time.monotonic()
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, time.monotonic() - started, str(error)
    return output, time.monotonic() - started, None


class Sampler:
    """Samples processes every 5 s and host state every 30 s into a JSONL file, in a thread."""

    def __init__(self, path, process_interval=PROCESS_INTERVAL_S, state_interval=STATE_INTERVAL_S):
        self.path, self.process_interval, self.state_interval = path, process_interval, state_interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.samples = 0

    def write(self, record):
        with self.path.open("a") as out:
            out.write(json.dumps(record) + "\n")
        self.samples += 1

    def loop(self):
        last_state = None
        while not self.stop_event.is_set():
            text, cost, error = run(PS)
            self.write({"kind": "processes", "utc": utc_now(), "cost_s": cost, "error": error,
                        **(parse_ps(text) if text is not None else {})})
            if last_state is None or time.monotonic() - last_state >= self.state_interval:
                last_state = time.monotonic()
                thermal, thermal_cost, thermal_error = run(THERMAL)
                sysctl, sysctl_cost, sysctl_error = run(["sysctl", "-n", *SYSCTL_KEYS])
                self.write({"kind": "state", "utc": utc_now(), "cost_s": thermal_cost + sysctl_cost,
                            "error": thermal_error or sysctl_error,
                            "thermal": parse_thermal(thermal) if thermal is not None else None,
                            "memory": parse_sysctl(sysctl) if sysctl is not None else None})
            self.stop_event.wait(self.process_interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join(timeout=COMMAND_TIMEOUT_S * 3)
        return False


def spotlight_starts(start_utc, end_utc):
    """Spotlight worker starts per minute between two UTC times, from the unified log."""
    def local(stamp):
        # `log show` takes local time; the run records UTC.
        epoch = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))
    text, cost, error = run([UNIFIED_LOG, "show", "--start", local(start_utc), "--end", local(end_utc),
                             "--style", "compact", "--predicate", SPOTLIGHT_PREDICATE], timeout=LOG_TIMEOUT_S)
    return {"start_utc": start_utc, "end_utc": end_utc, "error": error, "cost_s": cost,
            "clock": "log timestamps are local time",
            "per_minute": count_per_minute(text) if text is not None else None}
