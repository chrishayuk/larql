import { readFile } from "node:fs/promises";
import { adaptStandard } from "../lib/standard.ts";
import { reduceEvents } from "../lib/record.ts";
import { auditStandard } from "../lib/replay-audit.ts";
import { createHash } from "node:crypto";

try {
  if (!process.argv[2] || process.argv.slice(3).some(arg => !["--require-executor", "--require-complete"].includes(arg))) throw new Error("Usage: node scripts/check-standard.mjs RECORD.json [--require-executor] [--require-complete]");
  const bytes = await readFile(process.argv[2]);
  if (bytes.length > 10_000_000) throw new Error("Record exceeds the UI's 10 MB import limit.");
  const source = JSON.parse(bytes.toString("utf8"));
  const record = adaptStandard(source);
  const audit = auditStandard(source, record);
  if (process.argv.includes("--require-executor") && record.provenance !== "executor") throw new Error("Real-record check requires runner-declared executor provenance; synthetic input refused.");
  if (process.argv.includes("--require-complete") && record.standard.coverage !== "complete") throw new Error("Lossless replay check requires complete declared carrier-write coverage.");
  const state = reduceEvents(record.events);
  console.log(JSON.stringify({ run_id: record.id, provenance: record.provenance, status: state.status,
    coverage: record.standard.coverage, writes: state.samples.length, events: state.events.length,
    dropped: state.dropped, gaps: state.gaps, probe_method: record.standard.probe_method,
    source_sha256: createHash("sha256").update(bytes).digest("hex"), audit }, null, 2));
} catch (error) {
  console.error(error.message); process.exitCode = 1;
}
