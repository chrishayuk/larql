import type { Recording } from "../lib/record";

/** Run-wide realization classes; no invented assignment to a selected operator. */
export function ExecutionProvenance({ record }: { record: Recording }) {
  if (!record.standard) return null;
  const execution = record.execution;
  return <details className="execution-provenance">
    <summary>{execution ? `${execution.lowering.join(" · ")} · ${execution.arithmetic_arm}` : "Execution provenance unavailable"}<span>RUN-WIDE PROVENANCE +</span></summary>
    {execution && <div>
      <p>Recorded for the prepared image. These counts do not identify the realization of the selected operation.</p>
      <dl><dt>Execution fingerprint / runner declared</dt><dd>{record.standard.identity.lowering}</dd><dt>K-quant mode</dt><dd>{execution.kquant_execution}</dd><dt>Basis</dt><dd>{record.bases[0].id} · {record.bases[0].hash}</dd></dl>
      <table><caption>Pinned realization classes</caption><thead><tr><th>Stored representation</th><th>Codec</th><th>Executed form</th><th>Operands</th></tr></thead><tbody>{execution.realizations.map((r, i) => <tr key={i}><td>{r.representation}</td><td>{r.codec ?? "Not recorded"}</td><td>{r.form}</td><td>{r.operands}</td></tr>)}</tbody></table>
      <p>Observation method: {record.standard.norm_method} / {record.standard.probe_method}. {record.probeSource}</p>
      <p>Capture timing intrusive: {record.standard.timing_intrusive ? "yes" : "no"}. Realization names alone do not establish numerical equivalence.</p>
    </div>}
  </details>;
}
