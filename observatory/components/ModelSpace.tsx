"use client";
import type { Recording, Sample } from "../lib/record";
import { labelLayer, labelRole } from "../lib/record";

/** The declared fixture program exists independently of any recorded traversal. */
export function ModelSpace({ record, selectedKey, position, touched, select }: {
  record: Recording; selectedKey?: string; position: number; touched: Sample[];
  select: (key: string) => void;
}) {
  return <>
    <div className="field-caption"><div><p className="eyebrow">MODEL / WHAT EXISTS</p><h2>{record.vindex3 ? "Observed execution sites." : "Structure before traversal."}</h2></div></div>
    <p className="model-intro">{record.vindex3 ? "Sites explicitly present in the event log; this is not a full model inventory." : "The record’s declared operation graph."} Inspecting it does not play the recording or execute a model. Highlighted sites have been visited by the selected position.</p>
    <div className="model-lenses"><span>{record.vindex3 ? "OBSERVED SITES" : "OPERATION GRAPH"}</span><span title="Requires a runner-provided graph">Semantic graph · unavailable</span><span>Representations · unavailable</span><span>Placement · unavailable</span></div>
    <div className="model-program"><div className="model-embedding">EMBEDDING <span>→ carrier</span></div>{record.program.map(node => <div className="model-layer" key={node.layer}><span>{labelLayer(node.layer)}</span><div>{node.roles.map(role => {
      const key = `${position}:${node.layer}:${role}:main:residual`;
      const visited = touched.some(s => s.position === position && s.layer === node.layer && s.role === role);
      return <button key={role} className={`${visited ? "touched" : ""} ${key === selectedKey ? "selected" : ""}`} aria-pressed={key === selectedKey} onClick={() => select(key)}><small>{role === "attention_write" ? "CONNECTION / MIXING" : "TRANSFORMATION"}</small>{labelRole(role)}<span>→ carrier</span></button>;
    })}</div></div>)}<div className="model-embedding">OUTPUT HEAD <span>→ raw selected-token probes</span></div></div>
    <p className="trace-note">{record.vindex3 ? "Observed traversal only; the file carries no independent operation graph." : record.standard ? "Runner-declared program, not an independently verified model graph." : "Synthetic declared structure, not an extracted model graph."} FFN addressing and deterministic replacement remain research hypotheses until supplied with evidence.</p>
  </>;
}
