import type { Sample } from "../lib/record";

export function EvidenceLevels({ selected, answer, readoutLabel }: { selected?: Sample; answer: string; readoutLabel: string }) {
  const read = selected?.logits[answer];
  const write = selected?.write_logits?.[answer];
  return <div className="evidence-levels" aria-label="Readability, directional write, and causal evidence">
    <div><span className="evidence-dash">┄</span><h3>READABLE</h3><strong>{read === undefined ? "Unavailable" : read.toFixed(3)}</strong><p>{answer || "No probe selected"} · {readoutLabel}</p></div>
    <div><span className="evidence-dash">→</span><h3>WRITTEN</h3><strong>{write === undefined ? "Unavailable" : `${write >= 0 ? "+" : ""}${write.toFixed(3)}`}</strong><p>Applied write in the same reader direction</p></div>
    <div className="causal-unavailable"><span className="evidence-dash">∅</span><h3>CAUSAL</h3><strong>Not tested</strong><p>No intervention / control witness</p></div>
  </div>;
}
