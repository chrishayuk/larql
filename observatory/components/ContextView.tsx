import { useState } from 'react';
import type { Recording, Sample, Role } from '../lib/record';
import { labelLayer, labelRole, semanticKey } from '../lib/record';
import { contextRows } from '../lib/context';
import type { ContextMetric } from '../lib/context';
import type { Lens, LensRecord } from '../lib/lenses';
import { specificity, vocabularyAt } from '../lib/lenses';
import { VocabularyBars } from './ResearchView';

export function ContextView({ record, samples, all, selected, position, role, answer, select, seek, setRole, inspect, lenses, openLens }: {
  record: Recording; samples: Sample[]; all: Sample[]; selected?: Sample; position: number;
  role: Role; answer: string; select: (s: Sample) => void; seek: (layer: number) => void;
  setRole: (role: Role) => void; inspect: () => void; lenses?: LensRecord; openLens: (lens: Lens) => void;
}) {
  const [metric, setMetric] = useState<ContextMetric | 'entropy' | 'specificity' | 'attention'>('norm');
  const [hover, setHover] = useState<number>();
  const [queryHead, setQueryHead] = useState(0);
  const [topK, setTopK] = useState(5);
  const layer = Math.max(0, selected?.layer ?? 0);
  const { rows, maximum } = contextRows(samples, all, record.tokens, layer, role, metric === 'norm' || metric === 'delta' || metric === 'probe' ? metric : 'norm', answer);
  const heads = lenses?.heads?.rows.filter(h => samples.some(s => semanticKey(s) === h.site && s.layer === layer && s.position === position)) ?? [];
  const head = heads.find(h => h.head === queryHead) ?? heads[0];
  const painted = rows.map(r => {
    const v = vocabularyAt(lenses, r.sample);
    const value = metric === 'entropy' ? v?.entropy : metric === 'specificity' ? v?.entropy !== undefined ? specificity(v.entropy, lenses!.vocabulary!.vocab_size) : undefined : metric === 'attention' ? head?.sources.find(s => s.position === r.position)?.weight : r.value;
    const scale = metric === 'entropy' ? Math.log(lenses?.vocabulary?.vocab_size ?? 2) : metric === 'specificity' || metric === 'attention' ? 1 : maximum;
    return { ...r, value, intensity: value === undefined ? undefined : Math.abs(value) / (scale || 1) };
  });
  const inspected = rows.find(r => r.position === (hover ?? position));
  const vocab = vocabularyAt(lenses, inspected?.sample);
  return <section className="context-view">
    <div className="field-caption"><div><p className="eyebrow">CONTEXT / ONE TOKEN, THROUGH DEPTH</p><h2>{record.prompt}</h2></div></div>
    <div className="context-controls">
      <label>Colour by <select value={metric} onChange={e => setMetric(e.target.value as typeof metric)}><option value="norm">Carrier norm</option><option value="delta">Applied-write norm</option><option value="probe" disabled={!record.answers.length}>{record.standard ? 'Raw head-row probe' : 'Fixture readout'}</option><option value="entropy" disabled={!lenses?.vocabulary?.rows.some(r => r.entropy !== undefined)}>Entropy</option><option value="specificity" disabled={!lenses?.vocabulary?.rows.some(r => r.entropy !== undefined)}>Specificity</option><option value="attention" disabled={!lenses?.heads}>Query attention</option></select></label>
      <label>Boundary <select value={role} onChange={e => setRole(e.target.value as Role)}><option value="attention_write">After attention / mixer</option><option value="ffn_write" disabled={!record.program[layer]?.roles.includes('ffn_write')}>After FFN</option></select></label>
      {metric === 'attention' && <label>Query {position} / head <select value={head?.head ?? 0} onChange={e => setQueryHead(Number(e.target.value))}>{[...new Set(heads.map(h => h.head))].map(h => <option key={h} value={h}>H{h}</option>)}</select></label>}
    </div>
    <label className="context-scrubber">Recorded layer <input type="range" min="0" max={record.layers - 1} value={layer} onChange={e => seek(Number(e.target.value))} /><strong>{labelLayer(layer)}</strong></label>
    <div className="context-document" aria-label={`Tokens at ${labelLayer(layer)}, ${labelRole(role)}`}>{painted.map(row => <button key={row.position} disabled={!row.sample} aria-pressed={row.position === position} onClick={() => row.sample && select(row.sample)} onMouseEnter={() => setHover(row.position)} onMouseLeave={() => setHover(undefined)} onFocus={() => setHover(row.position)} onBlur={() => setHover(undefined)} title={`Position ${row.position} · ${row.value === undefined ? 'not captured' : row.value.toPrecision(8)}`} style={{ backgroundColor: row.intensity === undefined ? 'transparent' : `rgba(178,201,156,${0.04 + row.intensity * .3})` }}><span>{row.token.replaceAll('Ġ', ' ').replaceAll('▁', ' ')}</span><small>{row.position} / {row.value === undefined ? 'unavailable' : row.value.toPrecision(4)}</small></button>)}</div>
    <p className="context-legend">{metric === 'entropy' ? 'Full-vocabulary entropy in nats / scale 0 → ln(vocabulary size).' : metric === 'specificity' ? 'Specificity = 1 − entropy / ln(vocabulary size). Concentration is not correctness.' : metric === 'attention' ? `Recorded source weights for query position ${position}, ${head ? `H${head.head}` : 'head unavailable'} / fixed scale 0 → 1. Weights are not causal contribution.` : `Magnitude: 0 → ${maximum.toPrecision(4)} · fixed scale across this run and boundary.`}{metric === 'probe' && ` Colour shows absolute magnitude; signed values are printed. Target: ${answer}.`}</p>
    {!!lenses?.spans?.length && <div className="semantic-spans"><p className="eyebrow">PROVIDER-ANNOTATED SPANS</p>{lenses.spans.map((s,i) => <button key={i} title={s.method} onClick={() => { const sample = rows[s.start]?.sample; if (sample) select(sample); }} disabled={!rows[s.start]?.sample}>{s.label}<small>{s.start}–{s.end} / {record.tokens.slice(s.start,s.end+1).join(' ')}</small></button>)}</div>}
    <div className="context-token-detail"><div><p className="eyebrow">{hover === undefined ? 'SELECTED' : 'PREVIEW'} / POSITION {inspected?.position}</p><h3>{inspected?.token}</h3><p className="lens-note">Carrier {inspected?.sample?.norm.toPrecision(5) ?? 'unavailable'} · write {inspected?.sample?.delta.toPrecision(5) ?? 'unavailable'}</p>{vocab?.entropy !== undefined && <p className="lens-note">Entropy {vocab.entropy.toFixed(4)} nats · specificity {specificity(vocab.entropy, lenses!.vocabulary!.vocab_size).toFixed(4)}</p>}<button className="text-action" onClick={() => openLens('Logits')}>Follow selected token through depth →</button><button className="text-action" onClick={() => openLens('Compare')}>Compare two boundaries →</button></div><div><label>Top <select value={topK} onChange={e => setTopK(Number(e.target.value))}>{[5,10,20].map(k => <option key={k}>{k}</option>)}</select> / vocabulary readout</label>{vocab ? <VocabularyBars rows={vocab.top} limit={topK} /> : <p className="lens-note">No normalized vocabulary readout at this position. Raw probes cannot supply top-k or entropy.</p>}</div></div>
    <button className="context-inspect" disabled={!selected} onClick={inspect}>Inspect this write →</button>
    <p className="context-capabilities">{lenses ? `${lenses.provenance === 'synthetic' ? 'Synthetic walkthrough' : 'Runner-declared analysis'} / ${lenses.provider}` : 'Rich lenses require a source-bound analysis sidecar. Open Lenses → Open analysis.'} Missing observations remain unavailable.</p>
  </section>;
}
