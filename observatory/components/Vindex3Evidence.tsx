import type { Recording, Sample } from '../lib/record';
import type { LensRecord } from '../lib/lenses';
import { vocabularyAt } from '../lib/lenses';

/** Always-visible identity sufficient to distinguish the selected experimental object. */
export function Vindex3Evidence({ record }: { record: Recording }) {
  const v = record.vindex3;
  if (!v) return null;
  return <section className="v3-evidence" aria-label="VINDEX3 recording evidence">
    <div><strong>{v.schema}</strong><span>{v.complete ? 'SEALED RECORD / COMPLETE RECEIPT' : 'INCOMPLETE RECORD / PREFIX'}</span><span>Event hash + provenance fingerprint checked</span></div>
    <p>Run {record.id} · model identity {record.model} · component {v.component}</p>
    <p>Prompt token IDs: {v.prompt_tokens.join(', ')}. Prompt text, token labels and container content hash: not recorded.</p>
    <p>Basis {record.bases[0].id} · {record.bases[0].provider} · {record.bases[0].hidden} dimensions · SHA-256 <code>{record.bases[0].hash.slice(7)}</code></p>
    <p>Carrier: post-add / pre-layer-scale · write magnitude: applied delta L2. Coordinates are recorded; camera changes only their display.</p>
    <p>Lens: {v.lens_method ?? 'not captured'}{v.lens_method && ' · layer scale → prepared final norm and head (including scaling/softcap) → full log-softmax'}. Head passes: {v.head_passes} · live-tap drops: {v.live_dropped}.</p>
    <p>Token text, raw logits, entropy and generated output are not reconstructed. File integrity does not establish output parity or authenticate the runner.</p>
    {v.warnings.map(w => <p className="loss" role="status" key={w}>{w}</p>)}
    <details><summary>Exact record identity and terminal receipt</summary>
      <dl><dt>Source SHA-256</dt><dd>{v.source_sha256}</dd><dt>Event log SHA-256</dt><dd>{v.log_sha256}</dd><dt>Provenance fingerprint</dt><dd>{v.provenance_fingerprint}</dd><dt>Recorded events</dt><dd>{v.events}</dd><dt>Wall-clock start (Unix ms)</dt><dd>{v.started_unix_ms}</dd><dt>Lens failure</dt><dd>{v.lens_failure ?? 'none reported'}</dd></dl>
      <p>The receipt declares traversal completion. This format carries no independent program inventory or final sampled output. The Model view shows observed sites only.</p>
    </details>
  </section>;
}

export function Vindex3Readout({ lenses, selected }: { lenses?: LensRecord; selected?: Sample }) {
  const row = vocabularyAt(lenses, selected);
  return <section className="readout"><h3>Answer lens <span>HEAD-V1 / RECORDED</span></h3>
    {row ? <><dl>{row.targets.map(t => <span className="dl-row" key={t.token_id}><dt>Token {t.token_id}</dt><dd>rank {t.rank}<br />log p {t.logprob}<br />p {t.probability}</dd></span>)}</dl><p className="qualifier">Probability = exp(recorded log p), over the full vocabulary. Top retained: {row.top.map(t => `${t.token} (rank ${t.rank})`).join(', ') || 'none'}.</p></> : <p className="unavailable">No lens readout recorded at this write.</p>}
    <p className="qualifier">An intermediate head readout is not a sampled output or causal evidence. Raw logits and entropy are not recorded.</p>
  </section>;
}

/** The inspector displays source precision for canonical evidence. */
export function CarrierReadout({ record, selected }: { record: Recording; selected?: Sample }) {
  const basis = record.bases[0];
  const metric = (v?: number) => v === undefined ? '—' : record.vindex3 ? String(v) : v.toFixed(3);
  return <section className="readout"><h3>Carrier <span>RESIDUAL / L2</span></h3>
    {record.standard && <p className="qualifier">Post-add, before layer scale. Entering-state norm is not captured.</p>}
    <dl><dt>Before write</dt><dd>{metric(selected?.before)}</dd><dt>Applied write</dt><dd className="accent">{metric(selected?.delta)}</dd><dt>After write</dt><dd>{metric(selected?.norm)}</dd>{selected?.layer_scale !== undefined && <><dt>Layer scale (not applied here)</dt><dd>{metric(selected.layer_scale)}</dd></>}</dl>
    {basis && selected && <div className="coordinate-row">{selected.projections[basis.id].map((v,i) => <span key={i}>{'xyz'[i]} <b>{metric(v)}</b></span>)}</div>}
  </section>;
}
