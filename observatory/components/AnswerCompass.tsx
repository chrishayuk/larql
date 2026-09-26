import type { Atlas } from '../lib/atlas';
import type { VocabularyRow } from '../lib/lenses';
import { compassReadout } from '../lib/answer-compass';

export function AnswerCompass({ atlas, current, previous, tokenId, choose, position, token, finalPosition, jumpToFinal }: {
  atlas: Atlas; current?: VocabularyRow; previous?: VocabularyRow; tokenId?: number;
  choose: (id?: number) => void; position:number; token:string; finalPosition:number; jumpToFinal:()=>void;
}) {
  const r = compassReadout(current, previous, atlas, tokenId);
  const probability = r.prediction?.probability;
  return <aside className="answer-compass" aria-label="Answer Readout">
    <p className="eyebrow">DIRECTION / ANSWER READOUT</p>
    <h3>After “{token.trim()}”</h3><p className="compass-position">Position {position} · {position===finalPosition ? "final prompt position" : "intermediate prompt position"}</p>{position!==finalPosition && <button className="compass-final" onClick={jumpToFinal}>Inspect final prompt position →</button>}
    <button className="compass-follow" aria-pressed={tokenId === undefined} onClick={() => choose()}>Follow strongest answer</button>
    <div className="compass-dial" role="img" aria-label={r.prediction ? `${r.token}, probability ${(probability! * 100).toFixed(2)} percent, rank ${r.prediction.rank}` : 'Vocabulary readout unavailable'}>
      <div className="compass-ring" style={{ background: `conic-gradient(var(--mint) ${100 * (probability ?? 0)}%, #333a2c 0)` }} />
      <div className="compass-center"><span>{r.token ?? (tokenId === undefined ? 'No readout' : `Token ${tokenId}`)}</span><strong>{probability === undefined ? '—' : `${(probability * 100).toFixed(2)}%`}</strong><small>{r.prediction ? `RANK #${r.prediction.rank}` : 'NOT RETAINED AT THIS SITE'}</small></div>
    </div>
    <p className="compass-caption">Ring fill = vocabulary probability. No spatial bearing is encoded by the ring.</p>
    <div className="compass-candidates" aria-label="Top five answer candidates">{r.candidates.map(p => <button key={p.token_id} aria-pressed={r.id === p.token_id} onClick={() => choose(p.token_id)}><span>#{p.rank} {p.token}</span><b>{(p.probability * 100).toFixed(2)}%</b><i style={{ width: `${p.probability * 100}%` }} /></button>)}</div>
    <p className="compass-caption">{current ? `Top-five mass ${(r.retainedMass * 100).toFixed(2)}% · full-vocabulary normalization` : 'Vocabulary evidence unavailable at this state.'}</p>
    <dl className="compass-metrics"><dt>Logit</dt><dd>{r.prediction?.logit?.toFixed(3) ?? 'unavailable'}</dd><dt>Δ previous write</dt><dd>{r.delta === undefined ? 'unavailable' : `${r.delta > 0 ? '↑ +' : r.delta < 0 ? '↓ ' : ''}${(r.delta * 100).toFixed(2)} pp`}</dd><dt>Full-space cosine</dt><dd>{r.cosine?.toFixed(4) ?? 'not captured'}</dd><dt>Token-axis bearing</dt><dd>{r.bearing === undefined ? 'not captured' : `${r.bearing.toFixed(1)}°`}</dd></dl>
    <p className="compass-caption">Bearing is the fixed token direction in the declared plane, counterclockwise from +X. It is independent of probability and is not a route from the carrier.</p>
  </aside>;
}
