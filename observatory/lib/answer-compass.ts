import type { Atlas } from './atlas.ts';
import type { VocabularyRow } from './lenses.ts';

/** Vocabulary probabilities and geometry stay independent; missing retained values stay missing. */
export function compassReadout(current: VocabularyRow | undefined, previous: VocabularyRow | undefined, atlas: Atlas, tokenId?: number) {
  const candidates = current?.top.slice(0, 5) ?? [];
  const id = tokenId ?? candidates[0]?.token_id;
  const prediction = [...(current?.targets ?? []), ...(current?.top ?? [])].find(p => p.token_id === id);
  const prior = [...(previous?.targets ?? []), ...(previous?.top ?? [])].find(p => p.token_id === id);
  const landmarkIndex = atlas.landmarks.findIndex(l => l.token_id === id);
  const landmark = atlas.landmarks[landmarkIndex];
  const row = current && atlas.rows.find(r => r.site === current.site);
  const bearing = landmark && Math.hypot(...landmark.values) > 1e-12
    ? (Math.atan2(landmark.values[1], landmark.values[0]) * 180 / Math.PI + 360) % 360 : undefined;
  return {
    candidates, id, prediction, token: prediction?.token ?? landmark?.token,
    delta: prediction && prior ? prediction.probability - prior.probability : undefined,
    cosine: row && landmarkIndex >= 0 ? row.cosines[landmarkIndex] : undefined,
    bearing,
    retainedMass: candidates.reduce((sum, p) => sum + p.probability, 0),
  };
}
