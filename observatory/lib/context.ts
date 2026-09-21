import type { Sample, Role } from "./record.ts";

export type ContextMetric = "norm" | "delta" | "probe";
export function contextValue(sample: Sample | undefined, metric: ContextMetric, answer: string): number | undefined {
  if (!sample) return undefined;
  return metric === "probe" ? sample.logits[answer] : sample[metric];
}

/** A fixed run-wide scale for the declared role, so layer scrubbing does not
 * manufacture changes by re-normalizing each layer's colors. No missing zeros. */
export function contextRows(samples: Sample[], all: Sample[], tokens: string[], layer: number, role: Role, metric: ContextMetric, answer: string) {
  const maximum = all.filter(s => s.role === role).reduce((max, s) => Math.max(max, Math.abs(contextValue(s, metric, answer) ?? 0)), 0);
  const sites = new Map(samples.filter(s => s.layer === layer && s.role === role).map(s => [s.position, s]));
  return { maximum, rows: tokens.map((token, position) => {
    const sample = sites.get(position), value = contextValue(sample, metric, answer);
    return { token, position, sample, value, intensity: value === undefined ? undefined : maximum === 0 ? 0 : Math.abs(value) / maximum };
  }) };
}
