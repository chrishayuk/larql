import type { Basis, Sample } from "./record.ts";

/** Camera transform only; the run-wide domain and recorded coordinates never change. */
export type Point = { x: number; y: number; sample: Sample };
export function project(samples: Sample[], basis: Basis, spatial: boolean, domain: Sample[] = samples): Point[] {
  const plane = (s: Sample) => {
    const [x, y, z] = s.projections[basis.id];
    return [x + (spatial ? z * .48 : 0), y + (spatial ? z * .28 : 0)];
  };
  const all = domain.map(plane);
  const lo = [Math.min(...all.map(p => p[0]), 0), Math.min(...all.map(p => p[1]), 0)];
  const hi = [Math.max(...all.map(p => p[0]), 1), Math.max(...all.map(p => p[1]), 1)];
  return samples.map(sample => { const p = plane(sample); return { sample, x: 14 + (p[0] - lo[0]) / Math.max(hi[0] - lo[0], .01) * 68, y: 83 - (p[1] - lo[1]) / Math.max(hi[1] - lo[1], .01) * 66 }; });
}
