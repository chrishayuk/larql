"use client";
import { useEffect, useRef } from "react";
import type { Basis, Sample } from "../lib/record";
import { labelLayer } from "../lib/record";

import { project } from "../lib/projection";
import type { Point } from "../lib/projection";

export function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const paint = () => {
      const { width, height } = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio; canvas.height = height * ratio;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(ratio, ratio); ctx.clearRect(0, 0, width, height); draw(ctx, width, height);
    };
    const observer = new ResizeObserver(paint); observer.observe(canvas); paint();
    return () => observer.disconnect();
  }, [draw]);
  return ref;
}

export function Trajectory({ samples, allSamples, target = [], basis, spatial, selected, select }: {
  samples: Sample[]; allSamples: Sample[]; target?: Sample[]; basis: Basis;
  spatial: boolean; selected?: Sample; select: (sample: Sample) => void;
}) {
  const domain = [...allSamples, ...target];
  const points = project(samples, basis, spatial, domain);
  const other = project(target, basis, spatial, domain);
  const ref = useCanvas((ctx, w, h) => {
    ctx.strokeStyle = "#262723"; ctx.lineWidth = .65;
    for (let x = .14; x < .9; x += .17) { ctx.beginPath(); ctx.moveTo(w * x, h * .12); ctx.lineTo(w * x, h * .88); ctx.stroke(); }
    for (let y = .17; y < .9; y += .165) { ctx.beginPath(); ctx.moveTo(w * .1, h * y); ctx.lineTo(w * .9, h * y); ctx.stroke(); }
    const line = (p: Point[], color: string, dashed = false) => {
      if (!p.length) return;
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash(dashed ? [4, 5] : []);
      p.forEach((point, i) => { const x = point.x * w / 100, y = point.y * h / 100; if (!i || point.sample.layer !== p[i - 1].sample.layer + 1) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
      for (const point of p) { ctx.beginPath(); ctx.fillStyle = color; ctx.arc(point.x * w / 100, point.y * h / 100, 2.4, 0, Math.PI * 2); ctx.fill(); }
    };
    line(other, "#b89ec4", true); line(points, "#b2c99c");
    if (points.length) {
      const p = points[points.length - 1]; ctx.beginPath(); ctx.strokeStyle = "#b2c99c55"; ctx.arc(p.x * w / 100, p.y * h / 100, 9, 0, Math.PI * 2); ctx.stroke();
    }
  });
  return <div className="trajectory">
    <canvas ref={ref} aria-hidden="true" />
    {points.map((p, i) => <button key={`${p.sample.layer}:${p.sample.role}`} className={`plot-point ${selected?.layer === p.sample.layer ? "chosen" : ""}`} style={{ left: `${p.x}%`, top: `${p.y}%` }} onClick={() => select(p.sample)} aria-label={`Select ${labelLayer(p.sample.layer)}, ${p.sample.role}, position ${p.sample.position}`} aria-pressed={selected?.layer === p.sample.layer}>
      <span className="point-core" />{(i === 0 || i % 8 === 0 || i === points.length - 1 || selected?.layer === p.sample.layer) && <span className="point-label">{labelLayer(p.sample.layer)}</span>}
    </button>)}
    {!points.length && <p className="plot-empty">Waiting for this position’s captured state.</p>}
    <span className="axis axis-x">{basis.axes[0]} →</span><span className="axis axis-y">{basis.axes[1]} →</span>
    <div className="plot-footnote">PROJECTED STATE · {spatial ? "3D / FIXED OBLIQUE CAMERA" : "2D / AXES 1 + 2"}<br />{spatial && `${basis.axes[2]} / depth`}</div>
  </div>;
}

export function AnswerPlot({ samples, answer, selected, select, metric, against }: {
  samples: Sample[]; answer: string; selected?: Sample; select: (s: Sample) => void;
  metric: "readout" | "difference"; against: string;
}) {
  const value = (s: Sample) => s.logits[answer] - (metric === "difference" ? s.logits[against] : 0);
  const values = samples.map(value);
  const min = Math.min(...values, 0), max = Math.max(...values, 1);
  const ref = useCanvas((ctx, w, h) => {
    ctx.strokeStyle = "#393b35"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(12, h - 18); ctx.lineTo(w - 12, h - 18); ctx.stroke();
    ctx.beginPath(); ctx.strokeStyle = "#b2c99c"; ctx.lineWidth = 1.6;
    samples.forEach((s, i) => { const x = 12 + i / Math.max(samples.length - 1, 1) * (w - 24), y = h - 22 - (value(s) - min) / (max - min) * (h - 36); if (!i || s.layer !== samples[i - 1].layer + 1) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    if (selected) { const i = samples.findIndex(s => s.layer === selected.layer); if (i >= 0) { const x = 12 + i / Math.max(samples.length - 1, 1) * (w - 24); ctx.strokeStyle = "#b2c99c55"; ctx.beginPath(); ctx.moveTo(x, 10); ctx.lineTo(x, h - 18); ctx.stroke(); } }
  });
  return <div className="answer-chart"><canvas ref={ref} aria-hidden="true" />
    <div className="answer-sites">{samples.filter((s, i) => i === 0 || s.layer % 8 === 7 || i === samples.length - 1).map(s => <button key={s.layer} onClick={() => select(s)}>{labelLayer(s.layer)}</button>)}</div>
    <span className="answer-range">{max.toFixed(1)} / {min.toFixed(1)}</span>
  </div>;
}
