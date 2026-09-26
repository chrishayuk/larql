import { useRef, useState } from 'react';
import type { Recording, Sample } from '../lib/record';
import { labelLayer, labelRole } from '../lib/record';
import type { LensRecord } from '../lib/lenses';
import { vocabularyAt } from '../lib/lenses';
import type { Atlas } from '../lib/atlas';
import { atlasJourney, atlasPoint, atlasTerminal, atlasRemainder, atlasAdjacent, atlasOffPlane, atlasFrame } from '../lib/atlas';
import { useCanvas } from './Plots';
import { AnswerCompass } from './AnswerCompass';

export function ResidualAtlas({ atlas, record, lenses, samples, selected, position, select, inspect, completed, recordedAtlas, recordedLenses, recordedSamples, completeRecord, reveal, selectPosition }: {
  atlas: Atlas; record: Recording; lenses?: LensRecord; samples: Sample[]; selected?: Sample; position: number;
  select: (s: Sample) => void; inspect: () => void; completed: boolean;
  recordedAtlas: Atlas; recordedLenses?: LensRecord; recordedSamples: Sample[]; completeRecord: boolean; reveal: (s: Sample) => void; selectPosition: (position: number) => void;
}) {
  const [showAnswer, setShowAnswer] = useState(true);
  const [showFuture, setShowFuture] = useState(true);
  const [showDestination, setShowDestination] = useState(true);
  const [answerId, setAnswerId] = useState<number>();
  const [anchors, setAnchors] = useState(true);
  const [landmarkId, setLandmarkId] = useState(atlas.landmarks[0].token_id);
  const bounds = [0,1].map(i => [Math.min(0,...atlas.landmarks.map(l=>l.values[i])),Math.max(0,...atlas.landmarks.map(l=>l.values[i]))]);
  const fitCenter: [number,number] = [(bounds[0][0]+bounds[0][1])/2,(bounds[1][0]+bounds[1][1])/2];
  const fitZoom = Math.min(3,1.6/Math.max(bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],.1));
  const framePoints = completeRecord ? atlasJourney(recordedAtlas,recordedSamples,position).map(r=>r.values) : atlas.landmarks.map(l=>l.values);
  const routeFrame = atlasFrame(framePoints);
  const [camera, setCamera] = useState<{ position:number; zoom:number; center:[number,number] }>();
  const effectiveCamera = camera?.position === position ? camera : { position, ...routeFrame };
  const zoom = effectiveCamera.zoom, pan = effectiveCamera.center;
  const setZoom = (value: number | ((zoom:number)=>number)) => setCamera(c => {
    const base=c?.position===position ? c : {position,...routeFrame};
    return {...base,zoom:typeof value==='function'?value(base.zoom):value};
  });
  const setPan = (center:[number,number]) => setCamera(c=>({...((c?.position===position)?c:{position,...routeFrame}),center}));
  const [follow, setFollow] = useState(false);
  const [network, setNetwork] = useState(false);
  const [writes, setWrites] = useState(true);
  const drag = useRef<{ x:number; y:number; center:[number,number] } | null>(null);
  const journey = atlasJourney(atlas,samples,position,selected);
  const current = journey.at(-1);
  const retrospective = completeRecord && recordedAtlas.basis.hash === atlas.basis.hash;
  const remainder = atlasRemainder(recordedAtlas, record, recordedSamples, position, retrospective, current?.site);
  const future = showFuture ? remainder.future : [];
  const active = atlas.landmarks.find(l=>l.token_id===landmarkId) ?? atlas.landmarks[0];
  const center = follow && current ? current.values : pan;
  const point = (v: readonly number[]) => atlasPoint(v,zoom,center);
  const landmarks = atlas.landmarks.map(l=>({...l,...point(l.values)}));
  const labelled = new Set<number>();
  const occupied: {x:number;y:number}[] = [];
  for(const l of [...landmarks].sort((a,b)=>Number(b.token_id===active.token_id)-Number(a.token_id===active.token_id))) {
    if(!occupied.some(p=>Math.abs(p.x-l.x)<14&&Math.abs(p.y-l.y)<6)) {labelled.add(l.token_id);occupied.push(l);}
  }
  const route = journey.map(r=>({...r,...point(r.values)}));
  const vocabulary = vocabularyAt(lenses,current?.sample);
  const cosine = current?.cosines[atlas.landmarks.findIndex(l=>l.token_id===active.token_id)];
  const distance = current ? Math.hypot(current.values[0]-active.values[0],current.values[1]-active.values[1]) : undefined;
  const previous = journey.at(-2);
  const adjacent = previous && current && atlasAdjacent(record,previous.sample,current.sample);
  const previousVocabulary = adjacent ? vocabularyAt(lenses, previous?.sample) : undefined;
  const terminal = showDestination ? (remainder.terminal ?? atlasTerminal(atlas, record, samples, position, completed, selected)) : undefined;
  const terminalAnswer = terminal && vocabularyAt(retrospective ? recordedLenses : lenses,terminal.sample)?.top[0];
  const terminalPoint = terminal && point(terminal.values);
  const frameRoute = () => {
    const points = retrospective ? atlasJourney(recordedAtlas,recordedSamples,position) : journey;
    if (!points.length) return;
    setCamera({position,...atlasFrame(points.map(r=>r.values))});setFollow(false);
  };
  const arrived = terminal?.site === current?.site && !!terminal;
  const ref = useCanvas((ctx,w,h)=>{
    const line=(a:{x:number;y:number},b:{x:number;y:number})=>{ctx.beginPath();ctx.moveTo(a.x*w/100,a.y*h/100);ctx.lineTo(b.x*w/100,b.y*h/100);ctx.stroke();};
    ctx.strokeStyle='#2b3326';ctx.lineWidth=.7;ctx.setLineDash([]);
    for(let v=-1;v<=1.001;v+=.25) { line(point([v,-1]),point([v,1]));line(point([-1,v]),point([1,v])); }
    ctx.strokeStyle='#56654b';line(point([0,-1]),point([0,1]));line(point([-1,0]),point([1,0]));
    if(anchors && network) {
      ctx.setLineDash([2,7]);ctx.strokeStyle='#7b897365';ctx.lineWidth=1;
      for(const e of atlas.edges) { const a=landmarks.find(l=>l.token_id===e.from)!, b=landmarks.find(l=>l.token_id===e.to)!;line(a,b); }
    }
    const ghost = [...(current ? [current] : []),...future].map(r=>({...r,...point(r.values)}));
    ctx.strokeStyle='#b5c1a873';ctx.lineWidth=1.4;ctx.setLineDash([2,6]);
    for(let i=1;i<ghost.length;i++) if(atlasAdjacent(record,ghost[i-1].sample,ghost[i].sample)) line(ghost[i-1],ghost[i]);
    ctx.setLineDash([]);
    for(const p of ghost.slice(current ? 1 : 0)) {ctx.strokeStyle='#b5c1a873';ctx.beginPath();ctx.arc(p.x*w/100,p.y*h/100,2,0,Math.PI*2);ctx.stroke();}
    for(let i=1;i<route.length;i++) {
      const a=route[i-1],b=route[i];
      if(!atlasAdjacent(record,a.sample,b.sample))continue;
      const attention=b.sample.role==='attention_write';
      ctx.strokeStyle=writes && attention ? '#cfac8b' : '#b2c99c';ctx.lineWidth=i===route.length-1?2.8:1.6;ctx.setLineDash([]);line(a,b);
    }
    ctx.setLineDash([]);
    for(const p of route) {ctx.fillStyle=p.sample.role==='attention_write'&&writes?'#cfac8b':'#b2c99c';ctx.beginPath();ctx.arc(p.x*w/100,p.y*h/100,2.4,0,Math.PI*2);ctx.fill();}
  });
  return <section className="residual-atlas">
    <div className="field-caption atlas-heading"><h2>Residual Atlas</h2><p className="eyebrow">POSITION {position} · “{record.tokens[position].trim()}” · {current ? labelLayer(current.sample.layer) : "AWAITING STATE"}</p></div>
    <div className="atlas-view-controls" aria-label="Map layers"><span>MAP LAYERS</span><label><input type="checkbox" checked={showAnswer} onChange={e=>setShowAnswer(e.target.checked)} /> Answer readout</label><label><input type="checkbox" checked={showFuture && retrospective} disabled={!retrospective} onChange={e=>setShowFuture(e.target.checked)} /> Recorded remainder</label><label><input type="checkbox" checked={showDestination} onChange={e=>setShowDestination(e.target.checked)} /> Destination</label></div>
    {retrospective && (showFuture || showDestination) && <p className="atlas-retrospective">Recorded route · dotted segments show measured later states, not predictions.</p>}
    <div className="atlas-coupled both">
    <div className="atlas-state">
    <div className="atlas-tools"><label><input type="checkbox" checked={anchors} onChange={e=>setAnchors(e.target.checked)} /> Token-direction anchors</label><label><input type="checkbox" checked={network} disabled={!anchors} onChange={e=>setNetwork(e.target.checked)} /> Token similarity</label><label><input type="checkbox" checked={writes} onChange={e=>setWrites(e.target.checked)} /> Write stages</label><button aria-pressed={follow} onClick={()=>{setFollow(!follow);}}>Follow traveller</button><div><button aria-label="Zoom out" onClick={()=>setZoom(z=>Math.max(1,z/1.4))}>−</button><span>{zoom.toFixed(1)}×</span><button aria-label="Zoom in" onClick={()=>setZoom(z=>Math.min(32,z*1.4))}>+</button><button onClick={frameRoute}>Frame route</button><button onClick={()=>{setZoom(fitZoom);setPan(fitCenter);setFollow(false);}}>All anchors</button></div></div>
    <div className="atlas-canvas" onPointerDown={e=>{
      if((e.target as HTMLElement).closest('button'))return;
      drag.current={x:e.clientX,y:e.clientY,center:[...center]};setFollow(false);setPan([...center]);e.currentTarget.setPointerCapture(e.pointerId);
    }} onPointerMove={e=>{if(!drag.current)return;const rect=e.currentTarget.getBoundingClientRect();setPan([drag.current.center[0]-(e.clientX-drag.current.x)/rect.width*100/(38*zoom),drag.current.center[1]+(e.clientY-drag.current.y)/rect.height*100/(38*zoom)]);}} onPointerUp={()=>{drag.current=null;}} onPointerCancel={()=>{drag.current=null;}}>
      <canvas ref={ref} aria-hidden="true" />
      {anchors && landmarks.map(l=><button key={l.token_id} className={`atlas-landmark ${labelled.has(l.token_id)?"labelled":""} ${l.token_id===active.token_id?'selected-anchor':''}`} style={{left:`${l.x}%`,top:`${l.y}%`}} onClick={()=>setLandmarkId(l.token_id)} aria-pressed={l.token_id===active.token_id} title={`Token direction ${l.token_id} · [${l.values.map(v=>v.toFixed(4)).join(', ')}]`}><span>◇</span><b>{l.token}</b></button>)}
      {terminalPoint && <button className={`atlas-destination ${arrived?'arrived':''}`} style={{left:`${terminalPoint.x}%`,top:`${terminalPoint.y}%`}} onClick={()=>terminal && reveal(terminal.sample)} aria-label="Go to measured terminal state" data-label-side={terminalPoint.x>62?"left":"right"}><i /><b>TERMINAL STATE<small>{terminalAnswer ? `${terminalAnswer.token} · ${(terminalAnswer.probability*100).toFixed(2)}% readout` : 'Readout unavailable'}</small><small>{arrived?'ARRIVED':'RECORDED ENDPOINT'}</small></b></button>}
      {route.filter((r,i)=>i%8===0||i===route.length-1).map(r=><button key={r.site} className={`atlas-stop ${r===route.at(-1)?'traveller':''}`} style={{left:`${r.x}%`,top:`${r.y}%`}} onClick={()=>{select(r.sample);if(r===route.at(-1))inspect();}} aria-label={`${labelLayer(r.sample.layer)} ${labelRole(r.sample.role)}; inspect recorded write`}><span />{r===route.at(-1)&&<b className={`${arrived?'arrival-position':''} ${r.x>65?'label-left':''}`}>YOU ARE HERE<small>{labelLayer(r.sample.layer)} / {r.sample.role==='ffn_write'?'FFN':'attention'}</small></b>}</button>)}
      {!current&&<p className="plot-empty">Waiting for this position’s captured state.</p>}
      <span className="atlas-axis atlas-x">{atlas.basis.axes[0]} →</span><span className="atlas-axis atlas-y">↑ {atlas.basis.axes[1]}</span>
    </div>
    <div className="atlas-legend"><span>◇ token direction</span><span>● captured state, projected</span><span>— traversed route</span><span>┄ recorded remainder</span><span>◎ terminal state</span><span>Amber / attention · green / FFN</span></div>
    {retrospective && <div className="atlas-write-timeline"><p className="eyebrow">RECORDED WRITE TIMELINE / solid: traversed · hollow: later · ◎: terminal</p><div>{atlasJourney(recordedAtlas,recordedSamples,position).map(r=>{const passed=journey.some(p=>p.site===r.site), last=r.site===remainder.terminal?.site;return <button key={r.site} className={`${passed?'passed':''} ${last?'end':''}`} aria-pressed={current?.site===r.site} onClick={()=>reveal(r.sample)} aria-label={`${labelLayer(r.sample.layer)} ${labelRole(r.sample.role)}${last?', terminal state':''}${passed?', traversed':', recorded later state'}`} title={`${labelLayer(r.sample.layer)} / ${labelRole(r.sample.role)}`}><span /></button>;})}</div></div>}
    <div className="atlas-projection-meta"><span>2D projection / {atlas.basis.width.toLocaleString()}D normalized carrier</span><span>Basis: {atlas.basis.id} · {atlas.basis.hash.slice(0,19)}…</span><span>{current ? `${(atlasOffPlane(current.values)*100).toFixed(1)}% of current squared norm is off-plane` : 'Off-plane state not yet available'}</span><span>Route variance retained: not measured</span></div>
    {anchors && <div className="atlas-landmark-list" aria-label="Inspect token-direction anchor">{atlas.landmarks.map(l=><button key={l.token_id} aria-pressed={l.token_id===active.token_id} onClick={()=>setLandmarkId(l.token_id)}>◇ {l.token}</button>)}</div>}
    {anchors && network && <div className="atlas-neighbors"><span>Neighbors of {active.token} / full-space similarity</span>{atlas.edges.filter(e=>e.from===active.token_id||e.to===active.token_id).map(e=>{const id=e.from===active.token_id?e.to:e.from;const neighbor=atlas.landmarks.find(l=>l.token_id===id)!;return <button key={id} onClick={()=>setLandmarkId(id)}>{neighbor.token} <small>cos {e.cosine.toFixed(3)}</small> →</button>;})}</div>}
    {anchors && <div className="atlas-readout"><div><small>TOKEN-DIRECTION ANCHOR</small><strong>{active.token}</strong><span>Geometric reference, not the run endpoint</span></div><div><small>FULL-SPACE COSINE</small><strong>{cosine?.toFixed(4)??'—'}</strong></div><div><small>PROJECTED SEPARATION</small><strong>{distance?.toFixed(4)??'—'}</strong><span>Distance in this plane; not answer confidence</span></div></div>}
    <p className="lens-note">Token anchors and carrier states share this declared projection. The path ends at its measured terminal carrier, not at a token vector. No semantic regions or basins are inferred.</p>
    </div>
    <div className="atlas-answer-mount" style={{visibility:showAnswer?'visible':'hidden'}} aria-hidden={!showAnswer} inert={!showAnswer}><AnswerCompass position={position} token={record.tokens[position]} finalPosition={record.tokens.length-1} jumpToFinal={()=>{ const next=completeRecord && recordedSamples.find(s=>s.position===record.tokens.length-1 && s.layer===current?.sample.layer && s.role===current?.sample.role); if(next)reveal(next); else selectPosition(record.tokens.length-1); }} atlas={atlas} current={vocabulary} previous={previousVocabulary} tokenId={answerId} choose={setAnswerId} /></div>
    </div>
    <div className="atlas-terminal"><p className="eyebrow">DESTINATION / RECORDED TERMINAL STATE</p><p>{terminal ? `${terminalAnswer ? `${terminalAnswer.token} · ${(terminalAnswer.probability*100).toFixed(2)}% — ` : ''}terminal readout at position ${position}; final carrier [${terminal.values.map(v=>v.toFixed(4)).join(', ')}].` : 'Destination overlay is off or a complete endpoint is unavailable. No destination prediction is made.'}</p></div>
    <div className="atlas-stops"><p className="eyebrow">RECENT STOPS / POSITION {position} · {record.tokens[position]}</p>{journey.slice(-8).map(r=><button key={r.site} aria-pressed={r.site===current?.site} onClick={()=>select(r.sample)}>{labelLayer(r.sample.layer)}<small>{r.sample.role==='ffn_write'?'FFN':'attention'} · Δ norm {r.sample.delta.toPrecision(3)}</small></button>)}</div>
    <p className="lens-note">The route follows recorded write order. Attention/FFN styling identifies the operation; it does not establish a source-token route or a causal effect. Map distance can hide differences outside this plane.</p>
    <details className="lens-provenance"><summary>Geography / basis / evidence</summary><p>{atlas.method}</p><p>{atlas.graph_method}</p><p>Basis {atlas.basis.id} / {atlas.basis.hash}</p><p>{atlas.basis.construction} · {atlas.basis.width} → 2 dimensions. Fixed basis and scale for every token position and layer.</p><p>Semantic regions, FFN feature POIs, recurrent-state routes and causal road closures are unavailable in this capture.</p></details>
  </section>;
}
