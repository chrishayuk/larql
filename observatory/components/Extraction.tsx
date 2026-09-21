"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { RefusalReadout } from "@chrishayuk/hause/components/forms/Refusal";
import { admitted, canPlan, commands, factoryStages, object, parseBuild, parseCapabilities, parseInspection, parsePlan, parseSources, runnerBase, runnerRequest } from "../lib/extraction";
import type { Capabilities, Finding, Graph, Inspection, JsonObject, Plan } from "../lib/extraction";

type View = "Source" | "Admission" | "Model" | "Representations" | "Verify" | "Factory";
const display = (v: unknown) => v === undefined || v === null ? "Not reported" : typeof v === "string" ? v : JSON.stringify(v, null, 2);
const download = (v: unknown, name: string) => { const url = URL.createObjectURL(new Blob([JSON.stringify(v, null, 2)], { type: 'application/json' })); const link = document.createElement('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };

export default function Extraction() {
  const [view, setView] = useState<View>("Source");
  const [source, setSource] = useState("hf://google/gemma-3-4b-it");
  const [output, setOutput] = useState("output/gemma3-4b-it.vindex3");
  const [scope, setScope] = useState<'whole' | 'text'>('text');
  const [endpoint, setEndpoint] = useState('');
  const [token, setToken] = useState('');
  const [caps, setCaps] = useState<Capabilities>();
  const [plan, setPlan] = useState<Plan>();
  const [inspection, setInspection] = useState<Inspection>();
  const [build, setBuild] = useState<JsonObject>();
  const [facts, setFacts] = useState<Record<string, unknown>>({});
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [filter, setFilter] = useState('');
  const [category, setCategory] = useState('all');
  const [blockerIds, setBlockerIds] = useState<number[]>();
  const [selected, setSelected] = useState<Finding>();
  const [component, setComponent] = useState('');
  const [objectId, setObjectId] = useState('');
  const file = useRef<HTMLInputElement>(null);
  const flight = useRef<AbortController | null>(null);
  const epoch = useRef(0);
  const cancel = () => { ++epoch.current; flight.current?.abort(); setBusy(''); };
  useEffect(() => () => { flight.current?.abort(); ++epoch.current; }, []);
  let sources: string[] = [], inputError = '';
  try { sources = parseSources(source); } catch (e) { inputError = (e as Error).message; }
  const graph: Graph | undefined = inspection?.graph ?? plan?.graph;
  const componentId = graph?.components.some(c => c.id === component) ? component : String(graph?.components[0]?.id ?? '');
  const objects = graph?.objects.filter(o => o.component === componentId) ?? [];
  const selectedObject = objects.find(o => o.id === objectId) ?? objects[0];
  const matching = plan?.findings.filter(f => (!blockerIds || blockerIds.includes(f.id)) && (category === 'all' || f.category === category) && `${f.component} ${f.subject} ${f.detail} ${f.class}`.toLowerCase().includes(filter.toLowerCase())) ?? [];
  const commandSources = plan?.artifacts.map(a => a.source.path.startsWith('hf://') && a.source.revision ? `${a.source.path.split('@')[0]}@${a.source.revision}` : a.source.path) ?? sources;
  let command: ReturnType<typeof commands> | undefined;
  try { if (commandSources.length) command = commands(commandSources, output, scope); } catch { /* Form below keeps invalid output visibly unready. */ }
  const ready = admitted(plan, scope) && !!command;
  const resetConnection = () => { cancel(); setCaps(undefined); setFacts({}); setError(''); };
  const request = async (label: string, work: (base: string, signal: AbortSignal) => Promise<() => void>) => {
    cancel(); const id = epoch.current; const controller = new AbortController(); flight.current = controller;
    setError(''); setNotice(''); setBusy(label); const timeout = setTimeout(() => controller.abort(), 120000);
    try { const commit = await work(runnerBase(endpoint), controller.signal); if (epoch.current === id) commit(); }
    catch (e) { if (epoch.current === id) setError(controller.signal.aborted ? 'Request stopped or timed out. Runner work may continue; its result is not claimed here.' : `${(e as Error).message} Browser access requires runner CORS; a hosted HTTPS page may not reach a local HTTP runner.`); }
    finally { clearTimeout(timeout); if (epoch.current === id) setBusy(''); }
  };
  const connect = () => { setCaps(undefined); setFacts({}); void request('Checking runner capabilities', async (base, signal) => {
    const next = parseCapabilities(await runnerRequest(base, '/v1/capabilities', token, signal));
    return () => { setCaps(next); setFacts({}); setNotice('Runner capabilities received. No model execution started.'); };
  }); };
  const makePlan = () => {
    if (!caps || !canPlan(caps, sources) || inputError) return;
    void request('Planning source — headers and metadata', async (base, signal) => {
      const p = parsePlan(await runnerRequest(base, '/v1/plan', token, signal, { sources }));
      return () => { setPlan(p); setBlockerIds(undefined); setInspection(undefined); setSelected(undefined); setView('Admission'); };
    });
  };
  const inspectBound = () => {
    if (!caps) return;
    void request('Reading currently bound container', async (base, signal) => {
      const names = ['components', 'representations', 'provenance', 'authority'].filter(k => caps.explorer[k]);
      if (!names.length) throw new Error('This runner exposes none of the container fact views.');
      const results = await Promise.allSettled(names.map(k => runnerRequest(base, `/v1/${k}`, token, signal)));
      const snapshot: Record<string, unknown> = {};
      results.forEach((r, i) => { snapshot[names[i]] = r.status === 'fulfilled' ? r.value : { unavailable: r.reason instanceof Error ? r.reason.message : 'Request failed' }; });
      return () => { setFacts(snapshot); setView('Representations'); setNotice('These independent reads describe the runner’s currently bound container, not the extraction source. No atomic snapshot guarantee.'); };
    });
  };
  const importFile = async (files: FileList | null) => {
    const f = files?.[0]; if (!f) return; cancel(); const id = epoch.current;
    try {
      if (f.size > 10000000) throw new Error('Document exceeds the 10 MB import limit.');
      const value = object(JSON.parse(await f.text()), 'document');
      if (id !== epoch.current) return;
      if ('planner' in value) { const p = parsePlan(value); setPlan(p); setBlockerIds(undefined); setSource(p.artifacts.map(a => a.source.path).join('\n')); setInspection(undefined); setSelected(undefined); setView('Admission'); }
      else if ('index' in value && 'graph' in value) { setInspection(parseInspection(value)); setView('Model'); }
      else if ('build_id' in value) { setBuild(parseBuild(value)); setView('Factory'); }
      else throw new Error('Open SystemPlan v6, a VINDEX3 inspection, or a Factory BuildRecord.');
      setError(''); setNotice(`Opened ${f.name} locally. Import is not independent verification of its claims.`);
    } catch (e) { if (id === epoch.current) setError((e as Error).message); }
    if (file.current) file.current.value = '';
  };
  const copy = async (value: string) => { try { await navigator.clipboard.writeText(value); setNotice('Command copied. Run it on the machine holding the source/output paths.'); } catch { setError('Clipboard unavailable. Select and copy the displayed command.'); } };
  const changeSource = (value: string) => { cancel(); setSource(value); setPlan(undefined); setSelected(undefined); };

  return <main className="observatory extraction">
    <header className="masthead"><Link href="/" className="wordmark">LARQL <span>/</span> OBSERVATORY</Link><nav className="workspace-nav" aria-label="Workspace"><Link href="/">Observe</Link><Link href="/extract" aria-current="page">Extract</Link></nav><span className="connection">{caps ? `${caps.profile} / CONNECTED` : 'NO RUNNER CONNECTED'}</span></header>
    <input ref={file} type="file" accept="application/json,.json" hidden onChange={e => void importFile(e.target.files)} />
    <section className="extract-heading"><div><p className="eyebrow">MODEL / BEFORE THE FIRST TOKEN</p><h1>From weights<br /><em>to a world you can inspect.</em></h1><p>Admit the source. Build its container. Follow the structure.</p></div><button className="text-action" onClick={() => file.current?.click()}>Open plan / inspection / build ↗</button></section>
    <div className="extract-process" aria-label="Extraction workflow">{['SOURCE', 'PLAN', 'ENCODE', 'INSPECT', 'OBSERVE'].map((name, i) => <span key={name}><small>0{i + 1}</small>{name}{i < 4 && <b>→</b>}</span>)}</div>
    <div className="scene-bar"><nav aria-label="Extraction views">{(['Source','Admission','Model','Representations','Verify','Factory'] as View[]).map(v => <button key={v} aria-current={view === v ? 'page' : undefined} onClick={() => setView(v)}>{v}</button>)}</nav><span className="eyebrow">VINDEX3 / EXPLICIT GENERATION</span></div>
    {error && <div className="error" role="alert"><RefusalReadout title="Cannot proceed" lines={[error]} principle="Missing capability is never an implied pass." /><button onClick={() => setError('')}>Dismiss ×</button></div>}
    {(busy || notice) && <div className="extract-notice" role="status">{busy || notice}{busy && <button onClick={cancel}>Stop waiting ×</button>}</div>}
    <div className="extract-body">
      {view === 'Source' && <div className="extract-columns"><section><p className="eyebrow">01 / SOURCE CHECKPOINT</p><h2>Start with the model.</h2><div className="preset-row"><button onClick={() => { changeSource('hf://google/gemma-3-4b-it'); setOutput('output/gemma3-4b-it.vindex3'); }}>Gemma 3 4B</button><button onClick={() => { changeSource('hf://ibm-granite/granite-4.2-3b'); setOutput('output/granite-4.2-3b.vindex3'); }}>Granite 4.2 3B</button></div><label className="extract-label">Sources · one artifact per line<textarea rows={3} value={source} onChange={e => changeSource(e.target.value)} spellCheck={false} /></label><p className="qualifier">Hugging Face references or paths on the runner. Planning an hf:// source reads headers and metadata. Encoding reads the bound tensor ranges.</p>{inputError && <p role="status" className="unavailable">{inputError}</p>}<label className="extract-label">Admission scope<select value={scope} onChange={e => setScope(e.target.value as 'whole'|'text')}><option value="text">Text generation</option><option value="whole">Whole model</option></select></label><p className="qualifier">Text admission requires understood semantics, present operands, and execution support. It is distinct from whole-model completeness.</p><label className="extract-label">New output container · runner filesystem<input value={output} onChange={e => setOutput(e.target.value)} /></label><button className="run-button" onClick={makePlan} disabled={!!busy || !!inputError || !caps || !canPlan(caps, sources)}>PLAN SOURCE <span>↗</span></button>{(!caps || !canPlan(caps, sources)) && <p className="unavailable">Connect a runner that advertises planning for these source forms, or import a CLI plan.</p>}</section>
      <aside className="extract-runner"><p className="eyebrow">RUNNER / CAPABILITIES FIRST</p><h2>Where the work lives.</h2><label className="extract-label">Runner origin<input type="url" placeholder="https://your-runner.example" value={endpoint} onChange={e => { resetConnection(); setEndpoint(e.target.value); }} /></label><label className="extract-label">Bearer token · optional, memory only<input type="password" autoComplete="off" value={token} onChange={e => { resetConnection(); setToken(e.target.value); }} /></label><button className="text-action" disabled={!endpoint || !!busy} onClick={connect}>Connect runner ↗</button><p className="qualifier">Direct browser connection. Credentials go only to the entered runner; they are not saved or sent through the hosted Observatory.</p>{caps && <><dl><dt>Profile</dt><dd>{caps.profile}</dd><dt>Plan HF / local</dt><dd>{String(caps.sources.plan.hf)} / {String(caps.sources.plan.local)}</dd><dt>Encode HF / local</dt><dd>{String(caps.sources.encode.hf)} / {String(caps.sources.encode.local)}</dd></dl><button className="text-action" onClick={inspectBound} disabled={!!busy}>Inspect bound container ↗</button><details><summary>Runner declaration</summary><pre>{display(caps)}</pre></details></>}<div className="extract-boundary"><h3>Two container generations.</h3><p>Legacy <code>extract-index</code> produces VINDEX2. This workspace uses explicit <code>vindex3 plan / encode / inspect</code>. It does not change the legacy extraction default.</p></div></aside></div>}
      {view === 'Admission' && (plan ? <><div className="admission-verdict"><p className="eyebrow">PLANNER-REPORTED / SEMANTICS {display(plan.planner.semantics_version)}</p><h2>{admitted(plan, scope) ? 'This scope is admitted.' : 'This scope is not admitted.'}</h2><p>Whole model: {plan.admissible ? 'admitted' : `${plan.summary.blocking} blocking findings`} · selected scope: {scope === 'text' ? 'text generation' : 'whole model'}</p><label>Scope <select value={scope} onChange={e => setScope(e.target.value as 'whole'|'text')}><option value="text">Text generation</option><option value="whole">Whole model</option></select></label><button className="text-action" onClick={() => download(plan.raw, 'plan.json')}>Save original plan ↓</button></div><details><summary>Staging / transfer evidence</summary><pre>{display(plan.raw.staging)}</pre><p className="qualifier">Reported header/metadata transfers only; no progress or payload size is inferred.</p></details><div className="capability-lines">{plan.capabilities.map(c => <div key={c.capability}><strong>{c.capability}</strong><span>understood / {c.admissible ? 'yes' : 'no'}</span><span>present / {c.available ? 'yes' : 'no'}</span><span>supported / {c.supported ? 'yes' : 'no'}</span><button onClick={() => { setCategory('all'); setFilter(''); setBlockerIds(c.blocker_ids); setSelected(plan.findings.find(f => c.blocker_ids.includes(f.id))); }}>{c.blocking} blockers</button></div>)}</div><div className="extract-columns findings-layout"><section><div className="finding-controls"><input aria-label="Search findings" placeholder="Find a subject, component, or rule…" value={filter} onChange={e => setFilter(e.target.value)} /><select aria-label="Finding category" value={category} onChange={e => setCategory(e.target.value)}>{['all','mismatched','unrepresented','representable','interface'].map(v => <option key={v}>{v}</option>)}</select></div><div className="finding-list">{blockerIds && <button onClick={() => setBlockerIds(undefined)}>Clear capability blocker filter ×</button>}{matching.slice(0, 300).map(f => <button key={f.id} aria-pressed={selected?.id === f.id} onClick={() => setSelected(f)}><span>{f.category}</span><strong>{f.subject}</strong><small>{f.component} / {f.class}</small></button>)}{matching.length > 300 && <p>Showing the first 300 of {matching.length} matches. Narrow the search to inspect the rest.</p>}{!matching.length && <p>No findings match this filter.</p>}</div></section><aside className="finding-detail">{selected ? <><p className="eyebrow">FINDING {selected.id} / {selected.category}</p><h3>{selected.subject}</h3><p>{selected.detail}</p><h4>Declared</h4><pre>{display(selected.declared)}</pre><h4>Resolved</h4><pre>{display(selected.resolved)}</pre><h4>Carriage / semantic cluster</h4><pre>{display(selected.carriage)}{'\n'}{display(selected.cluster)}</pre></> : <p>Select a finding to compare its declared and resolved meaning.</p>}</aside></div><section className="encode-review"><p className="eyebrow">02 / ENCODE HANDOFF</p><h2>{ready ? 'Ready for the runner.' : 'Resolve admission before encoding.'}</h2><p>The encode HTTP route has no supported request contract in this UI. Encoding stays on the CLI; it is not simulated here.</p>{command && <><pre>{command.encode}</pre><button className="text-action" disabled={!ready} onClick={() => void copy(command.encode)}>Copy admitted encode command ↗</button></>}<details><summary>Source and planner identity</summary><pre>{display(plan.artifacts)}{'\n'}{display(plan.planner)}</pre></details></section></> : <Empty title="Admission comes before extraction." text="Plan a source with a connected runner, or open the JSON from larql vindex3 plan." />)}
      {view === 'Model' && (graph ? <><p className="eyebrow">{inspection ? 'CONTAINER-REPORTED' : 'PLANNED'} / STRUCTURE WITHOUT INFERENCE</p><h2>The model before a prompt.</h2><p className="qualifier">{inspection ? 'Reconstructed from the imported container inspection. Separate from the source plan.' : 'Logical objects placed by the planner. This is not proof that a container has been encoded.'}</p><div className="component-tabs">{graph.components.map(c => <button key={String(c.id)} aria-pressed={componentId === c.id} onClick={() => { setComponent(String(c.id)); setObjectId(''); }}>{String(c.id)}<small>{display(c.role)} · {String(c.num_layers)} layers · width {String(c.hidden_size)}</small></button>)}</div><div className="extract-columns"><div className="object-list">{objects.map(o => <button key={String(o.id)} aria-pressed={selectedObject?.id === o.id} onClick={() => setObjectId(String(o.id))}><span>{display(o.kind)}</span><strong>{String(o.id)}</strong></button>)}</div><aside><h3>Selected logical object</h3><pre>{display(selectedObject)}</pre><h3>Declared interfaces</h3>{graph.edges.filter(e => e.producer_component === componentId || e.consumer_component === componentId).map((e, i) => <div className="interface-edge" key={i}>{String(e.producer_component)} <span>→</span> {String(e.consumer_component)}<small>{String(e.consumer_object)}</small></div>)}{!graph.edges.some(e => e.producer_component === componentId || e.consumer_component === componentId) && <p>No cross-component interface reported.</p>}</aside></div></> : <Empty title="Structure needs a source." text="Open a plan or container inspection. Objects and relationships are never inferred from the model name." />)}
      {view === 'Representations' && <><p className="eyebrow">OBJECT → PHYSICAL REPRESENTATION</p><h2>How the model is held.</h2>{inspection ? <><p className="qualifier">Imported container / {String(inspection.index.model)}</p><div className="representation-list">{Object.entries(object(inspection.index.representations, 'representations')).map(([id, value]) => { const r = object(value, 'representation'); return <details key={id}><summary><strong>{id}</strong><span>{display(r.encoding)}</span><small>{display(r.payload_bytes)} bytes</small></summary><pre>{display(r)}</pre></details>; })}</div><details><summary>Container authority and profiles</summary><pre>{display({ authority: inspection.index.authority, profiles: inspection.index.profiles, derived_from_model: inspection.index.derived_from_model })}</pre></details></> : <p>Import a container inspection to inspect its representation directory and full digests.</p>}{Object.keys(facts).length > 0 && <section className="bound-facts"><h3>Runner’s currently bound container</h3><p>Independent endpoint responses. They are not linked to this extraction source or claimed to be an atomic snapshot.</p>{Object.entries(facts).map(([k, v]) => <details key={k}><summary>{k}</summary><pre>{display(v)}</pre></details>)}</section>}{!inspection && !Object.keys(facts).length && <button className="text-action" disabled={!caps || !!busy} onClick={inspectBound}>Read runner container facts ↗</button>}</>}
      {view === 'Verify' && <><p className="eyebrow">03 / INSPECT THE PRODUCED CONTAINER</p><h2>Keep each claim separate.</h2><p>Admission describes what the planner understands. Structural coherence, payload integrity, execution completeness, and numerical parity are different checks.</p>{inspection && <div className="inspection-verdict"><h3>{inspection.defects.length ? `${inspection.defects.length} reported inspection defects` : 'No defects reported in this inspection'}</h3><p>The JSON does not record whether payload rehashing was requested. Zero reported defects is not a payload-integrity or parity witness.</p><pre>{display(inspection.defects)}</pre><button className="text-action" onClick={() => download(inspection.raw, 'inspection.json')}>Save original inspection ↓</button></div>}{command && <div className="encode-review"><h3>Run on the container’s host</h3><pre>{command.inspect}</pre><button className="text-action" onClick={() => void copy(command.inspect)}>Copy inspection command ↗</button></div>}<button className="text-action" onClick={() => file.current?.click()}>Open inspection result ↗</button><p className="qualifier">Observed/unobserved parity remains an executor witness. Opening a container does not start inference.</p><Link className="text-action" href="/">Return to observation ↗</Link></>}
      {view === 'Factory' && <><p className="eyebrow">ROADMAP / RECIPE → BUILD RECORD</p><h2>A build with a history.</h2><p>The Factory invokes legacy extraction and publishing. Its records stay distinct from VINDEX3 admission. Open a real BuildRecord to inspect the terminal outcome and outputs.</p><div className="factory-stages">{factoryStages.map(s => <span key={s} className={build?.stage === s ? 'failed-stage' : ''}>{s}{build?.stage === s ? ' ×' : ''}</span>)}</div>{build ? <><h3>{String(build.recipe_name)} / {String(build.status)}</h3>{build.status === 'failed' && <p role="status">{String(build.stage)}: {String(build.message)}</p>}<p className="qualifier">Build ID / {String(build.build_id)}</p><pre>{display(build.outputs)}</pre><button className="text-action" onClick={() => download(build, 'build-record.json')}>Save build record ↓</button></> : <Empty title="No build record open." text="The stage sequence is the declared pipeline, not live progress. No jobs have been dispatched." />}<p className="qualifier">Factory VERIFY currently checks checksum integrity. Numerical reconstruction and logit parity are not implied. This UI does not publish artifacts or change their visibility.</p></>}
      {view === 'Source' && command && <details className="cli-handoff"><summary>Use the CLI without a connected runner</summary><pre>{command.plan}</pre><button className="text-action" onClick={() => void copy(command.plan)}>Copy planning command ↗</button><p>Open plan.json here afterwards. This does not execute or upload the command.</p></details>}
    </div><footer className="run-footer"><span>EXTRACTION / DECLARED SOURCE → INSPECTABLE CONTAINER</span><span>No simulated progress · no implicit inference</span></footer>
  </main>;
}
function Empty({ title, text }: { title: string; text: string }) { return <div className="extract-empty"><h2>{title}</h2><p>{text}</p></div>; }
