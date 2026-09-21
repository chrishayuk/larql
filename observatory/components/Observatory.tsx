"use client";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RefusalReadout } from "@chrishayuk/hause/components/forms/Refusal";
import { Evidence } from "@chrishayuk/hause/components/forms/Evidence";
import { compatibleBasis, labelLayer, labelRole, parseRecording, reduceEvents, samplesAt, semanticKey, stepIndex } from "../lib/record";
import type { Recording, Role, Sample, Scene } from "../lib/record";
import { ResidualAtlas } from "./ResidualAtlas";
import { AnswerPlot, Trajectory } from "./Plots";
import { ModelSpace } from "./ModelSpace";
import { ExecutionProvenance } from "./ExecutionProvenance";
import { ContextView } from "./ContextView";
import { ResearchView } from "./ResearchView";
import { availableLenses, parseLenses, sourceDigest, headKey } from "../lib/lenses";
import type { Lens, LensRecord } from "../lib/lenses";
import { EvidenceLevels } from "./EvidenceLevels";
import { openRecording as importRecording } from "../lib/standard";
import { openRecordingText } from "../lib/ingestion";
import { Vindex3Evidence, Vindex3Readout, CarrierReadout } from "./Vindex3Evidence";
import { auditStandard } from "../lib/replay-audit";
import type { ReplayAudit } from "../lib/replay-audit";

const stories = [
  { id: "answer", letter: "A", title: "An answer takes shape", subtitle: "Depth → selected-token readout", prompt: "The capital of France is" },
  { id: "writes", letter: "B", title: "A change of hands", subtitle: "Attention → FFN → carrier", prompt: "The capital of France is" },
  { id: "address", letter: "C", title: "An address resolves", subtitle: "Authored relation / entity / binding", prompt: "KA red MU" },
  { id: "counterfactual", letter: "D", title: "Where paths divide", subtitle: "Base → counterfactual", prompt: "KA red MU" },
];
const fmt = (value?: number) => value === undefined ? "—" : value.toFixed(3);
const friendly = (s?: Sample) => s ? `${labelLayer(s.layer)} · ${labelRole(s.role)}` : "No captured site";

export default function Observatory({ initialAtlas = false, initialHeadsDemo = false, initialHeads = false }: { initialAtlas?: boolean; initialHeadsDemo?: boolean; initialHeads?: boolean }) {
  const [story, setStory] = useState(stories[0]);
  const [prompt, setPrompt] = useState(stories[0].prompt);
  const [record, setRecord] = useState<Recording>();
  const [sourceRecord, setSourceRecord] = useState<unknown>();
  const [sourceBytes, setSourceBytes] = useState("");
  const [digest, setDigest] = useState("");
  const [lenses, setLenses] = useState<LensRecord>();
  const [lens, setLens] = useState<Lens>("Logits");
  const [selectedHead, setSelectedHead] = useState<string>();
  const analysisFile = useRef<HTMLInputElement>(null);
  const comparisonFile = useRef<HTMLInputElement>(null);
  const [replayAudit, setReplayAudit] = useState<ReplayAudit>();
  const [targetLenses, setTargetLenses] = useState<LensRecord>();
  const [targetDigest, setTargetDigest] = useState("");
  const targetAnalysisFile = useRef<HTMLInputElement>(null);
  const [target, setTarget] = useState<Recording>();
  const [cursor, setCursor] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [scene, setScene] = useState<Scene>("Map");
  const [plane, setPlane] = useState<"Model" | "Execution" | "Alternatives">("Execution");
  const [position, setPosition] = useState(4);
  const [selectedKey, setSelectedKey] = useState<string>();
  const [basisId, setBasisId] = useState("");
  const [atlasMode, setAtlasMode] = useState(true);
  const [spatial, setSpatial] = useState(true);
  const [role, setRole] = useState<Role>("ffn_write");
  const [answer, setAnswer] = useState("");
  const [metric, setMetric] = useState<"readout" | "difference">("readout");
  const [against, setAgainst] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [refreshNotice, setRefreshNotice] = useState("");
  const [details, setDetails] = useState(false);
  const [upstream, setUpstream] = useState(false);
  const [downstream, setDownstream] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const request = useRef(0);

  const state = useMemo(() => reduceEvents(record?.events.slice(0, cursor + 1) ?? []), [record, cursor]);
  const full = useMemo(() => reduceEvents(record?.events ?? []), [record]);
  const visibleLenses = useMemo(() => availableLenses(lenses, state.samples, state.events.at(-1)?.sequence), [lenses, state]);
  const targetState = useMemo(() => reduceEvents(target?.events ?? []), [target]);
  const visible = useMemo(() => samplesAt(state, position, role), [state, position, role]);
  const all = useMemo(() => samplesAt(full, position, role), [full, position, role]);
  const selected = selectedKey ? state.samples.find(s => semanticKey(s) === selectedKey) : state.samples.filter(s => s.position === position && ((playing && scene === "Map" && atlasMode) || s.role === role)).at(-1);
  const basis = record?.bases.find(b => b.id === basisId) ?? record?.bases[0];
  const otherBasis = target?.bases.find(b => b.id === basis?.id);
  const canCompare = !!basis && !!otherBasis && compatibleBasis(basis, otherBasis);
  const comparison = samplesAt(targetState, position, role).filter(s => s.layer <= (visible.at(-1)?.layer ?? -2));
  const matrixMax = useMemo(() => full.samples.filter(s => s.role === role).reduce((max,s) => Math.max(max,s.delta),0), [full,role]);
  const related = state.samples.filter(s => s.position === position && s.layer === selected?.layer);
  const attn = related.find(s => s.role === "attention_write");
  const ffn = related.find(s => s.role === "ffn_write");
  const capturedHeads = visibleLenses?.heads?.rows.filter(h => attn && h.site === semanticKey(attn)) ?? [];
  const inspectedHead = capturedHeads.find(h => headKey(h) === selectedHead);
  const distinctHeads = [...new Map(capturedHeads.map(h => [h.head, h])).values()];
  const meanSources = new Map<number, number>();
  for (const h of distinctHeads) for (const source of h.sources) meanSources.set(source.position, (meanSources.get(source.position) ?? 0) + source.weight / distinctHeads.length);
  const attentionSources = inspectedHead?.sources ?? (distinctHeads.length ? [...meanSources].map(([position, weight]) => ({ position, weight })) : attn?.sources);
  const attentionSourceLabel = inspectedHead ? `Measured H${inspectedHead.head} source weights` : distinctHeads.length ? `Mean of ${distinctHeads.length} captured heads` : "Synthetic head-mean fixture";

  const sourceLabel = record?.provenance === "executor" ? "RUNNER-DECLARED EXECUTION" : "SYNTHETIC FIXTURE";
  const readoutLabel = record?.standard ? "raw head-row probe" : "fixture linear readout";
  const complete = cursor === (record?.events.length ?? 0) - 1;

  const openRecord = useCallback((r: Recording, play: boolean) => {
    if (!r.vindex3) { try { sessionStorage.removeItem("observatory.v3-record"); } catch {} }
    setRefreshNotice("");
    setLenses(undefined); setSelectedHead(undefined); setLens("Logits"); setRecord(r); setPlane("Execution"); setCursor(play ? 1 : r.events.length - 1); setPlaying(play);
    setPosition(r.standard?.observed_positions.at(-1) ?? r.tokens.length - 1); setBasisId(r.bases[0].id); setAnswer(r.answers[0] ?? ""); setAgainst(r.answers[1] ?? r.answers[0] ?? "");
    setSelectedKey(undefined); setError(""); setScene("Map"); setRole(r.program.some(p => p.roles.includes("ffn_write")) ? "ffn_write" : "attention_write"); setUpstream(false); setDownstream(false);
  }, []);

  const launch = useCallback(async (browse = false, headsDemo = false) => {
    if (prompt.trim() !== story.prompt) { setError("No recording for this prompt. Choose a fixture story below. Live Gemma 3 4B / Granite 3B execution will connect through the runner adapter."); return; }
    const id = ++request.current; setLoading(true); setError("");
    try {
      const response = await fetch(`/fixtures/${story.id}.json`);
      if (!response.ok) throw new Error("The fixture recording could not be opened.");
      const bytes = await response.text();
      const r = parseRecording(JSON.parse(bytes));
      const hash = await sourceDigest(bytes);
      const analysisResponse = await fetch(`/fixtures/${story.id}.lenses.json`);
      const analysis = analysisResponse.ok ? parseLenses(await analysisResponse.json(), r, hash) : undefined;
      let t: Recording | undefined;
      let ta: LensRecord | undefined;
      let th = "";
      if (story.id === "counterfactual") {
        const response = await fetch("/fixtures/counterfactual-target.json");
        if (!response.ok) throw new Error("The target fixture could not be opened.");
        const bytes = await response.text();
        t = parseRecording(JSON.parse(bytes)); th = await sourceDigest(bytes);
        const a = await fetch("/fixtures/counterfactual-target.lenses.json");
        if (a.ok) ta = parseLenses(await a.json(), t, th);
      }
      if (id !== request.current) return;
      setReplayAudit(undefined); setTarget(t); setTargetLenses(ta); setTargetDigest(th); setSourceRecord(r); openRecord(r, !browse && !headsDemo); setDigest(hash); setSourceBytes(bytes); setLenses(analysis); if (browse) { setPlane("Model"); setCursor(1); } if (headsDemo) { setScene("Lenses"); setLens("Heads"); }
    } catch (e) { if (id === request.current) setError(e instanceof Error ? e.message : "Could not open recording."); }
    finally { if (id === request.current) setLoading(false); }
  }, [openRecord, prompt, story]);

  useEffect(() => {
    // This route loads a separately labelled synthetic recording.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (initialHeadsDemo) void launch(false, true);
  }, [initialHeadsDemo, launch]);

  const openMeasured = useCallback(async (heads = false) => {
    const id = ++request.current; setLoading(true); setError("");
    try {
      const response = await fetch("/recordings/granite-standard.json", { cache: "no-store" });
      if (!response.ok) throw new Error("The measured recording could not be opened.");
      const bytes = await response.text();
      const { record: r, source } = importRecording(JSON.parse(bytes));
      const hash = await sourceDigest(bytes);
      const audit = auditStandard(source, r);
      const lensResponse = await fetch("/recordings/granite-standard.lenses.json", { cache: "no-store" });
      if (!lensResponse.ok) throw new Error("The measured vocabulary analysis could not be opened.");
      const analysis = parseLenses(await lensResponse.json(), r, hash);
      if (r.provenance !== "executor" || r.standard?.coverage !== "complete") throw new Error("Measured study requires a complete executor record.");
      if (id !== request.current) return;
      setSourceRecord(source); setReplayAudit(audit); setTarget(undefined); setTargetLenses(undefined); setDetails(false); openRecord(r, false); setDigest(hash); setSourceBytes(bytes); setLenses(analysis); setScene(heads ? "Lenses" : "Map"); if (heads) setLens("Heads"); setAtlasMode(true);
    } catch (e) { if (id === request.current) setError(e instanceof Error ? e.message : "Could not open measured recording."); }
    finally { if (id === request.current) setLoading(false); }
  }, [openRecord]);

  useEffect(() => {
    // Route entry fetches the saved run; the shared loader also sets its pending state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (initialAtlas || initialHeads) void openMeasured(initialHeads);
  }, [initialAtlas, initialHeads, openMeasured]);

  useEffect(() => {
    if (!record || !playing) return;
    let frame: number;
    let start: number | undefined;
    const origin = Number(record.events[Math.max(0, cursor)].timestamp_ns) / 1e6;
    const tick = (now: number) => {
      start ??= now;
      const until = origin + (now - start) * speed;
      let next = Math.max(0, cursor);
      while (next + 1 < record.events.length && Number(record.events[next + 1].timestamp_ns) / 1e6 <= until) next++;
      setCursor(next);
      if (next === record.events.length - 1) setPlaying(false); else frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
    // Cursor advances within this playback clock; restart only when playback/speed changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record, playing, speed]);

  const select = useCallback((s: Sample) => { setPlaying(false); setPosition(s.position); setSelectedKey(semanticKey(s)); if (s.role !== "embedding") setRole(s.role); }, []);
  const selectPosition = useCallback((p: number) => {
    const sample = state.samples.find(s => s.position === p && s.layer === selected?.layer && s.role === role);
    setPosition(p); setSelectedKey(sample ? semanticKey(sample) : undefined);
  }, [state.samples, selected?.layer, role]);
  const jump = useCallback((index: number) => {
    if (!record) return;
    setPlaying(false); setCursor(Math.max(1, Math.min(record.events.length - 1, index))); setSelectedKey(undefined);
  }, [record]);
  const step = useCallback((direction: number, layerOnly = false) => {
    if (!record) return;
    const next = stepIndex(record, cursor, position, direction, layerOnly ? role : undefined);
    if (next !== undefined) jump(next);
  }, [record, position, role, cursor, jump]);
  const togglePlay = useCallback(() => {
    if (playing) {
      setPlaying(false);
      if (selected) { setSelectedKey(semanticKey(selected)); if (selected.role !== "embedding") setRole(selected.role); }
      return;
    }
    setSelectedKey(undefined);
    if (complete) setCursor(1);
    setPlaying(true);
  }, [complete, playing, selected]);

  useEffect(() => {
    const keys = (e: KeyboardEvent) => {
      if (!record || e.ctrlKey || e.metaKey || e.altKey || (e.target instanceof HTMLElement && e.target.closest("input,textarea,select,button,[contenteditable=true]"))) return;
      if (e.key === " ") { e.preventDefault(); togglePlay(); }
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); step(e.key === "ArrowRight" ? 1 : -1, e.shiftKey); }
      if (e.key.toLowerCase() === "j" || e.key.toLowerCase() === "k") { e.preventDefault(); selectPosition(Math.min(record.tokens.length - 1, Math.max(0, position + (e.key.toLowerCase() === "j" ? -1 : 1)))); }
      if (e.key === "Escape") setSelectedKey(undefined);
      if (e.key.toLowerCase() === "c" && target) setScene("Compare");
    };
    window.addEventListener("keydown", keys); return () => window.removeEventListener("keydown", keys);
  }, [record, target, togglePlay, step, selectPosition, position]);

  const mountBytes = useCallback(async (bytes: string, current: number) => {
    const opened = await openRecordingText(bytes);
    const { record: r, source } = opened;
    const audit = r.standard && !r.vindex3 ? auditStandard(source, r) : undefined;
    if (current !== request.current) return;
    setReplayAudit(audit); setSourceRecord(source); setDetails(!!audit);
    setLoading(false); setTarget(undefined); setTargetLenses(undefined); openRecord(r, false);
    setDigest(opened.digest); setSourceBytes(opened.bytes); setLenses(opened.lenses);
    if (r.vindex3) {
      setAtlasMode(false);
      try { sessionStorage.setItem("observatory.v3-record", bytes); setRefreshNotice("Recording retained in this tab for refresh. Files stay in your browser."); }
      catch { setRefreshNotice("This browser could not retain the recording for refresh. Save and reopen the original file to replay it."); }
    }
  }, [openRecord]);
  const restored = useRef(false);
  useEffect(() => {
    if (restored.current || initialAtlas || initialHeads || initialHeadsDemo) return;
    restored.current = true;
    let bytes: string | null = null;
    try { bytes = sessionStorage.getItem("observatory.v3-record"); } catch {}
    if (!bytes) return;
    const current = ++request.current;
    void mountBytes(bytes, current).catch(e => { if (current === request.current) setError(e instanceof Error ? e.message : "Saved recording refused."); });
  }, [initialAtlas, initialHeads, initialHeadsDemo, mountBytes]);
  const importFile = async (files: FileList | null) => {
    const chosen = files?.[0]; if (!chosen) return;
    const current = ++request.current; setLoading(true);
    try {
      if (chosen.size > 10_000_000) throw new Error("Recording exceeds the 10 MB import limit.");
      await mountBytes(await chosen.text(), current);
    } catch (e) { if (current === request.current) setError(e instanceof Error ? e.message : "Invalid recording."); }
    finally { if (current === request.current) setLoading(false); }
    if (file.current) file.current.value = "";
  };
  const openParis = async () => {
    const current = ++request.current; setLoading(true); setError("");
    try {
      const response = await fetch("/recordings/gemma-paris-v3.jsonl");
      if (!response.ok) throw new Error("Paris recording could not be opened.");
      await mountBytes(await response.text(), current);
    } catch (e) { if (current === request.current) setError(e instanceof Error ? e.message : "Paris recording refused."); }
    finally { if (current === request.current) setLoading(false); }
  };
  const loadAnalysis = async (files: FileList | null) => {
    const chosen = files?.[0]; if (!chosen || !record) return;
    const current = request.current;
    try {
      if (chosen.size > 10_000_000) throw new Error("Analysis exceeds the 10 MB limit.");
      if (record.vindex3) throw new Error("The canonical JSONL is the lens authority. Sidecar replacement is not supported for VINDEX3 recordings.");
      const analysis = parseLenses(JSON.parse(await chosen.text()), record, digest);
      if (current !== request.current) return;
      setLenses(analysis); setSelectedHead(undefined); setError("");
    } catch (e) { if (current === request.current) setError(e instanceof Error ? e.message : "Invalid analysis."); }
    if (analysisFile.current) analysisFile.current.value = "";
  };
  const loadComparison = async (files: FileList | null) => {
    const chosen = files?.[0]; if (!chosen) return;
    const current = request.current;
    try {
      if (chosen.size > 10_000_000) throw new Error("Comparison exceeds the 10 MB limit.");
      const bytes = await chosen.text();
      const opened = await openRecordingText(bytes);
      const { record: r, digest: hash } = opened;
      if (current !== request.current) return;
      setTarget(r); setTargetDigest(hash); setTargetLenses(opened.lenses); setError("");
    } catch (e) { if (current === request.current) setError(e instanceof Error ? e.message : "Invalid comparison."); }
    if (comparisonFile.current) comparisonFile.current.value = "";
  };
  const loadTargetAnalysis = async (files: FileList | null) => {
    const chosen = files?.[0]; if (!chosen || !target) return;
    const current = request.current;
    try {
      if (chosen.size > 10_000_000) throw new Error("Analysis exceeds the 10 MB limit.");
      if (target.vindex3) throw new Error("The comparison JSONL supplies its own lens evidence; sidecar replacement is not supported.");
      const parsed = parseLenses(JSON.parse(await chosen.text()), target, targetDigest);
      if (current !== request.current) return;
      setTargetLenses(parsed); setError("");
    } catch (e) { if (current === request.current) setError(e instanceof Error ? e.message : "Invalid target analysis."); }
    if (targetAnalysisFile.current) targetAnalysisFile.current.value = "";
  };
  const saveAnalysis = () => {
    if (!lenses) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(lenses)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "observatory.lenses.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const seekLayer = (layer: number) => {
    if (!record) return;
    const matching = record.events.map((e, i) => ({ sample: e.sample, i })).filter(x => x.sample?.layer === layer && x.sample.role === role);
    const site = matching.find(x => x.sample?.position === position);
    if (site?.sample) { jump(matching.at(-1)!.i); select(site.sample); }
  };
  const save = () => {
    if (!record) return;
    const url = URL.createObjectURL(new Blob([sourceBytes || JSON.stringify(sourceRecord ?? record)], { type: record.vindex3 ? "application/x-ndjson" : "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `${record.id.replace(/[^a-z0-9_-]/gi, "_")}.${record.vindex3 ? "jsonl" : "json"}`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const attention = <section className="readout"><h3>Attention sources <span>WEIGHT</span></h3>
    {attentionSources ? <><p className="qualifier">{attentionSourceLabel} · not causal contribution</p><div className="source-bars">{[...attentionSources].sort((a, b) => b.weight - a.weight).map(a => <button key={a.position} onClick={() => { setPosition(a.position); setSelectedKey(undefined); }} title={`Inspect source position ${a.position}`}><span>{record?.tokens[a.position]} <small>{a.position}</small></span><i style={{ width: `${a.weight * 100}%` }} /><b>{(a.weight * 100).toFixed(1)}%</b></button>)}</div></> : <p className="unavailable">ATTENTION SOURCE CAPTURE UNAVAILABLE<br /><span>Standard records writes. Source weights need the separate per-head capture rung.</span></p>}
    <dl><dt>Applied write · L2</dt><dd>{fmt(attn?.delta)}</dd></dl>
  </section>;

  return <main className={record ? "observatory has-run" : "observatory arrival"}>
    <header className="masthead"><button className="wordmark" onClick={() => { if (initialAtlas || initialHeadsDemo || initialHeads) { window.location.assign("/"); return; } ++request.current; try { sessionStorage.removeItem("observatory.v3-record"); } catch {} setLoading(false); setPlaying(false); setRecord(undefined); setTarget(undefined); setError(""); }}>LARQL <span>/</span> OBSERVATORY</button><nav className="workspace-nav" aria-label="Workspace"><Link href="/" aria-current="page">Observe</Link><Link href="/extract">Extract</Link></nav><span className="connection"><i /> {record ? "RECORD PLAYER" : "NO RUNNER CONNECTED"}</span></header>
    <input ref={file} type="file" accept="application/json,application/x-ndjson,.json,.jsonl" hidden onChange={e => void importFile(e.target.files)} />
    <input ref={targetAnalysisFile} type="file" accept="application/json,.json" hidden onChange={e => void loadTargetAnalysis(e.target.files)} />
    <input ref={analysisFile} type="file" accept="application/json,.json" hidden onChange={e => void loadAnalysis(e.target.files)} />
    <input ref={comparisonFile} type="file" accept="application/json,application/x-ndjson,.json,.jsonl" hidden onChange={e => void loadComparison(e.target.files)} />
    {error && <div className="error" role="alert"><RefusalReadout title="Recording unavailable" lines={[error]} principle="Missing evidence stays missing." /><button onClick={() => setError("")}>Dismiss ×</button></div>}
    {!record && (initialAtlas || initialHeadsDemo || initialHeads) && !error ? <section className="prompt-room" role="status" aria-live="polite"><p className="eyebrow">{initialHeadsDemo ? "SYNTHETIC HEAD DEMO / AUTHORED EVIDENCE" : "RESIDUAL ATLAS / GRANITE 4.2 3B"}</p><h1>Opening the<br /><em>{initialHeadsDemo ? "head study." : "token geography."}</em></h1><p className="intro">{initialHeadsDemo ? "A separate fixture demonstrating the head → content → experiment workflow." : "Loading eight landmarks and 400 recorded carrier states."}</p></section> : !record ? <>
      <section className="prompt-room"><p className="eyebrow">01 / ENTER THE OBSERVATORY</p><h1>Watch a computation<br /><em>take shape.</em></h1><p className="intro">A prompt. A path through depth. A moment worth inspecting.</p>
        <div className="atlas-entry"><div><p className="eyebrow">RESIDUAL ATLAS / REAL GRANITE RECORDING</p><p>Paris, Berlin, France and five more token landmarks. Follow the carrier through 40 layers.</p></div><Link href="/atlas">Open Residual Atlas ↗</Link></div>
        <div className="prompt-surface"><label htmlFor="prompt">PROMPT</label><textarea id="prompt" value={prompt} onChange={e => setPrompt(e.target.value)} rows={2} spellCheck={false} /><div className="prompt-actions"><div><span className="fixture-mark">SYNTHETIC FIXTURE</span><p>Live test beds: Gemma 3 4B · Granite 3B</p></div><button className="run-button" onClick={() => void launch()} disabled={loading}>{loading ? "OPENING…" : "RUN FIXTURE"}<span>↗</span></button></div></div>
        <div className="quiet-settings"><span>capture / {story.id === "writes" ? "rich fixture" : "standard"}</span><span>projection / {story.id === "address" ? "authored ADDRESS" : "fixed carrier basis"}</span><button onClick={() => void launch(true)} disabled={loading}>Browse fixture model ↗</button><button onClick={() => file.current?.click()}>Open recording ↗</button></div>
      </section>
      <section className="measured-study"><div><p className="eyebrow">VINDEX3 / PARIS GOLDEN RECORD</p><h2>Open the CLI evidence directly.</h2><p>408 writes · 204 head readouts · token 9079 (Paris in the capture study). Original recording, no model execution.</p></div><button disabled={loading} onClick={() => void openParis()}>Open Paris recording ↗</button><button onClick={() => file.current?.click()}>Open run.jsonl ↗</button><a href="/recordings/gemma-paris-v3.jsonl" download>Download original JSONL ↓</a></section>
      <section className="measured-study"><div><p className="eyebrow">RECORDED EXECUTION / GRANITE 4.2 3B</p><h2>A real computation, ready to inspect.</h2><p>The capital of France is · Measured carrier writes · vocabulary through depth · individual attention heads</p></div><button disabled={loading} onClick={() => void openMeasured()}>Open measured run ↗</button><Link href="/heads">Explore real Granite heads →</Link><a href="/recordings/observatory-granite-heads-compact-20260920.heads.json" download>Download raw head capture ↓</a><a href="/recordings/granite-standard.parity.json" target="_blank" rel="noreferrer">Separate parity witness ↗</a></section>
      <section className="stories"><div className="section-label"><span>FOUR STUDIES / ONE INSTRUMENT</span><span>Authored UI fixtures · no model execution</span></div><div className="story-list">{stories.map(s => <button key={s.id} className={s.id === story.id ? "active" : ""} onClick={() => { setStory(s); setPrompt(s.prompt); setError(""); }} aria-pressed={s.id === story.id}><span className="story-letter">{s.letter}</span><strong>{s.title}</strong><span>{s.subtitle}</span><span className="story-arrow">↗</span></button>)}</div></section>
      <footer className="arrival-footer"><span>VINDEX3 / EXECUTION &nbsp; · &nbsp; HAUSE / VISUAL LANGUAGE</span><span>VINDEX3 JSONL + STANDARD RECORD IMPORT</span></footer>
    </> : <>
      <div className="run-heading"><div><p className="eyebrow">{record.story} <span> / {sourceLabel}</span></p><div className="tokens" aria-label="Select token position">{record.tokens.map((token, p) => <button key={p} aria-pressed={p === position} onClick={() => selectPosition(p)}>{token}<small>{p}</small></button>)}</div></div><div className="run-state"><span className={playing ? "on" : ""}>●</span> {playing ? "PLAYING RECORDING" : complete ? (state.status === "completed" ? "EXPLORE" : state.status === "refused" ? "RUN REFUSED" : "INCOMPLETE RECORD") : "PAUSED RECORDING"}<p>{record.layers} {record.vindex3 ? "observed layers" : "layers"} · {record.capture === "standard" ? `Standard stats${lenses?.heads ? " + intrusive head capture" : ""}` : "Rich fixture only"}</p></div></div>
      <Vindex3Evidence record={record} />
      {record.vindex3 && <p className="qualifier">{refreshNotice}</p>}
      <ExecutionProvenance record={record} />
      <div className="plane-bar"><div>{(["Model", "Execution", "Alternatives"] as const).map(p => <button key={p} aria-pressed={plane === p} onClick={() => { setPlane(p); if (p === "Alternatives" && target) setScene("Compare"); else if (p === "Execution" && scene === "Compare") setScene("Map"); }}>{p === "Execution" ? "Run" : p === "Alternatives" ? "What if" : "Model"}<small>{p === "Model" ? "what exists" : p === "Execution" ? "what happens" : "what else"}</small></button>)}</div><span>ONE MODEL WORLD / THREE LENSES</span></div><div className="scene-bar"><nav aria-label="Research views">{(["Context", "Map", "Lenses", "Trace", "Graph", "Matrix", ...(target ? ["Compare"] : [])] as Scene[]).map(s => <button key={s} onClick={() => { setScene(s); setPlane(s === "Compare" ? "Alternatives" : "Execution"); }} aria-current={scene === s ? "page" : undefined}>{s === "Map" ? "Atlas" : s}</button>)}</nav><div className="scene-controls"><label>Position <select value={position} onChange={e => selectPosition(Number(e.target.value))}>{record.tokens.map((t, p) => <option key={p} value={p}>{p} / {t}</option>)}</select></label><button onClick={() => setDetails(d => !d)} aria-expanded={details}>Record {details ? "−" : "+"}</button></div></div>
      <div className="global-depth"><label>Depth <input type="range" min="0" max={record.layers - 1} value={Math.max(0, selected?.layer ?? 0)} onChange={e => seekLayer(Number(e.target.value))} /></label><span>{selected ? labelLayer(selected.layer) : "—"}</span><label>Boundary <select value={role} onChange={e => { const next = e.target.value as Role; const sites = record.events.map((event,i) => ({ sample: event.sample,i })).filter(x => x.sample?.layer === selected?.layer && x.sample?.role === next); const s = sites.find(x => x.sample?.position === position); if (s?.sample) { jump(sites.at(-1)!.i); select(s.sample); } }}><option value="attention_write">Attention / mixer</option><option value="ffn_write" disabled={!record.program.some(p => p.roles.includes("ffn_write"))}>FFN</option></select></label><button onClick={() => { setPlane("Execution"); setScene("Lenses"); setLens("Compare"); }}>Compare states →</button></div>
      {(state.dropped > 0 || state.gaps.length > 0 || (complete && state.status !== "completed") || record.standard?.coverage === "incomplete") && <div role="status" className="loss">INCOMPLETE EVIDENCE · {state.dropped} declared dropped events · {state.gaps.length} sequence gaps. {state.status !== "completed" && complete ? "No terminal completion event. " : ""}No interpolation across missing evidence.</div>}
      {details && <section className="record-details"><Evidence items={[{ label: "Run / " + record.id, status: "OPEN", detail: record.provenance === "executor" ? "Runner-declared source. Import does not verify execution or output parity." : "Authored synthetic data. Not a VINDEX3 execution witness." }, { label: "Capture / " + record.capture, status: "OPEN", detail: `${state.events.length} / ${record.events.length} retained events · ${record.model}` }]} />{record.standard && <div className="standard-provenance"><p>Source: {record.model} · Single carrier · post-add / pre-layer-scale</p><p>Coverage: {record.standard.coverage} · positions {record.standard.observed_positions.join(", ")} · carrier writes only</p><p>Capture timing intrusive: {record.standard.timing_intrusive ? "yes" : "no"} · device readbacks: {record.standard.device_readbacks ?? "not recorded"}</p><p>Basis: {basis?.provider} · width {basis?.hidden} · {basis?.hash}</p><dl>{Object.entries(record.standard.identity).map(([k, v]) => <span className="dl-row" key={k}><dt>{k}</dt><dd>{v}</dd></span>)}</dl><p>Terminal: {full.events.at(-1)?.reason ?? "not recorded"}</p><p>Output parity is a separate executor witness. Replay validation does not adjudicate it.</p>{replayAudit && <div role="status"><h3>Replay fidelity / checked</h3><p>{replayAudit.writes_checked} carrier writes checked exactly against source values · {replayAudit.events_checked} unique events · {replayAudit.duplicate_deliveries} repeated deliveries removed.</p><p>Save/reopen preserved the record. Replay checked at {replayAudit.replay_checkpoints} boundaries. Coverage: {replayAudit.coverage}.</p><p>This checks record-to-view fidelity. Model execution and output parity remain unverified by the UI.</p></div>}</div>}<div><button onClick={save}>Save original record ↓</button><button onClick={() => file.current?.click()}>Open recording ↗</button></div></section>}
      <div className={`workbench ${scene === "Map" && atlasMode && visibleLenses?.token_map && plane === "Execution" ? "atlas-workbench" : ""}`}><section className="main-field">
        {plane === "Model" && <ModelSpace record={record} selectedKey={selectedKey} position={position} touched={state.samples} select={setSelectedKey} />}
        {plane === "Alternatives" && !target && <div className="alternatives-empty"><p className="eyebrow">ALTERNATIVES / CAPABILITY BOUNDARY</p><h2>Every branch needs evidence.</h2><p>Open story D, “Where paths divide,” for a two-record comparison. This record has no alternative execution record.</p><ul><li>Reference ↔ graph walk / no runner capability</li><li>Representation A ↔ B / no runner capability</li><li>Normal operator ↔ WASM / no runner capability</li><li>Base ↔ intervention / no runner capability</li></ul><p>Deterministic does not mean equivalent. Parity witnesses will name their criterion and controls.</p></div>}
        {plane !== "Model" && (plane !== "Alternatives" || target) && <>
        {scene === "Lenses" && <ResearchView fullAnalysis={lenses} key={record.id} record={record} samples={state.samples} selected={selected} position={position} role={role} lenses={visibleLenses} lens={lens} setLens={setLens} selectedHead={selectedHead} setHead={setSelectedHead} select={select} load={() => analysisFile.current?.click()} save={saveAnalysis} loadTarget={() => comparisonFile.current?.click()} target={target} targetSamples={targetState.samples} targetLenses={targetLenses?.run_id === target?.id && targetLenses?.source_sha256 === targetDigest ? targetLenses : undefined} loadTargetAnalysis={() => targetAnalysisFile.current?.click()} />}
        {scene === "Context" && <ContextView lenses={visibleLenses} openLens={l => { setLens(l); setScene("Lenses"); }} record={record} samples={state.samples} all={full.samples} selected={selected} position={position} role={role} answer={answer} select={select} inspect={() => setScene("Trace")} seek={layer => {
          const activeRole = role;
          const matching = record.events.map((e, i) => ({ sample: e.sample, i })).filter(x => x.sample?.layer === layer && x.sample.role === activeRole);
          const site = matching.find(x => x.sample?.position === position) ?? matching.at(-1);
          if (site?.sample) { jump(matching.at(-1)!.i); select(site.sample); }
        }} setRole={nextRole => {
          setRole(nextRole);
          const sites = record.events.map((e, i) => ({ sample: e.sample, i })).filter(x => x.sample && x.sample.layer === selected?.layer && x.sample.role === nextRole);
          const site = sites.find(x => x.sample?.position === position);
          if (site?.sample) { jump(sites.at(-1)!.i); select(site.sample); }
        }} />}
        {scene === "Map" && <div className="atlas-switch"><button aria-pressed={atlasMode} disabled={!visibleLenses?.token_map} onClick={() => setAtlasMode(true)}>Residual Atlas</button><button aria-pressed={!atlasMode || !visibleLenses?.token_map} onClick={() => setAtlasMode(false)}>Carrier projection</button>{!visibleLenses?.token_map && <span>This record has no token landmarks. <Link href="/atlas">Open the Granite Atlas ↗</Link></span>}</div>}
        {scene === "Map" && atlasMode && visibleLenses?.token_map && <ResidualAtlas selectPosition={selectPosition} recordedAtlas={lenses!.token_map!} recordedLenses={lenses} recordedSamples={full.samples} completeRecord={full.status === "completed" && full.dropped === 0 && full.gaps.length === 0 && record.standard?.coverage !== "incomplete"} reveal={s => { const index = record.events.findIndex(e => e.sample && semanticKey(e.sample) === semanticKey(s)); if (index >= 0) { jump(index === record.events.findLastIndex(e => !!e.sample) ? record.events.length - 1 : index); select(s); } }} completed={state.status === "completed"} key={visibleLenses.token_map.basis.hash} atlas={visibleLenses.token_map} record={record} lenses={visibleLenses} samples={state.samples} selected={selected} position={position} select={select} inspect={() => setScene("Trace")} />}
        {((scene === "Map" && (!atlasMode || !visibleLenses?.token_map)) || scene === "Compare") && basis && <><div className="field-caption"><div><p className="eyebrow">{scene === "Compare" ? "TWO RUNS / ONE BASIS" : "WORLDLINE / CARRIER THROUGH DEPTH"}</p><h2>{scene === "Compare" ? "Where paths divide." : record.title}</h2></div><div className="map-controls"><label className="sr-only" htmlFor="basis">Projection</label><select id="basis" value={basis.id} onChange={e => setBasisId(e.target.value)}>{record.bases.map(b => <option key={b.id} value={b.id}>{b.label}</option>)}</select><button onClick={() => setSpatial(s => !s)}>{spatial ? "3D" : "2D"} ↔</button><label className="sr-only" htmlFor="site-role">Map carrier boundary</label><select id="site-role" value={role} onChange={e => setRole(e.target.value as Role)}><option value="ffn_write" disabled={!record.program.some(p => p.roles.includes("ffn_write"))}>After FFN</option><option value="attention_write">After attention</option></select></div></div>
          {scene === "Compare" && !canCompare ? <div className="blank-state">Cannot overlay trajectories: projection basis differs.</div> : <Trajectory samples={visible} allSamples={all} target={scene === "Compare" ? comparison : []} basis={basis} spatial={spatial} selected={selected} select={select} />}
          <div className="basis-line"><span>{basis.label} / {basis.hash.slice(0, 23)}…</span><span>{scene === "Compare" ? `— BASE: ${record.id} · - - TARGET: ${target?.id}` : "FIXED PROJECTION · NOT LITERAL MODEL GEOMETRY"}</span></div>
          {scene === "Compare" && <p className="comparison-note">Aligned by token position + semantic site. Dashed target is the comparison recording. Positional alignment does not imply semantic token equivalence. No causal claim.</p>}
        </>}
        {scene === "Trace" && <><div className="field-caption"><div><p className="eyebrow">DECOMPOSITION / SELECTED LAYER</p><h2>{selected ? labelLayer(selected.layer) : "Awaiting a site"}<em> / a change of state.</em></h2></div></div><EvidenceLevels selected={selected} answer={answer} readoutLabel={readoutLabel} /><div className="unfolding"><div className="write-node attention-node"><span>ATTENTION WRITE</span><strong>{fmt(attn?.delta)}</strong><small>L2 norm · applied delta</small></div><div className="flow-row"><span>BEFORE<br /><b>{fmt(attn?.before ?? selected?.before)}</b></span><i>→</i><div className="carrier-node"><span>CARRIER</span><strong>{fmt(ffn?.norm ?? attn?.norm ?? selected?.norm)}</strong><small>after available write</small></div><i>→</i><span>NEXT<br />SITE</span></div><div className="write-node ffn-node"><span>FFN WRITE</span><strong>{fmt(ffn?.delta)}</strong><small>{ffn ? "L2 norm · applied delta" : "No captured FFN write at this site"}</small></div></div><p className="trace-note">Write norms describe magnitude, not attribution. Norms do not add.</p><TraceTable samples={state.samples.filter(s => s.position === position)} selected={selected} select={select} /></>}
        {scene === "Graph" && <><div className="field-caption"><div><p className="eyebrow">STRUCTURE / FOLLOW THE COMPUTATION</p><h2>{selected ? friendly(selected) : "Select a captured site"}</h2></div></div><EvidenceLevels selected={selected} answer={answer} readoutLabel={readoutLabel} /><div className="graph-controls"><button onClick={() => setUpstream(v => !v)} aria-expanded={upstream}>{upstream ? "COLLAPSE" : "SHOW"} UPSTREAM {upstream ? "−" : "+"}</button><button onClick={() => setDownstream(v => !v)} aria-expanded={downstream}>{downstream ? "COLLAPSE" : "SHOW"} DOWNSTREAM {downstream ? "−" : "+"}</button></div><div className="graph-stage">
          {upstream && <><div className="graph-row">{attn?.sources ? attn.sources.filter(a => a.weight > .12).map(a => <button className="graph-token" key={a.position} onClick={() => { setPosition(a.position); setSelectedKey(undefined); }}>{record.tokens[a.position]}<small>fixture weight {(a.weight * 100).toFixed(0)}%</small></button>) : <div className="graph-node muted">Input position {position}<small>source attribution unavailable</small></div>}</div><div className="graph-edge"><span>{attn?.sources ? "attention weights · synthetic" : "declared program context"}</span></div></>}
          {selected && <button className="graph-node graph-selected" onClick={() => setScene("Trace")}>{friendly(selected)}<small>{record.tokens[position]} · position {position} · open Trace ↗</small></button>}
          {downstream && <><div className="graph-edge" /><div className="graph-row">{state.samples.filter(s => s.position === position && BigInt(s.sequence) > BigInt((selected as Sample & { sequence?: string })?.sequence ?? 0)).slice(0, 2).map(s => <button key={semanticKey(s)} className="graph-node" onClick={() => select(s)}>{friendly(s)}<small>executed successor</small></button>)}</div><div className="graph-edge derived"><span>derived readout of selected state</span></div><div className="graph-node answer-node">{answer}<small>{readoutLabel} {fmt(selected?.logits[answer])}</small></div></>}
        </div><div className="graph-legend"><span>— program dependency</span><span>┄ derived readout</span><span>NO CAUSAL EDGES</span></div><p className="trace-note">Program order is not proof of an answer’s cause. Rich source links are synthetic attention weights.</p></>}
        {scene === "Matrix" && <><div className="field-caption"><div><p className="eyebrow">TOKEN × DEPTH / WRITE MAGNITUDE</p><h2>One field. Every position.</h2></div></div><div className="matrix-legend">L2 applied-write norm · fixed run-wide scale 0–{matrixMax.toPrecision(4)} · — uncaptured</div><div className="matrix-scroll"><table className="matrix"><thead><tr><th>{role === "ffn_write" ? "FFN" : "ATTENTION"} / DEPTH</th>{record.tokens.map((t, i) => <th key={i}>{t}<small>{i}</small></th>)}</tr></thead><tbody>{Array.from({ length: record.layers }, (_, layer) => <tr key={layer}><th>{labelLayer(layer)}</th>{record.tokens.map((_, p) => { const s = state.samples.find(s => s.position === p && s.layer === layer && s.role === role); return <td key={p}><button disabled={!s} className={selected?.layer === layer && position === p ? "selected-cell" : ""} style={{ background: s ? `rgba(178,201,156,${.06 + (s.delta / (matrixMax || 1)) * .78})` : undefined }} onClick={() => s && select(s)} aria-label={`${record.tokens[p]}, ${labelLayer(layer)}, ${s ? fmt(s.delta) : "uncaptured"}`}>{s ? fmt(s.delta) : "—"}</button></td>; })}</tr>)}</tbody></table></div></>}
      </>}
      </section><aside className="inspector" aria-label="Selected site inspector"><div className="inspector-heading"><span className="eyebrow">SELECTED / {selectedKey ? "PINNED" : "FOLLOWING"}</span><h2>{selected ? labelLayer(selected.layer) : "—"}</h2><p>{record.tokens[position]} <span>POSITION {position}</span></p><div className="site-switch">{related.map(s => <button key={s.role} onClick={() => select(s)} aria-pressed={selected?.role === s.role}>{s.role === "embedding" ? "Embedding" : s.role === "attention_write" ? "Attention" : "FFN"}</button>)}</div></div>
        <CarrierReadout record={{ ...record, bases: basis ? [basis] : [] }} selected={selected} />
        {attention}
        <section className="readout"><h3>FFN <span>APPLIED WRITE</span></h3><dl><dt>Write · L2 norm</dt><dd>{fmt(ffn?.delta)}</dd><dt>Input carrier norm</dt><dd>{fmt(ffn?.before)}</dd><dt>Output carrier norm</dt><dd>{fmt(ffn?.norm)}</dd></dl>{!ffn && <p className="qualifier">No FFN write captured here. A layer boundary does not imply an FFN.</p>}</section>
        {record.vindex3 && <Vindex3Readout lenses={visibleLenses} selected={selected} />}
        <section className="readout"><h3>Selected readout <span>{record.standard ? "RAW / DOT-RAW-V1" : "DERIVED / FIXTURE"}</span></h3><dl>{record.answers.map(a => <span className="dl-row" key={a}><dt>{a}</dt><dd className={a === answer ? "accent" : ""}>{fmt(selected?.logits[a])}</dd></span>)}</dl><p className="qualifier">Predeclared tokens · {readoutLabel}.<br />No final norm / softmax. {record.vindex3 ? "Normalized readouts appear separately in Answer lens." : "Rank and probability unavailable."}</p></section>
        <div className="inspector-end">{selected && <><span>{labelRole(selected.role)}</span><span>{selected.duration_ns === undefined ? "Site duration unavailable" : `duration ${(Number(selected.duration_ns) / 1e6).toFixed(3)} ms · synthetic`}</span></>}<button onClick={() => { setSelectedKey(undefined); setPosition(record.tokens.length - 1); }}>Follow final position ↗</button></div>
      </aside></div>
      <section className="answer-strip"><div><p className="eyebrow">ANSWER / THROUGH DEPTH</p><label className="sr-only" htmlFor="answer">Selected answer token</label><select id="answer" value={answer} onChange={e => setAnswer(e.target.value)}>{record.answers.map(a => <option key={a}>{a}</option>)}</select><div className="answer-metrics"><button onClick={() => setMetric("readout")} aria-pressed={metric === "readout"}>readout</button><button onClick={() => setMetric("difference")} aria-pressed={metric === "difference"}>readout difference</button></div>{metric === "difference" && <label>versus <select value={against} onChange={e => setAgainst(e.target.value)}>{record.answers.map(a => <option key={a}>{a}</option>)}</select></label>}</div>{record.answers.length ? <AnswerPlot samples={visible} answer={answer} selected={selected} select={select} metric={metric} against={against} /> : <p className="unavailable">{record.vindex3 && lenses?.vocabulary ? <button onClick={() => { setScene("Lenses"); setLens("Logits"); }}>Inspect recorded answer lens →</button> : "SELECTED-TOKEN PROBES NOT CAPTURED"}</p>}<div className="output"><p className="eyebrow">RECORDED OUTPUT</p><strong>{state.output || (record.vindex3 ? "Not recorded" : "…")}</strong><span>{sourceLabel}</span></div></section>
      <section className="transport"><div className="layer-rail" aria-label="Layer progression">{[-1, ...Array.from({ length: record.layers }, (_, l) => l)].map(l => { const s = state.samples.filter(s => s.position === position && s.layer === l).at(-1); return <button key={l} disabled={!s} aria-pressed={selected?.layer === l} className={`${s ? "visited" : "future"} ${selected?.layer === l ? "selected" : ""}`} onClick={() => s && select(s)}>{labelLayer(l)}<span>{selected?.layer === l ? "●" : s ? "·" : ""}</span></button>; })}<button disabled={!state.output && !record.vindex3} className={state.status === "completed" ? "visited" : "future"} onClick={() => jump(record.events.length - 1)}>{record.vindex3 ? "END" : "OUT"}<span>{state.status === "completed" ? "✓" : ""}</span></button></div><div className="replay-controls"><button onClick={() => step(-1)} aria-label="Previous site">←</button><button className="play-button" onClick={togglePlay}>{playing ? "PAUSE" : complete ? "REPLAY" : "PLAY"}</button><button onClick={() => step(1)} aria-label="Next site">→</button><button onClick={() => step(1, true)}>STEP LAYER</button><label className="sr-only" htmlFor="timeline">Replay event cursor</label><input id="timeline" type="range" min={1} max={record.events.length - 1} value={Math.max(1, cursor)} onChange={e => jump(Number(e.target.value))} /><label className="sr-only" htmlFor="speed">Replay speed</label><select id="speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={.25}>0.25×</option><option value={1}>1×</option><option value={4}>4×</option></select><button onClick={() => jump(record.events.length - 1)}>INSTANT</button><span>{state.events.length} / {record.events.length}</span></div></section>
      <footer className="run-footer"><span>{sourceLabel} · {record.id} · RECORDED REPLAY</span><span>SPACE play / pause &nbsp; ← → site &nbsp; SHIFT ← → layer &nbsp; J K position</span></footer>
    </>}
  </main>;
}

function TraceTable({ samples, selected, select }: { samples: Sample[]; selected?: Sample; select: (s: Sample) => void }) {
  const layers = [...new Set(samples.map(s => s.layer))].filter(l => l >= 0);
  return <div className="trace-table-wrap"><table className="trace-table"><caption>Every captured layer · L2 norms, absolute values</caption><thead><tr><th>Layer</th><th>Attention write</th><th>FFN write</th><th>Carrier after</th></tr></thead><tbody>{layers.map(l => { const a = samples.find(s => s.layer === l && s.role === "attention_write"), f = samples.find(s => s.layer === l && s.role === "ffn_write"), last = f ?? a; return <tr key={l} className={selected?.layer === l ? "active" : ""}><th><button onClick={() => last && select(last)}>{labelLayer(l)}</button></th>{[a?.delta, f?.delta, last?.norm].map((v, i) => <td key={i}><span>{fmt(v)}</span>{v !== undefined && i < 2 && <i style={{ width: `${Math.min(v / .6, 1) * 50}%` }} />}</td>)}</tr>; })}</tbody></table></div>;
}
