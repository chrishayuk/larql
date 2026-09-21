/** Read adapters for the shipped capabilities, SystemPlan v6 and inspection surfaces. */
export type JsonObject = Record<string, unknown>;
export function object(v: unknown, label: string): JsonObject {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error(`Invalid ${label}.`);
  return v as JsonObject;
}
function string(v: unknown, label: string): string {
  if (typeof v !== 'string' || !v.length || v.length > 8192) throw new Error(`Invalid ${label}.`);
  return v;
}
function array(v: unknown, label: string): unknown[] {
  if (!Array.isArray(v) || v.length > 100000) throw new Error(`Invalid ${label}.`);
  return v;
}
function count(v: unknown): number {
  if (typeof v !== 'number' || !Number.isSafeInteger(v) || v < 0) throw new Error('Invalid count.');
  return v;
}
function bool(v: unknown): boolean {
  if (typeof v !== 'boolean') throw new Error('Missing explicit capability/verdict.');
  return v;
}
export interface Capabilities { schema: 1; profile: string; server: JsonObject; routes: string[]; sources: { plan: { local: boolean; hf: boolean }; encode: { local: boolean; hf: boolean } }; explorer: Record<string, boolean> }
export function parseCapabilities(v: unknown): Capabilities {
  const c = object(v, 'capabilities');
  if (c.schema !== 1 || c.object !== 'capabilities') throw new Error('Unsupported runner capabilities schema.');
  const sources = object(c.sources, 'sources'), plan = object(sources.plan, 'plan capabilities'), encode = object(sources.encode, 'encode capabilities');
  const routes = array(c.routes, 'routes').map(r => string(r, 'route'));
  const explorer = object(c.explorer, 'explorer');
  for (const b of Object.values(explorer)) bool(b);
  for (const [name, op] of [['plan', plan], ['encode', encode]] as const) {
    if ((bool(op.local) || bool(op.hf)) && !routes.includes(`/v1/${name}`)) throw new Error(`Runner advertises ${name} without its route.`);
  }
  for (const name of ['components', 'representations', 'provenance', 'authority']) if (explorer[name] === true && !routes.includes(`/v1/${name}`)) throw new Error(`Runner advertises ${name} without its route.`);
  return { schema: 1, profile: string(c.profile, 'profile'), server: object(c.server, 'server'), routes, sources: { plan: { local: bool(plan.local), hf: bool(plan.hf) }, encode: { local: bool(encode.local), hf: bool(encode.hf) } }, explorer: explorer as Record<string, boolean> };
}
export function parseSources(input: string): string[] {
  const sources = input.split('\n').map(s => s.trim()).filter(Boolean);
  if (!sources.length || sources.length > 8) throw new Error('Enter one to eight sources, one per line.');
  for (const s of sources) {
    if (s.length > 4096 || /[\x00-\x1f\x7f]/.test(s) || s.startsWith('-')) throw new Error('Invalid source path/reference.');
    if (s.includes('://') && !/^hf:\/\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:@[^\s]+)?$/.test(s)) throw new Error('Use hf://owner/repo[@revision] or a local checkpoint path.');
  }
  return sources;
}
export function canPlan(caps: Capabilities, sources: string[]): boolean {
  return sources.length > 0 && sources.every(s => caps.sources.plan[s.startsWith('hf://') ? 'hf' : 'local']);
}
export function runnerBase(raw: string): string {
  const url = new URL(raw);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || (url.pathname !== '/' && url.pathname !== '')) throw new Error('Use the runner HTTP(S) origin, without credentials, query or path.');
  return url.origin;
}
export interface Finding { id: number; artifact: string; component: string; subject: string; category: string; class: string; detail: string; declared?: unknown; resolved?: unknown; carriage?: unknown; cluster?: unknown }
export interface Plan { raw: JsonObject; artifacts: { name: string; source: { path: string; revision?: string | null }; model_type: string }[]; admissible: boolean; findings: Finding[]; summary: Record<string, number>; capabilities: { capability: string; admissible: boolean; available: boolean; supported: boolean; blocking: number; blocker_ids: number[] }[]; graph: Graph; planner: JsonObject }
export interface Graph { components: JsonObject[]; objects: JsonObject[]; edges: JsonObject[] }
export function parseGraph(value: unknown): Graph {
  const g = object(value, 'system graph');
  if (g.schema !== 6) throw new Error('Unsupported system graph schema; expected 6.');
  const components = array(g.components, 'components').map(v => object(v, 'component'));
  const objects = array(g.objects, 'objects').map(v => object(v, 'logical object'));
  const edges = array(g.edges, 'edges').map(v => object(v, 'edge'));
  const ids = new Set<string>();
  for (const c of components) { const id = string(c.id, 'component ID'); if (ids.has(id)) throw new Error('Duplicate component.'); ids.add(id); count(c.num_layers); count(c.hidden_size); }
  const objectIds = new Set<string>();
  for (const o of objects) { const id = string(o.id, 'object ID'); if (objectIds.has(id) || !ids.has(string(o.component, 'object component'))) throw new Error('Duplicate or unowned object.'); objectIds.add(id); }
  for (const e of edges) if (!ids.has(string(e.producer_component, 'producer')) || !ids.has(string(e.consumer_component, 'consumer')) || !objectIds.has(string(e.consumer_object, 'consumer object'))) throw new Error('Unresolved graph edge.');
  return { components, objects, edges };
}
export function parsePlan(v: unknown): Plan {
  const r = object(v, 'plan');
  if (r.schema !== 6) throw new Error('Unsupported SystemPlan schema; expected 6.');
  const planner = object(r.planner, 'planner'); string(planner.package, 'planner package'); string(planner.package_version, 'planner version'); count(planner.semantics_version);
  const findings: Finding[] = [], ids = new Set<number>();
  const artifacts = array(r.artifacts, 'artifacts').map(v => {
    const a = object(v, 'artifact'), src = object(a.source, 'source');
    const name = string(a.name, 'artifact name');
    if (src.revision !== undefined && src.revision !== null) string(src.revision, 'source revision');
    for (const value of array(a.findings, 'findings')) {
      const p = object(value, 'planned finding'), f = object(p.finding, 'finding'), id = count(p.id);
      if (ids.has(id)) throw new Error('Duplicate finding ID.'); ids.add(id);
      const category = string(f.category, 'category');
      if (!['representable', 'mismatched', 'unrepresented', 'interface'].includes(category)) throw new Error('Unknown finding category.');
      findings.push({ id, artifact: name, component: string(f.component, 'component'), subject: string(f.subject, 'subject'), category, class: string(f.class, 'semantic class'), detail: string(f.detail, 'finding detail'), declared: f.declared, resolved: f.resolved, carriage: f.carriage, cluster: p.cluster });
    }
    return { name, source: { path: string(src.path, 'source path'), revision: src.revision as string | null | undefined }, model_type: string(a.model_type, 'model type') };
  });
  if (!artifacts.length) throw new Error('Plan has no artifact.');
  const summary = object(r.summary, 'summary');
  const counts = Object.fromEntries(['representable', 'mismatched', 'unrepresented', 'interfaces', 'blocking'].map(k => [k, count(summary[k])]));
  const admissible = bool(r.admissible);
  if (admissible !== (counts.blocking === 0)) throw new Error('Plan admission contradicts its blocking count.');
  const capabilities = array(r.capabilities ?? [], 'capabilities').map(v => {
    const c = object(v, 'capability'), blockers = array(c.blocker_ids, 'blocker IDs').map(count), blocking = count(c.blocking);
    if (new Set(blockers).size !== blockers.length || blockers.some(id => !ids.has(id)) || blocking !== blockers.length || bool(c.admissible) !== (blocking === 0)) throw new Error('Capability blockers contradict findings.');
    return { capability: string(c.capability, 'capability'), admissible: bool(c.admissible), available: bool(c.available), supported: bool(c.supported), blocking, blocker_ids: blockers };
  });
  return { raw: r, artifacts, admissible, findings, summary: counts, capabilities, planner, graph: parseGraph(r.graph) };
}
export function admitted(plan: Plan | undefined, scope: 'whole' | 'text'): boolean {
  if (!plan) return false;
  if (scope === 'whole') return plan.admissible;
  const c = plan.capabilities.find(c => c.capability === 'text_generation');
  return !!c && c.admissible && c.available && c.supported;
}
export function shellQuote(s: string): string { return `'${s.replaceAll("'", "'\\''")}'`; }
export function commands(sources: string[], output: string, scope: 'whole' | 'text') {
  for (const source of sources) { if (parseSources(source).length !== 1) throw new Error('Invalid command source.'); }
  if (!output.trim() || output.startsWith('-') || /[\x00-\x1f\x7f]/.test(output)) throw new Error('Enter a new output container path.');
  const args = sources.map(shellQuote).join(' '), dest = shellQuote(output);
  return { plan: `larql vindex3 plan ${args} --output plan.json`, encode: `larql vindex3 encode ${args} --output ${dest}${scope === 'text' ? ' --capability text-generation' : ''}`, inspect: `larql vindex3 inspect ${dest} --verify --execution-complete --json > inspection.json` };
}
export interface Inspection { raw: JsonObject; graph: Graph; index: JsonObject; defects: unknown[] }
export function parseInspection(v: unknown): Inspection {
  const raw = object(v, 'inspection'), index = object(raw.index, 'container index');
  if (index.version !== 3) throw new Error('Expected a VINDEX3 container inspection.');
  string(index.model, 'model');
  for (const value of Object.values(object(index.representations, 'representations'))) {
    const r = object(value, 'representation'); string(r.object, 'representation object'); string(r.encoding, 'encoding');
    count(r.tensor_count); count(r.payload_bytes); string(r.payload_sha256, 'payload digest'); string(r.segment_sha256, 'segment digest');
  }
  return { raw, graph: parseGraph(raw.graph), index, defects: array(raw.defects, 'inspection defects') };
}
export const factoryStages = ['preflight', 'fetch', 'extract', 'slice', 'manifest', 'verify', 'publish', 'release'];
export function parseBuild(v: unknown): JsonObject {
  const r = object(v, 'Factory build'); string(r.build_id, 'build ID'); string(r.recipe_name, 'recipe name');
  if (!['passed', 'failed'].includes(String(r.status))) throw new Error('Unknown build status.');
  if (r.status === 'failed' && (!factoryStages.includes(string(r.stage, 'failed stage')) || typeof r.message !== 'string')) throw new Error('Missing failed stage/message.');
  for (const item of array(r.outputs, 'outputs')) { const o = object(item, 'output'); string(o.preset, 'preset'); bool(o.released); if (o.size_bytes !== null) count(o.size_bytes); if (o.repo !== null) string(o.repo, 'repo'); }
  return r;
}
/** Bounded, direct browser requests. Never proxies credentials through the hosted UI. */
export async function runnerRequest(base: string, path: string, token: string, signal: AbortSignal, body?: unknown): Promise<unknown> {
  const response = await fetch(`${base}${path}`, { method: body === undefined ? 'GET' : 'POST', headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...(token ? { Authorization: `Bearer ${token}` } : {}) }, body: body === undefined ? undefined : JSON.stringify(body), signal, credentials: 'omit', redirect: 'error' });
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Runner returned no response body.');
  let length = 0; const chunks: Uint8Array[] = [];
  try { while (true) { const { done, value } = await reader.read(); if (done) break; length += value.length; if (length > 10_000_000) throw new Error('Runner response exceeds 10 MB.'); chunks.push(value); } } finally { await reader.cancel(); }
  const bytes = new Uint8Array(length); let offset = 0; for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  const text = new TextDecoder().decode(bytes);
  if (!response.ok) throw new Error(`Runner returned ${response.status}: ${text.slice(0, 600)}`);
  return JSON.parse(text);
}
