import assert from 'node:assert/strict';
import test from 'node:test';
import { admitted, canPlan, commands, parseBuild, parseCapabilities, parseGraph, parseInspection, parsePlan, parseSources, runnerBase, runnerRequest } from '../lib/extraction.ts';
const graph = () => ({ schema: 6, components: [{ id: 'text', role: 'decoder', num_layers: 34, hidden_size: 2560 }], objects: [{ id: 'text.layers', component: 'text', kind: 'decoder_stack' }], edges: [] });
const plan = () => ({ schema: 6, planner: { package: 'larql-vindex', package_version: 'test', semantics_version: 22 }, artifacts: [{ name: 'contract', source: { path: 'hf://test/model', revision: 'a'.repeat(40) }, model_type: 'test', findings: [{ id: 0, cluster: 'vision', finding: { category: 'unrepresented', class: 'unsupported_component', component: 'vision', subject: 'tower', detail: 'Authored contract case' } }] }], interfaces: [], graph: graph(), admissible: false, summary: { representable: 0, mismatched: 0, unrepresented: 1, interfaces: 0, blocking: 1 }, capabilities: [{ capability: 'text_generation', admissible: true, available: true, supported: true, blocking: 0, blocker_ids: [] }] });
const caps = () => ({ object: 'capabilities', schema: 1, profile: 'public_explorer', server: { name: 'larql-server' }, routes: ['/v1/plan', '/v1/components'], sources: { plan: { local: false, hf: true }, encode: { local: false, hf: false } }, explorer: { components: true, representations: false } });
test('capabilities govern source forms; hostname never grants local planning or encoding', () => {
 const c = parseCapabilities(caps());
 assert.equal(canPlan(c, ['hf://test/model']), true); assert.equal(canPlan(c, ['/local/model']), false);
 assert.equal(canPlan(c, ['hf://test/model', '/local/model']), false); assert.equal(canPlan(c, []), false);
 const liar = caps(); liar.sources.encode.hf = true; assert.throws(() => parseCapabilities(liar), /without its route/);
 const malformed = caps(); malformed.sources.plan.hf = 'true'; assert.throws(() => parseCapabilities(malformed));
 assert.throws(() => parseCapabilities({ ...caps(), schema: 2 }), /Unsupported/);
});
test('scope separates whole-model completeness from executable text capability', () => {
 const p = parsePlan(plan()); assert.equal(admitted(p, 'whole'), false); assert.equal(admitted(p, 'text'), true);
 for (const key of ['admissible','available','supported']) { const x = plan(); x.capabilities[0][key] = false; if (key === 'admissible') { x.capabilities[0].blocking = 1; x.capabilities[0].blocker_ids = [0]; } assert.equal(admitted(parsePlan(x), 'text'), false); }
 assert.equal(admitted(undefined, 'text'), false);
});
test('plans refuse wrong schema, unattributed verdicts and dangling blocker references', () => {
 for (const mutate of [p => p.schema = 4, p => delete p.planner, p => p.admissible = true, p => p.capabilities[0].blocker_ids = [100], p => p.graph.schema = 99]) { const p = plan(); mutate(p); assert.throws(() => parsePlan(p)); }
 const p = parsePlan(plan()); assert.equal(p.artifacts[0].source.revision, 'a'.repeat(40)); assert.equal(p.findings[0].subject, 'tower');
 assert.deepEqual(parsePlan(JSON.parse(JSON.stringify(p.raw))), p);
});
test('model graph preserves real ownership and refuses invented or dangling relationships', () => {
 const g = graph(); g.objects[0].component = 'missing'; assert.throws(() => parseGraph(g), /unowned/);
 const h = graph(); h.edges = [{ producer_component: 'text', consumer_component: 'text', consumer_object: 'missing' }]; assert.throws(() => parseGraph(h), /Unresolved/);
 assert.equal(parseGraph(graph()).edges.length, 0);
});
test('inspection remains generation-aware and malformed representation data refuses before rendering', () => {
 const i = { index: { version: 3, model: 'test', representations: {} }, graph: graph(), defects: [] };
 assert.deepEqual(parseInspection(i).defects, []);
 assert.throws(() => parseInspection({ ...i, index: { ...i.index, version: 2 } }), /VINDEX3/);
 assert.throws(() => parseInspection({ ...i, index: { ...i.index, representations: { invalid: null } } }));
});
test('source validation and shell quoting preserve literal paths without command substitution', () => {
 assert.deepEqual(parseSources('hf://google/gemma-3-4b-it@revision\n /models/with space'), ['hf://google/gemma-3-4b-it@revision', '/models/with space']);
 for (const source of ['', 'https://example.com/model', '--help', Array(9).fill('/source').join('\n')]) assert.throws(() => parseSources(source));
 const c = commands(["/model's/$(touch never)"], '/output with space', 'text');
 assert.equal(c.encode, "larql vindex3 encode '/model'\\''s/$(touch never)' --output '/output with space' --capability text-generation");
 assert.match(c.inspect, /--verify --execution-complete --json/); assert.doesNotMatch(c.encode, /extract-index/);
 assert.throws(() => commands(['/source'], '--flag', 'whole'));
 assert.throws(() => commands(['--help'], '/output', 'whole'));
 assert.throws(() => commands(['/a\n/b'], '/output', 'whole'));
});
test('runner connection accepts origins only and never embedded credentials', () => {
 assert.equal(runnerBase('http://localhost:8080/'), 'http://localhost:8080');
 for (const url of ['https://u:p@example.com', 'file:///tmp', 'https://example.com/v1', 'https://example.com?token=secret']) assert.throws(() => runnerBase(url));
});
test('Factory record reports terminal failure without fabricating step timings', () => {
 const b = { build_id: 'test', recipe_name: 'test', status: 'failed', stage: 'verify', message: 'checksum mismatch', outputs: [{ preset: 'full', size_bytes: 12, repo: null, released: false }] };
 assert.deepEqual(parseBuild(b), b); assert.throws(() => parseBuild({ ...b, stage: 'made-up' }));
 assert.throws(() => parseBuild({ ...b, outputs: [{ preset: 'full', size_bytes: null, repo: null, released: 'true' }] }));
});
test('runner requests use exact existing API, omit cookies and disallow redirects', async t => {
 const calls = []; t.mock.method(globalThis, 'fetch', async (url, init) => { calls.push({ url, init }); return new Response(JSON.stringify({ ok: true })); });
 const controller = new AbortController();
 assert.deepEqual(await runnerRequest('https://runner.test', '/v1/plan', 'secret', controller.signal, { sources: ['hf://test/model'] }), { ok: true });
 assert.equal(calls[0].url, 'https://runner.test/v1/plan'); assert.equal(calls[0].init.method, 'POST');
 assert.equal(calls[0].init.credentials, 'omit'); assert.equal(calls[0].init.redirect, 'error');
 assert.equal(calls[0].init.headers.Authorization, 'Bearer secret'); assert.deepEqual(JSON.parse(calls[0].init.body), { sources: ['hf://test/model'] });
});
