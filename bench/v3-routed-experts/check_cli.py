#!/usr/bin/env python3
"""Correctness receipt only: local or exact HTTP expert-grid CLI runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--topology', choices=['local', 'one', 'experts', 'mixed'], default='local')
p.add_argument('--reference', type=Path)
p.add_argument('--port', type=int, default=19281)
p.add_argument('--max-tokens', type=int, default=32)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
a.out.mkdir(parents=True, exist_ok=False)
model = a.model.resolve()
cli, server = [root / 'target/release' / name for name in ['larql', 'larql-server']]
prompts = json.loads(Path(__file__).with_name('prompts.json').read_text())
settings = dict(LARQL_CPU_WORKERS='8', RAYON_NUM_THREADS='8', VECLIB_MAXIMUM_THREADS='1', TOKIO_WORKER_THREADS='2', LARQL_KV_ENGINE='')
env = dict(os.environ, **settings)
def save(name, value):
    (a.out / name).write_text(json.dumps(value, indent=2) + '\n')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
save('manifest.json', dict(topology=a.topology, model=str(model), max_tokens=a.max_tokens,
    revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
    settings=settings, binary_sha256={f.name:digest(f) for f in [cli,server]},
    metadata_sha256={f:digest(model/f) for f in ['index.json','system_graph.json']},
    purpose='correctness only; no exclusive timing window or performance claim'))
# GPT-OSS 20B: 24 layers and 32 experts per layer. CLI endpoints are inclusive.
layouts = {'local': [], 'one': [(0,23,0,31)], 'experts': [(0,23,0,15),(0,23,16,31)],
           'mixed': [(0,11,0,31),(12,23,0,7),(12,23,8,31)]}
processes, logs, urls, bindings = [], [], [], []
try:
    for i, (start,end,first,last) in enumerate(layouts[a.topology]):
        url = f'http://127.0.0.1:{a.port+i}'
        cmd = [str(server),str(model),'--host','127.0.0.1','--port',str(a.port+i),
               '--ffn-only','--layers',f'{start}-{end}','--experts',f'{first}-{last}']
        log = (a.out/f'worker-{i}.log').open('w'); logs.append(log)
        proc = subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append(proc); deadline = time.monotonic()+600
        while True:
            if proc.poll() is not None: raise RuntimeError(f'worker {i} exited')
            try:
                with urllib.request.urlopen(url+'/v1/vindex3/experts',timeout=2) as r: binding=json.load(r)
                break
            except OSError:
                if time.monotonic()>deadline: raise TimeoutError(f'worker {i}')
                time.sleep(.25)
        bindings.append(binding); urls.append(url); print(f'worker {i} ready',flush=True)
    save('bindings.json',bindings)
    reference = json.loads(a.reference.read_text()) if a.reference else None
    results = {}
    for name, prompt in prompts.items():
        cmd = [str(cli),'run',str(model),prompt['text'],'--max-tokens',str(a.max_tokens),'--emit-ids']
        if urls: cmd += ['--v3-ffn-shards',','.join(urls),'--v3-ffn-wire','binary']
        print(f'running {a.topology} {name}',flush=True)
        with (a.out/f'{name}.stdout').open('w') as stdout, (a.out/f'{name}.stderr').open('w') as stderr:
            subprocess.run(cmd,cwd=root,env=env,stdout=stdout,stderr=stderr,check=True,timeout=1200)
        text = (a.out/f'{name}.stderr').read_text()
        ids = {key:json.loads(re.search(rf'{key} ids: (\[.*\])',text)[1]) for key in ['prompt','generated']}
        assert ids['prompt']==prompt['ids'], (name,'tokenizer drift')
        if reference: assert ids==reference[name], (name,'token mismatch')
        results[name]=ids
        save('ids.json',results)
        print(f'{name}: {len(ids["prompt"])} prompt + {len(ids["generated"])} generated, matched={bool(reference)}',flush=True)
    save('verification.json',dict(complete=True,reference=str(a.reference) if a.reference else None,matched=bool(reference),prompts=len(results)))
finally:
    for proc in processes: proc.terminate()
    for proc in processes:
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill();proc.wait()
    for log in logs: log.close()
