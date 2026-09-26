"""Synthetic lens walkthrough. No model execution or experimental claims."""
import hashlib
import json
import math
from pathlib import Path
root = Path(__file__).resolve().parents[1] / 'public' / 'fixtures'
def distribution(words, values):
    exps = [math.exp(x-max(values)) for x in values]
    probs = [x/sum(exps) for x in exps]
    order = sorted(range(len(words)), key=lambda i: (-values[i], i))
    return [dict(token_id=i,token=words[i],probability=probs[i],logit=values[i],rank=order.index(i)+1) for i in order]
for name in ['answer','writes','address','counterfactual','counterfactual-target']:
    raw = (root / f'{name}.json').read_bytes()
    record = json.loads(raw)
    words = record['answers'] + ['Europe','London','France']
    vocabulary, heads = [], []
    for event in record['events']:
        s = event.get('sample')
        if not s: continue
        key = f"{s['position']}:{s['layer']}:{s['role']}:main:residual"
        dist = distribution(words,[s['logits'][t] for t in record['answers']]+[.2,-.1,.4])
        vocabulary.append(dict(site=key,top=dist,targets=[next(x for x in dist if x['token_id']==0)],entropy=-sum(p['probability']*math.log(p['probability']) for p in dist)))
        if s['role'] != 'attention_write': continue
        for head in range(8):
            peak = math.exp(-((s['layer']-(23 if head==3 else 18+head))/2.8)**2)
            amplitude = (-1 if head in [1,6] else 1)*(.02+peak*(2.4 if head==3 else .6))*(s['position']+1)/len(record['tokens'])
            coeffs = [amplitude*x for x in [1,.13,-.08,.03,.01,.004]]
            energy = sorted([c*c for c in coeffs],reverse=True)
            dims = [next(i+1 for i in range(6) if sum(energy[:i+1])/sum(energy)>=t) for t in [.95,.99,.999]]
            scores = [1+(8 if p==max(0,s['position']-1) else 0)+(head+1 if p==1 else 0) for p in range(s['position']+1)]
            heads.append(dict(site=key,head=head,target=words[0],dla=coeffs[0],sources=[dict(position=p,weight=w/sum(scores)) for p,w in enumerate(scores)],content=dict(method='Synthetic vector coordinates in orthonormal toy token basis',basis='fixture-six-axis-token-basis-v1',vector_norm=math.sqrt(sum(energy)),projections=[dict(token=t,token_id=i,coefficient=c) for i,(t,c) in enumerate(zip(words,coeffs))],dimensionality=dict(method='Cumulative squared-coordinate energy in six orthonormal fixture axes',d95=dims[0],d99=dims[1],d999=dims[2],dimensions=6),orthogonality=dict(method='Cosine between distinct orthonormal fixture axes; not model embeddings',pairs=[dict(label=f'{words[0]} axis / {t} axis',cosine=0) for t in words[1:3]]))))
    experiments=[]
    for h in heads:
        if h['head']!=3 or ':23:attention_write:' not in h['site']: continue
        before=next(v['top'] for v in vocabulary if v['site']==h['site'])
        values=[next(p['logit'] for p in before if p['token_id']==i) for i in range(6)]
        values[0]-=h['dla']
        after=distribution(words,values)
        kl=sum(p['probability']*math.log(p['probability']/next(q['probability'] for q in after if q['token_id']==p['token_id'])) for p in before)
        experiments.append(dict(site=h['site'],head=3,kind='ablate',parent=record['id'],fork=f"{record['id']}-authored-fork-{h['site'].split(':')[0]}",method='Synthetic illustration: subtract toy direct coefficient and re-softmax six logits. No model intervention.',witness='synthetic-six-token-distribution-v1',target=words[0],before=before,after=after,kl=dict(direction='before-to-after',scope='full-vocabulary',nats=kl)))
    kv_bytes=len(record['tokens'])*32*2*2*16*4
    sidecar=dict(schema='larql.observatory.lenses.v1',run_id=record['id'],source_sha256=hashlib.sha256(raw).hexdigest(),provenance='synthetic',provider='Authored Anatomist workflow / six-token toy vocabulary',vocabulary=dict(method='Softmax of six authored fixture logits; no model final normalization',basis='fixture-six-token-readout-v1',vocab_size=6,rows=vocabulary),heads=dict(method='Synthetic orthonormal target-direction dot product; not captured model heads',basis='fixture-six-axis-token-basis-v1',rows=heads),spans=[dict(start=0,end=min(1,len(record['tokens'])-1),label='Authored query span',method='Manual fixture annotation; not detected semantics')],kv=dict(sequence=record['events'][-1]['sequence'],method='Synthetic dense KV budget: positions × 32 layers × 2 KV heads × K/V × 16 dimensions × 4 bytes',total_bytes=kv_bytes,key_bytes=kv_bytes//2,value_bytes=kv_bytes//2,content=dict(bytes=12,method='One toy direction ID (4 bytes) plus coefficient (8 bytes)',scope='Single selected direction only; no demonstrated cache replacement',witness='Authored encoding example / not an experimental result')),experiments=experiments)
    (root/f'{name}.lenses.json').write_text(json.dumps(sidecar,separators=(',',':'))+'\n')
