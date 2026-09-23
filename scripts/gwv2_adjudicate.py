#!/usr/bin/env python3
"""Adjudicate the entire frozen V2 surface; sham subsets are diagnostic only."""
import argparse
import json
from pathlib import Path
import numpy as np
from gwv2_amend import validate, seal_json
from gwv2_io import array, checked_manifest
from gwv2_population import sha256
from gwv2_prepare import validate_execution

DEPTHS=[0,4,8,12,16,20,23]
RANKS=[8,16,32,64,128]


def distributions(logits, z=False):
    logits=np.asarray(logits,dtype=np.float64)
    if z:
        sd=logits.std(axis=-1,keepdims=True)
        if np.any(sd<=0) or not np.isfinite(sd).all():
            raise ValueError('invalid row z-score')
        logits=(logits-logits.mean(axis=-1,keepdims=True))/sd
    x=np.exp(logits-logits.max(axis=-1,keepdims=True))
    return x/x.sum(axis=-1,keepdims=True)


def js(a,b):
    middle=(a+b)*0.5
    def term(p):
        ratio=np.ones_like(p)
        np.divide(p,middle,out=ratio,where=p>0)
        return np.sum(p*np.log(ratio),axis=-1)
    return (term(a)+term(b))*0.5


def group_js(data,groups):
    a,b,c=(data[groups[:,i]] for i in range(3))
    return (js(a,b)+js(a,c)+js(b,c))/3


def effect(before,after,facts,controls):
    return (group_js(before,facts)-group_js(after,facts))-(group_js(before,controls)-group_js(after,controls))


def max_t_lower(point,draws):
    se=draws.std(axis=0,ddof=1)
    if not np.isfinite(draws).all() or not np.isfinite(se).all() or np.any(se<=0):
        raise ValueError('nonfinite or zero bootstrap standard error')
    t=((point-draws)/se).max(axis=1)
    critical=float(np.quantile(t,0.95))
    return point-critical*se,critical


def frontier(clears):
    result=[]
    for d in range(3):
        for r in range(4):
            cells=[d*5+r,d*5+r+1,(d+1)*5+r,(d+1)*5+r+1]
            if all(clears[i] for i in cells):
                result.append({'depths':DEPTHS[d:d+2],'ranks':RANKS[r:r+2],'cells':cells})
    return {'exists':bool(result),'coordinate':result[0] if result else None,
            'qualifying_tiles':result,'predictor_selected':None}


def summarize(subject_effects,seed,confirmatory):
    # subjects x arms x proximal/terminal x raw/z. Relations have already
    # been equally averaged inside each sampled subject, preserving clusters.
    point=subject_effects.mean(axis=0)
    rng=np.random.default_rng(seed)
    indices=rng.integers(len(subject_effects),size=(10000,len(subject_effects)))
    # Multinomial weights avoid an enormous [draw,subject,arm,surface] tensor.
    weights=np.stack([(indices==i).sum(axis=1) for i in range(len(subject_effects))],axis=1)/len(subject_effects)
    draws=np.einsum('bs,samz->bamz',weights,subject_effects)
    intervals=np.quantile(draws,[0.025,0.975],axis=0)
    denominator=point[2,0]
    estimable=bool(np.all(denominator>0))
    cells=[]; sham_cells=[]
    ratio_draws=None; ratio_point=None; lower=None; terminal_lower=None
    if estimable:
        ratio_point=point[5:40,0]/denominator
        ratio_draws=draws[:,5:40,0]/draws[:,2:3,0]
        if not np.isfinite(ratio_draws).all():
            if confirmatory:
                raise ValueError('nonfinite retention bootstrap')
            # A descriptive train subset cannot veto the held-out primary gate.
            estimable=False
        if confirmatory and estimable:
            lower,_=max_t_lower(ratio_point.reshape(-1),ratio_draws.reshape(10000,-1))
            lower=lower.reshape(35,2)
            terminal_lower,_=max_t_lower(point[5:40,1].reshape(-1),draws[:,5:40,1].reshape(10000,-1))
            terminal_lower=terminal_lower.reshape(35,2)
    for cell in range(35):
        a=cell+5
        cleared=bool(confirmatory and estimable and np.all(ratio_point[cell]>=0.8) and np.all(lower[cell]>=0.5)
                     and np.all(intervals[0,2,0]>0) and np.all(terminal_lower[cell]>0))
        value={'cell':cell,'depth':DEPTHS[cell//5],'rank':RANKS[cell%5],
               'effect':point[a].tolist(),'effect_ci95':intervals[:,a].tolist(),
               'retention':ratio_point[cell].tolist() if estimable else None,
               'retention_ci95':np.quantile(ratio_draws[:,cell],[0.025,0.975],axis=0).tolist() if estimable else None,
               'retention_simultaneous_lower95':lower[cell].tolist() if confirmatory and estimable else None,
               'terminal_effect_simultaneous_lower95':terminal_lower[cell].tolist() if confirmatory and estimable else None,
               'gate_pass':cleared if confirmatory else None}
        cells.append(value)
        s=cell+40
        sham_ratio=point[s,0]/denominator if estimable else None
        sham_draws=draws[:,s,0]/draws[:,2,0] if estimable else None
        sham_estimable=estimable and np.isfinite(sham_draws).all()
        sham_cells.append({'cell':cell,'depth':DEPTHS[cell//5],'rank':RANKS[cell%5],
                           'effect':point[s].tolist(),'effect_ci95':intervals[:,s].tolist(),
                           'retention':sham_ratio.tolist() if estimable else None,
                           'retention_ci95':np.quantile(sham_draws,[0.025,0.975],axis=0).tolist() if sham_estimable else None,
                           'diagnostic_only':True,'gate_pass':None})
    return {'subjects':len(subject_effects),'effect_axis_order':['proximal','terminal'],
            'metric_axis_order':['raw','zscored'],'point':point.tolist(),
            'effect_ci95':intervals.tolist(),'exact_effect_positive':estimable,
            'cells':cells,'sham_cells':sham_cells,'diagnostic_only':not confirmatory}


def run(binding,replay_path,fit_path,capture_path,output):
    e=validate_execution(binding); lineage=validate(Path(e['protocol_path']))
    replay=checked_manifest(replay_path,'larql.gwv2.replay.v1',lineage)
    fit=checked_manifest(fit_path,'larql.gwv2.fit.v1',lineage)
    capture=checked_manifest(capture_path,'larql.gwv2.capture.v1',lineage)
    if (e['lineage']!=lineage or replay['execution_file_sha256']!=sha256(binding)
        or replay['fit_file_sha256']!=sha256(fit_path) or replay['capture_file_sha256']!=sha256(capture_path)
        or replay['parity_bit_mismatches']!=0 or replay['intervention_firings']!=666*74):
        raise ValueError('incomplete or unbound replay')
    expected=['natural','noop','exact','identity','zero']+[f'{f}-{i}' for f in ['primary','sham'] for i in range(35)]
    if replay['arms']!=expected:
        raise ValueError('replay arm order changed')
    before=array(replay_path.parent,replay,'before.f32')
    after=[array(replay_path.parent,replay,n) for n in ['proximal.f32','terminal.f32']]
    if before.shape!=(666,142) or any(a.shape!=(666,75,142) for a in after):
        raise ValueError('incomplete grid coverage')
    rows=[item['row'] for item in e['rows']]
    lookup={r['edge_id']:i for i,r in enumerate(rows)}
    groups={}
    for i,r in enumerate(rows):
        groups.setdefault((r['split'],r['subject_id'],r['semantic_edge']['relation']),[]).append(i)
    group_keys=sorted(groups)
    facts=np.array([groups[k] for k in group_keys])
    if facts.shape!=(222,3):
        raise ValueError('incomplete semantic-edge groups')
    control_map={i:lookup[next(c['paired_edge_id'] for c in r['control_ids'] if c['kind']=='same_relation_different_subject')] for i,r in enumerate(rows)}
    controls=np.array([[control_map[i] for i in group] for group in facts])
    effects=np.empty((222,75,2,2))
    for z in range(2):
        b=distributions(before,z)
        for surface in range(2):
            for arm in range(75):
                effects[:,arm,surface,z]=effect(b,distributions(after[surface][:,arm],z),facts,controls)
    result={}; clears={}
    for split in ['train','validation','test']:
        subjects=sorted({k[1] for k in group_keys if k[0]==split})
        values=np.stack([effects[[i for i,k in enumerate(group_keys) if k[0]==split and k[1]==s]].mean(axis=0) for s in subjects])
        result[split]=summarize(values,27022032,split!='train')
        result[split]['relation_effects']={r:effects[[i for i,k in enumerate(group_keys) if k[0]==split and k[2]==r]].mean(axis=0).tolist() for r in ['capital','currency','language']}
        if split!='train': clears[split]=[c['gate_pass'] for c in result[split]['cells']]
        else:
            selected=np.array([s not in ['KNA','STP'] for s in subjects])
            result['train_shuffled_only']=summarize(values[selected],27022032,False)
            result['train_shuffled_only']['filter']='sealed donor subject differs; original matched controls retained'
    clear=[a and b for a,b in zip(clears['validation'],clears['test'],strict=True)]
    declared=frontier(clear)
    predictions=array(fit_path.parent,fit,'predictions.f32')
    natural_v=array(capture_path.parent,capture,'source-depth-v.f32')[:,7]
    subject_names=[]; splits=[]
    for item in e['rows']:
        for _ in item['roles']['subject_entity']:
            subject_names.append(item['row']['subject_id']); splits.append(item['row']['split'])
    splits=np.array(splits); subject_names=np.array(subject_names)
    diagnostic={}
    for name,mask in {**{s:splits==s for s in ['train','validation','test']},'train_shuffled_only':(splits=='train')&~np.isin(subject_names,['KNA','STP']),'complete_population':np.ones(len(splits),dtype=bool)}.items():
        a=predictions[:,:,mask]; b=natural_v[mask]
        mse=np.mean((a.astype(float)-b)**2,axis=(-2,-1))
        denom=np.linalg.norm(a.astype(float),axis=-1)*np.linalg.norm(b.astype(float),axis=-1)
        if np.any(denom<=0): raise ValueError('zero reconstruction norm')
        cosine=np.mean(np.sum(a.astype(float)*b,axis=-1)/denom,axis=-1)
        diagnostic[name]={'token_rows':int(mask.sum()),'mse':mse.tolist(),'cosine':cosine.tolist(),'diagnostic_only':True}
    document=dict(schema='larql.gwv2.adjudication.v1',lineage=lineage,
        authorities={str(p):sha256(p) for p in [binding,replay_path,fit_path,capture_path]},
        arms=expected,results=result,heldout_cells_clear=clear,early_frontier=declared,
        reconstruction_diagnostics=diagnostic,
        verdict='EARLY PAYLOAD FRONTIER EXISTS' if declared['exists'] else 'NO EARLY PAYLOAD FRONTIER ON THE FROZEN GRID',
        sham='partially shuffled; 42/44 identities and 522/603 train token rows; no primary decision uses sham diagnostics',
        efficiency_claim=False,physical_accounting_complete=False,
        latency='Operational row timing only; separate exclusive bracketed measurements required.',
        analysis_code_sha256=sha256(Path(__file__)))
    seal_json(output,document,'adjudication_sha256')
    print(json.dumps({'verdict':document['verdict'],'heldout_clearing_cells':sum(clear),'frontier':declared},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['execution','replay','fit','capture','output']:p.add_argument(name,type=Path)
    a=p.parse_args();run(a.execution,a.replay,a.fit,a.capture,a.output)
