#!/usr/bin/env python3
"""Fit every frozen GW-V2 primary/sham cell, with subject-excluded train diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gwv2_amend import validate, seal_json
from gwv2_io import array, checked_manifest, write_array
from gwv2_population import sha256
from gwread1_fit import fit_ridge
from gwv2_prepare import validate_execution

DEPTHS = [0,4,8,12,16,20,23]
RANKS = [8,16,32,64,128]


def factors(model, rank):
    xm, ym, u, singular, vt = model
    return tuple(np.asarray(v, dtype=np.float32) for v in (xm,ym,u[:,:rank]*singular[:rank],vt[:rank]))


def predict(x, packed):
    xm,ym,a,b = packed
    result = ((np.asarray(x,dtype=np.float32)-xm)@a)@b+ym
    if not np.isfinite(result).all():
        raise ValueError('nonfinite predictor output')
    return result


def fit_all(binding, capture_path, output):
    e = validate_execution(binding)
    lineage = validate(Path(e['protocol_path']))
    if e['lineage'] != lineage:
        raise ValueError('execution lineage mismatch')
    c = checked_manifest(capture_path,'larql.gwv2.capture.v1',lineage)
    if c['execution_file_sha256'] != sha256(binding) or c['outcomes_captured'] or c['source_and_carrier_bit_mismatches'] != 0:
        raise ValueError('capture is not the bound outcome-free capture')
    output.mkdir(parents=True, exist_ok=False)
    v = array(capture_path.parent,c,'source-depth-v.f32')
    subjects=[]; splits=[]; token_keys=[]
    for i,item in enumerate(e['rows']):
        for ordinal in range(len(item['roles']['subject_entity'])):
            subjects.append(item['row']['subject_id']); splits.append(item['row']['split']); token_keys.append((i,ordinal))
    subjects=np.array(subjects); train=np.array(splits)=='train'
    if v.shape!=(len(subjects),8,256) or train.sum()!=603:
        raise ValueError('V2 capture token geometry changed')
    lookup={key:i for i,key in enumerate(token_keys)}
    donor_doc=json.loads((Path(e['protocol_path']).parent/'gwv2-donor-map.json').read_text())
    donor=np.arange(len(subjects))
    shuffled=np.zeros(len(subjects),dtype=bool)
    for entry in donor_doc['entries']:
        target=lookup[entry['row'],entry['ordinal']]
        donor[target]=lookup[entry['donor_row'],entry['ordinal']]
        shuffled[target]=entry['shuffled']
    if shuffled.sum()!=522 or set(subjects[train & ~shuffled])!={'KNA','STP'}:
        raise ValueError('sham mapping coverage mismatch')
    output_values=np.empty((2,35,len(subjects),256),dtype=np.float32)
    descriptors=[]; model_rows=[]; loo=[]
    # Only train target rows enter these objectives. Held-out natural targets
    # are used exclusively in the later diagnostic report, after fit sealing.
    for family in range(2):
        targets=np.arange(len(subjects)) if family==0 else donor
        for d,depth in enumerate(DEPTHS):
            model=fit_ridge(v[train,d],v[targets[train],7])
            for r,rank in enumerate(RANKS):
                cell=d*5+r
                packed=factors(model,rank)
                flat=np.concatenate([x.ravel() for x in packed])
                name=f'{["primary","sham"][family]}-L{depth}-r{rank}.f32'
                desc=write_array(output/name,flat); descriptors.append(desc)
                model_rows.append(dict(family=family,cell=cell,depth=depth,rank=rank,artifact=name,
                                       sha256=desc['sha256'],cached_bytes=flat.nbytes,
                                       flops_per_token=4*256*rank+512))
                output_values[family,cell,~train]=predict(v[~train,d],packed)
            for subject in sorted(set(subjects[train])):
                held=train & (subjects==subject)
                fit_mask=train & ~held
                if family==1:
                    fit_mask &= subjects[donor]!=subject
                if (subjects[fit_mask]==subject).any() or (subjects[targets[fit_mask]]==subject).any():
                    raise ValueError('leave-subject-out target leakage')
                local=fit_ridge(v[fit_mask,d],v[targets[fit_mask],7])
                for r,rank in enumerate(RANKS):
                    packed=factors(local,rank)
                    output_values[family,d*5+r,held]=predict(v[held,d],packed)
                    digest=hashlib.sha256(b''.join(x.astype('<f4').tobytes() for x in packed)).hexdigest()
                    loo.append(dict(family=family,depth=depth,rank=rank,subject=subject,
                                    fit_rows=int(fit_mask.sum()),parameter_sha256='sha256:'+digest))
            print(f'GW-V2 fit family={family} depth={depth}; all five ranks sealed',flush=True)
    descriptors.append(write_array(output/'predictions.f32',output_values))
    result=dict(schema='larql.gwv2.fit.v1',lineage=lineage,
                execution_file_sha256=sha256(binding),capture_file_sha256=sha256(capture_path),
                models=model_rows,leave_subject_out_models=loo,artifacts=descriptors,
                train_subjects=44,train_fit_rows=603,validation_or_test_fit_rows=0,
                candidate_effects_observed=False,selection=None,
                software={'numpy':np.__version__},
                code_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('gwread1_fit.py')]})
    seal_json(output/'fit.json',result,'fit_sha256')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('execution',type=Path); p.add_argument('capture',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args(); fit_all(a.execution,a.capture,a.output)
