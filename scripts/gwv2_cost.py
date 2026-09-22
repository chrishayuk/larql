#!/usr/bin/env python3
"""Account for V2 constructor work from the bound executable plan, not tensor size alone."""
import argparse
import json
import math
from pathlib import Path
from gwv2_amend import seal_json
from gwv2_prepare import validate_execution
from gwv2_population import sha256


def layer_work(layer,positions):
    attention=layer['attention']; ffn=layer['ffn']
    matrices=[attention[k] for k in ['q','k','v','o']]+[ffn[k] for k in ['gate','up','down'] if k in ffn]
    matrix_flops=positions*sum(2*math.prod(m['shape']) for m in matrices)
    # All V2 prefixes are shorter than this model's sliding windows.
    pairs=positions*(positions+1)//2
    attention_flops=4*attention['num_q_heads']*attention['head_dim']*pairs
    kv_bytes=2*positions*attention['num_kv_heads']*attention['head_dim']*4
    return dict(matrix_flops=matrix_flops,attention_dot_and_value_flops=attention_flops,
                matrix_calls=positions*len(matrices),kv_state_bytes=kv_bytes)


def run(binding,image_path,fit_path,prefix_path,output):
    e=validate_execution(binding)
    image=json.loads(image_path.read_text()); fit=json.loads(fit_path.read_text())
    prefix=json.loads(prefix_path.read_text())
    if any(d['lineage']!=e['lineage'] for d in [image,fit,prefix]):
        raise ValueError('cost authority lineage mismatch')
    if any(d['bit_mismatches']!=0 or d['subject_positions_checked']!=1008 for d in prefix['depths']):
        raise ValueError('constructor has not passed complete source-V parity')
    plan=image['plan']; hidden=plan['embedding']['table']['shape'][1]; dim=256
    residency={d['depth']:d['prefix_prepared_resident_bytes'] for d in prefix['depths']}
    cells=[]
    for model in fit['models']:
        if model['family']!=0:continue
        depth,rank=model['depth'],model['rank']; records=[]
        for index,item in enumerate(e['rows']):
            token_count=len(item['row']['prompt']['token_ids'])
            positions=item['roles']['subject_entity']; n=max(positions)+1; m=len(positions)
            source=[layer_work(l,n) for l in plan['layers'][:depth]]
            baseline=[layer_work(l,token_count) for l in plan['layers'][:24]]
            projection=2*hidden*dim*m
            predictor=(4*dim*rank+2*dim)*m
            prefix_flops=sum(d['matrix_flops']+d['attention_dot_and_value_flops'] for d in source)+projection
            canonical=sum(d['matrix_flops']+d['attention_dot_and_value_flops'] for d in baseline)
            # Dynamic state means simultaneously represented logical vectors,
            # not measured DRAM traffic or allocator/process RSS.
            state_bytes=sum(d['kv_state_bytes'] for d in source)+n*hidden*4+m*dim*4
            records.append(dict(row=index,split=item['row']['split'],subject_id=item['row']['subject_id'],
                original_prompt_tokens=token_count,constructor_prefix_tokens=n,subject_tokens=m,
                canonical_layers_fully_executed=depth,
                source_layer_operations=['pre-attention normalization at subject positions','one KV-head V projection at subject positions'],
                constructor_matrix_calls=sum(d['matrix_calls'] for d in source)+m,
                predictor_matrix_calls=2*m,
                constructor_matvec_and_attention_flops=prefix_flops,predictor_flops=predictor,
                canonical_L0_L23_matvec_and_attention_flops=canonical,
                counterfactual_composition_avoided_matvec_and_attention_flops=canonical-prefix_flops-predictor,
                dynamic_prefix_state_bytes=state_bytes,
                predictor_dynamic_logical_read_write_bytes=4*(6*dim+2*rank)*m))
        cells.append(dict(cell=model['cell'],depth=depth,rank=rank,
            source_boundary=f'{depth} complete canonical layers from L0, then L{depth} subject pre-attention norm and KV-head-0 V; no later subject-dependent computation',
            shared_prefix_frontier=depth,cached_parameter_bytes=model['cached_bytes'],
            prefix_prepared_resident_bytes=residency[depth],
            actual_V2_full_prepared_resident_bytes=image['prepared_resident_bytes'],
            per_execution=records,
            latency={'predictor_only':None,'prefix_construction':None,'intervention_replay':None,'end_to_end':None},
            latency_status='awaiting exclusive bracketed measurements'))
    document=dict(schema='larql.gwv2.cost-accounting.v1',lineage=e['lineage'],cells=cells,
        authorities={str(p):sha256(p) for p in [binding,image_path,fit_path,prefix_path]},
        actual_avoided_canonical_work=0,efficiency_claim=False,
        arithmetic_scope='Matvec-equivalent multiply/add FLOPs and attention dot/weighted-value FLOPs. Norm, softmax, activation and RoPE scalar operations are not included in these explicitly bounded counters.',
        dynamic_byte_scope='Logical vector reads/writes for the factor predictor and logical prefix KV/carrier state capacity; not measured memory-bus traffic.',
        residency_scope='Prepared allocated representation, including shared embeddings. Prefix and full-image residency overlap and must not be summed as a minimal constructor residency claim.',
        shared_prefix_rule='Any successor must execute the deepest required common canonical prefix once; V2 still retains the full natural context.',
        physical_report_complete=False)
    seal_json(output,document,'cost_sha256')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['execution','image','fit','prefix','output']:p.add_argument(name,type=Path)
    a=p.parse_args();run(a.execution,a.image,a.fit,a.prefix,a.output)
