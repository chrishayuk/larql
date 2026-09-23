#!/usr/bin/env python3
"""Seal/verify GW-V2-AMEND-1 and its outcome-independent token donor map."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from gwkey1_preregister import role_row
from gwv2_population import canonical_bytes, canonical_hash, read_jsonl, sha256
from gwv2_preflight import subject_lengths
from gwv2_preregister import validate as original_validate

AMENDMENT_IDENTITY = 'sha256:bb5ebb48e52768e50ff5a26954bf8873489c1ab01d0fec1272e2f52caa3b2d5b'


def seal_json(path, document, field):
    document[field] = canonical_hash(document, field)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return document


def donor_map(rows):
    lengths = subject_lengths(rows)['train']
    strata = defaultdict(list)
    relations = ['capital', 'currency', 'language']
    for relation in relations:
        for subject, count in lengths.items():
            strata[relation, count].append(subject)
    mapping = {}
    for (relation, count), subjects in sorted(strata.items()):
        ordered = sorted(subjects, key=lambda s: (
            hashlib.sha256(canonical_bytes([27022031, relation, count, s])).hexdigest(), s))
        for index, subject in enumerate(ordered):
            mapping[relation, subject] = ordered[(index + 1) % len(ordered)]
    lookup = {(r['subject_id'], r['semantic_edge']['relation'],
               r['semantic_edge']['prompt_semantic_family']): (i, r)
              for i, r in enumerate(rows)}
    result = []
    for index, row in enumerate(rows):
        if row['split'] != 'train':
            continue
        subject = row['subject_id']
        relation = row['semantic_edge']['relation']
        family = row['semantic_edge']['prompt_semantic_family']
        donor = mapping[relation, subject]
        donor_index, donor_row = lookup[donor, relation, family]
        positions = role_row(row)['roles']['subject_entity']
        donor_positions = role_row(donor_row)['roles']['subject_entity']
        for ordinal, (position, donor_position) in enumerate(zip(positions, donor_positions, strict=True)):
            result.append(dict(row=index, edge_id=row['edge_id'], subject_id=subject,
                               relation=relation, prompt_family=family, ordinal=ordinal,
                               position=position, donor_row=donor_index,
                               donor_edge_id=donor_row['edge_id'], donor_subject_id=donor,
                               donor_position=donor_position, shuffled=subject != donor))
    return result


def amendment_body(protocol_path, donor_path):
    original = original_validate(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    population_path = protocol_path.parent / protocol['authorities']['population']['path']
    return {
        'schema': 'larql.gwv2.amendment.v1', 'name': 'GW-V2-AMEND-1',
        'status': 'frozen_before_capture_fit_replay', 'date': '2026-09-21',
        'scope': 'Only resolve fit-sham token correspondence discovered during outcome-free execution preflight on 21 September 2026, and freeze its diagnostic reporting.',
        'original_protocol': {'path': protocol_path.name, 'identity': original['protocol_sha256'], 'sha256': sha256(protocol_path)},
        'population': {'path': population_path.name, 'identity': original['population_sha256'], 'sha256': sha256(population_path)},
        'donor_map': {'path': donor_path.name, 'sha256': sha256(donor_path),
                      'identity': json.loads(donor_path.read_text())['donor_map_sha256']},
        'permutation': {'seed': 27022031, 'strata': ['relation', 'subject_token_count'],
            'order': 'SHA256 of UTF-8 compact JSON [27022031,relation,token_count,subject_id], ensure_ascii=false, separators comma/colon; tie by subject_id',
            'donor': 'cyclic successor in sorted order',
            'fixed': ['prompt family', 'token ordinal'],
            'reuse': 'identical mapping across prompt families, depths and ranks',
            'singletons': ['KNA', 'STP'], 'singleton_treatment': 'unchanged, explicitly unshuffled',
            'shuffled_subjects': 42, 'total_train_subjects': 44,
            'shuffled_fit_rows': 522, 'total_fit_rows': 603,
            'description': 'partially shuffled sham; not complete subject destruction'},
        'reporting': {
            'complete_population': 'All 666 registered executions; train, validation and test reported separately; all 35 sham cells.',
            'shuffled_rows_only': 'Train recipient rows whose sealed donor subject differs; 42 countries, 378 executions, 522 token positions. Report all 35 sham cells separately.',
            'heldout_membership': 'No held-out row has a training donor assignment. Do not invent a shuffled-only held-out subset.',
            'train_prediction': 'Leave recipient subject out. For sham fits also omit fit rows whose sealed donor is that subject; never rewire the sealed donor map.',
            'targets': 'Causal effects and reconstruction diagnostics use the natural recipient reference, not shuffled donor correctness.',
            'matched_controls': 'Retain the original bound control for each included recipient; never rematch after filtering.',
            'diagnostic_only': True, 'threshold': None, 'frontier_role': None, 'selection_role': None},
        'unchanged': ['original protocol bytes', 'population', 'hypothesis', 'primary 35-cell grid',
                      'predictor family', 'primary analysis', 'thresholds', 'frontier criterion'],
        'seal_order': ['donor map', 'amendment', 'capture', 'fit all primary and sham cells', 'replay', 'adjudication'],
        'pre_seal_v2_model_prompts_executed': 0,
        'pre_seal_v2_activations_examined': False,
        'pre_seal_v2_fits_or_replays': False,
    }


def freeze(protocol_path):
    original_validate(protocol_path)
    root = protocol_path.parent
    for name in ('gwv2-donor-map.json', 'gwv2-amend-1.json'):
        if (root / name).exists():
            raise ValueError('refusing to overwrite an existing amendment seal')
    rows = read_jsonl(root / 'input.jsonl')
    entries = donor_map(rows)
    if len(entries) != 603 or sum(e['shuffled'] for e in entries) != 522:
        raise ValueError('donor coverage differs from authorized amendment')
    donor_path = root / 'gwv2-donor-map.json'
    seal_json(donor_path, {'schema': 'larql.gwv2.donor-map.v1', 'seed': 27022031,
                          'input_rows_sha256': sha256(root / 'input.jsonl'), 'entries': entries}, 'donor_map_sha256')
    amendment = seal_json(root / 'gwv2-amend-1.json', amendment_body(protocol_path, donor_path), 'amendment_sha256')
    return {'amendment_sha256': amendment['amendment_sha256'], 'donor_map_sha256': amendment['donor_map']['identity']}


def validate(protocol_path):
    original = original_validate(protocol_path)
    root = protocol_path.parent
    path = root / 'gwv2-amend-1.json'
    amendment = json.loads(path.read_text())
    identity = canonical_hash(amendment, 'amendment_sha256')
    if AMENDMENT_IDENTITY is None or identity != AMENDMENT_IDENTITY or amendment['amendment_sha256'] != identity:
        raise ValueError('amendment identity is not the pinned pre-capture seal')
    donor_path = root / 'gwv2-donor-map.json'
    expected = amendment_body(protocol_path, donor_path)
    if {k:v for k,v in amendment.items() if k != 'amendment_sha256'} != expected:
        raise ValueError('amendment authority or reporting rule changed')
    donor = json.loads(donor_path.read_text())
    if canonical_hash(donor, 'donor_map_sha256') != donor['donor_map_sha256']:
        raise ValueError('donor map identity mismatch')
    if donor['entries'] != donor_map(read_jsonl(root / 'input.jsonl')):
        raise ValueError('donor correspondence differs from sealed deterministic mapping')
    return {'protocol_sha256': original['protocol_sha256'],
            'population_sha256': original['population_sha256'],
            'amendment_sha256': identity, 'donor_map_sha256': donor['donor_map_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'validate'])
    parser.add_argument('protocol', type=Path)
    args = parser.parse_args()
    print(json.dumps((freeze if args.command == 'freeze' else validate)(args.protocol), indent=2))
