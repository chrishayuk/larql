#!/usr/bin/env python3
"""Bind the frozen V2 lineage, role positions, model and inherited reference bank."""
import argparse
import json
from pathlib import Path
from gwkey1_preregister import role_row
from gwv2_amend import validate, seal_json
from gwv2_population import read_jsonl, sha256, canonical_hash


def validate_execution(path):
    document = json.loads(path.read_text())
    if canonical_hash(document, 'execution_sha256') != document['execution_sha256']:
        raise ValueError('execution binding identity mismatch')
    protocol_path = Path(document['protocol_path'])
    if document['lineage'] != validate(protocol_path):
        raise ValueError('execution lineage changed')
    original_rows = read_jsonl(protocol_path.parent / 'input.jsonl')
    subject_offset = source_offset = 0
    for item, row in zip(document['rows'], original_rows, strict=True):
        if (item['row'] != row or item['roles'] != role_row(row)['roles']
            or item['subject_offset'] != subject_offset or item['source_offset'] != source_offset):
            raise ValueError('execution rows/roles differ from frozen population')
        source_offset += len(row['prompt']['token_ids'])
        subject_offset += len(item['roles']['subject_entity'])
    if document['source_rows'] != source_offset or document['subject_rows'] != subject_offset:
        raise ValueError('execution coverage mismatch')
    for name, expected in document['container_metadata'].items():
        if sha256(Path(document['container']) / name) != expected:
            raise ValueError('bound model metadata changed')
    return document


def prepare(protocol_path, container, output):
    lineage = validate(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    root = protocol_path.parent
    rows = read_jsonl(root / 'input.jsonl')
    population = json.loads((root / 'population-manifest.json').read_text())
    # This authority is transitively pinned by the unchanged READ-1 protocol.
    read_path = (root / protocol['authorities']['gwread1_protocol']['path']).resolve()
    read = json.loads(read_path.read_text())
    source_descriptor = read['authorities']['gwkey1_source_capture']
    source_path = (read_path.parent / source_descriptor['path']).resolve()
    if sha256(source_path) != source_descriptor['sha256']:
        raise ValueError('inherited reference capture identity mismatch')
    source = json.loads(source_path.read_text())
    key_path = (root / protocol['authorities']['gwkey1_preregistration']['path']).resolve()
    key = json.loads(key_path.read_text())
    roles_path = (key_path.parent / key['roles']['artifact']['path']).resolve()
    if sha256(roles_path) != key['roles']['artifact']['sha256']:
        raise ValueError('inherited role map changed')
    if output.exists():
        raise ValueError('execution binding already exists')
    subject_offset = 0
    source_offset = 0
    execution_rows = []
    for row in rows:
        roles = role_row(row)
        positions = roles['roles']['subject_entity']
        execution_rows.append(dict(row=row, roles=roles['roles'],
                                   source_offset=source_offset, subject_offset=subject_offset))
        source_offset += len(row['prompt']['token_ids'])
        subject_offset += len(positions)
    image = {name: sha256(container / name) for name in ['index.json', 'system_graph.json']}
    document = dict(schema='larql.gwv2.execution.v1', lineage=lineage,
        protocol_path=str(protocol_path.resolve()), container=str(container.resolve()),
        container_metadata=image, plan_sha256=source['authorities']['plan_sha256'],
        candidate_token_ids=population['candidate_readout']['token_ids'],
        reference_source_manifest=str(source_path), reference_source_sha256=sha256(source_path),
        reference_roles=str(roles_path), reference_roles_sha256=sha256(roles_path),
        source_rows=source_offset, subject_rows=subject_offset,
        rows=execution_rows,
        numerical_execution='canonical production CPU, effective prepared Q8 operands',
        grid_order=[dict(depth=d, rank=r) for d in [0,4,8,12,16,20,23] for r in [8,16,32,64,128]])
    return seal_json(output, document, 'execution_sha256')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == 'validate':
        print(json.dumps(validate_execution(Path(sys.argv[2]))['lineage']))
        raise SystemExit(0)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('protocol', type=Path); p.add_argument('container', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args()
    d = prepare(a.protocol, a.container, a.output)
    print(json.dumps({k:d[k] for k in ['execution_sha256','lineage','source_rows','subject_rows']}, indent=2))
