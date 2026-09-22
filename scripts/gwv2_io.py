"""Checked artifact IO shared by the GW-V2 stages."""
import json
import math
from pathlib import Path
import numpy as np
from gwv2_population import sha256, canonical_hash
from gwv2_amend import validate, seal_json


def checked_manifest(path, schema, lineage):
    document = json.loads(Path(path).read_text())
    if document['schema'] != schema or document['lineage'] != lineage:
        raise ValueError(f'inadmissible stage authority: {path}')
    if schema == 'larql.gwv2.fit.v1' and document.get('fit_sha256') != canonical_hash(document, 'fit_sha256'):
        raise ValueError(f'stage manifest seal mismatch: {path}')
    return document


def array(root, manifest, name):
    entries = [item for item in manifest['artifacts'] if item['path'] == name]
    if len(entries) != 1:
        raise ValueError(f'expected one artifact: {name}')
    entry = entries[0]
    path = Path(root) / name
    if sha256(path) != entry['sha256'] or path.stat().st_size != math.prod(entry['shape']) * 4:
        raise ValueError(f'artifact identity/size mismatch: {path}')
    result = np.fromfile(path, dtype='<f4').reshape(entry['shape'])
    if not np.isfinite(result).all():
        raise ValueError(f'nonfinite tensor: {path}')
    return result


def write_array(path, data):
    data = np.asarray(data, dtype='<f4')
    if not np.isfinite(data).all():
        raise ValueError('nonfinite output')
    with Path(path).open('xb') as handle:
        data.tofile(handle)
    return dict(path=Path(path).name, dtype='f32-le', shape=list(data.shape),
                bytes=Path(path).stat().st_size, sha256=sha256(path))
