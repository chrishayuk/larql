#!/usr/bin/env python3
"""Generate/check manifest-derived crate inventory (Python 3.11+, no network).

Root and nested workspaces are distinct. Normal, build and dev dependencies
remain separate; optional dependencies do not become unconditional graph edges.
This records manifest declarations, not a platform/feature-resolved Cargo graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent
WORKSPACES = [Path('Cargo.toml'), Path('crates/larql-experts/Cargo.toml')]


def read(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def inventory() -> dict:
    workspaces = []
    packages = []
    for manifest in WORKSPACES:
        workspace = read(ROOT / manifest)['workspace']
        directory = (ROOT / manifest).parent
        members_set = set()
        for member in workspace['members']:
            matches = list(directory.glob(member))
            if not matches:
                raise ValueError(f'{manifest}: member {member!r} matches no directory')
            members_set.update(matches)
        excluded = {p for pattern in workspace.get('exclude', []) for p in directory.glob(pattern)}
        members = sorted(members_set - excluded)
        if not members:
            raise ValueError(f'{manifest}: no members')
        workspaces.append({'manifest': str(manifest), 'members': [str(p.relative_to(ROOT)) for p in members],
                           'default_members': workspace.get('default-members', workspace['members'])})
        for member in members:
            doc = read(member / 'Cargo.toml')
            package = doc['package']
            def field(name: str):
                value = package.get(name)
                return workspace['package'][name] if isinstance(value, dict) and value.get('workspace') else value
            deps = []
            tables = [(None, doc)] + sorted(doc.get('target', {}).items())
            for target, table in tables:
                for kind in ('dependencies', 'build-dependencies', 'dev-dependencies'):
                    for name, raw in sorted(table.get(kind, {}).items()):
                        if not isinstance(raw, dict):
                            continue
                        dep = dict(raw)
                        if dep.pop('workspace', False):
                            declaration = workspace['dependencies'][name]
                            inherited = {'version': declaration} if isinstance(declaration, str) else dict(declaration)
                            # Cargo inherits features additively. Resolve inherited paths
                            # from the workspace, not from the consuming member.
                            base = directory
                            features = inherited.pop('features', []) + dep.pop('features', [])
                            inherited.update(dep)
                            dep = inherited
                            dep['features'] = sorted(set(features))
                        else:
                            base = member
                        if 'path' not in dep:
                            continue
                        dep_path = (base / dep['path']).resolve()
                        dep_name = read(dep_path / 'Cargo.toml')['package']['name']
                        deps.append({'name': dep_name, 'alias': name, 'kind': kind,
                                     'path': str(dep_path.relative_to(ROOT)), 'target': target,
                                     'optional': dep.get('optional', False),
                                     'default_features': dep.get('default-features', True),
                                     'features': dep.get('features', [])})
            packages.append({'name': package['name'], 'path': str(member.relative_to(ROOT)),
                             'workspace': str(manifest), 'version': field('version'),
                             'description': field('description'), 'features': doc.get('features', {}),
                             'local_dependencies': deps,
                             'examples': doc.get('example', [])})
    return {'schema': 1, 'scope': 'manifest-declarations', 'workspaces': workspaces,
            'packages': sorted(packages, key=lambda p: p['name'])}


def markdown(data: dict) -> str:
    out = ['# Workspace crate facts', '', '**Class: CURRENT — generated.**', '',
           'Run `python3 scripts/workspace_facts.py --write`; CI runs `--check`.',
           'These are manifest declarations, not a resolved platform/feature build graph.',
           '[JSON](workspace-facts.json) retains normal, build and dev dependencies,',
           'target conditions, default-feature choices and explicit example targets.', '',
           '[Architecture guide](../architecture-stack.md) explains responsibility and capability.', '']
    for workspace in data['workspaces']:
        manifest = workspace['manifest']
        out += [f'## `{manifest}`', '', '| Package | Version | Normal local dependencies | Default features |', '|---|---|---|---|']
        for package in data['packages']:
            if package['workspace'] != manifest:
                continue
            deps = []
            for dep in package['local_dependencies']:
                if dep['kind'] != 'dependencies':
                    continue
                label = f"`{dep['name']}`"
                if dep['optional']:
                    label += ' (optional)'
                if dep['target']:
                    label += f" ({dep['target']})"
                deps.append(label)
            out.append(f"| [{package['name']}](../../{package['path']}/Cargo.toml) | `{package['version']}` | {', '.join(deps) or 'None'} | {', '.join('`'+f+'`' for f in package['features'].get('default', [])) or 'None'} |")
        out.append('')
    out += ['The experts workspace is separate: root workspace test/build/coverage sweeps do',
            'not include it. A package appearing here does not claim a published release or',
            'a supported VINDEX3 operator. See each crate README and its tests.', '']
    return '\n'.join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = inventory()
    outputs = {'workspace-facts.json': json.dumps(data, indent=2) + '\n',
               'workspace-facts.md': markdown(data)}
    directory = ROOT / 'docs/generated'
    if args.write:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (directory / name).write_text(content)
        return 0
    guides = [ROOT / member / 'README.md' for member in data['workspaces'][0]['members']]
    guides += [(ROOT / workspace['manifest']).parent / 'README.md' for workspace in data['workspaces'][1:]]
    missing = [str(path.relative_to(ROOT)) for path in guides
               if not path.exists() or '**Class: CURRENT.' not in path.read_text()]
    if missing:
        print('Missing CURRENT crate guides: ' + ', '.join(missing), file=sys.stderr)
        return 1
    stale = [name for name, content in outputs.items()
             if not (directory / name).exists() or (directory / name).read_text() != content]
    if stale:
        print('Stale workspace facts: ' + ', '.join(stale) + '; run python3 scripts/workspace_facts.py --write', file=sys.stderr)
        return 1
    print(f"Workspace facts match {len(data['packages'])} package manifests in {len(data['workspaces'])} workspaces.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
