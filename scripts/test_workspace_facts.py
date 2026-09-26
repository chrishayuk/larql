"""Manifest-boundary tests for the generated documentation inventory."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import workspace_facts as facts


class WorkspaceFactsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        root_patch = patch.object(facts, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.write('Cargo.toml', '''[workspace]
members = ["crates/consumer", "crates/leaf"]
[workspace.package]
version = "1.2.3"
[workspace.dependencies]
renamed = { package = "leaf", path = "crates/leaf", features = ["shared"] }
serde = "1"
''')
        self.write('crates/consumer/Cargo.toml', '''[package]
name = "consumer"
version.workspace = true
[dependencies]
serde.workspace = true
[dev-dependencies]
leaf = { path = "../leaf" }
[target.'cfg(unix)'.dependencies]
renamed = { workspace = true, optional = true, default-features = false, features = ["local"] }
''')
        self.write('crates/leaf/Cargo.toml', '[package]\nname = "leaf"\nversion = "2.0.0"\n')
        self.write('crates/larql-experts/Cargo.toml', '[workspace]\nmembers = ["guest"]\n')
        self.write('crates/larql-experts/guest/Cargo.toml', '[package]\nname = "guest"\nversion = "3.0.0"\n')
        for guide in ['crates/consumer/README.md', 'crates/leaf/README.md', 'crates/larql-experts/README.md']:
            self.write(guide, '**Class: CURRENT.**\n')

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def run_mode(self, mode):
        with patch('sys.argv', ['workspace_facts.py', mode]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return facts.main()

    def test_inherited_path_target_features_and_dev_edges_stay_distinct(self):
        data = facts.inventory()
        consumer = next(p for p in data['packages'] if p['name'] == 'consumer')
        self.assertEqual(consumer['version'], '1.2.3')
        deps = consumer['local_dependencies']
        self.assertEqual(len(deps), 2)
        runtime = next(d for d in deps if d['kind'] == 'dependencies')
        self.assertEqual(runtime['name'], 'leaf')
        self.assertEqual(runtime['alias'], 'renamed')
        self.assertEqual(runtime['path'], 'crates/leaf')
        self.assertEqual(runtime['features'], ['local', 'shared'])
        self.assertEqual(runtime['target'], 'cfg(unix)')
        self.assertTrue(runtime['optional'])
        self.assertFalse(runtime['default_features'])
        self.assertEqual(next(p for p in data['packages'] if p['name'] == 'guest')['workspace'],
                         'crates/larql-experts/Cargo.toml')
        self.assertNotIn('crates/larql-experts/guest', data['workspaces'][0]['members'])

    def test_check_rejects_manifest_drift_and_missing_current_readme(self):
        self.assertEqual(self.run_mode('--write'), 0)
        self.assertEqual(self.run_mode('--check'), 0)
        path = self.root / 'crates/consumer/Cargo.toml'
        path.write_text(path.read_text().replace('optional = true', 'optional = false'))
        self.assertEqual(self.run_mode('--check'), 1)
        self.assertEqual(self.run_mode('--write'), 0)
        self.assertEqual(self.run_mode('--check'), 0)
        self.write('crates/consumer/README.md', '# Undocumented status\n')
        self.assertEqual(self.run_mode('--check'), 1)

    def test_missing_member_fails_instead_of_silently_narrowing_inventory(self):
        path = self.root / 'Cargo.toml'
        path.write_text(path.read_text().replace('"crates/leaf"]', '"crates/absent"]'))
        with self.assertRaisesRegex(ValueError, 'matches no directory'):
            facts.inventory()


if __name__ == '__main__':
    unittest.main()
