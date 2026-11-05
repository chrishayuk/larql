import tempfile
from pathlib import Path
import unittest

from check_tls_dependencies import manifest_errors, system_tls_packages


class TlsDependencyTests(unittest.TestCase):
    def check(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Cargo.toml"
            path.write_text(manifest)
            return manifest_errors(path)

    def test_inheritance_and_extra_features_are_allowed(self):
        self.assertEqual(self.check('[dependencies]\nreqwest = { workspace = true, features = ["json"] }'), [])

    def test_independent_target_dev_edge_is_caught(self):
        self.assertTrue(self.check('[target.\'cfg(windows)\'.dev-dependencies]\nhttp = { package = "reqwest", version = "0.12" }'))

    def test_inherited_edge_cannot_reenable_default_tls(self):
        for setting in ('default-features = true', 'features = ["default-tls"]', 'features = ["native-tls-vendored"]'):
            self.assertTrue(self.check('[dependencies]\nreqwest = { workspace = true, ' + setting + ' }'))

    def test_graph_catches_transitive_system_tls(self):
        self.assertEqual(system_tls_packages('reqwest v0.12.28\nnative-tls v0.2.18 (*)\nopenssl-sys v0.9.114'), {"native-tls", "openssl-sys"})
        self.assertEqual(system_tls_packages('reqwest v0.12.28\nrustls v0.23.45'), set())


if __name__ == "__main__":
    unittest.main()
