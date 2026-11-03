"""The verifier's independence from LARQL, as far as it can be checked mechanically."""

import ast
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ALLOWED = set(sys.stdlib_module_names) | {"numpy", "gguf", "refcheck", "larql_contract_verifier", "_mini"}


def python_files():
    for top in ("larql_contract_verifier", "refcheck", "tests"):
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(d, f)


def imported_roots(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


class Boundaries(unittest.TestCase):
    def test_nothing_imports_anything_but_the_standard_library_numpy_and_gguf(self):
        for path in python_files():
            extra = set(imported_roots(path)) - ALLOWED
            self.assertFalse(extra, f"{os.path.relpath(path, ROOT)} imports {sorted(extra)}")

    def test_the_package_depends_on_nothing_from_larql(self):
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
            deps = re.search(r"dependencies = \[(.*?)\]", f.read(), re.S).group(1)
        self.assertNotIn("larql", deps)
        with open(os.path.join(ROOT, "requirements.lock"), encoding="utf-8") as f:
            names = re.findall(r"^([a-z0-9_.-]+)==", f.read(), re.M)
        self.assertFalse([n for n in names if "larql" in n])

    def test_the_verifier_never_runs_a_program(self):
        for path in python_files():
            if os.path.dirname(path).endswith("tests"):
                continue
            roots = set(imported_roots(path))
            self.assertFalse(roots & {"subprocess", "ctypes"}, os.path.relpath(path, ROOT))


if __name__ == "__main__":
    unittest.main()
