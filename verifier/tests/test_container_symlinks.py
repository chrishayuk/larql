"""How a container directory is read when it holds symlinks (`weight_patches_v4.resolve_container`), as the ledger,
pack and residency contracts read it.

The rule: a symlink is followed only to a regular file whose real path is inside the container directory or, for a
Hugging Face snapshot (`<cache>/models--<org>--<name>/snapshots/<revision>`, beside a real `blobs/`), inside that
repository's `blobs/`. Every place the chain passes through must be inside too. A link that escapes, dangles, loops or
ends at a directory or another non-regular file is refused, with a reason that says which.

Every symlink here is made at test time in a temporary directory: none is committed.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from larql_contract_verifier import cli  # noqa: E402
from larql_contract_verifier.contracts import q8_0_pack as P  # noqa: E402
from larql_contract_verifier.contracts import weight_patches_v4 as W  # noqa: E402

BASE = os.path.join(ROOT, "conformance", "structured_patch", "base")
COMPILED = os.path.join(ROOT, "conformance", "weight_patches_v4", "low_rank_k2", "compiled")
REPO = "models--larql--fixture"
REVISION = "91c35bfa9cd2c6d1dcad65b46f60a58eabb9cc1f"
SAT, CNE = "satisfied", "could_not_evaluate"


def hf_snapshot(source: str, cache: str, repo: str = REPO, revision: str = REVISION) -> str:
    """`source` laid out as the Hugging Face cache lays out a download: every file of `<repo>/snapshots/<revision>/`
    a relative symlink to `../../blobs/<sha256>` (one more `../` per directory level). Returns the snapshot."""
    blobs = os.path.join(cache, repo, "blobs")
    snapshot = os.path.join(cache, repo, "snapshots", revision)
    os.makedirs(blobs, exist_ok=True)
    for dirpath, _, filenames in os.walk(source):
        rel_dir = os.path.relpath(dirpath, source)
        os.makedirs(os.path.join(snapshot, rel_dir), exist_ok=True)
        for name in filenames:
            blob = W.sha256_file(os.path.join(dirpath, name))
            shutil.copyfile(os.path.join(dirpath, name), os.path.join(blobs, blob))
            link = os.path.normpath(os.path.join(snapshot, rel_dir, name))
            os.symlink(os.path.relpath(os.path.join(blobs, blob), os.path.dirname(link)), link)
    return snapshot


def count_files(root: str) -> int:
    return sum(len(f) for _, _, f in os.walk(root))


def without_resolution(out: dict) -> dict:
    """An answer with the parts that name where its inputs are removed: the paths given, the measurements and fields
    that say where the files were read, and the code digest."""
    out = dict(out)
    for side in ("base", "container"):
        if side in out:
            out[side] = {k: v for k, v in out[side].items() if k not in ("path", "resolution")}
    out["measurements"] = [m for m in out["measurements"] if m["name"] not in W.FILES_READ.values()]
    return out


class Fixture(unittest.TestCase):
    def setUp(self):
        self.work = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.work)

    def tree(self, name: str = "c") -> str:
        """A small ordinary container directory: two regular files, one in a subdirectory."""
        root = os.path.join(self.work, name)
        os.makedirs(os.path.join(root, "segments"))
        for rel, data in (("index.json", b"{}"), ("segments/a.bin", b"abc")):
            with open(os.path.join(root, rel), "wb") as f:
                f.write(data)
        return root

    def refused(self, root: str, *needles: str) -> str:
        with self.assertRaises(W.CannotEvaluate) as cm:
            W.file_digests(root)
        reason = str(cm.exception)
        for needle in needles:
            self.assertIn(needle, reason)
        return reason


class TheRule(Fixture):
    """`resolve_container` and `file_digests` on small trees, one kind of link each."""

    def test_regular_files_are_read_where_they_are(self):
        root = self.tree()
        files = W.file_digests(root)
        self.assertEqual(sorted(files), ["index.json", "segments/a.bin"])
        self.assertEqual(files["segments/a.bin"], (3, W.sha256_file(os.path.join(root, "segments", "a.bin"))))
        self.assertEqual(files.resolution.layout, W.LAYOUT_DIRECTORY)
        self.assertEqual(files.resolution.boundaries, (root,))
        self.assertEqual(files.resolution.followed, {})
        self.assertEqual(files.resolution.describe(len(files)),
                         f"0 of 2 files read through a symlink; a link is followed only into {root} (directory layout)")

    def test_a_link_inside_an_ordinary_container_is_followed_and_hashed_as_its_target(self):
        # Allowed: the boundary of an ordinary container is the container directory itself. The link's own path is
        # what the digest lists; the bytes are its target's, read at the real path that was checked.
        root = self.tree()
        os.symlink("a.bin", os.path.join(root, "segments", "b.bin"))
        files = W.file_digests(root)
        self.assertEqual(files["segments/b.bin"], files["segments/a.bin"])
        self.assertEqual(files.resolution.followed, {"segments/b.bin": os.path.join(root, "segments", "a.bin")})
        self.assertEqual(files.resolution.detail(),
                         {"layout": W.LAYOUT_DIRECTORY, "boundaries": [root], "symlinks_followed": 1})

    def test_a_link_out_of_the_container_is_refused(self):
        root = self.tree()
        outside = os.path.join(self.work, "outside.bin")
        with open(outside, "wb") as f:
            f.write(b"abc")
        os.symlink(outside, os.path.join(root, "segments", "b.bin"))
        reason = self.refused(root)
        self.assertEqual(reason, f"{os.path.join(root, 'segments', 'b.bin')} is a symlink that escapes its boundary: "
                                 f"it resolves to {outside}, outside {root} (directory layout)")

    def test_a_relative_link_that_climbs_out_is_refused(self):
        root = self.tree()
        with open(os.path.join(self.work, "outside.bin"), "wb") as f:
            f.write(b"abc")
        os.symlink("../../outside.bin", os.path.join(root, "segments", "b.bin"))
        self.refused(root, "escapes its boundary", os.path.join(self.work, "outside.bin"))

    def test_a_chain_that_passes_outside_and_comes_back_is_refused(self):
        # It ends inside, but its target is whatever the outside link says at the time: every hop must be inside.
        root = self.tree()
        hop = os.path.join(self.work, "hop.bin")
        os.symlink(os.path.join(root, "segments", "a.bin"), hop)
        os.symlink(hop, os.path.join(root, "segments", "b.bin"))
        self.refused(root, "escapes its boundary", f"resolves to {hop}, outside")

    def test_a_dangling_link_is_refused(self):
        root = self.tree()
        link = os.path.join(root, "segments", "b.bin")
        os.symlink("missing.bin", link)
        self.assertEqual(self.refused(root), f"{link} is a dangling symlink: its target 'missing.bin' does not exist")

    def test_a_link_through_a_file_as_if_it_were_a_directory_is_dangling(self):
        root = self.tree()
        os.symlink("a.bin/x", os.path.join(root, "segments", "b.bin"))
        self.refused(root, "is a dangling symlink")

    def test_a_link_to_a_directory_is_refused(self):
        root = self.tree()
        link = os.path.join(root, "linked")
        os.symlink("segments", link)
        self.assertEqual(self.refused(root), f"{link} is a symlink to a directory ({os.path.join(root, 'segments')}); "
                                             "this contract reads regular files only")

    def test_a_link_to_a_directory_out_of_the_container_is_refused_as_an_escape(self):
        root = self.tree()
        os.symlink(self.work, os.path.join(root, "linked"))
        self.refused(root, "escapes its boundary")

    def test_a_link_to_a_fifo_is_refused(self):
        root = self.tree()
        os.mkfifo(os.path.join(root, "segments", "fifo"))
        os.symlink("fifo", os.path.join(root, "segments", "b.bin"))
        with self.assertRaises(W.CannotEvaluate) as cm:
            W.resolve_container(root)
        self.assertIn("b.bin is a symlink to a file that is not a regular file", str(cm.exception))

    def test_a_fifo_itself_is_refused_and_never_opened(self):
        root = self.tree()
        fifo = os.path.join(root, "segments", "fifo")
        os.mkfifo(fifo)
        self.assertEqual(self.refused(root), f"{fifo} is not a regular file; this contract reads regular files only")

    def test_a_link_loop_is_refused(self):
        root = self.tree()
        a, b = os.path.join(root, "segments", "x.bin"), os.path.join(root, "segments", "y.bin")
        os.symlink("y.bin", a)
        os.symlink("x.bin", b)
        self.assertEqual(self.refused(root), f"{a} is a symlink loop: it never resolves to a file")

    def test_a_link_to_itself_is_a_loop(self):
        root = self.tree()
        os.symlink("self.bin", os.path.join(root, "segments", "self.bin"))
        self.refused(root, "is a symlink loop")

    def test_a_chain_longer_than_the_limit_is_refused_as_a_loop(self):
        root = self.tree()
        os.symlink("a.bin", os.path.join(root, "segments", "b.bin"))
        os.symlink("b.bin", os.path.join(root, "segments", "c.bin"))
        with mock.patch.object(W, "MAX_LINK_HOPS", 1):
            self.refused(root, "c.bin is a symlink loop")

    def test_an_unreadable_link_is_an_os_error_not_a_verdict(self):
        root = self.tree()
        os.symlink("a.bin", os.path.join(root, "segments", "b.bin"))
        with mock.patch.object(W.os, "stat", side_effect=PermissionError(13, "denied")):
            with self.assertRaises(PermissionError):
                W.resolve_container(root)


class HuggingFaceLayout(Fixture):
    """The layout `larql pull` leaves: each snapshot file a link into the repository's `blobs/`."""

    def test_a_snapshot_is_read_through_its_blobs(self):
        snapshot = hf_snapshot(self.tree("src"), os.path.join(self.work, "hub"))
        files = W.file_digests(snapshot)
        self.assertEqual(files, W.file_digests(os.path.join(self.work, "src")))
        blobs = os.path.join(self.work, "hub", REPO, "blobs")
        self.assertEqual(files.resolution.layout, W.LAYOUT_HF_SNAPSHOT)
        self.assertEqual(files.resolution.boundaries, (snapshot, blobs))
        self.assertEqual(len(files.resolution.followed), 2)
        self.assertTrue(all(p.startswith(blobs + os.sep) for p in files.resolution.followed.values()))

    def test_a_link_into_another_snapshot_or_refs_is_refused(self):
        # The boundary is the repository's blobs/, not the whole repository.
        hub = os.path.join(self.work, "hub")
        snapshot = hf_snapshot(self.tree("src"), hub)
        other = hf_snapshot(self.tree("src2"), hub, revision="0" * 40)
        os.symlink(os.path.join(other, "index.json"), os.path.join(snapshot, "other.json"))
        self.refused(snapshot, "escapes its boundary", "(hf_snapshot layout)")
        os.remove(os.path.join(snapshot, "other.json"))
        os.makedirs(os.path.join(hub, REPO, "refs"))
        with open(os.path.join(hub, REPO, "refs", "main"), "w") as f:
            f.write(REVISION)
        os.symlink("../../refs/main", os.path.join(snapshot, "ref"))
        self.refused(snapshot, "escapes its boundary")

    def test_a_snapshot_link_out_of_the_cache_is_refused(self):
        snapshot = hf_snapshot(self.tree("src"), os.path.join(self.work, "hub"))
        target = os.path.join(self.work, "src", "index.json")
        os.remove(os.path.join(snapshot, "index.json"))
        os.symlink(target, os.path.join(snapshot, "index.json"))
        reason = self.refused(snapshot, "escapes its boundary", target)
        self.assertIn(os.path.join(self.work, "hub", REPO, "blobs"), reason)

    def test_dangling_blob_and_blob_directory_are_refused(self):
        hub = os.path.join(self.work, "hub")
        snapshot = hf_snapshot(self.tree("src"), hub)
        blobs = os.path.join(hub, REPO, "blobs")
        os.symlink("../../blobs/" + "f" * 64, os.path.join(snapshot, "gone.json"))
        self.refused(snapshot, "is a dangling symlink")
        os.remove(os.path.join(snapshot, "gone.json"))
        os.makedirs(os.path.join(blobs, "dir"))
        os.symlink("../../blobs/dir", os.path.join(snapshot, "dir.json"))
        self.refused(snapshot, "is a symlink to a directory")

    def assertNotSnapshot(self, snapshot: str) -> None:
        """Not taken for a snapshot: its links into the neighbouring blobs/ leave the container, and are refused."""
        self.assertIsNone(W.hf_snapshot_blobs(snapshot))
        self.refused(snapshot, "escapes its boundary", "(directory layout)")

    def test_a_directory_merely_named_snapshots_is_not_a_snapshot(self):
        # The structure decides, not the name: `<repo>/snapshots/<revision>` with no real blobs/ beside snapshots/.
        hub = os.path.join(self.work, "hub")
        snapshot = hf_snapshot(self.tree("src"), hub)
        blobs = os.path.join(hub, REPO, "blobs")
        os.rename(blobs, os.path.join(hub, REPO, "stuff"))
        for link in ("index.json", os.path.join("segments", "a.bin")):
            target = os.readlink(os.path.join(snapshot, link)).replace("/blobs/", "/stuff/")
            os.remove(os.path.join(snapshot, link))
            os.symlink(target, os.path.join(snapshot, link))
        self.assertNotSnapshot(snapshot)
        with open(blobs, "w") as f:  # a blobs that is a file is no more one
            f.write("")
        self.assertNotSnapshot(snapshot)

    def test_blobs_that_is_a_symlink_does_not_make_a_snapshot(self):
        hub = os.path.join(self.work, "hub")
        snapshot = hf_snapshot(self.tree("src"), hub)
        blobs = os.path.join(hub, REPO, "blobs")
        shutil.move(blobs, os.path.join(self.work, "real-blobs"))
        os.symlink(os.path.join(self.work, "real-blobs"), blobs)
        self.assertNotSnapshot(snapshot)

    def test_a_repository_not_named_as_a_model_repository_is_not_a_snapshot(self):
        snapshot = hf_snapshot(self.tree("src"), os.path.join(self.work, "hub"), repo="larql--fixture")
        self.assertNotSnapshot(snapshot)

    def test_a_snapshot_one_level_too_deep_is_not_a_snapshot(self):
        hub = os.path.join(self.work, "hub")
        hf_snapshot(self.tree("src"), hub)
        deeper = os.path.join(hub, REPO, "snapshots", REVISION, "segments")
        self.assertIsNone(W.hf_snapshot_blobs(deeper))
        self.refused(deeper, "escapes its boundary")

    def test_a_snapshot_reached_through_a_symlinked_path_is_still_one(self):
        hub = os.path.join(self.work, "hub")
        snapshot = hf_snapshot(self.tree("src"), hub)
        alias = os.path.join(self.work, "alias")
        os.symlink(snapshot, alias)
        files = W.file_digests(alias)
        self.assertEqual(files.resolution.layout, W.LAYOUT_HF_SNAPSHOT)
        self.assertEqual(files, W.file_digests(snapshot))


class LedgerOnASnapshot(Fixture):
    """`larql-verifier ledger` with the base, the compiled container, or both in the Hugging Face layout."""

    def test_a_snapshot_base_verifies_exactly_as_a_regular_copy_does(self):
        copy = os.path.join(self.work, "copy")
        shutil.copytree(BASE, copy)
        snapshot = hf_snapshot(BASE, os.path.join(self.work, "hub"))
        regular = cli.evaluate_ledger(W.CONTRACT, copy, COMPILED)
        linked = cli.evaluate_ledger(W.CONTRACT, snapshot, COMPILED)
        self.assertEqual(regular["status"], SAT, regular["reasons"])
        self.assertEqual(linked["status"], SAT, linked["reasons"])
        self.assertEqual(without_resolution(linked), without_resolution(regular))
        self.assertEqual(linked["base"]["sha256"], regular["base"]["sha256"])
        n = count_files(BASE)
        self.assertEqual(linked["base"]["resolution"],
                         {"layout": W.LAYOUT_HF_SNAPSHOT, "symlinks_followed": n,
                          "boundaries": [snapshot, os.path.join(self.work, "hub", REPO, "blobs")]})
        self.assertEqual(regular["base"]["resolution"],
                         {"layout": W.LAYOUT_DIRECTORY, "symlinks_followed": 0, "boundaries": [copy]})
        measured = {m["name"]: m["value"] for m in linked["measurements"]}
        self.assertTrue(measured[W.FILES_READ["base"]].startswith(f"{n} of {n} files read through a symlink"))
        self.assertTrue(measured[W.FILES_READ["container"]].startswith("0 of "))

    def test_both_containers_as_snapshots(self):
        hub = os.path.join(self.work, "hub")
        base = hf_snapshot(BASE, hub)
        compiled = hf_snapshot(COMPILED, hub, repo="models--larql--compiled")
        out = cli.evaluate_ledger(W.CONTRACT, base, compiled)
        self.assertEqual(out["status"], SAT, out["reasons"])
        self.assertEqual(without_resolution(out), without_resolution(cli.evaluate_ledger(W.CONTRACT, BASE, COMPILED)))
        self.assertEqual(out["container"]["resolution"]["symlinks_followed"], count_files(COMPILED))

    def test_a_refused_link_is_could_not_evaluate_with_its_reason(self):
        for kind in ("escape", "dangling", "directory", "loop"):
            with self.subTest(link=kind):
                snapshot = hf_snapshot(BASE, os.path.join(self.work, f"hub-{kind}"))
                link = os.path.join(snapshot, "extra")
                if kind == "escape":
                    os.symlink(os.path.join(BASE, "index.json"), link)
                    expected = "is a symlink that escapes its boundary"
                elif kind == "dangling":
                    os.symlink("../../blobs/missing", link)
                    expected = "is a dangling symlink"
                elif kind == "directory":
                    os.symlink("segments", link)
                    expected = "is a symlink to a directory"
                else:
                    os.symlink("extra", link)
                    expected = "is a symlink loop"
                out = cli.evaluate_ledger(W.CONTRACT, snapshot, COMPILED)
                self.assertEqual(out["status"], CNE)
                self.assertEqual(out["verdicts"][W.NUMERICAL]["status"], CNE)
                self.assertEqual(out["verdicts"][W.LEDGER]["status"], CNE)
                self.assertTrue(any(f"{link} {expected}" in r for r in out["reasons"]), out["reasons"])
                self.assertNotIn("resolution", out["base"])


class PackOnASnapshot(Fixture):
    def test_a_snapshot_is_judged_as_its_regular_copy_and_says_where_it_was_read(self):
        snapshot = hf_snapshot(BASE, os.path.join(self.work, "hub"))
        linked, regular = cli.evaluate_pack(P.CONTRACT, snapshot), cli.evaluate_pack(P.CONTRACT, BASE)
        self.assertEqual(linked["status"], SAT, linked["reasons"])
        self.assertEqual(linked["container"]["sha256"], regular["container"]["sha256"])
        self.assertEqual(linked["container"]["resolution"]["symlinks_followed"], count_files(BASE))
        self.assertEqual(without_resolution(linked), without_resolution(regular))

    def test_a_link_inside_an_ordinary_container_is_followed(self):
        root = os.path.join(self.work, "c")
        shutil.copytree(BASE, root)
        os.symlink("index.json", os.path.join(root, "link.json"))
        out = cli.evaluate_pack(P.CONTRACT, root)
        self.assertEqual(out["status"], SAT, out["reasons"])
        self.assertEqual(out["container"]["resolution"]["symlinks_followed"], 1)


if __name__ == "__main__":
    unittest.main()
