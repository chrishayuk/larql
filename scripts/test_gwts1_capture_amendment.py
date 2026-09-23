import copy
import json
import os
import subprocess
import unittest

from scripts import gwts1_capture_amendment as a

# History-independent: CI checks out one commit, and a squash merge can drop
# the SHA a draft was built on. HEAD carries the frozen files; the unrelated
# commit below carries nothing and shares no history with HEAD.
AUTHORITY = a.resolve_commit("HEAD")


def unrelated_commit() -> str:
    def git(*args: str, stdin: bytes = b"") -> str:
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "gwts1-test",
            "GIT_AUTHOR_EMAIL": "gwts1-test@invalid",
            "GIT_COMMITTER_NAME": "gwts1-test",
            "GIT_COMMITTER_EMAIL": "gwts1-test@invalid",
        }
        return subprocess.run(
            ["git", "-C", str(a.REPO_ROOT), *args],
            input=stdin, capture_output=True, check=True, env=env,
        ).stdout.decode().strip()

    empty_tree = git("hash-object", "-t", "tree", "-w", "--stdin")
    return git("commit-tree", empty_tree, "-m", "gwts1 amendment test: no frozen files")


NON_AUTHORITY = unrelated_commit()


def sealed_witness() -> dict:
    return json.loads((a.REPO_ROOT / a.WITNESS_RELATIVE_PATH).read_text())


def rejected_control() -> dict:
    control = sealed_witness()
    control["gates"][a.SOURCE_ALIGNMENT_GATE]["pass"] = False
    control["overall_pass"] = False
    return control


class AmendmentBuildTests(unittest.TestCase):
    def test_amendment_on_disk_is_reproduced_byte_identically(self) -> None:
        path = a.AMENDMENT_PATH if a.AMENDMENT_PATH.exists() else a.DRAFT_PATH
        on_disk = json.loads(path.read_text())
        bound = on_disk["attr1d_authority"]["repository_commit"]
        draft = on_disk["status"].startswith("draft")
        self.assertEqual(a.build(bound, draft=draft), on_disk)

    def test_identity_is_the_canonical_hash(self) -> None:
        document = a.build(AUTHORITY, draft=True)
        self.assertEqual(
            document["amendment_sha256"],
            a.population.canonical_identity(document, "amendment_sha256"),
        )

    def test_amendment_changes_no_frozen_design_field(self) -> None:
        change = a.build(AUTHORITY, draft=True)["change"]
        flags = {key: value for key, value in change.items() if key != "kind"}
        self.assertTrue(flags)
        self.assertFalse(any(flags.values()))

    def test_l2_mass_is_diagnostic_and_never_selects(self) -> None:
        reading = a.build(AUTHORITY, draft=True)["child_contribution_reading"]
        self.assertEqual(reading["signed_contribution"], "dot(reader, child_write)")
        self.assertFalse(reading["vector_l2_mass"]["may_select_or_admit"])

    def test_amendment_does_not_authorize_assessment(self) -> None:
        authorization = a.build(AUTHORITY, draft=True)["authorization"]
        self.assertIn("not authorized", authorization["gwts1_extraction_or_assessment"])


class PreflightTests(unittest.TestCase):
    def test_draft_preflight_passes_on_the_authority_commit(self) -> None:
        checks = a.preflight(a.build(AUTHORITY, draft=True), AUTHORITY, require_main=False)
        self.assertIn("frozen population identities and counts", checks)

    def test_tampered_amendment_is_refused(self) -> None:
        document = a.build(AUTHORITY, draft=True)
        document["change"]["thresholds_changed"] = True
        with self.assertRaisesRegex(ValueError, "declared identity"):
            a.preflight(document, AUTHORITY, require_main=False)

    def test_authority_without_the_frozen_witness_is_refused(self) -> None:
        document = a.build(NON_AUTHORITY, draft=True)
        with self.assertRaisesRegex(ValueError, "absent at"):
            a.preflight(document, AUTHORITY, require_main=False)

    def test_wrong_frozen_bytes_are_refused(self) -> None:
        document = a.build(AUTHORITY, draft=True)
        document["attr1d_authority"]["frozen_files"][a.WITNESS_RELATIVE_PATH] = "sha256:" + "0" * 64
        document["amendment_sha256"] = a.population.canonical_identity(document, "amendment_sha256")
        with self.assertRaisesRegex(ValueError, "not the frozen bytes"):
            a.preflight(document, AUTHORITY, require_main=False)

    def test_capture_commit_that_does_not_descend_from_authority_is_refused(self) -> None:
        document = a.build(AUTHORITY, draft=True)
        with self.assertRaisesRegex(ValueError, "not an ancestor of the capture"):
            a.preflight(document, NON_AUTHORITY, require_main=False)


class BridgeTests(unittest.TestCase):
    def test_reproduced_witness_with_new_run_identities_is_bit_identical(self) -> None:
        candidate = sealed_witness()
        candidate["support_observation"]["execution_identity"] = "sha256:" + "1" * 64
        candidate["support_observation_id"] = "sha256:" + "2" * 64
        report = a.adjudicate_bridge(sealed_witness(), candidate, rejected_control())
        self.assertEqual(report["verdict"], "bit_identical")
        self.assertTrue(report["capture_authorized"])

    def test_one_ulp_in_one_source_contribution_is_caught(self) -> None:
        import math

        candidate = sealed_witness()
        vector = candidate["descriptive_support"]["heads"][3]["sources"][2]["contribution"]
        vector[100] = math.nextafter(vector[100], math.inf)
        report = a.adjudicate_bridge(sealed_witness(), candidate, rejected_control())
        self.assertEqual(report["verdict"], "measurements_differ")
        self.assertFalse(report["capture_authorized"])
        self.assertEqual(report["differing_field_count"], 1)
        self.assertIn("heads[3].sources[2].contribution[100]", report["differing_fields"][0]["field"])

    def test_int_float_type_change_is_a_difference(self) -> None:
        candidate = sealed_witness()
        candidate["support_coordinate"]["layer"] = 24.0
        report = a.adjudicate_bridge(sealed_witness(), candidate, rejected_control())
        self.assertEqual(report["verdict"], "measurements_differ")

    def test_moved_coordinate_is_a_difference(self) -> None:
        candidate = sealed_witness()
        candidate["support_coordinate"]["position"] = 4
        report = a.adjudicate_bridge(sealed_witness(), candidate, rejected_control())
        self.assertEqual(report["verdict"], "measurements_differ")

    def test_dropped_source_row_is_a_difference(self) -> None:
        candidate = sealed_witness()
        candidate["descriptive_support"]["heads"][0]["sources"].pop()
        report = a.adjudicate_bridge(sealed_witness(), candidate, rejected_control())
        self.assertEqual(report["verdict"], "measurements_differ")

    def test_reworded_gate_note_is_reported_not_counted(self) -> None:
        candidate = sealed_witness()
        candidate["gates"]["coverage"]["note"] = "reworded"
        report = a.adjudicate_bridge(sealed_witness(), candidate, rejected_control())
        self.assertEqual(report["verdict"], "bit_identical")
        self.assertEqual(report["prose_changed"], ["gates.coverage.note"])

    def test_control_that_passes_makes_the_bridge_blind(self) -> None:
        report = a.adjudicate_bridge(sealed_witness(), sealed_witness(), sealed_witness())
        self.assertEqual(report["verdict"], "control_not_rejected")
        self.assertFalse(report["capture_authorized"])

    def test_control_rejected_by_another_gate_does_not_count(self) -> None:
        control = sealed_witness()
        control["gates"]["head_sum_max_relative_l2"]["pass"] = False
        control["overall_pass"] = False
        report = a.adjudicate_bridge(sealed_witness(), sealed_witness(), control)
        self.assertEqual(report["verdict"], "control_not_rejected")

    def test_control_failing_source_split_plus_an_unrelated_gate_does_not_count(
        self,
    ) -> None:
        control = rejected_control()
        control["gates"]["coverage"]["pass"] = False
        report = a.adjudicate_bridge(sealed_witness(), sealed_witness(), control)
        self.assertEqual(report["verdict"], "control_not_rejected")

    def test_control_run_on_a_different_case_does_not_count(self) -> None:
        control = rejected_control()
        control["support_coordinate_id"] = "sha256:" + "3" * 64
        report = a.adjudicate_bridge(sealed_witness(), sealed_witness(), control)
        self.assertEqual(report["verdict"], "control_not_rejected")
        self.assertFalse(report["control"]["same_case"])

    def test_control_report_carries_the_complete_gate_vector(self) -> None:
        report = a.adjudicate_bridge(sealed_witness(), sealed_witness(), rejected_control())
        observed = report["control"]["observed_gate_vector"]
        self.assertEqual(set(observed), set(a.gate_vector(sealed_witness())))
        self.assertEqual([name for name, ok in observed.items() if not ok], [a.SOURCE_ALIGNMENT_GATE])

    def test_candidate_failing_a_gate_is_refused_first(self) -> None:
        candidate = copy.deepcopy(sealed_witness())
        candidate["gates"]["coverage"]["pass"] = False
        report = a.adjudicate_bridge(sealed_witness(), candidate, rejected_control())
        self.assertEqual(report["verdict"], "candidate_gates_fail")


if __name__ == "__main__":
    unittest.main()
