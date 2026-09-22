import json
import unittest
from pathlib import Path

from scripts import gwts1_population_manifest as m


class Gwts1PopulationManifestTests(unittest.TestCase):
    def test_sealed_manifest_and_contract_are_reproduced_byte_identically(self) -> None:
        manifest = m.build_manifest()
        contract = m.build_contract(manifest)
        sealed_manifest = json.loads((m.GW_TS_1_DIR / "population-manifest.json").read_text())
        sealed_contract = json.loads((m.GW_TS_1_DIR / "population-contract.json").read_text())
        self.assertEqual(manifest, sealed_manifest)
        self.assertEqual(contract, sealed_contract)

    def test_discovery_and_held_out_partition_every_eligible_identity_once(self) -> None:
        manifest = m.build_manifest()
        rows = manifest["assignment"]["rows"]
        keys = [(row["relation"], row["subject"], row["target"]) for row in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            len(rows),
            manifest["assignment"]["discovery_count"]
            + manifest["assignment"]["held_out_count"],
        )
        self.assertEqual(
            manifest["assignment"]["discovery_count"],
            sum(1 for row in rows if row["assignment"] == "discovery"),
        )
        self.assertEqual(
            manifest["assignment"]["held_out_count"],
            sum(1 for row in rows if row["assignment"] == "held_out"),
        )

    def test_apportionment_matches_the_target_discovery_fraction_within_one_per_stratum(
        self,
    ) -> None:
        manifest = m.build_manifest()
        rows = manifest["assignment"]["rows"]
        by_stratum: dict[str, list[dict]] = {}
        for row in rows:
            by_stratum.setdefault(row["stratum"], []).append(row)
        for stratum, members in by_stratum.items():
            discovery = sum(1 for row in members if row["assignment"] == "discovery")
            exact = len(members) * m.DISCOVERY_FRACTION
            self.assertLessEqual(
                abs(discovery - exact),
                1.0,
                f"{stratum}: discovery {discovery} strays >1 from exact {exact}",
            )

    def test_within_stratum_assignment_follows_ascending_canonical_hash(self) -> None:
        manifest = m.build_manifest()
        rows = manifest["assignment"]["rows"]
        by_stratum: dict[str, list[dict]] = {}
        for row in rows:
            by_stratum.setdefault(row["stratum"], []).append(row)
        for members in by_stratum.values():
            ordered = sorted(members, key=lambda row: row["stratum_rank"])
            hashes = [row["canonical_hash_order/v1"] for row in ordered]
            self.assertEqual(hashes, sorted(hashes))
            discovery_ranks = [
                row["stratum_rank"] for row in ordered if row["assignment"] == "discovery"
            ]
            held_out_ranks = [
                row["stratum_rank"] for row in ordered if row["assignment"] == "held_out"
            ]
            if discovery_ranks and held_out_ranks:
                self.assertLess(max(discovery_ranks), min(held_out_ranks))

    def test_source_top_k_is_the_max_token_length_of_the_eligible_population_only(
        self,
    ) -> None:
        _, rows = m.load_gw0()
        eligible = m.eligible_groups(rows)
        manifest = m.build_manifest()
        self.assertEqual(
            manifest["observation"]["attention"]["source_top_k"],
            m.max_visible_prompt_length(eligible),
        )
        eligible_edge_ids = {
            member["edge_id"] for members in eligible.values() for member in members
        }
        full_max = max(len(row["prompt"]["token_ids"]) for row in rows)
        eligible_max = max(
            len(row["prompt"]["token_ids"])
            for row in rows
            if row["edge_id"] in eligible_edge_ids
        )
        self.assertEqual(manifest["observation"]["attention"]["source_top_k"], eligible_max)
        self.assertLessEqual(eligible_max, full_max)

    def test_core_population_numbers_are_the_frozen_values(self) -> None:
        manifest = m.build_manifest()
        self.assertEqual(manifest["population"]["transition_identity_count"], 85)
        self.assertEqual(manifest["assignment"]["discovery_count"], 51)
        self.assertEqual(manifest["assignment"]["held_out_count"], 34)
        self.assertEqual(len(manifest["assignment"]["rows"]), 85)
        self.assertEqual(manifest["observation"]["attention"]["source_top_k"], 14)
        rows = manifest["assignment"]["rows"]
        by_relation: dict[str, int] = {}
        for row in rows:
            by_relation[row["relation"]] = by_relation.get(row["relation"], 0) + 1
        self.assertEqual(
            by_relation,
            {"capital": 23, "currency": 20, "hypernym": 24, "language": 18},
        )

    def test_manifest_and_contract_identities_are_canonical_hashes(self) -> None:
        manifest = m.build_manifest()
        contract = m.build_contract(manifest)
        self.assertEqual(
            manifest["manifest_sha256"],
            m.canonical_identity(manifest, "manifest_sha256"),
        )
        self.assertEqual(
            contract["population_contract_sha256"],
            m.canonical_identity(contract, "population_contract_sha256"),
        )

    def test_apportion_discovery_floors_do_not_exceed_the_exact_share(self) -> None:
        for count in range(0, 30):
            self.assertLessEqual(
                m.apportion_discovery(count, m.DISCOVERY_FRACTION),
                count * m.DISCOVERY_FRACTION,
            )

    def test_write_sealed_refuses_an_existing_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "already sealed"):
            m.write_sealed(m.GW_TS_1_DIR / "population-manifest.json", {})

    def test_algeria_provenance_fact_is_recorded_and_stays_held_out(self) -> None:
        manifest = m.build_manifest()
        facts = manifest["provenance_facts"]
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["identity"], m.INSTRUMENT_VALIDATION_EXPOSURE)
        self.assertIn("ATTR-1D", facts[0]["fact"])
        self.assertIn("no GW-TS-1 population assignment", facts[0]["assurance"])

        rows = manifest["assignment"]["rows"]
        exposed = [
            row
            for row in rows
            if (row["relation"], row["subject"], row["target"])
            == tuple(m.INSTRUMENT_VALIDATION_EXPOSURE.values())
        ]
        self.assertEqual(len(exposed), 1)
        self.assertEqual(exposed[0]["assignment"], "held_out")

    def test_leave_algeria_out_sensitivity_check_is_preregistered_as_secondary(
        self,
    ) -> None:
        manifest = m.build_manifest()
        checks = manifest["preregistered_sensitivity_checks"]
        self.assertEqual(len(checks), 1)
        check = checks[0]
        self.assertEqual(check["name"], "leave_algeria_out")
        self.assertEqual(check["excluded_identity"], m.INSTRUMENT_VALIDATION_EXPOSURE)
        self.assertIn("never in place of", check["authority"])

    def test_contract_carries_the_same_provenance_and_sensitivity_fields(self) -> None:
        manifest = m.build_manifest()
        contract = m.build_contract(manifest)
        self.assertEqual(contract["provenance_facts"], manifest["provenance_facts"])
        self.assertEqual(
            contract["preregistered_sensitivity_checks"],
            manifest["preregistered_sensitivity_checks"],
        )

    def test_contract_flags_population_capture_as_not_yet_authorized(self) -> None:
        contract = json.loads((m.GW_TS_1_DIR / "population-contract.json").read_text())
        self.assertIn("Population CAPTURE must not", contract["freeze"]["note"])


if __name__ == "__main__":
    unittest.main()
