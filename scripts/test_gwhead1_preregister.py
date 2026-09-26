import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("gwhead1_preregister.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("gwhead1_preregister", MODULE_PATH)
gwhead1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gwhead1)


class GwHead1PreregistrationTests(unittest.TestCase):
    def test_expected_gemma_geometry_is_accepted(self) -> None:
        attention = [
            {
                "operator": "softmax",
                "span": "sliding",
                "window": 1024,
                "geometry": {"head_dim": 256, "num_kv_heads": 4},
            }
            for _ in range(25)
        ]
        graph = {
            "components": [
                {
                    "id": "target",
                    "role": "primary_text",
                    "num_layers": 34,
                    "hidden_size": 2560,
                    "attention": attention,
                    "execution": {
                        "attention": {
                            "num_q_heads": 8,
                            "num_kv_heads": 4,
                            "head_dim": 256,
                        },
                        "norm": {
                            "placement": "pre_post",
                            "post": {
                                "kind": "rms_norm",
                                "eps": 9.999999974752427e-7,
                                "weight_offset": 1.0,
                            },
                        },
                    },
                }
            ]
        }
        self.assertEqual(gwhead1.geometry(graph)["num_q_heads"], 8)
        graph["components"][0]["execution"]["attention"]["num_q_heads"] = 16
        with self.assertRaisesRegex(ValueError, "unexpected L24 execution geometry"):
            gwhead1.geometry(graph)

    def test_gwconv_authority_requires_fixed_l24_for_every_relation(self) -> None:
        site = {"index": 48, "layer": 24, "site": "attention"}
        adjudication = {
            "schema": "larql.gwconv1.adjudication.v1",
            "gate": {"conditions": {"full_support": True}},
            "selection": {
                "global_site": site,
                "relation_sites": {
                    relation: dict(site)
                    for relation in ("capital", "currency", "language", "hypernym")
                },
            },
        }
        gwhead1.require_gwconv1(adjudication)
        adjudication["selection"]["relation_sites"]["capital"]["layer"] = 23
        with self.assertRaisesRegex(ValueError, "relation sites"):
            gwhead1.require_gwconv1(adjudication)

    def test_frozen_constants_do_not_define_small_post_hoc(self) -> None:
        self.assertEqual(gwhead1.EXPECTED_HEADS, 8)
        self.assertEqual(gwhead1.SMALL_HEAD_LIMIT, 4)
        self.assertEqual(gwhead1.TRAIN_RETENTION, 0.80)
        self.assertEqual(gwhead1.HELD_OUT_RETENTION_LOWER, 0.50)


if __name__ == "__main__":
    unittest.main()
