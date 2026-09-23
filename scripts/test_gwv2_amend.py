import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from gwv2_amend import donor_map, validate
from gwv2_population import read_jsonl

P=ROOT/'bench/gw-v2/gemma3-4b-it-phase1/gwv2-protocol.json'

class AmendmentTests(unittest.TestCase):
    def test_sealed_lineage_and_actual_donor_correspondence(self):
        self.assertEqual(len(validate(P)),4)
        rows=read_jsonl(P.parent/'input.jsonl')
        entries=donor_map(rows)
        self.assertEqual(len(entries),603)
        self.assertEqual(sum(e['shuffled'] for e in entries),522)
        for entry in entries:
            donor=rows[entry['donor_row']]
            recipient=rows[entry['row']]
            self.assertEqual(donor['split'],'train')
            self.assertEqual(donor['semantic_edge']['relation'],recipient['semantic_edge']['relation'])
            self.assertEqual(donor['semantic_edge']['prompt_semantic_family'],recipient['semantic_edge']['prompt_semantic_family'])
        self.assertEqual({e['subject_id'] for e in entries if not e['shuffled']},{'KNA','STP'})

    def test_mapping_reused_across_prompt_families(self):
        mapping={}
        for e in donor_map(read_jsonl(P.parent/'input.jsonl')):
            key=e['relation'],e['subject_id']
            self.assertEqual(mapping.setdefault(key,e['donor_subject_id']),e['donor_subject_id'])

    def test_tampered_amendment_is_refused(self):
        original=Path.read_text
        def altered(path,*args,**kwargs):
            data=original(path,*args,**kwargs)
            if path.name=='gwv2-amend-1.json':
                d=json.loads(data); d['reporting']['diagnostic_only']=False
                return json.dumps(d)
            return data
        with patch.object(Path,'read_text',altered),self.assertRaisesRegex(ValueError,'amendment identity'):
            validate(P)

    def test_rehashed_wrong_mapping_still_refused(self):
        import gwv2_amend
        bad=donor_map(read_jsonl(P.parent/'input.jsonl'))
        bad[0]['donor_subject_id']='KNA'
        with patch.object(gwv2_amend,'donor_map',return_value=bad),self.assertRaisesRegex(ValueError,'correspondence'):
            validate(P)

if __name__=='__main__': unittest.main()
