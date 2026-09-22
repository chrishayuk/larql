import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from gwv2_adjudicate import distributions,effect,frontier,max_t_lower,summarize
from gwv2_fit import factors,predict
from gwread1_fit import fit_ridge


class AnalysisTests(unittest.TestCase):
    def test_factual_convergence_only_is_positive_adjusted_effect(self):
        logits=np.array([[5.,0,0],[0,5.,0],[0,0,5.]]*2)
        before=distributions(logits)
        after=before.copy(); after[:3]=[1/3,1/3,1/3]
        measured=effect(before,after,np.array([[0,1,2]]),np.array([[3,4,5]]))
        self.assertGreater(measured[0],0.5)
        after[3:]=[1/3,1/3,1/3]
        np.testing.assert_allclose(effect(before,after,np.array([[0,1,2]]),np.array([[3,4,5]])),0)

    def test_isolated_and_late_cells_cannot_declare_early_frontier(self):
        isolated=[False]*35; isolated[0]=True
        self.assertFalse(frontier(isolated)['exists'])
        late=[False]*35
        for i in [15,16,20,21]:late[i]=True
        self.assertFalse(frontier(late)['exists'])
        early=[False]*35
        for i in [10,11,15,16]:early[i]=True
        self.assertTrue(frontier(early)['exists'])
        self.assertEqual(frontier(early)['coordinate']['depths'],[8,12])

    def test_bootstrap_refuses_zero_uncertainty(self):
        with self.assertRaisesRegex(ValueError,'zero bootstrap'):
            max_t_lower(np.ones(70),np.ones((100,70)))

    def test_diagnostic_subset_does_not_require_primary_uncertainty_gate(self):
        report=summarize(np.ones((3,75,2,2)),27022035,False)
        self.assertTrue(report['diagnostic_only'])
        self.assertTrue(all(c['gate_pass'] is None for c in report['cells']))
        self.assertTrue(all(c['gate_pass'] is None for c in report['sham_cells']))

    def test_f32_factor_execution_and_parameter_shape(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(40,8)); y=x@rng.normal(size=(8,8))
        packed=factors(fit_ridge(x,y),4)
        output=predict(x,packed)
        self.assertEqual(output.dtype,np.float32)
        self.assertEqual(sum(v.size for v in packed),2*8*4+2*8)
        xm,ym,a,b=packed
        np.testing.assert_allclose(output,((x.astype('f4')-xm)@a)@b+ym,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
