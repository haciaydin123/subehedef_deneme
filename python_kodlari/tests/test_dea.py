import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from verimlilik.solver import minimum_units,solve_peer_set,validate
from verimlilik.study import analyze,prepare


class DEATests(unittest.TestCase):
    def test_known_crs_productivity(self):
        result=solve_peer_set([[1],[1],[2]],[[1],[2],[2]])
        np.testing.assert_allclose([r['efficiency'] for r in result],[.5,1,.5])
        self.assertAlmostEqual(result[0]['radial_growth'],1)
    def test_crs_scale_assumption_visible_in_vrs(self):
        result=solve_peer_set([[1],[2]],[[1],[4]])
        self.assertAlmostEqual(result[0]['efficiency'],.5);self.assertAlmostEqual(result[0]['vrs_efficiency'],1)
    def test_three_outputs_share_radial_multiplier(self):
        result=solve_peer_set([[1],[1]],[[1,1,1],[2,2,2]])[0]
        np.testing.assert_allclose(result['radial_projection'],[2,2,2]);self.assertAlmostEqual(result['phi'],2)
    def test_score_one_can_have_output_slack(self):
        result=solve_peer_set([[1],[1]],[[2,1],[2,2]])[0]
        self.assertAlmostEqual(result['efficiency'],1)
        self.assertEqual(result['status'],'radially_efficient_with_slack')
        np.testing.assert_allclose(result['output_slack'],[0,1])
    def test_units_do_not_change_scores(self):
        X=np.array([[1,2],[2,1],[3,2]]);Y=np.array([[3,2],[2,4],[3,3]])
        a=solve_peer_set(X,Y);b=solve_peer_set(X*[1000,.001],Y*[1e6,.0001])
        np.testing.assert_allclose([r['phi'] for r in a],[r['phi'] for r in b],rtol=1e-7)
    def test_zero_output_is_not_epsilon_imputed(self):
        result=solve_peer_set([[1],[1]],[[0,1],[1,2]])[0]
        self.assertEqual(result['radial_projection'][0],0)
        self.assertGreater(result['output_projection'][0],0)
    def test_bad_values_rejected(self):
        for X,Y in [([[0]],[[1]]),([[1]],[[0]]),([[1]],[[-1]]),([[np.nan]],[[1]]),([[1]],[[1,0]])]:
            with self.subTest(X=X,Y=Y):
                with self.assertRaises(ValueError):validate(X,Y)
    def test_minimum_size_heuristic(self):
        self.assertEqual(minimum_units(1,3),12);self.assertEqual(minimum_units(2,3),15)
    def test_small_cluster_stops_instead_of_dropping_branches(self):
        panel=pd.DataFrame([{'__branch':str(i),'__period':'2026-12','__group':'g','cluster_id':'g__C1','x':1,'y':i+1} for i in range(5)])
        cfg={'inputs':[{'name':'x'}],'outputs':[{'name':'y'}],'tolerance':1e-7,'high_radial_growth_flag':1,'high_radial_efficiency_share_flag':.5}
        with self.assertRaisesRegex(ValueError,'tam kapsam'):analyze(panel,cfg)
    def test_small_cluster_is_detected_before_any_partial_calculation(self):
        from unittest.mock import patch
        panel=pd.DataFrame([{'__branch':f'{g}{i}','__period':'2026-12','__group':'g','cluster_id':g,'x':1,'y':i+1} for g,n in [('large',6),('small',5)] for i in range(n)])
        cfg={'inputs':[{'name':'x'}],'outputs':[{'name':'y'}],'tolerance':1e-7,'high_radial_growth_flag':1,'high_radial_efficiency_share_flag':.5}
        with patch('verimlilik.study.solve_peer_set') as solve:
            with self.assertRaises(ValueError):analyze(panel,cfg)
            solve.assert_not_called()
    def test_clusters_cannot_borrow_other_frontier(self):
        rows=[{'__branch':f'{g}{i}','__period':'2026-12','__group':g,'cluster_id':g,'x':1,'y':(i+1)*factor} for g,factor in [('a',1),('b',1000)] for i in range(6)]
        cfg={'inputs':[{'name':'x'}],'outputs':[{'name':'y'}],'tolerance':1e-7,'high_radial_growth_flag':1,'high_radial_efficiency_share_flag':.5}
        result=analyze(pd.DataFrame(rows),cfg)
        for p in result['peers']:self.assertEqual(p['branch'][0],p['peer_branch'][0])
        np.testing.assert_allclose([s['efficiency'] for s in result['scores'][:6]],[s['efficiency'] for s in result['scores'][6:]])
    def test_reverse_row_order_keeps_efficiency(self):
        X=np.array([[1],[2],[3]]);Y=np.array([[1,3],[4,1],[2,5]])
        a=solve_peer_set(X,Y);b=solve_peer_set(X[::-1],Y[::-1])
        np.testing.assert_allclose([r['phi'] for r in a],[r['phi'] for r in b[::-1]])
    def test_customer_transfers_use_calendar_exposure_not_closing_count(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'rows.csv'
            pd.DataFrame([{'b':'001','p':'2024-02','g':'g','customer_days':145,'closing_customers':1,'balance':100}]).to_csv(source,index=False)
            cfg={'input':{'keys':{'branch':'b','period':'p','group':'g'}},'inputs':[{'name':'x','source':'customer_days','transform':'divide_by_calendar_days'}],'outputs':[{'name':'y','source':'balance'}]}
            assignments=[{'__branch':'001','__period':'2024-02','__group':'g','mode':'frozen','method':'gower_kmedoids','cluster_id':'g__C1'}]
            result=prepare(source,cfg,assignments,'2024-02');self.assertEqual(result['x'].iloc[0],5);self.assertEqual(result['__branch'].iloc[0],'001')
    def test_assignment_geography_cannot_be_silently_mismatched(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'rows.csv';pd.DataFrame([{'b':'a','p':'2026-12','g':'different','x':1,'y':2}]).to_csv(source,index=False)
            cfg={'input':{'keys':{'branch':'b','period':'p','group':'g'}},'inputs':[{'name':'x','source':'x'}],'outputs':[{'name':'y','source':'y'}]}
            assignments=[{'__branch':'a','__period':'2026-12','__group':'g','mode':'frozen','method':'gower_kmedoids','cluster_id':'g__C1'}]
            with self.assertRaises(ValueError):prepare(source,cfg,assignments,'2026-12')

if __name__=='__main__':unittest.main()
