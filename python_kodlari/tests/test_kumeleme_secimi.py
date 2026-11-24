import copy
import unittest
from kumeleme_secimi.selection import choose,score_candidate,validate_config


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'validation_periods':['2026-10','2026-11'],'evaluation_periods':['2026-12'],
                  'weights':{'silhouette':.3,'refit_ari':.3,'subsample_ari':.25,'seed_ari':.15},
                  'thresholds':{'min_cluster_size':12,'silhouette':.1,'refit_ari':.6,'subsample_ari':.6,'seed_ari':.6,'worst_group_refit_ari':.4,'worst_group_silhouette':.1},'incumbent':'old','minimum_gain':.02}
        self.case={'case_id':'case','feature_ratio':.25,'feature_count':6,'bank_clusters':4,'k_by_group':{'a':2,'b':2},'group_counts':{'a':80,'b':120}}
        self.selected=[{'method':'gower_kmedoids','group':g,'converged':True,'min_cluster_size':40,'distinct_clusters':2,'cluster_sizes':[n//2,n//2]} for g,n in self.case['group_counts'].items()]
        self.monthly=[{'method':'gower_kmedoids','group':g,'period':p,'mode':mode,'ari':.8,'common_gower_silhouette':.15,'min_cluster_size':40,'distinct_clusters':2,'cluster_sizes':[n//2,n//2]} for g,n in self.case['group_counts'].items() for p in ['2026-10','2026-11','2026-12'] for mode in ['frozen','monthly_refit']]
        self.sensitivity=[{'group':g,'method':'gower_kmedoids','case':kind,'repeat':i,'ari':.8} for g in ['a','b'] for kind in ['seed','subsample'] for i in range(3)]
    def score(self):return score_candidate(self.case,self.selected,self.monthly,self.sensitivity,'gower_kmedoids',self.cfg,{'seed':3,'subsample':3})
    def test_alternative_can_be_main(self):
        result=choose([self.score()],self.cfg);self.assertEqual(result['selected']['method'],'gower_kmedoids')
    def test_future_metrics_do_not_change_choice(self):
        before=self.score()
        for row in self.monthly:
            if row['period']=='2026-12':row.update(ari=-99,common_gower_silhouette=99,min_cluster_size=0)
        self.assertEqual(before,self.score())
    def test_no_forced_bad_winner(self):
        self.cfg['thresholds']['silhouette']=.9;self.assertIsNone(choose([self.score()],self.cfg)['selected'])
    def test_small_cluster_excludes_good_score(self):
        self.monthly[0]['min_cluster_size']=2;self.assertFalse(self.score()['qualified'])
    def test_weak_group_cannot_hide_in_average(self):
        for row in self.monthly:
            if row['mode']=='monthly_refit':row['ari']=.3 if row['group']=='a' else 1
        result=self.score();self.assertGreater(result['metrics']['refit_ari'],.6);self.assertFalse(result['qualified'])
    def test_missing_month_disqualifies(self):
        self.monthly.pop(0);result=self.score();self.assertFalse(result['qualified']);self.assertIsNone(result['score'])
    def test_incomplete_repeats_disqualify(self):
        self.sensitivity.pop();self.assertFalse(self.score()['qualified'])
    def test_switch_margin_preserves_qualified_incumbent(self):
        a=self.score();b=copy.deepcopy(a);a['candidate_id']='old';b['candidate_id']='new';b['score']+=.01
        self.assertEqual(choose([a,b],self.cfg)['selected']['candidate_id'],'old')
    def test_bad_incumbent_replaced_even_for_small_gain(self):
        a=self.score();b=copy.deepcopy(a);a.update(candidate_id='old',qualified=False);b.update(candidate_id='new',score=a['score']+.001)
        self.assertEqual(choose([a,b],self.cfg)['selected']['candidate_id'],'new')
    def test_split_overlap_rejected(self):
        self.cfg['evaluation_periods']=['2026-11']
        with self.assertRaises(ValueError):validate_config(self.cfg,'2026-09',['2026-10','2026-11','2026-12'])
    def test_high_bank_average_cannot_hide_weak_geography(self):
        for row in self.monthly:
            if row['mode']=='frozen':row['common_gower_silhouette']=.05 if row['group']=='a' else .4
        r=self.score();self.assertGreater(r['metrics']['silhouette'],.1);self.assertFalse(r['qualified'])
        self.assertIn('worst_group_silhouette',r['reasons']);self.assertEqual(r['group_quality'][0]['passed'],False)
    def test_geographic_threshold_equality_passes(self):
        for row in self.monthly:row['common_gower_silhouette']=.1
        self.assertTrue(self.score()['qualified'])
    def test_missing_geographic_measure_cannot_be_ignored(self):
        self.monthly[0]['common_gower_silhouette']=None
        r=self.score();self.assertFalse(r['qualified']);self.assertIsNone(r['metrics']['worst_group_silhouette'])
    def test_geographic_threshold_can_be_configured_without_code_change(self):
        self.cfg['thresholds']['worst_group_silhouette']=.2
        self.assertFalse(self.score()['qualified'])
        self.cfg['thresholds']['worst_group_silhouette']=.15
        self.assertTrue(self.score()['qualified'])
    def test_invalid_geographic_threshold_rejected(self):
        for value in [float('nan'),1.1,-1.1]:
            self.cfg['thresholds']['worst_group_silhouette']=value
            with self.assertRaises(ValueError):validate_config(self.cfg,'2026-09',['2026-10','2026-11','2026-12'])

if __name__=='__main__':unittest.main()
