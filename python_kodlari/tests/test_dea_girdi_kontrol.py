import unittest
import numpy as np
from verimlilik_kontrol.audit import compare_same_branches,deterministic_noise,correlation,describe_changes,measured_profile_change,recommend


class AuditTests(unittest.TestCase):
    def test_cohort_average_excludes_unscored_branches(self):
        a=[{'branch':'a','cluster_id':'x','efficiency':.2},{'branch':'b','cluster_id':'y','efficiency':.9}]
        b=[{'branch':'a','cluster_id':'x','efficiency':.3},{'branch':'b','cluster_id':'y','efficiency':None}]
        summary,rows=compare_same_branches(a,b,.01,'2026-12')
        self.assertEqual(summary['common_scored'],1);self.assertAlmostEqual(summary['base_mean_same_cohort'],.2);self.assertFalse(summary['metric_is_performance_accuracy'])
    def test_shared_noise_reproducible_and_bounded(self):
        a=deterministic_noise(1,'cluster',0,5,4,.01);b=deterministic_noise(1,'cluster',0,5,4,.01)
        np.testing.assert_array_equal(a,b);self.assertTrue((np.abs(a)<=.01).all());self.assertFalse(np.array_equal(a,deterministic_noise(1,'cluster',1,5,4,.01)))
    def test_constant_staff_not_treated_as_valid_correlation(self):
        self.assertEqual(correlation([1,2,3],[2,2,2]),{'pearson':None,'spearman':None})
    def test_score_scale_is_points(self):
        self.assertAlmostEqual(describe_changes([-.01,.03])['mean_abs_change'],.02)
    def test_zero_to_positive_is_not_a_stable_profile(self):
        maximum,flag=measured_profile_change([0,1],[2,1]);self.assertTrue(flag)
    def test_input_addition_may_not_lower_same_cohort_score(self):
        a=[{'branch':'a','cluster_id':'x','efficiency':.5}];b=[{'branch':'a','cluster_id':'x','efficiency':.4}]
        with self.assertRaises(ValueError):compare_same_branches(a,b,.01,'2026-12')
    def test_higher_score_does_not_override_incomplete_coverage(self):
        policy={'minimum_global_coverage':1.,'local_score_p95_review':.03,'peer_removal_max_review':.1}
        coverage=[{'scenario':'customer','period':'p','coverage':1.},{'scenario':'customer_staff','period':'p','coverage':.8,'withheld_branches':40}]
        stress=[{'scenario':'customer','local':{'p95_abs_change':.01},'peer_removal':{'max_abs_change':.2},'peer_removal_size_rule_passed':True}]
        r=recommend(policy,'p',coverage,stress);self.assertEqual(r['recommended_scenario'],'customer');self.assertFalse(r['augmented_global_eligible']);self.assertIn('influential_peer_review',r['review_reasons'])

if __name__=='__main__':unittest.main()
