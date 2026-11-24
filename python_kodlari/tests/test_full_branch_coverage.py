import copy
import unittest
from kumeleme_secimi.coverage import build_contract,assignment_coverage,assert_spec_covered
from kumeleme_secimi.selection import score_candidate,choose


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.base={'inputs':[{'name':'customer','source':'customer_days','transform':'divide_by_calendar_days'}],
                   'outputs':[{'name':n,'source':n} for n in ['demand','term','loan']],'insufficient_cluster_policy':'error'}
        self.audit={'additional_inputs':[{'name':'fte','source':'fte'}],'minimum_global_coverage':1.0}
    def test_dimension_union_requires_fifteen(self):
        c=build_contract(self.base,self.audit,12)
        self.assertEqual([s['required_n'] for s in c['scenarios']],[12,15]);self.assertEqual(c['required_min_cluster_size'],15)
    def test_new_input_automatically_increases_requirement(self):
        self.audit['additional_inputs'].append({'name':'space','source':'space'})
        self.assertEqual(build_contract(self.base,self.audit,12)['required_min_cluster_size'],18)
    def test_configured_stricter_minimum_is_preserved(self):
        self.assertEqual(build_contract(self.base,self.audit,20)['required_min_cluster_size'],20)
    def test_partial_coverage_cannot_be_configured(self):
        self.audit['minimum_global_coverage']=.9
        with self.assertRaises(ValueError):build_contract(self.base,self.audit,12)
    def records(self,n=20):
        return [{'__period':'2027-01','__branch':str(i),'__group':'g','cluster':0} for i in range(n)]
    def test_all_branches_accounted_for(self):
        r=self.records();self.assertEqual(assignment_coverage(r,r,15,{'g':1})[0]['coverage'],1.)
    def test_missing_branch_is_not_hidden_by_large_cluster(self):
        r=self.records()
        with self.assertRaisesRegex(ValueError,'missing'):assignment_coverage(r[:-1],r,15,{'g':1})
    def test_duplicate_branch_rejected(self):
        r=self.records()
        with self.assertRaises(ValueError):assignment_coverage(r+[r[0]],r,15,{'g':1})
    def test_small_cluster_rejected_even_if_every_branch_assigned(self):
        r=self.records(14)
        with self.assertRaisesRegex(ValueError,'at least 15'):assignment_coverage(r,r,15,{'g':1})
    def test_empty_learned_cluster_rejected(self):
        r=self.records()
        with self.assertRaises(ValueError):assignment_coverage(r,r,15,{'g':2})
    def test_updated_dea_fields_require_reselection(self):
        c=build_contract(self.base,self.audit,12);assert_spec_covered(c,self.base)
        changed=copy.deepcopy(self.base);changed['outputs'][0]['source']='different_balance'
        with self.assertRaisesRegex(ValueError,'rerun'):assert_spec_covered(c,changed)
    def test_no_silent_reenable_of_partial_scoring(self):
        c=build_contract(self.base,self.audit,12);changed=copy.deepcopy(self.base);changed['insufficient_cluster_policy']='withhold_scores'
        with self.assertRaises(ValueError):assert_spec_covered(c,changed)

if __name__=='__main__':unittest.main()
