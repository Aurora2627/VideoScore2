import unittest
from expanded_metrics import summarize,prompt_bootstrap
from prepare_expanded_inputs import select_groups
from quality_metrics import LABELS

def result(score=3): return {'status':'completed','valid':True,'eos_reached':True,'scores':dict.fromkeys(LABELS,score)}
def samples(): return [{'video_name':name,'prompt':'same','ground_truth':dict.fromkeys(LABELS,score)} for name,score in [('a',1),('b',5)]]
class ExpandedTests(unittest.TestCase):
    def test_selection_excludes_seen_prompts_without_looking_at_labels(self):
        rows=[{'prompt':p,'video_name':p+str(i)} for p in ('old','new','later') for i in range(2)]
        self.assertEqual([r['video_name'] for r in select_groups(rows,{'old'},set(),1)],['new0','new1'])
    def test_partial_group_does_not_enter_bootstrap(self):
        report=summarize(samples(),{'a':result(1),'b':result(5)},{'a':result(1)},dict.fromkeys(LABELS,1/3))
        self.assertFalse(report['full_coverage']);self.assertEqual(report['paired_valid_videos'],1)
        self.assertIsNone(report['paired_prompt_bootstrap']);self.assertEqual(report['conclusion'],'incomplete_coverage')
    def test_paired_error_direction_and_tie_prediction(self):
        report=summarize(samples(),{'a':result(),'b':result()},{'a':result(1),'b':result(5)},dict.fromkeys(LABELS,1/3))
        self.assertEqual(report['baseline']['pair_correct'],0);self.assertEqual(report['more_detail']['pair_correct'],1)
        self.assertEqual(report['paired_prompt_bootstrap']['mean'],-2)
    def test_bootstrap_seed_is_reproducible(self):
        self.assertEqual(prompt_bootstrap([-1,0,1]),prompt_bootstrap([-1,0,1]))
class ResumeTests(unittest.TestCase):
    def test_changed_identity_cannot_resume(self):
        from run_expanded_evaluation import validate_resume
        with self.assertRaises(ValueError): validate_resume({'manifest':'a'},{'manifest':'b'})
    def test_each_clip_has_exactly_two_configurations(self):
        from run_expanded_evaluation import build_jobs
        jobs=build_jobs(samples())
        self.assertEqual(len(jobs),4)
        self.assertEqual({(phase,item['video_name']) for phase,side,item in jobs},{('baseline','a'),('baseline','b'),('more_detail','a'),('more_detail','b')})
if __name__=='__main__': unittest.main()
