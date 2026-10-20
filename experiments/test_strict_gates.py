import copy,unittest
from strict_gates import gates

def metrics(mae,pairs=1,correct=1):return {'macro_mae':mae,'sample_count':2,'valid_rate':1,'pair_total':pairs,'pair_correct':correct}
def fixture():
 original={'candidate_development':{'more_detail':metrics(1)},'regression_candidate':metrics(0.55),'regression_baseline':metrics(0.78),'fresh_candidate':metrics(0.33,0,0),'fresh_baseline':metrics(0.33,0,0)}
 supplement={'phases':{'baseline':{'metrics':metrics(1)},'more_detail':{'metrics':metrics(0.5)}}};return original,supplement
class StrictGateTests(unittest.TestCase):
 def test_tied_original_retained_with_independent_strict_check(self):
  original,supplement=fixture();self.assertTrue(all(gates(original,supplement).values()))
 def test_reject_original_fresh_regression_despite_better_average(self):
  original,supplement=fixture();original['fresh_candidate']['macro_mae']=0.5;self.assertFalse(gates(original,supplement)['original_fresh_not_worse'])
 def test_lower_mae_does_not_override_wrong_pair(self):
  original,supplement=fixture();supplement['phases']['more_detail']['metrics']['pair_correct']=0;self.assertFalse(gates(original,supplement)['supplement_pair_correct'])
 def test_reject_incomplete_outputs(self):
  original,supplement=fixture();supplement['phases']['more_detail']['metrics']['valid_rate']=0.5;self.assertFalse(gates(original,supplement)['all_outputs_complete'])
if __name__=='__main__':unittest.main()
