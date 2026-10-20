import unittest
from quality_metrics import evaluate,choose,promotion_gate,LABELS
W=dict(zip(LABELS,(0.3,0.4,0.3)))
def result(scores):return {'status':'completed','valid':True,'eos_reached':True,'scores':dict(zip(LABELS,scores))}
class QualityTests(unittest.TestCase):
 def test_incomplete_cannot_win(self):
  baseline={'macro_mae':1};candidate={'valid_rate':0.5,'macro_mae':0};self.assertEqual(choose(baseline,{'candidate':candidate},{'candidate':{'frames':8,'max_side':224}}),'baseline')
 def test_equal_error_keeps_baseline(self):
  self.assertEqual(choose({'macro_mae':1},{'candidate':{'valid_rate':1,'macro_mae':1}},{'candidate':{'frames':4,'max_side':448}}),'baseline')
 def test_tied_predictions_not_correct_for_strict_label_preference(self):
  samples=[{'video_name':'a','prompt':'same','ground_truth':dict(zip(LABELS,(5,5,5)))},{'video_name':'b','prompt':'same','ground_truth':dict(zip(LABELS,(1,1,1)))}]
  m=evaluate(samples,{'a':result((3,3,3)),'b':result((3,3,3))},W);self.assertEqual(m['pair_total'],1);self.assertEqual(m['pair_correct'],0)
 def test_gate_rejects_regression(self):
  base={'macro_mae':1,'valid_rate':1,'pair_total':1,'pair_correct':1};good=dict(base,macro_mae=0.5);bad=dict(base,macro_mae=1.2)
  gates=promotion_gate(good,base,bad,base,good);self.assertFalse(all(gates.values()));self.assertFalse(gates['regression_mae_not_worse'])
if __name__=='__main__':unittest.main()
