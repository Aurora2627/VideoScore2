import unittest
from evaluate_test import original_score_vector
class ScoreAdapterTests(unittest.TestCase):
 def test_actual_original_schema(self):
  r={'scores':{'visual quality':1,'text-to-video alignment':2,'physical/common-sense consistency':1}}
  self.assertEqual(original_score_vector(r),[1.,2.,1.])
 def test_head_schema_is_not_silently_treated_as_original(self):
  with self.assertRaises(KeyError):original_score_vector({'scores':{'visual_quality':3,'text_alignment':3,'physical_consistency':3}})
 def test_non_finite_or_out_of_range_rejected(self):
  for value in (float('nan'),0,6):
   r={'scores':dict.fromkeys(('visual quality','text-to-video alignment','physical/common-sense consistency'),value)}
   with self.assertRaises(ValueError):original_score_vector(r)
if __name__=='__main__':unittest.main()
