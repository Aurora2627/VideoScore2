import copy,unittest,tempfile,hashlib
from pathlib import Path
from selection import DEFAULT,LABELS,select_candidates,validate_policy
from run_batch import preflight

def result(scores=(4,4,4),**kw):
 d={'status':'completed','eos_reached':True,'valid':True,'scores':dict(zip(LABELS,scores)) if scores is not None else None};d.update(kw);return d
class SelectionTests(unittest.TestCase):
 def test_prompt_isolation(self):
  report=select_candidates([{'video_name':'a','prompt':'one'},{'video_name':'b','prompt':'two'}],{'a':result(),'b':result()},DEFAULT)
  self.assertEqual(len(report['groups']),2);self.assertTrue(all(not g['comparison_available'] for g in report['groups']));self.assertFalse(any(r['selected'] for g in report['groups'] for r in g['candidates']))
 def test_quality_gate_beats_weighted_score(self):
  report=select_candidates([{'video_name':n,'prompt':'same'} for n in 'ab'],{'a':result((5,5,1)),'b':result((3,3,3))},DEFAULT)
  rows={r['video_name']:r for r in report['groups'][0]['candidates']};self.assertEqual(rows['a']['decision'],'manual_review');self.assertTrue(rows['b']['selected'])
 def test_truncated_never_selected(self):
  g=select_candidates([{'video_name':n,'prompt':'same'} for n in 'ab'],{'a':result(eos_reached=False),'b':result()},DEFAULT)['groups'][0]
  self.assertFalse(g['comparison_available']);self.assertFalse(any(r['selected'] for r in g['candidates']))
 def test_ties_are_explicit(self):
  g=select_candidates([{'video_name':n,'prompt':'same'} for n in 'ab'],{'a':result(),'b':result()},DEFAULT)['groups'][0]
  self.assertEqual([r['rank'] for r in g['candidates']],[1,1]);self.assertTrue(all(r['selected'] for r in g['candidates']))
 def test_no_scores_safe(self):
  g=select_candidates([{'video_name':'a','prompt':'same'}],{'a':result(scores=None)},DEFAULT)['groups'][0];self.assertFalse(g['candidates'][0]['valid'])
 def test_invalid_weights(self):
  p=copy.deepcopy(DEFAULT);p['weights'][LABELS[0]]=float('nan')
  with self.assertRaises(ValueError):validate_policy(p)
 def test_manifest_hash_and_traversal(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'video.mp4';p.write_bytes(b'input');item={'video_name':'clip','prompt':'same','video_path':str(p),'video_sha256':hashlib.sha256(b'input').hexdigest()};preflight({'samples':[item]})
   bad=dict(item,video_sha256='bad')
   with self.assertRaises(ValueError):preflight({'samples':[bad]})
   bad=dict(item,video_name='../escape')
   with self.assertRaises(ValueError):preflight({'samples':[bad]})
 def test_duplicate_ids(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'clip';p.write_bytes(b'x');item={'video_name':'clip','prompt':'a','video_path':str(p),'video_sha256':hashlib.sha256(b'x').hexdigest()}
   with self.assertRaises(ValueError):preflight({'samples':[item,item]})
if __name__=='__main__':unittest.main()
