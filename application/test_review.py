import copy,json,tempfile,unittest
from pathlib import Path
from render_review import render

class ReviewTests(unittest.TestCase):
 def fixture(self,root):
  policy={'minimum_scores':{'visual quality':2,'text-to-video alignment':4,'physical/common-sense consistency':3},'weights':{'visual quality':0.2,'text-to-video alignment':0.5,'physical/common-sense consistency':0.3}}
  row={'video_name':'clip','scores':{'visual quality':4,'text-to-video alignment':4,'physical/common-sense consistency':4},'rank':None,'selected':False,'decision':'single_candidate_review','weighted_score':4,'review_reasons':[]}
  (root/'selection.json').write_text(json.dumps({'policy':policy,'groups':[{'prompt':'A <script>alert(1)</script> "quoted" prompt','comparison_available':False,'candidates':[row]}]}),encoding='utf-8')
  (root/'clip').mkdir();raw=root/'clip/raw_output.txt';raw.write_text('<img src=x onerror=alert(1)>',encoding='utf-8');return raw
 def test_escaped_content_and_preserved_raw(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);raw=self.fixture(root);before=raw.read_bytes();out=root/'review.html';render(root,out);text=out.read_text(encoding='utf-8')
   self.assertNotIn('<script>alert(1)</script>',text);self.assertNotIn('<img src=x',text);self.assertIn('&lt;script&gt;',text);self.assertIn('&lt;img',text);self.assertEqual(raw.read_bytes(),before)
 def test_actual_policy_is_displayed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root);out=root/'review.html';render(root,out);text=out.read_text(encoding='utf-8');self.assertIn('文本匹配 4 分',text);self.assertIn('画面质量 20%',text)
 def test_no_report_overwrite(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root);out=root/'review.html';render(root,out)
   with self.assertRaises(FileExistsError):render(root,out)
if __name__=='__main__':unittest.main()
