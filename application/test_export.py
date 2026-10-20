import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from selection import select_candidates,DEFAULT,LABELS
class ExportTests(unittest.TestCase):
 def fixture(self,root,failed):
  run=root/'run';run.mkdir();(run/'sources').mkdir();video=root/'input.mp4';video.write_bytes(b'synthetic-test-fixture')
  samples=[{'video_name':n,'prompt':'a new business prompt','video_path':str(video),'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest()} for n in 'ab']
  results={n:{'status':'completed','eos_reached':True,'valid':True,'scores':dict(zip(LABELS,(4,4,4))),'generation_seconds':0.1,'generated_token_ids':[1],'peak_allocated_bytes':1024,'video_sha256':samples[0]['video_sha256']} for n in 'ab'}
  if failed:results['b']={'status':'failed','valid':False,'eos_reached':False}
  write=lambda name,d:(run/name).write_text(json.dumps(d),encoding='utf-8')
  write('run.json',{'status':'completed_with_errors' if failed else 'completed','model_load_count':1,'source_sha256':{},'model_load_seconds':0.1,'wall_seconds':1})
  write('manifest.json',{'samples':samples});write('selection.json',select_candidates(samples,results,DEFAULT));write('metrics.json',{'attempted':2,'valid_outputs':1 if failed else 2,'mae':{k:None for k in LABELS}})
  for n,r in results.items():
   (run/n).mkdir();write(n+'/result.json',r)
   if r['valid']:(run/n/'raw_output.txt').write_text('test fixture output',encoding='utf-8')
  return run
 def export(self,root,failed):
  run=self.fixture(root,failed);out=root/'report';result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('export_run.py')),'--run-dir',str(run),'--output-dir',str(out)],capture_output=True,encoding='utf-8');self.assertEqual(result.returncode,0,result.stdout+result.stderr);return out
 def test_unlabeled_new_candidates_without_history(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=self.export(Path(tmp),False);summary=json.loads((out/'application-summary.json').read_text(encoding='utf-8'));self.assertEqual(summary['valid_outputs'],2);self.assertIsNone(summary['samples'][0]['ground_truth']);self.assertTrue((out/'media/a.mp4').exists());self.assertIn('查看视频',(out/'review.html').read_text(encoding='utf-8'))
 def test_failed_candidate_remains_reviewable(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=self.export(Path(tmp),True);report=json.loads((out/'selection.json').read_text(encoding='utf-8'));rows={r['video_name']:r for r in report['groups'][0]['candidates']};self.assertEqual(rows['b']['decision'],'manual_review');self.assertTrue((out/'review.html').exists())
if __name__=='__main__':unittest.main()
