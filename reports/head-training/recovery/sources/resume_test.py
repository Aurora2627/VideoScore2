"""Resume only the failed original-scorer comparison, preserving selected weights and test predictions."""
import argparse,datetime,os,shutil,subprocess,sys,traceback
from pathlib import Path
from prepare_data import read,save,sha
from run_pipeline import ROOT,publish
def now():return datetime.datetime.now().astimezone().isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args();run=a.run_dir;status=read(run/'status.json');assert status['status']=='needs_attention' and status['stage']=='test'
 folder=run/'recovery';folder.mkdir(exist_ok=False);save(folder/'status-before-recovery.json',status)
 protected=[run/'heads/selection.json',run/'test/head-test-predictions.json',run/'test/head-test-metrics.json']+list((run/'heads').glob('*.pt'))
 hashes={p.relative_to(run).as_posix():sha(p) for p in protected};save(folder/'protected-files.json',hashes)
 selection=read(run/'heads/selection.json')
 for name,digest in selection['checkpoint_sha256'].items():assert sha(run/'heads'/name)==digest
 for relative,digest in read(run/'identity.json')['source_sha256'].items():assert sha(run/'sources'/relative)==digest,'Original source snapshot changed'
 for name in ('evaluate_test.py','run_pipeline.py','resume_test.py','test_score_adapter.py'):
  target=folder/'sources'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'training/cliprank'/name,target)
 command=[sys.executable,'-X','utf8',str(ROOT/'training/cliprank/evaluate_test.py'),'--manifest',str(run/'manifest.json'),'--features',str(run/'features'),'--heads',str(run/'heads'),'--output-dir',str(run/'test'),'--resume-baseline-only']
 save(folder/'amendment.json',{'reason':'Original scorer uses human-readable dimension keys; head uses snake_case. Repair only schema conversion and resume cached original-scorer comparison. No retraining, checkpoint selection or repeated head test inference.','command':command,'started':now(),'source_sha256':{p.name:sha(p) for p in (folder/'sources').glob('*.py')}})
 status.update(status='running',stage='test_recovery',recovery_started=now(),recovery_pid=os.getpid());save(run/'status.json',status)
 try:
  env=os.environ.copy();env.update(PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  with (folder/'test-recovery.log').open('x',encoding='utf-8') as log:
   p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);status['recovery_child_pid']=p.pid;save(run/'status.json',status);code=p.wait()
  assert code==0,'Recovery evaluation failed'
  assert all(sha(run/name)==digest for name,digest in hashes.items()),'Protected training/test artifacts changed'
  save(folder/'verification.json',{'protected_files_unchanged':True,'finished':now(),'files':len(hashes)})
  status.update(status='completed',stage='test_recovery',finished=now());save(run/'status.json',status);publish(run);status.update(status='published',published=now());save(run/'status.json',status)
 except Exception as error:status.update(status='needs_attention',error=str(error),stopped=now());save(run/'status.json',status);(folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
