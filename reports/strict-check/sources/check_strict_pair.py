"""Fixed-resolution supplementary check, queued after the original experiment."""
import argparse,datetime,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'application'))
from run_batch import ResidentScorer,preflight
from quality_metrics import evaluate
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('manifest','after-session','model','output-dir'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--revision',required=True);a=p.parse_args()
 print('WAITING_FOR_ORIGINAL_EXPERIMENT',str(a.after_session),flush=True)
 deadline=time.monotonic()+3600
 while read(a.after_session)['status']=='started':
  if time.monotonic()>deadline:raise TimeoutError('Original experiment still running')
  time.sleep(10)
 parent=read(a.after_session);assert parent['status']=='completed';assert parent['recommendation']['chosen_on_development']=='more_detail'
 data=read(a.manifest);preflight(data);out=a.output_dir;out.mkdir(parents=True,exist_ok=False);sources=out/'sources';sources.mkdir()
 for source in [Path(__file__),Path(__file__).with_name('quality_metrics.py'),ROOT/'application/run_batch.py',ROOT/'application/selection.py',ROOT/'reproduction/infer_cuda.py',ROOT/'vs2_inference.py']:shutil.copy2(source,sources/source.name)
 policy=read(ROOT/'application/policy.json');save(out/'manifest.json',data);save(out/'protocol.json',read(Path(__file__).with_name('strict_check_protocol.json')));save(out/'policy.json',policy)
 session={'status':'started','started':datetime.datetime.now().isoformat(),'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip(),'source_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sources.iterdir()},'parent_session_sha256':hashlib.sha256(a.after_session.read_bytes()).hexdigest(),'configuration_selection_from_fresh_scores':False,'phases':{}};save(out/'session.json',session);start=time.perf_counter()
 try:
  a.frames=4;a.max_side=448;a.max_new_tokens=1024;engine=ResidentScorer(a);session.update(engine.info);save(out/'session.json',session)
  from transformers import AutoProcessor
  for name,side in [('baseline',224),('more_detail',448)]:
   directory=out/name;directory.mkdir();a.max_side=side;engine.processor=AutoProcessor.from_pretrained(a.model,min_pixels=56*56,max_pixels=side**2,revision=a.revision,local_files_only=True,trust_remote_code=False);results={}
   for index,item in enumerate(data['samples'],1):
    folder=directory/item['video_name'];folder.mkdir()
    try:result=engine.score({k:item[k] for k in ('video_name','prompt','video_path','video_sha256')},folder)
    except Exception as error:
     result={'status':'failed','valid':False,'eos_reached':False,'error_type':type(error).__name__,'error':str(error)};(folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
     if 'device-side assert' in str(error).lower() or 'illegal memory access' in str(error).lower():raise
    result['configuration']={'frames':4,'max_side':side};results[item['video_name']]=result;save(folder/'result.json',result);session['progress']={'phase':name,'completed':index,'total':len(data['samples']),'video_name':item['video_name']};save(out/'session.json',session);print('STRICT_CHECK_CLIP',name,index,item['video_name'],result['status'],flush=True)
   metrics=evaluate(data['samples'],results,policy['weights']);session['phases'][name]={'configuration':{'frames':4,'max_side':side},'metrics':metrics};save(directory/'metrics.json',metrics);save(out/'session.json',session);print('STRICT_CHECK_PHASE',name,json.dumps(metrics),flush=True)
  session['status']='completed'
 except Exception as error:session.update(status='failed',error=str(error));(out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
 session.update(finished=datetime.datetime.now().isoformat(),wall_seconds=time.perf_counter()-start);save(out/'session.json',session);print('STRICT_CHECK_FINISHED',str(out),session['status'],flush=True)
 return 0 if session['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
