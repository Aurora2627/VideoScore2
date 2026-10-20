"""Resident-model input ablation with fixed development selection and disjoint final checks."""
import argparse,datetime,gc,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'application'))
from run_batch import ResidentScorer,preflight
from quality_metrics import evaluate,choose,promotion_gate

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('inputs-root','baseline-root','model','output-dir'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--revision',required=True);a=p.parse_args()
 protocol=read(Path(__file__).with_name('quality_protocol.json'));assert a.revision==protocol['fixed']['model_revision']
 development=read(a.inputs_root/'development.json');regression=read(a.inputs_root/'regression.json');fresh=read(a.inputs_root/'untouched-check.json')
 for data in [development,regression,fresh]:preflight(data)
 assert set(x['prompt'] for x in fresh['samples']).isdisjoint(x['prompt'] for x in development['samples']+regression['samples'])
 baseline_run=read(a.baseline_root/'run.json');assert all(baseline_run['configuration'][k]==v for k,v in {'frames':4,'max_side':224,'max_new_tokens':1024,'revision':a.revision}.items())
 policy=read(a.baseline_root/'policy.json');weights=policy['weights'];baseline={s['video_name']:read(a.baseline_root/s['video_name']/'result.json') for s in development['samples']+regression['samples']}
 for s in development['samples']+regression['samples']:assert baseline[s['video_name']]['video_sha256']==s['video_sha256']
 out=a.output_dir;out.mkdir(parents=True,exist_ok=False);sources=out/'sources';sources.mkdir()
 for path in [*Path(__file__).parent.glob('*.py'),ROOT/'application/run_batch.py',ROOT/'application/selection.py',ROOT/'reproduction/infer_cuda.py',ROOT/'vs2_inference.py']:shutil.copy2(path,sources/path.name)
 save(out/'protocol.json',protocol);save(out/'policy.json',policy)
 for name,data in [('development',development),('regression',regression),('untouched-check',fresh)]:save(out/(name+'-manifest.json'),data)
 cached=out/'baseline-cached';cached.mkdir()
 for name in baseline:
  folder=cached/name;folder.mkdir()
  for filename in ('result.json','raw_output.txt'):shutil.copy2(a.baseline_root/name/filename,folder/filename)
 session={'status':'started','started':datetime.datetime.now().isoformat(),'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,encoding='utf-8'),'source_sha256':{x.name:sha(x) for x in sources.iterdir()},'baseline_source_commit':baseline_run['project_commit'],'baseline_result_sha256':{n:sha(cached/n/'result.json') for n in baseline},'phases':{},'model_load_count':0}
 save(out/'session.json',session);start=time.perf_counter()
 try:
  a.frames=4;a.max_side=448;a.max_new_tokens=1024;engine=ResidentScorer(a);session.update(engine.info);save(out/'session.json',session)
  print('QUALITY_MODEL_READY',json.dumps(engine.info),flush=True)
  from transformers import AutoProcessor
  def phase(name,data,configuration):
   directory=out/name;directory.mkdir();a.frames=configuration['frames'];a.max_side=configuration['max_side']
   engine.processor=AutoProcessor.from_pretrained(a.model,min_pixels=56*56,max_pixels=a.max_side**2,revision=a.revision,local_files_only=True,trust_remote_code=False)
   save(directory/'configuration.json',{'frames':a.frames,'max_side':a.max_side,'max_new_tokens':1024,'revision':a.revision,'shared_model_session':str(out)})
   results={};phase_start=time.perf_counter();stop=False
   for index,item in enumerate(data['samples'],1):
    folder=directory/item['video_name'];folder.mkdir()
    if stop:result={'status':'skipped_after_oom','valid':False,'eos_reached':False}
    else:
     try:result=engine.score({k:item[k] for k in ('video_name','prompt','video_path','video_sha256')},folder)
     except Exception as error:
      result={'status':'failed','valid':False,'eos_reached':False,'error_type':type(error).__name__,'error':str(error)};(folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
      if 'device-side assert' in str(error).lower() or 'illegal memory access' in str(error).lower():raise
      stop=isinstance(error,engine.torch.cuda.OutOfMemoryError)
    result['configuration']={'frames':a.frames,'max_side':a.max_side};results[item['video_name']]=result;save(folder/'result.json',result)
    session['progress']={'phase':name,'completed':index,'total':len(data['samples']),'video_name':item['video_name'],'status':result['status']};save(out/'session.json',session)
    print('QUALITY_CLIP',name,index,len(data['samples']),item['video_name'],result['status'],flush=True)
   metrics=evaluate(data['samples'],results,weights);metrics['phase_wall_seconds']=time.perf_counter()-phase_start
   session['phases'][name]={'configuration':configuration,'metrics':metrics};save(directory/'metrics.json',metrics);save(out/'session.json',session)
   print('QUALITY_PHASE',name,json.dumps(metrics),flush=True);return metrics
  base_dev=evaluate(development['samples'],baseline,weights);base_reg=evaluate(regression['samples'],baseline,weights)
  save(out/'baseline-development-metrics.json',base_dev);save(out/'baseline-regression-metrics.json',base_reg)
  candidates={}
  for name,configuration in protocol['candidates'].items():candidates[name]=phase('development_'+name,development,configuration)
  chosen=choose(base_dev,candidates,protocol['candidates'])
  decision={'chosen_on_development':chosen,'baseline_development':base_dev,'candidate_development':candidates,'fresh_labels_used_for_choice':False,'policy_or_weights_changed':False}
  save(out/'selection-before-fresh-check.json',decision);print('QUALITY_CHOSEN',chosen,flush=True)
  if chosen!='baseline':
   configuration=protocol['candidates'][chosen];reg_metrics=phase('regression_'+chosen,regression,configuration)
   check_base=phase('fresh_baseline',fresh,protocol['baseline']);check_metrics=phase('fresh_'+chosen,fresh,configuration)
   gates=promotion_gate(candidates[chosen],base_reg,reg_metrics,check_base,check_metrics)
   decision.update(regression_baseline=base_reg,regression_candidate=reg_metrics,fresh_baseline=check_base,fresh_candidate=check_metrics,promotion_gates=gates,eligible_for_default=all(gates.values()),recommended_configuration=configuration if all(gates.values()) else protocol['baseline'])
  else:decision.update(eligible_for_default=False,recommended_configuration=protocol['baseline'],reason='Neither input change strictly improved development MAE with complete outputs; fresh check remains unused.')
  save(out/'recommendation.json',decision);session['status']='completed';session['recommendation']=decision
 except Exception as error:
  session.update(status='failed',error_type=type(error).__name__,error=str(error));(out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
 session.update(finished=datetime.datetime.now().isoformat(),wall_seconds=time.perf_counter()-start);save(out/'session.json',session)
 print('QUALITY_FINISHED',str(out),session['status'],flush=True)
 return 0 if session['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
