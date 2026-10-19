"""Run a fixed diagnostic subset with saved inputs and independently compared labels."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
 p.add_argument('--revision',required=True);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--count',type=int,default=1);p.add_argument('--frames',type=int,default=4);p.add_argument('--max-new-tokens',type=int,default=1024)
 args=p.parse_args()
 data=json.loads(args.manifest.read_text(encoding='utf-8'))
 if not 1<=args.count<=len(data['samples']):p.error('Count exceeds manifest')
 args.output_dir.mkdir(parents=True,exist_ok=False)
 shutil.copy2(__file__,args.output_dir/'diagnostic_source.py')
 shutil.copy2(args.manifest,args.output_dir/'subset_manifest.json')
 env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUTF8='1')
 report={'purpose':'diagnostic_only_not_full_benchmark','created':datetime.datetime.now().isoformat(),'samples':[]}
 failed=False
 for index,sample in enumerate(data['samples'][:args.count]):
  path=Path(sample['video_path'])
  if hashlib.sha256(path.read_bytes()).hexdigest()!=sample['video_sha256']:raise ValueError('Input video hash changed')
  out=args.output_dir/(str(index)+'_'+sample['video_name'])
  cmd=[sys.executable,str(Path(__file__).with_name('infer_cuda.py')),'--model',str(args.model),'--revision',args.revision,'--video',str(path),'--prompt',sample['prompt'],'--frames',str(args.frames),'--max-side','224','--max-new-tokens',str(args.max_new_tokens),'--output-dir',str(out),'--local-files-only']
  # Ground-truth scores and teacher explanations are never passed to the model.
  with (args.output_dir/(sample['video_name']+'.log')).open('w',encoding='utf-8') as log:
   proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  result=json.loads((out/'result.json').read_text(encoding='utf-8')) if (out/'result.json').exists() else {'status':'failed_before_result'}
  row={'video_name':sample['video_name'],'exit':proc.returncode,'status':result['status'],'ground_truth':sample['ground_truth'],'predictions':result.get('scores'),'valid':result.get('valid',False)}
  if row['valid']:
   row['absolute_error']={k:abs(v-float(sample['ground_truth'][k])) for k,v in row['predictions'].items()}
  report['samples'].append(row)
  (args.output_dir/'diagnostic-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  print('SAMPLE_FINISHED',sample['video_name'],result['status'],flush=True)
  if proc.returncode:failed=True;break
 print('DIAGNOSTIC_FINISHED',str(args.output_dir),flush=True)
 return 1 if failed else 0
if __name__=='__main__':raise SystemExit(main())
