"""Evaluate the validation-selected head once and compare the original scorer on identical test videos."""
import argparse,collections,hashlib,itertools,json,sys,time
from pathlib import Path
import numpy as np
import torch
from prepare_data import LABELS,canonical,read,save,sha
from train_heads import ScoreHead,load_features,labels,evaluate
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'application'))
from run_batch import ResidentScorer
REV='09a2732cb64fa566a1f332f978368292ce5c295c'
def paired_interval(head,baseline,target,rows):
 errors=np.abs(head-target).mean(1)-np.abs(baseline-target).mean(1);groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[canonical(r['prompt'])].append(i)
 group_errors=np.array([errors[ids].mean() for ids in groups.values()]);rng=np.random.default_rng(20261022)
 means=np.array([rng.choice(group_errors,len(group_errors),replace=True).mean() for _ in range(5000)])
 return {'estimand':'Prompt-macro mean MAE difference: head minus original; negative favors head','groups':len(group_errors),'observed':float(group_errors.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'bootstrap_samples':5000,'seed':20261022}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--heads',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--model',default='D:/career-projects/models/VideoScore2/'+REV);ap.add_argument('--revision',default=REV);a=ap.parse_args();assert a.revision==REV
 a.output_dir.mkdir(parents=True,exist_ok=True);selection=read(a.heads/'selection.json');assert selection['manifest_sha256']==sha(a.manifest) and selection['feature_identity_sha256']==sha(a.features/'identity.json')
 for name,digest in selection['checkpoint_sha256'].items():assert sha(a.heads/name)==digest,'Checkpoint changed after selection'
 rows=[r for r in read(a.manifest)['samples'] if r['split']=='test'];assert rows and all(r['source_split']=='official_test' for r in rows)
 normalization=torch.load(a.heads/'normalization.pt',map_location='cpu',weights_only=True);raw=load_features(rows,a.features);x=((raw-normalization['mean'])/normalization['std']).cuda();predictions=[];states=[]
 for seed in selection['seeds']:
  checkpoint=torch.load(a.heads/(selection['selected_variant']+'_'+str(seed)+'.pt'),map_location='cpu',weights_only=True);model=ScoreHead(checkpoint['input_dim'],checkpoint['variant']).cuda();model.load_state_dict(checkpoint['state_dict']);model.eval()
  with torch.inference_mode():predictions.append(model(x).cpu())
  states.append(checkpoint)
 head=torch.stack(predictions).mean(0).numpy();target=labels(rows).numpy();head_metrics=evaluate(head,target,rows)
 save(a.output_dir/'head-test-predictions.json',{'selection_sha256':sha(a.heads/'selection.json'),'samples':[{'video_name':r['video_name'],'scores':dict(zip(LABELS,p.tolist()))} for r,p in zip(rows,head)]})
 save(a.output_dir/'head-test-metrics.json',head_metrics)
 torch.save({'models':states,'normalization':normalization,'model_revision':REV,'frames':4,'max_side':448,'pooling':'final_non_padding_token_before_answer','selection_sha256':sha(a.heads/'selection.json'),'feature_identity':read(a.features/'identity.json')},a.heads/'selected-ensemble.pt')
 del model,x,raw;torch.cuda.empty_cache()
 a.frames=4;a.max_side=448;a.max_new_tokens=1024;engine=ResidentScorer(a);baseline=[];valid_rows=[];valid_indices=[];failures=[]
 for index,row in enumerate(rows):
  folder=a.output_dir/'original-scorer'/row['video_name'];folder.mkdir(parents=True,exist_ok=True);p=folder/'result.json'
  if p.exists():r=read(p);assert r['video_sha256']==row['video_sha256']
  else:
   safe={k:row[k] for k in ('video_name','video_path','video_sha256','prompt')};r=engine.score(safe,folder);save(p,r)
  if r.get('valid'):baseline.append([r['scores'][k] for k in LABELS]);valid_rows.append(row);valid_indices.append(index)
  else:failures.append({'video_name':row['video_name'],'status':r.get('status'),'reason':r.get('reason')})
  save(a.output_dir/'progress.json',{'stage':'original_scorer_test','finished':index+1,'total':len(rows),'valid':len(baseline)});print('ORIGINAL_TEST_SCORED',index+1,len(rows),bool(r.get('valid')),flush=True)
 metrics={'head_all_test':head_metrics,'selected_variant':selection['selected_variant'],'seeds':selection['seeds'],'test_count':len(rows),'baseline_valid_count':len(baseline),'baseline_failures':failures,'scope':'Project-specific head training; pretrained backbone was already trained on official_train. Test is an unused official_test subset, not a new external dataset.'}
 mean=np.tile(np.asarray(selection['train_label_mean']),(len(rows),1));metrics['train_mean_baseline']=evaluate(mean,target,rows);metrics['constant_three_baseline']=evaluate(np.full_like(target,3),target,rows)
 if baseline:
  original=np.array(baseline);selected_target=target[valid_indices];metrics['original_on_valid_test']=evaluate(original,selected_target,valid_rows);metrics['head_on_same_valid_test']=evaluate(head[valid_indices],selected_target,valid_rows);metrics['prompt_bootstrap_difference']=paired_interval(head[valid_indices],original,selected_target,valid_rows)
 save(a.output_dir/'comparison.json',metrics);print('TEST_COMPARISON_COMPLETE',flush=True)
if __name__=='__main__':main()
