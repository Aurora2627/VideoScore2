"""Train score heads on frozen video-language features with validation-only selection."""
import argparse,collections,copy,itertools,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from prepare_data import LABELS,canonical,read,save,sha
VARIANTS=('linear','mlp','mlp_rank')
SEEDS=(42,43,44)
class ScoreHead(nn.Module):
 def __init__(self,dim,variant):
  super().__init__();self.net=nn.Linear(dim,3) if variant=='linear' else nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,3))
 def forward(self,x):return 1+4*torch.sigmoid(self.net(x))
def normalize_train(x):return x.mean(0),x.std(0,unbiased=False).clamp_min(.05)
def ranking_loss(pred,target,prompts):
 n=len(prompts);same=torch.tensor([[a==b for b in prompts] for a in prompts],device=pred.device)
 upper=torch.triu(torch.ones(n,n,device=pred.device,dtype=torch.bool),diagonal=1);delta=target[:,None,:]-target[None,:,:]
 mask=(same&upper)[:,:,None] & (delta!=0)
 if not bool(mask.any()):return pred.sum()*0
 return torch.nn.functional.softplus(-delta.sign()*(pred[:,None,:]-pred[None,:,:]))[mask].mean()
def load_features(rows,folder):
 values=[]
 for row in rows:
  p=folder/row['video_name'];info=read(p/'feature.json');assert info['video_sha256']==row['video_sha256'] and info['feature_sha256']==sha(p/'feature.pt')
  value=torch.load(p/'feature.pt',map_location='cpu',weights_only=True);assert value.ndim==1 and torch.isfinite(value).all();values.append(value)
 return torch.stack(values)
def labels(rows):return torch.tensor([[r['ground_truth'][k] for k in LABELS] for r in rows],dtype=torch.float32)
def correlations(pred,target):
 from scipy.stats import spearmanr,pearsonr
 result={}
 for j,k in enumerate(LABELS):
  x=pred[:,j];y=target[:,j];nonconstant=np.std(x)>1e-10 and np.std(y)>1e-10
  result[k]={'spearman':float(spearmanr(x,y).statistic) if nonconstant else None,'pearson':float(pearsonr(x,y).statistic) if nonconstant else None}
 return result
def evaluate(pred,target,rows):
 pred=np.asarray(pred,dtype=float);target=np.asarray(target,dtype=float);assert pred.shape==target.shape==(len(rows),3) and np.isfinite(pred).all()
 errors=np.abs(pred-target);groups=collections.defaultdict(list);weights=np.array([.3,.4,.3]);comparisons=[];ties=0
 for i,row in enumerate(rows):groups[canonical(row['prompt'])].append(i)
 for group,indices in groups.items():
  for a,b in itertools.combinations(indices,2):
   dy=float((target[a]-target[b])@weights);dp=float((pred[a]-pred[b])@weights)
   if abs(dy)<1e-8:ties+=1;continue
   comparisons.append({'videos':[rows[a]['video_name'],rows[b]['video_name']],'label_margin':dy,'predicted_margin':dp,'correct':bool(dp*dy>0 and abs(dp)>1e-8)})
 return {'samples':len(rows),'mean_mae':float(errors.mean()),'mae':dict(zip(LABELS,errors.mean(0).tolist())),'correlation':correlations(pred,target),'ranking_correct':sum(r['correct'] for r in comparisons),'ranking_pairs':len(comparisons),'annotation_ties_excluded':ties,'ranking_accuracy':sum(r['correct'] for r in comparisons)/len(comparisons) if comparisons else None,'pairs':comparisons}
def train_one(x,y,xv,yv,rows,variant,seed,out,epochs=100,patience=15):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 model=ScoreHead(x.shape[1],variant).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 groups=collections.defaultdict(list)
 for i,row in enumerate(rows):groups[row['group_id']].append(i)
 keys=list(groups);best=float('inf');best_state=None;history=[];stale=0;initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rng=random.Random(seed)
 for epoch in range(1,epochs+1):
  model.train();rng.shuffle(keys);loss_sum=reg_sum=rank_sum=0.;seen=0
  for start in range(0,len(keys),8):
   ids=[i for group in keys[start:start+8] for i in groups[group]];idx=torch.tensor(ids,device='cuda');p=model(x[idx]);reg=nn.functional.smooth_l1_loss(p,y[idx],beta=.5);rank=ranking_loss(p,y[idx],[canonical(rows[i]['prompt']) for i in ids]);loss=reg+(.1*rank if variant=='mlp_rank' else 0)
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();n=len(ids);seen+=n;loss_sum+=float(loss.detach())*n;reg_sum+=float(reg.detach())*n;rank_sum+=float(rank.detach())*n
  model.eval()
  with torch.inference_mode():pv=model(xv);val=float((pv-yv).abs().mean())
  history.append({'epoch':epoch,'train_loss':loss_sum/seen,'train_regression_loss':reg_sum/seen,'train_ranking_loss':rank_sum/seen,'validation_mae':val})
  if val<best-1e-6:best=val;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
  else:stale+=1
  if stale>=patience:break
 assert best_state is not None and any(not torch.equal(initial[k],v) for k,v in best_state.items()),'No actual parameter update'
 checkpoint={'state_dict':best_state,'variant':variant,'seed':seed,'input_dim':x.shape[1],'best_epoch':best_epoch,'validation_mae':best,'trainable_parameters':sum(p.numel() for p in model.parameters()),'backbone_trainable_parameters':0}
 path=out/(variant+'_'+str(seed)+'.pt');torch.save(checkpoint,path);save(out/(variant+'_'+str(seed)+'-history.json'),history);model.load_state_dict(best_state);model.eval()
 with torch.inference_mode():pv=model(xv).cpu()
 print('HEAD_TRAINED',variant,seed,'epoch',best_epoch,'validation_mae',round(best,5),flush=True)
 return checkpoint,pv

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();assert torch.cuda.is_available();torch.set_num_threads(4)
 a.output_dir.mkdir(parents=True,exist_ok=False);all_rows=read(a.manifest)['samples'];train=[r for r in all_rows if r['split']=='train'];val=[r for r in all_rows if r['split']=='validation'];assert train and val
 assert all(r['source_split']=='official_train' for r in train+val)
 assert not ({canonical(r['prompt']) for r in train}&{canonical(r['prompt']) for r in val})
 # No test features or test labels used during fitting, early stopping or model selection.
 xt=load_features(train,a.features);xv=load_features(val,a.features);mean,std=normalize_train(xt);torch.save({'mean':mean,'std':std},a.output_dir/'normalization.pt')
 x=((xt-mean)/std).cuda();xv=((xv-mean)/std).cuda();y=labels(train).cuda();yv=labels(val).cuda();candidates=[]
 for variant in VARIANTS:
  predictions=[];trials=[]
  for seed in SEEDS:
   checkpoint,pv=train_one(x,y,xv,yv,train,variant,seed,a.output_dir);predictions.append(pv);trials.append({k:v for k,v in checkpoint.items() if k!='state_dict'})
  ensemble=torch.stack(predictions).mean(0);metrics=evaluate(ensemble.numpy(),yv.cpu().numpy(),val);candidates.append({'variant':variant,'seeds':list(SEEDS),'validation':metrics,'trials':trials})
 selected=min(candidates,key=lambda c:c['validation']['mean_mae'])
 save(a.output_dir/'selection.json',{'selected_variant':selected['variant'],'selection_rule':'Lowest 3-seed ensemble validation mean MAE; fixed variant order breaks ties; no test access','seeds':list(SEEDS),'candidates':candidates,'manifest_sha256':sha(a.manifest),'feature_identity_sha256':sha(a.features/'identity.json'),'training_source_sha256':sha(__file__),'normalization_sha256':sha(a.output_dir/'normalization.pt'),'train_count':len(train),'validation_count':len(val),'train_label_mean':y.mean(0).cpu().tolist(),'checkpoint_sha256':{p.name:sha(p) for p in a.output_dir.glob('*.pt')}})
 print('MODEL_SELECTED',selected['variant'],flush=True)
if __name__=='__main__':main()
