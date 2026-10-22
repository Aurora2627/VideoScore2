"""Cache label-free last-token features from a frozen quantized multimodal backbone."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'application'))
from run_batch import ResidentScorer
from infer_cuda import read_video,official_prompt
from prepare_data import read,save,sha
REV='09a2732cb64fa566a1f332f978368292ce5c295c'
def extract(engine,item):
 torch=engine.torch;frames=inputs=output=None
 try:
  assert sha(item['video_path'])==item['video_sha256']
  frames,sampling=read_video(Path(item['video_path']),4,448)
  prompt=official_prompt(item['prompt'])
  messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}]
  text=engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  inputs=engine.processor(text=[text],videos=[frames],fps=[sampling['effective_fps']],padding=True,return_tensors='pt').to('cuda:0')
  assert inputs['video_grid_thw'].shape[0]==1
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():
   output=engine.model.model(**inputs,use_cache=False,return_dict=True)
   hidden=output.last_hidden_state
   position=int(inputs['attention_mask'][0].sum().item())-1
   feature=hidden[0,position].detach().float().cpu().clone()
  torch.cuda.synchronize();seconds=time.perf_counter()-start
  assert feature.ndim==1 and torch.isfinite(feature).all()
  return feature,{'video_name':item['video_name'],'video_sha256':item['video_sha256'],'generation_prompt_sha256':hashlib.sha256(item['prompt'].encode()).hexdigest(),'feature_dim':feature.numel(),'pooling':'final non-padding token before any assistant answer; no labels or rationales provided','sampling':sampling,'video_grid_thw':inputs['video_grid_thw'].detach().cpu().tolist(),'input_shapes':{k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
 finally:
  frames=inputs=output=None;gc.collect();torch.cuda.empty_cache()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--model',default='D:/career-projects/models/VideoScore2/'+REV);ap.add_argument('--revision',default=REV);ap.add_argument('--limit',type=int);a=ap.parse_args();assert a.revision==REV
 import torch
 assert torch.cuda.is_available();a.frames=4;a.max_side=448;a.max_new_tokens=1024
 a.output_dir.mkdir(parents=True,exist_ok=True)
 identity={'manifest_sha256':sha(a.manifest),'model_revision':a.revision,'model':str(Path(a.model).resolve()),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'application/run_batch.py',ROOT/'reproduction/infer_cuda.py',ROOT/'vs2_inference.py']},'frames':4,'max_side':448,'quantization':'NF4 double quant BF16','pooling':'final_non_padding_token_before_answer'}
 path=a.output_dir/'identity.json'
 if path.exists():assert read(path)==identity,'Feature identity mismatch'
 else:save(path,identity)
 items=read(a.manifest)['samples'];items=items[:a.limit] if a.limit else items
 engine=ResidentScorer(a);engine.model.requires_grad_(False);assert not any(p.requires_grad for p in engine.model.parameters())
 info=dict(engine.info,frozen_parameters=sum(p.numel() for p in engine.model.parameters()),trainable_backbone_parameters=0,total=len(items));save(a.output_dir/'session.json',info)
 for index,item in enumerate(items,1):
  # Explicit allowlist: annotations and dataset rationales never enter extraction.
  safe={k:item[k] for k in ('video_name','video_path','video_sha256','prompt')};folder=a.output_dir/safe['video_name'];folder.mkdir(exist_ok=True)
  if (folder/'feature.pt').exists() and (folder/'feature.json').exists():
   receipt=read(folder/'feature.json');assert receipt['video_sha256']==safe['video_sha256'] and receipt['feature_sha256']==sha(folder/'feature.pt')
  else:
   feature,receipt=extract(engine,safe);tmp=folder/'feature.pt.tmp';torch.save(feature,tmp);tmp.replace(folder/'feature.pt');receipt['feature_sha256']=sha(folder/'feature.pt');save(folder/'feature.json',receipt)
  info.update(finished=index,last_video=safe['video_name'],last_seconds=receipt['seconds']);save(a.output_dir/'session.json',info)
  print('FEATURE_READY',index,len(items),safe['video_name'],round(receipt['seconds'],3),flush=True)
 info['status']='completed';save(a.output_dir/'session.json',info)
if __name__=='__main__':main()
