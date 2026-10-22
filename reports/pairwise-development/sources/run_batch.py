"""ClipRank: resident CUDA scoring, grouped selection and auditable reports."""
import argparse,datetime,gc,hashlib,html,importlib.metadata,json,os,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction'))
from infer_cuda import read_video,official_prompt,classify_output
from selection import select_candidates,LABELS

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def preflight(data):
    names=set()
    if not data.get('samples'):raise ValueError('Empty candidate manifest')
    for item in data['samples']:
        name=item['video_name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or name in names:raise ValueError('Unsafe or duplicate video_name')
        names.add(name)
        if not isinstance(item['prompt'],str) or not item['prompt'].strip():raise ValueError('Original generation prompt required')
        if digest(item['video_path'])!=item['video_sha256']:raise ValueError('Video checksum mismatch: '+name)

class ResidentScorer:
    def __init__(self,args):
        import torch
        from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration,set_seed
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        self.torch=torch;self.args=args;self.set_seed=set_seed;set_seed(42)
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        common=dict(revision=args.revision,local_files_only=True,trust_remote_code=False)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        self.processor=AutoProcessor.from_pretrained(args.model,min_pixels=56*56,max_pixels=args.max_side**2,**common)
        start=time.perf_counter()
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,quantization_config=quant,torch_dtype=dtype,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True,**common).eval()
        torch.cuda.synchronize();self.load_seconds=time.perf_counter()-start
        self.info={'gpu':torch.cuda.get_device_name(0),'compute_dtype':str(dtype),'model_load_count':1,'model_load_seconds':self.load_seconds,'device_map':{k:str(v) for k,v in self.model.hf_device_map.items()},'environment':{k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate','bitsandbytes','numpy','opencv-python-headless']}}
    def score(self,item,directory):
        torch=self.torch;self.set_seed(42);inputs=generated=frames=None
        try:
            prompt=official_prompt(item['prompt']);(directory/'prompt.txt').write_text(prompt,encoding='utf-8')
            frames,sampling=read_video(Path(item['video_path']),self.args.frames,self.args.max_side)
            messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}]
            text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(text=[text],videos=[frames],fps=[sampling['effective_fps']],padding=True,return_tensors='pt').to('cuda:0')
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():generated=self.model.generate(**inputs,max_new_tokens=self.args.max_new_tokens,do_sample=False,use_cache=True)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            ids=generated[0,inputs['input_ids'].shape[1]:].tolist()
            raw=self.processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            (directory/'raw_output.txt').write_text(raw,encoding='utf-8')
            eos=self.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
            result={'video_name':item['video_name'],'video_sha256':item['video_sha256'],'video_sampling':sampling,'generation_seconds':seconds,'generated_token_ids':ids,'generated_tokens':len(ids),'eos_reached':bool(ids and ids[-1] in eos),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'input_shapes':{k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}}
            result.update(classify_output(raw,result['eos_reached']))
            return result
        finally:
            inputs=generated=frames=None;gc.collect();torch.cuda.empty_cache()

def reports(out,data,results,policy,run):
    decisions=select_candidates(data['samples'],results,policy);save(out/'selection.json',decisions)
    valid=[r for r in results.values() if r.get('valid')];errors={k:[] for k in LABELS}
    for item in data['samples']:
        result=results.get(item['video_name'],{})
        if result.get('valid') and 'ground_truth' in item:
            for k in LABELS:errors[k].append(abs(result['scores'][k]-float(item['ground_truth'][k])))
    metrics={'attempted':len(results),'manifest_count':len(data['samples']),'valid_outputs':len(valid),'valid_output_rate':len(valid)/len(results) if results else None,'mae':{k:sum(v)/len(v) if v else None for k,v in errors.items()},'labeled_counts':{k:len(v) for k,v in errors.items()},'scope':'diagnostic only; no held-out model-quality or production claim'}
    save(out/'metrics.json',metrics);save(out/'run.json',run)
    body=['<!doctype html><meta charset="utf-8"><title>ClipRank | AIGC 视频候选评估</title><style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#172033}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #d8dfe8;padding:10px;text-align:left}h2{margin-top:36px}code{background:#eef2f6;padding:3px}a{color:#245bba}</style><h1>ClipRank · AIGC 短视频质量评估与候选筛选</h1><p>按同一提示词分组。加权评分为可配置规则，并非概率；低分或失败项进入人工复核。该报告不自动发布视频。</p>', '<p>有效输出：'+str(len(valid))+'/'+str(len(data['samples']))+'；模型加载次数：'+str(run.get('model_load_count','pending'))+'</p>']
    for group in decisions['groups']:
        body+=['<h2>需求 '+group['prompt_sha256'][:10]+'</h2><p>'+html.escape(group['prompt'])+'</p><p>'+('具备多个候选' if group['comparison_available'] else '只有一个候选，暂不排名')+'</p><table><tr><th>视频</th><th>画质 / 对齐 / 物理</th><th>规则总分</th><th>排名</th><th>处理建议</th><th>原因与原始输出</th></tr>']
        for row in group['candidates']:
            name=row['video_name'];score=' / '.join(str(row['scores'].get(k,'—')) for k in LABELS) if isinstance(row['scores'],dict) else '—'
            body+=['<tr><td>'+html.escape(name)+'</td><td>'+score+'</td><td>'+str(row['weighted_score'])+'</td><td>'+str(row['rank'])+'</td><td>'+html.escape(row['decision'])+'</td><td>'+html.escape('; '.join(row['review_reasons']))+' <a href="'+name+'/raw_output.txt">查看分析</a></td></tr>']
        body+=['</table>']
    body+=['<p>样本、配置、代码版本、原始回答及评分记录均保存在本报告目录。模型来源与许可见项目说明。样本量和输入压缩会限制结论。</p>']
    (out/'report.html').write_text(chr(10).join(body),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','model','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--revision',required=True);p.add_argument('--frames',type=int,default=4);p.add_argument('--max-side',type=int,default=448);p.add_argument('--max-new-tokens',type=int,default=1024);p.add_argument('--policy',type=Path,default=Path(__file__).with_name('policy.json'))
    a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',a.revision):p.error('Pin model revision')
    if a.frames<2 or a.frames%2 or a.max_side<56 or a.max_new_tokens<1:p.error('Invalid input limits')
    data=read(a.manifest);preflight(data);policy=read(a.policy)
    from selection import validate_policy
    validate_policy(policy);out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    save(out/'manifest.json',data);save(out/'policy.json',policy)
    sources=out/'sources';sources.mkdir()
    for source in [*Path(__file__).parent.glob('*.py'),ROOT/'reproduction/infer_cuda.py',ROOT/'vs2_inference.py']:shutil.copy2(source,sources/source.name)
    run={'status':'started','started':datetime.datetime.now().isoformat(),'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,encoding='utf-8'),'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'source_sha256':{x.name:digest(x) for x in sources.iterdir()},'model_load_count':0}
    save(out/'run.json',run);results={};start=time.perf_counter()
    try:
        engine=ResidentScorer(a);run.update(engine.info);save(out/'run.json',run)
        print('MODEL_READY',json.dumps(engine.info),flush=True)
        for item in data['samples']:
            directory=out/item['video_name'];directory.mkdir()
            try:result=engine.score(item,directory)
            except Exception as error:
                result={'status':'failed','valid':False,'eos_reached':False,'error_type':type(error).__name__,'error':str(error)}
                (directory/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
                if isinstance(error,engine.torch.cuda.OutOfMemoryError):
                    results[item['video_name']]=result;save(directory/'result.json',result);raise
            results[item['video_name']]=result;save(directory/'result.json',result)
            print('CLIP_FINISHED',item['video_name'],result['status'],flush=True)
            reports(out,data,results,policy,run)
        run['status']='completed' if all(r.get('valid') for r in results.values()) else 'completed_with_errors'
    except Exception as error:
        run.update(status='failed',error_type=type(error).__name__,error=str(error));(out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
    run.update(finished=datetime.datetime.now().isoformat(),wall_seconds=time.perf_counter()-start)
    reports(out,data,results,policy,run);print('BATCH_FINISHED',str(out),run['status'],flush=True)
    return 0 if run['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
