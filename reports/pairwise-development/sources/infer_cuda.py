"""Memory-limited CUDA inference adaptation; not paper-metric reproduction."""
import argparse, ast, datetime, hashlib, importlib.metadata, json, re, shutil
import string, subprocess, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('visual quality', 'text-to-video alignment', 'physical/common-sense consistency')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def parse_scores(text):
    if re.search(r'<think>', text, re.I):
        if not re.search(r'</think>', text, re.I):
            return {'scores': {label: None for label in LABELS}, 'valid': False}
        text = re.split(r'</think>', text, flags=re.I)[-1]
    scores = {}
    for label in LABELS:
        found = re.findall(re.escape(label) + r'(?:\s+[–—-]\s+[^:\n;]{1,160})?\s*:\s*(\d+(?:\.\d+)?)', text, re.I)
        scores[label] = int(found[0]) if len(found) == 1 and found[0] in ('1','2','3','4','5') else None
    return {'scores': scores, 'valid': all(v is not None for v in scores.values())}

def classify_output(text, eos_reached):
    parsed = parse_scores(text)
    parsed['scores_valid'] = parsed['valid']
    parsed['valid'] = parsed['valid'] and eos_reached
    parsed['status'] = 'truncated_output' if not eos_reached else ('completed' if parsed['valid'] else 'invalid_output')
    return parsed

def official_prompt(prompt):
    source = ROOT / 'vs2_inference.py'
    for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VS2_QUERY_TEMPLATE' for t in node.targets):
            return string.Template(ast.literal_eval(node.value.args[0])).substitute(t2v_prompt=prompt)
    raise ValueError('Official prompt constant not found')

def read_video(path, count, max_side):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total < 2 or fps <= 0: raise ValueError('Video needs >=2 frames and valid FPS')
        if count < 2 or count % 2 or total < count: raise ValueError('Use an even frame count <= total frames')
        indices = np.linspace(0,total-1,count,dtype=int).tolist()
        frames=[]
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok, frame = cap.read()
            if not ok: raise ValueError(f'Cannot decode frame {index}')
            h,w=frame.shape[:2]
            ratio=min(1.0,max_side/max(h,w))
            frame=cv2.resize(frame,(max(1,round(w*ratio)),max(1,round(h*ratio))))
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        effective_fps=(count-1)*fps/(total-1)
        return np.stack(frames), {'frame_indices':indices,'source_fps':fps,'total_frames':total,'effective_fps':effective_fps}
    finally: cap.release()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='TIGER-Lab/VideoScore2')
    p.add_argument('--revision',required=True,help='Immutable 40-character model commit')
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--prompt',required=True,help='Original text-to-video generation prompt')
    p.add_argument('--frames',type=int,default=4)
    p.add_argument('--max-side',type=int,default=224)
    p.add_argument('--max-new-tokens',type=int,default=1024)
    p.add_argument('--cache-dir',type=Path,default=Path('D:/career-projects/cache/huggingface/hub'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--local-files-only',action='store_true')
    args=p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.revision): p.error('Pin an immutable model commit')
    if not args.video.is_file(): p.error('Video does not exist')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    out=args.output_dir
    result={'status':'started','paper_reproduction':False,'adaptations':['4-bit NF4 double quantization','uniform capped frames','224-pixel default longest side','greedy decoding','hard scores only; no upstream floating-score equivalence claimed'],'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    def save(): (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, set_seed
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
        set_seed(42)
        result['environment']={k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate','bitsandbytes','numpy','opencv-python-headless']}
        result['gpu']=torch.cuda.get_device_name(0)
        result['video_sha256']=digest(args.video)
        result['source_sha256']=digest(Path(__file__))
        result['project_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        result['upstream_commit']='9e0fcbaeee0283b03efaaf8cd906f6381a808c75'
        result['git_status']=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
        shutil.copy2(__file__,out/'inference_source.py')
        shutil.copy2(ROOT/'vs2_inference.py',out/'upstream_inference_source.py')
        prompt=official_prompt(args.prompt)
        (out/'prompt.txt').write_text(prompt,encoding='utf-8')
        frames,info=read_video(args.video,args.frames,args.max_side)
        result['video_sampling']=info
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        result['compute_dtype']=str(dtype)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        common=dict(revision=args.revision,cache_dir=str(args.cache_dir),local_files_only=args.local_files_only,trust_remote_code=False)
        processor=AutoProcessor.from_pretrained(args.model,min_pixels=56*56,max_pixels=args.max_side**2,**common)
        start=time.perf_counter()
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,quantization_config=quant,torch_dtype=dtype,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True,**common).eval()
        result['load_seconds']=time.perf_counter()-start
        result['device_map']={k:str(v) for k,v in model.hf_device_map.items()}
        messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],videos=[frames],fps=[info['effective_fps']],padding=True,return_tensors='pt').to('cuda:0')
        result['input_shapes']={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=True)
        torch.cuda.synchronize()
        result['generation_seconds']=time.perf_counter()-start
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        ids=generated[0,inputs['input_ids'].shape[1]:].tolist()
        raw=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        (out/'raw_output.txt').write_text(raw,encoding='utf-8')
        result['generated_token_ids']=ids
        result['eos_reached']=bool(ids and ids[-1] in ([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id,int) else model.generation_config.eos_token_id or []))
        result.update(classify_output(raw, result['eos_reached']))
        save()
        print(json.dumps(result,indent=2))
        if not result['valid']: raise SystemExit(2)
    except Exception as e:
        result.update(status='failed',error_type=type(e).__name__,error=str(e))
        (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        save()
        raise

if __name__=='__main__': main()
