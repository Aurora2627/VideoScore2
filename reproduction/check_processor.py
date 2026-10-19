"""Check actual Transformers video inputs without loading model weights."""
import argparse,json
from pathlib import Path
from infer_cuda import read_video,official_prompt

def main():
 from transformers import AutoProcessor
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 sample=json.loads(args.manifest.read_text(encoding='utf-8'))['samples'][0]
 frames,info=read_video(Path(sample['video_path']),4,224)
 processor=AutoProcessor.from_pretrained(str(args.model),min_pixels=56*56,max_pixels=224**2,trust_remote_code=False,local_files_only=True)
 messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':official_prompt(sample['prompt'])}]}]
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 batch=processor(text=[text],videos=[frames],fps=[info['effective_fps']],padding=True,return_tensors='pt')
 assert 'pixel_values_videos' in batch and batch['input_ids'].shape[0]==1
 report={'status':'passed','scope':'real Transformers preprocessing only; model weights not loaded','processor_class':type(processor).__name__,'video_name':sample['video_name'],'input_shapes':{k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')},'video_grid_thw':batch['video_grid_thw'].tolist(),'second_per_grid_ts':batch['second_per_grid_ts'].tolist() if hasattr(batch.get('second_per_grid_ts'),'tolist') else batch.get('second_per_grid_ts'),'effective_fps':info['effective_fps']}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
