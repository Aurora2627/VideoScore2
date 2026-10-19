"""Validate fixed real videos without loading the model or using their labels."""
import argparse,hashlib,json
from pathlib import Path
from infer_cuda import read_video

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 data=json.loads(args.manifest.read_text(encoding='utf-8'))
 report={'status':'passed','scope':'input decoding and uniform sampling only; no model inference','samples':[]}
 for sample in data['samples']:
  video=Path(sample['video_path'])
  assert hashlib.sha256(video.read_bytes()).hexdigest()==sample['video_sha256']
  frames,info=read_video(video,4,224)
  assert frames.ndim==4 and frames.shape[0]==4 and frames.shape[-1]==3
  assert frames.shape[1]<=224 and frames.shape[2]<=224
  assert str(frames.dtype)=='uint8'
  assert len(set(info['frame_indices']))==4 and info['effective_fps']>0
  report['samples'].append({'video_name':sample['video_name'],'video_sha256':sample['video_sha256'],'frame_shape':list(frames.shape),'sampling':info})
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
