"""Score a new video with the selected frozen-backbone head ensemble."""
import argparse,json
from pathlib import Path
import torch
from prepare_data import LABELS,sha
from extract_features import extract,ResidentScorer,REV
from train_heads import ScoreHead
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--video',type=Path,required=True);ap.add_argument('--prompt',required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--model',default='D:/career-projects/models/VideoScore2/'+REV);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 bundle=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert bundle['model_revision']==REV;a.revision=REV;a.frames=4;a.max_side=448;a.max_new_tokens=1024
 engine=ResidentScorer(a);engine.model.requires_grad_(False);item={'video_name':a.video.stem,'video_path':str(a.video),'video_sha256':sha(a.video),'prompt':a.prompt};feature,receipt=extract(engine,item);normalization=bundle['normalization'];x=((feature-normalization['mean'])/normalization['std']).unsqueeze(0).cuda();values=[]
 for checkpoint in bundle['models']:
  head=ScoreHead(checkpoint['input_dim'],checkpoint['variant']).cuda();head.load_state_dict(checkpoint['state_dict']);head.eval()
  with torch.inference_mode():values.append(head(x).cpu()[0])
 scores=torch.stack(values).mean(0);result={'video':str(a.video),'video_sha256':item['video_sha256'],'scores':dict(zip(LABELS,scores.tolist())),'checkpoint_sha256':sha(a.checkpoint),'feature_receipt':receipt,'note':'Continuous scores in [1,5], not calibrated probabilities; no generated rationale'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result['scores'],ensure_ascii=False))
if __name__=='__main__':main()
