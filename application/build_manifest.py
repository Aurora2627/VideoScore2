"""Create a candidate manifest for videos generated from the same creative prompt."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prompt',required=True);p.add_argument('--video',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not a.prompt.strip():p.error('Original generation prompt is required')
 if a.output.exists():raise FileExistsError(a.output)
 paths=[p.resolve() for p in a.video]
 if len(set(paths))!=len(paths):p.error('Repeated video path')
 samples=[]
 for index,path in enumerate(paths,1):
  with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  samples.append({'video_name':'clip_'+format(index,'03d'),'original_filename':path.name,'video_path':str(path),'prompt':a.prompt,'video_sha256':digest})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'purpose':'User-provided creative candidates','samples':samples},indent=2,ensure_ascii=False),encoding='utf-8');print('MANIFEST_READY',str(a.output),len(samples))
if __name__=='__main__':main()
