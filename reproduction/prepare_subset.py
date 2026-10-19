"""Fetch the first N fixed benchmark entries; annotations never enter model prompts."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--count',type=int,default=3);args=p.parse_args()
 root=args.data_dir
 rows=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
 provenance=json.loads((root/'metadata-provenance.json').read_text(encoding='utf-8'))
 assert hashlib.sha256((root/'metadata.json').read_bytes()).hexdigest()==provenance['sha256']
 if not 1<=args.count<=10:raise ValueError('Diagnostic subset limited to 1-10 samples')
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 manifest={'purpose':'diagnostic_only_not_full_benchmark','selection':'first N entries in pinned official metadata','provenance':provenance,'samples':[]}
 videos=root/'videos';videos.mkdir(exist_ok=True)
 for row in rows[:args.count]:
  name=row['video_name']
  if Path(name).name!=name:raise ValueError('Unsafe video name')
  target=videos/(name+'.mp4')
  if not target.exists():
   part=target.with_suffix('.mp4.part')
   with opener.open(row['video_url'],timeout=30) as response,part.open('wb') as out:
    while True:
     block=response.read(1024*1024)
     if not block:break
     out.write(block)
   part.rename(target)
  manifest['samples'].append({'video_name':name,'video_url':row['video_url'],'video_path':str(target),'prompt':row['prompt'],'ground_truth':{'visual quality':row['visual_score'],'text-to-video alignment':row['t2v_score'],'physical/common-sense consistency':row['phy_score']},'video_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
  print('VIDEO_READY',name,target.stat().st_size,flush=True)
 (root/'subset.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('SUBSET_READY',len(manifest['samples']))
if __name__=='__main__':main()
