"""Prepare deterministic development, regression and untouched prompt-disjoint check inputs."""
import argparse,collections,hashlib,json,urllib.request
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original-manifest',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--proxy',required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 old=read(a.original_manifest);groups=collections.defaultdict(list)
 for x in old['samples']:groups[x['prompt']].append(x)
 pair=next(v for v in groups.values() if len(v)==2);names={x['video_name'] for x in pair};regression=[x for x in old['samples'] if x['video_name'] not in names]
 provenance=read(a.data_root/'metadata-provenance.json');assert sha(a.data_root/'metadata.json')==provenance['sha256']
 fresh=collections.defaultdict(list)
 for row in read(a.data_root/'metadata.json'):
  if row['prompt'] not in groups:fresh[row['prompt']].append(row)
 untouched=next(v[:2] for v in fresh.values() if len(v)>=2)
 a.output_dir.mkdir(parents=True);videos=a.output_dir/'videos';videos.mkdir();opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':a.proxy,'https':a.proxy}));samples=[]
 for row in untouched:
  url=row['video_url'].replace('/resolve/main/','/resolve/'+provenance['revision']+'/');destination=videos/(row['video_name']+'.mp4');count=0
  if not url.startswith('https://'):raise ValueError('HTTPS source required')
  with opener.open(url,timeout=90) as response,destination.open('xb') as output:
   while chunk:=response.read(1024*1024):
    count+=len(chunk)
    if count>64*1024*1024:raise ValueError('Video exceeds diagnostic size cap')
    output.write(chunk)
  item={'video_name':row['video_name'],'prompt':row['prompt'],'video_url':url,'video_path':str(destination),'video_sha256':sha(destination),'ground_truth':dict(zip(('visual quality','text-to-video alignment','physical/common-sense consistency'),(row['visual_score'],row['t2v_score'],row['phy_score'])))};samples.append(item);print('UNTOUCHED_VIDEO_READY',row['video_name'],count,flush=True)
 for filename,items,purpose in [('development.json',pair,'Previously observed failure pair; development diagnosis only'),('regression.json',regression,'Previous three diagnostic videos; no new independence claimed'),('untouched-check.json',samples,'First unused same-prompt pair in pinned metadata order. Not training data or a full held-out benchmark.')]:
  (a.output_dir/filename).write_text(json.dumps({'samples':items,'purpose':purpose,'metadata_provenance':provenance},indent=2),encoding='utf-8')
 assert set(x['prompt'] for x in samples).isdisjoint(groups)
 print('INPUT_SPLITS_READY',len(pair),len(regression),len(samples),flush=True)
if __name__=='__main__':main()
