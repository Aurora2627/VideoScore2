"""Build a fixed diagnostic manifest with original checks and a real same-prompt pair."""
import argparse,collections,hashlib,json,urllib.request
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--proxy',default=None);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 provenance=read(a.data_root/'metadata-provenance.json');metadata=a.data_root/'metadata.json';assert sha(metadata)==provenance['sha256']
 records=read(metadata);base=read(a.data_root/'subset.json');groups=collections.defaultdict(list)
 for row in records:groups[row['prompt']].append(row)
 existing={x['video_name'] for x in base['samples']}
 pair=next(rows[:2] for rows in groups.values() if len(rows)>=2 and not any(x['video_name'] in existing for x in rows[:2]))
 a.output_dir.mkdir(parents=True);videos=a.output_dir/'videos';videos.mkdir()
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':a.proxy,'https':a.proxy} if a.proxy else {}))
 for row in pair:
  url=row['video_url'].replace('/resolve/main/','/resolve/'+provenance['revision']+'/')
  if not url.startswith('https://'):raise ValueError('HTTPS video source required')
  destination=videos/(row['video_name']+'.mp4');count=0
  with opener.open(url,timeout=90) as response,destination.open('xb') as f:
   while chunk:=response.read(1024*1024):
    count+=len(chunk)
    if count>64*1024*1024:raise ValueError('Diagnostic video exceeds 64 MiB')
    f.write(chunk)
  if count<1024:raise ValueError('Unexpected video response')
  item={k:row[k] for k in ('video_name','prompt')};item.update(video_url=url,video_path=str(destination.resolve()),video_sha256=sha(destination),bytes=count,ground_truth=dict(zip(('visual quality','text-to-video alignment','physical/common-sense consistency'),(row['visual_score'],row['t2v_score'],row['phy_score']))))
  base['samples'].append(item);print('PAIR_VIDEO_SAVED',row['video_name'],count,flush=True)
 base['metadata_provenance']=provenance;base['purpose']='Fixed diagnostic only: previous three plus first disjoint same-prompt pair in pinned metadata. Not a tuning or held-out evaluation set.'
 base['selection_rule']='Metadata order, no filtering on scores. Teacher thinking excluded.'
 (a.output_dir/'manifest.json').write_text(json.dumps(base,indent=2),encoding='utf-8');print('DEMO_MANIFEST_READY',len(base['samples']),flush=True)
if __name__=='__main__':main()
