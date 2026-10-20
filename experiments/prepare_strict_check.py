"""Prepare the first unused prompt pair with a non-tied fixed-weight annotation direction."""
import collections,hashlib,json,urllib.request
from pathlib import Path
root=Path('D:/career-projects/data');read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=read(root/'ClipRank-demo-20261020/manifest.json')['samples']+read(root/'ClipRank-quality-20261020/untouched-check.json')['samples'];seen={r['prompt'] for r in old};seen_hash={r['video_sha256'] for r in old}
provenance=read(root/'VideoScoreBench-v2/metadata-provenance.json');metadata=root/'VideoScoreBench-v2/metadata.json';assert sha(metadata)==provenance['sha256']
groups=collections.defaultdict(list)
for row in read(metadata):
 if row['prompt'] not in seen:groups[row['prompt']].append(row)
def margin(pair):return sum((float(pair[0][k])-float(pair[1][k]))*w for k,w in [('visual_score',0.3),('t2v_score',0.4),('phy_score',0.3)])
pair=next(rows[:2] for rows in groups.values() if len(rows)>=2 and abs(margin(rows[:2]))>1e-8)
out=root/'ClipRank-strict-check-20261020';out.mkdir(exist_ok=False);videos=out/'videos';videos.mkdir();opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}));samples=[]
for row in pair:
 url=row['video_url'].replace('/resolve/main/','/resolve/'+provenance['revision']+'/');path=videos/(row['video_name']+'.mp4');count=0
 if not url.startswith('https://'):raise ValueError('HTTPS required')
 with opener.open(url,timeout=90) as response,path.open('xb') as file:
  while chunk:=response.read(1024*1024):
   count+=len(chunk)
   if count>64*1024*1024:raise ValueError('Diagnostic size cap')
   file.write(chunk)
 digest=sha(path);assert digest not in seen_hash
 samples.append({'video_name':row['video_name'],'prompt':row['prompt'],'video_path':str(path),'video_url':url,'video_sha256':digest,'ground_truth':dict(zip(('visual quality','text-to-video alignment','physical/common-sense consistency'),(row['visual_score'],row['t2v_score'],row['phy_score'])))})
 print('STRICT_CHECK_VIDEO_READY',row['video_name'],count,flush=True)
(out/'manifest.json').write_text(json.dumps({'samples':samples,'metadata_provenance':provenance,'selection_rule':'First unused prompt pair in metadata order with nonzero fixed .3/.4/.3 label margin; no model predictions used for selection; original tied pair retained.'},indent=2),encoding='utf-8')
print('STRICT_CHECK_INPUT_READY')
