import sys,json,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path('training/cliprank').resolve()))
from prepare_data import read,download
p=Path('D:/career-projects/data/ClipRank-training');plan=read(p/'split-plan.json');bench=read('D:/career-projects/data/VideoScoreBench-v2/metadata.json')
for r in bench:
 letter=r['video_name'].split('_')[1]
 if letter not in plan['generator_map']:continue
 start=int(r['video_name'].split('_')[0])//500*500;derived='https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/'+str(start).zfill(4)+'_'+str(start+499).zfill(4)+'/'+plan['generator_map'][letter]+'/'+r['video_name']+'.mp4'
 assert derived==r['video_url'],(derived,r['video_url'])
print('URL_MAPPING_VALIDATED')
for row in read(p/'planned-samples.json')['splits']['train'][:2]:
 result=download(row,p);print('SMOKE_VIDEO',result['video_name'],Path(result['video_path']).stat().st_size)
