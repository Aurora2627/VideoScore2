from pathlib import Path
import json,hashlib,cv2
root=Path('D:/career-projects/data/ClipRank-quality-20261020');records={}
for split in ['development','regression','untouched-check']:
 data=json.loads((root/(split+'.json')).read_text(encoding='utf-8'));rows=[]
 for item in data['samples']:
  path=Path(item['video_path']);digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==item['video_sha256']
  cap=cv2.VideoCapture(str(path))
  try:rows.append({'video_name':item['video_name'],'sha256':digest,'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':cap.get(cv2.CAP_PROP_FPS)})
  finally:cap.release()
 records[split]=rows
seen={x['sha256'] for split in ['development','regression'] for x in records[split]};fresh={x['sha256'] for x in records['untouched-check']};assert seen.isdisjoint(fresh)
report={'fresh_video_hashes_disjoint':True,'all_input_hashes_match':True,'source_videos':records}
Path('D:/career-projects/videoscore2-setup-20261019/quality-data-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('QUALITY_DATA_AUDIT_PASSED; no labels read or printed')
