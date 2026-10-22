"""Select fresh prompt groups before inference, then download a pinned evaluation set."""
import argparse, collections, hashlib, json, re, urllib.request
from pathlib import Path
LABELS = ('visual quality','text-to-video alignment','physical/common-sense consistency')
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)
def select_groups(rows, excluded_prompts, excluded_names, count):
    groups=collections.defaultdict(list)
    for row in rows:
        if row['prompt'] not in excluded_prompts and row['video_name'] not in excluded_names:
            groups[row['prompt']].append(row)
    selected=[]
    for group in groups.values():
        if len(group)>=2:
            selected.extend(group[:2])
            if len(selected)==count*2: return selected
    raise ValueError('Not enough unused multi-candidate prompts')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--exclude',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--groups',type=int,default=10)
    p.add_argument('--proxy',default='http://127.0.0.1:7890');a=p.parse_args()
    if a.groups<1: p.error('groups must be positive')
    provenance=read(a.data_root/'metadata-provenance.json')
    assert sha(a.data_root/'metadata.json')==provenance['sha256']
    assert re.fullmatch('[0-9a-f]{40}',provenance['revision'])
    excluded=[item for path in a.exclude for item in read(path)['samples']]
    rows=select_groups(read(a.data_root/'metadata.json'),{x['prompt'] for x in excluded},{x['video_name'] for x in excluded},a.groups)
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True);(out/'videos').mkdir(exist_ok=True)
    plan={'selection':'First unused multi-candidate prompt groups in pinned metadata order; first two videos per group; no label or prediction filtering',
          'metadata_provenance':provenance,'excluded_manifests':{str(path):sha(path) for path in a.exclude},'groups':a.groups,
          'video_names':[row['video_name'] for row in rows]}
    if (out/'selection-plan.json').exists(): assert read(out/'selection-plan.json')==plan,'Selection changed; use a new output directory'
    else: save(out/'selection-plan.json',plan)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':a.proxy,'https':a.proxy}))
    samples=[];seen={x['video_sha256'] for x in excluded}
    for row in rows:
        name=row['video_name'];assert re.fullmatch('[A-Za-z0-9_-]+',name)
        url=row['video_url'].replace('/resolve/main/','/resolve/'+provenance['revision']+'/')
        assert url.startswith(('https://huggingface.co/','https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/'))
        path=out/'videos'/(name+'.mp4');receipt=path.with_suffix('.receipt.json')
        if receipt.exists():
            previous=read(receipt);assert previous['url']==url and previous['sha256']==sha(path)
        else:
            temp=path.with_suffix('.part');size=0
            with opener.open(url,timeout=90) as response,temp.open('wb') as handle:
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>64*1024*1024: raise ValueError('Video exceeds 64 MiB cap')
                    handle.write(chunk)
            assert size>0;temp.replace(path);save(receipt,{'url':url,'sha256':sha(path),'bytes':size})
        digest=sha(path);assert digest not in seen,'Duplicate video across evaluation inputs';seen.add(digest)
        samples.append({'video_name':name,'prompt':row['prompt'],'video_path':str(path.resolve()),'video_url':url,'video_sha256':digest,
                        'ground_truth':dict(zip(LABELS,(row['visual_score'],row['t2v_score'],row['phy_score'])))})
        print('EXPANDED_VIDEO_READY',name,path.stat().st_size,flush=True)
    import cv2
    audit=[]
    for item in samples:
        capture=cv2.VideoCapture(item['video_path']);ok,frame=capture.read()
        try:
            assert ok and frame is not None,'Video cannot be decoded'
            audit.append({'video_name':item['video_name'],'sha256':item['video_sha256'],'width':int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          'height':int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),'fps':capture.get(cv2.CAP_PROP_FPS),'frames':int(capture.get(cv2.CAP_PROP_FRAME_COUNT))})
        finally: capture.release()
    save(out/'input-audit.json',audit)
    save(out/'manifest.json',{'samples':samples,'purpose':'Expanded frozen-input comparison; no parameter selection from this evaluation','selection_plan':plan,'video_versioning':'Metadata revision is pinned. External object-storage video URLs are not revision-addressed; downloaded bytes are identified by captured SHA256 receipts.'})
    print('EXPANDED_INPUT_READY',len(samples),a.groups,flush=True)
if __name__=='__main__': main()
