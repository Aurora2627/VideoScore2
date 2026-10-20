from pathlib import Path
import hashlib,json,sys
root=Path('D:/career-projects/VideoScore2')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=Path('D:/career-projects/data/ClipRank-quality-20261020')
paths=[inputs/'development.json',inputs/'regression.json',inputs/'untouched-check.json',Path('D:/career-projects/data/ClipRank-strict-check-20261020/manifest.json')]
seen_prompts=set();seen_hashes=set();seen_names=set();audit=[]
for path in paths:
    data=read(path);prompts=set();hashes=set()
    for item in data['samples']:
        video=Path(item['video_path']);digest=sha(video)
        assert digest==item['video_sha256'],item['video_name']
        assert digest not in seen_hashes and digest not in hashes
        assert item['video_name'] not in seen_names
        seen_names.add(item['video_name']);hashes.add(digest);prompts.add(item['prompt'])
    assert not prompts.intersection(seen_prompts),str(path)
    seen_hashes.update(hashes);seen_prompts.update(prompts)
    audit.append({'manifest':str(path),'sha256':sha(path),'videos':len(data['samples']),'prompts':len(prompts)})
record={'splits':audit,'unique_videos':len(seen_hashes),'unique_prompts':len(seen_prompts),'split_prompt_overlap':False,'duplicate_video_hashes':False}
(root/'reports/quality-ablation/combined-input-audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('INPUT_AUDIT',json.dumps(record))
for name in ('experiments/export_strict_check.py','experiments/test_strict_gates.py'):
    path=root/name;path.write_text(path.read_text(encoding='utf-8-sig'),encoding='utf-8')
report=root/'reports/quality-ablation/RESULTS.md'
text=report.read_text(encoding='utf-8')
text+=chr(10)+'注：fresh 阶段的 pair_total=0 是两段视频加权标注总分打平，排序方向不可评估，并非方向判断错误。原门槛未通过记录保留，补充验证另见 strict-check 报告。'+chr(10)
report.write_text(text,encoding='utf-8')
