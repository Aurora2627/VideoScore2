from pathlib import Path
import json,collections,urllib.request
rows=json.loads(Path('D:/career-projects/data/VideoScoreBench-v2/metadata.json').read_text(encoding='utf-8-sig'))
m=collections.defaultdict(set)
for r in rows:m[r['video_name'].split('_')[-1]].add(r['video_url'].split('/')[-2])
print('GENERATOR_MAP',dict((k,sorted(v)) for k,v in sorted(m.items())))
print('GROUP_PREFIX_COUNTS',len(set(r['video_name'].split('_')[0] for r in rows)))
print('MODEL_CLASS_SOURCE')
import inspect
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
s=inspect.getsource(Qwen2_5_VLForConditionalGeneration.forward)
print(s[s.index('outputs = self.model'):s.index('outputs = self.model')+1100])
