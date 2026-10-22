from pathlib import Path
import json,urllib.request,urllib.parse,hashlib
ROOT=Path(__file__).resolve().parents[2]
REV='99b9935b00a205756c25b0555f2dcdc071281327'
data_dir=Path('D:/career-projects/data/ClipRank-training');data_dir.mkdir(exist_ok=True)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7890'}))
name='data_27k_train_RL.json';path=data_dir/name
if not path.exists():
 url='https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2/resolve/'+REV+'/'+urllib.parse.quote(name)
 with opener.open(url,timeout=90) as response:raw=response.read()
 json.loads(raw);path.write_bytes(raw)
 (data_dir/'metadata-provenance.json').write_text(json.dumps({'url':url,'revision':REV,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)},indent=2),encoding='utf-8')
rows=json.loads(path.read_text(encoding='utf-8'))
print('TRAIN_ROWS',len(rows),'KEYS',list(rows[0]))
print('FIRST_ROW',json.dumps(rows[0],ensure_ascii=False)[:4500])
import inspect
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
print('FORWARD',inspect.signature(Qwen2_5_VLModel.forward))
