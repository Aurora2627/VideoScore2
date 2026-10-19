"""Combine saved diagnostic runs without changing original inference artifacts."""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('first-dir','remaining-dir','manifest','output-dir'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();manifest=read(a.manifest);report={'scope':'Three fixed diagnostic videos, not full benchmark or training reproduction','samples':[]}
 sources=[a.first_dir]+sorted(x.parent for x in a.remaining_dir.glob('*/result.json'))
 assert len(sources)==3,'All three real inference records required'
 a.output_dir.mkdir(parents=True,exist_ok=False)
 shutil.copy2(a.manifest,a.output_dir/'subset-manifest.json')
 for sample,directory in zip(manifest['samples'],sources):
  result=read(directory/'result.json');assert result['video_sha256']==sample['video_sha256']
  assert sha(directory/'inference_source.py')==result['source_sha256']
  assert {k:result['args'][k] for k in ('frames','max_side','max_new_tokens','revision')}=={'frames':4,'max_side':224,'max_new_tokens':1024,'revision':'09a2732cb64fa566a1f332f978368292ce5c295c'}
  derived=read(directory/'reparsed-v013.json') if directory==a.first_dir else result
  assert derived['valid'] and derived['eos_reached'],'Complete validated output required'
  if directory==a.first_dir:
   assert derived['original_result_sha256']==sha(directory/'result.json')
   assert derived['raw_output_sha256']==sha(directory/'raw_output.txt')
  destination=a.output_dir/sample['video_name'];destination.mkdir()
  for name in ('result.json','raw_output.txt','inference_source.py','reparsed-v013.json'):
   if (directory/name).exists():shutil.copy2(directory/name,destination/name)
  row={k:result[k] for k in ('project_commit','source_sha256','video_sha256','load_seconds','generation_seconds','peak_allocated_bytes','peak_reserved_bytes','environment','video_sampling')}
  row.update(video_name=sample['video_name'],scores=derived['scores'],ground_truth=sample['ground_truth'],eos_reached=True,valid=True,postprocess_only_repair=directory==a.first_dir,original_status=result['status'],generated_tokens=len(result['generated_token_ids']),raw_output_sha256=sha(directory/'raw_output.txt'))
  row['absolute_error']={k:abs(v-float(sample['ground_truth'][k])) for k,v in row['scores'].items()}
  report['samples'].append(row)
 report['valid_outputs']=len(report['samples']);report['accuracy_claim']=False
 report['protocol']={'frames':4,'max_side':224,'max_new_tokens':1024,'quantization':'NF4 4-bit double quantization','decoding':'greedy','model_revision':'09a2732cb64fa566a1f332f978368292ce5c295c','labels_and_teacher_reasoning_in_model_input':False}
 (a.output_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 lines=['# Three-video diagnostic results','','Official weights; Windows RTX 4060 Laptop 8 GB; dedicated PyTorch CUDA environment. Valid output means EOS plus three parsed integer scores, not a correct prediction.','','Scores are visual quality / text alignment / physical consistency.','','| Video | Prediction | Dataset label | Generation (s) | Peak allocated (GiB) |','|---|---|---|---:|---:|']
 for row in report['samples']:
  vals=lambda d:' / '.join(str(d[k]) for k in ('visual quality','text-to-video alignment','physical/common-sense consistency'))
  lines.append('| '+row['video_name']+' | '+vals(row['scores'])+' | '+vals(row['ground_truth'])+' | '+format(row['generation_seconds'],'.1f')+' | '+format(row['peak_allocated_bytes']/2**30,'.2f')+' |')
 lines+=['','The first video reached EOS but its expanded score labels were rejected by the old parser. Its original result remains unchanged; reparsed-v013.json records the separate parser-only repair with hashes and commit. The other two videos ran using the repaired parser.','','This is a reduced-frame, reduced-resolution, quantized inference diagnostic. It does not reproduce paper benchmark metrics, floating-score evaluation, SFT or GRPO. Three videos cannot establish general scoring quality. See saved raw outputs and source snapshots for evidence.']
 (a.output_dir/'RESULTS.md').write_text('\n'.join(lines).replace('\\n','\n'),encoding='utf-8')
 print(json.dumps({'valid_outputs':report['valid_outputs'],'samples':[{k:r[k] for k in ('video_name','scores','ground_truth','generation_seconds')} for r in report['samples']]},indent=2))
if __name__=='__main__':main()
