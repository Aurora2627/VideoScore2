"""Publish immutable local evaluation evidence and a review dashboard."""
import argparse,hashlib,json,shutil
from pathlib import Path

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--previous-report',type=Path);a=p.parse_args()
 run=read(a.run_dir/'run.json');assert run['status'] in ('completed','completed_with_errors'),'Wait for finished batch';assert run['model_load_count']==1
 for name,digest in run['source_sha256'].items():assert sha(a.run_dir/'sources'/name)==digest,'Saved source changed'
 manifest=read(a.run_dir/'manifest.json');selection=read(a.run_dir/'selection.json');metrics=read(a.run_dir/'metrics.json')
 assert metrics['attempted']==len(manifest['samples']) and metrics['valid_outputs']<=metrics['attempted']
 previous=read(a.previous_report/'summary.json') if a.previous_report else {'samples':[]};comparison=[]
 for old in previous['samples']:
  if not (a.run_dir/old['video_name']/'result.json').exists():continue
  now=read(a.run_dir/old['video_name']/'result.json')
  original=read(a.previous_report/old['video_name']/'result.json')
  comparison.append({'video_name':old['video_name'],'scores_equal':now['scores']==old['scores'],'tokens_equal':now['generated_token_ids']==original['generated_token_ids'],'old_scores':old['scores'],'new_scores':now['scores'],'old_generation_seconds':old['generation_seconds'],'new_generation_seconds':now['generation_seconds'],'same_video_hash':now['video_sha256']==old['video_sha256']})
 assert all(r['same_video_hash'] for r in comparison)
 shutil.copytree(a.run_dir,a.output_dir)
 media=a.output_dir/'media';media.mkdir()
 for item in manifest['samples']:
     source=Path(item['video_path']);assert sha(source)==item['video_sha256'];shutil.copy2(source,media/(item['video_name']+'.mp4'))
 # Render presentation separately. Original report and model outputs are retained.
 from render_review import render
 render(a.output_dir,a.output_dir/'review.html')
 save=lambda name,value:(a.output_dir/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
 save('historical-comparison.json',{'scope':'Same three videos and inference settings across historical runs; not a controlled speed benchmark','samples':comparison})
 rows=[]
 for item in manifest['samples']:
  r=read(a.run_dir/item['video_name']/'result.json');rows.append({'video_name':item['video_name'],'scores':r.get('scores'),'ground_truth':item.get('ground_truth'),'generation_seconds':r.get('generation_seconds'),'peak_allocated_gib':r['peak_allocated_bytes']/2**30 if 'peak_allocated_bytes' in r else None})
 shortlist=[r['video_name'] for g in selection['groups'] for r in g['candidates'] if r['selected']]
 review=[r['video_name'] for g in selection['groups'] for r in g['candidates'] if r['decision']=='manual_review']
 summary={'valid_outputs':metrics['valid_outputs'],'prompt_groups':len(selection['groups']),'model_load_count':run['model_load_count'],'model_load_seconds':run['model_load_seconds'],'batch_wall_seconds':run['wall_seconds'],'samples':rows,'mae':metrics['mae'],'shortlist':shortlist,'manual_review':review,'scope':'Diagnostic evaluation only. No business acceptance, model improvement or full benchmark claim.'};save('application-summary.json',summary)
 lines=['# ClipRank 素材评估实测','','同一进程常驻模型，按提示词分组评估和辅助选片。','','- 有效完整输出：'+str(metrics['valid_outputs'])+'/'+str(len(rows)),'- 需求分组：'+str(len(selection['groups'])),'- 模型加载次数：1','- 模型加载耗时：'+format(run['model_load_seconds'],'.1f')+' 秒','- 整批耗时：'+format(run['wall_seconds'],'.1f')+' 秒（含导入、加载、预处理、推理及报告）','','分数顺序为画质 / 文本匹配 / 物理合理性。','','| 视频 | 预测 | 数据标签 | 生成秒数 | 峰值显存 GiB |','|---|---|---|---:|---:|']
 labels=('visual quality','text-to-video alignment','physical/common-sense consistency')
 for row in rows:
  display=lambda x:' / '.join(str(x.get(k,'—')) for k in labels) if x else '—'
  number=lambda x:'—' if x is None else format(x,'.1f')
  lines.append('| '+row['video_name']+' | '+display(row['scores'])+' | '+display(row['ground_truth'])+' | '+number(row['generation_seconds'])+' | '+number(row['peak_allocated_gib'])+' |')
 lines+=['','候选短名单：'+('、'.join(shortlist) or '无；不会强行选片。'),'待人工复核：'+('、'.join(review) or '无。'),'','原三个诊断样本的数值与 token 对照见 historical-comparison.json。不同运行时段的耗时仅供参考，不据此宣称稳定加速比例。','本轮权重与阈值为预先配置的启发式规则，未根据标签调整。当前诊断样本不构成泛化效果或业务收益证明。','','浏览器评审入口：[review.html](review.html)。原始 CLI 报告 report.html、run.json、原始输出及源码快照均保留。']
 (a.output_dir/'RESULTS.md').write_text(chr(10).join(lines),encoding='utf-8')
 save('export-receipt.json',{'original_run_json_sha256':sha(a.run_dir/'run.json'),'original_selection_sha256':sha(a.run_dir/'selection.json'),'export_source_sha256':sha(Path(__file__)),'renderer_source_sha256':sha(Path(__file__).with_name('render_review.py')),'raw_output_sha256':{r['video_name']:sha(a.output_dir/r['video_name']/'raw_output.txt') if (a.output_dir/r['video_name']/'raw_output.txt').exists() else None for r in rows},'model_outputs_modified':False})
 print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
