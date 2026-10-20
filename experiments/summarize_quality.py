"""Export completed input-quality evidence without changing model outputs."""
import argparse,hashlib,json,shutil
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 session=read(a.run_dir/'session.json');assert session['status']=='completed','Wait for the full experiment'
 for name,value in session['source_sha256'].items():assert sha(a.run_dir/'sources'/name)==value
 for name,value in session['baseline_result_sha256'].items():assert sha(a.run_dir/'baseline-cached'/name/'result.json')==value
 shutil.copytree(a.run_dir,a.output_dir)
 decision=read(a.output_dir/'recommendation.json');protocol=read(a.output_dir/'protocol.json');rows=[]
 rows.append({'name':'development_baseline','frames':4,'max_side':224,'metrics':read(a.output_dir/'baseline-development-metrics.json')})
 for name,phase in session['phases'].items():
  if name.startswith('regression_'):rows.append({'name':'regression_baseline','frames':4,'max_side':224,'metrics':read(a.output_dir/'baseline-regression-metrics.json')})
  rows.append({'name':name,**phase['configuration'],'metrics':phase['metrics']})
 save=lambda name,value:(a.output_dir/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
 sample_comparison=[]
 for name,phase in session['phases'].items():
  split='development' if name.startswith('development_') else ('regression' if name.startswith('regression_') else 'untouched-check')
  samples=read(a.output_dir/(split+'-manifest.json'))['samples']
  for item in samples:
   result=read(a.output_dir/name/item['video_name']/'result.json')
   sample_comparison.append({'phase':name,'split':split,'video_name':item['video_name'],'configuration':phase['configuration'],'status':result['status'],'scores':result.get('scores'),'ground_truth':item['ground_truth'],'generation_seconds':result.get('generation_seconds'),'peak_allocated_bytes':result.get('peak_allocated_bytes'),'input_shapes':result.get('input_shapes')})
 save('sample-comparison.json',sample_comparison)
 lines=['# ClipRank 评分输入对照实验','','固定模型、原始提示词、量化、随机种子、生成配置与选片规则，只分别增加帧数或空间细节。','','## 开发选择、回归与独立检查','','开发样本是已知错误候选对（2 条）；回归样本是已有诊断视频（3 条）；新检查是按固定元数据顺序选出的未使用提示词候选对（2 条）。新检查标签不用于选择候选配置。样本仍很少，不能据此推断整体业务效果。','','| 阶段 | 帧数 | 最长边 | 有效/总数 | 画质 MAE | 文本 MAE | 物理 MAE | 平均 MAE | 候选对方向正确数 |','|---|---:|---:|---|---:|---:|---:|---:|---|']
 labels=('visual quality','text-to-video alignment','physical/common-sense consistency')
 number=lambda x:'—' if x is None else format(x,'.3f')
 for row in rows:
  m=row['metrics'];lines.append('| '+row['name']+' | '+str(row['frames'])+' | '+str(row['max_side'])+' | '+str(m['valid_count'])+'/'+str(m['sample_count'])+' | '+' | '.join(number(m['per_dimension_mae'][k]) for k in labels)+' | '+number(m['macro_mae'])+' | '+str(m['pair_correct'])+'/'+str(m['pair_total'])+' |')
 lines+=['','## 配置结论','','开发阶段选中：'+decision['chosen_on_development']+'。','是否通过预先设定的默认配置更新门槛：'+('是' if decision['eligible_for_default'] else '否')+'。']
 if 'promotion_gates' in decision:
  lines+=['','| 条件 | 结果 |','|---|---|']
  lines+=['| '+k+' | '+('通过' if v else '未通过')+' |' for k,v in decision['promotion_gates'].items()]
 else:lines+=['','两个候选都未在完整输出条件下降低开发误差，保留基线；未开启新检查阶段。']
 lines+=['','推荐参数：'+json.dumps(decision['recommended_configuration'],ensure_ascii=False)+'。','所有原始回答、token、输入配置、源码快照和失败记录保留。未覆盖旧分数，也未根据标签调提示词、阈值或权重。','','## 使用边界','','这些结果检验输入压缩的影响，没有进行模型训练。开发对已被观察过；新检查只有一个提示词、两个视频，不能作为大规模泛化或业务采用率证明。','若某项改动降低平均误差但不能纠正排序或损害原样本，不能仅凭该项误差改善就统一替换默认配置。','','模型加载次数：'+str(session['model_load_count'])+'；本轮总耗时：'+format(session['wall_seconds'],'.1f')+' 秒。']
 (a.output_dir/'RESULTS.md').write_text(chr(10).join(lines),encoding='utf-8')
 save('export-receipt.json',{'original_session_sha256':sha(a.run_dir/'session.json'),'protocol_sha256':sha(a.run_dir/'protocol.json'),'export_source_sha256':sha(Path(__file__)),'model_results_modified':False})
 costs=[]
 for phase_name in session['phases']:
     results=[read(path) for path in (a.output_dir/phase_name).glob('*/result.json')]
     generated=[r for r in results if 'generation_seconds' in r]
     costs.append({'phase':phase_name,'generated_count':len(generated),'mean_generation_seconds':sum(r['generation_seconds'] for r in generated)/len(generated) if generated else None,'peak_allocated_gib':max((r['peak_allocated_bytes']/2**30 for r in generated),default=None),'max_input_tokens':max((r['input_shapes']['input_ids'][1] for r in generated),default=None),'max_video_patch_rows':max((r['input_shapes']['pixel_values_videos'][0] for r in generated),default=None)})
 save('resource-comparison.json',costs)
 extra=['','## 资源代价','','| 阶段 | 有生成记录数 | 平均生成秒数 | 峰值显存 GiB | 最大输入 token | 视频 patch 行数 |','|---|---:|---:|---:|---:|---:|']
 for cost in costs:
     extra.append('| '+cost['phase']+' | '+str(cost['generated_count'])+' | '+number(cost['mean_generation_seconds'])+' | '+number(cost['peak_allocated_gib'])+' | '+str(cost['max_input_tokens'])+' | '+str(cost['max_video_patch_rows'])+' |')
 report_path=a.output_dir/'RESULTS.md';report_path.write_text(report_path.read_text(encoding='utf-8')+chr(10)+chr(10).join(extra),encoding='utf-8')
 print(json.dumps({'selected':decision['chosen_on_development'],'eligible_for_default':decision['eligible_for_default'],'recommended_configuration':decision['recommended_configuration'],'phases':rows},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
