"""Publish all expanded-evaluation outcomes after the run has finished."""
import argparse,hashlib,json,shutil
from pathlib import Path
from expanded_metrics import summarize
from quality_metrics import LABELS
ROOT=Path(__file__).resolve().parents[1]
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def number(value): return '不可计算' if value is None else format(value,'.3f')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    run,out=a.run_dir,a.output_dir;session=read(run/'session.json')
    assert session['status'] in ('completed','completed_with_errors'),'Wait for the evaluation to finish'
    identity=read(run/'identity.json')
    for name,expected in identity['sources_sha256'].items(): assert sha(run/'sources'/name)==expected,name
    data=read(run/'manifest.json');policy=read(run/'policy.json');results={}
    for phase in ('baseline','more_detail'):
        results[phase]={item['video_name']:read(run/phase/item['video_name']/'result.json') for item in data['samples']}
    calculated=summarize(data['samples'],results['baseline'],results['more_detail'],policy['weights'])
    assert calculated==read(run/'summary.json'),'Summary must match the stored raw scores'
    shutil.copytree(run,out,ignore=shutil.ignore_patterns('run.lock','*.tmp'))
    shutil.copy2(Path(__file__),out/'sources'/Path(__file__).name)
    b,d=calculated['baseline'],calculated['more_detail'];uncertainty=calculated['paired_prompt_bootstrap']
    lines=['# ClipRank 扩大独立评测', '',
      '先固定清单与方案，再对 20 段新视频、10 个新提示词进行 224/448 像素成对比较。模型、原始提示词、4 帧采样、4-bit NF4、种子、生成上限和选片权重固定。按提示词交替两种配置的执行顺序。', '',
      '本轮没有训练、修改提示词、重新挑参数或自动修改默认配置。视频按元数据顺序选择，未按标注或预测筛选；所有失败、截断和并列结果均保留。', '',
      '| 指标 | 224 像素 | 448 像素 |','|---|---:|---:|',
      '| 有效视频数 / 计划数 | '+str(b['valid_count'])+'/20 | '+str(d['valid_count'])+'/20 |',
      '| 平均绝对误差 MAE | '+number(b['macro_mae'])+' | '+number(d['macro_mae'])+' |',
      '| 非平标注候选对方向正确数 | '+str(b['pair_correct'])+'/'+str(b['pair_total'])+' | '+str(d['pair_correct'])+'/'+str(d['pair_total'])+' |']
    for key in LABELS:
        lines.append('| '+key+' MAE | '+number(b['per_dimension_mae'][key])+' | '+number(d['per_dimension_mae'][key])+' |')
    for key,label in [('mean_generation_seconds','平均生成秒数'),('peak_allocated_gib','峰值分配显存 GiB')]:
        lines.append('| '+label+' | '+number(calculated['resources']['baseline'][key])+' | '+number(calculated['resources']['more_detail'][key])+' |')
    lines+=['', '## 不确定性与参照', '',
      '两种配置均有效的视频数：'+str(calculated['paired_valid_videos'])+'/20；完整提示词组：'+str(calculated['complete_prompt_groups'])+'/10。',
      '恒定输出 3 分参照的 MAE：'+number(calculated['constant_three_reference']['macro_mae'])+'。这仅用于检查评分是否优于无区分能力的简单参照。']
    if uncertainty:
        lines+=['448 减 224 的按提示词平均 MAE 差：'+number(uncertainty['mean'])+'；提示词分组 bootstrap 95% 区间：['+number(uncertainty['lower_95'])+', '+number(uncertainty['upper_95'])+']。负数表示 448 误差更低；区间跨 0 时，不支持稳定收益判断。']
    lines+=['', '## 评分相关性', '', '| 维度 | 224 Spearman | 448 Spearman | 224 Pearson | 448 Pearson |','|---|---:|---:|---:|---:|']
    for key in LABELS:
        cb=calculated['correlations']['baseline'][key];cd=calculated['correlations']['more_detail'][key]
        lines.append('| '+key+' | '+number(cb['spearman'])+' | '+number(cd['spearman'])+' | '+number(cb['pearson'])+' | '+number(cd['pearson'])+' |')
    lines+=['', '## 解释边界', '',
      '仅 10 个提示词，相关性与区间均应作为小样本诊断。加权标注总分打平的组不计排序分母；模型预测打平不算方向正确。若存在无效输出，不能用有效样本上的 MAE 宣称完整评测提升，应同时检查 paired_* 和覆盖率。',
      '元数据版本已固定；部分视频来自元数据指向的外部对象存储，URL 本身没有版本号，因此以下载记录中的 SHA256 标识实际字节。',
      '原始模型输出、token、采样位置、输入形状、每视频耗时及显存、源代码快照、输入清单和执行记录均已保留。未来训练或调参需另划开发集与新检验集。','']
    (out/'RESULTS.md').write_text(chr(10).join(lines),encoding='utf-8')
    for path in (ROOT/'README.md',ROOT/'reproduction/README.zh-CN.md'):
        link='reports/expanded-evaluation/RESULTS.md' if path.parent==ROOT else '../reports/expanded-evaluation/RESULTS.md'
        section=chr(10).join(['','## 扩大独立评测','','已按预先固定的方案评测 20 段新视频、10 个新提示词，比较 224 与 448 像素配置。',
          'MAE：'+number(b['macro_mae'])+' → '+number(d['macro_mae'])+'；两种配置均有效的视频 '+str(calculated['paired_valid_videos'])+'/20。完整排序、相关性、不确定性及失败记录见 [扩大评测报告]('+link+')。',''])
        source=path.read_text(encoding='utf-8');source=source.split('## 扩大独立评测')[0].rstrip();path.write_text(source+chr(10)+section,encoding='utf-8')
    (out/'delivery-checksums.json').write_text(json.dumps({'files_sha256':{str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.is_file()}},indent=2),encoding='utf-8')
    print('EXPANDED_EVIDENCE_PUBLISHED',session['status'],calculated['conclusion'])
if __name__=='__main__': main()
