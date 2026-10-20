"""Export a completed supplementary check and apply its predeclared decision."""
import argparse
import hashlib
import json
import py_compile
import shutil
from pathlib import Path
from quality_metrics import evaluate
from strict_gates import gates

ROOT = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--original-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run, original, out = args.run_dir, args.original_run, args.output_dir
    session = read(run / 'session.json')
    parent = read(original / 'session.json')
    assert session['status'] == parent['status'] == 'completed'
    assert session['parent_session_sha256'] == sha(original / 'session.json')
    for name, expected in session['source_sha256'].items():
        assert sha(run / 'sources' / name) == expected, name
    manifest = read(run / 'manifest.json')
    policy = read(run / 'policy.json')
    rows, resources = [], []
    for phase in ('baseline', 'more_detail'):
        results = {item['video_name']: read(run / phase / item['video_name'] / 'result.json') for item in manifest['samples']}
        metrics = evaluate(manifest['samples'], results, policy['weights'])
        assert metrics == session['phases'][phase]['metrics'], phase
        for item in manifest['samples']:
            result = results[item['video_name']]
            rows.append({'phase': phase, 'video_name': item['video_name'], 'ground_truth': item['ground_truth'], 'scores': result.get('scores'), 'status': result['status']})
        resources.append({'phase':phase, 'peak_allocated_gib':max(r.get('peak_allocated_bytes',0) for r in results.values()) / 2**30})
    checks = gates(parent['recommendation'], session)
    eligible = all(checks.values())
    configuration = {'frames':4, 'max_side':448 if eligible else 224}
    decision = {'eligible_for_default':eligible, 'recommended_configuration':configuration, 'promotion_gates':checks,
                'original_gate_preserved':parent['recommendation']['eligible_for_default'],
                'scope':'9 unique diagnostic videos, 6 prompts; 2 development, 3 regression, 4 fresh. No training and no population-level guarantee.'}
    shutil.copytree(run, out)
    shutil.copy2(Path(__file__), out / 'sources' / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('strict_gates.py'), out / 'sources' / 'strict_gates.py')
    save(out / 'sample-comparison.json', rows)
    save(out / 'resource-comparison.json', resources)
    save(out / 'recommendation.json', decision)
    application = ROOT / 'application/run_batch.py'
    before = sha(application)
    if eligible:
        source = application.read_text(encoding='utf-8')
        old = "p.add_argument('--max-side',type=int,default=224)"
        assert source.count(old) == 1
        shutil.copy2(application, out / 'sources/application_before_default.py')
        application.write_text(source.replace(old,"p.add_argument('--max-side',type=int,default=448)"), encoding='utf-8')
        py_compile.compile(str(application), doraise=True)
    save(out / 'application-default-receipt.json', {'changed':eligible, 'before_source_sha256':before, 'after_source_sha256':sha(application), 'default_configuration':configuration})
    save(ROOT / 'application/quality_recommendation.json', {**decision, 'evidence':'reports/strict-check/RESULTS.md'})
    old = parent['recommendation']
    b = session['phases']['baseline']['metrics']
    d = session['phases']['more_detail']['metrics']
    mean = lambda first, second: (first['macro_mae']*first['sample_count']+second['macro_mae']*second['sample_count'])/(first['sample_count']+second['sample_count'])
    combined_base = mean(old['fresh_baseline'], b)
    combined_detail = mean(old['fresh_candidate'], d)
    lines = ['# ClipRank 评分输入改进与独立候选验证', '',
      '固定 VideoScore2 权重、提示词、4-bit NF4 量化、生成参数和选片权重。开发阶段选定 4 帧、最长边 448 像素；补充检查只比较这一配置与 4 帧、224 像素基线。', '',
      '原新样例对的加权标注总分相同，因此无法检验排序方向。保留其全部结果和原门槛未通过记录，另按元数据顺序选取首个未使用且标注总分不相同的候选对。选取时未使用模型预测；补充方案先提交 Git，后运行。', '',
      '| 数据划分 | 视频数 | 224 像素 MAE | 448 像素 MAE |', '|---|---:|---:|---:|',
      f"| 已知错误开发样例 | 2 | {old['baseline_development']['macro_mae']:.3f} | {old['candidate_development']['more_detail']['macro_mae']:.3f} |",
      f"| 旧样例回归 | 3 | {old['regression_baseline']['macro_mae']:.3f} | {old['regression_candidate']['macro_mae']:.3f} |",
      f"| 首组新样例（标注总分打平） | 2 | {old['fresh_baseline']['macro_mae']:.3f} | {old['fresh_candidate']['macro_mae']:.3f} |",
      f"| 补充新样例（方向明确） | 2 | {b['macro_mae']:.3f} | {d['macro_mae']:.3f} |",
      f'| 合计新样例 | 4 | {combined_base:.3f} | {combined_detail:.3f} |', '',
      f"补充候选对方向：基线 {b['pair_correct']}/{b['pair_total']}，448 像素 {d['pair_correct']}/{d['pair_total']}。预测打平不记为方向正确。", '',
      '默认配置：4 帧、最长边 '+str(configuration['max_side'])+' 像素。'+('补充门槛全部通过。' if eligible else '补充门槛未全部通过，448 像素仍作为显式实验配置。'), '',
      '| 更新条件 | 通过 |', '|---|---|']
    lines += [f'| {name} | {value} |' for name,value in checks.items()]
    lines += ['', '这是 9 个视频、6 个提示词的小样本诊断，其中新检验仅 4 个视频、2 个提示词。不能将结果解释为完整基准成绩、线上效果或稳定泛化收益。物理一致性判断仍需扩大评估。', '',
      '完整源代码快照、输入哈希、原始文本、token、逐视频评分和模型加载信息保存在本目录。原对照实验见 [输入对照报告](../quality-ablation/RESULTS.md)。', '']
    (out / 'RESULTS.md').write_text(chr(10).join(lines), encoding='utf-8')
    for path in (ROOT/'README.md', ROOT/'reproduction/README.zh-CN.md'):
        source = path.read_text(encoding='utf-8')
        source = source.split('## 评分输入改进实验')[0].rstrip()
        if eligible:
            source = source.replace('当前默认 4 帧、最长边 224 像素','当前默认 4 帧、最长边 448 像素')
        link = 'reports/strict-check/RESULTS.md' if path.parent == ROOT else '../reports/strict-check/RESULTS.md'
        text = chr(10).join(['','', '## 评分输入改进实验', '',
            '对比增加帧数与提高分辨率，开发样例平均绝对误差从 1.833 降至 1.000，旧样例回归从 0.778 降至 0.556。',
            f'4 个新检验视频合计 MAE：224 像素 {combined_base:.3f}，448 像素 {combined_detail:.3f}。当前默认 4 帧、最长边 '+str(configuration['max_side'])+' 像素。',
            '完整条件、排序结果、资源代价与小样本限制见 [评分改进报告]('+link+')。主应用支持 --frames 4 --max-side 448 显式对照。',''])
        path.write_text(source+text, encoding='utf-8')
    save(out / 'export-receipt.json', {'source_run':str(run), 'files_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    print('STRICT_EVIDENCE_PUBLISHED', json.dumps(decision, ensure_ascii=False))

if __name__ == '__main__':
    main()
