"""Publish the terminal outcome of one development-only pairwise trial."""
import argparse,datetime,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,required=True);a=p.parse_args();setup=a.task.parent;task=read(a.task);run=Path(task['output_dir'])
    state={'status':'waiting_for_trial','run_dir':str(run)};save(setup/'pairwise-delivery-state.json',state)
    try:
        deadline=time.monotonic()+10800
        while True:
            if (run/'session.json').exists():
                session=read(run/'session.json')
                if session['status']!='started':break
            if time.monotonic()>deadline:raise TimeoutError('Trial still running; delivery stopped without terminating inference')
            time.sleep(10)
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Uncommitted edits require review'
        identity=read(run/'identity.json')
        for name,expected in identity['source_sha256'].items():assert sha(run/'sources'/name)==expected,name
        out=ROOT/'reports/pairwise-development';shutil.copytree(run,out,ignore=shutil.ignore_patterns('run.lock','*.tmp'))
        metrics=read(run/'metrics.json') if (run/'metrics.json').exists() else None
        lines=['# 双视频比较开发实验','','状态：'+session['status']+'。样本为此前已复核的 4 组错误候选，属于开发诊断；不是新的独立检验。',
               '同提示词双视频输入，AB/BA 交换顺序；每维结论一致、无 uncertain 且加权非平才选片。其余进入人工复核。原始单视频评分基线保持，当前应用默认未更改。','']
        if metrics:
            lines+=['计划候选对：4；实际尝试：'+str(metrics['attempted_pairs'])+'；格式有效输出：'+str(metrics['valid_order_outputs'])+'/'+str(metrics['order_outputs'])+'。',
                    '给出有效选择的候选对：'+str(metrics['selected_pairs'])+'/4；判对候选对：'+str(metrics['correct_pairs'])+'/4（弃权不计正确）。',
                    '仅在已选择样例中计算的准确率：'+str(metrics['selected_pair_accuracy'])+'；必须结合覆盖率 '+str(metrics['coverage'])+' 阅读。',
                    '弃权原因：'+json.dumps(metrics['abstention_reasons'],ensure_ascii=False)+'。',
                    '平均单次双视频生成时间：'+str(metrics['mean_generation_seconds'])+' 秒；PyTorch 峰值分配显存：'+str(metrics['peak_allocated_gib'])+' GiB。','']
        lines+=['先判断格式、显存和顺序一致性是否可用，再确定是否值得对原先判对的组和未使用提示词做回归与独立检验。不得把选定已知错误样例上的改善当作泛化提升。',
                '提示词与双视频上下文同时变化，因此这不是单一因素消融。采样、原始输出、token、解析结果、失败和拒绝选择原因均保留。','']
        (out/'RESULTS.md').write_text(chr(10).join(lines),encoding='utf-8')
        shutil.copy2(setup/'pairwise-development.log',out/'execution.log');shutil.copy2(a.task,out/'launch-command.json');shutil.copy2(Path(__file__),out/'sources'/Path(__file__).name)
        save(out/'delivery-checksums.json',{'files_sha256':{str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.is_file()}})
        subprocess.run(['git','add','--','reports/pairwise-development'],cwd=ROOT,check=True)
        logs=[str(path.relative_to(ROOT)) for path in out.rglob('*.log')]
        if logs:subprocess.run(['git','add','-f','--',*logs],cwd=ROOT,check=True)
        for relative,expected in read(out/'delivery-checksums.json')['files_sha256'].items():
            assert hashlib.sha256(subprocess.check_output(['git','show',':'+(out/relative).relative_to(ROOT).as_posix()],cwd=ROOT)).hexdigest()==expected,relative
        subprocess.run(['git','--no-pager','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','eval: record pairwise development feasibility and abstentions'],cwd=ROOT,check=True)
        subprocess.run(['git','tag','-a','v0.6.1-pairwise-development-results','-m','Development-only pairwise trial; retain all outputs and abstentions'],cwd=ROOT,check=True)
        subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7890','push','origin','main','--follow-tags'],cwd=ROOT,check=True,timeout=180)
        state.update(status='published',trial_status=session['status'],finished=datetime.datetime.now().isoformat());save(setup/'pairwise-delivery-state.json',state);print('PAIRWISE_DEVELOPMENT_PUBLISHED',flush=True)
    except Exception as error:
        state.update(status='needs_attention',error=str(error));save(setup/'pairwise-delivery-state.json',state);(setup/'pairwise-delivery-error.log').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
