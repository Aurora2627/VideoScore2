"""Finish this one evaluation: verify, export, commit and push its recorded results."""
import argparse,ctypes,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def process_running(pid):
    handle=ctypes.windll.kernel32.OpenProcess(0x00100000,False,pid)
    if not handle: return False
    try: return ctypes.windll.kernel32.WaitForSingleObject(handle,0)==258
    finally: ctypes.windll.kernel32.CloseHandle(handle)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,required=True);p.add_argument('--timeout-seconds',type=int,default=21600);a=p.parse_args()
    setup=a.task.parent;task=read(a.task);run=Path(task['output_dir']);state_path=setup/'delivery-state.json'
    state={'status':'waiting_for_evaluation','run_dir':str(run),'started':datetime.datetime.now().isoformat()};save(state_path,state)
    try:
        deadline=time.monotonic()+a.timeout_seconds
        while True:
            session=read(run/'session.json')
            if session['status']!='started': break
            if not process_running(session['attempts'][-1]['pid']): raise RuntimeError('Evaluation process exited unexpectedly; no automatic re-run')
            if time.monotonic()>deadline: raise TimeoutError('Evaluation still running; delivery timeout does not terminate it')
            time.sleep(10)
        assert session['status'] in ('completed','completed_with_errors'),session
        state['status']='verifying';save(state_path,state)
        identity=read(run/'identity.json')
        mapping={p.name:p for p in [ROOT/'experiments/run_expanded_evaluation.py',ROOT/'experiments/expanded_metrics.py',ROOT/'experiments/quality_metrics.py',ROOT/'application/run_batch.py',ROOT/'application/selection.py',ROOT/'reproduction/infer_cuda.py',ROOT/'vs2_inference.py']}
        for name,digest in identity['sources_sha256'].items(): assert sha(mapping[name])==digest,'Evaluated source changed: '+name
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Uncommitted user edits; export/publish requires review'
        out=ROOT/'reports/expanded-evaluation'
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'experiments/export_expanded_evaluation.py'),'--run-dir',str(run),'--output-dir',str(out)],cwd=ROOT,check=True)
        shutil.copy2(setup/'expanded-evaluation.log',out/'execution.log');shutil.copy2(a.task,out/'launch-command.json')
        for name in ('duplicate-launch-probe.json','duplicate-launch-probe.log'):
            if (setup/name).exists(): shutil.copy2(setup/name,out/name)
        shutil.copy2(Path(__file__),out/'sources'/Path(__file__).name)
        save(out/'delivery-checksums.json',{'files_sha256':{str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.is_file() and path.name!='delivery-checksums.json'}})
        subprocess.run(['git','add','--','README.md','reproduction/README.zh-CN.md','reports/expanded-evaluation'],cwd=ROOT,check=True)
        logs=[str(path.relative_to(ROOT)) for path in out.rglob('*.log')]
        if logs: subprocess.run(['git','add','-f','--',*logs],cwd=ROOT,check=True)
        for relative,expected in read(out/'delivery-checksums.json')['files_sha256'].items():
            staged=subprocess.check_output(['git','show',':'+(out/relative).relative_to(ROOT).as_posix()],cwd=ROOT)
            assert hashlib.sha256(staged).hexdigest()==expected,relative
        subprocess.run(['git','--no-pager','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','eval: publish frozen 20-video paired scoring results and uncertainty'],cwd=ROOT,check=True)
        subprocess.run(['git','tag','-a','v0.5.2-expanded-results','-m','Complete expanded evaluation with all outputs, paired metrics and uncertainty'],cwd=ROOT,check=True)
        state.update(status='pushing',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip());save(state_path,state)
        subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7890','push','origin','main','--follow-tags'],cwd=ROOT,check=True,timeout=180)
        state.update(status='published',finished=datetime.datetime.now().isoformat());save(state_path,state)
        print('EXPANDED_DELIVERY_PUBLISHED',state['commit'],flush=True);return 0
    except Exception as error:
        state.update(status='needs_attention',error=str(error),finished=datetime.datetime.now().isoformat());save(state_path,state)
        (setup/'delivery-error.log').write_text(traceback.format_exc(),encoding='utf-8');print('EXPANDED_DELIVERY_NEEDS_ATTENTION',str(error),flush=True);return 1
if __name__=='__main__': raise SystemExit(main())
