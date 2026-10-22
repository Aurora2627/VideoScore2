"""Resume a frozen paired CUDA evaluation without re-running completed jobs."""
import argparse,collections,datetime,hashlib,json,os,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'application'))
from run_batch import ResidentScorer,preflight
from expanded_metrics import summarize

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temporary.replace(path)
def validate_resume(saved,current):
    if saved!=current: raise ValueError('Frozen manifest, model, protocol, policy or source code changed; create a new run')
def build_jobs(samples):
    groups=collections.defaultdict(list)
    for item in samples: groups[item['prompt']].append(item)
    jobs=[]
    for index,items in enumerate(groups.values()):
        phases=[('baseline',224),('more_detail',448)]
        if index%2: phases.reverse()
        for phase,side in phases:
            for item in items: jobs.append((phase,side,item))
    return jobs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','model','output-dir'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--revision',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',args.revision)
    data=read(args.manifest);preflight(data)
    groups=collections.Counter(item['prompt'] for item in data['samples'])
    assert len(data['samples'])==20 and len(groups)==10 and set(groups.values())=={2}
    out=args.output_dir
    if out.exists() and not args.resume: raise FileExistsError('Use --resume for an existing run')
    out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1): lock.write(b'0');lock.flush()
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise RuntimeError('Another process owns this evaluation; refusing duplicate GPU work')
    sources=[Path(__file__),Path(__file__).with_name('expanded_metrics.py'),Path(__file__).with_name('quality_metrics.py'),
             ROOT/'application/run_batch.py',ROOT/'application/selection.py',ROOT/'reproduction/infer_cuda.py',ROOT/'vs2_inference.py']
    protocol=Path(__file__).with_name('expanded_protocol.json');policy_path=ROOT/'application/policy.json';policy=read(policy_path)
    assert args.revision==read(protocol)['model_revision'];assert policy['weights']==read(protocol)['fixed']['weights']
    identity={'manifest_sha256':sha(args.manifest),'model_revision':args.revision,'model_path':str(args.model.resolve()),
              'protocol_sha256':sha(protocol),'policy_sha256':sha(policy_path),'sources_sha256':{p.name:sha(p) for p in sources}}
    if (out/'identity.json').exists(): validate_resume(read(out/'identity.json'),identity)
    else:
        assert not args.resume,'No frozen run identity exists'
        (out/'sources').mkdir()
        for source in sources: shutil.copy2(source,out/'sources'/source.name)
        save(out/'identity.json',identity);save(out/'manifest.json',data);save(out/'policy.json',policy);save(out/'protocol.json',read(protocol))
    session=read(out/'session.json') if (out/'session.json').exists() else {'created':datetime.datetime.now().isoformat(),'attempts':[],
              'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip()}
    results={'baseline':{},'more_detail':{}}
    jobs=build_jobs(data['samples'])
    for phase,side,item in jobs:
        path=out/phase/item['video_name']/'result.json'
        if path.exists():
            value=read(path);assert value['video_sha256']==item['video_sha256'] and value['configuration']=={'frames':4,'max_side':side}
            results[phase][item['video_name']]=value
    pending=[job for job in jobs if job[2]['video_name'] not in results[job[0]]]
    print('EXPANDED_PENDING',len(pending),'TOTAL',len(jobs),flush=True)
    if not pending:
        save(out/'summary.json',summarize(data['samples'],results['baseline'],results['more_detail'],policy['weights']))
        print('EXPANDED_ALREADY_FINISHED',flush=True);return 0
    attempt={'pid':os.getpid(),'started':datetime.datetime.now().isoformat(),'pending_at_start':len(pending)}
    session['attempts'].append(attempt);session['status']='started';save(out/'session.json',session)
    started=time.perf_counter()
    try:
        args.frames=4;args.max_side=224;args.max_new_tokens=1024
        engine=ResidentScorer(args);attempt.update(engine.info);save(out/'session.json',session)
        from transformers import AutoProcessor
        current_side=224
        for index,(phase,side,item) in enumerate(pending,1):
            if side!=current_side:
                args.max_side=side
                engine.processor=AutoProcessor.from_pretrained(args.model,revision=args.revision,min_pixels=56*56,max_pixels=side**2,local_files_only=True,trust_remote_code=False)
                current_side=side
            folder=out/phase/item['video_name'];folder.mkdir(parents=True,exist_ok=True)
            artifact=folder/('attempt_'+str(len(session['attempts'])));artifact.mkdir(exist_ok=False)
            try: result=engine.score({k:item[k] for k in ('video_name','prompt','video_path','video_sha256')},artifact)
            except Exception as error:
                result={'status':'failed','valid':False,'eos_reached':False,'error':str(error),'error_type':type(error).__name__}
                (artifact/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
                if any(text in str(error).lower() for text in ('device-side assert','illegal memory access')): raise
            result.update(video_sha256=item['video_sha256'],configuration={'frames':4,'max_side':side},artifact_directory=artifact.name)
            save(folder/'result.json',result);results[phase][item['video_name']]=result
            session['progress']={'finished_jobs':sum(len(x) for x in results.values()),'total_jobs':len(jobs),'phase':phase,'video_name':item['video_name']}
            save(out/'session.json',session)
            print('EXPANDED_CLIP',session['progress']['finished_jobs'],len(jobs),phase,item['video_name'],result['status'],flush=True)
        report=summarize(data['samples'],results['baseline'],results['more_detail'],policy['weights']);save(out/'summary.json',report)
        session['status']='completed' if report['full_coverage'] else 'completed_with_errors'
    except Exception as error:
        session.update(status='failed',error=str(error));(out/('attempt_'+str(len(session['attempts']))+'_error.log')).write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        attempt.update(finished=datetime.datetime.now().isoformat(),wall_seconds=time.perf_counter()-started)
        save(out/'session.json',session);lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
    print('EXPANDED_FINISHED',str(out),session['status'],flush=True)
    return 0 if session['status']=='completed' else 1
if __name__=='__main__': raise SystemExit(main())
