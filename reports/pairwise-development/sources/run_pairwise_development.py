"""Development-only pairwise trial on previously reviewed failures; no holdout claim."""
import argparse,collections,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'application'))
from run_batch import ResidentScorer,preflight
from pairwise import compare,order_consensus,LABELS

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--revision',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    protocol=read(Path(__file__).with_name('pairwise_development_protocol.json'));assert a.revision==protocol['model_revision']
    source=ROOT/'reports/expanded-evaluation';data=read(source/'manifest.json');audit=read(source/'distribution-analysis.json');policy=read(source/'policy.json')
    groups=[group for group in audit['prompt_groups'] if group['direction_correct']['more_detail'] is False]
    assert policy['weights']==protocol['weights'];assert len(groups)==4;names={name for group in groups for name in group['videos']};samples=[item for item in data['samples'] if item['video_name'] in names]
    preflight({'samples':samples});items={item['video_name']:item for item in samples}
    out=a.output_dir
    if out.exists() and not a.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'run.lock').open('a+b')
    if os.fstat(lock.fileno()).st_size==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Another process owns this pairwise trial')
    sources=[Path(__file__),ROOT/'application/pairwise.py',ROOT/'application/run_batch.py',ROOT/'application/selection.py',ROOT/'reproduction/infer_cuda.py',ROOT/'vs2_inference.py']
    identity={'model_path':str(a.model.resolve()),'model_revision':a.revision,'source_sha256':{path.name:sha(path) for path in sources},'manifest_sha256':sha(source/'manifest.json'),'audit_sha256':sha(source/'distribution-analysis.json'),'protocol':protocol,'policy':policy}
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity,'Source or input changed; use a new trial'
    else:
        assert not a.resume;(out/'sources').mkdir()
        for path in sources:shutil.copy2(path,out/'sources'/path.name)
        save(out/'identity.json',identity);save(out/'manifest.json',{'samples':samples,'groups':groups,'scope':'Selected prior failures: development only'});save(out/'protocol.json',protocol)
    session=read(out/'session.json') if (out/'session.json').exists() else {'created':datetime.datetime.now().isoformat(),'attempts':[],'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip()}
    attempt={'started':datetime.datetime.now().isoformat(),'pid':os.getpid()};session['attempts'].append(attempt);session['status']='started';save(out/'session.json',session);start=time.perf_counter()
    results={};summaries=[]
    try:
        a.frames=4;a.max_side=448;a.max_new_tokens=1024;engine=ResidentScorer(a);attempt.update(engine.info);save(out/'session.json',session)
        for index,group in enumerate(groups,1):
            pair_names=group['videos'];ordered=[items[name] for name in pair_names];pair_dir=out/('pair_'+str(index));pair_dir.mkdir(exist_ok=True);orders=[]
            for label,ordered_items in [('AB',ordered),('BA',list(reversed(ordered)))]:
                folder=pair_dir/label;folder.mkdir(exist_ok=True);path=folder/'result.json'
                if path.exists():result=read(path)
                else:
                    artifacts=folder/('attempt_'+str(len(session['attempts'])));artifacts.mkdir(exist_ok=False)
                    try:result=compare(engine,[{k:item[k] for k in ('video_name','prompt','video_path','video_sha256')} for item in ordered_items],artifacts)
                    except Exception as error:
                        result={'status':'failed','valid':False,'parsed':{'valid':False,'reason':'inference_error'},'error':str(error)};(artifacts/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
                        if any(text in str(error).lower() for text in ('device-side assert','illegal memory access')):raise
                    result['artifact_directory']=artifacts.name;save(path,result)
                orders.append(result);results[str(index)+'_'+label]=result
                session['progress']={'finished_jobs':len(results),'total_jobs':8,'pair':index,'order':label};save(out/'session.json',session)
                print('PAIRWISE_ORDER',len(results),8,index,label,result['status'],result['valid'],flush=True)
            consensus=order_consensus(orders[0]['parsed'],orders[1]['parsed'],pair_names,policy['weights'])
            margin=group['annotation_margin'];expected=pair_names[0] if margin>0 else pair_names[1]
            row={'pair':index,'videos':pair_names,'consensus':consensus,'expected_from_annotation':expected,'correct':consensus['selected']==expected,'baseline_correct':False}
            summaries.append(row);save(pair_dir/'consensus.json',row);save(out/'pair-summary.json',summaries)
            if index==1 and not any(order['valid'] for order in orders):session['status']='stopped_invalid_format';break
        else:session['status']='completed'
        selected=[row for row in summaries if row['consensus']['selected'] is not None]
        metrics={'scope':'Development on 4 selected known failures; cannot claim held-out improvement','planned_pairs':4,'attempted_pairs':len(summaries),'valid_order_outputs':sum(r['valid'] for r in results.values()),'order_outputs':len(results),
                 'selected_pairs':len(selected),'coverage':len(selected)/4,'correct_pairs':sum(row['correct'] for row in summaries),'correct_over_all_planned_pairs':sum(row['correct'] for row in summaries)/4,
                 'selected_pair_accuracy':sum(row['correct'] for row in selected)/len(selected) if selected else None,'abstention_reasons':dict(collections.Counter(row['consensus']['reason'] for row in summaries if row['consensus']['selected'] is None)),
                 'mean_generation_seconds':sum(r.get('generation_seconds',0) for r in results.values())/len(results) if results else None,'peak_allocated_gib':max((r.get('peak_allocated_bytes',0) for r in results.values()),default=0)/2**30}
        save(out/'metrics.json',metrics)
    except Exception as error:session.update(status='failed',error=str(error));(out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        attempt.update(finished=datetime.datetime.now().isoformat(),wall_seconds=time.perf_counter()-start);save(out/'session.json',session);lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
    print('PAIRWISE_FINISHED',str(out),session['status'],flush=True)
    return 0 if session['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
