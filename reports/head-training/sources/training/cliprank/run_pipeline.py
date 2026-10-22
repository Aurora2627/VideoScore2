"""One-shot CUDA training pipeline with frozen sources, stage logs and optional Git publication."""
import argparse,datetime,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from prepare_data import read,save,sha
ROOT=Path(__file__).resolve().parents[2]
def stamp():return datetime.datetime.now().astimezone().isoformat()
def publish(run):
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Uncommitted edits require review before publication'
 out=ROOT/'reports/head-training';out.mkdir(parents=True,exist_ok=False)
 for name in ('sources','heads','test'):shutil.copytree(run/name,out/name)
 for name in ('manifest.json','protocol.json','identity.json','status.json','data-provenance.json','split-plan.json'):shutil.copy2(run/name,out/name)
 shutil.copytree(run/'logs',out/'logs')
 features=run/'features';save(out/'feature-audit.json',{'identity':read(features/'identity.json'),'session':read(features/'session.json'),'receipts':{p.parent.name:read(p) for p in features.glob('*/feature.json')},'tensor_cache':'Feature tensors retained in Windows run directory; regenerate with extract_features.py, not included in Git'})
 comparison=read(run/'test/comparison.json');selection=read(run/'heads/selection.json');plan=read(run/'split-plan.json')
 lines=['# ClipRank 冻结多模态骨干的评分头训练实验','','训练/验证数据来自 VideoFeedback2 官方训练划分；新测试子集来自官方测试划分，按提示词隔离，并排除先前项目调试使用的组。','本轮训练轻量评分头，未更新 VideoScore2 骨干，未执行 LoRA、SFT 或 GRPO。骨干此前已在官方训练集上训练过，因此验证集隔离针对本项目评分头，不代表骨干从未见过验证样本。','','## 数据与训练','','样本数量：'+json.dumps(plan['counts'],ensure_ascii=False)+'。','特征：四帧、448 输入、NF4 量化 VideoScore2，读取回答生成前最后一个非 padding token 的隐藏状态。模型输入不含人工评分或解释。','线性头、MLP、MLP+排序损失各三个随机种子；仅按验证集 MAE 选择三种子集成。选中：'+selection['selected_variant']+'。','','## 新测试子集结果','','| 方法 | 样本数 | 平均 MAE | 排序正确/可比较对 |','|---|---:|---:|---:|']
 for label,key in [('评分头（全部测试）','head_all_test'),('训练集均值','train_mean_baseline'),('固定三分','constant_three_baseline'),('原模型（有效输出）','original_on_valid_test'),('评分头（同一有效子集）','head_on_same_valid_test')]:
  m=comparison.get(key)
  if m:lines.append('| '+label+' | '+str(m['samples'])+' | '+str(round(m['mean_mae'],6))+' | '+str(m['ranking_correct'])+'/'+str(m['ranking_pairs'])+' |')
 lines+=['','原模型有效输出：'+str(comparison['baseline_valid_count'])+'/'+str(comparison['test_count'])+'；无效样本不计作正确，配对比较仅使用同一有效子集。','按提示词 bootstrap：'+json.dumps(comparison.get('prompt_bootstrap_difference'),ensure_ascii=False)+'。','','测试只在验证集选择完成后运行；这批结果不能用于继续调参后再次声称独立验证。测试规模有限，报告保留退步与不确定性，不自动替换当前应用默认评分器。','评分头、标准化参数、九次训练曲线、验证集对照、测试原始回答、采样和文件校验均保存。视频和完整骨干权重不提交 Git。','','基础模型与数据来源：[VideoScore2](https://huggingface.co/TIGER-Lab/VideoScore2)、[VideoFeedback2](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2)。代码与模型授权见项目 LICENSE、MODEL_SOURCES.md 和 CITATION.cff。','']
 (out/'RESULTS.zh-CN.md').write_text(chr(10).join(lines),encoding='utf-8')
 save(out/'checksums.json',{'files_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}})
 subprocess.run(['git','add','-f','--','reports/head-training'],cwd=ROOT,check=True)
 for name,digest in read(out/'checksums.json')['files_sha256'].items():assert __import__('hashlib').sha256(subprocess.check_output(['git','show',':reports/head-training/'+name],cwd=ROOT)).hexdigest()==digest,name
 subprocess.run(['git','--no-pager','diff','--cached','--check'],cwd=ROOT,check=True)
 subprocess.run(['git','commit','-m','eval: publish frozen-backbone head training and untouched test comparison'],cwd=ROOT,check=True)
 subprocess.run(['git','tag','-a','v0.7.1-head-training-results','-m','Score-head training, validation selection and new test comparison'],cwd=ROOT,check=True)
 subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7890','push','origin','main','--follow-tags'],cwd=ROOT,check=True,timeout=240)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=Path('D:/career-projects/data/ClipRank-training'));ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--publish',action='store_true');a=ap.parse_args();a.run_dir.mkdir(parents=True,exist_ok=False);(a.run_dir/'logs').mkdir()
 import msvcrt
 lock=(a.run_dir/'run.lock').open('w+b');lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 status={'status':'started','started':stamp(),'pid':os.getpid(),'stages':[]};save(a.run_dir/'status.json',status)
 files=list((ROOT/'training/cliprank').glob('*.py'))+[ROOT/'training/cliprank/protocol.json',ROOT/'application/run_batch.py',ROOT/'application/selection.py',ROOT/'reproduction/infer_cuda.py',ROOT/'vs2_inference.py']
 identity={'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'manifest_sha256':sha(a.data_dir/'manifest.json'),'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode('ascii').strip()};save(a.run_dir/'identity.json',identity)
 for p in files:
  target=a.run_dir/'sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 for src,dest in [('manifest.json','manifest.json'),('metadata-provenance.json','data-provenance.json'),('split-plan.json','split-plan.json')]:shutil.copy2(a.data_dir/src,a.run_dir/dest)
 shutil.copy2(ROOT/'training/cliprank/protocol.json',a.run_dir/'protocol.json')
 env=os.environ.copy();env.update(PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 def stage(name,script,args):
  for relative,digest in identity['source_sha256'].items():assert sha(ROOT/relative)==digest,'Source changed during experiment: '+relative
  entry={'name':name,'started':stamp(),'command':[sys.executable,'-X','utf8',str(ROOT/'training/cliprank'/script),*map(str,args)]};status.update(stage=name,status='running');status['stages'].append(entry)
  with (a.run_dir/'logs'/(name+'.log')).open('x',encoding='utf-8') as log:
   child=subprocess.Popen(entry['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);entry['pid']=child.pid;save(a.run_dir/'status.json',status);code=child.wait()
  entry.update(finished=stamp(),exit_code=code);save(a.run_dir/'status.json',status);assert code==0,name+' failed; see stage log';print('STAGE_COMPLETED',name,flush=True)
 try:
  manifest=a.run_dir/'manifest.json';features=a.run_dir/'features';heads=a.run_dir/'heads';test=a.run_dir/'test'
  stage('features','extract_features.py',['--manifest',manifest,'--output-dir',features])
  assert read(features/'session.json')['status']=='completed'
  stage('heads','train_heads.py',['--manifest',manifest,'--features',features,'--output-dir',heads])
  assert (heads/'selection.json').exists()
  stage('test','evaluate_test.py',['--manifest',manifest,'--features',features,'--heads',heads,'--output-dir',test])
  status.update(status='completed',finished=stamp());save(a.run_dir/'status.json',status)
  if a.publish:publish(a.run_dir);status.update(status='published',published=stamp());save(a.run_dir/'status.json',status)
 except Exception as error:status.update(status='needs_attention',error=str(error),stopped=stamp());save(a.run_dir/'status.json',status);(a.run_dir/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
