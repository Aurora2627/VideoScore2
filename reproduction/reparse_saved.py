"""Reparse a saved generation without changing the original result or rerunning a model."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from infer_cuda import classify_output

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sample-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.output.exists():raise FileExistsError(str(args.output))
 original=args.sample_dir/'result.json';raw=args.sample_dir/'raw_output.txt'
 data=json.loads(original.read_text(encoding='utf-8'))
 assert digest(args.sample_dir/'inference_source.py')==data['source_sha256'],'Original inference source changed'
 result={'postprocess_only':True,'model_inference_rerun':False,'original_result_sha256':digest(original),'raw_output_sha256':digest(raw),'original_status':data['status'],'parser_source_sha256':digest(Path(__file__).with_name('infer_cuda.py')),'parser_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],encoding='utf-8',errors='replace').strip(),'eos_reached':data.get('eos_reached',False)}
 result.update(classify_output(raw.read_text(encoding='utf-8'),result['eos_reached']))
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2))
 return 0 if result['valid'] else 2
if __name__=='__main__':raise SystemExit(main())
