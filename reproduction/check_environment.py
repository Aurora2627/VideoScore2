"""Verify the activated named environment and execute a small CUDA operation."""
import argparse,json,os,sys
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 expected=Path(os.environ.get('VIDEOSCORE2_ENV','D:/career-projects/envs/videoscore2')).resolve()
 assert Path(sys.prefix).resolve()==expected,'Wrong interpreter'
 assert Path(os.environ['VIRTUAL_ENV']).resolve()==expected,'Environment is not activated'
 assert os.environ.get('VIRTUAL_ENV_PROMPT')=='videoscore2-cuda','Unexpected environment name'
 assert sys.prefix!=sys.base_prefix,'Not an isolated venv'
 assert torch.cuda.is_available(),'CUDA unavailable'
 x=torch.ones((4,4),device='cuda');value=float((x@x)[0,0].item());assert value==4.0
 result={'status':'passed','display_name':'videoscore2-cuda','python':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cuda_matmul_value':value,'scope':'environment and small CUDA operation only; no model rerun'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
