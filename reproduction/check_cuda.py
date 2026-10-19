"""Check real CUDA and bitsandbytes kernels without loading model weights."""
import json,platform,sys,traceback
from pathlib import Path

def main():
    result={'python':sys.version,'platform':platform.platform(),'status':'failed'}
    try:
        import torch,bitsandbytes as bnb,transformers
        result.update(torch=torch.__version__,cuda_build=torch.version.cuda,transformers=transformers.__version__,bitsandbytes=bnb.__version__,cuda_available=torch.cuda.is_available())
        assert torch.cuda.is_available(),'CUDA unavailable'
        result['gpu']=torch.cuda.get_device_name(0)
        layer=bnb.nn.Linear4bit(128,128,bias=False,compute_dtype=torch.float16,quant_type='nf4',compress_statistics=True).to('cuda')
        with torch.inference_mode(): y=layer(torch.randn(2,128,device='cuda',dtype=torch.float16))
        torch.cuda.synchronize()
        assert torch.isfinite(y).all().item()
        result.update(status='passed',nf4_output_shape=list(y.shape))
    except Exception as e:
        result.update(error=repr(e),traceback=traceback.format_exc())
    dest=Path(sys.argv[1])
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1
if __name__=='__main__': raise SystemExit(main())
