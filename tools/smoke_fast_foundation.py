import sys,time,json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/foundation_runtime'))
sys.path.insert(0,str(ROOT/'third_party/Fast-FoundationStereo'))
from core.utils.utils import InputPadder
model=torch.load(ROOT/'checkpoints/Fast-FoundationStereo/23-36-37/model_best_bp2_serialize.pth',map_location='cpu',weights_only=False)
model.args.valid_iters=8;model.args.max_disp=192;model.cuda().eval()
print(type(model),model.args,flush=True)
from run_scissors_foundationstereo import PairReader
src=ROOT/'pointflow_outputs/scissors_foundationstereo_20261105'
episode=Path(json.loads((src/'source.json').read_text())['episode']);pair=json.loads((src/'ir_pair_index.json').read_text())[0]
left,right=PairReader(episode).get(pair)
tensors=[torch.as_tensor(np.repeat(x[...,None],3,2)).cuda().float()[None].permute(0,3,1,2).contiguous() for x in [left,right]]
padder=InputPadder(tensors[0].shape,divis_by=32,force_square=False);tensors=padder.pad(*tensors)
with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
 for j in range(3):
  torch.cuda.synchronize();start=time.perf_counter();disp=model.forward(*tensors,iters=8,test_mode=True,optimize_build_volume='pytorch1');torch.cuda.synchronize()
  d=padder.unpad(disp.float()).cpu().numpy().squeeze();print(j,(time.perf_counter()-start)*1000,d.shape,np.percentile(d,[5,50,95]),flush=True)
