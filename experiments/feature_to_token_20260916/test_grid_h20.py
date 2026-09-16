"""CPU checks for4->8 rank migration and identical full-grid sampling recipe."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('USE_TF','0')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import ast
import numpy as np
import torch
from bert_continue.data import FoldedSampler
from bert2d.runtime import RankBucketSampler, OffsetSampler
from experiments.feature_to_token_20260916.continue_grid_h20 import migration_rng


class Toy:
    k_values=np.repeat([64,128],448*4+123)


def test():
    for epoch in (0,1,2):
        old=[FoldedSampler(Toy(),r,0,micro=112,world=4) for r in range(4)]
        new=[RankBucketSampler(Toy(),56,8,r,0) for r in range(8)]
        for sampler in old+new: sampler.set_epoch(epoch)
        before,after=[list(s) for s in old],[list(s) for s in new]
        assert all(len(s)==len(before[0]) for s in after)
        for step in range(len(before[0])):
            a=sum([s[step] for s in before],[])
            b=sum([s[step] for s in after],[])
            assert a==b and len(set(b))==448
        for r,s in enumerate(new):
            for skip in (0,1,len(after[r])):
                offset=OffsetSampler(s);offset.offset=skip
                assert list(offset)==after[r][skip:]
    seeds=[]
    for rank in range(4,8):
        seed,state,_=migration_rng('test',rank,torch.device('cpu'))
        seeds.append(seed)
        assert torch.equal(state,migration_rng('test',rank,torch.device('cpu'))[1])
    assert len(set(seeds))==4
    root=Path(__file__).parent
    old=(root/'eval_grid.py').read_text();new=(root/'eval_grid_h20.py').read_text()
    # AST identity of the entire generation loop before decode: sampler untouched.
    def generation_prefix(source):
        tree=ast.parse(source)
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='offset')
        parts=[]
        for n in loop.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dense_features' for t in n.targets):break
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='images' for t in n.targets):break
            parts.append(ast.dump(n))
        return parts
    assert generation_prefix(old)==generation_prefix(new)
    assert not torch.cuda.is_initialized()
    print('PASS:4x112 vs8x56 exact global row order/coverage across3 epochs; offsets; unique new-rank RNG; generation AST unchanged')


if __name__=='__main__':test()
