"""Train ordinary dense-grid MaskGIT using frozen feature-to-token anchors."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('USE_TF','0')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from safetensors import safe_open
from bert2d.paths import ASSETS,atomic_json,sha256
from bert2d.features import PortableBaseFeatures
from bert2d.runtime import save_latest
from bert_continue.data import FoldedSampler
from h20.data import E117SparseCodeDataset,collate_e117_sparse
from experiments.feature_to_token_20260916.converter import load_converter
from experiments.feature_to_token_20260916.grid_maskgit import GridMaskGIT,GridObjective,load_titok_backbone
from experiments.feature_to_token_20260916.probe import nearest


def load_grid(directory,device='cpu'):
    directory=Path(directory)
    metadata=json.loads((directory/'latest.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    assert config==metadata['config'] and config['style']=='dense_proxy_grid_maskgit_v1'
    assert metadata['sha256']==sha256(directory/'latest.pt') and metadata['round_trip_exact']
    core=GridMaskGIT(**config['model_config'])
    with safe_open(str(directory/'latest.pt'),framework='pt',device='cpu') as reader:
        weights={k[4:]:reader.get_tensor(k) for k in reader.keys() if k.startswith('raw/')}
    core.load_state_dict(weights,strict=True)
    return core.to(device).eval().requires_grad_(False),metadata


def run(args):
    rank,world,local=(int(os.getenv(k,'0' if k!='WORLD_SIZE' else '1')) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    if world!=2:raise ValueError('bounded experiment requires two GPUs')
    if 448%(world*args.micro):raise ValueError('micro must divide global448')
    if not 1<=args.updates<=2000:raise ValueError('pilot capped at2000updates')
    torch.set_num_threads(4);torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    torch.cuda.set_per_process_memory_fraction(.90,device)
    if torch.cuda.mem_get_info(device)[0]<25*1024**3:raise RuntimeError('25GiB free required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    out=args.output.resolve();root=Path('/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results').resolve()
    if out==root or not out.is_relative_to(root):raise ValueError('named result output required')
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    status=out/f'train_rank{rank}.json';started=time.monotonic()
    atomic_json(status,dict(status='loading',pid=os.getpid()))
    torch.manual_seed(args.seed)
    core=GridMaskGIT();audit=load_titok_backbone(core)
    core.to(device).train();core.enable_recompute(args.recompute)
    provider=PortableBaseFeatures(str(device),chunk=16)
    converter=None;book=None;converter_audit={'mode':'nearest'}
    if args.converter:
        converter,meta=load_converter(args.converter,device)
        identity=json.loads(args.converter.with_name('latest.json').read_text())
        assert identity['sha256']==sha256(args.converter) and identity['round_trip_exact']
        assert meta['train_config']['frozen_audit']['identities']==provider.audit['identities']
        converter_audit=dict(mode='learned',path=str(args.converter.resolve()),sha256=identity['sha256'],step=meta['step'])
    else:
        from h20_joint.assets import FrozenAssets
        temporary=FrozenAssets(ASSETS,'cpu',chunk=1)
        with torch.no_grad():
            vq=temporary.shell.llamagen_vq
            emb=vq.quantize.get_codebook_entry(torch.arange(16384))
            book=vq.post_quant_conv(emb.T[None,:,None]).squeeze(0).squeeze(1).T.contiguous().to(device)
        del temporary
    fresh=set(audit['fresh_parameters'])
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in core.named_parameters() if n in fresh],'lr':1e-4},
        {'params':[p for n,p in core.named_parameters() if n not in fresh],'lr':1e-5}],
        betas=(.9,.96),eps=1e-8,weight_decay=.03)
    wrapped=DistributedDataParallel(GridObjective(core),device_ids=[local])
    data=E117SparseCodeDataset(ASSETS/'codes/train',ASSETS/'routes/train',split='all')
    assert len(data)==2562334
    sampler=FoldedSampler(data,rank,args.seed,micro=args.micro,world=world)
    loader=DataLoader(data,batch_sampler=sampler,collate_fn=collate_e117_sparse,num_workers=4,pin_memory=True)
    sampler.set_epoch(0);iterator=iter(loader);epoch=0;offset=0
    accum=448//(args.micro*world)
    config=dict(style='dense_proxy_grid_maskgit_v1',model_config=core.config,assets_root=str(ASSETS.resolve()),
        converter=converter_audit,sequence='class +256 grid tokens; no continuous context tokens',
        total_updates=args.updates,micro=args.micro,global_batch=448,world=world,accumulation=accum,seed=args.seed,
        recompute_layers=args.recompute,lr_new=1e-4,lr_pretrained=1e-5,warmup_updates=100,
        betas=[.9,.96],weight_decay=.03,gradient_clip=1.,class_dropout=.1,
        loss='same arccos mask + label smoothing.1 + visible selected weight.1 as 2D-only',
        unselected_gt_used=False,masked_scope='Router-selected positions only',
        precision='BERT bf16; frozen features and converter fp32; TF32 disabled',
        init_audit=audit,feature_audit=provider.audit,sampler='original global448 K-buckets, folded over two ranks',
        source_hashes={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('grid_maskgit.py')]})
    if rank==0:atomic_json(out/'config.json',config)
    ticks=[]
    for step in range(1,args.updates+1):
        tick=time.monotonic()
        if tick-started>5400:raise TimeoutError('90minute bounded MaskGIT pilot expired')
        optimizer.zero_grad(set_to_none=True);stats=torch.zeros(4,device=device)
        warmup=min(step/100,1.)
        optimizer.param_groups[0]['lr']=1e-4*warmup;optimizer.param_groups[1]['lr']=1e-5*warmup
        for micro_step in range(accum):
            try:batch=next(iterator)
            except StopIteration:
                epoch+=1;offset=0;sampler.set_epoch(epoch);iterator=iter(loader);batch=next(iterator)
            offset+=1
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            features=provider(batch['z1d'])
            with torch.no_grad():
                proxy=torch.cat([converter.tokens(x) for x in features.split(8)]) if converter is not None else nearest(features,book)[0]
            indices,valid=batch['route_indices'],batch['route_valid']
            gt=torch.zeros(len(proxy),256,dtype=torch.long,device=device)
            bi=torch.arange(len(proxy),device=device)[:,None].expand_as(indices)[valid]
            gt[bi,indices[valid]]=batch['z2d'][valid]
            with (wrapped.no_sync() if micro_step<accum-1 else nullcontext()):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,row=wrapped(proxy,gt,batch['label'],indices,valid)
                if not torch.isfinite(loss):raise ValueError('nonfinite MaskGIT objective')
                (loss/accum).backward()
            stats+=row/accum
            del features,loss,row
        norm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();ticks.append(time.monotonic()-tick)
        if step==1 or step%20==0 or step==args.updates:
            dist.all_reduce(stats);stats/=world
            record=dict(status='running',pid=os.getpid(),step=step,total=args.updates,loss=float(stats[0]),masked_nll=float(stats[1]),
                mask_ratio=float(stats[2]),class_dropout=float(stats[3]),grad_norm=float(norm),
                seconds=time.monotonic()-started,seconds_per_step=sum(ticks[-20:])/len(ticks[-20:]),
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
            atomic_json(status,record)
            if rank==0:
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
        if step%500==0 or step==args.updates:
            rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device)}
            all_rng=[None]*world;dist.all_gather_object(all_rng,rng)
            if rank==0:
                save_latest(out,core,None,optimizer,all_rng,dict(step=step,config=config,
                    cursor={'packed_pass':epoch,'next_microbatch_offset':offset}))
            dist.barrier()
    provider.assert_frozen()
    if converter is not None:assert all(not p.requires_grad and p.grad is None for p in converter.parameters())
    atomic_json(status,dict(status='complete',pid=os.getpid(),step=args.updates,seconds=time.monotonic()-started,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--converter',type=Path)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--micro',type=int,default=112)
    p.add_argument('--recompute',type=int,default=14)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():atomic_json(a.output/f'failure_rank{os.getenv("RANK","0")}.json',dict(error=repr(exc),pid=os.getpid()))
        raise
