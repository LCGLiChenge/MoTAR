"""Fixed-cohort free-generation FID; RGB proxies, selected-only direct replacement."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('USE_TF','0')
# The running trainer launches this file by path for every evaluation. Dispatch
# before Torch/TF imports so the live job can adopt parallel eval without a restart.
if __name__=='__main__' and not any(v.startswith('--worker-shard') for v in sys.argv):
    from experiments.feature_to_token_20260916.halton_parallel_eval import cli
    raise SystemExit(cli())
import numpy as np
import torch
from bert2d.paths import ASSETS,ONE_D,EVALUATOR,atomic_json,output_path,sha256
from bert2d.eval import ImageBert,OfficialSamplingView,to_uint8
from h20.assets import official_model
from h20_joint.assets import FrozenAssets
from experiments.feature_to_token_20260916.halton_proxy import HaltonProxy
from experiments.feature_to_token_20260916.halton_official_completion import sample_halton_completion
from experiments.feature_to_token_20260916.halton_parallel_eval import worker_batches
from experiments.feature_to_token_20260916.e117_parent_only import install_parent_only
from experiments.feature_to_token_20260916.converter import FeatureTokenConverter
from experiments.feature_to_token_20260916.converter_lowrank import LowRankFeatureTokenConverter


def main(args):
    if args.n not in (8,32,64,5000,50000): raise ValueError('unsupported evaluation size')
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=1: raise ValueError('one isolated evaluation GPU required')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    out=output_path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    shard_id=args.worker_shard
    batches=worker_batches(args.n,args.worker_count,shard_id)
    status_path=out/f'status_shard{shard_id}.json'
    if status_path.exists(): raise FileExistsError(status_path)
    started=time.monotonic()
    atomic_json(status_path,dict(status='loading',pid=os.getpid(),completed=0))
    digest=sha256(args.checkpoint)
    state=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    config=state['config'];step=state['step']
    model=HaltonProxy(Path(config['upstream_root']))
    model.load_state_dict(state['model'],strict=True)
    model=model.to(device).eval().requires_grad_(False)
    del state
    assets=FrozenAssets(ASSETS,str(device),chunk=4,keep_encoder=True)
    mapper=None;mapper_audit=None
    if args.mapper_checkpoint is not None:
        mapper_digest=sha256(args.mapper_checkpoint)
        mapper_state=torch.load(args.mapper_checkpoint,map_location='cpu',mmap=True,weights_only=True)
        if mapper_state.get('format') not in ('feature_proxy_token_converter_v1', 'feature_proxy_token_converter_lowrank_v1'):
            raise RuntimeError('wrong mapper checkpoint format')
        mapper_class=(LowRankFeatureTokenConverter if mapper_state['format'].endswith('lowrank_v1') else FeatureTokenConverter)
        mapper=mapper_class(**mapper_state['model_config'])
        mapper.load_state_dict(mapper_state['model'],strict=True)
        mapper=mapper.to(device).eval().requires_grad_(False)
        mapper_config=mapper_state['train_config']
        if mapper_config.get('target','').find('dense 256 proxy IDs')<0:
            raise RuntimeError('mapper was not trained on RGB-reencoded proxy IDs')
        if mapper_config['frozen_audit']['identities']!=assets.identities:
            raise RuntimeError('mapper and evaluation frozen assets differ')
        mapper_audit=dict(checkpoint=str(args.mapper_checkpoint.resolve()),sha256=mapper_digest,
                          step=int(mapper_state['step']),format=mapper_state['format'],
                          parameters=sum(p.numel() for p in mapper.parameters()))
        del mapper_state
    if args.router_mode == 'parent-only':
        router_audit=install_parent_only(assets)
    else:
        router_audit={'mode':'e117','parameters':sum(p.numel() for p in assets.adapter.student.parameters())}
    if assets.identities['weights/mot_latest.pt']!=config['codebook_audit']['mot_sha256']:
        raise RuntimeError('evaluation decoder differs from training proxy decoder')
    one_d,one_d_audit=official_model(ASSETS)
    one_d=one_d.to(device).eval().requires_grad_(False)
    view=OfficialSamplingView(one_d).eval()
    from bert2d.assets import FID,check
    for item in FID: check(ASSETS/item['local_path'],item)
    spec=importlib.util.spec_from_file_location('halton_proxy_adm',EVALUATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.INCEPTION_V3_PATH=str(ASSETS/'fid/classify_image_graph_def.pb')
    tf=module.tf;tf.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    cfg=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=2)
    cfg.gpu_options.allow_growth=True;cfg.gpu_options.visible_device_list='0'
    manifest=dict(n=args.n,step=step,checkpoint_sha256=digest,batch=8,logical_shards=4,seed=args.seed,
        stage1=ONE_D,one_d_audit=one_d_audit,router=router_audit,
        stage2={**config['sampling'],'cfg_w':args.cfg_w,'steps':args.stage2_steps},
        proxy=(dict(mode='learned_f1d_mapper',**mapper_audit) if mapper is not None else
               dict(mode='rgb_reencoded',precision='fp32 no TF32',
                    path='MoT EMA decoder RGB [0,1] -> 2*RGB-1 -> same EMA encoder/quantizer')),
        decoder_identities=assets.identities,primary='full_refine',
        full_refine='unselected continuous 1D features unchanged; selected directly replaced by 2D',
        no_gt_images_or_2d_used=True,source_sha256=sha256(Path(__file__)),
        reference=str(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz'))
    manifest['physical_gpu']=os.environ['CUDA_VISIBLE_DEVICES']
    atomic_json(out/f'manifest_shard{shard_id}.json',manifest)
    arms=('base','full_refine')
    completed_count=0
    with torch.inference_mode(),tf.Session(config=cfg) as session:
        evaluator=module.Evaluator(session,batch_size=8)
        gpu_checked=False
        for shard,ids_all in [(shard_id,np.concatenate(batches))]:
            pools={name:np.lib.format.open_memmap(out/f'{name}_shard{shard}.npy',mode='w+',dtype=np.float32,
                   shape=(len(ids_all),2048)) for name in arms}
            offset=0
            for ids in batches:
                labels=torch.as_tensor(ids%1000,device=device)
                torch.manual_seed(args.seed+2*int(ids[0]))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z1=ImageBert.generate(view,condition=labels,**ONE_D)
                bundle=assets.bundle(z1)
                base,index,valid=(bundle[k] for k in ('base','index','valid'))
                if not bool(((valid.sum(1)==64)|(valid.sum(1)==128)).all()): raise RuntimeError('unexpected router budget')
                selected=torch.zeros(len(ids),256,dtype=torch.bool,device=device)
                rows=torch.arange(len(ids),device=device)[:,None].expand_as(index)
                selected[rows[valid],index[valid]]=True
                rgb=torch.cat([assets.render(part) for part in base.split(4)])
                if mapper is not None:
                    proxy=torch.cat([mapper.tokens(part) for part in base.split(4)]).long()
                else:
                    # Match the train cache: FP32, clamped float RGB, no uint8 round-trip.
                    with torch.autocast('cuda',enabled=False):
                        proxy=torch.cat([assets.shell.llamagen_vq.encode(part*2-1)[2][2].reshape(len(part),256)
                                         for part in rgb.split(4)]).long()
                torch.manual_seed(args.seed+2*int(ids[0])+1)
                trace=[]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    completed=sample_halton_completion(
                        model, proxy, selected, labels, Path(config['upstream_root']),
                        steps=args.stage2_steps, cfg_w=args.cfg_w, trace=trace)
                if not torch.equal(completed[~selected],proxy[~selected]): raise RuntimeError('anchors changed')
                sparse=completed.gather(1,index.clamp_min(0))
                mixed=assets.mixed(base,sparse,index,valid)
                pixels=dict(base=to_uint8(rgb),full_refine=to_uint8(torch.cat([assets.render(part) for part in mixed.split(4)])))
                if offset==0:
                    np.savez_compressed(out/f'first_shard{shard}.npz',ids=ids,proxy=proxy.cpu().numpy(),
                                        completed=completed.cpu().numpy(),selected=selected.cpu().numpy(),**pixels)
                    atomic_json(out/f'trace_shard{shard}.json',trace)
                for name,images in pixels.items():
                    kwargs={}
                    if not gpu_checked:
                        metadata=tf.RunMetadata()
                        kwargs=dict(options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=metadata)
                    features=session.run(evaluator.pool_features,{evaluator.image_input:images.astype(np.float32)},**kwargs).reshape(len(ids),2048)
                    if not gpu_checked:
                        devices=[d.device for d in metadata.step_stats.dev_stats if 'GPU' in d.device.upper()
                                 and any('conv' in n.node_name.lower() for n in d.node_stats)]
                        if not devices: raise RuntimeError('ADM convolutions not on GPU')
                        atomic_json(out/f'adm_gpu_shard{shard}.json',dict(devices=devices));gpu_checked=True
                    if not np.isfinite(features).all(): raise RuntimeError('nonfinite FID features')
                    pools[name][offset:offset+len(ids)]=features
                completed_count+=len(ids)
                offset+=len(ids)
                atomic_json(status_path,dict(status='running',completed=completed_count,total=len(ids_all),
                    seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
            for pool in pools.values(): pool.flush()
            result=dict(status='complete',shard=shard,n=len(ids_all),ids=ids_all.tolist(),step=step,
                checkpoint_sha256=digest,anchors_unchanged=True,seconds=time.monotonic()-started,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                hashes={name:sha256(out/f'{name}_shard{shard}.npy') for name in arms})
            atomic_json(out/f'shard{shard}.json',result)
            atomic_json(status_path,dict(status='complete',completed=completed_count,total=len(ids_all),
                seconds=time.monotonic()-started,peak_reserved_gib=result['peak_reserved_gib']))
        return  # the CPU coordinator verifies all workers and computes FID


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20260914)
    p.add_argument('--cfg-w',type=float,default=1.5)
    p.add_argument('--stage2-steps',type=int,choices=(1,2,4,6,8,16,32),default=32)
    p.add_argument('--router-mode',choices=('e117','parent-only'),default='e117')
    p.add_argument('--mapper-checkpoint',type=Path)
    p.add_argument('--worker-shard',type=int,choices=range(8),required=True)
    p.add_argument('--worker-count',type=int,choices=(1,2,4,8),default=4)
    args=p.parse_args()
    try: main(args)
    except BaseException as exc:
        if args.output.exists(): atomic_json(args.output/f'failure_shard{args.worker_shard}.json',dict(error=repr(exc)))
        raise
