"""Free-generation ADM FID for ordinary MaskGIT with fixed converted anchors.

Four logical shards retain the previous 5k cohort's batching/seeds.
H20 uses one physical GPU per logical shard. Direct replacement is primary;
base is diagnostic. No all-discrete decoding or checkpoint snapshot.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));os.environ.setdefault('USE_TF','0')
# Do this before importing Torch or any vendor module. Even an import-time
# cuda.is_available() can initialize CUDA device enumeration without setting
# torch.cuda.is_initialized(), so isolation inside main is not early enough.
_BOUND_PHYSICAL_GPU=None
if __name__=='__main__':
    _visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    _BOUND_PHYSICAL_GPU=_visible[int(os.getenv('LOCAL_RANK','0'))]
    os.environ['CUDA_VISIBLE_DEVICES']=_BOUND_PHYSICAL_GPU
import numpy as np
import torch
from bert2d.paths import ASSETS,atomic_json,sha256,ONE_D,EVALUATOR
from bert2d.eval import ImageBert,OfficialSamplingView,to_uint8
from h20.assets import official_model
from h20_joint.assets import FrozenAssets
from experiments.feature_to_token_20260916.grid_maskgit import selected_mask,sample_grid
from experiments.feature_to_token_20260916.train_grid import load_grid
from experiments.feature_to_token_20260916.converter import load_converter
from experiments.feature_to_token_20260916.probe import nearest

ARMS=('base','full_refine')


def main(args):
    rank,world,local=(int(os.getenv(k,'1' if k=='WORLD_SIZE' else '0')) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    assert world==4 and args.n in (8,5000)
    # One physical GPU per process for BOTH Torch and TF, including TF's
    # auxiliary eager context used while constructing the ADM distance graph.
    # Session-only visible_device_list remapped rank1 GPU:0 inconsistently.
    assert _BOUND_PHYSICAL_GPU is not None, 'run evaluator as a standalone worker'
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    device=torch.device('cuda',0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.mem_get_info(device)[0]<20*1024**3:raise RuntimeError('20GiB free required')
    out=args.output.resolve();root=Path(os.environ['MOTAR_RESULTS']).resolve()
    if out==root or not out.is_relative_to(root):raise ValueError('named result output required')
    out.mkdir(parents=True,exist_ok=True)
    status=out/f'status_rank{rank}.json'
    if status.exists():raise FileExistsError(status)
    started=time.monotonic();atomic_json(status,dict(status='loading',pid=os.getpid()))
    model,meta=load_grid(args.checkpoint,device)
    core_cfg=meta['config'];conversion=core_cfg['converter'];converter=None
    if conversion['mode']=='learned':
        assert sha256(Path(conversion['path']))==conversion['sha256']
        converter,_=load_converter(Path(conversion['path']),device)
    assets=FrozenAssets(ASSETS,str(device),chunk=4)
    assert assets.identities==core_cfg['feature_audit']['identities']
    one_d,one_d_audit=official_model(ASSETS)
    one_d=one_d.to(device).eval().requires_grad_(False);view=OfficialSamplingView(one_d).eval()
    from bert2d.assets import FID,check
    for item in FID:check(ASSETS/item['local_path'],item)
    spec=importlib.util.spec_from_file_location('grid_adm_evaluator',EVALUATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.INCEPTION_V3_PATH=str(ASSETS/'fid/classify_image_graph_def.pb')
    tf=module.tf;tf.disable_eager_execution();tf.config.experimental.enable_tensor_float_32_execution(False)
    tf_cfg=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=2)
    tf_cfg.gpu_options.allow_growth=True;tf_cfg.gpu_options.visible_device_list='0'
    manifest=dict(n=args.n,rank=rank,world=world,logical_shards=4,batch=8,seed=args.seed,
        physical_gpu=_BOUND_PHYSICAL_GPU,torch_local_device=0,tf_local_device=0,
        checkpoint_sha256=meta['sha256'],step=meta['step'],conversion=conversion,one_d_audit=one_d_audit,
        stage1=ONE_D,stage2=dict(steps=16,cfg=4.5,schedule='cosine',draw='categorical',confidence='logprob',
            choice_temperature=1.,known_tokens_fixed=True),sequence='class+256, full bidirectional BERT',
        endpoints=list(ARMS),full_refine='keep continuous unselected base; directly replace selected positions',
        all_discrete='decode all256 token IDs: fixed proxy anchors plus generated selected tokens',
        no_gt_images_or_2d_used=True,source_sha256=sha256(Path(__file__)),
        reference=str(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz'))
    atomic_json(out/f'manifest_rank{rank}.json',manifest)
    with torch.inference_mode(),tf.Session(config=tf_cfg) as session:
        evaluator=module.Evaluator(session,batch_size=8)
        vq=assets.shell.llamagen_vq
        emb=vq.quantize.get_codebook_entry(torch.arange(16384,device=device))
        book=vq.post_quant_conv(emb.T[None,:,None]).squeeze(0).squeeze(1).T.contiguous()
        for shard in range(rank,4,world):
            ids_all=np.array_split(np.arange(args.n),4)[shard]
            pools={name:np.lib.format.open_memmap(out/f'{name}_shard{shard}.npy',mode='w+',dtype=np.float32,shape=(len(ids_all),2048)) for name in ARMS}
            done=np.zeros(len(ids_all),dtype=bool);traced=False;counts=[]
            for offset in range(0,len(ids_all),8):
                if time.monotonic()-started>3600:raise TimeoutError('60minute FID bound')
                ids=ids_all[offset:offset+8];labels=torch.as_tensor(ids%1000,device=device)
                torch.manual_seed(args.seed+2*int(ids[0]))
                with torch.autocast('cuda',dtype=torch.bfloat16):z1=ImageBert.generate(view,condition=labels,**ONE_D)
                bundle=assets.bundle(z1);base=bundle['base'];index=bundle['index'];valid=bundle['valid']
                selected=selected_mask(index,valid)
                assert bool(((valid.sum(1)==64)|(valid.sum(1)==128)).all())
                proxy=converter.tokens(base) if converter is not None else nearest(base,book)[0]
                torch.manual_seed(args.seed+2*int(ids[0])+1)
                trace=[]
                with torch.autocast('cuda',dtype=torch.bfloat16):completed=sample_grid(model,proxy,selected,labels,trace=trace)
                assert torch.equal(completed[~selected],proxy[~selected])
                sparse=completed.gather(1,index.clamp_min(0))
                images=dict(base=to_uint8(assets.render(base)),
                    full_refine=to_uint8(assets.render(assets.mixed(base,sparse,index,valid))))
                if offset==0:
                    np.savez_compressed(out/f'first_shard{shard}.npz',ids=ids,z1=z1.cpu().numpy(),proxy=proxy.cpu().numpy(),
                        completed=completed.cpu().numpy(),selected=selected.cpu().numpy(),**images)
                    atomic_json(out/f'sampling_trace_shard{shard}.json',trace)
                for name,pixels in images.items():
                    kwargs={}
                    if not traced:
                        metadata=tf.RunMetadata();kwargs=dict(options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=metadata)
                    values=session.run(evaluator.pool_features,{evaluator.image_input:pixels.astype(np.float32)},**kwargs).reshape(len(ids),2048)
                    if not traced:
                        devices=[d.device for d in metadata.step_stats.dev_stats if 'GPU' in d.device.upper() and any('conv' in node.node_name.lower() for node in d.node_stats)]
                        assert devices,'FID extraction did not run on GPU'
                        atomic_json(out/f'adm_gpu_shard{shard}.json',dict(convolution_devices=devices));traced=True
                    assert np.isfinite(values).all();pools[name][offset:offset+len(ids)]=values
                done[offset:offset+len(ids)]=True;counts.extend(valid.sum(1).cpu().tolist())
                atomic_json(status,dict(status='running',pid=os.getpid(),shard=shard,completed=int(done.sum()),
                    total=len(ids_all),seconds=time.monotonic()-started))
            assert done.all()
            for pool in pools.values():pool.flush()
            atomic_json(out/f'shard{shard}.json',dict(status='complete',ids=ids_all.tolist(),mean_k=float(np.mean(counts)),
                hashes={name:sha256(out/f'{name}_shard{shard}.npy') for name in ARMS}))
        atomic_json(status,dict(status='complete',pid=os.getpid(),seconds=time.monotonic()-started))
        if rank!=0:return
        while any(not (out/f'shard{s}.json').exists() for s in range(4)):
            if time.monotonic()-started>3600:raise TimeoutError('merge deadline')
            time.sleep(2)
        ids=sum([json.loads((out/f'shard{s}.json').read_text())['ids'] for s in range(4)],[])
        assert ids==list(range(args.n))
        metrics={}
        with np.load(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz') as reference:
            ref=module.FIDStatistics(reference['mu'],reference['sigma'])
            for name in ARMS:
                features=np.concatenate([np.load(out/f'{name}_shard{s}.npy') for s in range(4)])
                stats=evaluator.compute_statistics(features)
                metrics[name]=dict(fid=float(stats.frechet_distance(ref))) if args.n==5000 else dict(smoke_only=True)
        atomic_json(out/'summary.json',dict(status='complete',n=args.n,metrics=metrics,checkpoint_sha256=meta['sha256'],
            step=meta['step'],generated_1d_prefix=True,anchors_unchanged=True,full_class_balance=args.n==5000,
            logical_shards=4,seconds=time.monotonic()-started,smoke=args.n!=5000))
        print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20260914)
    a=p.parse_args()
    try:main(a)
    except BaseException as exc:
        if a.output.exists():atomic_json(a.output/f'failure_rank{os.getenv("RANK","0")}.json',dict(error=repr(exc),pid=os.getpid()))
        raise
