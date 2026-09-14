"""Explicit-GPU memory probe + save/resume gate + 40-epoch latest-only training."""
import argparse,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from .paths import ASSETS,RESULT_ROOT,ROOT,atomic_json,output_path
from .assets import verify
def run(cmd,env):subprocess.run(cmd,cwd=ROOT,env=env,check=True)
def main(a):
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    gpus=[s.strip() for s in visible.split(",") if s.strip()]
    if len(gpus) not in (1,2,4,8) or len(set(gpus))!=len(gpus):
        raise ValueError("set explicitly authorized CUDA_VISIBLE_DEVICES (1/2/4/8 distinct GPUs)")
    n=len(gpus)
    if a.epochs<1 or a.global_batch<1 or a.micro<0:raise ValueError("positive epoch/global batch and nonnegative micro required")
    if a.global_batch%n:raise ValueError("global batch must divide number of GPUs")
    if a.micro and a.global_batch%(a.micro*n):raise ValueError("micro*world must divide global batch")
    out=output_path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out.parent).free<20*1024**3:raise RuntimeError("at least 20 GiB free output space required before startup")
    env=os.environ.copy();env.update(USE_TF="0",PYTORCH_ALLOC_CONF="expandable_segments:True")
    if not a.skip_asset_verify:verify(a.assets_root)
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
    free={}
    for row in rows.strip().splitlines():
        idx,uid,value=[x.strip() for x in row.split(",")]
        free[idx]=int(value);free[uid]=int(value)
    if any(g not in free for g in gpus):raise ValueError("GPU identifiers must be nvidia-smi index or full UUID")
    limited=min(gpus,key=lambda x:free[x]);limit=free[limited]*.92
    resume=out if (out/"latest.pt").exists() else None
    if out.exists() and resume is None:raise FileExistsError("existing output has no checkpoint; inspect before reusing")
    if resume:
        old=json.loads((resume/"config.json").read_text())
        micro=old["micro"]
        if old["world"]!=n or old["global_batch"]!=a.global_batch:raise ValueError("resume DDP layout changed")
        if a.micro and a.micro!=micro:raise ValueError("resume microbatch changed")
    else:micro=a.micro
    candidates=[micro] if micro else [v for v in range(a.global_batch//n,0,-1) if a.global_batch%(v*n)==0]
    probe_dir=Path(tempfile.mkdtemp(prefix="bert2d_startup_",dir=out.parent))
    chosen=None
    for value in candidates:
        result=probe_dir/f"memory_m{value}.json"
        probe_env=dict(env,CUDA_VISIBLE_DEVICES=limited)
        cmd=[sys.executable,"-m","bert2d.probe","--micro",str(value),"--output",str(result),
             "--assets-root",str(a.assets_root),"--recompute-layers",str(a.recompute_layers),"--limit-mib",str(limit)]
        rc=subprocess.run(cmd,cwd=ROOT,env=probe_env).returncode
        if rc not in (0,2):raise RuntimeError("capacity probe failed; not a recoverable capacity result")
        row=json.loads(result.read_text())
        if row["status"]=="ok":chosen=value;break
    if chosen is None:raise RuntimeError("no allowed microbatch fits; no training started")
    base=[sys.executable,"-m","torch.distributed.run","--standalone",f"--nproc_per_node={n}","-m","bert2d.train"]
    common=["--micro",str(chosen),"--global-batch",str(a.global_batch),"--assets-root",str(a.assets_root),
            "--recompute-layers",str(a.recompute_layers)]
    if not resume:
        smoke=probe_dir/"save_resume_smoke"
        run(base+common+["--output",str(smoke),"--smoke-updates","2","--wandb-mode","disabled"],env)
        run(base+common+["--output",str(smoke),"--resume",str(smoke),"--smoke-updates","3","--wandb-mode","disabled"],env)
        report=json.loads((smoke/"latest.json").read_text())
        if report["step"]!=3 or not report["round_trip_exact"]:raise RuntimeError("save/resume gate failed")
        weight=smoke/"latest.pt"
        if weight.is_symlink() or weight.resolve().parent!=smoke.resolve():raise ValueError("unsafe smoke cleanup")
        atomic_json(probe_dir/"cleanup.json",dict(path=str(weight),sha256=report["sha256"],bytes=weight.stat().st_size,status="deleting"))
        weight.unlink()
        atomic_json(probe_dir/"cleanup.json",dict(path=str(weight),sha256=report["sha256"],status="deleted_after_verified_resume"))
    # Probe/smoke allocated only the explicitly requested GPUs. Recheck at the handoff boundary.
    now=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
    for row in now.strip().splitlines():
        idx,uid,value=[x.strip() for x in row.split(",")]
        if (idx in gpus or uid in gpus) and int(value)<row_peak(row=None,probe_dir=probe_dir,micro=chosen):
            raise RuntimeError("GPU memory availability changed before training")
    cmd=base+common+["--output",str(out),"--epochs",str(a.epochs),
        "--wandb-project",a.wandb_project,"--wandb-mode",a.wandb_mode]
    if resume:cmd+=["--resume",str(resume)]
    atomic_json(probe_dir/"launch.json",dict(command=cmd,global_batch=a.global_batch,micro=chosen,world=n,epochs=a.epochs))
    run(cmd,env)
def row_peak(row,probe_dir,micro):
    d=json.loads((probe_dir/f"memory_m{micro}.json").read_text())
    return d["peak_reserved_mib"]+1536
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=RESULT_ROOT/"bert_sparse2d_40epoch")
    p.add_argument("--assets-root",type=Path,default=ASSETS)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--global-batch",type=int,default=448)
    p.add_argument("--micro",type=int,default=0,help="0 probes divisors while preserving global batch")
    p.add_argument("--recompute-layers",type=int,default=10)
    p.add_argument("--wandb-project",default="motar-bert-sparse2d")
    p.add_argument("--wandb-mode",choices=("online","offline","disabled"),default="online")
    p.add_argument("--skip-asset-verify",action="store_true",help="only after explicit separate successful verification")
    main(p.parse_args())
