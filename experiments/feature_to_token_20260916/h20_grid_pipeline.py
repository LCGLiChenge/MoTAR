"""Authorized8-GPU6000->20000 continuation; paused4-GPU5k every4000."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bert2d.paths import ASSETS, atomic_json, sha256, output_path


def read(path): return json.loads(Path(path).read_text())


def validate_eval(folder, checkpoint, n, step):
    summary, meta = read(folder/'summary.json'), read(checkpoint/'latest.json')
    assert summary['status'] == 'complete' and summary['n'] == n and summary['step'] == step
    assert summary['checkpoint_sha256'] == meta['sha256'] == sha256(checkpoint/'latest.pt')
    assert summary['generated_1d_prefix'] and summary['anchors_unchanged']
    assert summary['smoke'] == (n != 5000)
    assert set(summary['metrics']) == {'base', 'full_refine'}
    ids = []
    for rank in range(4):
        assert read(folder/f'status_rank{rank}.json')['status'] == 'complete'
        manifest = read(folder/f'manifest_rank{rank}.json')
        assert manifest['seed'] == 20260914 and manifest['batch'] == 8
        assert manifest['checkpoint_sha256'] == meta['sha256'] and manifest['step'] == step
        assert manifest['stage2']['steps'] == 16 and manifest['stage2']['cfg'] == 4.5
        assert manifest['no_gt_images_or_2d_used']
        assert read(folder/f'adm_gpu_shard{rank}.json')['convolution_devices']
        shard = read(folder/f'shard{rank}.json')
        ids.extend(shard['ids'])
        for arm, value in shard['hashes'].items():
            assert sha256(folder/f'{arm}_shard{rank}.npy') == value
    assert ids == list(range(n))
    return summary


def run(a):
    out, source = output_path(a.output), a.resume.resolve()
    out.mkdir(exist_ok=False, parents=True)
    progress = out/'pipeline.json'
    def record(**kw): atomic_json(progress, dict(pid=os.getpid(), **kw))
    record(status='preflight', target=20000, global_batch=448, eval_steps=[8000,12000,16000,20000])
    initial = read(source/'latest.json')
    assert initial['step'] == 6000 and initial['sha256'] == sha256(source/'latest.pt')
    from bert2d.assets import verify
    verify(ASSETS, include_fid=True)
    manifest = read(Path(__file__).with_name('h20_asset_manifest.json'))
    for item in manifest['files']:
        path = ASSETS/item['path']
        assert path.stat().st_size == item['bytes'] and sha256(path) == item['sha256'], item['path']
    atomic_json(out/'asset_verification.json', dict(status='passed', files=manifest['files'], assets=str(ASSETS)))
    def execute(stage, script, arguments, world, limit):
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free','--format=csv,noheader,nounits'], text=True)
        rows = [r.split(',') for r in rows.strip().splitlines()]
        assert len(rows) == 8 and all('H20' in r[1] for r in rows)
        assert min(int(r[2]) for r in rows[:world]) >= 40*1024
        command = [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
                   '--',f'experiments/feature_to_token_20260916/{script}'] + arguments
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(world)),
                   USE_TF='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True,start_new_session=True)
        record(status='running',stage=stage,child_pid=child.pid,gpus=list(range(world)),command=command)
        begin = time.monotonic()
        try: stdout,stderr = child.communicate(timeout=limit)
        except BaseException:
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
            raise
        atomic_json(out/f'{stage}_process.json', dict(returncode=child.returncode,seconds=time.monotonic()-begin,
                    command=command,stdout_tail=stdout[-16000:],stderr_tail=stderr[-16000:]))
        assert child.returncode == 0, stage
    def train(target, resume):
        execute(f'train{target}','continue_grid_h20.py',['--resume',str(resume),'--output',str(out/'train'),
                '--target',str(target)],8,15000)
        meta = read(out/'train/latest.json')
        assert meta['step'] == target and meta['round_trip_exact']
        assert meta['sha256'] == sha256(out/'train/latest.pt')
        for rank in range(8): assert read(out/f'train/train_rank{rank}.json')['status'] == 'complete'
        return meta
    def evaluate(step,n):
        folder = out/(f'fid5k_step{step}' if n == 5000 else 'smoke8')
        execute(f'eval{step}_{n}','eval_grid_h20.py',['--checkpoint',str(out/'train'),'--output',str(folder),
                '--n',str(n),'--seed','20260914'],4,3900)
        return validate_eval(folder,out/'train',n,step)
    train(6002,source)
    evaluate(6002,8)
    # Only smoke feature arrays are disposable; keep tokens/previews + JSON audit.
    removed = []
    for rank in range(4):
        for arm in ('base','full_refine'):
            path = out/'smoke8'/f'{arm}_shard{rank}.npy'
            removed.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path)))
    atomic_json(out/'smoke8/cleanup.json',dict(status='planned',files=removed))
    for item in removed: Path(item['path']).unlink()
    assert all(not Path(item['path']).exists() for item in removed)
    atomic_json(out/'smoke8/cleanup.json',dict(status='complete',files=removed))
    points = []
    for target in (8000,12000,16000,20000):
        meta = train(target,out/'train')
        result = evaluate(target,5000)
        points.append(result)
        atomic_json(out/'fid_trend.json',dict(status='complete' if target==20000 else 'partial',points=points,
                    primary='full_refine',checkpoint_policy='rolling latest, no eval snapshot',
                    source6000_sha256=initial['sha256']))
        if target == 8000:
            # Only the transferred input copy is removed; original5090 source remains.
            assert source.name == 'input6000' and source.parent == out.parent
            assert source != out/'train'
            path = source/'latest.pt'
            assert sha256(path) == initial['sha256']
            cleanup = dict(path=str(path),bytes=path.stat().st_size,sha256=initial['sha256'],
                           reason='remote staging copy; successful8k save/resume/eval; source5090 retained')
            atomic_json(out/'input_cleanup.json',dict(status='planned',**cleanup))
            path.unlink()
            assert not path.exists()
            atomic_json(out/'input_cleanup.json',dict(status='complete',**cleanup))
    record(status='complete',step=20000,summary=str(out/'fid_trend.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): atomic_json(a.output/'pipeline.json',dict(status='failed',error=repr(exc),pid=os.getpid()))
        raise
