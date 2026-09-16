"""Download the pinned public6000-step release, outside the Git checkout."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bert2d.paths import RESULT_ROOT, atomic_json, output_path, sha256


def main():
    spec=json.loads((ROOT/'configs/dense_proxy_grid_h20.json').read_text())
    out=output_path(RESULT_ROOT/'input6000')
    out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_HOME',str(RESULT_ROOT/'.hf'))
    os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES','0')
    from huggingface_hub import HfApi, hf_hub_download
    if spec['revision']=='resolve-once':
        # The release may still be uploading. No CUDA context or GPU allocation.
        api=HfApi();begin=time.monotonic()
        while True:
            revision=api.repo_info(spec['repo']).sha
            paths=api.get_paths_info(spec['repo'],[spec['prefix']+'/'+n for n in ('latest.pt','latest.json','config.json')],revision=revision)
            if len(paths)==3:
                weight=next(p for p in paths if p.path.endswith('/latest.pt'))
                assert weight.size==spec['bytes']
                if weight.lfs:assert weight.lfs.sha256==spec['sha256']
                spec['revision']=revision
                break
            atomic_json(out/'download.json',dict(status='waiting_hf_publication',seconds=time.monotonic()-begin,**spec))
            if time.monotonic()-begin>21600:raise TimeoutError('HF publication unavailable after6h; no GPU allocated')
            time.sleep(60)
    assert len(spec['revision'])==40
    for name in ('latest.pt','latest.json','config.json'):
        target=out/name
        if target.exists():
            if name=='latest.pt':
                assert target.stat().st_size==spec['bytes'] and sha256(target)==spec['sha256'], 'existing checkpoint differs; inspect, never overwrite'
            continue
        source=Path(hf_hub_download(repo_id=spec['repo'],revision=spec['revision'],
                    filename=spec['prefix']+'/'+name,local_dir=str(out)))
        if name=='latest.pt':
            assert source.stat().st_size==spec['bytes'] and sha256(source)==spec['sha256']
        source.rename(target)
    meta=json.loads((out/'latest.json').read_text())
    config=json.loads((out/'config.json').read_text())
    assert meta['step']==6000 and meta['config']==config and meta['round_trip_exact']
    assert meta['sha256']==spec['sha256']==sha256(out/'latest.pt')
    atomic_json(out/'download.json',dict(status='verified',**spec))
    print(json.dumps(dict(status='verified',path=str(out),step=6000,sha256=spec['sha256'])))


if __name__=='__main__':main()
