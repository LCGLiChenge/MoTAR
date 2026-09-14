"""Free-generation smoke/local5k for Halton sparse 2D-only checkpoints."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

from experiments.maskgit_optimization_sweep_20260910.common import ASSETS, RESULT_ROOT, ROOT, atomic_json
from experiments.maskgit_optimization_sweep_20260910.fid50k import EVALUATOR, REFERENCE, GRAPH, ONE_D, digest
from experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_adapter import HaltonSparse2DAdapter, HaltonFullContextSparse2DAdapter, HaltonBaseContextSparse2DAdapter
from experiments.maskgit_optimization_sweep_20260910.sampling import BASE as SAMPLE_BASE, VARIANTS as SAMPLE_VARIANTS
from experiments.maskgit_optimization_sweep_20260910.sampling_followup import VARIANTS as FOLLOWUP_VARIANTS
from experiments.maskgit_optimization_sweep_20260910.sampling_followup import generate as sample_generate
from h20.model import OfficialSamplingView
from h20_joint.assets import FrozenAssets

for _titok_root in (ROOT / "third_party/TiTok", ROOT / "delivery_h20_20260910/third_party/TiTok"):
    if (_titok_root / "modeling/maskgit.py").exists():
        sys.path.insert(0, str(_titok_root))
        break
from modeling.maskgit import ImageBert  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def variant_map():
    values = {cfg.name: cfg for cfg in SAMPLE_VARIANTS}
    for cfg in FOLLOWUP_VARIANTS:
        values.setdefault(cfg.name, cfg)
    values.setdefault("baseline", SAMPLE_BASE)
    return values


def load_model(checkpoint_dir: Path, state: str, device: str):
    checkpoint_dir = checkpoint_dir.resolve()
    meta = json.loads((checkpoint_dir / "latest.json").read_text())
    config = json.loads((checkpoint_dir / "config.json").read_text())
    style = config.get("style")
    require(style in {"halton_sparse2d", "halton_fullctx_sparse2d", "halton_basectx_sparse2d"}, "checkpoint is not a supported sparse 2D trial")
    require(meta["sha256"] == digest(checkpoint_dir / "latest.safetensors"), "checkpoint sha mismatch")
    model = ({"halton_sparse2d": HaltonSparse2DAdapter, "halton_fullctx_sparse2d": HaltonFullContextSparse2DAdapter, "halton_basectx_sparse2d": HaltonBaseContextSparse2DAdapter}[style])()
    prefix = state + "/"
    with safe_open(str(checkpoint_dir / "latest.safetensors"), framework="pt", device="cpu") as reader:
        tensors = {key[len(prefix):]: reader.get_tensor(key) for key in reader.keys() if key.startswith(prefix)}
    missing, unexpected = model.load_state_dict(tensors, strict=True)
    require(not missing and not unexpected, "state load accounting failed")
    audit = dict(step=meta["step"], sha256=meta["sha256"], path=str(checkpoint_dir), state=state, model_class=model.__class__.__module__ + "." + model.__class__.__name__)
    return model.to(device).eval().requires_grad_(False), audit


def to_uint8(image01: torch.Tensor) -> np.ndarray:
    require(tuple(image01.shape[1:]) == (3, 256, 256), "invalid image shape")
    require(bool(torch.isfinite(image01).all()), "nonfinite decoded image")
    return image01.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def main(args):
    output = args.output.resolve()
    require(output.is_relative_to(RESULT_ROOT.resolve()) and output != RESULT_ROOT.resolve(), "output must be under approved result root")
    require(not output.exists(), "fresh output required")
    max_n = 50000 if args.allow_50k else 5000
    require(1 <= args.n <= max_n and 1 <= args.batch <= 32 and 1 <= args.feature_batch <= 32, "bounded local5k/explicit 50k only")
    if args.n > 5000:
        require(args.allow_50k and args.n == 50000, "50k evaluation requires --allow-50k and --n 50000")
    if args.smoke:
        require(args.n <= 64, "smoke capped at64")
    else:
        require(args.n >= 4096, "FID screen requires >=4096 samples; use --smoke for smaller tests")
    variants = variant_map()
    chosen = args.variants or ["margin4", "greedy_margin4_cfg1"]
    for name in chosen:
        require(name in variants, "unknown variant: " + name)
    endpoints = ["base"] + chosen
    output.mkdir(parents=True)

    model, model_audit = load_model(args.checkpoint, args.state, args.device)
    one_d_core = HaltonSparse2DAdapter().to(args.device).eval().requires_grad_(False)
    # The 1D generator must remain the official TiTok path, not this adapter.
    del one_d_core
    assets = FrozenAssets(ASSETS, args.device, chunk=args.feature_chunk)

    manifest = dict(
        format="halton_sparse2d_free_generation_local5k_v1",
        n=args.n,
        smoke=args.smoke,
        seed=args.seed,
        state=args.state,
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_audit=model_audit,
        variants=chosen,
        endpoints=endpoints,
        stage1=ONE_D,
        labels="global_sample_id % 1000",
        direct_replace=True,
        selected_sparse_2d_only=True,
        evaluator=str(EVALUATOR),
        reference=str(REFERENCE),
        graph=str(GRAPH),
        source_hashes={
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            str(ROOT / "experiments/maskgit_optimization_sweep_20260910/halton_sparse2d_adapter.py"): digest(ROOT / "experiments/maskgit_optimization_sweep_20260910/halton_sparse2d_adapter.py"),
            str(ROOT / "experiments/maskgit_optimization_sweep_20260910/sampling_followup.py"): digest(ROOT / "experiments/maskgit_optimization_sweep_20260910/sampling_followup.py"),
        },
        created_at=time.strftime("%Y-%m-%d %H:%M:%S %z"),
    )
    atomic_json(output / "manifest.json", manifest)

    spec = importlib.util.spec_from_file_location("mix_style_free_adm_eval", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INCEPTION_V3_PATH = str(GRAPH)
    tf = module.tf
    tf.disable_eager_execution()
    require(bool(tf.config.list_physical_devices("GPU")), "ADM GPU unavailable")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8, inter_op_parallelism_threads=2)
    config.gpu_options.allow_growth = True

    # Reuse the official TiTok MaskGIT 1D path from the portable MoTAR assets.
    from h20.assets import official_model

    one_d, one_d_report = official_model(ASSETS)
    one_d = one_d.to(args.device).eval().requires_grad_(False)
    view = OfficialSamplingView(one_d).eval()
    manifest["official_1d_report"] = one_d_report
    atomic_json(output / "manifest.json", manifest)

    torch.set_num_threads(4)
    features = {name: np.lib.format.open_memmap(output / (name + "_pool.npy"), mode="w+", dtype=np.float32, shape=(args.n, 2048)) for name in endpoints}
    written = np.zeros(args.n, dtype=bool)
    route_counts = np.zeros(args.n, dtype=np.int64)
    traced_gpu = False
    started = time.monotonic()
    z1_min, z1_max = 10**9, -1
    with tf.Session(config=config) as session:
        evaluator = module.Evaluator(session, batch_size=args.feature_batch)
        for start in range(0, args.n, args.batch):
            end = min(start + args.batch, args.n)
            ids = np.arange(start, end)
            labels = torch.as_tensor(ids % 1000, device=args.device)
            torch.manual_seed(args.seed + 2 * start)
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                z1 = ImageBert.generate(view, condition=labels, **ONE_D)
            require(z1.shape == (len(ids), 32), "invalid 1D generation shape")
            z1_min = min(z1_min, int(z1.min().item()))
            z1_max = max(z1_max, int(z1.max().item()))
            bundle = assets.bundle(z1)
            base, index, valid = bundle["base"], bundle["index"], bundle["valid"]
            counts = valid.sum(1)
            require(bool(((counts == 64) | (counts == 128)).all()), "unexpected route count")
            frozen_z1 = z1.detach().clone()

            def provider(query, frozen=frozen_z1, feat=base):
                if not torch.equal(query, frozen):
                    raise ValueError("wrong z1 bound to feature provider")
                return feat

            model.set_feature_provider(provider)
            images = {"base": to_uint8(assets.render(base))}
            for vi, variant in enumerate(chosen):
                torch.manual_seed(args.seed + 2 * start + 1 + 104729 * vi)
                with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    z2 = sample_generate(model, z1, index, valid, labels, variants[variant])
                images[variant] = to_uint8(assets.render(assets.mixed(base, z2, index, valid)))
            if start == 0:
                np.savez_compressed(output / "first_batch_debug.npz", endpoints=np.array(endpoints), ids=ids[:8], z1d=z1[:8].detach().cpu().numpy().astype(np.uint16), index=index[:8].detach().cpu().numpy().astype(np.uint16), valid=valid[:8].detach().cpu().numpy())
            for name, pixels in images.items():
                chunks = []
                for offset in range(0, len(ids), args.feature_batch):
                    stop = min(offset + args.feature_batch, len(ids))
                    kwargs = {}
                    if not traced_gpu:
                        trace = tf.RunMetadata()
                        kwargs = dict(options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=trace)
                    value = session.run(evaluator.pool_features, {evaluator.image_input: pixels[offset:stop].astype(np.float32)}, **kwargs).reshape(stop - offset, -1)
                    if not traced_gpu:
                        devices = [d.device for d in trace.step_stats.dev_stats if "GPU" in d.device.upper() and any("conv" in node.node_name.lower() for node in d.node_stats)]
                        require(devices, "ADM convolutions did not execute on GPU")
                        atomic_json(output / "adm_gpu_trace.json", dict(convolution_devices=devices, tensorflow=tf.__version__))
                        traced_gpu = True
                    require(value.shape == (stop - offset, 2048) and value.dtype == np.float32 and np.isfinite(value).all(), "invalid ADM features")
                    chunks.append(value)
                features[name][ids] = np.concatenate(chunks, axis=0)
            for arr in features.values():
                arr.flush()
            written[ids] = True
            route_counts[ids] = counts.cpu().numpy()
            progress = dict(status="running", completed=int(written.sum()), n=args.n, seconds=time.monotonic() - started, mean_k=float(route_counts[written].mean()), gpu_features_verified=traced_gpu)
            atomic_json(output / "progress.json", progress)
            if progress["completed"] % 256 == 0 or progress["completed"] == args.n:
                print(json.dumps(progress), flush=True)
        metrics = {}
        score_fid = args.n > 2048
        with np.load(REFERENCE, allow_pickle=False) as reference:
            ref = module.FIDStatistics(reference["mu"], reference["sigma"])
            for name in endpoints:
                stats = evaluator.compute_statistics(features[name])
                row = dict(feature_mean_l2_to_reference=float(np.linalg.norm(stats.mu - reference["mu"])))
                if score_fid:
                    row["fid"] = float(stats.frechet_distance(ref))
                    row["inception_score"] = float(evaluator.compute_inception_score(features[name], split_size=max(1, min(5000, args.n))))
                metrics[name] = row
                np.savez_compressed(output / (name + "_statistics.npz"), mu=stats.mu, sigma=stats.sigma)
                atomic_json(output / "partial_metrics.json", metrics)
        label_counts = np.bincount(np.arange(args.n) % 1000, minlength=1000)
        summary = dict(status="complete", n=args.n, smoke=args.smoke, metrics=metrics, endpoints=endpoints, complete_coverage=bool(written.all()), generated_prefix=True, complete_free_generation_fid=score_fid, fid_skipped_reason=None if score_fid else "smoke or n<=2048", gpu_features_verified=traced_gpu, full_class_balance=bool(np.all(label_counts == (args.n // 1000))) if args.n % 1000 == 0 else False, mean_k=float(route_counts.mean()), z1_min=z1_min, z1_max=z1_max, runtime_seconds=time.monotonic() - started, feature_hashes={name + "_pool": digest(output / (name + "_pool.npy")) for name in endpoints})
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "progress.json", dict(status="complete", completed=args.n, n=args.n, seconds=time.monotonic() - started))
        print(json.dumps(dict(status="complete", output=str(output), metrics=metrics), sort_keys=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state", choices=("raw", "ema"), default="raw")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--feature-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--device", default="cuda")
    p.add_argument("--feature-chunk", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--variants", nargs="*")
    p.add_argument("--allow-50k", action="store_true", help="explicitly permit a full 50k ImageNet FID run")
    main(p.parse_args())
