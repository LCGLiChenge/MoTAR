"""Train a small f_1d -> RGB-reencoded LlamaGen proxy-token mapper.

The target is deliberately *not* the original-image LlamaGen code.  It is the
deterministic 256-token cache produced by cache_rgb_proxy_tokens.py from the
same 1D base reconstruction.  Source images, rather than augmentations, own
the train/eval split so that the two ADM views cannot leak across the split.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from bert2d.paths import ASSETS, atomic_json, output_path, sha256
from experiments.feature_to_token_20260916.converter import FeatureTokenConverter
from experiments.feature_to_token_20260916.converter_lowrank import LowRankFeatureTokenConverter
from experiments.feature_to_token_20260916.probe import nearest
from h20_joint.assets import FrozenAssets


FORMAT = "feature_proxy_token_converter_v1"
PROXY_FORMAT = "mot199440_1d_base_rgb_reencoded_proxy_v1"
NUM_SOURCES = 1_281_167
NUM_AUG = 2
VOCABULARY = 16_384


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_split(seed: int, eval_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    cutoff = int(round(eval_fraction * 10_000))
    require(1 <= cutoff <= 9_999, "eval fraction must map to [1,9999]/10000")
    source = np.arange(NUM_SOURCES, dtype=np.uint64)
    hashed = source * np.uint64(6364136223846793005) + np.uint64(seed)
    is_eval = (hashed % np.uint64(10_000)) < np.uint64(cutoff)
    return np.flatnonzero(~is_eval).astype(np.int64), np.flatnonzero(is_eval).astype(np.int64)


def flat_rows(source: np.ndarray) -> np.ndarray:
    result = np.empty(len(source) * NUM_AUG, dtype=np.int64)
    result[0::2] = source * NUM_AUG
    result[1::2] = source * NUM_AUG + 1
    return result


def load_data_contract(assets_root: Path, proxy_root: Path):
    codes_root = assets_root / "codes/train"
    meta_path = codes_root / "meta.json"
    meta = json.loads(meta_path.read_text())
    require(meta.get("completed") is True, "packed cache is incomplete")
    require(int(meta.get("num_samples", -1)) == NUM_SOURCES, "wrong source count")
    require(int(meta.get("num_aug", -1)) == NUM_AUG, "two ADM views required")
    require(int(meta.get("mot_step", -1)) == 199440, "MoT step 199440 required")
    require(meta.get("mot_state_key") == "model_ema", "MoT EMA cache required")
    z1 = np.load(codes_root / "titok_codes.npy", mmap_mode="r", allow_pickle=False)
    written = np.load(codes_root / "written.npy", mmap_mode="r", allow_pickle=False)
    require(z1.shape == (NUM_SOURCES, NUM_AUG, 32) and z1.dtype == np.uint16,
            "bad TiTok cache layout")
    require(written.shape == (NUM_SOURCES,) and written.dtype == np.uint8
            and bool((written == 1).all()), "packed TiTok cache is incomplete")

    proxy_config_path = proxy_root / "config.json"
    proxy_summary_path = proxy_root / "summary.json"
    proxy_config = json.loads(proxy_config_path.read_text())
    proxy_summary = json.loads(proxy_summary_path.read_text())
    require(proxy_config.get("format") == PROXY_FORMAT, "wrong proxy semantics")
    require(proxy_config.get("flat_layout") == "flat_id=source_index*num_aug+augmentation",
            "wrong proxy indexing")
    require(proxy_summary.get("status") == "complete"
            and proxy_summary.get("complete_coverage") is True,
            "proxy cache is incomplete")
    require(proxy_summary.get("proxy_codes_sha256"), "proxy cache has no final SHA256")
    proxy = np.load(proxy_root / "proxy_codes.npy", mmap_mode="r", allow_pickle=False)
    require(proxy.shape == (NUM_SOURCES * NUM_AUG, 256) and proxy.dtype == np.uint16,
            "bad proxy cache layout")
    require(0 <= int(proxy_summary["minimum_id"])
            <= int(proxy_summary["maximum_id"]) < VOCABULARY,
            "proxy token range is invalid")
    audit = {
        "packed_meta_sha256": sha256(meta_path),
        "proxy_config_sha256": sha256(proxy_config_path),
        "proxy_summary_sha256": sha256(proxy_summary_path),
        "proxy_codes_sha256": proxy_summary["proxy_codes_sha256"],
    }
    return z1, proxy, audit


def make_features(ids: torch.Tensor, quantizer, decoder) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(ids.device.type, enabled=False):
        quantized = quantizer.get_codebook_entry(ids.reshape(-1))
        quantized = quantized.reshape(len(ids), 1, 32, -1).permute(0, 3, 1, 2).contiguous()
        features = decoder(quantized.float())
    require(features.shape == (len(ids), 256, 16, 16), "unexpected f_1d shape")
    return features.detach()


@torch.no_grad()
def evaluate(core, z1, proxy, rows: np.ndarray, batch: int, device, quantizer, decoder,
             rank: int, world: int) -> dict:
    was_training = core.training
    core.eval()
    local_rows = rows[rank::world]
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for offset in range(0, len(local_rows), batch):
        flat = local_rows[offset:offset + batch]
        source, augmentation = np.divmod(flat, NUM_AUG)
        ids = torch.from_numpy(np.array(z1[source, augmentation], dtype=np.int64)).to(device)
        target = torch.from_numpy(np.array(proxy[flat], dtype=np.int64)).to(device).reshape(-1, 16, 16)
        features = make_features(ids, quantizer, decoder)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = core(features)
        loss = F.cross_entropy(logits.float(), target, reduction="sum")
        correct = (logits.argmax(1) == target).sum()
        totals += torch.tensor([float(loss), int(correct), target.numel()],
                               dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(totals)
    if was_training:
        core.train()
    return {"nll": float(totals[0] / totals[2]),
            "token_accuracy": float(totals[1] / totals[2]),
            "tokens": int(totals[2])}


def save_checkpoint(out: Path, core, optimizer, step: int, config: dict, smoke: bool) -> dict:
    state = {"format": config["format"], "model_config": core.config,
             "model": {key: value.detach().cpu() for key, value in core.state_dict().items()},
             "step": step, "train_config": config, "optimizer": optimizer.state_dict()}
    temporary = out / "latest.pt.tmp"
    torch.save(state, temporary)
    loaded = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
    require(loaded["format"] == config["format"] and loaded["step"] == step, "checkpoint header mismatch")
    require(set(loaded["model"]) == set(state["model"]), "checkpoint keys mismatch")
    for name, value in state["model"].items():
        require(torch.equal(value, loaded["model"][name]), f"checkpoint tensor mismatch: {name}")
    receipt = {"step": step, "sha256": sha256(temporary),
               "bytes": temporary.stat().st_size, "round_trip_exact": True}
    if smoke:
        temporary.unlink()
        receipt["smoke_checkpoint_removed"] = True
    else:
        temporary.replace(out / "latest.pt")
        atomic_json(out / "latest.json", receipt)
    return receipt


def run(args) -> None:
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    require(world > 0 and 0 <= rank < world,
            "invalid distributed rank; do not use RANK for the mapper factor rank")
    require(args.steps > 0 and args.batch > 0 and args.eval_batch > 0, "positive sizes required")
    require(args.eval_sources > 0, "positive eval source count required")
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    require("H20" in torch.cuda.get_device_name(device), "this pilot expects an H20")
    free, total_memory = torch.cuda.mem_get_info(device)
    require(free >= args.min_free_gib * 1024**3, "insufficient free H20 memory")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=15))

    out = output_path(args.output)
    if rank == 0:
        require(not out.exists(), f"refusing to overwrite experiment: {out}")
        out.mkdir(parents=True)
    if world > 1:
        dist.barrier()
    status = out / f"status_rank{rank}.json"
    started = time.monotonic()
    atomic_json(status, {"status": "loading", "pid": os.getpid(), "rank": rank})

    z1, proxy, data_audit = load_data_contract(args.assets_root, args.proxy_root)
    train_sources, eval_sources = source_split(args.split_seed, args.eval_fraction)
    require(args.eval_sources <= len(eval_sources), "requested held-out set is too large")
    eval_rng = np.random.default_rng(args.eval_seed)
    chosen_eval_sources = np.sort(eval_rng.choice(eval_sources, args.eval_sources, replace=False))
    eval_rows = flat_rows(chosen_eval_sources)
    train_rows = flat_rows(train_sources)
    sample_count = args.steps * args.batch * world
    require(sample_count <= len(train_rows), "pilot requires no-replacement training rows")
    order = np.random.default_rng(args.seed).choice(train_rows, sample_count, replace=False)

    frozen = FrozenAssets(args.assets_root, "cpu", chunk=args.batch)
    quantizer = frozen.native.quantize.to(device)
    decoder = frozen.shell.latent_decoder.to(device)
    projection = frozen.shell.llamagen_vq.post_quant_conv
    with torch.no_grad():
        embedding = frozen.shell.llamagen_vq.quantize.get_codebook_entry(torch.arange(VOCABULARY))
        book = projection(embedding.T[None, :, None]).squeeze(0).squeeze(1).T.contiguous().to(device)
    if args.model == "full":
        core = FeatureTokenConverter().to(device)
        model_format = FORMAT
    else:
        core = LowRankFeatureTokenConverter(rank=args.rank).to(device)
        model_format = "feature_proxy_token_converter_lowrank_v1"
    core.initialize_nearest(book)
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    model = DDP(core, device_ids=[local], broadcast_buffers=False) if world > 1 else core

    config = {
        "format": model_format, "steps": args.steps, "micro_batch": args.batch,
        "global_batch": args.batch * world, "world": world, "seed": args.seed,
        "lr": args.lr, "warmup": args.warmup, "weight_decay": 0.01,
        "target": "same f_1d decoded to base RGB then frozen MoT199440 LlamaGen encoder; dense 256 proxy IDs",
        "input": "f_1d from frozen TiTok L32 codes through MoT199440 EMA latent_decoder",
        "loss": "equal-weight hard cross entropy on all 256 proxy positions",
        "split": "source-disjoint integer hash; both ADM views stay together",
        "split_seed": args.split_seed, "eval_fraction": args.eval_fraction,
        "eval_seed": args.eval_seed, "eval_sources": args.eval_sources,
        "eval_rows_sha256": hashlib.sha256(eval_rows.tobytes()).hexdigest(),
        "train_rows_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "train_samples": sample_count, "precision": "frozen FP32 noTF32; mapper BF16",
        "init": ("exact projected-codeword squared-distance logits" if args.model == "full" else
                 f"rank-{args.rank} truncated SVD of projected-codeword logits") +
                " plus zero-init spatial residual",
        "parameters": sum(parameter.numel() for parameter in core.parameters()),
        "model_config": core.config, "frozen_audit": frozen.audit, "data_audit": data_audit,
        "source_sha256": {Path(__file__).name: sha256(Path(__file__)),
                          ("converter.py" if args.model == "full" else "converter_lowrank.py"):
                          sha256(Path(__file__).with_name(
                              "converter.py" if args.model == "full" else "converter_lowrank.py"))},
        "smoke": args.smoke,
    }
    if rank == 0:
        atomic_json(out / "config.json", config)
        np.save(out / "eval_source_indices.npy", chosen_eval_sources)
        if not args.smoke:
            np.save(out / "train_row_indices.npy", order)

    run_wandb = None
    if rank == 0:
        import wandb
        run_wandb = wandb.init(project=args.wandb_project, name=out.name, dir=str(out),
                               mode="disabled" if args.smoke else "online", config=config)
        atomic_json(out / "wandb.json", {"id": run_wandb.id, "url": run_wandb.url,
                                         "smoke": args.smoke})

    def publish_eval(step: int) -> dict:
        result = evaluate(core, z1, proxy, eval_rows, args.eval_batch, device,
                          quantizer, decoder, rank, world)
        result["step"] = step
        if rank == 0:
            with (out / "eval_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            run_wandb.log({"eval/step": step, "eval/nll": result["nll"],
                           "eval/token_accuracy": result["token_accuracy"]})
            print(json.dumps({"evaluation": result}), flush=True)
        return result

    initial_eval = publish_eval(0)
    ticks: list[float] = []
    last_eval = initial_eval
    for step in range(1, args.steps + 1):
        if args.max_seconds and time.monotonic() - started > args.max_seconds:
            raise TimeoutError("explicit converter runtime limit reached")
        tick = time.monotonic()
        offset = ((step - 1) * world + rank) * args.batch
        flat = order[offset:offset + args.batch]
        source, augmentation = np.divmod(flat, NUM_AUG)
        ids = torch.from_numpy(np.array(z1[source, augmentation], dtype=np.int64)).to(device)
        target = torch.from_numpy(np.array(proxy[flat], dtype=np.int64)).to(device).reshape(-1, 16, 16)
        features = make_features(ids, quantizer, decoder)
        if step == 1:
            with torch.no_grad():
                exact = core.tokens(features[:1])
                nearest_ids, _ = nearest(features[:1], book)
                agreement = float((exact == nearest_ids).float().mean())
                if args.model == "full":
                    require(torch.equal(exact, nearest_ids), "nearest-codeword initialization mismatch")
            if rank == 0:
                atomic_json(out / "initialization.json", {"nearest_agreement": agreement,
                            "exact_nearest": args.model == "full",
                            "initial_eval": initial_eval, "parameters": config["parameters"]})
        progress = (step - 1) / max(args.steps - 1, 1)
        cosine = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        rate = args.lr * min(step / max(args.warmup, 1), 1.0) * cosine
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(features)
        loss = F.cross_entropy(logits.float(), target)
        require(bool(torch.isfinite(loss)), "non-finite converter loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        accuracy = (logits.argmax(1) == target).float().mean()
        values = torch.stack([loss.detach(), accuracy.detach(), grad_norm.detach()])
        if world > 1:
            dist.all_reduce(values)
            values /= world
        ticks.append(time.monotonic() - tick)
        if rank == 0 and (step == 1 or step % args.log_every == 0 or step == args.steps):
            record = {"status": "running", "step": step, "total": args.steps,
                      "loss": float(values[0]), "token_accuracy": float(values[1]),
                      "grad_norm": float(values[2]), "lr": rate,
                      "seconds": time.monotonic() - started,
                      "seconds_per_step": float(np.mean(ticks[-args.log_every:])),
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3}
            with (out / "train_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            run_wandb.log({"train/step": step, "train/loss": record["loss"],
                           "train/token_accuracy": record["token_accuracy"],
                           "train/lr": rate, "train/grad_norm": record["grad_norm"]})
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            last_eval = publish_eval(step)
        atomic_json(status, {"status": "running", "pid": os.getpid(), "rank": rank,
                    "step": step, "seconds": time.monotonic() - started,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3})
        del ids, target, features, logits, loss

    frozen.assert_frozen()
    if world > 1:
        dist.barrier()
    receipt = None
    if rank == 0:
        receipt = save_checkpoint(out, core, optimizer, args.steps, config, args.smoke)
        summary = {"status": "complete", "step": args.steps,
                   "initial_eval": initial_eval, "final_eval": last_eval,
                   "nll_improvement": initial_eval["nll"] - last_eval["nll"],
                   "accuracy_improvement": last_eval["token_accuracy"] - initial_eval["token_accuracy"],
                   "checkpoint": receipt, "seconds": time.monotonic() - started,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3}
        atomic_json(out / "summary.json", summary)
        run_wandb.log({"summary/nll_improvement": summary["nll_improvement"],
                       "summary/accuracy_improvement": summary["accuracy_improvement"]})
        run_wandb.finish()
    atomic_json(status, {"status": "complete", "pid": os.getpid(), "rank": rank,
                "step": args.steps, "seconds": time.monotonic() - started,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ASSETS)
    parser.add_argument("--model", choices=("full", "lowrank"), default="full")
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=768)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-sources", type=int, default=512)
    parser.add_argument("--eval-fraction", type=float, default=0.002)
    parser.add_argument("--split-seed", type=int, default=20260901)
    parser.add_argument("--eval-seed", type=int, default=20260917)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--memory-fraction", type=float, default=0.92)
    parser.add_argument("--min-free-gib", type=float, default=75.0)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--wandb-project", default="motar-proxy-converter")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    try:
        run(arguments)
    except BaseException as error:
        if arguments.output.exists():
            atomic_json(arguments.output / f"failure_rank{os.getenv('RANK', '0')}.json",
                        {"error": repr(error), "pid": os.getpid()})
        raise
