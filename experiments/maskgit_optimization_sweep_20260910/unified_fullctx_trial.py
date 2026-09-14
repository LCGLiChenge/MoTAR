"""Train the conservative unified MaskGIT with a full-canvas sparse 2D branch."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from experiments.maskgit_optimization_sweep_20260910.common import ASSETS, atomic_json, output_path, sha256
from experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_adapter import HALTON_BASE_CKPT
from experiments.maskgit_optimization_sweep_20260910.unified_fullctx_maskgit import (
    FullContextUnifiedMaskGIT,
    UnifiedHaltonMixMaskGIT,
    build_fullctx_unified,
    build_unified_halton_mix,
)
from h20.base_model import sample_arccos_mask
from h20.data import E117SparseCodeDataset, collate_e117_sparse
from h20.training import RankBucketSampler, optimizer_description, update_ema

SOURCE_N = 1_281_167
STAT_NAMES = (
    "loss_total",
    "loss1d",
    "loss2d",
    "masked_nll1d",
    "masked_nll2d",
    "mask_ratio1d",
    "mask_ratio2d",
)


def flatten_metrics(prefix: str, value) -> dict[str, float]:
    rows = {}
    if isinstance(value, dict):
        for key, child in value.items():
            rows.update(flatten_metrics(f"{prefix}/{key}" if prefix else str(key), child))
    elif isinstance(value, (int, float)):
        rows[prefix] = float(value)
    return rows


class OffsetSampler:
    def __init__(self, base) -> None:
        self.base = base
        self.offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __len__(self) -> int:
        return max(0, len(self.base) - self.offset)

    def __iter__(self):
        from itertools import islice

        return islice(iter(self.base), self.offset, None)


def endless(loader, sampler, origin):
    epoch, skip = origin
    while True:
        sampler.set_epoch(epoch)
        for offset, batch in enumerate(loader):
            if offset < skip:
                continue
            yield batch, {"packed_pass": epoch, "next_microbatch_offset": offset + 1}
        epoch += 1
        skip = 0


def masked_inputs_2d(model: FullContextUnifiedMaskGIT, targets: torch.Tensor, masked: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    tokens = torch.where(masked, model.mask_token_2d, targets)
    return torch.where(valid, tokens, torch.full_like(tokens, model.pad_token_2d))


def smoothed_objective(logits: torch.Tensor, targets: torch.Tensor, masked: torch.Tensor, valid: torch.Tensor):
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, targets[..., None]).squeeze(-1)
    ce = 0.9 * nll - 0.1 * logp.mean(-1)
    weights = torch.where(masked, 1.0, 0.1) * valid
    loss = ((ce * weights).sum(-1) / weights.sum(-1).clamp_min(1)).mean()
    hard_nll = ((nll * masked).sum(-1) / masked.sum(-1).clamp_min(1)).mean()
    return loss, hard_nll.detach()


class TrainingForward(nn.Module):
    def __init__(
        self,
        core: FullContextUnifiedMaskGIT,
        *,
        loss1d_weight: float,
        loss2d_weight: float,
    ) -> None:
        super().__init__()
        self.core = core
        self.loss1d_weight = float(loss1d_weight)
        self.loss2d_weight = float(loss2d_weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        core = self.core
        z1, z2 = batch["z1d"], batch["z2d"]
        valid1 = torch.ones_like(z1, dtype=torch.bool)
        valid2 = batch["route_valid"]
        mask1, _ = sample_arccos_mask(valid1)
        mask2, _ = sample_arccos_mask(valid2)
        logits1 = core.forward_1d(torch.where(mask1, core.mask_token_1d, z1), batch["label"])
        logits2 = core.forward_2d(
            z1,
            masked_inputs_2d(core, z2, mask2, valid2),
            batch["route_indices"],
            valid2,
            batch["label"],
        )
        loss1, nll1 = smoothed_objective(logits1, z1, mask1, valid1)
        loss2, nll2 = smoothed_objective(logits2, z2, mask2, valid2)
        total = self.loss1d_weight * loss1 + self.loss2d_weight * loss2
        stats = torch.stack((total.detach(), loss1.detach(), loss2.detach(), nll1, nll2, mask1.float().mean(), mask2.float().mean()))
        return total, stats


def fixed_masks(source: int, k: int):
    rng = np.random.default_rng(20260922 + int(source))
    order1 = rng.permutation(32)
    masks1 = []
    for count in (8, 16, 32):
        mask = np.zeros(32, dtype=bool)
        mask[order1[:count]] = True
        masks1.append(mask)
    count2 = min(k, max(1, int(np.rint(np.arccos(rng.uniform()) * 2 / np.pi * k))))
    mask2 = np.zeros(k, dtype=bool)
    mask2[rng.permutation(k)[:count2]] = True
    return np.stack(masks1), mask2


@torch.no_grad()
def evaluate(core: FullContextUnifiedMaskGIT, dataset: E117SparseCodeDataset, selected: np.ndarray) -> dict:
    was_training = core.training
    core.eval()
    sums1 = np.zeros(3, dtype=np.float64)
    sums2: dict[int, list[float]] = {64: [], 128: []}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for k in (64, 128):
            rows = [int(i) for i in selected if int(dataset.k_values[i]) == k]
            for start in range(0, len(rows), 32):
                chosen = rows[start : start + 32]
                cpu = collate_e117_sparse([dataset[i] for i in chosen])
                batch = {key: value.cuda(non_blocking=True) for key, value in cpu.items()}
                mask_rows = [fixed_masks(int(s), k) for s in cpu["source_index"]]
                mask1 = torch.from_numpy(np.stack([row[0] for row in mask_rows])).cuda()
                mask2 = torch.from_numpy(np.stack([row[1] for row in mask_rows])).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for regime in range(3):
                        current = mask1[:, regime]
                        out1 = core.forward_1d(torch.where(current, core.mask_token_1d, batch["z1d"]), batch["label"])
                        nll1 = F.cross_entropy(out1.float().transpose(1, 2), batch["z1d"], reduction="none")
                        sums1[regime] += float(((nll1 * current).sum(-1) / current.sum(-1)).sum())
                    out2 = core.forward_2d(
                        batch["z1d"],
                        masked_inputs_2d(core, batch["z2d"], mask2, batch["route_valid"]),
                        batch["route_indices"],
                        batch["route_valid"],
                        batch["label"],
                    )
                    nll2 = F.cross_entropy(out2.float().transpose(1, 2), batch["z2d"], reduction="none")
                    values = ((nll2 * mask2).sum(-1) / mask2.sum(-1)).float().cpu().tolist()
                    sums2[k].extend(values)
                del batch
    core.train(was_training)
    result = {f"nll1d_mask{count}": float(value / len(selected)) for count, value in zip((8, 16, 32), sums1)}
    result["nll1d"] = float(sums1.mean() / len(selected))
    result["nll2d"] = float(np.mean(sums2[64] + sums2[128]))
    result["nll2d_k64"] = float(np.mean(sums2[64]))
    result["nll2d_k128"] = float(np.mean(sums2[128]))
    result["n_k64"] = len(sums2[64])
    result["n_k128"] = len(sums2[128])
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite development metric")
    return result


def set_lrs(optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace) -> None:
    for group in optimizer.param_groups:
        name = group["name"]
        if name == "one_d":
            if step <= args.freeze_1d_updates:
                group["lr"] = 0.0
            else:
                group["lr"] = group["base_lr"] * min(1.0, (step - args.freeze_1d_updates) / max(1, args.one_d_ramp))
        elif name == "two_d_feature":
            group["lr"] = group["base_lr"] * min(1.0, step / max(1, args.feature_ramp))
        elif name == "two_d_pretrained":
            if step <= args.warmup_2d_pretrained:
                group["lr"] = 0.0
            else:
                group["lr"] = group["base_lr"] * min(1.0, (step - args.warmup_2d_pretrained) / max(1, args.two_d_ramp))
        else:
            raise ValueError("unknown optimizer group: " + name)


def optimizer_for(core: FullContextUnifiedMaskGIT, args: argparse.Namespace) -> torch.optim.Optimizer:
    one_d = [(n, p) for n, p in core.named_parameters() if n.startswith("one_d.") and p.requires_grad]
    two_d_feature = [
        (n, p)
        for n, p in core.named_parameters()
        if n.startswith(("two_d.feature_", "two_d.one_d_context_")) and p.requires_grad
    ]
    two_d_pretrained = [
        (n, p)
        for n, p in core.named_parameters()
        if n.startswith("two_d.") and not n.startswith(("two_d.feature_", "two_d.one_d_context_")) and p.requires_grad
    ]
    if not one_d or not two_d_feature or not two_d_pretrained:
        raise ValueError("missing expected parameter groups")
    return torch.optim.AdamW(
        [
            {"params": [p for _, p in one_d], "lr": 0.0, "base_lr": args.lr_1d, "name": "one_d"},
            {"params": [p for _, p in two_d_feature], "lr": 0.0, "base_lr": args.lr_feature, "name": "two_d_feature"},
            {"params": [p for _, p in two_d_pretrained], "lr": 0.0, "base_lr": args.lr_2d, "name": "two_d_pretrained"},
        ],
        betas=(0.9, 0.96),
        weight_decay=args.weight_decay,
    )


def checkpoint_step(path: Path) -> int:
    sidecar = path.parent / "latest.json"
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
        if "step" in data:
            return int(data["step"])
    return 0


def save_latest(output: Path, core: nn.Module, ema: nn.Module | None, optimizer: torch.optim.Optimizer, rng_states: list, metadata: dict) -> dict:
    if shutil.disk_usage(output).free < 10 * 1024**3:
        raise RuntimeError("less than 10GiB checkpoint safety headroom")
    tensors = {"raw/" + name: value.detach().cpu().contiguous() for name, value in core.state_dict().items()}
    if ema is not None:
        tensors.update({"ema/" + name: value.detach().cpu().contiguous() for name, value in ema.state_dict().items()})
    for name, parameter in core.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                tensors["adam/" + name + "/" + key] = value.detach().cpu().contiguous()
    for rank, state in enumerate(rng_states):
        tensors[f"rng/{rank}/cpu"] = state["cpu"]
        tensors[f"rng/{rank}/cuda"] = state["cuda"]
    temporary = output / "latest.pt.tmp"
    format_name = metadata.get("config", {}).get("style", "fullctx_unified_maskgit_v1")
    save_file(tensors, str(temporary), metadata={"format": str(format_name)})
    with safe_open(str(temporary), framework="pt", device="cpu") as reader:
        if set(reader.keys()) != set(tensors):
            raise ValueError("checkpoint key mismatch")
    result = dict(
        metadata,
        format=str(format_name),
        model_class=core.__class__.__module__ + "." + core.__class__.__name__,
        optimizer=optimizer_description(optimizer, core),
        tensors=len(tensors),
        bytes=temporary.stat().st_size,
        sha256=sha256(temporary),
        round_trip_exact=True,
    )
    os.replace(temporary, output / "latest.pt")
    atomic_json(output / "latest.json", result)
    return {key: result[key] for key in ("step", "sha256", "bytes", "tensors", "round_trip_exact")}


def load_checkpoint(path: Path, core: nn.Module, ema: nn.Module | None, optimizer: torch.optim.Optimizer, *, state: str, restore_rng: bool, rank: int) -> dict:
    tensors = load_file(str(path), device="cpu")
    prefix = state + "/"
    model_state = {key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)}
    if not model_state:
        raise ValueError(f"checkpoint {path} does not contain {prefix} tensors")
    core.load_state_dict(model_state, strict=True)
    if ema is not None:
        ema_state = {key[len("ema/") :]: value for key, value in tensors.items() if key.startswith("ema/")}
        ema.load_state_dict(ema_state if ema_state else model_state, strict=True)
    optimizer_states = 0
    for name, parameter in core.named_parameters():
        base = "adam/" + name + "/"
        state_tensors = {key[len(base) :]: value.to(parameter.device) for key, value in tensors.items() if key.startswith(base)}
        if state_tensors:
            optimizer.state[parameter] = state_tensors
            optimizer_states += 1
    rng_restored = False
    if restore_rng:
        cpu_key = f"rng/{rank}/cpu"
        cuda_key = f"rng/{rank}/cuda"
        if cpu_key in tensors and cuda_key in tensors:
            torch.set_rng_state(tensors[cpu_key])
            torch.cuda.set_rng_state(tensors[cuda_key].cuda())
            rng_restored = True
    return {"resume_from": str(path), "resume_sha256": sha256(path), "resume_state": state, "resume_step": checkpoint_step(path), "optimizer_states": optimizer_states, "rng_restored": rng_restored}


def main(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world < 1 or args.micro < 1 or args.global_batch % (args.micro * world):
        raise ValueError("global_batch must be divisible by micro*world")
    accumulation = args.global_batch // (args.micro * world)
    updates_per_epoch = math.ceil((SOURCE_N * 2) / args.global_batch)
    target_updates = int(args.updates)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("epochs must be positive")
        target_updates = int(args.epochs) * updates_per_epoch
    if not args.allow_long and not 1 <= target_updates <= 500:
        raise ValueError("bounded local screen requires 1..500 updates; pass --allow-long for longer runs")
    output = output_path(args.output)
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20), device_id=torch.device("cuda", local))
    step = 0
    started = time.monotonic()
    wandb_run = None
    try:
        config = {
            "style": "unified_haltonmix_v2" if args.variant == "haltonmix_v2" else "fullctx_unified_maskgit_v1",
            "variant": args.variant,
            "detach_1d_context": args.detach_1d_context,
            "assets_root": str(args.assets_root.resolve()),
            "halton_ckpt": str(args.halton_ckpt.resolve()),
            "output": str(output),
            "train_data": str((args.assets_root / "codes/train").resolve()),
            "train_routes": str((args.assets_root / "routes/train").resolve()),
            "micro": args.micro,
            "world": world,
            "accumulation": accumulation,
            "global_batch": args.global_batch,
            "updates": target_updates,
            "requested_updates": args.updates,
            "epochs": args.epochs,
            "updates_per_epoch": updates_per_epoch,
            "save_every": args.save_every,
            "eval_every": args.eval_every,
            "eval_n": args.eval_n,
            "lr_1d": args.lr_1d,
            "lr_2d": args.lr_2d,
            "lr_feature": args.lr_feature,
            "loss1d_weight": args.loss1d_weight,
            "loss2d_weight": args.loss2d_weight,
            "freeze_1d_updates": args.freeze_1d_updates,
            "warmup_2d_pretrained": args.warmup_2d_pretrained,
            "feature_chunk": args.feature_chunk,
            "ema": args.ema,
            "wandb_project": args.wandb_project,
            "wandb_name": args.wandb_name,
            "wandb_mode": args.wandb_mode,
        }
        if rank == 0:
            if output.exists():
                raise FileExistsError(output)
            output.mkdir(parents=True)
            atomic_json(output / "config.json", config)
        dist.barrier()
        torch.manual_seed(args.seed + rank)
        if args.variant == "haltonmix_v2":
            core, init_audit = build_unified_halton_mix(
                assets_root=args.assets_root,
                halton_ckpt=args.halton_ckpt,
                device="cuda",
                feature_chunk=args.feature_chunk,
                attention_implementation=args.attention_implementation,
                detach_1d_context=args.detach_1d_context,
            )
        else:
            core, init_audit = build_fullctx_unified(
                assets_root=args.assets_root,
                halton_ckpt=args.halton_ckpt,
                device="cuda",
                feature_chunk=args.feature_chunk,
                attention_implementation=args.attention_implementation,
            )
        core.train()
        ema = deepcopy(core).eval().requires_grad_(False) if args.ema and rank == 0 else None
        if ema is not None:
            ema.two_d.set_feature_provider(core.two_d._feature_provider)
        optimizer = optimizer_for(core, args)
        resume_report = None
        start_step = 0
        if args.resume_from:
            resume_report = load_checkpoint(args.resume_from, core, ema, optimizer, state=args.resume_state, restore_rng=args.restore_rng, rank=rank)
            start_step = int(resume_report["resume_step"])
            if start_step >= target_updates:
                raise ValueError(f"resume step {start_step} is already >= target updates {target_updates}")
        if rank == 0:
            atomic_json(output / "init_audit_rank0.json", {"init": init_audit, "resume": resume_report})
            sources = [Path(__file__), Path(__file__).with_name("unified_fullctx_maskgit.py"), args.halton_ckpt]
            atomic_json(output / "manifest.json", {"config": config, "sources": {str(p): sha256(p) for p in sources if p.exists()}})
            if args.wandb_project:
                import wandb
                (output / "wandb").mkdir(parents=True, exist_ok=True)
                wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name or output.name, mode=args.wandb_mode, dir=str(output / "wandb"), config=config)
                atomic_json(output / "wandb_run.json", {"project": wandb_run.project, "name": wandb_run.name, "id": wandb_run.id, "mode": args.wandb_mode})
        dist.barrier()
        train = E117SparseCodeDataset(args.assets_root / "codes/train", args.assets_root / "routes/train", split="all")
        val = E117SparseCodeDataset(args.assets_root / "codes/val", args.assets_root / "routes/val", split="all") if rank == 0 else None
        if len(train) != SOURCE_N * 2:
            raise ValueError("full ImageNet TRAIN code cache required")
        sampler = OffsetSampler(RankBucketSampler(train, args.micro, world, rank, 0))
        loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_e117_sparse, num_workers=args.workers, pin_memory=True, generator=torch.Generator().manual_seed(90000 + rank), **({"persistent_workers": True, "prefetch_factor": 2} if args.workers else {}))
        stream = endless(loader, sampler, (0, 0))
        if rank == 0:
            selected = np.random.default_rng(20260922).choice(len(val), args.eval_n, replace=False)
            atomic_json(output / "dev_cohort.json", {"source_indices": selected.tolist(), "mask_seed": 20260922, "split": "val"})
        else:
            selected = np.empty(0, dtype=np.int64)
        forward = TrainingForward(core, loss1d_weight=args.loss1d_weight, loss2d_weight=args.loss2d_weight)
        ddp = DDP(forward, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=False)
        history = []

        def development(current_step: int) -> None:
            if rank == 0:
                values = {"raw": evaluate(core, val, selected)}
                if ema is not None:
                    values["ema"] = evaluate(ema, val, selected)
                row = {"step": current_step, **values}
                history.append(row)
                atomic_json(output / "development.json", history)
                print(json.dumps({"stage": "development", **row}, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(flatten_metrics("dev", row), step=current_step)
            dist.barrier()

        if args.eval_initial:
            development(start_step)
        window = torch.zeros(len(STAT_NAMES), device="cuda")
        window_count = 0
        window_start = time.monotonic()
        checkpoint = None
        cursor = None
        for step in range(start_step + 1, target_updates + 1):
            set_lrs(optimizer, step, args)
            core.train()
            ddp.zero_grad(set_to_none=True)
            stats = torch.zeros(len(STAT_NAMES), device="cuda")
            for micro in range(accumulation):
                cpu, cursor = next(stream)
                batch = {key: value.cuda(non_blocking=True) for key, value in cpu.items()}
                with ddp.no_sync() if micro + 1 < accumulation else nullcontext():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, row = ddp(batch)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("nonfinite training loss")
                    (loss / accumulation).backward()
                stats += row / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if ema is not None:
                update_ema(ema, core)
            dist.all_reduce(stats)
            stats /= world
            window += stats
            window_count += 1
            if step <= 3 or step % args.log_every == 0 or step == target_updates:
                divisor = max(1, window_count)
                row = {
                    "step": step,
                    "phase": "two_d_protected" if step <= args.freeze_1d_updates else "joint",
                    **{name: float(value / divisor) for name, value in zip(STAT_NAMES, window)},
                    "grad_norm": float(grad_norm),
                    "lr_1d": optimizer.param_groups[0]["lr"],
                    "lr_feature": optimizer.param_groups[1]["lr"],
                    "lr_2d": optimizer.param_groups[2]["lr"],
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "seconds_per_step": (time.monotonic() - window_start) / divisor,
                }
                if rank == 0:
                    atomic_json(output / "status.json", {"status": "running", "pid": os.getpid(), **row})
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(flatten_metrics("train", row), step=step)
                window.zero_()
                window_count = 0
                window_start = time.monotonic()
            if args.eval_every > 0 and (step % args.eval_every == 0 or step == target_updates):
                development(step)
                window_start = time.monotonic()
            if args.save_every > 0 and step % args.save_every == 0 and step != args.updates:
                rng_states = [None] * world
                dist.all_gather_object(rng_states, {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()})
                if rank == 0:
                    checkpoint = save_latest(output, core, ema, optimizer, rng_states, {"step": step, "config": config, "cursor": cursor, "init_audit": init_audit, "resume": resume_report})
                    print(json.dumps({"stage": "checkpoint", **checkpoint}, sort_keys=True), flush=True)
                dist.barrier()
                window_start = time.monotonic()
        if args.no_final_save:
            checkpoint = None
        else:
            rng_states = [None] * world
            dist.all_gather_object(rng_states, {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()})
            if rank == 0:
                checkpoint = save_latest(output, core, ema, optimizer, rng_states, {"step": step, "config": config, "cursor": cursor, "init_audit": init_audit, "resume": resume_report})
                print(json.dumps({"stage": "checkpoint", **checkpoint}, sort_keys=True), flush=True)
            dist.barrier()
        if rank == 0:
            summary = {"status": "complete", "step": step, "config": config, "checkpoint": checkpoint, "elapsed_seconds": time.monotonic() - started, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3, "development_initial": history[0] if history else None, "development_final": history[-1] if history else None, "final_save_skipped": bool(args.no_final_save)}
            atomic_json(output / "summary.json", summary)
            atomic_json(output / "status.json", summary)
            if wandb_run is not None:
                wandb_run.log(flatten_metrics("summary", summary), step=step)
                wandb_run.finish()
        dist.barrier()
    except BaseException as exc:
        if output.exists():
            atomic_json(output / f"failure_rank{rank}.json", {"type": type(exc).__name__, "error": str(exc), "step": step})
        raise
    finally:
        dist.destroy_process_group()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, default=ASSETS)
    p.add_argument("--halton-ckpt", type=Path, default=HALTON_BASE_CKPT)
    p.add_argument("--variant", choices=("haltonmix_v2", "fullctx_v1"), default="haltonmix_v2")
    p.add_argument("--detach-1d-context", action="store_true")
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--epochs", type=int, help="override --updates with epochs * ceil(full_train_samples/global_batch)")
    p.add_argument("--allow-long", action="store_true")
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--resume-state", choices=("raw", "ema"), default="raw")
    p.add_argument("--restore-rng", action="store_true")
    p.add_argument("--micro", type=int, default=64)
    p.add_argument("--global-batch", type=int, default=768)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval-initial", action="store_true")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--feature-chunk", type=int, default=16)
    p.add_argument("--lr-1d", type=float, default=1e-5)
    p.add_argument("--lr-2d", type=float, default=1e-5)
    p.add_argument("--lr-feature", type=float, default=1e-4)
    p.add_argument("--loss1d-weight", type=float, default=1.0)
    p.add_argument("--loss2d-weight", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--freeze-1d-updates", type=int, default=50)
    p.add_argument("--warmup-2d-pretrained", type=int, default=20)
    p.add_argument("--one-d-ramp", type=int, default=100)
    p.add_argument("--two-d-ramp", type=int, default=100)
    p.add_argument("--feature-ramp", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--no-final-save", action="store_true")
    p.add_argument("--attention-implementation", choices=("eager", "sdpa"))
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-name")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
