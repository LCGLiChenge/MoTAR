"""Short sparse-2D-only fine-tuning of pretrained Halton/LlamaGen MaskGIT."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from experiments.maskgit_optimization_sweep_20260910.common import ASSETS, ROOT, RESULT_ROOT, atomic_json, sha256
from experiments.maskgit_optimization_sweep_20260910.portable_base_features import PortableBaseFeatures
from experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_adapter import (
    HALTON_BASE_CKPT,
    HaltonSparse2DAdapter,
    HaltonFullContextSparse2DAdapter,
    HaltonBaseContextSparse2DAdapter,
    load_halton_base_state,
)
from experiments.maskgit_optimization_sweep_20260910.sampling_followup import spatial_order_ranks

from h20.data import E117SparseCodeDataset, collate_e117_sparse
from h20.training import RankBucketSampler, optimizer_description, update_ema
from h20.base_model import _mask_from_counts, sample_arccos_mask


SOURCE_N = 1281167
STAT_NAMES = ("loss2d", "masked_nll2d", "mask_ratio2d")


def flatten_metrics(prefix: str, value) -> dict[str, float]:
    rows = {}
    if isinstance(value, dict):
        for key, child in value.items():
            rows.update(flatten_metrics(f"{prefix}/{key}" if prefix else str(key), child))
    elif isinstance(value, (int, float)):
        rows[prefix] = float(value)
    return rows


class OffsetSampler:
    def __init__(self, base):
        self.base = base
        self.offset = 0

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __len__(self):
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
            yield batch, dict(packed_pass=epoch, next_microbatch_offset=offset + 1)
        epoch += 1
        skip = 0


def output_path(path: Path) -> Path:
    path = Path(path).resolve()
    if path == RESULT_ROOT.resolve() or not path.is_relative_to(RESULT_ROOT.resolve()):
        raise ValueError("output must be a specific child of " + str(RESULT_ROOT))
    return path


def set_lrs(optimizer: torch.optim.Optimizer, step: int, warmup: int) -> None:
    for group in optimizer.param_groups:
        if group["name"] == "feature":
            group["lr"] = group["base_lr"] * min(1.0, step / 50)
        elif step <= warmup:
            group["lr"] = 0.0
        else:
            group["lr"] = group["base_lr"] * min(1.0, (step - warmup) / 100)


def checkpoint_step(path: Path) -> int:
    for name in ("latest.json", "summary.json"):
        sidecar = path.parent / name
        if not sidecar.exists():
            continue
        with sidecar.open("r") as handle:
            data = json.load(handle)
        if name == "latest.json" and "step" in data:
            return int(data["step"])
        if name == "summary.json":
            if "step" in data:
                return int(data["step"])
            if isinstance(data.get("checkpoint"), dict) and "step" in data["checkpoint"]:
                return int(data["checkpoint"]["step"])
    return 0


def load_trial_checkpoint(
    path: Path,
    core: nn.Module,
    ema: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    *,
    state: str,
    restore_rng: bool,
    rank: int,
) -> dict:
    if state not in {"raw", "ema"}:
        raise ValueError("state must be raw or ema")
    tensors = load_file(str(path), device="cpu")
    prefix = state + "/"
    model_state = {key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)}
    if not model_state:
        raise ValueError(f"checkpoint {path} does not contain {prefix} tensors")
    core.load_state_dict(model_state, strict=True)
    ema_loaded = False
    if ema is not None:
        ema_state = {key[len("ema/") :]: value for key, value in tensors.items() if key.startswith("ema/")}
        ema.load_state_dict(ema_state if ema_state else model_state, strict=True)
        ema_loaded = bool(ema_state)

    named_params = dict(core.named_parameters())
    optimizer_states = 0
    for name, parameter in named_params.items():
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

    return dict(
        resume_from=str(path),
        resume_sha256=sha256(path),
        resume_state=state,
        resume_step=checkpoint_step(path),
        loaded_model_tensors=len(model_state),
        ema_loaded=ema_loaded,
        optimizer_states=optimizer_states,
        rng_restored=rng_restored,
    )


def smoothed_objective(logits, targets, masked, valid):
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, targets[..., None]).squeeze(-1)
    ce = 0.9 * nll - 0.1 * logp.mean(-1)
    weights = torch.where(masked, 1.0, 0.1) * valid
    loss = ((ce * weights).sum(-1) / weights.sum(-1).clamp_min(1)).mean()
    hard = ((nll * masked).sum(-1) / masked.sum(-1).clamp_min(1)).mean()
    return loss, hard.detach()


class TrainingForward(nn.Module):
    def __init__(
        self,
        core: HaltonSparse2DAdapter,
        *,
        train_mask_order: str = "random",
        rollin_visible_prob: float = 0.0,
        rollin_draw: str = "argmax",
    ):
        super().__init__()
        if train_mask_order not in {"random", "halton", "halton_fixed", "raster"}:
            raise ValueError("unknown train_mask_order")
        if not 0.0 <= rollin_visible_prob <= 1.0:
            raise ValueError("rollin_visible_prob must be in [0,1]")
        if rollin_draw not in {"argmax", "categorical"}:
            raise ValueError("unknown rollin_draw")
        self.core = core
        self.train_mask_order = train_mask_order
        self.rollin_visible_prob = float(rollin_visible_prob)
        self.rollin_draw = rollin_draw

    def _mask(self, z1: torch.Tensor, index: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.train_mask_order == "random":
            mask, _ = sample_arccos_mask(valid)
            return mask
        counts = valid.sum(1)
        ratio = torch.rand(valid.shape[0], device=valid.device).clamp_(1e-6, 1.0 - 1e-6)
        remain = torch.acos(ratio) / (math.pi * 0.5)
        target = torch.floor(counts.float() * remain).long().clamp_min(1)
        target = torch.minimum(target, counts)
        ranks = spatial_order_ranks(index, valid, z1, labels, self.train_mask_order)
        return _mask_from_counts(valid, target, -ranks.float())

    def _rollin(self, z1, z2, index, valid, labels, mask2, tokens):
        if self.rollin_visible_prob <= 0.0:
            return tokens
        visible = (~mask2) & valid
        if not bool(visible.any()):
            return tokens
        with torch.no_grad():
            logits = self.core.forward_2d(z1, tokens, index, valid, labels)
            if self.rollin_draw == "categorical":
                sampled = torch.distributions.Categorical(logits=logits.float()).sample()
            else:
                sampled = logits.argmax(-1)
        roll = (torch.rand(visible.shape, device=visible.device) < self.rollin_visible_prob) & visible
        return torch.where(roll, sampled, tokens)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        core = self.core
        z1, z2, valid = batch["z1d"], batch["z2d"], batch["route_valid"]
        index, labels = batch["route_indices"], batch["label"]
        mask2 = self._mask(z1, index, valid, labels)
        tokens = torch.where(mask2, core.mask_token_2d, z2)
        tokens = self._rollin(z1, z2, index, valid, labels, mask2, tokens)
        logits = core.forward_2d(z1, tokens, index, valid, labels)
        loss, nll = smoothed_objective(logits, z2, mask2, valid)
        return loss, torch.stack((loss.detach(), nll, mask2.float().mean()))


def fixed_mask(source: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(20260922 + int(source))
    count = min(k, max(1, int(np.rint(np.arccos(rng.uniform()) * 2 / np.pi * k))))
    mask = np.zeros(k, dtype=bool)
    mask[rng.permutation(k)[:count]] = True
    return mask


@torch.no_grad()
def evaluate(core: HaltonSparse2DAdapter, dataset: E117SparseCodeDataset, selected: np.ndarray) -> dict:
    was_training = core.training
    core.eval()
    sums: dict[int, list[float]] = {64: [], 128: []}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for k in (64, 128):
            rows = [int(i) for i in selected if int(dataset.k_values[i]) == k]
            for start in range(0, len(rows), 32):
                chosen = rows[start : start + 32]
                cpu = collate_e117_sparse([dataset[i] for i in chosen])
                batch = {key: val.cuda(non_blocking=True) for key, val in cpu.items()}
                mask2 = torch.from_numpy(np.stack([fixed_mask(int(s), k) for s in cpu["source_index"]])).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = core.forward_2d(
                        batch["z1d"],
                        torch.where(mask2, core.mask_token_2d, batch["z2d"]),
                        batch["route_indices"],
                        batch["route_valid"],
                        batch["label"],
                    )
                    nll = F.cross_entropy(out.float().transpose(1, 2), batch["z2d"], reduction="none")
                    values = ((nll * mask2).sum(-1) / mask2.sum(-1)).float().cpu().tolist()
                    sums[k].extend(values)
    core.train(was_training)
    result = {
        "nll2d": float(np.mean(sums[64] + sums[128])),
        "nll2d_k64": float(np.mean(sums[64])),
        "nll2d_k128": float(np.mean(sums[128])),
        "n_k64": len(sums[64]),
        "n_k128": len(sums[128]),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite development metric")
    return result


def save_latest(output: Path, core: nn.Module, ema: nn.Module, optimizer: torch.optim.Optimizer, rng_states: list, metadata: dict) -> dict:
    tensors = {}
    for label, model in (("raw", core), ("ema", ema)):
        tensors.update({label + "/" + name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()})
    for name, parameter in core.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                tensors["adam/" + name + "/" + key] = value.detach().cpu().contiguous()
    for rank, state in enumerate(rng_states):
        tensors[f"rng/{rank}/cpu"] = state["cpu"]
        tensors[f"rng/{rank}/cuda"] = state["cuda"]
    tmp = output / "latest.safetensors.tmp"
    save_file(tensors, str(tmp), metadata={"format": "halton_sparse2d_trial_v1"})
    with safe_open(str(tmp), framework="pt", device="cpu") as reader:
        if set(reader.keys()) != set(tensors):
            raise ValueError("checkpoint key mismatch")
    result = dict(
        metadata,
        format="halton_sparse2d_trial_v1",
        model_class=core.__class__.__module__ + "." + core.__class__.__name__,
        optimizer=optimizer_description(optimizer, core),
        tensors=len(tensors),
        bytes=tmp.stat().st_size,
        sha256=sha256(tmp),
        round_trip_exact=True,
    )
    os.replace(tmp, output / "latest.safetensors")
    atomic_json(output / "latest.json", result)
    return {key: result[key] for key in ("step", "sha256", "bytes", "tensors", "round_trip_exact")}


def main(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world < 1 or args.micro < 1 or args.global_batch % (args.micro * world):
        raise ValueError("global_batch must be divisible by micro*world")
    if not args.allow_long and not 1 <= args.updates <= 500:
        raise ValueError("bounded local screen requires 1..500 updates; pass --allow-long for longer runs")
    if not 1 <= args.updates or not 0 <= args.warmup < args.updates:
        raise ValueError("updates must be positive and warmup < updates")
    accumulation = args.global_batch // (args.micro * world)
    output = output_path(args.output)
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=torch.device("cuda", local))
    step = 0
    started = time.monotonic()
    try:
        config = dict(
            style=("halton_basectx_sparse2d" if args.context_style == "basectx" else "halton_fullctx_sparse2d" if args.context_style == "fullctx" else "halton_sparse2d"),
            context_style=args.context_style,
            source=str(args.halton_ckpt),
            micro=args.micro,
            world=world,
            accumulation=accumulation,
            global_batch=args.global_batch,
            updates=args.updates,
            warmup=args.warmup,
            eval_every=args.eval_every,
            eval_n=args.eval_n,
            save_every=args.save_every,
            lr_feature=args.lr_feature,
            lr_pretrained=args.lr_pretrained,
            resume_from=str(args.resume_from) if args.resume_from else None,
            resume_state=args.resume_state,
            teacher_cache=str(args.teacher_cache.resolve()) if args.teacher_cache else None,
            teacher_dev=str(args.teacher_dev.resolve()) if args.teacher_cache else None,
            teacher_arm=args.teacher_arm,
            train_mask_order=args.train_mask_order,
            rollin_visible_prob=args.rollin_visible_prob,
            rollin_draw=args.rollin_draw,
            train_data=str((ASSETS / "codes/train").resolve()),
            train_routes=str((ASSETS / "routes/train").resolve()),
            output=str(output),
            objective="selected sparse 2D only",
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            wandb_mode=args.wandb_mode,
        )
        if rank == 0:
            if output.exists():
                raise FileExistsError(output)
            output.mkdir(parents=True)
            atomic_json(output / "config.json", config)
            sources = [
                Path(__file__),
                ROOT / "experiments/maskgit_optimization_sweep_20260910/halton_sparse2d_adapter.py",
                ROOT / "experiments/maskgit_optimization_sweep_20260910/portable_base_features.py",
                args.halton_ckpt,
            ]
            if args.resume_from:
                sources.append(args.resume_from)
            if args.teacher_cache:
                sources.append(args.teacher_cache / "meta.json")
            if args.teacher_cache and args.teacher_dev:
                sources.append(args.teacher_dev / "manifest.json")
            atomic_json(output / "manifest.json", dict(config=config, sources={str(p): sha256(p) for p in sources if p.exists()}))
        wandb_run = None
        if rank == 0 and args.wandb_project:
            import wandb
            (output / "wandb").mkdir(parents=True, exist_ok=True)
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or output.name,
                mode=args.wandb_mode,
                dir=str(output / "wandb"),
                config=config,
            )
            atomic_json(output / "wandb_run.json", dict(project=wandb_run.project, name=wandb_run.name, id=wandb_run.id, mode=args.wandb_mode))
        dist.barrier()
        torch.manual_seed(0)
        model_cls = {"sparse": HaltonSparse2DAdapter, "fullctx": HaltonFullContextSparse2DAdapter, "basectx": HaltonBaseContextSparse2DAdapter}[args.context_style]
        core = model_cls().cuda().train()
        load_report = load_halton_base_state(core, args.halton_ckpt)
        provider = PortableBaseFeatures("cuda", chunk=args.feature_chunk)
        core.set_feature_provider(provider)
        ema = deepcopy(core).eval().requires_grad_(False) if rank == 0 else None
        if ema is not None:
            ema.set_feature_provider(provider)
        feature_params = [(n, p) for n, p in core.named_parameters() if n.startswith("feature_")]
        base_params = [(n, p) for n, p in core.named_parameters() if not n.startswith("feature_")]
        optimizer = torch.optim.AdamW(
            [
                dict(params=[p for _, p in feature_params], lr=args.lr_feature, base_lr=args.lr_feature, name="feature"),
                dict(params=[p for _, p in base_params], lr=0.0, base_lr=args.lr_pretrained, name="pretrained"),
            ],
            betas=(0.9, 0.96),
            weight_decay=0.03,
        )
        resume_report = None
        start_step = 0
        if args.resume_from:
            resume_report = load_trial_checkpoint(
                args.resume_from,
                core,
                ema,
                optimizer,
                state=args.resume_state,
                restore_rng=args.restore_rng,
                rank=rank,
            )
            start_step = int(resume_report["resume_step"])
            if start_step >= args.updates:
                raise ValueError(f"resume step {start_step} is already >= target updates {args.updates}")
        if rank == 0:
            atomic_json(
                output / "init_audit_rank0.json",
                dict(load_report, resume=resume_report, feature_parameters=[n for n, _ in feature_params]),
            )
        if args.teacher_cache:
            from experiments.generated_prefix_training_20260910.data import HeldoutTeacherDataset, TeacherDataset
            train = TeacherDataset(args.teacher_cache, args.teacher_arm, allow_smoke=args.allow_teacher_smoke)
            val = HeldoutTeacherDataset(args.teacher_dev, args.teacher_arm) if rank == 0 else None
        else:
            train = E117SparseCodeDataset(ASSETS / "codes/train", ASSETS / "routes/train", split="all")
            val = E117SparseCodeDataset(ASSETS / "codes/val", ASSETS / "routes/val", split="all") if rank == 0 else None
            if len(train) != SOURCE_N * 2:
                raise ValueError("full ImageNet TRAIN code cache required")
        sampler = OffsetSampler(RankBucketSampler(train, args.micro, world, rank, 0))
        loader = DataLoader(
            train,
            batch_sampler=sampler,
            collate_fn=collate_e117_sparse,
            num_workers=args.workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(90000 + rank),
            **({"persistent_workers": True, "prefetch_factor": 2} if args.workers else {}),
        )
        stream = endless(loader, sampler, (0, 0))
        if rank == 0:
            eval_population = len(val)
            if args.eval_n > eval_population:
                raise ValueError(f"eval_n {args.eval_n} exceeds dev population {eval_population}")
            selected = np.random.default_rng(20260922).choice(eval_population, args.eval_n, replace=False)
            atomic_json(output / "dev_cohort.json", dict(source_indices=selected.tolist(), mask_seed=20260922, split="teacher_dev" if args.teacher_cache else "val"))
        else:
            selected = np.empty(0, dtype=np.int64)
        forward = TrainingForward(
            core,
            train_mask_order=args.train_mask_order,
            rollin_visible_prob=args.rollin_visible_prob,
            rollin_draw=args.rollin_draw,
        )
        ddp = DDP(forward, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=True)
        history = []

        def development(current_step: int) -> None:
            if rank == 0:
                values = {label: evaluate(model, val, selected) for label, model in (("raw", core), ("ema", ema))}
                row = dict(step=current_step, **values)
                history.append(row)
                atomic_json(output / "development.json", history)
                print(json.dumps(dict(stage="development", **row), sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(flatten_metrics("dev", row), step=current_step)
            dist.barrier()

        development(start_step)
        window = torch.zeros(len(STAT_NAMES), device="cuda")
        window_count = 0
        window_start = time.monotonic()
        checkpoint = None
        for step in range(start_step + 1, args.updates + 1):
            set_lrs(optimizer, step, args.warmup)
            core.train()
            ddp.zero_grad(set_to_none=True)
            stats = torch.zeros(len(STAT_NAMES), device="cuda")
            cursor = None
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
            grad_norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0, error_if_nonfinite=True)
            optimizer.step()
            if rank == 0:
                update_ema(ema, core)
            dist.all_reduce(stats)
            stats /= world
            window += stats
            window_count += 1
            if step % 10 == 0 or step <= 3 or step == args.updates:
                divisor = max(1, window_count)
                row = dict(
                    step=step,
                    phase="feature_warmup" if step <= args.warmup else "joint_finetune",
                    **{name: float(value / divisor) for name, value in zip(STAT_NAMES, window)},
                    grad_norm=float(grad_norm),
                    lr_feature=optimizer.param_groups[0]["lr"],
                    lr_pretrained=optimizer.param_groups[1]["lr"],
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                    seconds_per_step=(time.monotonic() - window_start) / divisor,
                )
                if rank == 0:
                    atomic_json(output / "status.json", dict(status="running", pid=os.getpid(), **row))
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(flatten_metrics("train", row), step=step)
                window.zero_()
                window_count = 0
                window_start = time.monotonic()
            if step % args.eval_every == 0 or step == args.updates:
                development(step)
                window_start = time.monotonic()
            if args.save_every > 0 and step % args.save_every == 0 and step != args.updates:
                rng_states = [None] * world
                dist.all_gather_object(rng_states, dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()))
                if rank == 0:
                    checkpoint = save_latest(
                        output,
                        core,
                        ema,
                        optimizer,
                        rng_states,
                        dict(step=step, config=config, cursor=cursor, init_audit=load_report, resume=resume_report),
                    )
                    print(json.dumps(dict(stage="checkpoint", **checkpoint), sort_keys=True), flush=True)
                dist.barrier()
                window_start = time.monotonic()
        rng_states = [None] * world
        dist.all_gather_object(rng_states, dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()))
        if rank == 0:
            checkpoint = save_latest(
                output,
                core,
                ema,
                optimizer,
                rng_states,
                dict(step=step, config=config, cursor=cursor, init_audit=load_report, resume=resume_report),
            )
            print(json.dumps(dict(stage="checkpoint", **checkpoint), sort_keys=True), flush=True)
            summary = dict(
                status="complete",
                step=step,
                config=config,
                checkpoint=checkpoint,
                elapsed_seconds=time.monotonic() - started,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                development_initial=history[0],
                development_final=history[-1],
            )
            atomic_json(output / "summary.json", summary)
            atomic_json(output / "status.json", summary)
            if wandb_run is not None:
                wandb_run.log(flatten_metrics("summary", summary), step=step)
                wandb_run.finish()
        dist.barrier()
    except BaseException as exc:
        if output.exists():
            atomic_json(output / f"failure_rank{rank}.json", dict(type=type(exc).__name__, error=str(exc), step=step))
        raise
    finally:
        dist.destroy_process_group()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--halton-ckpt", type=Path, default=HALTON_BASE_CKPT)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--allow-long", action="store_true")
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--resume-state", choices=("raw", "ema"), default="raw")
    p.add_argument("--restore-rng", action="store_true")
    p.add_argument("--micro", type=int, default=96)
    p.add_argument("--global-batch", type=int, default=1152)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--feature-chunk", type=int, default=16)
    p.add_argument("--lr-feature", type=float, default=1e-4)
    p.add_argument("--lr-pretrained", type=float, default=1e-5)
    p.add_argument("--teacher-cache", type=Path)
    p.add_argument("--teacher-dev", type=Path, default=ROOT / "experiments/generated_prefix_teacher_20260910/paired2000_seed20261011")
    p.add_argument("--teacher-arm", choices=("generated", "reencoded"), default="generated")
    p.add_argument("--context-style", choices=("sparse", "fullctx", "basectx"), default="sparse")
    p.add_argument("--allow-teacher-smoke", action="store_true")
    p.add_argument("--train-mask-order", choices=("random", "halton", "halton_fixed", "raster"), default="random")
    p.add_argument("--rollin-visible-prob", type=float, default=0.0)
    p.add_argument("--rollin-draw", choices=("argmax", "categorical"), default="argmax")
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-name")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
