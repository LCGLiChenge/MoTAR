#!/usr/bin/env python3
"""Train unified TiTok-L32 + MoT/LlamaGen AR with a disjoint 1D sidecar on packed ImageNet codes."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from motar.checkpoint_disjoint import (
    recover_checkpoint_tree,
    save_latest_checkpoint,
)
from motar.data import PackedJointCodeDataset
from motar.model import TiTokLlamaGenUnifiedAR, split_new_and_base_params
from motar.sidecar import TiTok1DDisjointSidecar, detached_sidecar_inputs

logger = get_logger(__name__)


def flatten_config(obj: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_config(value, name))
    elif isinstance(obj, (list, tuple)):
        result[prefix] = ",".join(str(value) for value in obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        result[prefix] = "None" if obj is None else obj
    else:
        result[prefix] = str(obj)
    return result


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scale_learning_rate(
    base_lr: float,
    global_batch_size: int,
    reference_global_batch_size: int,
    mode: str,
) -> float:
    ratio = float(global_batch_size) / float(reference_global_batch_size)
    if mode == "none":
        factor = 1.0
    elif mode == "linear":
        factor = ratio
    elif mode == "sqrt":
        factor = math.sqrt(ratio)
    else:
        raise ValueError(f"Unsupported lr_scaling={mode!r}; use none, sqrt, or linear")
    return float(base_lr) * factor


def validate_registered_config(config, total_epochs: int) -> None:
    expected = {
        "model.n_layer": 24,
        "model.n_head": 16,
        "model.dim": 1024,
        "model.vocab_size": 16384,
        "model.block_size": 256,
        "model.titok_vocab_size": 4096,
        "model.titok_num_tokens": 32,
        "model.titok_conditioning": "prefix",
        "model.loss_1d_weight": 1.5,
        "model.loss_2d_weight": 1.0,
        "sidecar.depth": 4,
        "sidecar.aux_weight": 1.0,
        "optimizer.base_lr": 1.0e-4,
        "optimizer.new_lr": 2.0e-4,
        "optimizer.lr_scaling": "none",
        "scheduler.warmup_epochs": 0.225,
        "model.grad_checkpointing": True,
        "training.gradient_accumulation_steps": 1,
        "training.mixed_precision": "bf16",
        "training.log_with": "wandb",
        "h200.required_gpu_count": 8,
        "checkpoint.save_every_epochs": 1,
        "checkpoint.stable_name": "latest",
    }
    if total_epochs != 150:
        raise ValueError(
            f"This handoff is registered for exactly 150 total epochs, got {total_epochs}"
        )
    mismatches = {
        key: (OmegaConf.select(config, key), value)
        for key, value in expected.items()
        if OmegaConf.select(config, key) != value
    }
    if mismatches:
        raise ValueError(f"Registered H200 config invariants changed: {mismatches}")


def resolve_wandb_run_id(
    experiment_dir: Path,
    accelerator: Accelerator,
    *,
    resume: bool,
) -> str:
    run_id_path = experiment_dir / "wandb_run_id.txt"
    if accelerator.is_main_process:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        if resume and run_id_path.is_file():
            run_id = run_id_path.read_text().strip()
        else:
            run_id = os.environ.get("WANDB_RUN_ID", "").strip() or uuid.uuid4().hex[:16]
            run_id_path.write_text(run_id + "\n")
        if not run_id:
            raise RuntimeError(f"Empty W&B run id in {run_id_path}")
    accelerator.wait_for_everyone()
    run_id = run_id_path.read_text().strip()
    if not run_id:
        raise RuntimeError(f"Empty W&B run id in {run_id_path}")
    return run_id


def build_optimizer(model, config, global_batch_size: int):
    base_decay, base_nodecay, new_decay, new_nodecay = split_new_and_base_params(model)
    opt = config.optimizer
    scaled_base_lr = scale_learning_rate(
        float(opt.base_lr),
        global_batch_size,
        int(opt.reference_global_batch_size),
        str(opt.lr_scaling),
    )
    scaled_new_lr = scale_learning_rate(
        float(opt.new_lr),
        global_batch_size,
        int(opt.reference_global_batch_size),
        str(opt.lr_scaling),
    )
    definitions = (
        (base_decay, float(opt.weight_decay), scaled_base_lr, "base"),
        (base_nodecay, 0.0, scaled_base_lr, "base"),
        (new_decay, float(opt.weight_decay), scaled_new_lr, "new"),
        (new_nodecay, 0.0, scaled_new_lr, "new"),
    )
    groups = [
        {
            "params": parameters,
            "weight_decay": weight_decay,
            "lr": lr,
            "initial_lr": lr,
        }
        for parameters, weight_decay, lr, _role in definitions
        if parameters
    ]
    roles = [role for parameters, _weight_decay, _lr, role in definitions if parameters]
    fused_supported = "fused" in inspect.signature(torch.optim.AdamW).parameters
    kwargs = {"fused": True} if fused_supported and torch.cuda.is_available() else {}
    optimizer = torch.optim.AdamW(
        groups,
        betas=(float(opt.beta1), float(opt.beta2)),
        **kwargs,
    )
    return optimizer, roles, scaled_base_lr, scaled_new_lr


def build_sidecar_optimizer(sidecar, config, global_batch_size: int):
    opt = config.optimizer
    sidecar_lr = scale_learning_rate(
        float(config.sidecar.lr),
        global_batch_size,
        int(opt.reference_global_batch_size),
        str(opt.lr_scaling),
    )
    decay = [
        parameter
        for parameter in sidecar.parameters()
        if parameter.requires_grad and parameter.dim() >= 2
    ]
    nodecay = [
        parameter
        for parameter in sidecar.parameters()
        if parameter.requires_grad and parameter.dim() < 2
    ]
    groups = [
        {
            "params": parameters,
            "weight_decay": weight_decay,
            "lr": sidecar_lr,
            "initial_lr": sidecar_lr,
        }
        for parameters, weight_decay in (
            (decay, float(opt.weight_decay)),
            (nodecay, 0.0),
        )
        if parameters
    ]
    fused_supported = "fused" in inspect.signature(torch.optim.AdamW).parameters
    kwargs = {"fused": True} if fused_supported and torch.cuda.is_available() else {}
    optimizer = torch.optim.AdamW(
        groups,
        betas=(float(opt.beta1), float(opt.beta2)),
        **kwargs,
    )
    return optimizer, sidecar_lr


def reset_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    roles: list[str],
    base_lr: float,
    new_lr: float,
) -> None:
    if len(optimizer.param_groups) != len(roles):
        raise RuntimeError(
            f"Optimizer group count changed: {len(optimizer.param_groups)} != {len(roles)}"
        )
    for group, role in zip(optimizer.param_groups, roles):
        lr = new_lr if role == "new" else base_lr
        group["lr"] = lr
        group["initial_lr"] = lr


def cosine_epoch_lambda(
    *,
    start_progress_epochs: float,
    total_epochs: int,
    warmup_epochs: float,
    steps_per_epoch: int,
    min_lr_ratio: float,
    num_cycles: float,
):
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if not 0.0 <= start_progress_epochs < total_epochs:
        raise ValueError(
            f"start progress {start_progress_epochs} is outside [0,{total_epochs})"
        )

    def factor(update_index: int) -> float:
        absolute_epoch = start_progress_epochs + float(update_index) / steps_per_epoch
        if warmup_epochs > 0 and absolute_epoch < warmup_epochs:
            return max(absolute_epoch / warmup_epochs, 1.0e-8)
        denominator = max(float(total_epochs) - warmup_epochs, 1.0e-8)
        progress = min(max((absolute_epoch - warmup_epochs) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return factor


def resolve_resume(
    args,
    checkpoint_root: Path,
    accelerator: Accelerator,
) -> tuple[Path | None, float, int, int]:
    """Resolve only a native H200 latest checkpoint saved at an epoch boundary."""

    resume_path: Path | None = None
    progress_epochs = 0.0
    display_step = 0

    if args.resume:
        if accelerator.is_main_process:
            recover_checkpoint_tree(checkpoint_root)
        accelerator.wait_for_everyone()
        candidate = checkpoint_root / "latest"
        if candidate.is_dir():
            resume_path = candidate
        if resume_path is not None:
            metadata = json.loads((resume_path / "metadata.json").read_text())
            progress_epochs = float(metadata["completed_epochs"])
            if not progress_epochs.is_integer():
                raise ValueError(
                    "Native latest checkpoints must be saved at an epoch boundary"
                )
            display_step = int(metadata.get("global_step", 0))

    start_epoch = int(progress_epochs)
    return resume_path, progress_epochs, display_step, start_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h200_8gpu_150epoch_disjoint.yaml")
    parser.add_argument("--exp-name", default="motar_titok_l32_mot199440_disjoint_150ep_h200")
    parser.add_argument("--packed-code-root", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument(
        "--resume",
        choices=("auto",),
        default=None,
        help="resume only this experiment's checkpoints/latest after an interruption",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--total-epochs", type=int, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    config_path = Path(args.config).expanduser().resolve()
    config_text = config_path.read_text()
    config = OmegaConf.load(config_path)
    total_epochs = int(args.total_epochs or config.training.total_epochs)
    validate_registered_config(config, total_epochs)
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be positive")

    experiment_dir = Path(args.results_dir).expanduser().resolve() / args.exp_name
    checkpoint_root = experiment_dir / "checkpoints"
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(experiment_dir)),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
        mixed_precision=str(config.training.mixed_precision),
        log_with=config.training.get("log_with", None),
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
    )
    set_seed(int(config.training.seed) + accelerator.process_index)

    gpu_name = torch.cuda.get_device_name(accelerator.device)
    if "H200" not in gpu_name.upper():
        raise RuntimeError(f"Production training requires H200, found {gpu_name!r}")

    world_size = int(accelerator.num_processes)
    expected_world_size = int(config.h200.required_gpu_count)
    if world_size != expected_world_size:
        raise ValueError(
            f"Expected {expected_world_size} training processes for 8xH200, got {world_size}"
        )
    accumulation = int(config.training.gradient_accumulation_steps)
    global_batch_size = args.micro_batch_size * world_size * accumulation

    dataset = PackedJointCodeDataset(args.packed_code_root, validate=True)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=accelerator.process_index,
        shuffle=True,
        seed=int(config.training.seed),
        drop_last=True,
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else config.dataloader.num_workers
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.micro_batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(config.dataloader.pin_memory),
        "drop_last": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.dataloader.prefetch_factor)
    loader = DataLoader(**loader_kwargs)
    batches_per_epoch = len(loader)
    if batches_per_epoch < 1:
        raise ValueError(
            f"No full batches: samples={len(dataset)}, global_batch={global_batch_size}"
        )
    if accumulation != 1:
        raise ValueError("The H200 handoff currently requires gradient_accumulation_steps=1")
    steps_per_epoch = batches_per_epoch

    model = TiTokLlamaGenUnifiedAR(**OmegaConf.to_container(config.model, resolve=True))
    sidecar = TiTok1DDisjointSidecar(
        model,
        depth=int(config.sidecar.depth),
        drop_path=float(config.sidecar.drop_path),
    )
    optimizer, optimizer_roles, base_lr, new_lr = build_optimizer(
        model, config, global_batch_size
    )
    sidecar_optimizer, sidecar_lr = build_sidecar_optimizer(
        sidecar, config, global_batch_size
    )

    resume_path, start_progress, global_step, start_epoch = resolve_resume(
        args,
        checkpoint_root,
        accelerator,
    )
    if start_progress >= total_epochs:
        raise ValueError(
            f"Checkpoint progress={start_progress:.6f} already reached {total_epochs} epochs"
        )

    model, sidecar, optimizer, sidecar_optimizer = accelerator.prepare(
        model, sidecar, optimizer, sidecar_optimizer
    )
    if resume_path is not None:
        accelerator.load_state(str(resume_path))
        reset_optimizer_lrs(optimizer, optimizer_roles, base_lr, new_lr)
        for group in sidecar_optimizer.param_groups:
            group["lr"] = sidecar_lr
            group["initial_lr"] = sidecar_lr

    lr_lambda = cosine_epoch_lambda(
        start_progress_epochs=start_progress,
        total_epochs=total_epochs,
        warmup_epochs=float(config.scheduler.warmup_epochs),
        steps_per_epoch=steps_per_epoch,
        min_lr_ratio=float(config.scheduler.min_lr_ratio),
        num_cycles=float(config.scheduler.num_cycles),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    sidecar_scheduler = LambdaLR(sidecar_optimizer, lr_lambda=lr_lambda)
    sidecar_aux_weight = float(config.sidecar.aux_weight)
    wandb_project = os.environ.get("WANDB_PROJECT", "motar-unified-ar")
    wandb_run_id = resolve_wandb_run_id(
        experiment_dir, accelerator, resume=resume_path is not None
    )

    if accelerator.is_main_process:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        resolved_config = OmegaConf.to_yaml(config, resolve=True)
        (experiment_dir / "resolved_config.yaml").write_text(resolved_config)
        plan = {
            "config": str(config_path),
            "config_sha256": sha256_text(config_text),
            "dataset_root": str(dataset.root),
            "dataset_size": len(dataset),
            "world_size": world_size,
            "micro_batch_size_per_gpu": args.micro_batch_size,
            "gradient_accumulation_steps": accumulation,
            "global_batch_size": global_batch_size,
            "batches_per_epoch": batches_per_epoch,
            "total_epochs": total_epochs,
            "start_progress_epochs": start_progress,
            "start_epoch": start_epoch,
            "base_lr": base_lr,
            "new_lr": new_lr,
            "sidecar_lr": sidecar_lr,
            "sidecar_depth": int(config.sidecar.depth),
            "sidecar_aux_weight": sidecar_aux_weight,
            "architecture": "disjoint_1d_sidecar",
            "lr_scaling": str(config.optimizer.lr_scaling),
            "resume": str(resume_path) if resume_path else None,
            "checkpoint_policy": "one stable checkpoints/latest, replaced every epoch",
            "wandb_project": wandb_project,
            "wandb_run_id": wandb_run_id,
        }
        (experiment_dir / "run_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
        logger.info("Run plan: %s", json.dumps(plan, sort_keys=True))

    accelerator.init_trackers(
        project_name=wandb_project,
        config=flatten_config(OmegaConf.to_container(config, resolve=True)),
        init_kwargs={
            "wandb": {
                "name": args.exp_name,
                "id": wandb_run_id,
                "resume": "allow",
            }
        },
    )

    log_every = int(args.log_every or config.training.log_every)
    running = {
        "loss": 0.0,
        "loss_1d": 0.0,
        "loss_1d_main": 0.0,
        "loss_2d": 0.0,
        "acc_1d": 0.0,
        "acc_1d_main": 0.0,
        "acc_2d": 0.0,
        "grad_norm_main": 0.0,
        "grad_norm_sidecar": 0.0,
    }
    running_updates = 0
    started = time.time()
    model.train()
    sidecar.train()

    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            z1d, z2d, labels, _indices = batch
            z1d = z1d.to(accelerator.device, non_blocking=True)
            z2d = z2d.to(accelerator.device, non_blocking=True)
            labels = labels.to(accelerator.device, non_blocking=True)

            with accelerator.accumulate(model, sidecar):
                output = model(
                    z1d=z1d,
                    z2d=z2d,
                    cond_idx=labels,
                    targets_1d=z1d,
                    targets_2d=z2d,
                )
                devices = [z1d.device] if z1d.is_cuda else []
                with torch.random.fork_rng(devices=devices, enabled=True):
                    prefix_h, prefix_freqs = detached_sidecar_inputs(
                        accelerator.unwrap_model(model),
                        z1d,
                        labels,
                    )
                    sidecar_logits = sidecar(prefix_h, prefix_freqs)
                combined_logits_1d = output["logits_1d"].detach() + sidecar_logits
                sidecar_loss = F.cross_entropy(
                    combined_logits_1d.reshape(-1, combined_logits_1d.shape[-1]),
                    z1d.reshape(-1),
                )
                main_loss = output["loss"]
                if not torch.isfinite(main_loss) or not torch.isfinite(sidecar_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch} batch={batch_index}: "
                        f"main={main_loss} sidecar={sidecar_loss}"
                    )

                accelerator.backward(main_loss)
                accelerator.backward(sidecar_aux_weight * sidecar_loss)
                grad_norm_main = torch.zeros([], device=accelerator.device)
                grad_norm_sidecar = torch.zeros([], device=accelerator.device)
                if float(config.optimizer.max_grad_norm) != 0.0:
                    grad_norm_main = accelerator.clip_grad_norm_(
                        model.parameters(), float(config.optimizer.max_grad_norm)
                    )
                    grad_norm_sidecar = accelerator.clip_grad_norm_(
                        sidecar.parameters(), float(config.optimizer.max_grad_norm)
                    )
                absolute_progress = epoch + float(batch_index + 1) / batches_per_epoch
                before_skip_window = (
                    absolute_progress < float(config.optimizer.skip_grad_epochs)
                )
                if (
                    float(grad_norm_main.detach()) < float(config.optimizer.skip_grad_norm)
                    or before_skip_window
                ):
                    optimizer.step()
                if (
                    float(grad_norm_sidecar.detach()) < float(config.optimizer.skip_grad_norm)
                    or before_skip_window
                ):
                    sidecar_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                sidecar_optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                sidecar_scheduler.step()

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            combined_acc_1d = (
                combined_logits_1d.argmax(dim=-1) == z1d
            ).float().mean().detach()
            logging_loss = main_loss.detach() + sidecar_aux_weight * sidecar_loss.detach()
            gathered = {
                "loss": accelerator.gather(logging_loss).mean().item(),
                "loss_1d": accelerator.gather(sidecar_loss.detach()).mean().item(),
                "loss_1d_main": accelerator.gather(output["loss_1d"]).mean().item(),
                "loss_2d": accelerator.gather(output["loss_2d"]).mean().item(),
                "acc_1d": accelerator.gather(combined_acc_1d).mean().item(),
                "acc_1d_main": accelerator.gather(output["acc_1d"]).mean().item(),
                "acc_2d": accelerator.gather(output["acc_2d"]).mean().item(),
                "grad_norm_main": accelerator.gather(
                    grad_norm_main.detach()
                ).mean().item(),
                "grad_norm_sidecar": accelerator.gather(
                    grad_norm_sidecar.detach()
                ).mean().item(),
            }
            for key, value in gathered.items():
                running[key] += value
            running_updates += 1

            if running_updates >= log_every:
                elapsed = (time.time() - started) / running_updates
                metrics = {key: value / running_updates for key, value in running.items()}
                metrics.update(
                    {
                        "lr": scheduler.get_last_lr()[0],
                        "lr_main": scheduler.get_last_lr()[0],
                        "lr_sidecar": sidecar_scheduler.get_last_lr()[0],
                        "time_per_step": elapsed,
                        "epoch_progress": epoch
                        + float(batch_index + 1) / batches_per_epoch,
                        "global_batch_size": global_batch_size,
                    }
                )
                if accelerator.is_main_process:
                    message = (
                        "epoch {epoch_progress:.4f}/{total_epochs} "
                        "| step {global_step:08d} | loss {loss:.4f} "
                        "| 1d-combined {loss_1d:.4f}/{acc_1d:.4f} "
                        "| 1d-main {loss_1d_main:.4f}/{acc_1d_main:.4f} "
                        "| 2d {loss_2d:.4f}/{acc_2d:.4f} "
                        "| grad main/side {grad_norm_main:.4f}/{grad_norm_sidecar:.4f} "
                        "| lr main/side {lr_main:.6g}/{lr_sidecar:.6g} "
                        "| {elapsed:.3f}s"
                    ).format(
                        total_epochs=total_epochs,
                        global_step=global_step,
                        elapsed=elapsed,
                        **metrics,
                    )
                    accelerator.print(message)
                    logger.info(message)
                accelerator.log(metrics, step=global_step)
                running = {key: 0.0 for key in running}
                running_updates = 0
                started = time.time()

        completed_epochs = epoch + 1
        metadata = {
            "completed_epochs": completed_epochs,
            "total_epochs": total_epochs,
            "global_step": global_step,
            "dataset_size": len(dataset),
            "world_size": world_size,
            "micro_batch_size_per_gpu": args.micro_batch_size,
            "gradient_accumulation_steps": accumulation,
            "global_batch_size": global_batch_size,
            "batches_per_epoch": batches_per_epoch,
            "base_lr": base_lr,
            "new_lr": new_lr,
            "sidecar_lr": sidecar_lr,
            "sidecar_depth": int(config.sidecar.depth),
            "sidecar_aux_weight": sidecar_aux_weight,
            "architecture": "disjoint_1d_sidecar",
            "lr_scaling": str(config.optimizer.lr_scaling),
            "config_sha256": sha256_text(config_text),
            "loss_1d_weight": float(config.model.loss_1d_weight),
            "loss_2d_weight": float(config.model.loss_2d_weight),
            "source_checkpoint": str(resume_path) if resume_path else None,
            "initial_progress_epochs": start_progress,
            "checkpoint_policy": "latest-only; replaced after every epoch",
            "wandb_project": wandb_project,
            "wandb_run_id": wandb_run_id,
            "saved_unix_time": time.time(),
        }
        latest = save_latest_checkpoint(accelerator, checkpoint_root, metadata)
        if accelerator.is_main_process:
            logger.info(
                "Completed epoch %d/%d; updated %s",
                completed_epochs,
                total_epochs,
                latest,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
