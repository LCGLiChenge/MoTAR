#!/usr/bin/env python3
"""Train and validate hierarchical E117-routed sparse unified MaskGIT."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from e117_sparse_data import (
    E117SparseCodeDataset,
    KBucketBatchSampler,
    PackedOneDCodeDataset,
    collate_e117_sparse,
)
from e117_sparse_maskgit import (
    E117SparseUnifiedMaskGIT,
    fixed_ratio_mask,
    masked_token_metrics,
    sample_arccos_mask,
)


def flatten_config(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_config(item, name))
    elif isinstance(value, (list, tuple)):
        result[prefix] = ",".join(str(item) for item in value)
    else:
        result[prefix] = value
    return result


def optimizer_for(model, config):
    decay, no_decay = [], []
    no_decay_terms = (
        "ln",
        "bias",
        "latent_tokens",
        "mask_token",
        "embedding",
        "norm",
        "gamma",
        "embed",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = parameter.ndim < 2 or any(term in name.lower() for term in no_decay_terms)
        (no_decay if excluded else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": float(config.weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    return torch.optim.AdamW(
        groups,
        lr=float(config.lr),
        betas=(float(config.beta1), float(config.beta2)),
        fused=fused,
    )


@torch.no_grad()
def update_ema_model(ema_model, source_model, decay):
    """Update a full-model fp32 EMA exactly once per optimizer step."""

    ema_parameters = dict(ema_model.named_parameters())
    source_parameters = dict(source_model.named_parameters())
    if ema_parameters.keys() != source_parameters.keys():
        raise ValueError("EMA/source parameter names do not match")
    for name, ema_parameter in ema_parameters.items():
        source_parameter = source_parameters[name].detach().to(dtype=ema_parameter.dtype)
        ema_parameter.mul_(float(decay)).add_(source_parameter, alpha=1.0 - float(decay))
    ema_buffers = dict(ema_model.named_buffers())
    source_buffers = dict(source_model.named_buffers())
    if ema_buffers.keys() != source_buffers.keys():
        raise ValueError("EMA/source buffer names do not match")
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(source_buffers[name].detach().to(dtype=ema_buffer.dtype))


def scheduler_for(optimizer, warmup_steps, max_steps, min_lr_ratio):
    def multiplier(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / float(max(1, max_steps - warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def scheduler_step_index(scheduler):
    """Return the underlying PyTorch step, including after Accelerate wrapping."""

    return int(getattr(scheduler, "scheduler", scheduler).last_epoch)


def resolved_config_sha256(config):
    payload = json.dumps(
        OmegaConf.to_container(config, resolve=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(state["torch_cuda"])


def exposure_audit(config, world_size, num_train_sources):
    accumulation = int(config.training.gradient_accumulation_steps)
    global_batch = int(config.training.per_gpu_batch_size) * int(world_size) * accumulation
    replay_per_gpu_batch = int(
        config.training.get(
            "replay_1d_per_gpu_batch_size",
            config.training.per_gpu_batch_size,
        )
    )
    replay_global_batch = replay_per_gpu_batch * int(world_size) * accumulation
    max_steps = int(config.training.max_steps)
    target = int(config.training.get("target_image_exposures", max_steps * global_batch))
    observed = max_steps * global_batch
    exposure_error = observed - target
    # H200 batch selection is empirical. An arbitrary winning micro-batch may
    # not divide the fixed TiTok reference exposure exactly, so the launcher
    # rounds to the nearest optimizer step. The discrepancy must stay within
    # half a global batch (one rounded step).
    if abs(exposure_error) > global_batch // 2:
        raise ValueError(
            "2D exposure mismatch exceeds one rounded optimizer step: "
            f"max_steps*global_batch={observed:,}, target={target:,}, "
            f"error={exposure_error:+,}"
        )
    if bool(config.data.get("full_1d_replay", False)) and replay_global_batch != global_batch:
        if bool(config.training.get("require_equal_branch_exposures", False)):
            raise ValueError(
                f"1D/2D global batches differ: {replay_global_batch} != {global_batch}"
            )
    expected_sources = config.training.get("reference_num_train_sources", None)
    if expected_sources is not None and int(expected_sources) != int(num_train_sources):
        raise ValueError(
            f"training source count mismatch: {num_train_sources:,} != {int(expected_sources):,}"
        )
    reference_steps = config.training.get("reference_max_steps", None)
    reference_batch = config.training.get("reference_global_batch_size", None)
    if reference_steps is not None and reference_batch is not None:
        reference_exposures = int(reference_steps) * int(reference_batch)
        if target != reference_exposures:
            raise ValueError(
                f"reference exposure mismatch: target={target:,}, reference={reference_exposures:,}"
            )
    return {
        "global_batch_2d": global_batch,
        "global_batch_1d": replay_global_batch,
        "target_image_exposures": target,
        "planned_image_exposures": observed,
        "exposure_error": exposure_error,
        "num_train_sources": int(num_train_sources),
        "source_equivalent_epochs": float(observed) / float(num_train_sources),
    }


def load_pretrained_1d_stage(model, checkpoint_path, state_key="model"):
    """Load only the verified 1D/shared state from a pre-norm control.

    The 1D control wraps this same unified class under ``core`` but instantiates
    one-code placeholder 2D modules. Those four module families must remain
    newly initialized for the real 16k sparse stage. Every other persistent
    tensor is required and shape-checked so a partial architectural mismatch
    cannot silently become an experiment.
    """

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if state_key not in checkpoint or not isinstance(checkpoint[state_key], dict):
        raise KeyError(f"checkpoint has no state dict at {state_key!r}: {checkpoint_path}")
    source = checkpoint[state_key]
    target = model.state_dict()
    two_d_prefixes = ("embedding_2d.", "pos_embedding_2d.", "budget_embedding.", "output_2d.")
    mapped = {}
    skipped_2d = []
    unexpected = []
    for source_name, tensor in source.items():
        name = source_name.removeprefix("module.").removeprefix("core.")
        if name.startswith(two_d_prefixes):
            skipped_2d.append(source_name)
            continue
        if name not in target:
            unexpected.append(source_name)
            continue
        if tensor.shape != target[name].shape:
            raise ValueError(
                f"1D initialization shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(target[name].shape)}"
            )
        mapped[name] = tensor

    required = {name for name in target if not name.startswith(two_d_prefixes)}
    missing_required = sorted(required - mapped.keys())
    if missing_required or unexpected:
        raise RuntimeError(
            "incompatible 1D initialization: "
            f"missing_required={missing_required}, unexpected={sorted(unexpected)}"
        )
    incompatible = model.load_state_dict(mapped, strict=False)
    expected_missing = sorted(name for name in target if name.startswith(two_d_prefixes))
    if sorted(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "unexpected load_state_dict result: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    report = {
        "checkpoint": str(checkpoint_path),
        "state_key": str(state_key),
        "loaded_tensors": len(mapped),
        "loaded_parameters": int(sum(target[name].numel() for name in mapped)),
        "random_2d_tensors": expected_missing,
        "skipped_source_2d_tensors": sorted(skipped_2d),
        "source_step": int(checkpoint.get("global_step", -1)),
    }
    del checkpoint, source, mapped
    gc.collect()
    return report


def masked_inputs_1d(model, targets, mask):
    return torch.where(mask, torch.full_like(targets, model.mask_token_1d), targets)


def masked_inputs_2d(model, targets, mask, valid):
    inputs = torch.where(mask, torch.full_like(targets, model.mask_token_2d), targets)
    return torch.where(valid, inputs, torch.full_like(inputs, model.pad_token_2d))


@torch.no_grad()
def evaluate_global_1d(model, loader, accelerator, config):
    """Evaluate retention on the original source-disjoint full-packed holdout."""

    if loader is None:
        return {}
    unwrapped = accelerator.unwrap_model(model)
    totals = {
        "global_loss_1d": 0.0,
        "global_acc_1d": 0.0,
        "global_top5_1d": 0.0,
        "global_loss_1d_shuffled_visible": 0.0,
        "global_loss_1d_shuffled_label": 0.0,
        "count": 0.0,
    }
    ratio = float(config.feasibility.eval_mask_ratio)
    eval_seed = int(config.feasibility.eval_seed) + 10_000_000
    max_batches = int(config.feasibility.get("eval_1d_batches", 0))
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        z1d = batch["z1d"]
        labels = batch["label"]
        valid = torch.ones_like(z1d, dtype=torch.bool)
        masked = fixed_ratio_mask(valid, ratio, seed=eval_seed + batch_index)
        inputs = masked_inputs_1d(unwrapped, z1d, masked)
        permutation = torch.arange(z1d.shape[0], device=z1d.device).roll(1)
        shuffled_visible = torch.where(
            masked,
            torch.full_like(z1d, unwrapped.mask_token_1d),
            z1d[permutation],
        )
        keep_condition = torch.zeros_like(labels)
        with accelerator.autocast():
            logits = model(
                stage="1d", input_tokens=inputs, labels=labels, force_drop_ids=keep_condition
            )
            logits_shuffled_visible = model(
                stage="1d",
                input_tokens=shuffled_visible,
                labels=labels,
                force_drop_ids=keep_condition,
            )
            logits_shuffled_label = model(
                stage="1d",
                input_tokens=inputs,
                labels=labels[permutation],
                force_drop_ids=keep_condition,
            )
        metric_args = (
            z1d,
            masked,
            valid,
            config.training.label_smoothing,
            0.0,
            config.training.get("loss_normalization", "per_sample"),
        )
        metrics = masked_token_metrics(logits, *metric_args)
        shuffled_visible_metrics = masked_token_metrics(logits_shuffled_visible, *metric_args)
        shuffled_label_metrics = masked_token_metrics(logits_shuffled_label, *metric_args)
        batch_size = float(z1d.shape[0])
        values = torch.stack(
            [
                metrics["masked_loss"].double(),
                metrics["masked_acc"].double(),
                metrics["masked_top5"].double(),
                shuffled_visible_metrics["masked_loss"].double(),
                shuffled_label_metrics["masked_loss"].double(),
                torch.tensor(batch_size, device=accelerator.device, dtype=torch.float64),
            ]
        )
        values[:-1] *= batch_size
        gathered = accelerator.gather(values[None]).sum(dim=0).cpu().tolist()
        for key, value in zip(totals, gathered):
            totals[key] += float(value)
    count = max(totals.pop("count"), 1.0)
    result = {key: value / count for key, value in totals.items()}
    result["global_context_loss_delta_1d"] = (
        result["global_loss_1d_shuffled_visible"] - result["global_loss_1d"]
    )
    result["global_label_loss_delta_1d"] = (
        result["global_loss_1d_shuffled_label"] - result["global_loss_1d"]
    )
    return result


@torch.no_grad()
def evaluate(model, loader, accelerator, config, global_step, retention_loader=None):
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    totals = {
        "loss_1d": 0.0,
        "loss_2d": 0.0,
        "acc_1d": 0.0,
        "acc_2d": 0.0,
        "top5_1d": 0.0,
        "top5_2d": 0.0,
        "loss_1d_shuffled_visible": 0.0,
        "loss_1d_shuffled_label": 0.0,
        "loss_2d_shuffled_visible": 0.0,
        "loss_2d_shuffled_label": 0.0,
        "loss_2d_shuffled_1d_prefix": 0.0,
        "count": 0.0,
    }
    ratio = float(config.feasibility.eval_mask_ratio)
    eval_seed = int(config.feasibility.eval_seed)
    max_eval_batches = int(config.feasibility.eval_batches)
    for batch_index, batch in enumerate(loader):
        if max_eval_batches > 0 and batch_index >= max_eval_batches:
            break
        z1d = batch["z1d"]
        z2d = batch["z2d"]
        labels = batch["label"]
        route_indices = batch["route_indices"]
        route_valid = batch["route_valid"]
        valid_1d = torch.ones_like(z1d, dtype=torch.bool)
        mask_1d = fixed_ratio_mask(valid_1d, ratio, seed=eval_seed + 2 * batch_index)
        mask_2d = fixed_ratio_mask(route_valid, ratio, seed=eval_seed + 2 * batch_index + 1)
        input_1d = masked_inputs_1d(unwrapped, z1d, mask_1d)
        input_2d = masked_inputs_2d(unwrapped, z2d, mask_2d, route_valid)
        permutation = torch.arange(z1d.shape[0], device=z1d.device).roll(1)
        shuffled_visible_1d = torch.where(
            mask_1d,
            torch.full_like(z1d, unwrapped.mask_token_1d),
            z1d[permutation],
        )
        shuffled_visible_2d = torch.where(
            mask_2d,
            torch.full_like(z2d, unwrapped.mask_token_2d),
            z2d[permutation],
        )
        shuffled_visible_2d = torch.where(
            route_valid,
            shuffled_visible_2d,
            torch.full_like(shuffled_visible_2d, unwrapped.pad_token_2d),
        )
        keep_condition = torch.zeros_like(labels)
        with accelerator.autocast():
            logits_1d = model(
                stage="1d",
                input_tokens=input_1d,
                labels=labels,
                force_drop_ids=keep_condition,
            )
            logits_2d = model(
                stage="2d",
                completed_1d=z1d,
                input_tokens=input_2d,
                route_indices=route_indices,
                route_valid=route_valid,
                labels=labels,
                force_drop_ids=keep_condition,
            )
            logits_1d_shuffled_visible = model(
                stage="1d",
                input_tokens=shuffled_visible_1d,
                labels=labels,
                force_drop_ids=keep_condition,
            )
            logits_1d_shuffled_label = model(
                stage="1d",
                input_tokens=input_1d,
                labels=labels[permutation],
                force_drop_ids=keep_condition,
            )
            logits_2d_shuffled_visible = model(
                stage="2d",
                completed_1d=z1d,
                input_tokens=shuffled_visible_2d,
                route_indices=route_indices,
                route_valid=route_valid,
                labels=labels,
                force_drop_ids=keep_condition,
            )
            logits_2d_shuffled_label = model(
                stage="2d",
                completed_1d=z1d,
                input_tokens=input_2d,
                route_indices=route_indices,
                route_valid=route_valid,
                labels=labels[permutation],
                force_drop_ids=keep_condition,
            )
            logits_2d_shuffled_1d_prefix = model(
                stage="2d",
                completed_1d=z1d[permutation],
                input_tokens=input_2d,
                route_indices=route_indices,
                route_valid=route_valid,
                labels=labels,
                force_drop_ids=keep_condition,
            )
        metrics_1d = masked_token_metrics(
            logits_1d,
            z1d,
            mask_1d,
            valid_1d,
            config.training.label_smoothing,
            0.0,
            config.training.get("loss_normalization", "per_sample"),
        )
        metrics_2d = masked_token_metrics(
            logits_2d,
            z2d,
            mask_2d,
            route_valid,
            config.training.label_smoothing,
            0.0,
            config.training.get("loss_normalization", "per_sample"),
        )
        metrics_1d_shuffled_visible = masked_token_metrics(
            logits_1d_shuffled_visible,
            z1d,
            mask_1d,
            valid_1d,
            config.training.label_smoothing,
            0.0,
            config.training.get("loss_normalization", "per_sample"),
        )
        metrics_1d_shuffled_label = masked_token_metrics(
            logits_1d_shuffled_label,
            z1d,
            mask_1d,
            valid_1d,
            config.training.label_smoothing,
            0.0,
            config.training.get("loss_normalization", "per_sample"),
        )
        metrics_2d_shuffled_visible = masked_token_metrics(
            logits_2d_shuffled_visible,
            z2d,
            mask_2d,
            route_valid,
            config.training.label_smoothing,
            0.0,
            config.training.get("loss_normalization", "per_sample"),
        )
        metrics_2d_shuffled_label = masked_token_metrics(
            logits_2d_shuffled_label,
            z2d,
            mask_2d,
            route_valid,
            config.training.label_smoothing,
            0.0,
            config.training.get("loss_normalization", "per_sample"),
        )
        metrics_2d_shuffled_1d_prefix = masked_token_metrics(
            logits_2d_shuffled_1d_prefix,
            z2d,
            mask_2d,
            route_valid,
            config.training.label_smoothing,
            0.0,
            config.training.get("loss_normalization", "per_sample"),
        )
        batch_size = float(z1d.shape[0])
        values = torch.stack(
            [
                metrics_1d["masked_loss"].double(),
                metrics_2d["masked_loss"].double(),
                metrics_1d["masked_acc"].double(),
                metrics_2d["masked_acc"].double(),
                metrics_1d["masked_top5"].double(),
                metrics_2d["masked_top5"].double(),
                metrics_1d_shuffled_visible["masked_loss"].double(),
                metrics_1d_shuffled_label["masked_loss"].double(),
                metrics_2d_shuffled_visible["masked_loss"].double(),
                metrics_2d_shuffled_label["masked_loss"].double(),
                metrics_2d_shuffled_1d_prefix["masked_loss"].double(),
                torch.tensor(batch_size, device=accelerator.device, dtype=torch.float64),
            ],
        )
        values[:-1] *= batch_size
        gathered = accelerator.gather(values[None]).sum(dim=0).cpu().tolist()
        for key, value in zip(totals, gathered):
            totals[key] += float(value)
    count = max(totals.pop("count"), 1.0)
    result = {key: value / count for key, value in totals.items()}
    result["context_loss_delta_1d"] = result["loss_1d_shuffled_visible"] - result["loss_1d"]
    result["label_loss_delta_1d"] = result["loss_1d_shuffled_label"] - result["loss_1d"]
    result["context_loss_delta_2d"] = result["loss_2d_shuffled_visible"] - result["loss_2d"]
    result["label_loss_delta_2d"] = result["loss_2d_shuffled_label"] - result["loss_2d"]
    result["prefix_loss_delta_2d"] = result["loss_2d_shuffled_1d_prefix"] - result["loss_2d"]

    sample_batch = next(iter(loader))
    sample_labels = sample_batch["label"][: int(config.feasibility.sample_count)]
    sample_routes = sample_batch["route_indices"][: sample_labels.shape[0]]
    sample_valid = sample_batch["route_valid"][: sample_labels.shape[0]]
    generated_1d = unwrapped.generate_1d(
        sample_labels,
        num_steps=int(config.sampling.steps_1d),
        cfg_scale=1.0,
        randomize_temperature=0.0,
        guidance_decay="constant",
        cfg_formula="standard",
        softmax_temperature_annealing=False,
    )
    generated_2d = unwrapped.generate_2d(
        generated_1d,
        sample_routes,
        sample_valid,
        sample_labels,
        num_steps=int(config.sampling.steps_2d),
        cfg_scale=1.0,
        randomize_temperature=0.0,
        guidance_decay="constant",
        cfg_formula="standard",
        softmax_temperature_annealing=False,
    )
    result["sample_valid"] = bool(
        generated_1d.shape == (sample_labels.shape[0], 32)
        and torch.all((generated_1d >= 0) & (generated_1d < unwrapped.titok_vocab_size))
        and torch.all(
            (~sample_valid)
            | ((generated_2d >= 0) & (generated_2d < unwrapped.llamagen_vocab_size))
        )
    )
    result.update(evaluate_global_1d(model, retention_loader, accelerator, config))
    model.train()
    return result


def feasibility_status(baseline, current, config):
    drop_1d = 1.0 - current["loss_1d"] / baseline["loss_1d"]
    drop_2d = 1.0 - current["loss_2d"] / baseline["loss_2d"]
    random_top5_1d = 5.0 / 4096.0
    random_top5_2d = 5.0 / 16384.0
    checks = {
        "finite": all(math.isfinite(float(value)) for key, value in current.items() if key != "sample_valid"),
        "top5_1d": current["top5_1d"] >= float(config.feasibility.random_top5_multiplier) * random_top5_1d,
        "top5_2d": current["top5_2d"] >= float(config.feasibility.random_top5_multiplier) * random_top5_2d,
        "sample_valid": bool(current["sample_valid"]),
    }
    if config.feasibility.get("min_relative_loss_drop_1d", None) is not None:
        checks["loss_drop_1d"] = drop_1d >= float(config.feasibility.min_relative_loss_drop_1d)
    if config.feasibility.get("min_relative_loss_drop_2d", None) is not None:
        checks["loss_drop_2d"] = drop_2d >= float(config.feasibility.min_relative_loss_drop_2d)
    relative_increase_1d = current["loss_1d"] / baseline["loss_1d"] - 1.0
    if config.feasibility.get("max_relative_loss_increase_1d", None) is not None:
        checks["retain_loss_1d"] = relative_increase_1d <= float(
            config.feasibility.max_relative_loss_increase_1d
        )
    global_relative_increase_1d = None
    if "global_loss_1d" in current:
        global_relative_increase_1d = current["global_loss_1d"] / baseline["global_loss_1d"] - 1.0
    if config.feasibility.get("max_global_relative_loss_increase_1d", None) is not None:
        if global_relative_increase_1d is None:
            raise KeyError("global 1D retention gate configured without global evaluation metrics")
        checks["retain_global_loss_1d"] = global_relative_increase_1d <= float(
            config.feasibility.max_global_relative_loss_increase_1d
        )
    if config.feasibility.get("min_global_context_loss_delta_1d", None) is not None:
        checks["global_context_delta_1d"] = current["global_context_loss_delta_1d"] >= float(
            config.feasibility.min_global_context_loss_delta_1d
        )
    if config.feasibility.get("min_global_label_loss_delta_1d", None) is not None:
        checks["global_label_delta_1d"] = current["global_label_loss_delta_1d"] >= float(
            config.feasibility.min_global_label_loss_delta_1d
        )
    if config.feasibility.get("min_global_top5_multiplier_1d", None) is not None:
        checks["global_top5_1d"] = current["global_top5_1d"] >= float(
            config.feasibility.min_global_top5_multiplier_1d
        ) * random_top5_1d
    optional_control_checks = {
        "context_delta_1d": "min_context_loss_delta_1d",
        "label_delta_1d": "min_label_loss_delta_1d",
        "context_delta_2d": "min_context_loss_delta_2d",
        "label_delta_2d": "min_label_loss_delta_2d",
        "prefix_delta_2d": "min_prefix_loss_delta_2d",
    }
    for metric_suffix, config_key in optional_control_checks.items():
        if config.feasibility.get(config_key, None) is not None:
            checks[metric_suffix] = current[f"{metric_suffix.replace('_delta', '_loss_delta')}"] >= float(
                config.feasibility[config_key]
            )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_loss_drop_1d": drop_1d,
        "relative_loss_drop_2d": drop_2d,
        "relative_loss_increase_1d": relative_increase_1d,
        "global_relative_loss_increase_1d": global_relative_increase_1d,
    }


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_latest(
    accelerator,
    model,
    ema_model,
    optimizer,
    scheduler,
    output_dir,
    config,
    step,
    epoch,
    replay_generator=None,
):
    local_rng_state = capture_rng_state()
    if dist.is_available() and dist.is_initialized():
        rng_states = [None] * accelerator.num_processes
        dist.all_gather_object(rng_states, local_rng_state)
    else:
        rng_states = [local_rng_state]
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_config = config.get("checkpoint", {})
        compact = bool(checkpoint_config.get("compact", False))
        state = {
            "format": "e117_sparse_unified_maskgit_v3",
            "model": accelerator.get_state_dict(model),
            "global_step": int(step),
            "epoch": int(epoch),
            "config": OmegaConf.to_container(config, resolve=True),
            "config_sha256": resolved_config_sha256(config),
            "resume_capable": not compact,
            "rng_states": rng_states,
        }
        if not compact:
            state["optimizer"] = optimizer.state_dict()
            state["scheduler"] = scheduler.state_dict()
        if ema_model is not None:
            state["model_ema"] = ema_model.state_dict()
        if replay_generator is not None:
            state["replay_generator_state"] = replay_generator.get_state()
        destination = output_dir / "latest.pt"
        temporary = output_dir / "latest.pt.tmp"
        torch.save(state, temporary)
        os.replace(temporary, destination)
    accelerator.wait_for_everyone()


def resolve_resume_path(output_dir, config):
    checkpoint_config = config.get("checkpoint", {})
    explicit = checkpoint_config.get("resume_from", None)
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidate = output_dir / "latest.pt"
    if bool(checkpoint_config.get("auto_resume", False)) and candidate.is_file():
        return candidate
    return None


def restore_training_state(
    accelerator,
    model,
    ema_model,
    optimizer,
    scheduler,
    checkpoint_path,
    config,
    replay_generator=None,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not bool(checkpoint.get("resume_capable", False)):
        raise ValueError(f"checkpoint is not resume-capable: {checkpoint_path}")
    required = {"model", "optimizer", "scheduler", "global_step", "epoch", "rng_states"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise KeyError(f"resume checkpoint is missing fields: {missing}")
    expected_digest = resolved_config_sha256(config)
    if checkpoint.get("config_sha256") != expected_digest:
        raise ValueError("resume config SHA-256 does not match the current resolved config")
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if ema_model is not None:
        if "model_ema" not in checkpoint:
            raise KeyError("EMA is enabled but resume checkpoint has no model_ema")
        ema_model.load_state_dict(checkpoint["model_ema"], strict=True)
    elif "model_ema" in checkpoint:
        raise ValueError("resume checkpoint has EMA but current config disables it")
    if replay_generator is not None:
        if "replay_generator_state" not in checkpoint:
            raise KeyError("full replay resume requires replay_generator_state")
        replay_generator.set_state(checkpoint["replay_generator_state"])
    global_step = int(checkpoint["global_step"])
    epoch = int(checkpoint["epoch"])
    if scheduler_step_index(scheduler) != global_step:
        raise RuntimeError(
            f"resumed scheduler/global-step mismatch: {scheduler_step_index(scheduler)} != {global_step}"
        )
    rng_states = checkpoint["rng_states"]
    if len(rng_states) != accelerator.num_processes:
        raise ValueError(
            f"resume world size changed: {len(rng_states)} != {accelerator.num_processes}"
        )
    local_rng_state = rng_states[accelerator.process_index]
    del checkpoint
    gc.collect()
    return global_step, epoch, local_rng_state


def main(args):
    config = OmegaConf.load(args.config)
    enable_tf32 = bool(config.training.get("enable_tf32", True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
    output_dir = Path(args.output_dir or config.experiment.output_dir).resolve()
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        mixed_precision=str(config.training.mixed_precision),
        log_with=config.logging.get("log_with", None),
        project_config=ProjectConfiguration(project_dir=str(output_dir)),
        # We manually step once per synchronized optimizer update.  Accelerate's
        # default otherwise advances a prepared scheduler once per DDP rank.
        step_scheduler_with_optimizer=False,
    )
    required_world_size = config.training.get("required_world_size", None)
    if required_world_size is not None and accelerator.num_processes != int(required_world_size):
        raise RuntimeError(
            f"this config requires {int(required_world_size)} processes, "
            f"found {accelerator.num_processes}"
        )
    if bool(config.logging.get("require_wandb", False)) and config.logging.get(
        "log_with", None
    ) != "wandb":
        raise RuntimeError("this config requires logging.log_with=wandb")
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")
    accelerator.wait_for_everyone()
    set_seed(int(config.training.seed) + accelerator.process_index)

    train_all_sources = bool(config.data.get("train_all_sources", False))
    eval_packed_root = config.data.get("eval_packed_code_root", config.data.packed_code_root)
    eval_route_cache = config.data.get("eval_route_cache", config.data.route_cache)
    independent_eval = (
        Path(eval_packed_root).resolve() != Path(config.data.packed_code_root).resolve()
        or Path(eval_route_cache).resolve() != Path(config.data.route_cache).resolve()
    )
    train_dataset = E117SparseCodeDataset(
        config.data.packed_code_root,
        config.data.route_cache,
        split="all" if train_all_sources else "train",
        eval_fraction=config.data.eval_fraction,
        split_seed=config.data.split_seed,
    )
    eval_dataset = E117SparseCodeDataset(
        eval_packed_root,
        eval_route_cache,
        split="all" if independent_eval else "eval",
        eval_fraction=config.data.eval_fraction,
        split_seed=config.data.split_seed,
    )
    replay_dataset = None
    retention_dataset = None
    if bool(config.data.get("full_1d_replay", False)):
        replay_eval_fraction = float(config.data.get("replay_eval_fraction", 0.002))
        if independent_eval:
            replay_dataset = PackedOneDCodeDataset(
                config.data.packed_code_root,
                split="all" if train_all_sources else "train",
                eval_fraction=replay_eval_fraction,
                split_seed=config.data.split_seed,
            )
            retention_dataset = PackedOneDCodeDataset(
                eval_packed_root,
                split="all",
                eval_fraction=replay_eval_fraction,
                split_seed=config.data.split_seed,
            )
        else:
            route_eval_sources = np.asarray(
                eval_dataset.source_indices[eval_dataset.local_rows], dtype=np.int64
            )
            replay_dataset = PackedOneDCodeDataset(
                config.data.packed_code_root,
                split="train",
                eval_fraction=replay_eval_fraction,
                split_seed=config.data.split_seed,
                excluded_source_indices=route_eval_sources,
            )
            retention_dataset = PackedOneDCodeDataset(
                config.data.packed_code_root,
                split="eval",
                eval_fraction=replay_eval_fraction,
                split_seed=config.data.split_seed,
            )
            if np.intersect1d(replay_dataset.source_indices, route_eval_sources).size:
                raise RuntimeError("full 1D replay leaks a route-evaluation source")
            if np.intersect1d(
                replay_dataset.source_indices, retention_dataset.source_indices
            ).size:
                raise RuntimeError("full 1D replay leaks an original 1D-evaluation source")
    train_sampler = KBucketBatchSampler(
        train_dataset,
        batch_size=int(config.training.per_gpu_batch_size),
        shuffle=True,
        drop_last=True,
        seed=int(config.training.seed),
    )
    eval_sampler = KBucketBatchSampler(
        eval_dataset,
        batch_size=int(config.feasibility.eval_batch_size),
        shuffle=False,
        drop_last=False,
        seed=int(config.training.seed),
        interleave=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=int(config.data.num_workers),
        pin_memory=True,
        persistent_workers=int(config.data.num_workers) > 0,
        collate_fn=collate_e117_sparse,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        num_workers=min(int(config.data.num_workers), 4),
        pin_memory=True,
        persistent_workers=int(config.data.num_workers) > 0,
        collate_fn=collate_e117_sparse,
    )
    replay_loader = None
    retention_loader = None
    replay_generator = None
    if replay_dataset is not None:
        replay_generator = torch.Generator().manual_seed(
            int(config.training.seed) + 1_000_000
        )
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=int(config.training.replay_1d_per_gpu_batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(config.data.num_workers),
            pin_memory=True,
            persistent_workers=int(config.data.num_workers) > 0,
            generator=replay_generator,
        )
        retention_loader = DataLoader(
            retention_dataset,
            batch_size=int(config.feasibility.eval_1d_batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=min(int(config.data.num_workers), 4),
            pin_memory=True,
            persistent_workers=int(config.data.num_workers) > 0,
        )
    model = E117SparseUnifiedMaskGIT(**OmegaConf.to_container(config.model, resolve=True))
    initialization_report = None
    if config.get("initialization", {}).get("checkpoint", None):
        initialization_report = load_pretrained_1d_stage(
            model,
            config.initialization.checkpoint,
            config.initialization.get("state_key", "model"),
        )
    optimizer = optimizer_for(model, config.optimizer)
    scheduler = scheduler_for(
        optimizer,
        int(config.optimizer.warmup_steps),
        int(config.training.max_steps),
        float(config.optimizer.min_lr_ratio),
    )
    if replay_loader is None:
        model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, scheduler
        )
    else:
        (
            model,
            optimizer,
            train_loader,
            eval_loader,
            replay_loader,
            retention_loader,
            scheduler,
        ) = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            eval_loader,
            replay_loader,
            retention_loader,
            scheduler,
        )
    ema_model = None
    if bool(config.training.get("use_ema", False)):
        ema_model = copy.deepcopy(accelerator.unwrap_model(model))
        ema_model.to(accelerator.device).eval().requires_grad_(False)
    audit = exposure_audit(
        config,
        accelerator.num_processes,
        int(train_dataset.local_rows.size),
    )
    if accelerator.is_main_process:
        accelerator.print(
            f"parameters={accelerator.unwrap_model(model).parameter_count():,} "
            f"train={len(train_dataset):,} eval={len(eval_dataset):,} gpu_processes={accelerator.num_processes}"
        )
        accelerator.print(f"exposure_audit={json.dumps(audit, sort_keys=True)}")
        if replay_dataset is not None:
            accelerator.print(
                f"full_1d_replay={len(replay_dataset):,} "
                f"global_1d_eval={len(retention_dataset):,} "
                f"replay_batch_per_gpu={int(config.training.replay_1d_per_gpu_batch_size)}"
            )
        if initialization_report is not None:
            accelerator.print(f"initialization={json.dumps(initialization_report, sort_keys=True)}")
    if config.logging.get("log_with", None):
        wandb_kwargs = {
            "name": str(config.experiment.name),
            "dir": str(output_dir),
        }
        if config.logging.get("run_id", None):
            wandb_kwargs.update(id=str(config.logging.run_id), resume="allow")
        if config.logging.get("entity", None):
            wandb_kwargs["entity"] = str(config.logging.entity)
        accelerator.init_trackers(
            project_name=str(config.logging.project),
            config=flatten_config(OmegaConf.to_container(config, resolve=True)),
            init_kwargs={"wandb": wandb_kwargs},
        )

    global_step = 0
    epoch = 0
    resume_path = resolve_resume_path(output_dir, config)
    pending_rng_state = None
    if resume_path is not None:
        global_step, epoch, pending_rng_state = restore_training_state(
            accelerator,
            model,
            ema_model,
            optimizer,
            scheduler,
            resume_path,
            config,
            replay_generator=replay_generator,
        )
        report_path = output_dir / "feasibility.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"resume requires the matching feasibility report: {report_path}"
            )
        report = json.loads(report_path.read_text())
        baseline = report["baseline"]
        accelerator.print(
            f"resumed={resume_path} global_step={global_step} completed_epochs={epoch}"
        )
    else:
        baseline = evaluate(
            model,
            eval_loader,
            accelerator,
            config,
            global_step=0,
            retention_loader=retention_loader,
        )
        baseline_status = feasibility_status(baseline, baseline, config)
        report = {
            "baseline": baseline,
            "initialization": initialization_report,
            "exposure_audit": audit,
            "evaluations": [{"step": 0, "metrics": baseline, "status": baseline_status}],
        }
        if accelerator.is_main_process:
            atomic_json(output_dir / "feasibility.json", report)
            accelerator.log({f"eval/{key}": value for key, value in baseline.items()}, step=0)
    if pending_rng_state is not None:
        restore_rng_state(pending_rng_state)

    consecutive_passes = 0
    for evaluation in reversed(report.get("evaluations", [])):
        if evaluation.get("status", {}).get("passed", False):
            consecutive_passes += 1
        else:
            break
    last_time = time.time()
    replay_iterator = None
    model.train()
    while global_step < int(config.training.max_steps):
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        else:
            train_sampler.set_epoch(epoch)
        reset_replay_each_epoch = bool(config.training.get("reset_replay_each_epoch", False))
        if replay_loader is not None and (replay_iterator is None or reset_replay_each_epoch):
            if hasattr(replay_loader, "set_epoch"):
                replay_loader.set_epoch(epoch)
            replay_iterator = iter(replay_loader)
        epoch_completed = True
        for batch_index, batch in enumerate(train_loader):
            replay_batch = None
            if replay_iterator is not None:
                try:
                    replay_batch = next(replay_iterator)
                except StopIteration:
                    replay_iterator = iter(replay_loader)
                    replay_batch = next(replay_iterator)
            with accelerator.accumulate(model):
                route_z1d, z2d = batch["z1d"], batch["z2d"]
                route_labels = batch["label"]
                if replay_batch is None:
                    z1d, labels_1d = route_z1d, route_labels
                else:
                    z1d, labels_1d = replay_batch["z1d"], replay_batch["label"]
                route_indices, route_valid = batch["route_indices"], batch["route_valid"]
                valid_1d = torch.ones_like(z1d, dtype=torch.bool)
                mask_1d, _ = sample_arccos_mask(valid_1d)
                mask_2d, _ = sample_arccos_mask(route_valid)
                input_1d = masked_inputs_1d(accelerator.unwrap_model(model), z1d, mask_1d)
                input_2d = masked_inputs_2d(accelerator.unwrap_model(model), z2d, mask_2d, route_valid)
                with accelerator.autocast():
                    logits_1d = model(stage="1d", input_tokens=input_1d, labels=labels_1d)
                    logits_2d = model(
                        stage="2d",
                        completed_1d=route_z1d,
                        input_tokens=input_2d,
                        route_indices=route_indices,
                        route_valid=route_valid,
                        labels=route_labels,
                    )
                    metrics_1d = masked_token_metrics(
                        logits_1d,
                        z1d,
                        mask_1d,
                        valid_1d,
                        config.training.label_smoothing,
                        config.training.unmasked_loss_weight,
                        config.training.get("loss_normalization", "per_sample"),
                    )
                    metrics_2d = masked_token_metrics(
                        logits_2d,
                        z2d,
                        mask_2d,
                        route_valid,
                        config.training.label_smoothing,
                        config.training.unmasked_loss_weight,
                        config.training.get("loss_normalization", "per_sample"),
                    )
                    loss = float(config.training.loss_1d_weight) * metrics_1d["loss"] + float(
                        config.training.loss_2d_weight
                    ) * metrics_2d["loss"]
                accelerator.backward(loss)
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), float(config.optimizer.max_grad_norm))
                if not bool(torch.isfinite(loss.detach())) or not bool(
                    torch.isfinite(torch.as_tensor(grad_norm, device=loss.device))
                ):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"non-finite training value at step={global_step}: "
                        f"loss={float(loss.detach())}, grad_norm={float(grad_norm)}"
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            if ema_model is not None:
                update_ema_model(
                    ema_model,
                    accelerator.unwrap_model(model),
                    float(config.training.get("ema_decay", 0.999)),
                )
            global_step += 1
            if scheduler_step_index(scheduler) != global_step:
                raise RuntimeError(
                    f"scheduler/global-step mismatch: {scheduler_step_index(scheduler)} != {global_step}"
                )
            if global_step % int(config.logging.log_every) == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                train_metrics = {
                    "train/loss": float(loss.detach()),
                    "train/loss_1d": float(metrics_1d["masked_loss"]),
                    "train/loss_2d": float(metrics_2d["masked_loss"]),
                    "train/acc_1d": float(metrics_1d["masked_acc"]),
                    "train/acc_2d": float(metrics_2d["masked_acc"]),
                    "train/top5_1d": float(metrics_1d["masked_top5"]),
                    "train/top5_2d": float(metrics_2d["masked_top5"]),
                    "train/mask_ratio_1d": float(metrics_1d["mask_ratio"]),
                    "train/mask_ratio_2d": float(metrics_2d["mask_ratio"]),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": float(scheduler.get_last_lr()[0]),
                    "train/seconds_per_logged_step": elapsed / int(config.logging.log_every),
                    "train/source_equivalent_epoch": (
                        global_step * audit["global_batch_2d"] / audit["num_train_sources"]
                    ),
                }
                accelerator.log(train_metrics, step=global_step)
                accelerator.print(
                    f"step={global_step} loss={train_metrics['train/loss']:.4f} "
                    f"1d={train_metrics['train/loss_1d']:.4f}/{train_metrics['train/top5_1d']:.4f} "
                    f"2d={train_metrics['train/loss_2d']:.4f}/{train_metrics['train/top5_2d']:.4f} "
                    f"grad={train_metrics['train/grad_norm']:.3f}"
                )

            should_eval = global_step % int(config.feasibility.eval_every) == 0
            if should_eval or global_step == int(config.training.max_steps):
                current = evaluate(
                    model,
                    eval_loader,
                    accelerator,
                    config,
                    global_step,
                    retention_loader=retention_loader,
                )
                status = feasibility_status(baseline, current, config)
                consecutive_passes = consecutive_passes + 1 if status["passed"] else 0
                will_stop_on_pass = (
                    bool(config.feasibility.stop_on_pass)
                    and global_step >= int(config.feasibility.min_steps)
                    and consecutive_passes >= int(config.feasibility.required_consecutive_passes)
                )
                report["evaluations"].append({"step": global_step, "metrics": current, "status": status})
                if accelerator.is_main_process:
                    atomic_json(output_dir / "feasibility.json", report)
                    accelerator.log({f"eval/{key}": value for key, value in current.items()}, step=global_step)
                    accelerator.log(
                        {
                            "feasibility/passed": int(status["passed"]),
                            "feasibility/loss_drop_1d": status["relative_loss_drop_1d"],
                            "feasibility/loss_drop_2d": status["relative_loss_drop_2d"],
                        },
                        step=global_step,
                    )
                checkpoint_config = config.get("checkpoint", {})
                save_on_pass_only = bool(checkpoint_config.get("save_on_pass_only", False))
                if bool(checkpoint_config.get("save_on_eval", True)) and bool(
                    checkpoint_config.get("save_latest", True)
                ) and (
                    not save_on_pass_only or will_stop_on_pass
                ):
                    save_latest(
                        accelerator,
                        model,
                        ema_model,
                        optimizer,
                        scheduler,
                        output_dir,
                        config,
                        global_step,
                        epoch,
                        replay_generator=replay_generator,
                    )
                accelerator.print(f"feasibility step={global_step}: {json.dumps(status, sort_keys=True)}")
                if will_stop_on_pass:
                    report["final"] = {"step": global_step, "reason": "preregistered_feasibility_gate_passed"}
                    if accelerator.is_main_process:
                        atomic_json(output_dir / "feasibility.json", report)
                    accelerator.end_training()
                    return
            if global_step >= int(config.training.max_steps):
                epoch_completed = batch_index + 1 >= len(train_loader)
                break
        if epoch_completed:
            epoch += 1
            checkpoint_config = config.get("checkpoint", {})
            save_every_epochs = int(checkpoint_config.get("save_every_epochs", 0))
            if (
                bool(checkpoint_config.get("save_latest", True))
                and not bool(checkpoint_config.get("save_on_pass_only", False))
                and save_every_epochs > 0
                and epoch % save_every_epochs == 0
            ):
                save_latest(
                    accelerator,
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    output_dir,
                    config,
                    global_step,
                    epoch,
                    replay_generator=replay_generator,
                )
                accelerator.print(
                    f"checkpoint latest.pt updated at completed_epoch={epoch} step={global_step}"
                )

    report["final"] = {"step": global_step, "reason": "max_steps_reached_without_gate"}
    if accelerator.is_main_process:
        atomic_json(output_dir / "feasibility.json", report)
    checkpoint_config = config.get("checkpoint", {})
    if bool(checkpoint_config.get("save_latest", True)) and not bool(
        checkpoint_config.get("save_on_pass_only", False)
    ):
        save_latest(
            accelerator,
            model,
            ema_model,
            optimizer,
            scheduler,
            output_dir,
            config,
            global_step,
            epoch,
            replay_generator=replay_generator,
        )
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    main(parser.parse_args())
