#!/usr/bin/env python3
"""Resolve paths and an empirically selected H200 batch into an immutable config."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="configs/e117_maskgit_h200_8gpu.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--packed-code-root", required=True)
    parser.add_argument("--route-cache", required=True)
    parser.add_argument("--eval-packed-code-root", required=True)
    parser.add_argument("--eval-route-cache", required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--wandb-project", default="motar-maskgit")
    parser.add_argument("--wandb-entity", default="")
    return parser.parse_args()


def canonical(config) -> dict:
    return OmegaConf.to_container(config, resolve=True)


def main() -> None:
    args = parse_args()
    if args.micro_batch_size < 1:
        raise ValueError("micro batch must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_config.yaml"
    run_id_path = output_dir / "wandb_run_id.txt"
    if (output_dir / "latest.pt").is_file() and not resolved_path.is_file():
        raise RuntimeError("latest.pt exists without resolved_config.yaml; refusing unsafe resume")

    if run_id_path.is_file():
        run_id = run_id_path.read_text().strip()
        if not run_id:
            raise RuntimeError(f"empty W&B run id: {run_id_path}")
    else:
        run_id = uuid.uuid4().hex
        run_id_path.write_text(run_id + "\n")

    config = OmegaConf.load(args.template)
    world_size = int(config.h200.required_gpu_count)
    global_batch = args.micro_batch_size * world_size
    target_exposures = int(config.training.target_image_exposures)
    max_steps = int(math.floor(target_exposures / global_batch + 0.5))
    planned_exposures = max_steps * global_batch
    warmup_steps = max(
        1,
        int(
            math.floor(
                max_steps
                * int(config.training.reference_warmup_steps)
                / int(config.training.reference_max_steps)
                + 0.5
            )
        ),
    )
    eval_every = max(
        1,
        int(math.floor(float(config.h200.reference_eval_exposures) / global_batch + 0.5)),
    )
    min_steps = max(
        1,
        int(
            math.floor(
                float(config.h200.reference_min_gate_exposures) / global_batch + 0.5
            )
        ),
    )
    eval_batch = min(args.micro_batch_size, 64)
    eval_batches = max(1, math.ceil(2048 / (eval_batch * world_size)))

    config.experiment.name = args.experiment_name
    config.experiment.output_dir = str(output_dir)
    config.data.packed_code_root = str(Path(args.packed_code_root).expanduser().resolve())
    config.data.route_cache = str(Path(args.route_cache).expanduser().resolve())
    config.data.eval_packed_code_root = str(
        Path(args.eval_packed_code_root).expanduser().resolve()
    )
    config.data.eval_route_cache = str(Path(args.eval_route_cache).expanduser().resolve())
    config.data.num_workers = int(args.num_workers)
    config.training.per_gpu_batch_size = int(args.micro_batch_size)
    config.training.replay_1d_per_gpu_batch_size = int(args.micro_batch_size)
    config.training.max_steps = max_steps
    config.optimizer.warmup_steps = warmup_steps
    config.feasibility.eval_batch_size = eval_batch
    config.feasibility.eval_batches = eval_batches
    config.feasibility.eval_1d_batch_size = eval_batch
    config.feasibility.eval_1d_batches = eval_batches
    config.feasibility.eval_every = eval_every
    config.feasibility.min_steps = min_steps
    config.logging.project = args.wandb_project
    config.logging.run_id = run_id
    if args.wandb_entity:
        config.logging.entity = args.wandb_entity

    if resolved_path.is_file():
        existing = OmegaConf.load(resolved_path)
        if canonical(existing) != canonical(config):
            raise RuntimeError(
                "resolved config differs from the existing run. Use a new EXP_NAME "
                "instead of changing an in-progress experiment."
            )
    else:
        OmegaConf.save(config, resolved_path)

    plan = {
        "format": "e117_maskgit_h200_run_plan_v1",
        "experiment_name": args.experiment_name,
        "output_dir": str(output_dir),
        "world_size": world_size,
        "micro_batch_size": int(args.micro_batch_size),
        "global_batch_size": global_batch,
        "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
        "max_steps": max_steps,
        "target_image_exposures": target_exposures,
        "planned_image_exposures": planned_exposures,
        "exposure_error": planned_exposures - target_exposures,
        "source_equivalent_epochs": planned_exposures
        / int(config.training.reference_num_train_sources),
        "warmup_steps": warmup_steps,
        "learning_rate": float(config.optimizer.lr),
        "loss_1d_weight": float(config.training.loss_1d_weight),
        "loss_2d_weight": float(config.training.loss_2d_weight),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity or None,
        "wandb_run_id_file": str(run_id_path),
        "checkpoint": str(output_dir / "latest.pt"),
        "checkpoint_policy": "replace latest.pt after every completed data epoch",
        "created_by_pid": os.getpid(),
    }
    plan_path = output_dir / "run_plan.json"
    if plan_path.is_file():
        old_plan = json.loads(plan_path.read_text())
        # Process id is diagnostic and naturally changes on a restart.
        old_plan.pop("created_by_pid", None)
        comparable = dict(plan)
        comparable.pop("created_by_pid", None)
        if old_plan != comparable:
            raise RuntimeError("run_plan.json differs from the requested immutable run")
    else:
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
