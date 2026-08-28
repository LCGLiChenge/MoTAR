#!/usr/bin/env python3
"""Teacher-forced TiTok-1D position diagnostics for a MoTAR latest checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file

from motar.model import TiTokLlamaGenUnifiedAR, interleave_tokens


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def prefix_logits(model, z1d, labels):
    batch_size = z1d.shape[0]
    device = z1d.device
    model.freqs_cis = model.freqs_cis.to(device)
    condition = model.cls_embedding(labels, train=False)[:, : model.cls_token_num]
    query, content = model._titok_prefix(z1d)
    hidden = torch.cat((condition, interleave_tokens(query, content)), dim=1)

    cls_freqs = model.freqs_cis[: model.cls_token_num]
    cls_freqs = cls_freqs.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    titok_freqs = model._titok_freqs(batch_size, device)
    titok_freqs = interleave_tokens(titok_freqs, titok_freqs)
    frequencies = torch.cat((cls_freqs, titok_freqs), dim=1)
    for layer in model.layers:
        hidden = layer(hidden, frequencies, None, None)
    hidden = model.norm(hidden)
    start = model.cls_token_num
    end = start + model.titok_num_tokens * 2
    return model.titok_output(hidden[:, start:end:2]).float()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h200_8gpu_150epoch.yaml")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--packed-code-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--augmentation", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank, world_size, device = init_distributed()
    config = OmegaConf.load(args.config)
    model = TiTokLlamaGenUnifiedAR(
        **OmegaConf.to_container(config.model, resolve=True)
    )
    state = load_file(
        str(Path(args.checkpoint_dir).expanduser().resolve() / "model.safetensors"),
        device="cpu",
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.grad_checkpointing = False

    root = Path(args.packed_code_root).expanduser().resolve()
    codes = np.load(root / "titok_codes.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")
    count = min(int(args.num_samples), int(codes.shape[0]))
    if count < 1:
        raise ValueError("--num-samples must select at least one row")
    if not 0 <= args.augmentation < codes.shape[1]:
        raise ValueError("augmentation index is out of range")

    positions = int(config.model.titok_num_tokens)
    ce_sum = torch.zeros(positions, dtype=torch.float64, device=device)
    top1_sum = torch.zeros_like(ce_sum)
    top5_sum = torch.zeros_like(ce_sum)
    local_count = 0
    local_ids = np.arange(rank, count, world_size, dtype=np.int64)

    for start in range(0, len(local_ids), args.batch_size):
        indices = local_ids[start : start + args.batch_size]
        z1d = torch.from_numpy(
            np.array(codes[indices, args.augmentation], dtype=np.int64, copy=True)
        ).to(device)
        class_labels = torch.from_numpy(
            np.array(labels[indices], dtype=np.int64, copy=True)
        ).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = prefix_logits(model, z1d, class_labels)
        ce = F.cross_entropy(logits.transpose(1, 2), z1d, reduction="none")
        predictions = logits.argmax(dim=-1)
        top5 = logits.topk(5, dim=-1).indices.eq(z1d.unsqueeze(-1)).any(dim=-1)
        ce_sum += ce.double().sum(dim=0)
        top1_sum += predictions.eq(z1d).double().sum(dim=0)
        top5_sum += top5.double().sum(dim=0)
        local_count += len(indices)

    count_tensor = torch.tensor(local_count, dtype=torch.int64, device=device)
    if world_size > 1:
        for tensor in (ce_sum, top1_sum, top5_sum, count_tensor):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        total = int(count_tensor.item())
        ce = (ce_sum / total).cpu().tolist()
        top1 = (top1_sum / total).cpu().tolist()
        top5 = (top5_sum / total).cpu().tolist()
        ranges = ((0, 4), (4, 8), (8, 16), (16, 24), (24, 32))
        report = {
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "packed_code_root": str(root),
            "selection": f"fixed sample indices [0,{count})",
            "augmentation": args.augmentation,
            "num_samples": total,
            "world_size": world_size,
            "overall": {
                "cross_entropy": sum(ce) / positions,
                "perplexity": math.exp(min(sum(ce) / positions, 50.0)),
                "top1_accuracy": sum(top1) / positions,
                "top5_accuracy": sum(top5) / positions,
            },
            "groups": {
                f"positions_{begin:02d}_{end - 1:02d}": {
                    "cross_entropy": sum(ce[begin:end]) / (end - begin),
                    "top1_accuracy": sum(top1[begin:end]) / (end - begin),
                    "top5_accuracy": sum(top5[begin:end]) / (end - begin),
                }
                for begin, end in ranges
            },
            "positions": [
                {
                    "position": index,
                    "cross_entropy": ce[index],
                    "top1_accuracy": top1[index],
                    "top5_accuracy": top5[index],
                }
                for index in range(positions)
            ],
        }
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
