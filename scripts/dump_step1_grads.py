#!/usr/bin/env python3
"""Dump step-1 parameter gradients, then run the usual one-step trainer.

The parent process configures the same checkpoint load path as
``run_acc.py``. Each torchrun worker patches ``TrainingEngine.optim_step``
and writes its local shards after backward, before clipping and the
optimizer update. Model math is unchanged.
"""

import argparse
import json
import os
import subprocess
import sys

import torch
import torch.distributed as dist


def _local_grad(grad: torch.Tensor) -> tuple[torch.Tensor, list[int], list[str] | None]:
    from torch.distributed.tensor import DTensor

    if isinstance(grad, DTensor):
        placements = [str(placement) for placement in grad.placements]
        return grad.to_local(), list(grad.shape), placements
    return grad, list(grad.shape), None


def _dump(engine) -> None:
    out_dir = os.environ["STEP1_GRAD_DUMP_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    summary = []
    tensors = {}
    for part_index, part in enumerate(engine.model_parts):
        for name, param in part.named_parameters():
            key = name if len(engine.model_parts) == 1 else f"part{part_index}.{name}"
            grad = param.grad
            row = {
                "name": key,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "has_grad": grad is not None,
            }
            if grad is not None:
                local, global_shape, placements = _local_grad(grad)
                values = local.detach().float().cpu().contiguous()
                finite = torch.isfinite(values)
                row["global_shape"] = global_shape
                row["local_shape"] = list(values.shape)
                row["placements"] = placements
                row["norm"] = float(torch.linalg.vector_norm(values)) if values.numel() else 0.0
                row["max_abs"] = float(values.abs().max()) if values.numel() else 0.0
                row["mean"] = float(values.mean()) if values.numel() else 0.0
                row["nonfinite"] = int((~finite).sum())
                tensors[key] = values
            summary.append(row)
    payload = {
        "rank": rank,
        "world_size": world,
        "num_completed_steps": engine.num_completed_steps,
        "config": os.environ.get("STEP1_GRAD_DUMP_CONFIG", ""),
        "grads": tensors,
    }
    stem = os.path.join(out_dir, f"rank{rank}")
    torch.save(payload, stem + ".pt.tmp")
    os.replace(stem + ".pt.tmp", stem + ".pt")
    with open(stem + ".json.tmp", "w", encoding="utf-8") as handle:
        json.dump({"rank": rank, "world_size": world, "params": summary}, handle)
    os.replace(stem + ".json.tmp", stem + ".json")
    missing = sum(1 for row in summary if not row["has_grad"])
    print(
        f"STEP1_GRAD_DUMP rank={rank} world={world} params={len(summary)} "
        f"missing={missing} out={stem}.pt",
        flush=True,
    )


def _install() -> None:
    from torchtitan.training_engine import TrainingEngine

    if getattr(TrainingEngine.optim_step, "_step1_grad_dump", False):
        return
    original = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0:
            _dump(self)
        return original(self)

    optim_step._step1_grad_dump = True
    TrainingEngine.optim_step = optim_step


def _worker() -> None:
    _install()
    from torchtitan.train import main

    main()


def _parent() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ngpu", required=True, type=int)
    parser.add_argument("--load", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tb", required=True)
    args = parser.parse_args()
    load_path = os.path.abspath(args.load)
    if not os.path.isdir(load_path):
        print(f"missing checkpoint: {load_path}", file=sys.stderr)
        return 2

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    from _checkpoint_test_config import configure_training_run

    env = os.environ.copy()
    env["TORCHTITAN_CHECKPOINT_INITIAL_LOAD_PATH"] = load_path
    module, config = configure_training_run(
        env,
        module=args.module,
        config=args.config,
        steps=1,
        tb_folder=args.tb,
        checkpoint_mode="load",
        export_dtype="bfloat16",
    )
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    env["STEP1_GRAD_DUMP_WORKER"] = "1"
    env["STEP1_GRAD_DUMP_DIR"] = out_dir
    env["STEP1_GRAD_DUMP_CONFIG"] = args.config
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["LOG_RANK"] = env.get("LOG_RANK", "0")
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(args.ngpu),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "--local-ranks-filter",
        env["LOG_RANK"],
        "--role",
        "rank",
        "--tee",
        "3",
        os.path.abspath(__file__),
        "--module",
        module,
        "--config",
        config,
    ]
    print(
        f"step1 dump module={args.module} config={args.config} "
        f"ngpu={args.ngpu} load={load_path} out={out_dir}",
        flush=True,
    )
    return subprocess.run(cmd, cwd=root, env=env).returncode


if __name__ == "__main__":
    if os.environ.get("STEP1_GRAD_DUMP_WORKER") == "1":
        _worker()
    else:
        raise SystemExit(_parent())
