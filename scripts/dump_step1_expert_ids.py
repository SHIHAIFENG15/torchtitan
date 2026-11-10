#!/usr/bin/env python3
"""Dump step-1 routed expert ids for one Muse Spark recipe.

Saves, per rank, the first forward's top-k expert ids for every
TokenChoiceTopKRouter. Ids are sorted on the expert axis because topk
uses sorted=False. Model math is unchanged.
"""

import argparse
import os
import subprocess
import sys

import torch
from torch.distributed.tensor import DTensor
import torch.distributed as dist


_SAVED: dict[str, dict[str, torch.Tensor]] = {}
_SEEN: set[str] = set()
_STAMPED = False


def _stamp(engine) -> None:
    global _STAMPED
    if _STAMPED:
        return
    from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter

    for part_index, part in enumerate(engine.model_parts):
        for name, module in part.named_modules():
            if isinstance(module, TokenChoiceTopKRouter):
                prefix = "" if len(engine.model_parts) == 1 else f"part{part_index}."
                module._expert_dump_name = prefix + name
    _STAMPED = True


def _install() -> None:
    from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter
    from torchtitan.training_engine import TrainingEngine

    original_forward = TokenChoiceTopKRouter.forward

    def forward(self, x_TD, expert_bias_E=None, *, padding_mask_T=None, **kwargs):
        result = original_forward(
            self,
            x_TD,
            expert_bias_E,
            padding_mask_T=padding_mask_T,
            **kwargs,
        )
        name = getattr(self, "_expert_dump_name", None)
        if name and name not in _SEEN and os.environ.get("STEP1_EXPERT_DUMP_DIR"):
            _SEEN.add(name)
            t = result[1].detach()
            if isinstance(t, DTensor):
                t = t.to_local()
            import spmd_types as spmd
            with spmd.local():
                expert_ids = torch.sort(t, dim=-1).values.to(torch.int32).cpu()
                entry = {"expert_ids": expert_ids}
                if padding_mask_T is not None:
                    entry["padding_mask"] = padding_mask_T.detach().to(torch.bool).cpu()
            _SAVED[name] = entry
        return result

    TokenChoiceTopKRouter.forward = forward

    original_forward_backward = TrainingEngine.forward_backward

    def forward_backward(self, *args, **kwargs):
        _stamp(self)
        return original_forward_backward(self, *args, **kwargs)

    TrainingEngine.forward_backward = forward_backward

    original_optim = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0 and _SAVED:
            _write()
        return original_optim(self)

    TrainingEngine.optim_step = optim_step


def _write() -> None:
    out_dir = os.environ["STEP1_EXPERT_DUMP_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    payload = {
        "rank": rank,
        "world_size": world,
        "config": os.environ.get("STEP1_EXPERT_DUMP_CONFIG", ""),
        "routers": _SAVED,
    }
    stem = os.path.join(out_dir, f"rank{rank}")
    torch.save(payload, stem + ".pt.tmp")
    os.replace(stem + ".pt.tmp", stem + ".pt")
    print(
        f"STEP1_EXPERT_DUMP rank={rank} world={world} routers={len(_SAVED)} out={stem}.pt",
        flush=True,
    )


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
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(item for item in (root, prior) if item)
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
    env["STEP1_EXPERT_DUMP_WORKER"] = "1"
    env["STEP1_EXPERT_DUMP_DIR"] = out_dir
    env["STEP1_EXPERT_DUMP_CONFIG"] = args.config
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
        f"expert dump module={args.module} config={args.config} "
        f"ngpu={args.ngpu} load={load_path} out={out_dir}",
        flush=True,
    )
    return subprocess.run(cmd, cwd=root, env=env).returncode


if __name__ == "__main__":
    if os.environ.get("STEP1_EXPERT_DUMP_WORKER") == "1":
        _worker()
    else:
        raise SystemExit(_parent())
