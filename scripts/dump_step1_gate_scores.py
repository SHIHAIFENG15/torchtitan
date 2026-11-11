#!/usr/bin/env python3
"""Dump step-1 router gate operands and scores for one Muse Spark recipe.

Same launch path as scripts/dump_step1_expert_ids.py. The first forward of
each TokenChoiceTopKRouter is unchanged; the hook only records tensors from
that forward. topk stays sorted=False. Saved expert_ids are sorted on K so
they can be compared with outputs/step1_expert_ids.
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


def _to_cpu(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().to(dtype).cpu()


def _install() -> None:
    from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter
    from torchtitan.training_engine import TrainingEngine

    original_forward = TokenChoiceTopKRouter.forward

    def forward(self, x_TD, expert_bias_E=None, *, padding_mask_T=None, **kwargs):
        name = getattr(self, "_expert_dump_name", None)
        record = bool(name) and name not in _SEEN and bool(os.environ.get("STEP1_EXPERT_DUMP_DIR"))
        captured: dict[str, torch.Tensor] = {}
        orig_linear = None
        orig_score = None
        if record:
            if not hasattr(self.gate, "_linear"):
                raise RuntimeError(f"{type(self.gate).__name__} has no _linear")
            orig_linear = self.gate._linear
            orig_score = self.score_func

            def _linear(inp, weight, bias):
                out = orig_linear(inp, weight, bias)
                captured["gate_input"] = inp.detach()
                captured["gate_weight"] = weight.detach()
                captured["gate_logits"] = out.detach()
                return out

            class _RecordScore:
                def __call__(self, logits_TE, **kw):
                    out = orig_score(logits_TE, **kw)
                    captured["scores"] = out.detach()
                    return out

            self.gate._linear = _linear
            self.score_func = _RecordScore()
        try:
            result = original_forward(
                self,
                x_TD,
                expert_bias_E,
                padding_mask_T=padding_mask_T,
                **kwargs,
            )
        finally:
            if orig_linear is not None:
                self.gate._linear = orig_linear
            if orig_score is not None:
                self.score_func = orig_score

        if record:
            _SEEN.add(name)
            import spmd_types as spmd

            with spmd.local():
                topk_ids = result[1].detach()
                topk_scores = result[0].detach()
                gate_input = _to_cpu(captured["gate_input"], torch.bfloat16)
                gate_weight = _to_cpu(captured["gate_weight"], torch.bfloat16).flatten(0, -2)
                gate_logits = _to_cpu(captured["gate_logits"], torch.float32)
                gate_input = gate_input.reshape(-1, gate_input.shape[-1])
                gate_logits = gate_logits.reshape(-1, gate_logits.shape[-1])
                entry = {
                    "expert_ids": torch.sort(_to_cpu(topk_ids, torch.int32), dim=-1).values,
                    "topk_ids": _to_cpu(topk_ids, torch.int32),
                    "topk_scores": _to_cpu(topk_scores, torch.float32),
                    "gate_input": gate_input,
                    "gate_weight": gate_weight,
                    "gate_logits": gate_logits,
                    "scores": _to_cpu(captured["scores"], torch.float32),
                }
                if expert_bias_E is not None:
                    bias = _to_cpu(expert_bias_E, torch.float32)
                    entry["expert_bias"] = bias
                    entry["scores_for_choice"] = entry["scores"] + bias
                if padding_mask_T is not None:
                    entry["padding_mask"] = _to_cpu(padding_mask_T, torch.bool)
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
        "git_rev": os.environ.get("STEP1_EXPERT_DUMP_GIT", ""),
        "torch_version": torch.__version__,
        "routers": _SAVED,
    }
    stem = os.path.join(out_dir, f"rank{rank}")
    torch.save(payload, stem + ".pt.tmp")
    os.replace(stem + ".pt.tmp", stem + ".pt")
    print(
        f"STEP1_GATE_DUMP rank={rank} world={world} routers={len(_SAVED)} out={stem}.pt",
        flush=True,
    )


def _check(out_dir: str, expect_dir: str) -> int:
    required = (
        "expert_ids",
        "topk_ids",
        "topk_scores",
        "gate_input",
        "gate_weight",
        "gate_logits",
        "scores",
        "scores_for_choice",
        "expert_bias",
        "padding_mask",
    )
    mismatched = 0
    compared = 0
    for rank in range(8):
        path = os.path.join(out_dir, f"rank{rank}.pt")
        if not os.path.isfile(path):
            print(f"missing {path}", file=sys.stderr)
            return 3
        payload = torch.load(path, weights_only=False)
        routers = payload["routers"]
        names = [f"layers.{i}.feed_forward.router" for i in range(8)]
        if set(routers) != set(names) or len(routers) != len(names):
            print(f"rank{rank} routers={list(routers)}", file=sys.stderr)
            return 3
        expect = None
        expect_path = os.path.join(expect_dir, f"rank{rank}.pt")
        if os.path.isfile(expect_path):
            expect = torch.load(expect_path, weights_only=False)["routers"]
        for name in names:
            entry = routers[name]
            for key in required:
                if key not in entry:
                    print(f"rank{rank} {name} missing {key}", file=sys.stderr)
                    return 3
            tokens = entry["gate_input"].shape[0]
            experts = entry["scores"].shape[-1]
            if entry["gate_input"].shape != (tokens, entry["gate_weight"].shape[-1]):
                print(f"rank{rank} {name} gate_input {tuple(entry['gate_input'].shape)}", file=sys.stderr)
                return 3
            if entry["gate_weight"].shape[0] != experts or entry["scores"].shape != (tokens, experts):
                print(f"rank{rank} {name} scores {tuple(entry['scores'].shape)}", file=sys.stderr)
                return 3
            if entry["gate_logits"].shape != (tokens, experts):
                print(f"rank{rank} {name} gate_logits {tuple(entry['gate_logits'].shape)}", file=sys.stderr)
                return 3
            if entry["scores_for_choice"].shape != (tokens, experts):
                print(f"rank{rank} {name} scores_for_choice {tuple(entry['scores_for_choice'].shape)}", file=sys.stderr)
                return 3
            if entry["expert_ids"].shape != (tokens, 2) or entry["topk_ids"].shape != (tokens, 2):
                print(f"rank{rank} {name} expert_ids {tuple(entry['expert_ids'].shape)}", file=sys.stderr)
                return 3
            if entry["padding_mask"].shape != (tokens,):
                print(f"rank{rank} {name} padding_mask {tuple(entry['padding_mask'].shape)}", file=sys.stderr)
                return 3
            if entry["gate_input"].dtype != torch.bfloat16 or entry["gate_weight"].dtype != torch.bfloat16:
                print(f"rank{rank} {name} gate dtype", file=sys.stderr)
                return 3
            if entry["gate_logits"].dtype != torch.float32 or entry["scores"].dtype != torch.float32:
                print(f"rank{rank} {name} score dtype", file=sys.stderr)
                return 3
            if expect is not None:
                old = expect[name]["expert_ids"]
                compared += old.numel()
                mismatched += int((old != entry["expert_ids"]).sum())
        print(
            f"rank{rank} tokens={routers[names[0]]['gate_input'].shape[0]} "
            f"experts={routers[names[0]]['scores'].shape[-1]}",
            flush=True,
        )
    if compared == 0:
        print(f"no expert-id baseline in {expect_dir}", file=sys.stderr)
        return 3
    print(f"expert_ids mismatch {mismatched}/{compared} vs {expect_dir}", flush=True)
    if mismatched:
        return 3
    print("STEP1_GATE_DUMP check ok", flush=True)
    return 0


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
    parser.add_argument(
        "--expect-ids",
        default="outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4",
    )
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
    git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    env["STEP1_EXPERT_DUMP_WORKER"] = "1"
    env["STEP1_EXPERT_DUMP_DIR"] = out_dir
    env["STEP1_EXPERT_DUMP_CONFIG"] = args.config
    env["STEP1_EXPERT_DUMP_GIT"] = git_rev
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["LOG_RANK"] = env.get("LOG_RANK", "0")
    for key, value in (
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        ("TORCH_DEVICE_BACKEND_AUTOLOAD", "1"),
        ("MUSE_DEBUG_DETERMINISTIC", "1"),
    ):
        env.setdefault(key, value)
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
        f"gate dump module={args.module} config={args.config} "
        f"ngpu={args.ngpu} load={load_path} out={out_dir}",
        flush=True,
    )
    code = subprocess.run(cmd, cwd=root, env=env).returncode
    if code != 0:
        return code
    return _check(out_dir, os.path.abspath(args.expect_ids))


if __name__ == "__main__":
    if os.environ.get("STEP1_EXPERT_DUMP_WORKER") == "1":
        _worker()
    else:
        raise SystemExit(_parent())
